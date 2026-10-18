from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .events import EventCalendar, ResourcePool
from .policy import RateBelief, choose_robust_strategy
from .state import BladderBandKernel, ObservationGate


SCENARIOS = {
    "baseline": {"description": "Reference route and usual preparation"},
    "route_direct": {
        "description": "Same commute and preparation behavior; transfers and crossings removed",
        "route_direct": True,
    },
    "route_adjacent": {
        "description": "Adjacent lodging route; prior preparation behavior retained",
        "route_adjacent": True,
    },
    "adaptive_support": {
        "description": "History-based adaptive preparation support on the reference route",
        "adaptive_support": True,
    },
    "route_adjacent_plus_adaptive": {
        "description": "Adjacent route plus history-based adaptive preparation support",
        "route_adjacent": True,
        "adaptive_support": True,
    },
    "gate_fast_track": {
        "description": "Reference route and preparation with one additional checking resource",
        "gate_fast_track": True,
    },
}


def finite(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def lognormal_mean_cv(rng, mean, cv):
    if mean <= 0:
        return 0.0
    cv = max(0.0, float(cv))
    sigma = math.sqrt(math.log1p(cv * cv))
    mu = math.log(float(mean)) - 0.5 * sigma * sigma
    return float(rng.lognormal(mu, sigma))


def transform_scenario(frame, scenario):
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario}")
    d = frame.copy()
    spec = SCENARIOS[scenario]
    # This flag is computed from the reference configuration before a
    # counterfactual route is applied.  Consequently, relocating a patient does
    # not silently grant the preparation coaching/alignment already available
    # to on-campus or no-road-crossing pathways.
    d["stable_pathway_baseline"] = (
        d.get("commute_min_model", pd.Series(np.inf, index=d.index)).fillna(np.inf).le(10.0)
        & d.get("transfers", pd.Series(np.inf, index=d.index)).fillna(np.inf).eq(0)
    ).astype(float)
    d["behavior_friction_proxy"] = d.get(
        "mobility_friction_proxy", pd.Series(0.5, index=d.index)
    ).fillna(0.5)
    if spec.get("route_direct"):
        d["transfers"] = 0.0
        d["crossings_derived"] = 0.0
        d["route_entropy"] = 0.0
        d["route_duration_cv"] = 0.0
    if spec.get("route_adjacent"):
        d["commute_min_model"] = 6.0
        d["transfers"] = 0.0
        d["crossings_derived"] = 0.0
        d["route_entropy"] = 0.0
        d["route_duration_cv"] = 0.0
        d["mobility_friction_proxy"] = 0.10
    return d, spec


def _kernel(config):
    return BladderBandKernel(
        target_low=config["state_target_low"],
        target_high=config["state_target_high"],
        target_center=config["state_target_center"],
        capacity=config["state_capacity"],
        readiness_softness=config["state_readiness_softness"],
        reset_state=config["state_reset"],
    )


def _clinical_signal(frame):
    """Return an image-derived patient signal with no geography/group input."""

    signal = frame.get("ready_soft_data", pd.Series(0.50, index=frame.index))
    observed = frame.get("n_ready_evidence", pd.Series(0, index=frame.index)).fillna(0).gt(0)
    signal = signal.where(observed, 0.50).fillna(0.50).clip(0.05, 0.95)
    return signal.to_numpy(float)


def _travel_parameters(row, pars):
    """Assemble travel uncertainty with an auditable structure weight.

    ``route_structure_scale`` is a mechanism weight rather than a direct
    transfer effect. Setting it to zero removes every incremental
    transfer/crossing/entropy/alternative-route term while retaining common
    travel noise and the common disruption process.  A value of one reproduces
    the full route mechanism.
    """

    commute = max(0.0, finite(row.get("commute_min_model"), 0.0))
    transfers = max(0.0, finite(row.get("transfers"), 0.0))
    crossings = max(0.0, finite(row.get("crossings_derived"), 0.0))
    entropy = max(0.0, finite(row.get("route_entropy"), 0.0))
    structure_scale = np.clip(finite(pars.get("route_structure_scale"), 1.0), 0.0, 4.0)
    structural_cv = (
        pars["travel_cv_base"]
        + structure_scale
        * (
            pars["travel_cv_transfer"] * transfers
            + pars["travel_cv_crossing"] * crossings
            + pars.get("travel_cv_entropy", 0.03) * entropy
        )
    )
    alternative_cv = (
        structure_scale * max(0.0, finite(row.get("route_duration_cv"), 0.0))
    )
    cv = math.sqrt(structural_cv**2 + alternative_cv**2)
    disruption = np.clip(
        pars.get("disruption_probability_base", 0.03)
        + structure_scale
        * (
            pars.get("disruption_probability_transfer", 0.06) * transfers
            + pars.get("disruption_probability_crossing", 0.015) * crossings
            + pars.get("disruption_probability_entropy", 0.05) * entropy
        ),
        0.0,
        0.80,
    )
    return commute, transfers, crossings, float(cv), float(disruption), float(structure_scale)


def simulate_cohort(
    frame,
    pars,
    config,
    scenario="baseline",
    seed=0,
    fractions=None,
    fraction_offset=0,
    strategy_override=None,
    course_drift_override=None,
    target_band_override=None,
    gate_override=None,
):
    """Run the network-uncertain, history-adaptive ABM--DES.

    Geography affects route reliability and ordinary timing behavior.  It never
    enters the physiological rate equation.  Adaptive support can read only
    prior measured states and elapsed preparation; it cannot read the current
    latent rate or future random stream.
    """

    rng = np.random.default_rng(seed)
    d, spec = transform_scenario(frame.reset_index(drop=True), scenario)
    n = len(d)
    if n == 0:
        return pd.DataFrame()
    n_fx = int(fractions or config["fractions"])
    clinical_signal = _clinical_signal(d)
    behavior_friction = d["behavior_friction_proxy"].fillna(0.5).clip(0, 1).to_numpy(float)

    clinical_rate_factor = np.exp(pars["clinical_rate_weight"] * (clinical_signal - 0.50))
    patient_rate_factor = clinical_rate_factor * rng.lognormal(
        -0.5 * pars["patient_rate_sd"] ** 2,
        pars["patient_rate_sd"],
        n,
    )
    base_rate = config["state_target_center"] / np.maximum(
        pars["required_prep"] / patient_rate_factor, 1.0
    )
    stable_pathway = d["stable_pathway_baseline"].to_numpy(float)
    strategy_mean = (
        pars["base_prep"]
        + pars["buffer_coef"] * behavior_friction
    )
    alignment = np.clip(pars.get("stable_pathway_alignment", 0.0), 0.0, 1.0) * stable_pathway
    strategy_mean = (1.0 - alignment) * strategy_mean + alignment * pars["required_prep"]
    stable_multiplier = (
        1.0
        - stable_pathway
        + stable_pathway * pars.get("stable_pathway_variability_multiplier", 1.0)
    )
    strategy = strategy_mean + rng.normal(
        0.0,
        pars["patient_strategy_sd"]
        * (0.35 + 0.65 * behavior_friction)
        * stable_multiplier,
        n,
    )
    if strategy_override is not None:
        if np.isscalar(strategy_override):
            strategy = np.full(n, float(strategy_override))
        else:
            strategy = np.asarray(strategy_override, dtype=float)
            if len(strategy) != n:
                raise ValueError("strategy_override must be scalar or match frame length")
    strategy = np.clip(strategy, config["strategy_min_min"], config["strategy_max_min"])

    belief_prior_mean = math.log(config["state_target_center"] / pars["required_prep"])
    beliefs = [
        RateBelief(belief_prior_mean, float(config["adaptive_prior_log_sd"]) ** 2)
        for _ in range(n)
    ]
    kernel = _kernel(config)
    if target_band_override:
        for key, value in target_band_override.items():
            setattr(kernel, key, float(value))
    gate_values = {
        "coverage": config["qa_coverage"],
        "sensitivity": pars.get("qa_sensitivity", config["qa_sensitivity"]),
        "specificity": pars.get("qa_specificity", config["qa_specificity"]),
    }
    gate_values.update(gate_override or {})
    gate = ObservationGate(**gate_values)
    logs = []

    for fx in range(n_fx):
        course_fraction = int(fraction_offset) + fx
        drift = (
            float(config["course_state_drift_per_fraction"])
            if course_drift_override is None
            else float(course_drift_override)
        )
        course_scale = math.exp(-drift * course_fraction)
        order = rng.permutation(n)
        for day_start in range(0, n, int(config["patients_per_day"])):
            batch = order[day_start : day_start + int(config["patients_per_day"])]
            qa_capacity = int(config["qa_resources"]) + int(bool(spec.get("gate_fast_track")))
            qa = ResourcePool(qa_capacity)
            treatment = ResourcePool(int(config["treatment_resources"]))
            calendar = EventCalendar()
            states = {}

            for slot, i in enumerate(batch):
                row = d.iloc[i]
                appointment = float(slot) * float(config["slot_interval_min"])
                (
                    commute,
                    transfers,
                    crossings,
                    travel_cv,
                    disruption_probability,
                    route_structure_scale,
                ) = _travel_parameters(row, pars)
                arrival_buffer = pars["arrival_buffer"] + pars["arrival_buffer_friction"] * behavior_friction[i]

                recommended = np.nan
                policy_expected = np.nan
                policy_tail = np.nan
                policy_overfill = np.nan
                eligible_adaptive = bool(
                    spec.get("adaptive_support")
                    and fx >= int(config["adaptive_warmup_fractions"])
                    and strategy_override is None
                )
                # Day-to-day execution is a behavioral process.  Its variance
                # grows with route/coordination friction, while physiological
                # rate variability below remains route-independent.
                scheduled_strategy = (
                    (1.0 - alignment[i]) * strategy[i]
                    + alignment[i] * pars["required_prep"] / max(course_scale, 0.35)
                )
                episode_strategy = float(
                    scheduled_strategy
                    + rng.normal(
                        0.0,
                        pars.get("fraction_strategy_sd", 0.0)
                        * (0.25 + behavior_friction[i])
                        * stable_multiplier[i],
                    )
                )
                if eligible_adaptive:
                    recommendation = choose_robust_strategy(
                        beliefs[i],
                        strategy[i],
                        course_scale,
                        commute,
                        travel_cv,
                        arrival_buffer,
                        disruption_probability,
                        pars["disruption_delay_mean"],
                        kernel,
                        config,
                        rng,
                    )
                    recommended = recommendation["strategy"]
                    policy_expected = recommendation["expected_ready"]
                    policy_tail = recommendation["tail_ready"]
                    policy_overfill = recommendation["overfill_risk"]
                    adheres = rng.random() < float(config["adaptive_adherence_probability"])
                    if adheres:
                        episode_strategy = recommended + rng.normal(
                            0.0, float(config["adaptive_strategy_noise_min"])
                        )
                episode_strategy = float(
                    np.clip(episode_strategy, config["strategy_min_min"], config["strategy_max_min"])
                )

                actual_travel = lognormal_mean_cv(rng, commute, travel_cv)
                disrupted = bool(rng.random() < disruption_probability)
                disruption_delay = (
                    float(rng.gamma(2.0, pars["disruption_delay_mean"] / 2.0)) if disrupted else 0.0
                )
                crossing_delay = float(
                    rng.gamma(
                        max(crossings, 1.0),
                        pars.get("crossing_wait_mean", 0.8) * route_structure_scale,
                    )
                ) if crossings > 0 and route_structure_scale > 0 else 0.0
                actual_travel += disruption_delay + crossing_delay
                leave = appointment - commute - arrival_buffer
                arrival = leave + actual_travel
                late_grace = pars.get("late_grace", config["late_grace_min"])
                on_time = bool(arrival <= appointment + late_grace)

                episode_rate = base_rate[i] * course_scale * rng.lognormal(
                    -0.5 * pars["fraction_rate_sd"] ** 2,
                    pars["fraction_rate_sd"],
                )
                prep_start = appointment - episode_strategy
                qa_request = max(arrival, appointment)
                states[int(i)] = {
                    "row": row,
                    "appointment": appointment,
                    "arrival": arrival,
                    "leave": leave,
                    "arrival_buffer": arrival_buffer,
                    "actual_travel": actual_travel,
                    "travel_cv": travel_cv,
                    "route_structure_scale": route_structure_scale,
                    "disrupted": disrupted,
                    "disruption_delay": disruption_delay,
                    "crossing_delay": crossing_delay,
                    "on_time": on_time,
                    "prep_start": prep_start,
                    "episode_strategy": episode_strategy,
                    "recommended_strategy": recommended,
                    "policy_expected": policy_expected,
                    "policy_tail": policy_tail,
                    "policy_overfill": policy_overfill,
                    "adaptive_eligible": eligible_adaptive,
                    "rate": episode_rate,
                    "course_scale": course_scale,
                    "latent_quantile": rng.random(),
                    "reference_state": 0.0,
                    "reference_time": prep_start,
                    "qa_request": qa_request,
                    "qa_queue": 0.0,
                    "correction_count": 0,
                    "correction_minutes": 0.0,
                    "correction_action": "none",
                    "first_gate_true_ready": False,
                    "first_gate_state": np.nan,
                    "first_gate_probability": np.nan,
                    "release": np.nan,
                    "gate_limit_release": False,
                }
                calendar.schedule(qa_request, "qa", i, priority=1)

            def state_at(state, time):
                return kernel.advance(
                    state["reference_state"],
                    max(0.0, float(time) - float(state["reference_time"])),
                    state["rate"],
                )

            def schedule_correction(state, agent, qa_end, observed_state):
                state_end = kernel.advance(
                    observed_state,
                    qa_end - state["last_gate_start"],
                    state["rate"],
                )
                plan = kernel.correction(state_end, state["rate"])
                minutes = max(float(config["minimum_recheck_min"]), float(plan.minutes))
                if plan.reset_state is not None:
                    state["reference_state"] = float(plan.reset_state)
                    state["reference_time"] = float(qa_end)
                state["correction_count"] += 1
                state["correction_minutes"] += minutes
                state["correction_action"] = plan.action
                calendar.schedule(qa_end + minutes, "recheck", agent, priority=0)

            while calendar:
                event = calendar.pop()
                state = states[event.agent]
                if event.kind in {"qa", "recheck"}:
                    qa_start, qa_end = qa.acquire(
                        event.time,
                        rng.gamma(4.0, pars["qa_service"] / 4.0),
                    )
                    observed_state = state_at(state, qa_start)
                    probability = kernel.readiness_probability(observed_state)
                    true_ready = bool(state["latent_quantile"] < probability)
                    gate_result = gate.inspect_truth(true_ready, rng)
                    inadequate = bool(gate_result["inadequate"])
                    state["last_gate_start"] = qa_start
                    if event.kind == "qa":
                        state["qa_queue"] = max(0.0, qa_start - state["qa_request"])
                        state["first_gate_true_ready"] = true_ready
                        state["first_gate_state"] = observed_state
                        state["first_gate_probability"] = probability
                        state["first_gate_time"] = qa_start

                    if inadequate and state["correction_count"] < int(config["max_corrections"]):
                        schedule_correction(state, event.agent, qa_end, observed_state)
                    else:
                        state["gate_limit_release"] = bool(
                            inadequate and state["correction_count"] >= int(config["max_corrections"])
                        )
                        state["release"] = qa_end
                        calendar.schedule(qa_end, "treatment", event.agent, priority=1)

                elif event.kind == "treatment":
                    treatment_start, treatment_end = treatment.acquire(
                        event.time,
                        rng.gamma(6.0, pars["treatment_service"] / 6.0),
                    )
                    final_state = state_at(state, treatment_start)
                    final_probability = kernel.readiness_probability(final_state)
                    final_true_ready = bool(state["latent_quantile"] < final_probability)
                    direct = state["correction_count"] == 0
                    first_pass_success = bool(state["on_time"] and direct and final_true_ready)
                    row = state["row"]
                    logs.append(
                        {
                            "patient_id": row["patient_id"],
                            "grp": row.get("grp", "standardized"),
                            "fraction": course_fraction + 1,
                            "scenario": scenario,
                            "adaptive_support": int(state["adaptive_eligible"]),
                            "course_state_scale": state["course_scale"],
                            "strategy_min": state["episode_strategy"],
                            "recommended_strategy_min": state["recommended_strategy"],
                            "policy_expected_ready": state["policy_expected"],
                            "policy_tail_ready": state["policy_tail"],
                            "policy_overfill_risk": state["policy_overfill"],
                            "hold_total_min": max(0.0, treatment_start - state["prep_start"]),
                            "wait_total_min": max(0.0, treatment_start - state["arrival"]),
                            "travel_realized_min": state["actual_travel"],
                            "travel_cv": state["travel_cv"],
                            "route_structure_scale": state["route_structure_scale"],
                            "route_disruption": int(state["disrupted"]),
                            "route_disruption_delay_min": state["disruption_delay"],
                            "route_crossing_delay_min": state["crossing_delay"],
                            "arrival_buffer_min": state["arrival_buffer"],
                            "on_time": int(state["on_time"]),
                            "first_gate_state": state["first_gate_state"],
                            "first_gate_probability": state["first_gate_probability"],
                            "first_gate_true_ready": int(state["first_gate_true_ready"]),
                            "rework": int(state["correction_count"] > 0),
                            "correction_count": state["correction_count"],
                            "correction_action": state["correction_action"],
                            "correction_minutes": state["correction_minutes"],
                            "gate_limit_release": int(state["gate_limit_release"]),
                            "final_state": final_state,
                            "final_ready_probability": final_probability,
                            "final_true_ready": int(final_true_ready),
                            "protocol_ready_first_pass": int(first_pass_success),
                            "preappointment_wait_min": max(0.0, state["appointment"] - state["arrival"]),
                            "qa_queue_min": state["qa_queue"],
                            "treatment_queue_min": max(0.0, treatment_start - state["release"]),
                            "system_queue_min": state["qa_queue"]
                            + max(0.0, treatment_start - state["release"]),
                            "treatment_end": treatment_end,
                        }
                    )

                    # Every completed fraction contributes a past-state update.
                    # The controller is enabled only after the prespecified
                    # warm-up, but it never discards the warm-up observations.
                    elapsed = state["first_gate_time"] - state["prep_start"]
                    beliefs[event.agent].update(
                        state["first_gate_state"],
                        elapsed,
                        state["course_scale"],
                        config["adaptive_measurement_log_sd"],
                    )
                    if state["adaptive_eligible"]:
                        if np.isfinite(state["recommended_strategy"]):
                            strategy[event.agent] = 0.65 * strategy[event.agent] + 0.35 * state[
                                "recommended_strategy"
                            ]
                    elif strategy_override is None:
                        learning = pars["learning_rate"]
                        if state["correction_count"] > 0 and state["correction_action"] == "continue_filling":
                            strategy[event.agent] += learning * 8.0
                        elif state["correction_count"] > 0 and state["correction_action"] == "void_and_refill":
                            strategy[event.agent] -= learning * 8.0
                        elif treatment_start - state["prep_start"] > pars["long_hold_threshold"]:
                            strategy[event.agent] -= learning * 4.0
                    strategy[event.agent] = float(
                        np.clip(
                            strategy[event.agent],
                            config["strategy_min_min"],
                            config["strategy_max_min"],
                        )
                    )

    return pd.DataFrame(logs)


def summarize_run(logs):
    if logs.empty:
        return pd.DataFrame(), pd.DataFrame()
    patient = logs.groupby(["patient_id", "grp"], as_index=False).agg(
        hold_pred=("hold_total_min", "median"),
        wait_pred=("wait_total_min", "median"),
        strategy_pred=("strategy_min", "median"),
        rework_probability=("rework", "mean"),
        correction_minutes=("correction_minutes", "mean"),
        first_gate_ready=("first_gate_true_ready", "mean"),
        final_ready=("final_true_ready", "mean"),
        on_time_probability=("on_time", "mean"),
        protocol_ready_probability=("protocol_ready_first_pass", "mean"),
        system_queue=("system_queue_min", "mean"),
    )
    group = patient.groupby("grp", as_index=False).agg(
        n=("patient_id", "size"),
        hold_pred=("hold_pred", "median"),
        wait_pred=("wait_pred", "median"),
        strategy_pred=("strategy_pred", "median"),
        rework_probability=("rework_probability", "mean"),
        final_ready=("final_ready", "mean"),
        on_time_probability=("on_time_probability", "mean"),
        protocol_ready_probability=("protocol_ready_probability", "mean"),
    )
    return patient, group
