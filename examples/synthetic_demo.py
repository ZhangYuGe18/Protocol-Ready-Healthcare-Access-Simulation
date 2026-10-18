"""Run the model on fictional agents and illustrative, uncalibrated settings."""

import pandas as pd

from protocol_access import simulate_cohort, summarize_run
from protocol_access.calibration import BOUNDS, FIXED


def example_parameters():
    """Use range midpoints solely to demonstrate the software interface."""
    parameters = dict(FIXED)
    parameters.update({name: (low + high) / 2 for name, (low, high) in BOUNDS.items()})
    return parameters


def example_config():
    """Fictional service setting; not a published study configuration."""
    return {
        "fractions": 3,
        "patients_per_day": 4,
        "slot_interval_min": 18.0,
        "qa_resources": 2,
        "treatment_resources": 2,
        "qa_coverage": 1.0,
        "qa_sensitivity": 0.92,
        "qa_specificity": 0.94,
        "late_grace_min": 6.0,
        "state_target_low": 0.75,
        "state_target_center": 1.0,
        "state_target_high": 1.25,
        "state_capacity": 1.8,
        "state_reset": 0.05,
        "state_readiness_softness": 0.04,
        "course_state_drift_per_fraction": 0.01,
        "minimum_recheck_min": 5.0,
        "max_corrections": 2,
        "strategy_min_min": 20.0,
        "strategy_max_min": 160.0,
        "adaptive_warmup_fractions": 1,
        "adaptive_action_cap_min": 15.0,
        "adaptive_adherence_probability": 0.8,
        "adaptive_strategy_noise_min": 3.0,
        "adaptive_measurement_log_sd": 0.12,
        "adaptive_prior_log_sd": 0.3,
        "adaptive_target_probability": 0.7,
        "adaptive_tail_probability": 0.15,
        "adaptive_work_weight": 0.08,
        "adaptive_rework_weight": 1.0,
        "adaptive_overfill_weight": 0.35,
        "adaptive_candidates_min": list(range(20, 161, 5)),
        "policy_samples": 32,
    }


def example_agents():
    """Fabricated route descriptors; no real places, journeys, or patients."""
    return pd.DataFrame(
        {
            "patient_id": ["SYN-A", "SYN-B", "SYN-C", "SYN-D"],
            "grp": ["illustrative"] * 4,
            "commute_min_model": [15.0, 25.0, 35.0, 45.0],
            "transfers": [0.0, 1.0, 0.0, 2.0],
            "crossings_derived": [0.0, 1.0, 2.0, 1.0],
            "route_entropy": [0.1, 0.3, 0.2, 0.4],
            "route_duration_cv": [0.05, 0.1, 0.08, 0.12],
            "mobility_friction_proxy": [0.2, 0.4, 0.6, 0.8],
            "ready_soft_data": [0.5] * 4,
            "n_ready_evidence": [0] * 4,
        }
    )


def main():
    agents = example_agents()
    pars = example_parameters()
    config = example_config()
    for scenario in ("baseline", "adaptive_support"):
        logs = simulate_cohort(agents, pars, config, scenario=scenario, seed=7)
        _, summary = summarize_run(logs)
        print(f"{scenario}: {summary.to_dict(orient='records')}")


if __name__ == "__main__":
    main()
