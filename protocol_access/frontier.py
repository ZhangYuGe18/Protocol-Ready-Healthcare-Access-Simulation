from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .core import simulate_cohort


INTERVAL_BOUNDARY_COLUMNS = (
    "frontier_first",
    "frontier_last",
    "largest_interval_start",
    "largest_interval_end",
)
INTERVAL_MEASURE_COLUMNS = (
    "largest_interval_span",
    "feasible_total_measure",
)


def standardized_location_frame(
    commute,
    transfers,
    crossings,
    friction,
    n_agents,
    route_entropy=0.0,
    route_duration_cv=0.0,
    prefix="location",
):
    return pd.DataFrame(
        {
            "patient_id": [f"{prefix}_{i:03d}" for i in range(int(n_agents))],
            "grp": "standardized_location",
            "commute_min_model": float(commute),
            "transfers": float(transfers),
            "crossings_derived": float(crossings),
            "mobility_friction_proxy": float(friction),
            "route_entropy": float(route_entropy),
            "route_duration_cv": float(route_duration_cv),
            "ready_soft_data": 0.50,
            "n_ready_evidence": 0,
        }
    )


def _run_fixed(frame, pars, config, work_min, seed, fraction_offset=None):
    offset = (
        int(config["frontier_reference_fraction"]) - 1
        if fraction_offset is None
        else int(fraction_offset)
    )
    logs = simulate_cohort(
        frame,
        pars,
        config,
        scenario="baseline",
        seed=int(seed),
        fractions=1,
        fraction_offset=offset,
        strategy_override=float(work_min),
    )
    n = max(int(len(logs)), 1)
    access = float(logs["protocol_ready_first_pass"].mean())
    return {
        "access": access,
        "access_mcse": float(np.sqrt(max(access * (1.0 - access), 0.0) / n)),
        "on_time": float(logs["on_time"].mean()),
        "rework": float(logs["rework"].mean()),
        "final_ready": float(logs["final_true_ready"].mean()),
        "hold": float(logs["hold_total_min"].mean()),
        "system_queue": float(logs["system_queue_min"].mean()),
    }


def estimate_location_fixed_work(
    ensemble,
    config,
    commute,
    transfers,
    crossings,
    friction,
    seed,
    work_min=None,
    n_agents=None,
    route_entropy=0.0,
    route_duration_cv=0.0,
):
    n_agents = int(n_agents or config["frontier_agents"])
    work_min = float(config["fixed_work_min"] if work_min is None else work_min)
    actual = standardized_location_frame(
        commute, transfers, crossings, friction, n_agents,
        route_entropy=route_entropy, route_duration_cv=route_duration_cv
    )
    simple = standardized_location_frame(
        commute, 0.0, 0.0, friction, n_agents,
        route_entropy=0.0, route_duration_cv=0.0
    )
    rows = []
    for member, pars in enumerate(ensemble):
        member_seed = int(seed + member * 1009)
        observed = _run_fixed(actual, pars, config, work_min, member_seed)
        counterfactual = _run_fixed(simple, pars, config, work_min, member_seed)
        rows.append(
            {
                "member": member,
                **{f"{key}_actual": value for key, value in observed.items()},
                **{f"{key}_simple": value for key, value in counterfactual.items()},
                "route_structure_penalty": counterfactual["access"] - observed["access"],
            }
        )
    result = pd.DataFrame(rows)
    output = {"work_min": float(work_min)}
    for column in [
        "access_actual",
        "access_simple",
        "route_structure_penalty",
        "access_mcse_actual",
        "on_time_actual",
        "rework_actual",
        "final_ready_actual",
        "hold_actual",
        "system_queue_actual",
    ]:
        output[f"{column}_p05"] = float(result[column].quantile(0.05))
        output[f"{column}_p50"] = float(result[column].median())
        output[f"{column}_p95"] = float(result[column].quantile(0.95))
    return output


def extract_connected_components(candidates, success, target):
    """Return connected components of the linearly interpolated feasible set.

    Threshold crossings are interpolated on each segment, so infeasible gaps
    remain separate and component lengths are defined on the evaluated domain.
    """

    candidates = np.asarray(candidates, dtype=float)
    success = np.asarray(success, dtype=float)
    valid = np.isfinite(candidates) & np.isfinite(success)
    candidates = candidates[valid]
    success = success[valid]
    if not len(candidates):
        return []
    order = np.argsort(candidates)
    candidates = candidates[order]
    success = success[order]
    target = float(target)
    feasible = success >= target
    components = []
    start = float(candidates[0]) if feasible[0] else None
    for index in range(len(candidates) - 1):
        left, right = bool(feasible[index]), bool(feasible[index + 1])
        if left == right:
            continue
        x0, x1 = float(candidates[index]), float(candidates[index + 1])
        y0, y1 = float(success[index]), float(success[index + 1])
        if abs(y1 - y0) < 1e-12:
            crossing = 0.5 * (x0 + x1)
        else:
            crossing = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        crossing = float(np.clip(crossing, x0, x1))
        if (not left) and right:
            start = crossing
        elif left and (not right) and start is not None:
            end = crossing
            n_points = int(np.sum(feasible & (candidates >= start) & (candidates <= end)))
            components.append(
                {
                    "start": float(start),
                    "end": end,
                    "span": float(max(0.0, end - start)),
                    "measure": float(max(0.0, end - start)),
                    "n_points": n_points,
                }
            )
            start = None
    if feasible[-1] and start is not None:
        end = float(candidates[-1])
        n_points = int(np.sum(feasible & (candidates >= start) & (candidates <= end)))
        components.append(
            {
                "start": float(start),
                "end": end,
                "span": float(max(0.0, end - start)),
                "measure": float(max(0.0, end - start)),
                "n_points": n_points,
            }
        )
    return components


def interval_summary(candidates, success, target):
    components = extract_connected_components(candidates, success, target)
    if not components:
        return {
            "attained": False,
            "all_components": [],
            "first_boundary": np.nan,
            "last_boundary": np.nan,
            "largest_start": np.nan,
            "largest_end": np.nan,
            # The target-attaining set is empty.  Its measure and the length of
            # its largest component are both zero; only its boundaries are
            # undefined.  Keeping these two derived quantities on the same
            # member rows is necessary for the mathematical invariant M >= L.
            "largest_span": 0.0,
            "total_measure": 0.0,
            "hole_count": 0,
        }
    largest = max(components, key=lambda item: (item["measure"], -item["start"]))
    result = {
        "attained": True,
        "all_components": components,
        "first_boundary": components[0]["start"],
        "last_boundary": components[-1]["end"],
        "largest_start": largest["start"],
        "largest_end": largest["end"],
        "largest_span": largest["span"],
        "total_measure": float(sum(item["measure"] for item in components)),
        "hole_count": max(0, len(components) - 1),
    }
    if result["total_measure"] + 1e-10 < result["largest_span"]:
        raise AssertionError("Connected-set invariant failed: total measure M is smaller than largest interval L")
    return result


def validate_interval_member_rows(members, key_columns=("member",), tolerance=1e-10):
    """Validate paired member-level connected-set measures.

    Empty feasible sets must be represented as ``M=L=0``.  Boundary columns
    may remain missing because an empty set has no start or end.  The function
    deliberately rejects duplicated pairing keys and asymmetric missingness so
    downstream quantiles cannot silently use different posterior members.
    """

    required = set(key_columns).union(INTERVAL_MEASURE_COLUMNS)
    missing = required.difference(members.columns)
    if missing:
        raise ValueError(f"Missing interval audit columns: {sorted(missing)}")
    if members.duplicated(list(key_columns)).any():
        duplicated = members.loc[members.duplicated(list(key_columns), keep=False), list(key_columns)]
        raise ValueError(f"Duplicated interval pairing keys: {duplicated.head().to_dict('records')}")
    measures = members.loc[:, list(INTERVAL_MEASURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if measures.isna().any().any():
        counts = measures.isna().sum()
        raise ValueError(f"Missing paired interval measures: {counts[counts.gt(0)].to_dict()}")
    if (measures < -float(tolerance)).any().any():
        raise AssertionError("Connected-set measures must be non-negative")
    invalid = measures["feasible_total_measure"] + float(tolerance) < measures["largest_interval_span"]
    if invalid.any():
        keys = members.loc[invalid, list(key_columns)].head().to_dict("records")
        raise AssertionError(f"Connected-set invariant M >= L failed for paired rows: {keys}")
    return True


def summarize_paired_route_groups(member_rows, quantiles=(0.05, 0.50, 0.95)):
    """Aggregate route groups without breaking member/configuration pairing.

    The aggregation takes paired posterior quantiles within configuration,
    then the equal-configuration median within route group.  Every operation is
    monotone, so M >= L is retained at the member, configuration, route-group,
    and reported-quantile levels.
    """

    required = {"origin_id", "member", "transfers"}.union(INTERVAL_MEASURE_COLUMNS)
    missing = required.difference(member_rows.columns)
    if missing:
        raise ValueError(f"Missing route-group interval columns: {sorted(missing)}")
    data = member_rows.copy()
    validate_interval_member_rows(data, key_columns=("origin_id", "member"))
    member_sets = data.groupby("origin_id")["member"].agg(lambda values: tuple(sorted(values)))
    if member_sets.nunique() != 1:
        raise ValueError("Route configurations do not contain the same posterior-member rows")
    data["route_group"] = np.select(
        [data["transfers"].le(0), data["transfers"].eq(1)],
        ["No transfer", "One transfer"],
        default="At least two transfers",
    )

    rows = []
    group_order = ["No transfer", "One transfer", "At least two transfers", "All configurations"]
    for route_group in group_order:
        selected = data if route_group == "All configurations" else data[data["route_group"].eq(route_group)]
        if selected.empty:
            continue
        row = {
            "route_group": route_group,
            "n_configurations": int(selected["origin_id"].nunique()),
            "n_posterior_members": int(selected["member"].nunique()),
            "n_matched_member_configuration_rows": int(len(selected)),
            "aggregation_order": "paired posterior quantile within configuration; equal-configuration median within route group",
        }
        for quantile in quantiles:
            suffix = f"p{int(round(100 * float(quantile))):02d}"
            configuration = selected.groupby("origin_id", as_index=False).agg(
                **{
                    column: (column, lambda values, q=float(quantile): values.quantile(q))
                    for column in INTERVAL_MEASURE_COLUMNS
                }
            )
            validate_interval_member_rows(configuration, key_columns=("origin_id",))
            for column in INTERVAL_MEASURE_COLUMNS:
                row[f"{column}_{suffix}"] = float(configuration[column].median())
        rows.append(row)
    result = pd.DataFrame(rows)
    for quantile in quantiles:
        suffix = f"p{int(round(100 * float(quantile))):02d}"
        invalid = result[f"feasible_total_measure_{suffix}"] + 1e-10 < result[f"largest_interval_span_{suffix}"]
        if invalid.any():
            raise AssertionError(f"Route-group connected-set invariant M >= L failed at {suffix}")
    return result


def estimate_location_frontier(
    ensemble,
    config,
    commute,
    transfers,
    crossings,
    friction,
    seed,
    n_agents=None,
    route_entropy=0.0,
    route_duration_cv=0.0,
):
    """Estimate all connected target-attaining preparation intervals."""

    n_agents = int(n_agents or config["frontier_agents"])
    candidates = np.asarray(config["frontier_candidates_min"], dtype=float)
    frame = standardized_location_frame(
        commute,
        transfers,
        crossings,
        friction,
        n_agents,
        route_entropy=route_entropy,
        route_duration_cv=route_duration_cv,
    )
    member_rows = []
    curves = []
    mcse_curves = []
    for member, pars in enumerate(ensemble):
        member_seed = int(seed + member * 1009)
        success = []
        access_mcse = []
        rework = []
        final_ready = []
        for work in candidates:
            metrics = _run_fixed(frame, pars, config, work, member_seed)
            success.append(metrics["access"])
            access_mcse.append(metrics["access_mcse"])
            rework.append(metrics["rework"])
            final_ready.append(metrics["final_ready"])
        success = np.asarray(success, dtype=float)
        summary = interval_summary(candidates, success, config["frontier_target"])
        peak_index = int(np.argmax(success))
        fixed_index = int(np.argmin(np.abs(candidates - config["fixed_work_min"])))
        member_rows.append(
            {
                "member": member,
                "attained": summary["attained"],
                "frontier_first": summary["first_boundary"],
                "frontier_last": summary["last_boundary"],
                "largest_interval_start": summary["largest_start"],
                "largest_interval_end": summary["largest_end"],
                "largest_interval_span": summary["largest_span"],
                "feasible_total_measure": summary["total_measure"],
                "hole_count": summary["hole_count"],
                "components_json": json.dumps(summary["all_components"]),
                "peak_access": float(success[peak_index]),
                "peak_work": float(candidates[peak_index]),
                "access_at_fixed": float(success[fixed_index]),
                "mcse_at_fixed": float(access_mcse[fixed_index]),
                "rework_at_fixed": float(rework[fixed_index]),
                "final_ready_at_fixed": float(final_ready[fixed_index]),
            }
        )
        curves.append(success)
        mcse_curves.append(access_mcse)

    members = pd.DataFrame(member_rows)
    validate_interval_member_rows(members, key_columns=("member",))
    curves = np.vstack(curves)
    mcse_curves = np.vstack(mcse_curves)
    attainable = members["attained"].astype(bool)
    output = {
        "frontier_target": float(config["frontier_target"]),
        "frontier_candidate_min": float(candidates.min()),
        "frontier_candidate_max": float(candidates.max()),
        "frontier_unattained_share": float((~attainable).mean()),
        "any_hole_share": float((members["hole_count"] > 0).mean()),
    }
    columns = [
        "frontier_first",
        "frontier_last",
        "largest_interval_start",
        "largest_interval_end",
        "largest_interval_span",
        "feasible_total_measure",
        "hole_count",
        "peak_access",
        "peak_work",
        "access_at_fixed",
        "mcse_at_fixed",
        "rework_at_fixed",
        "final_ready_at_fixed",
    ]
    for column in columns:
        # Boundaries are conditional on attainment; set measures, hole counts,
        # and all outcome quantities use the full, identical member set.  In
        # particular, an unattained member contributes M=L=0.
        values = members.loc[attainable, column] if column in INTERVAL_BOUNDARY_COLUMNS else members[column]
        output[f"{column}_p05"] = float(values.quantile(0.05)) if len(values) else np.nan
        output[f"{column}_p50"] = float(values.median()) if len(values) else np.nan
        output[f"{column}_p95"] = float(values.quantile(0.95)) if len(values) else np.nan
    output["curve_candidates_min"] = candidates.tolist()
    output["curve_access_p05"] = np.quantile(curves, 0.05, axis=0).tolist()
    output["curve_access_p50"] = np.quantile(curves, 0.50, axis=0).tolist()
    output["curve_access_p95"] = np.quantile(curves, 0.95, axis=0).tolist()
    output["curve_mcse_p50"] = np.median(mcse_curves, axis=0).tolist()
    output["member_intervals"] = members.to_dict("records")
    return output
