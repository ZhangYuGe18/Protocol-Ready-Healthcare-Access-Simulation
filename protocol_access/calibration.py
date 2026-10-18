from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import qmc, wasserstein_distance

from .core import simulate_cohort, summarize_run


BOUNDS = {
    "base_prep": (30.0, 74.0),
    "buffer_coef": (0.0, 18.0),
    "arrival_buffer": (5.0, 32.0),
    "arrival_buffer_friction": (0.0, 22.0),
    "required_prep": (50.0, 96.0),
    "patient_strategy_sd": (2.0, 20.0),
    "fraction_strategy_sd": (0.0, 20.0),
    "stable_pathway_alignment": (0.0, 1.0),
    "stable_pathway_variability_multiplier": (0.05, 0.70),
    "patient_rate_sd": (0.03, 0.25),
    "fraction_rate_sd": (0.03, 0.20),
    "clinical_rate_weight": (0.0, 1.0),
    "travel_cv_base": (0.05, 0.30),
    # One mechanism weight summarizes correlated route-structure terms.
    # Zero removes their incremental effects; one retains them at their
    # nominal levels, while larger values allow sensitivity analysis.
    "route_structure_scale": (0.0, 2.0),
    "qa_sensitivity": (0.75, 1.00),
    "qa_specificity": (0.85, 1.00),
    "qa_service": (1.5, 6.0),
    "treatment_service": (6.0, 14.0),
}


FIXED = {
    "travel_cv_transfer": 0.060,
    "travel_cv_crossing": 0.025,
    "travel_cv_entropy": 0.030,
    "disruption_probability_base": 0.030,
    "disruption_probability_transfer": 0.060,
    "disruption_probability_crossing": 0.015,
    "disruption_probability_entropy": 0.050,
    "disruption_delay_mean": 14.0,
    "crossing_wait_mean": 0.8,
    "learning_rate": 0.15,
    "long_hold_threshold": 110.0,
    "late_grace": 5.0,
}


def sobol_candidates(n, seed):
    dimension = len(BOUNDS)
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    power = int(math.ceil(math.log2(max(int(n), 2))))
    unit = sampler.random_base2(power)[: int(n)]
    lower = np.array([bounds[0] for bounds in BOUNDS.values()])
    upper = np.array([bounds[1] for bounds in BOUNDS.values()])
    scaled = qmc.scale(unit, lower, upper)
    rows = []
    for candidate, values in enumerate(scaled):
        pars = dict(FIXED)
        pars.update(dict(zip(BOUNDS, values)))
        pars["candidate"] = int(candidate)
        rows.append(pars)
    return pd.DataFrame(rows)


def observed_scales(frame):
    def robust_scale(column, default):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 3:
            return float(default)
        mad = float(np.median(np.abs(values - np.median(values)))) * 1.4826
        return max(mad, float(default))

    return {
        "hold": robust_scale("hold_patient_median", 12.0),
        "wait": robust_scale("wait_patient_median", 10.0),
        "rework": 0.20,
    }


def simulate_candidate(frame, pars, config, seed, replicates=None):
    rows = []
    replicates = int(replicates or config["calibration_replicates"])
    for replicate in range(replicates):
        logs = simulate_cohort(
            frame,
            pars,
            config,
            scenario="baseline",
            seed=int(seed) + replicate * 100003,
            fractions=int(config["calibration_fractions"]),
        )
        patient, _ = summarize_run(logs)
        patient["replicate"] = replicate
        rows.append(patient)
    combined = pd.concat(rows, ignore_index=True)
    summary = combined.groupby(["patient_id", "grp"], as_index=False).agg(
        hold_pred=("hold_pred", "mean"),
        wait_pred=("wait_pred", "mean"),
        rework_pred=("rework_probability", "mean"),
        access_pred=("protocol_ready_probability", "mean"),
        final_ready_pred=("final_ready", "mean"),
    )
    return summary


def discrepancy(observed, predicted, scales=None):
    scales = scales or observed_scales(observed)
    joined = observed.merge(predicted, on=["patient_id", "grp"], how="inner")
    components = {}
    for observed_column, predicted_column, key in [
        ("hold_patient_median", "hold_pred", "hold"),
        ("wait_patient_median", "wait_pred", "wait"),
    ]:
        valid = joined.dropna(subset=[observed_column, predicted_column])
        components[f"{key}_patient"] = float(
            np.mean(((valid[predicted_column] - valid[observed_column]) / scales[key]) ** 2)
        ) if len(valid) else 0.0

    rework = joined.dropna(subset=["rework_observed_rate", "rework_pred"])
    components["rework_brier"] = float(
        np.mean((rework["rework_pred"] - rework["rework_observed_rate"]) ** 2)
        / scales["rework"] ** 2
    ) if len(rework) else 0.0

    group_obs = joined.groupby("grp", as_index=False).agg(
        hold_obs=("hold_patient_median", "median"),
        wait_obs=("wait_patient_median", "median"),
        rework_obs=("rework_observed_rate", "mean"),
    )
    group_pred = joined.groupby("grp", as_index=False).agg(
        hold_pred=("hold_pred", "median"),
        wait_pred=("wait_pred", "median"),
        rework_pred=("rework_pred", "mean"),
    )
    group = group_obs.merge(group_pred, on="grp", how="inner")
    components["hold_group"] = float(
        np.nanmean(((group["hold_pred"] - group["hold_obs"]) / scales["hold"]) ** 2)
    )
    components["wait_group"] = float(
        np.nanmean(((group["wait_pred"] - group["wait_obs"]) / scales["wait"]) ** 2)
    )
    components["rework_group"] = float(
        np.nanmean(((group["rework_pred"] - group["rework_obs"]) / scales["rework"]) ** 2)
    )
    weights = {
        "hold_patient": 0.20,
        "wait_patient": 0.20,
        "rework_brier": 0.20,
        "hold_group": 0.15,
        "wait_group": 0.10,
        "rework_group": 0.15,
    }
    loss = float(sum(weights[key] * components[key] for key in weights))
    return loss, components


def abc_weights(candidate_scores, bandwidth_quantile=0.20):
    scores = np.asarray(candidate_scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("No finite calibration scores")
    minimum = float(np.min(scores[finite]))
    excess = scores - minimum
    bandwidth = float(np.quantile(excess[finite], float(bandwidth_quantile)))
    bandwidth = max(bandwidth, 1e-8)
    # Gaussian-kernel ABC weighting on the excess composite discrepancy.
    log_weight = -0.5 * (excess / bandwidth) ** 2
    log_weight[~finite] = -np.inf
    log_weight -= np.max(log_weight[finite])
    weight = np.exp(log_weight)
    weight /= weight.sum()
    ess = float(1.0 / np.sum(weight**2))
    return weight, bandwidth, ess


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]
    if not len(values) or weights.sum() <= 0:
        return np.full(len(np.atleast_1d(quantiles)), np.nan)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= weights.sum()
    return np.interp(np.atleast_1d(quantiles), cdf, values)


def posterior_predictive(observed, predictions, weights):
    predictions = predictions.merge(weights[["candidate", "weight"]], on="candidate", how="left")
    rows = []
    for (patient_id, grp), group in predictions.groupby(["patient_id", "grp"]):
        row = {"patient_id": patient_id, "grp": grp}
        for metric in ["hold_pred", "wait_pred", "rework_pred", "access_pred", "final_ready_pred"]:
            q = weighted_quantile(group[metric], [0.05, 0.10, 0.50, 0.90, 0.95], group["weight"])
            for label, value in zip(["p05", "p10", "p50", "p90", "p95"], q):
                row[f"{metric}_{label}"] = float(value)
        rows.append(row)
    return observed.merge(pd.DataFrame(rows), on=["patient_id", "grp"], how="left")


def validation_metrics(patient):
    out = {"n_patients": int(patient["patient_id"].nunique())}
    for observed_column, predicted_stem, label in [
        ("hold_patient_median", "hold_pred", "hold"),
        ("wait_patient_median", "wait_pred", "wait"),
    ]:
        valid = patient.dropna(subset=[observed_column, f"{predicted_stem}_p50"])
        error = valid[f"{predicted_stem}_p50"] - valid[observed_column]
        out[f"{label}_mae_min"] = float(np.mean(np.abs(error)))
        out[f"{label}_rmse_min"] = float(np.sqrt(np.mean(error**2)))
        out[f"{label}_wasserstein_min"] = float(
            wasserstein_distance(valid[observed_column], valid[f"{predicted_stem}_p50"])
        )
        for level, lo, hi in [("80", "p10", "p90"), ("90", "p05", "p95")]:
            out[f"{label}_{level}_coverage"] = float(
                np.mean(
                    (valid[observed_column] >= valid[f"{predicted_stem}_{lo}"])
                    & (valid[observed_column] <= valid[f"{predicted_stem}_{hi}"])
                )
            )

    rework = patient.dropna(subset=["rework_observed_rate", "rework_pred_p50"])
    p = rework["rework_pred_p50"].clip(1e-6, 1 - 1e-6)
    y = rework["rework_observed_rate"].clip(0, 1)
    out["rework_brier"] = float(np.mean((p - y) ** 2))
    out["rework_log_score"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    if len(rework) >= 5 and np.std(logit(p)) > 0:
        x = logit(p).to_numpy()
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]
        out["rework_calibration_intercept_linear"] = float(beta[0])
        out["rework_calibration_slope_linear"] = float(beta[1])
    return out


def identifiability(candidate_table):
    rows = []
    for name, (lower, upper) in BOUNDS.items():
        values = candidate_table[name].to_numpy(float)
        weights = candidate_table["weight"].to_numpy(float)
        mean = float(np.sum(values * weights))
        variance = float(np.sum(weights * (values - mean) ** 2))
        prior_variance = float((upper - lower) ** 2 / 12.0)
        q05, q50, q95 = weighted_quantile(values, [0.05, 0.50, 0.95], weights)
        rows.append(
            {
                "parameter": name,
                "prior_low": lower,
                "prior_high": upper,
                "posterior_p05": q05,
                "posterior_p50": q50,
                "posterior_p95": q95,
                "variance_ratio_posterior_to_prior": variance / prior_variance,
                "identified_flag": bool(variance / prior_variance < 0.60),
            }
        )
    return pd.DataFrame(rows)
