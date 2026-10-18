from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class RateBelief:
    """Log-rate belief available to the patient/support process.

    This is deliberately not initialized from residential group.  The only
    updates come from prior fractions' measured state and elapsed preparation.
    """

    mean: float
    variance: float

    def update(self, observed_state, elapsed_min, course_scale, measurement_log_sd):
        elapsed = max(float(elapsed_min), 1e-6)
        scale = max(float(course_scale), 1e-6)
        state = float(observed_state)
        if not np.isfinite(state) or state <= 0.02 or state >= 1.75:
            return self
        observation = math.log(state / elapsed / scale)
        obs_var = max(float(measurement_log_sd) ** 2, 1e-6)
        prior_precision = 1.0 / max(float(self.variance), 1e-8)
        obs_precision = 1.0 / obs_var
        variance = 1.0 / (prior_precision + obs_precision)
        mean = variance * (prior_precision * float(self.mean) + obs_precision * observation)
        self.mean = float(mean)
        self.variance = float(variance)
        return self


def draw_route_delay(
    rng,
    n,
    commute,
    travel_cv,
    arrival_buffer,
    disruption_probability,
    disruption_delay_mean,
):
    commute = max(float(commute), 0.0)
    travel_cv = max(float(travel_cv), 0.0)
    if commute <= 0:
        travel = np.zeros(int(n))
    else:
        sigma = math.sqrt(math.log1p(travel_cv * travel_cv))
        mu = math.log(commute) - 0.5 * sigma * sigma
        travel = rng.lognormal(mu, sigma, int(n))
    disrupted = rng.random(int(n)) < np.clip(disruption_probability, 0.0, 0.80)
    delay = rng.gamma(2.0, max(float(disruption_delay_mean), 0.1) / 2.0, int(n))
    travel = travel + disrupted * delay
    return np.maximum(0.0, travel - commute - float(arrival_buffer))


def choose_robust_strategy(
    belief,
    current_strategy,
    course_scale,
    commute,
    travel_cv,
    arrival_buffer,
    disruption_probability,
    disruption_delay_mean,
    kernel,
    config,
    rng,
):
    """Choose a bounded next-fraction strategy under rate and route uncertainty.

    Candidate policies are compared on three objectives: direct readiness,
    planned patient work, and overfill risk.  A lower-tail readiness penalty
    prevents the controller from optimizing only for the mean.  The result is
    action-capped and subject to imperfect adherence in ``core.py``.
    """

    n = int(config["policy_samples"])
    rate = np.exp(rng.normal(float(belief.mean), math.sqrt(max(belief.variance, 1e-8)), n))
    delay = draw_route_delay(
        rng,
        n,
        commute,
        travel_cv,
        arrival_buffer,
        disruption_probability,
        disruption_delay_mean,
    )
    target = float(config["adaptive_target_probability"])
    tail_p = float(config["adaptive_tail_probability"])
    action_cap = float(config["adaptive_action_cap_min"])
    lower_action = max(float(config["strategy_min_min"]), float(current_strategy) - action_cap)
    upper_action = min(float(config["strategy_max_min"]), float(current_strategy) + action_cap)
    candidates = np.asarray(config["adaptive_candidates_min"], dtype=float)
    candidates = candidates[(candidates >= lower_action) & (candidates <= upper_action)]
    if not len(candidates):
        candidates = np.array([float(np.clip(current_strategy, lower_action, upper_action))])

    rows = []
    for work in candidates:
        state = np.clip(rate * float(course_scale) * (float(work) + delay), 0.0, kernel.capacity)
        ready = np.array([kernel.readiness_probability(value) for value in state])
        mean_ready = float(np.mean(ready))
        tail_ready = float(np.quantile(ready, tail_p))
        overfill = float(np.mean(state > kernel.target_high))
        # A squared target shortfall makes the requested probability q an
        # operative policy choice instead of an additive constant whenever all
        # candidates fall below q.  Mean readiness carries the service target;
        # the lower tail retains a separate robustness target at 75% of q.
        target_shortfall = max(0.0, target - mean_ready)
        tail_shortfall = max(0.0, 0.75 * target - tail_ready)
        cost = (
            float(config["adaptive_work_weight"]) * float(work) / 120.0
            + float(config["adaptive_rework_weight"]) * (1.0 - mean_ready)
            + float(config["adaptive_overfill_weight"]) * overfill
            + 6.0 * target_shortfall**2
            + 3.0 * tail_shortfall**2
        )
        rows.append((cost, -mean_ready, float(work), mean_ready, tail_ready, overfill))
    # Chance-constrained choice: when q is attainable, select the least-burden
    # feasible candidate while retaining lower-tail and overfill penalties.  If
    # q is unattainable under the action cap, minimize the smooth shortfall
    # objective above.  This makes q both interpretable and testable.
    feasible = [row for row in rows if row[3] >= target]
    if feasible:
        feasible.sort(
            key=lambda row: (
                row[2],
                row[5],
                -row[4],
            )
        )
        best = feasible[0]
    else:
        rows.sort()
        best = rows[0]
    return {
        "strategy": best[2],
        "expected_ready": best[3],
        "tail_ready": best[4],
        "overfill_risk": best[5],
        "objective": best[0],
    }
