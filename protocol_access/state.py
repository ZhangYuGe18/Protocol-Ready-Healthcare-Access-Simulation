from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class CorrectionPlan:
    action: str
    minutes: float
    reset_state: float | None = None


@dataclass
class BladderBandKernel:
    """Normalized accumulative bladder state with an upper and lower target.

    The state is relative to the planning target rather than an absolute volume.
    Course drift is supplied as a multiplicative episode input and never uses a
    mobility variable.  The same kernel is used by calibration, scenarios and
    spatial frontiers.
    """

    target_low: float
    target_high: float
    target_center: float
    capacity: float
    readiness_softness: float
    reset_state: float

    def advance(self, state, minutes, rate):
        return float(np.clip(float(state) + max(0.0, float(minutes)) * float(rate), 0.0, self.capacity))

    def readiness_probability(self, state):
        softness = max(float(self.readiness_softness), 1e-6)
        lower = expit((float(state) - self.target_low) / softness)
        upper = expit((self.target_high - float(state)) / softness)
        return float(lower * upper)

    def band_status(self, state):
        if float(state) < self.target_low:
            return "underfilled"
        if float(state) > self.target_high:
            return "overfilled"
        return "in_band"

    def correction(self, state, rate):
        rate = max(float(rate), 1e-6)
        if float(state) > self.target_high:
            minutes = max(0.0, (self.target_center - self.reset_state) / rate)
            return CorrectionPlan("void_and_refill", minutes, self.reset_state)
        minutes = max(0.0, (self.target_center - float(state)) / rate)
        return CorrectionPlan("continue_filling", minutes, None)


@dataclass
class ObservationGate:
    coverage: float
    sensitivity: float
    specificity: float

    def inspect_truth(self, true_ready, rng):
        if rng.random() >= float(self.coverage):
            return {"inspected": False, "inadequate": False}
        inadequate_probability = 1.0 - float(self.specificity) if true_ready else float(self.sensitivity)
        return {"inspected": True, "inadequate": bool(rng.random() < inadequate_probability)}
