"""Model checks on fictional inputs only."""

import unittest

import numpy as np
import pandas as pd

from examples.synthetic_demo import example_agents, example_config, example_parameters
from protocol_access import simulate_cohort
from protocol_access.core import transform_scenario
from protocol_access.frontier import interval_summary
from protocol_access.state import BladderBandKernel


class ModelTests(unittest.TestCase):
    def test_reproducible_for_same_seed(self):
        args = (example_agents(), example_parameters(), example_config())
        first = simulate_cohort(*args, seed=21)
        second = simulate_cohort(*args, seed=21)
        pd.testing.assert_frame_equal(first, second)

    def test_first_pass_requires_on_time_no_rework_and_readiness(self):
        logs = simulate_cohort(example_agents(), example_parameters(), example_config(), seed=4)
        passed = logs.loc[logs["protocol_ready_first_pass"].eq(1)]
        self.assertTrue(passed["on_time"].eq(1).all())
        self.assertTrue(passed["rework"].eq(0).all())
        self.assertTrue(passed["final_true_ready"].eq(1).all())

    def test_direct_route_preserves_commute(self):
        source = example_agents()
        transformed, _ = transform_scenario(source, "route_direct")
        self.assertTrue(transformed["commute_min_model"].equals(source["commute_min_model"]))
        self.assertTrue(transformed["transfers"].eq(0).all())
        self.assertTrue(transformed["crossings_derived"].eq(0).all())

    def test_target_band_is_nonmonotone(self):
        kernel = BladderBandKernel(0.75, 1.25, 1.0, 1.8, 0.04, 0.05)
        middle = kernel.readiness_probability(1.0)
        self.assertGreater(middle, kernel.readiness_probability(0.5))
        self.assertGreater(middle, kernel.readiness_probability(1.5))

    def test_feasible_set_keeps_gaps(self):
        result = interval_summary(
            np.array([0, 1, 2, 3, 4], dtype=float),
            np.array([0.8, 0.8, 0.5, 0.8, 0.8]),
            0.7,
        )
        self.assertEqual(result["hole_count"], 1)
        self.assertEqual(len(result["all_components"]), 2)
        self.assertLess(result["total_measure"], result["last_boundary"] - result["first_boundary"])


if __name__ == "__main__":
    unittest.main()
