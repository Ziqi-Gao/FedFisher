import types
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.dslassocov_benchmark import OptimizerConfig
from effect_modifier_experiment.one_shot_local_logistic import _fit_path


def optimizer_report(converged, proximal, kkt, threshold, iterations):
    return {
        "iterations": iterations,
        "converged": converged,
        "initial_objective": 1.0,
        "final_objective": 0.5,
        "max_coefficient_update": 1e-4 if not converged else proximal,
        "proximal_gradient_mapping_inf": proximal,
        "kkt_residual_inf": kkt,
        "unpenalized_kkt_residual_inf": kkt,
        "penalized_kkt_residual_inf": kkt,
        "stationarity_threshold": threshold,
        "final_step_size": 1.0,
        "backtracking_loss_rounds": iterations + 1,
        "gradient_rounds": iterations + 1,
        "total_backtracks": 0,
        "adaptive_restarts": 0,
        "site_level_messages_per_gradient_round": 1,
    }


class OneShotLogisticPolishTests(unittest.TestCase):
    def setUp(self):
        self.sites = [types.SimpleNamespace(design=np.zeros((2, 3)))]
        self.optimizer = OptimizerConfig(
            max_iter=2000,
            tolerance=1e-6,
            initial_step=1.0,
            backtrack_factor=0.5,
            min_step=1e-12,
        )

    def test_stationary_legacy_false_negative_gets_one_momentum_reset(self):
        primary = np.array([0.0, 0.0, 1.0])
        polished = primary + np.array([1e-7, -2e-7, 3e-7])
        primary_report = optimizer_report(False, 2e-7, 3e-7, 1e-6, 2000)
        polish_report = optimizer_report(True, 1e-7, 2e-7, 1e-6, 1)
        with mock.patch(
            "effect_modifier_experiment.one_shot_local_logistic.proximal_gradient_fit",
            side_effect=[(primary, primary_report), (polished, polish_report)],
        ) as mocked:
            coefficients, reports = _fit_path(
                self.sites,
                penalty_start=2,
                lambda_values=np.array([0.1]),
                initial_coefficients=np.zeros(3),
                optimizer=self.optimizer,
                polish_max_iter=50,
            )
        self.assertEqual(mocked.call_count, 2)
        np.testing.assert_array_equal(coefficients[0], polished)
        report = reports[0]
        self.assertTrue(report["converged"])
        self.assertEqual(report["iterations"], 2001)
        audit = report["momentum_reset_polish"]
        self.assertTrue(audit["attempted"])
        self.assertFalse(audit["trigger_legacy_converged"])
        self.assertTrue(audit["trigger_final_stationarity_passed"])
        self.assertTrue(audit["post_converged"])
        self.assertEqual(audit["post_iterations"], 1)
        self.assertAlmostEqual(audit["coefficient_max_drift"], 3e-7)

    def test_genuinely_nonstationary_fit_is_not_reclassified(self):
        primary = np.array([0.0, 0.0, 1.0])
        primary_report = optimizer_report(False, 2e-5, 3e-5, 1e-6, 2000)
        with mock.patch(
            "effect_modifier_experiment.one_shot_local_logistic.proximal_gradient_fit",
            return_value=(primary, primary_report),
        ) as mocked:
            coefficients, reports = _fit_path(
                self.sites,
                penalty_start=2,
                lambda_values=np.array([0.1]),
                initial_coefficients=np.zeros(3),
                optimizer=self.optimizer,
                polish_max_iter=50,
            )
        self.assertEqual(mocked.call_count, 1)
        np.testing.assert_array_equal(coefficients[0], primary)
        self.assertFalse(reports[0]["converged"])
        audit = reports[0]["momentum_reset_polish"]
        self.assertFalse(audit["attempted"])
        self.assertFalse(audit["trigger_final_stationarity_passed"])


if __name__ == "__main__":
    unittest.main()
