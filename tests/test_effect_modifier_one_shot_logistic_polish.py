import json
import os
import types
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.dslassocov_benchmark import OptimizerConfig
from effect_modifier_experiment.one_shot_lasso_benchmark import (
    _validated_selected_logistic_reports,
)
from effect_modifier_experiment.one_shot_local_logistic import (
    _fit_path,
    _unpenalized_start,
)
from effect_modifier_experiment.run_one_shot_lasso_effect_modifier import (
    _add_stationarity_cell_context,
)


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
        self.sites = [
            types.SimpleNamespace(
                design=np.zeros((2, 3)),
                labels=np.array([0.25, 0.75]),
                site_idx=7,
            )
        ]
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

    def test_unpenalized_false_flag_gets_same_objective_momentum_reset(self):
        primary = np.array([0.2, -0.1])
        polished = primary + np.array([1e-7, -2e-7])
        primary_report = optimizer_report(False, 2e-7, 3e-7, 1e-6, 2000)
        polish_report = optimizer_report(True, 1e-7, 2e-7, 1e-6, 2)
        with mock.patch(
            "effect_modifier_experiment.one_shot_local_logistic.proximal_gradient_fit",
            side_effect=[(primary, primary_report), (polished, polish_report)],
        ) as mocked:
            coefficients, report = _unpenalized_start(
                self.sites,
                penalty_start=2,
                optimizer=self.optimizer,
                polish_max_iter=50,
                stage="full_unpenalized",
            )

        self.assertEqual(mocked.call_count, 2)
        np.testing.assert_array_equal(
            coefficients, np.array([polished[0], polished[1], 0.0])
        )
        primary_call, polish_call = mocked.call_args_list
        self.assertEqual(primary_call.kwargs["lambda_value"], 0.0)
        self.assertEqual(primary_call.kwargs["penalty_start"], 2)
        self.assertEqual(primary_call.args[0][0].design.shape, (2, 2))
        self.assertEqual(polish_call.kwargs["lambda_value"], 0.0)
        self.assertEqual(polish_call.kwargs["penalty_start"], 2)
        self.assertEqual(polish_call.kwargs["config"].max_iter, 50)
        self.assertEqual(
            polish_call.kwargs["config"].tolerance,
            self.optimizer.tolerance,
        )
        self.assertEqual(
            polish_call.kwargs["config"].initial_step,
            self.optimizer.initial_step,
        )
        self.assertEqual(
            polish_call.kwargs["config"].backtrack_factor,
            self.optimizer.backtrack_factor,
        )
        self.assertEqual(
            polish_call.kwargs["config"].min_step,
            self.optimizer.min_step,
        )
        np.testing.assert_array_equal(
            polish_call.kwargs["initial_coefficients"], primary
        )
        self.assertTrue(report["converged"])
        self.assertTrue(report["momentum_reset_polish"]["attempted"])

    def test_genuinely_nonstationary_unpenalized_start_fails_with_context(self):
        primary = np.array([0.2, -0.1])
        primary_report = optimizer_report(False, 2e-5, 3e-5, 1e-6, 2000)
        with mock.patch(
            "effect_modifier_experiment.one_shot_local_logistic.proximal_gradient_fit",
            return_value=(primary, primary_report),
        ) as mocked:
            with self.assertRaisesRegex(
                RuntimeError, "one-shot logistic stationarity failure"
            ) as raised:
                _unpenalized_start(
                    self.sites,
                    penalty_start=2,
                    optimizer=self.optimizer,
                    polish_max_iter=50,
                    stage="fold_unpenalized",
                    fold_idx=3,
                )

        self.assertEqual(mocked.call_count, 1)
        diagnostic = json.loads(str(raised.exception).split(": ", 1)[1])
        self.assertEqual(diagnostic["stage"], "fold_unpenalized")
        self.assertEqual(diagnostic["site_idx"], 7)
        self.assertEqual(diagnostic["fold_idx"], 3)
        self.assertEqual(diagnostic["lambda"], 0.0)
        self.assertFalse(diagnostic["momentum_reset_polish"]["attempted"])
        self.assertEqual(diagnostic["schema_version"], 1)
        self.assertEqual(
            diagnostic["error_type"],
            "one_shot_logistic_stationarity_failure",
        )
        for key in (
            "initial_objective",
            "final_objective",
            "unpenalized_kkt_residual_inf",
            "penalized_kkt_residual_inf",
            "final_step_size",
            "total_backtracks",
            "adaptive_restarts",
        ):
            self.assertIn(key, diagnostic)

        raised.exception.add_context(
            run_id="run-test",
            family="one_shot_local_logistic_lasso_fedavg",
            variant="control",
            seed=150,
            split="noniid",
            model="linear",
            cell="oneshot_seed_000150_noniid_linear",
            scheduler_job_id="job-test",
            scheduler_attempt=2,
        )
        enriched = json.loads(str(raised.exception).split(": ", 1)[1])
        self.assertEqual(enriched["run_id"], "run-test")
        self.assertEqual(enriched["seed"], 150)
        self.assertEqual(
            enriched["cell"], "oneshot_seed_000150_noniid_linear"
        )
        self.assertEqual(enriched["scheduler_job_id"], "job-test")
        self.assertEqual(enriched["scheduler_attempt"], 2)

    def test_runner_adds_cell_and_scheduler_attempt_context(self):
        report = optimizer_report(False, 2e-5, 3e-5, 1e-6, 2000)
        with mock.patch(
            "effect_modifier_experiment.one_shot_local_logistic.proximal_gradient_fit",
            return_value=(np.array([0.2, -0.1]), report),
        ):
            with self.assertRaises(RuntimeError) as raised:
                _unpenalized_start(
                    self.sites,
                    penalty_start=2,
                    optimizer=self.optimizer,
                    stage="full_unpenalized",
                )
        args = types.SimpleNamespace(
            run_id="run-test",
            family="one_shot_local_logistic_lasso_fedavg",
            variant="control",
            seed=150,
            split="noniid",
            model="linear",
        )
        with mock.patch.dict(
            os.environ,
            {
                "SERVER_SCHEDULER_JOB_ID": "job-test",
                "SERVER_SCHEDULER_ATTEMPT": "3",
            },
            clear=False,
        ):
            _add_stationarity_cell_context(
                raised.exception,
                args,
                "oneshot_seed_000150_noniid_linear",
            )

        diagnostic = json.loads(str(raised.exception).split(": ", 1)[1])
        self.assertEqual(
            diagnostic["cell"], "oneshot_seed_000150_noniid_linear"
        )
        self.assertEqual(diagnostic["scheduler_job_id"], "job-test")
        self.assertEqual(diagnostic["scheduler_attempt"], 3)

    def test_unpenalized_start_still_fails_when_polish_is_nonstationary(self):
        primary = np.array([0.2, -0.1])
        polished = primary + np.array([2e-5, -3e-5])
        primary_report = optimizer_report(False, 2e-7, 3e-7, 1e-6, 2000)
        polish_report = optimizer_report(False, 2e-5, 3e-5, 1e-6, 50)
        with mock.patch(
            "effect_modifier_experiment.one_shot_local_logistic.proximal_gradient_fit",
            side_effect=[(primary, primary_report), (polished, polish_report)],
        ):
            with self.assertRaises(RuntimeError) as raised:
                _unpenalized_start(
                    self.sites,
                    penalty_start=2,
                    optimizer=self.optimizer,
                    polish_max_iter=50,
                    stage="full_unpenalized",
                )

        diagnostic = json.loads(str(raised.exception).split(": ", 1)[1])
        polish = diagnostic["momentum_reset_polish"]
        self.assertTrue(polish["attempted"])
        self.assertFalse(polish["post_final_stationarity_passed"])
        self.assertFalse(diagnostic["converged"])

    def test_converged_unpenalized_start_does_not_refit(self):
        primary = np.array([0.2, -0.1])
        primary_report = optimizer_report(True, 2e-7, 3e-7, 1e-6, 5)
        with mock.patch(
            "effect_modifier_experiment.one_shot_local_logistic.proximal_gradient_fit",
            return_value=(primary, primary_report),
        ) as mocked:
            coefficients, report = _unpenalized_start(
                self.sites,
                penalty_start=2,
                optimizer=self.optimizer,
                polish_max_iter=50,
            )

        self.assertEqual(mocked.call_count, 1)
        np.testing.assert_array_equal(coefficients, np.array([0.2, -0.1, 0.0]))
        self.assertFalse(report["momentum_reset_polish"]["attempted"])

    def test_selected_fold_failure_keeps_original_cv_index(self):
        converged = optimizer_report(True, 1e-7, 2e-7, 1e-6, 3)
        failed = optimizer_report(False, 2e-5, 3e-5, 1e-6, 2000)
        fit = {
            "cv": {
                "selected_index": 1,
                "selected_lambda": 0.25,
                "fold_reports": [
                    {
                        "fold_idx": 4,
                        "lambda_fits": [converged, failed],
                    }
                ],
            },
            "optimizer": {"selected_lambda_fit": converged},
        }

        with self.assertRaises(RuntimeError) as raised:
            _validated_selected_logistic_reports(fit, site_idx=9)
        diagnostic = json.loads(str(raised.exception).split(": ", 1)[1])
        self.assertEqual(diagnostic["stage"], "fold_lambda")
        self.assertEqual(diagnostic["site_idx"], 9)
        self.assertEqual(diagnostic["fold_idx"], 4)
        self.assertEqual(diagnostic["lambda_index"], 1)
        self.assertEqual(diagnostic["lambda"], 0.25)

    def test_selected_final_failure_has_final_stage_context(self):
        converged = optimizer_report(True, 1e-7, 2e-7, 1e-6, 3)
        failed = optimizer_report(False, 2e-5, 3e-5, 1e-6, 2000)
        fit = {
            "cv": {
                "selected_index": 0,
                "selected_lambda": 0.5,
                "fold_reports": [
                    {"fold_idx": 0, "lambda_fits": [converged]},
                ],
            },
            "optimizer": {"selected_lambda_fit": failed},
        }

        with self.assertRaises(RuntimeError) as raised:
            _validated_selected_logistic_reports(fit, site_idx=11)
        diagnostic = json.loads(str(raised.exception).split(": ", 1)[1])
        self.assertEqual(diagnostic["stage"], "final_lambda")
        self.assertEqual(diagnostic["site_idx"], 11)
        self.assertIsNone(diagnostic["fold_idx"])
        self.assertEqual(diagnostic["lambda_index"], 0)


if __name__ == "__main__":
    unittest.main()
