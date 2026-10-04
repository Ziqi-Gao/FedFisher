"""Bounded analytic/manual fixtures; no default-size simulation is run here."""

import copy
from dataclasses import asdict
from io import BytesIO
import math
import unittest
from unittest.mock import patch

import numpy as np

from effect_modifier_experiment.survival.data import SurvivalRows, SurvivalSite
from effect_modifier_experiment.survival.dslasso import DslassoConfig
from effect_modifier_experiment.survival.pseudo import pseudo_survival
from effect_modifier_experiment.survival.recovery_method import (
    OptimizationFailure, RecoveryPolicy, fit_recovery_dslasso,
)
from effect_modifier_experiment.survival_pooled.methods import (
    _responses, fit_pooled, validate_pooled_fit,
)


def intercept_sites():
    sites = []
    for site_id in range(2):
        n = 16
        rows = SurvivalRows(np.zeros((n, 2)), np.zeros(n),
                            np.tile([1., 2., 4., 5.], 4), np.ones(n),
                            np.arange(site_id * n, (site_id + 1) * n))
        sites.append(SurvivalSite(site_id, rows))
    return sites


def linear_options():
    return dict(optimizer=asdict(DslassoConfig(num_folds=2, num_lambdas=2, max_iter=1000)),
                recovery_policy=asdict(RecoveryPolicy(max_step=64.)))


class SurvivalPooledMethodsTests(unittest.TestCase):
    def test_pseudo_scopes_preserve_unbounded_values_and_change_preprocessing(self):
        rows = [SurvivalRows(np.zeros((3, 2)), np.zeros(3), np.arange(1., 4.),
                            event, np.arange(i * 3, i * 3 + 3))
                for i, event in enumerate(([0, 1, 1], [1, 0, 1]))]
        local = _responses(rows, 2., "site_local_pv")
        pooled = _responses(rows, 2., "pooled_km_pv")
        np.testing.assert_allclose(local[0], [.5, -.25, 1.25])
        expected = pseudo_survival(np.concatenate([r.time for r in rows]),
                                   np.concatenate([r.event for r in rows]), 2.)
        np.testing.assert_array_equal(np.concatenate(pooled), expected)
        self.assertFalse(np.array_equal(np.concatenate(local), expected))

    def test_central_site_labels_match_existing_federated_linear_objective(self):
        sites, method = intercept_sites(), linear_options()
        model, report, arrays = fit_pooled(sites, 3., 0, "linear", "site_local_pv",
                                           method, math.log(.12 * 3 ** 1.5))
        reference, original = fit_recovery_dslasso(sites, 3., seed=0,
            config=DslassoConfig(**method["optimizer"]),
            policy=RecoveryPolicy(**method["recovery_policy"]))
        self.assertEqual(model.intercept, reference.intercept)
        np.testing.assert_array_equal(model.coefficients, reference.coefficients)
        self.assertEqual(report["cv_loss"], original["cv_loss"])
        self.assertEqual(report["selected_lambda_fraction"], original["selected_lambda_fraction"])
        self.assertAlmostEqual(model.intercept, np.log(-np.log(.5)), places=5)
        self.assertEqual(report["communication_uploads"], 0)
        validate_pooled_fit(report, arrays, "linear", "site_local_pv", method, [16, 16])
        altered = copy.deepcopy(report)
        altered["paths"][-1]["fits"][-1]["converged"] = False
        with self.assertRaises(ValueError):
            validate_pooled_fit(altered, arrays, "linear", "site_local_pv", method, [16, 16])

    def test_cv_recomputes_pooled_km_from_disjoint_partitions(self):
        sites, method = intercept_sites(), linear_options()
        calls = []

        def observe(rows, tau, scope):
            calls.append([r.row_ids.copy() for r in rows])
            return _responses(rows, tau, scope)

        with patch("effect_modifier_experiment.survival_pooled.methods._responses", side_effect=observe):
            _, report, arrays = fit_pooled(sites, 3., 0, "linear", "pooled_km_pv", method, 0.)
        self.assertEqual(len(calls), 5)  # train+validation for two folds, then full refit
        all_ids = np.concatenate([site.rows.row_ids for site in sites])
        for index in (0, 2):
            training, validation = np.concatenate(calls[index]), np.concatenate(calls[index + 1])
            self.assertFalse(len(np.intersect1d(training, validation)))
            np.testing.assert_array_equal(np.sort(np.r_[training, validation]), all_ids)
            self.assertLess(len(training), len(all_ids))
            self.assertLess(len(validation), len(all_ids))
        np.testing.assert_array_equal(np.concatenate(calls[-1]), all_ids)
        validate_pooled_fit(report, arrays, "linear", "pooled_km_pv", method, [16, 16])

    def test_linear_nonstationary_fit_fails_closed(self):
        rng = np.random.default_rng(183)
        rows = SurvivalRows(rng.normal(size=(20, 2)), np.tile([0., 1.], 10),
                            np.tile([1., 2., 4., 5.], 5), np.ones(20), np.arange(20))
        method = dict(optimizer=asdict(DslassoConfig(num_folds=2, num_lambdas=2,
                                                    max_iter=1, tolerance=1e-12)),
                      recovery_policy=asdict(RecoveryPolicy(max_step=64.)))
        with self.assertRaises(OptimizationFailure):
            fit_pooled([SurvivalSite(0, rows)], 3., 0, "linear", "site_local_pv", method, 0.)

    def test_one_site_pooled_neural_reproduces_original_local_sgd_without_fisher(self):
        import torch
        from effect_modifier_experiment.survival.fisher import (
            FisherConfig, fit_local_fisher, public_initial_vector,
        )
        torch.set_num_threads(1)
        rng = np.random.default_rng(811)
        rows = SurvivalRows(rng.normal(size=(8, 2)), np.tile([0., 1.], 4),
                            np.tile([1., 2., 4., 5.], 2), np.ones(8), np.arange(8))
        cfg = FisherConfig(architecture="shallow", epochs=2, batch_size=4)
        method = dict(optimizer=asdict(cfg))
        initial_eta = math.log(.12 * 3 ** 1.5)
        with patch("effect_modifier_experiment.survival.fisher.working_fisher",
                   side_effect=AssertionError("pooled training must not compute Fisher")):
            _, report, arrays = fit_pooled([SurvivalSite(0, rows)], 3., 7, "shallow",
                                           "site_local_pv", method, initial_eta)
        initial = public_initial_vector(5, cfg, 7, initial_eta)
        upload, original = fit_local_fisher(rows.design, _responses([rows], 3., "site_local_pv")[0],
            initial_vector=initial, config=cfg, seed=7 + 1009, method="diag")
        with np.load(BytesIO(upload), allow_pickle=False) as archive:
            np.testing.assert_array_equal(arrays["vector"], archive["vector"])
        self.assertEqual(report["initial_loss"], original["initial_loss"])
        self.assertEqual(report["final_loss"], original["final_loss"])
        self.assertEqual(report["optimizer_updates"], 4)
        self.assertEqual(report["processed_rows"], 16)
        self.assertIs(report["fisher_computed"], False)
        validate_pooled_fit(report, arrays, "shallow", "site_local_pv", method, [8])
        altered = copy.deepcopy(report)
        altered["optimizer_updates"] += 1
        with self.assertRaises(ValueError):
            validate_pooled_fit(altered, arrays, "shallow", "site_local_pv", method, [8])


if __name__ == "__main__":
    unittest.main()
