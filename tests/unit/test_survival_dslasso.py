import copy
import unittest

import numpy as np

from effect_modifier_experiment.survival.data import SurvivalRows, SurvivalSite
from effect_modifier_experiment.survival.dslasso import (
    DslassoConfig, SurvivalLinearModel, aggregate_loss_gradient, fit_dslasso,
    proximal_fit, validate_fit_diagnostics,
)


class SurvivalDslassoTests(unittest.TestCase):
    def test_gradient_and_site_weighting_with_out_of_range_responses(self):
        x = np.array([[1., -1.], [1., .2], [1., .7], [1., 1.5]])
        y = np.array([-.2, .5, 1.2, .1])
        beta = np.array([-.3, .2])
        sites = [(x[:1], y[:1]), (x[1:], y[1:])]
        loss, gradient = aggregate_loss_gradient(sites, beta)
        pooled, pooled_gradient = aggregate_loss_gradient([(x, y)], beta)
        self.assertAlmostEqual(loss, pooled)
        np.testing.assert_allclose(gradient, pooled_gradient, atol=1e-15)
        difference = []
        for j in range(2):
            offset = np.zeros(2); offset[j] = 1e-6
            difference.append((aggregate_loss_gradient(sites, beta + offset)[0] -
                               aggregate_loss_gradient(sites, beta - offset)[0]) / 2e-6)
        np.testing.assert_allclose(gradient, difference, rtol=1e-7, atol=1e-9)

    def test_no_censoring_intercept_solution_and_stationarity(self):
        sites = [(np.ones((4, 1)), np.array([0., 0., 1., 1.]))]
        beta, evidence = proximal_fit(sites, np.zeros(1), 1, 0.,
                                      DslassoConfig(max_iter=1000, tolerance=1e-8))
        self.assertTrue(evidence["converged"])
        self.assertAlmostEqual(beta[0], np.log(-np.log(.5)), places=5)
        self.assertLessEqual(evidence["kkt_residual_inf"], evidence["stationarity_threshold"])

    def test_budget_exhaustion_is_not_convergence(self):
        sites = [(np.ones((4, 1)), np.array([0., 0., 1., 1.]))]
        _, evidence = proximal_fit(sites, np.array([1.]), 1, 0.,
                                   DslassoConfig(max_iter=1, tolerance=1e-12))
        self.assertFalse(evidence["converged"])

    def test_tiny_cv_and_saved_selection(self):
        # Identical features isolate the intercept, with fixed uncensored times.
        sites = []
        for site_id in range(2):
            n = 16
            rows = SurvivalRows(np.zeros((n, 2)), np.zeros(n),
                                np.tile([1., 2., 4., 5.], 4), np.ones(n),
                                np.arange(site_id * n, (site_id + 1) * n))
            sites.append(SurvivalSite(site_id, rows))
        cfg = DslassoConfig(num_folds=2, num_lambdas=2, max_iter=1000)
        model, diagnostics = fit_dslasso(sites, 3., config=cfg)
        arrays = dict(intercept=np.array(model.intercept), coefficients=model.coefficients)
        validate_fit_diagnostics(diagnostics, cfg, arrays, [16, 16])
        self.assertEqual(diagnostics["selected_lambda_index"], 0)
        self.assertAlmostEqual(model.intercept, np.log(-np.log(.5)), places=5)
        self.assertFalse(diagnostics["communication"]["one_shot"])
        bad = copy.deepcopy(diagnostics)
        bad["final_fit"]["converged"] = False
        with self.assertRaises(ValueError):
            validate_fit_diagnostics(bad, cfg, arrays, [16, 16])

    def test_prediction_requires_continuous_single_score(self):
        model = SurvivalLinearModel(.3, np.array([.1, .2]))
        np.testing.assert_allclose(model.predict_eta(np.array([[1., 2.]])), [.8])
        with self.assertRaises(ValueError):
            model.predict_eta(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
