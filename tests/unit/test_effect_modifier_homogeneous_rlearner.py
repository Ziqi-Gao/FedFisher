"""Small numerical and leakage fixtures; never an experiment/pilot launcher."""

from dataclasses import replace
from copy import deepcopy
import json
import math
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.causal_comparison import rlearner as r
from effect_modifier_experiment.dslassocov_benchmark import RawSite


class HomogeneousRlearnerTests(unittest.TestCase):
    def setUp(self):
        self.config = r.RlearnerConfig(
            crossfit_folds=2, nuisance_cv_folds=2,
            nuisance_lambda_count=2, nuisance_lambda_max=0.15,
            nuisance_lambda_min_ratio=0.5, nuisance_max_iter=1500,
            nuisance_tolerance=1e-5, effect_lambda_count=8,
        )

    @staticmethod
    def site(n, idx=0):
        rng = np.random.default_rng(173 + idx)
        x = rng.normal(size=(n, 2))
        a = (np.arange(n) % 2).astype(np.float64)
        y = ((np.arange(n) // 2) % 2).astype(np.float64)
        return r.CausalSite(x, a, y, idx)

    @staticmethod
    def raw(site):
        return RawSite(
            np.column_stack((site.treatment, site.covariates, site.treatment[:, None] * site.covariates)),
            site.outcome, site.site_idx,
        )

    def test_summary_matches_weighted_pooled_residual_objective(self):
        summaries, residual_data = [], []
        for n, idx in ((19, 0), (31, 1)):
            site = self.site(n, idx)
            m = 1 / (1 + np.exp(-0.2 * site.covariates[:, 0]))
            e = np.full(n, 0.45)
            summary, report = r.build_site_summary(site, m, e, self.config)
            z = (site.treatment - e)[:, None] * np.column_stack((np.ones(n), site.covariates))
            y = site.outcome - m
            variance = np.square(y - z.dot(summary.beta)).sum() / n
            np.testing.assert_allclose(summary.precision, z.T.dot(z) / variance, rtol=1e-12)
            self.assertEqual(report["payload_bytes"], 16 + 8 * (3 + 3 * 3))
            summaries.append(summary)
            residual_data.append((z, y, variance))
        h = sum(z.T.dot(z) / variance for z, y, variance in residual_data)
        rhs = sum(z.T.dot(y) / variance for z, y, variance in residual_data)
        expected = np.linalg.solve(h, rhs)
        model, _ = r.fit_shared_effect(summaries, "none", self.config)
        np.testing.assert_allclose(model.coefficients, expected, atol=1e-12)
        beta_a, beta_b = np.array([0.2, -0.3, 0.7]), np.array([-0.1, 0.9, 0.05])

        def pooled_objective(beta):
            return sum(np.square(y - z.dot(beta)).sum() / variance for z, y, variance in residual_data) / 100

        self.assertAlmostEqual(
            r.summary_cd_loss(summaries, beta_a) - r.summary_cd_loss(summaries, beta_b),
            pooled_objective(beta_a) - pooled_objective(beta_b), places=11,
        )

    def test_zero_penalty_uses_precision_without_double_sample_weighting(self):
        p1 = np.array([[5, 1, 0], [1, 3, 0.2], [0, 0.2, 2]], dtype=np.float64)
        p2 = np.diag([3.0, 8.0, 4.0])
        beta1, beta2 = np.array([1.0, -2.0, 0.5]), np.array([0.0, 1.0, -0.5])
        summaries = [r.SiteSummary(0, 20, beta1, p1), r.SiteSummary(1, 80, beta2, p2)]
        expected = np.linalg.solve(p1 + p2, p1.dot(beta1) + p2.dot(beta2))
        wrong = np.linalg.solve(20 * p1 + 80 * p2, 20 * p1.dot(beta1) + 80 * p2.dot(beta2))
        model, report = r.fit_shared_effect(summaries, "none", self.config)
        np.testing.assert_allclose(model.coefficients, expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(expected - wrong), 0.2)
        self.assertEqual(report["selected_lambda"], 0.0)
        self.assertEqual(report["selected_fit"]["iterations"], 0)

    def test_lasso_soft_threshold_and_unpenalized_intercept(self):
        h = np.diag([2.0, 4.0, 1.0])
        b = np.array([2.4, 3.0, 0.1])
        coefficients, report = r.solve_shared_quadratic(h, b, 0.5, np.array([0.0, 1.0, 1.0]), self.config)
        np.testing.assert_allclose(coefficients, [1.2, 0.625, 0.0], atol=1e-12)
        self.assertTrue(report["converged"])
        self.assertLessEqual(report["kkt_residual_inf"], report["stationarity_threshold"])

    def test_adaptive_ebic_counts_one_shared_intercept_and_active_slopes(self):
        summaries = [
            r.SiteSummary(0, 400, np.array([1.2, 0.6, 1e-5]), np.eye(3) * 400),
            r.SiteSummary(1, 600, np.array([1.2, 0.6, 1e-5]), np.eye(3) * 600),
        ]
        model, report = r.fit_shared_effect(summaries, "adaptive_lasso", self.config)
        self.assertAlmostEqual(model.coefficients[0], 1.2)
        self.assertEqual(model.coefficients[2], 0.0)
        self.assertGreater(model.coefficients[1], 0.5)
        chosen = report["selected_fit"]
        self.assertEqual(chosen["degrees_of_freedom"], 2)
        self.assertAlmostEqual(chosen["ebic"], chosen["cd_discrepancy"] + 2 * math.log(1000) + 2 * math.log(2))
        self.assertTrue(all(item["converged"] for item in report["lambda_fits"]))

    def test_actual_crossfit_prediction_does_not_use_heldout_outcomes(self):
        site, seed = self.site(40), 112
        m1, e1, _ = r.crossfit_site_nuisances(site, self.config, seed)
        heldout = r.deterministic_folds(40, 2, seed) == 0
        changed_y = site.outcome.copy()
        changed_y[heldout] = 1 - changed_y[heldout]
        changed = r.CausalSite(site.covariates, site.treatment, changed_y, site.site_idx)
        m2, e2, _ = r.crossfit_site_nuisances(changed, self.config, seed)
        # This exercises the real inner-CV selection and refit, not a fake fit.
        np.testing.assert_array_equal(m1[heldout], m2[heldout])
        np.testing.assert_array_equal(e1, e2)

    def test_outer_training_boundary_hides_heldout_labels_from_selector(self):
        site, seed = self.site(24), 92
        folds = r.deterministic_folds(24, 2, seed)
        seen = []

        def fitted(train_x, target, predict_x, config, inner_seed, stages):
            fold = len(seen) // 2
            training = folds != fold
            np.testing.assert_array_equal(train_x, site.covariates[training])
            np.testing.assert_array_equal(predict_x, site.covariates[~training])
            expected_target = site.outcome if len(seen) % 2 == 0 else site.treatment
            np.testing.assert_array_equal(target, expected_target[training])
            seen.append((train_x.shape[0], target.shape[0]))
            return np.full(predict_x.shape[0], 0.5), {"converged": True}

        with mock.patch.object(r, "_fit_nuisance_cv", side_effect=fitted):
            r.crossfit_site_nuisances(site, self.config, seed)
        self.assertEqual(len(seen), 4)

    def test_inner_standardization_fits_only_its_training_rows(self):
        site, seed = self.site(24), 17
        splits = r.deterministic_folds(24, 2, seed)
        original = r._local_design
        fitted = []

        def local_design(x, target):
            fitted.append(x.copy())
            return original(x, target)

        with mock.patch.object(r, "_local_design", side_effect=local_design):
            r._fit_nuisance_cv(site.covariates, site.outcome, np.ones((3, 2)) * 9999, self.config, seed, {})
        self.assertEqual(len(fitted), 3)
        np.testing.assert_array_equal(fitted[0], site.covariates[splits != 0])
        np.testing.assert_array_equal(fitted[1], site.covariates[splits != 1])
        np.testing.assert_array_equal(fitted[2], site.covariates)

    def test_constant_binary_response_is_train_only_and_explicit(self):
        site = self.site(20)
        constant = r.CausalSite(site.covariates, site.treatment, np.ones(20), 0)
        m, e, report = r.crossfit_site_nuisances(constant, self.config, 81)
        np.testing.assert_allclose(m, (10 + 0.5) / 11)
        self.assertTrue(np.isfinite(e).all())
        self.assertTrue(all(fold["outcome"]["fit_kind"] == "train_only_smoothed_bernoulli_constant" for fold in report["outer_folds"]))

    def test_effect_prediction_is_unclipped_probability_scale(self):
        model = r.LinearProbabilityEffectModel(np.array([0.1, 2.0, -1.0]))
        np.testing.assert_array_equal(model.predict_tau(np.array([[1.0, 0.0], [0.0, 1.0]])), [2.1, -0.9])
        with self.assertRaises(ValueError):
            model.predict_tau(np.zeros((2, 5)))

    def test_full_fit_has_json_metadata_and_one_summary_upload_per_site(self):
        raw_sites = [self.raw(self.site(32, 0)), self.raw(self.site(40, 1))]
        ledger = mock.Mock()
        model, report = r.fit_homogeneous_rlearner(raw_sites, 2, "none", self.config, 55, communication=ledger)
        json.dumps(report, allow_nan=False)
        r.validate_fit_report(report, self.config, "none", model.coefficients)
        self.assertEqual(model.coefficients.size, 3)
        self.assertEqual(report["communication"]["synchronization_rounds"], 1)
        self.assertEqual(report["communication"]["upload_messages"], 2)
        self.assertEqual(report["communication"]["upload_bytes"], 2 * (16 + 8 * 12))
        self.assertEqual(ledger.add.call_count, 2)
        self.assertEqual({c.kwargs["round_id"] for c in ledger.add.call_args_list}, {"r_summary_upload"})
        self.assertFalse(report["server_uses_individual_rows"])
        self.assertGreater(report["full_fit_wall_seconds"], 0)
        self.assertLessEqual(sum(s["wall_seconds"] for s in report["stage_timing"].values()), report["full_fit_wall_seconds"])

    def test_semantic_validation_rejects_tampered_successful_fit_evidence(self):
        sites = [self.raw(self.site(32, 0)), self.raw(self.site(40, 1))]
        model, report = r.fit_homogeneous_rlearner(sites, 2, "adaptive_lasso", self.config, 55)
        r.validate_fit_report(report, report["configuration"], "adaptive_lasso", model.coefficients)

        def change(mapping, path, value):
            changed = deepcopy(mapping)
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            return changed

        edits = [
            (("configuration", "propensity_clip"), 0.2),
            (("effect_penalty",), "none"),
            (("converged",), False),
            (("shared_effect_coefficients", 0), 99.0),
            (("adaptive_weights", 0), 1.0),
            (("lambda_fits", 0, "converged"), False),
            (("lambda_fits", 0, "kkt_residual_inf"), 99.0),
            (("lambda_fits", 0, "degrees_of_freedom"), 999),
            (("local_reports", 0, "nuisance", "outer_folds", 0, "outcome", "max_stationarity_ratio"), 2.0),
            (("local_reports", 0, "nuisance", "test_rows_used"), 1),
            (("local_reports", 0, "summary", "residual_variance"), 0.0),
            (("communication", "synchronization_rounds"), 2),
            (("full_fit_wall_seconds",), 0.0),
        ]
        for path, value in edits:
            with self.subTest(path=path), self.assertRaises(ValueError):
                r.validate_fit_report(change(report, path, value), self.config, "adaptive_lasso", model.coefficients)

    def test_degenerate_inputs_fail_without_ridge_or_pseudoinverse_fallback(self):
        site = self.site(20)
        bad_overlap = replace(site, treatment=np.zeros(20))
        with self.assertRaisesRegex(ValueError, "overlap"):
            r.crossfit_site_nuisances(bad_overlap, self.config, 0)
        singular = replace(site, covariates=np.ones((20, 2)))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            r.build_site_summary(singular, np.full(20, 0.5), np.full(20, 0.5), self.config)
        with self.assertRaisesRegex(ValueError, "variance is degenerate"):
            r.build_site_summary(site, site.outcome.copy(), np.full(20, 0.5), self.config)
        with self.assertRaisesRegex(ValueError, "more observations"):
            r.build_site_summary(self.site(3), np.full(3, 0.5), np.full(3, 0.5), self.config)
        with self.assertRaisesRegex(ValueError, "SiteSummary"):
            r.fit_shared_effect([site], "none", self.config)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            r.fit_shared_effect([r.SiteSummary(0, 20, np.ones(3), np.zeros((3, 3)))], "none", self.config)

    def test_nonconverged_optimizers_fail_closed(self):
        site = self.site(40)
        with self.assertRaisesRegex(RuntimeError, "stationarity"):
            r.crossfit_site_nuisances(site, replace(self.config, nuisance_max_iter=1), 27)
        with self.assertRaisesRegex(RuntimeError, "iteration limit"):
            r.solve_shared_quadratic(
                np.array([[2.0, 1.0], [1.0, 2.0]]), np.ones(2),
                0.01, np.array([0.0, 1.0]), replace(self.config, effect_max_iter=1),
            )

    def test_raw_interaction_inconsistency_and_invalid_configuration_rejected(self):
        raw = self.raw(self.site(20))
        changed = raw.features.copy()
        changed[0, -1] = 99
        with self.assertRaisesRegex(ValueError, "interactions"):
            r.raw_sites_to_causal_sites([RawSite(changed, raw.labels, 0)], 2)
        for config in (
            replace(self.config, crossfit_folds=1),
            replace(self.config, propensity_clip=0.5),
            replace(self.config, effect_tolerance=float("nan")),
            replace(self.config, condition_limit=float("inf")),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                r.fit_shared_effect([], "none", config)


if __name__ == "__main__":
    unittest.main()
