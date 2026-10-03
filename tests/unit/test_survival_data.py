"""Tiny deterministic data-contract fixtures, never default-sized simulations."""

from dataclasses import fields, replace
import unittest

import numpy as np

from effect_modifier_experiment.survival.data import (
    DATASET_ID, SurvivalConfig, SurvivalRows, generate_survival,
    site_gaussian_parameters, survival_truth,
)


class SurvivalDataTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(SurvivalConfig(num_sites=3, site_min_samples=9, site_max_samples=14,
                                      covariate_dim=5, num_test=11), **kwargs)

    def test_default_contract_and_normalized_original_coefficients(self):
        defaults = SurvivalConfig()
        self.assertEqual((defaults.num_sites, defaults.site_min_samples, defaults.site_max_samples,
                          defaults.covariate_dim), (10, 500, 1000, 100))
        self.assertEqual((defaults.weibull_shape, defaults.weibull_lambda, defaults.censor_rate,
                          defaults.tau), (1.5, 0.12, 0.05, 3.0))
        cfg = self.config()
        x = np.eye(cfg.covariate_dim)
        truth = survival_truth(x, cfg)
        np.testing.assert_allclose(truth.delta, [np.sqrt(2), -np.sqrt(2), 0, 0, 0])
        baseline = np.log(0.12) + 1.5 * np.log(3)
        np.testing.assert_allclose(truth.eta0 - baseline, [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0], atol=5e-16)
        np.testing.assert_array_equal(truth.modifier_mask, [True, True, False, False, False])
        self.assertTrue(np.all(truth.propensity[:2] > 0.5))
        np.testing.assert_array_equal(truth.propensity[2:], [0.5, 0.5, 0.5])

    def test_truth_obeys_ph_and_noise_exclusion(self):
        cfg = self.config()
        x = np.array([[0.2, -0.3, 9, 8, 7], [-0.1, 0.4, 3, 2, 1]], dtype=float)
        first = survival_truth(x, cfg)
        x[:, 2:] = -1000
        second = survival_truth(x, cfg)
        for field in fields(first):
            np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name))
        np.testing.assert_allclose(np.log(-np.log(first.survival1))
                                   - np.log(-np.log(first.survival0)), first.delta)
        alternate_tau = survival_truth(x, replace(cfg, tau=2.0))
        np.testing.assert_allclose(alternate_tau.delta, first.delta)
        np.testing.assert_allclose(alternate_tau.eta0 - first.eta0, 1.5 * np.log(2 / 3))
        at_zero = survival_truth(np.zeros((1, 5)), cfg)
        np.testing.assert_array_equal(at_zero.delta, [0])  # No treatment main effect.

    def test_site_first_shift_preserves_sizes_and_common_test(self):
        cfg = self.config()
        iid = generate_survival(cfg, 13)
        shifted = generate_survival(cfg, 13, "covariate_shift")
        self.assertEqual(iid.config.dataset_id, DATASET_ID)
        for base, other, (mean, std) in zip(iid.sites, shifted.sites, site_gaussian_parameters(cfg)):
            self.assertTrue(9 <= len(base.rows) <= 14)
            self.assertEqual(len(base.rows), len(other.rows))
            np.testing.assert_allclose(other.rows.x, mean + std * base.rows.x, atol=1e-15)
            np.testing.assert_array_equal(base.rows.row_ids, other.rows.row_ids)
        for field in fields(SurvivalRows):
            np.testing.assert_array_equal(getattr(iid.evaluation, field.name),
                                          getattr(shifted.evaluation, field.name))
        sizes = np.array([len(s.rows) for s in iid.sites])
        np.testing.assert_array_equal(iid.site_weights, sizes / sizes.sum())

    def test_outcome_parameters_do_not_change_sites_covariates_or_treatment(self):
        base = generate_survival(self.config(), 17)
        changed = generate_survival(self.config(weibull_shape=2.0, weibull_lambda=0.4,
                                               censor_rate=0.2, signal_strength=0.5), 17)
        for first, second in zip(base.sites, changed.sites):
            np.testing.assert_array_equal(first.rows.row_ids, second.rows.row_ids)
            np.testing.assert_array_equal(first.rows.x, second.rows.x)
            np.testing.assert_array_equal(first.rows.a, second.rows.a)
            self.assertFalse(np.array_equal(first.rows.time, second.rows.time))

    def test_independent_test_and_reproducibility(self):
        base = generate_survival(self.config(), 19)
        repeated = generate_survival(self.config(), 19)
        more_test = generate_survival(self.config(num_test=15), 19)
        for first, second, third in zip(base.sites, repeated.sites, more_test.sites):
            for field in fields(SurvivalRows):
                np.testing.assert_array_equal(getattr(first.rows, field.name), getattr(second.rows, field.name))
                np.testing.assert_array_equal(getattr(first.rows, field.name), getattr(third.rows, field.name))
        training_ids = np.concatenate([site.rows.row_ids for site in base.sites])
        self.assertEqual(len(np.unique(training_ids)), len(training_ids))
        self.assertFalse(np.isin(base.evaluation.row_ids, training_ids).any())
        self.assertEqual({field.name for field in fields(SurvivalRows)},
                         {"x", "a", "time", "event", "row_ids"})

    def test_observed_rows_design_validation_and_immutability(self):
        source = np.arange(6, dtype=float).reshape(3, 2)
        rows = SurvivalRows(source, [0, 1, 0], [1, 2, 3], [1, 0, 1], np.arange(3))
        source[:] = 99
        np.testing.assert_array_equal(rows.design[:, 1:3], np.arange(6).reshape(3, 2))
        np.testing.assert_array_equal(rows.design[:, 3:], [[0, 0], [2, 3], [0, 0]])
        with self.assertRaises(ValueError):
            rows.x[0, 0] = 2
        np.testing.assert_array_equal(rows.subset([2, 0]).row_ids, [2, 0])
        for changes in ({"a": [0, 2, 1]}, {"event": [0, -1, 1]}, {"time": [1, np.nan, 3]},
                        {"row_ids": [0, 0, 2]}, {"row_ids": [0.0, 1.0, 2.0]}, {"x": np.ones((3, 1))}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(rows, **changes)

    def test_invalid_config_and_scenario_rejected(self):
        for changes in ({"num_sites": True}, {"site_max_samples": 2}, {"censor_rate": 0},
                        {"tau": np.nan}, {"weibull_shape": -1}, {"propensity_clip": 0.5},
                        {"dataset_id": "SyntheticEffectModifier"}, {"covariate_dim": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        with self.assertRaises(ValueError):
            generate_survival(self.config(), 0, "noniid")
        with self.assertRaises(ValueError):
            generate_survival(self.config(), -1)


if __name__ == "__main__":
    unittest.main()
