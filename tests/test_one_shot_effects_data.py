"""Small data/isolation contracts; no fitting, inference or scientific pilots."""

from contextlib import redirect_stdout
from dataclasses import asdict, fields, replace
import io
from types import SimpleNamespace
import unittest

import numpy as np

from effect_modifier_experiment.one_shot_effects.data import (
    DATASET_ID, SCENARIOS, LegacyConfig, SCMConfig, fixed_site_means,
    generate_legacy, generate_scm, partition_fingerprint, partition_rows,
    rows_fingerprint,
)
from effect_modifier_experiment.one_shot_effects.types import Rows, SitePartitions


def observed(n=20, d=5):
    x = np.arange(n * d, dtype=float).reshape(n, d) / 10
    return Rows(x, np.arange(n) % 2, np.arange(n) % 3 == 0, np.arange(n, dtype=np.int64))


def original_order(site):
    blocks = (site.train, site.validation, site.holdout)
    ids = np.concatenate([b.row_ids for b in blocks])
    order = np.argsort(ids)
    return Rows(*(np.concatenate([getattr(b, name) for b in blocks])[order]
                  for name in ("x", "a", "y", "row_ids")))


class ObservedRowsTests(unittest.TestCase):
    def test_strict_shapes_labels_identifiers_and_copies(self):
        source = np.ones((8, 5))
        rows = Rows(source, np.zeros(8), np.ones(8), np.arange(8))
        source[:] = 99
        np.testing.assert_array_equal(rows.x, np.ones((8, 5)))
        self.assertEqual((len(rows), rows.n_samples, rows.n_features), (8, 8, 5))
        with self.assertRaises(ValueError):
            rows.x[0, 0] = 1
        for changes in ({"a": np.zeros((8, 1))}, {"y": np.full(8, 2)},
                        {"x": np.full((8, 5), np.nan)}, {"row_ids": np.zeros(8, dtype=int)},
                        {"row_ids": np.arange(8, dtype=float)}, {"y": np.zeros(7)}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                replace(rows, **changes)
        with self.assertRaises(TypeError):
            Rows(source, np.zeros(8), np.ones(8), np.arange(8), truth={})

    def test_partition_is_disjoint_exhaustive_and_independent_of_values(self):
        for n in (7, 21, 100):
            first = observed(n)
            changed = Rows(np.zeros_like(first.x), 1-first.a, 1-first.y, first.row_ids)
            a, b = partition_rows(first, 2, 7), partition_rows(changed, 2, 7)
            self.assertEqual((len(a.train), len(a.validation), len(a.holdout)),
                             (70*n//100, 15*n//100, n-70*n//100-15*n//100))
            self.assertEqual(partition_fingerprint(a), partition_fingerprint(b))
            self.assertNotEqual(rows_fingerprint(a.train), rows_fingerprint(b.train))
            np.testing.assert_array_equal(original_order(a).row_ids, np.arange(n))
            np.testing.assert_array_equal(original_order(a).x, first.x)
            with self.assertRaises(ValueError):
                replace(a, validation=a.holdout)
        self.assertNotEqual(partition_fingerprint(partition_rows(observed(), 0, 1)),
                            partition_fingerprint(partition_rows(observed(), 0, 2)))

    def test_method_inputs_expose_only_observations(self):
        self.assertEqual({f.name for f in fields(Rows)}, {"x", "a", "y", "row_ids"})
        self.assertEqual({f.name for f in fields(SitePartitions)},
                         {"site_id", "n_total", "train", "validation", "holdout"})


class SCMDataTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(SCMConfig(num_sites=3, site_min_samples=31, site_max_samples=39,
                                 covariate_dim=7, num_test=113), **kwargs)

    def test_all_five_mechanisms_probabilities_cate_and_null_mask(self):
        cfg = self.config()
        self.assertEqual(cfg.dataset_id, DATASET_ID)
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                experiment = generate_scm(cfg, 12, scenario)
                evaluation, x = experiment.evaluation, experiment.evaluation.x
                truth = evaluation.truth
                expected_e = np.clip(1/(1+np.exp(-(0.8*x[:, 2]-0.6*x[:, 4]))), 0.05, 0.95)
                baseline = 0.25+0.5/(1+np.exp(-(0.8*np.tanh(x[:, 2])+0.5*np.tanh(x[:, 3]))))
                if scenario == "null":
                    expected_tau = np.zeros(len(x))
                elif scenario == "interaction":
                    expected_tau = 0.2*np.tanh(x[:, 0])*np.tanh(x[:, 1])
                else:
                    expected_tau = 0.1*np.sin(x[:, 0])+0.1*np.tanh(x[:, 1])
                np.testing.assert_allclose(truth.e, expected_e, rtol=0, atol=1e-15)
                np.testing.assert_allclose(truth.tau, expected_tau, rtol=0, atol=1e-15)
                np.testing.assert_allclose(truth.mu_0, baseline-expected_tau/2, rtol=0, atol=1e-15)
                np.testing.assert_allclose(truth.mu_1-truth.mu_0, expected_tau, rtol=0, atol=1e-15)
                self.assertTrue(np.all((truth.mu_0 >= 0.15) & (truth.mu_0 <= 0.85)))
                self.assertTrue(np.all((truth.mu_1 >= 0.15) & (truth.mu_1 <= 0.85)))
                self.assertEqual(np.flatnonzero(truth.modifier_mask).tolist(), [] if scenario == "null" else [0, 1])
                self.assertEqual(truth.variable_roles[2:5],
                                 ("prognostic_and_confounder", "prognostic", "treatment_only"))
                sizes = np.array([site.n_total for site in experiment.sites], dtype=float)
                np.testing.assert_array_equal(evaluation.target_weights, sizes/sizes.sum())
                self.assertTrue(all(not hasattr(site, "truth") for site in experiment.sites))

    def test_outcomes_cannot_change_sites_partitions_or_treatment(self):
        experiments = [generate_scm(self.config(), 2, name) for name in ("base", "interaction", "null")]
        for base, other in zip(experiments[0].sites, experiments[1].sites):
            np.testing.assert_array_equal(original_order(base).x, original_order(other).x)
        for other in experiments[1:]:
            np.testing.assert_array_equal(other.evaluation.x, experiments[0].evaluation.x)
            for first, second in zip(experiments[0].sites, other.sites):
                self.assertEqual(partition_fingerprint(first), partition_fingerprint(second))
                np.testing.assert_array_equal(original_order(first).a, original_order(second).a)

    def test_fixed_shift_and_ar1_covariance_use_the_declared_target_mixture(self):
        cfg = self.config(num_test=2400)
        base = generate_scm(cfg, 19, "base")
        shifted = generate_scm(cfg, 19, "covariate_shift")
        correlated = generate_scm(cfg, 19, "correlated_shift")
        means = fixed_site_means(cfg)
        np.testing.assert_array_equal(means[:, 0], [-0.75, 0, 0.75])
        np.testing.assert_array_equal(means, np.repeat(means[:, :1], cfg.covariate_dim, axis=1))
        ids = shifted.evaluation.site_ids
        np.testing.assert_array_equal(base.evaluation.site_ids, ids)
        np.testing.assert_allclose(shifted.evaluation.x-base.evaluation.x, means[ids], rtol=0, atol=5e-16)
        residual = correlated.evaluation.x-means[ids]
        covariance = np.cov(residual, rowvar=False)
        expected = 0.5**np.abs(np.arange(cfg.covariate_dim)[:, None]-np.arange(cfg.covariate_dim))
        np.testing.assert_allclose(covariance, expected, rtol=0, atol=0.09)
        frequencies = np.bincount(ids, minlength=cfg.num_sites)/len(ids)
        np.testing.assert_allclose(frequencies, shifted.evaluation.target_weights, rtol=0, atol=0.04)

    def test_independent_external_test_cannot_change_training(self):
        small = generate_scm(self.config(num_test=17), 4, "base")
        large = generate_scm(self.config(num_test=119), 4, "base")
        for first, second in zip(small.sites, large.sites):
            self.assertEqual(partition_fingerprint(first), partition_fingerprint(second))
            self.assertEqual(rows_fingerprint(original_order(first)), rows_fingerprint(original_order(second)))
        train_ids = set(np.concatenate([original_order(s).row_ids for s in small.sites]).tolist())
        self.assertTrue(train_ids.isdisjoint(small.evaluation.rows.row_ids.tolist()))
        repeated = generate_scm(self.config(num_test=17), 4, "base")
        self.assertEqual(rows_fingerprint(small.evaluation.rows), rows_fingerprint(repeated.evaluation.rows))

    def test_invalid_scm_configs_fail(self):
        for changes in ({"covariate_dim": 4}, {"num_sites": True}, {"site_min_samples": 2},
                        {"site_mean_span": float("nan")}, {"ar1_rho": 0.2},
                        {"dataset_id": "SyntheticEffectModifier"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        with self.assertRaises(ValueError):
            generate_scm(self.config(), 0, "outcome_sorted")


class LegacyCompatibilityTests(unittest.TestCase):
    def test_original_sites_test_and_four_variant_pairing_are_unchanged(self):
        from effect_modifier_experiment.run_dslassocov_effect_modifier import generate_datasets
        cfg = LegacyConfig(num_sites=2, site_min_samples=30, site_max_samples=35,
                           covariate_dim=5, num_test=41)
        variants = ("control", "site_gaussian_shift", "local_permutation", "site_gaussian_local_permutation")
        for split in ("iid", "noniid"):
            prepared = {}
            for variant in variants:
                with self.subTest(split=split, variant=variant), redirect_stdout(io.StringIO()):
                    original_sites, original_test, _ = generate_datasets(SimpleNamespace(
                        **asdict(cfg), seed=8, variant=variant, split=split))
                    adapted = generate_legacy(cfg, 8, variant, split)
                    prepared[variant] = adapted
                    for original, site in zip(original_sites, adapted.sites):
                        values = original_order(site)
                        design, y = (tensor.numpy() for tensor in original.tensors)
                        np.testing.assert_array_equal(values.x, design[:, 1:6])
                        np.testing.assert_array_equal(values.a, design[:, 0])
                        np.testing.assert_array_equal(values.y, y)
                        self.assertEqual(site.n_total, len(original))
                    design, y = (tensor.numpy() for tensor in original_test.tensors)
                    np.testing.assert_array_equal(adapted.evaluation.x, design[:, 1:6])
                    np.testing.assert_array_equal(adapted.evaluation.rows.a, design[:, 0])
                    np.testing.assert_array_equal(adapted.evaluation.rows.y, y)
                    self.assertIsNone(adapted.evaluation.truth)
                    self.assertIsNone(adapted.evaluation.site_ids)
            for first, second in (("control", "local_permutation"),
                                  ("site_gaussian_shift", "site_gaussian_local_permutation")):
                self.assertEqual(rows_fingerprint(prepared[first].evaluation.rows), rows_fingerprint(prepared[second].evaluation.rows))
                for a, b in zip(prepared[first].sites, prepared[second].sites):
                    self.assertEqual(rows_fingerprint(a.train), rows_fingerprint(b.train))
                    self.assertEqual(partition_fingerprint(a), partition_fingerprint(b))


if __name__ == "__main__":
    unittest.main()
