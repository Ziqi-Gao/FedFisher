"""Tiny single-thread numerical replay fixtures, never scientific pilots."""
import copy
from dataclasses import asdict, replace
import json
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.one_shot_effects.basis import BasisConfig, SharedBasis
from effect_modifier_experiment.one_shot_effects.diagnostics.dr import (
    diagnose_dr, validate_dr_diagnostic, _ranking,
)
from effect_modifier_experiment.one_shot_effects.dr_nuisance import DRNuisanceConfig
from effect_modifier_experiment.one_shot_effects.dr_sieve import (
    DRSieveConfig, DRSieveModel, aggregate, build_local_payload,
)
from effect_modifier_experiment.one_shot_effects.evaluation import exact_scores
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations
from effect_modifier_experiment.one_shot_effects.types import (
    EvaluationData, EvaluationTruth, ExperimentData, Rows, SitePartitions,
)


def _rows(n, offset, seed):
    x = np.random.default_rng(seed).normal(size=(n, 3))
    return Rows(x, np.arange(n) % 2, (np.arange(n) // 2) % 2,
                np.arange(offset, offset + n, dtype=np.int64))


def _site(site_id, counts):
    offset = site_id * 1000
    blocks = []
    for index, count in enumerate(counts):
        blocks.append(_rows(count, offset, 250 + 10 * site_id + index))
        offset += count
    return SitePartitions(site_id, sum(counts), *blocks)


def _fixed_psi(train, validation, *_):
    return .15 * train.x[:, 1], .15 * validation.x[:, 1], dict(converged=True)


def _fixture():
    sites = (_site(0, (24, 8, 8)), _site(1, (36, 12, 12)))
    plans = [make_permutations(len(site.holdout), 3, 333 + site.site_id)
             for site in sites]
    weights = np.array([.4, .6])
    sources = {}
    configuration = DRSieveConfig(lambda_count=30, tolerance=2e-6)
    for name, kind in (("linear", "linear"), ("spline", "additive_cubic_bspline")):
        basis_config = BasisConfig(kind, 3)
        with mock.patch(
                "effect_modifier_experiment.one_shot_effects.dr_sieve.fit_dr_pseudo_outcomes",
                side_effect=_fixed_psi):
            payloads = [build_local_payload(site, basis_config, DRNuisanceConfig(),
                        configuration, 0, plan)[0] for site, plan in zip(sites, plans)]
        model, scores, fit = aggregate(payloads, configuration)
        sources[name] = dict(
            config=dict(dataset="scmv2_base", seed=0, recipe="dr_" + name,
                        dr_sieve=asdict(configuration)),
            result=dict(models=dict(primary=dict(fit=fit, holdout=dict(
                summary_squared_scores=scores.tolist())))),
            arrays={"primary.beta": model.beta}, payloads=payloads,
            models={"primary": model},
        )
    rows = _rows(12, 10000, 777)
    # Deliberately make the test-preferred vector differ from V selection.
    # Truth affects only diagnostics; the saved path is constructed first.
    intercept = sources["linear"]["result"]["models"]["primary"]["fit"]["coefficient_path"][0][0]
    tau = np.full(len(rows), intercept)
    truth = EvaluationTruth(np.full(len(rows), .5), .5 - tau / 2, .5 + tau / 2,
                            tau, np.array([True, True, False]), ("modifier", "modifier", "other"))
    evaluation = EvaluationData(rows, np.r_[np.zeros(9, dtype=int), np.ones(3, dtype=int)],
                                weights, truth)
    observation_weights = np.r_[np.full(9, .4 / 9), np.full(3, .6 / 3)]
    for source in sources.values():
        predicted = source["models"]["primary"].predict_tau(rows.x)
        mean = observation_weights @ predicted
        source["result"]["models"]["primary"]["test"] = dict(
            cate_rmse=float(np.sqrt(observation_weights @ ((predicted - tau) ** 2))),
            estimated_heterogeneity=float(observation_weights @ ((predicted - mean) ** 2)),
            effect_out_of_bounds_fraction=float(observation_weights @ (np.abs(predicted) > 1)),
        )
    return dict(dataset="scmv2_base", seed=0), ExperimentData(sites, evaluation), plans, sources


class DRReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.data, cls.plans, cls.sources = _fixture()

    def test_replays_all_candidates_without_fitting_or_test_selection(self):
        before = copy.deepcopy(self.sources["linear"]["result"])
        patches = [mock.patch(target, side_effect=AssertionError("replay must never fit"))
                   for target in (
                       "effect_modifier_experiment.one_shot_effects.dr_sieve.aggregate",
                       "effect_modifier_experiment.one_shot_effects.dr_sieve.solve_group_lasso",
                       "effect_modifier_experiment.one_shot_effects.dr_sieve.fit_dr_pseudo_outcomes",
                       "effect_modifier_experiment.one_shot_effects.dr_nuisance.fit_dr_pseudo_outcomes",
                   )]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            result = diagnose_dr(self.config, self.data, self.plans, self.sources)
        json.dumps(result, allow_nan=False)
        self.assertEqual(self.sources["linear"]["result"], before)
        self.assertFalse(result["test_used_for_selection"])
        self.assertFalse(result["nuisance_refit"])
        self.assertFalse(result["coefficient_refit"])
        for name in ("linear", "spline"):
            replay = result["basis_results"][name]
            original = self.sources[name]["result"]["models"]["primary"]["fit"]
            self.assertEqual(len(replay["path"]), 30)
            self.assertEqual(replay["original_selected_index"], original["selected_index"])
            self.assertEqual(sum(row["selected_by_original_V_rule"] for row in replay["path"]), 1)
            np.testing.assert_allclose([row["validation_risk"] for row in replay["path"]],
                                       original["validation_risks"], rtol=0, atol=1e-12)
            self.assertEqual([row["nonzero_groups"] for row in replay["path"]],
                             [row["nonzero_groups"] for row in original["path"]])
            self.assertEqual(replay["true_modifier_mask"], [True, True, False])
        linear = result["basis_results"]["linear"]
        test_best = int(np.argmin([row["test"]["cate_rmse"] for row in linear["path"]]))
        self.assertNotEqual(test_best, linear["original_selected_index"])

    def test_H_path_scores_equal_direct_fixed_candidate_scores(self):
        result = diagnose_dr(self.config, self.data, self.plans, self.sources)
        for name in ("linear", "spline"):
            basis_config = self.sources[name]["models"]["primary"].basis_config
            for index in (0, 14, 29):
                row = result["basis_results"][name]["path"][index]
                model = DRSieveModel(np.array(row["coefficients"]), basis_config)
                direct = sum(weight * exact_scores(model, site.holdout.x, plan, squared=True)
                             for weight, site, plan in zip(self.data.evaluation.target_weights,
                                                           self.data.sites, self.plans))
                np.testing.assert_allclose(row["H_scores"], direct, atol=1e-14)
                self.assertTrue(row["feature_groups"][0]["true_modifier"])
                self.assertTrue(row["feature_groups"][1]["true_modifier"])
                self.assertFalse(row["feature_groups"][2]["true_modifier"])

    def test_test_weighting_uses_target_site_means_not_test_site_frequencies(self):
        result = diagnose_dr(self.config, self.data, self.plans, self.sources)
        row = result["basis_results"]["linear"]["path"][-1]
        model = DRSieveModel(np.array(row["coefficients"]), BasisConfig("linear", 3))
        error = model.predict_tau(self.data.evaluation.rows.x) - self.data.evaluation.truth.tau
        expected = np.sqrt(.4 * np.mean(error[:9] ** 2) + .6 * np.mean(error[9:] ** 2))
        self.assertAlmostEqual(row["test"]["cate_rmse"], expected, places=14)
        self.assertNotAlmostEqual(expected, np.sqrt(np.mean(error ** 2)), places=6)

    def test_tie_ranks_do_not_promote_leading_true_features(self):
        ranks = _ranking(np.array([0., 2., 2., 0.]))
        self.assertEqual(ranks["rank_min"], [3, 1, 1, 3])
        self.assertEqual(ranks["rank_max"], [4, 2, 2, 4])
        self.assertEqual(ranks["average_rank"], [3.5, 1.5, 1.5, 3.5])
        replay = diagnose_dr(self.config, self.data, self.plans, self.sources)
        first = replay["basis_results"]["linear"]["path"][0]
        self.assertEqual(first["nonzero_groups"], 0)
        self.assertEqual(first["H_recovery"]["auroc"], .5)

    def test_modified_saved_selection_fails_before_test_inference(self):
        sources = copy.deepcopy(self.sources)
        sources["linear"]["result"]["models"]["primary"]["fit"]["selected_index"] = 0
        with mock.patch.object(SharedBasis, "transform", side_effect=AssertionError("must reject first")):
            with self.assertRaisesRegex(ValueError, "selected model mismatch"):
                diagnose_dr(self.config, self.data, self.plans, sources)

    def test_changed_H_permutation_fails_before_test_inference(self):
        plans = [plan.copy() for plan in self.plans]
        plans[0][0, 0] = plans[0][0, 0][::-1]
        with mock.patch.object(SharedBasis, "transform", side_effect=AssertionError("must reject first")):
            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                diagnose_dr(self.config, self.data, plans, self.sources)

    def test_saved_H_block_disagrees_with_regenerated_rows(self):
        site = self.data.sites[0]
        changed = Rows(site.holdout.x * 2, site.holdout.a, site.holdout.y, site.holdout.row_ids)
        data = replace(self.data, sites=(replace(site, holdout=changed), self.data.sites[1]))
        with self.assertRaisesRegex(ValueError, "H blocks disagree"):
            diagnose_dr(self.config, data, self.plans, self.sources)

    def test_scope_guard_rejects_missing_basis_or_other_dataset(self):
        for config, sources in (
                ({**self.config, "dataset": "legacy_control_iid"}, self.sources),
                ({**self.config, "seed": 1}, self.sources),
                (self.config, {"linear": self.sources["linear"]})):
            with self.assertRaises(ValueError):
                diagnose_dr(config, self.data, self.plans, sources)

    def test_saved_diagnostic_validator_has_no_row_inference_or_fitting(self):
        result = diagnose_dr(self.config, self.data, self.plans, self.sources)
        with mock.patch.object(SharedBasis, "transform", side_effect=AssertionError("no row inference")), \
                mock.patch.object(SharedBasis, "blocks", side_effect=AssertionError("no row evaluation")), \
                mock.patch("effect_modifier_experiment.one_shot_effects.dr_sieve.solve_group_lasso",
                           side_effect=AssertionError("no solver")):
            validate_dr_diagnostic(result, self.config, self.sources)

    def test_saved_diagnostic_validator_rejects_mutated_path_or_scope(self):
        original = diagnose_dr(self.config, self.data, self.plans, self.sources)
        mutations = (
            lambda r: r.update(test_used_for_selection=True),
            lambda r: r["basis_results"]["linear"].update(original_selected_index=0),
            lambda r: r["basis_results"]["linear"]["path"][5].update(validation_risk_gap_to_minimum=10.),
            lambda r: r["basis_results"]["linear"]["path"][5]["feature_groups"][1].update(coefficient_norm=99.),
            lambda r: r["basis_results"]["linear"]["path"][5]["H_scores"].__setitem__(1, 99.),
            lambda r: r["basis_results"]["linear"]["path"][5]["H_ranking"]["rank_min"].__setitem__(1, 99),
            lambda r: r["basis_results"]["linear"]["path"][5]["test"].update(cate_rmse=float("nan")),
            lambda r: r["basis_results"]["linear"]["path"][5]["test"].update(diagnostic_only=False),
            lambda r: r["basis_results"]["linear"].update(true_modifier_mask=[False, True, True]),
            lambda r: r["basis_results"]["linear"].update(H_summary_replay_max_absolute_error=.1),
            lambda r: r["timing"].update(wall_seconds=-1.),
        )
        for index, mutation in enumerate(mutations):
            changed = copy.deepcopy(original)
            mutation(changed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_dr_diagnostic(changed, self.config, self.sources)


if __name__ == "__main__":
    unittest.main()
