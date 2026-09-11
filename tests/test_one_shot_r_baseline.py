"""Tiny common-target adapter checks; historical R implementation is unchanged."""
import copy
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.causal_comparison.rlearner import RlearnerConfig, summary_cd_loss
from effect_modifier_experiment.one_shot_effects import r_baseline
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations, perturbed_covariates
from effect_modifier_experiment.one_shot_effects.types import Rows, SitePartitions
from effect_modifier_experiment.one_shot_effects.wire import unpack_payload


def rows(n, offset, seed):
    rng = np.random.default_rng(seed)
    return Rows(rng.normal(size=(n, 2)), (np.arange(n)%2).astype(float),
                (rng.random(n) < .45).astype(float), np.arange(offset, offset+n))


def fixture():
    sites = []
    for j, counts in enumerate(((20, 4, 6), (30, 10, 8))):
        offset = 1000*j
        t = rows(counts[0], offset, 11+j)
        v = rows(counts[1], offset+counts[0], 21+j)
        h = rows(counts[2], offset+sum(counts[:2]), 31+j)
        sites.append(SitePartitions(j, sum(counts), t, v, h))
    config = RlearnerConfig(crossfit_folds=2, nuisance_cv_folds=2, nuisance_lambda_count=2,
                            effect_lambda_count=4, effect_lambda_min_ratio=.01)
    plans = [make_permutations(len(s.holdout), 2, 80+s.site_id) for s in sites]
    def nuisances(site, config, seed, stages):
        return np.full(len(site.outcome), .4), np.full(len(site.outcome), .45), {"test_rows_used": 0}
    with mock.patch.object(r_baseline, "crossfit_site_nuisances", side_effect=nuisances):
        payloads = [r_baseline.build_local_payload(s, config, 9, plan)[0] for s, plan in zip(sites, plans)]
    return sites, plans, payloads, config


class CommonRTests(unittest.TestCase):
    def test_target_weighted_objective_and_T_only_EBIC_count(self):
        sites, _, payloads, config = fixture()
        summaries, _, target = r_baseline.adapted_summaries(payloads, config)
        beta = np.array([.2, -.1, .3])
        total = sum(s.n_total for s in sites)
        direct = 0.
        for site in sites:
            design = (site.train.a-.45)[:, None]*np.column_stack((np.ones(len(site.train)), site.train.x))
            residual_y = site.train.y-.4
            local_beta = np.linalg.solve(design.T@design, design.T@residual_y)
            variance = np.mean((residual_y-design@local_beta)**2)
            precision = design.T@design/variance
            delta = local_beta-beta
            direct += site.n_total/total * (delta@precision@delta)/(2*len(site.train))
        self.assertAlmostEqual(summary_cd_loss(summaries, beta), direct, places=14)
        self.assertEqual(target["ebic_num_examples"], 50)
        self.assertEqual(target["target_num_examples"], 78)
        np.testing.assert_allclose(target["site_weights"], np.array([30., 48.])/78)
        self.assertNotEqual(target["site_weights"][0], .4)

    def test_both_penalties_selected_KKT_exact_score_and_tamper_rejection(self):
        sites, plans, payloads, config = fixture()
        for penalty in ("none", "adaptive_lasso"):
            model, scores, report = r_baseline.aggregate(payloads, penalty, config)
            r_baseline.validate_server_result(payloads, model, scores, report, penalty, config)
            expected = np.zeros(2)
            for site, plan in zip(sites, plans):
                weight = site.n_total/sum(s.n_total for s in sites)
                original = model.predict_tau(site.holdout.x)
                for j, _, changed in perturbed_covariates(site.holdout.x, plan):
                    expected[j] += weight*np.mean((original-model.predict_tau(changed))**2)/10
            np.testing.assert_allclose(scores, expected, atol=1e-14)
            self.assertEqual(report["communication"]["upload_rounds"], 1)
            self.assertEqual(report["communication"]["upload_bytes"], sum(map(len, payloads)))
            self.assertEqual(report["fit"]["num_examples"], 50)
            self.assertFalse(report["validation_used"])
            broken = copy.deepcopy(report)
            broken["target"]["site_weights"][0] += .01
            with self.assertRaises(ValueError):
                r_baseline.validate_server_result(payloads, model, scores, broken, penalty, config)
            broken = copy.deepcopy(report)
            broken["fit"]["unpenalized_shared_coefficients"][0] += 1
            with self.assertRaises(ValueError):
                r_baseline.validate_server_result(payloads, model, scores, broken, penalty, config)
            with self.assertRaises(ValueError):
                r_baseline.validate_server_result(payloads, model, scores+1, report, penalty, config)

    def test_observed_T_only_local_inputs(self):
        sites, plans, _, config = fixture()
        site = sites[0]
        records = []
        def nuisances(local, config, seed, stages):
            records.append((local.covariates.copy(), local.treatment.copy(), local.outcome.copy()))
            return np.full(len(local.outcome), .4), np.full(len(local.outcome), .45), {}
        changed = replace(site,
            validation=Rows(site.validation.x+100, 1-site.validation.a, 1-site.validation.y, site.validation.row_ids),
            holdout=Rows(site.holdout.x*2, 1-site.holdout.a, 1-site.holdout.y, site.holdout.row_ids))
        with mock.patch.object(r_baseline, "crossfit_site_nuisances", side_effect=nuisances):
            first, _ = r_baseline.build_local_payload(site, config, 12, plans[0])
            second, _ = r_baseline.build_local_payload(changed, config, 12, plans[0])
        for original, modified in zip(records[0], records[1]):
            np.testing.assert_array_equal(original, modified)
        one, _ = unpack_payload(first, "r_common_target_v1")
        two, _ = unpack_payload(second, "r_common_target_v1")
        np.testing.assert_array_equal(one["beta"], two["beta"])
        np.testing.assert_array_equal(one["precision"], two["precision"])


if __name__ == "__main__":
    unittest.main()
