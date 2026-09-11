"""Small single-CPU numerical and leakage contracts; no scientific pilots."""
import copy
from dataclasses import replace
import inspect
import json
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.one_shot_effects.basis import (
    BasisConfig, SharedBasis, cubic_bspline_values,
)
from effect_modifier_experiment.one_shot_effects.dr_nuisance import (
    DRNuisanceConfig, aipw_pseudo_outcome, fit_dr_pseudo_outcomes,
)
from effect_modifier_experiment.one_shot_effects import dr_nuisance
from effect_modifier_experiment.one_shot_effects.dr_sieve import (
    DRSieveConfig, DRSieveModel, aggregate, aggregate_moments, build_local_payload,
    group_kkt_residual, group_objective, holdout_score_blocks, solve_group_lasso,
    validate_server_result, validation_risk,
)
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations, perturbed_covariates
from effect_modifier_experiment.one_shot_effects.types import Rows, SitePartitions
from effect_modifier_experiment.one_shot_effects.wire import pack_payload, unpack_payload


def make_rows(n, p, offset, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    a = (np.arange(n) % 2).astype(float)
    y = ((np.arange(n)//2) % 2).astype(float)
    return Rows(x=x, a=a, y=y, row_ids=np.arange(offset, offset+n, dtype=np.int64))


def make_site(site_id, counts=(28, 8, 9), p=2):
    offset = site_id * 10000
    train = make_rows(counts[0], p, offset, 100+site_id)
    validation = make_rows(counts[1], p, offset+counts[0], 200+site_id)
    holdout = make_rows(counts[2], p, offset+counts[0]+counts[1], 300+site_id)
    return SitePartitions(site_id, sum(counts), train, validation, holdout)


def fixed_psi(train, validation, config, seed):
    def effect(rows):
        return .12 + .25*np.tanh(rows.x[:, 0]) - .08*rows.x[:, 1]
    return effect(train), effect(validation), dict(converged=True)


def fixture_payloads(kind="linear"):
    basis_config = BasisConfig(kind=kind, covariate_dim=2)
    config = DRSieveConfig(lambda_count=4, lambda_min_ratio=.05, tolerance=2e-7)
    sites = [make_site(0), make_site(1, (42, 11, 12))]
    plans = [make_permutations(len(s.holdout.x), 2, 90+s.site_id) for s in sites]
    with mock.patch("effect_modifier_experiment.one_shot_effects.dr_sieve.fit_dr_pseudo_outcomes", side_effect=fixed_psi):
        payloads = [build_local_payload(s, basis_config, DRNuisanceConfig(), config, 11, plan)[0]
                    for s, plan in zip(sites, plans)]
    return sites, plans, payloads, basis_config, config


class BasisTests(unittest.TestCase):
    def test_clamped_partition_and_fixed_drop(self):
        u = np.r_[-1., np.linspace(-.999, .999, 100), 1.]
        values = cubic_bspline_values(u)
        np.testing.assert_allclose(values.sum(axis=1), 1., atol=1e-15)
        self.assertTrue(np.all(values >= 0))
        np.testing.assert_array_equal(values[0], [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(values[-1], [0, 0, 0, 0, 0, 1])
        basis = SharedBasis(BasisConfig(covariate_dim=2))
        design = basis.transform(np.array([[-100., 100.], [0., 0.]]))
        self.assertEqual(design.shape, (2, 11))
        np.testing.assert_array_equal(design[0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(basis.metadata()["dropped_basis_index"], 5)
        x = np.random.default_rng(5).normal(size=(50, 2))
        self.assertEqual(np.linalg.matrix_rank(basis.transform(x)), 11)

    def test_linear_raw_and_invalid(self):
        x = np.array([[2., -3.], [0., 9.]])
        np.testing.assert_array_equal(SharedBasis(BasisConfig("linear", 2)).transform(x),
                                      [[1., 2., -3.], [1., 0., 9.]])
        with self.assertRaises(ValueError):
            BasisConfig("site_quantiles", 2)
        with self.assertRaises(ValueError):
            cubic_bspline_values(np.array([1.1]))
        with self.assertRaises(ValueError):
            SharedBasis(BasisConfig("linear", 2)).transform(np.array([[np.nan, 0.]]))


class DRNuisanceTests(unittest.TestCase):
    def test_aipw_formula(self):
        a = np.array([0., 1.])
        value = aipw_pseudo_outcome(a, np.array([1., 0.]), np.array([.25, .75]),
                                   np.array([.2, .3]), np.array([.7, .6]))
        np.testing.assert_allclose(value, [.5-.8/.75, .3-.6/.75])

    def test_outer_scope_separate_outcome_arms_and_no_validation_leakage(self):
        site = make_site(2)
        config = DRNuisanceConfig(crossfit_folds=2, nuisance_cv_folds=2, nuisance_lambda_count=2)
        calls = []

        def fit(x, target, predict_x, legacy_config, seed, stages):
            calls.append((x.copy(), target.copy(), predict_x.copy(), seed))
            return np.full(len(predict_x), (target.sum()+.5)/(len(target)+1)), dict(converged=True)

        with mock.patch.object(dr_nuisance, "_fit_nuisance_cv", side_effect=fit):
            psi, vpsi, report = fit_dr_pseudo_outcomes(site.train, site.validation, config, 18)
            first = copy.deepcopy(calls)
            calls.clear()
            changed = Rows(site.validation.x+100., 1-site.validation.a, 1-site.validation.y, site.validation.row_ids)
            psi2, _, report2 = fit_dr_pseudo_outcomes(site.train, changed, config, 18)
        np.testing.assert_array_equal(psi, psi2)
        self.assertEqual(report["training_psi_sha256"], report2["training_psi_sha256"])
        self.assertEqual(len(first), 9)
        for old, new in zip(first, calls):
            np.testing.assert_array_equal(old[0], new[0])
            np.testing.assert_array_equal(old[1], new[1])
            self.assertEqual(old[3], new[3])
        for i in range(0, 9, 3):
            propensity, mu0, mu1 = first[i:i+3]
            # Each outcome model is fitted solely within its observed arm.
            np.testing.assert_array_equal(mu0[0], propensity[0][propensity[1] == 0])
            np.testing.assert_array_equal(mu1[0], propensity[0][propensity[1] == 1])
            if i < 6:
                self.assertFalse(any(np.array_equal(x, z) for x in propensity[0] for z in propensity[2]))
        self.assertEqual(first[-3][0].shape[0], len(site.train.x))
        self.assertEqual(vpsi.shape, (len(site.validation.x),))
        self.assertFalse(report["holdout_used"])

    def test_empty_arm_fails_and_nonempty_constant_outcome_is_explicit(self):
        site = make_site(0, (40, 8, 9))
        config = DRNuisanceConfig(crossfit_folds=2, nuisance_cv_folds=2, nuisance_lambda_count=2,
                                  nuisance_lambda_max=.5, nuisance_lambda_min_ratio=.8)
        constant = Rows(site.train.x, site.train.a, np.zeros(len(site.train.x)), site.train.row_ids)
        _, _, report = fit_dr_pseudo_outcomes(constant, site.validation, config, 31)
        for triplet in report["crossfit_folds"]+[report["validation_nuisance_refit"]]:
            self.assertEqual(triplet["models"]["mu0"]["fit_kind"], "train_only_smoothed_bernoulli_constant")
            self.assertEqual(triplet["models"]["mu1"]["fit_kind"], "train_only_smoothed_bernoulli_constant")
            self.assertGreater(min(triplet["treatment_arm_counts"]), 0)
        missing = Rows(site.train.x, np.zeros(len(site.train.x)), site.train.y, site.train.row_ids)
        with self.assertRaisesRegex(ValueError, "empty arm"):
            fit_dr_pseudo_outcomes(missing, site.validation, config, 31)
        self.assertNotIn("holdout", inspect.signature(fit_dr_pseudo_outcomes).parameters)
        self.assertNotIn("truth", inspect.signature(fit_dr_pseudo_outcomes).parameters)
        with self.assertRaises(TypeError):
            fit_dr_pseudo_outcomes(site.train, site.validation, config, 31, truth=np.zeros(2))

    def test_nonconstant_three_logistic_models_end_to_end(self):
        site = make_site(4, (60, 8, 9))
        nuisance = DRNuisanceConfig(crossfit_folds=2, nuisance_cv_folds=2, nuisance_lambda_count=3,
                                    nuisance_lambda_max=.1, nuisance_lambda_min_ratio=.5)
        config = DRSieveConfig(lambda_count=3, lambda_min_ratio=.1)
        payload, local = build_local_payload(site, BasisConfig("linear", 2), nuisance, config, 15,
                                             make_permutations(len(site.holdout.x), 2, 8))
        for triplet in local["nuisance"]["crossfit_folds"]+[local["nuisance"]["validation_nuisance_refit"]]:
            for name in ("propensity", "mu0", "mu1"):
                self.assertEqual(triplet["models"][name]["fit_kind"], "site_local_logistic_lasso_nested_cv")
                self.assertTrue(triplet["models"][name]["converged"])
        model, scores, report = aggregate([payload], config)
        validate_server_result([payload], model, scores, report, config)
        json.dumps(local, allow_nan=False)


class SummaryAndSolverTests(unittest.TestCase):
    def test_main_spline_optimization_and_exact_selected_score(self):
        sites, plans, payloads, basis_config, config = fixture_payloads("additive_cubic_bspline")
        model, scores, report = aggregate(payloads, config)
        validate_server_result(payloads, model, scores, report, config)
        expected = np.zeros(2)
        for site, plan in zip(sites, plans):
            weight = site.n_total/sum(s.n_total for s in sites)
            baseline = model.predict_tau(site.holdout.x)
            for j, _, changed in perturbed_covariates(site.holdout.x, plan):
                expected[j] += weight*np.mean((baseline-model.predict_tau(changed))**2)/10
        np.testing.assert_allclose(scores, expected, atol=1e-13)

    def test_weighted_pooled_objective_validation_and_exact_H(self):
        sites, plans, payloads, basis_config, config = fixture_payloads("additive_cubic_bspline")
        moments = aggregate_moments(payloads, config)
        basis = SharedBasis(basis_config)
        beta = np.linspace(-.6, .7, basis.dimension)
        total = sum(s.n_total for s in sites)
        direct_train, direct_validation, direct_scores = 0., 0., np.zeros(2)
        for site, plan in zip(sites, plans):
            weight = site.n_total/total
            psi, vpsi, _ = fixed_psi(site.train, site.validation, None, None)
            pred = basis.transform(site.train.x) @ beta
            direct_train += weight * np.mean(.5*pred**2-psi*pred)
            direct_validation += weight * np.mean((basis.transform(site.validation.x)@beta-vpsi)**2)
            baseline = basis.transform(site.holdout.x) @ beta
            full_blocks = np.zeros((2, basis.dimension, basis.dimension))
            for j, repeat, changed in perturbed_covariates(site.holdout.x, plan):
                design_delta = basis.transform(site.holdout.x)-basis.transform(changed)
                full_blocks[j] += design_delta.T@design_delta/(2*len(changed)*5)
                direct_scores[j] += weight*np.mean((baseline-basis.transform(changed)@beta)**2)/10
            blocks = holdout_score_blocks(site.holdout.x, basis, plan)
            for j, group in enumerate(basis.groups):
                np.testing.assert_allclose(blocks[j], full_blocks[j, group, group], atol=1e-15)
                self.assertAlmostEqual(beta@full_blocks[j]@beta, beta[group]@blocks[j]@beta[group], places=14)
        summary_train = .5*beta@moments["train_gram"]@beta-moments["train_linear"]@beta
        self.assertAlmostEqual(summary_train, direct_train, places=14)
        self.assertAlmostEqual(validation_risk(beta, moments["validation_gram"], moments["validation_linear"],
                                              moments["validation_constant"]), direct_validation, places=14)
        reshaped = beta[1:].reshape(2, 5)
        np.testing.assert_allclose(np.einsum("pi,pij,pj->p", reshaped, moments["holdout_blocks"], reshaped), direct_scores, atol=1e-14)
        self.assertNotAlmostEqual(moments["weights"][0], len(sites[0].train.x)/sum(len(s.train.x) for s in sites))

    def test_group_gradient_soft_threshold_and_kkt(self):
        gram = np.eye(5)
        linear = np.array([.7, 3., 4., 1., 0.])
        weights = np.array([1., 2.])
        config = DRSieveConfig(tolerance=1e-10)
        beta, report = solve_group_lasso(gram, linear, weights, 2, 1., config)
        expected = np.r_[.7, np.array([3., 4.])*(1-np.sqrt(2)/5), 0., 0.]
        np.testing.assert_allclose(beta, expected, atol=1e-12)
        self.assertLessEqual(group_kkt_residual(beta, gram, linear, 1., weights, 2), report["stationarity_threshold"])
        interior = np.array([.1, .5, -.3, .4, .2])
        derivative = gram@interior-linear
        for j in range(2):
            part = slice(1+2*j, 3+2*j)
            derivative[part] += weights[j]*np.sqrt(2)*interior[part]/np.linalg.norm(interior[part])
        eps = 1e-6
        numerical = []
        for j in range(5):
            delta = np.eye(5)[j]*eps
            numerical.append((group_objective(interior+delta, gram, linear, 1., weights, 2)
                              -group_objective(interior-delta, gram, linear, 1., weights, 2))/(2*eps))
        np.testing.assert_allclose(numerical, derivative, atol=1e-9)
        with self.assertRaisesRegex(RuntimeError, "safety iteration limit"):
            solve_group_lasso(np.diag([1., 2., 5.]), np.array([1., 2., 3.]), np.ones(2), 1, .01,
                              replace(config, max_iter=1))

    def test_server_selection_certificates_tamper_and_single_upload(self):
        sites, plans, payloads, basis_config, config = fixture_payloads()
        model, scores, report = aggregate(payloads, config)
        validate_server_result(payloads, model, scores, report, config)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["communication"]["upload_messages"], len(sites))
        self.assertEqual(report["communication"]["upload_rounds"], 1)
        self.assertEqual(report["communication"]["upload_bytes"], sum(map(len, payloads)))
        selected = np.flatnonzero(np.array(report["validation_risks"]) <= min(report["validation_risks"])+config.validation_tie_tolerance)[0]
        self.assertEqual(selected, report["selected_index"])
        first = np.array(report["coefficient_path"])[0]
        np.testing.assert_array_equal(first[1:], np.zeros(len(first)-1))
        for field in ("selected_index", "initial_coefficients", "coefficient_path", "adaptive_weights"):
            broken = copy.deepcopy(report)
            if field == "selected_index":
                broken[field] = (selected+1) % config.lambda_count
            elif field == "coefficient_path":
                broken[field][1][1] += 1.
            else:
                broken[field][0] += 1.
            with self.assertRaises(ValueError):
                validate_server_result(payloads, model, scores, broken, config)
        with self.assertRaises(ValueError):
            validate_server_result(payloads, model, scores+1., report, config)
        with self.assertRaises(ValueError):
            aggregate(sites, config)
        with self.assertRaises(ValueError):
            aggregate([payloads[0], payloads[0]], config)
        arrays, metadata = unpack_payload(payloads[0], "dr_sieve")
        wrong = dict(arrays)
        wrong["train_gram"] = arrays["train_gram"].copy()
        wrong["train_gram"][1, 1] = -1
        with self.assertRaises(ValueError):
            aggregate([pack_payload("dr_sieve", wrong, metadata)], config)
        self.assertFalse(report["refit_after_selection"])
        unbounded = DRSieveModel(np.array([3., 0., 0.]), basis_config)
        np.testing.assert_array_equal(unbounded.predict_tau(sites[0].holdout.x), np.full(len(sites[0].holdout.x), 3.))

    def test_singular_local_gram_constant_psi_ties_larger_lambda(self):
        site = make_site(0, (2, 2, 2), p=5)
        config = DRSieveConfig(lambda_count=3)
        def constant(train, validation, *args):
            return np.zeros(len(train.x)), np.zeros(len(validation.x)), {}
        with mock.patch("effect_modifier_experiment.one_shot_effects.dr_sieve.fit_dr_pseudo_outcomes", side_effect=constant):
            payload, audit = build_local_payload(site, BasisConfig("linear", 5), DRNuisanceConfig(), config, 2,
                                                 make_permutations(2, 5, 7))
        arrays, _ = unpack_payload(payload, "dr_sieve")
        self.assertLess(np.linalg.matrix_rank(arrays["train_gram"]), arrays["train_gram"].shape[0])
        model, scores, report = aggregate([payload], config)
        validate_server_result([payload], model, scores, report, config)
        self.assertEqual(report["selected_index"], 0)
        self.assertEqual(report["lambda_max"], 0.)
        np.testing.assert_array_equal(scores, np.zeros(5))

    def test_holdout_and_validation_changes_cannot_change_T_summaries(self):
        site = make_site(0)
        config, basis_config = DRSieveConfig(lambda_count=3), BasisConfig("linear", 2)
        nuisance = DRNuisanceConfig(crossfit_folds=2, nuisance_cv_folds=2, nuisance_lambda_count=2,
                                    nuisance_lambda_max=.5, nuisance_lambda_min_ratio=.8)
        plan = make_permutations(len(site.holdout.x), 2, 17)
        def fit(x, target, predict_x, legacy_config, seed, stages):
            return np.full(len(predict_x), (target.sum()+.5)/(len(target)+1)), dict(converged=True)
        with mock.patch.object(dr_nuisance, "_fit_nuisance_cv", side_effect=fit):
            first, audit1 = build_local_payload(site, basis_config, nuisance, config, 10, plan)
            changed = replace(site,
                validation=Rows(site.validation.x+10, 1-site.validation.a, 1-site.validation.y, site.validation.row_ids),
                holdout=Rows(site.holdout.x*2, 1-site.holdout.a, 1-site.holdout.y, site.holdout.row_ids))
            second, audit2 = build_local_payload(changed, basis_config, nuisance, config, 10, plan)
        a, _ = unpack_payload(first, "dr_sieve")
        b, _ = unpack_payload(second, "dr_sieve")
        for name in ("train_gram", "train_linear", "train_constant"):
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(audit1["nuisance"]["training_psi_sha256"], audit2["nuisance"]["training_psi_sha256"])


if __name__ == "__main__":
    unittest.main()
