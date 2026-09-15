"""One-thread synthetic numerical fixtures; no saved scientific pilot is run."""
from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from effect_modifier_experiment.one_shot_effects import fedfisher_em as em
from effect_modifier_experiment.one_shot_effects.diagnostics.fisher import (
    diagnose_fisher, sampled_interventions)
from effect_modifier_experiment.one_shot_effects.diagnostics.fisher_metrics import (
    at_vector, cate_metrics, combine_moments, error_moments, gradient_check,
    local_derivatives, outputs, sigmoid)
from effect_modifier_experiment.one_shot_effects.diagnostics.fisher_validation import validate_fisher_diagnostic
from effect_modifier_experiment.one_shot_effects.evaluation import summarize_linearization
from effect_modifier_experiment.one_shot_effects.fisher_models import (
    OutcomeEffectModel, array_hash, make_model, model_from_vector, model_vector)
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations
from effect_modifier_experiment.one_shot_effects.types import (
    Rows, SitePartitions, EvaluationTruth, EvaluationData, ExperimentData)


def tiny_relu(bias=2.):
    model = nn.Sequential(nn.Linear(3, 1), nn.ReLU(), nn.Linear(1, 2)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[.5, 1., 0.]], dtype=torch.float64))
        model[0].bias.fill_(bias)
        model[2].weight.copy_(torch.tensor([[0.], [1.]], dtype=torch.float64))
        model[2].bias.zero_()
    return model


def configuration():
    return dict(seed=0, sample_rows_per_site=2, sample_seed_stream=8600021,
        direction_seed_stream=8700023, t_grid=[0, .001, .01, .03, .1, .25, .5, .75, 1],
        fd_steps=[1e-4, 1e-5, 1e-6], fd_atol=1e-7, fd_rtol=1e-4,
        replay_tolerance=dict(rtol=1e-9, atol=1e-10))


def replay_fixture(legacy=False):
    """Hand-constructed observed rows and parameters; no fitting or DGP call."""
    sites, locals_, payloads = [], [], []
    fisher_config = em.FedFisherEMConfig(local_epochs=1)
    for site_id in range(2):
        blocks = []
        start = 100 * site_id
        for count in (4, 2, 3):
            raw = np.linspace(-.8, .9, count * 2).reshape(count, 2) + .1 * site_id
            treatment = np.arange(count) % 2
            blocks.append(Rows(raw, treatment, 1 - treatment, np.arange(start, start + count)))
            start += count
        site = SitePartitions(site_id, 9, *blocks)
        sites.append(site)
        model = make_model('SyntheticMLP', 2, 0)
        theta = model_vector(model)
        initial_hash = array_hash(theta)
        theta = theta + np.random.default_rng(11 + site_id).normal(size=len(theta)) * .0001
        model = model_from_vector('SyntheticMLP', 2, theta)
        fisher = np.ones_like(theta)  # Algebra fixture, not estimated scientific Fisher.
        audit = dict(fisher_status='computed_mean_true_categorical_diag', theta_sha256=array_hash(theta),
                     fisher_sha256=array_hash(fisher), train_row_ids_sha256=array_hash(site.train.row_ids),
                     train_x_sha256=array_hash(site.train.x))
        local = em.LocalFisherState(model, 'SyntheticMLP', 2, theta, fisher, site.train,
                                   0, site_id, fisher_config, initial_hash, True, audit)
        locals_.append(local)
        payloads.append(em.build_training_payload(local, 9, site_id))
    test_rows = Rows(np.array([[-.6, .2], [.2, -.3], [.7, .4], [-.1, -.5]]),
                     [0, 1, 0, 1], [0, 1, 1, 0], np.arange(300, 304))
    true_tau = np.array([-.05, .02, .08, -.01])
    truth = None if legacy else EvaluationTruth(np.full(4, .5), .5 - true_tau / 2,
        .5 + true_tau / 2, true_tau, np.array([True, True]), ('modifier', 'modifier'))
    evaluation = EvaluationData(test_rows, None if legacy else np.array([0, 0, 1, 1]), np.array([.5, .5]), truth)
    data = ExperimentData(tuple(sites), evaluation)
    plans = [make_permutations(3, 2, 9 + site.site_id, 5) for site in sites]
    theta = np.mean([local.theta for local in locals_], axis=0)
    model = OutcomeEffectModel('SyntheticMLP', 2, model_from_vector('SyntheticMLP', 2, theta))
    gate = dict(relative_tau_rmse_max=.25, relative_score_linf_max=.25, denominator_floor=1e-8)
    original_config = dict(seed=0, fisher=asdict(fisher_config), dgp=dict(covariate_dim=2),
                           dataset_definition=dict(family='legacy' if legacy else 'scmv2'), linearity_gate=gate)
    local_diagnostics = [em.full_gradient_diagnostics(local, theta, site.holdout.x, plan)
                         for local, site, plan in zip(locals_, sites, plans)]
    saved_test = cate_metrics(model.predict_tau(test_rows.x), evaluation)
    result = dict(data_fingerprints=dict(sites=[dict(site_id=site.site_id,
        counts=dict(train=4, validation=2, holdout=3)) for site in sites], target_weights=[.5, .5]),
        diagnostics=dict(local_linearization=local_diagnostics,
                         linearization=summarize_linearization(local_diagnostics, [.5, .5], [True, True], gate)),
        models={name: dict(test={key: saved_test[key] for key in ('cate_rmse', 'estimated_heterogeneity')})
                for name in ('primary', 'matched_fedavg', 'legacy_validation_selected', 'strict_legacy_uploaded_objective')})
    source = dict(config=original_config, result=result, arrays={}, payloads=payloads,
                  models={name: deepcopy(model) for name in result['models']})
    return data, plans, dict(aggregation=source, linearity=deepcopy(source))


class FisherReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_and_random_finite_differences_pass_in_stable_regions(self):
        model = tiny_relu()
        theta = model_vector(model)
        direction = np.linspace(-.2, .3, len(theta))
        report = gradient_check(model, np.array([[-.2], [.3]]), direction, direction[::-1].copy(),
                                [1e-4, 1e-5, 1e-6], 1e-7, 1e-4)
        self.assertTrue(report['passed'])
        self.assertEqual(report['stable_checks'], 36)
        self.assertEqual(report['boundary_checks'], 0)
        self.assertEqual(len(report['checks']), 6)
        np.testing.assert_array_equal(model_vector(model), theta)

    def test_relu_crossings_are_recorded_and_not_called_gradient_failures(self):
        model = tiny_relu(bias=0.)
        direction = np.zeros_like(model_vector(model))
        direction[3] = 1.  # Hidden bias crosses zero for the control x=0 row.
        report = gradient_check(model, np.array([[0.], [.3]]), direction, direction,
                                [1e-4, 1e-5, 1e-6], 1e-7, 1e-4)
        self.assertGreater(report['boundary_checks'], 0)
        self.assertEqual(report['stable_failures'], 0)
        self.assertTrue(report['passed'])
        self.assertFalse(report['checks'][0]['stable_activation_mask'][0][0])
        self.assertGreater(report['checks'][0]['absolute_error'][0][0], .1)

    def test_sigmoid_of_linear_logits_is_exact_for_a_logit_bias_move(self):
        model = tiny_relu()
        theta = model_vector(model)
        delta = np.zeros_like(theta)
        delta[-1] = 1.5
        x = np.array([[-.2], [.3]])
        baseline, derivative = local_derivatives(model, x, delta[:, None])
        exact = outputs(at_vector(model, theta + delta), x)
        linked = sigmoid(baseline[:, :2] + derivative[:, :2, 0])
        linked_tau = linked[:, 1] - linked[:, 0]
        direct_tau = baseline[:, 2] + derivative[:, 2, 0]
        np.testing.assert_allclose(linked_tau, exact['tau'], atol=1e-15)
        self.assertGreater(np.max(np.abs(direct_tau - exact['tau'])), .001)

    def test_prediction_average_is_not_parameter_average(self):
        first = tiny_relu()
        theta = model_vector(first)
        delta = np.zeros_like(theta)
        delta[-1] = 4.
        second = at_vector(first, theta + delta)
        parameter_average = at_vector(first, theta + .5 * delta)
        x = np.array([[-.2], [.3]])
        ensemble = .5 * outputs(first, x)['tau'] + .5 * outputs(second, x)['tau']
        self.assertGreater(np.max(np.abs(ensemble - outputs(parameter_average, x)['tau'])), .001)

    def test_subsample_keeps_full_holdout_permutation_donors(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        plan = np.tile(np.array([5, 4, 3, 2, 1, 0]), (2, 5, 1))
        actual = sampled_interventions(x, plan, np.array([0, 1]))
        np.testing.assert_array_equal(actual[2:4, 0], x[[5, 4], 0])
        np.testing.assert_array_equal(actual[2:4, 1], x[[0, 1], 1])
        self.assertEqual(actual.shape, (22, 2))

    def test_centering_separates_constant_prediction_shift(self):
        exact = np.array([-.1, .2, .8])
        report = error_moments(exact, exact + 2.)
        self.assertAlmostEqual(report['rmse'], 2.)
        self.assertAlmostEqual(report['centered_rmse'], 0.)
        combined = combine_moments([report, error_moments(exact, exact - 2.)], [.5, .5])
        self.assertAlmostEqual(combined['mean_error'], 0.)
        self.assertAlmostEqual(combined['centered_rmse'], 2.)

    def test_full_tiny_replay_and_read_only_validation(self):
        data, plans, sources = replay_fixture()
        with patch.object(em, 'fit_local', side_effect=AssertionError('replay must not train')), \
             patch.object(em, 'mean_categorical_fisher_diag', side_effect=AssertionError('replay must not recompute Fisher')), \
             patch.object(em, 'legacy_validation_diagnostic', side_effect=AssertionError('replay must not select iterates')):
            report = diagnose_fisher(configuration(), data, plans, sources)
        self.assertTrue(report['implementation_gate']['passed'])
        self.assertTrue(report['full_H_endpoint']['replay_matches_source'])
        self.assertEqual(len(report['parameter_path']['target_weighted_points']), 9)
        self.assertIn('target_weighted_prediction_average', report['model_quality']['test'])
        with patch('torch.nn.Module._call_impl', side_effect=AssertionError('validator cannot infer')):
            validate_fisher_diagnostic(report, configuration(), sources)
        broken = deepcopy(report)
        broken['gradient_checks'][0]['checks'][0]['finite_difference'][0][0] += .1
        with self.assertRaises(ValueError):
            validate_fisher_diagnostic(broken, configuration(), sources)
        broken = deepcopy(report)
        broken['parameter_path']['target_weighted_points'][0]['score_vectors']['exact'][0] += 1.
        with self.assertRaises(ValueError):
            validate_fisher_diagnostic(broken, configuration(), sources)
        broken = deepcopy(report)
        broken['full_H_endpoint']['linearization']['gate']['compression_eligible'] = False
        with self.assertRaises(ValueError):
            validate_fisher_diagnostic(broken, configuration(), sources)
        # Preserve each model's elementary moment identities while breaking
        # the cross-model/checkpoint identities checked only by the replay.
        for case in ('zero_mean', 'zero_variance', 'zero_rmse', 'truth_variance', 'ensemble_mean', 'saved_variance'):
            with self.subTest(quality_tamper=case):
                broken = deepcopy(report)
                tests = broken['model_quality']['test']
                if case in ('zero_mean', 'ensemble_mean'):
                    value = tests['constant_zero_CATE' if case == 'zero_mean' else 'target_weighted_prediction_average']
                    value['predicted_mean'] += .1
                    value['prediction_mean_bias'] += .1
                    value['cate_rmse'] = float(np.sqrt(value['centered_cate_rmse'] ** 2 + value['prediction_mean_bias'] ** 2))
                elif case in ('zero_variance', 'saved_variance'):
                    value = tests['constant_zero_CATE' if case == 'zero_variance' else 'strict_FedFisher']
                    value['estimated_heterogeneity'] += .1
                    value['estimated_standard_deviation'] = float(np.sqrt(value['estimated_heterogeneity']))
                elif case == 'zero_rmse':
                    value = tests['constant_zero_CATE']
                    value['centered_cate_rmse'] += .1
                    value['cate_rmse'] = float(np.sqrt(value['centered_cate_rmse'] ** 2 + value['prediction_mean_bias'] ** 2))
                else:
                    value = tests['local_site_000']
                    value['true_heterogeneity'] += .1
                    value['true_standard_deviation'] = float(np.sqrt(value['true_heterogeneity']))
                with self.assertRaises(ValueError):
                    validate_fisher_diagnostic(broken, configuration(), sources)

    def test_legacy_replay_has_no_fabricated_cate_truth(self):
        data, plans, sources = replay_fixture(legacy=True)
        report = diagnose_fisher(configuration(), data, plans, sources)
        validate_fisher_diagnostic(report, configuration(), sources)
        for value in report['model_quality']['test'].values():
            self.assertIsNone(value['cate_rmse'])
            self.assertFalse(value['true_cate_available'])

    def test_bad_stable_gradient_stops_diagnostics_and_remains_a_valid_finding(self):
        data, plans, sources = replay_fixture()
        original = local_derivatives
        def incorrect(*args, **kwargs):
            baseline, derivative = original(*args, **kwargs)
            derivative[:, :, :] += .1
            return baseline, derivative
        with patch('effect_modifier_experiment.one_shot_effects.diagnostics.fisher_metrics.local_derivatives', side_effect=incorrect):
            report = diagnose_fisher(configuration(), data, plans, sources)
        self.assertFalse(report['implementation_gate']['passed'])
        self.assertIsNone(report['parameter_path'])
        self.assertIsNone(report['full_H_endpoint'])
        self.assertIsNone(report['model_quality'])
        validate_fisher_diagnostic(report, configuration(), sources)


if __name__ == '__main__':
    unittest.main()
