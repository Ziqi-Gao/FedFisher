"""Adversarial, one-thread fixtures for the original pilot gradient gate.

These construct tiny rows/checkpoints in memory, never regenerate pilot data,
fit a model, or read a scientific checkpoint from disk.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from effect_modifier_experiment.one_shot_effects import fedfisher_em as em
from effect_modifier_experiment.one_shot_effects import fisher_summary
from effect_modifier_experiment.one_shot_effects.diagnostics import fisher
from effect_modifier_experiment.one_shot_effects.diagnostics.fisher_metrics import (
    cate_metrics, gradient_check)
from effect_modifier_experiment.one_shot_effects.diagnostics.fisher_validation import (
    validate_fisher_diagnostic)
from effect_modifier_experiment.one_shot_effects.evaluation import summarize_linearization
from effect_modifier_experiment.one_shot_effects.fisher_models import (
    OutcomeEffectModel, array_hash, model_from_vector, model_vector)
from tests.unit.test_effects_replay_fisher import configuration, replay_fixture, tiny_relu


OUTPUTS = ('control_binary_log_odds', 'treated_binary_log_odds', 'probability_CATE')


def boundary_replay_fixture():
    """Put a first-layer treated unit exactly at zero for every fixed H row."""
    data, plans, sources = replay_fixture()
    source = sources['aggregation']
    locals_ = fisher.restore_locals(source, data)
    for local in locals_:
        first = next(layer for layer in local.model.modules() if isinstance(layer, nn.Linear))
        with torch.no_grad():
            first.weight[0].zero_()
            first.weight[0, 0] = .5
            first.bias[0] = -.5
        local.theta = model_vector(local.model)
        local.report['theta_sha256'] = array_hash(local.theta)
    theta = np.mean([local.theta for local in locals_], axis=0)
    model = OutcomeEffectModel('SyntheticMLP', 2, model_from_vector('SyntheticMLP', 2, theta))
    source['payloads'] = [em.build_training_payload(local, site.n_total, site.site_id)
                          for local, site in zip(locals_, data.sites)]
    source['models'] = {name: deepcopy(model) for name in source['models']}
    local_diagnostics = [em.full_gradient_diagnostics(local, theta, site.holdout.x, plan)
                         for local, site, plan in zip(locals_, data.sites, plans)]
    source['result']['diagnostics'] = dict(local_linearization=local_diagnostics,
        linearization=summarize_linearization(local_diagnostics, [.5, .5], [True, True],
                                              source['config']['linearity_gate']))
    saved_test = cate_metrics(model.predict_tau(data.evaluation.rows.x), data.evaluation)
    source['result']['models'] = {name: dict(test={key: saved_test[key]
        for key in ('cate_rmse', 'estimated_heterogeneity')}) for name in source['models']}
    return data, plans, dict(aggregation=source, linearity=deepcopy(source))


class OriginalPilotGradientRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_stopped(self, report, status):
        self.assertFalse(report['implementation_gate']['passed'])
        self.assertEqual(report['implementation_gate']['status'], status)
        for key in ('parameter_path', 'full_H_endpoint', 'model_quality'):
            self.assertIsNone(report[key])
        self.assertFalse(report['downstream_training_allowed_by_this_replay'])

    def check_model(self, model, x, actual, random):
        return gradient_check(model, x, actual, random, [1e-4, 1e-5, 1e-6], 1e-7, 1e-4)

    def test_original_gradient_doubled_in_reference_and_replay_is_detected(self):
        original = fisher_summary.tau_and_directional_derivatives

        def doubled(*args, **kwargs):
            tau, derivative = original(*args, **kwargs)
            return tau, 2 * derivative

        # Patch the alias actually called by fisher_summary.diagnostic_scores,
        # including when the supposedly reassuring saved reference is created.
        with patch.object(fisher_summary, 'tau_and_directional_derivatives', side_effect=doubled):
            data, plans, sources = replay_fixture()
            locals_ = fisher.restore_locals(sources['aggregation'], data)
            theta = model_vector(sources['aggregation']['models']['primary'].model)
            self.assertTrue(fisher.full_holdout_endpoint(locals_, theta, data, plans,
                sources['linearity'], configuration())['replay_matches_source'])
            report = fisher.diagnose_fisher(configuration(), data, plans, sources)
        self.assert_stopped(report, 'failed')
        self.assertEqual(report['implementation_gate']['stable_failures'], 0)
        self.assertTrue(all(record['original_pilot']['tau_passed'] for record in report['gradient_checks']))
        self.assertTrue(any(not record['original_pilot']['derivative_passed']
                            for record in report['gradient_checks']))
        self.assertGreater(sum(record['original_pilot']['stable_fd_failures']
                               for record in report['gradient_checks']), 0)
        with patch('torch.nn.Module._call_impl', side_effect=AssertionError('validation must not infer')):
            validate_fisher_diagnostic(report, configuration(), sources)
        forged = deepcopy(report)
        forged['implementation_gate'].update(passed=True, status='passed')
        for record in forged['gradient_checks']:
            record.update(passed=True, status='passed')
            record['original_pilot'].update(passed=True, derivative_passed=True)
        with self.assertRaises(ValueError):
            validate_fisher_diagnostic(forged, configuration(), sources)

    def test_original_tau_corruption_is_detected_even_with_correct_derivative(self):
        original = fisher_summary.tau_and_directional_derivatives

        def shifted(*args, **kwargs):
            tau, derivative = original(*args, **kwargs)
            return tau + .02, derivative

        with patch.object(fisher_summary, 'tau_and_directional_derivatives', side_effect=shifted):
            data, plans, sources = replay_fixture()
            report = fisher.diagnose_fisher(configuration(), data, plans, sources)
        self.assert_stopped(report, 'failed')
        self.assertTrue(all(not record['original_pilot']['tau_passed'] for record in report['gradient_checks']))
        self.assertTrue(all(record['original_pilot']['derivative_passed'] for record in report['gradient_checks']))
        validate_fisher_diagnostic(report, configuration(), sources)

    def test_control_only_finite_differences_cannot_pass_cate_audit(self):
        model = tiny_relu(bias=-.5)
        actual = np.zeros_like(model_vector(model))
        actual[3] = 1.  # Treated preactivation is exactly zero at both x=0 rows.
        report = self.check_model(model, np.zeros((2, 1)), actual, actual)
        self.assertEqual(report['stable_checks'], 12)
        self.assertEqual(report['stable_failures'], 0)
        self.assertFalse(report['passed'])
        self.assertEqual(report['status'], 'inconclusive')
        self.assertFalse(report['coverage_complete'])
        for direction in ('actual_aggregation', 'fixed_random'):
            coverage = report['coverage'][direction]
            self.assertEqual(coverage['status'], 'inconclusive')
            self.assertEqual(coverage['stable_checks_by_output'], dict(zip(OUTPUTS, [6, 0, 0])))
            self.assertEqual(coverage['missing_outputs'], list(OUTPUTS[1:]))
        self.assertEqual(report['original_pilot']['stable_fd_checks'], 0)

    def test_random_direction_coverage_cannot_rescue_actual_direction(self):
        model = tiny_relu(bias=-.5 + 1e-7)
        actual = np.zeros_like(model_vector(model))
        actual[3] = 1.  # All three actual-direction h values cross the tiny margin.
        random = np.zeros_like(actual)
        random[-1] = 1.  # An output-bias move leaves the ReLU activation stable.
        report = self.check_model(model, np.zeros((2, 1)), actual, random)
        self.assertEqual(report['stable_failures'], 0)
        self.assertFalse(report['passed'])
        self.assertEqual(report['status'], 'inconclusive')
        self.assertEqual(report['coverage']['actual_aggregation']['missing_outputs'], list(OUTPUTS[1:]))
        self.assertEqual(report['coverage']['fixed_random']['status'], 'complete')
        self.assertEqual(report['coverage']['fixed_random']['stable_checks_by_output'],
                         dict(zip(OUTPUTS, [6, 6, 6])))

    def test_zero_actual_direction_is_excluded_without_vacuous_coverage(self):
        model = tiny_relu()
        zero = np.zeros_like(model_vector(model))
        random = np.linspace(-.2, .3, len(zero))
        report = self.check_model(model, np.array([[-.2], [.3]]), zero, random)
        self.assertTrue(report['passed'])
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['stable_checks'], 18)
        self.assertEqual(report['boundary_checks'], 0)
        coverage = report['coverage']['actual_aggregation']
        self.assertEqual(coverage['status'], 'not_applicable_zero_direction')
        self.assertFalse(coverage['required'])
        self.assertEqual(coverage['stable_checks_by_output'], dict.fromkeys(OUTPUTS, 0))
        self.assertEqual(coverage['missing_outputs'], [])
        self.assertEqual(report['coverage']['fixed_random']['status'], 'complete')

    def test_zero_random_direction_is_inconclusive_despite_actual_coverage(self):
        model = tiny_relu()
        zero = np.zeros_like(model_vector(model))
        actual = np.linspace(-.2, .3, len(zero))
        report = self.check_model(model, np.array([[-.2], [.3]]), actual, zero)
        self.assertEqual(report['stable_checks'], 18)
        self.assertEqual(report['stable_failures'], 0)
        self.assertEqual(report['boundary_checks'], 0)
        self.assertFalse(report['passed'])
        self.assertEqual(report['status'], 'inconclusive')
        coverage = report['coverage']['fixed_random']
        self.assertTrue(coverage['required'])
        self.assertEqual(coverage['status'], 'inconclusive')
        self.assertEqual(coverage['stable_checks_by_output'], dict.fromkeys(OUTPUTS, 0))
        self.assertEqual(coverage['missing_outputs'], list(OUTPUTS))

    def test_saved_replay_zero_actual_direction_is_not_counted_as_validation(self):
        data, plans, sources = replay_fixture()
        source = sources['aggregation']
        locals_ = fisher.restore_locals(source, data)
        theta = locals_[0].theta.copy()
        model = OutcomeEffectModel('SyntheticMLP', 2, deepcopy(locals_[0].model))
        source['models']['primary'] = model
        saved_test = cate_metrics(model.predict_tau(data.evaluation.rows.x), data.evaluation)
        source['result']['models']['primary'] = dict(test={key: saved_test[key]
            for key in ('cate_rmse', 'estimated_heterogeneity')})
        local_diagnostics = [em.full_gradient_diagnostics(local, theta, site.holdout.x, plan)
                             for local, site, plan in zip(locals_, data.sites, plans)]
        source['result']['diagnostics'] = dict(local_linearization=local_diagnostics,
            linearization=summarize_linearization(local_diagnostics, [.5, .5], [True, True],
                                                  source['config']['linearity_gate']))
        sources = dict(aggregation=source, linearity=deepcopy(source))
        report = fisher.diagnose_fisher(configuration(), data, plans, sources)
        self.assertTrue(report['implementation_gate']['passed'])
        self.assertTrue(report['full_H_endpoint']['replay_matches_source'])
        record = report['gradient_checks'][0]
        coverage = record['coverage']['actual_aggregation']
        self.assertEqual(coverage['status'], 'not_applicable_zero_direction')
        self.assertEqual(coverage['stable_checks_by_output'], dict.fromkeys(OUTPUTS, 0))
        self.assertEqual(record['stable_checks'], 18)
        with patch('torch.nn.Module._call_impl', side_effect=AssertionError('validation must not infer')):
            validate_fisher_diagnostic(report, configuration(), sources)
        for mutation in ('counts', 'status', 'required'):
            with self.subTest(zero_actual_tamper=mutation):
                forged = deepcopy(report)
                coverage = forged['gradient_checks'][0]['coverage']['actual_aggregation']
                if mutation == 'counts':
                    coverage['stable_checks_by_output']['probability_CATE'] = 6
                elif mutation == 'status':
                    coverage['status'] = 'complete'
                else:
                    coverage['required'] = True
                with self.assertRaises(ValueError):
                    validate_fisher_diagnostic(forged, configuration(), sources)

    def test_inconclusive_boundary_replay_stops_and_remains_valid_saved_finding(self):
        data, plans, sources = boundary_replay_fixture()
        with patch.object(fisher, 'parameter_path', side_effect=AssertionError('inconclusive gate must stop paths')), \
             patch.object(fisher, 'full_holdout_endpoint', side_effect=AssertionError('inconclusive gate must stop full H')), \
             patch.object(fisher, 'model_quality', side_effect=AssertionError('inconclusive gate must stop quality')):
            report = fisher.diagnose_fisher(configuration(), data, plans, sources)
        self.assert_stopped(report, 'inconclusive')
        self.assertEqual(report['implementation_gate']['stable_failures'], 0)
        self.assertFalse(report['implementation_gate']['coverage_complete'])
        with patch('torch.nn.Module._call_impl', side_effect=AssertionError('validation must not infer')):
            validate_fisher_diagnostic(report, configuration(), sources)
        for field, value in (('status', 'passed'), ('coverage_complete', True), ('passed', True)):
            with self.subTest(forged_global_field=field):
                forged = deepcopy(report)
                forged['implementation_gate'][field] = value
                with self.assertRaises(ValueError):
                    validate_fisher_diagnostic(forged, configuration(), sources)

    def test_saved_coverage_cannot_disagree_with_observed_masks(self):
        data, plans, sources = replay_fixture()
        report = fisher.diagnose_fisher(configuration(), data, plans, sources)
        self.assertTrue(report['implementation_gate']['passed'])
        for mutation in ('counts', 'missing', 'status', 'original_projection', 'original_fd_mask'):
            with self.subTest(mutation=mutation):
                forged = deepcopy(report)
                record = forged['gradient_checks'][0]
                coverage = record['coverage']['fixed_random']
                if mutation == 'counts':
                    coverage['stable_checks_by_output']['probability_CATE'] += 1
                elif mutation == 'missing':
                    coverage['missing_outputs'] = ['probability_CATE']
                elif mutation == 'status':
                    coverage['status'] = 'inconclusive'
                elif mutation == 'original_projection':
                    record['original_pilot']['directional_derivatives'][0][0] += .1
                else:
                    record['checks'][0]['original_pilot_within_fd_tolerance'][0] = False
                with self.assertRaises(ValueError):
                    validate_fisher_diagnostic(forged, configuration(), sources)


if __name__ == '__main__':
    unittest.main()
