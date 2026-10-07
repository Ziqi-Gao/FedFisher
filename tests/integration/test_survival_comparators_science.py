"""Tiny observed fixtures test paired comparator composition and saved-only checks."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
import torch

from effect_modifier_experiment.survival.artifacts import array_hash, dataset_fingerprints
from effect_modifier_experiment.survival.data import (
    SurvivalConfig, SurvivalDataset, SurvivalRows, SurvivalSite, survival_truth,
)
from effect_modifier_experiment.survival.dslasso import SurvivalLinearModel
from effect_modifier_experiment.survival.evaluation import evaluate_model
from effect_modifier_experiment.survival.pseudo import site_pseudo_response
from effect_modifier_experiment.components.fingerprints import array_hash as numeric_array_hash
from effect_modifier_experiment.survival_mlp8.methods import public_initial_vector
from effect_modifier_experiment.survival_pooled.evaluation import probability_predictions
from effect_modifier_experiment.survival_comparators import artifacts, computation, fedavg, inputs, protocol


def fixture():
    """Handwritten records and a known linear predictor, with no dataset generation."""
    cfg = SurvivalConfig(num_sites=2, site_min_samples=4, site_max_samples=5,
                         covariate_dim=3, num_test=5)
    x = np.array([[-1., .2, .7], [.4, -.3, -.1], [.5, .6, .2], [-.2, -.7, .3], [.1, .3, -.4]])
    sites = tuple(SurvivalSite(i, SurvivalRows(x[:n] + .1 * i, np.arange(n) % 2,
        np.array([1., 2., 4., 5., 6.])[:n], np.array([1, 1, 0, 1, 0])[:n],
        np.arange(10 * i, 10 * i + n))) for i, n in enumerate((4, 5)))
    observed = SurvivalRows(x * .8, np.array([1., 0., 1., 0., 1.]),
        np.array([2., 3., 4., 5., 6.]), np.array([1, 0, 1, 0, 1]), np.arange(100, 105))
    data = SurvivalDataset(sites, observed, survival_truth(observed.x, cfg), np.array([4., 5.]) / 9,
                           cfg, 0, 'iid')
    model = SurvivalLinearModel(-.2, np.array([.15, .12, -.04, .02, .6, -.4, 0.]))
    config = protocol.run_config(protocol.PILOT_TASK, dict(scenario='iid', seed=0))
    config['dgp'] = asdict(cfg)
    fingerprints = dataset_fingerprints(data)
    reference = dict(identity=dict(seed=0, scenario='iid', cell='dslasso'))
    linear = dict(model=model, fingerprints=dict(fingerprints, model='3' * 64), binding=reference,
        result=dict(site_sizes=[4, 5], evaluation=evaluate_model(model.predict_eta, data, seed=0),
                    fit=dict(method='dslassocov_style_survival_nls_lasso', fixture=True),
                    timing=dict(stages=dict(fit=dict(wall_seconds=9., cpu_seconds=7.)))))
    arrays = {'truth__' + key: np.asarray(getattr(data.evaluation_truth, key)).copy()
              for key in artifacts.TRUTH_FIELDS}
    arrays.update(factual_a=np.asarray(observed.a, dtype=np.float64), discovery__x=observed.x.copy())
    for scope in protocol.PSEUDO_SCOPES:
        # These predictions are supplied as saved controls, not fitted baselines.
        prediction = probability_predictions(model.predict_eta, data)
        arrays.update({'pooled_' + scope + '__prediction__' + key: value for key, value in prediction.items()})
    arrays['fedfisher_diag__fit__initial_vector'] = public_initial_vector(7,
        fedavg.FedAvgConfig().local_config(), 0, config['public_initial_eta'])
    local = [dict(training_pseudo_sha256=numeric_array_hash(site_pseudo_response(site.rows, cfg.tau)))
             for site in sites]
    baseline = dict(result=dict(site_sizes=[4, 5], fit=dict(fedfisher_diag=dict(local=local))), arrays=arrays, config=config,
                    fingerprints=deepcopy(fingerprints), binding=dict(accepted_mlp8_fixture=True))
    return config, data, linear, baseline


class SurvivalComparatorScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config, self.data, self.linear, self.baseline = fixture()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (computation, artifacts):
            self.stack.enter_context(patch.object(module, 'load_historical',
                return_value={'dslasso': self.linear}))
        self.stack.enter_context(patch.object(inputs, 'load_mlp8_reference', return_value=self.baseline))
        self.stack.enter_context(patch.object(artifacts, 'selected_fingerprints',
            return_value=deepcopy(self.baseline['fingerprints'])))
        self.generate = self.stack.enter_context(patch.object(computation, 'generate_survival', return_value=self.data))
        self.identity = dict(run_id='survival_comparators_v1_iid_10seeds', config_sha256='1' * 64,
            provenance_sha256='2' * 64, seed=0, cell='comparators', scenario='iid')

    def completed(self):
        result, arrays = computation.compute_cell(self.config, 0)
        result['identity'] = self.identity
        return result, arrays

    def test_only_fedavg_is_fitted_probability_precedes_discovery_and_ds_scores_are_reused(self):
        events = []
        fit = computation.fit_fedavg
        probability = computation.probability_predictions
        discover = computation.evaluate_model

        def fitting(*args):
            self.assertIs(args[0], self.data.sites)
            self.assertEqual(args[1:3], (3., 0))
            self.assertTrue(all(site.rows is not self.data.evaluation for site in args[0]))
            events.append('fit')
            return fit(*args)

        def probabilities(predict, dataset):
            self.assertIs(dataset, self.data)
            events.append('probability')
            return probability(predict, dataset)

        def discovering(*args, **kwargs):
            self.assertEqual(events, ['fit', 'probability', 'probability'])
            events.append('discovery')
            return discover(*args, **kwargs)

        with patch.object(computation, 'fit_fedavg', side_effect=fitting) as fitted, \
             patch.object(computation, 'probability_predictions', side_effect=probabilities), \
             patch.object(computation, 'evaluate_model', side_effect=discovering) as discovered, \
             patch.object(SurvivalLinearModel, 'predict_eta', wraps=self.linear['model'].predict_eta) as linear_predictions:
            result, arrays = self.completed()
        self.assertEqual(fitted.call_count, 1)
        self.assertEqual(discovered.call_count, 1)
        self.assertEqual(linear_predictions.call_count, 2)  # One batch per arm; no permutation replay.
        self.assertEqual(result['fit_origin'], artifacts.FIT_ORIGIN)
        self.assertEqual(result['discovery']['dslasso'], self.linear['result']['evaluation'])
        self.assertEqual(result['fit']['dslasso'], self.linear['result']['fit'])
        self.assertEqual(result['timing']['reused_dslasso_fit'], dict(wall_seconds=9., cpu_seconds=7.))
        self.assertEqual(result['fit']['fedavg']['training_rounds_executed'], 6)
        self.assertEqual(result['fit']['fedavg']['total_local_epochs_per_site'], 30)
        self.assertEqual(result['prediction_arrays_sha256'], array_hash(arrays))
        self.assertTrue(artifacts.validate_result(result, arrays, self.identity, self.config))

    def test_each_accepted_cohort_mismatch_stops_before_fitting_or_inference(self):
        for reference in (self.linear, self.baseline):
            for key in ('training', 'evaluation', 'evaluation_truth'):
                original = reference['fingerprints'][key]
                reference['fingerprints'][key] = '0' * 64
                with self.subTest(cohort=reference['binding'], key=key), \
                     patch.object(computation, 'fit_fedavg', side_effect=AssertionError('fit')), \
                     patch.object(computation, 'probability_predictions', side_effect=AssertionError('inference')), \
                     self.assertRaisesRegex(ValueError, 'data differ'):
                    computation.compute_cell(self.config, 0)
                reference['fingerprints'][key] = original
            reference['result']['site_sizes'] = [5, 4]
            with patch.object(computation, 'fit_fedavg', side_effect=AssertionError('fit')), \
                 self.assertRaisesRegex(ValueError, 'site sizes differ'):
                computation.compute_cell(self.config, 0)
            reference['result']['site_sizes'] = [4, 5]

    def test_saved_validation_has_no_fitting_prediction_generator_rng_or_permutation_replay(self):
        result, arrays = self.completed()
        with patch.object(computation, 'fit_fedavg', side_effect=AssertionError('fit')), \
             patch.object(fedavg, 'fit_fedavg', side_effect=AssertionError('fit')), \
             patch.object(computation, 'generate_survival', side_effect=AssertionError('generator')), \
             patch('effect_modifier_experiment.survival.data.generate_survival', side_effect=AssertionError('generator')), \
             patch('effect_modifier_experiment.survival_mlp8.methods.MLP8Fit.predict_eta', side_effect=AssertionError('inference')), \
             patch.object(SurvivalLinearModel, 'predict_eta', side_effect=AssertionError('inference')), \
             patch('effect_modifier_experiment.survival.evaluation.evaluate_model', side_effect=AssertionError('permutations')), \
             patch('numpy.random.default_rng', side_effect=AssertionError('RNG')):
            self.assertTrue(artifacts.validate_result(result, arrays, self.identity, self.config))

    def test_initialization_and_local_pseudo_mismatch_stop_before_prediction(self):
        expected = self.baseline['arrays']['fedfisher_diag__fit__initial_vector'].copy()
        self.baseline['arrays']['fedfisher_diag__fit__initial_vector'][0] += .1
        with patch.object(computation, 'probability_predictions', side_effect=AssertionError('inference')), \
             self.assertRaisesRegex(ValueError, 'public initialization'):
            computation.compute_cell(self.config, 0)
        self.baseline['arrays']['fedfisher_diag__fit__initial_vector'] = expected
        self.baseline['result']['fit']['fedfisher_diag']['local'][0]['training_pseudo_sha256'] = '0' * 64
        with patch.object(computation, 'probability_predictions', side_effect=AssertionError('inference')), \
             self.assertRaisesRegex(ValueError, 'local pseudo-values'):
            computation.compute_cell(self.config, 0)

    def test_rehashed_fit_inventory_dtype_and_saved_cost_tampering_fail_closed(self):
        result, arrays = self.completed()
        altered = {key: value.copy() for key, value in arrays.items()}
        altered['fedavg__fit__initial_vector'] = altered['fedavg__fit__initial_vector'].astype(np.float32)
        changed = deepcopy(result)
        changed['prediction_arrays_sha256'] = array_hash(altered)
        with self.assertRaises(ValueError):
            artifacts.validate_result(changed, altered, self.identity, self.config)
        for section, mutation in (('timing', lambda r: r['timing']['stages']['fit'].update(wall_seconds=0.)),
                                  ('memory', lambda r: r['memory'].update(scope='scheduler allocation')),
                                  ('references', lambda r: r['references']['dslasso']['identity'].update(seed=1))):
            changed = deepcopy(result)
            mutation(changed)
            with self.subTest(section=section), self.assertRaises(ValueError):
                artifacts.validate_result(changed, arrays, self.identity, self.config)

    def test_rehashed_evaluator_probability_and_replayed_model_tampering_fails_closed(self):
        result, arrays = self.completed()
        for name in ('discovery__x', 'factual_a', 'truth__eta0', 'dslasso__fit__intercept',
                     'dslasso__fit__coefficients', 'fedavg__prediction__s0', 'fedavg__prediction__eta1'):
            changed = {key: value.copy() for key, value in arrays.items()}
            changed[name].flat[0] += .1
            mutated = deepcopy(result)
            mutated['prediction_arrays_sha256'] = array_hash(changed)
            with self.subTest(array=name), self.assertRaises(ValueError):
                artifacts.validate_result(mutated, changed, self.identity, self.config)
        changed = dict(arrays, extra=np.array([0.]))
        mutated = deepcopy(result)
        mutated['prediction_arrays_sha256'] = array_hash(changed)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            artifacts.validate_result(mutated, changed, self.identity, self.config)

    def test_modified_discovery_ties_streams_replay_fit_and_timing_are_rejected(self):
        result, arrays = self.completed()
        for tag in protocol.TAGS:
            for field, value in (('permutation_sha256', '0' * 64), ('estimated_contrast_sha256', '0' * 64),
                                 ('contrast_rmse', 0.)):
                changed = deepcopy(result)
                changed['discovery'][tag][field] = value
                with self.subTest(tag=tag, field=field), self.assertRaises(ValueError):
                    artifacts.validate_result(changed, arrays, self.identity, self.config)
        changed = deepcopy(result)
        changed['discovery']['fedavg']['recovery']['probability_all_signals_selected'] = .17
        with self.assertRaises(ValueError):
            artifacts.validate_result(changed, arrays, self.identity, self.config)
        for name in ('fit_origin', 'fit', 'timing'):
            changed = deepcopy(result)
            if name == 'fit_origin':
                changed[name]['dslasso'] = 'new_training'
            elif name == 'fit':
                changed[name]['dslasso']['fixture'] = False
            else:
                changed[name]['reused_dslasso_fit']['wall_seconds'] = 0.
            with self.subTest(section=name), self.assertRaises(ValueError):
                artifacts.validate_result(changed, arrays, self.identity, self.config)

    def test_changed_saved_pooled_predictions_invalidate_gap_and_invalid_seed_loads_nothing(self):
        result, arrays = self.completed()
        self.baseline['arrays']['pooled_site_local_pv__prediction__eta0'][0] += .1
        with self.assertRaises(ValueError):
            artifacts.validate_result(result, arrays, self.identity, self.config)
        for seed in (True, -1, 10):
            with patch.object(computation, 'load_historical', side_effect=AssertionError('load')), \
                 self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'outside'):
                computation.compute_cell(self.config, seed)


if __name__ == '__main__':
    unittest.main()
