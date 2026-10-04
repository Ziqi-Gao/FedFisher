"""Tiny orchestration fixtures with mocked trainers, models and scheduler storage."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import os
import unittest
from unittest import mock

import numpy as np
import torch

from effect_modifier_experiment.survival.artifacts import array_hash, dataset_fingerprints
from effect_modifier_experiment.survival.data import (
    SurvivalConfig, SurvivalDataset, SurvivalRows, SurvivalSite, survival_truth,
)
from effect_modifier_experiment.survival_pooled import computation, inputs, protocol, runner
from effect_modifier_experiment.survival_pooled.evaluation import compare_predictions


class AnalyticPredictor:
    """An explicit fixture function; no fitted model is loaded or applied."""
    def __init__(self, offset=0., events=None, name='fixture'):
        self.offset, self.events, self.name = offset, events, name
        self.calls = []

    def predict_eta(self, design):
        design = np.asarray(design)
        self.calls.append(design.copy())
        if self.events is not None:
            self.events.append('prediction:' + self.name)
        return self.offset + .2 * design[:, 1] + .5 * design[:, 0] + .4 * design[:, 3]


def manual_dataset():
    """Handwritten observed arrays and closed-form truth, without a generator."""
    cfg = SurvivalConfig(num_sites=2, site_min_samples=4, site_max_samples=4,
                          covariate_dim=2, num_test=4)
    sites = tuple(SurvivalSite(i, SurvivalRows(
        np.array([[-1., .2], [.4, -.3], [.5, .6], [-.2, -.7]]) + .1 * i,
        np.array([0., 1., 0., 1.]), np.array([1., 2., 4., 5.]), np.array([1, 1, 0, 1]),
        np.arange(4 * i, 4 * i + 4))) for i in range(2))
    evaluation = SurvivalRows(np.array([[-.7, .1], [.2, -.4], [.8, .3], [-.5, -.2]]),
        np.array([1., 0., 1., 0.]), np.array([2., 3., 4., 5.]), np.array([1, 0, 1, 0]),
        np.arange(100, 104))
    return SurvivalDataset(sites, evaluation, survival_truth(evaluation.x, cfg),
                           np.array([.5, .5]), cfg, 0, 'iid')


class SurvivalPooledComputationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = manual_dataset()
        self.config = protocol.run_config(protocol.PILOT_TASK, dict(family='shallow', scenario='iid'))
        self.config['dgp'] = asdict(self.dataset.config)
        self.events = []
        fingerprints = dataset_fingerprints(self.dataset)
        self.models = {tag: AnalyticPredictor(.05 * index, self.events, tag)
                       for index, tag in enumerate(protocol.historical_tags('shallow'))}
        self.historical = {tag: dict(model=model, fingerprints=deepcopy(fingerprints),
            result=dict(site_sizes=[4, 4]), binding=dict(identity=dict(seed=0, scenario='iid',
                cell=tag + '_shallow'))) for tag, model in self.models.items()}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.load = self.stack.enter_context(mock.patch.object(computation, 'load_historical',
            side_effect=lambda *args: self.events.append('historical') or self.historical))
        self.generate = self.stack.enter_context(mock.patch.object(computation, 'generate_survival',
            side_effect=lambda *args: self.events.append('dataset') or self.dataset))
        self.fit = self.stack.enter_context(mock.patch.object(computation, 'fit_pooled', side_effect=self.fake_fit))

    def fake_fit(self, *args):
        sites, tau, seed, family, scope, method, initial_eta = args
        self.assertIs(sites, self.dataset.sites)
        self.assertEqual((tau, seed, family), (3., 0, 'shallow'))
        self.assertIs(method, self.config['method'])
        self.assertEqual(initial_eta, self.config['public_initial_eta'])
        self.assertNotIn(id(self.dataset.evaluation), [id(value) for value in args])
        self.assertNotIn(id(self.dataset.evaluation_truth), [id(value) for value in args])
        for site in sites:
            self.assertIsInstance(site.rows, SurvivalRows)
            self.assertFalse(np.intersect1d(site.rows.row_ids, self.dataset.evaluation.row_ids).size)
        self.events.append('fit:' + scope)
        model = AnalyticPredictor(.1 if scope == 'site_local_pv' else -.1, self.events, 'pooled_' + scope)
        self.models['pooled_' + scope] = model
        return model, dict(family=family, pseudo_scope=scope, seed=seed, total_fit_seconds=0.), {
            'vector': np.array([1., 2.], dtype=np.float64)}

    def test_complete_pipeline_passes_only_observed_training_and_saves_all_evidence(self):
        result, arrays = computation.compute_cell(self.config, 0)
        self.load.assert_called_once_with('shallow', 'iid', 0)
        self.generate.assert_called_once_with(self.dataset.config, 0, 'iid')
        self.assertEqual(self.fit.call_count, 2)
        self.assertEqual(self.events[:4], ['historical', 'dataset', 'fit:site_local_pv', 'fit:pooled_km_pv'])
        self.assertTrue(all(event.startswith('prediction:') for event in self.events[4:]))
        self.assertEqual(set(result['timing']['stages']), {
            'historical_input_validation', 'data_preparation', 'fit', 'probability_evaluation'})
        for stage in result['timing']['stages'].values():
            self.assertGreaterEqual(stage['wall_seconds'], 0)
            self.assertGreaterEqual(stage['cpu_seconds'], 0)
        self.assertEqual(result['fingerprints'], dataset_fingerprints(self.dataset))
        self.assertEqual(result['historical_inputs'], inputs.historical_binding())
        self.assertEqual(result['prediction_arrays_sha256'], array_hash(arrays))
        fields = ('propensity', 'eta0', 'eta1', 'survival0', 'survival1', 'delta', 'modifier_mask')
        truth = {key: arrays['truth__' + key] for key in fields}
        self.assertEqual(array_hash(truth), result['fingerprints']['evaluation_truth'])
        for key in fields:
            np.testing.assert_array_equal(truth[key], getattr(self.dataset.evaluation_truth, key))
            self.assertEqual(truth[key].dtype, np.bool_ if key == 'modifier_mask' else np.float64)
        np.testing.assert_array_equal(arrays['factual_a'], self.dataset.evaluation.a)
        np.testing.assert_array_equal(arrays['plot_x12'], self.dataset.evaluation.x[:, :2])
        for tag, model in self.models.items():
            self.assertEqual(len(model.calls), 2)
            for call in model.calls:
                np.testing.assert_array_equal(call[:, 3:], call[:, :1] * call[:, 1:3])
            self.assertEqual(result['prediction'][tag]['primary'],
                             'oracle_survival_probability_accuracy_before_contrast')
            self.assertIs(result['prediction'][tag]['end_to_end_one_round_claim'], False)
            self.assertIn(tag + '__prediction__s0', arrays)
            self.assertIn(tag + '__prediction__s1', arrays)
        for tag in self.historical:
            self.assertEqual(result['historical'][tag], self.historical[tag]['binding'])
            for scope in self.config['pseudo_scopes']:
                left = {key: arrays[tag + '__prediction__' + key] for key in ('eta0', 'eta1', 's0', 's1')}
                right = {key: arrays['pooled_' + scope + '__prediction__' + key]
                         for key in ('eta0', 'eta1', 's0', 's1')}
                self.assertEqual(result['comparisons'][tag][scope], compare_predictions(left, right))

    def test_all_frozen_dataset_mismatches_fail_before_training_or_predictions(self):
        for key in ('training', 'evaluation', 'evaluation_truth'):
            with self.subTest(key=key):
                reference = self.historical['fedfisher_diag']['fingerprints'][key]
                self.historical['fedfisher_diag']['fingerprints'][key] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'data differ'):
                    computation.compute_cell(self.config, 0)
                self.historical['fedfisher_diag']['fingerprints'][key] = reference
        self.fit.assert_not_called()
        self.assertTrue(all(not model.calls for model in self.models.values()))

    def test_site_size_mismatch_fails_before_training_or_predictions(self):
        self.historical['fedfisher_kfac']['result']['site_sizes'] = [3, 5]
        with self.assertRaisesRegex(ValueError, 'site sizes differ'):
            computation.compute_cell(self.config, 0)
        self.fit.assert_not_called()
        self.assertTrue(all(not model.calls for model in self.models.values()))

    def test_invalid_historical_evidence_fails_before_dataset_preparation(self):
        self.load.side_effect = ValueError('invalid historical evidence')
        with self.assertRaisesRegex(ValueError, 'invalid historical'):
            computation.compute_cell(self.config, 0)
        self.generate.assert_not_called()
        self.fit.assert_not_called()

    def test_seed_outside_fixed_recipe_fails_before_loading_inputs(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            computation.compute_cell(self.config, 1)
        self.load.assert_not_called()
        self.generate.assert_not_called()


class SurvivalPooledRunnerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(torch, 'set_num_threads'))
        self.stack.enter_context(mock.patch.object(torch, 'set_num_interop_threads'))
        self.stack.enter_context(mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}))
        self.source = dict(git_commit='1' * 40, provenance_sha256='2' * 64)
        self.provenance = self.stack.enter_context(mock.patch.object(runner, 'current_provenance', return_value=self.source))
        self.gate = self.stack.enter_context(mock.patch.object(inputs, 'validate_historical_inputs', return_value={}))
        self.completed = set()
        self.store = mock.Mock()
        self.store.status.side_effect = lambda seed: 'validated' if seed in self.completed else 'missing'
        self.store.identity.side_effect = lambda seed: dict(seed=seed, fixture='pooled-runner')
        self.store.complete.side_effect = lambda result, arrays, seed: self.completed.add(seed)
        self.store_class = self.stack.enter_context(mock.patch.object(runner, 'ArtifactStore', return_value=self.store))
        self.compute = self.stack.enter_context(mock.patch.object(computation, 'compute_cell',
            side_effect=lambda config, seed: (dict(status='complete'), dict(vector=np.array([0.])))))
        self.job = dict(task=protocol.PILOT_TASK, parameters=dict(family='shallow', scenario='iid'),
            job_id='fedfisher-survival-pooled-runner-fixture', execution_profile='cpu-1core',
            allocation={}, estimated_runtime_seconds=10.)

    def test_runner_validates_inputs_before_publication_and_binds_seed_identity(self):
        events = []
        self.gate.side_effect = lambda: events.append('historical_gate')
        self.store.initialize.side_effect = lambda: events.append('initialize')
        self.assertEqual(runner.run_job(self.job), 0)
        self.assertEqual(events, ['historical_gate', 'initialize'])
        self.assertEqual([call.args[1] for call in self.compute.call_args_list], [0, 49])
        self.assertEqual(self.completed, {0, 49})
        for call in self.store.complete.call_args_list:
            result, arrays, seed = call.args
            self.assertEqual(result['identity'], dict(seed=seed, fixture='pooled-runner'))
        self.store.fail.assert_not_called()

    def test_invalid_startup_reference_does_not_create_store_or_compute(self):
        self.gate.side_effect = ValueError('changed frozen anchor')
        with self.assertRaisesRegex(ValueError, 'changed frozen'):
            runner.run_job(self.job)
        self.store_class.assert_not_called()
        self.compute.assert_not_called()

    def test_validated_units_are_skipped_without_recomputing(self):
        self.completed.update((0, 49))
        self.assertEqual(runner.run_job(self.job), 0)
        self.compute.assert_not_called()
        self.store.start.assert_not_called()
        self.store.complete.assert_not_called()

    def test_source_change_stops_before_starting_scientific_unit(self):
        self.provenance.side_effect = [self.source, dict(self.source, git_commit='9' * 40)]
        with self.assertRaisesRegex(ValueError, 'source changed'):
            runner.run_job(self.job)
        self.store.start.assert_not_called()
        self.compute.assert_not_called()

    def test_scientific_exception_records_failure_and_stops_shard(self):
        error = ValueError('fixture scientific failure')
        self.compute.side_effect = error
        with self.assertRaisesRegex(ValueError, 'fixture scientific'):
            runner.run_job(self.job)
        self.store.fail.assert_called_once_with(error, 0)
        self.store.complete.assert_not_called()
        self.assertEqual(self.compute.call_count, 1)

    def test_formal_requires_same_source_pilot_before_output_initialization(self):
        from effect_modifier_experiment.survival_pooled import acceptance
        self.job['task'] = protocol.FORMAL_TASK
        self.job['parameters'].update(shard_index=0, shard_count=16)
        with mock.patch.object(acceptance, 'default_acceptance_path', return_value='fixture'), \
                mock.patch.object(acceptance, 'verify_pilot_acceptance', side_effect=ValueError('missing pilot')):
            with self.assertRaisesRegex(ValueError, 'missing pilot'):
                runner.run_job(self.job)
        self.store_class.assert_not_called()
        self.compute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
