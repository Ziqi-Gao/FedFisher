"""Small actual MLP8 fits and saved-only, immutable lifecycle regression fixtures."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from effect_modifier_experiment.infrastructure.provenance import PROVENANCE_KEYS, canonical_text
from effect_modifier_experiment.survival import artifacts as original_artifacts
from effect_modifier_experiment.survival.data import (
    SurvivalConfig, SurvivalDataset, SurvivalRows, SurvivalSite, survival_truth,
)
from effect_modifier_experiment.survival_pooled import inputs
from effect_modifier_experiment.survival_mlp8 import artifacts, computation, methods, protocol, runner

SCRATCH = Path('/scr/del6500/FedFisher/survival-mlp8-validation/lifecycle-fixtures')


def provenance():
    value = dict(git_commit='1' * 40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                 source_tree_sha256='2' * 64)
    value['provenance_sha256'] = hashlib.sha256(canonical_text(value, PROVENANCE_KEYS).encode()).hexdigest()
    return value


def execution():
    return dict(job_id='fedfisher-survival-mlp8-lifecycle-fixture', execution_profile='cpu-1core',
        attempt=1, estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192,
        gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def dataset():
    """Handwritten observed records, with evaluation truth kept in a separate object."""
    config = SurvivalConfig(num_sites=2, site_min_samples=8, site_max_samples=8,
                            covariate_dim=2, num_test=4)
    x = np.array([[-1., 0.], [.1, .3], [.2, -.1], [.4, .5],
                  [.3, .8], [-.2, -.9], [.7, -.1], [-.5, .4]])
    sites = tuple(SurvivalSite(index, SurvivalRows(x + .1 * index,
        np.tile([0., 1.], 4), np.tile([1., 2., 4., 5.], 2), np.ones(8),
        np.arange(8 * index, 8 * index + 8))) for index in range(2))
    evaluation = SurvivalRows(np.array([[-.7, .1], [.2, -.4], [.8, .3], [-.5, -.2]]),
        np.array([1., 0., 1., 0.]), np.array([2., 3., 4., 5.]), np.array([1, 0, 1, 0]), np.arange(100, 104))
    return SurvivalDataset(sites, evaluation, survival_truth(evaluation.x, config),
                           np.array([.5, .5]), config, 0, 'iid')


def tiny_example():
    observed = dataset()
    config = protocol.run_config(protocol.PILOT_TASK, dict(scenario='iid'))
    config['dgp'] = asdict(observed.config)
    config['method']['optimizer'].update(epochs=1, batch_size=4)
    reference = dict(model=mock.Mock(), fingerprints=original_artifacts.dataset_fingerprints(observed),
        result=dict(site_sizes=[8, 8]), binding=dict(identity=dict(seed=0, scenario='iid', cell='dslasso'),
        artifacts={'result.json': dict(path=str(SCRATCH / 'historical-reference.json'))}))
    reference['model'].predict_eta.side_effect = AssertionError('historical baseline is never inferred')
    with mock.patch.object(computation, 'load_historical', return_value={'dslasso': reference}), \
            mock.patch.object(computation, 'generate_survival', return_value=observed):
        result, arrays = computation.compute_cell(config, 0)
    reference['model'].predict_eta.assert_not_called()
    return config, result, arrays, reference


class SurvivalMLP8ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.example = tiny_example()

    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.config, self.result, self.arrays, self.reference = deepcopy(self.example)
        self.frozen_fingerprints = deepcopy(self.result['fingerprints'])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(original_artifacts, 'current_provenance', return_value=provenance()))
        self.stack.enter_context(mock.patch.object(protocol, 'run_config', return_value=deepcopy(self.config)))
        self.selected = self.stack.enter_context(mock.patch.object(inputs, 'selected_fingerprints',
            side_effect=lambda receipt: deepcopy(self.frozen_fingerprints)))
        self.stack.enter_context(mock.patch.object(artifacts, 'read_json', return_value=dict(site_sizes=[8, 8])))
        self.stack.enter_context(mock.patch.object(methods.MLP8Fit, 'predict_eta', side_effect=AssertionError('no inference')))
        self.directory = Path(self.temporary.name) / self.config['run_id']
        self.store = artifacts.ArtifactStore(self.config, provenance(), directory=self.directory)
        self.result['identity'] = self.store.identity(0)

    def validate(self):
        return artifacts.validate_result(self.result, self.arrays, self.store.identity(0), self.config)

    def rehash(self):
        self.result['prediction_arrays_sha256'] = original_artifacts.array_hash(self.arrays)

    def publish(self):
        self.store.initialize()
        self.store.start(execution(), 0)
        self.store.complete(self.result, self.arrays, 0)

    def test_actual_four_small_fits_validate_without_model_inference(self):
        self.assertTrue(self.validate())
        self.selected.assert_called_once_with(self.result['reference'])
        self.assertEqual(set(self.result['fit']), set(artifacts.METHOD_TAGS))
        for tag in artifacts.METHOD_TAGS:
            self.assertEqual(self.result['prediction'][tag]['primary'],
                'oracle_survival_probability_accuracy_before_contrast')
            report = self.result['fit'][tag]
            training = report['local'] if tag in artifacts.FEDERATED_TAGS else [report]
            self.assertTrue(all(item['stop_reason'] == 'fixed_epoch_budget' for item in training))

    def test_saved_probability_metrics_and_each_gap_are_recomputed(self):
        self.result['prediction']['fedfisher_diag']['combined']['rmse'] += .1
        with self.assertRaisesRegex(ValueError, 'metrics differ'):
            self.validate()
        self.result['prediction'] = deepcopy(self.example[1]['prediction'])
        self.result['comparisons']['fedfisher_kfac']['pooled_km_pv']['arms']['0']['rmse'] += .1
        with self.assertRaisesRegex(ValueError, 'gap arithmetic'):
            self.validate()

    def test_dataset_fingerprints_cannot_be_changed_after_rehashing(self):
        self.result['fingerprints']['training'] = '8' * 64
        with self.assertRaisesRegex(ValueError, 'data differ'):
            self.validate()

    def test_all_oracle_arrays_are_bound_to_the_accepted_truth_hash(self):
        self.arrays['truth__propensity'][0] += .01
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'oracle fingerprint'):
            self.validate()

    def test_truth_link_shape_dtype_and_addition_are_required(self):
        for name, replacement in [('truth__survival0', np.ones(4)),
                ('truth__modifier_mask', np.array([1., 1.])),
                ('factual_a', self.arrays['factual_a'].astype(np.int64))]:
            with self.subTest(name=name):
                saved = self.arrays[name]
                self.arrays[name] = replacement
                self.rehash()
                with self.assertRaises(ValueError):
                    self.validate()
                self.arrays[name] = saved
                self.rehash()

    def test_fit_seed_dimension_and_initial_eta_are_config_bound(self):
        report = self.result['fit']['fedfisher_diag']
        for key, value in [('seed', 49), ('seed', True), ('input_dim', 7),
                           ('initial_eta', self.config['public_initial_eta'] + .1)]:
            with self.subTest(key=key):
                saved = report[key]
                report[key] = value
                with self.assertRaisesRegex(ValueError, 'seed/design/initialization'):
                    self.validate()
                report[key] = saved

    def test_curvature_choice_must_preserve_actual_local_vectors(self):
        name = 'fedfisher_kfac__fit__local_0__vector'
        self.arrays[name][0] += .01
        self.rehash()
        # Isolate the cross-method invariant from each method's separate hash
        # check, so a future relaxing of one cannot erase the pairing contract.
        with mock.patch.object(methods, 'validate_federated_fit', return_value=True):
            with self.assertRaisesRegex(ValueError, 'local fitting mismatch'):
                self.validate()

    def test_pooled_and_federated_public_initializations_are_identical(self):
        self.arrays['pooled_site_local_pv__fit__initial_vector'][0] += .01
        self.rehash()
        with mock.patch.object(methods, 'validate_pooled_fit', return_value=True):
            with self.assertRaisesRegex(ValueError, 'public initialization mismatch'):
                self.validate()

    def test_curvature_choice_preserves_response_fingerprints(self):
        self.result['fit']['fedfisher_diag']['local'][0]['training_pseudo_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'pseudo-response mismatch'):
            self.validate()

    def test_fit_cost_must_include_all_four_models(self):
        self.result['timing']['stages']['fit']['wall_seconds'] = 1e-12
        with self.assertRaisesRegex(ValueError, 'excludes fitting work'):
            self.validate()

    def test_reference_identity_and_site_sizes_are_checked(self):
        self.result['reference']['identity']['cell'] = 'fedfisher_diag_shallow'
        with self.assertRaisesRegex(ValueError, 'another data unit'):
            self.validate()
        self.result['reference']['identity']['cell'] = 'dslasso'
        with mock.patch.object(artifacts, 'read_json', return_value=dict(site_sizes=[7, 9])):
            with self.assertRaisesRegex(ValueError, 'site sizes differ'):
                self.validate()

    def test_unrecognized_arrays_and_missing_evidence_fail_closed(self):
        self.arrays['unreviewed_payload'] = np.array([0.])
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            self.validate()
        del self.arrays['unreviewed_payload']
        del self.arrays['truth__delta']
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'malformed'):
            self.validate()

    def test_completion_is_hash_bound_immutable_and_safe_to_skip(self):
        self.publish()
        self.assertEqual(self.store.status(0), 'validated')
        before = {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()}
        with self.assertRaises(FileExistsError):
            self.store.start(execution(), 0)
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays, 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()})
        (self.store.cell_dir(0) / 'model.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.store.status(0)

    def test_failed_partial_unit_is_preserved_and_cannot_be_resumed(self):
        self.store.initialize()
        self.store.start(execution(), 0)
        self.store.fail(RuntimeError('fixture failure'), 0)
        before = {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()}
        self.assertEqual(set(before), {'started.json', 'failure.json'})
        for operation in [lambda: self.store.status(0), lambda: self.store.start(execution(), 0),
                          lambda: self.store.complete(self.result, self.arrays, 0)]:
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()})

    def test_rejected_server_endpoint_keeps_diagnostics_without_completion(self):
        self.store.initialize()
        self.store.start(execution(), 0)
        self.result['fit']['fedfisher_kfac']['server']['converged'] = False
        with self.assertRaisesRegex(ValueError, 'convergence mismatch') as rejected:
            self.store.complete(self.result, self.arrays, 0)
        self.store.fail(rejected.exception, 0)
        failure = json.loads((self.store.cell_dir(0) / 'failure.json').read_text())
        self.assertEqual(failure['optimizer_diagnostics'], self.result['fit'])
        self.assertFalse((self.store.cell_dir(0) / 'done.json').exists())
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status(0)


class SurvivalMLP8ComputationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.observed = dataset()
        self.config = protocol.run_config(protocol.PILOT_TASK, dict(scenario='iid'))
        self.config['dgp'] = asdict(self.observed.config)
        self.reference = dict(fingerprints=original_artifacts.dataset_fingerprints(self.observed),
                              result=dict(site_sizes=[8, 8]))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.load = self.stack.enter_context(mock.patch.object(computation, 'load_historical',
            return_value={'dslasso': self.reference}))
        self.generate = self.stack.enter_context(mock.patch.object(computation, 'generate_survival', return_value=self.observed))
        self.fit = self.stack.enter_context(mock.patch.object(computation, 'fit_federated'))
        self.pooled = self.stack.enter_context(mock.patch.object(computation, 'fit_pooled'))

    def test_changed_data_fail_before_any_training_or_prediction(self):
        for key in self.reference['fingerprints']:
            original = self.reference['fingerprints'][key]
            self.reference['fingerprints'][key] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'data differ'):
                computation.compute_cell(self.config, 0)
            self.reference['fingerprints'][key] = original
        self.fit.assert_not_called()
        self.pooled.assert_not_called()

    def test_changed_reference_stops_before_data_generation(self):
        self.load.side_effect = ValueError('changed reference')
        with self.assertRaisesRegex(ValueError, 'changed reference'):
            computation.compute_cell(self.config, 0)
        self.generate.assert_not_called()
        self.fit.assert_not_called()


class SurvivalMLP8RunnerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(torch, 'set_num_threads'))
        self.stack.enter_context(mock.patch.object(torch, 'set_num_interop_threads'))
        self.stack.enter_context(mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}))
        self.source = provenance()
        self.provenance = self.stack.enter_context(mock.patch.object(runner, 'current_provenance', return_value=self.source))
        self.gate = self.stack.enter_context(mock.patch.object(inputs, 'validate_historical_inputs', return_value=True))
        self.completed = set()
        self.store = mock.Mock()
        self.store.status.side_effect = lambda seed: 'validated' if seed in self.completed else 'missing'
        self.store.identity.side_effect = lambda seed: dict(seed=seed, fixture='mlp8-runner')
        self.store.complete.side_effect = lambda result, arrays, seed: self.completed.add(seed)
        self.factory = self.stack.enter_context(mock.patch.object(runner, 'ArtifactStore', return_value=self.store))
        self.compute = self.stack.enter_context(mock.patch.object(computation, 'compute_cell',
            side_effect=lambda config, seed: (dict(status='complete'), dict(vector=np.array([0.])))))
        self.job = dict(task=protocol.PILOT_TASK, parameters=dict(scenario='iid'),
            job_id='fedfisher-survival-mlp8-runner-fixture', execution_profile='cpu-1core',
            allocation={}, estimated_runtime_seconds=10.)

    def test_both_owned_seeds_run_and_receipts_bind_identity(self):
        self.assertEqual(runner.run_job(self.job), 0)
        self.assertEqual([call.args[1] for call in self.compute.call_args_list], [0, 49])
        self.assertEqual(self.completed, {0, 49})
        for call in self.store.complete.call_args_list:
            self.assertEqual(call.args[0]['identity'], self.store.identity(call.args[2]))
        self.store.fail.assert_not_called()

    def test_validated_results_are_skipped_without_scientific_computation(self):
        self.completed.update((0, 49))
        self.assertEqual(runner.run_job(self.job), 0)
        self.compute.assert_not_called()
        self.store.start.assert_not_called()

    def test_source_drift_stops_before_unit_start(self):
        self.provenance.side_effect = [self.source, dict(self.source, git_commit='9' * 40)]
        with self.assertRaisesRegex(ValueError, 'source changed'):
            runner.run_job(self.job)
        self.store.start.assert_not_called()

    def test_scientific_error_is_preserved_and_stops_the_shard(self):
        error = ValueError('fixture scientific failure')
        self.compute.side_effect = error
        with self.assertRaisesRegex(ValueError, 'fixture scientific'):
            runner.run_job(self.job)
        self.store.fail.assert_called_once_with(error, 0)
        self.assertEqual(self.compute.call_count, 1)

    def test_reference_gate_rejection_prevents_any_output_creation(self):
        self.gate.side_effect = ValueError('changed frozen anchor')
        with self.assertRaisesRegex(ValueError, 'changed frozen'):
            runner.run_job(self.job)
        self.factory.assert_not_called()
        self.compute.assert_not_called()

    def test_formal_rejects_missing_new_source_pilot_before_output(self):
        from effect_modifier_experiment.survival_mlp8 import acceptance
        self.job['task'] = protocol.FORMAL_TASK
        self.job['parameters'].update(shard_index=0, shard_count=16)
        with mock.patch.object(acceptance, 'default_acceptance_path', return_value='fixture'), \
                mock.patch.object(acceptance, 'verify_pilot_acceptance', side_effect=ValueError('missing MLP8 pilot')):
            with self.assertRaisesRegex(ValueError, 'missing MLP8 pilot'):
                runner.run_job(self.job)
        self.factory.assert_not_called()
        self.compute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
