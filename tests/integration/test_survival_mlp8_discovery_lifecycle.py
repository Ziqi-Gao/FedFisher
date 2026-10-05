"""Bounded real four-fit discovery and saved-only V2 lifecycle fixtures."""
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
from effect_modifier_experiment.survival import artifacts as original_artifacts, evaluation
from effect_modifier_experiment.survival.data import (
    SurvivalConfig, SurvivalDataset, SurvivalRows, SurvivalSite, survival_truth,
)
from effect_modifier_experiment.survival_pooled import inputs
from effect_modifier_experiment.survival_mlp8 import (
    artifacts, computation, discovery_evaluation, discovery_protocol, methods, protocol, runner,
)

SCRATCH = Path('/scr/del6500/FedFisher/survival-mlp8-validation/discovery-lifecycle-fixtures')


def provenance():
    value = dict(git_commit='1' * 40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                 source_tree_sha256='2' * 64)
    value['provenance_sha256'] = hashlib.sha256(canonical_text(value, PROVENANCE_KEYS).encode()).hexdigest()
    return value


def execution():
    return dict(job_id='fedfisher-survival-mlp8-discovery-lifecycle-fixture', execution_profile='cpu-1core',
        attempt=1, estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192,
        gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def dataset():
    """Two planted modifier columns and one nuisance, without dataset generation."""
    config = SurvivalConfig(num_sites=2, site_min_samples=8, site_max_samples=8,
                            covariate_dim=3, num_test=4)
    x = np.array([[-1., 0., .2], [.1, .3, -.7], [.2, -.1, .8], [.4, .5, -.1],
                  [.3, .8, .4], [-.2, -.9, .6], [.7, -.1, -.3], [-.5, .4, -.5]])
    sites = tuple(SurvivalSite(index, SurvivalRows(x + .1 * index,
        np.tile([0., 1.], 4), np.tile([1., 2., 4., 5.], 2), np.ones(8),
        np.arange(8 * index, 8 * index + 8))) for index in range(2))
    rows = SurvivalRows(np.array([[-.7, .1, .8], [.2, -.4, -.5], [.8, .3, -.2], [-.5, -.2, .6]]),
        np.array([1., 0., 1., 0.]), np.array([2., 3., 4., 5.]), np.array([1, 0, 1, 0]), np.arange(100, 104))
    return SurvivalDataset(sites, rows, survival_truth(rows.x, config),
                           np.array([.5, .5]), config, 0, 'iid')


def tiny_example(directory):
    observed = dataset()
    config = discovery_protocol.run_config(discovery_protocol.PILOT_TASK, dict(scenario='iid'))
    config['dgp'] = asdict(observed.config)
    config['method']['optimizer'].update(epochs=1, batch_size=4)
    # This handwritten analytic witness supplies only original evaluator
    # hashes and score arithmetic. No historical fitted model is inferred.
    reference_result = dict(identity=dict(seed=0, scenario='iid', cell='dslasso'), status='complete',
        site_sizes=[8, 8], evaluation=evaluation.evaluate_model(
            lambda design: design[:, 0] + design[:, 4] - design[:, 5], observed, seed=0))
    path = directory / 'historical-reference.json'
    encoded = json.dumps(reference_result, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(encoded)
    reference = dict(model=mock.Mock(), fingerprints=original_artifacts.dataset_fingerprints(observed),
        result=reference_result, binding=dict(identity=reference_result['identity'],
        artifacts={'result.json': dict(path=str(path), sha256=hashlib.sha256(encoded).hexdigest())}))
    reference['model'].predict_eta.side_effect = AssertionError('historical baseline is never inferred')
    with mock.patch.object(computation, 'load_historical', return_value={'dslasso': reference}), \
            mock.patch.object(computation, 'generate_survival', return_value=observed):
        result, arrays = computation.compute_discovery_cell(config, 0)
    reference['model'].predict_eta.assert_not_called()
    return config, result, arrays, reference


class DiscoveryArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.example_directory = tempfile.TemporaryDirectory(dir=SCRATCH)
        cls.addClassCleanup(cls.example_directory.cleanup)
        cls.example = tiny_example(Path(cls.example_directory.name))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.config, self.result, self.arrays, self.reference = deepcopy(self.example)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(original_artifacts, 'current_provenance', return_value=provenance()))
        self.stack.enter_context(mock.patch.object(discovery_protocol, 'run_config', return_value=deepcopy(self.config)))
        self.stack.enter_context(mock.patch.object(inputs, 'selected_fingerprints',
            return_value=deepcopy(self.result['fingerprints'])))
        self.stack.enter_context(mock.patch.object(methods.MLP8Fit, 'predict_eta', side_effect=AssertionError('no inference')))
        self.stack.enter_context(mock.patch.object(computation, 'generate_survival', side_effect=AssertionError('no generation')))
        self.stack.enter_context(mock.patch.object(computation, 'fit_federated', side_effect=AssertionError('no fitting')))
        self.stack.enter_context(mock.patch.object(computation, 'fit_pooled', side_effect=AssertionError('no fitting')))
        self.stack.enter_context(mock.patch.object(discovery_evaluation, 'evaluate_discovery', side_effect=AssertionError('no re-evaluation')))
        self.store = artifacts.DiscoveryArtifactStore(self.config, provenance(),
            directory=Path(self.temporary.name) / self.config['run_id'])
        self.result['identity'] = self.store.identity(0)

    def validate(self):
        return artifacts.validate_discovery_result(self.result, self.arrays, self.store.identity(0), self.config)

    def rehash(self):
        self.result['prediction_arrays_sha256'] = original_artifacts.array_hash(self.arrays)

    def test_four_actual_fits_and_original_rankings_validate_without_inference(self):
        before_result = deepcopy(self.result)
        before_arrays = {name: value.tobytes() for name, value in self.arrays.items()}
        self.assertTrue(self.validate())
        self.assertEqual(self.result, before_result)
        self.assertEqual({name: value.tobytes() for name, value in self.arrays.items()}, before_arrays)
        self.assertEqual(set(self.result['discovery']), set(protocol.TAGS))
        for row in self.result['discovery'].values():
            self.assertEqual(np.asarray(row['importance_by_repeat']).shape, (5, 3))
            self.assertGreaterEqual(row['recovery']['auroc'], 0.)
            self.assertLessEqual(row['recovery']['probability_all_signals_selected'], 1.)

    def test_completion_namespace_is_new_and_immutable(self):
        self.store.initialize()
        self.store.start(execution(), 0)
        self.store.complete(self.result, self.arrays, 0)
        self.assertEqual(self.store.manifest['protocol'], discovery_protocol.PROTOCOL)
        self.assertIn('survival_mlp8_v2_', self.store.run_id)
        self.assertEqual(self.store.status(0), 'validated')
        before = {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()}
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays, 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()})

    def test_missing_ranking_or_full_x_fails_closed(self):
        ranking = self.result.pop('discovery')
        with self.assertRaisesRegex(ValueError, 'missing modifier ranking'):
            self.validate()
        self.result['discovery'] = ranking
        del self.arrays['discovery__x']
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'discovery saved arrays'):
            self.validate()

    def test_full_outer_hash_is_checked_before_base_projection(self):
        self.arrays['discovery__x'][0, 2] += .1
        with mock.patch.object(artifacts, 'validate_result', side_effect=AssertionError('projection reached')) as base:
            with self.assertRaisesRegex(ValueError, 'discovery array fingerprint'):
                self.validate()
        base.assert_not_called()
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'covariates or oracle contrast'):
            self.validate()

    def test_missing_methods_unknown_arrays_and_probability_tampering_fail(self):
        row = self.result['discovery'].pop('fedfisher_kfac')
        with self.assertRaisesRegex(ValueError, 'exactly the four'):
            self.validate()
        self.result['discovery']['fedfisher_kfac'] = row
        self.arrays['discovery__unreviewed'] = np.array([1.])
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            self.validate()
        del self.arrays['discovery__unreviewed']
        self.rehash()
        self.result['prediction']['fedfisher_diag']['combined']['rmse'] += .1
        with self.assertRaisesRegex(ValueError, 'metrics differ'):
            self.validate()

    def test_discovery_time_and_recipe_scope_are_mandatory(self):
        stage = self.result['timing']['stages'].pop('discovery_evaluation')
        with self.assertRaisesRegex(ValueError, 'timing scope'):
            self.validate()
        self.result['timing']['stages']['discovery_evaluation'] = stage
        stage['wall_seconds'] = 0.
        with self.assertRaisesRegex(ValueError, 'stage timing'):
            self.validate()
        stage['wall_seconds'] = 1.
        self.config['evaluation']['modifier_permutation_performed'] = False
        with self.assertRaisesRegex(ValueError, 'fixed original'):
            self.validate()

    def test_reference_second_read_is_checksum_bound(self):
        with mock.patch.object(artifacts, 'file_binding', return_value=dict(path='changed', sha256='0' * 64)):
            with self.assertRaisesRegex(ValueError, 'reference checksum'):
                self.validate()

    def test_v1_result_validator_rejects_unprojected_v2_evidence(self):
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            artifacts.validate_result(self.result, self.arrays, self.store.identity(0), self.config)

    def test_failed_ranking_keeps_fit_diagnostics_without_done_marker(self):
        self.store.initialize()
        self.store.start(execution(), 0)
        self.result['discovery']['fedfisher_diag']['recovery']['auroc'] = 2.
        with self.assertRaises(ValueError) as rejected:
            self.store.complete(self.result, self.arrays, 0)
        self.store.fail(rejected.exception, 0)
        failure = json.loads((self.store.cell_dir(0) / 'failure.json').read_text())
        self.assertEqual(failure['optimizer_diagnostics'], self.result['fit'])
        self.assertFalse((self.store.cell_dir(0) / 'done.json').exists())
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status(0)


class DiscoveryComputationTests(unittest.TestCase):
    def test_probability_is_assessed_before_discovery_and_v1_contract_is_unchanged(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            config, result, arrays, reference = tiny_example(Path(directory))
            old_config = protocol.run_config(protocol.PILOT_TASK, dict(scenario='iid'))
            old_config['dgp'] = deepcopy(config['dgp'])
            old_config['method'] = deepcopy(config['method'])
            events = []
            probability = computation.probability_predictions
            discovery = discovery_evaluation.evaluate_discovery

            def record_probability(*args, **kwargs):
                events.append('probability')
                return probability(*args, **kwargs)

            def record_discovery(*args, **kwargs):
                events.append('discovery')
                return discovery(*args, **kwargs)

            with mock.patch.object(computation, 'load_historical', return_value={'dslasso': reference}), \
                    mock.patch.object(computation, 'generate_survival', return_value=dataset()), \
                    mock.patch.object(computation, 'probability_predictions', side_effect=record_probability), \
                    mock.patch.object(discovery_evaluation, 'evaluate_discovery', side_effect=record_discovery):
                new_result, new_arrays = computation.compute_discovery_cell(config, 0)
                self.assertEqual(events, ['probability'] * 4 + ['discovery'] * 4)
                events.clear()
                old_result, old_arrays = computation.compute_cell(old_config, 0)
                self.assertEqual(events, ['probability'] * 4)
            self.assertNotIn('discovery', old_result)
            self.assertNotIn('discovery_evaluation', old_result['timing']['stages'])
            self.assertEqual(set(old_arrays), set(new_arrays) - {'discovery__x'})
            for name, value in old_arrays.items():
                self.assertEqual(value.tobytes(), new_arrays[name].tobytes(), name)
            for field in ('prediction', 'comparisons', 'fingerprints', 'reference', 'historical_inputs', 'site_sizes'):
                self.assertEqual(old_result[field], new_result[field], field)
            self.assertEqual(result['discovery'], new_result['discovery'])
            self.assertEqual(arrays['discovery__x'].tobytes(), new_arrays['discovery__x'].tobytes())


class DiscoveryAdmissionTests(unittest.TestCase):
    def test_store_rejects_v1_namespace_and_changed_discovery_recipe(self):
        config = discovery_protocol.run_config(discovery_protocol.PILOT_TASK, dict(scenario='iid'))
        invalid = [protocol.run_config(protocol.PILOT_TASK, dict(scenario='iid')),
                   deepcopy(config), deepcopy(config)]
        invalid[1]['evaluation']['discovery']['repeats'] = 1
        invalid[2]['timing']['stages'].remove('discovery_evaluation')
        for changed in invalid:
            with self.subTest(config=changed['protocol']):
                with self.assertRaises(ValueError):
                    artifacts.DiscoveryArtifactStore(changed, provenance())

    def test_formal_store_requires_same_source_and_pilot_rejects_prerequisites(self):
        source = provenance()
        formal = discovery_protocol.run_config(discovery_protocol.FORMAL_TASK,
            dict(scenario='iid', shard_index=0, shard_count=16))
        pilot = discovery_protocol.run_config(discovery_protocol.PILOT_TASK, dict(scenario='iid'))
        proof = dict(pilot_acceptance_sha256='3' * 64, provenance_sha256=source['provenance_sha256'])
        with mock.patch.object(original_artifacts, 'current_provenance', return_value=source):
            accepted = artifacts.DiscoveryArtifactStore(formal, source, prerequisite=proof)
            self.assertEqual(accepted.prerequisite, proof)
            for prerequisite in (None, dict(proof, provenance_sha256='4' * 64),
                                 dict(proof, pilot_acceptance_sha256='bad')):
                with self.subTest(prerequisite=prerequisite):
                    with self.assertRaisesRegex(ValueError, 'same-source discovery pilot'):
                        artifacts.DiscoveryArtifactStore(formal, source, prerequisite=prerequisite)
            with self.assertRaisesRegex(ValueError, 'pilot cannot use formal'):
                artifacts.DiscoveryArtifactStore(pilot, source, prerequisite=proof)


class DiscoveryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(torch, 'set_num_threads'))
        self.stack.enter_context(mock.patch.object(torch, 'set_num_interop_threads'))
        self.stack.enter_context(mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}))
        self.source = provenance()
        self.stack.enter_context(mock.patch.object(runner, 'current_provenance', return_value=self.source))
        self.stack.enter_context(mock.patch.object(inputs, 'validate_historical_inputs', return_value=True))
        self.completed = set()
        self.store = mock.Mock()
        self.store.status.side_effect = lambda seed: 'validated' if seed in self.completed else 'missing'
        self.store.identity.side_effect = lambda seed: dict(seed=seed, fixture='discovery-runner')
        self.store.complete.side_effect = lambda result, arrays, seed: self.completed.add(seed)
        self.factory = self.stack.enter_context(mock.patch.object(runner, 'DiscoveryArtifactStore', return_value=self.store))
        self.old_factory = self.stack.enter_context(mock.patch.object(runner, 'ArtifactStore', side_effect=AssertionError('old store')))
        self.compute = self.stack.enter_context(mock.patch.object(computation, 'compute_discovery_cell',
            side_effect=lambda config, seed: (dict(status='complete'), dict(vector=np.array([0.])))))
        self.old_compute = self.stack.enter_context(mock.patch.object(computation, 'compute_cell', side_effect=AssertionError('old compute')))
        self.job = dict(task=discovery_protocol.PILOT_TASK, parameters=dict(scenario='iid'),
            job_id='fedfisher-survival-mlp8-discovery-runner-fixture', execution_profile='cpu-1core',
            allocation={}, estimated_runtime_seconds=10.)

    def test_discovery_pilot_owns_only_its_two_seeds_and_store(self):
        self.assertEqual(runner.run_discovery_job(self.job), 0)
        self.assertEqual([call.args[1] for call in self.compute.call_args_list], [0, 49])
        self.assertEqual(self.factory.call_args.args[0]['protocol'], discovery_protocol.PROTOCOL)
        self.old_factory.assert_not_called()
        self.old_compute.assert_not_called()

    def test_saved_validated_units_skip_all_fitting_and_inference(self):
        self.completed.update((0, 49))
        self.assertEqual(runner.run_discovery_job(self.job), 0)
        self.compute.assert_not_called()
        self.store.start.assert_not_called()

    def test_formal_requires_fresh_discovery_proof_before_creating_outputs(self):
        from effect_modifier_experiment.survival_mlp8 import acceptance
        self.job['task'] = discovery_protocol.FORMAL_TASK
        self.job['parameters'].update(shard_index=0, shard_count=16)
        with mock.patch.object(acceptance, 'default_acceptance_path', return_value='v2-proof') as path, \
                mock.patch.object(acceptance, 'verify_pilot_acceptance', side_effect=ValueError('missing V2 proof')) as verify:
            with self.assertRaisesRegex(ValueError, 'missing V2 proof'):
                runner.run_discovery_job(self.job)
        path.assert_called_once_with(self.source, discovery=True)
        verify.assert_called_once_with('v2-proof', discovery=True)
        self.factory.assert_not_called()
        self.compute.assert_not_called()

    def test_formal_scientific_prerequisite_binds_discovery_proof_digest(self):
        from effect_modifier_experiment.infrastructure.artifacts import digest
        from effect_modifier_experiment.survival_mlp8 import acceptance
        proof = dict(scope='survival_mlp8_discovery_v2_pilot_scientific_acceptance', passed=True)
        self.job['task'] = discovery_protocol.FORMAL_TASK
        self.job['parameters'].update(shard_index=3, shard_count=16)
        with mock.patch.object(acceptance, 'default_acceptance_path', return_value='v2-proof'), \
                mock.patch.object(acceptance, 'verify_pilot_acceptance', return_value=proof):
            self.assertEqual(runner.run_discovery_job(self.job), 0)
        self.assertEqual([call.args[1] for call in self.compute.call_args_list], list(range(3, 300, 16)))
        self.assertEqual(self.factory.call_args.kwargs['prerequisite'],
            dict(pilot_acceptance_sha256=digest(proof), provenance_sha256=self.source['provenance_sha256']))


if __name__ == '__main__':
    unittest.main()
