"""Bounded numerical composition and immutable survival evidence fixtures."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import os
import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest import mock

import numpy as np
import torch

from effect_modifier_experiment.dslassocov_run_contract import canonical_text
from effect_modifier_experiment.survival import artifacts, protocol, runner

SCRATCH = Path('/scr/del6500/FedFisher/survival-validation/contract')


def provenance():
    values = dict(git_commit='1'*40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                  source_tree_sha256='2'*64)
    values['provenance_sha256'] = hashlib.sha256(canonical_text(values, artifacts.PROVENANCE_KEYS).encode()).hexdigest()
    return values


def execution():
    return dict(job_id='fedfisher-survival-contract-fixture', execution_profile='cpu-1core', attempt=1,
        estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192, gpu_count=0,
        gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def tiny_config(cell='fedfisher_diag_shallow'):
    config = protocol.run_config(protocol.PILOT_TASK, dict(cell=cell, scenario='iid'))
    config['dgp'].update(num_sites=2, site_min_samples=48, site_max_samples=48,
        covariate_dim=3, num_test=12, signal_strength=.2, main_effect_strength=.2, confounding_strength=.2)
    config['evaluation']['repeats'] = 1
    if cell == 'dslasso':
        config['method']['optimizer'].update(num_folds=2, num_lambdas=2, max_iter=2000, tolerance=1e-4)
    else:
        config['method']['optimizer'].update(epochs=1, batch_size=16, server_ridge=.001)
    return config


class SurvivalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.examples = {}
        for cell in ('dslasso', 'fedfisher_diag_shallow', 'fedfisher_kfac_shallow'):
            config = tiny_config(cell)
            result, arrays = runner.compute_cell(config)
            result['identity'] = dict(seed=0, fixture='tiny_survival_contract')
            artifacts.validate_result(result, arrays, result['identity'], config)
            cls.examples[cell] = (config, result, arrays)

    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.config, self.result, self.arrays = deepcopy(self.examples['fedfisher_diag_shallow'])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(artifacts, 'current_provenance', return_value=provenance()))
        # Only the test's explicit fixture config is admitted; production recipes stay strict.
        self.stack.enter_context(mock.patch.object(protocol, 'run_config', return_value=deepcopy(self.config)))
        self.directory = Path(self.temp.name)/self.config['run_id']
        self.store = artifacts.ArtifactStore(self.config, provenance(), self.directory)
        self.result['identity'] = self.store.identity()

    def publish(self):
        self.store.initialize()
        self.store.start(execution())
        self.store.complete(self.result, self.arrays)

    def rehash(self, filename):
        path = self.store.cell_dir()
        marker = json.loads((path/'done.json').read_text())
        marker['artifacts'][filename] = hashlib.sha256((path/filename).read_bytes()).hexdigest()
        (path/'done.json').write_text(json.dumps(marker))

    def test_actual_tiny_dslasso_diag_and_kfac_results_validate_without_prediction(self):
        from effect_modifier_experiment.survival import fisher, dslasso, evaluation
        with mock.patch.object(fisher.FisherFit, 'predict_eta', side_effect=AssertionError('no inference')), \
                mock.patch.object(dslasso.SurvivalLinearModel, 'predict_eta', side_effect=AssertionError('no inference')):
            for config, result, arrays in self.examples.values():
                artifacts.validate_result(result, arrays, result['identity'], config)

    def test_method_families_bind_identical_training_and_independent_evaluation(self):
        values = [value[1]['fingerprints'] for value in self.examples.values()]
        for name in ('training', 'evaluation', 'evaluation_truth'):
            self.assertEqual(len({value[name] for value in values}), 1)
        self.assertEqual(self.result['site_sizes'], [48, 48])
        self.assertIs(self.result['evaluation']['end_to_end_one_round_claim'], False)

    def test_roundtrip_retry_and_no_overwrite_preserve_producer_attempt(self):
        self.store.initialize()
        self.assertEqual(self.store.status(), 'missing')
        self.store.start(execution())
        self.store.complete(self.result, self.arrays)
        before = {path.name: path.read_bytes() for path in self.store.cell_dir().iterdir()}
        self.assertEqual(self.store.status(), 'validated')
        with self.assertRaises(FileExistsError):
            self.store.start(dict(execution(), attempt=2))
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.cell_dir().iterdir()})
        self.assertEqual(json.loads(before['started.json'])['execution']['attempt'], 1)

    def test_partial_and_failed_attempts_are_never_completion(self):
        self.store.initialize()
        self.store.start(execution())
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status()
        self.store.fail(RuntimeError('fixture failure'))
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status()
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays)
        self.assertFalse((self.store.cell_dir()/'done.json').exists())

    def test_changed_source_and_config_fail_closed(self):
        self.publish()
        with mock.patch.object(artifacts, 'current_provenance', return_value=dict(provenance(), source_tree_sha256='3'*64)):
            with self.assertRaisesRegex(ValueError, 'current source'):
                self.store.status()
        self.store.config['dgp']['tau'] += 1
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            self.store.status()

    def test_dirty_source_and_unreviewed_tau_recipe_rejected_before_manifest(self):
        with self.assertRaisesRegex(ValueError, 'finalized Git'):
            artifacts.ArtifactStore(self.config, dict(provenance(), git_dirty='1'), self.directory)
        changed = deepcopy(self.config); changed['dgp']['tau'] += 1
        with self.assertRaisesRegex(ValueError, 'reviewed fixed recipe'):
            artifacts.ArtifactStore(changed, provenance(), self.directory)
        self.assertFalse(self.directory.exists())

    def test_nonempty_unmanifested_directory_and_symlinks_rejected(self):
        self.directory.mkdir()
        (self.directory/'unowned.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.store.initialize()
        link = Path(self.temp.name)/'alias'
        link.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            artifacts.safe_path(link/'result.json')

    def test_rehashed_scientific_identity_and_endpoint_tampering_rejected(self):
        self.publish()
        for mutate in (
            lambda r: r['identity'].update(seed=1),
            lambda r: r['evaluation'].update(primary='permute/probability_contrast'),
            lambda r: r['evaluation'].update(end_to_end_one_round_claim=True),
            lambda r: r['evaluation']['recovery'].update(probability_all_signals_selected=.123456),
            lambda r: r['fit'].update(pseudo_scope='pooled'),
            lambda r: r['fit']['config'].update(epochs=99),
            lambda r: r['fit']['server'].update(converged=False),
            lambda r: r['fit']['server'].update(client_to_server_upload_phases=2),
            lambda r: r['fit']['local'][0].update(uploads=2),
            lambda r: r['fit'].update(initial_eta=0.),
            lambda r: r['timing']['stages']['fit'].update(wall_seconds=1e-12),
            lambda r: r['pseudo_diagnostics'][0].update(tau=99),
            lambda r: r['pseudo_diagnostics'][0].update(all_one=True),
            lambda r: r['pseudo_diagnostics'][0].update(censored_count=48),
            lambda r: r['pseudo_diagnostics'][0].update(pv_mean=0),
        ):
            changed = deepcopy(self.result); mutate(changed)
            (self.store.cell_dir()/'result.json').write_text(json.dumps(changed))
            self.rehash('result.json')
            with self.subTest(changed=changed['fit'].get('pseudo_scope')), self.assertRaises(ValueError):
                self.store.status()

    def test_rehashed_model_and_upload_tampering_fail_semantic_validation(self):
        self.publish()
        for target in ('vector', 'upload_000'):
            arrays = {name: value.copy() for name, value in self.arrays.items()}
            if target == 'vector':
                arrays[target][0] += 1.
            else:
                arrays[target][0] ^= np.uint8(1)
            buffer = io.BytesIO(); np.savez(buffer, **arrays)
            (self.store.cell_dir()/'model.npz').write_bytes(buffer.getvalue())
            result = deepcopy(self.result)
            result['fingerprints']['model'] = artifacts.array_hash(arrays)
            (self.store.cell_dir()/'result.json').write_text(json.dumps(result))
            self.rehash('model.npz'); self.rehash('result.json')
            with self.subTest(target=target), self.assertRaises((ValueError, OSError)):
                self.store.status()

    def test_invalid_execution_never_starts(self):
        self.store.initialize()
        for key, value in (('attempt', 0), ('execution_profile', 'gpu'),
                           ('job_id', 'unrelated-job'), ('estimated_runtime_seconds', 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.start(dict(execution(), **{key: value}))
        self.assertFalse(self.store.cell_dir().exists())

    def test_foreground_runner_validates_completed_retry_without_fitting(self):
        self.publish()
        job = dict(execution(), task=protocol.PILOT_TASK,
                   parameters=dict(cell=self.config['cell'], scenario=self.config['scenario']))
        before = {path.name: path.read_bytes() for path in self.store.cell_dir().iterdir()}
        with mock.patch.object(artifacts, 'ArtifactStore', return_value=self.store), \
                mock.patch.object(runner, 'compute_cell', side_effect=AssertionError('completed retry must skip')), \
                mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '2'}):
            self.assertEqual(runner.run_job(job), 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.cell_dir().iterdir()})

    def test_foreground_runner_preserves_failed_cell_and_does_not_retry_it(self):
        job = dict(execution(), task=protocol.PILOT_TASK,
                   parameters=dict(cell=self.config['cell'], scenario=self.config['scenario']))
        with mock.patch.object(artifacts, 'ArtifactStore', return_value=self.store), \
                mock.patch.object(runner, 'compute_cell', side_effect=RuntimeError('fixture-only failure')) as compute, \
                mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'fixture-only failure'):
                runner.run_job(job)
            with self.assertRaisesRegex(ValueError, 'partial'):
                runner.run_job(job)
            self.assertEqual(compute.call_count, 1)
        self.assertTrue((self.store.cell_dir()/'failure.json').is_file())
        self.assertFalse((self.store.cell_dir()/'done.json').exists())

    def test_lasso_false_convergence_wrong_scope_or_selection_rejected(self):
        config, result, arrays = deepcopy(self.examples['dslasso'])
        for change in (
            lambda r: r['fit'].update(pseudo_scope='pooled'),
            lambda r: r['fit']['final_fit'].update(converged=False),
            lambda r: r['fit'].update(selected_lambda_fraction=.123),
            lambda r: r['fit']['communication'].update(one_shot=True),
        ):
            modified = deepcopy(result); change(modified)
            with self.assertRaises(ValueError):
                artifacts.validate_result(modified, arrays, modified['identity'], config)

    def formal_config(self):
        config = deepcopy(self.config)
        config.update(task=protocol.FORMAL_TASK, seeds=list(range(300)))
        config['run_id'] = config['run_id'].removesuffix('_pilot')+'_300seeds'
        del config['seed']
        return config

    def formal_prerequisite(self):
        return dict(pilot_acceptance_sha256='a'*64, provenance_sha256=provenance()['provenance_sha256'])

    def test_formal_store_binds_distinct_seed_cells_and_source_bound_pilot_proof(self):
        config = self.formal_config()
        proof = self.formal_prerequisite()
        with mock.patch.object(protocol, 'run_config', return_value=deepcopy(config)):
            store = artifacts.ArtifactStore(config, provenance(), Path(self.temp.name)/config['run_id'],
                                            prerequisite=proof)
            store.initialize()
            result, arrays = runner.compute_cell(config, 17)
            result['identity'] = store.identity(17)
            store.start(execution(), 17)
            store.complete(result, arrays, 17)
            self.assertEqual(store.status(17), 'validated')
            self.assertEqual(store.status(0), 'missing')
            self.assertEqual(store.status(33), 'missing')
            self.assertEqual(store.cell_dir(17).name, 'seed_017')
            receipt = json.loads((store.cell_dir(17)/'started.json').read_text())
            self.assertEqual(receipt['prerequisite'], proof)
            self.assertEqual(receipt['identity']['seed'], 17)
            self.assertNotEqual(store.identity(17)['config_sha256'], self.store.identity()['config_sha256'])
            with self.assertRaises(ValueError):
                store.identity(300)
            with self.assertRaises(ValueError):
                store.identity(True)
            with self.assertRaises(ValueError):
                runner.compute_cell(config)
            with self.assertRaises(ValueError):
                runner.compute_cell(config, 300)
            receipt['prerequisite']['pilot_acceptance_sha256'] = 'b'*64
            (store.cell_dir(17)/'started.json').write_text(json.dumps(receipt))
            marker = json.loads((store.cell_dir(17)/'done.json').read_text())
            marker['artifacts']['started.json'] = hashlib.sha256((store.cell_dir(17)/'started.json').read_bytes()).hexdigest()
            (store.cell_dir(17)/'done.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, 'prerequisite mismatch'):
                store.status(17)

    def test_formal_constructor_requires_pilot_proof_from_same_source(self):
        config = self.formal_config()
        for proof in (None, {}, dict(self.formal_prerequisite(), provenance_sha256='b'*64),
                      dict(self.formal_prerequisite(), pilot_acceptance_sha256='bad')):
            with mock.patch.object(protocol, 'run_config', return_value=config), self.assertRaisesRegex(ValueError, 'pilot acceptance'):
                artifacts.ArtifactStore(config, provenance(), Path(self.temp.name)/config['run_id'], prerequisite=proof)

    def test_concurrent_formal_shards_initialize_one_manifest_without_internal_temps(self):
        config = self.formal_config()
        proof = self.formal_prerequisite()
        directory = Path(self.temp.name)/config['run_id']
        with mock.patch.object(protocol, 'run_config', return_value=config):
            stores = [artifacts.ArtifactStore(config, provenance(), directory, prerequisite=proof)
                      for _ in range(2)]
        barrier = threading.Barrier(2)
        errors = []
        original_link = os.link

        def delayed_link(source, target):
            if Path(target).name == 'run_manifest.env':
                self.assertEqual(Path(source).parent, directory.parent)
                self.assertNotIn(Path(source), list(directory.iterdir()))
                barrier.wait(timeout=5)
            return original_link(source, target)

        def initialize(store):
            try:
                store.initialize()
            except Exception as error:
                errors.append(error)

        with mock.patch.object(artifacts.os, 'link', side_effect=delayed_link):
            threads = [threading.Thread(target=initialize, args=(store,)) for store in stores]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual({path.name for path in directory.iterdir()}, {'run_manifest.env'})
        self.assertEqual({path.name for path in directory.parent.iterdir()}, {directory.name})
        self.assertEqual(stores[0].status(0), 'missing')
        self.assertEqual(stores[1].status(1), 'missing')

    def test_formal_runtime_gate_failure_precedes_output_and_computation(self):
        gate_module = ModuleType('effect_modifier_experiment.survival.acceptance')
        gate_module.check_current_pilot_gate = mock.Mock(side_effect=ValueError('pilot evidence missing'))
        job = dict(execution(), task=protocol.FORMAL_TASK,
            parameters=dict(cell=self.config['cell'], scenario='iid', shard_index=3, shard_count=16))
        with mock.patch.dict(sys.modules, {gate_module.__name__: gate_module}), \
                mock.patch.object(protocol, 'run_config', return_value=self.formal_config()), \
                mock.patch.object(artifacts, 'ArtifactStore') as store, \
                mock.patch.object(runner, 'compute_cell') as compute:
            with self.assertRaisesRegex(ValueError, 'pilot evidence missing'):
                runner.run_job(job)
        gate_module.check_current_pilot_gate.assert_called_once_with()
        store.assert_not_called()
        compute.assert_not_called()

    def test_formal_runner_executes_only_its_deterministic_seeds_after_gate(self):
        config = self.formal_config()
        proof = dict(fixture='accepted pilot proof', provenance=provenance())
        gate_module = ModuleType('effect_modifier_experiment.survival.acceptance')
        gate_module.check_current_pilot_gate = mock.Mock(return_value=proof)
        job = dict(execution(), task=protocol.FORMAL_TASK,
            parameters=dict(cell=self.config['cell'], scenario='iid', shard_index=3, shard_count=16))
        seeds = protocol.seed_sequence(job['task'], job['parameters'])
        store = mock.Mock()
        store.status.side_effect = [value for _ in seeds for value in ('missing', 'validated')]
        store.identity.side_effect = lambda seed: dict(seed=seed)
        with mock.patch.dict(sys.modules, {gate_module.__name__: gate_module}), \
                mock.patch.object(protocol, 'run_config', return_value=config), \
                mock.patch.object(artifacts, 'ArtifactStore', return_value=store) as factory, \
                mock.patch.object(runner, 'compute_cell', side_effect=lambda conf, seed: ({}, {})) as compute, \
                mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}):
            self.assertEqual(runner.run_job(job), 0)
        gate_module.check_current_pilot_gate.assert_called_once_with()
        factory.assert_called_once_with(config, provenance(), prerequisite=dict(
            pilot_acceptance_sha256=artifacts.digest(proof), provenance_sha256=provenance()['provenance_sha256']))
        self.assertEqual([call.args[1] for call in compute.call_args_list], list(seeds))
        self.assertEqual([call.args[2] for call in store.complete.call_args_list], list(seeds))


if __name__ == '__main__':
    unittest.main()
