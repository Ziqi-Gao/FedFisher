"""Small numerical fixtures for separate diagnostic and formal completion contracts."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.infrastructure.provenance import canonical_text
from effect_modifier_experiment.survival import artifacts, protocol, recovery

SCRATCH = Path('/scr/del6500/FedFisher/survival-validation/recovery-contract')


def provenance():
    values = dict(git_commit='1' * 40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                  source_tree_sha256='2' * 64)
    values['provenance_sha256'] = hashlib.sha256(canonical_text(values, artifacts.PROVENANCE_KEYS).encode()).hexdigest()
    return values


def execution():
    return dict(job_id='fedfisher-survival-recovery-contract', execution_profile='cpu-1core', attempt=1,
        estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192, gpu_count=0,
        gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def tiny_config(formal=False, failure=False):
    if formal:
        config = recovery.run_config(recovery.FORMAL_TASK,
            dict(scenario='iid', shard_index=0, shard_count=16), recipe='budget_only')
    else:
        config = recovery.run_config(recovery.DIAGNOSTIC_TASK, dict(case='iid_000', recipe='baseline'))
    config['dgp'].update(num_sites=2, site_min_samples=48, site_max_samples=48,
        covariate_dim=3, num_test=12, signal_strength=.2, main_effect_strength=.2, confounding_strength=.2)
    config['evaluation']['repeats'] = 1
    config['method']['optimizer'].update(num_folds=2, num_lambdas=2,
        max_iter=1 if failure else 2000, tolerance=1e-12 if failure else 1e-4)
    return config


class RecoveryProtocolTests(unittest.TestCase):
    def test_fixed_case_selection_matches_archived_failures_and_controls(self):
        self.assertEqual(len(recovery.CASES), 27)
        self.assertEqual(sum(row['role'] == 'observed_failure' for row in recovery.CASES.values()), 19)
        self.assertEqual({row['scenario'] for row in recovery.CASES.values()}, set(protocol.SCENARIOS))
        ids = set()
        for case, row in recovery.CASES.items():
            for recipe in recovery.RECIPES:
                p = dict(case=case, recipe=recipe)
                self.assertEqual(recovery.seed_sequence(recovery.DIAGNOSTIC_TASK, p), (row['seed'],))
                config = recovery.run_config(recovery.DIAGNOSTIC_TASK, p)
                self.assertEqual(config['scenario'], row['scenario'])
                self.assertEqual(config['recovery_recipe'], recipe)
                ids.add((config['run_id'], row['seed']))
        self.assertEqual(len(ids), 108)

    def test_formal_coverage_is_exact_and_shards_do_not_change_scientific_config(self):
        for scenario in recovery.SCENARIOS:
            for count in (8, 16):
                seen = []
                configs = []
                for index in range(count):
                    p = dict(scenario=scenario, shard_index=index, shard_count=count)
                    seen.extend(recovery.seed_sequence(recovery.FORMAL_TASK, p))
                    configs.append(recovery.run_config(recovery.FORMAL_TASK, p, recipe='budget_only'))
                self.assertEqual(sorted(seen), list(range(300)))
                self.assertTrue(all(config == configs[0] for config in configs))
        with self.assertRaises(ValueError):
            recovery.run_config(recovery.FORMAL_TASK, dict(scenario='iid', shard_index=0, shard_count=16))
        with self.assertRaises(ValueError):
            recovery.run_config(recovery.FORMAL_TASK,
                dict(scenario='iid', shard_index=0, shard_count=16), recipe='baseline')

    def test_unreviewed_parameters_and_optimization_overrides_rejected(self):
        for p in ({'case': 'iid_001', 'recipe': 'baseline'}, {'case': 'iid_000', 'recipe': 'new'},
                  {'case': 'iid_000', 'recipe': 'baseline', 'max_iter': 8},
                  {'case': 'iid_000', 'recipe': 'baseline', 'resources': {}}):
            with self.assertRaises(ValueError):
                recovery.validate_parameters(recovery.DIAGNOSTIC_TASK, p)
        for p in (dict(scenario='iid', shard_index=True, shard_count=16),
                  dict(scenario='iid', shard_index=0, shard_count=4),
                  dict(scenario='iid', shard_index=0, shard_count=16, recipe='budget_only')):
            with self.assertRaises(ValueError):
                recovery.validate_parameters(recovery.FORMAL_TASK, p)

    def test_numerical_candidates_preserve_old_science_and_unchanged_v1_recipe(self):
        old = protocol.run_config(protocol.FORMAL_TASK,
            dict(cell='dslasso', scenario='iid', shard_index=0, shard_count=16))
        before = artifacts.canonical(old)
        for recipe in recovery.RECIPES:
            config = recovery.run_config(recovery.DIAGNOSTIC_TASK, dict(case='iid_000', recipe=recipe))
            for field in ('dgp', 'pseudo', 'mean_link', 'evaluation', 'public_initial_eta'):
                self.assertEqual(config[field], old[field])
            optimizer = dict(config['method']['optimizer']); optimizer.pop('max_iter')
            original = dict(old['method']['optimizer']); original.pop('max_iter')
            self.assertEqual(optimizer, original)
        self.assertEqual(before, artifacts.canonical(protocol.run_config(protocol.FORMAL_TASK,
            dict(cell='dslasso', scenario='iid', shard_index=0, shard_count=16))))


class RecoveryLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {}
        for label, formal, failure in (('stationary', False, False), ('iteration_limit', False, True),
                                      ('formal', True, False)):
            config = tiny_config(formal=formal, failure=failure)
            result, arrays = recovery.compute_cell(config, 0)
            result['identity'] = dict(seed=0, fixture=label)
            recovery.validate_result(result, arrays, result['identity'], config)
            cls.examples[label] = (config, result, arrays)

    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(artifacts, 'current_provenance', return_value=provenance()))

    def store(self, label, root=None):
        config, result, arrays = deepcopy(self.examples[label])
        prerequisite = (dict(diagnostic_acceptance_sha256='a' * 64,
            provenance_sha256=provenance()['provenance_sha256'], selected_recipe='budget_only')
            if label == 'formal' else None)
        with mock.patch.object(recovery, 'run_config', return_value=deepcopy(config)):
            store = recovery.RecoveryStore(config, provenance(), Path(root or self.temp.name)/config['run_id'],
                                             prerequisite=prerequisite)
        result['identity'] = store.identity(0)
        return store, result, arrays

    def publish(self, label, root=None):
        store, result, arrays = self.store(label, root)
        store.initialize()
        store.start(execution(), 0)
        store.complete(result, arrays, 0)
        return store, result, arrays

    def rehash(self, store, name):
        marker = json.loads((store.cell_dir(0)/store.done_name).read_text())
        marker['artifacts'][name] = hashlib.sha256((store.cell_dir(0)/name).read_bytes()).hexdigest()
        (store.cell_dir(0)/store.done_name).write_text(json.dumps(marker))

    def test_stationary_and_iteration_limit_are_distinct_valid_diagnostic_observations(self):
        for label in ('stationary', 'iteration_limit'):
            with self.subTest(label=label):
                store, result, arrays = self.publish(label, Path(self.temp.name)/label)
                self.assertEqual(result['outcome'], label)
                self.assertEqual(store.status(0), 'validated')
                self.assertEqual({path.name for path in store.cell_dir(0).iterdir()},
                    {'started.json', 'fit_arrays.npz', 'diagnostic_result.json', 'diagnostic_done.json'})
                self.assertFalse((store.cell_dir(0)/'done.json').exists())
                self.assertNotIn('evaluation', result)
                before = {path.name: path.read_bytes() for path in store.cell_dir(0).iterdir()}
                with self.assertRaises(FileExistsError):
                    store.start(dict(execution(), attempt=2), 0)
                self.assertEqual(before, {path.name: path.read_bytes() for path in store.cell_dir(0).iterdir()})

    def test_saved_validation_does_not_generate_data_fit_or_predict(self):
        from effect_modifier_experiment.survival import data, dslasso, recovery_method
        with mock.patch.object(data, 'generate_survival', side_effect=AssertionError('no regeneration')), \
                mock.patch.object(recovery_method, 'fit_recovery_dslasso', side_effect=AssertionError('no fitting')), \
                mock.patch.object(dslasso.SurvivalLinearModel, 'predict_eta', side_effect=AssertionError('no inference')):
            for config, result, arrays in self.examples.values():
                recovery.validate_result(result, arrays, result['identity'], config)

    def test_formal_completion_pins_gate_in_manifest_and_started_receipt(self):
        store, result, arrays = self.publish('formal')
        self.assertEqual(store.status(0), 'validated')
        self.assertEqual(store.manifest['prerequisite_sha256'], artifacts.digest(store.prerequisite))
        receipt = json.loads((store.cell_dir(0)/'started.json').read_text())
        self.assertEqual(receipt['prerequisite'], store.prerequisite)
        receipt['prerequisite']['diagnostic_acceptance_sha256'] = 'b' * 64
        (store.cell_dir(0)/'started.json').write_text(json.dumps(receipt))
        self.rehash(store, 'started.json')
        with self.assertRaisesRegex(ValueError, 'prerequisite mismatch'):
            store.status(0)

    def test_formal_store_requires_valid_same_source_selected_recipe_prerequisite(self):
        config = self.examples['formal'][0]
        good = dict(diagnostic_acceptance_sha256='a'*64,
            provenance_sha256=provenance()['provenance_sha256'], selected_recipe='budget_only')
        for proof in (None, {}, dict(good, selected_recipe='step_only'),
                      dict(good, provenance_sha256='b'*64), dict(good, diagnostic_acceptance_sha256='bad')):
            with mock.patch.object(recovery, 'run_config', return_value=deepcopy(config)):
                with self.assertRaisesRegex(ValueError, 'same-source diagnostic acceptance'):
                    recovery.RecoveryStore(config, provenance(), Path(self.temp.name)/config['run_id'],
                                           prerequisite=proof)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_rehashed_diagnostic_outcome_or_identity_cannot_fabricate_completion(self):
        store, result, arrays = self.publish('iteration_limit')
        for change in (lambda r: r.update(outcome='stationary'),
                       lambda r: r['identity'].update(case='iid_057'),
                       lambda r: r['fit']['context'].update(seed=57),
                       lambda r: r['fit'].update(outcome='stationary'),
                       lambda r: r['timing']['stages']['fit'].update(wall_seconds=1e-12)):
            modified = deepcopy(result); change(modified)
            (store.cell_dir(0)/store.result_name).write_text(json.dumps(modified))
            self.rehash(store, store.result_name)
            with self.assertRaises(ValueError):
                store.status(0)

    def test_rehashed_terminal_gradient_is_checked_against_saved_arithmetic(self):
        store, result, arrays = self.publish('iteration_limit')
        arrays['terminal_gradient'][0] += 1
        buffer = io.BytesIO(); np.savez(buffer, **arrays)
        (store.cell_dir(0)/store.model_name).write_bytes(buffer.getvalue())
        result['fingerprints']['arrays'] = artifacts.array_hash(arrays)
        (store.cell_dir(0)/store.result_name).write_text(json.dumps(result))
        self.rehash(store, store.model_name); self.rehash(store, store.result_name)
        with self.assertRaises(ValueError):
            store.status(0)

    def test_saved_endpoint_dimension_is_bound_to_declared_dgp(self):
        config, result, arrays = deepcopy(self.examples['iteration_limit'])
        result['fit']['paths'][0]['means'].append(0.)
        result['fit']['paths'][0]['scales'].append(1.)
        with self.assertRaisesRegex(ValueError, 'declared covariate dimension'):
            recovery.validate_result(result, arrays, result['identity'], config)

    def test_source_changes_partial_output_and_wrong_completion_filename_fail_closed(self):
        store, result, arrays = self.store('stationary')
        store.initialize(); store.start(execution(), 0)
        with self.assertRaisesRegex(ValueError, 'partial'):
            store.status(0)
        store.fail(RuntimeError('unexpected scientific failure'), 0)
        self.assertFalse((store.cell_dir(0)/store.done_name).exists())
        with self.assertRaises(ValueError):
            store.complete(result, arrays, 0)
        with mock.patch.object(artifacts, 'current_provenance', return_value=dict(provenance(), source_tree_sha256='f'*64)):
            with self.assertRaisesRegex(ValueError, 'current source'):
                store.status(0)

    def test_diagnostic_done_marker_cannot_be_renamed_as_formal_completion(self):
        store, _, _ = self.publish('stationary')
        (store.cell_dir(0)/store.done_name).rename(store.cell_dir(0)/'done.json')
        with self.assertRaisesRegex(ValueError, 'partial'):
            store.status(0)

    def test_original_data_disagreement_blocks_fitting(self):
        from effect_modifier_experiment.survival import recovery_method
        config, result, _ = deepcopy(self.examples['stationary'])
        result['fingerprints']['training'] = '0' * 64
        with mock.patch.object(recovery_method, 'fit_recovery_dslasso') as fit:
            with self.assertRaisesRegex(ValueError, 'accepted original training'):
                recovery.compute_cell(config, 0, reference=result)
        fit.assert_not_called()

    def test_formal_runtime_cannot_initialize_without_diagnostic_acceptance(self):
        gate = ModuleType('effect_modifier_experiment.survival.recovery_acceptance')
        gate.check_original_closeout = mock.Mock(return_value={})
        gate.check_current_diagnostic_gate = mock.Mock(side_effect=ValueError('diagnostic gate missing'))
        gate.original_reference = mock.Mock()
        job = dict(execution(), task=recovery.FORMAL_TASK,
            parameters=dict(scenario='iid', shard_index=3, shard_count=16))
        with mock.patch.dict(sys.modules, {gate.__name__: gate}), \
                mock.patch.object(recovery, 'RecoveryStore') as store, \
                mock.patch.object(recovery, 'compute_cell') as compute:
            with self.assertRaisesRegex(ValueError, 'diagnostic gate missing'):
                recovery.run_job(job)
        store.assert_not_called(); compute.assert_not_called()

    def test_completed_diagnostic_retry_preserves_producer_and_skips_all_science(self):
        store, _, _ = self.publish('stationary')
        gate = ModuleType('effect_modifier_experiment.survival.recovery_acceptance')
        gate.check_original_closeout = mock.Mock(return_value={})
        gate.check_current_diagnostic_gate = mock.Mock(side_effect=AssertionError('not formal'))
        gate.original_reference = mock.Mock(side_effect=AssertionError('no regeneration'))
        job = dict(execution(), task=recovery.DIAGNOSTIC_TASK,
            parameters=dict(case='iid_000', recipe='baseline'))
        before = {path.name: path.read_bytes() for path in store.cell_dir(0).iterdir()}
        with mock.patch.dict(sys.modules, {gate.__name__: gate}), \
                mock.patch.object(recovery, 'run_config', return_value=store.config), \
                mock.patch.object(recovery, 'RecoveryStore', return_value=store), \
                mock.patch.object(recovery, 'compute_cell', side_effect=AssertionError('no fit')), \
                mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '2'}):
            self.assertEqual(recovery.run_job(job), 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in store.cell_dir(0).iterdir()})
        self.assertEqual(json.loads(before['started.json'])['execution']['attempt'], 1)

    def test_formal_iteration_limit_stays_failed_with_structured_evidence_and_no_done(self):
        from effect_modifier_experiment.survival.recovery_method import OptimizationFailure
        config = tiny_config(formal=True, failure=True)
        proof = dict(selected_recipe='budget_only')
        prerequisite = dict(diagnostic_acceptance_sha256=artifacts.digest(proof),
            provenance_sha256=provenance()['provenance_sha256'], selected_recipe='budget_only')
        with mock.patch.object(recovery, 'run_config', return_value=deepcopy(config)):
            store = recovery.RecoveryStore(config, provenance(), Path(self.temp.name)/config['run_id'],
                                           prerequisite=prerequisite)
        gate = ModuleType('effect_modifier_experiment.survival.recovery_acceptance')
        gate.check_original_closeout = mock.Mock(return_value={})
        gate.check_current_diagnostic_gate = mock.Mock(return_value=proof)
        gate.original_reference = mock.Mock(return_value=self.examples['stationary'][1])
        job = dict(execution(), task=recovery.FORMAL_TASK,
            parameters=dict(scenario='iid', shard_index=0, shard_count=16))
        with mock.patch.dict(sys.modules, {gate.__name__: gate}), \
                mock.patch.object(recovery, 'run_config', return_value=config), \
                mock.patch.object(recovery, 'RecoveryStore', return_value=store), \
                mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}):
            with self.assertRaises(OptimizationFailure):
                recovery.run_job(job)
        failure = json.loads((store.cell_dir(0)/'failure.json').read_text())
        self.assertEqual(failure['optimization_diagnostics']['outcome'], 'iteration_limit')
        self.assertFalse((store.cell_dir(0)/'done.json').exists())
        with self.assertRaisesRegex(ValueError, 'partial'):
            store.status(0)

    def test_formal_runtime_executes_only_owned_seeds_under_one_selected_recipe(self):
        gate = ModuleType('effect_modifier_experiment.survival.recovery_acceptance')
        gate.check_original_closeout = mock.Mock(return_value={})
        gate.check_current_diagnostic_gate = mock.Mock(return_value=dict(selected_recipe='budget_only'))
        gate.original_reference = mock.Mock(return_value={'fixture': True})
        job = dict(execution(), task=recovery.FORMAL_TASK,
            parameters=dict(scenario='iid', shard_index=3, shard_count=16))
        seeds = recovery.seed_sequence(job['task'], job['parameters'])
        store = mock.Mock()
        store.status.side_effect = [value for _ in seeds for value in ('missing', 'validated')]
        store.identity.side_effect = lambda seed: dict(seed=seed)
        with mock.patch.dict(sys.modules, {gate.__name__: gate}), \
                mock.patch.object(recovery, 'RecoveryStore', return_value=store) as factory, \
                mock.patch.object(recovery, 'compute_cell', side_effect=lambda *a, **kw: ({}, {})) as compute, \
                mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}):
            self.assertEqual(recovery.run_job(job), 0)
        self.assertEqual(factory.call_args.args[0]['recovery_recipe'], 'budget_only')
        self.assertEqual([call.args[1] for call in compute.call_args_list], list(seeds))
        self.assertTrue(all(call.kwargs['reference'] == {'fixture': True} for call in compute.call_args_list))


if __name__ == '__main__':
    unittest.main()
