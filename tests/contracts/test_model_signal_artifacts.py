"""Fail-closed lifecycle and dispatch checks using a tiny real FedAvg result."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from effect_modifier_experiment.causal_comparison.fedavg import FedAvgConfig
from effect_modifier_experiment.fedfisher_timing import artifacts as timing_artifacts
from effect_modifier_experiment.infrastructure.artifacts import PROVENANCE_KEYS, safe_path
from effect_modifier_experiment.infrastructure.provenance import canonical_text
from effect_modifier_experiment.model_signal_ablation import artifacts, protocol, runner


def parameters(**changes):
    result = dict(setting='weak_signal_05', cell='fedavg_small_e1_budget30',
                  variant='control', shard_index=0, shard_count=1)
    result.update(changes)
    return result


def tiny_configuration():
    config = protocol.run_config(protocol.PILOT_TASK, parameters())
    config['dgp'].update(covariate_dim=3, num_sites=2, num_test=16,
                         site_min_samples=12, site_max_samples=16)
    config['evaluation'].update(repeats=1, batch_size=8)
    config['method']['optimizer'] = asdict(FedAvgConfig(
        covariate_dim=3, local_epochs=1, batch_size=8, min_rounds=1, max_rounds=2,
        stopping_mode='fixed_total_epochs', total_local_epochs=2))
    return config


def provenance():
    values = dict(git_commit='1' * 40, git_dirty='0',
                  git_status_sha256=hashlib.sha256(b'').hexdigest(), source_tree_sha256='2' * 64)
    values['provenance_sha256'] = hashlib.sha256(canonical_text(values, PROVENANCE_KEYS).encode()).hexdigest()
    return values


def execution():
    return dict(job_id='fedfisher-ms-lifecycle-fixture', execution_profile='cpu-1core', attempt=1,
        estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192, gpu_count=0,
        gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


class ModelSignalArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture_config = tiny_configuration()
        with redirect_stdout(io.StringIO()):
            cls.fixture_result, cls.fixture_arrays = runner.compute_cell(cls.fixture_config, 0, 'iid')

    def setUp(self):
        parent = safe_path(Path('/scr/del6500/FedFisher/model-signal-validation/contracts'))
        parent.mkdir(parents=True, exist_ok=True)
        safe_path(parent)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=parent)))
        self.config = deepcopy(self.fixture_config)
        self.provenance = provenance()
        self.stack.enter_context(mock.patch.object(protocol, 'run_config', return_value=deepcopy(self.config)))
        self.current = self.stack.enter_context(mock.patch.object(
            timing_artifacts, 'current_provenance', return_value=self.provenance))
        self.store = artifacts.ArtifactStore(self.config, self.provenance, self.root / self.config['run_id'])
        self.store.initialize()
        self.result, self.arrays = deepcopy(self.fixture_result), deepcopy(self.fixture_arrays)
        self.result['identity'] = self.store.identity(0, 'iid')

    def complete(self):
        self.store.start(0, 'iid', execution())
        self.store.complete(0, 'iid', self.result, self.arrays)

    @staticmethod
    def rehash(path, name):
        marker = json.loads((path / 'done.json').read_text())
        marker['artifacts'][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        (path / 'done.json').write_text(json.dumps(marker))

    def test_real_small_fedavg_completes_and_revalidation_is_read_only(self):
        self.assertEqual(self.store.status(0, 'iid'), 'missing')
        self.complete()
        before = {str(path): path.read_bytes() for path in self.store.directory.rglob('*') if path.is_file()}
        retry = artifacts.ArtifactStore(self.config, self.provenance, self.store.directory)
        retry.initialize()
        self.assertEqual(retry.status(0, 'iid'), 'validated')
        self.assertEqual(retry.status(0, 'noniid'), 'missing')
        self.assertEqual(before, {str(path): path.read_bytes()
                                 for path in self.store.directory.rglob('*') if path.is_file()})
        with self.assertRaises(FileExistsError):
            retry.start(0, 'iid', execution())
        with self.assertRaisesRegex(ValueError, 'overwritten'):
            retry.complete(0, 'iid', self.result, self.arrays)

    def test_partial_and_failed_units_never_acquire_success_markers(self):
        self.store.start(0, 'iid', execution())
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status(0, 'iid')
        invalid = deepcopy(self.result)
        invalid['fit']['completion_eligible'] = False
        with self.assertRaises(ValueError):
            self.store.complete(0, 'iid', invalid, self.arrays)
        self.store.fail(0, 'iid', RuntimeError('bounded lifecycle fixture'))
        self.assertEqual({path.name for path in self.store.cell_dir(0, 'iid').iterdir()},
                         {'started.json', 'failure.json'})
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.start(0, 'iid', execution())

    def test_rehashing_wrong_recovery_memory_or_execution_does_not_make_it_valid(self):
        self.complete()
        path = self.store.cell_dir(0, 'iid')
        originals = {name: (path / name).read_bytes() for name in ('result.json', 'started.json', 'done.json')}
        for kind in ('recovery', 'memory', 'attempt', 'namespace'):
            with self.subTest(kind=kind):
                for name, value in originals.items():
                    (path / name).write_bytes(value)
                name = 'result.json' if kind in ('recovery', 'memory') else 'started.json'
                value = json.loads((path / name).read_text())
                if kind == 'recovery':
                    value['evaluation']['metrics']['permute/tau_prob']['recovery']['expected_topk_hits'] = -1
                elif kind == 'memory':
                    value['memory']['process_peak_rss_kib'] = 0
                elif kind == 'attempt':
                    value['execution']['attempt'] = 0
                else:
                    value['execution']['job_id'] = 'fedfisher-timing-foreign-cohort'
                (path / name).write_text(json.dumps(value))
                self.rehash(path, name)
                with self.assertRaises(ValueError):
                    self.store.status(0, 'iid')

    def test_source_change_during_validation_prevents_done_marker(self):
        self.store.start(0, 'iid', execution())
        validator = artifacts.validate_result
        calls = []

        def changing_validation(*args):
            validator(*args)
            calls.append(1)
            if len(calls) == 2:
                self.current.return_value = dict(self.provenance, source_tree_sha256='9' * 64)

        with mock.patch.object(artifacts, 'validate_result', side_effect=changing_validation):
            with self.assertRaisesRegex(ValueError, 'current source provenance'):
                self.store.complete(0, 'iid', self.result, self.arrays)
        self.assertFalse((self.store.cell_dir(0, 'iid') / 'done.json').exists())
        with self.assertRaisesRegex(ValueError, 'current source provenance'):
            self.store.status(0, 'iid')

    def test_wrong_signal_identity_and_symlinks_fail_closed(self):
        changed = deepcopy(self.config)
        changed['dgp']['signal_strength'] = 1.
        with self.assertRaisesRegex(ValueError, 'reviewed fixed recipe'):
            artifacts.ArtifactStore(changed, self.provenance, self.root / changed['run_id'])
        self.store.start(0, 'iid', execution())
        changed_result = deepcopy(self.result)
        changed_result['identity']['run_id'] = changed_result['identity']['run_id'].replace('weak_signal_05', 'weak_signal_1')
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.store.complete(0, 'iid', changed_result, self.arrays)
        alias = self.root / 'alias'
        alias.symlink_to(self.store.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            artifacts.ArtifactStore(self.config, self.provenance, alias / self.config['run_id'])


class ModelSignalRunnerTests(unittest.TestCase):
    def test_existing_family_runner_receives_the_exact_configuration(self):
        for cell in ('fedavg_small_e1_budget30', 'fedfisher_diag_small', 'fedfisher_kfac_deep', 'dslassocov'):
            config = protocol.run_config(protocol.PILOT_TASK, parameters(cell=cell))
            module = ('effect_modifier_experiment.fedfisher_timing.computation' if cell.startswith('fedfisher_')
                      else 'effect_modifier_experiment.causal_comparison.computation')
            arrays = {'fixture': object()}
            with self.subTest(cell=cell), mock.patch(module + '.compute_cell', return_value=({'status': 'fixture'}, arrays)) as compute:
                result, actual_arrays = runner.compute_cell(config, 3, 'noniid')
            compute.assert_called_once_with(config, 3, 'noniid')
            self.assertIs(actual_arrays, arrays)
            self.assertEqual(result['memory']['scope'], artifacts.MEMORY_SCOPE)
            self.assertGreater(result['memory']['process_peak_rss_kib'], 0)

    def test_formal_gate_fails_before_output_creation_or_scientific_work(self):
        job = dict(execution(), task=protocol.FORMAL_TASK, parameters=parameters(shard_count=16))
        with mock.patch.object(runner, 'current_provenance', return_value=provenance()), \
                mock.patch.object(runner, 'ArtifactStore') as store, \
                mock.patch.object(runner, 'compute_cell') as compute, \
                mock.patch('effect_modifier_experiment.model_signal_ablation.acceptance.check_current_pilot_gate',
                           side_effect=ValueError('pilot evidence absent')) as gate, \
                mock.patch.object(torch, 'set_num_threads'), mock.patch.object(torch, 'set_num_interop_threads'):
            with self.assertRaisesRegex(ValueError, 'pilot evidence absent'):
                runner.run_shard(job)
            gate.assert_called_once_with()
            store.assert_not_called()
            compute.assert_not_called()

    def test_shard_dispatch_honors_deterministic_seeds_splits_and_validated_skips(self):
        job = dict(execution(), task=protocol.FORMAL_TASK,
                   parameters=parameters(shard_index=15, shard_count=16))
        store = mock.Mock()
        store.status.side_effect = lambda seed, split: 'validated' if (seed, split) == (15, 'iid') else 'missing'
        store.identity.side_effect = lambda seed, split: dict(seed=seed, split=split)
        with mock.patch.object(runner, 'current_provenance', return_value=provenance()), \
                mock.patch.object(runner, 'ArtifactStore', return_value=store), \
                mock.patch.object(runner, 'compute_cell', side_effect=lambda *args: ({}, {})) as compute, \
                mock.patch('effect_modifier_experiment.model_signal_ablation.acceptance.check_current_pilot_gate'), \
                mock.patch.dict('os.environ', {'SERVER_SCHEDULER_ATTEMPT': '2'}), \
                mock.patch.object(torch, 'set_num_threads'), mock.patch.object(torch, 'set_num_interop_threads'):
            self.assertEqual(runner.run_shard(job), 0)
        expected = [(seed, split) for seed in range(15, 300, 16)
                    for split in ('iid', 'noniid') if (seed, split) != (15, 'iid')]
        self.assertEqual([call.args[1:] for call in compute.call_args_list], expected)
        self.assertEqual([call.args[:2] for call in store.start.call_args_list], expected)
        self.assertEqual(store.complete.call_count, len(expected))
        self.assertTrue(all(call.args[2]['attempt'] == 2 for call in store.start.call_args_list))
        store.fail.assert_not_called()


if __name__ == '__main__':
    unittest.main()
