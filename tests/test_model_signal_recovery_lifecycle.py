"""A capped fit stays immutable, resumable as noncompletion, and never success."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from effect_modifier_experiment.fedfisher_timing import artifacts as lifecycle
from effect_modifier_experiment.infrastructure.artifacts import PROVENANCE_KEYS, canonical, safe_path
from effect_modifier_experiment.infrastructure.provenance import canonical_text
from effect_modifier_experiment.model_signal_ablation import recovery, recovery_protocol as protocol
from effect_modifier_experiment.causal_comparison.noncompletion import ScientificNoncompletion
from test_effect_modifier_fedavg_noncompletion import capped_fixture


def provenance():
    values = dict(git_commit='1' * 40, git_dirty='0',
                  git_status_sha256=hashlib.sha256(b'').hexdigest(), source_tree_sha256='2' * 64)
    values['provenance_sha256'] = hashlib.sha256(canonical_text(values, PROVENANCE_KEYS).encode()).hexdigest()
    return values


def parameters(**changes):
    p = dict(setting='small_mlp', cell='fedavg_small_e1_patience', variant='control',
             shard_index=0, shard_count=1, case='baseline')
    p.update(changes)
    return p


def execution(attempt=1):
    return dict(job_id='fedfisher-ms-recovery-lifecycle-fixture', execution_profile='cpu-1core',
                attempt=attempt, estimated_runtime_seconds=20., allocation=dict(cpu_cores=1,
                memory_mib=8192, gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def prerequisites(formal=False):
    binding = dict(path='/scr/del6500/FedFisher/model-signal-recovery-validation/fixture-closeout.json',
                   sha256='a' * 64)
    return dict(closeout=binding, pilot=dict(binding, sha256='b' * 64) if formal else None)


class RecoveryLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture_config, cls.fixture_result, cls.fixture_arrays = capped_fixture(seed=0)

    def setUp(self):
        parent = safe_path(Path('/scr/del6500/FedFisher/model-signal-recovery-validation/lifecycle'))
        parent.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=parent)))
        self.config = deepcopy(self.fixture_config)
        template = protocol.run_config(protocol.PILOT_TASK, parameters())
        for key in ('protocol', 'task', 'run_id', 'case', 'seeds'):
            self.config[key] = template[key]
        self.provenance = provenance()
        self.stack.enter_context(mock.patch.object(protocol, 'run_config', return_value=self.config))
        self.current = self.stack.enter_context(mock.patch.object(lifecycle, 'current_provenance',
                                                                  return_value=self.provenance))
        self.store = recovery.RecoveryStore(self.config, self.provenance, prerequisites(),
                                            self.root / self.config['run_id'])
        self.store.initialize()
        self.result, self.arrays = deepcopy(self.fixture_result), deepcopy(self.fixture_arrays)
        self.result['identity'] = self.store.identity(0, 'iid')

    def publish(self):
        self.store.start(0, 'iid', execution())
        self.store.noncomplete(0, 'iid', self.result, self.arrays)

    def rehash(self, name):
        path = self.store.cell_dir(0, 'iid')
        marker = json.loads((path / 'noncompletion_marker.json').read_text())
        marker['artifacts'][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        (path / 'noncompletion_marker.json').write_text(json.dumps(marker))

    def test_real_capped_fit_has_distinct_validated_terminal_files_and_retry_is_read_only(self):
        self.publish()
        path = self.store.cell_dir(0, 'iid')
        self.assertEqual(set(p.name for p in path.iterdir()), set(recovery.NONCOMPLETE_FILES))
        self.assertFalse((path / 'done.json').exists())
        self.assertNotIn('evaluation', self.result)
        before = {str(p): p.read_bytes() for p in self.store.directory.rglob('*') if p.is_file()}
        retry = recovery.RecoveryStore(self.config, self.provenance, prerequisites(), self.store.directory)
        retry.initialize()
        self.assertEqual(retry.status(0, 'iid'), 'noncompleted')
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.store.directory.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            retry.start(0, 'iid', execution(2))
        with self.assertRaisesRegex(ValueError, 'overwritten'):
            retry.noncomplete(0, 'iid', self.result, self.arrays)
        with self.assertRaises(ValueError):
            retry.complete(0, 'iid', self.result, self.arrays)

    def test_absent_closeout_binding_is_rejected_before_store_initialization(self):
        with self.assertRaisesRegex(ValueError, 'closeout binding is mandatory'):
            recovery.RecoveryStore(self.config, self.provenance, dict(closeout=None, pilot=None),
                                  self.root / self.config['run_id'])

    def test_rehashed_wrong_prerequisite_stale_history_and_model_still_fail(self):
        self.publish()
        path = self.store.cell_dir(0, 'iid')
        originals = {p.name: p.read_bytes() for p in path.iterdir()}
        for change in ('prerequisite', 'history', 'performance'):
            with self.subTest(change=change):
                for name, data in originals.items():
                    (path / name).write_bytes(data)
                name = 'started.json' if change == 'prerequisite' else 'noncompletion.json'
                value = json.loads((path / name).read_text())
                if change == 'prerequisite':
                    value['prerequisites']['closeout']['sha256'] = 'f' * 64
                elif change == 'history':
                    value['fit']['round_history'][-1]['stale_rounds'] = 10
                else:
                    value['evaluation'] = {'invented_performance': 1.0}
                (path / name).write_text(json.dumps(value))
                self.rehash(name)
                with self.assertRaises(ValueError):
                    self.store.status(0, 'iid')

    def test_partial_mixed_symlink_and_source_drift_are_not_terminal(self):
        self.store.start(0, 'iid', execution())
        with self.assertRaises(ValueError):
            self.store.status(0, 'iid')
        self.store.noncomplete(0, 'iid', self.result, self.arrays)
        path = self.store.cell_dir(0, 'iid')
        (path / 'done.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'partial or mixed'):
            self.store.status(0, 'iid')
        (path / 'done.json').unlink()
        original = (path / 'noncompletion.json').read_bytes()
        (path / 'noncompletion.json').unlink()
        target = self.root / 'alias-target.json'
        target.write_bytes(original)
        (path / 'noncompletion.json').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.store.status(0, 'iid')
        (path / 'noncompletion.json').unlink()
        (path / 'noncompletion.json').write_bytes(original)
        self.current.return_value = dict(self.provenance, source_tree_sha256='f' * 64)
        with self.assertRaisesRegex(ValueError, 'current source'):
            self.store.status(0, 'iid')

    def test_shard_summary_requires_exact_coverage_and_does_not_call_exhaustion_success(self):
        self.publish()
        job = dict(execution(), task=protocol.PILOT_TASK, parameters=parameters())
        with self.assertRaisesRegex(ValueError, 'nonterminal'):
            recovery.publish_shard_summary(self.store, job, execution())
        result = deepcopy(self.result)
        result['identity'] = self.store.identity(0, 'noniid')
        self.store.start(0, 'noniid', execution())
        self.store.noncomplete(0, 'noniid', result, self.arrays)
        summary = recovery.publish_shard_summary(self.store, job, execution())
        self.assertTrue(summary['execution_complete'])
        self.assertFalse(summary['whole_shard_scientific_success'])
        self.assertEqual(summary['counts'], dict(complete=0, budget_exhausted=2,
                                               unattempted=0, errors=0, planned=2))
        path = recovery.shard_summary_path(self.store, job)
        before = path.read_bytes()
        recovery.publish_shard_summary(self.store, job, execution(2))
        self.assertEqual(path.read_bytes(), before)
        summary['whole_shard_scientific_success'] = True
        path.write_text(canonical(summary))
        with self.assertRaisesRegex(ValueError, 'differs'):
            recovery.validate_shard_summary(self.store, job)


class RecoveryRunnerTests(unittest.TestCase):
    def run_with(self, compute, *, status=None):
        from effect_modifier_experiment.model_signal_ablation import recovery_acceptance as gate
        job = dict(execution(), task=protocol.PILOT_TASK,
                   parameters=parameters(variant='site_gaussian_shift', case='cap_diagnostic'))
        store = mock.Mock()
        store.status.side_effect = status or (lambda seed, split: 'missing')
        store.identity.side_effect = lambda seed, split: dict(seed=seed, split=split)
        summary = dict(whole_shard_scientific_success=False, counts={'planned': 6})
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(recovery, 'current_provenance', return_value=provenance()))
            stack.enter_context(mock.patch.object(gate, 'verify_closeout', return_value={'binding': prerequisites()['closeout']}))
            stack.enter_context(mock.patch.object(recovery, 'RecoveryStore', return_value=store))
            calls = stack.enter_context(mock.patch.object(recovery, 'compute_cell', side_effect=compute))
            finish = stack.enter_context(mock.patch.object(recovery, 'publish_shard_summary', return_value=summary))
            stack.enter_context(mock.patch.dict('os.environ', {'SERVER_SCHEDULER_ATTEMPT': '2'}))
            stack.enter_context(mock.patch.object(torch, 'set_num_threads'))
            stack.enter_context(mock.patch.object(torch, 'set_num_interop_threads'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                result = recovery.run_shard(job)
            except Exception as error:
                result = error
        return result, store, calls, finish

    @staticmethod
    def capped_error():
        error = ScientificNoncompletion.__new__(ScientificNoncompletion)
        RuntimeError.__init__(error, 'fixture cap')
        error.result = {'status': 'scientific_noncompletion'}
        error.arrays = {}
        return error

    def test_validated_cap_continues_to_seed_41_and_retry_skips_both_terminal_outcomes(self):
        def compute(config, seed, split):
            if (seed, split) == (25, 'noniid'):
                raise self.capped_error()
            return {}, {}
        result, store, calls, finish = self.run_with(compute)
        self.assertEqual(result, 0)
        self.assertEqual([call.args[1:] for call in calls.call_args_list],
                         [(s, split) for s in (9, 25, 41) for split in protocol.SPLITS])
        store.noncomplete.assert_called_once()
        self.assertEqual(store.complete.call_count, 5)
        store.fail.assert_not_called()
        finish.assert_called_once()
        result, store, calls, finish = self.run_with(compute,
            status=lambda seed, split: 'noncompleted' if seed == 25 else 'validated')
        self.assertEqual(result, 0)
        calls.assert_not_called()
        store.start.assert_not_called()

    def test_unknown_or_nonfinite_error_aborts_before_later_seeds(self):
        for error in (RuntimeError('unknown fit failure'), FloatingPointError('nonfinite loss')):
            with self.subTest(error=type(error).__name__):
                result, store, calls, finish = self.run_with(lambda *args: (_ for _ in ()).throw(error))
                self.assertIs(result, error)
                self.assertEqual(calls.call_count, 1)
                store.fail.assert_called_once()
                store.noncomplete.assert_not_called()
                finish.assert_not_called()

    def test_missing_new_pilot_proof_blocks_formal_work_before_output(self):
        from effect_modifier_experiment.model_signal_ablation import recovery_acceptance as gate
        p = parameters(shard_index=9, shard_count=16)
        p.pop('case')
        job = dict(execution(), task=protocol.FORMAL_TASK, parameters=p)
        with mock.patch.object(recovery, 'current_provenance', return_value=provenance()), \
                mock.patch.object(gate, 'verify_closeout', return_value={'binding': prerequisites()['closeout']}), \
                mock.patch.object(gate, 'verify_pilot_acceptance', side_effect=ValueError('new pilot absent')), \
                mock.patch.object(recovery, 'RecoveryStore') as store, \
                mock.patch.object(recovery, 'compute_cell') as compute, \
                mock.patch.object(torch, 'set_num_threads'), mock.patch.object(torch, 'set_num_interop_threads'):
            with self.assertRaisesRegex(ValueError, 'new pilot absent'):
                recovery.run_shard(job)
            store.assert_not_called()
            compute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
