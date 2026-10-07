"""Actual tiny fits cross the new immutable store's scientific boundary."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure.provenance import canonical_text, PROVENANCE_KEYS
from effect_modifier_experiment.survival import artifacts as original_store
from effect_modifier_experiment.survival_comparators import protocol
from effect_modifier_experiment.survival_comparators.artifacts import ArtifactStore
from tests.integration import test_survival_comparators_science as science_fixture


class ComparatorLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.helper = science_fixture.SurvivalComparatorScienceTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        source = dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64, source_tree_sha256='c' * 64)
        source['provenance_sha256'] = hashlib.sha256(canonical_text(source, PROVENANCE_KEYS).encode()).hexdigest()
        self.source = source
        self.stack.enter_context(mock.patch.object(original_store, 'current_provenance', return_value=source))
        self.stack.enter_context(mock.patch.object(protocol, 'run_config', return_value=self.helper.config))
        scratch = Path('/scr/del6500/FedFisher/survival-comparator-validation/lifecycle')
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.store = ArtifactStore(self.helper.config, source, directory=root/self.helper.config['run_id'])
        self.store.initialize()
        self.execution = dict(job_id='fedfisher-survival-comparators-fixture', attempt=1,
            execution_profile='cpu-1core', estimated_runtime_seconds=20., allocation=dict(cpu_cores=1,
                memory_mib=8192, gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))

    def complete(self):
        self.store.start(self.execution, 0)
        result, arrays = self.helper.completed()
        result['identity'] = self.store.identity(0)
        self.store.complete(result, arrays, 0)
        return result, arrays

    def test_actual_fit_publishes_then_restores_without_inference_or_overwrite(self):
        self.assertEqual(self.store.status(0), 'missing')
        result, arrays = self.complete()
        with mock.patch('effect_modifier_experiment.survival_mlp8.methods.MLP8Fit.predict_eta', side_effect=AssertionError('inference')):
            self.assertEqual(self.store.status(0), 'validated')
        with self.assertRaises(FileExistsError):
            self.store.start(self.execution, 0)
        with self.assertRaises(ValueError):
            self.store.complete(result, arrays, 0)
        with self.assertRaises(ValueError):
            self.store.fail(RuntimeError('completed is immutable'), 0)

    def test_partial_failed_and_wrong_source_outputs_fail_closed(self):
        self.store.start(self.execution, 0)
        with self.assertRaises(ValueError):
            self.store.status(0)
        self.store.fail(RuntimeError('fixture numerical failure'), 0)
        self.assertFalse((self.store.cell_dir(0)/'done.json').exists())
        with self.assertRaises(ValueError):
            self.store.start(self.execution, 0)
        with mock.patch.object(original_store, 'current_provenance', return_value=dict(self.source, git_dirty='1')):
            with self.assertRaises(ValueError):
                self.store.status(1)

    def test_rehashed_metrics_cannot_be_promoted_to_scientific_completion(self):
        self.complete()
        cell = self.store.cell_dir(0)
        result = json.loads((cell/'result.json').read_bytes())
        result['prediction']['fedavg']['combined']['rmse'] += 1
        (cell/'result.json').write_text(json.dumps(result))
        marker = json.loads((cell/'done.json').read_bytes())
        marker['artifacts']['result.json'] = hashlib.sha256((cell/'result.json').read_bytes()).hexdigest()
        (cell/'done.json').write_text(json.dumps(marker))
        with self.assertRaises(ValueError):
            self.store.status(0)


if __name__ == '__main__':
    unittest.main()
