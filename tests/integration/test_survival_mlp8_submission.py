"""CPU allocation fixtures and immutable request publication; no science launch."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from tests.paths import PROJECT_ROOT
from tests.integration.request_fixtures import bind_review_sources
import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.scheduling import catalog
from effect_modifier_experiment.scheduling.publication import survival_mlp8 as builder
from effect_modifier_experiment.survival_mlp8 import protocol


def provenance():
    return dict(git_commit='1' * 40, git_dirty='0', git_status_sha256='2' * 64,
        source_tree_sha256='3' * 64, provenance_sha256='4' * 64)


class MLP8AdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        self.path, self.job = self.helper.fixture('survival-mlp8-pilot-running-job.json')

    def test_full_v2_routes_dispatch_after_runtime(self):
        self.assertEqual(protocol.TASKS, catalog.SURVIVAL_MLP8_TASKS)
        for phase in ('pilot', 'shard'):
            path, job = self.helper.fixture('survival-mlp8-' + phase + '-running-job.json')
            env = self.helper.environment(path, job)
            normalized = fixture.ADAPTER.load_running_job(path, env)
            runner = ModuleType('effect_modifier_experiment.survival_mlp8.runner')
            events = []
            runner.run_job = mock.Mock(side_effect=lambda row: events.append('runner') or 0)
            with mock.patch.dict(sys.modules, {runner.__name__: runner}), mock.patch.object(
                    fixture.ADAPTER, '_safe_runtime', side_effect=lambda *args: events.append('runtime')):
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, env), 0)
            runner.run_job.assert_called_once_with(normalized)
            self.assertEqual(events, ['runtime', 'runner'])

    def test_rejects_architecture_and_resource_overrides(self):
        for field in ('family', 'cell', 'seed', 'epochs', 'architecture', 'cpu_cores', 'gpu_count', 'output_dir'):
            job = dict(self.job, parameters=dict(self.job['parameters'], **{field: 'deep'}))
            with self.subTest(field=field), self.assertRaises(fixture.ADAPTER.AdapterError):
                fixture.ADAPTER.validate_running_job(job, self.helper.environment(self.path, job), self.path)
        for field, value in [('schema_version', 1), ('execution_profile', 'other'), ('estimated_runtime_seconds', 0)]:
            job = dict(self.job, **{field: value})
            with self.subTest(field=field), self.assertRaises(fixture.ADAPTER.AdapterError):
                fixture.ADAPTER.validate_running_job(job, self.helper.environment(self.path, job), self.path)
        job = deepcopy(self.job)
        job['allocation']['gpu_count'] = 1
        with self.assertRaises(fixture.ADAPTER.AdapterError):
            fixture.ADAPTER.validate_running_job(job, self.helper.environment(self.path, job), self.path)

    def test_guard_is_lazy_and_schemas_are_closed(self):
        code = "from effect_modifier_experiment.survival_mlp8 import protocol; import sys; protocol.validate_parameters(protocol.PILOT_TASK,{'scenario':'iid'}); assert 'numpy' not in sys.modules and 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT_ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-mlp8-v1'
        for phase, fields in [('pilot', {'scenario'}), ('shard', {'scenario', 'shard_index', 'shard_count'})]:
            schema = json.loads((directory / (phase + '-parameters.schema.json')).read_text())
            self.assertFalse(schema['additionalProperties'])
            self.assertEqual(set(schema['required']), fields)
            self.assertEqual(set(schema['properties']), fields)
        self.assertIn('enabled = false', (directory / 'registration-proposal.toml').read_text())


class MLP8PublicationTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/survival-mlp8-validation/publication')
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        bind_review_sources(self.stack, builder)
        for name, value in [('OUTBOX_ROOT', self.root / 'outbox'), ('REPORT_ROOT', self.root / 'reports')]:
            self.stack.enter_context(mock.patch.object(builder, name, value))
        self.stack.enter_context(mock.patch.object(builder, 'current_provenance', return_value=provenance()))
        self.stack.enter_context(mock.patch.object(builder, 'historical_binding', return_value={'fixture': True}))

    def test_exact_pilot_and_formal_seed_coverage(self):
        for phase, shards, requests, units in [('pilot', None, 2, 4), ('formal', 16, 32, 600), ('formal', 8, 16, 600)]:
            specs = builder.build_requests(cohort=phase, shard_count=shards)
            self.assertEqual(len(specs), requests)
            self.assertEqual(sum(len(s['seed_sequence']) for s in specs), units)
            coverage = {}
            for spec in specs:
                raw, config = spec['request'], spec['configuration']
                self.assertEqual(set(raw), {'schema_version', 'project', 'task', 'job_id', 'parameters', 'priority'})
                self.assertEqual(config['method']['optimizer']['architecture'], 'mlp8')
                coverage.setdefault(raw['parameters']['scenario'], []).extend(spec['seed_sequence'])
            for seeds in coverage.values():
                self.assertEqual(sorted(seeds), [0, 49] if phase == 'pilot' else list(range(300)))
        first, second = builder.build_requests(1), builder.build_requests(2)
        self.assertTrue({s['request']['job_id'] for s in first}.isdisjoint({s['request']['job_id'] for s in second}))

    def test_inventory_last_no_overwrite_and_new_pilot_gate(self):
        writes, original = [], builder.atomic_create
        def write(path, data):
            writes.append(path)
            return original(path, data)
        with mock.patch.object(builder, 'atomic_create', side_effect=write):
            receipt = builder.write_requests()
        self.assertEqual(len(writes), 3)
        self.assertEqual(writes[-1], Path(receipt['inventory_path']))
        self.assertEqual(receipt['scientific_cell_count'], 4)
        self.assertEqual(receipt['fitted_model_count'], 16)
        self.assertFalse(receipt['submitted'])
        self.assertFalse(receipt['formal_plan']['publication_authorized'])
        data = Path(receipt['inventory_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            builder.write_requests()
        self.assertEqual(Path(receipt['inventory_path']).read_bytes(), data)
        with self.assertRaisesRegex(ValueError, 'same-source MLP8 pilot'):
            builder.write_requests(cohort='formal')
        self.assertFalse((self.root / 'outbox' / builder.bundle_name(cohort='formal')).exists())

    def test_dirty_or_drifting_sources_do_not_publish_ready_inventory(self):
        with mock.patch.object(builder, 'current_provenance', side_effect=ValueError('dirty')):
            with self.assertRaisesRegex(ValueError, 'dirty'):
                builder.write_requests()
        self.assertFalse((self.root / 'outbox').exists())
        with mock.patch.object(builder, 'current_provenance', side_effect=[provenance(), dict(provenance(), source_tree_sha256='5' * 64)]):
            with self.assertRaisesRegex(ValueError, 'changed'):
                builder.write_requests()
        self.assertFalse((self.root / 'reports' / builder.inventory_name()).exists())

    def test_source_drift_after_inventory_preserves_bytes_but_refuses_delivery(self):
        with mock.patch.object(builder, 'current_provenance', side_effect=[provenance(), provenance(),
                dict(provenance(), source_tree_sha256='5' * 64)]):
            with self.assertRaisesRegex(ValueError, 'after publication'):
                builder.write_requests()
        inventory = self.root / 'reports' / builder.inventory_name()
        self.assertTrue(inventory.exists())
        self.assertEqual(json.loads(inventory.read_text())['provenance'], provenance())


if __name__ == '__main__':
    unittest.main()
