"""Real immutable outbox/adapter fixtures, with no scientific execution."""
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
from tests.integration import test_fedfisher_server_scheduler_adapter as fixture
from tests.integration.test_survival_mlp8_submission import provenance
from effect_modifier_experiment.infrastructure.requests import review_sources as source_bindings
from effect_modifier_experiment.scheduling import catalog
from effect_modifier_experiment.scheduling.publication import survival_comparators as builder
from effect_modifier_experiment.survival_comparators import inputs, protocol, requests


class ComparatorAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        self.path, self.job = self.helper.fixture('survival-comparators-pilot-running-job.json')

    def test_full_v2_dispatches_only_after_allocated_runtime_setup(self):
        self.assertEqual(protocol.TASKS, catalog.SURVIVAL_COMPARATOR_TASKS)
        env, events = self.helper.environment(self.path, self.job), []
        normalized = fixture.ADAPTER.load_running_job(self.path, env)
        runner = ModuleType('effect_modifier_experiment.survival_comparators.runner')
        runner.run_job = mock.Mock(side_effect=lambda job: events.append('runner') or 0)
        with mock.patch.dict(sys.modules, {runner.__name__: runner}), mock.patch.object(
                fixture.ADAPTER, '_safe_runtime', side_effect=lambda *args: events.append('runtime')):
            self.assertEqual(fixture.ADAPTER.run_from_manifest(self.path, env), 0)
        runner.run_job.assert_called_once_with(normalized)
        self.assertEqual(events, ['runtime', 'runner'])

    def test_closed_ten_seed_parameters_and_false_gpu_overrides(self):
        for scenario in protocol.SCENARIOS:
            for seed in range(10):
                p = dict(scenario=scenario, seed=seed)
                self.assertEqual(protocol.seed_sequence(protocol.PILOT_TASK, p), (seed,))
        for bad in (dict(scenario='iid', seed=True), dict(scenario='iid', seed=10),
                dict(scenario='iid', seed=-1), dict(scenario='combine', seed=0),
                dict(scenario='iid', seed=1.0), dict(scenario='iid')):
            with self.subTest(parameters=bad), self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.PILOT_TASK, bad)
        for field in ('epochs', 'architecture', 'rounds', 'cpu_cores', 'gpu_count', 'output_dir'):
            job = dict(self.job, parameters=dict(self.job['parameters'], **{field: 'override'}))
            with self.subTest(field=field), self.assertRaises(fixture.ADAPTER.AdapterError):
                fixture.ADAPTER.validate_running_job(job, self.helper.environment(self.path, job), self.path)
        mutations = [('schema_version', 1), ('execution_profile', 'other'), ('estimated_runtime_seconds', 0)]
        for key, value in mutations:
            job = dict(self.job, **{key: value})
            with self.assertRaises(fixture.ADAPTER.AdapterError):
                fixture.ADAPTER.validate_running_job(job, self.helper.environment(self.path, job), self.path)
        for key, value in (('gpu_count', 1), ('cpu_cores', 2), ('memory_mib', 4096)):
            job = deepcopy(self.job)
            job['allocation'][key] = value
            with self.assertRaises(fixture.ADAPTER.AdapterError):
                fixture.ADAPTER.validate_running_job(job, self.helper.environment(self.path, job), self.path)
        for resource in (dict(cpu_cores=2), dict(gpu_count=1), dict(memory_mib=4096)):
            job = dict(self.job, resources=resource)
            with self.assertRaises(fixture.ADAPTER.AdapterError):
                fixture.ADAPTER.validate_running_job(job, self.helper.environment(self.path, job), self.path)
        env = self.helper.environment(self.path, self.job)
        env['SERVER_SCHEDULER_GPU_INDICES'] = '0'
        with self.assertRaises(fixture.ADAPTER.AdapterError):
            fixture.ADAPTER.validate_running_job(self.job, env, self.path)

    def test_task_schema_proposal_and_fresh_process_guard_remain_lazy(self):
        code = ("from effect_modifier_experiment.scheduling.validation import _validate_parameters; "
                "from effect_modifier_experiment.survival_comparators import protocol; import sys; "
                "_validate_parameters(protocol.PILOT_TASK, {'scenario':'iid','seed':9}); "
                "assert 'numpy' not in sys.modules and 'torch' not in sys.modules; "
                "assert 'effect_modifier_experiment.survival_comparators.runner' not in sys.modules")
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT_ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-comparators-v1'
        schema = json.loads((directory / 'pilot-parameters.schema.json').read_text())
        self.assertIs(schema['additionalProperties'], False)
        self.assertEqual(set(schema['required']), {'scenario', 'seed'})
        self.assertEqual(set(schema['properties']), {'scenario', 'seed'})
        self.assertEqual(schema['properties']['scenario']['enum'], list(protocol.SCENARIOS))
        self.assertEqual(schema['properties']['seed'], dict(type='integer', minimum=0, maximum=9))
        proposal = (directory / 'registration-proposal.toml').read_text()
        self.assertIn('enabled = false', proposal)
        self.assertIn('name = "' + protocol.PILOT_TASK + '"', proposal)
        self.assertNotIn('comparators-shard', proposal)


class ComparatorPublicationTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/survival-comparators-validation/submission')
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        # Hash real files from this checkout; no foreign canonical checkout is
        # borrowed when this bounded test executes in a review worktree.
        bindings = lambda: source_bindings({'submission': Path(__file__).resolve(),
            'protocol': PROJECT_ROOT / 'effect_modifier_experiment/survival_comparators/protocol.py'}, PROJECT_ROOT)
        for name, value in (('OUTBOX_ROOT', self.root / 'outbox'), ('REPORT_ROOT', self.root / 'reports')):
            self.stack.enter_context(mock.patch.object(builder, name, value))
        self.stack.enter_context(mock.patch.object(builder, 'review_sources', side_effect=bindings))
        self.stack.enter_context(mock.patch.object(builder, 'current_provenance', return_value=provenance()))
        self.stack.enter_context(mock.patch.object(builder, 'validate_inputs', return_value=True))
        for module in (builder, inputs):
            self.stack.enter_context(mock.patch.object(module, 'historical_binding', return_value={'fixture': True}))

    def test_exact_twenty_requests_same_seed_independent_recipe_no_machine_choices(self):
        specs, coverage, configurations = requests.build_requests(), {}, {}
        self.assertEqual(len(specs), 20)
        for item in specs:
            raw, config = item['request'], item['configuration']
            self.assertEqual(set(raw), {'schema_version', 'project', 'task', 'job_id', 'parameters', 'priority'})
            self.assertEqual(set(raw['parameters']), {'scenario', 'seed'})
            self.assertEqual(item['seed_sequence'], [raw['parameters']['seed']])
            scenario = raw['parameters']['scenario']
            coverage.setdefault(scenario, []).extend(item['seed_sequence'])
            self.assertEqual(configurations.setdefault(scenario, config), config)
            self.assertEqual(config['method']['optimizer']['local_epochs'], 5)
            self.assertEqual(config['method']['optimizer']['total_local_epochs'], 30)
            self.assertEqual(config['fit_origin']['dslasso'], 'accepted_r3_model_replay')
        self.assertEqual(coverage, {s: list(range(10)) for s in protocol.SCENARIOS})
        self.assertEqual(len({item['request']['job_id'] for item in specs}), 20)
        self.assertTrue({s['filename'] for s in specs}.isdisjoint(s['filename'] for s in requests.build_requests(2)))
        for bad in (True, 0, -1):
            with self.assertRaises(ValueError):
                requests.build_requests(bad)

    def test_inventory_last_readback_and_no_overwrite(self):
        writes, original = [], builder.atomic_create
        def create(path, data):
            writes.append(path)
            original(path, data)
        with mock.patch.object(builder, 'atomic_create', side_effect=create):
            receipt = builder.write_requests()
        inventory = Path(receipt['inventory_path'])
        self.assertEqual(len(writes), 21)
        self.assertEqual(writes[-1], inventory)
        self.assertEqual(receipt['request_count'], 20)
        self.assertIs(receipt['submitted'], False)
        self.assertIs(receipt['formal_expansion_authorized'], False)
        for item in receipt['requests']:
            self.assertEqual(json.loads(Path(item['path']).read_text())['parameters'], item['parameters'])
        before = inventory.read_bytes()
        with self.assertRaises(FileExistsError):
            builder.write_requests()
        self.assertEqual(before, inventory.read_bytes())

    def test_source_or_historical_drift_cannot_publish_ready_inventory(self):
        cases = [('current_provenance', [provenance(), dict(provenance(), source_tree_sha256='5' * 64)]),
                 ('historical_binding', [{'fixture': True}, {'fixture': False}])]
        for generation, (name, values) in enumerate(cases, 1):
            with mock.patch.object(builder, name, side_effect=values), self.assertRaises(ValueError):
                builder.write_requests(generation)
            self.assertFalse((self.root / 'reports' / (requests.bundle_name(generation) + '.inventory.json')).exists())
        with mock.patch.object(builder, 'current_provenance', side_effect=ValueError('dirty')), self.assertRaises(ValueError):
            builder.write_requests(3)
        self.assertFalse((self.root / 'outbox' / requests.bundle_name(3)).exists())

    def test_mutated_request_bytes_fail_readback_without_ready_inventory(self):
        original = builder.atomic_create
        def create(path, data):
            if path.name.endswith('.job-request.json'):
                raw = json.loads(data)
                raw['priority'] = 1
                data = json.dumps(raw).encode()
            original(path, data)
        with mock.patch.object(builder, 'atomic_create', side_effect=create), self.assertRaisesRegex(ValueError, 'readback'):
            builder.write_requests()
        self.assertFalse((self.root / 'reports' / (requests.bundle_name() + '.inventory.json')).exists())


if __name__ == '__main__':
    unittest.main()
