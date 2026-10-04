"""Pooled task admission fixtures; no scientific launch or central mutation."""
from copy import deepcopy
import json
import os
import subprocess
import sys
from types import ModuleType
import unittest
from unittest import mock

from tests.paths import PROJECT_ROOT
import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.scheduling import catalog
from effect_modifier_experiment.survival_pooled import protocol

PROPOSAL_ROOT = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-pooled-v1'


class SurvivalPooledAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        self.path, self.job = self.helper.fixture('survival-pooled-pilot-running-job.json')

    def validate(self, job=None, environment=None):
        job = self.job if job is None else job
        return fixture.ADAPTER.validate_running_job(job,
            self.helper.environment(self.path, job) if environment is None else environment, self.path)

    def test_both_routes_validate_and_dispatch_after_runtime_setup(self):
        self.assertEqual(protocol.TASKS, catalog.SURVIVAL_POOLED_TASKS)
        for phase in ('pilot', 'shard'):
            path, job = self.helper.fixture('survival-pooled-' + phase + '-running-job.json')
            environment = self.helper.environment(path, job)
            normalized = fixture.ADAPTER.load_running_job(path, environment)
            runner = ModuleType('effect_modifier_experiment.survival_pooled.runner')
            runner.run_job = mock.Mock(return_value=23)
            events = []
            runner.run_job.side_effect = lambda normalized: events.append('runner') or 23
            with mock.patch.dict(sys.modules, {runner.__name__: runner}), mock.patch.object(
                    fixture.ADAPTER, '_safe_runtime', side_effect=lambda *args: events.append('runtime')) as runtime:
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, environment), 23)
            runtime.assert_called_once_with(job['job_id'], 1)
            runner.run_job.assert_called_once_with(normalized)
            self.assertEqual(events, ['runtime', 'runner'])

    def test_fixed_two_seed_pilots_three_families_two_scenarios(self):
        ids = set()
        for family in protocol.FAMILIES:
            for scenario in protocol.SCENARIOS:
                p = dict(family=family, scenario=scenario)
                self.assertEqual(self.validate(dict(self.job, parameters=p))['parameters'], p)
                self.assertEqual(protocol.seed_sequence(protocol.PILOT_TASK, p), (0, 49))
                ids.add(protocol.run_id(protocol.PILOT_TASK, p))
        self.assertEqual(len(ids), 6)
        for key in ('cell', 'seed', 'epochs', 'tau', 'output_dir', 'cpu_cores', 'gpu_count',
                    'resources', 'execution_profile', 'command', 'skip_pilot_gate', 'shard_count'):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, parameters=dict(self.job['parameters'], **{key: 1})))
        for key in self.job['parameters']:
            for value in (None, True, [], {}, 'unknown'):
                with self.subTest(key=key, value=value), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(dict(self.job, parameters=dict(self.job['parameters'], **{key: value})))

    def test_full_plural_gpu_allocation_environment_guards_apply(self):
        for field, value in (('schema_version', 1), ('execution_profile', 'other'),
                             ('estimated_runtime_seconds', 0), ('state', 'pending')):
            with self.subTest(field=field), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, **{field: value}))
        for field, value in (('cpu_cores', 2), ('memory_mib', 4096), ('gpu_count', 1)):
            job = deepcopy(self.job)
            job['allocation'][field] = value
            with self.subTest(field=field), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)
        environment = self.helper.environment(self.path, self.job)
        for key in environment:
            changed = dict(environment)
            del changed[key]
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(environment=changed)
        for key, value in (('SERVER_SCHEDULER_ATTEMPT', '0'), ('CUDA_VISIBLE_DEVICES', '0'),
                           ('SERVER_SCHEDULER_GPU_INDEX', ''), ('SERVER_SCHEDULER_GPU_UUIDS', 'foreign')):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(environment=dict(environment, **{key: value}))

    def test_formal_shards_cover_300_seeds_with_partition_independent_config(self):
        for family in protocol.FAMILIES:
            for scenario in protocol.SCENARIOS:
                configs, ids = [], set()
                for count in (8, 16):
                    seeds = []
                    for index in range(count):
                        p = dict(family=family, scenario=scenario, shard_index=index, shard_count=count)
                        job = dict(self.job, task=protocol.FORMAL_TASK, parameters=p)
                        self.assertEqual(self.validate(job)['parameters'], p)
                        seeds.extend(protocol.seed_sequence(protocol.FORMAL_TASK, p))
                        configs.append(protocol.run_config(protocol.FORMAL_TASK, p))
                        ids.add(protocol.run_id(protocol.FORMAL_TASK, p))
                    self.assertEqual(sorted(seeds), list(range(300)))
                self.assertTrue(all(config == configs[0] for config in configs))
                self.assertEqual(len(ids), 1)
                self.assertEqual(configs[0]['cell'], family)
                self.assertEqual(configs[0]['pseudo_scopes'], ['site_local_pv', 'pooled_km_pv'])
                pilot = protocol.run_config(protocol.PILOT_TASK, dict(family=family, scenario=scenario))
                for key in ('dgp', 'method', 'evaluation', 'mean_link', 'historical_reference', 'public_initial_eta'):
                    self.assertEqual(configs[0][key], pilot[key])
                self.assertEqual(pilot['seeds'], [0, 49])
                self.assertFalse(pilot['evaluation']['modifier_permutation_performed'])

    def test_task_guard_is_lightweight_before_thread_setup(self):
        code = """import sys
from effect_modifier_experiment.survival_pooled import protocol
protocol.validate_parameters(protocol.PILOT_TASK, dict(family='deep',scenario='iid'))
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT_ROOT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_schemas_and_disabled_additive_proposal_match_task_guard(self):
        pilot = json.loads((PROPOSAL_ROOT / 'pilot-parameters.schema.json').read_text())
        formal = json.loads((PROPOSAL_ROOT / 'shard-parameters.schema.json').read_text())
        for schema in (pilot, formal):
            self.assertIs(schema['additionalProperties'], False)
            self.assertEqual(schema['properties']['family']['enum'], list(protocol.FAMILIES))
            self.assertEqual(schema['properties']['scenario']['enum'], list(protocol.SCENARIOS))
        self.assertEqual(set(pilot['required']), {'family', 'scenario'})
        self.assertEqual(set(formal['required']), {'family', 'scenario', 'shard_index', 'shard_count'})
        self.assertEqual(formal['properties']['shard_count']['enum'], [8, 16])
        proposal = (PROPOSAL_ROOT / 'registration-proposal.toml').read_text()
        self.assertIn('enabled = false', proposal)
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        self.assertIn('unmeasured admission estimates', proposal)
        self.assertIn('Do not admit formal work before that gate', proposal)
        for task in protocol.TASKS:
            self.assertIn('name = "' + task + '"', proposal)


if __name__ == '__main__':
    unittest.main()
