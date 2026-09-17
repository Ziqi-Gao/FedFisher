"""Survival pilot/formal protocol-v2 fixtures; no scientific launch or central mutation."""
from copy import deepcopy
import json
import re
import sys
from types import ModuleType
import unittest
from unittest import mock

import test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.survival import protocol

PROPOSAL_ROOT = fixture.PROJECT_ROOT/'effect_modifier_experiment/server_scheduler/survival-v1'


class SurvivalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        self.path, self.job = self.helper.fixture('survival-pilot-running-job.json')

    def validate(self, job=None, environment=None):
        job = self.job if job is None else job
        return fixture.ADAPTER.validate_running_job(job,
            self.helper.environment(self.path, job) if environment is None else environment, self.path)

    def test_pilot_and_formal_tasks_dispatch_in_foreground_through_existing_adapter(self):
        self.assertEqual(protocol.TASKS, fixture.ADAPTER.SURVIVAL_TASKS)
        for phase in ('pilot', 'shard'):
            path, job = self.helper.fixture('survival-'+phase+'-running-job.json')
            environment = self.helper.environment(path, job)
            normalized = fixture.ADAPTER.load_running_job(path, environment)
            runner = ModuleType('effect_modifier_experiment.survival.runner')
            runner.run_job = mock.Mock(return_value=19)
            with mock.patch.dict(sys.modules, {runner.__name__: runner}), mock.patch.object(
                    fixture.ADAPTER, '_safe_runtime') as runtime:
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, environment), 19)
            runtime.assert_called_once_with(job['job_id'], 1)
            runner.run_job.assert_called_once_with(normalized)

    def test_five_methods_two_scenarios_fixed_seed_without_caller_hardware_choices(self):
        coverage = set()
        for cell in protocol.CELLS:
            for scenario in protocol.SCENARIOS:
                p = dict(cell=cell, scenario=scenario)
                job = dict(self.job, parameters=p)
                self.assertEqual(self.validate(job)['parameters'], p)
                self.assertEqual(protocol.seed_sequence(protocol.PILOT_TASK, p), (0,))
                coverage.add(protocol.run_id(protocol.PILOT_TASK, p))
        self.assertEqual(len(coverage), 10)
        for key in ('seed', 'shard_index', 'shard_count', 'cpu_cores', 'gpu_count',
                    'resources', 'execution_profile', 'output_dir', 'epochs', 'command'):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, parameters=dict(self.job['parameters'], **{key: 1})))
        for key in self.job['parameters']:
            p = dict(self.job['parameters']); del p[key]
            with self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, parameters=p))
            for value in (None, True, [], {}, 'unknown'):
                with self.subTest(key=key, value=value), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(dict(self.job, parameters=dict(self.job['parameters'], **{key: value})))

    def test_full_v2_allocation_environment_attempt_and_runtime_guards_apply(self):
        for key, value in (('schema_version', 1), ('execution_profile', 'other'), ('state', 'pending'),
                           ('estimated_runtime_seconds', 0)):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, **{key: value}))
        for key, value in (('cpu_cores', 2), ('memory_mib', 4096), ('gpu_count', 1), ('gpu_memory_mib', 1)):
            job = deepcopy(self.job); job['allocation'][key] = value
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)
        environment = self.helper.environment(self.path, self.job)
        for key in environment:
            changed = dict(environment); del changed[key]
            with self.subTest(missing=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(environment=changed)
        for key, value in (('SERVER_SCHEDULER_ATTEMPT', '0'), ('CUDA_VISIBLE_DEVICES', '0'),
                           ('SERVER_SCHEDULER_GPU_INDEX', ''), ('SERVER_SCHEDULER_GPU_UUIDS', 'foreign')):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(environment=dict(environment, **{key: value}))

    def test_old_task_names_do_not_accept_survival_scientific_parameters(self):
        for task in fixture.ADAPTER.ALLOWED_TASKS - set(protocol.TASKS):
            with self.subTest(task=task), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, task=task))

    def test_formal_shards_cover_exactly_300_seeds_with_shared_run_identity(self):
        for cell in protocol.CELLS:
            for scenario in protocol.SCENARIOS:
                configs, ids = [], set()
                for count in (8, 16):
                    seeds = []
                    for index in range(count):
                        p = dict(cell=cell, scenario=scenario, shard_index=index, shard_count=count)
                        job = dict(self.job, task=protocol.FORMAL_TASK, parameters=p)
                        self.assertEqual(self.validate(job)['parameters'], p)
                        shard = protocol.seed_sequence(protocol.FORMAL_TASK, p)
                        self.assertTrue(all(seed % count == index for seed in shard))
                        seeds.extend(shard)
                        configs.append(protocol.run_config(protocol.FORMAL_TASK, p))
                        ids.add(protocol.run_id(protocol.FORMAL_TASK, p))
                    self.assertEqual(sorted(seeds), list(range(300)))
                self.assertTrue(all(value == configs[0] for value in configs))
                self.assertEqual(configs[0]['seeds'], list(range(300)))
                self.assertNotIn('seed', configs[0])
                self.assertEqual(ids, {'survival_v1_'+cell+'_'+scenario+'_300seeds'})
                pilot = protocol.run_config(protocol.PILOT_TASK, dict(cell=cell, scenario=scenario))
                for key in ('dgp', 'method', 'evaluation', 'pseudo', 'mean_link', 'public_initial_eta', 'timing'):
                    self.assertEqual(pilot[key], configs[0][key])

    def test_formal_cannot_override_science_seed_resources_or_shard_partition(self):
        good = dict(cell='dslasso', scenario='iid', shard_index=0, shard_count=16)
        cases = [dict(good, **{key: 1}) for key in (
            'seed', 'tau', 'cpu_cores', 'gpu_count', 'output_dir', 'acceptance_path', 'skip_pilot_gate')]
        cases.extend(dict(good, shard_count=count, shard_index=index) for count, index in (
            (1, 0), (4, 0), (8, 8), (16, 16), (16, -1), (True, 0), (16, False)))
        for key in good:
            missing = dict(good); del missing[key]; cases.append(missing)
        for p in cases:
            with self.subTest(parameters=p), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, task=protocol.FORMAL_TASK, parameters=p))

    def test_strict_schema_examples_and_disabled_proposal_are_consistent(self):
        schema = json.loads((PROPOSAL_ROOT/'pilot-parameters.schema.json').read_text())
        self.assertEqual(schema['type'], 'object')
        self.assertIs(schema['additionalProperties'], False)
        self.assertEqual(set(schema['required']), {'cell', 'scenario'})
        self.assertEqual(set(schema['properties']), set(schema['required']))
        self.assertEqual(schema['properties']['cell']['enum'], list(protocol.CELLS))
        self.assertEqual(schema['properties']['scenario']['enum'], list(protocol.SCENARIOS))
        coverage = set()
        examples = list((PROPOSAL_ROOT/'examples').glob('*.job-request.json'))
        self.assertEqual(len(examples), 10)
        for path in examples:
            request = json.loads(path.read_text())
            self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
            self.assertEqual(request['schema_version'], 2)
            self.assertEqual(request['task'], protocol.PILOT_TASK)
            self.assertEqual(path.name, request['job_id']+'.job-request.json')
            p = protocol.validate_parameters(request['task'], request['parameters'])
            coverage.add((p['cell'], p['scenario']))
        self.assertEqual(coverage, {(cell, scenario) for cell in protocol.CELLS for scenario in protocol.SCENARIOS})
        formal = json.loads((PROPOSAL_ROOT/'shard-parameters.schema.json').read_text())
        self.assertIs(formal['additionalProperties'], False)
        self.assertEqual(set(formal['required']), {'cell', 'scenario', 'shard_index', 'shard_count'})
        self.assertEqual(set(formal['properties']), set(formal['required']))
        self.assertEqual(formal['properties']['cell'], schema['properties']['cell'])
        self.assertEqual(formal['properties']['scenario'], schema['properties']['scenario'])
        self.assertEqual(formal['properties']['shard_count']['enum'], [8, 16])
        self.assertEqual(formal['allOf'][0]['then']['properties']['shard_index']['maximum'], 7)
        formal_examples = list((PROPOSAL_ROOT/'formal-examples').glob('*.job-request.json'))
        self.assertEqual(len(formal_examples), 10)
        formal_coverage = set()
        for path in formal_examples:
            request = json.loads(path.read_text())
            self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
            self.assertEqual(request['task'], protocol.FORMAL_TASK)
            self.assertEqual(request['schema_version'], 2)
            self.assertEqual(path.name, request['job_id']+'.job-request.json')
            p = protocol.validate_parameters(request['task'], request['parameters'])
            formal_coverage.add((p['cell'], p['scenario']))
        self.assertEqual(formal_coverage, coverage)
        proposal = (PROPOSAL_ROOT/'registration-proposal.toml').read_text()
        self.assertEqual(re.findall(r'(?m)^enabled = (.+)$', proposal), ['false'])
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        self.assertEqual(proposal.count('[[tasks.execution_profiles]]'), 2)
        self.assertIn('name = "'+protocol.PILOT_TASK+'"', proposal)
        self.assertIn('unmeasured admission estimates', proposal)
        self.assertIn('not outbox requests or submitted jobs', proposal)
        self.assertIn('name = "'+protocol.FORMAL_TASK+'"', proposal)
        self.assertIn('do not admit formal work before that gate', proposal)
        self.assertEqual(re.findall(r'(?m)^estimated_runtime_seconds = (.+)$', proposal), ['86400', '604800'])


if __name__ == '__main__':
    unittest.main()
