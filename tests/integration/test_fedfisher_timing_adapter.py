"""No-compute contracts for original FedFisher CPU timing admission."""
import copy
import json
import re
import sys
from types import ModuleType
import unittest
from unittest import mock

import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.fedfisher_timing import protocol


PROPOSAL_ROOT = fixture.PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/fedfisher-timing-v1'
EXAMPLE_ROOT = fixture.PROJECT_ROOT / 'tests/fixtures/server_scheduler_requests/fedfisher-timing-v1'
EXPECTED_CELLS = {'fedfisher_diag_shallow', 'fedfisher_kfac_shallow',
                  'fedfisher_diag_deep', 'fedfisher_kfac_deep'}


class FedFisherTimingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        self.path, self.job = self.helper.fixture('fedfisher-timing-pilot-running-job.json')

    def validate(self, job, environment=None):
        return fixture.ADAPTER.validate_running_job(
            job, self.helper.environment(self.path, job) if environment is None else environment, self.path)

    def test_pilot_and_formal_fixtures_dispatch_foreground_to_the_new_runner(self):
        self.assertEqual(set(protocol.TASKS), set(fixture.ADAPTER.FEDFISHER_TIMING_TASKS))
        for phase, task in (('pilot', protocol.PILOT_TASK), ('shard', protocol.FORMAL_TASK)):
            path, raw = self.helper.fixture('fedfisher-timing-'+phase+'-running-job.json')
            environment = self.helper.environment(path, raw)
            job = fixture.ADAPTER.load_running_job(path, environment)
            self.assertEqual(job['task'], task)
            self.assertEqual(job['allocation']['cpu_cores'], 1)
            self.assertEqual(job['allocation']['memory_mib'], 8192)
            self.assertEqual(job['allocation']['gpu_count'], 0)
            self.assertIsNone(job['resources'])
            runner = ModuleType('effect_modifier_experiment.fedfisher_timing.runner')
            runner.run_shard = mock.Mock(return_value=29)
            with mock.patch.dict(sys.modules, {runner.__name__: runner}), mock.patch.object(
                    fixture.ADAPTER, '_safe_runtime') as runtime:
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, environment), 29)
            runtime.assert_called_once_with(job['job_id'], 1)
            runner.run_shard.assert_called_once_with(job)

    def test_exact_pilot_scope_and_formal_seed_partition(self):
        self.assertEqual(set(protocol.CELLS), EXPECTED_CELLS)
        self.assertEqual(tuple(protocol.SPLITS), ('iid', 'noniid'))
        self.assertEqual(tuple(protocol.VARIANTS), fixture.ADAPTER.VARIANTS)
        for cell in protocol.CELLS:
            for variant in protocol.VARIANTS:
                parameters = dict(cell=cell, variant=variant, shard_index=0, shard_count=1)
                job = copy.deepcopy(self.job)
                job['parameters'] = parameters
                self.assertEqual(self.validate(job)['parameters'], parameters)
                self.assertEqual(list(protocol.seed_sequence(protocol.PILOT_TASK, parameters)), [0])
                self.assertEqual(protocol.run_id(protocol.PILOT_TASK, parameters),
                                 'fedfisher_timing_'+cell+'_'+variant+'_v1_pilot')
                for count in (8, 16):
                    all_seeds = []
                    run_ids = set()
                    for index in range(count):
                        formal = dict(parameters, shard_index=index, shard_count=count)
                        seeds = list(protocol.seed_sequence(protocol.FORMAL_TASK, formal))
                        self.assertEqual(seeds, list(range(index, 300, count)))
                        all_seeds.extend(seeds)
                        run_ids.add(protocol.run_id(protocol.FORMAL_TASK, formal))
                    self.assertEqual(sorted(all_seeds), list(range(300)))
                    self.assertEqual(run_ids, {'fedfisher_timing_'+cell+'_'+variant+'_v1_300seeds'})

    def test_unknown_science_and_caller_machine_or_path_parameters_are_rejected(self):
        cases = []
        for key in ('seed', 'epochs', 'cpu_cores', 'gpu_count', 'resources', 'execution_profile',
                    'output_dir', 'run_id', 'device', 'command'):
            cases.append(dict(self.job['parameters'], **{key: 1}))
        for key in self.job['parameters']:
            missing = dict(self.job['parameters']); del missing[key]
            cases.append(missing)
            for value in ('unknown', None, True, [], {}):
                cases.append(dict(self.job['parameters'], **{key: value}))
        cases.extend([None, [], dict(self.job['parameters'], shard_count=8),
                      dict(self.job['parameters'], shard_index=1)])
        for parameters in cases:
            job = dict(self.job, parameters=parameters)
            with self.subTest(parameters=parameters), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)
        for count, index in ((1, 0), (4, 0), (8, 8), (16, 16), (16, -1), (True, 0)):
            job = dict(self.job, task=protocol.FORMAL_TASK,
                       parameters=dict(self.job['parameters'], shard_count=count, shard_index=index))
            with self.subTest(count=count, index=index), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)

    def test_wrong_allocation_environment_and_protocol_are_rejected(self):
        for key, value in (('schema_version', 1), ('execution_profile', 'other'), ('state', 'pending')):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, **{key: value}))
        for key, value in (('cpu_cores', 2), ('memory_mib', 4096), ('gpu_count', 1), ('gpu_memory_mib', 1)):
            job = copy.deepcopy(self.job); job['allocation'][key] = value
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)
        environment = self.helper.environment(self.path, self.job)
        for key in environment:
            altered = dict(environment); del altered[key]
            with self.subTest(missing=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(self.job, altered)
        for key, value in (('SERVER_SCHEDULER_ATTEMPT', '0'), ('CUDA_VISIBLE_DEVICES', '0'),
                           ('SERVER_SCHEDULER_GPU_INDEX', ''), ('SERVER_SCHEDULER_GPU_UUIDS', 'foreign')):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(self.job, dict(environment, **{key: value}))

    def test_old_tasks_cannot_be_repurposed_to_launch_fisher_timing(self):
        for task in fixture.ADAPTER.ALLOWED_TASKS - set(protocol.TASKS):
            with self.subTest(task=task), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(dict(self.job, task=task))

    def test_strict_schemas_examples_and_bounded_plan_agree(self):
        for phase in ('pilot', 'shard'):
            schema = json.loads((PROPOSAL_ROOT / (phase+'-parameters.schema.json')).read_text())
            self.assertEqual(schema['type'], 'object')
            self.assertIs(schema['additionalProperties'], False)
            self.assertEqual(set(schema['required']), {'cell', 'variant', 'shard_index', 'shard_count'})
            self.assertEqual(set(schema['properties']), set(schema['required']))
            self.assertEqual(set(schema['properties']['cell']['enum']), EXPECTED_CELLS)
            self.assertEqual(schema['properties']['variant']['enum'], list(protocol.VARIANTS))
            if phase == 'pilot':
                self.assertEqual(schema['properties']['shard_count']['const'], 1)
                self.assertEqual(schema['properties']['shard_index']['const'], 0)
            else:
                self.assertEqual(schema['properties']['shard_count']['enum'], [8, 16])
                self.assertEqual(schema['properties']['shard_index']['maximum'], 15)
                self.assertEqual(schema['allOf'][0]['then']['properties']['shard_index']['maximum'], 7)
        examples = list(EXAMPLE_ROOT.glob('*.job-request.json'))
        self.assertEqual(len(examples), 16)
        coverage = set()
        for path in examples:
            request = json.loads(path.read_text())
            self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
            self.assertEqual(request['task'], protocol.PILOT_TASK)
            self.assertEqual(request['schema_version'], 2)
            self.assertEqual(path.name, request['job_id']+'.job-request.json')
            parameters = protocol.validate_parameters(request['task'], request['parameters'])
            coverage.add((parameters['cell'], parameters['variant']))
        self.assertEqual(coverage, {(cell, variant) for cell in protocol.CELLS for variant in protocol.VARIANTS})
        plan = json.loads((PROPOSAL_ROOT / 'timing-plan.json').read_text())
        self.assertEqual(plan['pilot']['scientific_cell_count'], 32)
        self.assertEqual(plan['formal']['scientific_cell_count'], 9600)
        self.assertFalse(plan['formal']['publication_authorized'])
        self.assertFalse(plan['direct_execution_allowed'])

    def test_disabled_proposal_requires_measured_review_before_formal_admission(self):
        proposal = (PROPOSAL_ROOT / 'registration-proposal.toml').read_text()
        self.assertEqual(re.findall(r'(?m)^enabled = (.+)$', proposal), ['false'])
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        self.assertEqual(proposal.count('[[tasks.execution_profiles]]'), 2)
        for task in protocol.TASKS:
            self.assertIn('name = "'+task+'"', proposal)
        for key, value in {'kind': '"cpu"', 'resource_mode': '"fixed"', 'cpu_cores_min': '1',
                           'cpu_cores_preferred': '1', 'cpu_cores_max': '1',
                           'host_memory_default_mib': '8192', 'gpu_count': '0',
                           'gpu_memory_mib': '0', 'gpu_utilization_pct': '0'}.items():
            self.assertEqual(re.findall(r'(?m)^'+key+r' = (.+)$', proposal), [value, value])
        self.assertEqual(re.findall(r'(?m)^estimated_runtime_seconds = (.+)$', proposal), ['86400', '604800'])
        self.assertIn('unmeasured admission estimates', proposal)
        self.assertIn('same-source pilot scientific acceptance', proposal)
        self.assertIn('do not register/admit formal work', proposal)
        self.assertIn('not outbox requests or submitted jobs', proposal)


if __name__ == '__main__':
    unittest.main()
