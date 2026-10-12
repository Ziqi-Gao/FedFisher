"""Recovery admission fixtures; no scientific fitting or central scheduler calls."""
from copy import deepcopy
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.survival import recovery
from effect_modifier_experiment.survival_mlp8 import protocol as mlp8
from effect_modifier_experiment.survival_mlp8 import discovery_protocol as discovery


PROPOSAL_ROOT = fixture.PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-recovery-v1'
REQUEST_FIXTURES = fixture.PROJECT_ROOT / 'tests/fixtures/server_scheduler_requests'
FIXTURES = ('survival-dslasso-diagnostic-running-job.json',
            'survival-dslasso-recovery-shard-running-job.json',
            'survival-dslasso-repair-pilot-running-job.json',
            'survival-dslasso-repair-shard-running-job.json',
            'survival-dslasso-integrated-pilot-running-job.json',
            'survival-dslasso-integrated-shard-running-job.json')


class SurvivalRecoveryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()

    def validate(self, path, job, environment=None):
        if environment is None:
            environment = self.helper.environment(path, job)
        return fixture.ADAPTER.validate_running_job(job, environment, path)

    def test_both_tasks_dispatch_in_foreground_after_full_manifest_validation(self):
        self.assertEqual(recovery.TASKS, fixture.ADAPTER.SURVIVAL_RECOVERY_TASKS)
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            environment = self.helper.environment(path, job)
            expected = self.validate(path, job)
            with mock.patch.object(recovery, 'run_job', return_value=19) as run, mock.patch.object(
                    fixture.ADAPTER, '_safe_runtime') as runtime:
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, environment), 19)
            runtime.assert_called_once_with(job['job_id'], 1)
            run.assert_called_once_with(expected)

    def test_manifest_validation_does_not_import_blas_before_runtime_thread_limits(self):
        script = '''
import importlib.util
import json
import pathlib
import sys
spec = importlib.util.spec_from_file_location('adapter', sys.argv[1])
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
fixture = json.loads(sys.argv[2])
adapter.load_running_job(pathlib.Path(fixture['path']), fixture['environment'])
assert 'effect_modifier_experiment.survival.recovery_protocol' in sys.modules
for module in ('numpy', 'torch', 'effect_modifier_experiment.survival.recovery',
               'effect_modifier_experiment.survival.artifacts'):
    assert module not in sys.modules, module + ' imported before runtime thread setup'
'''
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            payload = dict(path=str(path), environment=self.helper.environment(path, job))
            result = subprocess.run([sys.executable, '-B', '-c', script,
                str(fixture.ADAPTER_PATH), json.dumps(payload)],
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_diagnostic_enum_covers_fixed_cases_and_recipes_without_free_scientific_choices(self):
        path, job = self.helper.fixture(FIXTURES[0])
        coverage = set()
        for case, context in recovery.CASES.items():
            for recipe in recovery.RECIPES:
                parameters = dict(case=case, recipe=recipe)
                candidate = dict(job, parameters=parameters)
                self.assertEqual(self.validate(path, candidate)['parameters'], parameters)
                self.assertEqual(recovery.seed_sequence(recovery.DIAGNOSTIC_TASK, parameters),
                                 (context['seed'],))
                coverage.add((context['scenario'], context['seed'], recipe))
        self.assertEqual(len(coverage), 108)
        for key in ('seed', 'scenario', 'fold', 'lambda_index', 'max_iter', 'max_step',
                    'tolerance', 'output_dir', 'cpu_cores', 'resources', 'execution_profile'):
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(path, dict(job, parameters=dict(job['parameters'], **{key: 1})))
        for key in job['parameters']:
            parameters = dict(job['parameters']); del parameters[key]
            with self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(path, dict(job, parameters=parameters))
            for value in (None, True, [], {}, 'unreviewed'):
                with self.subTest(key=key, value=value), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, parameters=dict(job['parameters'], **{key: value})))

    def test_formal_shards_cover_600_cells_with_one_scientific_identity_per_scenario(self):
        path, job = self.helper.fixture(FIXTURES[1])
        total = set()
        for scenario in ('iid', 'covariate_shift'):
            run_ids, configs = set(), []
            for count in (8, 16):
                seeds = []
                for index in range(count):
                    parameters = dict(scenario=scenario, shard_index=index, shard_count=count)
                    self.assertEqual(self.validate(path, dict(job, parameters=parameters))['parameters'],
                                     parameters)
                    owned = recovery.seed_sequence(recovery.FORMAL_TASK, parameters)
                    self.assertTrue(all(seed % count == index for seed in owned))
                    seeds.extend(owned)
                    total.update((scenario, seed) for seed in owned)
                    run_ids.add(recovery.run_id(recovery.FORMAL_TASK, parameters))
                    configs.append(recovery.run_config(recovery.FORMAL_TASK, parameters,
                                                       recipe='budget_only'))
                self.assertEqual(sorted(seeds), list(range(300)))
            self.assertEqual(len(run_ids), 1)
            self.assertTrue(all(config == configs[0] for config in configs))
        self.assertEqual(len(total), 600)

    def test_formal_rejects_recipe_override_and_invalid_shard_or_additional_parameters(self):
        path, job = self.helper.fixture(FIXTURES[1])
        good = job['parameters']
        cases = [dict(good, **{key: 1}) for key in (
            'case', 'seed', 'recipe', 'cell', 'max_iter', 'max_step', 'tolerance',
            'output_dir', 'acceptance_path', 'skip_diagnostic_gate', 'cpu_cores', 'gpu_count')]
        cases.extend(dict(good, shard_count=count, shard_index=index) for count, index in (
            (1, 0), (4, 0), (8, 8), (16, 16), (16, -1), (True, 0), (16, False)))
        for key in good:
            missing = dict(good); del missing[key]; cases.append(missing)
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(path, dict(job, parameters=parameters))

    def test_full_v2_environment_allocation_attempt_and_runtime_guards_apply_to_both_tasks(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            for key, value in (('schema_version', 1), ('execution_profile', 'other'),
                               ('state', 'pending'), ('estimated_runtime_seconds', 0)):
                with self.subTest(name=name, key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, **{key: value}))
            for key, value in (('cpu_cores', 2), ('memory_mib', 4096), ('gpu_count', 1)):
                candidate = deepcopy(job); candidate['allocation'][key] = value
                with self.subTest(name=name, key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, candidate)
            environment = self.helper.environment(path, job)
            for key in environment:
                changed = dict(environment); del changed[key]
                with self.subTest(name=name, missing=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, job, changed)
            for key, value in (('SERVER_SCHEDULER_ATTEMPT', '0'), ('CUDA_VISIBLE_DEVICES', '0'),
                               ('SERVER_SCHEDULER_GPU_INDEX', ''), ('SERVER_SCHEDULER_GPU_UUIDS', 'foreign')):
                with self.subTest(name=name, key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, job, dict(environment, **{key: value}))

    def test_new_scientific_parameters_cannot_enter_old_task_names(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            # MLP8 is a newer independent recipe. Its formal task deliberately
            # shares scenario/shard field names, with its own runner and namespace.
            for task in fixture.ADAPTER.ALLOWED_TASKS - set(recovery.TASKS) - set(mlp8.TASKS) - set(discovery.TASKS):
                with self.subTest(name=name, task=task), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, task=task))

    def test_same_shard_fields_preserve_distinct_new_mlp8_recipe(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            if set(job['parameters']) == {'scenario', 'shard_index', 'shard_count'}:
                candidate = self.validate(path, dict(job, task=mlp8.FORMAL_TASK))
                self.assertEqual(candidate['task'], mlp8.FORMAL_TASK)
                config = mlp8.run_config(candidate['task'], candidate['parameters'])
                self.assertEqual(config['protocol'], mlp8.PROTOCOL)
                self.assertTrue(config['run_id'].startswith('survival_mlp8_v1_'))
                self.assertEqual(config['method']['optimizer']['architecture'], 'mlp8')

    def test_strict_schemas_and_two_examples_match_protocol_and_omit_hardware(self):
        diagnostic = json.loads((PROPOSAL_ROOT / 'diagnostic-parameters.schema.json').read_text())
        formal = json.loads((PROPOSAL_ROOT / 'shard-parameters.schema.json').read_text())
        for schema, required in ((diagnostic, {'case', 'recipe'}),
                                 (formal, {'scenario', 'shard_index', 'shard_count'})):
            self.assertEqual(schema['type'], 'object')
            self.assertIs(schema['additionalProperties'], False)
            self.assertEqual(set(schema['required']), required)
            self.assertEqual(set(schema['properties']), required)
        self.assertEqual(set(diagnostic['properties']['case']['enum']), set(recovery.CASES))
        self.assertEqual(diagnostic['properties']['recipe']['enum'], list(recovery.RECIPES))
        self.assertEqual(formal['properties']['scenario']['enum'], ['iid', 'covariate_shift'])
        self.assertEqual(formal['properties']['shard_count']['enum'], [8, 16])
        self.assertEqual(formal['allOf'][0]['then']['properties']['shard_index']['maximum'], 7)
        examples = list((REQUEST_FIXTURES / 'survival-recovery-v1').glob('*.job-request.json'))
        self.assertEqual(len(examples), 2)
        tasks = set()
        for path in examples:
            request = json.loads(path.read_text())
            self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
            self.assertEqual(request['schema_version'], 2)
            self.assertEqual(request['project'], 'FedFisher')
            self.assertEqual(path.name, request['job_id'] + '.job-request.json')
            recovery.validate_parameters(request['task'], request['parameters'])
            tasks.add(request['task'])
        self.assertEqual(tasks, set(recovery.tasks_for_epoch(1)))

    def test_registration_is_disabled_and_requires_diagnostic_and_operator_review(self):
        proposal = (PROPOSAL_ROOT / 'registration-proposal.toml').read_text()
        self.assertEqual(re.findall(r'(?m)^enabled = (.+)$', proposal), ['false'])
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        self.assertEqual(proposal.count('[[tasks.execution_profiles]]'), 2)
        for task in recovery.tasks_for_epoch(1):
            self.assertIn('name = "' + task + '"', proposal)
        self.assertEqual(re.findall(r'(?m)^cpu_cores_max = (.+)$', proposal), ['1', '1'])
        self.assertEqual(re.findall(r'(?m)^host_memory_max_mib = (.+)$', proposal), ['8192', '8192'])
        self.assertEqual(re.findall(r'(?m)^gpu_count = (.+)$', proposal), ['0', '0'])
        self.assertIn('same-source diagnostic acceptance', proposal)
        self.assertIn('operator runtime/memory review', proposal)
        self.assertIn('unmeasured admission estimates', proposal)
        self.assertIn('not outbox requests or submitted jobs', proposal)

    def test_repair_pilot_has_exact_cases_and_no_recipe_or_runtime_override(self):
        path, job = self.helper.fixture(FIXTURES[2])
        self.assertEqual(len(recovery.REPAIR_CASES), 30)
        for case in recovery.REPAIR_CASES:
            self.assertEqual(self.validate(path, dict(job, parameters=dict(case=case)))['parameters'],
                             dict(case=case))
        for p in ({}, {'case': 'iid_001'}, {'case': True},
                  {'case': 'covariate_shift_049', 'recipe': 'budget_only'},
                  {'case': 'covariate_shift_049', 'max_iter': 16000}):
            with self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(path, dict(job, parameters=p))

    def test_repair_proposal_and_schemas_are_separate_and_disabled(self):
        root = PROPOSAL_ROOT.parent / 'survival-repair-v2'
        schema = json.loads((root / 'diagnostic-parameters.schema.json').read_text())
        self.assertEqual(schema['required'], ['case'])
        self.assertEqual(set(schema['properties']), {'case'})
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(set(schema['properties']['case']['enum']), set(recovery.REPAIR_CASES))
        self.assertEqual(json.loads((root / 'shard-parameters.schema.json').read_text()),
                         json.loads((PROPOSAL_ROOT / 'shard-parameters.schema.json').read_text()))
        proposal = (root / 'registration-proposal.toml').read_text()
        self.assertEqual(re.findall(r'(?m)^enabled = (.+)$', proposal), ['false'])
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        for task in recovery.tasks_for_epoch(2):
            self.assertIn('name = "' + task + '"', proposal)
        requests = [json.loads(p.read_text())
                    for p in (REQUEST_FIXTURES / 'survival-repair-v2').glob('*.json')]
        self.assertEqual({r['task'] for r in requests}, set(recovery.tasks_for_epoch(2)))
        for request in requests:
            self.assertNotIn('resources', request)
            self.assertNotIn('execution_profile', request)
            recovery.validate_parameters(request['task'], request['parameters'])


if __name__ == '__main__':
    unittest.main()
