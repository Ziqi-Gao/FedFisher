"""Admission fixtures for model/signal ablations; no scheduler or scientific jobs."""
from copy import deepcopy
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.model_signal_ablation import protocol


PROPOSAL_ROOT = fixture.PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/model-signal-ablation-v1'
FIXTURES = ('model-signal-pilot-running-job.json', 'model-signal-shard-running-job.json')


class ModelSignalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        # Fixture imports must use this checkout; production stays canonical.
        patch = mock.patch.object(fixture.ADAPTER, 'PROJECT_ROOT', fixture.PROJECT_ROOT)
        patch.start()
        self.addCleanup(patch.stop)

    def validate(self, path, job, environment=None):
        if environment is None:
            environment = self.helper.environment(path, job)
        return fixture.ADAPTER.validate_running_job(job, environment, path)

    def test_both_tasks_dispatch_in_foreground_after_complete_manifest_validation(self):
        from effect_modifier_experiment.model_signal_ablation import runner
        self.assertEqual(protocol.TASKS, fixture.ADAPTER.MODEL_SIGNAL_TASKS)
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            environment = self.helper.environment(path, job)
            expected = self.validate(path, job)
            with mock.patch.object(runner, 'run_shard', return_value=19) as run, mock.patch.object(
                    fixture.ADAPTER, '_safe_runtime') as runtime:
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, environment), 19)
            runtime.assert_called_once_with(job['job_id'], 1)
            run.assert_called_once_with(expected)

    def test_manifest_validation_does_not_import_numerics_before_thread_setup(self):
        script = '''
import importlib.util
import json
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('adapter', sys.argv[1])
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
adapter.PROJECT_ROOT = Path(sys.argv[1]).resolve().parents[1]
fixture = json.loads(sys.argv[2])
adapter.load_running_job(Path(fixture['path']), fixture['environment'])
assert 'effect_modifier_experiment.model_signal_ablation.protocol' in sys.modules
for module in ('numpy', 'torch', 'effect_modifier_experiment.model_signal_ablation.runner',
               'effect_modifier_experiment.model_signal_ablation.artifacts'):
    assert module not in sys.modules, module + ' imported before runtime thread setup'
'''
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            payload = dict(path=str(path), environment=self.helper.environment(path, job))
            result = subprocess.run([sys.executable, '-B', '-c', script,
                str(fixture.ADAPTER_PATH), json.dumps(payload)],
                cwd=str(fixture.PROJECT_ROOT),
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_pilot_has_exact_288_requests_and_576_units_without_dsLasso_duplication(self):
        path, job = self.helper.fixture(FIXTURES[0])
        requests, units = set(), set()
        self.assertEqual({setting: len(spec['cells']) for setting, spec in protocol.SETTINGS.items()},
                         {'small_mlp': 10, 'weak_signal_1': 31, 'weak_signal_05': 31})
        for setting, spec in protocol.SETTINGS.items():
            self.assertEqual(sum(cell == 'dslassocov' for cell in spec['cells']),
                             0 if setting == 'small_mlp' else 1)
            for cell in spec['cells']:
                for variant in protocol.VARIANTS:
                    parameters = dict(setting=setting, cell=cell, variant=variant,
                                      shard_index=0, shard_count=1)
                    self.assertEqual(self.validate(path, dict(job, parameters=parameters))['parameters'],
                                     parameters)
                    self.assertEqual(protocol.seed_sequence(protocol.PILOT_TASK, parameters), (0,))
                    requests.add(protocol.run_id(protocol.PILOT_TASK, parameters))
                    units.update((setting, cell, variant, 0, split) for split in protocol.SPLITS)
        self.assertEqual(len(requests), 288)
        self.assertEqual(len(units), 576)

    def test_formal_shards_partition_all_seeds_without_changing_run_identity(self):
        path, job = self.helper.fixture(FIXTURES[1])
        for setting, cell in (('small_mlp', 'fedavg_small_e30_budget30'),
                              ('weak_signal_1', 'fedfisher_kfac_deep'),
                              ('weak_signal_05', 'dslassocov')):
            run_ids = set()
            for count in (8, 16):
                seeds = []
                for index in range(count):
                    parameters = dict(setting=setting, cell=cell, variant='control',
                                      shard_index=index, shard_count=count)
                    self.assertEqual(self.validate(path, dict(job, parameters=parameters))['parameters'],
                                     parameters)
                    owned = protocol.seed_sequence(protocol.FORMAL_TASK, parameters)
                    self.assertTrue(all(seed % count == index for seed in owned))
                    seeds.extend(owned)
                    run_ids.add(protocol.run_id(protocol.FORMAL_TASK, parameters))
                self.assertEqual(sorted(seeds), list(range(300)))
            self.assertEqual(len(run_ids), 1)

    def test_rejects_extra_fields_invalid_types_and_incompatible_setting_cell_pairs(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            good = job['parameters']
            candidates = [dict(good, **{key: 1}) for key in (
                'seed', 'signal_strength', 'model', 'hidden_dims', 'epochs', 'max_iter',
                'tolerance', 'output_dir', 'acceptance_path', 'cpu_cores', 'gpu_count',
                'resources', 'execution_profile')]
            candidates.extend(dict(good, setting='small_mlp', cell=cell) for cell in (
                'dslassocov', 'fedavg_shallow_e1_patience', 'fedfisher_diag_deep'))
            for key in good:
                missing = dict(good); del missing[key]; candidates.append(missing)
                candidates.extend(dict(good, **{key: value}) for value in (None, True, [], {}))
            candidates.extend(dict(good, **{key: 'unreviewed'}) for key in ('setting', 'cell', 'variant'))
            counts = ((0, 0), (1, 1), (8, 0)) if name == FIXTURES[0] else (
                (1, 0), (4, 0), (8, 8), (16, 16), (16, -1))
            candidates.extend(dict(good, shard_count=count, shard_index=index) for count, index in counts)
            for parameters in candidates:
                with self.subTest(name=name, parameters=parameters), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, parameters=parameters))

    def test_complete_v2_environment_allocation_attempt_and_runtime_guards_apply(self):
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

    def test_new_parameters_cannot_enter_old_task_names(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            for task in fixture.ADAPTER.ALLOWED_TASKS - set(protocol.TASKS) - set(
                    fixture.ADAPTER.MODEL_SIGNAL_RECOVERY_TASKS):
                with self.subTest(name=name, task=task), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, task=task))

    def test_schemas_and_source_examples_match_fixed_protocol(self):
        required = {'setting', 'cell', 'variant', 'shard_index', 'shard_count'}
        for phase in ('pilot', 'shard'):
            schema = json.loads((PROPOSAL_ROOT / (phase+'-parameters.schema.json')).read_text())
            self.assertEqual(schema['type'], 'object')
            self.assertIs(schema['additionalProperties'], False)
            self.assertEqual(set(schema['required']), required)
            self.assertEqual(set(schema['properties']), required)
            self.assertEqual(schema['properties']['setting']['enum'], list(protocol.SETTINGS))
            self.assertEqual(schema['properties']['cell']['enum'], list(protocol.CELLS))
            self.assertEqual(schema['properties']['variant']['enum'], list(protocol.VARIANTS))
            restriction = schema['allOf'][0]
            self.assertEqual(restriction['if']['properties']['setting'], {'const': 'small_mlp'})
            self.assertEqual(restriction['then']['properties']['cell']['enum'],
                             list(protocol.SETTINGS['small_mlp']['cells']))
            if phase == 'pilot':
                self.assertEqual(schema['properties']['shard_index']['const'], 0)
                self.assertEqual(schema['properties']['shard_count']['const'], 1)
            else:
                self.assertEqual(schema['properties']['shard_index']['maximum'], 15)
                self.assertEqual(schema['properties']['shard_count']['enum'], [8, 16])
                self.assertEqual(schema['allOf'][1], {
                    'if': {'properties': {'shard_count': {'const': 8}}},
                    'then': {'properties': {'shard_index': {'maximum': 7}}}})
        examples = list((PROPOSAL_ROOT / 'examples').glob('*.job-request.json'))
        self.assertEqual(len(examples), 2)
        tasks = set()
        for path in examples:
            request = json.loads(path.read_text())
            self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
            self.assertEqual(request['schema_version'], 2)
            self.assertEqual(request['project'], 'FedFisher')
            self.assertEqual(path.name, request['job_id'] + '.job-request.json')
            self.assertTrue(request['job_id'].startswith('fedfisher-ms-'))
            protocol.validate_parameters(request['task'], request['parameters'])
            tasks.add(request['task'])
        self.assertEqual(tasks, set(protocol.TASKS))

    def test_plan_counts_and_disabled_proposal_do_not_authorize_admission(self):
        plan = json.loads((PROPOSAL_ROOT / 'ablation-plan.json').read_text())
        self.assertEqual(plan['protocol'], protocol.PROTOCOL)
        self.assertEqual(plan['variants'], list(protocol.VARIANTS))
        self.assertEqual(plan['splits'], list(protocol.SPLITS))
        for setting, spec in protocol.SETTINGS.items():
            self.assertEqual(plan['settings'][setting], dict(signal_strength=spec['signal_strength'],
                                                           cells=list(spec['cells'])))
        combinations = sum(len(spec['cells']) for spec in protocol.SETTINGS.values())
        self.assertEqual(plan['pilot']['task'], protocol.PILOT_TASK)
        self.assertEqual(plan['pilot']['request_count'], combinations * 4)
        self.assertEqual(plan['pilot']['scientific_cell_count'], combinations * 4 * 2)
        self.assertEqual(plan['formal']['task'], protocol.FORMAL_TASK)
        self.assertEqual(plan['formal']['scientific_cell_count'], combinations * 4 * 2 * 300)
        for count in (8, 16):
            self.assertEqual(plan['formal']['request_count_by_shard_count'][str(count)],
                             combinations * 4 * count)
        self.assertIs(plan['formal']['publication_authorized'], False)
        self.assertIs(plan['direct_execution_allowed'], False)
        self.assertIs(plan['hardware_parameters_allowed'], False)
        proposal = (PROPOSAL_ROOT / 'registration-proposal.toml').read_text()
        self.assertEqual(re.findall(r'(?m)^enabled = (.+)$', proposal), ['false'])
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        self.assertEqual(proposal.count('[[tasks.execution_profiles]]'), 2)
        for task in protocol.TASKS:
            self.assertIn('name = "' + task + '"', proposal)
        self.assertEqual(re.findall(r'(?m)^cpu_cores_max = (.+)$', proposal), ['1', '1'])
        self.assertEqual(re.findall(r'(?m)^host_memory_max_mib = (.+)$', proposal), ['8192', '8192'])
        self.assertEqual(re.findall(r'(?m)^gpu_count = (.+)$', proposal), ['0', '0'])
        self.assertIn('same-source pilot acceptance', proposal)
        self.assertIn('operator runtime/memory review', proposal)
        self.assertIn('unmeasured admission estimates', proposal)
        self.assertIn('not outbox requests or submitted jobs', proposal)
        self.assertIn('entrypoint = "/home/del6500/projects/FedFisher/scripts/', proposal)


if __name__ == '__main__':
    unittest.main()
