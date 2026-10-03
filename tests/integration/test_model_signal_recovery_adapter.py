"""Closed recovery contracts and foreground dispatch; no scientific execution."""
from copy import deepcopy
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.model_signal_ablation import protocol as original
from effect_modifier_experiment.model_signal_ablation import recovery_protocol as protocol

PROPOSALS = fixture.PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2'
FIXTURES = tuple('model-signal-recovery-' + phase + '-running-job.json'
                 for phase in ('pilot', 'diagnostic', 'shard'))


class RecoveryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        patch = mock.patch.object(fixture.ADAPTER, 'PROJECT_ROOT', fixture.PROJECT_ROOT)
        patch.start()
        self.addCleanup(patch.stop)

    def validate(self, path, job, environment=None):
        return fixture.ADAPTER.validate_running_job(job,
            self.helper.environment(path, job) if environment is None else environment, path)

    def test_new_tasks_dispatch_after_shared_v2_validation(self):
        from effect_modifier_experiment.model_signal_ablation import recovery
        self.assertEqual(fixture.ADAPTER.MODEL_SIGNAL_RECOVERY_TASKS, protocol.TASKS)
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            expected = self.validate(path, job)
            with mock.patch.object(recovery, 'run_shard', return_value=23) as run, mock.patch.object(
                    fixture.ADAPTER, '_safe_runtime') as runtime:
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, self.helper.environment(path, job)), 23)
            runtime.assert_called_once_with(job['job_id'], 1)
            run.assert_called_once_with(expected)

    def test_validation_keeps_numerical_imports_after_thread_configuration(self):
        code = '''
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('adapter', sys.argv[1])
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
adapter.PROJECT_ROOT = Path(sys.argv[1]).resolve().parents[1]
data = json.loads(sys.argv[2])
adapter.load_running_job(Path(data['path']), data['environment'])
assert 'effect_modifier_experiment.model_signal_ablation.recovery_protocol' in sys.modules
for name in ('numpy', 'torch', 'effect_modifier_experiment.model_signal_ablation.recovery'):
    assert name not in sys.modules, name
'''
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            result = subprocess.run([sys.executable, '-B', '-c', code, str(fixture.ADAPTER_PATH),
                json.dumps(dict(path=str(path), environment=self.helper.environment(path, job)))],
                cwd=str(fixture.PROJECT_ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_parameter_allowlists_do_not_expand_recipes_or_accept_arbitrary_seeds(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            good = job['parameters']
            bad = [dict(good, **{key: value}) for key, value in (
                ('seed', 25), ('seeds', [9, 25, 41]), ('max_rounds', 201),
                ('patience', 5), ('hidden_dims', [16]), ('signal_strength', .25),
                ('output_dir', '/scr/del6500/FedFisher'), ('resources', {}), ('cpu_cores', 1))]
            for key in good:
                missing = dict(good); del missing[key]; bad.append(missing)
                bad.extend(dict(good, **{key: value}) for value in (None, True, [], {}))
            bad.extend(dict(good, setting='small_mlp', cell=cell)
                       for cell in ('dslassocov', 'fedfisher_diag_deep'))
            if job['task'] == protocol.FORMAL_TASK:
                bad.extend([dict(good, case='baseline'), dict(good, shard_count=8)])
            else:
                bad.extend([dict(good, shard_count=16), dict(good, shard_index=1),
                            dict(good, case='unreviewed')])
            if good.get('case') == 'cap_diagnostic':
                bad.extend([dict(good, variant='control'), dict(good, variant='local_permutation'),
                            dict(good, cell='fedavg_small_e5_patience'), dict(good, setting='weak_signal_1')])
            for p in bad:
                with self.subTest(name=name, parameters=p), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, parameters=p))

    def test_exact_seed_sequences_and_old_tasks_unchanged(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            p = job['parameters']
            seeds = protocol.seed_sequence(job['task'], p)
            expected = ((0,) if p.get('case') == 'baseline' else (9, 25, 41)) if 'case' in p else tuple(range(9, 300, 16))
            self.assertEqual(seeds, expected)
            if 'case' in p:
                with self.assertRaises(ValueError):
                    original.validate_parameters(original.PILOT_TASK, p)
        for setting, recipe in protocol.SETTINGS.items():
            for cell in recipe['cells']:
                p = dict(setting=setting, cell=cell, variant='control', shard_index=0, shard_count=16)
                old = original.run_config(original.FORMAL_TASK, p)
                new = protocol.run_config(protocol.FORMAL_TASK, p)
                for key in ('protocol', 'task', 'run_id'):
                    self.assertNotEqual(old.pop(key), new.pop(key))
                self.assertEqual(old, new)
        self.assertEqual(protocol.method_config('fedavg_small_e1_patience')['optimizer']['max_rounds'], 200)
        ids, seeds = set(), []
        for index in range(16):
            p = dict(setting='weak_signal_05', cell='dslassocov', variant='control', shard_index=index, shard_count=16)
            ids.add(protocol.run_id(protocol.FORMAL_TASK, p))
            seeds.extend(protocol.seed_sequence(protocol.FORMAL_TASK, p))
        self.assertEqual(len(ids), 1)
        self.assertEqual(sorted(seeds), list(range(300)))

    def test_shared_manifest_environment_guards_cover_new_tasks(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            for key, value in (('schema_version', 1), ('state', 'pending'),
                               ('execution_profile', 'other'), ('estimated_runtime_seconds', 0)):
                with self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, **{key: value}))
            for key, value in (('cpu_cores', 2), ('memory_mib', 4096), ('gpu_count', 1)):
                changed = deepcopy(job); changed['allocation'][key] = value
                with self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, changed)
            env = self.helper.environment(path, job)
            for key in env:
                changed = dict(env); del changed[key]
                with self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, job, changed)
            for key, value in (('SERVER_SCHEDULER_ATTEMPT', '0'), ('SERVER_SCHEDULER_GPU_INDEX', ''),
                               ('CUDA_VISIBLE_DEVICES', '0')):
                with self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, job, dict(env, **{key: value}))

    def test_disabled_proposal_and_schemas_match_closed_scope(self):
        for phase in ('pilot', 'shard'):
            schema = json.loads((PROPOSALS / (phase + '-parameters.schema.json')).read_text())
            required = {'setting', 'cell', 'variant', 'shard_index', 'shard_count'}
            if phase == 'pilot': required.add('case')
            self.assertEqual(set(schema['required']), required)
            self.assertEqual(set(schema['properties']), required)
            self.assertIs(schema['additionalProperties'], False)
            if phase == 'pilot':
                self.assertEqual(schema['properties']['case']['enum'], list(protocol.CASES))
                restricted = schema['allOf'][1]['then']['properties']
                self.assertEqual(restricted['variant']['enum'], list(protocol.DIAGNOSTIC_VARIANTS))
                self.assertEqual(restricted['cell']['const'], 'fedavg_small_e1_patience')
            else:
                self.assertEqual(schema['properties']['shard_count']['const'], 16)
        examples = list((PROPOSALS / 'examples').glob('*.job-request.json'))
        self.assertEqual(len(examples), 3)
        for path in examples:
            request = json.loads(path.read_text())
            self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
            protocol.validate_parameters(request['task'], request['parameters'])
        proposal = (PROPOSALS / 'registration-proposal.toml').read_text()
        self.assertEqual(re.findall(r'(?m)^enabled = (.+)$', proposal), ['false'])
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        self.assertEqual(re.findall(r'(?m)^cpu_cores_max = (.+)$', proposal), ['1', '1'])
        self.assertEqual(re.findall(r'(?m)^host_memory_max_mib = (.+)$', proposal), ['8192', '8192'])
        self.assertEqual(re.findall(r'(?m)^gpu_count = (.+)$', proposal), ['0', '0'])
        for task in protocol.TASKS: self.assertIn('name = "' + task + '"', proposal)
        plan = json.loads((PROPOSALS / 'recovery-plan.json').read_text())
        self.assertEqual(plan['source_closeout'], protocol.SOURCE_CLOSEOUT)
        self.assertEqual(plan['pilot']['request_count'], 290)
        self.assertEqual(plan['formal']['request_count'], 4479)
        self.assertIs(plan['formal']['publication_authorized'], False)


if __name__ == '__main__':
    unittest.main()
