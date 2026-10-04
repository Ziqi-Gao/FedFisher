"""Closed completion plans and lazy scheduler routes without scientific runs."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure.requests import canonical
from effect_modifier_experiment.model_signal_ablation import completion_protocol as protocol
from effect_modifier_experiment.model_signal_ablation import completion_requests as plan
from effect_modifier_experiment.model_signal_ablation import recovery_protocol as previous
from effect_modifier_experiment.scheduling import catalog, dispatch
import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from tests.paths import PROJECT_ROOT


PROPOSAL_ROOT = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/model-signal-completion-v3'
FIXTURES = ('model-signal-completion-pilot-running-job.json',
            'model-signal-completion-shard-running-job.json')


def key(parameters):
    return tuple(parameters[name] for name in ('setting', 'cell', 'variant', 'shard_index', 'shard_count'))


class CompletionProtocolAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()

    def validate(self, path, job, environment=None):
        return fixture.ADAPTER.validate_running_job(job,
            self.helper.environment(path, job) if environment is None else environment, path)

    def test_exact_failed_whole_shards_and_units(self):
        specs = plan.build_requests(cohort='formal')
        expected = {(protocol.BUDGET_CELL, 'control', index) for index in range(16)}
        expected.add((protocol.BUDGET_CELL, 'site_gaussian_shift', 0))
        expected.update((protocol.PATIENCE_CELL, 'site_gaussian_local_permutation', index)
                        for index in range(9, 16))
        observed, units = set(), set()
        for spec in specs:
            p = spec['request']['parameters']
            observed.add((p['cell'], p['variant'], p['shard_index']))
            self.assertEqual(p['setting'], 'weak_signal_05')
            self.assertEqual(p['shard_count'], 16)
            self.assertEqual(spec['seed_sequence'], list(range(p['shard_index'], 300, 16)))
            for seed in spec['seed_sequence']:
                for split in spec['splits']:
                    identity = (p['setting'], p['cell'], p['variant'], seed, split)
                    self.assertNotIn(identity, units)
                    units.add(identity)
        self.assertEqual(observed, expected)
        self.assertEqual(len(specs), 24)
        self.assertEqual(len(units), 896)
        self.assertEqual(171898 + 6 + len(units), 172800)
        self.assertEqual(len({s['run_id'] for s in specs}), 3)

    def test_pilot_covers_both_recipes_all_variants_and_both_splits(self):
        specs = plan.build_requests()
        self.assertEqual(len(specs), 8)
        self.assertEqual({(s['request']['parameters']['cell'], s['request']['parameters']['variant'])
                          for s in specs},
                         {(cell, variant) for cell in protocol.CELLS for variant in protocol.VARIANTS})
        for spec in specs:
            self.assertEqual(spec['seed_sequence'], [0])
            self.assertEqual(spec['splits'], ['iid', 'noniid'])
            self.assertTrue(spec['run_id'].endswith('_v3_pilot'))
            self.assertNotIn('case', spec['configuration'])

    def test_formal_allowlist_rejects_every_unowned_recipe_variant_shard(self):
        allowed = {key(p) for p in protocol.FORMAL_PARAMETERS}
        count = 0
        for cell in protocol.CELLS:
            for variant in protocol.VARIANTS:
                for index in range(16):
                    p = dict(setting='weak_signal_05', cell=cell, variant=variant,
                             shard_index=index, shard_count=16)
                    if key(p) in allowed:
                        self.assertEqual(protocol.validate_parameters(protocol.FORMAL_TASK, p), p)
                        count += 1
                    else:
                        with self.assertRaisesRegex(ValueError, '24 archived failed shards'):
                            protocol.validate_parameters(protocol.FORMAL_TASK, p)
        self.assertEqual(count, 24)

    def test_scientific_recipes_match_v2_with_type_exact_configuration(self):
        for cohort in ('pilot', 'formal'):
            for spec in plan.build_requests(cohort=cohort):
                task, p = spec['request']['task'], spec['request']['parameters']
                old_p = dict(p)
                if cohort == 'pilot': old_p['case'] = 'baseline'
                old = previous.run_config(previous.PILOT_TASK if cohort == 'pilot'
                                          else previous.FORMAL_TASK, old_p)
                new = deepcopy(spec['configuration'])
                for field in ('protocol', 'task', 'run_id'):
                    self.assertNotEqual(new.pop(field), old.pop(field))
                old.pop('case', None)
                self.assertEqual(canonical(new), canonical(old))
                representative = protocol.parameters_for_config(spec['configuration'])
                self.assertEqual(canonical(protocol.run_config(task, representative)),
                                 canonical(spec['configuration']))
        p = protocol.FORMAL_PARAMETERS[-1]
        self.assertEqual(protocol.parameters_for_config(protocol.run_config(protocol.FORMAL_TASK, p))['shard_index'], 9)
        recipe = protocol.method_config(protocol.PATIENCE_CELL)
        self.assertEqual(recipe['optimizer']['max_rounds'], 200)
        self.assertEqual(recipe['optimizer']['patience'], 10)
        self.assertEqual(recipe['optimizer']['min_delta'], 1e-4)

    def test_fixed_contract_rejects_extra_fields_and_type_confusion(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            good = job['parameters']
            bad = [dict(good, **{field: value}) for field, value in (
                ('case', 'baseline'), ('seed', 25), ('seeds', [0]), ('max_rounds', 1000),
                ('signal_strength', .25), ('hidden_dims', [16]), ('output_dir', '/scr/del6500/FedFisher'),
                ('resources', {}), ('setting', 'small_mlp'), ('cell', 'fedavg_small_e1_patience'),
                ('shard_count', 8), ('shard_index', -1), ('shard_index', 16))]
            for field in good:
                missing = dict(good); del missing[field]; bad.append(missing)
                bad.extend(dict(good, **{field: value}) for value in (None, True, [], {}))
            for candidate in bad:
                with self.subTest(name=name, parameters=candidate), self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, parameters=candidate))

    def test_request_id_limits_and_changes_never_reuse_old_namespaces(self):
        for cohort in ('pilot', 'formal'):
            first, second = plan.build_requests(1, cohort), plan.build_requests(2, cohort)
            for a, b in zip(first, second):
                self.assertNotEqual(a['request']['job_id'], b['request']['job_id'])
                self.assertEqual(a['configuration'], b['configuration'])
                request = a['request']
                self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
                self.assertRegex(request['job_id'], r'^fedfisher-ms-v3-')
                self.assertLessEqual(len(request['job_id']), 128)
                self.assertEqual(a['request_sha256'], plan.sha256(plan.encoded(request)))
                self.assertEqual(a['configuration_sha256'], plan.sha256(canonical(a['configuration']).encode()))
                self.assertNotEqual(a['run_id'], protocol.previous_config(request['task'], request['parameters'])['run_id'])
        for value in (0, -1, True, '1', None, 10 ** 80):
            with self.assertRaises(ValueError): plan.build_requests(value)
        with self.assertRaises(ValueError): plan.build_requests(cohort='remaining')

    def test_plan_rejects_numeric_or_coverage_drift(self):
        original = protocol.run_config
        def changed(task, parameters):
            config = original(task, parameters)
            config['dgp']['signal_strength'] = 0.25
            return config
        with mock.patch.object(protocol, 'run_config', side_effect=changed), self.assertRaisesRegex(ValueError, 'numerical recipe'):
            plan.build_requests()
        with mock.patch.object(protocol, 'seed_sequence', return_value=(1,)), self.assertRaisesRegex(ValueError, 'coverage'):
            plan.build_requests()
        with mock.patch.object(protocol, 'FORMAL_PARAMETERS', protocol.FORMAL_PARAMETERS[:-1]), self.assertRaisesRegex(ValueError, 'coverage'):
            plan.build_requests(cohort='formal')

    def test_new_routes_load_runner_only_after_runtime_configuration(self):
        self.assertEqual(catalog.MODEL_SIGNAL_COMPLETION_TASKS, protocol.TASKS)
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            expected = self.validate(path, job)
            events = []
            run = mock.Mock(return_value=23)
            def load(module):
                self.assertEqual(events, ['runtime'])
                self.assertEqual(module, 'effect_modifier_experiment.model_signal_ablation.completion')
                events.append('runner_import')
                return SimpleNamespace(run_shard=run)
            with mock.patch.object(fixture.ADAPTER, '_safe_runtime', side_effect=lambda *a: events.append('runtime')) as runtime, \
                    mock.patch.object(dispatch, 'import_module', side_effect=load):
                self.assertEqual(fixture.ADAPTER.run_from_manifest(path, self.helper.environment(path, job)), 23)
            runtime.assert_called_once_with(job['job_id'], 1)
            run.assert_called_once_with(expected)

    def test_validation_stays_lightweight_and_planning_never_loads_execution(self):
        code = '''
import importlib, json, sys
from pathlib import Path
from effect_modifier_experiment.scheduling import adapter
from effect_modifier_experiment.model_signal_ablation import completion_requests
data = json.loads(sys.argv[1])
adapter.load_running_job(Path(data['path']), data['environment'])
for name in sys.modules:
    assert name.split('.')[0] not in {'numpy', 'torch', 'nngeometry', 'models', 'data'}, name
completion_requests.build_requests()
completion_requests.build_requests(cohort='formal')
for name in sys.modules:
    if name.startswith('effect_modifier_experiment.'):
        assert not name.endswith(('.runner', '.acceptance', '.completion', '.completion_acceptance', '.completion_archive', '.recovery_acceptance')), name
    assert not name.startswith('effect_modifier_experiment.scheduling.publication'), name
'''
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            result = subprocess.run([sys.executable, '-B', '-c', code,
                json.dumps(dict(path=str(path), environment=self.helper.environment(path, job)))],
                cwd=PROJECT_ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_tasks_inherit_shared_manifest_and_environment_guards(self):
        for name in FIXTURES:
            path, job = self.helper.fixture(name)
            for field, value in (('schema_version', 1), ('state', 'pending'),
                                 ('execution_profile', 'other'), ('estimated_runtime_seconds', 0)):
                with self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, dict(job, **{field: value}))
            for field, value in (('cpu_cores', 2), ('memory_mib', 4096), ('gpu_count', 1)):
                bad = deepcopy(job); bad['allocation'][field] = value
                with self.assertRaises(fixture.ADAPTER.AdapterError): self.validate(path, bad)
            env = self.helper.environment(path, job)
            for field in env:
                bad = dict(env); del bad[field]
                with self.assertRaises(fixture.ADAPTER.AdapterError): self.validate(path, job, bad)
            for field, value in (('SERVER_SCHEDULER_GPU_INDEX', ''), ('SERVER_SCHEDULER_ATTEMPT', '0'),
                                 ('CUDA_VISIBLE_DEVICES', '0')):
                with self.assertRaises(fixture.ADAPTER.AdapterError):
                    self.validate(path, job, dict(env, **{field: value}))

    def test_disabled_proposal_schema_and_examples_have_exact_scope(self):
        for phase, task, expected in (('pilot', protocol.PILOT_TASK, protocol.PILOT_PARAMETERS),
                                      ('shard', protocol.FORMAL_TASK, protocol.FORMAL_PARAMETERS)):
            schema = json.loads((PROPOSAL_ROOT / (phase+'-parameters.schema.json')).read_text())
            fields = {'setting', 'cell', 'variant', 'shard_index', 'shard_count'}
            self.assertEqual(set(schema['required']), fields)
            self.assertEqual(set(schema['properties']), fields)
            self.assertIs(schema['additionalProperties'], False)
            self.assertEqual({canonical(p) for p in schema['enum']}, {canonical(p) for p in expected})
            self.assertEqual(len(schema['enum']), len(expected))
            for p in schema['enum']: protocol.validate_parameters(task, p)
            for field in ('shard_index', 'shard_count'):
                self.assertEqual(schema['properties'][field]['type'], 'integer')
        proposal = (PROPOSAL_ROOT / 'registration-proposal.toml').read_text()
        self.assertEqual(re.findall(r'(?m)^enabled = (.+)$', proposal), ['false'])
        self.assertEqual(proposal.count('[[tasks]]'), 2)
        for field, expected in (('cpu_cores_max', ['1', '1']), ('host_memory_max_mib', ['8192', '8192']),
                                 ('gpu_count', ['0', '0'])):
            self.assertEqual(re.findall(r'(?m)^'+field+r' = (.+)$', proposal), expected)
        examples = list((PROPOSAL_ROOT / 'examples').glob('*.job-request.json'))
        self.assertEqual(len(examples), 2)
        for path in examples:
            value = json.loads(path.read_text())
            protocol.validate_parameters(value['task'], value['parameters'])
            self.assertEqual(set(value), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
        metadata = json.loads((PROPOSAL_ROOT / 'completion-plan.json').read_text())
        self.assertEqual(metadata['source_closeout'], protocol.SOURCE_CLOSEOUT)
        self.assertEqual(metadata['formal']['parameters'], list(protocol.FORMAL_PARAMETERS))
        self.assertFalse(metadata['formal']['publication_authorized'])
        self.assertFalse(metadata['new_execution_authorized_by_plan'])

    def test_review_inventory_covers_routes_publication_and_shared_lifecycle(self):
        for label, path in catalog.adapter_review_sources().items():
            self.assertEqual(plan.REVIEW_SOURCES[label], path)
        for label in ('archive', 'acceptance', 'protocol', 'runner', 'request_plan', 'request_publication',
                      'request_publication_support', 'shared_recovery_runner', 'shared_artifacts',
                      'shared_provenance', 'shared_evidence', 'shared_requests', 'shared_proof_publisher'):
            self.assertIn(label, plan.REVIEW_SOURCES)
            self.assertIn(catalog.PROJECT_ROOT, plan.REVIEW_SOURCES[label].parents)


if __name__ == '__main__':
    unittest.main()
