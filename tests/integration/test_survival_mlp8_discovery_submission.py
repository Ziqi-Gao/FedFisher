"""Additive V2 admission/publication fixtures; never execute scientific work.

Reuse the original allocation and scratch-publication helpers, without
inheriting or duplicating their V1 tests.  Actual request bytes, source review
hashes, publication protections and proof-namespace rejection remain real.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest import mock

from tests.paths import PROJECT_ROOT
from tests.integration import test_fedfisher_server_scheduler_adapter as adapter_fixture
from tests.integration import test_survival_mlp8_submission as v1_fixture
from effect_modifier_experiment.scheduling import catalog
from effect_modifier_experiment.scheduling.publication import survival_mlp8 as builder
from effect_modifier_experiment.survival_mlp8 import acceptance as gate
from effect_modifier_experiment.survival_mlp8 import discovery_protocol as recipe
from effect_modifier_experiment.survival_mlp8 import protocol as prediction


class DiscoveryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = adapter_fixture.FedFisherServerSchedulerAdapterTests()

    def test_new_running_manifests_route_discovery_only_after_runtime(self):
        self.assertEqual(recipe.TASKS, catalog.SURVIVAL_MLP8_DISCOVERY_TASKS)
        for phase in ('pilot', 'shard'):
            path, job = self.helper.fixture('survival-mlp8-discovery-' + phase + '-running-job.json')
            env = self.helper.environment(path, job)
            normalized = adapter_fixture.ADAPTER.load_running_job(path, env)
            runner, events = ModuleType('effect_modifier_experiment.survival_mlp8.runner'), []
            runner.run_discovery_job = mock.Mock(side_effect=lambda row: events.append('discovery') or 0)
            runner.run_job = mock.Mock(side_effect=AssertionError('V1 dispatch'))
            with mock.patch.dict(sys.modules, {runner.__name__: runner}), mock.patch.object(
                    adapter_fixture.ADAPTER, '_safe_runtime', side_effect=lambda *args: events.append('runtime')):
                self.assertEqual(adapter_fixture.ADAPTER.run_from_manifest(path, env), 0)
            runner.run_discovery_job.assert_called_once_with(normalized)
            runner.run_job.assert_not_called()
            self.assertEqual(events, ['runtime', 'discovery'])

    def test_new_tasks_reject_discovery_recipe_and_machine_overrides(self):
        for phase in ('pilot', 'shard'):
            path, original = self.helper.fixture('survival-mlp8-discovery-' + phase + '-running-job.json')
            for field in ('repeats', 'top2', 'pseudo_scope', 'architecture', 'epochs', 'seed',
                          'cpu_cores', 'gpu_count', 'output_dir'):
                job = dict(original, parameters=dict(original['parameters'], **{field: 'override'}))
                with self.subTest(phase=phase, field=field), self.assertRaises(adapter_fixture.ADAPTER.AdapterError):
                    adapter_fixture.ADAPTER.validate_running_job(job, self.helper.environment(path, job), path)
            for field, value in (('schema_version', 1), ('execution_profile', 'other'),
                                 ('estimated_runtime_seconds', 0)):
                job = dict(original, **{field: value})
                with self.subTest(phase=phase, field=field), self.assertRaises(adapter_fixture.ADAPTER.AdapterError):
                    adapter_fixture.ADAPTER.validate_running_job(job, self.helper.environment(path, job), path)
            for field, value in (('gpu_count', 1), ('cpu_cores', 2), ('memory_mib', 4096)):
                job = deepcopy(original)
                job['allocation'][field] = value
                with self.subTest(phase=phase, allocation=field), self.assertRaises(adapter_fixture.ADAPTER.AdapterError):
                    adapter_fixture.ADAPTER.validate_running_job(job, self.helper.environment(path, job), path)

    def test_discovery_parameter_admission_remains_lazy_and_closed(self):
        code = ("from effect_modifier_experiment.survival_mlp8 import discovery_protocol as p; "
                "import sys; p.validate_parameters(p.PILOT_TASK, {'scenario':'iid'}); "
                "p.validate_parameters(p.FORMAL_TASK, {'scenario':'covariate_shift','shard_index':7,'shard_count':8}); "
                "assert 'numpy' not in sys.modules and 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT_ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-mlp8-v2'
        for phase, fields in (('pilot', {'scenario'}), ('shard', {'scenario', 'shard_index', 'shard_count'})):
            schema = json.loads((directory / (phase + '-parameters.schema.json')).read_text())
            self.assertIs(schema['additionalProperties'], False)
            self.assertEqual(set(schema['required']), fields)
            self.assertEqual(set(schema['properties']), fields)
            self.assertEqual(schema['properties']['scenario']['enum'], list(recipe.SCENARIOS))
        proposal = (directory / 'registration-proposal.toml').read_text()
        self.assertIn('enabled = false', proposal)
        for task in recipe.TASKS:
            self.assertIn('name = "' + task + '"', proposal)
        self.assertNotIn('name = "' + prediction.PILOT_TASK + '"', proposal)
        for bad in ({'scenario': 'combined'}, {'scenario': 'iid', 'repeats': 5}):
            with self.assertRaises(ValueError):
                recipe.validate_parameters(recipe.PILOT_TASK, bad)
        for index, count in ((8, 8), (True, 8), (0, 32)):
            with self.assertRaises(ValueError):
                recipe.validate_parameters(recipe.FORMAL_TASK,
                    dict(scenario='iid', shard_index=index, shard_count=count))


class DiscoveryPublicationTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the V1 scratch/source-binding setup; no V1 tests inherit.
        v1_fixture.MLP8PublicationTests.setUp(self)
        self.stack.enter_context(mock.patch.object(gate, 'PROOF_ROOT', self.root / 'proofs'))
        self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=v1_fixture.provenance()))

    def test_exact_v2_coverage_preserves_every_v1_data_and_training_setting(self):
        for cohort, shards, expected_requests, expected_units in (
                ('pilot', None, 2, 4), ('formal', 16, 32, 600), ('formal', 8, 16, 600)):
            old = builder.build_requests(cohort=cohort, shard_count=shards)
            new = builder.build_requests(cohort=cohort, shard_count=shards, discovery=True)
            self.assertEqual(len(new), expected_requests)
            self.assertEqual(sum(len(s['seed_sequence']) for s in new), expected_units)
            self.assertTrue({s['filename'] for s in old}.isdisjoint(s['filename'] for s in new))
            coverage = {}
            for before, after in zip(old, new):
                self.assertEqual(before['seed_sequence'], after['seed_sequence'])
                self.assertEqual(before['request']['parameters'], after['request']['parameters'])
                self.assertEqual(set(after['request']), {'schema_version', 'project', 'task', 'job_id', 'parameters', 'priority'})
                self.assertEqual(after['request']['task'], recipe.PILOT_TASK if cohort == 'pilot' else recipe.FORMAL_TASK)
                restored = deepcopy(after['configuration'])
                self.assertEqual(restored.pop('protocol'), recipe.PROTOCOL)
                self.assertEqual(restored.pop('task'), after['request']['task'])
                self.assertEqual(restored.pop('run_id'), after['run_id'])
                restored['protocol'] = before['configuration']['protocol']
                restored['task'] = before['configuration']['task']
                restored['run_id'] = before['configuration']['run_id']
                discovery = restored['evaluation'].pop('discovery')
                self.assertEqual(discovery['repeats'], 5)
                self.assertEqual(discovery['top2'], 'probability_all_signals_selected')
                self.assertIs(discovery['auroc'], True)
                self.assertIs(restored['evaluation']['modifier_permutation_performed'], True)
                restored['evaluation']['modifier_permutation_performed'] = False
                self.assertEqual(restored['timing']['stages'].pop(), 'discovery_evaluation')
                self.assertEqual(restored, before['configuration'])
                coverage.setdefault(after['request']['parameters']['scenario'], []).extend(after['seed_sequence'])
            self.assertEqual(set(coverage), set(recipe.SCENARIOS))
            for seeds in coverage.values():
                self.assertEqual(sorted(seeds), [0, 49] if cohort == 'pilot' else list(range(300)))

    def test_review_bindings_cover_new_and_original_discovery_sources(self):
        old, new = builder.review_sources(), builder.review_sources(discovery=True)
        for name in ('discovery_protocol', 'discovery_evaluation', 'survival_discovery'):
            self.assertNotIn(name, old)
            self.assertIn(name, new)
            self.assertEqual(len(new[name]['sha256']), 64)
        self.assertTrue(new['proposal']['path'].endswith('/survival-mlp8-v2/registration-proposal.toml'))
        self.assertTrue(old['proposal']['path'].endswith('/survival-mlp8-v1/registration-proposal.toml'))

    def test_v2_pilot_inventory_publishes_last_and_cannot_overwrite(self):
        writes, original = [], builder.atomic_create
        def write(path, data):
            writes.append(path)
            return original(path, data)
        with mock.patch.object(builder, 'atomic_create', side_effect=write):
            receipt = builder.write_requests(discovery=True)
        self.assertEqual(len(writes), 3)
        self.assertEqual(writes[-1], Path(receipt['inventory_path']))
        self.assertEqual(receipt['bundle'], 'fedfisher-survival-mlp8-v2-pilot')
        self.assertEqual(receipt['task'], recipe.PILOT_TASK)
        self.assertEqual(receipt['scientific_cell_count'], 4)
        self.assertEqual(receipt['fitted_model_count'], 16)
        self.assertIs(receipt['submitted'], False)
        self.assertIs(receipt['formal_plan']['publication_authorized'], False)
        self.assertNotIn('pilot_acceptance', receipt)
        data = Path(receipt['inventory_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            builder.write_requests(discovery=True)
        self.assertEqual(Path(receipt['inventory_path']).read_bytes(), data)
        # The additive bundle leaves the old recipe independently publishable.
        old = builder.write_requests()
        self.assertNotEqual(old['inventory_path'], receipt['inventory_path'])
        self.assertEqual(old['task'], prediction.PILOT_TASK)

    def test_v1_proof_cannot_open_v2_formal_gate_even_if_copied(self):
        source = v1_fixture.provenance()
        old_path = gate.default_acceptance_path(source)
        old_proof = dict(passed=True, provenance=source, scope='survival_mlp8_pilot_scientific_acceptance')
        old_path.parent.mkdir(parents=True)
        old_path.write_text(json.dumps(old_proof))
        with self.assertRaisesRegex(ValueError, 'canonical MLP8 proof'):
            builder.write_requests(cohort='formal', pilot_acceptance=old_path, discovery=True)
        new_path = gate.default_acceptance_path(source, discovery=True)
        new_path.write_bytes(old_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'invalid same-source MLP8 pilot proof'):
            builder.write_requests(cohort='formal', pilot_acceptance=new_path, discovery=True)
        self.assertFalse((self.root / 'outbox').exists())
        with self.assertRaisesRegex(ValueError, 'same-source MLP8 pilot acceptance'):
            builder.write_requests(cohort='formal', discovery=True)

    def test_v2_formal_publication_replays_its_current_pilot_proof(self):
        source = v1_fixture.provenance()
        path = gate.default_acceptance_path(source, discovery=True)
        proof = dict(passed=True, provenance=source,
                     scope='survival_mlp8_discovery_v2_pilot_scientific_acceptance')
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(proof))
        # This fixture replaces only the successful scientific proof replay;
        # separate acceptance tests exercise its full saved-evidence checks.
        with mock.patch.object(gate, 'verify_pilot_acceptance', return_value=proof) as replay:
            receipt = builder.write_requests(cohort='formal', pilot_acceptance=path, discovery=True)
        self.assertEqual(replay.call_count, 3)
        for call in replay.call_args_list:
            self.assertEqual(call, mock.call(path, discovery=True))
        self.assertEqual(receipt['task'], recipe.FORMAL_TASK)
        self.assertEqual(receipt['request_count'], 32)
        self.assertEqual(receipt['pilot_acceptance']['evidence'], proof)
        self.assertEqual(receipt['pilot_acceptance']['path'], str(path))

    def test_v2_proof_filename_scope_and_publication_are_immutable(self):
        source = v1_fixture.provenance()
        proof = dict(passed=True, provenance=source,
                     scope='survival_mlp8_discovery_v2_pilot_scientific_acceptance')
        path = gate.publish_acceptance(proof, discovery=True)
        self.assertEqual(path, gate.default_acceptance_path(source, discovery=True))
        self.assertNotEqual(path, gate.default_acceptance_path(source))
        self.assertIn('pilot-discovery-v2-', path.name)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'existing or partial'):
            gate.publish_acceptance(dict(proof, extra='changed'), discovery=True)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'scope/source mismatch'):
            gate.publish_acceptance(proof)
        path.unlink()  # Only this disposable fixture is removed.
        pending = path.with_name('.' + path.name + '.pending')
        pending.write_bytes(b'preserve partial V2 fixture')
        with self.assertRaisesRegex(ValueError, 'existing or partial'):
            gate.publish_acceptance(proof, discovery=True)
        self.assertFalse(path.exists())
        self.assertEqual(pending.read_bytes(), b'preserve partial V2 fixture')


if __name__ == '__main__':
    unittest.main()
