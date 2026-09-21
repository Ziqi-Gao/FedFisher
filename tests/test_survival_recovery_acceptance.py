"""Gate and outbox fixtures; no scientific run or central-state writes."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.survival import recovery, recovery_acceptance as gate
from scripts import prepare_survival_recovery_requests as builder


def provenance():
    return dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                source_tree_sha256='c' * 64, provenance_sha256='d' * 64)


def observations():
    return {(case, recipe): dict(outcome='iteration_limit' if recipe == 'baseline'
                and context['role'] == 'observed_failure' else 'stationary',
                loss_gradient_calls=100, baseline_numeric_parity=True)
            for case, context in recovery.CASES.items() for recipe in recovery.RECIPES}


class NumericalGateTests(unittest.TestCase):
    def test_all_cases_and_fixed_tie_order(self):
        rows = observations()
        accepted = gate.select_recipe(rows)
        self.assertTrue(accepted['formal_eligible'])
        self.assertEqual(accepted['selected_recipe'], 'budget_only')
        for (case, recipe), row in rows.items():
            if recipe == 'step_only':
                row['loss_gradient_calls'] = 90
            # Scientific recovery measures cannot influence recipe selection.
            row['auroc'] = 1. if recipe == 'budget_only' else 0.
        self.assertEqual(gate.select_recipe(rows)['selected_recipe'], 'step_only')

    def test_missing_case_and_invalid_cost_rejected(self):
        rows = observations()
        rows.pop(next(iter(rows)))
        with self.assertRaisesRegex(ValueError, '108'):
            gate.select_recipe(rows)
        rows = observations()
        rows[next(iter(recovery.CASES)), 'budget_only']['loss_gradient_calls'] = True
        with self.assertRaisesRegex(ValueError, 'cost'):
            gate.select_recipe(rows)

    def test_failed_reproduction_or_no_qualifying_candidate_blocks_formal(self):
        rows = observations()
        case = next(key for key, row in recovery.CASES.items() if row['role'] == 'observed_failure')
        rows[case, 'baseline']['outcome'] = 'stationary'
        proof = gate.select_recipe(rows)
        self.assertFalse(proof['formal_eligible'])
        self.assertIsNone(proof['selected_recipe'])
        rows = observations()
        for recipe in gate.SELECTION_ORDER:
            rows[case, recipe]['outcome'] = 'iteration_limit'
        self.assertFalse(gate.select_recipe(rows)['formal_eligible'])
        rows = observations()
        case = next(key for key, row in recovery.CASES.items() if row['role'] == 'original_pilot_control')
        rows[case, 'baseline']['baseline_numeric_parity'] = False
        self.assertFalse(gate.select_recipe(rows)['formal_eligible'])

    def test_baseline_observers_may_add_fields_but_not_change_old_numeric_values(self):
        old = dict(fingerprints={'model': 'same'}, fit=dict(cv_losses=[1., 2.], selected=0,
                   timing_seconds={'full_fit': 1.}, communication={'events': 4}))
        new = deepcopy(old)
        new['fit']['timing_seconds']['full_fit'] = 2.
        new['fit']['new_observer'] = {'endpoint': [0.]}
        new['fit']['communication']['loss_gradient_calls'] = 10
        self.assertTrue(gate.baseline_parity(new, old))
        new['fit']['cv_losses'][0] += .001
        self.assertFalse(gate.baseline_parity(new, old))

    def test_mixed_version_pairing_allows_model_change_but_rejects_science_change(self):
        config = recovery.run_config(recovery.DIAGNOSTIC_TASK,
                                    dict(case=next(iter(recovery.CASES)), recipe='baseline'))
        result = dict(fingerprints=dict(training='a', evaluation='b', evaluation_truth='c', model='new'),
            site_sizes=[500], pseudo_diagnostics=[{'pv_mean': .4}],
            evaluation=dict(raw_x_sha256='d', contrast_truth_sha256='e', permutation_sha256='f'))
        reference = deepcopy(result)
        reference['fingerprints']['model'] = 'old'
        gate.validate_pair(result, reference, config, config, formal=True)
        for change in ('data', 'pseudo', 'permutation', 'config'):
            bad, recipe = deepcopy(result), deepcopy(config)
            if change == 'data': bad['fingerprints']['training'] = 'different'
            if change == 'pseudo': bad['pseudo_diagnostics'][0]['pv_mean'] = .5
            if change == 'permutation': bad['evaluation']['permutation_sha256'] = 'different'
            if change == 'config': recipe['mean_link']['inverse'] = 'different'
            with self.subTest(change=change), self.assertRaises(ValueError):
                gate.validate_pair(bad, reference, recipe, config, formal=True)

    def test_proof_boolean_cannot_bypass_replay_and_ineligible_proof_is_blocked(self):
        proof = dict(provenance=provenance(), diagnostic_evidence_complete=True,
                     inventory={'path': 'fixture'}, formal_eligible=True, selected_recipe='step_only')
        path = gate.default_acceptance_path(provenance())
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
            stack.enter_context(mock.patch.object(gate.old, '_ordinary_path', side_effect=lambda p, **k: Path(p)))
            stack.enter_context(mock.patch.object(gate.old, '_binding', return_value={'hash': 'fixture'}))
            stack.enter_context(mock.patch.object(gate.old, '_read_json', return_value=proof))
            collect = stack.enter_context(mock.patch.object(gate, 'collect_diagnostic_acceptance', return_value=proof))
            self.assertEqual(gate.verify_diagnostic_acceptance(path), proof)
            collect.return_value = dict(proof, formal_eligible=False, selected_recipe=None)
            with self.assertRaisesRegex(ValueError, 'differs'):
                gate.verify_diagnostic_acceptance(path)
        with mock.patch.object(gate, 'default_acceptance_path', return_value=path), \
             mock.patch.object(gate, 'verify_diagnostic_acceptance', return_value=collect.return_value):
            with self.assertRaisesRegex(ValueError, 'does not qualify'):
                gate.check_current_diagnostic_gate()

    def test_formal_collect_requires_all_600_new_units_and_rechecks_old_archive(self):
        specs = builder.build_requests(cohort='formal', recipe='step_only')
        diagnostic = dict(selected_recipe='step_only', provenance=provenance())
        proof_path = Path('/scr/del6500/FedFisher/survival-recovery-validation/fabricated-proof.json')
        binding = lambda path: dict(path=str(path), sha256='e' * 64)
        inventory = dict(review_sources={}, pilot_acceptance=dict(path=str(proof_path), sha256='e'*64,
                          evidence=diagnostic), requests=[dict(item['request'], configuration=item['configuration'],
                          seeds=item['seed_sequence']) for item in specs])
        configs = {item['configuration']['scenario']: item['configuration'] for item in specs}
        original = dict(runs=[dict(config=dict(config, cell='fedfisher_diag_shallow'))
                             for config in configs.values()])
        saved = dict(fingerprints=dict(training='t', evaluation='e', evaluation_truth='v'),
                     site_sizes=[500], pseudo_diagnostics=[], timing={}, memory={},
                     fit=dict(loss_gradient_calls=1, timing_seconds={}, communication={}),
                     evaluation=dict(raw_x_sha256='x', contrast_truth_sha256='d', permutation_sha256='p'))
        owner = self

        class Store:
            model_name, result_name, done_name = 'model.npz', 'result.json', 'done.json'

            def __init__(self, config, source, prerequisite):
                self.config = config
                owner.assertEqual(prerequisite['selected_recipe'], 'step_only')
                owner.assertEqual(prerequisite['diagnostic_acceptance_sha256'], gate.digest(diagnostic))

            def cell_dir(self, seed):
                return proof_path.parent / self.config['run_id'] / ('seed_%03d' % seed)

            def status(self, seed):
                return 'validated'

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
            stack.enter_context(mock.patch.object(gate, 'check_original_closeout', return_value=original))
            stack.enter_context(mock.patch.object(gate, 'check_current_diagnostic_gate', return_value=diagnostic))
            stack.enter_context(mock.patch.object(gate, 'default_acceptance_path', return_value=proof_path))
            stack.enter_context(mock.patch.object(gate, '_inventory', return_value=(inventory, binding(proof_path), [])))
            stack.enter_context(mock.patch.object(gate, '_completed_job', return_value=({}, dict(job=binding(proof_path), logs={}))))
            stack.enter_context(mock.patch.object(gate.old, '_binding', side_effect=binding))
            stack.enter_context(mock.patch.object(gate.old, '_read_json', side_effect=lambda path:
                {'execution': {}} if Path(path).name == 'started.json' else deepcopy(saved)))
            stack.enter_context(mock.patch.object(gate.old, '_producer_evidence', return_value={}))
            stack.enter_context(mock.patch.object(gate.old, '_recheck_producer'))
            stack.enter_context(mock.patch.object(gate, '_checked', return_value=deepcopy(saved)))
            stack.enter_context(mock.patch.object(gate, '_old_unit', side_effect=lambda proof, cell, scenario, seed:
                {'artifacts': {'result.json': binding(proof_path.parent / ('old-%s-%s' % (scenario, seed)))}}))
            stack.enter_context(mock.patch.object(recovery, 'RecoveryStore', Store))
            archive = stack.enter_context(mock.patch.object(gate, '_recheck_original_archive'))
            proof = gate.collect_formal_acceptance(proof_path)
            self.assertEqual(proof['validated_unit_count'], 600)
            self.assertEqual(proof['paired_original_cases'], 600)
            self.assertFalse(proof['original_dslasso_successes_used_as_formal_results'])
            archive.assert_called_once_with(original)
            inventory['requests'][-1]['seeds'] = []
            with self.assertRaisesRegex(ValueError, 'coverage'):
                gate.collect_formal_acceptance(proof_path)


class RequestGateTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/survival-recovery-validation/requests')
        gate.old._ordinary_path(scratch, exists=False)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        for module, key, value in (
            (builder, 'OUTBOX_ROOT', self.root / 'outbox'), (builder, 'REPORT_ROOT', self.root / 'reports'),
            (gate.old, 'OUTBOX_ROOT', self.root / 'outbox'), (gate.old, 'INVENTORY_ROOT', self.root / 'reports')):
            self.stack.enter_context(mock.patch.object(module, key, value))
        self.stack.enter_context(mock.patch.object(builder, 'current_provenance', return_value=provenance()))
        self.stack.enter_context(mock.patch.object(gate, 'check_original_closeout', return_value={}))

    def test_exact_diagnostic_and_formal_coverage(self):
        diagnostic = builder.build_requests()
        self.assertEqual(len(diagnostic), 108)
        self.assertEqual(len({row['request']['job_id'] for row in diagnostic}), 108)
        for row in diagnostic:
            self.assertEqual(set(row['request']), {'schema_version', 'project', 'task', 'job_id', 'parameters', 'priority'})
            self.assertEqual(set(row['request']['parameters']), {'case', 'recipe'})
        formal = builder.build_requests(cohort='formal', recipe='step_only')
        self.assertEqual(len(formal), 32)
        for scenario in ('iid', 'covariate_shift'):
            rows = [row for row in formal if row['request']['parameters']['scenario'] == scenario]
            self.assertEqual(sorted(seed for row in rows for seed in row['seed_sequence']), list(range(300)))
            self.assertEqual(len({row['configuration_sha256'] for row in rows}), 1)
            self.assertTrue(all(set(row['request']['parameters']) == {'scenario', 'shard_index', 'shard_count'} for row in rows))
        self.assertEqual(len(builder.build_requests(cohort='formal', recipe='step_only', shard_count=8)), 16)

    def test_invalid_overrides_and_missing_formal_gate_leave_no_output(self):
        for options in ({'generation': True}, {'shard_count': 8}, {'recipe': 'baseline'},
                        {'cohort': 'formal'}, {'cohort': 'formal', 'recipe': 'baseline'},
                        {'cohort': 'formal', 'recipe': 'step_only', 'shard_count': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                builder.build_requests(**options)
        with self.assertRaisesRegex(ValueError, 'acceptance first'):
            builder.write_requests(cohort='formal')
        self.assertFalse((self.root / 'outbox').exists())

    def test_immutable_publication_inventory_readback_and_tampered_request(self):
        result = builder.write_requests()
        path = Path(result['inventory_path'])
        self.assertFalse(result['submitted'])
        self.assertFalse(result['formal_publication_authorized'])
        checked, _, _ = gate._inventory(path, provenance(), 'diagnostic')
        self.assertEqual(checked, result)
        with self.assertRaises(FileExistsError):
            builder.write_requests()
        target = Path(result['requests'][0]['path'])
        target.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            gate._inventory(path, provenance(), 'diagnostic')

    def test_duplicate_coverage_and_source_mutation_rejected(self):
        result = builder.write_requests()
        result['requests'][-1] = result['requests'][0]
        Path(result['inventory_path']).write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            gate._inventory(result['inventory_path'], provenance(), 'diagnostic')
        with mock.patch.object(builder, 'current_provenance', side_effect=[provenance(), dict(provenance(), git_commit='e'*40)]):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                builder.write_requests(generation=2)
        self.assertFalse((self.root / 'reports' / (builder.bundle_name(2) + '.inventory.json')).exists())

    def test_formal_publication_rechecks_and_binds_selected_recipe(self):
        accepted = dict(path='fixture', sha256='e'*64, evidence=dict(provenance=provenance(), selected_recipe='step_only'))
        with mock.patch.object(builder, 'acceptance_binding', return_value=accepted) as check:
            result = builder.write_requests(cohort='formal', diagnostic_acceptance='fixture')
        self.assertEqual(check.call_count, 2)
        self.assertEqual(result['pilot_acceptance'], accepted)
        self.assertEqual(result['scientific_cell_count'], 600)
        self.assertEqual(result['selected_recipe'], 'step_only')


class DiagnosticEvidenceTests(unittest.TestCase):
    """Exercise actual file/producer checks; numerical store has separate tests."""
    def setUp(self):
        RequestGateTests.setUp(self)
        self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
        for name, relative in (('CENTRAL_JOBS', 'jobs'), ('CENTRAL_LOGS', 'logs'),
                               ('CENTRAL_RETRIES', 'retries'), ('CENTRAL_AUDIT', 'audit.jsonl')):
            self.stack.enter_context(mock.patch.object(gate.old, name, self.root / 'central' / relative))
        self.stack.enter_context(mock.patch.object(gate, 'PROOF_ROOT', self.root / 'proofs'))
        self.inventory = builder.write_requests()
        self.inventory_path = Path(self.inventory['inventory_path'])
        self.directories, old_units, old_runs, audit = {}, {}, {}, []

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))

        for request in self.inventory['requests']:
            case, recipe = (request['parameters'][name] for name in ('case', 'recipe'))
            context, config, seed = recovery.CASES[case], request['configuration'], request['seeds'][0]
            directory = self.root / 'outputs' / config['run_id'] / ('seed_%03d' % seed)
            directory.mkdir(parents=True)
            self.directories[case, recipe] = directory
            write(directory.parent / 'run_manifest.env', config)
            job_id = request['job_id']
            execution = dict(job_id=job_id, attempt=1, execution_profile='cpu-1core',
                allocation=dict(cpu_cores=1, memory_mib=8192, gpu_count=0, gpu_memory_mib=0,
                                gpu_utilization_pct=0, exclusive_gpu=False), estimated_runtime_seconds=20.)
            write(directory / 'started.json', dict(execution=execution))
            result = dict(identity={'scenario': context['scenario'], 'seed': seed},
                outcome=observations()[case, recipe]['outcome'],
                fingerprints=dict(training=case, evaluation=case, evaluation_truth=case, model='saved-model'),
                site_sizes=[500], pseudo_diagnostics=[{'mean': .5}],
                fit=dict(loss_gradient_calls=100, timing_seconds={'full_fit': 2.},
                         communication={'events': 10}, numeric_endpoint=[.25, .5]),
                timing=dict(stages={'fit': dict(wall_seconds=3., cpu_seconds=2.)}),
                memory=dict(process_peak_rss_kib=1000))
            write(directory / 'diagnostic_result.json', result)
            (directory / 'fit_arrays.npz').write_bytes(b'fabricated saved arrays; never load or infer')
            self.rehash(directory)
            key = context['scenario'], seed
            for cell in ('fedfisher_diag_shallow', 'dslasso'):
                if (cell, *key) not in old_units:
                    original_path = self.root / 'archive' / cell / case / 'result.json'
                    original_result = deepcopy(result)
                    original_result['fit'].pop('loss_gradient_calls')
                    write(original_path, original_result)
                    old_units[(cell, *key)] = dict(cell=cell, scenario=key[0], seed=seed,
                        identity=original_result['identity'], artifacts={'result.json': gate.old._binding(original_path)})
            old_runs[key[0]] = dict(config=dict(config, cell='fedfisher_diag_shallow'))
            body = json.loads(Path(request['path']).read_bytes())
            job = dict(body, state='completed', exit_code=0,
                       **{k: v for k, v in execution.items() if k not in ('job_id', 'attempt')})
            for kind in ('stdout', 'stderr'):
                path = gate.old.CENTRAL_LOGS / ('%s.attempt-001.%s.log' % (job_id, kind))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('')
                job[kind + '_log'] = str(path)
            write(gate.old.CENTRAL_JOBS / (job_id + '.json'), job)
            write(gate.old.CENTRAL_RETRIES / (job_id + '.json'), dict(schema_version=2, job_id=job_id,
                attempts_started=1, max_attempts=3, next_attempt_at=None,
                last_failure_reason=None, updated_at='fixture'))
            audit.extend([dict(schema_version=2, job_id=job_id, event='job_attempt_started', attempt=1),
                dict(schema_version=2, job_id=job_id, event='job_transition', project='FedFisher',
                     source='pending', destination='running')])
        gate.old.CENTRAL_AUDIT.write_text(''.join(json.dumps(row) + '\n' for row in audit))
        original = dict(validated_units=list(old_units.values()), runs=list(old_runs.values()))
        self.stack.enter_context(mock.patch.object(gate, 'check_original_closeout', return_value=original))
        owner = self

        class Store:
            result_name, model_name, done_name = 'diagnostic_result.json', 'fit_arrays.npz', 'diagnostic_done.json'

            def __init__(self, config, source, prerequisite=None):
                owner.assertEqual(source, provenance())
                owner.assertIsNone(prerequisite)
                self.config = config

            def cell_dir(self, seed):
                return owner.directories['%s_%03d' % (self.config['scenario'], seed), self.config['recovery_recipe']]

            def status(self, seed):
                directory = self.cell_dir(seed)
                done = json.loads((directory / self.done_name).read_text())
                if any(gate.old._binding(directory / name)['sha256'] != value for name, value in done.items()):
                    raise ValueError('fabricated scientific artifact checksum mismatch')
                return 'validated'

        self.stack.enter_context(mock.patch.object(recovery, 'RecoveryStore', Store))

    @staticmethod
    def rehash(directory):
        files = ('started.json', 'fit_arrays.npz', 'diagnostic_result.json')
        (directory / 'diagnostic_done.json').write_text(json.dumps(
            {name: gate.old._binding(directory / name)['sha256'] for name in files}))

    def test_full_108_observation_gate_and_exact_proof_replay(self):
        proof = gate.collect_diagnostic_acceptance(self.inventory_path)
        self.assertEqual(proof['validated_unit_count'], 108)
        self.assertEqual(proof['paired_original_cases'], 27)
        self.assertTrue(proof['diagnostic_evidence_complete'])
        self.assertTrue(proof['formal_eligible'])
        self.assertEqual(proof['selected_recipe'], 'budget_only')
        path = gate.default_acceptance_path(provenance())
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(proof))
        self.assertEqual(gate.check_current_diagnostic_gate(), proof)

    def test_rehashed_data_mismatch_cannot_inherit_completion(self):
        directory = self.directories[next(iter(recovery.CASES)), 'baseline']
        path = directory / 'diagnostic_result.json'
        result = json.loads(path.read_text())
        result['fingerprints']['training'] = 'different'
        path.write_text(json.dumps(result))
        self.rehash(directory)
        with self.assertRaisesRegex(ValueError, 'fingerprints differ'):
            gate.collect_diagnostic_acceptance(self.inventory_path)

    def test_central_failure_and_wrong_producer_attempt_block_acceptance(self):
        request = self.inventory['requests'][0]
        path = gate.old.CENTRAL_JOBS / (request['job_id'] + '.json')
        body = json.loads(path.read_text())
        body['state'] = 'failed'
        path.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, 'completed central'):
            gate.collect_diagnostic_acceptance(self.inventory_path)
        body['state'] = 'completed'
        path.write_text(json.dumps(body))
        directory = self.directories[next(iter(recovery.CASES)), 'baseline']
        started = json.loads((directory / 'started.json').read_text())
        started['execution']['attempt'] = 2
        (directory / 'started.json').write_text(json.dumps(started))
        self.rehash(directory)
        with self.assertRaisesRegex(ValueError, 'producer attempt is later'):
            gate.collect_diagnostic_acceptance(self.inventory_path)


if __name__ == '__main__':
    unittest.main()
