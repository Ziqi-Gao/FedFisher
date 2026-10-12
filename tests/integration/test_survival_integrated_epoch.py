"""Integrated-source survival admission without fitting or central-state writes."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.survival import recovery, recovery_acceptance as gate
from effect_modifier_experiment.survival import recovery_protocol as protocol
from effect_modifier_experiment.scheduling.publication import survival_recovery as builder
from effect_modifier_experiment.survival import recovery_requests as request_plan
import tests.integration.test_survival_recovery_acceptance as legacy
import tests.integration.test_survival_repair_epoch as repair_fixture
SCRATCH = Path('/scr/del6500/FedFisher/survival-current-compat-validation')
EXAMPLE_ROOT = builder.PROJECT_ROOT / 'tests/fixtures/server_scheduler_requests/survival-integrated-v3'


class IntegratedProtocolTests(unittest.TestCase):
    def test_disabled_proposal_and_schemas_match_the_fixed_contract(self):
        root = builder.PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-integrated-v3'
        pilot = json.loads((root / 'diagnostic-parameters.schema.json').read_text())
        formal = json.loads((root / 'shard-parameters.schema.json').read_text())
        for schema, fields in ((pilot, {'case'}), (formal, {'scenario', 'shard_index', 'shard_count'})):
            self.assertIs(schema['additionalProperties'], False)
            self.assertEqual(set(schema['required']), fields)
            self.assertEqual(set(schema['properties']), fields)
        self.assertEqual(set(pilot['properties']['case']['enum']), set(protocol.REPAIR_CASES))
        self.assertEqual(set(formal['properties']['scenario']['enum']), set(protocol.SCENARIOS))
        self.assertEqual(formal['properties']['shard_count']['enum'], [8, 16])
        self.assertEqual(formal['properties']['shard_index']['type'], 'integer')
        self.assertEqual(formal['allOf'][0]['then']['properties']['shard_index']['maximum'], 7)
        proposal = (root / 'registration-proposal.toml').read_text()
        self.assertIn('\nenabled = false\n', proposal)
        self.assertIn('\nallow_gpu_jobs = false\n', proposal)
        for task in protocol.tasks_for_epoch(3):
            self.assertIn('name = "' + task + '"', proposal)
        examples = list(EXAMPLE_ROOT.glob('*.json'))
        self.assertEqual(len(examples), 2)
        for path in examples:
            request = json.loads(path.read_text())
            self.assertIn(request['task'], protocol.tasks_for_epoch(3))
            protocol.validate_parameters(request['task'], request['parameters'])
            self.assertNotIn('resources', request)
            self.assertNotIn('execution_profile', request)

    def test_new_pilot_preserves_all_thirty_cases_and_scientific_recipe(self):
        specs = builder.build_requests(epoch=3)
        self.assertEqual(len(specs), 30)
        self.assertEqual({row['request']['parameters']['case'] for row in specs},
                         set(protocol.REPAIR_CASES))
        self.assertEqual(len({row['request']['job_id'] for row in specs}), 30)
        old_specs = builder.build_requests(epoch=2)
        self.assertTrue({row['request']['job_id'] for row in specs}.isdisjoint(
                        row['request']['job_id'] for row in old_specs))
        for row in specs:
            request, config = row['request'], row['configuration']
            self.assertEqual(request['task'], protocol.INTEGRATED_DIAGNOSTIC_TASK)
            self.assertEqual(set(request['parameters']), {'case'})
            self.assertEqual(set(request), {'schema_version', 'project', 'task', 'job_id',
                                            'parameters', 'priority'})
            self.assertEqual(config['protocol'], protocol.INTEGRATED_PROTOCOL)
            self.assertEqual(config['required_evidence_schema'], protocol.REPAIR_EVIDENCE_SCHEMA)
            self.assertEqual(config['r2_diagnostic_acceptance'], protocol.R2_DIAGNOSTIC_ACCEPTANCE)
            self.assertIn('_r3_', config['run_id'])
            old = protocol.run_config(protocol.REPAIR_DIAGNOSTIC_TASK, request['parameters'])
            for field in ('dgp', 'pseudo', 'mean_link', 'evaluation', 'method',
                          'public_initial_eta', 'seeds', 'recovery_recipe'):
                self.assertEqual(config[field], old[field])
            self.assertEqual(protocol.run_config(config['task'], protocol.config_parameters(config)), config)

    def test_formal_shards_cover_six_hundred_without_reusing_old_results(self):
        for count in (8, 16):
            specs = builder.build_requests(epoch=3, cohort='formal', shard_count=count,
                                           recipe=protocol.REPAIR_RECIPE)
            self.assertEqual(len(specs), 2 * count)
            for scenario in protocol.SCENARIOS:
                rows = [row for row in specs if row['configuration']['scenario'] == scenario]
                self.assertEqual(sorted(seed for row in rows for seed in row['seed_sequence']),
                                 list(range(300)))
                self.assertEqual(len({row['configuration_sha256'] for row in rows}), 1)
                for row in rows:
                    self.assertEqual(row['request']['task'], protocol.INTEGRATED_FORMAL_TASK)
                    self.assertIn('_r3_', row['run_id'])
                    index = row['request']['parameters']['shard_index']
                    self.assertTrue(all(seed % count == index for seed in row['seed_sequence']))
        for recipe in ('baseline', 'budget_only', 'step_only', None):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                builder.build_requests(epoch=3, cohort='formal', recipe=recipe)

    def test_unreviewed_epoch_case_or_resource_and_type_overrides_fail(self):
        for epoch in (True, False, 0, 4, '3', 3.0):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                builder.build_requests(epoch=epoch)
        for parameters in ({'case': 'iid_001'}, {'case': True},
                           {'case': 'iid_000', 'recipe': 'budget_and_step'},
                           {'case': 'iid_000', 'seed': 0}, {'case': 'iid_000', 'cpu_cores': 1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.INTEGRATED_DIAGNOSTIC_TASK, parameters)
        for value in (True, False, 0.0, '0'):
            with self.subTest(index=value), self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.INTEGRATED_FORMAL_TASK,
                    dict(scenario='iid', shard_index=value, shard_count=16))


class IntegratedNumericalGateTests(unittest.TestCase):
    def test_all_thirty_stationary_models_require_r2_parity_without_retuning(self):
        observations = {case: dict(outcome='stationary', r2_numeric_parity=True)
                        for case in protocol.REPAIR_CASES}
        accepted = gate.select_integration(observations)
        self.assertTrue(accepted['formal_eligible'])
        self.assertEqual(accepted['selected_recipe'], protocol.REPAIR_RECIPE)
        self.assertEqual(set(accepted['r2_numeric_parity_checks']), set(protocol.REPAIR_CASES))
        for case in protocol.REPAIR_CASES:
            for field, value in (('outcome', 'iteration_limit'), ('r2_numeric_parity', False),
                                 ('r2_numeric_parity', 1)):
                rows = deepcopy(observations)
                rows[case].update({field: value, 'auroc': 1., 'loss_gradient_calls': 1})
                with self.subTest(case=case, field=field, value=value):
                    self.assertFalse(gate.select_integration(rows)['formal_eligible'])
                    self.assertIsNone(gate.select_integration(rows)['selected_recipe'])
        observations.pop('covariate_shift_240')
        with self.assertRaisesRegex(ValueError, '30'):
            gate.select_integration(observations)

    def test_model_schema_witness_and_path_changes_cannot_hide_behind_timing(self):
        reference = dict(fingerprints={'model': 'same'}, fit=dict(
            recovery_schema=protocol.REPAIR_EVIDENCE_SCHEMA, cv_loss=[.1, .2], selected_index=1,
            timing_seconds={'full_fit': 1.}, timing_cpu_seconds={'full_fit': .9},
            diagnostic_fingerprint_seconds=.1, paths=[dict(fits=[dict(
                iterations=7, terminal_gradient=[.01], elapsed_wall_seconds=1., elapsed_cpu_seconds=.8,
                majorization_evidence={'accepted_checks': 7, 'max_guard_witness': {
                    'candidate_smooth': .1, 'bound': .1, 'accepted_bound': .10000000000001,
                    'guard_excess': -1e-14}})])]))
        timing_only = deepcopy(reference)
        timing_only['fit'].update(timing_seconds={'full_fit': 9.}, timing_cpu_seconds={'full_fit': 8.},
                                  diagnostic_fingerprint_seconds=2.)
        timing_only['fit']['paths'][0]['fits'][0].update(elapsed_wall_seconds=8., elapsed_cpu_seconds=7.)
        self.assertTrue(gate.integrated_numeric_parity(timing_only, reference))
        for mutation in ('model', 'schema', 'witness', 'gradient', 'added', 'missing', 'path_length'):
            bad = deepcopy(timing_only)
            report = bad['fit']['paths'][0]['fits'][0]
            if mutation == 'model': bad['fingerprints']['model'] = 'changed'
            if mutation == 'schema': bad['fit']['recovery_schema'] = 'survival_dslasso_numerical_recovery_v1'
            if mutation == 'witness': report['majorization_evidence']['max_guard_witness']['bound'] += 1e-15
            if mutation == 'gradient': report['terminal_gradient'][0] += 1e-15
            if mutation == 'added': report['new_observer'] = 0
            if mutation == 'missing': del report['majorization_evidence']
            if mutation == 'path_length': bad['fit']['paths'][0]['fits'].append(deepcopy(report))
            with self.subTest(mutation=mutation):
                self.assertFalse(gate.integrated_numeric_parity(bad, reference))

    def test_archived_r2_anchor_retains_old_source_and_rejects_incomplete_or_changed_science(self):
        binding = lambda path: dict(path=str(path), sha256='e' * 64)
        specs = builder.build_requests(epoch=2)
        requests = [dict(row['request'], path=row['filename'], sha256='e' * 64,
                        configuration=row['configuration'], configuration_sha256=row['configuration_sha256'],
                        run_id=row['run_id'], seeds=row['seed_sequence']) for row in specs]
        scientific_root = gate.old.PROJECT_ROOT / 'effect_modifier_experiment/survival'
        inventory = dict(provenance=legacy.provenance(), task=protocol.REPAIR_DIAGNOSTIC_TASK,
            bundle_ready=True, epoch=2, request_count=30, scientific_cell_count=30, requests=requests,
            review_sources=dict(dslasso=binding(scientific_root / 'dslasso.py'),
                                method=binding(scientific_root / 'recovery_method.py')))
        proof = dict(schema_version=1, epoch=2, scope='survival_dslasso_repair_diagnostic_acceptance',
            provenance=legacy.provenance(), diagnostic_evidence_complete=True, formal_eligible=True,
            validated_request_count=30, validated_unit_count=30, paired_original_cases=30,
            selected_recipe=protocol.REPAIR_RECIPE, original_closeout=protocol.ORIGINAL_CLOSEOUT,
            repair_closeout=protocol.REPAIR_CLOSEOUT, r1_diagnostic_acceptance=protocol.R1_DIAGNOSTIC_ACCEPTANCE,
            stationary_checks=dict.fromkeys(protocol.REPAIR_CASES, True),
            r1_numeric_parity_checks=dict.fromkeys(protocol.CASES, True), inventory=binding('old-inventory'),
            units=[dict(parameters=row['parameters']) for row in requests])

        def checked(record):
            if record == protocol.R2_DIAGNOSTIC_ACCEPTANCE:
                return proof
            self.assertEqual(record, proof['inventory'])
            return inventory

        with mock.patch.object(gate, '_checked', side_effect=checked), \
             mock.patch.object(gate.old, '_binding', side_effect=binding), \
             mock.patch.object(gate, 'current_provenance', side_effect=AssertionError('must retain old source')):
            self.assertEqual(gate.check_integration_reference(), proof)
            proof['validated_unit_count'] = 29
            with self.assertRaisesRegex(ValueError, 'historical r2 pilot'):
                gate.check_integration_reference()
            proof['validated_unit_count'] = 30
            inventory['review_sources']['method']['sha256'] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'optimizer implementation'):
                gate.check_integration_reference()
            inventory['review_sources']['method'] = binding(scientific_root / 'recovery_method.py')
            inventory['requests'][0]['configuration']['required_evidence_schema'] = 'old'
            with self.assertRaisesRegex(ValueError, 'scientific configuration'):
                gate.check_integration_reference()


class IntegratedRequestTests(unittest.TestCase):
    def setUp(self):
        gate.old._ordinary_path(SCRATCH, exists=False)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=SCRATCH)))
        for module, name, value in (
            (builder, 'OUTBOX_ROOT', self.root / 'outbox'),
            (builder, 'REPORT_ROOT', self.root / 'reports'),
            (gate.old, 'OUTBOX_ROOT', self.root / 'outbox'),
            (gate.old, 'INVENTORY_ROOT', self.root / 'reports')):
            self.stack.enter_context(mock.patch.object(module, name, value))
        for module, name, value in (
            (builder, 'current_provenance', legacy.provenance()),
            (gate, 'check_original_closeout', {}), (gate, 'check_repair_closeout', {}),
            (gate, 'check_integration_reference', {}), (builder, 'review_sources', {}),
            (request_plan, 'review_sources', {})):
            self.stack.enter_context(mock.patch.object(module, name, return_value=value))

    def test_immutable_pilot_inventory_and_r2_r3_namespace_separation(self):
        inventory = builder.write_requests(epoch=3)
        self.assertEqual(inventory['request_count'], 30)
        self.assertEqual(inventory['scientific_cell_count'], 30)
        self.assertEqual(inventory['r2_diagnostic_acceptance'], protocol.R2_DIAGNOSTIC_ACCEPTANCE)
        self.assertFalse(inventory['submitted'])
        self.assertFalse(inventory['formal_publication_authorized'])
        checked, _, _ = gate._inventory(inventory['inventory_path'], legacy.provenance(), 'diagnostic', epoch=3)
        self.assertEqual(checked, inventory)
        with self.assertRaises(ValueError):
            gate._inventory(inventory['inventory_path'], legacy.provenance(), 'diagnostic', epoch=2)
        with self.assertRaises(FileExistsError):
            builder.write_requests(epoch=3)
        self.assertNotEqual(gate.default_acceptance_path(legacy.provenance(), epoch=2),
                            gate.default_acceptance_path(legacy.provenance(), epoch=3))
        target = Path(inventory['requests'][0]['path'])
        target.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            gate._inventory(inventory['inventory_path'], legacy.provenance(), 'diagnostic', epoch=3)

    def test_formal_publication_requires_actual_current_integrated_pilot(self):
        for proof_path in (None, protocol.R2_DIAGNOSTIC_ACCEPTANCE['path'],
                           gate.default_acceptance_path(legacy.provenance(), epoch=2)):
            with self.subTest(proof_path=proof_path), self.assertRaisesRegex(ValueError, 'acceptance first'):
                builder.write_requests(epoch=3, cohort='formal', diagnostic_acceptance=proof_path)
        self.assertFalse((self.root / 'outbox').exists())
        accepted = dict(path='fixture', sha256='e' * 64,
                        evidence=dict(provenance=legacy.provenance(), selected_recipe=protocol.REPAIR_RECIPE))
        with mock.patch.object(builder, 'acceptance_binding', return_value=accepted) as check:
            inventory = builder.write_requests(epoch=3, cohort='formal', diagnostic_acceptance='fixture')
        self.assertEqual(check.call_count, 2)
        self.assertTrue(all(call.args[-1] == 3 for call in check.call_args_list))
        self.assertEqual(inventory['pilot_acceptance'], accepted)
        self.assertEqual(inventory['scientific_cell_count'], 600)
        self.assertEqual(inventory['request_count'], 32)

    def test_relabelled_old_proof_and_changed_source_fail_before_evidence_replay(self):
        proof = dict(provenance=legacy.provenance(), diagnostic_evidence_complete=True,
                     inventory={'path': 'fixture'}, formal_eligible=True, selected_recipe=protocol.REPAIR_RECIPE,
                     epoch=2, scope='survival_dslasso_repair_diagnostic_acceptance')
        path = gate.default_acceptance_path(legacy.provenance(), epoch=3)
        with mock.patch.object(gate, 'current_provenance', return_value=legacy.provenance()), \
             mock.patch.object(gate.old, '_ordinary_path', side_effect=lambda p, **k: Path(p)), \
             mock.patch.object(gate.old, '_binding', return_value={}), \
             mock.patch.object(gate.old, '_read_json', return_value=proof), \
             mock.patch.object(gate, 'collect_diagnostic_acceptance') as collect:
            with self.assertRaises(ValueError):
                gate.verify_diagnostic_acceptance(path, epoch=3)
            collect.assert_not_called()
            proof.update(epoch=3, scope='survival_dslasso_integrated_diagnostic_acceptance')
            proof['provenance'] = dict(legacy.provenance(), git_commit='e' * 40)
            with self.assertRaisesRegex(ValueError, 'another source'):
                gate.verify_diagnostic_acceptance(path, epoch=3)
            collect.assert_not_called()

    def test_formal_runtime_cannot_reach_store_or_fitting_without_new_pilot(self):
        job = dict(task=protocol.INTEGRATED_FORMAL_TASK,
                   parameters=dict(scenario='iid', shard_index=0, shard_count=16))
        with mock.patch.object(recovery.artifacts, 'current_provenance', return_value=legacy.provenance()), \
             mock.patch.object(gate, 'check_current_diagnostic_gate', side_effect=ValueError('pilot absent')) as check, \
             mock.patch.object(recovery, 'RecoveryStore') as store, \
             mock.patch.object(recovery, 'compute_cell') as compute:
            with self.assertRaisesRegex(ValueError, 'pilot absent'):
                recovery.run_job(job)
            check.assert_called_once_with(epoch=3)
            store.assert_not_called()
            compute.assert_not_called()

    def test_pilot_rejects_changed_historical_reference_before_fitting(self):
        job = dict(task=protocol.INTEGRATED_DIAGNOSTIC_TASK,
                   parameters=dict(case='covariate_shift_049'))
        with mock.patch.object(recovery.artifacts, 'current_provenance', return_value=legacy.provenance()), \
             mock.patch.object(gate, '_integration_reference', side_effect=ValueError('archive changed')) as check, \
             mock.patch.object(recovery, 'RecoveryStore') as store, \
             mock.patch.object(recovery, 'compute_cell') as compute:
            with self.assertRaisesRegex(ValueError, 'archive changed'):
                recovery.run_job(job)
            check.assert_called_once_with({}, 'covariate_shift_049')
            store.assert_not_called()
            compute.assert_not_called()


class IntegratedCollectionTests(unittest.TestCase):
    """The existing collector fixture supplies saved records, never real models."""
    def setUp(self):
        original_builder = builder.build_requests
        with mock.patch.object(builder, 'build_requests',
                               side_effect=lambda **kw: original_builder(**dict(kw, epoch=3))):
            repair_fixture.RepairCollectionTests.setUp(self)
        self.stack.enter_context(mock.patch.object(gate, 'check_integration_reference', return_value={}))
        self.stack.enter_context(mock.patch.object(gate, '_integration_reference', side_effect=lambda proof, case:
            (deepcopy(self.references[case]), dict(manifest=dict(path='r2.env', sha256='e' * 64),
                artifacts=[], producer_evidence={}))))

    def test_complete_current_pilot_binds_all_thirty_archived_r2_references(self):
        proof = gate.collect_diagnostic_acceptance(self.root / 'inventory.json', epoch=3)
        self.assertEqual(proof['validated_unit_count'], 30)
        self.assertEqual(proof['paired_original_cases'], 30)
        self.assertEqual(len(proof['r2_diagnostic_reference_units']), 30)
        self.assertEqual(proof['epoch'], 3)
        self.assertEqual(proof['scope'], 'survival_dslasso_integrated_diagnostic_acceptance')
        self.assertEqual(proof['r2_diagnostic_acceptance'], protocol.R2_DIAGNOSTIC_ACCEPTANCE)
        self.assertTrue(proof['formal_eligible'])
        self.assertEqual(proof['selected_recipe'], protocol.REPAIR_RECIPE)
        self.assertFalse(proof['inference_or_fitting_performed'])
        # The three previously failed seeds are now parity controls too.
        case = 'covariate_shift_240'
        self.references[case]['fit']['paths'][0]['beta'][0] += .01
        changed = gate.collect_diagnostic_acceptance(self.root / 'inventory.json', epoch=3)
        self.assertFalse(changed['formal_eligible'])
        self.assertFalse(changed['r2_numeric_parity_checks'][case])

    def test_nonstationary_or_missing_current_case_prevents_formal_eligibility(self):
        path, saved = next(iter(self.saved.items()))
        saved['outcome'] = 'iteration_limit'
        self.assertFalse(gate.collect_diagnostic_acceptance(
            self.root / 'inventory.json', epoch=3)['formal_eligible'])
        saved['outcome'] = 'stationary'
        self.inventory['requests'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            gate.collect_diagnostic_acceptance(self.root / 'inventory.json', epoch=3)


if __name__ == '__main__':
    unittest.main()
