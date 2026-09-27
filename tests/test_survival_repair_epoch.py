"""Repair-epoch coverage, historical parity and actual-evidence publication gates."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest import mock

from effect_modifier_experiment.survival import recovery, recovery_acceptance as gate
from effect_modifier_experiment.survival import recovery_protocol as protocol
from scripts import prepare_survival_recovery_requests as builder
import test_survival_recovery_acceptance as legacy


class RepairProtocolTests(unittest.TestCase):
    def test_fixed_repair_cases_namespaces_and_evidence(self):
        self.assertEqual(set(protocol.REPAIR_CASES) - set(protocol.CASES),
                         {'covariate_shift_049', 'covariate_shift_063', 'covariate_shift_240'})
        self.assertEqual(len(protocol.REPAIR_CASES), 30)
        self.assertEqual(protocol.tasks_for_epoch(1), (protocol.DIAGNOSTIC_TASK, protocol.FORMAL_TASK))
        specs = builder.build_requests(epoch=2)
        self.assertEqual(len(specs), 30)
        self.assertEqual(len({row['request']['job_id'] for row in specs}), 30)
        for row in specs:
            request, config = row['request'], row['configuration']
            self.assertEqual(set(request['parameters']), {'case'})
            self.assertNotIn('resources', request)
            self.assertNotIn('execution_profile', request)
            self.assertEqual(config['recovery_recipe'], protocol.REPAIR_RECIPE)
            self.assertEqual(config['required_evidence_schema'], protocol.REPAIR_EVIDENCE_SCHEMA)
            self.assertEqual(config['repair_closeout'], protocol.REPAIR_CLOSEOUT)
            self.assertEqual(config['r1_diagnostic_acceptance'], protocol.R1_DIAGNOSTIC_ACCEPTANCE)
            self.assertIn('_r2_', config['run_id'])
            self.assertEqual(protocol.run_config(config['task'], protocol.config_parameters(config)), config)

    def test_formal_full_coverage_has_no_partial_old_source_substitution(self):
        for count in (8, 16):
            specs = builder.build_requests(epoch=2, cohort='formal', shard_count=count,
                                           recipe=protocol.REPAIR_RECIPE)
            self.assertEqual(len(specs), 2 * count)
            for scenario in protocol.SCENARIOS:
                rows = [row for row in specs if row['configuration']['scenario'] == scenario]
                self.assertEqual(sorted(seed for row in rows for seed in row['seed_sequence']), list(range(300)))
                self.assertEqual(len({row['configuration_sha256'] for row in rows}), 1)
                config = rows[0]['configuration']
                self.assertEqual(protocol.run_config(config['task'], protocol.config_parameters(config),
                                                    recipe=protocol.REPAIR_RECIPE), config)
        for recipe in ('baseline', 'budget_only', 'step_only', None):
            with self.assertRaises(ValueError):
                builder.build_requests(epoch=2, cohort='formal', recipe=recipe)

    def test_unreviewed_cases_recipe_overrides_and_epochs_rejected(self):
        for parameters in ({'case': 'iid_001'}, {'case': 'iid_000', 'recipe': 'budget_and_step'},
                           {'case': 'iid_000', 'seed': 0}):
            with self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.REPAIR_DIAGNOSTIC_TASK, parameters)
        for epoch in (True, 0, 3, '2'):
            with self.assertRaises(ValueError):
                builder.build_requests(epoch=epoch)
        old = protocol.run_config(protocol.DIAGNOSTIC_TASK, dict(case='iid_000', recipe='budget_and_step'))
        new = protocol.run_config(protocol.REPAIR_DIAGNOSTIC_TASK, dict(case='iid_000'))
        for field in ('dgp', 'pseudo', 'mean_link', 'evaluation', 'method', 'public_initial_eta'):
            self.assertEqual(new[field], old[field])


class RepairNumericalGateTests(unittest.TestCase):
    @staticmethod
    def observations():
        return {case: dict(outcome='stationary', r1_numeric_parity=True) for case in protocol.REPAIR_CASES}

    def test_all_30_stationary_and_27_parity_are_independent_requirements(self):
        observations = self.observations()
        accepted = gate.select_repair(observations)
        self.assertTrue(accepted['formal_eligible'])
        self.assertEqual(accepted['selected_recipe'], protocol.REPAIR_RECIPE)
        self.assertEqual(len(accepted['r1_numeric_parity_checks']), 27)
        observations.pop('covariate_shift_049')
        with self.assertRaisesRegex(ValueError, '30'):
            gate.select_repair(observations)
        for case, field, value in (('covariate_shift_049', 'outcome', 'iteration_limit'),
                                   ('iid_000', 'r1_numeric_parity', False)):
            observations = self.observations()
            observations[case][field] = value
            observations[case]['auroc'] = 1.
            self.assertFalse(gate.select_repair(observations)['formal_eligible'])

    def test_parity_excludes_only_timings_schema_and_added_observer_fields(self):
        old = dict(fingerprints={'model': 'same'}, fit=dict(recovery_schema='old',
            cv_loss=[.1, .2], timing_seconds={'full_fit': 1.}, timing_cpu_seconds={'full_fit': 1.},
            diagnostic_fingerprint_seconds=.1, paths=[dict(fits=[dict(terminal_beta=[1., 2.],
                terminal_gradient=[.01, .02], iterations=7, majorization_max_gap=1e-15,
                elapsed_wall_seconds=1., elapsed_cpu_seconds=.8)])]))
        new = deepcopy(old)
        new['fit'].update(recovery_schema='new', timing_seconds={'full_fit': 2.},
            timing_cpu_seconds={'full_fit': 2.}, diagnostic_fingerprint_seconds=.2)
        report = new['fit']['paths'][0]['fits'][0]
        report.update(elapsed_wall_seconds=2., elapsed_cpu_seconds=1.8, new_witness={'margin': 0.})
        self.assertTrue(gate.repair_numeric_parity(new, old))
        report['terminal_gradient'][0] += 1e-15
        self.assertFalse(gate.repair_numeric_parity(new, old))
        new = deepcopy(old)
        new['fit']['paths'][0]['fits'][0]['majorization_max_gap'] += 1e-15
        self.assertFalse(gate.repair_numeric_parity(new, old))
        new = deepcopy(old)
        new['fingerprints']['model'] = 'different'
        self.assertFalse(gate.repair_numeric_parity(new, old))

    def test_historical_closeout_requires_complete_source_release_evidence(self):
        closeout = dict(scope='survival_dslasso_recovery_r1_failed_cohort_closeout',
            source_release_gate_passed=True, whole_cohort_passed=False,
            counts=dict(validated=565, failed=3, never_started=32),
            diagnostic_acceptance=protocol.R1_DIAGNOSTIC_ACCEPTANCE,
            original_closeout=protocol.ORIGINAL_CLOSEOUT,
            diagnostic_prerequisite_fully_replayed=True, original_archive_fully_rechecked=True,
            provenance=legacy.provenance(), review={'path': 'review', 'sha256': 'a' * 64})
        diagnostic = dict(provenance=legacy.provenance(), diagnostic_evidence_complete=True,
            formal_eligible=True, validated_unit_count=108, selected_recipe=protocol.REPAIR_RECIPE)
        def checked(binding):
            if binding == protocol.REPAIR_CLOSEOUT:
                return closeout
            if binding == protocol.R1_DIAGNOSTIC_ACCEPTANCE:
                return diagnostic
            if binding == closeout['review']:
                return {'saved': 'review'}
            self.fail('unexpected historical evidence')
        with mock.patch.object(gate, '_checked', side_effect=checked) as check:
            self.assertEqual(gate.check_repair_closeout(), closeout)
            self.assertEqual(check.call_count, 3)
            closeout['counts']['validated'] = 564
            with self.assertRaisesRegex(ValueError, 'source-release'):
                gate.check_repair_closeout()
            closeout['counts']['validated'] = 565
            diagnostic['selected_recipe'] = 'step_only'
            with self.assertRaisesRegex(ValueError, 'source-release'):
                gate.check_repair_closeout()
class RepairRequestTests(unittest.TestCase):
    def setUp(self):
        legacy.RequestGateTests.setUp(self)
        self.stack.enter_context(mock.patch.object(gate, 'check_repair_closeout', return_value={}))
        self.stack.enter_context(mock.patch.object(builder, 'review_sources', return_value={}))

    def test_pilot_publication_pins_epoch_and_cannot_be_read_as_r1(self):
        inventory = builder.write_requests(epoch=2)
        self.assertEqual(inventory['request_count'], 30)
        self.assertEqual(inventory['scientific_cell_count'], 30)
        self.assertEqual(inventory['repair_closeout'], protocol.REPAIR_CLOSEOUT)
        self.assertFalse(inventory['formal_publication_authorized'])
        checked, _, _ = gate._inventory(inventory['inventory_path'], legacy.provenance(), 'diagnostic', epoch=2)
        self.assertEqual(checked, inventory)
        with self.assertRaises(ValueError):
            gate._inventory(inventory['inventory_path'], legacy.provenance(), 'diagnostic')
        with self.assertRaises(FileExistsError):
            builder.write_requests(epoch=2)

    def test_formal_cannot_publish_without_actual_same_source_r2_proof(self):
        with self.assertRaisesRegex(ValueError, 'acceptance first'):
            builder.write_requests(epoch=2, cohort='formal')
        old = gate.default_acceptance_path(legacy.provenance(), epoch=1)
        with self.assertRaisesRegex(ValueError, 'acceptance first'):
            builder.write_requests(epoch=2, cohort='formal', diagnostic_acceptance=old)
        self.assertFalse((self.root / 'outbox').exists())
        proof = dict(provenance=legacy.provenance(), diagnostic_evidence_complete=True,
                     inventory={'path': 'fixture'}, formal_eligible=True, selected_recipe='budget_and_step')
        with mock.patch.object(gate, 'current_provenance', return_value=legacy.provenance()), \
             mock.patch.object(gate.old, '_ordinary_path', side_effect=lambda p, **k: Path(p)), \
             mock.patch.object(gate.old, '_binding', return_value={}), \
             mock.patch.object(gate.old, '_read_json', return_value=proof), \
             mock.patch.object(gate, 'collect_diagnostic_acceptance') as collect:
            with self.assertRaisesRegex(ValueError, 'earlier diagnostic epoch'):
                gate.verify_diagnostic_acceptance(gate.default_acceptance_path(legacy.provenance(), epoch=2), epoch=2)
            collect.assert_not_called()


class RepairCollectionTests(unittest.TestCase):
    """Exercise the shared collector's full new coverage without fitting or inference."""
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        source, self.root = legacy.provenance(), Path('/scr/del6500/FedFisher/survival-repair-validation/fixture')
        binding = lambda p: dict(path=str(p), sha256='e' * 64)
        self.specs = builder.build_requests(epoch=2)
        self.inventory = dict(review_sources={}, requests=[dict(row['request'], configuration=row['configuration'],
            seeds=row['seed_sequence']) for row in self.specs])
        self.saved, self.references, configs = {}, {}, {}
        for spec in self.specs:
            config, case = spec['configuration'], spec['request']['parameters']['case']
            directory = self.root / config['run_id'] / ('seed_%03d' % spec['seed_sequence'][0])
            saved = dict(outcome='stationary', fingerprints=dict(training=case, evaluation=case,
                evaluation_truth=case, model='same'), site_sizes=[500], pseudo_diagnostics=[], timing={}, memory={},
                fit=dict(loss_gradient_calls=1, timing_seconds={}, communication={}, paths=[dict(beta=[.25])]))
            self.saved[str(directory / 'diagnostic_result.json')] = saved
            self.references[case] = deepcopy(saved)
            configs[config['scenario']] = dict(config, cell='fedfisher_diag_shallow')
        original = dict(runs=[dict(config=config) for config in configs.values()])
        for module, name, value in (
            (gate, 'current_provenance', source), (gate, 'check_original_closeout', original),
            (gate, 'check_repair_closeout', {}),
            (gate, '_inventory', (self.inventory, binding(self.root / 'inventory.json'), [])),
            (gate, '_completed_job', ({}, dict(job=binding(self.root / 'job.json'), logs={})))):
            self.stack.enter_context(mock.patch.object(module, name, return_value=value))
        self.stack.enter_context(mock.patch.object(gate.old, '_binding', side_effect=binding))
        self.stack.enter_context(mock.patch.object(gate.old, '_read_json', side_effect=lambda p:
            {'execution': {}} if Path(p).name == 'started.json' else deepcopy(self.saved[str(p)])))
        self.stack.enter_context(mock.patch.object(gate.old, '_producer_evidence', return_value={}))
        self.stack.enter_context(mock.patch.object(gate.old, '_recheck_producer'))
        self.stack.enter_context(mock.patch.object(gate, '_old_unit', side_effect=lambda proof, cell, scenario, seed:
            dict(artifacts={'result.json': binding(self.root / ('%s_%03d' % (scenario, seed)))})))
        self.stack.enter_context(mock.patch.object(gate, '_checked', side_effect=lambda b:
            {} if b == protocol.R1_DIAGNOSTIC_ACCEPTANCE else deepcopy(self.references[Path(b['path']).name])))
        self.stack.enter_context(mock.patch.object(gate, '_repair_reference', side_effect=lambda proof, case:
            (deepcopy(self.references[case]), dict(manifest=binding(self.root / 'r1.env'), artifacts=[],
                                                 producer_evidence={}))))
        owner = self

        class Store:
            model_name, result_name, done_name = 'fit_arrays.npz', 'diagnostic_result.json', 'diagnostic_done.json'

            def __init__(self, config, provenance, prerequisite=None):
                owner.assertIsNone(prerequisite)
                owner.assertEqual(provenance, source)
                self.config = config

            def cell_dir(self, seed):
                return owner.root / self.config['run_id'] / ('seed_%03d' % seed)

            def status(self, seed):
                return 'validated'

        self.stack.enter_context(mock.patch.object(recovery, 'RecoveryStore', Store))

    def test_all_new_cases_and_reference_parity_enter_source_bound_proof(self):
        proof = gate.collect_diagnostic_acceptance(self.root / 'inventory.json', epoch=2)
        self.assertEqual(proof['validated_unit_count'], 30)
        self.assertEqual(proof['paired_original_cases'], 30)
        self.assertEqual(len(proof['r1_diagnostic_reference_units']), 27)
        self.assertTrue(proof['formal_eligible'])
        self.assertEqual(proof['selected_recipe'], protocol.REPAIR_RECIPE)
        self.assertEqual(proof['epoch'], 2)
        next(iter(self.saved.values()))['fit']['paths'][0]['beta'][0] += .01
        self.assertFalse(gate.collect_diagnostic_acceptance(self.root / 'inventory.json', epoch=2)['formal_eligible'])

    def test_missing_repair_case_and_nonstationary_new_case_do_not_pass(self):
        spec = next(row for row in self.specs if row['request']['parameters']['case'] == 'covariate_shift_049')
        config, seed = spec['configuration'], spec['seed_sequence'][0]
        path = self.root / config['run_id'] / ('seed_%03d' % seed) / 'diagnostic_result.json'
        self.saved[str(path)]['outcome'] = 'iteration_limit'
        self.assertFalse(gate.collect_diagnostic_acceptance(self.root / 'inventory.json', epoch=2)['formal_eligible'])
        self.inventory['requests'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            gate.collect_diagnostic_acceptance(self.root / 'inventory.json', epoch=2)


if __name__ == '__main__':
    unittest.main()
