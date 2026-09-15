"""Presentation fixtures use fabricated small JSON, never scientific replay."""
from contextlib import ExitStack
import csv
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('report_effects_replay', ROOT / 'scripts/report_effects_replay.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def fisher_record(recipe='fisher_shallow', dataset='scmv2_base', gate=True, endpoint=True, status=None):
    points = []
    for t in (0., .1, 1.):
        moments = dict(rmse=.2*t, centered_rmse=.1*t, mean_error=.05*t,
                       mean_squared_error=.04*t*t, exact_mean=.1, approximate_mean=.1+.05*t)
        errors = {method: moments for method in ('control_logit', 'treated_logit', 'direct_tau',
                                                  'sigmoid_of_linear_logits')}
        agreement = dict(maximum_absolute_error=.1*t, mean_absolute_error=.05*t,
                         spearman_rank_correlation=None if t == 0 else .8,
                         top2_set_agreement_probability=1.-.1*t,
                         per_variable_error=[0., .1*t])
        points.append(dict(t=t, prediction_errors=errors, original_rows_prediction_errors=errors,
                           score_agreement={method: agreement for method in
                                            ('direct_tau', 'sigmoid_of_linear_logits')},
                           relu_flip_from_local={'relu0': dict(control=.1*t, treated=.2*t)}))
    outcome = dict(rows=2, cross_entropy=.6, brier_score=.2)
    truth = dataset == 'scmv2_base'
    metric = dict(cate_rmse=.15 if truth else None, predicted_mean=.1,
                  true_cate_available=truth, estimated_heterogeneity=.02)
    quality = dict(test={name: dict(metric) for name in ('constant_zero_CATE', 'local_site_000',
                   'target_weighted_prediction_average', 'parameter_FedAvg', 'strict_FedFisher')},
        local_outcome_metrics=[dict(site_id=0, T=outcome, V=outcome, H=outcome)],
        saved_model_outcome_metrics={'parameter_FedAvg': dict(test=outcome,
                                  sites=[dict(site_id=0, T=outcome, V=outcome, H=outcome)])})
    status = status or ('passed' if gate else 'failed')
    coverage = {name: dict(required=True, status='complete', missing_outputs=[],
        stable_checks_by_output=dict(control_binary_log_odds=6, treated_binary_log_odds=6,
                                     probability_CATE=6)) for name in ('actual_aggregation', 'fixed_random')}
    if status == 'inconclusive':
        for direction in coverage.values():
            direction.update(status='inconclusive', missing_outputs=['treated_binary_log_odds', 'probability_CATE'])
            direction['stable_checks_by_output'].update(treated_binary_log_odds=0, probability_CATE=0)
    pilot = dict(function='fisher_summary.tau_and_directional_derivatives', passed=status != 'failed',
        tau_passed=True, derivative_passed=status != 'failed', stable_fd_checks=12 if status != 'inconclusive' else 0,
        stable_fd_failures=1 if status == 'failed' else 0, tau_absolute_error_vs_new=[0., 0.],
        tau_absolute_error_vs_model=[0., 0.], derivative_absolute_error_vs_new=[[.2 if status == 'failed' else 0., 0.]])
    diagnostic = dict(implementation_gate=dict(passed=gate, status=status, stable_checks=36,
            stable_failures=1 if status == 'failed' else 0, coverage_complete=status != 'inconclusive',
            original_pilot_passed=pilot['passed'], original_pilot_stable_fd_failures=pilot['stable_fd_failures']),
        gradient_checks=[dict(site_id=0, audit_version=2, status=status, coverage=coverage, original_pilot=pilot)],
        parameter_path=dict(target_weighted_points=points, sites=[dict(site_id=0, points=points)]) if gate else None,
        full_H_endpoint=dict(replay_matches_source=endpoint,
            linearization=dict(gate=dict(relative_tau_rmse=.4, relative_score_linf=.5,
                                        compression_eligible=False))) if gate else None,
        model_quality=quality if gate and endpoint else None)
    return dict(recipe=recipe, dataset=dataset, source_path='/scr/del6500/FedFisher/fabricated-result.json',
                source_sha256='a'*64, result=dict(diagnostic=diagnostic))


def dr_record():
    path = []
    for index, value in enumerate((.1, .01, .001)):
        groups = [dict(feature_index=j, feature_label='X'+str(j+1), true_modifier=j==0,
                       coefficients=[float(index+j)], coefficient_norm=float(index+j),
                       nonzero=index+j>0, H_importance=.01*(index+j)) for j in range(2)]
        path.append(dict(index=index, lambda_value=value, validation_risk=.5+.01*abs(index-1),
            validation_risk_gap_to_minimum=.01*abs(index-1), nonzero_groups=index,
            selected_by_original_V_rule=index==1, feature_groups=groups,
            H_ranking=dict(rank_min=[1, 2], rank_max=[1, 2], average_rank=[1., 2.]),
            test=dict(cate_rmse=.1+.01*index, diagnostic_only=True)))
    detail = dict(path=path, original_selected_index=1, original_selected_lambda=.01)
    return dict(recipe='dr_path', dataset='scmv2_base', source_path='/scr/del6500/FedFisher/fabricated-dr.json',
                source_sha256='b'*64, result=dict(diagnostic=dict(basis_results=dict(linear=detail, spline=detail))))


class ReplayReportingTests(unittest.TestCase):
    def setUp(self):
        scratch = REPORT.safe_path(Path('/scr/del6500/FedFisher/effects-replay-validation/report-tests'))
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temporary = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.stack.enter_context(mock.patch.dict(os.environ))
        REPORT.configure_runtime()

    def test_tables_preserve_model_names_moments_and_original_DR_selection(self):
        tables = REPORT.tables([fisher_record(), dr_record()])
        self.assertEqual(len(tables['fisher_prediction_errors']), 48)
        self.assertEqual(len(tables['fisher_activation_flips']), 12)
        self.assertEqual(len(tables['outcome_prediction']), 7)
        names = {row['model'] for row in tables['model_quality']}
        self.assertIn('target_weighted_prediction_average', names)
        self.assertIn('parameter_FedAvg', names)
        self.assertEqual(len(tables['dr_path']), 6)
        self.assertEqual(len(tables['dr_feature_path']), 12)
        selected = [row for row in tables['dr_path'] if row['selected_by_original_V_rule']]
        self.assertEqual([row['index'] for row in selected], [1, 1])
        self.assertTrue(all(row['test_diagnostic_only'] for row in tables['dr_path']))
        feature = tables['dr_feature_path'][0]
        self.assertEqual(feature['coefficients'], [0.])
        self.assertEqual(feature['H_rank_min'], 1)

    def test_failure_status_omits_uncomputed_quality_without_relabeling_as_success(self):
        tables = REPORT.tables([fisher_record(gate=False), fisher_record(gate=False, status='inconclusive'),
                               fisher_record(recipe='fisher_deep', endpoint=False)])
        self.assertEqual([row['diagnostic_state'] for row in tables['status']],
                         ['gradient_check_failed', 'gradient_check_inconclusive', 'full_H_endpoint_mismatch'])
        self.assertEqual([row['implementation_gate_status'] for row in tables['status']],
                         ['failed', 'inconclusive', 'passed'])
        self.assertEqual(tables['model_quality'], [])
        self.assertEqual(tables['outcome_prediction'], [])
        self.assertEqual(len(tables['fisher_prediction_errors']), 48)

    def test_gradient_tables_expose_missing_CATE_coverage_and_original_gradient_failure(self):
        tables = REPORT.tables([fisher_record(gate=False, status='inconclusive'),
                                fisher_record(recipe='fisher_deep', gate=False)])
        missing = [row for row in tables['fisher_gradient_coverage'] if row['missing_output']]
        self.assertEqual(len(missing), 4)
        self.assertEqual({row['direction'] for row in missing}, {'actual_aggregation', 'fixed_random'})
        self.assertTrue(all(row['stable_checks'] == 0 and row['required'] for row in missing))
        self.assertEqual({row['output'] for row in missing}, {'treated_binary_log_odds', 'probability_CATE'})
        self.assertFalse(tables['status'][0]['coverage_complete'])
        original = tables['fisher_original_pilot_audit'][1]
        self.assertFalse(original['passed'])
        self.assertFalse(original['derivative_passed'])
        self.assertEqual(original['stable_fd_failures'], 1)
        self.assertEqual(original['maximum_derivative_absolute_error_vs_new'], .2)

    def test_zero_actual_direction_keeps_not_applicable_coverage_label(self):
        record = fisher_record()
        coverage = record['result']['diagnostic']['gradient_checks'][0]['coverage']['actual_aggregation']
        coverage.update(required=False, status='not_applicable_zero_direction', missing_outputs=[])
        coverage['stable_checks_by_output'] = dict.fromkeys(coverage['stable_checks_by_output'], 0)
        rows = [row for row in REPORT.tables([record])['fisher_gradient_coverage']
                if row['direction'] == 'actual_aggregation']
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row['coverage_status'] == 'not_applicable_zero_direction'
                            and not row['required'] and not row['missing_output']
                            and row['stable_checks'] == 0 for row in rows))

    def test_fabricated_results_render_figures_csv_and_hashed_presentation_only_index(self):
        records = [fisher_record(), fisher_record(recipe='fisher_deep', dataset='legacy_control_iid'), dr_record()]
        output = self.temporary/'figures'
        summary = REPORT.write_report(records, {'fixture': True}, output)
        self.assertEqual(summary['kind'], 'presentation_only_not_scientific_completion')
        self.assertFalse(summary['selection_changed'])
        self.assertFalse(summary['expansion_authorized'])
        self.assertEqual(json.loads((output/'report.json').read_text()), summary)
        for name, digest in summary['files'].items():
            self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
        figures = [name for name in summary['files'] if name.endswith('.png')]
        self.assertEqual(len(figures), 4)
        self.assertNotIn('fisher_deep_legacy_control_iid_cate_quality.png', figures)
        self.assertTrue(all((output/name).read_bytes().startswith(b'\x89PNG') for name in figures))
        with (output/'dr_feature_path.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(json.loads(rows[0]['coefficients']), [0.])
        self.assertEqual(rows[0]['H_rank_min'], '1')
        with (output/'fisher_gradient_coverage.csv').open() as stream:
            coverage = list(csv.DictReader(stream))
        self.assertEqual({row['output'] for row in coverage},
                         {'control_binary_log_odds', 'treated_binary_log_odds', 'probability_CATE'})
        self.assertIn('fisher_original_pilot_audit.csv', summary['files'])
        self.assertFalse((output/'done.json').exists())
        self.assertFalse((output/'run_manifest.env').exists())
        with self.assertRaises(FileExistsError):
            REPORT.write_report(records, {}, output)

    def test_empty_diagnostics_produce_status_without_uncomputed_figures(self):
        output = self.temporary/'failed-gate'
        summary = REPORT.write_report([fisher_record(gate=False)], {}, output)
        self.assertEqual(set(summary['files']),
                         {'status.csv', 'fisher_gradient_coverage.csv', 'fisher_original_pilot_audit.csv'})

    def test_symlink_and_outside_paths_fail_before_output(self):
        real = self.temporary/'real'
        real.mkdir()
        link = self.temporary/'link'
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            REPORT.write_report([], {}, link/'new')
        self.assertFalse((real/'new').exists())
        for path in (Path('relative'), Path('/home/del6500/projects/FedFisher/forbidden-report')):
            with self.assertRaisesRegex(ValueError, 'below'):
                REPORT.safe_path(path)

    def test_cache_paths_are_redirected_before_plot_import(self):
        for key in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TMPDIR'):
            self.assertTrue(Path(os.environ[key]).is_relative_to(REPORT.SCRATCH_ROOT))
        self.assertEqual(os.environ['MPLBACKEND'], 'Agg')

    def test_renderer_handoff_is_fixed_awaited_and_cache_confined(self):
        with mock.patch.object(REPORT.importlib.util, 'find_spec', return_value=None), mock.patch.object(
                REPORT.subprocess, 'run', return_value=SimpleNamespace(stdout='{"fixture":true}')) as child:
            result = REPORT.write_report([fisher_record()], {}, self.temporary/'handoff')
        self.assertEqual(result, {'fixture': True})
        arguments, options = child.call_args
        self.assertEqual(arguments[0][:3], ['/opt/anaconda3/bin/python', '-B', '-c'])
        self.assertIs(options['check'], True)
        self.assertEqual(options['cwd'], REPORT.PROJECT_ROOT)
        self.assertNotIn('shell', options)
        for key in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TMPDIR'):
            self.assertTrue(Path(options['env'][key]).is_relative_to(REPORT.SCRATCH_ROOT))
        self.assertEqual(json.loads(options['input'])['records'][0]['recipe'], 'fisher_shallow')
        self.assertFalse((self.temporary/'handoff').exists())

    def test_missing_matplotlib_in_fixed_renderer_cannot_recurse(self):
        with mock.patch.object(REPORT.importlib.util, 'find_spec', return_value=None), mock.patch.object(
                REPORT.sys, 'executable', str(REPORT.PRESENTATION_PYTHON)), mock.patch.object(
                REPORT.subprocess, 'run') as child:
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                REPORT.write_report([fisher_record()], {}, self.temporary/'unavailable')
        child.assert_not_called()
        self.assertFalse((self.temporary/'unavailable').exists())

    def test_loader_validates_exact_five_before_returning_and_never_initializes(self):
        from effect_modifier_experiment.one_shot_effects.diagnostics import protocol
        provenance = {'fixture': 'same-source'}
        stores, status = [], {'value': 'validated'}
        for params in protocol.cohort_plan():
            directory = self.temporary/protocol.run_id(protocol.PILOT_TASK, params)/'seed_000'
            directory.mkdir(parents=True)
            body = json.dumps({'fixture': params})
            (directory/'result.json').write_text(body)
            (directory/'done.json').write_text(json.dumps(dict(artifacts={
                'result.json': hashlib.sha256(body.encode()).hexdigest()})))

        def store_factory(config, received):
            self.assertEqual(received, provenance)
            store = SimpleNamespace(unit_dir=self.temporary/config['run_id']/'seed_000',
                                    status=mock.Mock(side_effect=lambda: status['value']))
            stores.append(store)
            return store

        original = ModuleType('effect_modifier_experiment.one_shot_effects.artifacts')
        original.current_provenance = mock.Mock(return_value=provenance)
        original.read_json = lambda path: json.loads(Path(path).read_text())
        replay = ModuleType('effect_modifier_experiment.one_shot_effects.diagnostics.artifacts')
        replay.ReplayArtifactStore = mock.Mock(side_effect=store_factory)
        with mock.patch.dict(sys.modules, {original.__name__: original, replay.__name__: replay}):
            records, actual = REPORT.load_completed()
            self.assertEqual(actual, provenance)
            self.assertEqual(len(records), 5)
            self.assertTrue(all(store.status.call_count == 1 for store in stores))
            status['value'] = 'missing'
            with self.assertRaisesRegex(ValueError, 'all five'):
                REPORT.load_completed()
            status['value'] = 'validated'
            path = stores[0].unit_dir/'result.json'
            path.write_text('modified')
            with self.assertRaisesRegex(ValueError, 'changed after validation'):
                REPORT.load_completed()

    def recovery_fixture(self):
        from effect_modifier_experiment.dslassocov_run_contract import env_text
        from effect_modifier_experiment.one_shot_effects.diagnostics import protocol
        from effect_modifier_experiment.one_shot_effects.wire import canonical
        old = dict(git_commit='a'*40, provenance_sha256='b'*64, git_dirty='0',
                   git_status_sha256='c'*64, source_tree_sha256='d'*64)
        current = dict(old, git_commit='e'*40, provenance_sha256='f'*64)
        inventory = dict(provenance=old, requests=[])
        retained = []
        for params in protocol.cohort_plan():
            config = protocol.run_config(protocol.PILOT_TASK, params)
            if params['recipe'] != 'fisher_deep':
                job_id = 'fedfisher-effects-replay-fixture-' + config['run_id']
                digest = hashlib.sha256(canonical(config).encode()).hexdigest()
                request = dict(job_id=job_id, configuration=config, configuration_sha256=digest)
                inventory['requests'].append(request)
                identity = dict(run_id=config['run_id'], config_sha256=digest,
                    provenance_sha256=old['provenance_sha256'], seed=0, **params)
                directory = self.temporary/config['run_id']
                unit = directory/'seed_000'
                unit.mkdir(parents=True)
                manifest = dict(format_version='2', protocol=protocol.PROTOCOL, run_id=config['run_id'],
                    config_json=canonical(config), config_sha256=digest, **old)
                (directory/'run_manifest.env').write_text(env_text(manifest.items()))
                (unit/'started.json').write_text(json.dumps(dict(identity=identity, execution=dict(job_id=job_id))))
                (unit/'result.json').write_text(json.dumps(dict(identity=identity, fixture=params)))
                done = dict(identity=identity, artifacts={name: hashlib.sha256((unit/name).read_bytes()).hexdigest()
                    for name in ('started.json', 'result.json')})
                (unit/'done.json').write_text(json.dumps(done))
                artifacts = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    for path in (directory/'run_manifest.env', unit/'started.json', unit/'result.json', unit/'done.json')]
                retained.append(dict(job_id=job_id, parameters=params, artifacts=artifacts))
            else:
                unit = self.temporary/config['run_id']/'seed_000'
                unit.mkdir(parents=True)
                body = json.dumps(dict(fixture=params))
                (unit/'result.json').write_text(body)
                (unit/'done.json').write_text(json.dumps(dict(artifacts={
                    'result.json': hashlib.sha256(body.encode()).hexdigest()})))
        inventory_path = self.temporary/'inventory.json'
        inventory_path.write_text(json.dumps(inventory))
        envelope = dict(source_inventory=dict(path=str(inventory_path),
            sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest()), retained=retained)
        evidence_path = self.temporary/'recovery-evidence.json'
        evidence_path.write_text(json.dumps(envelope))
        return current, inventory, envelope, evidence_path

    def test_recovery_loader_retains_three_frozen_semantic_checks_and_two_current_stores(self):
        current, inventory, envelope, evidence_path = self.recovery_fixture()
        stores = []

        def store_factory(config, received):
            self.assertEqual(received, current)
            self.assertEqual(config['recipe'], 'fisher_deep')
            store = SimpleNamespace(unit_dir=self.temporary/config['run_id']/'seed_000',
                                    status=mock.Mock(return_value='validated'))
            stores.append(store)
            return store

        with mock.patch('effect_modifier_experiment.one_shot_effects.artifacts.current_provenance', return_value=current), \
                mock.patch('effect_modifier_experiment.one_shot_effects.diagnostics.artifacts.ReplayArtifactStore',
                           side_effect=store_factory), \
                mock.patch('effect_modifier_experiment.one_shot_effects.diagnostics.recovery.verify_recovery_evidence',
                           return_value=envelope) as verify, \
                mock.patch('effect_modifier_experiment.one_shot_effects.diagnostics.validation.validate_result') as semantic:
            records, provenance = REPORT.load_completed(evidence_path)
        self.assertEqual(provenance, current)
        self.assertEqual(len(stores), 2)
        self.assertEqual(semantic.call_count, 3)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(len(records), 5)
        self.assertEqual([item['origin'] for item in records].count('retained_p2'), 3)
        self.assertEqual([item['origin'] for item in records].count('replayed_current'), 2)
        for result, config, identity in (call.args for call in semantic.call_args_list):
            self.assertEqual(result['identity'], identity)
            self.assertEqual(identity['provenance_sha256'], inventory['provenance']['provenance_sha256'])
            self.assertNotEqual(identity['provenance_sha256'], current['provenance_sha256'])
        self.assertTrue(all(item['recovery_evidence']['path'] == str(evidence_path) for item in records))

    def test_retained_semantic_failure_or_changed_hash_stops_reporting(self):
        _, inventory, envelope, _ = self.recovery_fixture()
        retained = envelope['retained'][0]
        with mock.patch('effect_modifier_experiment.one_shot_effects.diagnostics.validation.validate_result',
                        side_effect=ValueError('invalid saved CATE gate')) as semantic:
            with self.assertRaisesRegex(ValueError, 'invalid saved CATE'):
                REPORT._retained_record(retained, inventory)
            semantic.assert_called_once()
        result = next(item for item in retained['artifacts'] if Path(item['path']).name == 'result.json')
        Path(result['path']).write_text('{}')
        with mock.patch('effect_modifier_experiment.one_shot_effects.diagnostics.validation.validate_result') as semantic:
            with self.assertRaisesRegex(ValueError, 'changed after validation'):
                REPORT._retained_record(retained, inventory)
            semantic.assert_not_called()

    def test_retained_manifest_identity_and_complete_inventory_cannot_be_bypassed(self):
        _, inventory, envelope, _ = self.recovery_fixture()
        retained = envelope['retained'][0]
        extra = Path(retained['artifacts'][1]['path']).parent/'unexpected.json'
        extra.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'paths/inventory differ'):
            REPORT._retained_record(retained, inventory)
        extra.unlink()
        altered = copy.deepcopy(inventory)
        altered['provenance']['git_commit'] = '9'*40
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            REPORT._retained_record(retained, altered)
        marker = next(item for item in retained['artifacts'] if Path(item['path']).name == 'done.json')
        body = json.loads(Path(marker['path']).read_text())
        body['identity']['provenance_sha256'] = '0'*64
        Path(marker['path']).write_text(json.dumps(body))
        marker['sha256'] = hashlib.sha256(Path(marker['path']).read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'completion/execution identities differ'):
            REPORT._retained_record(retained, inventory)

    def test_mixed_epoch_report_preserves_execution_sources_and_origin_in_rows_and_index(self):
        records = [fisher_record(gate=False), fisher_record(recipe='fisher_deep', gate=False)]
        for record, origin, commit, digest in zip(records, ('retained_p2', 'replayed_current'),
                ('a'*40, 'b'*40), ('c'*64, 'd'*64)):
            record.update(origin=origin, execution_run_id=origin+'-run',
                execution_provenance=dict(git_commit=commit, provenance_sha256=digest),
                recovery_evidence=dict(path=str(self.temporary/'evidence.json'), sha256='e'*64))
        current = records[1]['execution_provenance']
        summary = REPORT.write_report(records, current, self.temporary/'mixed')
        self.assertTrue(summary['mixed_execution_epochs'])
        self.assertIsNone(summary['diagnostic_provenance'])
        self.assertEqual(summary['reporting_source_provenance'], current)
        self.assertEqual(len(summary['execution_provenances']), 2)
        for row, source, record in zip(summary['status'], summary['inputs'], records):
            self.assertEqual(row['origin'], record['origin'])
            self.assertEqual(row['execution_git_commit'], record['execution_provenance']['git_commit'])
            self.assertEqual(source['execution_provenance'], record['execution_provenance'])
            self.assertEqual(source['recovery_evidence'], record['recovery_evidence'])

    def test_recovery_option_selects_separate_default_output(self):
        with mock.patch.object(REPORT, 'load_completed', return_value=([], {})) as loader, \
                mock.patch.object(REPORT, 'write_report', return_value=dict(files={})) as writer, \
                mock.patch('builtins.print'):
            evidence = self.temporary/'evidence.json'
            self.assertEqual(REPORT.main(['--recovery-evidence', str(evidence)]), 0)
            loader.assert_called_once_with(evidence)
            self.assertEqual(writer.call_args.args[2], REPORT.RECOVERY_OUTPUT)
            loader.reset_mock()
            REPORT.main([])
            loader.assert_called_once_with(None)
            self.assertEqual(writer.call_args.args[2], REPORT.DEFAULT_OUTPUT)


if __name__ == '__main__':
    unittest.main()
