"""Presentation fixtures use fabricated small JSON, never scientific replay."""
from contextlib import ExitStack
import csv
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


def fisher_record(recipe='fisher_shallow', dataset='scmv2_base', gate=True, endpoint=True):
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
    diagnostic = dict(implementation_gate=dict(passed=gate, stable_checks=6, stable_failures=0 if gate else 1),
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
        tables = REPORT.tables([fisher_record(gate=False), fisher_record(recipe='fisher_deep', endpoint=False)])
        self.assertEqual([row['diagnostic_state'] for row in tables['status']],
                         ['gradient_check_failed_or_inconclusive', 'full_H_endpoint_mismatch'])
        self.assertEqual(tables['model_quality'], [])
        self.assertEqual(tables['outcome_prediction'], [])
        self.assertEqual(len(tables['fisher_prediction_errors']), 48)

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
        self.assertFalse((output/'done.json').exists())
        self.assertFalse((output/'run_manifest.env').exists())
        with self.assertRaises(FileExistsError):
            REPORT.write_report(records, {}, output)

    def test_empty_diagnostics_produce_status_without_uncomputed_figures(self):
        output = self.temporary/'failed-gate'
        summary = REPORT.write_report([fisher_record(gate=False)], {}, output)
        self.assertEqual(set(summary['files']), {'status.csv'})

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


if __name__ == '__main__':
    unittest.main()
