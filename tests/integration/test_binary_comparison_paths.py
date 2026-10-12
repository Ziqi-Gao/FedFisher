"""Saved-summary CLI paths remain usable without repository-local artifacts."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from reporting.binary import comparison
from tests.paths import PROJECT_ROOT


class BinaryComparisonPathsTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/tests')
        self.assertEqual(scratch.resolve(), scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_table(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def archive(self):
        root = self.root / 'archive'
        paths = {
            'original': root / 'synthetic_binary_experiment/outputs/original_fedfisher_1000seeds/figures/original_compare_summary.csv',
            'alpha': root / 'synthetic_binary_experiment/outputs/original_fedfisher_alpha_sweep_1000seeds/figures/alpha_sweep_summary.csv',
            'model': root / 'results/compare' / comparison.MODEL_SUMMARY,
            'primary': root / 'results/compare' / comparison.PRIMARY_SUMMARY,
            'feature': root / 'results/compare' / comparison.FEATURE_SUMMARY,
        }
        for name, path in paths.items():
            self.write_table(path, [{'fixture': name, 'value': '1.25'}])
        return root, paths

    def fail_cli(self, args):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            comparison.main(args)
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_archived_tables_and_explicit_paths_route_to_selected_output(self):
        archive, paths = self.archive()
        explicit = self.root / 'custom-original.csv'
        self.write_table(explicit, [{'fixture': 'explicit-original', 'value': '2.5'}])
        output = self.root / 'presentation'
        renderers = {
            'plot_original_gain': 'explicit-original', 'plot_alpha_sweep': 'alpha',
            'plot_prediction_accuracy': 'model', 'plot_prediction_signal_recovery': 'primary',
            'plot_feature_importance_heatmap': 'feature',
        }
        with ExitStack() as stack:
            calls = {name: stack.enter_context(mock.patch.object(comparison, name)) for name in renderers}
            stack.enter_context(mock.patch.object(comparison, 'write_synthetic_readme'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(comparison.main(['--archive-root', str(archive), '--original-summary',
                str(explicit), '--output-dir', str(output)]), 0)
        for name, expected in renderers.items():
            rows, destination = calls[name].call_args.args
            self.assertEqual(rows[0]['fixture'], expected)
            self.assertEqual(destination, output)
        for name in ('model', 'primary', 'feature'):
            self.assertEqual((output / paths[name].name).read_bytes(), paths[name].read_bytes())

    def test_missing_archive_input_fails_before_output_creation(self):
        archive, paths = self.archive()
        paths['feature'].unlink()
        output = self.root / 'missing-output'
        error = self.fail_cli(['--archive-root', str(archive), '--output-dir', str(output)])
        self.assertIn(str(paths['feature']), error)
        self.assertIn('--help', error)
        self.assertFalse(output.exists())

    def test_explicit_missing_raw_inputs_do_not_fall_back_to_archive(self):
        archive, _ = self.archive()
        output = self.root / 'missing-output'
        error = self.fail_cli(['--archive-root', str(archive), '--prediction-dir',
            str(self.root / 'absent-raw'), '--output-dir', str(output)])
        self.assertIn('No saved input CSVs match', error)
        self.assertFalse(output.exists())

    def test_source_archive_and_input_overwrite_destinations_are_rejected(self):
        archive, _ = self.archive()
        for destination in (PROJECT_ROOT / 'results/compare', comparison.ARCHIVE_ROOT / 'new-report',
                            archive / 'results/compare'):
            with self.subTest(destination=destination):
                self.fail_cli(['--archive-root', str(archive), '--output-dir', str(destination)])
        alias = self.root / 'source-link'
        alias.symlink_to(PROJECT_ROOT, target_is_directory=True)
        error = self.fail_cli(['--archive-root', str(archive), '--output-dir', str(alias / 'results/compare')])
        self.assertIn('Report output must remain below', error)

    def test_raw_prediction_aggregation_keeps_original_mean_and_count(self):
        raw = self.root / 'raw'
        base = dict(model_source='fedavg', model='SyntheticMLP', split='iid',
                    predicted_positive_rate='0.5', mean_class1_probability='0.4', mean_abs_logit_score='1.5')
        for seed, accuracy in enumerate(('0.6', '0.8')):
            self.write_table(raw / ('seed%d_prediction_intervention_model_summary.csv' % seed),
                             [dict(base, baseline_accuracy=accuracy)])
        output = self.root / 'tables'
        output.mkdir()
        rows = comparison.aggregate_prediction_model(raw, output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['n'], 2)
        self.assertAlmostEqual(rows[0]['baseline_accuracy_mean'], 0.7)
        self.assertTrue((output / comparison.MODEL_SUMMARY).is_file())

    def test_wrapper_help_works_without_repository_outputs(self):
        completed = subprocess.run([sys.executable, '-B', str(PROJECT_ROOT /
            'scripts/make_synthetic_compare_visualizations.py'), '--help'], cwd=self.root,
            check=True, capture_output=True, text=True)
        for option in ('--archive-root', '--original-summary', '--alpha-summary', '--summary-dir',
                       '--prediction-dir', '--feature-importance-dir', '--output-dir'):
            self.assertIn(option, completed.stdout)


if __name__ == '__main__':
    unittest.main()
