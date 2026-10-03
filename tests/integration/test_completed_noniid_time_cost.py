"""Reporting-only fixtures; no scientific fitting, inference or scheduler calls."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from reporting.effect_modifier import time_cost as plot


class TimeCostTests(unittest.TestCase):
    @staticmethod
    def fixture(cell='dslassocov'):
        aggregation = cell == 'local_logistic_lasso'
        stages = {s: dict(measured=False, calls=0, failed_calls=0,
                          wall_seconds=0, cpu_seconds=0, components={}) for s in plot.STAGES}
        for name, wall, cpu in (('data_preparation', 2, 1), ('train', 10, 8), ('evaluation', 3, 2)):
            stages[name].update(measured=True, calls=1, wall_seconds=wall, cpu_seconds=cpu)
        if aggregation:
            stages['aggregation'].update(measured=True, calls=1, wall_seconds=1, cpu_seconds=0.5)
        result = dict(identity=dict(cell=cell, variant='control', seed=0, split='noniid',
                                    run_id=f'comparison_{cell}_control_v1_300seeds'),
                      status='complete', fit=dict(completion_eligible=True,
                      timing_contract=dict(training_includes_all_selection_and_refits=True)),
                      timing=dict(schema_version=1, method=cell, execution_model='single_process_simulation',
                                  actual_network_measured=False, clock=dict(wall='perf_counter', cpu='process_time'),
                                  stages=stages, measured_stage_sum=dict(wall_seconds=15+int(aggregation),
                                                                       cpu_seconds=11+0.5*aggregation)))
        execution = dict(execution_profile='cpu-1core', job_id='fixture',
                         allocation=dict(cpu_cores=1, gpu_count=0))
        return result, execution

    @staticmethod
    def rows():
        return [dict(cell=c, variant=v, seed=s, split='noniid',
                     **{k: 10.0 + s % 2 for k in plot.METRICS})
                for c in plot.base.CELLS for v in plot.base.VARIANTS for s in range(300)]

    def test_fit_excludes_data_and_evaluation(self):
        row = plot.timing_row(*self.fixture())
        self.assertEqual(row['fit_wall_seconds'], 10)
        self.assertEqual(row['fit_cpu_seconds'], 8)
        self.assertEqual(row['evaluation_wall_seconds'], 3)
        self.assertEqual(row['data_preparation_wall_seconds'], 2)
        self.assertIn('included_in_fit', row['tuning_scope'])

    def test_separate_aggregation_is_added_once(self):
        row = plot.timing_row(*self.fixture('local_logistic_lasso'))
        self.assertEqual(row['fit_wall_seconds'], 11)
        self.assertEqual(row['fit_cpu_seconds'], 8.5)

    def test_nested_components_not_double_counted(self):
        result, execution = self.fixture()
        result['timing']['stages']['train']['components']['refit'] = dict(wall_seconds=5, cpu_seconds=4)
        self.assertEqual(plot.timing_row(result, execution)['fit_wall_seconds'], 10)

    def test_missing_measured_stage_is_not_zero_cost(self):
        result, execution = self.fixture()
        result['timing']['stages']['train'].update(measured=False, calls=0, wall_seconds=0, cpu_seconds=0)
        with self.assertRaisesRegex(ValueError, 'missing required'):
            plot.timing_row(result, execution)

    def test_bad_times_and_sums_rejected(self):
        for bad in (float('nan'), float('inf'), -1, True, None):
            with self.assertRaises(ValueError):
                plot.finite_time(bad)
        result, execution = self.fixture()
        result['timing']['measured_stage_sum']['wall_seconds'] = 20
        with self.assertRaisesRegex(ValueError, 'sum mismatch'):
            plot.timing_row(result, execution)

    def test_pilot_iid_gpu_and_incomplete_cost_rejected(self):
        result, execution = self.fixture()
        for field, value in (('split', 'iid'), ('run_id', 'pilot')):
            bad = copy.deepcopy(result)
            bad['identity'][field] = value
            with self.assertRaises(ValueError):
                plot.timing_row(bad, execution)
        bad_execution = copy.deepcopy(execution)
        bad_execution['allocation']['gpu_count'] = 1
        with self.assertRaises(ValueError):
            plot.timing_row(result, bad_execution)
        result['fit']['timing_contract']['training_includes_all_selection_and_refits'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete selection'):
            plot.timing_row(result, execution)

    def test_coverage_and_seed_standard_error(self):
        rows = self.rows()
        summaries = plot.summarize(rows)
        self.assertEqual(len(summaries), 88)
        self.assertEqual(summaries[0]['fit_wall_seconds_mean'], 10.5)
        self.assertAlmostEqual(summaries[0]['fit_wall_seconds_se'], 0.5/(299**0.5))
        for bad in (rows[:-1], rows[:-1]+[rows[-2]]):
            with self.assertRaisesRegex(ValueError, 'coverage'):
                plot.summarize(bad)

    def test_figures_have_correct_scope_and_valid_svg(self):
        summaries = plot.summarize(self.rows())
        documents = {}
        with patch.object(plot.base, 'save_svg', side_effect=lambda canvas, dest, name:
                          documents.update({name: canvas.document()})):
            plot.plot_overview(summaries, Path('/unused'))
            plot.plot_fedavg(summaries, Path('/unused'))
            plot.plot_lasso(summaries, Path('/unused'))
        self.assertEqual(set(documents), set(plot.FIGURES))
        for document in documents.values():
            ET.fromstring(document)
        self.assertIn('N/A', documents[plot.FIGURES[0]])
        self.assertIn('NOT a lambda-specific', documents[plot.FIGURES[2]])
        self.assertIn('Selection and final refit included', documents[plot.FIGURES[1]])


if __name__ == '__main__':
    unittest.main()
