"""In-memory CPU-report regression checks; no model or filesystem execution."""

from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from reporting.effect_modifier import unified as plot


class UnifiedCpuTimingTests(unittest.TestCase):
    @staticmethod
    def fisher_rows():
        return [dict(
            cell=cell, variant=variant, seed=seed, split='noniid', metric=plot.METRIC,
            run_id=f'fedfisher_timing_{cell}_{variant}_v1_300seeds',
            top2_probability=float(seed % 2), expected_topk_precision=.75, auroc=.8,
            timing_status='measured', execution_profile='cpu-1core',
            **{metric: 10. + index + seed % 2
               for index, metric in enumerate(plot.timing.METRICS)})
            for cell in plot.FISHER_CELLS for variant in plot.base.VARIANTS
            for seed in range(300)]

    @staticmethod
    def existing_cpu_inputs():
        scores = [dict(cell=cell, variant=variant, split='noniid', n=300,
                       metric=plot.METRIC, top2_pct=60., top2_se_pct=2.,
                       auroc=.85, auroc_se=.02)
                  for cell in plot.base.CELLS for variant in plot.base.VARIANTS]
        times = [dict(cell=row['cell'], variant=row['variant'],
                      **{f'{metric}_{statistic}': 20. if statistic == 'mean' else .2
                         for metric in plot.timing.METRICS for statistic in ('mean', 'se')})
                 for row in scores]
        return scores, times

    def test_complete_cpu_recovery_coverage_and_seed_se(self):
        rows = self.fisher_rows()
        self.assertEqual(len(rows), 4800)
        summaries = plot.summarize_fisher(rows, cpu=True)
        self.assertEqual(len(summaries), 16)
        for summary in summaries:
            self.assertEqual(summary['n'], 300)
            self.assertEqual(summary['top2_pct'], 50.)
            self.assertAlmostEqual(summary['top2_se_pct'], 50. / (299 ** .5))

    def test_missing_duplicate_or_mixed_identity_rejected(self):
        rows = self.fisher_rows()
        bad_rows = {
            'missing seed': rows[:-1],
            'duplicate seed': rows[:-1] + [rows[-2]],
            'IID row': [{**rows[0], 'split': 'iid'}] + rows[1:],
            'old GPU run': [{**rows[0], 'run_id': plot.legacy.OLD_RUNS[rows[0]['variant']]}]
                           + rows[1:],
            'pilot run': [{**rows[0], 'run_id': rows[0]['run_id'].replace('300seeds', 'pilot')}]
                         + rows[1:],
        }
        for label, bad in bad_rows.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                plot.summarize_fisher(bad, cpu=True)

    def test_all_signal_probability_is_not_topk_precision(self):
        rows = [{**row, 'top2_probability': .5, 'expected_topk_precision': .75}
                for row in self.fisher_rows()]
        summaries = plot.summarize_fisher(rows, cpu=True)
        self.assertTrue(all(row['top2_pct'] == 50. for row in summaries))
        self.assertTrue(all(row['top2_se_pct'] == 0. for row in summaries))

    def test_all_six_timing_metrics_keep_seed_se(self):
        summaries = plot.summarize_fisher_times(self.fisher_rows())
        self.assertEqual(len(summaries), 16)
        for summary in summaries:
            for index, metric in enumerate(plot.timing.METRICS):
                self.assertEqual(summary[f'{metric}_mean'], 10.5 + index)
                self.assertAlmostEqual(summary[f'{metric}_se'], .5 / (299 ** .5))

    def test_merge_preserves_other_methods_and_adds_real_fisher_times(self):
        cpu, times = self.existing_cpu_inputs()
        rows = self.fisher_rows()
        fisher = plot.summarize_fisher(rows, cpu=True)
        legacy_rows = [{**row, 'run_id': plot.legacy.OLD_RUNS[row['variant']]} for row in rows]
        old = plot.merge_rows(cpu, times, plot.summarize_fisher(legacy_rows))
        merged = plot.merge_rows(cpu, times, fisher, plot.summarize_fisher_times(rows))
        self.assertEqual(len(merged), 104)
        self.assertEqual([row for row in old if row['cell'] not in plot.FISHER_CELLS],
                         [row for row in merged if row['cell'] not in plot.FISHER_CELLS])
        for row in merged:
            self.assertEqual(row['hardware'], 'CPU 1 core')
            self.assertEqual(row['timing_status'], 'measured')
            for metric in plot.timing.METRICS:
                self.assertGreater(row[f'{metric}_mean'], 0.)
            if row['cell'] in plot.FISHER_CELLS:
                self.assertEqual(row['run_id'],
                                 f"fedfisher_timing_{row['cell']}_{row['variant']}_v1_300seeds")

    def test_partial_fisher_timing_cannot_silently_mix_sources(self):
        cpu, times = self.existing_cpu_inputs()
        rows = self.fisher_rows()
        fisher = plot.summarize_fisher(rows, cpu=True)
        with self.assertRaises(ValueError):
            plot.merge_rows(cpu, times, fisher, plot.summarize_fisher_times(rows)[:-1])

    def test_cpu_figure_has_no_gpu_or_missing_time_labels_and_legacy_still_works(self):
        cpu, times = self.existing_cpu_inputs()
        rows = self.fisher_rows()
        current = plot.merge_rows(cpu, times, plot.summarize_fisher(rows, cpu=True),
                                  plot.summarize_fisher_times(rows))
        legacy_rows = [{**row, 'run_id': plot.legacy.OLD_RUNS[row['variant']]} for row in rows]
        archived = plot.merge_rows(cpu, times, plot.summarize_fisher(legacy_rows))
        documents = []
        with patch.object(plot.base, 'save_svg', side_effect=lambda canvas, dest, name:
                          documents.append(canvas.document())):
            plot.plot_unified(current, Path('/unused'))
            plot.plot_unified(archived, Path('/unused'))
        for document in documents:
            ET.fromstring(document)
        self.assertNotIn('[GPU]', documents[0])
        self.assertNotIn('N/A', documents[0])
        self.assertIn('FedFisher KFAC / deep', documents[0])
        self.assertEqual(documents[1].count('[GPU]*'), 4)
        self.assertEqual(documents[1].count('N/A: not recorded'), 8)


if __name__ == '__main__':
    unittest.main()
