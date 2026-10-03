"""In-memory reporting tests: never fit, infer, submit, or generate data."""

import copy
import csv
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from reporting.effect_modifier import unified as plot


class UnifiedComparisonTests(unittest.TestCase):
    @staticmethod
    def fisher_rows():
        return [dict(cell=c, variant=v, seed=s, split='noniid', metric=plot.METRIC,
                     run_id=plot.legacy.OLD_RUNS[v], top2_probability=float(s % 2), auroc=.75)
                for c in plot.FISHER_CELLS for v in plot.base.VARIANTS for s in range(300)]

    @classmethod
    def inputs(cls):
        cpu = [dict(cell=c, variant=v, split='noniid', n=300, metric=plot.METRIC,
                    top2_pct=60., top2_se_pct=2., auroc=.85, auroc_se=.02)
               for c in plot.base.CELLS for v in plot.base.VARIANTS]
        times = [dict(cell=r['cell'], variant=r['variant'],
                      **{f'{m}_{s}': (10. if s == 'mean' else .1)
                         for m in plot.timing.METRICS for s in ('mean', 'se')}) for r in cpu]
        return cpu, times, plot.summarize_fisher(cls.fisher_rows())

    @staticmethod
    def score_file(split='noniid', duplicate=False):
        rows = [dict(seed=0, model_source=method, model='SyntheticMLP', split=split,
                     permutation_scope='global', feature_idx=i, is_signal=int(i in (1, 2)),
                     intervention_mode='permute', metric='tau_intervention_prob', importance=101-i)
                for method in ('fedfisher_diag', 'fedfisher_kfac') for i in range(1, 101)]
        if duplicate:
            rows.append(rows[0])
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode()

    def test_method_inventory_interleaves_fisher_with_fedavg(self):
        self.assertEqual(len(plot.CELLS), 26)
        self.assertEqual(set(plot.CELLS), set(plot.base.CELLS) | set(plot.FISHER_CELLS))
        for depth in ('shallow', 'deep'):
            start = plot.CELLS.index(f'one_shot_fedavg_{depth}')
            self.assertEqual(plot.CELLS[start+1:start+3],
                             (f'fedfisher_diag_{depth}', f'fedfisher_kfac_{depth}'))

    def test_exact_ties_not_index_based(self):
        scores = {i: 0. for i in range(1, 101)}
        probability, auc, tied = plot.saved_score_recovery(scores)
        self.assertAlmostEqual(probability, 1/4950)
        self.assertEqual((auc, tied), (.5, 100))
        scores[1] = 1.
        self.assertAlmostEqual(plot.saved_score_recovery(scores)[0], 1/99)
        scores[2] = 1.
        self.assertEqual(plot.saved_score_recovery(scores)[0], 1.)
        scores[3] = 2.
        self.assertEqual(plot.saved_score_recovery(scores)[0], 0.)

    def test_saved_score_validation(self):
        good = {i: 101-i for i in range(1, 101)}
        self.assertEqual(plot.saved_score_recovery(good), (1., 1., 1))
        for bad in (-1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'invalid saved'):
                plot.saved_score_recovery({**good, 100: bad})
        with self.assertRaisesRegex(ValueError, 'raw-X'):
            plot.saved_score_recovery({i: 0 for i in range(100)})

    def test_parser_checks_identity_and_duplicate_features(self):
        args = ('control', plot.legacy.OLD_RUNS['control'], 'global')
        rows = plot.parse_scores(self.score_file(), *args)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['metric'] == plot.METRIC and r['top2_probability'] == 1 for r in rows))
        with self.assertRaisesRegex(ValueError, 'identity'):
            plot.parse_scores(self.score_file('iid'), *args)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            plot.parse_scores(self.score_file(duplicate=True), *args)

    def test_complete_seed_coverage_and_se(self):
        rows = self.fisher_rows()
        summaries = plot.summarize_fisher(rows)
        self.assertEqual(len(summaries), 16)
        self.assertAlmostEqual(summaries[0]['top2_se_pct'], 50 / (299**.5))
        for bad in (rows[:-1], rows[:-1] + [rows[-2]]):
            with self.assertRaisesRegex(ValueError, 'coverage'):
                plot.summarize_fisher(bad)

    def test_missing_times_remain_none(self):
        rows = plot.merge_rows(*self.inputs())
        self.assertEqual(len(rows), 104)
        missing = [r for r in rows if r['timing_status'] == 'not_recorded']
        self.assertEqual(len(missing), 16)
        self.assertTrue(all(r['fit_wall_seconds_mean'] is None for r in missing))
        self.assertTrue(all(r['evaluation_cpu_seconds_mean'] is None for r in missing))
        self.assertTrue(all(r['top2_pct'] == 50 for r in missing))

    def test_logit_cannot_silently_enter_common_endpoint(self):
        cpu, times, fisher = self.inputs()
        bad = copy.deepcopy(cpu)
        bad[0]['metric'] = 'permute/tau_intervention_logit'
        with self.assertRaisesRegex(ValueError, 'probability-scale'):
            plot.merge_rows(bad, times, fisher)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            plot.merge_rows(cpu, times[:-1], fisher)

    def test_unified_figure_retains_fisher_and_missing_time_labels(self):
        documents = []
        with patch.object(plot.base, 'save_svg', side_effect=lambda canvas, dest, name:
                          documents.append(canvas.document())):
            plot.plot_unified(plot.merge_rows(*self.inputs()), Path('/unused'))
        document = documents[0]
        ET.fromstring(document)
        self.assertEqual(document.count('N/A: not recorded'), 8)
        self.assertEqual(document.count('[GPU]*'), 4)
        self.assertIn('FedFisher KFAC / deep', document)
        self.assertIn('Probability-scale Top-2', document)
        self.assertNotIn('New formal comparison', document)
        self.assertNotIn('Historical formal comparison', document)


if __name__ == '__main__':
    unittest.main()
