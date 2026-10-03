"""In-memory reporting fixtures only; no fitting, inference or scheduler calls."""
import unittest
import xml.etree.ElementTree as ET

from reporting.effect_modifier import completed as plot


class CompletedNonIIDVisualizationTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [dict(cell=cell, variant=variant, seed=seed, split='noniid',
                     top2_probability=float(seed % 2), auroc=0.75)
                for cell in plot.CELLS for variant in plot.VARIANTS for seed in range(300)]

    def test_formal_inventory_excludes_pilots_and_group_lasso(self):
        self.assertEqual(len(plot.CELLS), 22)
        self.assertEqual(len(set(plot.CELLS)), 22)
        self.assertFalse(any('pilot' in cell or 'group_lasso' in cell for cell in plot.CELLS))
        for cell in plot.CELLS + plot.OLD_CELLS:
            self.assertTrue(plot.label(cell))

    def test_complete_summary_uses_seed_standard_errors(self):
        rows = plot.summarize(self.rows())
        self.assertEqual(len(rows), 88)
        self.assertTrue(all(row['top2_pct'] == 50 and row['n'] == 300 for row in rows))
        self.assertAlmostEqual(rows[0]['top2_se_pct'], 50 / (299 ** 0.5))
        self.assertTrue(all(row['metric'] == 'permute/tau_prob' for row in rows))

    def test_missing_and_duplicate_seeds_fail_closed(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            plot.summarize(rows[:-1])
        rows[-1] = dict(rows[-2])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            plot.summarize(rows)

    def test_iid_cannot_enter_requested_figures(self):
        rows = self.rows()
        rows[0]['split'] = 'iid'
        with self.assertRaisesRegex(ValueError, 'IID'):
            plot.summarize(rows)

    def test_adaptive_ties_remain_probabilities_not_binary_hits(self):
        rows = self.rows()
        for row in rows:
            row['top2_probability'] = 1 / 4950
        summaries = plot.summarize(rows)
        self.assertAlmostEqual(summaries[0]['top2_pct'], 100 / 4950)
        self.assertEqual(summaries[0]['top2_se_pct'], 0)

    def test_output_paths_cannot_escape_allowed_roots(self):
        with self.assertRaisesRegex(ValueError, 'outside allowed'):
            plot.safe_path('/home/unsafe-figure.png')
        self.assertEqual(plot.safe_path(plot.DEST / 'example.png'), plot.DEST / 'example.png')

    def test_svg_text_is_escaped_and_document_is_valid_xml(self):
        canvas = plot.SVG(400, 300)
        canvas.text(20, 40, 'A < B & C > D')
        canvas.marker(80, 60, '#0072B2', 2)
        root = ET.fromstring(canvas.document())
        texts = root.findall('.//{http://www.w3.org/2000/svg}text')
        self.assertEqual(texts[0].text, 'A < B & C > D')
        self.assertEqual(root.attrib['width'], '400')

    def test_heat_scale_is_fixed_zero_to_one_hundred(self):
        self.assertEqual(plot.heat_color(-1), plot.heat_color(0))
        self.assertEqual(plot.heat_color(101), plot.heat_color(100))
        self.assertNotEqual(plot.heat_color(0), plot.heat_color(100))
        for value in range(101):
            self.assertRegex(plot.heat_color(value), r'^#[0-9a-f]{6}$')


if __name__ == '__main__':
    unittest.main()
