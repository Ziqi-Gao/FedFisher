import unittest

from effect_modifier_experiment.run_one_shot_lasso_effect_modifier import parse_args


class OneShotManifestSeedRangeTests(unittest.TestCase):
    def _arguments(self, seed):
        return [
            "--family",
            "one_shot_local_logistic_lasso_fedavg",
            "--variant",
            "control",
            "--run-id",
            "one_shot_local_logistic_lasso_fedavg_control_300seeds",
            "--output-dir",
            "/data/del6500/FedFisher/effect_modifier_experiment/outputs/"
            "one_shot_local_logistic_lasso_fedavg_control_300seeds",
            "--seed",
            str(seed),
            "--seed-start",
            "0",
            "--seed-end",
            "299",
            "--split",
            "iid",
            "--model",
            "linear",
        ]

    def test_formal_seed_range_is_manifest_bound_across_cells(self):
        first = parse_args(self._arguments(0))
        last = parse_args(self._arguments(299))
        self.assertEqual((first.seed_start, first.seed_end), (0, 299))
        self.assertEqual((last.seed_start, last.seed_end), (0, 299))

    def test_cell_seed_outside_manifest_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "within the manifest seed range"):
            parse_args(self._arguments(300))


if __name__ == "__main__":
    unittest.main()
