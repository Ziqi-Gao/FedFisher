import copy
import unittest

from effect_modifier_experiment.causal_comparison import protocol
from effect_modifier_experiment.causal_comparison.reporting import aggregate, metrics, validate_pairs


class ComparisonReportingTests(unittest.TestCase):
    def test_fit_time_includes_tuning_and_aggregation_once_and_excludes_data(self):
        stages = {name: dict(wall_seconds=value, cpu_seconds=value / 2)
                  for name, value in (("data_preparation", 100), ("train", 3),
                                      ("tuning", 5), ("aggregation", 2), ("evaluation", 7))}
        result = dict(timing=dict(stages=stages), evaluation=dict(metrics={
            "permute/tau_prob": dict(recovery=dict(expected_topk_hits=0.04,
                expected_topk_precision=0.02, probability_all_signals_selected=1/4950, auroc=0.5))}))
        observed = metrics(result)
        self.assertEqual(observed["fit_wall_seconds"], 10)
        self.assertEqual(observed["fit_cpu_seconds"], 5)
        self.assertEqual(observed["evaluation_wall_seconds"], 7)
        self.assertEqual(observed["expected_topk_hits"], 0.04)

    def test_standard_errors_are_across_seeds_and_never_fabricated_for_single_seed(self):
        rows = [dict(cell="r_unpenalized", split="iid", variant_or_contrast="control",
                     seed=seed, values=dict(auroc=value)) for seed, value in enumerate((0.5, 0.9))]
        summary = aggregate(rows)[0]
        self.assertAlmostEqual(summary["mean"], 0.7)
        self.assertAlmostEqual(summary["standard_error"], 0.2)
        self.assertIsNone(aggregate(rows[:1])[0]["standard_error"])

    def test_factorial_pairs_require_identical_fit_and_matching_interventions(self):
        records = {}
        for variant in protocol.VARIANTS:
            training = "gaussian" if "gaussian" in variant else "standard"
            permutation = "local" if "local_permutation" in variant else "global"
            records[variant] = dict(fingerprints=dict(test="test", training=training, model=training),
                                   fit=dict(fit_fingerprint=training),
                                   evaluation=dict(permutation_sha256=permutation))
        validate_pairs(records)
        for section, field in (("fingerprints", "test"), ("fingerprints", "training"),
                               ("fingerprints", "model"), ("fit", "fit_fingerprint"),
                               ("evaluation", "permutation_sha256")):
            bad = copy.deepcopy(records)
            bad["control"][section][field] = "changed"
            with self.assertRaises(ValueError):
                validate_pairs(bad)
        with self.assertRaises(ValueError):
            validate_pairs({"control": records["control"]})


if __name__ == "__main__":
    unittest.main()
