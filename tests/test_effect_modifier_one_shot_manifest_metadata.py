import unittest

from effect_modifier_experiment.one_shot_lasso_run_contract import METHOD_NAMES
from effect_modifier_experiment.run_one_shot_lasso_effect_modifier import (
    method_config,
    parse_args,
)


class OneShotManifestMetadataTests(unittest.TestCase):
    def _args(self, family, model):
        run_id = family + "_control_metadata_test"
        return parse_args(
            [
                "--family",
                family,
                "--variant",
                "control",
                "--run-id",
                run_id,
                "--output-dir",
                "/data/del6500/FedFisher/effect_modifier_experiment/outputs/"
                + run_id,
                "--split",
                "iid",
                "--model",
                model,
                "--covariate-dim",
                "6",
            ]
        )

    def test_logistic_manifest_description_is_dimension_aware(self):
        config = method_config(
            self._args("one_shot_local_logistic_lasso_fedavg", "linear")
        )
        self.assertEqual(config["unpenalized_block"], "intercept,A,6_raw_X_main_effects")
        self.assertEqual(config["penalized_block"], "6_A_by_X_interactions_only")
        self.assertIn("14_float64", config["payload"])
        self.assertEqual(config["payload_bytes_per_site"], 128)
        self.assertNotIn("202", config["payload"])

    def test_mlp_manifest_limits_interpretation_and_records_matched_reference(self):
        family = "one_shot_mlp_group_lasso_fedavg"
        config = method_config(self._args(family, "SyntheticMLP"))
        self.assertIn("6_first_layer_explicit", config["penalty"])
        self.assertEqual(
            config["regularization_interpretation"],
            "explicit_A_by_X_input_columns_only_not_identifiable_modifier_selection",
        )
        self.assertFalse(config["legacy_GPU_FedAvg_bitwise_equivalence_claim"])
        self.assertEqual(config["execution_device"], "cpu")
        self.assertIn("explicit A-by-X input", METHOD_NAMES[family])


if __name__ == "__main__":
    unittest.main()
