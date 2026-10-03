import tempfile
import unittest
from pathlib import Path

from effect_modifier_experiment.dslassocov_run_contract import env_text
from effect_modifier_experiment.one_shot_lasso.invariants import (
    expected_evaluation_rounds,
    validate_marker,
)
from effect_modifier_experiment.one_shot_lasso_run_contract import sha256_file


RUNTIME_ROOT = Path(
    "/scr/del6500/FedFisher/effect_modifier_experiment/runtime"
)


class OneShotCheckerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def _fixture(self, include_site_artifact, include_site_hash):
        temporary = tempfile.TemporaryDirectory(
            dir=str(RUNTIME_ROOT), prefix="one_shot_checker_test_"
        )
        root = Path(temporary.name)
        completed = root / "completed_cells"
        completed.mkdir()
        stem = "oneshot_seed_000000_iid_linear"
        manifest = {
            "run_id": "one_shot_local_logistic_lasso_fedavg_control_test",
            "provenance_sha256": "a" * 64,
            "run_config_sha256": "b" * 64,
            "method_config_sha256": "c" * 64,
            "variant_config_sha256": "d" * 64,
        }
        suffixes = {
            "detail_csv": "_prediction_intervention.csv",
            "summary_csv": "_prediction_intervention_summary.csv",
            "model_summary_csv": "_prediction_intervention_model_summary.csv",
            "fit_json": "_fit.json",
            "result_env": "_result.env",
        }
        fields = [
            ("format_version", "2"),
            ("run_id", manifest["run_id"]),
            ("cell", stem),
            ("provenance_sha256", manifest["provenance_sha256"]),
            ("run_config_sha256", manifest["run_config_sha256"]),
            ("method_config_sha256", manifest["method_config_sha256"]),
            ("variant_config_sha256", manifest["variant_config_sha256"]),
        ]
        for key, suffix in suffixes.items():
            path = root / (stem + suffix)
            path.write_text(key + "\n", encoding="utf-8")
            fields.append((key + "_sha256", sha256_file(path)))
        site_path = root / (stem + "_prediction_intervention_site_summary.csv")
        if include_site_artifact:
            site_path.write_text("site\n", encoding="utf-8")
        if include_site_hash:
            fields.append(("site_summary_csv_sha256", sha256_file(site_path)))
        (completed / (stem + ".done")).write_text(
            env_text(fields), encoding="utf-8"
        )
        return temporary, root, stem, manifest

    def test_variant_evaluation_round_contract(self):
        self.assertEqual(expected_evaluation_rounds("control"), "0")
        self.assertEqual(expected_evaluation_rounds("site_gaussian_shift"), "0")
        self.assertEqual(expected_evaluation_rounds("local_permutation"), "1")
        self.assertEqual(
            expected_evaluation_rounds("site_gaussian_local_permutation"), "1"
        )

    def test_local_marker_requires_site_summary_hash(self):
        temporary, root, stem, manifest = self._fixture(True, False)
        with temporary:
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_marker(root, stem, manifest, require_site_summary=True)

    def test_global_marker_rejects_site_summary_artifact(self):
        temporary, root, stem, manifest = self._fixture(True, True)
        with temporary:
            with self.assertRaisesRegex(ValueError, "must not contain site summary"):
                validate_marker(root, stem, manifest, require_site_summary=False)


if __name__ == "__main__":
    unittest.main()
