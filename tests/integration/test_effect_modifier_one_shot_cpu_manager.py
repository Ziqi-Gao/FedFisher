import json
import tempfile
import unittest
from pathlib import Path

from effect_modifier_experiment.dslassocov_run_contract import env_text
from effect_modifier_experiment.manage_one_shot_lasso_cpu_run import (
    cell_stem,
    formal_runner_args,
    partial_cell_paths,
    validate_completed_cell,
)
from effect_modifier_experiment.one_shot_lasso_run_contract import sha256_file


RUNTIME_ROOT = Path("/scr/del6500/FedFisher/effect_modifier_experiment/runtime")


class OneShotCpuManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def _completed_fixture(self):
        temporary = tempfile.TemporaryDirectory(
            dir=str(RUNTIME_ROOT), prefix="one_shot_cpu_manager_test_"
        )
        root = Path(temporary.name)
        completed = root / "completed_cells"
        completed.mkdir()
        args = formal_runner_args(
            "one_shot_local_logistic_lasso_fedavg",
            "control",
            seed=0,
            split="iid",
            model="linear",
        )
        stem = cell_stem(args.seed, args.split, args.model)
        manifest = {
            "run_id": args.run_id,
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
        result_fields = [
            ("format_version", 2),
            ("run_id", args.run_id),
            ("family", args.family),
            ("variant", args.variant),
            ("seed", args.seed),
            ("split", args.split),
            ("model", args.model),
            ("provenance_sha256", manifest["provenance_sha256"]),
            ("run_config_sha256", manifest["run_config_sha256"]),
            ("method_config_sha256", manifest["method_config_sha256"]),
            ("variant_config_sha256", manifest["variant_config_sha256"]),
            ("training_communication_rounds", 1),
            ("site_upload_count", 10),
            ("post_upload_training_site_accesses", 0),
            ("evaluation_communication_rounds", 0),
            ("primary_metric", "permute/tau_intervention_logit"),
            ("primary_top2_hits", 2),
            ("primary_auroc", 1.0),
        ]
        fit = {
            "run_id": args.run_id,
            "family": args.family,
            "variant": args.variant,
            "seed": args.seed,
            "split": args.split,
            "model": args.model,
            "training_communication": {
                "training_communication_rounds": 1,
                "site_upload_count": 10,
                "observed_post_upload_training_site_accesses": 0,
            },
        }
        for key, suffix in suffixes.items():
            path = root / (stem + suffix)
            if key == "fit_json":
                path.write_text(json.dumps(fit) + "\n", encoding="utf-8")
            elif key == "result_env":
                path.write_text(env_text(result_fields), encoding="utf-8")
            else:
                path.write_text(key + "\n", encoding="utf-8")
        marker_fields = [
            ("format_version", 2),
            ("run_id", args.run_id),
            ("cell", stem),
            ("provenance_sha256", manifest["provenance_sha256"]),
            ("run_config_sha256", manifest["run_config_sha256"]),
            ("method_config_sha256", manifest["method_config_sha256"]),
            ("variant_config_sha256", manifest["variant_config_sha256"]),
        ]
        for key, suffix in suffixes.items():
            marker_fields.append((key + "_sha256", sha256_file(root / (stem + suffix))))
        (completed / (stem + ".done")).write_text(
            env_text(marker_fields), encoding="utf-8"
        )
        return temporary, root, args, manifest, stem

    def test_completed_cell_validates_semantics_and_all_hashes(self):
        temporary, root, args, manifest, _ = self._completed_fixture()
        with temporary:
            result = validate_completed_cell(root, manifest, args)
            self.assertEqual(result["training_communication_rounds"], "1")
            self.assertEqual(result["site_upload_count"], "10")
            self.assertEqual(result["primary_top2_hits"], "2")

    def test_hash_tampering_fails_closed(self):
        temporary, root, args, manifest, stem = self._completed_fixture()
        with temporary:
            result_path = root / (stem + "_result.env")
            result_path.write_text(result_path.read_text() + "tampered=1\n")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_completed_cell(root, manifest, args)

    def test_partial_artifact_is_detected_without_marker(self):
        temporary = tempfile.TemporaryDirectory(
            dir=str(RUNTIME_ROOT), prefix="one_shot_cpu_partial_test_"
        )
        with temporary:
            root = Path(temporary.name)
            stem = "oneshot_seed_000000_iid_linear"
            artifact = root / (stem + "_fit.json")
            artifact.write_text("{}\n", encoding="utf-8")
            self.assertEqual(partial_cell_paths(root, stem), [artifact])


if __name__ == "__main__":
    unittest.main()
