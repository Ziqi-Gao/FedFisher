import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from data import (
    build_effect_modifier_site_evaluation_datasets,
    get_dataset,
    get_effect_modifier_site_gaussian_parameters,
)
from models import SyntheticMLP
from scripts.summarize_effect_modifier_ablations import (
    RESULTS_ROOT,
    VARIANT_CONTRACT,
    resolve_within,
    validate_comparison_manifests,
)
from utils.prediction_intervention import (
    make_intervened_X,
    site_local_treatment_contrast_intervention_importance,
    treatment_contrast_intervention_importance,
)


class EffectModifierAblationTests(unittest.TestCase):
    def _main_command(self):
        project_root = Path(__file__).resolve().parents[1]
        return [
            sys.executable,
            "-B",
            str(project_root / "main.py"),
            "--dataset",
            "SyntheticEffectModifier",
            "--model",
            "SyntheticMLP",
            "--algs_to_run",
            "fedavg",
            "--prediction_intervention",
        ]

    def _run_main_parser(self, *extra_args):
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            self._main_command() + list(extra_args),
            cwd=str(project_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_cli_rejects_combined_ablations(self):
        result = self._run_main_parser(
            "--effect_modifier_site_gaussian_shift",
            "--prediction_intervention_scope",
            "site",
            "--output_dir",
            str(Path(__file__).resolve().parents[1] / "results"),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("mutually exclusive ablations", result.stderr)

    def test_cli_rejects_output_outside_allowed_roots(self):
        result = self._run_main_parser("--output_dir", "/tmp")
        self.assertEqual(result.returncode, 2)
        self.assertIn("output_dir resolves outside", result.stderr)

    def _write_test_manifest(self, directory, variant, provenance, run_config):
        contract = VARIANT_CONTRACT[variant]
        variant_text = (
            "effect_modifier_site_gaussian_shift=%s\n"
            "prediction_intervention_scope=%s\n"
            % (
                contract["effect_modifier_site_gaussian_shift"],
                contract["prediction_intervention_scope"],
            )
        )
        variant_sha = hashlib.sha256(variant_text.encode()).hexdigest()
        (directory / "run_manifest.env").write_text(
            "\n".join(
                [
                    "format_version=2",
                    "run_id=%s" % directory.name,
                    "effect_modifier_variant=%s" % variant,
                    "provenance_sha256=%s" % provenance,
                    "run_config_sha256=%s" % run_config,
                    "variant_config_sha256=%s" % variant_sha,
                    "seed_start=3",
                    "seed_end=7",
                    "effect_modifier_site_gaussian_shift=%s"
                    % contract["effect_modifier_site_gaussian_shift"],
                    "prediction_intervention_scope=%s"
                    % contract["prediction_intervention_scope"],
                ]
            )
            + "\n"
        )

    def _comparison_manifest_fixture(self):
        temporary = tempfile.TemporaryDirectory(dir="/scr/del6500/FedFisher")
        root = Path(temporary.name)
        inputs = {
            "control": root / "prediction_intervention_test_control",
            "site_gaussian_shift": root / "site_gaussian_shift_test",
            "local_permutation": root / "local_permutation_test",
        }
        provenance = "a" * 64
        run_config = "b" * 64
        for variant, directory in inputs.items():
            directory.mkdir()
            self._write_test_manifest(directory, variant, provenance, run_config)
        return temporary, inputs

    def test_summary_requires_comparable_v2_manifests(self):
        temporary, inputs = self._comparison_manifest_fixture()
        with temporary:
            manifests = validate_comparison_manifests(inputs)
            self.assertEqual(set(manifests), set(inputs))

            site_manifest = inputs["site_gaussian_shift"] / "run_manifest.env"
            text = site_manifest.read_text().replace(
                "provenance_sha256=" + "a" * 64,
                "provenance_sha256=" + "c" * 64,
            )
            site_manifest.write_text(text)
            with self.assertRaisesRegex(ValueError, "provenance_sha256 differs"):
                validate_comparison_manifests(inputs)

    def test_summary_rejects_legacy_control_without_manifest(self):
        temporary, inputs = self._comparison_manifest_fixture()
        with temporary:
            (inputs["control"] / "run_manifest.env").unlink()
            with self.assertRaisesRegex(ValueError, "requires a regular non-symlink"):
                validate_comparison_manifests(inputs)

    def test_summary_rejects_output_outside_results_root(self):
        with self.assertRaisesRegex(ValueError, "resolves outside"):
            resolve_within(Path("/tmp"), RESULTS_ROOT, "output directory")

    def _dataset(self, site_gaussian_shift=False, split="noniid"):
        with redirect_stdout(io.StringIO()):
            return get_dataset(
                datatype="SyntheticEffectModifier",
                n_client=10,
                n_c=2,
                alpha=0.1,
                synthetic_split=split,
                synthetic_num_test=250,
                effect_modifier_covariate_dim=8,
                effect_modifier_signal_dim=2,
                effect_modifier_signal_strength=2.0,
                effect_modifier_main_effect_strength=1.0,
                effect_modifier_confounding_strength=1.0,
                effect_modifier_intercept=0.0,
                effect_modifier_treatment_prob=0.5,
                effect_modifier_propensity_clip=0.05,
                effect_modifier_site_min_samples=30,
                effect_modifier_site_max_samples=40,
                effect_modifier_site_gaussian_shift=site_gaussian_shift,
                seed=17,
            )

    def test_default_site_gaussian_design_is_five_by_two(self):
        parameters = get_effect_modifier_site_gaussian_parameters(10)
        expected_means = [-0.75, -0.375, 0.0, 0.375, 0.75]
        self.assertEqual(
            parameters,
            [
                (site_mean, site_std)
                for site_mean in expected_means
                for site_std in (0.75, 1.25)
            ],
        )

    def test_site_gaussian_changes_only_training_dgp(self):
        for split in ["iid", "noniid"]:
            baseline = self._dataset(site_gaussian_shift=False, split=split)
            shifted = self._dataset(site_gaussian_shift=True, split=split)

            baseline_sites, _, baseline_test, _ = baseline
            shifted_sites, _, shifted_test, _ = shifted
            self.assertEqual(
                [len(dataset) for dataset in baseline_sites],
                [len(dataset) for dataset in shifted_sites],
            )
            for baseline_site, shifted_site in zip(baseline_sites, shifted_sites):
                torch.testing.assert_close(
                    torch.bincount(baseline_site.tensors[1], minlength=2),
                    torch.bincount(shifted_site.tensors[1], minlength=2),
                )

            torch.testing.assert_close(
                baseline_test.tensors[0],
                shifted_test.tensors[0],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                baseline_test.tensors[1],
                shifted_test.tensors[1],
                rtol=0.0,
                atol=0.0,
            )

    def test_local_test_shards_are_disjoint_and_exhaustive(self):
        for split in ["iid", "noniid"]:
            sites, _, test_dataset, _ = self._dataset(split=split)
            shards = build_effect_modifier_site_evaluation_datasets(
                test_dataset,
                sites,
                split=split,
                alpha=0.1,
                seed=17,
            )
            indices = [index for shard in shards for index in shard.indices]
            self.assertEqual(len(shards), 10)
            self.assertEqual(sorted(indices), list(range(len(test_dataset))))
            self.assertEqual(len(indices), len(set(indices)))
            self.assertTrue(all(len(shard) > 0 for shard in shards))

    def test_weighted_local_zero_equals_global_zero(self):
        sites, _, test_dataset, _ = self._dataset()
        shards = build_effect_modifier_site_evaluation_datasets(
            test_dataset,
            sites,
            split="noniid",
            alpha=0.1,
            seed=17,
        )
        torch.manual_seed(5)
        model = SyntheticMLP(input_dim=17, hidden_dims=(8,), n_out=2)
        args = {
            "device": "cpu",
            "bs": 64,
            "dataset": "SyntheticEffectModifier",
            "effect_modifier_covariate_dim": 8,
        }
        global_result = treatment_contrast_intervention_importance(
            model,
            test_dataset,
            args,
            modes=["zero"],
            repeats=1,
            seed=1,
            candidate_indices=range(1, 9),
        )
        local_result = site_local_treatment_contrast_intervention_importance(
            model,
            shards,
            args,
            modes=["zero"],
            repeats=1,
            seed=1,
            candidate_indices=range(1, 9),
        )

        for metric in global_result["scores"]["zero"]:
            np.testing.assert_allclose(
                global_result["scores"]["zero"][metric][1:9],
                local_result["scores"]["zero"][metric][1:9],
                rtol=2e-6,
                atol=2e-7,
            )

    def test_permutation_cannot_cross_site_boundaries(self):
        site_a = TensorDataset(
            torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
            torch.zeros(4, dtype=torch.long),
        )
        site_b = TensorDataset(
            torch.tensor([[100.0], [101.0], [102.0], [103.0]]),
            torch.zeros(4, dtype=torch.long),
        )
        generator = torch.Generator().manual_seed(9)
        for dataset in [site_a, site_b]:
            changed = make_intervened_X(
                dataset.tensors[0],
                feature_idx=0,
                mode="permute",
                generator=generator,
            )
            self.assertEqual(
                sorted(changed[:, 0].tolist()),
                sorted(dataset.tensors[0][:, 0].tolist()),
            )


if __name__ == "__main__":
    unittest.main()
