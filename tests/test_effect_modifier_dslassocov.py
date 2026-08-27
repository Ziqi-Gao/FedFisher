import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock
import time

import numpy as np
import torch

from data import build_effect_modifier_site_evaluation_datasets, get_dataset
from effect_modifier_experiment.dslassocov_benchmark import (
    FederatedSite,
    OptimizerConfig,
    aggregate_loss_and_gradient,
    aggregate_standardization,
    backtransform_coefficients,
    dataset_fingerprint,
    deterministic_site_folds,
    federated_logistic_lasso_cv,
    get_variant_contract,
    proximal_gradient_fit,
    standardize_sites,
    tensor_datasets_to_raw_sites,
    training_fingerprint,
)
from effect_modifier_experiment.dslassocov_run_contract import (
    BENCHMARK_CONFIG_KEYS,
    PRIVACY_LABEL,
    RUN_CONFIG_KEYS,
    VARIANT_CONFIG_KEYS,
    validate_manifest_self_consistency,
    config_hashes,
)
from effect_modifier_experiment import manage_dslassocov_cpu_run as run_manager
from effect_modifier_experiment.manage_dslassocov_cpu_run import validate_config
from effect_modifier_experiment import run_dslassocov_effect_modifier as runner


SCRATCH_TEST_ROOT = Path("/scr/del6500/FedFisher")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CPU_LOCK_ROOT = SCRATCH_TEST_ROOT / "effect_modifier_experiment" / "cpu_locks"
TEST_RUNTIME_ROOT = SCRATCH_TEST_ROOT / "effect_modifier_experiment" / "runtime" / "dslassocov_tests"
CPU_LOCK_HELPER = PROJECT_ROOT / "scripts" / "effect_modifier_dslassocov_cpu_lock.sh"


class DsLassoCovCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _datasets(self, variant, split="noniid", seed=17):
        contract = get_variant_contract(variant)
        with redirect_stdout(io.StringIO()):
            site_datasets, _, test_dataset, _ = get_dataset(
                datatype="SyntheticEffectModifier",
                n_client=10,
                n_c=2,
                alpha=0.1,
                synthetic_split=split,
                synthetic_num_test=120,
                effect_modifier_covariate_dim=6,
                effect_modifier_signal_dim=2,
                effect_modifier_signal_strength=2.0,
                effect_modifier_main_effect_strength=1.0,
                effect_modifier_confounding_strength=1.0,
                effect_modifier_intercept=0.0,
                effect_modifier_treatment_prob=0.5,
                effect_modifier_propensity_clip=0.05,
                effect_modifier_site_min_samples=20,
                effect_modifier_site_max_samples=25,
                effect_modifier_site_gaussian_shift=contract[
                    "site_gaussian_shift"
                ],
                seed=seed,
            )
        return site_datasets, test_dataset

    def _fit(self, site_datasets):
        raw_sites = tensor_datasets_to_raw_sites(site_datasets, 6)
        fit = federated_logistic_lasso_cv(
            raw_sites,
            covariate_dim=6,
            num_folds=2,
            lambda_count=3,
            lambda_min_ratio=0.1,
            cv_seed=4000054,
            optimizer_config=OptimizerConfig(
                max_iter=2000,
                tolerance=1e-5,
            ),
        )
        return raw_sites, fit

    def test_small_update_does_not_bypass_stationarity_check(self):
        with redirect_stdout(io.StringIO()):
            site_datasets, _, _, _ = get_dataset(
                datatype="SyntheticEffectModifier",
                n_client=10,
                n_c=2,
                alpha=0.1,
                synthetic_split="iid",
                synthetic_num_test=500,
                effect_modifier_covariate_dim=100,
                effect_modifier_signal_dim=2,
                effect_modifier_signal_strength=2.0,
                effect_modifier_main_effect_strength=1.0,
                effect_modifier_confounding_strength=1.0,
                effect_modifier_intercept=0.0,
                effect_modifier_treatment_prob=0.5,
                effect_modifier_propensity_clip=0.05,
                effect_modifier_site_min_samples=500,
                effect_modifier_site_max_samples=1000,
                effect_modifier_site_gaussian_shift=True,
                effect_modifier_site_gaussian_mean_span=0.75,
                effect_modifier_site_gaussian_std_low=0.75,
                effect_modifier_site_gaussian_std_high=1.25,
                seed=0,
            )
        raw_sites = tensor_datasets_to_raw_sites(site_datasets, 100)
        means, scales, _ = aggregate_standardization(raw_sites)
        standardized_sites = standardize_sites(raw_sites, means, scales)
        penalty_start = 102
        unpenalized_sites = [
            FederatedSite(
                design=site.design[:, :penalty_start],
                labels=site.labels,
                site_idx=site.site_idx,
            )
            for site in standardized_sites
        ]
        _, report = proximal_gradient_fit(
            unpenalized_sites,
            penalty_start=penalty_start,
            lambda_value=0.0,
            config=OptimizerConfig(max_iter=2000, tolerance=1e-6),
        )
        self.assertTrue(report["converged"])
        self.assertLessEqual(
            report["proximal_gradient_mapping_inf"],
            report["stationarity_threshold"],
        )
        self.assertLessEqual(
            report["kkt_residual_inf"],
            report["stationarity_threshold"],
        )

    def test_site_aggregated_loss_gradient_equals_pooled_math(self):
        rng = np.random.default_rng(9)
        sites = []
        pooled_design = []
        pooled_labels = []
        for site_idx, size in enumerate((7, 11, 5)):
            design = rng.normal(size=(size, 8))
            design[:, 0] = 1.0
            labels = rng.integers(0, 2, size=size).astype(np.float64)
            sites.append(FederatedSite(design, labels, site_idx))
            pooled_design.append(design)
            pooled_labels.append(labels)
        coefficients = rng.normal(size=8)
        loss, gradient, summaries = aggregate_loss_and_gradient(
            sites, coefficients
        )
        design = np.concatenate(pooled_design, axis=0)
        labels = np.concatenate(pooled_labels)
        scores = design.dot(coefficients)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        expected_loss = np.mean(np.logaddexp(0.0, scores) - labels * scores)
        expected_gradient = design.T.dot(probabilities - labels) / len(labels)
        self.assertAlmostEqual(loss, expected_loss, places=13)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-13, atol=1e-13)
        self.assertEqual(sum(row["num_examples"] for row in summaries), len(labels))

    def test_standardization_backtransform_preserves_scores(self):
        sites, _ = self._datasets("control", split="iid")
        raw_sites = tensor_datasets_to_raw_sites(sites, 6)
        feature_dim = raw_sites[0].features.shape[1]
        means = np.linspace(-0.3, 0.2, feature_dim)
        scales = np.linspace(0.7, 1.3, feature_dim)
        standardized = standardize_sites(raw_sites, means, scales)
        rng = np.random.default_rng(4)
        standardized_coefficients = rng.normal(size=feature_dim + 1)
        intercept, coefficients = backtransform_coefficients(
            standardized_coefficients, means, scales
        )
        original_scores = raw_sites[0].features.dot(coefficients) + intercept
        standardized_scores = standardized[0].design.dot(
            standardized_coefficients
        )
        np.testing.assert_allclose(
            original_scores, standardized_scores, rtol=1e-12, atol=1e-12
        )

    def test_site_local_folds_are_deterministic_and_exhaustive(self):
        sites, _ = self._datasets("control")
        raw_sites = tensor_datasets_to_raw_sites(sites, 6)
        first = deterministic_site_folds(raw_sites, 3, 71)
        second = deterministic_site_folds(raw_sites, 3, 71)
        for site, left, right in zip(raw_sites, first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(left.shape, (site.features.shape[0],))
            self.assertEqual(set(left.tolist()), {0, 1, 2})

    def test_factorial_pairs_have_identical_training_fit_and_test(self):
        pairs = (
            ("control", "local_permutation"),
            ("site_gaussian_shift", "site_gaussian_local_permutation"),
        )
        all_test_fingerprints = []
        for left_variant, right_variant in pairs:
            left_sites, left_test = self._datasets(left_variant)
            right_sites, right_test = self._datasets(right_variant)
            left_raw, left_fit = self._fit(left_sites)
            right_raw, right_fit = self._fit(right_sites)
            self.assertEqual(
                training_fingerprint(left_raw),
                training_fingerprint(right_raw),
            )
            self.assertEqual(
                left_fit["fit_fingerprint"], right_fit["fit_fingerprint"]
            )
            self.assertEqual(
                dataset_fingerprint(left_test),
                dataset_fingerprint(right_test),
            )
            all_test_fingerprints.extend(
                (dataset_fingerprint(left_test), dataset_fingerprint(right_test))
            )
        self.assertEqual(len(set(all_test_fingerprints)), 1)

    def test_selected_lambda_passes_proximal_and_kkt_stationarity(self):
        sites, _ = self._datasets("control")
        _, fit = self._fit(sites)
        selected_index = fit["cv"]["selected_index"]
        reports = [
            fold["lambda_fits"][selected_index]
            for fold in fit["cv"]["fold_reports"]
        ]
        reports.append(fit["optimizer"]["selected_lambda_fit"])
        for report in reports:
            self.assertTrue(report["converged"])
            self.assertLessEqual(
                report["proximal_gradient_mapping_inf"],
                report["stationarity_threshold"],
            )
            self.assertLessEqual(
                report["kkt_residual_inf"],
                report["stationarity_threshold"],
            )
        self.assertEqual(fit["penalty_start"], 8)
        self.assertEqual(fit["num_penalized"], 6)

    def test_site_local_test_shards_are_disjoint_and_exhaustive(self):
        sites, test_dataset = self._datasets("local_permutation")
        shards = build_effect_modifier_site_evaluation_datasets(
            test_dataset,
            sites,
            split="noniid",
            alpha=0.1,
            seed=17,
        )
        indices = [index for shard in shards for index in shard.indices]
        self.assertEqual(len(shards), 10)
        self.assertEqual(sorted(indices), list(range(len(test_dataset))))
        self.assertEqual(len(indices), len(set(indices)))

    def test_site_gaussian_audit_records_intended_and_realized_moments(self):
        sites, _ = self._datasets("site_gaussian_shift")
        args = mock.Mock(
            variant="site_gaussian_shift",
            num_sites=10,
            covariate_dim=6,
            site_gaussian_mean_span=0.75,
            site_gaussian_std_low=0.75,
            site_gaussian_std_high=1.25,
        )
        audit = runner.build_site_training_audit(args, sites)
        self.assertEqual(audit["proposal"], "fixed_site_gaussian_mean_by_scale")
        self.assertEqual(len(audit["sites"]), 10)
        self.assertEqual(
            [row["intended_raw_x_mean"] for row in audit["sites"][:2]],
            [-0.75, -0.75],
        )
        self.assertEqual(
            [row["intended_raw_x_std"] for row in audit["sites"][:2]],
            [0.75, 1.25],
        )
        self.assertTrue(
            all(len(row["empirical_raw_x_feature_means"]) == 6 for row in audit["sites"])
        )


class DsLassoCovRunContractTests(unittest.TestCase):
    def _args(self):
        args = runner.build_parser().parse_args(
            [
                "--run-id",
                "dslassocov_control_test",
                "--variant",
                "control",
                "--output-dir",
                str(
                    Path(
                        "/data/del6500/FedFisher/effect_modifier_experiment/outputs"
                    )
                    / "dslassocov_control_test"
                ),
                "--seed",
                "3",
                "--split",
                "iid",
                "--covariate-dim",
                "6",
                "--site-min-samples",
                "20",
                "--site-max-samples",
                "25",
                "--num-test",
                "120",
                "--cv-folds",
                "2",
                "--lambda-count",
                "3",
                "--lambda-min-ratio",
                "0.1",
                "--optimizer-max-iter",
                "2000",
                "--optimizer-tolerance",
                "0.00001",
            ]
        )
        return args

    def test_manifest_canonical_hash_rejects_parameter_drift(self):
        args = self._args()
        manifest = {"seed_start": "0", "seed_end": "9"}
        values = runner.manifest_config_values(args, manifest)
        manifest.update(values)
        manifest.update(config_hashes(values))
        runner.validate_args_against_manifest(args, manifest)
        manifest["lambda_count"] = "4"
        with self.assertRaisesRegex(ValueError, "lambda_count"):
            runner.validate_args_against_manifest(args, manifest)

    def test_manifest_self_consistency_recomputes_hashes_and_variant_contract(self):
        args = self._args()
        manifest = {
            "format_version": "2",
            "run_id": "dslassocov_control_test",
            "effect_modifier_variant": "control",
            "seed_start": "0",
            "seed_end": "9",
        }
        values = runner.manifest_config_values(args, manifest)
        manifest.update(values)
        manifest.update(config_hashes(values))
        self.assertEqual(
            validate_manifest_self_consistency(manifest),
            values,
        )

        tampered_hash = dict(manifest)
        tampered_hash["lambda_count"] = "4"
        with self.assertRaisesRegex(ValueError, "canonical configuration hash"):
            validate_manifest_self_consistency(tampered_hash)

        tampered_variant = dict(manifest)
        tampered_variant["effect_modifier_site_gaussian_shift"] = "1"
        tampered_variant.update(config_hashes(tampered_variant))
        with self.assertRaisesRegex(ValueError, "variant contract"):
            validate_manifest_self_consistency(tampered_variant)

        noncanonical = dict(manifest)
        noncanonical["seed_start"] = "00"
        with self.assertRaisesRegex(ValueError, "not canonical"):
            validate_manifest_self_consistency(noncanonical)

        wrong_namespace = dict(manifest)
        wrong_namespace["run_id"] = "dslassocov_site_gaussian_shift_test"
        with self.assertRaisesRegex(ValueError, "RUN_ID must start"):
            validate_manifest_self_consistency(wrong_namespace)

    def test_config_front_door_rejects_invalid_bounds(self):
        args = self._args()
        values = runner.manifest_config_values(
            args, {"seed_start": "0", "seed_end": "9"}
        )
        validate_config(values)
        alternate_spelling = dict(values)
        alternate_spelling["seed_start"] = "000"
        alternate_spelling["optimizer_tolerance"] = "0.000001"
        normalized = run_manager.normalize_config(alternate_spelling)
        self.assertEqual(normalized["seed_start"], "0")
        self.assertEqual(normalized["optimizer_tolerance"], "1e-06")

        for key, bad_value in (
            ("cv_folds", "1"),
            ("lambda_count", "1"),
            ("lambda_min_ratio", "1.0"),
            ("optimizer_tolerance", "0"),
            ("effect_modifier_propensity_clip", "0.5"),
            ("effect_modifier_site_gaussian_std_low", "0"),
            ("optimizer_initial_step", "nan"),
        ):
            broken = dict(values)
            broken[key] = bad_value
            with self.assertRaises(ValueError, msg=key):
                validate_config(broken)
        oversubscribed = dict(values)
        oversubscribed["cpu_workers"] = "4"
        oversubscribed["threads_per_worker"] = "2"
        with self.assertRaisesRegex(ValueError, "must be <= 4"):
            validate_config(oversubscribed)

    def test_launcher_rejects_oversubscribed_cpu_product(self):
        environment = os.environ.copy()
        environment.update(
            {
                "EFFECT_MODIFIER_VARIANT": "control",
                "RUN_ID": "dslassocov_control_oversubscription_test",
                "PYTHON_BIN": "/bin/true",
                "CPU_WORKERS": "4",
                "THREADS_PER_WORKER": "2",
            }
        )
        completed = subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts" / "launch_effect_modifier_dslassocov_cpu.sh")],
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("CPU_WORKERS * THREADS_PER_WORKER must be <= 4", completed.stderr)


    def test_atomic_json_success_and_existing_target_refusal(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH_TEST_ROOT) as directory:
            path = Path(directory) / "result.json"
            runner.atomic_json(path, {"ok": True})
            self.assertEqual(path.read_text(), '{\n  "ok": true\n}\n')
            with self.assertRaises(FileExistsError):
                runner.atomic_json(path, {"ok": False})
            self.assertFalse(any(Path(directory).glob(".*.tmp.*")))

    def test_atomic_json_failure_preserves_original_exception_and_cleans_temp(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH_TEST_ROOT) as directory:
            path = Path(directory) / "result.json"
            with mock.patch.object(
                runner.json,
                "dump",
                side_effect=RuntimeError("injected-json-failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected-json-failure"):
                    runner.atomic_json(path, {"ok": True})
            self.assertFalse(path.exists())
            self.assertFalse(any(Path(directory).glob(".*.tmp.*")))

    def test_csv_failure_cleans_temporaries_and_creates_no_completion(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH_TEST_ROOT) as directory:
            root = Path(directory)
            paths = {
                key: root / ("%s.csv" % key)
                for key in ("detail", "summary", "model_summary", "site_summary")
            }

            def injected_failure(*call_args, **call_kwargs):
                Path(call_args[3]).write_text("partial")
                raise RuntimeError("injected-csv-failure")

            with mock.patch.object(
                runner,
                "write_prediction_intervention_outputs",
                side_effect=injected_failure,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected-csv-failure"):
                    runner.atomic_prediction_outputs([], [], [], [], paths)
            self.assertFalse(any(path.exists() for path in paths.values()))
            self.assertFalse(any(root.glob(".*.tmp.*")))
    def test_done_marker_requires_both_semantic_results_and_fingerprints(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH_TEST_ROOT) as directory:
            root = Path(directory)
            manifest = {
                "run_id": "dslassocov_control_test",
                "provenance_sha256": "a" * 64,
                "run_config_sha256": "b" * 64,
                "variant_config_sha256": "c" * 64,
                "benchmark_config_sha256": "d" * 64,
            }
            results = {}
            for split in ("iid", "noniid"):
                prefix = "dslassocov_seed_000003_%s" % split
                fingerprints = {
                    "training_fingerprint": ("%s-training" % split),
                    "test_fingerprint": ("%s-test" % split),
                    "fit_fingerprint": ("%s-fit" % split),
                }
                artifact_paths = {
                    key: root / (prefix + suffix)
                    for key, suffix in run_manager.RESULT_HASH_FIELDS.items()
                }
                for key in (
                    "detail_csv_sha256",
                    "summary_csv_sha256",
                    "model_summary_csv_sha256",
                    "auxiliary_json_sha256",
                ):
                    artifact_paths[key].write_text("%s\n" % key)
                runner.atomic_json(
                    artifact_paths["fit_json_sha256"],
                    {
                        "benchmark_id": run_manager.BENCHMARK_ID,
                        "method": run_manager.METHOD_NAME,
                        "implementation": run_manager.IMPLEMENTATION_LABEL,
                        "privacy_boundary": run_manager.PRIVACY_LABEL,
                        "seed": 3,
                        "split": split,
                        "variant": "control",
                        "training_fingerprint": fingerprints[
                            "training_fingerprint"
                        ],
                        "test_fingerprint": fingerprints["test_fingerprint"],
                        "evaluation": {
                            "primary_metric": "permute/tau_intervention_logit",
                            "uses_test_for_hyperparameter_selection": False,
                            "intervention_scope": "global",
                        },
                        "fit": {
                            "fit_fingerprint": fingerprints["fit_fingerprint"]
                        },
                    },
                )
                result_fields = [
                    ("format_version", 2),
                    ("benchmark_id", run_manager.BENCHMARK_ID),
                    ("method_name", run_manager.METHOD_NAME),
                    ("implementation_label", run_manager.IMPLEMENTATION_LABEL),
                    ("privacy_label", run_manager.PRIVACY_LABEL),
                    ("seed", 3),
                    ("split", split),
                    ("run_id", manifest["run_id"]),
                    ("effect_modifier_variant", "control"),
                    ("provenance_sha256", manifest["provenance_sha256"]),
                    ("run_config_sha256", manifest["run_config_sha256"]),
                    ("variant_config_sha256", manifest["variant_config_sha256"]),
                    (
                        "benchmark_config_sha256",
                        manifest["benchmark_config_sha256"],
                    ),
                    ("training_fingerprint", fingerprints["training_fingerprint"]),
                    ("test_fingerprint", fingerprints["test_fingerprint"]),
                    ("fit_fingerprint", fingerprints["fit_fingerprint"]),
                    ("selected_lambda", "0.1"),
                    ("selected_cv_folds_converged", 1),
                    ("selected_final_fit_converged", 1),
                    ("selected_final_proximal_gradient_mapping_inf", "1e-8"),
                    ("selected_final_kkt_residual_inf", "1e-8"),
                    ("selected_final_stationarity_threshold", "1e-6"),
                    ("primary_metric", "permute/tau_intervention_logit"),
                    ("primary_top2_hits", 2),
                    ("primary_top2_precision", 1.0),
                    ("primary_auroc", 1.0),
                    ("site_summary_status", "not_applicable_global"),
                    ("site_summary_csv_sha256", "none"),
                ]
                result_fields.extend(
                    (key, runner.sha256_file(path))
                    for key, path in artifact_paths.items()
                )
                result_path = root / (prefix + "_result.env")
                runner.atomic_env(result_path, result_fields)
                results[split] = (result_path, fingerprints)

            done_dir = root / "completed_seeds"
            done_dir.mkdir()
            marker_fields = [
                ("format_version", 2),
                ("seed", 3),
                ("run_id", manifest["run_id"]),
                ("effect_modifier_variant", "control"),
                ("provenance_sha256", manifest["provenance_sha256"]),
                ("run_config_sha256", manifest["run_config_sha256"]),
                ("variant_config_sha256", manifest["variant_config_sha256"]),
                ("benchmark_config_sha256", manifest["benchmark_config_sha256"]),
            ]
            for split, (result_path, fingerprints) in results.items():
                marker_fields.append(
                    ("%s_result_env_sha256" % split, runner.sha256_file(result_path))
                )
                marker_fields.extend(
                    ("%s_%s" % (split, key), value)
                    for key, value in fingerprints.items()
                )
            marker = run_manager.marker_path(root, 3)
            runner.atomic_env(marker, marker_fields)
            run_manager.validate_marker(root, manifest, "control", 3)
            marker.write_text(
                marker.read_text().replace(
                    "iid_fit_fingerprint=iid-fit",
                    "iid_fit_fingerprint=tampered",
                )
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                run_manager.validate_marker(root, manifest, "control", 3)
    def _new_lock_run(self, label):
        CPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
        self.assertFalse(CPU_LOCK_ROOT.is_symlink())
        self.assertEqual(CPU_LOCK_ROOT.resolve(), CPU_LOCK_ROOT)
        run_id = "dslassocov_control_%s_%d_%d" % (
            label,
            os.getpid(),
            time.time_ns(),
        )
        lock_dir = CPU_LOCK_ROOT / run_id
        lock_dir.mkdir()
        return run_id, lock_dir

    def _wait_for_path(self, path, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(0.02)
        self.fail("timed out waiting for %s" % path)

    def test_two_processes_compete_for_one_seed_without_mixed_completion(self):
        TEST_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        run_id, lock_dir = self._new_lock_run("seed_lock_race")
        holder = None
        try:
            with tempfile.TemporaryDirectory(dir=TEST_RUNTIME_ROOT) as directory:
                root = Path(directory)
                entry = root / "runner_entries"
                ready = root / "holder_ready"
                result = root / "fake_result"
                done = root / "fake_done"
                holder_script = r"""
set -euo pipefail
source "$1"
dslassocov_acquire_seed_lock "$2" 0 "holder"
trap 'dslassocov_release_cpu_lock "holder"' EXIT
printf 'holder=%s\n' "$$" >> "$3"
touch "$4"
sleep 0.8
printf 'owner=%s\n' "$$" > "$5"
printf 'owner=%s\n' "$$" > "$6"
dslassocov_release_cpu_lock "holder"
trap - EXIT
"""
                holder = subprocess.Popen(
                    [
                        "bash",
                        "-c",
                        holder_script,
                        "bash",
                        str(CPU_LOCK_HELPER),
                        run_id,
                        str(entry),
                        str(ready),
                        str(result),
                        str(done),
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._wait_for_path(ready)
                contender_script = r"""
set -uo pipefail
source "$1"
if dslassocov_acquire_seed_lock "$2" 0 "contender"; then
    printf 'contender=%s\n' "$$" >> "$3"
    dslassocov_release_cpu_lock "contender"
    exit 0
else
    code=$?
    exit "${code}"
fi
"""
                contender = subprocess.run(
                    [
                        "bash",
                        "-c",
                        contender_script,
                        "bash",
                        str(CPU_LOCK_HELPER),
                        run_id,
                        str(entry),
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(contender.returncode, 73, contender.stderr)
                holder_stdout, holder_stderr = holder.communicate(timeout=5)
                self.assertEqual(holder.returncode, 0, holder_stdout + holder_stderr)
                holder = None
                entries = entry.read_text().splitlines()
                self.assertEqual(len(entries), 1)
                self.assertTrue(entries[0].startswith("holder="))
                self.assertEqual(result.read_text(), done.read_text())

                reacquire = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; dslassocov_acquire_seed_lock "$2" 0 "reacquire"; '
                        'dslassocov_release_cpu_lock "reacquire"',
                        "bash",
                        str(CPU_LOCK_HELPER),
                        run_id,
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(reacquire.returncode, 0, reacquire.stderr)
        finally:
            if holder is not None and holder.poll() is None:
                holder.terminate()
                holder.wait(timeout=5)
            shutil.rmtree(lock_dir, ignore_errors=True)

    def test_multiple_seed_release_does_not_leak_descriptors(self):
        TEST_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        run_id, lock_dir = self._new_lock_run("multi_seed_release")
        owner = None
        try:
            with tempfile.TemporaryDirectory(dir=TEST_RUNTIME_ROOT) as directory:
                ready = Path(directory) / "released_all"
                owner_script = r"""
set -euo pipefail
source "$1"
for seed in 4 5; do
    dslassocov_acquire_seed_lock "$2" "$seed" "sequential-owner"
    dslassocov_release_cpu_lock "sequential-owner"
done
touch "$3"
sleep 0.8
"""
                owner = subprocess.Popen(
                    [
                        "bash",
                        "-c",
                        owner_script,
                        "bash",
                        str(CPU_LOCK_HELPER),
                        run_id,
                        str(ready),
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._wait_for_path(ready)
                competitor = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'set -euo pipefail; source "$1"; '
                        'for seed in 4 5; do '
                        'dslassocov_acquire_seed_lock "$2" "$seed" "post-release"; '
                        'dslassocov_release_cpu_lock "post-release"; done',
                        "bash",
                        str(CPU_LOCK_HELPER),
                        run_id,
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(competitor.returncode, 0, competitor.stderr)
                owner_stdout, owner_stderr = owner.communicate(timeout=5)
                self.assertEqual(owner.returncode, 0, owner_stdout + owner_stderr)
                owner = None
                for seed in (4, 5):
                    events = (lock_dir / ("seed_%d.lock" % seed)).read_text()
                    self.assertEqual(events.count("event=acquired"), 2)
                    self.assertEqual(events.count("event=released"), 2)
        finally:
            if owner is not None and owner.poll() is None:
                owner.terminate()
                owner.wait(timeout=5)
            shutil.rmtree(lock_dir, ignore_errors=True)

    def test_screen_child_does_not_inherit_launch_lock_fd(self):
        self.assertIsNotNone(shutil.which("screen"))
        TEST_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        run_id, lock_dir = self._new_lock_run("launch_fd")
        session = "dslassocov_locktest_%d_%d" % (os.getpid(), time.time_ns())
        try:
            with tempfile.TemporaryDirectory(dir=TEST_RUNTIME_ROOT) as directory:
                ready = Path(directory) / "screen_child_ready"
                launch_script = r"""
set -euo pipefail
source "$1"
dslassocov_acquire_launch_lock "$2" "test-launcher"
screen -dmS "$3" bash -c 'touch "$1"; sleep 1' bash "$4" {DSLASSOCOV_CPU_LOCK_FD}>&-
dslassocov_release_cpu_lock "test-launcher"
"""
                launched = subprocess.run(
                    [
                        "bash",
                        "-c",
                        launch_script,
                        "bash",
                        str(CPU_LOCK_HELPER),
                        run_id,
                        session,
                        str(ready),
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self._wait_for_path(ready)
                active = subprocess.run(
                    ["screen", "-ls"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                ).stdout
                self.assertIn(session, active)
                reacquire = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; dslassocov_acquire_launch_lock "$2" "second-launcher"; '
                        'dslassocov_release_cpu_lock "second-launcher"',
                        "bash",
                        str(CPU_LOCK_HELPER),
                        run_id,
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(reacquire.returncode, 0, reacquire.stderr)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    listing = subprocess.run(
                        ["screen", "-ls"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    ).stdout
                    if session not in listing:
                        break
                    time.sleep(0.05)
                self.assertNotIn(session, listing)
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)


    def test_cpu_scripts_do_not_touch_gpu_or_gpu_locks(self):
        project_root = Path(__file__).resolve().parents[1]
        script_names = (
            "launch_effect_modifier_dslassocov_cpu.sh",
            "run_effect_modifier_dslassocov_cpu_worker.sh",
            "status_effect_modifier_dslassocov_cpu.sh",
            "effect_modifier_dslassocov_cpu_lock.sh",
        )
        forbidden = ("CUDA_VISIBLE_DEVICES", "nvidia-smi", "gpu_", "/locks/")
        for name in script_names:
            text = (project_root / "scripts" / name).read_text()
            for token in forbidden:
                self.assertNotIn(token, text, msg="%s in %s" % (token, name))
        self.assertIn("not a deployment privacy", PRIVACY_LABEL)
        launcher_text = (
            project_root / "scripts" / "launch_effect_modifier_dslassocov_cpu.sh"
        ).read_text()
        self.assertIn("{DSLASSOCOV_CPU_LOCK_FD}>&-", launcher_text)


if __name__ == "__main__":
    unittest.main()
