#!/usr/bin/env python3
"""Run one CPU-only dsLassoCov-style effect-modifier benchmark cell.

The data generator and treatment-contrast intervention evaluator are imported
from the existing experiment.  This runner never trains from the pooled
``dataset_train_global`` object; the optimizer receives the site datasets and
aggregates site loss/gradient summaries by total sample count.
"""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import (
    build_effect_modifier_site_evaluation_datasets,
    get_dataset,
    get_effect_modifier_site_gaussian_parameters,
)
from effect_modifier_experiment.dslassocov_benchmark import (
    IMPLEMENTATION_LABEL,
    METHOD_NAME,
    FederatedLogisticLassoModel,
    OptimizerConfig,
    VARIANT_CONTRACT,
    dataset_fingerprint,
    federated_logistic_lasso_cv,
    get_variant_contract,
    interaction_coefficient_recovery,
    json_safe,
    tensor_datasets_to_raw_sites,
    training_fingerprint,
)
from effect_modifier_experiment.dslassocov_run_contract import (
    BENCHMARK_CONFIG_KEYS,
    BENCHMARK_ID,
    IMPLEMENTATION_LABEL as CONTRACT_IMPLEMENTATION_LABEL,
    METHOD_NAME as CONTRACT_METHOD_NAME,
    OUTPUT_ROOT,
    PRIVACY_LABEL,
    RUN_CONFIG_KEYS,
    VARIANT_CONFIG_KEYS,
    VARIANT_CONTRACT as RUN_VARIANT_CONTRACT,
    compute_provenance,
    config_hashes,
    parse_env_file,
    validate_manifest_self_consistency,
)
from utils.prediction_intervention import (
    add_model_prediction_intervention,
    write_prediction_intervention_outputs,
)


DATA_OUTPUT_ROOT = OUTPUT_ROOT
MODEL_SOURCE = "dslassocov_style_federated"
if METHOD_NAME != CONTRACT_METHOD_NAME or IMPLEMENTATION_LABEL != CONTRACT_IMPLEMENTATION_LABEL:
    raise RuntimeError("benchmark method labels disagree with the run contract")


def parse_env_manifest(path):
    return parse_env_file(path)


def validate_output_and_manifest(output_dir, run_id, variant):
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_id):
        raise ValueError("run_id contains unsupported characters")
    expected = (DATA_OUTPUT_ROOT.resolve(strict=True) / run_id).resolve(strict=False)
    output_dir = Path(output_dir)
    if output_dir.resolve(strict=False) != expected:
        raise ValueError("output_dir must resolve exactly to %s" % expected)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("output_dir must be an initialized non-symlink directory")
    unsafe_symlink = next((path for path in output_dir.rglob("*") if path.is_symlink()), None)
    if unsafe_symlink is not None:
        raise ValueError("output tree may not contain symlinks: %s" % unsafe_symlink)

    manifest_path = output_dir / "run_manifest.env"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("run_manifest.env must be a regular non-symlink file")
    manifest = parse_env_manifest(manifest_path)
    expected_fields = {
        "format_version": "2",
        "run_id": run_id,
        "effect_modifier_variant": variant,
        "benchmark_id": BENCHMARK_ID,
        "method_name": METHOD_NAME,
        "implementation_label": IMPLEMENTATION_LABEL,
    }
    contract = get_variant_contract(variant)
    expected_fields.update(
        {
            "effect_modifier_site_gaussian_shift": str(
                int(contract["site_gaussian_shift"])
            ),
            "prediction_intervention_scope": str(contract["intervention_scope"]),
        }
    )
    for key, expected_value in expected_fields.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                "manifest mismatch for %s: expected %r, found %r"
                % (key, expected_value, manifest.get(key))
            )
    for required_hash in (
        "git_status_sha256",
        "source_tree_sha256",
        "provenance_sha256",
        "run_config_sha256",
        "variant_config_sha256",
        "benchmark_config_sha256",
    ):
        value = manifest.get(required_hash, "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("manifest %s must be a lowercase SHA256" % required_hash)
    validate_manifest_self_consistency(manifest, run_id, variant)
    current_provenance = compute_provenance(PROJECT_ROOT)
    for key, value in current_provenance.items():
        if manifest.get(key) != value:
            raise ValueError("current source provenance differs for %s" % key)
    return output_dir, manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_CONTRACT), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("iid", "noniid"), required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num-sites", type=int, default=10)
    parser.add_argument("--num-test", type=int, default=10000)
    parser.add_argument("--covariate-dim", type=int, default=100)
    parser.add_argument("--signal-dim", type=int, default=2)
    parser.add_argument("--signal-strength", type=float, default=2.0)
    parser.add_argument("--main-effect-strength", type=float, default=1.0)
    parser.add_argument("--confounding-strength", type=float, default=1.0)
    parser.add_argument("--outcome-intercept", type=float, default=0.0)
    parser.add_argument("--treatment-prob", type=float, default=0.5)
    parser.add_argument("--propensity-clip", type=float, default=0.05)
    parser.add_argument("--site-min-samples", type=int, default=500)
    parser.add_argument("--site-max-samples", type=int, default=1000)
    parser.add_argument("--site-gaussian-mean-span", type=float, default=0.75)
    parser.add_argument("--site-gaussian-std-low", type=float, default=0.75)
    parser.add_argument("--site-gaussian-std-high", type=float, default=1.25)
    parser.add_argument("--intervention-modes", nargs="+", choices=("permute", "zero"), default=("permute", "zero"))
    parser.add_argument("--intervention-repeats", type=int, default=5)
    parser.add_argument("--intervention-batch-size", type=int, default=1024)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed-offset", type=int, default=4000037)
    parser.add_argument("--lambda-count", type=int, default=30)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-3)
    parser.add_argument("--standardization-scale-floor", type=float, default=1e-12)
    parser.add_argument("--optimizer-max-iter", type=int, default=500)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-6)
    parser.add_argument("--optimizer-initial-step", type=float, default=1.0)
    parser.add_argument("--optimizer-backtrack-factor", type=float, default=0.5)
    parser.add_argument("--optimizer-min-step", type=float, default=1e-12)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=1)
    return parser


def validate_args(args):
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.num_sites < 1 or args.num_test < args.num_sites:
        raise ValueError("num_sites must be positive and num_test must cover every site")
    if args.covariate_dim < 2 or args.signal_dim != 2:
        raise ValueError("the benchmark requires exactly X1/X2 signals and at least 2 covariates")
    if args.intervention_repeats < 1 or args.intervention_batch_size < 1:
        raise ValueError("intervention repeats and batch size must be positive")
    finite_fields = (
        "alpha",
        "signal_strength",
        "main_effect_strength",
        "confounding_strength",
        "outcome_intercept",
        "treatment_prob",
        "propensity_clip",
        "site_gaussian_mean_span",
        "site_gaussian_std_low",
        "site_gaussian_std_high",
        "lambda_min_ratio",
        "standardization_scale_floor",
        "optimizer_tolerance",
        "optimizer_initial_step",
        "optimizer_backtrack_factor",
        "optimizer_min_step",
    )
    if any(not math.isfinite(getattr(args, field)) for field in finite_fields):
        raise ValueError("all continuous benchmark arguments must be finite")
    if args.cv_seed_offset < 0:
        raise ValueError("cv_seed_offset must be nonnegative")
    if args.alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if args.site_min_samples < 1 or args.site_max_samples < args.site_min_samples:
        raise ValueError("site sample bounds must satisfy 1 <= min <= max")
    if args.cv_folds < 2 or args.cv_folds > args.site_min_samples:
        raise ValueError("cv_folds must be in [2, site_min_samples]")
    if args.lambda_count < 2:
        raise ValueError("lambda_count must be at least 2")
    if not 0.0 < args.lambda_min_ratio < 1.0:
        raise ValueError("lambda_min_ratio must lie in (0, 1)")
    if args.optimizer_max_iter < 1 or args.optimizer_tolerance <= 0.0:
        raise ValueError("optimizer max_iter and tolerance must be positive")
    if args.optimizer_initial_step <= 0.0 or args.optimizer_min_step <= 0.0:
        raise ValueError("optimizer steps must be positive")
    if args.optimizer_min_step >= args.optimizer_initial_step:
        raise ValueError("optimizer_min_step must be below optimizer_initial_step")
    if not 0.0 < args.optimizer_backtrack_factor < 1.0:
        raise ValueError("optimizer_backtrack_factor must lie in (0, 1)")
    if args.standardization_scale_floor <= 0.0:
        raise ValueError("standardization_scale_floor must be positive")
    if not 0.0 < args.treatment_prob < 1.0:
        raise ValueError("treatment_prob must lie in (0, 1)")
    if not 0.0 <= args.propensity_clip < 0.5:
        raise ValueError("propensity_clip must lie in [0, 0.5)")
    if args.site_gaussian_mean_span < 0.0:
        raise ValueError("site Gaussian mean span must be nonnegative")
    if (
        args.site_gaussian_std_low <= 0.0
        or args.site_gaussian_std_high < args.site_gaussian_std_low
    ):
        raise ValueError("site Gaussian stds must satisfy 0 < low <= high")
    if not 1 <= args.threads_per_worker <= 4:
        raise ValueError("threads_per_worker must be in [1, 4]")
    if not 1 <= args.cpu_workers <= 4:
        raise ValueError("cpu_workers must be in [1, 4]")
    if args.cpu_workers * args.threads_per_worker > 4:
        raise ValueError("cpu_workers * threads_per_worker must be <= 4")
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(env_name) != str(args.threads_per_worker):
            raise ValueError(
                "%s must equal --threads-per-worker for an auditable CPU run"
                % env_name
            )


def manifest_config_values(args, manifest):
    values = {
        "seed_start": manifest.get("seed_start", ""),
        "seed_end": manifest.get("seed_end", ""),
        "splits": "iid,noniid",
        "synthetic_alpha": str(args.alpha),
        "effect_modifier_num_sites": str(args.num_sites),
        "synthetic_num_test": str(args.num_test),
        "effect_modifier_covariate_dim": str(args.covariate_dim),
        "effect_modifier_signal_dim": str(args.signal_dim),
        "effect_modifier_signal_strength": str(args.signal_strength),
        "effect_modifier_main_effect_strength": str(args.main_effect_strength),
        "effect_modifier_confounding_strength": str(args.confounding_strength),
        "effect_modifier_intercept": str(args.outcome_intercept),
        "effect_modifier_treatment_prob": str(args.treatment_prob),
        "effect_modifier_propensity_clip": str(args.propensity_clip),
        "effect_modifier_site_min_samples": str(args.site_min_samples),
        "effect_modifier_site_max_samples": str(args.site_max_samples),
        "effect_modifier_site_gaussian_mean_span": str(args.site_gaussian_mean_span),
        "effect_modifier_site_gaussian_std_low": str(args.site_gaussian_std_low),
        "effect_modifier_site_gaussian_std_high": str(args.site_gaussian_std_high),
        "prediction_intervention_modes": ",".join(args.intervention_modes),
        "prediction_intervention_repeats": str(args.intervention_repeats),
        "prediction_intervention_batch_size": str(args.intervention_batch_size),
        "benchmark_id": BENCHMARK_ID,
        "method_name": METHOD_NAME,
        "implementation_label": IMPLEMENTATION_LABEL,
        "privacy_label": PRIVACY_LABEL,
        "cv_folds": str(args.cv_folds),
        "cv_seed_offset": str(args.cv_seed_offset),
        "lambda_count": str(args.lambda_count),
        "lambda_min_ratio": str(args.lambda_min_ratio),
        "lambda_selection_rule": "minimum_weighted_validation_loss_ties_largest_lambda",
        "optimizer_algorithm": "fista_proximal_gradient_adaptive_restart_backtracking",
        "standardization_scale_floor": str(args.standardization_scale_floor),
        "optimizer_max_iter": str(args.optimizer_max_iter),
        "optimizer_tolerance": str(args.optimizer_tolerance),
        "optimizer_initial_step": str(args.optimizer_initial_step),
        "optimizer_backtrack_factor": str(args.optimizer_backtrack_factor),
        "optimizer_min_step": str(args.optimizer_min_step),
        "standardization_rule": "training_only_site_count_sum_sumsq",
        "unpenalized_block": "intercept,A,raw_X_main_effects",
        "penalized_block": "A_by_raw_X_interactions_only",
        "threads_per_worker": str(args.threads_per_worker),
        "cpu_workers": str(args.cpu_workers),
    }
    values.update(RUN_VARIANT_CONTRACT[args.variant])
    values.pop("run_id_prefix", None)
    return values


def validate_args_against_manifest(args, manifest):
    try:
        seed_start = int(manifest["seed_start"])
        seed_end = int(manifest["seed_end"])
    except (KeyError, ValueError) as error:
        raise ValueError("manifest seed range is missing or invalid") from error
    if not seed_start <= args.seed <= seed_end:
        raise ValueError("seed lies outside the manifest seed range")
    values = manifest_config_values(args, manifest)
    for key in RUN_CONFIG_KEYS + BENCHMARK_CONFIG_KEYS + VARIANT_CONFIG_KEYS:
        if manifest.get(key) != values[key]:
            raise ValueError(
                "runner argument/config does not match manifest field %s" % key
            )
    for key, expected_hash in config_hashes(values).items():
        if manifest.get(key) != expected_hash:
            raise ValueError(
                "manifest canonical configuration hash mismatch for %s" % key
            )


def generate_datasets(args):
    contract = get_variant_contract(args.variant)
    datasets = get_dataset(
        datatype="SyntheticEffectModifier",
        n_client=args.num_sites,
        n_c=2,
        alpha=args.alpha,
        synthetic_split=args.split,
        synthetic_num_test=args.num_test,
        effect_modifier_covariate_dim=args.covariate_dim,
        effect_modifier_signal_dim=args.signal_dim,
        effect_modifier_signal_strength=args.signal_strength,
        effect_modifier_main_effect_strength=args.main_effect_strength,
        effect_modifier_confounding_strength=args.confounding_strength,
        effect_modifier_intercept=args.outcome_intercept,
        effect_modifier_treatment_prob=args.treatment_prob,
        effect_modifier_propensity_clip=args.propensity_clip,
        effect_modifier_site_min_samples=args.site_min_samples,
        effect_modifier_site_max_samples=args.site_max_samples,
        effect_modifier_site_gaussian_shift=contract["site_gaussian_shift"],
        effect_modifier_site_gaussian_mean_span=args.site_gaussian_mean_span,
        effect_modifier_site_gaussian_std_low=args.site_gaussian_std_low,
        effect_modifier_site_gaussian_std_high=args.site_gaussian_std_high,
        seed=args.seed,
    )
    site_datasets, _, test_dataset, label_counts = datasets
    return site_datasets, test_dataset, label_counts

def build_site_training_audit(args, site_datasets):
    contract = get_variant_contract(args.variant)
    if contract["site_gaussian_shift"]:
        intended_parameters = get_effect_modifier_site_gaussian_parameters(
            args.num_sites,
            mean_span=args.site_gaussian_mean_span,
            std_low=args.site_gaussian_std_low,
            std_high=args.site_gaussian_std_high,
        )
        proposal = "fixed_site_gaussian_mean_by_scale"
    else:
        intended_parameters = [(0.0, 1.0)] * args.num_sites
        proposal = "common_standard_gaussian"
    rows = []
    for site_idx, (dataset, intended) in enumerate(
        zip(site_datasets, intended_parameters)
    ):
        features = dataset.tensors[0].detach().cpu().numpy()
        labels = dataset.tensors[1].detach().cpu().numpy()
        raw_x = np.asarray(
            features[:, 1 : 1 + args.covariate_dim],
            dtype=np.float64,
        )
        label_counts = np.bincount(labels.astype(np.int64), minlength=2)
        rows.append(
            {
                "site_idx": site_idx,
                "proposal": proposal,
                "intended_raw_x_mean": float(intended[0]),
                "intended_raw_x_std": float(intended[1]),
                "num_examples": int(features.shape[0]),
                "label_0_count": int(label_counts[0]),
                "label_1_count": int(label_counts[1]),
                "treatment_1_count": int(np.count_nonzero(features[:, 0])),
                "empirical_raw_x_grand_mean": float(raw_x.mean()),
                "empirical_raw_x_grand_std": float(raw_x.std(ddof=0)),
                "empirical_raw_x_feature_means": raw_x.mean(axis=0).tolist(),
                "empirical_raw_x_feature_stds": raw_x.std(axis=0, ddof=0).tolist(),
            }
        )
    if len(rows) != args.num_sites:
        raise RuntimeError("site proposal audit did not cover every training site")
    return {
        "proposal": proposal,
        "site_gaussian_shift": bool(contract["site_gaussian_shift"]),
        "audit_scope": (
            "intended proposal plus realized site size, labels, treatment count, "
            "and empirical raw-X moments"
        ),
        "sites": rows,
    }



def fit_benchmark(args, site_datasets):
    raw_sites = tensor_datasets_to_raw_sites(site_datasets, args.covariate_dim)
    optimizer_config = OptimizerConfig(
        max_iter=args.optimizer_max_iter,
        tolerance=args.optimizer_tolerance,
        initial_step=args.optimizer_initial_step,
        backtrack_factor=args.optimizer_backtrack_factor,
        min_step=args.optimizer_min_step,
    )
    fit_result = federated_logistic_lasso_cv(
        raw_sites,
        covariate_dim=args.covariate_dim,
        num_folds=args.cv_folds,
        lambda_count=args.lambda_count,
        lambda_min_ratio=args.lambda_min_ratio,
        cv_seed=args.cv_seed_offset + args.seed,
        scale_floor=args.standardization_scale_floor,
        optimizer_config=optimizer_config,
    )
    selected_index = fit_result["cv"]["selected_index"]
    selected_cv_reports = [
        fold_report["lambda_fits"][selected_index]
        for fold_report in fit_result["cv"]["fold_reports"]
    ]
    selected_final_report = fit_result["optimizer"]["selected_lambda_fit"]
    fit_result["selected_convergence"] = {
        "all_cv_fold_fits_converged": bool(
            all(report["converged"] for report in selected_cv_reports)
        ),
        "cv_fold_proximal_gradient_mapping_inf": [
            report["proximal_gradient_mapping_inf"] for report in selected_cv_reports
        ],
        "cv_fold_kkt_residual_inf": [
            report["kkt_residual_inf"] for report in selected_cv_reports
        ],
        "final_fit_converged": bool(selected_final_report["converged"]),
        "final_fit_proximal_gradient_mapping_inf": selected_final_report[
            "proximal_gradient_mapping_inf"
        ],
        "final_fit_kkt_residual_inf": selected_final_report["kkt_residual_inf"],
        "final_fit_stationarity_threshold": selected_final_report[
            "stationarity_threshold"
        ],
    }
    if not fit_result["selected_convergence"]["all_cv_fold_fits_converged"]:
        raise RuntimeError("selected lambda did not converge in every CV training fold")
    if not fit_result["selected_convergence"]["final_fit_converged"]:
        raise RuntimeError("selected final full-training fit did not converge")
    return raw_sites, fit_result


def evaluate_benchmark(args, site_datasets, test_dataset, fit_result):
    contract = get_variant_contract(args.variant)
    model = FederatedLogisticLassoModel(
        fit_result["original_intercept"], fit_result["original_coefficients"]
    ).to("cpu")
    evaluation_args = {
        "device": "cpu",
        "bs": args.intervention_batch_size,
        "dataset": "SyntheticEffectModifier",
        "effect_modifier_covariate_dim": args.covariate_dim,
        "prediction_intervention_batch_size": args.intervention_batch_size,
    }
    metadata = {
        "seed": args.seed,
        "model": METHOD_NAME,
        "split": args.split,
        "alpha": args.alpha,
        "synthetic_dim": 1 + 2 * args.covariate_dim,
        "synthetic_signal_dim": args.signal_dim,
    }
    site_evaluation_datasets = None
    if contract["intervention_scope"] == "site":
        site_evaluation_datasets = build_effect_modifier_site_evaluation_datasets(
            test_dataset,
            site_datasets,
            split=args.split,
            alpha=args.alpha,
            seed=args.seed,
        )

    detail_rows = []
    summary_rows = []
    model_summary_rows = []
    site_summary_rows = []
    add_model_prediction_intervention(
        detail_rows,
        summary_rows,
        model_summary_rows,
        MODEL_SOURCE,
        model,
        test_dataset,
        evaluation_args,
        metadata,
        modes=args.intervention_modes,
        repeats=args.intervention_repeats,
        seed=args.seed + sum(ord(character) for character in MODEL_SOURCE),
        permutation_scope=contract["intervention_scope"],
        site_datasets=site_evaluation_datasets,
        site_summary_rows=site_summary_rows,
    )
    return detail_rows, summary_rows, model_summary_rows, site_summary_rows


def _temporary_path(path):
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("temporary output already exists: %s" % temporary)
    if path.is_symlink():
        raise ValueError("refusing to replace output symlink: %s" % path)
    return temporary


def atomic_json(path, value):
    temporary = _temporary_path(path)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_prediction_outputs(
    detail_rows,
    summary_rows,
    model_summary_rows,
    site_summary_rows,
    paths,
):
    temporary_paths = {key: _temporary_path(path) for key, path in paths.items()}
    try:
        write_prediction_intervention_outputs(
            detail_rows,
            summary_rows,
            model_summary_rows,
            temporary_paths["detail"],
            temporary_paths["summary"],
            temporary_paths["model_summary"],
            site_summary_rows=site_summary_rows,
            site_summary_path=temporary_paths["site_summary"],
        )
        required = ("detail", "summary", "model_summary")
        for key in required:
            if not temporary_paths[key].is_file():
                raise RuntimeError("evaluation did not produce %s output" % key)
        for key in required:
            os.link(temporary_paths[key], paths[key])
            temporary_paths[key].unlink()
        if site_summary_rows:
            if not temporary_paths["site_summary"].is_file():
                raise RuntimeError("site-local evaluation did not produce site summary")
            os.link(temporary_paths["site_summary"], paths["site_summary"])
            temporary_paths["site_summary"].unlink()
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()


def atomic_env(path, fields):
    temporary = _temporary_path(path)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for key, value in fields:
                text = str(value)
                if not key or "=" in key or "\n" in key or "\n" in text:
                    raise ValueError("env metadata keys/values must be one line")
                handle.write("%s=%s\n" % (key, text))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _primary_recovery_row(summary_rows):
    rows = [
        row
        for row in summary_rows
        if row["model_source"] == MODEL_SOURCE
        and row["intervention_mode"] == "permute"
        and row["metric"] == "tau_intervention_logit"
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "expected exactly one primary permute/tau_intervention_logit row"
        )
    return rows[0]


def _refuse_existing_outputs(paths):
    existing = [
        path
        for path in paths
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing/partial benchmark outputs: %s"
            % ", ".join(str(path) for path in existing)
        )


def _configure_torch_threads(threads_per_worker):
    """Apply the fixed CPU thread contract safely across repeated cells."""
    torch.set_num_threads(threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def run(args):
    validate_args(args)
    _configure_torch_threads(args.threads_per_worker)
    output_dir, manifest = validate_output_and_manifest(
        args.output_dir, args.run_id, args.variant
    )
    validate_args_against_manifest(args, manifest)

    prefix = "dslassocov_seed_%06d_%s" % (args.seed, args.split)
    paths = {
        "detail": output_dir / (prefix + "_prediction_intervention.csv"),
        "summary": output_dir / (prefix + "_prediction_intervention_summary.csv"),
        "model_summary": output_dir
        / (prefix + "_prediction_intervention_model_summary.csv"),
        "site_summary": output_dir
        / (prefix + "_prediction_intervention_site_summary.csv"),
    }
    fit_path = output_dir / (prefix + "_fit.json")
    auxiliary_path = output_dir / (prefix + "_interaction_coefficient_auxiliary.json")
    result_path = output_dir / (prefix + "_result.env")
    _refuse_existing_outputs(
        list(paths.values()) + [fit_path, auxiliary_path, result_path]
    )

    site_datasets, test_dataset, label_counts = generate_datasets(args)
    site_training_audit = build_site_training_audit(args, site_datasets)
    raw_sites, fit_result = fit_benchmark(args, site_datasets)
    training_hash = training_fingerprint(raw_sites)
    test_hash = dataset_fingerprint(test_dataset)
    detail_rows, summary_rows, model_summary_rows, site_summary_rows = (
        evaluate_benchmark(args, site_datasets, test_dataset, fit_result)
    )
    primary = _primary_recovery_row(summary_rows)
    scope = get_variant_contract(args.variant)["intervention_scope"]
    if scope == "site" and not site_summary_rows:
        raise RuntimeError("site-local evaluation must produce site summary rows")
    if scope == "global" and site_summary_rows:
        raise RuntimeError("global evaluation must not produce site summary rows")

    atomic_prediction_outputs(
        detail_rows,
        summary_rows,
        model_summary_rows,
        site_summary_rows,
        paths,
    )
    atomic_json(
        fit_path,
        {
            "benchmark_id": BENCHMARK_ID,
            "method": METHOD_NAME,
            "implementation": IMPLEMENTATION_LABEL,
            "privacy_boundary": PRIVACY_LABEL,
            "method_scope": (
                "mathematical objective and site-aggregation simulation only; "
                "not official software and not a deployment privacy claim"
            ),
            "seed": args.seed,
            "split": args.split,
            "variant": args.variant,
            "training_fingerprint": training_hash,
            "test_fingerprint": test_hash,
            "site_label_counts": label_counts,
            "site_training_proposal_audit": site_training_audit,
            "evaluation": {
                "primary_metric": "permute/tau_intervention_logit",
                "primary_candidate_block": "100 raw covariates",
                "intervention_scope": scope,
                "intervention_seed": args.seed
                + sum(ord(character) for character in MODEL_SOURCE),
                "uses_test_for_hyperparameter_selection": False,
            },
            "fit": fit_result,
        },
    )
    atomic_json(
        auxiliary_path,
        interaction_coefficient_recovery(
            fit_result, args.covariate_dim, args.signal_dim
        ),
    )

    site_summary_hash = "none"
    site_summary_status = "not_applicable_global"
    if site_summary_rows:
        site_summary_hash = sha256_file(paths["site_summary"])
        site_summary_status = "written_site_local"
    selected = fit_result["selected_convergence"]
    result_fields = [
        ("format_version", 2),
        ("benchmark_id", BENCHMARK_ID),
        ("method_name", METHOD_NAME),
        ("implementation_label", IMPLEMENTATION_LABEL),
        ("privacy_label", PRIVACY_LABEL),
        ("seed", args.seed),
        ("split", args.split),
        ("run_id", args.run_id),
        ("effect_modifier_variant", args.variant),
        ("provenance_sha256", manifest["provenance_sha256"]),
        ("run_config_sha256", manifest["run_config_sha256"]),
        ("variant_config_sha256", manifest["variant_config_sha256"]),
        ("benchmark_config_sha256", manifest["benchmark_config_sha256"]),
        ("training_fingerprint", training_hash),
        ("test_fingerprint", test_hash),
        ("fit_fingerprint", fit_result["fit_fingerprint"]),
        ("selected_lambda", fit_result["cv"]["selected_lambda"]),
        ("selected_cv_folds_converged", 1),
        ("selected_final_fit_converged", 1),
        (
            "selected_final_proximal_gradient_mapping_inf",
            selected["final_fit_proximal_gradient_mapping_inf"],
        ),
        ("selected_final_kkt_residual_inf", selected["final_fit_kkt_residual_inf"]),
        (
            "selected_final_stationarity_threshold",
            selected["final_fit_stationarity_threshold"],
        ),
        ("primary_metric", "permute/tau_intervention_logit"),
        ("primary_top2_hits", primary["topk_hits"]),
        ("primary_top2_precision", primary["topk_precision"]),
        ("primary_auroc", primary["auroc"]),
        ("site_summary_status", site_summary_status),
        ("detail_csv_sha256", sha256_file(paths["detail"])),
        ("summary_csv_sha256", sha256_file(paths["summary"])),
        ("model_summary_csv_sha256", sha256_file(paths["model_summary"])),
        ("site_summary_csv_sha256", site_summary_hash),
        ("fit_json_sha256", sha256_file(fit_path)),
        ("auxiliary_json_sha256", sha256_file(auxiliary_path)),
        ("completed_at", datetime.now(timezone.utc).isoformat()),
    ]
    atomic_env(result_path, result_fields)
    print("benchmark_result=%s" % result_path)
    print("fit_fingerprint=%s" % fit_result["fit_fingerprint"])
    print("selected_lambda=%s" % fit_result["cv"]["selected_lambda"])
    print(
        "primary_top2_hits=%s primary_auroc=%s"
        % (primary["topk_hits"], primary["auroc"])
    )
    return result_path


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
