#!/usr/bin/env python3
"""Initialize, validate, complete, and report CPU dsLassoCov-style runs."""

import argparse
import json
import math
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.dslassocov_run_contract import (
    BENCHMARK_ID,
    CPU_LOCK_ROOT,
    DATA_ROOT,
    IMPLEMENTATION_LABEL,
    LOG_ROOT,
    METHOD_NAME,
    OUTPUT_ROOT,
    PRIVACY_LABEL,
    PROJECT_ROOT,
    RUNTIME_ROOT,
    SCRATCH_ROOT,
    VARIANT_CONTRACT,
    atomic_create_text,
    canonicalize_config_values,
    compute_provenance,
    config_hashes,
    env_text,
    expected_paths,
    parse_env_file,
    sha256_file,
    validate_exact_path,
    validate_run_id,
    validate_manifest_self_consistency,
)


CONFIG_ENV = {
    "seed_start": "SEED_START",
    "seed_end": "SEED_END",
    "splits": "SPLITS",
    "synthetic_alpha": "SYNTHETIC_ALPHA",
    "effect_modifier_num_sites": "EFFECT_MODIFIER_NUM_SITES",
    "synthetic_num_test": "SYNTHETIC_NUM_TEST",
    "effect_modifier_covariate_dim": "EFFECT_MODIFIER_COVARIATE_DIM",
    "effect_modifier_signal_dim": "EFFECT_MODIFIER_SIGNAL_DIM",
    "effect_modifier_signal_strength": "EFFECT_MODIFIER_SIGNAL_STRENGTH",
    "effect_modifier_main_effect_strength": "EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH",
    "effect_modifier_confounding_strength": "EFFECT_MODIFIER_CONFOUNDING_STRENGTH",
    "effect_modifier_intercept": "EFFECT_MODIFIER_INTERCEPT",
    "effect_modifier_treatment_prob": "EFFECT_MODIFIER_TREATMENT_PROB",
    "effect_modifier_propensity_clip": "EFFECT_MODIFIER_PROPENSITY_CLIP",
    "effect_modifier_site_min_samples": "EFFECT_MODIFIER_SITE_MIN_SAMPLES",
    "effect_modifier_site_max_samples": "EFFECT_MODIFIER_SITE_MAX_SAMPLES",
    "effect_modifier_site_gaussian_mean_span": "EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN",
    "effect_modifier_site_gaussian_std_low": "EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW",
    "effect_modifier_site_gaussian_std_high": "EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH",
    "prediction_intervention_modes": "PREDICTION_INTERVENTION_MODES",
    "prediction_intervention_repeats": "PREDICTION_INTERVENTION_REPEATS",
    "prediction_intervention_batch_size": "PREDICTION_INTERVENTION_BATCH_SIZE",
    "cv_folds": "CV_FOLDS",
    "cv_seed_offset": "CV_SEED_OFFSET",
    "lambda_count": "LAMBDA_COUNT",
    "lambda_min_ratio": "LAMBDA_MIN_RATIO",
    "standardization_scale_floor": "STANDARDIZATION_SCALE_FLOOR",
    "optimizer_max_iter": "OPTIMIZER_MAX_ITER",
    "optimizer_tolerance": "OPTIMIZER_TOLERANCE",
    "optimizer_initial_step": "OPTIMIZER_INITIAL_STEP",
    "optimizer_backtrack_factor": "OPTIMIZER_BACKTRACK_FACTOR",
    "optimizer_min_step": "OPTIMIZER_MIN_STEP",
    "threads_per_worker": "THREADS_PER_WORKER",
    "cpu_workers": "CPU_WORKERS",
}

FIXED_BENCHMARK_CONFIG = {
    "benchmark_id": BENCHMARK_ID,
    "method_name": METHOD_NAME,
    "implementation_label": IMPLEMENTATION_LABEL,
    "privacy_label": PRIVACY_LABEL,
    "lambda_selection_rule": "minimum_weighted_validation_loss_ties_largest_lambda",
    "optimizer_algorithm": "fista_proximal_gradient_adaptive_restart_backtracking",
    "standardization_rule": "training_only_site_count_sum_sumsq",
    "unpenalized_block": "intercept,A,raw_X_main_effects",
    "penalized_block": "A_by_raw_X_interactions_only",
}

RESULT_HASH_FIELDS = {
    "detail_csv_sha256": "_prediction_intervention.csv",
    "summary_csv_sha256": "_prediction_intervention_summary.csv",
    "model_summary_csv_sha256": "_prediction_intervention_model_summary.csv",
    "fit_json_sha256": "_fit.json",
    "auxiliary_json_sha256": "_interaction_coefficient_auxiliary.json",
}



def _integer(values, key, minimum=None):
    try:
        value = int(values[key])
    except (KeyError, ValueError) as error:
        raise ValueError("%s must be an integer" % key) from error
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %d" % (key, minimum))
    return value


def _float(values, key, lower=None, upper=None, strict_lower=False, strict_upper=False):
    try:
        value = float(values[key])
    except (KeyError, ValueError) as error:
        raise ValueError("%s must be numeric" % key) from error
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % key)
    if lower is not None and (value <= lower if strict_lower else value < lower):
        operator = ">" if strict_lower else ">="
        raise ValueError("%s must be %s %s" % (key, operator, lower))
    if upper is not None and (value >= upper if strict_upper else value > upper):
        operator = "<" if strict_upper else "<="
        raise ValueError("%s must be %s %s" % (key, operator, upper))
    return value


def normalize_config(values):
    """Return the unique text representation hashed into manifest-v2."""
    return canonicalize_config_values(values)


def validate_config(values):
    seed_start = _integer(values, "seed_start", 0)
    seed_end = _integer(values, "seed_end", 0)
    if seed_end < seed_start:
        raise ValueError("seed_end must be >= seed_start")
    if values.get("splits") != "iid,noniid":
        raise ValueError("splits must be exactly iid,noniid")
    _float(values, "synthetic_alpha", 0.0, strict_lower=True)
    num_sites = _integer(values, "effect_modifier_num_sites", 1)
    if _integer(values, "synthetic_num_test", 1) < num_sites:
        raise ValueError("synthetic_num_test must be at least the number of sites")
    covariate_dim = _integer(values, "effect_modifier_covariate_dim", 2)
    signal_dim = _integer(values, "effect_modifier_signal_dim", 1)
    if signal_dim != 2 or covariate_dim < signal_dim:
        raise ValueError("this benchmark requires exactly X1/X2 as signals")
    for key in (
        "effect_modifier_signal_strength",
        "effect_modifier_main_effect_strength",
        "effect_modifier_confounding_strength",
    ):
        _float(values, key, 0.0)
    _float(values, "effect_modifier_intercept")
    _float(
        values,
        "effect_modifier_treatment_prob",
        0.0,
        1.0,
        strict_lower=True,
        strict_upper=True,
    )
    _float(
        values,
        "effect_modifier_propensity_clip",
        0.0,
        0.5,
        strict_upper=True,
    )
    site_min = _integer(values, "effect_modifier_site_min_samples", 1)
    site_max = _integer(values, "effect_modifier_site_max_samples", 1)
    if site_max < site_min:
        raise ValueError("site max samples must be >= site min samples")
    _float(values, "effect_modifier_site_gaussian_mean_span", 0.0)
    gaussian_low = _float(
        values, "effect_modifier_site_gaussian_std_low", 0.0, strict_lower=True
    )
    gaussian_high = _float(
        values, "effect_modifier_site_gaussian_std_high", 0.0, strict_lower=True
    )
    if gaussian_high < gaussian_low:
        raise ValueError("site Gaussian std high must be >= std low")
    modes = values.get("prediction_intervention_modes", "").split(",")
    if not modes or any(mode not in {"permute", "zero"} for mode in modes):
        raise ValueError("prediction intervention modes must contain permute/zero")
    if len(modes) != len(set(modes)):
        raise ValueError("prediction intervention modes may not repeat")
    _integer(values, "prediction_intervention_repeats", 1)
    _integer(values, "prediction_intervention_batch_size", 1)
    cv_folds = _integer(values, "cv_folds", 2)
    if cv_folds > site_min:
        raise ValueError("cv_folds may not exceed site_min_samples")
    _integer(values, "cv_seed_offset", 0)
    _integer(values, "lambda_count", 2)
    _float(values, "lambda_min_ratio", 0.0, 1.0, strict_lower=True, strict_upper=True)
    _float(values, "standardization_scale_floor", 0.0, strict_lower=True)
    _integer(values, "optimizer_max_iter", 1)
    _float(values, "optimizer_tolerance", 0.0, strict_lower=True)
    _float(values, "optimizer_initial_step", 0.0, strict_lower=True)
    _float(
        values,
        "optimizer_backtrack_factor",
        0.0,
        1.0,
        strict_lower=True,
        strict_upper=True,
    )
    min_step = _float(values, "optimizer_min_step", 0.0, strict_lower=True)
    if min_step >= float(values["optimizer_initial_step"]):
        raise ValueError("optimizer_min_step must be below optimizer_initial_step")
    threads = _integer(values, "threads_per_worker", 1)
    workers = _integer(values, "cpu_workers", 1)
    if threads > 4 or workers > 4:
        raise ValueError("CPU workers and threads per worker must each be <= 4")
    if workers * threads > 4:
        raise ValueError("cpu_workers * threads_per_worker must be <= 4")


def config_from_environment(variant):
    missing = [env_name for env_name in CONFIG_ENV.values() if env_name not in os.environ]
    if missing:
        raise ValueError("missing configuration environment variables: %s" % ",".join(missing))
    values = {
        key: os.environ[env_name]
        for key, env_name in CONFIG_ENV.items()
    }
    values.update(FIXED_BENCHMARK_CONFIG)
    values.update(VARIANT_CONTRACT[variant])
    values.pop("run_id_prefix", None)
    validate_config(values)
    values = normalize_config(values)
    validate_config(values)
    return values


def _validate_tree_has_no_symlinks(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("directory may not be a symlink: %s" % path)
    if path.exists():
        unsafe = next((entry for entry in path.rglob("*") if entry.is_symlink()), None)
        if unsafe is not None:
            raise ValueError("tree may not contain symlinks: %s" % unsafe)


def _paths(run_id):
    paths = expected_paths(run_id)
    paths["output_dir"] = validate_exact_path(
        paths["output_dir"], OUTPUT_ROOT / run_id, DATA_ROOT, "output_dir"
    )
    paths["log_dir"] = validate_exact_path(
        paths["log_dir"], LOG_ROOT / run_id, SCRATCH_ROOT, "log_dir"
    )
    paths["runtime_dir"] = validate_exact_path(
        paths["runtime_dir"], RUNTIME_ROOT / run_id, SCRATCH_ROOT, "runtime_dir"
    )
    paths["lock_dir"] = validate_exact_path(
        paths["lock_dir"], CPU_LOCK_ROOT / run_id, SCRATCH_ROOT, "lock_dir"
    )
    return paths


def expected_manifest(run_id, variant, values, provenance):
    fields = {
        "format_version": "2",
        "run_id": run_id,
        "effect_modifier_variant": variant,
    }
    fields.update(provenance)
    fields.update(config_hashes(values))
    fields.update(values)
    return fields


def validate_manifest_fields(manifest, expected):
    for key, expected_value in expected.items():
        if manifest.get(key) != str(expected_value):
            raise ValueError(
                "run manifest mismatch for %s: expected %r, found %r"
                % (key, str(expected_value), manifest.get(key))
            )


def initialize_run(run_id, variant):
    validate_run_id(run_id, variant)
    values = config_from_environment(variant)
    paths = _paths(run_id)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        _validate_tree_has_no_symlinks(path)
    manifest_path = paths["output_dir"] / "run_manifest.env"
    provenance = compute_provenance(PROJECT_ROOT)
    expected = expected_manifest(run_id, variant, values, provenance)
    if not manifest_path.exists() and not manifest_path.is_symlink():
        unexpected = next(paths["output_dir"].iterdir(), None)
        if unexpected is not None:
            raise ValueError(
                "refusing nonempty output directory without run_manifest.env: %s"
                % paths["output_dir"]
            )
        fields = list(expected.items()) + [
            ("created_at", datetime.now(timezone.utc).isoformat())
        ]
        atomic_create_text(manifest_path, env_text(fields))
    manifest = parse_env_file(manifest_path)
    validate_manifest_fields(manifest, expected)
    stored_values = validate_manifest_self_consistency(manifest, run_id, variant)
    validate_config(stored_values)
    for path in paths.values():
        _validate_tree_has_no_symlinks(path)
    return paths, manifest


def validate_current_run(run_id, variant):
    validate_run_id(run_id, variant)
    values = config_from_environment(variant)
    paths = _paths(run_id)
    manifest = parse_env_file(paths["output_dir"] / "run_manifest.env")
    provenance = compute_provenance(PROJECT_ROOT)
    expected = expected_manifest(run_id, variant, values, provenance)
    validate_manifest_fields(manifest, expected)
    stored_values = validate_manifest_self_consistency(manifest, run_id, variant)
    validate_config(stored_values)
    for path in paths.values():
        _validate_tree_has_no_symlinks(path)
    return paths, manifest


def _result_prefix(seed, split):
    return "dslassocov_seed_%06d_%s" % (seed, split)


def validate_result(output_dir, manifest, variant, seed, split):
    prefix = _result_prefix(seed, split)
    result_path = output_dir / (prefix + "_result.env")
    result = parse_env_file(result_path)
    expected = {
        "format_version": "2",
        "benchmark_id": BENCHMARK_ID,
        "method_name": METHOD_NAME,
        "implementation_label": IMPLEMENTATION_LABEL,
        "privacy_label": PRIVACY_LABEL,
        "seed": str(seed),
        "split": split,
        "run_id": manifest["run_id"],
        "effect_modifier_variant": variant,
        "provenance_sha256": manifest["provenance_sha256"],
        "run_config_sha256": manifest["run_config_sha256"],
        "variant_config_sha256": manifest["variant_config_sha256"],
        "benchmark_config_sha256": manifest["benchmark_config_sha256"],
        "selected_cv_folds_converged": "1",
        "selected_final_fit_converged": "1",
        "primary_metric": "permute/tau_intervention_logit",
    }
    validate_manifest_fields(result, expected)
    threshold = float(result["selected_final_stationarity_threshold"])
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("result stationarity threshold must be finite and positive")
    for key in (
        "selected_final_proximal_gradient_mapping_inf",
        "selected_final_kkt_residual_inf",
    ):
        residual = float(result[key])
        if not math.isfinite(residual) or residual < 0.0 or residual > threshold:
            raise ValueError("%s exceeds stationarity threshold: %s" % (key, result_path))
    selected_lambda = float(result["selected_lambda"])
    if not math.isfinite(selected_lambda) or selected_lambda <= 0.0:
        raise ValueError("selected lambda must be finite and positive")
    top2_hits = int(result["primary_top2_hits"])
    top2_precision = float(result["primary_top2_precision"])
    auroc = float(result["primary_auroc"])
    if (
        top2_hits < 0
        or top2_hits > 2
        or not math.isfinite(top2_precision)
        or top2_precision < 0.0
        or top2_precision > 1.0
        or not math.isfinite(auroc)
        or auroc < 0.0
        or auroc > 1.0
    ):
        raise ValueError("primary Top-2/AUROC is outside its valid range")
    for hash_field, suffix in RESULT_HASH_FIELDS.items():
        artifact = output_dir / (prefix + suffix)
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("result artifact is missing or unsafe: %s" % artifact)
        if sha256_file(artifact) != result[hash_field]:
            raise ValueError("result artifact hash mismatch: %s" % artifact)
    fit_path = output_dir / (prefix + "_fit.json")
    with fit_path.open("r", encoding="utf-8") as handle:
        fit_document = json.load(handle)
    expected_fit = {
        "benchmark_id": BENCHMARK_ID,
        "method": METHOD_NAME,
        "implementation": IMPLEMENTATION_LABEL,
        "privacy_boundary": PRIVACY_LABEL,
        "seed": seed,
        "split": split,
        "variant": variant,
        "training_fingerprint": result["training_fingerprint"],
        "test_fingerprint": result["test_fingerprint"],
    }
    for key, expected_value in expected_fit.items():
        if fit_document.get(key) != expected_value:
            raise ValueError("fit metadata mismatch for %s: %s" % (key, fit_path))
    evaluation = fit_document.get("evaluation", {})
    if evaluation.get("primary_metric") != "permute/tau_intervention_logit":
        raise ValueError("fit metadata does not identify the primary metric")
    if evaluation.get("uses_test_for_hyperparameter_selection") is not False:
        raise ValueError("fit metadata must confirm training-only hyperparameter selection")
    expected_scope = VARIANT_CONTRACT[variant]["prediction_intervention_scope"]
    if evaluation.get("intervention_scope") != expected_scope:
        raise ValueError("fit metadata has the wrong intervention scope")
    if fit_document.get("fit", {}).get("fit_fingerprint") != result["fit_fingerprint"]:
        raise ValueError("fit fingerprint does not match result metadata")
    site_path = output_dir / (prefix + "_prediction_intervention_site_summary.csv")
    site_hash = result["site_summary_csv_sha256"]
    if VARIANT_CONTRACT[variant]["prediction_intervention_scope"] == "site":
        if site_path.is_symlink() or not site_path.is_file():
            raise ValueError("site-local variant requires a site summary: %s" % site_path)
        if sha256_file(site_path) != site_hash:
            raise ValueError("site summary hash mismatch: %s" % site_path)
        if result.get("site_summary_status") != "written_site_local":
            raise ValueError("site-local result has the wrong site-summary status")
    else:
        if site_path.exists() or site_path.is_symlink() or site_hash != "none":
            raise ValueError("global variant must not have a site-summary artifact")
        if result.get("site_summary_status") != "not_applicable_global":
            raise ValueError("global result has the wrong site-summary status")
    return result_path, result


def marker_path(output_dir, seed):
    return output_dir / "completed_seeds" / ("seed_%d.done" % seed)


def validate_marker(output_dir, manifest, variant, seed):
    marker = marker_path(output_dir, seed)
    values = parse_env_file(marker)
    expected = {
        "format_version": "2",
        "seed": str(seed),
        "run_id": manifest["run_id"],
        "effect_modifier_variant": variant,
        "provenance_sha256": manifest["provenance_sha256"],
        "run_config_sha256": manifest["run_config_sha256"],
        "variant_config_sha256": manifest["variant_config_sha256"],
        "benchmark_config_sha256": manifest["benchmark_config_sha256"],
    }
    validate_manifest_fields(values, expected)
    for split in ("iid", "noniid"):
        result_path, result = validate_result(output_dir, manifest, variant, seed, split)
        if values.get("%s_result_env_sha256" % split) != sha256_file(result_path):
            raise ValueError("done marker result hash mismatch for %s" % split)
        for fingerprint in ("training_fingerprint", "test_fingerprint", "fit_fingerprint"):
            key = "%s_%s" % (split, fingerprint)
            if values.get(key) != result[fingerprint]:
                raise ValueError("done marker fingerprint mismatch for %s" % key)
    return marker


def partial_seed_artifacts(output_dir, seed):
    prefix = "dslassocov_seed_%06d_" % seed
    return sorted(path for path in output_dir.glob(prefix + "*"))


def complete_seed(run_id, variant, seed):
    paths, manifest = validate_current_run(run_id, variant)
    output_dir = paths["output_dir"]
    done_dir = output_dir / "completed_seeds"
    done_dir.mkdir(exist_ok=True)
    _validate_tree_has_no_symlinks(done_dir)
    marker = marker_path(output_dir, seed)
    if marker.exists() or marker.is_symlink():
        validate_marker(output_dir, manifest, variant, seed)
        return marker
    fields = [
        ("format_version", 2),
        ("seed", seed),
        ("run_id", run_id),
        ("effect_modifier_variant", variant),
        ("provenance_sha256", manifest["provenance_sha256"]),
        ("run_config_sha256", manifest["run_config_sha256"]),
        ("variant_config_sha256", manifest["variant_config_sha256"]),
        ("benchmark_config_sha256", manifest["benchmark_config_sha256"]),
    ]
    for split in ("iid", "noniid"):
        result_path, result = validate_result(output_dir, manifest, variant, seed, split)
        fields.append(("%s_result_env_sha256" % split, sha256_file(result_path)))
        for fingerprint in ("training_fingerprint", "test_fingerprint", "fit_fingerprint"):
            fields.append(("%s_%s" % (split, fingerprint), result[fingerprint]))
        fields.append(("%s_primary_top2_hits" % split, result["primary_top2_hits"]))
        fields.append(("%s_primary_auroc" % split, result["primary_auroc"]))
    fields.append(("completed_at", datetime.now(timezone.utc).isoformat()))
    atomic_create_text(marker, env_text(fields))
    validate_marker(output_dir, manifest, variant, seed)
    return marker


def seed_status(run_id, variant, seed):
    validate_run_id(run_id, variant)
    paths = _paths(run_id)
    manifest = parse_env_file(paths["output_dir"] / "run_manifest.env")
    stored_values = validate_manifest_self_consistency(manifest, run_id, variant)
    validate_config(stored_values)
    marker = marker_path(paths["output_dir"], seed)
    if marker.exists() or marker.is_symlink():
        validate_marker(paths["output_dir"], manifest, variant, seed)
        print("seed_status=validated_complete")
        return 0
    partial = partial_seed_artifacts(paths["output_dir"], seed)
    if partial:
        print("seed_status=partial_refuse_overwrite")
        for path in partial:
            print("partial_artifact=%s" % path)
        return 2
    print("seed_status=missing")
    return 3


def status(run_id, variant):
    validate_run_id(run_id, variant)
    paths = _paths(run_id)
    manifest_path = paths["output_dir"] / "run_manifest.env"
    if not manifest_path.exists():
        print("manifest_status=missing")
        print("validated_completed_seeds=0")
        print("output_dir=%s" % paths["output_dir"])
        print("log_dir=%s" % paths["log_dir"])
        return 0
    manifest = parse_env_file(manifest_path)
    stored_values = validate_manifest_self_consistency(manifest, run_id, variant)
    validate_config(stored_values)
    current = compute_provenance(PROJECT_ROOT)
    source_matches = int(current["provenance_sha256"] == manifest["provenance_sha256"])
    seed_start = int(manifest["seed_start"])
    seed_end = int(manifest["seed_end"])
    completed = 0
    invalid = 0
    for seed in range(seed_start, seed_end + 1):
        marker = marker_path(paths["output_dir"], seed)
        if not marker.exists() and not marker.is_symlink():
            continue
        try:
            validate_marker(paths["output_dir"], manifest, variant, seed)
            completed += 1
        except (OSError, ValueError):
            invalid += 1
    print("manifest_status=validated_v2")
    print("source_matches_manifest=%d" % source_matches)
    print("provenance_sha256=%s" % manifest["provenance_sha256"])
    print("run_config_sha256=%s" % manifest["run_config_sha256"])
    print("benchmark_config_sha256=%s" % manifest["benchmark_config_sha256"])
    print("variant_config_sha256=%s" % manifest["variant_config_sha256"])
    print("configured_seed_range=%d..%d" % (seed_start, seed_end))
    print("validated_completed_seeds=%d" % completed)
    print("invalid_completed_markers=%d" % invalid)
    print("output_dir=%s" % paths["output_dir"])
    print("log_dir=%s" % paths["log_dir"])
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("init", "validate-current", "seed-status", "complete-seed", "status"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_CONTRACT), required=True)
    parser.add_argument("--seed", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command in {"seed-status", "complete-seed"} and (
        args.seed is None or args.seed < 0
    ):
        raise ValueError("%s requires a nonnegative --seed" % args.command)
    if args.command == "init":
        paths, manifest = initialize_run(args.run_id, args.variant)
        print("manifest_status=validated_v2")
        print("provenance_sha256=%s" % manifest["provenance_sha256"])
        print("run_config_sha256=%s" % manifest["run_config_sha256"])
        print("benchmark_config_sha256=%s" % manifest["benchmark_config_sha256"])
        print("variant_config_sha256=%s" % manifest["variant_config_sha256"])
        print("output_dir=%s" % paths["output_dir"])
        print("log_dir=%s" % paths["log_dir"])
    elif args.command == "validate-current":
        validate_current_run(args.run_id, args.variant)
        print("current_run_validation=ok")
    elif args.command == "seed-status":
        raise SystemExit(seed_status(args.run_id, args.variant, args.seed))
    elif args.command == "complete-seed":
        print("done_marker=%s" % complete_seed(args.run_id, args.variant, args.seed))
    else:
        raise SystemExit(status(args.run_id, args.variant))


if __name__ == "__main__":
    main()
