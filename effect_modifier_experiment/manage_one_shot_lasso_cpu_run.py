#!/usr/bin/env python3
"""Initialize, validate, and report formal one-shot LASSO CPU runs."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.dslassocov_run_contract import (
    compute_provenance,
    parse_env_file,
)
from effect_modifier_experiment.one_shot_lasso_benchmark import (
    FAMILIES,
    VARIANT_CONTRACT,
)
from effect_modifier_experiment.one_shot_lasso_run_contract import (
    OUTPUT_ROOT,
    _manifest_fields,
    expected_output_dir,
    initialize_or_validate_run,
    sha256_file,
    validate_manifest,
)
from effect_modifier_experiment.run_one_shot_lasso_effect_modifier import (
    method_config,
    parse_args as parse_runner_args,
    run_config,
)


FORMAL_SEED_START = 0
FORMAL_SEED_END = 299
FORMAL_RUN_SUFFIX = "stationarity_polish_300seeds"
MODELS = {
    "one_shot_local_logistic_lasso_fedavg": ("linear",),
    "one_shot_mlp_group_lasso_fedavg": ("SyntheticMLP", "SyntheticMLPDeep"),
}
VARIANTS = tuple(VARIANT_CONTRACT)
MISSING_EXIT = 3
INVALID_OR_PARTIAL_EXIT = 4


def formal_run_id(family, variant):
    return "%s_%s_%s" % (family, variant, FORMAL_RUN_SUFFIX)


def formal_runner_args(family, variant, seed=0, split="iid", model=None):
    """Build the manifest-bound, full scientific configuration."""
    if family not in FAMILIES or variant not in VARIANT_CONTRACT:
        raise ValueError("unknown one-shot family or variant")
    if model is None:
        model = MODELS[family][0]
    run_id = formal_run_id(family, variant)
    output_dir = expected_output_dir(run_id)
    argv = [
        "--family", family,
        "--variant", variant,
        "--run-id", run_id,
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--seed-start", str(FORMAL_SEED_START),
        "--seed-end", str(FORMAL_SEED_END),
        "--split", split,
        "--model", model,
        "--num-sites", "10",
        "--site-min-samples", "500",
        "--site-max-samples", "1000",
        "--test-size", "10000",
        "--covariate-dim", "100",
        "--signal-dim", "2",
        "--signal-strength", "2.0",
        "--main-effect-strength", "1.0",
        "--confounding-strength", "1.0",
        "--outcome-intercept", "0.0",
        "--treatment-prob", "0.5",
        "--propensity-clip", "0.05",
        "--alpha", "0.1",
        "--gaussian-mean-span", "0.75",
        "--gaussian-std-low", "0.75",
        "--gaussian-std-high", "1.25",
        "--intervention-repeats", "5",
        "--intervention-batch-size", "1024",
        "--threads", "1",
    ]
    if family == "one_shot_local_logistic_lasso_fedavg":
        argv.extend(
            [
                "--cv-folds", "5",
                "--lambda-count", "30",
                "--lambda-min-ratio", "0.001",
                "--cv-seed-offset", "4000037",
                "--optimizer-max-iter", "2000",
                "--optimizer-polish-max-iter", "50",
                "--optimizer-tolerance", "1e-6",
                "--optimizer-initial-step", "1.0",
                "--optimizer-backtrack-factor", "0.5",
                "--optimizer-min-step", "1e-12",
                "--standardization-scale-floor", "1e-12",
            ]
        )
    else:
        argv.extend(
            [
                "--local-epochs", "30",
                "--batch-size", "64",
                "--learning-rate", "0.01",
                "--momentum", "0.9",
                "--validation-fraction", "0.2",
                "--lambda-ratios", "0,1,3,10,30,100,300",
                "--split-seed-offset", "5100037",
                "--loader-seed-offset", "5200037",
            ]
        )
    return parse_runner_args(argv)


def expected_manifest(args):
    return _manifest_fields(
        args.run_id,
        args.family,
        args.variant,
        run_config(args),
        method_config(args),
    )


def _validate_output_tree(output_dir):
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("output directory is missing or unsafe: %s" % output_dir)
    if output_dir.resolve(strict=True) != output_dir:
        raise ValueError("output directory may not resolve through a symlink")
    unsafe = next((path for path in output_dir.rglob("*") if path.is_symlink()), None)
    if unsafe is not None:
        raise ValueError("output tree contains a symlink: %s" % unsafe)


def validate_current_manifest(args):
    output_dir = expected_output_dir(args.run_id)
    _validate_output_tree(output_dir)
    manifest_path = output_dir / "run_manifest.env"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("run manifest is missing or unsafe")
    manifest = parse_env_file(manifest_path)
    validate_manifest(manifest, expected_manifest(args))
    current = compute_provenance(PROJECT_ROOT)
    for key, value in current.items():
        if manifest.get(key) != value:
            raise ValueError("manifest current provenance mismatch for %s" % key)
    return manifest


def cell_stem(seed, split, model):
    return "oneshot_seed_%06d_%s_%s" % (seed, split, model)


def expected_evaluation_rounds(variant):
    return 1 if VARIANT_CONTRACT[variant]["intervention_scope"] == "site" else 0


def _cell_paths(output_dir, stem, require_site_summary):
    suffixes = {
        "detail_csv": "_prediction_intervention.csv",
        "summary_csv": "_prediction_intervention_summary.csv",
        "model_summary_csv": "_prediction_intervention_model_summary.csv",
        "fit_json": "_fit.json",
        "result_env": "_result.env",
    }
    if require_site_summary:
        suffixes["site_summary_csv"] = "_prediction_intervention_site_summary.csv"
    return {
        key: Path(output_dir) / (stem + suffix) for key, suffix in suffixes.items()
    }


def partial_cell_paths(output_dir, stem):
    output_dir = Path(output_dir)
    paths = list(output_dir.glob(stem + "_*"))
    marker = output_dir / "completed_cells" / (stem + ".done")
    if marker.exists() or marker.is_symlink():
        paths.append(marker)
    return sorted(set(paths))


def validate_completed_cell(output_dir, manifest, args):
    """Validate one marker, every artifact hash, and semantic result fields."""
    stem = cell_stem(args.seed, args.split, args.model)
    marker_path = Path(output_dir) / "completed_cells" / (stem + ".done")
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("completion marker is missing or unsafe")
    marker = parse_env_file(marker_path)
    expected_marker = {
        "format_version": "2",
        "run_id": args.run_id,
        "cell": stem,
        "provenance_sha256": manifest["provenance_sha256"],
        "run_config_sha256": manifest["run_config_sha256"],
        "method_config_sha256": manifest["method_config_sha256"],
        "variant_config_sha256": manifest["variant_config_sha256"],
    }
    require_site = expected_evaluation_rounds(args.variant) == 1
    paths = _cell_paths(output_dir, stem, require_site)
    expected_marker_keys = set(expected_marker)
    for key, value in expected_marker.items():
        if marker.get(key) != value:
            raise ValueError("completion marker mismatch for %s" % key)
    for key, path in paths.items():
        hash_key = key + "_sha256"
        expected_marker_keys.add(hash_key)
        if path.is_symlink() or not path.is_file():
            raise ValueError("cell artifact is missing or unsafe: %s" % path)
        if marker.get(hash_key) != sha256_file(path):
            raise ValueError("cell artifact hash mismatch: %s" % path)
    if set(marker) != expected_marker_keys:
        raise ValueError("completion marker has missing or unexpected fields")
    site_path = Path(output_dir) / (stem + "_prediction_intervention_site_summary.csv")
    if not require_site and (site_path.exists() or site_path.is_symlink()):
        raise ValueError("global cell must not contain a site-summary artifact")

    result = parse_env_file(paths["result_env"])
    expected_result = {
        "format_version": "2",
        "run_id": args.run_id,
        "family": args.family,
        "variant": args.variant,
        "seed": str(args.seed),
        "split": args.split,
        "model": args.model,
        "provenance_sha256": manifest["provenance_sha256"],
        "run_config_sha256": manifest["run_config_sha256"],
        "method_config_sha256": manifest["method_config_sha256"],
        "variant_config_sha256": manifest["variant_config_sha256"],
        "training_communication_rounds": "1",
        "site_upload_count": "10",
        "post_upload_training_site_accesses": "0",
        "evaluation_communication_rounds": str(
            expected_evaluation_rounds(args.variant)
        ),
        "primary_metric": "permute/tau_intervention_logit",
    }
    for key, value in expected_result.items():
        if result.get(key) != value:
            raise ValueError("result identity mismatch for %s" % key)
    top2 = int(result["primary_top2_hits"])
    auroc = float(result["primary_auroc"])
    if not 0 <= top2 <= 2 or not 0.0 <= auroc <= 1.0:
        raise ValueError("primary metrics are outside their valid ranges")

    with paths["fit_json"].open("r", encoding="utf-8") as handle:
        fit = json.load(handle)
    if fit.get("run_id") != args.run_id or fit.get("family") != args.family:
        raise ValueError("fit record identity mismatch")
    if fit.get("variant") != args.variant or fit.get("seed") != args.seed:
        raise ValueError("fit record variant/seed mismatch")
    if fit.get("split") != args.split or fit.get("model") != args.model:
        raise ValueError("fit record split/model mismatch")
    communication = fit.get("training_communication", {})
    if communication.get("training_communication_rounds") != 1:
        raise ValueError("fit record training communication mismatch")
    if communication.get("site_upload_count") != 10:
        raise ValueError("fit record site upload count mismatch")
    if communication.get("observed_post_upload_training_site_accesses") != 0:
        raise ValueError("fit record post-upload site access mismatch")
    return result


def initialize_formal(args):
    output_dir = expected_output_dir(args.run_id)
    return initialize_or_validate_run(
        output_dir,
        args.run_id,
        args.family,
        args.variant,
        run_config(args),
        method_config(args),
    )


def cell_status(args):
    manifest = validate_current_manifest(args)
    output_dir = expected_output_dir(args.run_id)
    stem = cell_stem(args.seed, args.split, args.model)
    marker = output_dir / "completed_cells" / (stem + ".done")
    if not marker.exists() and not marker.is_symlink():
        partial = partial_cell_paths(output_dir, stem)
        if partial:
            for path in partial:
                print("partial_artifact=%s" % path)
            return INVALID_OR_PARTIAL_EXIT
        print("cell_status=missing cell=%s" % stem)
        return MISSING_EXIT
    try:
        result = validate_completed_cell(output_dir, manifest, args)
    except Exception as error:
        print("cell_status=invalid cell=%s error=%s" % (stem, error), file=sys.stderr)
        return INVALID_OR_PARTIAL_EXIT
    print(
        "cell_status=validated cell=%s Top2=%s AUROC=%s"
        % (stem, result["primary_top2_hits"], result["primary_auroc"])
    )
    return 0


def status(args):
    manifest = validate_current_manifest(args)
    output_dir = expected_output_dir(args.run_id)
    counts = {"validated": 0, "missing": 0, "invalid_or_partial": 0}
    for seed in range(FORMAL_SEED_START, FORMAL_SEED_END + 1):
        for split in ("iid", "noniid"):
            for model in MODELS[args.family]:
                cell_args = formal_runner_args(args.family, args.variant, seed, split, model)
                stem = cell_stem(seed, split, model)
                marker = output_dir / "completed_cells" / (stem + ".done")
                if not marker.exists() and not marker.is_symlink():
                    if partial_cell_paths(output_dir, stem):
                        counts["invalid_or_partial"] += 1
                    else:
                        counts["missing"] += 1
                    continue
                try:
                    validate_completed_cell(output_dir, manifest, cell_args)
                    counts["validated"] += 1
                except Exception as error:
                    counts["invalid_or_partial"] += 1
                    print("invalid_cell=%s error=%s" % (stem, error))
    print("run_id=%s" % args.run_id)
    print("family=%s variant=%s" % (args.family, args.variant))
    print("provenance_sha256=%s" % manifest["provenance_sha256"])
    print("run_config_sha256=%s" % manifest["run_config_sha256"])
    print("method_config_sha256=%s" % manifest["method_config_sha256"])
    print("validated_cells=%d" % counts["validated"])
    print("missing_cells=%d" % counts["missing"])
    print("invalid_or_partial_cells=%d" % counts["invalid_or_partial"])
    return 0 if counts["invalid_or_partial"] == 0 else INVALID_OR_PARTIAL_EXIT


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "validate-current", "cell-status", "status"))
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("iid", "noniid"), default="iid")
    parser.add_argument("--model", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    args = formal_runner_args(cli.family, cli.variant, cli.seed, cli.split, cli.model)
    if cli.command == "init":
        manifest = initialize_formal(args)
        print("initialized_or_validated_run=%s" % args.run_id)
        print("provenance_sha256=%s" % manifest["provenance_sha256"])
        print("run_config_sha256=%s" % manifest["run_config_sha256"])
        print("method_config_sha256=%s" % manifest["method_config_sha256"])
        return 0
    if cli.command == "validate-current":
        manifest = validate_current_manifest(args)
        print("validated_current_run=%s" % args.run_id)
        print("provenance_sha256=%s" % manifest["provenance_sha256"])
        return 0
    if cli.command == "cell-status":
        return cell_status(args)
    return status(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("one_shot_cpu_manager_error=%s" % error, file=sys.stderr)
        raise SystemExit(INVALID_OR_PARTIAL_EXIT)
