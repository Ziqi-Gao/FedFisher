#!/usr/bin/env python3
"""Validate four-cell one-shot LASSO smoke or formal result families."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.dslassocov_run_contract import (
    compute_provenance,
    parse_env_file,
)
from effect_modifier_experiment.one_shot_lasso_benchmark import FAMILIES
from effect_modifier_experiment.one_shot_lasso_run_contract import (
    OUTPUT_ROOT,
    sha256_file,
    validate_manifest,
)


VARIANTS = (
    "control",
    "site_gaussian_shift",
    "local_permutation",
    "site_gaussian_local_permutation",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--control-run-id", required=True)
    parser.add_argument("--gaussian-run-id", required=True)
    parser.add_argument("--local-run-id", required=True)
    parser.add_argument("--combined-run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def expected_evaluation_rounds(variant):
    if variant in {"local_permutation", "site_gaussian_local_permutation"}:
        return "1"
    if variant in {"control", "site_gaussian_shift"}:
        return "0"
    raise ValueError("unknown effect-modifier variant")


def requires_site_summary(variant):
    return expected_evaluation_rounds(variant) == "1"


def validate_marker(root, stem, manifest, require_site_summary):
    marker = parse_env_file(root / "completed_cells" / (stem + ".done"))
    expected = {
        "format_version": "2",
        "run_id": manifest["run_id"],
        "cell": stem,
        "provenance_sha256": manifest["provenance_sha256"],
        "run_config_sha256": manifest["run_config_sha256"],
        "method_config_sha256": manifest["method_config_sha256"],
        "variant_config_sha256": manifest["variant_config_sha256"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError("completion marker mismatch for %s" % key)
    suffixes = {
        "detail_csv": "_prediction_intervention.csv",
        "summary_csv": "_prediction_intervention_summary.csv",
        "model_summary_csv": "_prediction_intervention_model_summary.csv",
        "fit_json": "_fit.json",
        "result_env": "_result.env",
    }
    site_summary_path = root / (stem + "_prediction_intervention_site_summary.csv")
    if require_site_summary:
        suffixes["site_summary_csv"] = "_prediction_intervention_site_summary.csv"
    else:
        if "site_summary_csv_sha256" in marker:
            raise ValueError("global variant marker must not contain site summary hash")
        if site_summary_path.exists() or site_summary_path.is_symlink():
            raise ValueError("global variant must not contain a site summary artifact")
    for key, suffix in suffixes.items():
        path = root / (stem + suffix)
        if marker.get(key + "_sha256") != sha256_file(path):
            raise ValueError("artifact hash mismatch for %s" % path)


def main(argv=None):
    args = parse_args(argv)
    run_ids = dict(
        zip(
            VARIANTS,
            (
                args.control_run_id,
                args.gaussian_run_id,
                args.local_run_id,
                args.combined_run_id,
            ),
        )
    )
    current_provenance = compute_provenance(PROJECT_ROOT)["provenance_sha256"]
    manifests = {}
    for variant, run_id in run_ids.items():
        root = OUTPUT_ROOT / run_id
        if root.is_symlink() or not root.is_dir():
            raise ValueError("invalid output root: %s" % root)
        manifest = parse_env_file(root / "run_manifest.env")
        validate_manifest(manifest)
        if manifest["experiment_family"] != args.family:
            raise ValueError("family mismatch")
        if manifest["effect_modifier_variant"] != variant:
            raise ValueError("variant mismatch")
        if manifest["provenance_sha256"] != current_provenance:
            raise ValueError("source provenance no longer matches smoke manifest")
        manifests[variant] = manifest
    for key in (
        "provenance_sha256",
        "run_config_sha256",
        "method_config_sha256",
    ):
        if len({manifest[key] for manifest in manifests.values()}) != 1:
            raise ValueError("four-cell manifests disagree on %s" % key)

    models = (
        ("linear",)
        if args.family == "one_shot_local_logistic_lasso_fedavg"
        else ("SyntheticMLP", "SyntheticMLPDeep")
    )
    results = {}
    validated = 0
    for variant, run_id in run_ids.items():
        root = OUTPUT_ROOT / run_id
        for split in ("iid", "noniid"):
            for model in models:
                stem = "oneshot_seed_%06d_%s_%s" % (args.seed, split, model)
                validate_marker(
                    root,
                    stem,
                    manifests[variant],
                    require_site_summary=requires_site_summary(variant),
                )
                result = parse_env_file(root / (stem + "_result.env"))
                expected = {
                    "format_version": "2",
                    "run_id": run_id,
                    "family": args.family,
                    "variant": variant,
                    "seed": str(args.seed),
                    "split": split,
                    "model": model,
                    "training_communication_rounds": "1",
                    "site_upload_count": "10",
                    "post_upload_training_site_accesses": "0",
                    "evaluation_communication_rounds": expected_evaluation_rounds(variant),
                    "primary_metric": "permute/tau_intervention_logit",
                }
                for key, value in expected.items():
                    if result.get(key) != value:
                        raise ValueError("result identity mismatch for %s" % key)
                top2 = int(result["primary_top2_hits"])
                auroc = float(result["primary_auroc"])
                if not 0 <= top2 <= 2 or not 0.0 <= auroc <= 1.0:
                    raise ValueError("primary recovery metric is invalid")
                results[(variant, split, model)] = result
                validated += 1
                print(
                    "%s %s %s Top2=%d AUROC=%.12g training_rounds=1 evaluation_rounds=%s"
                    % (
                        variant,
                        split,
                        model,
                        top2,
                        auroc,
                        result["evaluation_communication_rounds"],
                    )
                )
    for split in ("iid", "noniid"):
        for model in models:
            for left, right in (
                ("control", "local_permutation"),
                ("site_gaussian_shift", "site_gaussian_local_permutation"),
            ):
                lresult = results[(left, split, model)]
                rresult = results[(right, split, model)]
                for key in (
                    "training_fingerprint",
                    "test_fingerprint",
                    "payload_fingerprint",
                    "aggregate_model_fingerprint",
                ):
                    if lresult[key] != rresult[key]:
                        raise ValueError(
                            "paired %s/%s mismatch for %s %s %s"
                            % (left, right, split, model, key)
                        )
            if len(
                {results[(variant, split, model)]["test_fingerprint"] for variant in VARIANTS}
            ) != 1:
                raise ValueError("four variants do not share the common test set")
    print("paired_invariants=validated")
    print("validated_result_cells=%d" % validated)


if __name__ == "__main__":
    main()
