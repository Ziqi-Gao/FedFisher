#!/usr/bin/env python3
"""Validate paired training/fit/test invariants across four dsLassoCov cells."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.dslassocov_run_contract import (
    OUTPUT_ROOT,
    parse_env_file,
    validate_run_id,
    validate_manifest_self_consistency,
)
from effect_modifier_experiment.manage_dslassocov_cpu_run import (
    marker_path,
    validate_config,
    validate_marker,
)


VARIANTS = (
    "control",
    "site_gaussian_shift",
    "local_permutation",
    "site_gaussian_local_permutation",
)

DEFAULT_RUN_IDS = {
    "control": "dslassocov_control_300seeds",
    "site_gaussian_shift": "dslassocov_site_gaussian_shift_300seeds",
    "local_permutation": "dslassocov_local_permutation_300seeds",
    "site_gaussian_local_permutation": (
        "dslassocov_site_gaussian_local_permutation_300seeds"
    ),
}


def load_manifest(variant, run_id):
    validate_run_id(run_id, variant)
    directory = OUTPUT_ROOT / run_id
    if directory.resolve(strict=False) != (OUTPUT_ROOT.resolve(strict=True) / run_id):
        raise ValueError("run directory escapes the required output root")
    manifest = parse_env_file(directory / "run_manifest.env")
    stored_values = validate_manifest_self_consistency(manifest, run_id, variant)
    validate_config(stored_values)
    return directory, manifest


def result_env(directory, seed, split):
    return parse_env_file(
        directory
        / ("dslassocov_seed_%06d_%s_result.env" % (seed, split))
    )


def validate_common_manifests(manifests):
    reference = manifests["control"]
    for variant in VARIANTS[1:]:
        candidate = manifests[variant]
        for key in (
            "git_commit",
            "git_dirty",
            "git_status_sha256",
            "source_tree_sha256",
            "provenance_sha256",
            "run_config_sha256",
            "benchmark_config_sha256",
            "seed_start",
            "seed_end",
        ):
            if candidate.get(key) != reference.get(key):
                raise ValueError("%s differs across manifests for %s" % (key, variant))


def validate_pair(left, right, pair_name, seed, split):
    for key in ("training_fingerprint", "test_fingerprint", "fit_fingerprint"):
        if left[key] != right[key]:
            raise ValueError(
                "%s invariant failed for seed=%d split=%s field=%s"
                % (pair_name, seed, split, key)
            )


def check(run_ids):
    directories = {}
    manifests = {}
    for variant in VARIANTS:
        directories[variant], manifests[variant] = load_manifest(
            variant, run_ids[variant]
        )
    validate_common_manifests(manifests)
    seed_start = int(manifests["control"]["seed_start"])
    seed_end = int(manifests["control"]["seed_end"])
    checked_results = 0
    metric_lines = []
    for seed in range(seed_start, seed_end + 1):
        rows = {}
        for variant in VARIANTS:
            marker = marker_path(directories[variant], seed)
            if not marker.exists():
                raise ValueError("missing done marker: %s" % marker)
            validate_marker(
                directories[variant],
                manifests[variant],
                variant,
                seed,
            )
            rows[variant] = {
                split: result_env(directories[variant], seed, split)
                for split in ("iid", "noniid")
            }
        for split in ("iid", "noniid"):
            validate_pair(
                rows["control"][split],
                rows["local_permutation"][split],
                "control/local_permutation",
                seed,
                split,
            )
            validate_pair(
                rows["site_gaussian_shift"][split],
                rows["site_gaussian_local_permutation"][split],
                "site_gaussian/combined",
                seed,
                split,
            )
            test_fingerprints = {
                rows[variant][split]["test_fingerprint"] for variant in VARIANTS
            }
            if len(test_fingerprints) != 1:
                raise ValueError(
                    "common test invariant failed for seed=%d split=%s"
                    % (seed, split)
                )
            checked_results += 4
            metric_lines.append(
                "seed=%d split=%s %s"
                % (
                    seed,
                    split,
                    " ".join(
                        "%s:Top2=%s,AUROC=%s"
                        % (
                            variant,
                            rows[variant][split]["primary_top2_hits"],
                            rows[variant][split]["primary_auroc"],
                        )
                        for variant in VARIANTS
                    ),
                )
            )
    print("paired_invariants=validated")
    print("validated_result_cells=%d" % checked_results)
    for line in metric_lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in VARIANTS:
        parser.add_argument(
            "--%s-run-id" % variant.replace("_", "-"),
            default=DEFAULT_RUN_IDS[variant],
        )
    args = parser.parse_args()
    run_ids = {
        variant: getattr(args, "%s_run_id" % variant)
        for variant in VARIANTS
    }
    check(run_ids)


if __name__ == "__main__":
    main()
