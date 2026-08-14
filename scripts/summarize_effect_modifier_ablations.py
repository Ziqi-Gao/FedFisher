#!/usr/bin/env python3
"""Aggregate and pair the two SyntheticEffectModifier ablations with control."""

import argparse
import csv
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path


RECOVERY_FIELDS = [
    "topk_hits",
    "topk_precision",
    "mean_signal_rank",
    "median_signal_rank",
    "mean_noise_rank",
    "auroc",
]
MODEL_FIELDS = [
    "num_examples",
    "baseline_accuracy",
    "predicted_positive_rate",
    "mean_class1_probability",
    "mean_abs_logit_score",
]
PAIR_KEY_FIELDS = [
    "seed",
    "model",
    "split",
    "alpha",
    "model_source",
    "intervention_mode",
    "metric",
]
HIGHER_IS_BETTER = {"topk_hits", "topk_precision", "mean_noise_rank", "auroc"}
DATA_ROOT = Path("/data/del6500/FedFisher")
EXPERIMENT_OUTPUT_ROOT = DATA_ROOT / "effect_modifier_experiment" / "outputs"
RESULTS_ROOT = DATA_ROOT / "results"
VARIANT_CONTRACT = {
    "control": {
        "run_id_prefix": "prediction_intervention_",
        "effect_modifier_site_gaussian_shift": "0",
        "prediction_intervention_scope": "global",
    },
    "site_gaussian_shift": {
        "run_id_prefix": "site_gaussian_shift_",
        "effect_modifier_site_gaussian_shift": "1",
        "prediction_intervention_scope": "global",
    },
    "local_permutation": {
        "run_id_prefix": "local_permutation_",
        "effect_modifier_site_gaussian_shift": "0",
        "prediction_intervention_scope": "site",
    },
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def resolve_within(path, allowed_root, label, require_directory=False):
    resolved_data_root = DATA_ROOT.resolve(strict=True)
    relative_root = allowed_root.relative_to(DATA_ROOT)
    resolved_root = resolved_data_root / relative_root
    resolved = path.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(
            "%s resolves outside %s: %s" % (label, resolved_root, resolved)
        )
    if require_directory and not resolved.is_dir():
        raise ValueError("%s is not an existing directory: %s" % (label, resolved))
    return resolved


def read_manifest(directory, expected_variant):
    manifest_path = directory / "run_manifest.env"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            "%s requires a regular non-symlink run_manifest.env: %s"
            % (expected_variant, manifest_path)
        )

    metadata = {}
    for line in manifest_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in metadata:
            raise ValueError("Malformed or duplicate manifest entry in %s" % manifest_path)
        metadata[key] = value

    required = {
        "format_version",
        "run_id",
        "effect_modifier_variant",
        "provenance_sha256",
        "run_config_sha256",
        "variant_config_sha256",
        "seed_start",
        "seed_end",
        "effect_modifier_site_gaussian_shift",
        "prediction_intervention_scope",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError("Manifest %s is missing: %s" % (manifest_path, ", ".join(missing)))
    if metadata["format_version"] != "2":
        raise ValueError("Manifest %s must use format_version=2" % manifest_path)
    if metadata["effect_modifier_variant"] != expected_variant:
        raise ValueError("Manifest variant mismatch in %s" % manifest_path)
    if metadata["run_id"] != directory.name:
        raise ValueError("Manifest run_id must match its output directory: %s" % manifest_path)

    contract = VARIANT_CONTRACT[expected_variant]
    if not metadata["run_id"].startswith(contract["run_id_prefix"]):
        raise ValueError("Manifest RUN_ID namespace mismatch in %s" % manifest_path)
    for field in [
        "effect_modifier_site_gaussian_shift",
        "prediction_intervention_scope",
    ]:
        if metadata[field] != contract[field]:
            raise ValueError("Manifest %s violates the %s contract" % (manifest_path, field))

    for field in ["provenance_sha256", "run_config_sha256", "variant_config_sha256"]:
        if not SHA256_PATTERN.fullmatch(metadata[field]):
            raise ValueError("Manifest %s has an invalid %s" % (manifest_path, field))
    variant_text = (
        "effect_modifier_site_gaussian_shift=%s\n"
        "prediction_intervention_scope=%s\n"
        % (
            metadata["effect_modifier_site_gaussian_shift"],
            metadata["prediction_intervention_scope"],
        )
    )
    expected_variant_sha = hashlib.sha256(variant_text.encode()).hexdigest()
    if metadata["variant_config_sha256"] != expected_variant_sha:
        raise ValueError("Manifest %s has an inconsistent variant_config_sha256" % manifest_path)

    for field in ["seed_start", "seed_end"]:
        if not metadata[field].isdigit():
            raise ValueError("Manifest %s has a non-numeric %s" % (manifest_path, field))
    if int(metadata["seed_end"]) < int(metadata["seed_start"]):
        raise ValueError("Manifest %s has an invalid seed range" % manifest_path)
    return metadata


def validate_comparison_manifests(inputs):
    manifests = {
        experiment: read_manifest(directory, experiment)
        for experiment, directory in inputs.items()
    }
    for field in ["provenance_sha256", "run_config_sha256"]:
        values = {metadata[field] for metadata in manifests.values()}
        if len(values) != 1:
            raise ValueError(
                "Refusing incomparable experiments: %s differs across manifests" % field
            )
    return manifests


def prepare_output_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    resolved = resolve_within(path, RESULTS_ROOT, "output directory")
    if resolved != path:
        raise ValueError("Output directory changed after creation: %s" % path)
    unsafe_symlink = next((entry for entry in path.rglob("*") if entry.is_symlink()), None)
    if unsafe_symlink is not None:
        raise ValueError("Output tree may not contain symlinks: %s" % unsafe_symlink)


def parse_args():
    experiment_root = DATA_ROOT / "effect_modifier_experiment"
    parser = argparse.ArgumentParser(
        description="Summarize effect-modifier control and one-change ablations."
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=experiment_root / "outputs" / "prediction_intervention_ablation_control_300seeds",
    )
    parser.add_argument(
        "--site-gaussian-dir",
        type=Path,
        default=experiment_root / "outputs" / "site_gaussian_shift_300seeds",
    )
    parser.add_argument(
        "--local-permutation-dir",
        type=Path,
        default=experiment_root / "outputs" / "local_permutation_300seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "results" / "effect_modifier_ablations",
    )
    args = parser.parse_args()
    try:
        args.control_dir = resolve_within(
            args.control_dir, EXPERIMENT_OUTPUT_ROOT, "control directory", True
        )
        args.site_gaussian_dir = resolve_within(
            args.site_gaussian_dir, EXPERIMENT_OUTPUT_ROOT, "site-Gaussian directory", True
        )
        args.local_permutation_dir = resolve_within(
            args.local_permutation_dir,
            EXPERIMENT_OUTPUT_ROOT,
            "local-permutation directory",
            True,
        )
        args.output_dir = resolve_within(args.output_dir, RESULTS_ROOT, "output directory")
    except ValueError as error:
        parser.error(str(error))
    return args


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    average = mean(values)
    return math.sqrt(
        sum((value - average) ** 2 for value in values) / (len(values) - 1)
    )


def standard_error(values):
    return sample_std(values) / math.sqrt(len(values))


def read_csv_rows(directory, pattern, experiment):
    rows = []
    if not directory.exists():
        return rows
    for path in sorted(directory.rglob(pattern)):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                normalized = dict(row)
                normalized["experiment"] = experiment
                normalized["permutation_scope"] = row.get(
                    "permutation_scope",
                    "site" if experiment == "local_permutation" else "global",
                )
                rows.append(normalized)
    return rows


def numeric(row, field):
    value = row.get(field, "")
    if value == "":
        return float("nan")
    return float(value)


def aggregate(rows, group_fields, value_fields):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        for field in value_fields:
            value = numeric(row, field)
            if math.isfinite(value):
                buckets[key][field].append(value)

    output = []
    for key in sorted(buckets):
        result = dict(zip(group_fields, key))
        counts = [len(buckets[key][field]) for field in value_fields]
        result["n"] = max(counts) if counts else 0
        for field in value_fields:
            values = buckets[key][field]
            result[field + "_mean"] = mean(values) if values else float("nan")
            result[field + "_std"] = sample_std(values) if values else float("nan")
            result[field + "_se"] = standard_error(values) if values else float("nan")
        output.append(result)
    return output


def unique_by(rows, fields, label):
    indexed = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in indexed:
            raise ValueError("Duplicate %s row for key %r" % (label, key))
        indexed[key] = row
    return indexed


def build_paired_rows(control_rows, ablation_rows):
    control_index = unique_by(control_rows, PAIR_KEY_FIELDS, "control")
    unique_by(ablation_rows, PAIR_KEY_FIELDS, "ablation")

    paired = []
    for row in ablation_rows:
        key = tuple(row.get(field, "") for field in PAIR_KEY_FIELDS)
        control = control_index.get(key)
        if control is None:
            continue
        result = {
            "experiment": row["experiment"],
            "permutation_scope": row["permutation_scope"],
        }
        result.update({field: row.get(field, "") for field in PAIR_KEY_FIELDS})
        for field in RECOVERY_FIELDS:
            control_value = numeric(control, field)
            ablation_value = numeric(row, field)
            raw_delta = ablation_value - control_value
            improvement_delta = (
                raw_delta if field in HIGHER_IS_BETTER else -raw_delta
            )
            result["control_" + field] = control_value
            result["ablation_" + field] = ablation_value
            result["raw_delta_" + field] = raw_delta
            result["improvement_delta_" + field] = improvement_delta
        paired.append(result)
    return paired


def aggregate_paired_rows(paired_rows):
    group_fields = [
        "experiment",
        "permutation_scope",
        "model",
        "split",
        "model_source",
        "intervention_mode",
        "metric",
    ]
    buckets = defaultdict(list)
    for row in paired_rows:
        key = tuple(row.get(field, "") for field in group_fields)
        buckets[key].append(row)

    output = []
    for key in sorted(buckets):
        rows = buckets[key]
        result = dict(zip(group_fields, key))
        result["n_pairs"] = len(rows)
        for field in RECOVERY_FIELDS:
            for prefix in [
                "control_",
                "ablation_",
                "raw_delta_",
                "improvement_delta_",
            ]:
                output_field = prefix + field
                values = [float(row[output_field]) for row in rows]
                result[output_field + "_mean"] = mean(values)
                result[output_field + "_std"] = sample_std(values)
                result[output_field + "_se"] = standard_error(values)
        output.append(result)
    return output


def write_csv(path, rows, fields):
    if path.is_symlink():
        raise ValueError("Refusing to overwrite symlink: %s" % path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    serialized[field] = "%.10g" % value
                else:
                    serialized[field] = value
            writer.writerow(serialized)


def aggregate_fields(group_fields, value_fields):
    fields = list(group_fields) + ["n"]
    for field in value_fields:
        fields.extend([field + "_mean", field + "_std", field + "_se"])
    return fields


def paired_fields():
    fields = ["experiment", "permutation_scope"] + PAIR_KEY_FIELDS
    for field in RECOVERY_FIELDS:
        fields.extend(
            [
                "control_" + field,
                "ablation_" + field,
                "raw_delta_" + field,
                "improvement_delta_" + field,
            ]
        )
    return fields


def paired_summary_fields():
    fields = [
        "experiment",
        "permutation_scope",
        "model",
        "split",
        "model_source",
        "intervention_mode",
        "metric",
        "n_pairs",
    ]
    for field in RECOVERY_FIELDS:
        for prefix in [
            "control_",
            "ablation_",
            "raw_delta_",
            "improvement_delta_",
        ]:
            fields.extend(
                [
                    prefix + field + "_mean",
                    prefix + field + "_std",
                    prefix + field + "_se",
                ]
            )
    return fields


def main():
    args = parse_args()
    inputs = {
        "control": args.control_dir,
        "site_gaussian_shift": args.site_gaussian_dir,
        "local_permutation": args.local_permutation_dir,
    }
    manifests = validate_comparison_manifests(inputs)
    prepare_output_directory(args.output_dir)
    write_csv(
        args.output_dir / "comparison_manifests.csv",
        [
            {
                "experiment": experiment,
                "run_id": metadata["run_id"],
                "provenance_sha256": metadata["provenance_sha256"],
                "run_config_sha256": metadata["run_config_sha256"],
                "variant_config_sha256": metadata["variant_config_sha256"],
                "effect_modifier_site_gaussian_shift": metadata[
                    "effect_modifier_site_gaussian_shift"
                ],
                "prediction_intervention_scope": metadata[
                    "prediction_intervention_scope"
                ],
            }
            for experiment, metadata in manifests.items()
        ],
        [
            "experiment",
            "run_id",
            "provenance_sha256",
            "run_config_sha256",
            "variant_config_sha256",
            "effect_modifier_site_gaussian_shift",
            "prediction_intervention_scope",
        ],
    )

    recovery_rows = []
    model_rows = []
    site_rows = []
    for experiment, directory in inputs.items():
        recovery_rows.extend(
            read_csv_rows(
                directory,
                "*_prediction_intervention_summary.csv",
                experiment,
            )
        )
        model_rows.extend(
            read_csv_rows(
                directory,
                "*_prediction_intervention_model_summary.csv",
                experiment,
            )
        )
        site_rows.extend(
            read_csv_rows(
                directory,
                "*_prediction_intervention_site_summary.csv",
                experiment,
            )
        )

    recovery_group_fields = [
        "experiment",
        "permutation_scope",
        "model",
        "split",
        "model_source",
        "intervention_mode",
        "metric",
    ]
    recovery_summary = aggregate(
        recovery_rows,
        recovery_group_fields,
        RECOVERY_FIELDS,
    )
    write_csv(
        args.output_dir / "recovery_summary.csv",
        recovery_summary,
        aggregate_fields(recovery_group_fields, RECOVERY_FIELDS),
    )

    model_group_fields = [
        "experiment",
        "permutation_scope",
        "model",
        "split",
        "model_source",
    ]
    model_summary = aggregate(model_rows, model_group_fields, MODEL_FIELDS)
    write_csv(
        args.output_dir / "model_summary.csv",
        model_summary,
        aggregate_fields(model_group_fields, MODEL_FIELDS),
    )

    control_rows = [
        row
        for row in recovery_rows
        if row["experiment"] == "control"
    ]
    paired_rows = []
    for experiment in ["site_gaussian_shift", "local_permutation"]:
        paired_rows.extend(
            build_paired_rows(
                control_rows,
                [
                    row
                    for row in recovery_rows
                    if row["experiment"] == experiment
                ],
            )
        )
    write_csv(
        args.output_dir / "paired_seed_deltas.csv",
        paired_rows,
        paired_fields(),
    )
    write_csv(
        args.output_dir / "paired_delta_summary.csv",
        aggregate_paired_rows(paired_rows),
        paired_summary_fields(),
    )

    site_group_fields = [
        "experiment",
        "permutation_scope",
        "model",
        "split",
        "model_source",
        "site_idx",
        "intervention_mode",
        "metric",
    ]
    site_summary = aggregate(site_rows, site_group_fields, RECOVERY_FIELDS)
    write_csv(
        args.output_dir / "site_recovery_summary.csv",
        site_summary,
        aggregate_fields(site_group_fields, RECOVERY_FIELDS),
    )

    print(
        "Wrote summaries to %s from %d recovery rows, %d paired rows, and %d site rows"
        % (args.output_dir, len(recovery_rows), len(paired_rows), len(site_rows))
    )


if __name__ == "__main__":
    main()
