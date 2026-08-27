"""Manifest-v2 and path contract for isolated one-shot LASSO benchmarks."""

import hashlib
import json
import os
from pathlib import Path
import re

from effect_modifier_experiment.dslassocov_run_contract import (
    atomic_create_text,
    compute_provenance,
    env_text,
    parse_env_file,
)
from effect_modifier_experiment.one_shot_lasso_benchmark import (
    FAMILIES,
    VARIANT_CONTRACT,
)


PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")
DATA_ROOT = Path("/data/del6500/FedFisher")
SCRATCH_ROOT = Path("/scr/del6500/FedFisher")
OUTPUT_ROOT = DATA_ROOT / "effect_modifier_experiment" / "outputs"
RUNTIME_ROOT = SCRATCH_ROOT / "effect_modifier_experiment" / "runtime"
LOG_ROOT = SCRATCH_ROOT / "effect_modifier_experiment" / "logs"

FORMAT_VERSION = "2"
PRIVACY_LABEL = (
    "single-process cross-silo simulation; not a deployment privacy or "
    "DataSHIELD disclosure-control implementation"
)
METHOD_NAMES = {
    "one_shot_local_logistic_lasso_fedavg": (
        "one-shot site-local logistic LASSO with FedAvg coefficient averaging"
    ),
    "one_shot_mlp_group_lasso_fedavg": (
        "one-shot site-local MLP explicit A-by-X input group regularization with FedAvg parameter averaging"
    ),
}


def canonical_json(values):
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hashes(run_config, method_config, variant_config):
    return {
        "run_config_sha256": sha256_text(canonical_json(run_config)),
        "method_config_sha256": sha256_text(canonical_json(method_config)),
        "variant_config_sha256": sha256_text(canonical_json(variant_config)),
    }


def validate_run_id(run_id, family, variant):
    if family not in FAMILIES:
        raise ValueError("unknown one-shot family")
    if variant not in VARIANT_CONTRACT:
        raise ValueError("unknown effect-modifier variant")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id or ""):
        raise ValueError("RUN_ID contains unsupported characters")
    expected = "%s_%s_" % (family, variant)
    if not run_id.startswith(expected):
        raise ValueError("RUN_ID must start with %s" % expected)


def expected_output_dir(run_id):
    return OUTPUT_ROOT / run_id


def _validate_exact_output_path(output_dir, run_id):
    output_dir = Path(output_dir)
    expected = expected_output_dir(run_id)
    if not output_dir.is_absolute() or output_dir.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError("output_dir must resolve exactly to %s" % expected)
    root = OUTPUT_ROOT.resolve(strict=True)
    current = output_dir
    while current != root:
        if current.exists() and current.is_symlink():
            raise ValueError("output path may not contain symlinks: %s" % current)
        if root not in current.resolve(strict=False).parents:
            raise ValueError("output path escapes approved root")
        current = current.parent
    return output_dir


def _manifest_fields(run_id, family, variant, run_config, method_config):
    validate_run_id(run_id, family, variant)
    variant_config = {
        "effect_modifier_site_gaussian_shift": bool(
            VARIANT_CONTRACT[variant]["site_gaussian_shift"]
        ),
        "prediction_intervention_scope": VARIANT_CONTRACT[variant][
            "intervention_scope"
        ],
    }
    provenance = compute_provenance(PROJECT_ROOT)
    hashes = config_hashes(run_config, method_config, variant_config)
    fields = {
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
        "experiment_family": family,
        "method_name": METHOD_NAMES[family],
        "privacy_label": PRIVACY_LABEL,
        "effect_modifier_variant": variant,
        "training_communication_rounds": "1",
        "post_upload_training_site_access_allowed": "0",
        "run_config_json": canonical_json(run_config),
        "method_config_json": canonical_json(method_config),
        "variant_config_json": canonical_json(variant_config),
    }
    fields.update(provenance)
    fields.update(hashes)
    return fields


def _ordered_manifest_fields(fields):
    order = (
        "format_version",
        "run_id",
        "experiment_family",
        "method_name",
        "privacy_label",
        "effect_modifier_variant",
        "training_communication_rounds",
        "post_upload_training_site_access_allowed",
        "git_commit",
        "git_dirty",
        "git_status_sha256",
        "source_tree_sha256",
        "provenance_sha256",
        "run_config_sha256",
        "method_config_sha256",
        "variant_config_sha256",
        "run_config_json",
        "method_config_json",
        "variant_config_json",
    )
    return [(key, fields[key]) for key in order]


def initialize_or_validate_run(
    output_dir,
    run_id,
    family,
    variant,
    run_config,
    method_config,
):
    output_dir = _validate_exact_output_path(output_dir, run_id)
    expected_fields = _manifest_fields(
        run_id, family, variant, run_config, method_config
    )
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("output path must be a non-symlink directory")
        unsafe = next((path for path in output_dir.rglob("*") if path.is_symlink()), None)
        if unsafe is not None:
            raise ValueError("output tree contains a symlink: %s" % unsafe)
        manifest_path = output_dir / "run_manifest.env"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("existing output directory lacks a valid manifest")
        manifest = parse_env_file(manifest_path)
        validate_manifest(manifest, expected_fields)
        return manifest

    output_dir.mkdir(mode=0o700, parents=False)
    try:
        atomic_create_text(
            output_dir / "run_manifest.env",
            env_text(_ordered_manifest_fields(expected_fields)),
        )
        (output_dir / "completed_cells").mkdir(mode=0o700)
    except Exception:
        # A newly-created directory is left fail-closed for audit.  Never reuse
        # it without a valid manifest.
        raise
    return expected_fields


def validate_manifest(manifest, expected_fields=None):
    required_identity = {
        "format_version": FORMAT_VERSION,
        "training_communication_rounds": "1",
        "post_upload_training_site_access_allowed": "0",
    }
    for key, value in required_identity.items():
        if manifest.get(key) != value:
            raise ValueError("manifest identity mismatch for %s" % key)
    family = manifest.get("experiment_family", "")
    variant = manifest.get("effect_modifier_variant", "")
    run_id = manifest.get("run_id", "")
    validate_run_id(run_id, family, variant)
    if manifest.get("method_name") != METHOD_NAMES[family]:
        raise ValueError("manifest method name mismatch")
    if manifest.get("privacy_label") != PRIVACY_LABEL:
        raise ValueError("manifest privacy label mismatch")
    try:
        run_config = json.loads(manifest["run_config_json"])
        method_config = json.loads(manifest["method_config_json"])
        variant_config = json.loads(manifest["variant_config_json"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("manifest configuration JSON is invalid") from error
    if canonical_json(run_config) != manifest["run_config_json"]:
        raise ValueError("run config is not canonical")
    if canonical_json(method_config) != manifest["method_config_json"]:
        raise ValueError("method config is not canonical")
    if canonical_json(variant_config) != manifest["variant_config_json"]:
        raise ValueError("variant config is not canonical")
    contract = VARIANT_CONTRACT[variant]
    expected_variant = {
        "effect_modifier_site_gaussian_shift": bool(contract["site_gaussian_shift"]),
        "prediction_intervention_scope": contract["intervention_scope"],
    }
    if variant_config != expected_variant:
        raise ValueError("manifest variant contract mismatch")
    for key, value in config_hashes(run_config, method_config, variant_config).items():
        if manifest.get(key) != value:
            raise ValueError("manifest config hash mismatch for %s" % key)
    if expected_fields is not None:
        for key, value in expected_fields.items():
            if manifest.get(key) != str(value):
                raise ValueError("manifest current-source/config mismatch for %s" % key)
    return run_config, method_config, variant_config


def atomic_json(path, value):
    text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    atomic_create_text(path, text)


def atomic_csv_text(path, text):
    atomic_create_text(path, text)


def complete_cell(output_dir, stem, manifest, artifact_hashes):
    marker = Path(output_dir) / "completed_cells" / (stem + ".done")
    fields = [
        ("format_version", FORMAT_VERSION),
        ("run_id", manifest["run_id"]),
        ("cell", stem),
        ("provenance_sha256", manifest["provenance_sha256"]),
        ("run_config_sha256", manifest["run_config_sha256"]),
        ("method_config_sha256", manifest["method_config_sha256"]),
        ("variant_config_sha256", manifest["variant_config_sha256"]),
    ]
    for key in sorted(artifact_hashes):
        fields.append((key + "_sha256", artifact_hashes[key]))
    atomic_create_text(marker, env_text(fields))
    return marker


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
