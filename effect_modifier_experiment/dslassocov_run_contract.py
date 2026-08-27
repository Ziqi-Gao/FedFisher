"""Pure-standard-library run contract for the CPU dsLassoCov-style benchmark."""

import hashlib
import math
import os
from pathlib import Path
import re
import stat
import subprocess


PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")
DATA_ROOT = Path("/data/del6500/FedFisher")
SCRATCH_ROOT = Path("/scr/del6500/FedFisher")
OUTPUT_ROOT = DATA_ROOT / "effect_modifier_experiment" / "outputs"
LOG_ROOT = SCRATCH_ROOT / "effect_modifier_experiment" / "logs"
RUNTIME_ROOT = SCRATCH_ROOT / "effect_modifier_experiment" / "runtime"
CPU_LOCK_ROOT = SCRATCH_ROOT / "effect_modifier_experiment" / "cpu_locks"

BENCHMARK_ID = "dslassocov_style_federated_logistic_lasso"
METHOD_NAME = "dsLassoCov-style federated logistic LASSO"
IMPLEMENTATION_LABEL = "Python benchmark; not official R/DataSHIELD dsLassoCov"
PRIVACY_LABEL = (
    "single-process cross-silo simulation; not a deployment privacy or "
    "DataSHIELD disclosure-control implementation"
)

VARIANT_CONTRACT = {
    "control": {
        "run_id_prefix": "dslassocov_control_",
        "effect_modifier_site_gaussian_shift": "0",
        "prediction_intervention_scope": "global",
    },
    "site_gaussian_shift": {
        "run_id_prefix": "dslassocov_site_gaussian_shift_",
        "effect_modifier_site_gaussian_shift": "1",
        "prediction_intervention_scope": "global",
    },
    "local_permutation": {
        "run_id_prefix": "dslassocov_local_permutation_",
        "effect_modifier_site_gaussian_shift": "0",
        "prediction_intervention_scope": "site",
    },
    "site_gaussian_local_permutation": {
        "run_id_prefix": "dslassocov_site_gaussian_local_permutation_",
        "effect_modifier_site_gaussian_shift": "1",
        "prediction_intervention_scope": "site",
    },
}

RUN_CONFIG_KEYS = (
    "seed_start",
    "seed_end",
    "splits",
    "synthetic_alpha",
    "effect_modifier_num_sites",
    "synthetic_num_test",
    "effect_modifier_covariate_dim",
    "effect_modifier_signal_dim",
    "effect_modifier_signal_strength",
    "effect_modifier_main_effect_strength",
    "effect_modifier_confounding_strength",
    "effect_modifier_intercept",
    "effect_modifier_treatment_prob",
    "effect_modifier_propensity_clip",
    "effect_modifier_site_min_samples",
    "effect_modifier_site_max_samples",
    "effect_modifier_site_gaussian_mean_span",
    "effect_modifier_site_gaussian_std_low",
    "effect_modifier_site_gaussian_std_high",
    "prediction_intervention_modes",
    "prediction_intervention_repeats",
    "prediction_intervention_batch_size",
)

BENCHMARK_CONFIG_KEYS = (
    "benchmark_id",
    "method_name",
    "implementation_label",
    "privacy_label",
    "cv_folds",
    "cv_seed_offset",
    "lambda_count",
    "lambda_min_ratio",
    "lambda_selection_rule",
    "optimizer_algorithm",
    "standardization_scale_floor",
    "optimizer_max_iter",
    "optimizer_tolerance",
    "optimizer_initial_step",
    "optimizer_backtrack_factor",
    "optimizer_min_step",
    "standardization_rule",
    "unpenalized_block",
    "penalized_block",
    "threads_per_worker",
    "cpu_workers",
)

VARIANT_CONFIG_KEYS = (
    "effect_modifier_site_gaussian_shift",
    "prediction_intervention_scope",
)
INTEGER_CONFIG_KEYS = (
    "seed_start",
    "seed_end",
    "effect_modifier_num_sites",
    "synthetic_num_test",
    "effect_modifier_covariate_dim",
    "effect_modifier_signal_dim",
    "effect_modifier_site_min_samples",
    "effect_modifier_site_max_samples",
    "prediction_intervention_repeats",
    "prediction_intervention_batch_size",
    "cv_folds",
    "cv_seed_offset",
    "lambda_count",
    "cpu_workers",
    "optimizer_max_iter",
    "threads_per_worker",
)

FLOAT_CONFIG_KEYS = (
    "synthetic_alpha",
    "effect_modifier_signal_strength",
    "effect_modifier_main_effect_strength",
    "effect_modifier_confounding_strength",
    "effect_modifier_intercept",
    "effect_modifier_treatment_prob",
    "effect_modifier_propensity_clip",
    "effect_modifier_site_gaussian_mean_span",
    "effect_modifier_site_gaussian_std_low",
    "effect_modifier_site_gaussian_std_high",
    "lambda_min_ratio",
    "standardization_scale_floor",
    "optimizer_tolerance",
    "optimizer_initial_step",
    "optimizer_backtrack_factor",
    "optimizer_min_step",
)



def _run(args, cwd=PROJECT_ROOT):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text(values, ordered_keys):
    lines = []
    for key in ordered_keys:
        if key not in values:
            raise ValueError("configuration is missing %s" % key)
        value = str(values[key])
        if not value or "\n" in value or "\r" in value or "=" in key:
            raise ValueError("configuration fields must be nonempty single lines")
        lines.append("%s=%s" % (key, value))
    return "\n".join(lines) + "\n"


def config_hashes(values):
    return {
        "run_config_sha256": sha256_text(canonical_text(values, RUN_CONFIG_KEYS)),
        "benchmark_config_sha256": sha256_text(
            canonical_text(values, BENCHMARK_CONFIG_KEYS)
        ),
        "variant_config_sha256": sha256_text(
            canonical_text(values, VARIANT_CONFIG_KEYS)
        ),
    }


def canonicalize_config_values(values):
    """Normalize numeric config fields to their unique manifest text."""
    normalized = dict(values)
    for key in INTEGER_CONFIG_KEYS:
        try:
            normalized[key] = str(int(values[key]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("%s must be an integer" % key) from error
    for key in FLOAT_CONFIG_KEYS:
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("%s must be numeric" % key) from error
        if not math.isfinite(value):
            raise ValueError("%s must be finite" % key)
        normalized[key] = str(value)
    return normalized


def validate_run_id(run_id, variant):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id or ""):
        raise ValueError("RUN_ID may contain only letters, digits, dot, underscore, hyphen")
    if variant not in VARIANT_CONTRACT:
        raise ValueError("unknown effect-modifier variant: %s" % variant)
    prefix = VARIANT_CONTRACT[variant]["run_id_prefix"]
    if not run_id.startswith(prefix):
        raise ValueError("%s RUN_ID must start with %s" % (variant, prefix))


def validate_manifest_self_consistency(manifest, run_id=None, variant=None):
    """Recompute canonical config hashes and enforce the variant namespace."""
    actual_run_id = run_id if run_id is not None else manifest.get("run_id", "")
    actual_variant = (
        variant
        if variant is not None
        else manifest.get("effect_modifier_variant", "")
    )
    validate_run_id(actual_run_id, actual_variant)
    expected_identity = {
        "format_version": "2",
        "run_id": actual_run_id,
        "effect_modifier_variant": actual_variant,
        "benchmark_id": BENCHMARK_ID,
        "method_name": METHOD_NAME,
        "implementation_label": IMPLEMENTATION_LABEL,
        "privacy_label": PRIVACY_LABEL,
    }
    for key, expected_value in expected_identity.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                "manifest identity mismatch for %s: expected %r, found %r"
                % (key, expected_value, manifest.get(key))
            )

    ordered_config_keys = (
        RUN_CONFIG_KEYS + BENCHMARK_CONFIG_KEYS + VARIANT_CONFIG_KEYS
    )
    stored_values = {
        key: manifest.get(key, "")
        for key in ordered_config_keys
    }
    canonical_values = canonicalize_config_values(stored_values)
    for key in ordered_config_keys:
        if stored_values[key] != canonical_values[key]:
            raise ValueError("manifest config field is not canonical: %s" % key)

    variant_contract = VARIANT_CONTRACT[actual_variant]
    for key in VARIANT_CONFIG_KEYS:
        if canonical_values[key] != variant_contract[key]:
            raise ValueError(
                "manifest variant contract mismatch for %s" % key
            )
    for key, expected_hash in config_hashes(canonical_values).items():
        if manifest.get(key) != expected_hash:
            raise ValueError(
                "manifest canonical configuration hash mismatch for %s" % key
            )
    return canonical_values

def expected_paths(run_id):
    return {
        "output_dir": OUTPUT_ROOT / run_id,
        "log_dir": LOG_ROOT / run_id,
        "runtime_dir": RUNTIME_ROOT / run_id,
        "lock_dir": CPU_LOCK_ROOT / run_id,
    }


def _assert_no_symlink_components(path, stop_root):
    path = Path(path)
    stop_root = Path(stop_root).resolve(strict=True)
    if path.resolve(strict=False) != path.absolute():
        raise ValueError("path resolves through a symlink: %s" % path)
    current = path
    while current != stop_root:
        if current.exists() and current.is_symlink():
            raise ValueError("path component may not be a symlink: %s" % current)
        if stop_root not in current.parents:
            raise ValueError("path escapes required root %s: %s" % (stop_root, path))
        current = current.parent


def validate_exact_path(candidate, expected, root, label):
    candidate = Path(candidate)
    expected = Path(expected)
    root = Path(root).resolve(strict=True)
    if not candidate.is_absolute():
        raise ValueError("%s must be absolute" % label)
    if candidate.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError("%s must resolve exactly to %s" % (label, expected))
    _assert_no_symlink_components(candidate, root)
    return candidate


def parse_env_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("metadata must be a regular non-symlink file: %s" % path)
    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError("line %d is not key=value: %s" % (line_number, path))
            key, value = line.split("=", 1)
            if not key or key in values:
                raise ValueError("empty or duplicate key at line %d: %s" % (line_number, path))
            values[key] = value
    return values


def env_text(fields):
    lines = []
    for key, value in fields:
        text = str(value)
        if not key or "=" in key or "\n" in key or "\n" in text or "\r" in text:
            raise ValueError("env metadata keys and values must be single lines")
        lines.append("%s=%s" % (key, text))
    return "\n".join(lines) + "\n"


def atomic_create_text(path, text, mode=0o600):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite existing path: %s" % path)
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("temporary path already exists: %s" % temporary)
    try:
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def compute_source_tree_sha256(repo_root=PROJECT_ROOT):
    repo_root = Path(repo_root).resolve(strict=True)
    raw_paths = _run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
    ).split(b"\0")
    digest = hashlib.sha256()
    digest.update(b"fedfisher-source-tree-v1\0")
    for raw_path in sorted(path for path in raw_paths if path):
        relative_path = os.fsdecode(raw_path)
        full_path = repo_root / relative_path
        file_stat = full_path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            payload = os.fsencode(os.readlink(full_path))
            kind = b"L"
        elif stat.S_ISREG(file_stat.st_mode):
            payload = full_path.read_bytes()
            kind = b"F"
        else:
            raise ValueError("source tree contains unsupported entry: %s" % full_path)
        digest.update(raw_path)
        digest.update(b"\0" + kind)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def compute_provenance(repo_root=PROJECT_ROOT):
    repo_root = Path(repo_root).resolve(strict=True)
    commit = _run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo_root).decode().strip()
    status_bytes = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=repo_root,
    )
    status_sha256 = hashlib.sha256(status_bytes).hexdigest()
    source_tree_sha256 = compute_source_tree_sha256(repo_root)
    values = {
        "git_commit": commit,
        "git_dirty": "1" if status_bytes else "0",
        "git_status_sha256": status_sha256,
        "source_tree_sha256": source_tree_sha256,
    }
    values["provenance_sha256"] = sha256_text(
        canonical_text(
            values,
            (
                "git_commit",
                "git_dirty",
                "git_status_sha256",
                "source_tree_sha256",
            ),
        )
    )
    return values
