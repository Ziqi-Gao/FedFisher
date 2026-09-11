"""Immutable, provenance-bound scientific cells with fail-closed retry checks."""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from ..dslassocov_run_contract import (
    atomic_create_text, canonical_text, compute_provenance, env_text, parse_env_file, sha256_file)
from .evaluation import model_hash, signal_recovery_with_ties
from .protocol import CELLS, VARIANTS
from models import SYNTHETIC_MLP_HIDDEN_DIMS

OUTPUT_ROOT = Path("/data/del6500/FedFisher/effect_modifier_experiment/outputs")
SCRATCH_ROOT = Path("/scr/del6500/FedFisher")
PROVENANCE_KEYS = ("git_commit", "git_dirty", "git_status_sha256", "source_tree_sha256")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    safe_path(path)
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not any(
            root in path.parents for root in (OUTPUT_ROOT, SCRATCH_ROOT)):
        raise ValueError("comparison artifacts must remain below FedFisher output/scratch roots")
    if path.resolve(strict=False) != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink in comparison artifact path")
    return path


def create_bytes(path, data):
    path = safe_path(path)
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    safe_path(temporary)
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def make_manifest(run_id, config, provenance):
    return dict(format_version="2", run_id=run_id, protocol="causal-comparison-v1",
                config_json=canonical(config), config_sha256=digest(config), **provenance)


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid %s SHA-256" % label)


def _finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid nonnegative finite %s" % label)


def _validate_model_arrays(arrays, method, covariate_dim):
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("model artifact must contain named arrays")
    family = method["family"]
    if family == "rlearner":
        expected = {"effect_coefficients": (covariate_dim + 1,)}
        dtype = np.dtype("float64")
    elif family in ("fedavg", "one_shot_fedavg", "one_shot_group_lasso"):
        widths = (1 + 2 * covariate_dim, *SYNTHETIC_MLP_HIDDEN_DIMS[method["model"]], 2)
        expected = {}
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            expected["net.%d.weight" % (2 * index)] = (right, left)
            expected["net.%d.bias" % (2 * index)] = (right,)
        dtype = np.dtype("float32")
    elif family in ("dslassocov", "local_logistic_lasso"):
        expected = {"intercept": (), "coefficients": (1 + 2 * covariate_dim,)}
        dtype = np.dtype("float32")
    else:
        raise ValueError("unknown comparison model family")
    if set(arrays) != set(expected):
        raise ValueError("model artifact fields disagree with the configured method")
    for name, shape in expected.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError("invalid model array shape, dtype or values: %s" % name)


def validate_result(result, arrays, identity, config):
    """Check scientific semantics before writing a completion marker or retrying."""
    canonical(result)  # Reject NaN/Infinity anywhere, including unused fields.
    try:
        if result["identity"] != identity or result["status"] != "complete":
            raise ValueError("scientific result identity/status mismatch")
        if result["fit"].get("completion_eligible") is not True or result["fit"]["seed"] != identity["seed"]:
            raise ValueError("scientific fit did not complete")
        d, signal_dim = config["dgp"]["covariate_dim"], config["dgp"]["signal_dim"]
        _validate_model_arrays(arrays, config["method"], d)
        family = config["method"]["family"]
        if family == "rlearner":
            from .rlearner import validate_fit_report
            validate_fit_report(result["fit"], config["method"]["optimizer"],
                                config["method"]["penalty"], arrays["effect_coefficients"])
        elif family == "fedavg":
            from .fedavg import validate_fit_report
            validate_fit_report(result["fit"], config["method"]["optimizer"],
                                config["method"]["model"], identity["seed"], arrays)
        else:
            from .legacy import validate_fit_report
            validate_fit_report(result["fit"], config["method"], arrays)
        fingerprints = result["fingerprints"]
        for key in ("training", "test", "model"):
            _hash(fingerprints[key], key)
        if model_hash(arrays) != fingerprints["model"]:
            raise ValueError("model semantic fingerprint mismatch")
        evaluation = result["evaluation"]
        effect_only = config["method"]["family"] == "rlearner"
        if evaluation["primary"] != "permute/tau_prob" or evaluation["effect_only"] is not effect_only:
            raise ValueError("comparison probability endpoint or effect-only identity mismatch")
        scales = ("tau_prob",) if effect_only else ("tau_prob", "tau_logit")
        expected_metrics = {mode + "/" + scale for mode in config["evaluation"]["modes"] for scale in scales}
        if set(evaluation["metrics"]) != expected_metrics:
            raise ValueError("method evaluation scales or modes do not match the contract")
        for name, metric in evaluation["metrics"].items():
            values = np.asarray(metric["importance"], dtype=np.float64)
            if values.shape != (d,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("invalid raw-X importance vector")
            scale = name.split("/")[1].removeprefix("tau_")
            expected_recovery = signal_recovery_with_ties(values, signal_dim, "tau_intervention_" + scale)
            if metric["recovery"] != expected_recovery:
                raise ValueError("recovery metrics disagree with the importance ranking")
        for key in ("raw_x_sha256", "permutation_sha256"):
            _hash(evaluation[key], key)
        shard_sizes = evaluation["shard_sizes"]
        shard_hashes = evaluation["shard_indices_sha256"]
        expected_shards = config["dgp"]["num_sites"] if "local_permutation" in config["variant"] else 1
        if (len(shard_sizes) != expected_shards or len(shard_hashes) != expected_shards
                or any(type(size) is not int or size < 1 for size in shard_sizes)
                or sum(shard_sizes) != config["dgp"]["num_test"]):
            raise ValueError("evaluation shards do not cover the prescribed common test set")
        for value in shard_hashes:
            _hash(value, "shard")
        sizes = result["site_sizes"]
        if len(sizes) != config["dgp"]["num_sites"] or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("invalid training site sizes")
        if family == "rlearner":
            local_reports = result["fit"]["local_reports"]
            if (result["fit"]["num_sites"] != len(sizes) or result["fit"]["num_examples"] != sum(sizes)
                    or [local["site_idx"] for local in local_reports] != list(range(len(sizes)))
                    or [local["summary"]["num_examples"] for local in local_reports] != sizes):
                raise ValueError("R fit and generated training site counts disagree")
        elif family == "fedavg":
            if [site["num_examples"] for site in result["fit"]["site_splits"]] != sizes:
                raise ValueError("FedAvg fit and generated training site counts disagree")
        counts = result["label_counts"]
        if set(counts) != {str(index) for index in range(len(sizes))}:
            raise ValueError("training label counts have the wrong site identity")
        for index, size in enumerate(sizes):
            site_counts = counts[str(index)]
            if (not set(site_counts).issubset({"0", "1"}) or not site_counts
                    or any(type(value) is not int or value < 0 for value in site_counts.values())
                    or sum(site_counts.values()) != size):
                raise ValueError("training site size and label counts disagree")
        timing = result["timing"]
        if timing["method"] != config["cell"]:
            raise ValueError("timing belongs to a different method")
        for stage in ("train", "evaluation"):
            measured = timing["stages"][stage]
            if measured["measured"] is not True or measured["failed_calls"] != 0:
                raise ValueError("missing successful full method timing")
            if type(measured["calls"]) is not int or measured["calls"] < 1:
                raise ValueError("missing method timing observation")
            for key in ("wall_seconds", "cpu_seconds"):
                _finite_nonnegative(measured[key], key)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed comparison result") from error


class ArtifactStore:
    def __init__(self, directory, manifest):
        self.directory = safe_path(directory)
        self.manifest = dict(manifest)
        if self.directory.name != self.manifest["run_id"]:
            raise ValueError("run directory and manifest identity differ")
        if self.manifest["format_version"] != "2" or self.manifest["protocol"] != "causal-comparison-v1":
            raise ValueError("invalid comparison manifest version")
        expected_fields = {"format_version", "run_id", "protocol", "config_json", "config_sha256", "provenance_sha256", *PROVENANCE_KEYS}
        if set(self.manifest) != expected_fields:
            raise ValueError("comparison manifest fields do not match the contract")
        config = json.loads(self.manifest["config_json"], object_pairs_hook=_pairs,
                            parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        if digest(config) != self.manifest["config_sha256"]:
            raise ValueError("configuration hash mismatch")
        if config["cell"] not in CELLS or config["variant"] not in VARIANTS:
            raise ValueError("invalid comparison cell or variant")
        if self.manifest["git_dirty"] != "0" or re.fullmatch(r"[0-9a-f]{40}", self.manifest["git_commit"]) is None:
            raise ValueError("comparison manifest must bind a finalized Git commit")
        for key in ("git_status_sha256", "source_tree_sha256", "provenance_sha256"):
            _hash(self.manifest[key], key)
        expected_provenance = hashlib.sha256(canonical_text(self.manifest, PROVENANCE_KEYS).encode()).hexdigest()
        if expected_provenance != self.manifest["provenance_sha256"]:
            raise ValueError("provenance hash mismatch")

    def initialize(self):
        path = self.directory / "run_manifest.env"
        if self.directory.exists() and not path.exists() and any(self.directory.iterdir()):
            raise ValueError("nonempty run directory lacks manifest")
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_path(path)
        if not path.exists():
            try:
                atomic_create_text(path, env_text(self.manifest.items()))
            except FileExistsError:
                pass  # Another centrally admitted shard may initialize this run.
        self.validate_manifest()

    def validate_manifest(self):
        path = safe_path(self.directory / "run_manifest.env")
        if not path.is_file() or parse_env_file(path) != self.manifest:
            raise ValueError("run manifest/provenance/configuration mismatch")

    def identity(self, seed, split):
        if type(seed) is not int or seed < 0 or split not in ("iid", "noniid"):
            raise ValueError("invalid scientific unit")
        config = json.loads(self.manifest["config_json"])
        if not config["seeds"][0] <= seed <= config["seeds"][1]:
            raise ValueError("seed outside manifest")
        return dict(run_id=self.manifest["run_id"], seed=seed, split=split,
                    cell=config["cell"], variant=config["variant"],
                    config_sha256=self.manifest["config_sha256"],
                    provenance_sha256=self.manifest["provenance_sha256"])

    def cell_dir(self, seed, split):
        self.identity(seed, split)
        return safe_path(self.directory / ("seed_%03d_%s" % (seed, split)))

    def status(self, seed, split):
        self.validate_manifest()
        path = self.cell_dir(seed, split)
        if not path.exists():
            return "missing"
        expected = {"started.json", "model.npz", "result.json", "done.json"}
        if not path.is_dir() or set(p.name for p in path.iterdir()) != expected:
            raise ValueError("partial or invalid comparison output; preserve without overwrite")
        identity = self.identity(seed, split)
        marker = read_json(path / "done.json")
        if set(marker) != {"identity", "artifacts"} or marker["identity"] != identity:
            raise ValueError("completion identity mismatch")
        if set(marker["artifacts"]) != {"started.json", "model.npz", "result.json"}:
            raise ValueError("completion artifact inventory mismatch")
        for name, checksum in marker["artifacts"].items():
            target = safe_path(path / name)
            if not target.is_file() or sha256_file(target) != checksum:
                raise ValueError("comparison artifact checksum mismatch")
        started = read_json(path / "started.json")
        result = read_json(path / "result.json")
        if started["identity"] != identity or result["identity"] != identity:
            raise ValueError("scientific result identity mismatch")
        with np.load(path / "model.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        config = json.loads(self.manifest["config_json"])
        validate_result(result, arrays, identity, config)
        return "validated"

    def start(self, seed, split, execution):
        if self.status(seed, split) != "missing":
            raise FileExistsError("cell already completed")
        path = self.cell_dir(seed, split)
        path.mkdir()  # At-least-once concurrent delivery cannot overwrite a cell.
        atomic_create_text(path / "started.json", canonical(dict(
            identity=self.identity(seed, split), execution=execution)))

    def complete(self, seed, split, result, arrays):
        self.validate_manifest()
        path = self.cell_dir(seed, split)
        if set(p.name for p in path.iterdir()) != {"started.json"}:
            raise ValueError("partial output cannot be overwritten")
        identity = self.identity(seed, split)
        if read_json(path / "started.json")["identity"] != identity:
            raise ValueError("started identity mismatch")
        validate_result(result, arrays, identity, json.loads(self.manifest["config_json"]))
        encoded_result = canonical(result)  # Reject NaN before creating artifacts.
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        create_bytes(path / "model.npz", buffer.getvalue())
        atomic_create_text(path / "result.json", encoded_result)
        marker = dict(identity=self.identity(seed, split), artifacts={
            name: sha256_file(path / name) for name in ("started.json", "model.npz", "result.json")})
        atomic_create_text(path / "done.json", canonical(marker))
        if self.status(seed, split) != "validated":
            raise ValueError("new comparison completion failed validation")

    def fail(self, seed, split, error):
        path = self.cell_dir(seed, split) / "failure.json"
        safe_path(path)
        atomic_create_text(path, canonical(dict(identity=self.identity(seed, split),
            error_type=type(error).__name__, message=str(error))))


def current_provenance():
    provenance = compute_provenance()
    if provenance["git_dirty"] != "0":
        raise ValueError("comparison execution requires a finalized clean source tree")
    return provenance
