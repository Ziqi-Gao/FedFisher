"""Immutable one-shot-effect cells with current-source and semantic validation."""

import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np

from ..causal_comparison.artifacts import read_json, safe_path
from ..dslassocov_run_contract import (
    canonical_text, compute_provenance, env_text, parse_env_file,
)
from .wire import canonical, communication_report


PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")
OUTPUT_ROOT = Path("/data/del6500/FedFisher/effect_modifier_experiment/outputs")
PROTOCOL = "one-shot-effects-v1"
PROVENANCE_KEYS = ("git_commit", "git_dirty", "git_status_sha256", "source_tree_sha256")


def _sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def current_provenance():
    values = compute_provenance(PROJECT_ROOT)
    _require(values["git_dirty"] == "0", "one-shot effects require a finalized clean source tree")
    return values


def _atomic_create(path, data):
    """Exclusive publish; preserve any partial write for explicit inspection."""
    path = safe_path(path)
    pending = safe_path(path.with_name("." + path.name + ".pending"))
    if path.exists() or pending.exists():
        raise FileExistsError("refusing existing or partial artifact: " + str(path))
    descriptor = os.open(str(pending), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    safe_path(path)
    safe_path(pending)
    os.link(pending, path)
    pending.unlink()


def _arrays(values):
    _require(isinstance(values, dict) and bool(values), "named model arrays are required")
    result = {}
    for name, value in values.items():
        _require(isinstance(name, str) and bool(name) and name != "file"
                 and "/" not in name and "\\" not in name,
                 "unsafe model array name")
        array = np.asarray(value)
        _require(array.dtype.kind in "biuf" and np.isfinite(array).all(),
                 "model arrays must contain finite real numerical values")
        result[name] = np.array(array, order="C", copy=True)
    return result


class EffectArtifactStore:
    """One recipe/dataset/seed cell below a manifest-bound run directory.

    The optional directory supports isolated scratch fixtures; its basename
    must still match the configured run_id. Status never creates files.
    """

    def __init__(self, config, provenance, directory=None):
        _require(isinstance(config, dict), "configuration must be a mapping")
        self.config = json.loads(canonical(config))
        for key in ("run_id", "task", "recipe", "dataset"):
            _require(isinstance(self.config.get(key), str) and bool(self.config[key]),
                     "missing configuration identity: " + key)
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.config["run_id"]) is not None,
                 "invalid one-shot-effect run identifier")
        _require(type(self.config.get("seed")) is int and self.config["seed"] == 0,
                 "this bounded artifact contract accepts seed zero only")
        self.provenance = dict(provenance)
        _require(set(self.provenance) == {*PROVENANCE_KEYS, "provenance_sha256"},
                 "provenance fields do not match the contract")
        _require(self.provenance["git_dirty"] == "0"
                 and re.fullmatch(r"[0-9a-f]{40}", self.provenance["git_commit"]) is not None,
                 "provenance must bind a clean Git commit")
        for key in ("git_status_sha256", "source_tree_sha256", "provenance_sha256"):
            _require(isinstance(self.provenance[key], str)
                     and re.fullmatch(r"[0-9a-f]{64}", self.provenance[key]) is not None,
                     "invalid provenance hash: " + key)
        _require(_sha256(canonical_text(self.provenance, PROVENANCE_KEYS).encode())
                 == self.provenance["provenance_sha256"], "inconsistent provenance digest")
        self.run_id = self.config["run_id"]
        self.directory = safe_path(OUTPUT_ROOT / self.run_id if directory is None else directory)
        _require(self.directory.name == self.run_id, "directory and run identity differ")
        self.unit_dir = safe_path(self.directory / "seed_000")
        self.config_sha256 = _sha256(canonical(self.config).encode())
        self.manifest = dict(format_version="2", protocol=PROTOCOL, run_id=self.run_id,
                             config_json=canonical(self.config), config_sha256=self.config_sha256,
                             **self.provenance)
        self._current()

    @property
    def identity(self):
        return dict(run_id=self.run_id, config_sha256=self.config_sha256,
                    provenance_sha256=self.provenance["provenance_sha256"],
                    seed=self.config["seed"], recipe=self.config["recipe"], dataset=self.config["dataset"])

    def _current(self):
        _require(canonical(self.config) == self.manifest["config_json"]
                 and _sha256(canonical(self.config).encode()) == self.config_sha256,
                 "store configuration changed after construction")
        _require(current_provenance() == self.provenance,
                 "current source provenance differs from the scientific manifest")
        safe_path(self.directory)
        safe_path(self.unit_dir)

    def _manifest(self):
        self._current()
        path = safe_path(self.directory / "run_manifest.env")
        _require(path.is_file() and parse_env_file(path) == self.manifest,
                 "run manifest/provenance/configuration mismatch")
        _require({p.name for p in self.directory.iterdir()} <= {"run_manifest.env", "seed_000"},
                 "unexpected or partial artifacts in run directory")

    def initialize(self):
        self._current()
        path = safe_path(self.directory / "run_manifest.env")
        if self.directory.exists():
            _require(self.directory.is_dir(), "run path must be a directory")
            if not path.exists() and any(self.directory.iterdir()):
                raise ValueError("nonempty run directory lacks its manifest")
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_path(self.directory)
        if not path.exists():
            _atomic_create(path, env_text(self.manifest.items()).encode())
        self._manifest()

    def _started(self):
        started = read_json(self.unit_dir / "started.json")
        _require(set(started) == {"identity", "execution"}
                 and canonical(started["identity"]) == canonical(self.identity)
                 and isinstance(started["execution"], dict), "started identity or execution mismatch")
        return started

    def _semantic(self, result, arrays, payloads):
        _require(isinstance(result, dict) and result.get("status") == "complete"
                 and canonical(result.get("identity")) == canonical(self.identity),
                 "only identity-matched complete scientific results may finish")
        _require(isinstance(payloads, (list, tuple)) and bool(payloads)
                 and all(type(blob) is bytes and len(blob) > 0 for blob in payloads),
                 "nonempty serialized site payload bytes are required")
        communication = result.get("communication")
        _require(isinstance(communication, dict), "actual wire communication report is required")
        expected = communication_report(payloads, communication.get("protocol_distribution_bytes", 0),
                                        communication.get("final_model_distribution_bytes", 0))
        _require(all(key in communication and canonical(communication[key]) == canonical(value)
                     for key, value in expected.items()), "wire payload hashes/bytes/counts disagree")
        # Lazy import keeps the artifact lifecycle independent of runner/model code.
        from .validation import validate_result
        validate_result(result, arrays, list(payloads), self.config, self.identity)

    def _read_content(self, payload_names):
        self._started()
        result = read_json(self.unit_dir / "result.json")
        with np.load(safe_path(self.unit_dir / "model.npz"), allow_pickle=False) as archive:
            _require(len(set(archive.files)) == len(archive.files), "duplicate model archive fields")
            arrays = _arrays({name: archive[name] for name in archive.files})
        payloads = [safe_path(self.unit_dir / name).read_bytes() for name in payload_names]
        self._semantic(result, arrays, payloads)
        return result, arrays, payloads

    def status(self):
        self._current()
        if not self.directory.exists():
            return "missing"
        self._manifest()
        if not self.unit_dir.exists():
            return "missing"
        _require(self.unit_dir.is_dir(), "scientific unit must be a directory")
        marker_path = safe_path(self.unit_dir / "done.json")
        _require(marker_path.is_file(), "partial scientific unit has no valid completion marker")
        marker = read_json(marker_path)
        _require(set(marker) == {"identity", "artifacts"}
                 and canonical(marker["identity"]) == canonical(self.identity)
                 and isinstance(marker["artifacts"], dict), "completion marker identity mismatch")
        digests = marker["artifacts"]
        payload_count = len(digests) - 3
        _require(payload_count > 0, "completion marker lacks site payloads")
        payload_names = ["site_%03d.payload" % i for i in range(payload_count)]
        expected = {"started.json", "model.npz", "result.json", *payload_names}
        _require(set(digests) == expected, "completion artifact inventory mismatch")
        _require({p.name for p in self.unit_dir.iterdir()} == expected | {"done.json"},
                 "missing, extra or partial scientific artifacts")
        for name, digest in digests.items():
            path = safe_path(self.unit_dir / name)
            _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                     and path.is_file() and _sha256(path.read_bytes()) == digest,
                     "scientific artifact checksum mismatch: " + name)
        self._read_content(payload_names)
        self._current()
        return "validated"

    def start(self, execution):
        self._manifest()
        if self.status() != "missing":
            raise FileExistsError("scientific unit already exists")
        _require(isinstance(execution, dict), "scheduler execution metadata must be a mapping")
        encoded = canonical(dict(identity=self.identity, execution=execution)).encode()
        self.unit_dir.mkdir()
        safe_path(self.unit_dir)
        _atomic_create(self.unit_dir / "started.json", encoded)
        self._current()

    def complete(self, result, arrays, payloads):
        self._manifest()
        _require(self.unit_dir.is_dir()
                 and {p.name for p in self.unit_dir.iterdir()} == {"started.json"},
                 "partial or completed output cannot be overwritten")
        self._started()
        arrays = _arrays(arrays)
        payloads = list(payloads)
        self._semantic(result, arrays, payloads)
        encoded_result = canonical(result).encode()
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        payload_names = ["site_%03d.payload" % i for i in range(len(payloads))]
        content = {"model.npz": buffer.getvalue(), "result.json": encoded_result,
                   **dict(zip(payload_names, payloads))}
        for name, data in content.items():
            _atomic_create(self.unit_dir / name, data)
        for name, data in content.items():
            _require(safe_path(self.unit_dir / name).read_bytes() == data, "artifact readback mismatch")
        self._read_content(payload_names)
        self._current()
        names = ["started.json", *content]
        marker = dict(identity=self.identity, artifacts={
            name: _sha256(safe_path(self.unit_dir / name).read_bytes()) for name in names})
        _atomic_create(self.unit_dir / "done.json", canonical(marker).encode())
        _require(self.status() == "validated", "new completion marker did not validate")

    def fail(self, error):
        """Preserve a started attempt's failure; never replace completion evidence."""
        self._manifest()
        _require(isinstance(error, BaseException), "failure evidence requires an exception")
        _require(self.unit_dir.is_dir(), "only a started scientific unit may fail")
        self._started()
        if safe_path(self.unit_dir / "done.json").exists():
            raise FileExistsError("a completed unit cannot acquire a failure marker")
        failure = dict(identity=self.identity, error_type=type(error).__name__, message=str(error))
        _atomic_create(self.unit_dir / "failure.json", canonical(failure).encode())
        self._current()
