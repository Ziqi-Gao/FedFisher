"""Stable source fingerprints and env manifests, independent of any experiment.

The serialization and source-tree framing preserve the existing format-v2
contracts. Importing this module performs no filesystem or scheduler operation.
"""

import hashlib
import os
from pathlib import Path
import stat
import subprocess

PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")
DATA_ROOT = Path("/data/del6500/FedFisher")
SCRATCH_ROOT = Path("/scr/del6500/FedFisher")
PROVENANCE_KEYS = ("git_commit", "git_dirty", "git_status_sha256", "source_tree_sha256")


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
