"""Strict JSON and immutable file primitives shared by scientific experiments.

No models, data generators or experiment validators are imported here. The
legacy comparison error wording is retained for compatibility with callers.
"""

import hashlib
import json
import os
from pathlib import Path

from .provenance import PROVENANCE_KEYS, SCRATCH_ROOT

OUTPUT_ROOT = Path("/data/del6500/FedFisher/effect_modifier_experiment/outputs")


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
