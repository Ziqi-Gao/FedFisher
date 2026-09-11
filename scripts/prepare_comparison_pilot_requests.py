#!/usr/bin/env python3
"""Publish an immutable, unsubmitted 96-request comparison pilot review bundle.

Only request JSON is written under the project outbox; inventory evidence is
written under project scratch reports. This program never invokes a scheduler,
registers a task, fits a model, or admits work. A complete inventory is published
last. Any earlier failure leaves partial files for inspection and prevents a
same-generation retry from overwriting them.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")
SCRATCH_ROOT = Path("/scr/del6500/FedFisher")
OUTBOX_ROOT = SCRATCH_ROOT / "scheduler/outbox"
REPORT_ROOT = SCRATCH_ROOT / "scheduler/reports"
sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.causal_comparison import protocol
from effect_modifier_experiment.dslassocov_run_contract import compute_provenance


REVIEW_SOURCES = {
    "proposal": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/comparison-registration-proposal.toml",
    "pilot_schema": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/comparison-pilot-parameters.schema.json",
    "adapter": PROJECT_ROOT / "scripts/fedfisher_server_scheduler_adapter.py",
    "protocol": PROJECT_ROOT / "effect_modifier_experiment/causal_comparison/protocol.py",
    "request_builder": PROJECT_ROOT / "scripts/prepare_comparison_pilot_requests.py",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _generation(value):
    if type(value) is not int or value < 1:
        raise ValueError("generation must be a positive integer")
    return value


def bundle_name(generation=1):
    return "fedfisher-comparison-v1-pilot-cohort-p%d" % _generation(generation)


def inventory_name(generation=1):
    return bundle_name(generation) + ".inventory.json"


def build_requests(generation=1):
    """Build deterministic scientific specs in memory, without executing fits."""
    generation = _generation(generation)
    specs = []
    for cell in protocol.CELLS:
        for variant in protocol.VARIANTS:
            parameters = dict(variant=variant, cell=cell, shard_index=0, shard_count=1)
            protocol.validate_parameters(protocol.PILOT_TASK, parameters)
            job_id = "fedfisher-comparison-v1-pilot-cohort-p%d-%s-%s-s00of01" % (
                generation, cell.replace("_", "-"), variant.replace("_", "-"))
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
                raise ValueError("generation makes the job identifier exceed protocol limits")
            request = dict(schema_version=2, job_id=job_id, project="FedFisher",
                           task=protocol.PILOT_TASK, parameters=parameters, priority=0)
            configuration = json.loads(_canonical(protocol.run_config(protocol.PILOT_TASK, parameters)))
            specs.append(dict(filename=job_id + ".job-request.json", request=request,
                              request_sha256=_sha256(_encoded(request)),
                              run_id=protocol.run_id(protocol.PILOT_TASK, parameters),
                              configuration=configuration,
                              configuration_sha256=_sha256(_canonical(configuration).encode()),
                              seed_sequence=list(protocol.seed_sequence(protocol.PILOT_TASK, parameters)),
                              splits=list(protocol.SPLITS)))
    expected = {(cell, variant) for cell in protocol.CELLS for variant in protocol.VARIANTS}
    actual = {(item["request"]["parameters"]["cell"], item["request"]["parameters"]["variant"]) for item in specs}
    if len(specs) != 96 or len(expected) != 96 or actual != expected:
        raise ValueError("pilot contract must cover exactly 24 cells by four variants")
    if len({item["filename"] for item in specs}) != len(specs):
        raise ValueError("pilot request identifiers are not unique")
    if any(item["seed_sequence"] != [0] or item["splits"] != ["iid", "noniid"] for item in specs):
        raise ValueError("pilot scope must remain seed zero and both splits")
    return tuple(specs)


def _safe_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or SCRATCH_ROOT not in path.parents:
        raise ValueError("bundle paths must remain below the FedFisher scratch root")
    if path.resolve(strict=False) != path or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("bundle path must not contain symlinks")
    return path


def _pending_path(path):
    return path.with_name("." + path.name + ".pending")


def _preflight_new(path):
    path = _safe_path(path)
    pending = _safe_path(_pending_path(path))
    for target in (path, pending):
        if target.exists():
            raise FileExistsError("refusing existing or partial bundle target: %s" % target)
    return path


def _atomic_create(path, data):
    path = _preflight_new(path)
    pending = _pending_path(path)
    # No cleanup on failure: preserve partial evidence and fail closed next time.
    descriptor = os.open(str(pending), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _safe_path(path)
    _safe_path(pending)
    os.link(pending, path)
    pending.unlink()


def _provenance():
    provenance = compute_provenance()
    if provenance.get("git_dirty") != "0":
        raise ValueError("refusing pilot requests from a dirty source tree")
    if re.fullmatch(r"[0-9a-f]{40}", provenance.get("git_commit", "")) is None:
        raise ValueError("invalid source commit provenance")
    for key in ("git_status_sha256", "source_tree_sha256", "provenance_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", provenance.get(key, "")) is None:
            raise ValueError("invalid source provenance digest: %s" % key)
    return dict(provenance)


def _source_hashes():
    records = {}
    for name, path in REVIEW_SOURCES.items():
        if PROJECT_ROOT not in path.parents or path.resolve(strict=True) != path or any(
                item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("review source must be an ordinary canonical project file")
        records[name] = dict(path=str(path), sha256=_sha256(path.read_bytes()))
    return records


def write_requests(generation=1):
    """Create one clean-source review bundle, with its ready inventory last."""
    generation = _generation(generation)
    before = _provenance()
    specs = build_requests(generation)
    sources = _source_hashes()
    outbox, reports = _safe_path(OUTBOX_ROOT), _safe_path(REPORT_ROOT)
    # Existing root-only automatic intake must not prematurely ingest an
    # unregistered task. The operator submits the exact nested paths after
    # registration review; this does not implement a project queue or launcher.
    bundle = _safe_path(outbox / bundle_name(generation))
    if bundle.exists():
        raise FileExistsError("refusing existing or partial bundle directory: %s" % bundle)
    targets = [bundle / item["filename"] for item in specs]
    inventory_path = reports / inventory_name(generation)
    for target in (*targets, inventory_path):
        _preflight_new(target)
    # Directory validation precedes creation, and is repeated after it.
    bundle.mkdir(parents=True, exist_ok=False)
    reports.mkdir(parents=True, exist_ok=True)
    _safe_path(outbox)
    _safe_path(bundle)
    _safe_path(reports)
    records = []
    for item, path in zip(specs, targets):
        _atomic_create(path, _encoded(item["request"]))
        records.append(dict(path=str(path), sha256=item["request_sha256"],
                            job_id=item["request"]["job_id"], task=item["request"]["task"],
                            parameters=item["request"]["parameters"], run_id=item["run_id"],
                            configuration_sha256=item["configuration_sha256"],
                            configuration=item["configuration"],
                            seeds=item["seed_sequence"], splits=item["splits"]))
    # Verify bytes actually materialized, rather than trusting the construction
    # records. The full immutable inventory is the only bundle-ready marker.
    for item, path in zip(specs, targets):
        _safe_path(path)
        body = path.read_bytes()
        if body != _encoded(item["request"]) or _sha256(body) != item["request_sha256"]:
            raise ValueError("request readback differs from planned bytes: %s" % path)
    if _provenance() != before or _source_hashes() != sources:
        raise ValueError("source changed while publishing requests; partial bundle preserved")
    inventory = dict(
        schema_version=1, bundle="fedfisher-comparison-v1-pilot", generation=generation,
        task=protocol.PILOT_TASK, request_count=len(records), scientific_cell_count=192,
        coverage=dict(method_cells=24, variants=4, seeds=[0], splits=["iid", "noniid"]),
        bundle_ready=True, submitted=False, requires_operator_registration=True,
        publication_mode="nested_operator_review_bundle_not_automatic_intake",
        registration_state="project proposal only; no central registration or admission implied",
        provenance=before, project_head=before["git_commit"], worktree_status="clean",
        review_sources=sources, requests=records,
        operator_gate="verify exact source provenance, review/register pilot task only, validate exact request hashes, then submit through ServerScheduler",
        freeze_rule="keep this canonical source unchanged until all pilots and paired scientific validation finish",
        inventory_path=str(inventory_path),
    )
    _atomic_create(inventory_path, _encoded(inventory))
    if inventory_path.read_bytes() != _encoded(inventory):
        raise ValueError("inventory readback differs from planned bytes")
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        inventory = write_requests(args.generation)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print("request_generation_error=%s" % error)
        return 2
    path = Path(inventory["inventory_path"])
    print(json.dumps(dict(inventory_path=str(path), inventory_sha256=_sha256(path.read_bytes()),
                          project_head=inventory["project_head"], request_count=inventory["request_count"],
                          scientific_cell_count=inventory["scientific_cell_count"],
                          submitted=False, requires_operator_registration=True), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
