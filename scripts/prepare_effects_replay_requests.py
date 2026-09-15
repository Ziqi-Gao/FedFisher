#!/usr/bin/env python3
"""Publish an immutable full replay or two-cell deep recovery review bundle.

Only nested project outbox requests and a last-written scratch inventory are
created. This does not register, admit, launch, regenerate data or infer models.
Original pilot artifact hashes and the clean new diagnostic source are both
checked before publication and again before the inventory becomes ready.
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

from effect_modifier_experiment.dslassocov_run_contract import compute_provenance
from effect_modifier_experiment.one_shot_effects.diagnostics import protocol
from effect_modifier_experiment.one_shot_effects.diagnostics.recovery import (
    REPLACED_CELLS, verify_recovery_evidence)


DIAGNOSTICS = PROJECT_ROOT / "effect_modifier_experiment/one_shot_effects/diagnostics"
PROPOSAL = PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/effects-replay-v1"
REVIEW_SOURCES = {
    "proposal": PROPOSAL / "registration-proposal.toml",
    "pilot_schema": PROPOSAL / "pilot-parameters.schema.json",
    "adapter": PROJECT_ROOT / "scripts/fedfisher_server_scheduler_adapter.py",
    "protocol": DIAGNOSTICS / "protocol.py",
    "source_bindings": DIAGNOSTICS / "source_bindings.json",
    "source_validator": DIAGNOSTICS / "sources.py",
    "request_builder": PROJECT_ROOT / "scripts/prepare_effects_replay_requests.py",
    "recovery_validator": DIAGNOSTICS / "recovery.py",
}
EXPECTED_CELLS = frozenset({
    ("fisher_shallow", "scmv2_base"), ("fisher_deep", "scmv2_base"),
    ("fisher_shallow", "legacy_control_iid"), ("fisher_deep", "legacy_control_iid"),
    ("dr_path", "scmv2_base"),
})


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


def _cohort(value):
    if value not in ("full", "deep-recovery"):
        raise ValueError("cohort must be full or deep-recovery")
    return value


def bundle_name(generation=1, cohort="full"):
    name = "pilot" if _cohort(cohort) == "full" else "deep-recovery"
    return "fedfisher-effects-replay-v1-%s-cohort-p%d" % (name, _generation(generation))


def inventory_name(generation=1, cohort="full"):
    return bundle_name(generation, cohort) + ".inventory.json"


def build_requests(generation=1, cohort="full"):
    """Build the reviewed science-only requests without touching checkpoints."""
    generation = _generation(generation)
    cohort = _cohort(cohort)
    if cohort == "deep-recovery" and generation < 3:
        raise ValueError("deep recovery requires generation 3 or later")
    plan = protocol.cohort_plan()
    if (len(plan) != 5 or {(item["recipe"], item["dataset"]) for item in plan} != EXPECTED_CELLS):
        raise ValueError("replay contract must cover exactly the five reviewed cells")
    expected = EXPECTED_CELLS if cohort == "full" else REPLACED_CELLS
    specs = []
    for parameters in plan:
        parameters = protocol.validate_parameters(protocol.PILOT_TASK, parameters)
        if (parameters["recipe"], parameters["dataset"]) not in expected:
            continue
        job_id = "%s-%s-%s" % (
            bundle_name(generation, cohort), parameters["recipe"].replace("_", "-"),
            parameters["dataset"].replace("_", "-"))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
            raise ValueError("generation makes the job identifier exceed protocol limits")
        request = dict(schema_version=2, job_id=job_id, project="FedFisher",
                       task=protocol.PILOT_TASK, parameters=parameters, priority=0)
        configuration = json.loads(_canonical(protocol.run_config(protocol.PILOT_TASK, parameters)))
        if (type(configuration.get("seed")) is not int or configuration["seed"] != 0
                or configuration.get("fitting_allowed") is not False
                or configuration.get("training_tasks") != 0
                or configuration.get("expansion_authorized") is not False):
            raise ValueError("replay scope must remain seed zero, no fitting and no expansion")
        specs.append(dict(filename=job_id + ".job-request.json", request=request,
                          request_sha256=_sha256(_encoded(request)),
                          run_id=protocol.run_id(protocol.PILOT_TASK, parameters),
                          configuration=configuration,
                          configuration_sha256=_sha256(_canonical(configuration).encode())))
    actual = {(item["request"]["parameters"]["recipe"],
               item["request"]["parameters"]["dataset"]) for item in specs}
    if len(specs) != len(expected) or actual != expected:
        raise ValueError("replay contract must cover exactly the reviewed cohort")
    if len({item["filename"] for item in specs}) != len(expected):
        raise ValueError("replay request identifiers are not unique")
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
        raise ValueError("refusing replay requests from a dirty source tree")
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


def _verify_inputs(specs):
    from effect_modifier_experiment.one_shot_effects.diagnostics.sources import verify_source_bindings
    return {item["run_id"]: verify_source_bindings(item["configuration"]) for item in specs}


def _recovery_binding(cohort, path, specs):
    if cohort == "full":
        if path is not None:
            raise ValueError("recovery evidence is only valid for the deep-recovery cohort")
        return None
    if path is None:
        raise ValueError("deep-recovery publication requires frozen P2 recovery evidence")
    evidence = verify_recovery_evidence(path)
    old = {item["run_id"] for item in evidence["replaced"]}
    if old & {item["run_id"] for item in specs}:
        raise ValueError("deep recovery requires new result namespaces")
    return dict(path=str(Path(path)), sha256=_sha256(Path(path).read_bytes()),
                evidence=evidence)


def write_requests(generation=1, cohort="full", recovery_evidence=None):
    """Create a complete immutable review bundle; retain partial failures."""
    generation = _generation(generation)
    cohort = _cohort(cohort)
    before = _provenance()
    specs = build_requests(generation, cohort)
    sources = _source_hashes()
    original_inputs = _verify_inputs(specs)
    recovery = _recovery_binding(cohort, recovery_evidence, specs)
    outbox, reports = _safe_path(OUTBOX_ROOT), _safe_path(REPORT_ROOT)
    bundle = _safe_path(outbox / bundle_name(generation, cohort))
    if bundle.exists():
        raise FileExistsError("refusing existing or partial bundle directory: %s" % bundle)
    targets = [bundle / item["filename"] for item in specs]
    inventory_path = reports / inventory_name(generation, cohort)
    for target in (*targets, inventory_path):
        _preflight_new(target)
    bundle.mkdir(parents=True, exist_ok=False)
    reports.mkdir(parents=True, exist_ok=True)
    for path in (outbox, bundle, reports):
        _safe_path(path)
    records = []
    for item, path in zip(specs, targets):
        _atomic_create(path, _encoded(item["request"]))
        records.append(dict(path=str(path), sha256=item["request_sha256"],
                            job_id=item["request"]["job_id"], task=item["request"]["task"],
                            parameters=item["request"]["parameters"], run_id=item["run_id"],
                            configuration_sha256=item["configuration_sha256"],
                            configuration=item["configuration"], seeds=[0]))
    for item, path in zip(specs, targets):
        _safe_path(path)
        body = path.read_bytes()
        if body != _encoded(item["request"]) or _sha256(body) != item["request_sha256"]:
            raise ValueError("request readback differs from planned bytes: %s" % path)
    if _verify_inputs(specs) != original_inputs:
        raise ValueError("original pilot inputs changed; partial bundle preserved")
    if _recovery_binding(cohort, recovery_evidence, specs) != recovery:
        raise ValueError("recovery evidence changed; partial bundle preserved")
    if _provenance() != before or _source_hashes() != sources:
        raise ValueError("source changed while publishing requests; partial bundle preserved")
    inventory = dict(
        schema_version=1, bundle="fedfisher-effects-replay-v1-" + ("pilot" if cohort == "full" else cohort),
        generation=generation, cohort=cohort, task=protocol.PILOT_TASK, request_count=len(specs),
        coverage=dict(fisher_replays=sum(item["request"]["parameters"]["recipe"].startswith("fisher_") for item in specs),
                      dr_path_replays=sum(item["request"]["parameters"]["recipe"] == "dr_path" for item in specs), seeds=[0]),
        fitting_allowed=False, training_tasks=0, expansion_authorized=False,
        bundle_ready=True, submitted=False, requires_operator_registration=True,
        publication_mode="nested_operator_review_bundle_not_automatic_intake",
        registration_state="disabled project proposal; no central admission implied",
        provenance=before, project_head=before["git_commit"], worktree_status="clean",
        review_sources=sources, original_inputs=original_inputs, requests=records,
        operator_gate="verify exact diagnostic source and original artifact hashes; review replay task admission; validate and submit exactly these %d requests through ServerScheduler" % len(specs),
        freeze_rule="keep canonical source unchanged through replay and scientific acceptance; no training or compression expansion",
        inventory_path=str(inventory_path),
    )
    if recovery is not None:
        inventory.update(recovery=recovery, retained_request_count=3,
                         requires_operator_registration=False,
                         requires_operator_source_review=True,
                         registration_state="existing task requires operator source/profile review and live registration verification; no central admission implied")
    _atomic_create(inventory_path, _encoded(inventory))
    if inventory_path.read_bytes() != _encoded(inventory):
        raise ValueError("inventory readback differs from planned bytes")
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--cohort", choices=("full", "deep-recovery"), default="full")
    parser.add_argument("--recovery-evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        inventory = write_requests(args.generation, args.cohort, args.recovery_evidence)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print("request_generation_error=%s" % error)
        return 2
    path = Path(inventory["inventory_path"])
    print(json.dumps(dict(inventory_path=str(path), inventory_sha256=_sha256(path.read_bytes()),
                          project_head=inventory["project_head"], request_count=inventory["request_count"],
                          submitted=False, requires_operator_registration=inventory["requires_operator_registration"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
