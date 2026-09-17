#!/usr/bin/env python3
"""Publish an immutable, unsubmitted original FedFisher timing review bundle.

Only request JSON is written under the project outbox; inventory evidence is
written under project scratch reports. This program never invokes a scheduler,
registers a task, fits a model, or admits work. A complete inventory is published
last. Any earlier failure leaves partial files for inspection and prevents a
same-generation retry from overwriting them.
"""

import argparse
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

from effect_modifier_experiment.fedfisher_timing import protocol
from effect_modifier_experiment.infrastructure.provenance import compute_provenance
from effect_modifier_experiment.infrastructure.requests import (
    canonical as _canonical, encoded as _encoded, sha256 as _sha256,
    safe_path as _safe_path, pending_path as _pending_path,
    preflight_new as _preflight_new, atomic_create as _atomic_create,
    clean_provenance, review_sources, publish_bundle,
)


REVIEW_SOURCES = {
    "proposal": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/fedfisher-timing-v1/registration-proposal.toml",
    "pilot_schema": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/fedfisher-timing-v1/pilot-parameters.schema.json",
    "adapter": PROJECT_ROOT / "scripts/fedfisher_server_scheduler_adapter.py",
    "protocol": PROJECT_ROOT / "effect_modifier_experiment/fedfisher_timing/protocol.py",
    "request_builder": PROJECT_ROOT / "scripts/prepare_fedfisher_timing_requests.py",
    "scope_plan": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/fedfisher-timing-v1/timing-plan.json",
}
EXPECTED_CELLS = frozenset({"fedfisher_diag_shallow", "fedfisher_kfac_shallow",
                            "fedfisher_diag_deep", "fedfisher_kfac_deep"})
EXPECTED_VARIANTS = frozenset({"control", "site_gaussian_shift", "local_permutation",
                              "site_gaussian_local_permutation"})


def _generation(value):
    if type(value) is not int or value < 1:
        raise ValueError("generation must be a positive integer")
    return value


def _cohort(value):
    if value not in ("pilot", "formal"):
        raise ValueError("cohort must be pilot or formal")
    return value


def bundle_name(generation=1, cohort="pilot"):
    return "fedfisher-timing-v1-%s-cohort-p%d" % (_cohort(cohort), _generation(generation))


def inventory_name(generation=1, cohort="pilot"):
    return bundle_name(generation, cohort) + ".inventory.json"


def build_requests(generation=1, cohort="pilot", shard_count=None):
    """Build deterministic scientific specs in memory, without executing fits."""
    generation = _generation(generation)
    cohort = _cohort(cohort)
    pilot = cohort == "pilot"
    if pilot and shard_count is not None:
        raise ValueError("pilot shard count is fixed; shard-count is a formal option")
    shard_count = 1 if pilot else (16 if shard_count is None else shard_count)
    if not pilot and (type(shard_count) is not int or shard_count not in (8, 16)):
        raise ValueError("formal shard count must be 8 or 16")
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    if (len(protocol.CELLS) != 4 or set(protocol.CELLS) != EXPECTED_CELLS
            or len(protocol.VARIANTS) != 4 or set(protocol.VARIANTS) != EXPECTED_VARIANTS):
        raise ValueError("timing pilot must retain exactly the four methods and four legacy variants")
    specs = []
    for cell in protocol.CELLS:
        for variant in protocol.VARIANTS:
            for shard_index in range(shard_count):
                parameters = dict(variant=variant, cell=cell, shard_index=shard_index, shard_count=shard_count)
                protocol.validate_parameters(task, parameters)
                job_id = "%s-%s-%s-s%02dof%02d" % (
                    bundle_name(generation, cohort), cell.replace("_", "-"),
                    variant.replace("_", "-"), shard_index, shard_count)
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
                    raise ValueError("generation makes the job identifier exceed protocol limits")
                request = dict(schema_version=2, job_id=job_id, project="FedFisher",
                               task=task, parameters=parameters, priority=0)
                configuration = json.loads(_canonical(protocol.run_config(task, parameters)))
                specs.append(dict(filename=job_id + ".job-request.json", request=request,
                                  request_sha256=_sha256(_encoded(request)),
                                  run_id=protocol.run_id(task, parameters),
                                  configuration=configuration,
                                  configuration_sha256=_sha256(_canonical(configuration).encode()),
                                  seed_sequence=list(protocol.seed_sequence(task, parameters)),
                                  splits=list(protocol.SPLITS)))
    expected = {(cell, variant) for cell in protocol.CELLS for variant in protocol.VARIANTS}
    actual = {(item["request"]["parameters"]["cell"], item["request"]["parameters"]["variant"]) for item in specs}
    if len(specs) != 16 * shard_count or len(expected) != 16 or actual != expected:
        raise ValueError("pilot contract must cover exactly four method/architecture cells by four variants")
    if len({item["filename"] for item in specs}) != len(specs):
        raise ValueError("pilot request identifiers are not unique")
    for item in specs:
        expected_seeds = ([0] if pilot else list(range(item["request"]["parameters"]["shard_index"], 300, shard_count)))
        if item["seed_sequence"] != expected_seeds or item["splits"] != ["iid", "noniid"]:
            raise ValueError("timing scope must retain seed zero pilot or exact 300-seed partitions and both splits")
    return tuple(specs)


def _provenance():
    return clean_provenance(compute_provenance())


def _source_hashes():
    return review_sources(REVIEW_SOURCES, PROJECT_ROOT)


def _acceptance_binding(cohort, path, provenance):
    if cohort == "pilot":
        if path is not None:
            raise ValueError("pilot-acceptance is valid only for formal publication")
        return None
    if path is None:
        raise ValueError("formal publication requires complete same-source pilot acceptance")
    from effect_modifier_experiment.fedfisher_timing.acceptance import verify_pilot_acceptance
    accepted = verify_pilot_acceptance(path)
    if accepted.get("provenance") != provenance:
        raise ValueError("pilot acceptance differs from the current clean source provenance")
    return dict(path=str(Path(path)), sha256=_sha256(Path(path).read_bytes()), evidence=accepted)


def write_requests(generation=1, cohort="pilot", pilot_acceptance=None, shard_count=None):
    """Create one clean-source review bundle, with its ready inventory last."""
    generation = _generation(generation)
    cohort = _cohort(cohort)
    before = _provenance()
    specs = build_requests(generation, cohort, shard_count)
    accepted = _acceptance_binding(cohort, pilot_acceptance, before)
    sources = _source_hashes()
    pilot = cohort == "pilot"
    metadata = dict(
        schema_version=1, bundle="fedfisher-timing-v1-"+cohort, generation=generation,
        task=protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK,
        scientific_cell_count=32 if pilot else 9600,
        coverage=dict(method_cells=4, variants=4, seeds=[0] if pilot else list(range(300)), splits=["iid", "noniid"]),
        formal_plan=dict(task=protocol.FORMAL_TASK, seed_start=0, seed_end=299,
                         allowed_shard_counts=[8, 16], scientific_cell_count=9600,
                         publication_authorized=not pilot,
                         gate="complete same-source pilot scientific and paired acceptance plus operator profile review"),
        operator_gate="verify exact source provenance and scientific acceptance gate, review/register only the admitted stage, validate exact request hashes, then submit through ServerScheduler",
        freeze_rule="keep this canonical source unchanged through pilot/formal execution and paired scientific acceptance",
    )
    return publish_bundle(
        specs, bundle=OUTBOX_ROOT / bundle_name(generation, cohort),
        inventory_path=REPORT_ROOT / inventory_name(generation, cohort),
        metadata=metadata, provenance=before, sources=sources, acceptance=accepted,
        check_provenance=_provenance, check_sources=_source_hashes,
        check_acceptance=lambda: _acceptance_binding(cohort, pilot_acceptance, before),
        create=_atomic_create,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--cohort", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--pilot-acceptance", type=Path)
    parser.add_argument("--shard-count", choices=(8, 16), type=int)
    args = parser.parse_args(argv)
    try:
        inventory = write_requests(args.generation, args.cohort, args.pilot_acceptance, args.shard_count)
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
