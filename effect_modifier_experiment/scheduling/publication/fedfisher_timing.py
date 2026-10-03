"""Publish an immutable, unsubmitted original FedFisher timing review bundle.

Only request JSON is written under the project outbox; inventory evidence is
written under project scratch reports. This program never invokes a scheduler,
registers a task, fits a model, or admits work. A complete inventory is published
last. Any earlier failure leaves partial files for inspection and prevents a
same-generation retry from overwriting them.
"""

import argparse
from effect_modifier_experiment.scheduling.publication import require_canonical_source

import json

import os

from pathlib import Path

import re

import subprocess


from effect_modifier_experiment.fedfisher_timing import protocol

from effect_modifier_experiment.infrastructure.provenance import compute_provenance

from effect_modifier_experiment.infrastructure.requests import (
    canonical as _canonical, encoded as _encoded, sha256 as _sha256,
    safe_path as _safe_path, pending_path as _pending_path,
    preflight_new as _preflight_new, atomic_create as _atomic_create,
    clean_provenance, review_sources, publish_bundle,
)

SCRATCH_ROOT = Path("/scr/del6500/FedFisher")

OUTBOX_ROOT = SCRATCH_ROOT / "scheduler/outbox"

REPORT_ROOT = SCRATCH_ROOT / "scheduler/reports"

from effect_modifier_experiment.fedfisher_timing.requests import (
    PROJECT_ROOT,
    REVIEW_SOURCES,
    EXPECTED_CELLS,
    EXPECTED_VARIANTS,
    _generation,
    _cohort,
    bundle_name,
    inventory_name,
    build_requests,
    _source_hashes,
    review_sources,
)


def _provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance())


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
