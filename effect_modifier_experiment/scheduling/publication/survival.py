"""Publish immutable survival requests for ServerScheduler operator review.

Pilot publication requires clean frozen source. Formal publication additionally
requires freshly verified same-source pilot acceptance. No central commands,
scientific fitting, resource discovery or background execution occur here.
"""

import argparse
from effect_modifier_experiment.scheduling.publication import require_canonical_source

import json

import os

from pathlib import Path

import re

import subprocess


from effect_modifier_experiment.infrastructure.provenance import compute_provenance

from effect_modifier_experiment.infrastructure.requests import (
    canonical, encoded, sha256, safe_path, pending_path as _pending,
    preflight_new as _preflight_new, atomic_create, clean_provenance,
    review_sources as source_bindings, publish_bundle,
)

from effect_modifier_experiment.survival import protocol

SCRATCH_ROOT = Path('/scr/del6500/FedFisher')

OUTBOX_ROOT = SCRATCH_ROOT / 'scheduler/outbox'

REPORT_ROOT = SCRATCH_ROOT / 'scheduler/reports'

from effect_modifier_experiment.survival.requests import (
    PROJECT_ROOT,
    REVIEW_SOURCES,
    EXPECTED_CELLS,
    EXPECTED_SCENARIOS,
    _generation,
    _cohort,
    bundle_name,
    inventory_name,
    build_requests,
    review_sources,
)


def current_provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def acceptance_binding(cohort, path, provenance):
    if cohort == 'pilot':
        if path is not None:
            raise ValueError('pilot publication must not take a prior-pilot override')
        return None
    if path is None:
        raise ValueError('formal requests require successful same-source pilot acceptance first')
    from effect_modifier_experiment.survival.acceptance import (
        default_acceptance_path, verify_pilot_acceptance,
    )
    path = Path(path)
    if path != default_acceptance_path(provenance):
        raise ValueError('formal requests must bind the canonical source-specific pilot acceptance')
    evidence = verify_pilot_acceptance(path)
    if evidence.get('provenance') != provenance:
        raise ValueError('pilot source differs from current clean source')
    return dict(path=str(path), sha256=sha256(path.read_bytes()), evidence=evidence)


def write_requests(generation=1, cohort='pilot', pilot_acceptance=None, shard_count=None):
    """Publish inventory last; a partial bundle is never ready for submission."""
    _generation(generation)
    _cohort(cohort)
    before = current_provenance()
    accepted = acceptance_binding(cohort, pilot_acceptance, before)
    specs = build_requests(generation, cohort, shard_count)
    sources = review_sources()
    pilot = cohort == 'pilot'
    metadata = dict(
        schema_version=1, bundle='fedfisher-survival-v1-' + cohort, generation=generation,
        task=protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK,
        scientific_cell_count=10 if pilot else 3000,
        coverage=dict(method_cells=5, scenarios=2, seeds=[0] if pilot else list(range(300))),
        formal_plan=dict(task=protocol.FORMAL_TASK, seed_start=0, seed_end=299,
                         allowed_shard_counts=[8, 16], default_shard_count=16,
                         default_request_count=160, scientific_cell_count=3000,
                         publication_authorized=not pilot,
                         gate='complete same-source pilot scientific and paired acceptance plus operator profile review'),
        operator_gate='verify exact source and request hashes, review/register the admitted stage, then validate and submit through ServerScheduler',
        freeze_rule='keep canonical source unchanged through pilot and formal execution and scientific acceptance',
    )
    return publish_bundle(
        specs, bundle=OUTBOX_ROOT / bundle_name(generation, cohort),
        inventory_path=REPORT_ROOT / inventory_name(generation, cohort),
        metadata=metadata, provenance=before, sources=sources, acceptance=accepted,
        check_provenance=current_provenance, check_sources=review_sources,
        check_acceptance=lambda: acceptance_binding(cohort, pilot_acceptance, before),
        create=atomic_create,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    parser.add_argument('--pilot-acceptance', type=Path)
    parser.add_argument('--shard-count', type=int, choices=(8, 16))
    args = parser.parse_args(argv)
    try:
        result = write_requests(args.generation, args.cohort, args.pilot_acceptance, args.shard_count)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print('request_generation_error=%s' % error)
        return 2
    path = Path(result['inventory_path'])
    print(json.dumps(dict(inventory_path=str(path), inventory_sha256=sha256(path.read_bytes()),
                          project_head=result['project_head'], request_count=result['request_count'],
                          scientific_cell_count=result['scientific_cell_count'], submitted=False), indent=2))
    return 0
