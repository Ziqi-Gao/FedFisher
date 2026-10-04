"""Publish a fixed pooled-survival bundle for operator registration/admission.

This command performs no model fitting, model inference or central scheduler
operation.  Historical input hashes and current clean source are verified both
before publication and before the final immutable inventory becomes visible.
"""
import argparse
import json
from pathlib import Path
import subprocess

from . import require_canonical_source
from ...infrastructure.provenance import compute_provenance
from ...infrastructure.requests import (
    sha256, atomic_create, clean_provenance, publish_bundle,
)
from ...survival_pooled import protocol
from ...survival_pooled.requests import (
    PROJECT_ROOT, REVIEW_SOURCES, _generation, _cohort, bundle_name,
    inventory_name, build_requests, review_sources,
)

SCRATCH_ROOT = Path('/scr/del6500/FedFisher')
OUTBOX_ROOT = SCRATCH_ROOT / 'scheduler/outbox'
REPORT_ROOT = SCRATCH_ROOT / 'scheduler/reports'


def current_provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def historical_binding():
    from ...survival_pooled.inputs import historical_binding as bind, validate_historical_inputs
    validate_historical_inputs()
    return bind()


def acceptance_binding(cohort, path, provenance):
    if cohort == 'pilot':
        if path is not None:
            raise ValueError('pilot publication cannot override historical pilot admission')
        return None
    if path is None:
        raise ValueError('formal requests require successful same-source pilot acceptance first')
    from ...survival_pooled.acceptance import default_acceptance_path, verify_pilot_acceptance
    path = Path(path)
    if path != default_acceptance_path(provenance):
        raise ValueError('formal requests must bind the canonical source-specific pilot acceptance')
    evidence = verify_pilot_acceptance(path)
    if evidence.get('provenance') != provenance:
        raise ValueError('pooled survival pilot source differs from current clean source')
    return dict(path=str(path), sha256=sha256(path.read_bytes()), evidence=evidence)


def write_requests(generation=1, cohort='pilot', pilot_acceptance=None, shard_count=None):
    _generation(generation)
    _cohort(cohort)
    before = current_provenance()
    accepted = acceptance_binding(cohort, pilot_acceptance, before)
    historical = historical_binding()
    sources = review_sources()
    specs = build_requests(generation, cohort, shard_count)
    pilot = cohort == 'pilot'

    def stable_sources():
        if historical_binding() != historical:
            raise ValueError('historical survival inputs changed during publication')
        return review_sources()

    metadata = dict(
        schema_version=1, bundle='fedfisher-survival-pooled-v1-' + cohort,
        generation=generation, task=protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK,
        scientific_cell_count=12 if pilot else 1800,
        pooled_fit_count=24 if pilot else 3600,
        coverage=dict(families=list(protocol.FAMILIES), scenarios=list(protocol.SCENARIOS),
                      seeds=list(protocol.PILOT_SEEDS if pilot else protocol.FORMAL_SEEDS),
                      pseudo_scopes=list(protocol.PSEUDO_SCOPES)),
        historical_inputs=historical,
        formal_plan=dict(task=protocol.FORMAL_TASK, seed_start=0, seed_end=299,
                         allowed_shard_counts=[8, 16], default_shard_count=16,
                         default_request_count=96, scientific_cell_count=1800,
                         pooled_fit_count=3600, publication_authorized=not pilot,
                         gate='complete same-source pilot scientific and paired acceptance plus operator profile review'),
        operator_gate='review additive disabled proposal, schemas and fixtures; register/admit pilot, then validate and submit exact inventory requests through ServerScheduler',
        freeze_rule='keep canonical source unchanged through pilot and formal execution and scientific acceptance',
    )
    return publish_bundle(
        specs, bundle=OUTBOX_ROOT / bundle_name(generation, cohort),
        inventory_path=REPORT_ROOT / inventory_name(generation, cohort),
        metadata=metadata, provenance=before, sources=sources, acceptance=accepted,
        check_provenance=current_provenance, check_sources=stable_sources,
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
