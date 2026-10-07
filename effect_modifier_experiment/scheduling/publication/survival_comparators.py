"""Publish the fixed ten-seed comparator cohort into the operator outbox."""
import argparse
import json
from pathlib import Path

from . import require_canonical_source
from ...infrastructure.provenance import compute_provenance
from ...infrastructure.requests import atomic_create, clean_provenance, publish_bundle, sha256
from ...survival_comparators import protocol
from ...survival_comparators.inputs import historical_binding, validate_inputs
from ...survival_comparators.requests import PROJECT_ROOT, build_requests, bundle_name, review_sources

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')


def current_provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def write_requests(generation=1):
    specs, source = build_requests(generation), current_provenance()
    validate_inputs()
    inputs, sources = historical_binding(), review_sources()

    def stable_sources():
        validate_inputs()
        if historical_binding() != inputs:
            raise ValueError('historical comparator inputs changed')
        return review_sources()

    metadata = dict(schema_version=1, bundle='fedfisher-survival-comparators-v1-10seed',
        generation=generation, task=protocol.PILOT_TASK, scientific_cell_count=20,
        fitted_model_count=20, replayed_model_count=20, referenced_baseline_model_count=80,
        coverage=dict(scenarios=list(protocol.SCENARIOS), seeds=list(protocol.SEEDS), models=list(protocol.TAGS)),
        historical_inputs=inputs, formal_expansion_authorized=False,
        operator_gate='review disabled proposal/schema/fixtures; register bounded pilot and submit exact twenty requests',
        freeze_rule='keep canonical source unchanged through this small cohort and saved-artifact acceptance')
    return publish_bundle(specs, bundle=OUTBOX_ROOT / bundle_name(generation),
        inventory_path=REPORT_ROOT / (bundle_name(generation) + '.inventory.json'),
        metadata=metadata, provenance=source, sources=sources, acceptance=None,
        check_provenance=current_provenance, check_sources=stable_sources,
        check_acceptance=lambda: None, create=atomic_create)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    args = parser.parse_args(argv)
    result = write_requests(args.generation)
    print(json.dumps(dict(inventory=result['inventory_path'], request_count=result['request_count'],
        inventory_sha256=sha256(Path(result['inventory_path']).read_bytes()), submitted=False), sort_keys=True))
    return 0
