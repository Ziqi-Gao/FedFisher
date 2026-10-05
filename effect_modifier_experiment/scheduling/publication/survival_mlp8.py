"""Publish reviewed MLP8 requests; never register or submit central work."""
import argparse
import json
from pathlib import Path

from . import require_canonical_source
from ...infrastructure.provenance import compute_provenance
from ...infrastructure.requests import sha256, atomic_create, clean_provenance, publish_bundle
from ...survival_mlp8 import protocol
from ...survival_mlp8.requests import PROJECT_ROOT, REVIEW_SOURCES, build_requests, bundle_name, inventory_name, review_sources

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')


def current_provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def historical_binding():
    from ...survival_pooled.inputs import historical_binding as bind, validate_historical_inputs
    validate_historical_inputs()
    return bind()


def acceptance_binding(cohort, path, provenance, *, discovery=False):
    if cohort == 'pilot':
        if path is not None:
            raise ValueError('pilot cannot supply a formal prerequisite')
        return None
    if path is None:
        raise ValueError('formal requests require same-source MLP8 pilot acceptance')
    from ...survival_mlp8.acceptance import default_acceptance_path, verify_pilot_acceptance
    path = Path(path)
    if path != default_acceptance_path(provenance, discovery=discovery):
        raise ValueError('formal requests require the canonical MLP8 proof')
    proof = verify_pilot_acceptance(path, discovery=discovery)
    return dict(path=str(path), sha256=sha256(path.read_bytes()), evidence=proof)


def write_requests(generation=1, cohort='pilot', pilot_acceptance=None, shard_count=None, *, discovery=False):
    if discovery:
        from ...survival_mlp8 import discovery_protocol as recipe
    else:
        recipe = protocol
    specs = build_requests(generation, cohort, shard_count, discovery=discovery)
    source = current_provenance()
    accepted = acceptance_binding(cohort, pilot_acceptance, source, discovery=discovery)
    historical, sources = historical_binding(), review_sources(discovery=discovery)
    pilot = cohort == 'pilot'
    units = 4 if pilot else 600

    def stable_sources():
        if historical_binding() != historical:
            raise ValueError('historical fingerprint reference changed')
        return review_sources(discovery=discovery)

    metadata = dict(schema_version=1, bundle=('fedfisher-survival-mlp8-v2-' if discovery else 'fedfisher-survival-mlp8-v1-') + cohort,
        generation=generation, task=recipe.PILOT_TASK if pilot else recipe.FORMAL_TASK,
        scientific_cell_count=units, fitted_model_count=4 * units,
        coverage=dict(scenarios=list(protocol.SCENARIOS), seeds=list(protocol.PILOT_SEEDS if pilot else protocol.FORMAL_SEEDS),
            models=list(protocol.TAGS), pseudo_scopes=list(protocol.PSEUDO_SCOPES)),
        historical_inputs=historical,
        formal_plan=dict(task=recipe.FORMAL_TASK, seed_count=300, scientific_cell_count=600,
            fitted_model_count=2400, default_request_count=32, default_shard_count=16,
            publication_authorized=not pilot, gate='fresh same-source MLP8 pilot and operator profile review'),
        operator_gate='review disabled proposal/schema/fixtures; register pilot centrally, validate and submit exact inventory requests',
        freeze_rule='keep canonical source unchanged through pilot and its acceptance')
    result = publish_bundle(specs, bundle=OUTBOX_ROOT / bundle_name(generation, cohort, discovery=discovery),
        inventory_path=REPORT_ROOT / inventory_name(generation, cohort, discovery=discovery), metadata=metadata,
        provenance=source, sources=sources, acceptance=accepted, check_provenance=current_provenance,
        check_sources=stable_sources, check_acceptance=lambda: acceptance_binding(cohort, pilot_acceptance, source, discovery=discovery),
        create=atomic_create)
    if current_provenance() != source or stable_sources() != sources or acceptance_binding(cohort, pilot_acceptance, source, discovery=discovery) != accepted:
        raise ValueError('bindings changed after publication; preserve inventory and do not admit it')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    parser.add_argument('--shard-count', type=int, choices=(8, 16))
    parser.add_argument('--pilot-acceptance', type=Path)
    parser.add_argument('--discovery', action='store_true', help='V2: probability checks followed by Top-2/AUROC discovery')
    args = parser.parse_args(argv)
    result = write_requests(args.generation, args.cohort, args.pilot_acceptance, args.shard_count, discovery=args.discovery)
    print(json.dumps(dict(inventory=result['inventory_path'], request_count=result['request_count'],
        inventory_sha256=sha256(Path(result['inventory_path']).read_bytes()), submitted=False), sort_keys=True))
    return 0
