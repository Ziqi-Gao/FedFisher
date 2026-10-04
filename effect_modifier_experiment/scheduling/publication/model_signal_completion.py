"""Publish reviewed new-source completion requests; central admission is external."""
import argparse
from pathlib import Path

from . import require_canonical_source
from ...model_signal_ablation import completion_protocol as protocol
from ...model_signal_ablation.completion_requests import (
    PROJECT_ROOT, REVIEW_SOURCES, build_requests, bundle_name, inventory_name, review_sources,
)
from ...infrastructure.provenance import compute_provenance
from ...infrastructure.requests import clean_provenance, publish_bundle

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')


def _provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def _closeout():
    from ...model_signal_ablation.completion_acceptance import verify_closeout
    return verify_closeout()


def _acceptance_binding(cohort, path, provenance, closeout):
    if cohort == 'pilot':
        if path is not None:
            raise ValueError('pilot acceptance applies only to formal publication')
        return None
    if path is None:
        raise ValueError('formal completion requires actual same-source completion pilot acceptance')
    from ...model_signal_ablation import completion_acceptance as gate
    proof = gate.verify_pilot_acceptance(path)
    if (proof.get('passed') is not True or proof.get('formal_eligible') is not True
            or proof.get('provenance') != provenance or proof.get('closeout') != closeout['binding']
            or proof.get('validated_request_count') != 8 or proof.get('validated_unit_count') != 16):
        raise ValueError('completion pilot proof has incompatible source, scope or archive')
    return dict(gate._binding(path), evidence=proof)


def write_requests(generation=1, cohort='pilot', pilot_acceptance=None):
    before = _provenance()
    closeout = _closeout()
    specs = build_requests(generation, cohort)
    accepted = _acceptance_binding(cohort, pilot_acceptance, before, closeout)
    sources = review_sources()
    pilot = cohort == 'pilot'
    metadata = dict(schema_version=1, bundle=protocol.PROTOCOL + '-' + cohort,
        generation=generation, task=protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK,
        closeout=closeout['binding'], scientific_cell_count=16 if pilot else 896,
        target_scientific_cell_count=172800, retained_request_count=4584,
        retained_scientific_cell_count=171904, retained_complete_units=171898,
        retained_noncompleted_units=6,
        coverage=dict(pilot_requests=8, pilot_units=16, replacement_formal_requests=24,
                      replacement_formal_units=896, original_formal_requests=4608),
        formal_plan=dict(task=protocol.FORMAL_TASK, shard_count=16, request_count=24,
            scientific_cell_count=896, publication_authorized=not pilot,
            gate='same-source 16-unit numerical parity and paired pilot acceptance; operator profile review'),
        operator_gate='additive new task review and registration; admit pilot before formal',
        freeze_rule='keep canonical source unchanged through completion pilot, formal execution and acceptance',
        scientific_completion_rule='training caps remain scientific noncompletion; partial and invalid output fails closed')

    def recheck():
        current = _closeout()
        if current['binding'] != closeout['binding']:
            raise ValueError('sealed historical anchor changed during publication')
        checked = _acceptance_binding(cohort, pilot_acceptance, before, current)
        if _provenance() != before or review_sources() != sources:
            raise ValueError('completion source changed during final archive/pilot verification')
        return checked

    return publish_bundle(specs, bundle=OUTBOX_ROOT / bundle_name(generation, cohort),
        inventory_path=REPORT_ROOT / inventory_name(generation, cohort), metadata=metadata,
        provenance=before, sources=sources, acceptance=accepted,
        check_provenance=_provenance, check_sources=review_sources, check_acceptance=recheck)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    parser.add_argument('--pilot-acceptance', type=Path)
    args = parser.parse_args(argv)
    inventory = write_requests(args.generation, args.cohort, args.pilot_acceptance)
    print(inventory['inventory_path'])
    return 0
