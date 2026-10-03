"""Publish immutable recovery review bundles; never submit scheduler jobs."""

import argparse
from effect_modifier_experiment.scheduling.publication import require_canonical_source

import json

from pathlib import Path

import re


from effect_modifier_experiment.model_signal_ablation import recovery_protocol as protocol

from effect_modifier_experiment.model_signal_ablation import protocol as original

from effect_modifier_experiment.infrastructure.provenance import compute_provenance

from effect_modifier_experiment.infrastructure.requests import (
    canonical, encoded, sha256, safe_path, atomic_create, clean_provenance,
    review_sources as _review_sources, publish_bundle)

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')

REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')

from effect_modifier_experiment.model_signal_ablation.recovery_requests import (
    PROJECT_ROOT,
    REVIEW_SOURCES,
    bundle_name,
    inventory_name,
    _scope,
    _parameters_key,
    _formal_source_rows,
    build_requests,
    review_sources,
    _source_hashes,
)


def _provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def _closeout(path=None):
    from effect_modifier_experiment.model_signal_ablation.recovery_acceptance import verify_closeout
    return verify_closeout(path)


def _acceptance_binding(cohort, path, provenance, closeout):
    if cohort == 'pilot':
        if path is not None:
            raise ValueError('pilot acceptance is only valid for formal publication')
        return None
    if path is None:
        raise ValueError('formal publication requires complete same-source recovery pilot acceptance')
    from effect_modifier_experiment.model_signal_ablation.recovery_acceptance import verify_pilot_acceptance
    proof = verify_pilot_acceptance(path)
    if (proof.get('provenance') != provenance or proof.get('closeout') != closeout['binding']
            or proof.get('passed') is not True or proof.get('formal_eligible') is not True
            or proof.get('validated_request_count') != 290 or proof.get('validated_unit_count') != 588):
        raise ValueError('recovery pilot acceptance differs from source, closeout or required coverage')
    return dict(path=str(Path(path)), sha256=sha256(Path(path).read_bytes()), evidence=proof)


def write_requests(generation=1, cohort='pilot', pilot_acceptance=None, closeout_path=None):
    before = _provenance()
    closeout = _closeout(closeout_path)
    specs = build_requests(generation, cohort, closeout)
    accepted = _acceptance_binding(cohort, pilot_acceptance, before, closeout)
    sources = _source_hashes()
    pilot = cohort == 'pilot'
    metadata = dict(schema_version=1, bundle=protocol.PROTOCOL + '-' + cohort,
        generation=generation, task=protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK,
        closeout=closeout['binding'], scientific_cell_count=588 if pilot else 167962,
        target_scientific_cell_count=172800, retained_request_count=129,
        retained_scientific_cell_count=4838,
        coverage=dict(baseline_pilot_requests=288, diagnostic_pilot_requests=2,
            diagnostic_seeds=list(protocol.DIAGNOSTIC_SEEDS), splits=list(protocol.SPLITS),
            replacement_formal_requests=4479, original_formal_requests=4608),
        formal_plan=dict(task=protocol.FORMAL_TASK, shard_count=16, request_count=4479,
            scientific_cell_count=167962, publication_authorized=not pilot,
            gate='same-source 290-request pilot acceptance, old/new numerical parity and operator profile review'),
        operator_gate='review and register the two new tasks before admitting any request; preserve all old tasks',
        freeze_rule='keep canonical source unchanged through recovery pilot/formal execution and acceptance',
        scientific_completion_rule='execution may record strictly validated noncompletion; never count it as scientific success')

    def recheck():
        current = _closeout(closeout_path)
        if (current['binding'] != closeout['binding']
                or current['retained_request_ids'] != closeout['retained_request_ids']
                or current['old_inventory'] != closeout['old_inventory']):
            raise ValueError('source closeout changed during publication')
        return _acceptance_binding(cohort, pilot_acceptance, before, current)

    return publish_bundle(specs, bundle=OUTBOX_ROOT / bundle_name(generation, cohort),
        inventory_path=REPORT_ROOT / inventory_name(generation, cohort), metadata=metadata,
        provenance=before, sources=sources, acceptance=accepted, check_provenance=_provenance,
        check_sources=_source_hashes, check_acceptance=recheck, create=atomic_create)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    parser.add_argument('--pilot-acceptance', type=Path)
    parser.add_argument('--closeout', type=Path)
    args = parser.parse_args(argv)
    inventory = write_requests(args.generation, args.cohort, args.pilot_acceptance, args.closeout)
    print(json.dumps(dict(inventory_path=inventory['inventory_path'], request_count=inventory['request_count'],
        scientific_cell_count=inventory['scientific_cell_count'], submitted=False), sort_keys=True))
    return 0
