"""Publish immutable model/signal requests for later operator review only."""

import argparse
from effect_modifier_experiment.scheduling.publication import require_canonical_source

import json

from pathlib import Path

import re


from effect_modifier_experiment.model_signal_ablation import protocol

from effect_modifier_experiment.infrastructure.provenance import compute_provenance

from effect_modifier_experiment.infrastructure.requests import (
    canonical, encoded, sha256, safe_path, atomic_create, clean_provenance,
    review_sources, publish_bundle)

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')

REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')

from effect_modifier_experiment.model_signal_ablation.requests import (
    PROJECT_ROOT,
    REVIEW_SOURCES,
    bundle_name,
    inventory_name,
    build_requests,
    _source_hashes,
    review_sources,
)


def _provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def _acceptance_binding(cohort, path, provenance):
    if cohort == 'pilot':
        if path is not None:
            raise ValueError('pilot acceptance is only valid for formal publication')
        return None
    if path is None:
        raise ValueError('formal publication requires complete same-source pilot acceptance')
    from effect_modifier_experiment.model_signal_ablation.acceptance import verify_pilot_acceptance
    proof = verify_pilot_acceptance(path)
    if proof.get('provenance') != provenance:
        raise ValueError('pilot acceptance differs from current source provenance')
    return dict(path=str(Path(path)), sha256=sha256(Path(path).read_bytes()), evidence=proof)


def write_requests(generation=1, cohort='pilot', pilot_acceptance=None, shard_count=None):
    before = _provenance()
    specs = build_requests(generation, cohort, shard_count)
    accepted = _acceptance_binding(cohort, pilot_acceptance, before)
    sources = _source_hashes()
    pilot = cohort == 'pilot'
    metadata = dict(schema_version=1, bundle='model-signal-ablation-v1-' + cohort,
        generation=generation, task=protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK,
        scientific_cell_count=576 if pilot else 172800,
        coverage=dict(settings={name: dict(signal_strength=recipe['signal_strength'], cells=list(recipe['cells']))
            for name, recipe in protocol.SETTINGS.items()}, setting_method_combinations=72,
            variants=4, seeds=[0] if pilot else list(range(300)), splits=list(protocol.SPLITS)),
        formal_plan=dict(task=protocol.FORMAL_TASK, seeds=[0, 299], allowed_shard_counts=[8, 16],
            scientific_cell_count=172800, publication_authorized=not pilot,
            gate='complete same-source pilot scientific and paired acceptance plus operator profile review'),
        operator_gate='review/register admitted stage, verify exact clean source and request hashes, then submit through ServerScheduler',
        freeze_rule='keep canonical source unchanged through pilot/formal execution and paired scientific acceptance')
    return publish_bundle(specs, bundle=OUTBOX_ROOT / bundle_name(generation, cohort),
        inventory_path=REPORT_ROOT / inventory_name(generation, cohort), metadata=metadata,
        provenance=before, sources=sources, acceptance=accepted, check_provenance=_provenance,
        check_sources=_source_hashes,
        check_acceptance=lambda: _acceptance_binding(cohort, pilot_acceptance, before), create=atomic_create)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    parser.add_argument('--shard-count', type=int, choices=(8, 16))
    parser.add_argument('--pilot-acceptance', type=Path)
    args = parser.parse_args(argv)
    inventory = write_requests(args.generation, args.cohort, args.pilot_acceptance, args.shard_count)
    print(json.dumps(dict(inventory_path=inventory['inventory_path'], request_count=inventory['request_count'],
        scientific_cell_count=inventory['scientific_cell_count'], submitted=False), sort_keys=True))
    return 0
