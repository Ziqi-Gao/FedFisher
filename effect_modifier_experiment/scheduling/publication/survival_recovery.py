"""Create immutable recovery outbox requests; never submit central jobs."""

import argparse
from effect_modifier_experiment.scheduling.publication import require_canonical_source

import json

from pathlib import Path

import re


from effect_modifier_experiment.infrastructure.provenance import compute_provenance

from effect_modifier_experiment.infrastructure.requests import (
    canonical, encoded, sha256, clean_provenance, review_sources as source_bindings,
    atomic_create, publish_bundle)

from effect_modifier_experiment.survival import recovery_protocol as recovery

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')

REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')

from effect_modifier_experiment.survival.recovery_requests import (
    PROJECT_ROOT,
    REVIEW_SOURCES,
    bundle_prefix,
    bundle_name,
    build_requests,
    review_sources,
)


def current_provenance():
    require_canonical_source(__file__, PROJECT_ROOT)
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def acceptance_binding(cohort, path, provenance, epoch=1):
    from effect_modifier_experiment.survival import recovery_acceptance as gate
    gate.check_original_closeout()
    if epoch >= 2:
        gate.check_repair_closeout()
    if epoch == 3:
        gate.check_integration_reference()
    if cohort == 'diagnostic':
        if path is not None:
            raise ValueError('diagnostic publication must not take a prerequisite override')
        return None
    if path is None or Path(path) != gate.default_acceptance_path(provenance, epoch=epoch):
        raise ValueError('formal publication requires actual source-bound diagnostic acceptance first')
    proof = gate.check_current_diagnostic_gate(epoch=epoch)
    if proof['provenance'] != provenance:
        raise ValueError('diagnostic source differs from current clean source')
    return dict(path=str(path), sha256=sha256(Path(path).read_bytes()), evidence=proof)


def write_requests(generation=1, cohort='diagnostic', diagnostic_acceptance=None, shard_count=None, epoch=1):
    name = bundle_name(generation, cohort, epoch)
    before = current_provenance()
    accepted = acceptance_binding(cohort, diagnostic_acceptance, before, epoch)
    recipe = None if accepted is None else accepted['evidence']['selected_recipe']
    specs = build_requests(generation, cohort, shard_count, recipe=recipe, epoch=epoch)
    metadata = dict(schema_version=1, bundle=bundle_prefix(epoch) + '-' + cohort,
        generation=generation, task=specs[0]['request']['task'], scientific_cell_count=len(specs) if cohort == 'diagnostic' else 600,
        shard_count=None if cohort == 'diagnostic' else (16 if shard_count is None else shard_count),
        original_closeout=recovery.ORIGINAL_CLOSEOUT, case_inventory=recovery.CASE_INVENTORY,
        selected_recipe=recipe, formal_publication_authorized=cohort == 'formal',
        operator_gate='review disabled proposal, fixtures and profile; centrally admit diagnostic first; '
                      'formal requires actual same-source diagnostic acceptance and operator cost review',
        freeze_rule='keep canonical source unchanged through diagnostic, formal and full scientific acceptance')
    if epoch >= 2:
        metadata.update(epoch=epoch, repair_closeout=recovery.REPAIR_CLOSEOUT,
                        r1_diagnostic_acceptance=recovery.R1_DIAGNOSTIC_ACCEPTANCE)
    if epoch == 3:
        metadata['r2_diagnostic_acceptance'] = recovery.R2_DIAGNOSTIC_ACCEPTANCE
    return publish_bundle(specs, bundle=OUTBOX_ROOT / name,
        inventory_path=REPORT_ROOT / (name + '.inventory.json'), metadata=metadata,
        provenance=before, sources=review_sources(epoch), acceptance=accepted,
        check_provenance=current_provenance, check_sources=lambda: review_sources(epoch),
        check_acceptance=lambda: acceptance_binding(cohort, diagnostic_acceptance, before, epoch), create=atomic_create)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--epoch', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--cohort', choices=('diagnostic', 'formal'), default='diagnostic')
    parser.add_argument('--diagnostic-acceptance', type=Path)
    parser.add_argument('--shard-count', type=int, choices=(8, 16))
    args = parser.parse_args(argv)
    result = write_requests(args.generation, args.cohort, args.diagnostic_acceptance, args.shard_count, args.epoch)
    print(json.dumps(dict(inventory=result['inventory_path'], request_count=result['request_count'],
                         scientific_cell_count=result['scientific_cell_count'], submitted=False), sort_keys=True))
    return 0
