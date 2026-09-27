#!/usr/bin/env python3
"""Publish immutable model/signal requests for later operator review only."""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
if __name__ == '__main__':
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.model_signal_ablation import protocol
from effect_modifier_experiment.infrastructure.provenance import compute_provenance
from effect_modifier_experiment.infrastructure.requests import (
    canonical, encoded, sha256, safe_path, atomic_create, clean_provenance,
    review_sources, publish_bundle)

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')
REVIEW_SOURCES = {name: PROJECT_ROOT / path for name, path in {
    'proposal': 'effect_modifier_experiment/server_scheduler/model-signal-ablation-v1/registration-proposal.toml',
    'pilot_schema': 'effect_modifier_experiment/server_scheduler/model-signal-ablation-v1/pilot-parameters.schema.json',
    'shard_schema': 'effect_modifier_experiment/server_scheduler/model-signal-ablation-v1/shard-parameters.schema.json',
    'adapter': 'scripts/fedfisher_server_scheduler_adapter.py',
    'protocol': 'effect_modifier_experiment/model_signal_ablation/protocol.py',
    'request_builder': 'scripts/prepare_model_signal_requests.py',
    'acceptance': 'effect_modifier_experiment/model_signal_ablation/acceptance.py',
    'scientific_guide': 'effect_modifier_experiment/model_signal_ablation/README.md',
}.items()}


def bundle_name(generation=1, cohort='pilot'):
    if type(generation) is not int or generation < 1 or cohort not in ('pilot', 'formal'):
        raise ValueError('positive integer generation and pilot/formal cohort required')
    return 'fedfisher-ms-v1-%s-cohort-p%d' % (cohort, generation)


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def build_requests(generation=1, cohort='pilot', shard_count=None):
    """Pure exact-scope specifications; does not run science or write requests."""
    bundle = bundle_name(generation, cohort)
    pilot = cohort == 'pilot'
    if pilot and shard_count is not None:
        raise ValueError('pilot shard count is fixed')
    count = 1 if pilot else (16 if shard_count is None else shard_count)
    if type(count) is not int or count not in ((1,) if pilot else (8, 16)):
        raise ValueError('formal shard count must be 8 or 16')
    expected_cells = {'dslassocov'} | {
        'fedavg_%s_e%d_%s' % (model, epochs, stopping)
        for model in ('small', 'shallow', 'deep') for epochs in (1, 5, 10, 30)
        for stopping in ('patience', 'budget30')} | {
        'fedfisher_%s_%s' % (kind, model)
        for kind in ('diag', 'kfac') for model in ('small', 'shallow', 'deep')}
    expected_settings = dict(small_mlp=(2.0, {cell for cell in expected_cells if '_small' in cell}),
        weak_signal_1=(1.0, expected_cells), weak_signal_05=(0.5, expected_cells))
    if (set(protocol.CELLS) != expected_cells or set(protocol.SETTINGS) != set(expected_settings)
            or set(protocol.VARIANTS) != {'control', 'site_gaussian_shift', 'local_permutation',
                                         'site_gaussian_local_permutation'}
            or tuple(protocol.SPLITS) != ('iid', 'noniid')):
        raise ValueError('model/signal scope changed from the reviewed experiment matrix')
    for setting, (strength, cells) in expected_settings.items():
        recipe = protocol.SETTINGS[setting]
        if (type(recipe['signal_strength']) is not float or recipe['signal_strength'] != strength
                or set(recipe['cells']) != cells or len(recipe['cells']) != len(cells)):
            raise ValueError('model/signal setting changed from the reviewed experiment matrix')
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    specs = []
    for setting, recipe in protocol.SETTINGS.items():
        for cell in recipe['cells']:
            for variant in protocol.VARIANTS:
                for index in range(count):
                    parameters = dict(setting=setting, cell=cell, variant=variant,
                                      shard_index=index, shard_count=count)
                    protocol.validate_parameters(task, parameters)
                    job_id = '%s-%s-%s-%s-s%02dof%02d' % (
                        bundle, setting.replace('_', '-'), cell.replace('_', '-'),
                        variant.replace('_', '-'), index, count)
                    if re.fullmatch(r'fedfisher-ms-[A-Za-z0-9_.-]{1,115}', job_id) is None:
                        raise ValueError('generation exceeds scheduler job-identity limits')
                    request = dict(schema_version=2, job_id=job_id, project='FedFisher',
                                   task=task, parameters=parameters, priority=0)
                    config = json.loads(canonical(protocol.run_config(task, parameters)))
                    seeds = list(protocol.seed_sequence(task, parameters))
                    if seeds != ([0] if pilot else list(range(index, 300, count))):
                        raise ValueError('pilot seed zero or deterministic formal seed coverage changed')
                    specs.append(dict(filename=job_id + '.job-request.json', request=request,
                        request_sha256=sha256(encoded(request)), run_id=protocol.run_id(task, parameters),
                        configuration=config, configuration_sha256=sha256(canonical(config).encode()),
                        seed_sequence=seeds, splits=list(protocol.SPLITS)))
    if len(specs) != 288 * count or len({item['filename'] for item in specs}) != len(specs):
        raise ValueError('model/signal request coverage or job uniqueness changed')
    return tuple(specs)


def _provenance():
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def _source_hashes():
    return review_sources(REVIEW_SOURCES, PROJECT_ROOT)


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


if __name__ == '__main__':
    raise SystemExit(main())
