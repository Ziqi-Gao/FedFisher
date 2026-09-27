#!/usr/bin/env python3
"""Publish immutable recovery review bundles; never submit scheduler jobs."""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
if __name__ == '__main__':
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.model_signal_ablation import recovery_protocol as protocol
from effect_modifier_experiment.model_signal_ablation import protocol as original
from effect_modifier_experiment.infrastructure.provenance import compute_provenance
from effect_modifier_experiment.infrastructure.requests import (
    canonical, encoded, sha256, safe_path, atomic_create, clean_provenance,
    review_sources as _review_sources, publish_bundle)

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')
REVIEW_SOURCES = {name: PROJECT_ROOT / path for name, path in {
    'proposal': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/registration-proposal.toml',
    'pilot_schema': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/pilot-parameters.schema.json',
    'shard_schema': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/shard-parameters.schema.json',
    'adapter': 'scripts/fedfisher_server_scheduler_adapter.py',
    'protocol': 'effect_modifier_experiment/model_signal_ablation/recovery_protocol.py',
    'runner': 'effect_modifier_experiment/model_signal_ablation/recovery.py',
    'noncompletion': 'effect_modifier_experiment/causal_comparison/noncompletion.py',
    'comparison_runner': 'effect_modifier_experiment/causal_comparison/runner.py',
    'request_builder': 'scripts/prepare_model_signal_recovery_requests.py',
    'acceptance': 'effect_modifier_experiment/model_signal_ablation/recovery_acceptance.py',
    'validator_cli': 'scripts/validate_model_signal_recovery.py',
    'scientific_guide': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/README.md',
}.items()}


def bundle_name(generation=1, cohort='pilot'):
    if type(generation) is not int or generation < 1 or cohort not in ('pilot', 'formal'):
        raise ValueError('positive integer generation and pilot/formal cohort required')
    return 'fedfisher-ms-v2-%s-cohort-p%d' % (cohort, generation)


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def _scope():
    expected = {'dslassocov'} | {
        'fedavg_%s_e%d_%s' % (model, epochs, stopping)
        for model in ('small', 'shallow', 'deep') for epochs in (1, 5, 10, 30)
        for stopping in ('patience', 'budget30')} | {
        'fedfisher_%s_%s' % (kind, model)
        for kind in ('diag', 'kfac') for model in ('small', 'shallow', 'deep')}
    settings = dict(small_mlp=(2.0, {cell for cell in expected if '_small' in cell}),
                    weak_signal_1=(1.0, expected), weak_signal_05=(0.5, expected))
    if (set(protocol.CELLS) != expected or set(protocol.SETTINGS) != set(settings)
            or tuple(protocol.VARIANTS) != original.VARIANTS
            or tuple(protocol.SPLITS) != ('iid', 'noniid')):
        raise ValueError('recovery experiment matrix changed')
    for name, (strength, cells) in settings.items():
        recipe = protocol.SETTINGS[name]
        if (type(recipe['signal_strength']) is not float or recipe['signal_strength'] != strength
                or set(recipe['cells']) != cells or len(recipe['cells']) != len(cells)):
            raise ValueError('recovery setting changed')


def _parameters_key(parameters):
    return tuple(parameters[key] for key in ('setting', 'cell', 'variant', 'shard_index', 'shard_count'))


def _formal_source_rows(closeout):
    if not isinstance(closeout, dict):
        raise ValueError('formal recovery requires the verified source closeout')
    retained = closeout.get('retained_request_ids')
    rows = closeout.get('old_inventory', {}).get('requests')
    if (not isinstance(retained, (list, tuple)) or len(retained) != 129
            or any(not isinstance(jid, str) for jid in retained) or len(set(retained)) != 129
            or not isinstance(rows, list) or len(rows) != 4608):
        raise ValueError('closeout must retain exactly 129 complete requests from the original 4608')
    by_key, ids = {}, set()
    for row in rows:
        p = original.validate_parameters(original.FORMAL_TASK, row['parameters'])
        key = _parameters_key(p)
        expected = json.loads(canonical(original.run_config(original.FORMAL_TASK, p)))
        if (p['shard_count'] != 16 or row['task'] != original.FORMAL_TASK
                or key in by_key or row['job_id'] in ids
                or row['configuration'] != expected or row['run_id'] != expected['run_id']
                or row['configuration_sha256'] != sha256(canonical(expected).encode())
                or row['seeds'] != list(original.seed_sequence(original.FORMAL_TASK, p))
                or row['splits'] != list(protocol.SPLITS)):
            raise ValueError('closeout original formal coverage/configuration changed')
        by_key[key] = row
        ids.add(row['job_id'])
    if not set(retained) <= ids:
        raise ValueError('retained request is outside the original cohort')
    retained_units = sum(len(row['seeds']) * len(row['splits']) for row in rows
                         if row['job_id'] in set(retained))
    if retained_units != 4838:
        raise ValueError('retained requests must cover exactly 4838 successful units')
    return by_key, set(retained)


def build_requests(generation=1, cohort='pilot', closeout=None):
    """Pure specifications; formal input must be the verified normalized closeout."""
    bundle = bundle_name(generation, cohort)
    _scope()
    pilot = cohort == 'pilot'
    old_rows, retained = ({}, set()) if pilot else _formal_source_rows(closeout)
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    parameters = []
    for setting, recipe in protocol.SETTINGS.items():
        for cell in recipe['cells']:
            for variant in protocol.VARIANTS:
                for index in range(1 if pilot else 16):
                    p = dict(setting=setting, cell=cell, variant=variant,
                             shard_index=index, shard_count=1 if pilot else 16)
                    if pilot:
                        p['case'] = 'baseline'
                    elif old_rows[_parameters_key(p)]['job_id'] in retained:
                        continue
                    parameters.append(p)
    if pilot:
        parameters.extend(dict(setting='small_mlp', cell='fedavg_small_e1_patience',
            variant=variant, shard_index=0, shard_count=1, case='cap_diagnostic')
            for variant in protocol.DIAGNOSTIC_VARIANTS)
    specs = []
    for p in parameters:
        protocol.validate_parameters(task, p)
        diagnostic = p.get('case') == 'cap_diagnostic'
        job_id = '%s-%s-%s-%s%s-s%02dof%02d' % (
            bundle, p['setting'].replace('_', '-'), p['cell'].replace('_', '-'),
            p['variant'].replace('_', '-'), '-cap-diagnostic' if diagnostic else '',
            p['shard_index'], p['shard_count'])
        if re.fullmatch(r'fedfisher-ms-v2-[A-Za-z0-9_.-]+', job_id) is None or len(job_id) > 128:
            raise ValueError('generation exceeds scheduler job-identity limits')
        request = dict(schema_version=2, job_id=job_id, project='FedFisher',
                       task=task, parameters=p, priority=0)
        config = json.loads(canonical(protocol.run_config(task, p)))
        seeds = list(protocol.seed_sequence(task, p))
        expected_seeds = ([9, 25, 41] if diagnostic else [0]) if pilot else list(range(p['shard_index'], 300, 16))
        if seeds != expected_seeds:
            raise ValueError('recovery deterministic seed coverage changed')
        if not pilot:
            old_config = old_rows[_parameters_key(p)]['configuration']
            ignored = {'protocol', 'task', 'run_id'}
            if ({k: v for k, v in config.items() if k not in ignored}
                    != {k: v for k, v in old_config.items() if k not in ignored}):
                raise ValueError('recovery changed a scientific recipe')
        specs.append(dict(filename=job_id + '.job-request.json', request=request,
            request_sha256=sha256(encoded(request)), run_id=protocol.run_id(task, p),
            configuration=config, configuration_sha256=sha256(canonical(config).encode()),
            seed_sequence=seeds, splits=list(protocol.SPLITS)))
    expected_count, expected_units = (290, 588) if pilot else (4479, 167962)
    if (len(specs) != expected_count or len({s['filename'] for s in specs}) != expected_count
            or sum(len(s['seed_sequence']) * len(s['splits']) for s in specs) != expected_units):
        raise ValueError('recovery request/unit coverage changed')
    return tuple(specs)


def _provenance():
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def review_sources():
    return _review_sources(REVIEW_SOURCES, PROJECT_ROOT)


def _source_hashes():
    return review_sources()


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


if __name__ == '__main__':
    raise SystemExit(main())
