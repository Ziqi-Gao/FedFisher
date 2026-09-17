#!/usr/bin/env python3
"""Publish immutable survival requests for ServerScheduler operator review.

Pilot publication requires clean frozen source. Formal publication additionally
requires freshly verified same-source pilot acceptance. No central commands,
scientific fitting, resource discovery or background execution occur here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
SCRATCH_ROOT = Path('/scr/del6500/FedFisher')
OUTBOX_ROOT = SCRATCH_ROOT / 'scheduler/outbox'
REPORT_ROOT = SCRATCH_ROOT / 'scheduler/reports'
sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.dslassocov_run_contract import compute_provenance
from effect_modifier_experiment.survival import protocol


REVIEW_SOURCES = {
    'proposal': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-v1/registration-proposal.toml',
    'pilot_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-v1/pilot-parameters.schema.json',
    'formal_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-v1/shard-parameters.schema.json',
    'adapter': PROJECT_ROOT / 'scripts/fedfisher_server_scheduler_adapter.py',
    'protocol': PROJECT_ROOT / 'effect_modifier_experiment/survival/protocol.py',
    'runner': PROJECT_ROOT / 'effect_modifier_experiment/survival/runner.py',
    'artifacts': PROJECT_ROOT / 'effect_modifier_experiment/survival/artifacts.py',
    'acceptance': PROJECT_ROOT / 'effect_modifier_experiment/survival/acceptance.py',
    'request_builder': PROJECT_ROOT / 'scripts/prepare_survival_requests.py',
    'acceptance_cli': PROJECT_ROOT / 'scripts/validate_survival_pilot.py',
    'experiment_description': PROJECT_ROOT / 'effect_modifier_experiment/survival/README.md',
}
EXPECTED_CELLS = frozenset({'dslasso', 'fedfisher_diag_shallow', 'fedfisher_diag_deep',
                            'fedfisher_kfac_shallow', 'fedfisher_kfac_deep'})
EXPECTED_SCENARIOS = frozenset({'iid', 'covariate_shift'})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _generation(generation):
    if type(generation) is not int or generation < 1:
        raise ValueError('generation must be a positive integer')
    return generation


def _cohort(cohort):
    if cohort not in ('pilot', 'formal'):
        raise ValueError('cohort must be pilot or formal')
    return cohort


def bundle_name(generation=1, cohort='pilot'):
    return 'fedfisher-survival-v1-%s-cohort-p%d' % (_cohort(cohort), _generation(generation))


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def build_requests(generation=1, cohort='pilot', shard_count=None):
    """Construct only in-memory specs; writing formal files has a separate gate."""
    _generation(generation)
    pilot = _cohort(cohort) == 'pilot'
    if pilot and shard_count is not None:
        raise ValueError('pilot has no caller-selected shards')
    count = 1 if pilot else (16 if shard_count is None else shard_count)
    if not pilot and (type(count) is not int or count not in (8, 16)):
        raise ValueError('formal shard count must be 8 or 16')
    if (len(protocol.CELLS) != 5 or set(protocol.CELLS) != EXPECTED_CELLS
            or len(protocol.SCENARIOS) != 2 or set(protocol.SCENARIOS) != EXPECTED_SCENARIOS):
        raise ValueError('survival scope must retain five methods and two scenarios')
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    specs = []
    for cell in protocol.CELLS:
        for scenario in protocol.SCENARIOS:
            seen_seeds = []
            for index in range(count):
                parameters = dict(cell=cell, scenario=scenario)
                if not pilot:
                    parameters.update(shard_index=index, shard_count=count)
                protocol.validate_parameters(task, parameters)
                job_id = '%s-%s-%s' % (bundle_name(generation, cohort),
                                       cell.replace('_', '-'), scenario.replace('_', '-'))
                if not pilot:
                    job_id += '-s%02dof%02d' % (index, count)
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', job_id) is None:
                    raise ValueError('invalid or too long scheduler job identifier')
                request = dict(schema_version=2, project='FedFisher', task=task,
                               job_id=job_id, parameters=parameters, priority=0)
                configuration = json.loads(canonical(protocol.run_config(task, parameters)))
                seeds = list(protocol.seed_sequence(task, parameters))
                if seeds != ([0] if pilot else list(range(index, 300, count))):
                    raise ValueError('incorrect deterministic scientific shard coverage')
                seen_seeds.extend(seeds)
                specs.append(dict(filename=job_id + '.job-request.json', request=request,
                                  request_sha256=sha256(encoded(request)),
                                  run_id=protocol.run_id(task, parameters),
                                  configuration=configuration,
                                  configuration_sha256=sha256(canonical(configuration).encode()),
                                  seed_sequence=seeds))
            if sorted(seen_seeds) != ([0] if pilot else list(range(300))):
                raise ValueError('missing or duplicate seeds in a method/scenario cohort')
    if len(specs) != 10 * count or len({item['filename'] for item in specs}) != len(specs):
        raise ValueError('incomplete cohort or duplicate request identities')
    return tuple(specs)


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or SCRATCH_ROOT not in path.parents:
        raise ValueError('request publication must stay below the FedFisher scratch root')
    if any(p.is_symlink() for p in (path, *path.parents)) or path.resolve(strict=False) != path:
        raise ValueError('publication path must not contain symlinks')
    return path


def _pending(path):
    return path.with_name('.' + path.name + '.pending')


def _preflight_new(path):
    path = safe_path(path)
    for target in (path, safe_path(_pending(path))):
        if target.exists():
            raise FileExistsError('preserving existing or partial publication: %s' % target)
    return path


def atomic_create(path, data):
    path = _preflight_new(path)
    pending = _pending(path)
    fd = os.open(str(pending), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    safe_path(path)
    safe_path(pending)
    os.link(pending, path)
    pending.unlink()


def current_provenance():
    value = compute_provenance(PROJECT_ROOT)
    if value.get('git_dirty') != '0':
        raise ValueError('request publication requires finalized clean source')
    if re.fullmatch('[0-9a-f]{40}', value.get('git_commit', '')) is None:
        raise ValueError('invalid source commit')
    for key in ('git_status_sha256', 'source_tree_sha256', 'provenance_sha256'):
        if re.fullmatch('[0-9a-f]{64}', value.get(key, '')) is None:
            raise ValueError('invalid source provenance: ' + key)
    return dict(value)


def review_sources():
    result = {}
    for label, path in REVIEW_SOURCES.items():
        if (PROJECT_ROOT not in path.parents or path.resolve(strict=True) != path
                or any(p.is_symlink() for p in (path, *path.parents))):
            raise ValueError('review source must be a regular canonical project file')
        result[label] = dict(path=str(path), sha256=sha256(path.read_bytes()))
    return result


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
    outbox, reports = safe_path(OUTBOX_ROOT), safe_path(REPORT_ROOT)
    bundle = safe_path(outbox / bundle_name(generation, cohort))
    if bundle.exists():
        raise FileExistsError('preserving existing or partial cohort directory: %s' % bundle)
    inventory_path = reports / inventory_name(generation, cohort)
    targets = [bundle / item['filename'] for item in specs]
    for path in (*targets, inventory_path):
        _preflight_new(path)
    bundle.mkdir(parents=True, exist_ok=False)
    reports.mkdir(parents=True, exist_ok=True)
    safe_path(bundle); safe_path(reports)
    records = []
    for item, path in zip(specs, targets):
        atomic_create(path, encoded(item['request']))
        records.append(dict(path=str(path), sha256=item['request_sha256'],
                            job_id=item['request']['job_id'], task=item['request']['task'],
                            parameters=item['request']['parameters'], run_id=item['run_id'],
                            configuration=item['configuration'],
                            configuration_sha256=item['configuration_sha256'], seeds=item['seed_sequence']))
    for item, path in zip(specs, targets):
        safe_path(path)
        actual = path.read_bytes()
        if actual != encoded(item['request']) or sha256(actual) != item['request_sha256']:
            raise ValueError('request readback mismatch; partial cohort preserved')
    if current_provenance() != before or review_sources() != sources:
        raise ValueError('source changed during publication; partial cohort preserved')
    if acceptance_binding(cohort, pilot_acceptance, before) != accepted:
        raise ValueError('pilot proof changed during publication; partial cohort preserved')
    pilot = cohort == 'pilot'
    inventory = dict(
        schema_version=1, bundle='fedfisher-survival-v1-' + cohort, generation=generation,
        task=protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK,
        request_count=len(records), scientific_cell_count=10 if pilot else 3000,
        coverage=dict(method_cells=5, scenarios=2, seeds=[0] if pilot else list(range(300))),
        formal_plan=dict(task=protocol.FORMAL_TASK, seed_start=0, seed_end=299,
                         allowed_shard_counts=[8, 16], default_shard_count=16,
                         default_request_count=160, scientific_cell_count=3000,
                         publication_authorized=not pilot,
                         gate='complete same-source pilot scientific and paired acceptance plus operator profile review'),
        bundle_ready=True, submitted=False, requires_operator_registration=True,
        publication_mode='nested_operator_review_bundle_not_automatic_intake',
        registration_state='project proposal only; no central registration or admission implied',
        provenance=before, project_head=before['git_commit'], worktree_status='clean',
        review_sources=sources, requests=records, inventory_path=str(inventory_path),
        operator_gate='verify exact source and request hashes, review/register the admitted stage, then validate and submit through ServerScheduler',
        freeze_rule='keep canonical source unchanged through pilot and formal execution and scientific acceptance',
    )
    if accepted is not None:
        inventory['pilot_acceptance'] = accepted
    atomic_create(inventory_path, encoded(inventory))
    if inventory_path.read_bytes() != encoded(inventory):
        raise ValueError('inventory readback mismatch')
    return inventory


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


if __name__ == '__main__':
    raise SystemExit(main())
