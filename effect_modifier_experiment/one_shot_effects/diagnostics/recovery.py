"""Read-only evidence for retaining three P2 cells and replacing two failures.

This verifies an already reviewed frozen-source audit and its immutable file
bindings. It does not recreate scientific acceptance, relax a current-source
guard, or admit computation. Reporting must still validate retained results
against the original inventory's configuration and provenance.
"""
import hashlib
import json
from pathlib import Path
import re

from .protocol import PILOT_TASK


SCRATCH_ROOT = Path('/scr/del6500/FedFisher')
DATA_ROOT = Path('/data/del6500/FedFisher')
CENTRAL_JOBS_ROOT = Path('/data/del6500/ServerScheduler/state/jobs')
OUTPUT_ROOT = DATA_ROOT / 'effect_modifier_experiment/outputs'
REPLACED_CELLS = frozenset({('fisher_deep', 'scmv2_base'),
                            ('fisher_deep', 'legacy_control_iid')})
RETAINED_CELLS = frozenset({('fisher_shallow', 'scmv2_base'),
                            ('fisher_shallow', 'legacy_control_iid'),
                            ('dr_path', 'scmv2_base')})
P2_PREFIX = 'fedfisher-effects-replay-v1-pilot-cohort-p2'


def _read_file(path, central=False):
    path = Path(path)
    roots = (SCRATCH_ROOT, DATA_ROOT) + ((CENTRAL_JOBS_ROOT,) if central else ())
    if (not path.is_absolute() or '..' in path.parts
            or not any(root in path.parents for root in roots)
            or any(part.is_symlink() for part in (path, *path.parents))
            or path.resolve(strict=True) != path or not path.is_file()):
        raise ValueError('recovery evidence requires an ordinary bound file in an allowed root')
    return path.read_bytes()


def _bound_file(binding, central=False):
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
            or not isinstance(binding['path'], str)
            or not isinstance(binding['sha256'], str)
            or re.fullmatch(r'[0-9a-f]{64}', binding['sha256']) is None):
        raise ValueError('invalid recovery file binding')
    body = _read_file(binding['path'], central=central)
    if hashlib.sha256(body).hexdigest() != binding['sha256']:
        raise ValueError('recovery evidence hash mismatch: '+binding['path'])
    return body


def _cell(record):
    if not isinstance(record, dict):
        raise ValueError('recovery cell must be a JSON object')
    parameters = record.get('parameters')
    if not isinstance(parameters, dict) or set(parameters) != {'recipe', 'dataset'}:
        raise ValueError('recovery cells require exactly recipe/dataset')
    return parameters['recipe'], parameters['dataset']


def verify_recovery_evidence(path):
    """Return the checked P2 recovery envelope without writing or inference."""
    evidence = json.loads(_read_file(path))
    required = {'schema_version', 'cohort', 'source_inventory', 'audit_evidence',
                'retained', 'replaced'}
    if (not isinstance(evidence, dict) or set(evidence) != required
            or type(evidence['schema_version']) is not int or evidence['schema_version'] != 1
            or evidence['cohort'] != 'deep-recovery'):
        raise ValueError('invalid deep-recovery evidence envelope')
    inventory = json.loads(_bound_file(evidence['source_inventory']))
    audit = json.loads(_bound_file(evidence['audit_evidence']))
    if (not isinstance(inventory, dict) or inventory.get('generation') != 2
            or inventory.get('bundle') != 'fedfisher-effects-replay-v1-pilot'
            or inventory.get('task') != PILOT_TASK or inventory.get('request_count') != 5
            or inventory.get('bundle_ready') is not True
            or inventory.get('worktree_status') != 'clean'
            or inventory.get('provenance', {}).get('git_dirty') != '0'):
        raise ValueError('recovery requires the complete frozen P2 inventory')
    source = audit.get('source') if isinstance(audit, dict) else None
    if (not isinstance(source, dict)
            or source.get('bound_provenance') != inventory['provenance']
            or source.get('inventory_sha256') != evidence['source_inventory']['sha256']
            or source.get('all_tracked_bytes_equal_bound_git_tree') is not True
            or source.get('tracked_source_sha256') != inventory['provenance'].get('source_tree_sha256')
            or audit.get('all_scientific_artifacts_unchanged_during_audit') is not True):
        raise ValueError('recovery audit does not certify the frozen inventory and source')
    requests = inventory.get('requests')
    if not isinstance(requests, list) or len(requests) != 5:
        raise ValueError('recovery inventory must contain exactly five original requests')
    original = {}
    for item in requests:
        cell = _cell(item)
        if cell in original or cell not in RETAINED_CELLS | REPLACED_CELLS:
            raise ValueError('recovery inventory does not cover the original five cells exactly')
        recipe, dataset = cell
        expected_job = P2_PREFIX+'-'+recipe.replace('_', '-')+'-'+dataset.replace('_', '-')
        expected_run = 'ose_replay_v1_'+recipe+'_'+dataset+'_pilot'
        if item.get('job_id') != expected_job or item.get('run_id') != expected_run:
            raise ValueError('recovery original job or run identity mismatch')
        request = json.loads(_bound_file({key: item[key] for key in ('path', 'sha256')}))
        if (request != dict(schema_version=2, project='FedFisher', task=PILOT_TASK,
                            job_id=expected_job, parameters=item['parameters'], priority=0)):
            raise ValueError('recovery original request identity mismatch')
        original[cell] = item
    audited = {}
    audit_cells = audit.get('cells')
    if not isinstance(audit_cells, list) or len(audit_cells) != 5:
        raise ValueError('recovery audit requires exactly five original cells')
    for record in audit_cells:
        cell = _cell(record)
        if cell in audited or cell not in original:
            raise ValueError('recovery audit cell coverage mismatch')
        old = original[cell]
        if record.get('job_id') != old['job_id'] or record.get('run_id') != old['run_id']:
            raise ValueError('recovery audit original identity mismatch')
        audited[cell] = record
    seen = set()
    for group, expected, status in (
            ('retained', RETAINED_CELLS, 'validated_frozen_source'),
            ('replaced', REPLACED_CELLS, 'unaccepted_failed_execution')):
        records = evidence[group]
        if not isinstance(records, list) or len(records) != len(expected):
            raise ValueError('recovery must retain three and replace two cells exactly')
        group_cells = set()
        for record in records:
            cell = _cell(record)
            record_keys = {'job_id', 'parameters', 'run_id', 'scientific_status',
                           'artifacts', 'central_job'}
            if set(record) not in (record_keys, record_keys | {'central_state'}):
                raise ValueError('recovery cell has unsupported evidence fields')
            if cell not in expected or cell in seen:
                raise ValueError('recovery retained/replaced partition mismatch')
            seen.add(cell)
            group_cells.add(cell)
            old = original[cell]
            audit_cell = audited[cell]
            if (record.get('job_id') != old['job_id'] or record.get('run_id') != old['run_id']
                    or record.get('scientific_status') != status):
                raise ValueError('recovery cell identity or scientific status mismatch')
            retained = group == 'retained'
            if (audit_cell.get('historical_scientific_acceptance') is not retained
                    or (retained and (audit_cell.get('manifest_and_started_valid') is not True
                                      or audit_cell.get('artifact_integrity_valid') is not True
                                      or audit_cell.get('pure_semantic_validation') != 'passed'))):
                raise ValueError('recovery retention disagrees with frozen scientific acceptance')
            artifacts = record.get('artifacts')
            if (not isinstance(artifacts, list) or not artifacts
                    or not all(isinstance(item, dict) for item in artifacts)
                    or len({item.get('path') for item in artifacts}) != len(artifacts)):
                raise ValueError('recovery cells require unique nonempty artifact bindings')
            files = audit_cell.get('scientific_files')
            partial_files = {'run_manifest.env', 'seed_000/started.json'}
            complete_files = partial_files | {'seed_000/result.json', 'seed_000/done.json'}
            if (not isinstance(files, dict)
                    or set(files) not in ((complete_files,) if retained else (partial_files, complete_files))):
                raise ValueError('recovery audit scientific file inventory is incomplete or unsupported')
            expected_artifacts = {str(OUTPUT_ROOT / old['run_id'] / relative): digest
                                  for relative, digest in files.items()}
            if {item.get('path'): item.get('sha256') for item in artifacts} != expected_artifacts:
                raise ValueError('recovery artifacts differ from the frozen scientific file inventory')
            for artifact in artifacts:
                _bound_file(artifact)
            central = audit_cell.get('central')
            expected_path = str(CENTRAL_JOBS_ROOT / (old['job_id']+'.json'))
            if (not isinstance(central, dict) or central.get('path') != expected_path
                    or record['central_job'] != {key: central.get(key) for key in ('path', 'sha256')}):
                raise ValueError('recovery central record is not the audited authoritative job path')
            job = json.loads(_bound_file(record.get('central_job'), central=True))
            if (job.get('job_id') != old['job_id'] or job.get('project') != 'FedFisher'
                    or job.get('task') != PILOT_TASK or job.get('parameters') != old['parameters']):
                raise ValueError('recovery central job identity mismatch')
            expected_state = 'completed' if group == 'retained' else 'failed'
            exit_code = job.get('exit_code')
            if (job.get('state') != expected_state
                    or central.get('state') != expected_state or central.get('exit_code') != exit_code
                    or record.get('central_state', expected_state) != expected_state
                    or type(exit_code) is not int
                    or (exit_code != 0 if group == 'retained' else exit_code == 0)):
                raise ValueError('recovery central termination disagrees with retention policy')
        if group_cells != expected:
            raise ValueError('recovery retained/replaced coverage mismatch')
    return evidence
