"""Read-only file and producing-attempt evidence, independent of experiments.

Callers supply their JSON, log-content and execution-profile policies.  In
particular, sharing attempt checks must not silently broaden an experiment's
accepted logs or alter the shape/bytes of its existing acceptance evidence.
"""
import hashlib
import json
from pathlib import Path
import re


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary_path(path, exists=True):
    path = Path(path)
    _require(path.is_absolute() and '..' not in path.parts
             and not any(item.is_symlink() for item in (path, *path.parents))
             and path.resolve(strict=exists) == path,
             'pilot evidence must use an ordinary absolute path without symlinks')
    if exists:
        _require(path.is_file(), 'pilot evidence must be an ordinary file')
    return path


def file_binding(path, *, check_path=ordinary_path):
    path = check_path(path)
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return dict(path=str(path), sha256=checksum.hexdigest())


def attempt_audit(path, job_id, attempt, *, check_path=ordinary_path,
                  parse_json=json.loads, project='FedFisher'):
    """Bind exact start/running lines, allowing unrelated append-only growth.

    Earlier allocation/runtime snapshots are not archived centrally.  These
    audit lines prove the attempt ran; the separately validated scientific
    receipt supplies its original allocation and runtime fields.
    """
    path = check_path(path)
    found, active_attempt, offset = {}, None, 0
    with path.open('rb') as stream:
        for line in stream:
            start, offset = offset, offset + len(line)
            if not line.endswith(b'\n') or job_id.encode() not in line:
                continue
            record = parse_json(line)
            if record.get('job_id') != job_id:
                continue
            _require(record.get('schema_version') == 2, 'invalid producer audit schema')
            event, name = record.get('event'), None
            if event == 'job_attempt_started':
                active_attempt = record.get('attempt')
                _require(type(active_attempt) is int and active_attempt > 0,
                         'invalid producer attempt in central audit')
                if active_attempt == attempt:
                    name = 'attempt_started'
            elif (event == 'job_transition' and record.get('destination') == 'running'
                  and active_attempt == attempt):
                _require(record.get('project') == project and record.get('source') == 'pending',
                         'producer running transition has unexpected ownership/state')
                name = 'running_transition'
            if name is not None:
                _require(name not in found, 'duplicate producer attempt audit evidence')
                found[name] = dict(path=str(path), offset=start, length=len(line),
                                  sha256=hashlib.sha256(line).hexdigest(), record=record)
    _require(set(found) == {'attempt_started', 'running_transition'},
             'producer attempt lacks exact central start/running audit evidence')
    return found


def producer_evidence(request, job, started, *, logs_root, retries_root,
                      validate_execution, binding, read_json, log_evidence, audit):
    """Separate the actual producer from a later successful validation retry."""
    validate_execution(started)
    job_id = request['job_id']
    _require(started['job_id'] == job_id, 'scientific producer belongs to another scheduler job')
    final_attempt = None
    for kind in ('stdout', 'stderr'):
        path = Path(job[kind + '_log'])
        match = re.fullmatch(re.escape(job_id) + r'\.attempt-(\d{3,})\.' + kind + r'\.log', path.name)
        _require(path.parent == logs_root and match is not None,
                 'final central log has an invalid job/attempt identity')
        number = int(match.group(1))
        _require(number > 0 and (final_attempt is None or final_attempt == number),
                 'final stdout/stderr attempt identities disagree')
        final_attempt = number
    validate_execution(dict(job_id=job_id, attempt=final_attempt,
        **{name: job[name] for name in ('execution_profile', 'allocation', 'estimated_runtime_seconds')}))
    retry_path = retries_root / (job_id + '.json')
    retry_binding, retry = binding(retry_path), read_json(retry_path)
    _require(set(retry) == {'schema_version', 'job_id', 'attempts_started', 'max_attempts',
                           'next_attempt_at', 'last_failure_reason', 'updated_at'}
             and retry['schema_version'] == 2 and retry['job_id'] == job_id
             and type(retry['attempts_started']) is int and retry['attempts_started'] == final_attempt
             and type(retry['max_attempts']) is int and retry['max_attempts'] >= final_attempt
             and retry['next_attempt_at'] is None,
             'final central attempt differs from durable retry evidence')
    _require(started['attempt'] <= final_attempt,
             'scientific producer attempt is later than the final completed attempt')
    resumed = started['attempt'] < final_attempt
    if not resumed:
        for name in ('execution_profile', 'allocation', 'estimated_runtime_seconds'):
            _require(started[name] == job[name], 'producer receipt differs from its final central attempt')
    logs = {kind + '_log': log_evidence(logs_root / ('%s.attempt-%03d.%s.log' %
            (job_id, started['attempt'], kind)), kind) for kind in ('stdout', 'stderr')}
    return dict(producer_attempt=started['attempt'], final_completed_attempt=final_attempt,
                resumed_from_earlier_attempt=resumed, execution_receipt=started,
                retry=retry_binding, logs=logs, audit=audit(job_id, started['attempt']),
                allocation_runtime_evidence=(
                    'hash_bound_started_receipt_validated_against_fixed_task_profile; central_audit_and_logs_prove_attempt; '
                    'original_running_snapshot_not_archived' if resumed else
                    'started_receipt_matches_final_central_job_snapshot'))


def recheck_producer_evidence(proof, *, audit_path, binding,
                              check_path=ordinary_path, parse_json=json.loads):
    for item in [proof['retry'], *proof['logs'].values()]:
        _require(binding(item['path']) == item, 'producer attempt evidence changed during acceptance')
    for item in proof['audit'].values():
        path = check_path(item['path'])
        _require(path == audit_path, 'producer audit evidence moved outside central ownership')
        with path.open('rb') as stream:
            stream.seek(item['offset'])
            line = stream.read(item['length'])
        _require(len(line) == item['length'] and line.endswith(b'\n')
                 and hashlib.sha256(line).hexdigest() == item['sha256']
                 and parse_json(line) == item['record'],
                 'producer audit line changed during acceptance')
