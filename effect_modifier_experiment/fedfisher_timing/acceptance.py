"""Read-only pilot acceptance and a reproducible gate for formal requests."""
import hashlib
import json
from pathlib import Path
import re

from ..causal_comparison.artifacts import canonical, read_json, safe_path
from . import protocol
from .artifacts import ArtifactStore, current_provenance, validate_execution

CENTRAL_JOBS = Path('/data/del6500/ServerScheduler/state/jobs')
CENTRAL_LOGS = Path('/scr/del6500/ServerScheduler/logs')
CENTRAL_RETRIES = Path('/data/del6500/ServerScheduler/state/retries')
CENTRAL_AUDIT = Path('/data/del6500/ServerScheduler/audit/events.jsonl')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ordinary_file(path):
    path = Path(path)
    _require(path.is_absolute() and '..' not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents))
             and path.resolve(strict=True) == path and path.is_file(),
             'pilot evidence requires an ordinary absolute file')
    return path


def _binding(path):
    path = _ordinary_file(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _attempt_audit(job_id, attempt):
    """Bind exact immutable lines, allowing unrelated append-only audit growth.

    ServerScheduler retains only its final job/history snapshot. Its audit
    proves the attempt was started and entered running; it does not retain the
    original allocation/runtime snapshot. The independently hash-bound started
    receipt and its fixed-profile validator provide those scientific fields.
    """
    path = _ordinary_file(CENTRAL_AUDIT)
    found, active_attempt, offset = {}, None, 0
    with path.open('rb') as stream:
        for line in stream:
            start, offset = offset, offset+len(line)
            if not line.endswith(b'\n'):
                continue  # A concurrently appended unrelated tail is not evidence.
            if job_id.encode() not in line:
                continue
            record = json.loads(line)
            if record.get('job_id') != job_id:
                continue
            _require(record.get('schema_version') == 2, 'invalid producer audit schema')
            event = record.get('event')
            name = None
            if event == 'job_attempt_started':
                active_attempt = record.get('attempt')
                _require(type(active_attempt) is int and active_attempt > 0,
                         'invalid producer attempt in central audit')
                if active_attempt == attempt:
                    name = 'attempt_started'
            elif (event == 'job_transition' and record.get('destination') == 'running'
                  and active_attempt == attempt):
                _require(record.get('project') == 'FedFisher' and record.get('source') == 'pending',
                         'producer running transition has unexpected ownership/state')
                name = 'running_transition'
            if name is not None:
                _require(name not in found, 'duplicate producer attempt audit evidence')
                found[name] = dict(path=str(path), offset=start, length=len(line),
                    sha256=hashlib.sha256(line).hexdigest(), record=record)
    _require(set(found) == {'attempt_started', 'running_transition'},
             'producer attempt lacks exact central start/running audit evidence')
    return found


def _producer_evidence(request, job, started):
    """Distinguish the producing attempt from a later successful skip/resume."""
    validate_execution(started)
    job_id = request['job_id']
    _require(started['job_id'] == job_id, 'scientific producer belongs to another scheduler job')
    final_attempt = None
    for kind in ('stdout', 'stderr'):
        path = Path(job[kind+'_log'])
        match = re.fullmatch(re.escape(job_id)+r'\.attempt-(\d{3,})\.'+kind+r'\.log', path.name)
        _require(path.parent == CENTRAL_LOGS and match is not None,
                 'final central log has an invalid job/attempt identity')
        number = int(match.group(1))
        _require(number > 0 and (final_attempt is None or final_attempt == number),
                 'final stdout/stderr attempt identities disagree')
        final_attempt = number
    validate_execution(dict(job_id=job_id, attempt=final_attempt,
        **{name: job[name] for name in ('execution_profile', 'allocation', 'estimated_runtime_seconds')}))
    retry_path = CENTRAL_RETRIES/(job_id+'.json')
    retry_binding = _binding(retry_path)
    retry = json.loads(retry_path.read_bytes())
    _require(set(retry) == {'schema_version', 'job_id', 'attempts_started', 'max_attempts',
                           'next_attempt_at', 'last_failure_reason', 'updated_at'}
             and retry.get('schema_version') == 2 and retry.get('job_id') == job_id
             and type(retry.get('attempts_started')) is int
             and retry['attempts_started'] == final_attempt
             and type(retry.get('max_attempts')) is int and retry['max_attempts'] >= final_attempt
             and retry.get('next_attempt_at') is None,
             'final central attempt differs from durable retry evidence')
    _require(started['attempt'] <= final_attempt,
             'scientific producer attempt is later than the final completed attempt')
    # Final attempt metadata can be matched exactly. Earlier attempts can have
    # different runtime estimates and CPU placement; never rewrite their receipt
    # or pretend the final central snapshot certifies their old allocation.
    resumed = started['attempt'] < final_attempt
    if not resumed:
        for name in ('execution_profile', 'allocation', 'estimated_runtime_seconds'):
            _require(started[name] == job[name], 'producer receipt differs from its final central attempt')
    logs = {kind+'_log': _binding(CENTRAL_LOGS/('%s.attempt-%03d.%s.log' %
                     (job_id, started['attempt'], kind))) for kind in ('stdout', 'stderr')}
    return dict(producer_attempt=started['attempt'], final_completed_attempt=final_attempt,
        resumed_from_earlier_attempt=resumed, execution_receipt=started,
        retry=retry_binding, logs=logs, audit=_attempt_audit(job_id, started['attempt']),
        allocation_runtime_evidence=('hash_bound_started_receipt_validated_against_fixed_task_profile; '
            'central_audit_and_logs_prove_attempt; original_running_snapshot_not_archived'
            if resumed else 'started_receipt_matches_final_central_job_snapshot'))


def _recheck_producer_evidence(proof):
    for binding in [proof['retry'], *proof['logs'].values()]:
        _require(_binding(binding['path']) == binding, 'producer attempt evidence changed during acceptance')
    for binding in proof['audit'].values():
        path = _ordinary_file(binding['path'])
        _require(path == CENTRAL_AUDIT, 'producer audit evidence moved outside central ownership')
        with path.open('rb') as stream:
            stream.seek(binding['offset'])
            line = stream.read(binding['length'])
        _require(len(line) == binding['length'] and line.endswith(b'\n')
                 and hashlib.sha256(line).hexdigest() == binding['sha256']
                 and json.loads(line) == binding['record'],
                 'producer audit line changed during acceptance')


def validate_pairs(records):
    """Require paired data/local models and the intended intervention factor."""
    expected = {(c, v, s) for c in protocol.CELLS for v in protocol.VARIANTS for s in protocol.SPLITS}
    _require(set(records) == expected, 'pilot pair validation requires exactly 32 cells')
    paired = 0
    for cell in protocol.CELLS:
        for split in protocol.SPLITS:
            for left, right in (('control', 'local_permutation'),
                                ('site_gaussian_shift', 'site_gaussian_local_permutation')):
                a, b = (records[cell, variant, split] for variant in (left, right))
                _require(a['fingerprints'] == b['fingerprints'], 'intervention-only arm changed data/model')
                for key in ('initial_model_fingerprint', 'local_model_fingerprints', 'validation'):
                    _require(a['fit'][key] == b['fit'][key], 'intervention-only arm changed local fitting')
                _require(a['evaluation']['raw_x_sha256'] == b['evaluation']['raw_x_sha256'],
                         'paired arms have different common test rows')
                paired += 1
    local_pairs = 0
    for depth in ('shallow', 'deep'):
        for variant in protocol.VARIANTS:
            for split in protocol.SPLITS:
                a, b = (records['fedfisher_'+kind+'_'+depth, variant, split] for kind in ('diag', 'kfac'))
                for key in ('training', 'test'):
                    _require(a['fingerprints'][key] == b['fingerprints'][key], 'Fisher methods use different data')
                for key in ('initial_model_fingerprint', 'local_model_fingerprints', 'validation'):
                    _require(a['fit'][key] == b['fit'][key], 'selected Fisher altered local SGD/validation pairing')
                for key in ('raw_x_sha256', 'permutation_sha256', 'shard_indices_sha256', 'shard_sizes'):
                    _require(a['evaluation'][key] == b['evaluation'][key], 'Fisher method evaluation plans differ')
                local_pairs += 1
    for split in protocol.SPLITS:
        _require(len({record['fingerprints']['test'] for key, record in records.items() if key[2] == split}) == 1,
                 'all factorial arms must share the independent test set')
    return dict(intervention_only_pairs=paired, diag_kfac_local_fit_pairs=local_pairs,
                common_test_sets=2, passed=True)


def _completed_job(request):
    job_path = CENTRAL_JOBS / (request['job_id']+'.json')
    binding = _binding(job_path)
    job = json.loads(job_path.read_bytes())
    _require(job.get('schema_version') == 2 and job.get('project') == 'FedFisher'
             and job.get('job_id') == request['job_id'] and job.get('task') == protocol.PILOT_TASK
             and job.get('parameters') == request['parameters'] and job.get('state') == 'completed'
             and type(job.get('exit_code')) is int and job['exit_code'] == 0,
             'pilot needs matching completed central jobs with exit zero')
    logs = {}
    for name in ('stdout_log', 'stderr_log'):
        path = Path(job[name])
        _require(CENTRAL_LOGS in path.parents, 'unexpected central log location')
        logs[name] = _binding(path)
    stderr = Path(job['stderr_log']).read_text()
    if stderr:
        _require('RuntimeWarning: networkx backend defined more than once: nx-loopback' in stderr
                 and len(stderr.splitlines()) == 2,
                 'unexpected pilot stderr requires review before formal expansion')
    return job, dict(job=binding, logs=logs)


def collect_pilot_acceptance(inventory_path):
    """Revalidate completed artifacts without fitting, regeneration or inference."""
    inventory_path = safe_path(Path(inventory_path))
    inventory_binding = _binding(inventory_path)
    inventory = read_json(inventory_path)
    provenance = current_provenance()
    _require(inventory.get('provenance') == provenance and inventory.get('bundle_ready') is True
             and inventory.get('request_count') == 16
             and inventory.get('task') == protocol.PILOT_TASK,
             'pilot inventory must bind this exact clean source and 16 requests')
    requests = inventory.get('requests')
    _require(isinstance(requests, list) and len(requests) == 16, 'pilot request inventory incomplete')
    seen, results, units, jobs = set(), {}, [], []
    for request in requests:
        p = protocol.validate_parameters(protocol.PILOT_TASK, request['parameters'])
        key = (p['cell'], p['variant'])
        _require(key not in seen, 'duplicate pilot method/variant')
        seen.add(key)
        config = protocol.run_config(protocol.PILOT_TASK, p)
        _require(request['run_id'] == config['run_id'] and request['configuration'] == config
                 and request['configuration_sha256'] == hashlib.sha256(canonical(config).encode()).hexdigest(),
                 'pilot configuration differs from frozen science recipe')
        path = safe_path(Path(request['path']))
        _require(_binding(path)['sha256'] == request['sha256'], 'pilot request bytes changed')
        actual = read_json(path)
        _require(actual == dict(schema_version=2, project='FedFisher', task=protocol.PILOT_TASK,
                 job_id=request['job_id'], parameters=p, priority=0), 'pilot request contract differs')
        job, central = _completed_job(request)
        jobs.append(central)
        store = ArtifactStore(config, provenance)
        for split in protocol.SPLITS:
            _require(store.status(0, split) == 'validated', 'pilot scientific output is incomplete')
            directory = store.cell_dir(0, split)
            started = read_json(directory/'started.json')['execution']
            producer = _producer_evidence(request, job, started)
            result = read_json(directory/'result.json')
            results[p['cell'], p['variant'], split] = result
            units.append(dict(cell=p['cell'], variant=p['variant'], seed=0, split=split,
                run_id=config['run_id'], artifacts=[_binding(directory/name)
                    for name in ('started.json', 'model.npz', 'result.json', 'done.json')],
                manifest=_binding(directory.parent/'run_manifest.env'),
                timing=result['timing'], memory=result['memory'], producer_evidence=producer))
    pairs = validate_pairs(results)
    _require(current_provenance() == provenance and _binding(inventory_path) == inventory_binding,
             'source or pilot inventory changed during acceptance')
    for unit in units:
        for binding in [unit['manifest'], *unit['artifacts']]:
            _require(_binding(binding['path']) == binding, 'scientific artifact changed during acceptance')
        _recheck_producer_evidence(unit['producer_evidence'])
    for central in jobs:
        for binding in [central['job'], *central['logs'].values()]:
            _require(_binding(binding['path']) == binding, 'central evidence changed during acceptance')
    return dict(schema_version=1, protocol=protocol.PROTOCOL, scope='pilot_scientific_acceptance',
        passed=True, inference_performed=False, provenance=provenance,
        pilot_inventory=inventory_binding, validated_request_count=16, validated_unit_count=32,
        paired_checks=pairs, units=units, central_jobs=jobs,
        formal_scope=dict(method_cells=4, variants=4, seeds=[0, 299], splits=list(protocol.SPLITS),
                          scientific_units=9600),
        admission_note='scientific gate only; operator must review measured runtime/memory before formal admission')


def verify_pilot_acceptance(path):
    path = safe_path(Path(path))
    binding = _binding(path)
    proof = read_json(path)
    _require(isinstance(proof, dict) and proof.get('passed') is True
             and proof.get('scope') == 'pilot_scientific_acceptance'
             and isinstance(proof.get('pilot_inventory'), dict), 'invalid pilot acceptance proof')
    fresh = collect_pilot_acceptance(proof['pilot_inventory']['path'])
    _require(fresh == proof and _binding(path) == binding, 'pilot acceptance differs from current verified evidence')
    return proof
