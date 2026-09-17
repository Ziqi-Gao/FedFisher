"""Read-only pilot acceptance and a reproducible gate for formal requests."""
import hashlib
import json
from pathlib import Path

from ..infrastructure.artifacts import canonical, read_json, safe_path
from ..infrastructure import evidence
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
    return evidence.ordinary_path(path)


def _binding(path):
    return evidence.file_binding(path, check_path=_ordinary_file)


def _attempt_audit(job_id, attempt):
    return evidence.attempt_audit(CENTRAL_AUDIT, job_id, attempt,
                                  check_path=_ordinary_file)


def _producer_evidence(request, job, started):
    return evidence.producer_evidence(request, job, started,
        logs_root=CENTRAL_LOGS, retries_root=CENTRAL_RETRIES,
        validate_execution=validate_execution, binding=_binding,
        read_json=lambda path: json.loads(path.read_bytes()),
        log_evidence=lambda path, kind: _binding(path), audit=_attempt_audit)


def _recheck_producer_evidence(proof):
    evidence.recheck_producer_evidence(proof, audit_path=CENTRAL_AUDIT,
        binding=_binding, check_path=_ordinary_file)


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
