"""Saved-evidence pilot acceptance for the model/signal sensitivity experiments."""
import json
from pathlib import Path
import re

from ..infrastructure import evidence
from ..infrastructure.artifacts import canonical, digest, _pairs
from . import protocol
from .artifacts import ArtifactStore, current_provenance, validate_execution

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')
CENTRAL_JOBS = Path('/data/del6500/ServerScheduler/state/jobs')
CENTRAL_LOGS = Path('/scr/del6500/ServerScheduler/logs')
CENTRAL_RETRIES = Path('/data/del6500/ServerScheduler/state/retries')
CENTRAL_AUDIT = Path('/data/del6500/ServerScheduler/audit/events.jsonl')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ordinary_path(path, exists=True):
    return evidence.ordinary_path(path, exists=exists)


def _binding(path):
    return evidence.file_binding(path, check_path=_ordinary_path)


def _json_bytes(data):
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _read_json(path):
    path = _ordinary_path(path)
    _require(path.stat().st_size <= 128 * 1024 * 1024, 'oversized pilot JSON requires review')
    return _json_bytes(path.read_bytes())


def default_acceptance_path(provenance=None):
    provenance = current_provenance() if provenance is None else provenance
    checksum = provenance.get('provenance_sha256')
    _require(provenance.get('git_dirty') == '0' and isinstance(checksum, str)
             and re.fullmatch('[0-9a-f]{64}', checksum) is not None,
             'pilot proof requires clean source provenance')
    return _ordinary_path(REPORT_ROOT / ('model-signal-ablation-v1-' + checksum + '.acceptance.json'),
                          exists=False)


def _log_evidence(path, kind):
    path = _ordinary_path(path)
    _require(path.parent == CENTRAL_LOGS and path.stat().st_size <= 16 * 1024 * 1024,
             'unexpected central log location or oversized pilot log')
    binding, text = _binding(path), path.read_text(encoding='utf-8')
    if kind == 'stderr' and text:
        lines = text.splitlines()
        _require(len(lines) == 2 and re.fullmatch(
            r'.*networkx/utils/backends\.py:\d+: RuntimeWarning: '
            r'networkx backend defined more than once: nx-loopback', lines[0])
            and lines[1].strip() == 'backends.update(_get_backends("networkx.backends"))',
            'unexpected pilot stderr requires review')
    if kind == 'stdout':
        _require(re.search(r'Traceback \(most recent call last\)|CUDA out of memory|'
            r'(^|\n)(?:ValueError|RuntimeError|FloatingPointError|MemoryError):',
            text, re.IGNORECASE) is None, 'pilot stdout contains an unexplained failure signature')
    return binding


def _attempt_audit(job_id, attempt):
    return evidence.attempt_audit(CENTRAL_AUDIT, job_id, attempt,
        check_path=_ordinary_path, parse_json=_json_bytes)


def _completed_job(request):
    path = CENTRAL_JOBS / (request['job_id'] + '.json')
    binding, job = _binding(path), _read_json(path)
    _require(job.get('schema_version') == 2 and job.get('project') == 'FedFisher'
        and job.get('job_id') == request['job_id'] and job.get('task') == protocol.PILOT_TASK
        and job.get('parameters') == request['parameters'] and job.get('state') == 'completed'
        and type(job.get('exit_code')) is int and job['exit_code'] == 0,
        'pilot needs matching completed central jobs with exit zero')
    return job, dict(job=binding, logs={kind + '_log': _log_evidence(job[kind + '_log'], kind)
                                       for kind in ('stdout', 'stderr')})


def _producer_evidence(request, job, started):
    return evidence.producer_evidence(request, job, started,
        logs_root=CENTRAL_LOGS, retries_root=CENTRAL_RETRIES,
        validate_execution=validate_execution, binding=_binding, read_json=_read_json,
        log_evidence=_log_evidence, audit=_attempt_audit)


def _recheck_producer(proof):
    evidence.recheck_producer_evidence(proof, audit_path=CENTRAL_AUDIT,
        binding=_binding, check_path=_ordinary_path, parse_json=_json_bytes)


def validate_pairs(records):
    """Compare data within each signal; weaker outcomes need not match baseline."""
    expected = {(setting, cell, variant, split)
        for setting, recipe in protocol.SETTINGS.items() for cell in recipe['cells']
        for variant in protocol.VARIANTS for split in protocol.SPLITS}
    _require(set(records) == expected, 'pilot pairing requires exactly all 576 scientific units')
    intervention_pairs = method_pairs = fisher_pairs = common_tests = 0
    evaluation_keys = ('raw_x_sha256', 'permutation_sha256', 'shard_indices_sha256', 'shard_sizes')
    for setting, recipe in protocol.SETTINGS.items():
        cells = recipe['cells']
        for split in protocol.SPLITS:
            _require(len({records[setting, cell, variant, split]['fingerprints']['test']
                for cell in cells for variant in protocol.VARIANTS}) == 1,
                'all methods and factorial arms must share one test set within each setting')
            common_tests += 1
            for variant in protocol.VARIANTS:
                reference = records[setting, cells[0], variant, split]
                for cell in cells[1:]:
                    record = records[setting, cell, variant, split]
                    _require(all(reference['fingerprints'][key] == record['fingerprints'][key]
                                 for key in ('training', 'test'))
                        and reference['site_sizes'] == record['site_sizes']
                        and reference['label_counts'] == record['label_counts']
                        and all(reference['evaluation'][key] == record['evaluation'][key]
                                for key in evaluation_keys), 'methods changed paired data or evaluation plans')
                    method_pairs += 1
            for cell in cells:
                fisher = protocol.CELLS[cell]['family'] == 'legacy_fedfisher'
                for left, right in (('control', 'local_permutation'),
                                    ('site_gaussian_shift', 'site_gaussian_local_permutation')):
                    a, b = (records[setting, cell, variant, split] for variant in (left, right))
                    _require(a['fingerprints'] == b['fingerprints'],
                             'intervention-only arm changed data/model')
                    fit_keys = ('initial_model_fingerprint', 'local_model_fingerprints', 'validation') if fisher else ('fit_fingerprint',)
                    _require(all(a['fit'][key] == b['fit'][key] for key in fit_keys),
                             'intervention-only arm changed fitting')
                    _require(a['evaluation']['raw_x_sha256'] == b['evaluation']['raw_x_sha256'],
                             'intervention-only arm changed common test covariates')
                    intervention_pairs += 1
                for left, right in (('control', 'site_gaussian_shift'),
                                    ('local_permutation', 'site_gaussian_local_permutation')):
                    a, b = (records[setting, cell, variant, split] for variant in (left, right))
                    _require(all(a['evaluation'][key] == b['evaluation'][key] for key in evaluation_keys),
                             'matched factorial scopes changed evaluation plans')
            for cell in cells:
                method = protocol.CELLS[cell]
                if method['family'] != 'legacy_fedfisher' or method['algorithm'] != 'fedfisher_diag':
                    continue
                peer = next(name for name in cells if protocol.CELLS[name]['family'] == 'legacy_fedfisher'
                    and protocol.CELLS[name]['algorithm'] == 'fedfisher_kfac'
                    and protocol.CELLS[name]['model'] == method['model'])
                for variant in protocol.VARIANTS:
                    a, b = (records[setting, name, variant, split] for name in (cell, peer))
                    _require(all(a['fit'][key] == b['fit'][key] for key in
                        ('initial_model_fingerprint', 'local_model_fingerprints', 'validation')),
                        'Diag/KFAC changed paired local fitting or validation')
                    fisher_pairs += 1
    return dict(intervention_only_pairs=intervention_pairs, method_data_pairs=method_pairs,
        diag_kfac_local_fit_pairs=fisher_pairs, common_test_sets=common_tests, passed=True)


def collect_pilot_acceptance(inventory_path):
    """Validate immutable artifacts and producer evidence without inference."""
    inventory_path = _ordinary_path(inventory_path)
    _require(REPORT_ROOT in inventory_path.parents, 'pilot inventory must be in project scheduler reports')
    inventory_binding, inventory = _binding(inventory_path), _read_json(inventory_path)
    provenance = current_provenance()
    _require(inventory.get('provenance') == provenance and inventory.get('bundle_ready') is True
        and inventory.get('task') == protocol.PILOT_TASK and inventory.get('request_count') == 288
        and inventory.get('scientific_cell_count') == 576, 'pilot inventory must bind this source and exact scope')
    sources = inventory.get('review_sources')
    _require(isinstance(sources, dict) and bool(sources), 'pilot inventory lacks reviewed source bindings')
    for binding in sources.values():
        _require(PROJECT_ROOT in Path(binding['path']).parents and _binding(binding['path']) == binding,
                 'review source binding changed or is outside canonical source')
    requests = inventory.get('requests')
    _require(isinstance(requests, list) and len(requests) == 288, 'pilot request inventory incomplete')
    seen, paths, job_ids, records, units, jobs, request_bindings = set(), set(), set(), {}, [], [], []
    for request in requests:
        p = protocol.validate_parameters(protocol.PILOT_TASK, request['parameters'])
        key = p['setting'], p['cell'], p['variant']
        _require(key not in seen, 'duplicate pilot setting/cell/variant')
        seen.add(key)
        config = protocol.run_config(protocol.PILOT_TASK, p)
        _require(request['run_id'] == config['run_id'] and canonical(request['configuration']) == canonical(config)
            and request['configuration_sha256'] == digest(config) and request['task'] == protocol.PILOT_TASK
            and request['seeds'] == [0] and request['splits'] == list(protocol.SPLITS),
            'pilot scientific configuration or seed/split scope changed')
        job_id = request['job_id']
        _require(isinstance(job_id, str) and re.fullmatch(r'fedfisher-ms-[A-Za-z0-9_.-]{1,115}', job_id)
                 and job_id not in job_ids, 'invalid or duplicate pilot job identity')
        job_ids.add(job_id)
        path = _ordinary_path(request['path'])
        _require(OUTBOX_ROOT in path.parents and path not in paths
                 and path.name == job_id + '.job-request.json', 'pilot request path is not a unique outbox identity')
        paths.add(path)
        binding = _binding(path)
        _require(binding['sha256'] == request['sha256'], 'pilot request bytes changed')
        request_bindings.append(binding)
        _require(_read_json(path) == dict(schema_version=2, project='FedFisher', task=protocol.PILOT_TASK,
            job_id=job_id, parameters=p, priority=0), 'pilot request contract changed')
        job, central = _completed_job(request)
        jobs.append(central)
        store = ArtifactStore(config, provenance)
        for split in protocol.SPLITS:
            directory = store.cell_dir(0, split)
            artifact_bindings = [_binding(directory / name)
                for name in ('started.json', 'model.npz', 'result.json', 'done.json')]
            manifest_binding = _binding(directory.parent / 'run_manifest.env')
            _require(store.status(0, split) == 'validated', 'pilot scientific output incomplete')
            result = _read_json(directory / 'result.json')
            started = _read_json(directory / 'started.json')['execution']
            producer = _producer_evidence(request, job, started)
            records[(*key, split)] = result
            units.append(dict(setting=p['setting'], cell=p['cell'], variant=p['variant'], seed=0, split=split,
                run_id=config['run_id'], artifacts=artifact_bindings, manifest=manifest_binding,
                timing=result['timing'], memory=result.get('memory'), producer_evidence=producer))
    pairs = validate_pairs(records)
    _require(current_provenance() == provenance and _binding(inventory_path) == inventory_binding,
             'source or pilot inventory changed during acceptance')
    for binding in [*sources.values(), *request_bindings]:
        _require(_binding(binding['path']) == binding, 'review source or request changed during acceptance')
    for unit in units:
        for binding in [unit['manifest'], *unit['artifacts']]:
            _require(_binding(binding['path']) == binding, 'scientific artifact changed during acceptance')
        _recheck_producer(unit['producer_evidence'])
    for central in jobs:
        for binding in [central['job'], *central['logs'].values()]:
            _require(_binding(binding['path']) == binding, 'central evidence changed during acceptance')
    return dict(schema_version=1, protocol=protocol.PROTOCOL, scope='model_signal_pilot_scientific_acceptance',
        passed=True, inference_performed=False, provenance=provenance, pilot_inventory=inventory_binding,
        validated_request_count=288, validated_unit_count=576, paired_checks=pairs,
        units=units, central_jobs=jobs,
        formal_scope=dict(setting_method_combinations=72, variants=4, seeds=[0, 299],
                          splits=list(protocol.SPLITS), scientific_units=172800),
        admission_note='scientific gate only; operator reviews measured runtime/memory before formal admission')


def verify_pilot_acceptance(path):
    provenance = current_provenance()
    path = _ordinary_path(path)
    _require(path == default_acceptance_path(provenance), 'pilot proof must use canonical source-addressed path')
    binding, proof = _binding(path), _read_json(path)
    _require(isinstance(proof, dict) and proof.get('passed') is True
        and proof.get('scope') == 'model_signal_pilot_scientific_acceptance'
        and proof.get('provenance') == provenance and isinstance(proof.get('pilot_inventory'), dict),
        'invalid pilot acceptance proof')
    fresh = collect_pilot_acceptance(proof['pilot_inventory']['path'])
    _require(fresh == proof and _binding(path) == binding, 'pilot proof differs from freshly verified evidence')
    return proof


def check_current_pilot_gate():
    return verify_pilot_acceptance(default_acceptance_path())
