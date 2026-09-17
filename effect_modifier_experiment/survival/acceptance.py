"""Pure, reproducible pilot acceptance before formal publication or execution.

No fitting, data regeneration, inference, scheduler CLI, or central mutation.
Proofs bind saved scientific artifacts and exact producing-attempt evidence;
their saved boolean alone is never sufficient to open the formal gate.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np

from ..infrastructure import evidence
from ..infrastructure.artifacts import canonical
from . import protocol
from .artifacts import ArtifactStore, array_hash, current_provenance, validate_execution


PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
INVENTORY_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')
PROOF_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1')
CENTRAL_JOBS = Path('/data/del6500/ServerScheduler/state/jobs')
CENTRAL_LOGS = Path('/scr/del6500/ServerScheduler/logs')
CENTRAL_RETRIES = Path('/data/del6500/ServerScheduler/state/retries')
CENTRAL_AUDIT = Path('/data/del6500/ServerScheduler/audit/events.jsonl')
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ordinary_path(path, exists=True):
    return evidence.ordinary_path(path, exists=exists)


def _binding(path):
    return evidence.file_binding(path, check_path=_ordinary_path)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, 'duplicate key in pilot JSON evidence')
        result[key] = value
    return result


def _json_bytes(data):
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _read_json(path):
    path = _ordinary_path(path)
    _require(path.stat().st_size <= MAX_JSON_BYTES, 'pilot JSON exceeds bounded evidence size')
    return _json_bytes(path.read_bytes())


def default_acceptance_path(provenance=None):
    provenance = current_provenance() if provenance is None else provenance
    checksum = provenance.get('provenance_sha256')
    _require(provenance.get('git_dirty') == '0' and isinstance(checksum, str)
             and re.fullmatch('[0-9a-f]{64}', checksum) is not None,
             'pilot acceptance path requires clean source provenance')
    return _ordinary_path(PROOF_ROOT / ('pilot-' + checksum + '.acceptance.json'), exists=False)


def _log_evidence(path, kind):
    path = _ordinary_path(path)
    _require(path.parent == CENTRAL_LOGS and path.stat().st_size <= MAX_LOG_BYTES,
             'unexpected central log location or oversized pilot log requiring review')
    binding = _binding(path)
    text = path.read_text(encoding='utf-8')
    if kind == 'stderr' and text:
        lines = text.splitlines()
        _require(len(lines) == 2
                 and re.fullmatch(r'.*networkx/utils/backends\.py:\d+: RuntimeWarning: '
                                  r'networkx backend defined more than once: nx-loopback', lines[0])
                 and lines[1].strip() == 'backends.update(_get_backends("networkx.backends"))',
                 'unexpected pilot stderr requires review before formal expansion')
    if kind == 'stdout':
        _require(re.search(r'Traceback \(most recent call last\)|CUDA out of memory|'
                           r'(^|\n)(?:ValueError|RuntimeError|FloatingPointError|MemoryError):',
                           text, re.IGNORECASE) is None,
                 'pilot stdout contains an unexplained failure signature')
    return binding


def _attempt_audit(job_id, attempt):
    return evidence.attempt_audit(CENTRAL_AUDIT, job_id, attempt,
                                  check_path=_ordinary_path, parse_json=_json_bytes)


def _completed_job(request):
    job_path = CENTRAL_JOBS / (request['job_id'] + '.json')
    binding, job = _binding(job_path), _read_json(job_path)
    _require(job.get('schema_version') == 2 and job.get('project') == 'FedFisher'
             and job.get('job_id') == request['job_id'] and job.get('task') == protocol.PILOT_TASK
             and job.get('parameters') == request['parameters'] and job.get('state') == 'completed'
             and type(job.get('exit_code')) is int and job['exit_code'] == 0,
             'pilot needs matching completed central jobs with exit zero')
    logs = {kind + '_log': _log_evidence(job[kind + '_log'], kind) for kind in ('stdout', 'stderr')}
    return job, dict(job=binding, logs=logs)


def _producer_evidence(request, job, started):
    return evidence.producer_evidence(request, job, started,
        logs_root=CENTRAL_LOGS, retries_root=CENTRAL_RETRIES,
        validate_execution=validate_execution, binding=_binding, read_json=_read_json,
        log_evidence=_log_evidence, audit=_attempt_audit)


def _recheck_producer(proof):
    evidence.recheck_producer_evidence(proof, audit_path=CENTRAL_AUDIT,
        binding=_binding, check_path=_ordinary_path, parse_json=_json_bytes)


def _local_model_hashes(model_path, site_count):
    """Decode saved vectors only; no model construction or prediction."""
    hashes = []
    with np.load(_ordinary_path(model_path), allow_pickle=False) as archive:
        for site in range(site_count):
            blob = archive['upload_%03d' % site].tobytes()
            with np.load(io.BytesIO(blob), allow_pickle=False) as upload:
                hashes.append(array_hash({'vector': upload['vector']}))
    return hashes


def validate_pairs(records, local_models):
    expected = {(cell, scenario) for cell in protocol.CELLS for scenario in protocol.SCENARIOS}
    _require(set(records) == expected, 'survival pilot pair validation needs exactly ten cells')
    data_pairs = 0
    for scenario in protocol.SCENARIOS:
        reference = records[protocol.CELLS[0], scenario]
        for cell in protocol.CELLS[1:]:
            other = records[cell, scenario]
            for key in ('training', 'evaluation', 'evaluation_truth'):
                _require(reference['fingerprints'][key] == other['fingerprints'][key],
                         'survival methods differ in paired training/evaluation/truth data')
            _require(reference['site_sizes'] == other['site_sizes']
                     and reference['pseudo_diagnostics'] == other['pseudo_diagnostics'],
                     'survival methods differ in site sizes or pseudo-values')
            for key in ('raw_x_sha256', 'contrast_truth_sha256', 'permutation_sha256'):
                _require(reference['evaluation'][key] == other['evaluation'][key],
                         'survival methods differ in evaluator inputs or permutations')
            data_pairs += 1
    reference = next(iter(records.values()))
    for other in records.values():
        _require(reference['site_sizes'] == other['site_sizes'], 'scenario changed sampled site sizes')
        for key in ('evaluation', 'evaluation_truth'):
            _require(reference['fingerprints'][key] == other['fingerprints'][key],
                     'scenarios must share their common independent test and truth')
        for key in ('raw_x_sha256', 'contrast_truth_sha256', 'permutation_sha256'):
            _require(reference['evaluation'][key] == other['evaluation'][key],
                     'scenario changed common evaluator inputs or permutations')
    local_pairs = 0
    for scenario in protocol.SCENARIOS:
        for depth in ('shallow', 'deep'):
            keys = [('fedfisher_' + kind + '_' + depth, scenario) for kind in ('diag', 'kfac')]
            left, right = (records[key]['fit'] for key in keys)
            _require(left['initial_sha256'] == right['initial_sha256']
                     and local_models[keys[0]] == local_models[keys[1]],
                     'Diag/KFAC selected Fisher changed paired local model vectors')
            left_rows, right_rows = left['local'], right['local']
            _require(len(left_rows) == len(right_rows), 'Diag/KFAC local fit counts differ')
            for first, second in zip(left_rows, right_rows):
                for name in ('site_id', 'sample_count', 'optimizer_updates', 'epochs', 'stop_reason',
                             'initial_loss', 'final_loss'):
                    _require(first[name] == second[name], 'Diag/KFAC local fitting recipe/evidence differs')
            local_pairs += 1
    return dict(method_data_pairs=data_pairs, diag_kfac_local_fit_pairs=local_pairs,
                common_test_sets=1, passed=True)


def _quality_observations(result):
    """Descriptive only: never turn low recovery into a fabricated pass bar."""
    diagnostics = result['pseudo_diagnostics']
    fit = result['fit']
    cv = ([dict(fold=path['fold'], lambda_max=path['lambda_max'],
                zero_lambda_max=path['lambda_max'] == 0,
                stationarity_reports=path['fits']) for path in fit['cv_paths']]
          if 'cv_paths' in fit else None)
    return dict(recovery=result['evaluation']['recovery'],
                contrast_rmse=result['evaluation']['contrast_rmse'],
                sites_beyond_tau_support=[row['site_id'] for row in diagnostics if row['tau_beyond_observed_max']],
                sites_without_risk_at_tau=[row['site_id'] for row in diagnostics if row['n_at_risk_tau'] == 0],
                sites_with_degenerate_pseudo_values=[row['site_id'] for row in diagnostics
                                                    if row['all_zero'] or row['all_one']],
                pseudo_diagnostics=diagnostics,
                dslasso_cv_path_diagnostics=cv,
                full_site_diagnostics_establish_cv_fold_support=False,
                performance_threshold_applied=False)


def _fit_timing_details(result):
    fit = result['fit']
    if 'timing_seconds' in fit:
        return dict(method_timing_seconds=fit['timing_seconds'])
    return dict(total_fit_seconds=fit['total_fit_seconds'],
                server_aggregation_seconds=fit['server']['aggregation_seconds'],
                local=[{key: row[key] for key in ('site_id', 'pseudo_seconds', 'training_seconds',
                    'curvature_seconds', 'serialization_seconds')} for row in fit['local']])


def collect_pilot_acceptance(inventory_path):
    """Recompute ten-cell acceptance from actual evidence under current clean source."""
    inventory_path = _ordinary_path(inventory_path)
    _require(INVENTORY_ROOT in inventory_path.parents, 'pilot inventory must be in project scratch reports')
    inventory_binding, inventory = _binding(inventory_path), _read_json(inventory_path)
    provenance = current_provenance()
    _require(inventory.get('schema_version') == 1
             and inventory.get('bundle') == 'fedfisher-survival-v1-pilot'
             and type(inventory.get('generation')) is int and inventory['generation'] > 0
             and inventory.get('provenance') == provenance and inventory.get('bundle_ready') is True
             and inventory.get('request_count') == 10 and inventory.get('scientific_cell_count') == 10
             and inventory.get('task') == protocol.PILOT_TASK
             and inventory.get('project_head') == provenance['git_commit']
             and inventory.get('worktree_status') == 'clean'
             and inventory.get('coverage') == dict(method_cells=5, scenarios=2, seeds=[0]),
             'pilot inventory must bind exact clean source and ten seed-zero requests')
    sources = inventory.get('review_sources')
    _require(isinstance(sources, dict) and bool(sources), 'pilot reviewed source bindings are missing')
    for binding in sources.values():
        path = _ordinary_path(binding['path'])
        _require(PROJECT_ROOT in path.parents and _binding(path) == binding,
                 'pilot reviewed source differs from current canonical file')
    requests = inventory.get('requests')
    _require(isinstance(requests, list) and len(requests) == 10, 'pilot request inventory is incomplete')
    records, local_models, units, jobs, seen_jobs, seen_paths = {}, {}, [], [], set(), set()
    request_bindings = []
    for request in requests:
        parameters = protocol.validate_parameters(protocol.PILOT_TASK, request['parameters'])
        key = (parameters['cell'], parameters['scenario'])
        _require(key not in records and request['job_id'] not in seen_jobs,
                 'duplicate pilot cell or scheduler identity')
        seen_jobs.add(request['job_id'])
        config = protocol.run_config(protocol.PILOT_TASK, parameters)
        _require(request['run_id'] == config['run_id'] and request['configuration'] == config
                 and request['configuration_sha256'] == hashlib.sha256(canonical(config).encode()).hexdigest()
                 and request['task'] == protocol.PILOT_TASK and request['seeds'] == [0],
                 'pilot configuration differs from frozen science recipe')
        path = _ordinary_path(request['path'])
        _require(OUTBOX_ROOT in path.parents and str(path) not in seen_paths
                 and path.name == request['job_id'] + '.job-request.json',
                 'pilot request path differs from unique project outbox identity')
        seen_paths.add(str(path))
        binding = _binding(path)
        _require(binding['sha256'] == request['sha256'], 'pilot request bytes changed')
        request_bindings.append(binding)
        _require(_read_json(path) == dict(schema_version=2, project='FedFisher', task=protocol.PILOT_TASK,
                 job_id=request['job_id'], parameters=parameters, priority=0), 'pilot request contract differs')
        job, central = _completed_job(request)
        jobs.append(central)
        store = ArtifactStore(config, provenance)
        directory = store.cell_dir()
        # Bind before semantic validation and rereading, then recheck at the
        # end: a changed record cannot inherit an earlier store.status pass.
        artifact_bindings = [_binding(directory / name)
                             for name in ('started.json', 'model.npz', 'result.json', 'done.json')]
        manifest_binding = _binding(directory.parent / 'run_manifest.env')
        _require(store.status() == 'validated', 'pilot scientific output is incomplete')
        result = _read_json(directory / 'result.json')
        started = _read_json(directory / 'started.json')['execution']
        producer = _producer_evidence(request, job, started)
        records[key] = result
        if parameters['cell'] != 'dslasso':
            local_models[key] = _local_model_hashes(directory / 'model.npz', len(result['site_sizes']))
        timing = result['timing']
        # Store validation owns all detailed semantic checks, including positive
        # measured durations/RSS and optimizer convergence; retain the evidence.
        observed_wall = sum(stage['wall_seconds'] for stage in timing['stages'].values())
        _require(math.isfinite(observed_wall) and observed_wall > 0, 'pilot timing must be measured and positive')
        units.append(dict(cell=parameters['cell'], scenario=parameters['scenario'], seed=0,
            run_id=config['run_id'], artifacts=artifact_bindings,
            manifest=manifest_binding,
            timing=timing, measured_stage_wall_seconds=observed_wall,
            fit_timing_details=_fit_timing_details(result),
            memory=result['memory'], producer_evidence=producer,
            quality_observations=_quality_observations(result),
            local_model_sha256=local_models.get(key)))
    paired = validate_pairs(records, local_models)
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
    return dict(schema_version=1, protocol=protocol.PROTOCOL, scope='survival_pilot_scientific_acceptance',
        passed=True, inference_performed=False, provenance=provenance,
        pilot_inventory=inventory_binding, validated_request_count=10, validated_unit_count=10,
        paired_checks=paired, units=units, central_jobs=jobs,
        formal_scope=dict(method_cells=5, scenarios=2, seed_start=0, seed_end=299,
                          seed_count=300, scientific_units=3000),
        admission_note='technical and numerical evidence only; operator reviews observed quality, support, runtime and memory before formal admission; no method-superiority or clinical claim')


validate_pilot_inventory = collect_pilot_acceptance


def verify_pilot_acceptance(path):
    provenance = current_provenance()
    path = _ordinary_path(path)
    _require(path == default_acceptance_path(provenance),
             'pilot proof must use the canonical path for current source provenance')
    binding, proof = _binding(path), _read_json(path)
    _require(isinstance(proof, dict) and proof.get('passed') is True
             and proof.get('scope') == 'survival_pilot_scientific_acceptance'
             and proof.get('provenance') == provenance and isinstance(proof.get('pilot_inventory'), dict),
             'invalid pilot acceptance proof')
    fresh = collect_pilot_acceptance(proof['pilot_inventory']['path'])
    _require(fresh == proof and _binding(path) == binding,
             'pilot acceptance differs from freshly verified evidence')
    return proof


def check_current_pilot_gate():
    """Fixed source-addressed gate, used before formal publication and fitting."""
    return verify_pilot_acceptance(default_acceptance_path())
