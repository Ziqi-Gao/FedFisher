"""Saved-only scientific acceptance and same-source pilot gate.

Completed scheduler jobs are necessary but not sufficient. The immutable run
manifest, producing attempt, fitted model, prediction arrays and recomputed
probability summaries must also agree. No dataset or prediction is regenerated.
"""
from pathlib import Path
import json
import os

from ..infrastructure import evidence
from ..infrastructure.artifacts import digest
from ..survival.acceptance import (
    CENTRAL_JOBS, CENTRAL_LOGS, CENTRAL_RETRIES, INVENTORY_ROOT, OUTBOX_ROOT, _binding,
    _ordinary_path, _read_json, _log_evidence, _attempt_audit, _recheck_producer,
)
from ..survival.artifacts import current_provenance, validate_execution
from . import protocol
from .artifacts import ArtifactStore, require
from .inputs import historical_binding, validate_historical_inputs
from .requests import build_requests, review_sources

PROOF_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-pooled-v1')


def default_acceptance_path(provenance=None, cohort='pilot'):
    provenance = current_provenance() if provenance is None else provenance
    require(cohort in ('pilot', 'formal') and provenance.get('git_dirty') == '0',
            'pooled acceptance needs a clean source and known cohort')
    return _ordinary_path(PROOF_ROOT / (cohort + '-' + provenance['provenance_sha256']
                                        + '.acceptance.json'), exists=False)


def _completed_job(request):
    path = CENTRAL_JOBS / (request['job_id'] + '.json')
    binding, job = _binding(path), _read_json(path)
    require(job.get('schema_version') == 2 and job.get('project') == 'FedFisher'
            and job.get('job_id') == request['job_id'] and job.get('task') == request['task']
            and job.get('parameters') == request['parameters'] and job.get('state') == 'completed'
            and type(job.get('exit_code')) is int and job['exit_code'] == 0,
            'pooled survival requires matching completed central jobs with exit zero')
    logs = {kind + '_log': _log_evidence(job[kind + '_log'], kind) for kind in ('stdout', 'stderr')}
    return job, dict(job=binding, logs=logs)


def validate_pairs(records, seeds):
    expected = {(family, scenario, seed) for family in protocol.FAMILIES
                for scenario in protocol.SCENARIOS for seed in seeds}
    require(set(records) == expected, 'incomplete pooled survival paired coverage')
    comparisons = 0
    for seed in seeds:
        common = records[protocol.FAMILIES[0], protocol.SCENARIOS[0], seed]
        for scenario in protocol.SCENARIOS:
            reference = records[protocol.FAMILIES[0], scenario, seed]
            for family in protocol.FAMILIES:
                other = records[family, scenario, seed]
                require(reference['site_sizes'] == other['site_sizes']
                        and reference['fingerprints'] == other['fingerprints'],
                        'pooled model families have different training/test/truth data')
                require(common['site_sizes'] == other['site_sizes']
                        and all(common['fingerprints'][name] == other['fingerprints'][name]
                                for name in ('evaluation', 'evaluation_truth')),
                        'pooled scenarios have different independent test rows or oracle truth')
                comparisons += 1
    return dict(passed=True, complete_seed_count=len(seeds), scientific_units=comparisons,
                within_scenario_training_paired=True, common_independent_test_paired=True,
                historical_FL_training_test_and_truth_paired=True)


def _observations(result):
    # Preserve accuracy and measured costs; poor predictions are scientific
    # observations, never a hidden method-superiority threshold for completion.
    return dict(probability_accuracy={tag: {key: summary[key] for key in
        ('arms', 'combined', 'factual', 'contrast')} for tag, summary in result['prediction'].items()},
        FL_pooled_prediction_gaps={tag: {scope: gap['combined'] for scope, gap in scopes.items()}
                                  for tag, scopes in result['comparisons'].items()},
        fitting_budgets={scope: {key: fit[key] for key in
            ('family', 'pseudo_scope', 'sample_count', 'total_fit_seconds', 'stop_reason')}
                        for scope, fit in result['fit'].items()},
        performance_threshold_applied=False)


def collect_acceptance(inventory_path, cohort='pilot'):
    """Validate exact source-bound request coverage and immutable saved outputs."""
    require(cohort in ('pilot', 'formal'), 'unknown pooled survival cohort')
    pilot = cohort == 'pilot'
    inventory_path = _ordinary_path(inventory_path)
    require(INVENTORY_ROOT in inventory_path.parents, 'inventory must be in project scratch reports')
    inventory_binding, inventory = _binding(inventory_path), _read_json(inventory_path)
    source = current_provenance()
    validate_historical_inputs()
    seeds = list(protocol.PILOT_SEEDS if pilot else protocol.FORMAL_SEEDS)
    units_count = 6 * len(seeds)
    require(inventory.get('schema_version') == 1 and inventory.get('bundle') ==
            'fedfisher-survival-pooled-v1-' + cohort and inventory.get('bundle_ready') is True
            and inventory.get('provenance') == source and inventory.get('project_head') == source['git_commit']
            and inventory.get('worktree_status') == 'clean'
            and inventory.get('scientific_cell_count') == units_count
            and inventory.get('pooled_fit_count') == 2 * units_count
            and inventory.get('coverage') == dict(families=list(protocol.FAMILIES),
                scenarios=list(protocol.SCENARIOS), seeds=seeds, pseudo_scopes=list(protocol.PSEUDO_SCOPES))
            and inventory.get('historical_inputs') == historical_binding(),
            'pooled inventory differs from frozen source, coverage or historical inputs')
    sources = inventory.get('review_sources')
    require(sources == review_sources(), 'reviewed pooled sources differ from current canonical files')
    records = inventory.get('requests')
    require(isinstance(records, list) and records, 'missing pooled request records')
    shard_count = None if pilot else records[0]['parameters'].get('shard_count')
    specs = build_requests(inventory['generation'], cohort, shard_count)
    require(inventory.get('task') == (protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK)
            and inventory.get('request_count') == len(specs) == len(records), 'incomplete pooled request inventory')
    pilot_binding, prerequisite = None, None
    if not pilot:
        path = default_acceptance_path(source)
        proof = verify_pilot_acceptance(path)
        pilot_binding = dict(path=str(path), sha256=_binding(path)['sha256'], evidence=proof)
        require(inventory.get('pilot_acceptance') == pilot_binding, 'formal inventory pilot proof mismatch')
        prerequisite = dict(pilot_acceptance_sha256=digest(proof), provenance_sha256=source['provenance_sha256'])
    else:
        require('pilot_acceptance' not in inventory, 'pilot cannot claim formal prerequisite evidence')
    results, units, jobs, request_bindings = {}, [], [], []
    audit_cache = {}

    def cached_audit(job_id, attempt):
        # A formal shard owns many seeds. Scan the append-only audit once for
        # each producing attempt, then recheck exact byte-range witnesses below.
        key = (job_id, attempt)
        if key not in audit_cache:
            audit_cache[key] = _attempt_audit(job_id, attempt)
        return audit_cache[key]

    for request, spec in zip(records, specs):
        p, raw = spec['request']['parameters'], spec['request']
        path = _ordinary_path(request['path'])
        require(OUTBOX_ROOT in path.parents and path.name == spec['filename']
                and _read_json(path) == raw and request['job_id'] == raw['job_id']
                and request['task'] == raw['task'] and request['parameters'] == p
                and request['configuration'] == spec['configuration']
                and request['configuration_sha256'] == spec['configuration_sha256']
                and request['run_id'] == spec['run_id'] and request['seeds'] == spec['seed_sequence'],
                'request differs from fixed pooled survival recipe or deterministic partition')
        binding = _binding(path)
        require(binding['sha256'] == request['sha256'] == spec['request_sha256'], 'pooled request bytes changed')
        request_bindings.append(binding)
        job, central = _completed_job(request)
        jobs.append(central)
        store = ArtifactStore(spec['configuration'], source, prerequisite=prerequisite)
        for seed in spec['seed_sequence']:
            key = (p['family'], p['scenario'], seed)
            require(key not in results, 'duplicate pooled scientific unit')
            directory = store.cell_dir(seed)
            files = [_binding(directory / name) for name in (*store.artifact_names, store.done_name)]
            manifest = _binding(store.directory / 'run_manifest.env')
            require(store.status(seed) == 'validated', 'pooled prediction output is partial or invalid')
            result = _read_json(directory / store.result_name)
            started = _read_json(directory / 'started.json')['execution']
            producer = evidence.producer_evidence(request, job, started,
                logs_root=CENTRAL_LOGS, retries_root=CENTRAL_RETRIES, validate_execution=validate_execution,
                binding=_binding, read_json=_read_json, log_evidence=_log_evidence, audit=cached_audit)
            wall = sum(row['wall_seconds'] for row in result['timing']['stages'].values())
            require(wall > 0, 'pooled benchmark needs positive observed timing')
            results[key] = result
            units.append(dict(family=p['family'], scenario=p['scenario'], seed=seed,
                run_id=spec['run_id'], artifacts=files, manifest=manifest,
                producer_evidence=producer, timing=result['timing'], memory=result['memory'],
                measured_stage_wall_seconds=wall, observations=_observations(result)))
    paired = validate_pairs(results, seeds)
    require(current_provenance() == source and _binding(inventory_path) == inventory_binding
            and review_sources() == sources, 'source or inventory changed during pooled acceptance')
    validate_historical_inputs()
    for binding in request_bindings:
        require(_binding(binding['path']) == binding, 'pooled request changed during acceptance')
    for unit in units:
        for binding in [unit['manifest'], *unit['artifacts']]:
            require(_binding(binding['path']) == binding, 'pooled scientific artifacts changed during acceptance')
        _recheck_producer(unit['producer_evidence'])
    for central in jobs:
        for binding in [central['job'], *central['logs'].values()]:
            require(_binding(binding['path']) == binding, 'central completion evidence changed during acceptance')
    if pilot_binding is not None:
        require(_binding(pilot_binding['path'])['sha256'] == pilot_binding['sha256'], 'pilot proof changed')
    return dict(schema_version=1, protocol=protocol.PROTOCOL,
        scope='survival_pooled_' + cohort + '_scientific_acceptance', passed=True,
        inference_performed=False, provenance=source, inventory=inventory_binding,
        historical_inputs=historical_binding(), validated_request_count=len(records),
        validated_unit_count=len(units), paired_checks=paired, units=units, central_jobs=jobs,
        formal_scope=dict(families=3, scenarios=2, seed_count=300, scientific_units=1800, pooled_fits=3600),
        admission_note='technical completion and saved probability arithmetic; operator reviews pilot runtime, memory and numerical diagnostics before formal admission; no performance superiority claim')


def collect_pilot_acceptance(inventory_path):
    return collect_acceptance(inventory_path, 'pilot')


def verify_pilot_acceptance(path):
    source, path = current_provenance(), _ordinary_path(path)
    require(path == default_acceptance_path(source), 'pilot proof must be canonical for current clean source')
    binding, proof = _binding(path), _read_json(path)
    require(proof.get('scope') == 'survival_pooled_pilot_scientific_acceptance'
            and proof.get('passed') is True and proof.get('provenance') == source,
            'invalid pooled pilot scientific proof')
    require(collect_pilot_acceptance(proof['inventory']['path']) == proof and _binding(path) == binding,
            'pooled pilot proof differs from freshly verified evidence')
    return proof


def publish_acceptance(proof, cohort='pilot'):
    """Create a source-addressed report proof, preserving every partial write.

    The model-artifact writer is deliberately restricted to outputs/scratch;
    acceptance reports have their own canonical durable report namespace.
    Publishing a proof never substitutes for its fresh scientific verification.
    """
    source = current_provenance()
    require(proof.get('provenance') == source and proof.get('passed') is True
            and proof.get('scope') == 'survival_pooled_' + cohort + '_scientific_acceptance',
            'acceptance publication needs the current source and matching scientific scope')
    output = default_acceptance_path(source, cohort)
    pending = output.with_name('.' + output.name + '.pending')
    _ordinary_path(output, exists=False)
    _ordinary_path(pending, exists=False)
    require(not output.exists() and not pending.exists(), 'preserving existing or partial pooled acceptance proof')
    data = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_path(output, exists=False)
    _ordinary_path(pending, exists=False)
    require(current_provenance() == source, 'source changed before acceptance publication')
    with pending.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(current_provenance() == source, 'source changed while publishing acceptance; partial proof preserved')
    _ordinary_path(output, exists=False)
    _ordinary_path(pending)
    os.link(pending, output)
    pending.unlink()
    require(output.read_bytes() == data and current_provenance() == source,
            'acceptance readback or source changed; preserve proof for review')
    return output
