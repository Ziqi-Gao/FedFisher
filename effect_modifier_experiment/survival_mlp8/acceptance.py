"""Verify immutable predictions, numerical fits and actual producing attempts."""
from pathlib import Path
import json
import os

from ..infrastructure import evidence
from ..infrastructure.artifacts import digest
from ..survival.acceptance import (CENTRAL_LOGS, CENTRAL_RETRIES, INVENTORY_ROOT, OUTBOX_ROOT,
    _binding, _ordinary_path, _read_json, _attempt_audit, _log_evidence, _recheck_producer)
from ..survival.artifacts import current_provenance, validate_execution
from ..survival_pooled.acceptance import _completed_job
from ..survival_pooled.inputs import historical_binding, validate_historical_inputs
from . import protocol
from .artifacts import ArtifactStore, require
from .requests import build_requests, review_sources

PROOF_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-mlp8-v1')


def default_acceptance_path(provenance=None, cohort='pilot', *, discovery=False):
    source = current_provenance() if provenance is None else provenance
    require(source.get('git_dirty') == '0' and cohort in ('pilot', 'formal'), 'MLP8 acceptance requires a clean source and known cohort')
    prefix = cohort + ('-discovery-v2-' if discovery else '-')
    return _ordinary_path(PROOF_ROOT / (prefix + source['provenance_sha256'] + '.acceptance.json'), exists=False)


def validate_pairs(records, seeds, *, discovery=False):
    require(set(records) == {(s, seed) for s in protocol.SCENARIOS for seed in seeds}, 'incomplete MLP8 paired coverage')
    for seed in seeds:
        left, right = (records[s, seed] for s in protocol.SCENARIOS)
        require(left['site_sizes'] == right['site_sizes'] and all(left['fingerprints'][key] == right['fingerprints'][key]
            for key in ('evaluation', 'evaluation_truth')), 'MLP8 scenarios do not share independent test rows and truth')
        if discovery:
            common = left['discovery'][protocol.TAGS[0]]
            require(all(row['discovery'][tag][key] == common[key] for row in (left, right)
                for tag in protocol.TAGS for key in ('raw_x_sha256', 'contrast_truth_sha256', 'permutation_sha256')),
                'MLP8 discovery methods/scenarios do not share test rows, truth and permutations')
    return dict(passed=True, complete_seed_count=len(seeds), scientific_units=len(records),
        common_independent_test_paired=True, local_diag_kfac_training_paired=True,
        historical_training_test_and_truth_paired=True)


def _observations(result):
    observations = dict(probability_accuracy={tag: {key: row[key] for key in ('arms', 'combined', 'factual', 'contrast')}
        for tag, row in result['prediction'].items()},
        FL_pooled_prediction_gaps={tag: {scope: row['combined'] for scope, row in scopes.items()}
            for tag, scopes in result['comparisons'].items()},
        fitting_budgets={tag: dict({key: row.get(key) for key in ('config', 'sample_count', 'total_fit_seconds', 'stop_reason', 'server')},
            local_training_budgets=[{key: local.get(key) for key in
                ('site_id', 'sample_count', 'epochs', 'optimizer_updates', 'processed_rows', 'stop_reason')}
                for local in row.get('local', [])])
            for tag, row in result['fit'].items()}, performance_threshold_applied=False)
    if 'discovery' in result:
        observations['modifier_recovery'] = {tag: row['recovery'] for tag, row in result['discovery'].items()}
    return observations


def collect_acceptance(inventory_path, cohort='pilot', *, discovery=False):
    require(cohort in ('pilot', 'formal'), 'unknown MLP8 cohort')
    inventory_path = _ordinary_path(inventory_path)
    require(INVENTORY_ROOT in inventory_path.parents, 'MLP8 inventory must be in project scratch reports')
    inventory_binding, inventory = _binding(inventory_path), _read_json(inventory_path)
    source = current_provenance()
    if discovery:
        from . import discovery_protocol as recipe
        from .artifacts import DiscoveryArtifactStore as store_type
    else:
        recipe, store_type = protocol, ArtifactStore
    validate_historical_inputs()
    pilot = cohort == 'pilot'
    seeds = list(protocol.PILOT_SEEDS if pilot else protocol.FORMAL_SEEDS)
    require(inventory.get('schema_version') == 1 and inventory.get('bundle_ready') is True
        and inventory.get('bundle') == ('fedfisher-survival-mlp8-v2-' if discovery else 'fedfisher-survival-mlp8-v1-') + cohort
        and inventory.get('provenance') == source and inventory.get('project_head') == source['git_commit']
        and inventory.get('worktree_status') == 'clean' and inventory.get('scientific_cell_count') == 2 * len(seeds)
        and inventory.get('fitted_model_count') == 8 * len(seeds)
        and inventory.get('coverage') == dict(scenarios=list(protocol.SCENARIOS), seeds=seeds,
            models=list(protocol.TAGS), pseudo_scopes=list(protocol.PSEUDO_SCOPES))
        and inventory.get('historical_inputs') == historical_binding(), 'MLP8 inventory differs from fixed coverage or frozen source')
    sources = review_sources(discovery=True) if discovery else review_sources()
    require(inventory.get('review_sources') == sources, 'MLP8 reviewed sources changed')
    requests = inventory.get('requests')
    require(isinstance(requests, list) and requests, 'missing MLP8 requests')
    spec_args = (inventory['generation'], cohort, None if pilot else requests[0]['parameters'].get('shard_count'))
    specs = build_requests(*spec_args, discovery=True) if discovery else build_requests(*spec_args)
    require(inventory.get('request_count') == len(requests) == len(specs)
        and inventory.get('task') == (recipe.PILOT_TASK if pilot else recipe.FORMAL_TASK), 'incomplete MLP8 request inventory')
    prerequisite, pilot_binding = None, None
    if not pilot:
        path = default_acceptance_path(source, discovery=discovery)
        proof = verify_pilot_acceptance(path, discovery=discovery)
        pilot_binding = dict(path=str(path), sha256=_binding(path)['sha256'], evidence=proof)
        require(inventory.get('pilot_acceptance') == pilot_binding, 'MLP8 formal pilot prerequisite changed')
        prerequisite = dict(pilot_acceptance_sha256=digest(proof), provenance_sha256=source['provenance_sha256'])
    else:
        require('pilot_acceptance' not in inventory, 'pilot cannot claim formal prerequisites')
    results, units, jobs, bindings, audit_cache = {}, [], [], [], {}

    def audit(job_id, attempt):
        key = job_id, attempt
        if key not in audit_cache:
            audit_cache[key] = _attempt_audit(job_id, attempt)
        return audit_cache[key]

    for request, spec in zip(requests, specs):
        raw, config = spec['request'], spec['configuration']
        path = _ordinary_path(request['path'])
        require(OUTBOX_ROOT in path.parents and path.name == spec['filename'] and _read_json(path) == raw
            and all(request[key] == raw[key] for key in ('job_id', 'task', 'parameters'))
            and request['configuration'] == config and request['configuration_sha256'] == spec['configuration_sha256']
            and request['run_id'] == spec['run_id'] and request['seeds'] == spec['seed_sequence'], 'MLP8 request differs from the fixed recipe')
        binding = _binding(path)
        require(binding['sha256'] == request['sha256'] == spec['request_sha256'], 'MLP8 request bytes changed')
        bindings.append(binding)
        job, central = _completed_job(request)
        jobs.append(central)
        store = store_type(config, source, prerequisite=prerequisite)
        for seed in spec['seed_sequence']:
            key = config['scenario'], seed
            require(key not in results, 'duplicate MLP8 unit')
            directory = store.cell_dir(seed)
            files = [_binding(directory / name) for name in (*store.artifact_names, store.done_name)]
            manifest = _binding(store.directory / 'run_manifest.env')
            require(store.status(seed) == 'validated', 'partial or invalid MLP8 unit')
            result = _read_json(directory / store.result_name)
            started = _read_json(directory / 'started.json')['execution']
            producer = evidence.producer_evidence(request, job, started, logs_root=CENTRAL_LOGS,
                retries_root=CENTRAL_RETRIES, validate_execution=validate_execution, binding=_binding,
                read_json=_read_json, log_evidence=_log_evidence, audit=audit)
            results[key] = result
            units.append(dict(family='mlp8', scenario=config['scenario'], seed=seed, run_id=spec['run_id'],
                artifacts=files, manifest=manifest, producer_evidence=producer, timing=result['timing'],
                memory=result['memory'], measured_stage_wall_seconds=sum(r['wall_seconds'] for r in result['timing']['stages'].values()),
                observations=_observations(result)))
    pairs = validate_pairs(results, seeds, discovery=True) if discovery else validate_pairs(results, seeds)
    require(current_provenance() == source and _binding(inventory_path) == inventory_binding
        and (review_sources(discovery=True) if discovery else review_sources()) == sources, 'MLP8 source or inventory changed during acceptance')
    validate_historical_inputs()
    for binding in bindings:
        require(_binding(binding['path']) == binding, 'MLP8 request changed during acceptance')
    for unit in units:
        for binding in [unit['manifest'], *unit['artifacts']]:
            require(_binding(binding['path']) == binding, 'MLP8 scientific artifact changed')
        _recheck_producer(unit['producer_evidence'])
    for central in jobs:
        for binding in [central['job'], *central['logs'].values()]:
            require(_binding(binding['path']) == binding, 'MLP8 central completion evidence changed')
    if pilot_binding:
        require(_binding(pilot_binding['path'])['sha256'] == pilot_binding['sha256'], 'MLP8 pilot proof changed')
    validate_historical_inputs()
    require(current_provenance() == source and (review_sources(discovery=True) if discovery else review_sources()) == sources
        and _binding(inventory_path) == inventory_binding, 'MLP8 bindings changed before acceptance return')
    scope_prefix = 'survival_mlp8_discovery_v2_' if discovery else 'survival_mlp8_'
    return dict(schema_version=1, protocol=recipe.PROTOCOL, scope=scope_prefix + cohort + '_scientific_acceptance',
        passed=True, inference_performed=False, provenance=source, inventory=inventory_binding,
        historical_inputs=historical_binding(), validated_request_count=len(requests), validated_unit_count=len(units),
        paired_checks=pairs, units=units, central_jobs=jobs,
        admission_note='saved numerical/probability verification; operator reviews runtime and memory; no accuracy cutoff or superiority claim')


def verify_pilot_acceptance(path, *, discovery=False):
    source, path = current_provenance(), _ordinary_path(path)
    binding, proof = _binding(path), _read_json(path)
    scope = 'survival_mlp8_discovery_v2_pilot_scientific_acceptance' if discovery else 'survival_mlp8_pilot_scientific_acceptance'
    require(path == default_acceptance_path(source, discovery=discovery) and proof.get('scope') == scope
        and proof.get('passed') is True and proof.get('provenance') == source, 'invalid same-source MLP8 pilot proof')
    fresh = collect_acceptance(proof['inventory']['path'], discovery=True) if discovery else collect_acceptance(proof['inventory']['path'])
    require(fresh == proof and _binding(path) == binding, 'MLP8 pilot proof differs from fresh verification')
    return proof


def publish_acceptance(proof, cohort='pilot', *, discovery=False):
    source = current_provenance()
    scope_prefix = 'survival_mlp8_discovery_v2_' if discovery else 'survival_mlp8_'
    require(proof.get('passed') is True and proof.get('provenance') == source
        and proof.get('scope') == scope_prefix + cohort + '_scientific_acceptance', 'MLP8 proof publication scope/source mismatch')
    output = default_acceptance_path(source, cohort, discovery=discovery)
    pending = output.with_name('.' + output.name + '.pending')
    _ordinary_path(pending, exists=False)
    require(not output.exists() and not pending.exists(), 'preserve existing or partial MLP8 proof')
    data = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_path(output, exists=False)
    _ordinary_path(pending, exists=False)
    require(current_provenance() == source, 'source changed before MLP8 proof publication')
    with pending.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(current_provenance() == source, 'source changed; preserve partial MLP8 proof')
    _ordinary_path(pending)
    _ordinary_path(output, exists=False)
    os.link(pending, output)
    pending.unlink()
    require(output.read_bytes() == data and current_provenance() == source, 'MLP8 proof readback/source mismatch')
    return output
