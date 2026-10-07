"""Accept all twenty saved units and their actual scheduler producers."""
import argparse
import json
import os
from pathlib import Path

from ..infrastructure import evidence
from ..survival.acceptance import (CENTRAL_LOGS, CENTRAL_RETRIES, INVENTORY_ROOT, OUTBOX_ROOT,
    _binding, _ordinary_path, _read_json, _attempt_audit, _log_evidence, _recheck_producer)
from ..survival.artifacts import current_provenance, validate_execution
from ..survival_pooled.acceptance import _completed_job
from . import protocol
from .artifacts import ArtifactStore
from .inputs import historical_binding, validate_inputs
from .requests import build_requests, review_sources

PROOF_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-comparators-v1')
SCOPE = 'survival_comparators_v1_ten_seed_scientific_acceptance'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_pairs(records):
    require(set(records) == {(s, seed) for s in protocol.SCENARIOS for seed in protocol.SEEDS},
            'incomplete ten-seed comparator coverage')
    for seed in protocol.SEEDS:
        left, right = (records[s, seed] for s in protocol.SCENARIOS)
        require(left['site_sizes'] == right['site_sizes'] and all(left['fingerprints'][k] == right['fingerprints'][k]
            for k in ('evaluation', 'evaluation_truth')), 'comparator scenarios have different test data/truth')
        for key in ('raw_x_sha256', 'contrast_truth_sha256', 'permutation_sha256'):
            expected = left['discovery']['fedavg'][key]
            require(all(result['discovery'][tag][key] == expected for result in (left, right) for tag in protocol.TAGS),
                    'comparator methods/scenarios do not share covariates, truth and permutations')
    return dict(passed=True, complete_seed_count=10, scientific_units=20,
                historical_training_test_truth_paired=True, common_permutations_paired=True)


def collect_acceptance(inventory_path):
    path = _ordinary_path(inventory_path)
    require(INVENTORY_ROOT in path.parents, 'comparator inventory must be in project scratch reports')
    inventory_binding, inventory = _binding(path), _read_json(path)
    source = current_provenance()
    validate_inputs()
    sources = review_sources()
    require(inventory.get('schema_version') == 1 and inventory.get('bundle_ready') is True
        and inventory.get('bundle') == 'fedfisher-survival-comparators-v1-10seed'
        and inventory.get('provenance') == source and inventory.get('project_head') == source['git_commit']
        and inventory.get('worktree_status') == 'clean' and inventory.get('task') == protocol.PILOT_TASK
        and inventory.get('scientific_cell_count') == 20 and inventory.get('fitted_model_count') == 20
        and inventory.get('replayed_model_count') == 20 and inventory.get('formal_expansion_authorized') is False
        and inventory.get('coverage') == dict(scenarios=list(protocol.SCENARIOS), seeds=list(protocol.SEEDS), models=list(protocol.TAGS))
        and inventory.get('historical_inputs') == historical_binding()
        and inventory.get('review_sources') == sources, 'comparator inventory differs from fixed source/coverage')
    specs = build_requests(inventory['generation'])
    requests = inventory.get('requests')
    require(isinstance(requests, list) and len(requests) == len(specs) == inventory.get('request_count') == 20,
            'missing comparator requests')
    units, jobs, records, bindings, audit_cache = [], [], {}, [], {}

    def audit(job_id, attempt):
        key = job_id, attempt
        if key not in audit_cache:
            audit_cache[key] = _attempt_audit(job_id, attempt)
        return audit_cache[key]

    for request, spec in zip(requests, specs):
        raw, config = spec['request'], spec['configuration']
        request_path = _ordinary_path(request['path'])
        binding = _binding(request_path)
        require(OUTBOX_ROOT in request_path.parents and request_path.name == spec['filename']
            and _read_json(request_path) == raw and binding['sha256'] == request['sha256'] == spec['request_sha256']
            and all(request[k] == raw[k] for k in ('job_id', 'task', 'parameters'))
            and request['configuration'] == config and request['configuration_sha256'] == spec['configuration_sha256']
            and request['run_id'] == spec['run_id'] and request['seeds'] == spec['seed_sequence'],
            'comparator request changed from the reviewed recipe')
        bindings.append(binding)
        job, central = _completed_job(request)
        jobs.append(central)
        seed = spec['seed_sequence'][0]
        key = config['scenario'], seed
        require(key not in records, 'duplicate comparator unit')
        store = ArtifactStore(config, source)
        require(store.status(seed) == 'validated', 'invalid comparator scientific unit')
        directory = store.cell_dir(seed)
        result = _read_json(directory / store.result_name)
        started = _read_json(directory / 'started.json')['execution']
        producer = evidence.producer_evidence(request, job, started, logs_root=CENTRAL_LOGS,
            retries_root=CENTRAL_RETRIES, validate_execution=validate_execution, binding=_binding,
            read_json=_read_json, log_evidence=_log_evidence, audit=audit)
        records[key] = result
        units.append(dict(scenario=config['scenario'], seed=seed, run_id=store.run_id,
            manifest=_binding(store.directory / 'run_manifest.env'),
            artifacts=[_binding(directory / name) for name in (*store.artifact_names, store.done_name)],
            producer_evidence=producer, references=result['references'], fit_origin=result['fit_origin'],
            prediction=result['prediction'], discovery=result['discovery'], timing=result['timing'], memory=result['memory']))
    pairs = validate_pairs(records)
    for binding in bindings:
        require(_binding(binding['path']) == binding, 'comparator request changed during acceptance')
    for unit in units:
        for binding in [unit['manifest'], *unit['artifacts']]:
            require(_binding(binding['path']) == binding, 'comparator artifact changed during acceptance')
        _recheck_producer(unit['producer_evidence'])
    for central in jobs:
        for binding in [central['job'], *central['logs'].values()]:
            require(_binding(binding['path']) == binding, 'central comparator evidence changed')
    validate_inputs()
    require(current_provenance() == source and review_sources() == sources and _binding(path) == inventory_binding,
            'comparator source/inventory changed during acceptance')
    return dict(schema_version=1, protocol=protocol.PROTOCOL, scope=SCOPE, passed=True,
        inference_performed=False, provenance=source, inventory=inventory_binding,
        historical_inputs=historical_binding(), validated_request_count=20, validated_unit_count=20,
        paired_checks=pairs, units=units, central_jobs=jobs, formal_expansion_authorized=False,
        admission_note='ten paired seeds per scenario; saved scientific arithmetic and producer checks; no accuracy cutoff or convergence/superiority claim')


def publish_acceptance(proof):
    source = current_provenance()
    require(proof.get('passed') is True and proof.get('provenance') == source and proof.get('scope') == SCOPE,
            'comparator proof must match current clean source')
    output = _ordinary_path(PROOF_ROOT / ('ten-seed-' + source['provenance_sha256'] + '.acceptance.json'), exists=False)
    pending = output.with_name('.' + output.name + '.pending')
    _ordinary_path(pending, exists=False)
    require(not output.exists() and not pending.exists(), 'preserve existing or partial comparator proof')
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _ordinary_path(output, exists=False)
    _ordinary_path(pending, exists=False)
    require(current_provenance() == source, 'source changed before comparator proof publication')
    with pending.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(current_provenance() == source, 'source changed; preserve partial proof')
    _ordinary_path(pending)
    _ordinary_path(output, exists=False)
    os.link(pending, output)
    pending.unlink()
    require(output.read_bytes() == data and current_provenance() == source, 'comparator proof readback changed')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args(argv)
    proof = collect_acceptance(args.inventory)
    output = publish_acceptance(proof)
    print(json.dumps(dict(passed=True, proof=str(output), validated_units=20, new_fedavg_models=20,
                         restored_dslasso_models=20), sort_keys=True))
    return 0
