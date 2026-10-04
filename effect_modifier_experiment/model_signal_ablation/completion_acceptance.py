"""Saved-evidence admission and exact mixed-source completion of the sealed epoch."""
from collections import Counter, defaultdict
from pathlib import Path
import re

from ..infrastructure.artifacts import canonical, digest
from . import completion_archive as archive, completion_protocol as protocol
from . import recovery_acceptance as previous, recovery_protocol as original
from .artifacts import current_provenance

PROOF_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/model-signal-ablation-v1/completion-v3')
_require, _binding, _read_json = previous._require, previous._binding, previous._read_json
old = previous.old
verify_closeout = archive.verify_closeout


def default_acceptance_path(provenance=None):
    provenance = current_provenance() if provenance is None else provenance
    checksum = provenance.get('provenance_sha256')
    _require(provenance.get('git_dirty') == '0' and isinstance(checksum, str)
        and re.fullmatch('[0-9a-f]{64}', checksum), 'completion acceptance requires clean source')
    return previous.old._ordinary_path(PROOF_ROOT / ('pilot-' + checksum + '.acceptance.json'), exists=False)


def _inventory(path, provenance, cohort, closeout, snapshot):
    from .completion_requests import build_requests, review_sources
    path = previous.old._ordinary_path(path)
    _require(previous.REPORT_ROOT in path.parents, 'completion inventory must be in scheduler reports')
    binding = snapshot.add(_binding(path))
    inventory = _read_json(path)
    pilot = cohort == 'pilot'
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    count, units = (8, 16) if pilot else (24, 896)
    _require(inventory.get('schema_version') == 1 and inventory.get('bundle_ready') is True
        and inventory.get('provenance') == provenance and inventory.get('task') == task
        and inventory.get('closeout') == closeout['binding']
        and inventory.get('review_sources') == review_sources()
        and inventory.get('request_count') == count and inventory.get('scientific_cell_count') == units,
        'completion inventory differs from reviewed source, archive or exact scope')
    snapshot.walk(inventory['review_sources'])
    expected = {spec['request']['job_id']: spec for spec in build_requests(inventory['generation'], cohort)}
    rows, seen, paths = inventory['requests'], set(), set()
    _require(len(rows) == len(expected) == count, 'completion request coverage is incomplete')
    for row in rows:
        job_id = row['job_id']
        _require(job_id in expected and job_id not in seen, 'duplicate or unknown completion request')
        seen.add(job_id)
        spec = expected[job_id]
        path = previous.old._ordinary_path(row['path'])
        _require(previous.old.OUTBOX_ROOT in path.parents and path.name == spec['filename'] and path not in paths,
                 'completion request is not a unique canonical outbox file')
        paths.add(path)
        item = snapshot.add(_binding(path))
        _require(item['sha256'] == row['sha256'] == spec['request_sha256']
            and canonical(_read_json(path)) == canonical(spec['request'])
            and all(canonical(row[key]) == canonical(spec[key]) for key in ('configuration', 'configuration_sha256', 'run_id'))
            and row['parameters'] == spec['request']['parameters'] and row['task'] == task
            and row['seeds'] == list(protocol.seed_sequence(task, row['parameters']))
            and row['splits'] == list(protocol.SPLITS), 'completion request configuration or bytes changed')
    return inventory, binding


def validate_pilot_observations(records, references):
    expected = {(p['setting'], p['cell'], p['variant'], 0, split)
                for p in protocol.PILOT_PARAMETERS for split in protocol.SPLITS}
    _require(set(records) == set(references) == expected and len(expected) == 16,
             'completion pilot must cover all sixteen bounded units')
    thin = {}
    for key, row in records.items():
        _require(row['outcome'] == 'validated', 'previous successful pilot became a noncompletion')
        previous.validate_numeric_parity(row['result'], references[key])
        thin[key] = previous._terminal_thin(row['result'], 'validated')
    return dict(passed=True, exact_success_parity_units=16,
        terminal_training_pairs=previous.validate_terminal_training_pairs(thin),
        paired_checks=validate_bounded_pairs(thin))


def validate_bounded_pairs(records):
    """The bounded pilot's two recipes, four variants and both splits only."""
    for split in protocol.SPLITS:
        group = {(cell, variant): records[protocol.SETTING, cell, variant, 0, split]
                 for cell in protocol.CELLS for variant in protocol.VARIANTS}
        _require(len({row['fingerprints']['test'] for row in group.values()}) == 1,
                 'bounded pilot common test set changed')
        for variant in protocol.VARIANTS:
            a, b = [group[cell, variant] for cell in protocol.CELLS]
            _require(all(a['fingerprints'][key] == b['fingerprints'][key] for key in ('training', 'test'))
                and all(a[key] == b[key] for key in ('site_sizes', 'label_counts', 'evaluation')),
                'bounded pilot method data or evaluation plans differ')
        for cell in protocol.CELLS:
            for left, right in (('control', 'site_gaussian_shift'),
                                ('local_permutation', 'site_gaussian_local_permutation')):
                a, b = group[cell, left], group[cell, right]
                _require(all(a[key] == b[key] for key in ('site_sizes', 'label_counts', 'evaluation')),
                         'bounded pilot Gaussian factor changed evaluation or site sizes')
    return dict(passed=True, common_test_groups=2, method_data_pairs=8, factorial_scope_pairs=8)


def validate_composite_coverage(retained, replacement, replacement_keys):
    expected = {(setting, cell, variant, seed, split)
        for setting, recipe in original.SETTINGS.items() for cell in recipe['cells']
        for variant in original.VARIANTS for seed in range(300) for split in original.SPLITS}
    _require(len(retained) == 171904 and len(replacement) == 896
        and set(replacement) == replacement_keys and not (set(retained) & set(replacement))
        and set(retained) | set(replacement) == expected, 'mixed-source coverage has gaps, duplicates or audit-only retention')
    counts, conditions = Counter(), []
    for setting, recipe in original.SETTINGS.items():
        for cell in recipe['cells']:
            for variant in original.VARIANTS:
                for split in original.SPLITS:
                    units = [replacement.get((setting, cell, variant, seed, split),
                             retained.get((setting, cell, variant, seed, split))) for seed in range(300)]
                    state = Counter(row['outcome'] for row in units)
                    _require(set(state) <= {'validated', 'noncompleted'}, 'unknown terminal classification')
                    if state['noncompleted']:
                        method = original.method_config(cell)
                        _require(method['family'] == 'fedavg'
                            and method['optimizer']['stopping_mode'] == 'validation_patience'
                            and method['optimizer']['max_rounds'] == 200,
                            'noncompletion is outside the unchanged patience budget')
                    counts.update(state)
                    conditions.append(dict(setting=setting, cell=cell, variant=variant, split=split,
                        planned=300, completed=state['validated'], noncompleted=state['noncompleted']))
    old_caps = sum(row['outcome'] == 'noncompleted' for row in retained.values())
    _require(old_caps == 6, 'six historical noncompletions were lost or promoted')
    return dict(exact_coverage=True, scientific_units=172800, retained_units=171904, replacement_units=896,
        completed_units=counts['validated'], noncompleted_units=counts['noncompleted'], retained_noncompleted_units=6,
        whole_cohort_scientific_success=counts['noncompleted'] == 0, conditions=conditions,
        excluded_failed_shard_saved_models=287, diagnostic_checkpoints_used_for_performance=False)


def _collect(inventory_path, cohort):
    from .completion import CompletionStore
    from .recovery import validate_shard_summary, shard_summary_path
    provenance = current_provenance()
    closeout = verify_closeout()
    snapshot = previous._EvidenceSnapshot()
    inventory, inventory_binding = _inventory(inventory_path, provenance, cohort, closeout, snapshot)
    prerequisites = dict(closeout=closeout['binding'], pilot=None)
    if cohort == 'formal':
        path = default_acceptance_path(provenance)
        prerequisite = verify_pilot_acceptance(path)
        binding = snapshot.add(_binding(path))
        _require(prerequisite['closeout'] == closeout['binding']
            and inventory.get('pilot_acceptance') == dict(binding, evidence=prerequisite),
            'formal completion lacks fresh same-source pilot acceptance')
        prerequisites['pilot'] = binding
    attempts = previous._attempt_index({row['job_id'] for row in inventory['requests']})
    by_run = defaultdict(list)
    for row in inventory['requests']:
        by_run[row['run_id']].append(row)
    units, shards, records, pilot_results = [], [], {}, {}
    for run_id, requests in by_run.items():
        config = requests[0]['configuration']
        store = CompletionStore(config, provenance, prerequisites)
        store._manifest()
        manifest = snapshot.add(_binding(store.directory / 'run_manifest.env'))
        expected_names = {'run_manifest.env', 'shard_summaries'} | {
            'seed_%03d_%s' % (seed, split) for row in requests for seed in row['seeds'] for split in protocol.SPLITS}
        _require({p.name for p in store.directory.iterdir()} == expected_names,
                 'completion run contains missing or unplanned outputs')
        snapshot.layouts[str(store.directory)] = expected_names
        summary_root = store.directory / 'shard_summaries'
        expected_summaries = {row['job_id'] + '.json' for row in requests}
        _require(summary_root.is_dir() and {p.name for p in summary_root.iterdir()} == expected_summaries,
                 'completion summaries are missing or contain foreign shards')
        snapshot.layouts[str(summary_root)] = expected_summaries
        for request in requests:
            job, central = previous._completed_job(request, snapshot)
            terminal_units = []
            for seed in request['seeds']:
                for split in protocol.SPLITS:
                    status, result, receipt, artifacts, terminal = previous._read_terminal(store, seed, split, snapshot)
                    producer = previous._producer(request, job, receipt, attempts, snapshot)
                    terminal_units.append(terminal)
                    row = dict(setting=config['setting'], cell=config['cell'], variant=config['variant'],
                        seed=seed, split=split, outcome=status, run_id=run_id, job_id=job['job_id'])
                    key = previous._unit_key(row)
                    _require(key not in records, 'duplicate replacement scientific unit')
                    records[key] = previous._terminal_thin(result, status)
                    if cohort == 'pilot':
                        pilot_results[key] = dict(outcome=status, result=result)
                        row.update(artifacts=artifacts, manifest=manifest, producer_evidence=producer,
                            timing=result['timing'], memory=result.get('memory'))
                    units.append(row)
            summary = validate_shard_summary(store, job, verified_units=terminal_units)
            summary_binding = snapshot.add(_binding(shard_summary_path(store, job)))
            writer = previous._producer(request, job, summary['producer_execution'], attempts, snapshot)
            shards.append(dict(job_id=job['job_id'], run_id=run_id, manifest=manifest, summary=summary_binding,
                central=central, summary_producer=writer, counts=summary['counts']))
        store._manifest()
    if cohort == 'pilot':
        references = {key: archive.reference_result(closeout, key) for key in closeout['pilot_references']}
        checks = validate_pilot_observations(pilot_results, references)
        body = dict(scope='model_signal_completion_pilot_acceptance', passed=True, formal_eligible=True,
            pilot_inventory=inventory_binding, validated_request_count=8, validated_unit_count=16,
            complete_units=16, noncompleted_units=0, checks=checks, units=units,
            historical_reference=closeout['pilot_reference_binding'])
    else:
        retained, historical = archive.load_retained_evidence(closeout)
        coverage = validate_composite_coverage(retained, records, closeout['replacement_keys'])
        records.update(retained)
        terminal_pairs = previous.validate_terminal_training_pairs(records)
        complete = {key: row for key, row in records.items() if row['outcome'] == 'validated'}
        body = dict(scope='model_signal_completion_formal_acceptance', passed=True, formal_eligible=True,
            formal_inventory=inventory_binding, pilot_acceptance=prerequisites['pilot'],
            validated_request_count=24, validated_unit_count=896, coverage=coverage,
            terminal_training_pairs=terminal_pairs, paired_checks=previous.validate_available_pairs(complete),
            historical_evidence=historical, historical_provenance=closeout['proof']['source_provenance'])
    snapshot.recheck()
    closeout['reader'].recheck()
    _require(archive.verify_science_sources() == closeout['source_guard'] and current_provenance() == provenance,
             'source changed during completion acceptance')
    return dict(schema_version=1, protocol=protocol.PROTOCOL, provenance=provenance,
        closeout=closeout['binding'], source_guard=closeout['source_guard'],
        inference_or_fitting_performed=False, shards=shards,
        admission_note='saved-evidence acceptance; operator separately reviews resources and admission', **body)


def collect_pilot_acceptance(inventory_path):
    return _collect(inventory_path, 'pilot')


def collect_formal_acceptance(inventory_path):
    return _collect(inventory_path, 'formal')


def verify_pilot_acceptance(path):
    provenance = current_provenance()
    path = previous.old._ordinary_path(path)
    _require(path == default_acceptance_path(provenance), 'pilot proof must use its source-addressed canonical path')
    binding, proof = _binding(path), _read_json(path)
    _require(proof.get('schema_version') == 1 and proof.get('protocol') == protocol.PROTOCOL
        and proof.get('scope') == 'model_signal_completion_pilot_acceptance'
        and proof.get('passed') is True and proof.get('formal_eligible') is True
        and proof.get('provenance') == provenance and proof.get('closeout') == protocol.SOURCE_CLOSEOUT,
        'invalid or stale completion pilot proof')
    fresh = collect_pilot_acceptance(proof['pilot_inventory']['path'])
    _require(canonical(fresh) == canonical(proof) and _binding(path) == binding,
             'completion pilot differs from freshly verified complete evidence')
    return proof


def check_current_pilot_gate():
    return verify_pilot_acceptance(default_acceptance_path())
