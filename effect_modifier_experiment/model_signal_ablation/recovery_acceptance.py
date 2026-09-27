"""Saved-evidence gates for the unchanged-science recovery epoch.

Historical success belongs to its original source and producing attempt.  A
verified round cap is an execution terminal state, never a completed model.
This module performs no fitting, data generation, or model inference.
"""
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path
import re
import subprocess

from ..infrastructure import evidence
from ..infrastructure.artifacts import canonical, digest
from . import acceptance as old, protocol as original, recovery_protocol as protocol
from .artifacts import current_provenance

PROJECT_ROOT = old.PROJECT_ROOT
REPORT_ROOT = old.REPORT_ROOT
PROOF_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/model-signal-ablation-v1/recovery-v2')
MUTABLE_HISTORICAL_SOURCES = frozenset({
    'effect_modifier_experiment/causal_comparison/runner.py',
    'scripts/fedfisher_server_scheduler_adapter.py',
    'tests/test_model_signal_adapter.py',
    'docs/codex/SCHEDULER_HANDOFF.md',
    'effect_modifier_experiment/model_signal_ablation/README.md',
})
COUNTS = dict(whole_requests=4608, whole_units=172800, retained_requests=129,
    retained_units=4838, replacement_requests=4479, replacement_units=167962,
    excluded_old_valid_units=44, unadmitted_requests=4476, unadmitted_units=167848)
ACCEPTED_OLD = 'accepted_from_centrally_successful_shard'
_require = old._require
_binding = old._binding
_read_json = old._read_json


def _unit_key(row):
    return tuple(row[key] for key in ('setting', 'cell', 'variant', 'seed', 'split'))


def _request_key(parameters):
    return tuple(parameters[key] for key in ('setting', 'cell', 'variant', 'shard_index', 'shard_count'))


def _checked(binding):
    _require(_binding(binding['path']) == binding, 'bound recovery evidence changed')
    return _read_json(binding['path'])


def _historical_sources(proof):
    """Verify archived bytes against Git; only named observers may now differ."""
    source = proof['source_files']
    _require(isinstance(source, dict) and len(source) == 440, 'historical source inventory changed')
    commit = proof['provenance']['git_commit']
    _require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'invalid historical commit')
    names = sorted(source)
    for name in names:
        path = PROJECT_ROOT / name
        _require(not Path(name).is_absolute() and '..' not in Path(name).parts
                 and source[name]['path'] == str(path), 'historical source escaped canonical tree')
    # A single read-only Git process avoids hundreds of subprocesses or source scans.
    response = subprocess.run(['git', 'cat-file', '--batch'], cwd=PROJECT_ROOT,
        input=''.join(commit + ':' + name + '\n' for name in names).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    stream = io.BytesIO(response)
    for name in names:
        header = stream.readline().split()
        _require(len(header) == 3 and header[1] == b'blob', 'historical source missing from frozen Git commit')
        payload = stream.read(int(header[2]))
        _require(stream.read(1) == b'\n' and hashlib.sha256(payload).hexdigest() == source[name]['sha256'],
                 'historical source hash disagrees with frozen Git commit')
        if name not in MUTABLE_HISTORICAL_SOURCES:
            _require(_binding(source[name]['path']) == source[name],
                     'recovery changed a preserved scientific or supporting source: ' + name)
    _require(stream.read() == b'', 'unexpected historical Git object output')
    return {item['path']: item for item in source.values()}


class _EvidenceSnapshot:
    """Deduplicated byte/producers snapshot, with a full final reread."""
    def __init__(self, historical_sources=None):
        self.files = {}
        self.producers = {}
        self.layouts = {}
        self.historical_sources = historical_sources

    def add(self, binding):
        binding = {key: binding[key] for key in ('path', 'sha256')}
        path = binding['path']
        _require(path not in self.files or self.files[path] == binding, 'conflicting evidence bindings')
        if path not in self.files:
            self._check(binding)
            self.files[path] = binding
        return binding

    def _check(self, binding):
        if self.historical_sources is not None and PROJECT_ROOT in Path(binding['path']).parents:
            _require(self.historical_sources.get(binding['path']) == binding,
                     'historical review source is absent from the frozen source inventory')
        else:
            _require(_binding(binding['path']) == binding, 'historical or scientific evidence bytes changed')

    def walk(self, value):
        if isinstance(value, dict):
            if {'retry', 'logs', 'audit', 'producer_attempt'} <= set(value):
                key = (value['retry']['path'], value['producer_attempt'])
                _require(key not in self.producers or self.producers[key] == value,
                         'inconsistent producing-attempt evidence')
                self.producers[key] = value
                return  # Audit sha256 binds a byte range, not the entire growing file.
            if {'path', 'sha256'} <= set(value) and 'offset' not in value:
                self.add(value)
            for part in value.values():
                self.walk(part)
        elif isinstance(value, list):
            for part in value:
                self.walk(part)

    def recheck(self):
        for binding in self.files.values():
            self._check(binding)
        for producer in self.producers.values():
            old._recheck_producer(producer)
        for path, names in self.layouts.items():
            _require({entry.name for entry in Path(path).iterdir()} == names,
                     'scientific directory changed during acceptance')


def _validate_retention(proof, formal, inventory, pilot):
    """Close every old request/unit exactly; failed-shard successes stay excluded."""
    _require(proof.get('counts') == COUNTS and formal.get('source_provenance') == proof['provenance']
        and inventory.get('provenance') == proof['provenance'] and pilot.get('provenance') == proof['provenance']
        and formal.get('existing_saved_artifact_checks_passed') is True
        and formal.get('whole_cohort_accepted') is False and formal.get('no_active_cohort_jobs') is True
        and formal.get('accepted_successful_shard_count') == 129
        and formal.get('accepted_successful_unit_count') == 4838
        and formal.get('formal_inventory') == proof['formal_inventory']
        and inventory.get('task') == original.FORMAL_TASK and inventory.get('request_count') == 4608
        and inventory.get('scientific_cell_count') == 172800
        and pilot.get('passed') is True and pilot.get('validated_request_count') == 288
        and pilot.get('validated_unit_count') == 576, 'source closeout has incompatible historical scope')
    requests = inventory['requests']
    by_id = {row['job_id']: row for row in requests}
    snapshots = {row['job_id']: row for row in formal['requests']}
    _require(len(requests) == len(by_id) == len(snapshots) == 4608 and set(by_id) == set(snapshots),
             'historical request coverage is incomplete or duplicated')
    expected = {(setting, cell, variant, index, 16)
        for setting, recipe in original.SETTINGS.items() for cell in recipe['cells']
        for variant in original.VARIANTS for index in range(16)}
    seen, planned = set(), {}
    for job_id, row in by_id.items():
        p = original.validate_parameters(original.FORMAL_TASK, row['parameters'])
        key = _request_key(p)
        _require(key not in seen and key in expected, 'historical shard coverage changed')
        seen.add(key)
        config = original.run_config(original.FORMAL_TASK, p)
        seeds = list(original.seed_sequence(original.FORMAL_TASK, p))
        _require(row['configuration'] == config and row['configuration_sha256'] == digest(config)
            and row['run_id'] == config['run_id'] and row['task'] == original.FORMAL_TASK
            and row['seeds'] == seeds and row['splits'] == list(original.SPLITS)
            and snapshots[job_id]['parameters'] == p and snapshots[job_id]['seeds'] == seeds
            and snapshots[job_id]['planned_unit_count'] == 2 * len(seeds),
            'historical scientific recipe or deterministic ownership changed')
        for seed in seeds:
            for split in original.SPLITS:
                unit = (p['setting'], p['cell'], p['variant'], seed, split)
                _require(unit not in planned, 'historical unit has multiple owners')
                planned[unit] = job_id
    _require(seen == expected and len(planned) == 172800, 'historical total coverage changed')
    retained = {job_id for job_id, row in snapshots.items() if row['state'] == 'completed'}
    replacements = set(by_id) - retained
    _require(len(retained) == 129 and len(replacements) == 4479
        and len(proof['retained_job_ids']) == 129 and set(proof['retained_job_ids']) == retained
        and len(proof['replacement_job_ids']) == 4479 and set(proof['replacement_job_ids']) == replacements,
        'retained/replacement request partition changed')
    accepted, excluded, seen_units = [], [], set()
    for unit in formal['units']:
        key = _unit_key(unit)
        _require(key in planned and key not in seen_units and unit['job_id'] == planned[key],
                 'historical scientific unit identity or ownership changed')
        seen_units.add(key)
        if unit['classification'] == ACCEPTED_OLD:
            _require(unit['job_id'] in retained and unit['producer_key'] in formal['producer_evidence'],
                     'retained unit lacks a centrally successful producer')
            producer = formal['producer_evidence'][unit['producer_key']]
            _require(producer['execution_receipt']['job_id'] == unit['job_id'], 'old producer identity mismatch')
            accepted.append(unit)
        else:
            _require(unit['job_id'] in replacements and unit['producer_key'] is None,
                     'failed or unknown shard was counted as retained success')
            excluded.append(unit)
    required = {key for key, owner in planned.items() if owner in retained}
    _require({_unit_key(row) for row in accepted} == required and len(accepted) == 4838
             and len(excluded) == 44, 'historical retained/excluded scientific coverage changed')
    controls = [row for row in excluded if row['classification'] == 'artifact_valid_in_failed_scientific_shard']
    control_keys = {('small_mlp', 'fedavg_small_e1_patience', variant, seed, split)
        for variant in protocol.DIAGNOSTIC_VARIANTS for seed, split in ((9, 'iid'), (9, 'noniid'), (25, 'iid'))}
    _require(len(controls) == 6 and {_unit_key(row) for row in controls} == control_keys
        and proof['known_success_diagnostic_controls'] == controls,
        'the six historical diagnostic success controls changed')
    _require(proof['known_exhaustion_controls'] == formal['failures'] and len(formal['failures']) == 2,
             'known cap controls changed')
    return accepted, excluded


def verify_closeout(path=None):
    """Fully recheck the immutable historical anchor; no current-source v1 fiction."""
    path = protocol.SOURCE_CLOSEOUT['path'] if path is None else str(path)
    _require(path == protocol.SOURCE_CLOSEOUT['path'], 'recovery requires the fixed reviewed source closeout')
    proof = _checked(protocol.SOURCE_CLOSEOUT)
    _require(proof.get('schema_version') == 1 and proof.get('scope') == 'model_signal_v1_source_closeout'
        and proof.get('source_release_passed') is True and proof.get('whole_cohort_accepted') is False
        and proof.get('no_active_project_jobs') is True
        and proof.get('new_execution_authorized_by_closeout') is False,
        'historical source closeout is not the reviewed release anchor')
    source = _historical_sources(proof)
    snapshot = _EvidenceSnapshot(source)
    snapshot.walk(protocol.SOURCE_CLOSEOUT)
    for key in ('formal_acceptance', 'formal_inventory', 'pilot_acceptance', 'survival_acceptance'):
        snapshot.walk(proof[key])
    snapshot.walk(proof['verification_reports'])
    formal, inventory, pilot = (_checked(proof[key]) for key in
                                ('formal_acceptance', 'formal_inventory', 'pilot_acceptance'))
    accepted, excluded = _validate_retention(proof, formal, inventory, pilot)
    snapshot.walk(formal['evidence_bindings'])
    snapshot.walk(formal['producer_evidence'])
    snapshot.walk(inventory)
    snapshot.walk(pilot)
    # The current source may change observers only; archived science is checked above.
    snapshot.recheck()
    for row in formal['requests']:
        if row['state'] == 'not_admitted':
            _require(not (old.CENTRAL_JOBS / (row['job_id'] + '.json')).exists(),
                     'an old unadmitted request was admitted after source closeout')
    _require(_binding(path) == protocol.SOURCE_CLOSEOUT, 'historical closeout changed during verification')
    return dict(binding=dict(protocol.SOURCE_CLOSEOUT), proof=proof, old_inventory=inventory,
        old_pilot=pilot, formal_acceptance=formal, retained_request_ids=sorted(proof['retained_job_ids']),
        retained_units=accepted, excluded_units=excluded, evidence_snapshot=snapshot)


_MEASUREMENTS = frozenset({'timing', 'aggregation_cpu_seconds', 'aggregation_wall_seconds',
    'training_cpu_seconds', 'training_wall_seconds', 'validation_cpu_seconds', 'validation_wall_seconds'})


def _numeric(value):
    if isinstance(value, dict):
        return {key: _numeric(part) for key, part in value.items() if key not in _MEASUREMENTS}
    if isinstance(value, list):
        return [_numeric(part) for part in value]
    return value


def validate_numeric_parity(result, reference):
    """Compare all scientific fields, excluding only identity/RSS and elapsed time."""
    for value in (result, reference):
        _require(value.get('status') == 'complete', 'historical success must remain a completed model')
    keys = {'status', 'fingerprints', 'fit', 'evaluation', 'site_sizes', 'label_counts'}
    _require(all(key in result and key in reference for key in keys), 'incomplete success parity evidence')
    _require(canonical(_numeric({key: result[key] for key in keys}))
             == canonical(_numeric({key: reference[key] for key in keys})),
             'recovery changed historical model, data, evaluation, or numeric fitting path')
    return True


def _unit_result(unit):
    bindings = unit.get('artifacts', unit.get('artifact_bindings'))
    matches = [item for item in bindings if Path(item['path']).name == 'result.json']
    _require(len(matches) == 1, 'historical reference lacks one result binding')
    return _checked(matches[0])


def _pilot_sets():
    baseline = {(setting, cell, variant, 0, split)
        for setting, recipe in protocol.SETTINGS.items() for cell in recipe['cells']
        for variant in protocol.VARIANTS for split in protocol.SPLITS}
    diagnostic = {('small_mlp', 'fedavg_small_e1_patience', variant, seed, split)
        for variant in protocol.DIAGNOSTIC_VARIANTS for seed in protocol.DIAGNOSTIC_SEEDS
        for split in protocol.SPLITS}
    controls = {key for key in diagnostic if key[3:] in ((9, 'iid'), (9, 'noniid'), (25, 'iid'))}
    caps = {key for key in diagnostic if key[3:] == (25, 'noniid')}
    return baseline, diagnostic, controls, caps


def validate_pilot_observations(records, references):
    """Exact 588-unit gate: 582 preserved successes, two required caps, four new cases."""
    baseline, diagnostic, controls, caps = _pilot_sets()
    _require(set(records) == baseline | diagnostic and len(records) == 588,
             'recovery pilot requires exactly all 588 assigned scientific units')
    _require(set(references) == baseline | controls and len(references) == 582,
             'recovery pilot requires 576 original pilot and six diagnostic success references')
    for key in baseline | controls:
        _require(records[key]['outcome'] == 'validated', 'historical pilot/control success became noncompleted')
        validate_numeric_parity(records[key]['result'], references[key])
    return _pilot_terminal_checks(records, baseline | controls)


def _pilot_terminal_checks(records, parity_checked):
    """Finish checks after one-at-a-time full historical numeric comparisons."""
    baseline, diagnostic, controls, caps = _pilot_sets()
    _require(set(records) == baseline | diagnostic and len(records) == 588
        and parity_checked == baseline | controls,
        'recovery pilot coverage or completed numeric parity is incomplete')
    for key in baseline | controls:
        _require(records[key]['outcome'] == 'validated', 'historical pilot/control success became noncompleted')
    for key in caps:
        _require(records[key]['outcome'] == 'noncompleted', 'known seed25/nonIID cap did not reproduce')
    for key in diagnostic:
        _require(records[key]['outcome'] in ('validated', 'noncompleted'), 'diagnostic lacks a verified terminal state')
    baseline_records = {(setting, cell, variant, split): records[setting, cell, variant, 0, split]['result']
                        for setting, cell, variant, _, split in baseline}
    pairs = old.validate_pairs(baseline_records)
    for seed in protocol.DIAGNOSTIC_SEEDS:
        for split in protocol.SPLITS:
            a, b = (records['small_mlp', 'fedavg_small_e1_patience', variant, seed, split]
                    for variant in protocol.DIAGNOSTIC_VARIANTS)
            _require(a['outcome'] == b['outcome'], 'paired Gaussian diagnostics changed completion state')
            _require(all(_numeric(a['result'][key]) == _numeric(b['result'][key])
                for key in ('fingerprints', 'fit', 'site_sizes', 'label_counts')),
                'paired Gaussian diagnostics changed training data or numeric path')
            _require(a.get('numeric_fit_sha256') == b.get('numeric_fit_sha256'),
                     'paired Gaussian diagnostics changed full numeric fitting path')
    return dict(baseline_success_parity=576, diagnostic_success_parity=6,
        expected_caps_reproduced=2, seed41_terminal_units=4, baseline_pairs=pairs,
        diagnostic_paired_paths=6, passed=True)


def validate_composite_coverage(retained, replacement):
    """Retain the full 300-seed denominator, including verified missing performance."""
    expected = {(setting, cell, variant, seed, split)
        for setting, recipe in protocol.SETTINGS.items() for cell in recipe['cells']
        for variant in protocol.VARIANTS for seed in range(300) for split in protocol.SPLITS}
    observed, counts = set(), defaultdict(Counter)
    _require(len(retained) == 4838 and len(replacement) == 167962,
             'formal composite must contain 4838 historical and 167962 recovery units')
    for old_epoch, rows in ((True, retained), (False, replacement)):
        for row in rows:
            key = _unit_key(row)
            _require(key in expected and key not in observed, 'duplicate or outside-scope composite unit')
            observed.add(key)
            if old_epoch:
                _require(row.get('classification') == ACCEPTED_OLD and row.get('producer_key'),
                         'excluded old partial/unknown results cannot fill formal coverage')
                outcome = 'validated'
            else:
                outcome = row.get('outcome')
                _require(outcome in ('validated', 'noncompleted'), 'new formal unit lacks validated terminal evidence')
                if outcome == 'noncompleted':
                    _require(protocol.CELLS[row['cell']]['family'] == 'fedavg'
                        and row['cell'].endswith('_patience'), 'noncompletion broadened beyond patience FedAvg')
            counts[key[:3] + (key[4],)][outcome] += 1
    _require(observed == expected, 'composite scientific coverage has missing planned units')
    conditions = []
    for key, value in sorted(counts.items()):
        _require(sum(value.values()) == 300, 'condition lost the 300-seed denominator')
        conditions.append(dict(zip(('setting', 'cell', 'variant', 'split'), key),
            planned_seeds=300, completed=value['validated'], noncompleted=value['noncompleted'],
            completion_rate=value['validated'] / 300,
            performance_scope='conditional_on_scientifically_completed_seeds'))
    capped = sum(row['noncompleted'] for row in conditions)
    return dict(exact_coverage=True, scientific_units=172800, retained_units=4838,
        recovery_units=167962, completed_units=172800-capped, noncompleted_units=capped,
        whole_cohort_scientific_success=capped == 0, conditions=conditions,
        diagnostic_checkpoints_used_for_performance=False)


def default_acceptance_path(provenance=None):
    provenance = current_provenance() if provenance is None else provenance
    checksum = provenance.get('provenance_sha256')
    _require(provenance.get('git_dirty') == '0' and isinstance(checksum, str)
        and re.fullmatch('[0-9a-f]{64}', checksum), 'recovery acceptance requires clean source')
    return old._ordinary_path(PROOF_ROOT / ('pilot-' + checksum + '.acceptance.json'), exists=False)


def _inventory(path, provenance, cohort, closeout, snapshot):
    from scripts.prepare_model_signal_recovery_requests import build_requests, review_sources
    path = old._ordinary_path(path)
    _require(REPORT_ROOT in path.parents, 'recovery inventory must be in scheduler reports')
    binding = snapshot.add(_binding(path))
    inventory = _read_json(path)
    task = protocol.PILOT_TASK if cohort == 'pilot' else protocol.FORMAL_TASK
    _require(inventory.get('schema_version') == 1 and inventory.get('bundle_ready') is True
        and inventory.get('provenance') == provenance and inventory.get('task') == task
        and inventory.get('closeout') == closeout['binding']
        and inventory.get('review_sources') == review_sources(),
        'recovery inventory differs from reviewed current source or historical anchor')
    _require(inventory.get('request_count') == (290 if cohort == 'pilot' else 4479)
        and inventory.get('scientific_cell_count') == (588 if cohort == 'pilot' else 167962),
        'recovery inventory has incomplete scientific scope')
    snapshot.walk(inventory['review_sources'])
    specs = build_requests(inventory['generation'], cohort, closeout=closeout)
    expected = {row['request']['job_id']: row for row in specs}
    requests, seen, paths = inventory['requests'], set(), set()
    _require(len(requests) == len(expected), 'duplicate or missing recovery requests')
    for row in requests:
        job_id = row['job_id']
        _require(job_id in expected and job_id not in seen, 'unknown or duplicate recovery job')
        seen.add(job_id)
        spec = expected[job_id]
        path = old._ordinary_path(row['path'])
        _require(old.OUTBOX_ROOT in path.parents and path.name == spec['filename'] and path not in paths,
                 'recovery request is not a unique canonical outbox file')
        paths.add(path)
        item = snapshot.add(_binding(path))
        _require(item['sha256'] == row['sha256'] == spec['request_sha256']
            and _read_json(path) == spec['request']
            and all(row[key] == spec[key] for key in ('configuration', 'configuration_sha256', 'run_id'))
            and row['parameters'] == spec['request']['parameters'] and row['task'] == task
            and row['seeds'] == list(protocol.seed_sequence(task, row['parameters']))
            and row['splits'] == list(protocol.SPLITS), 'recovery request recipe, bytes, or scope changed')
    return inventory, binding


def _attempt_index(job_ids):
    """Read the central append-only audit once, preserving exact attempt bindings."""
    path = old._ordinary_path(old.CENTRAL_AUDIT)
    active, found, offset = {}, {}, 0
    with path.open('rb') as stream:
        for line in stream:
            start, offset = offset, offset + len(line)
            if not line.endswith(b'\n'):
                continue
            row = old._json_bytes(line)
            job_id = row.get('job_id')
            if job_id not in job_ids:
                continue
            _require(row.get('schema_version') == 2, 'invalid central producing audit schema')
            name = None
            if row.get('event') == 'job_attempt_started':
                attempt = row.get('attempt')
                _require(type(attempt) is int and attempt > 0, 'invalid central attempt')
                active[job_id], name = attempt, 'attempt_started'
            elif row.get('event') == 'job_transition' and row.get('destination') == 'running':
                _require(job_id in active and row.get('project') == 'FedFisher' and row.get('source') == 'pending',
                         'central running transition lacks a valid producing attempt')
                name = 'running_transition'
            if name is not None:
                key = job_id, active[job_id]
                parts = found.setdefault(key, {})
                _require(name not in parts, 'duplicate central producing-attempt event')
                parts[name] = dict(path=str(path), offset=start, length=len(line),
                    sha256=hashlib.sha256(line).hexdigest(), record=row)
    return found


def _completed_job(request, snapshot):
    binding = snapshot.add(_binding(old.CENTRAL_JOBS / (request['job_id'] + '.json')))
    job = _read_json(binding['path'])
    _require(job.get('schema_version') == 2 and job.get('project') == 'FedFisher'
        and job.get('job_id') == request['job_id'] and job.get('task') == request['task']
        and job.get('parameters') == request['parameters'] and job.get('state') == 'completed'
        and type(job.get('exit_code')) is int and job['exit_code'] == 0,
        'recovery execution requires a matching completed central job with exit zero')
    logs = {kind + '_log': snapshot.add(old._log_evidence(job[kind + '_log'], kind))
            for kind in ('stdout', 'stderr')}
    return job, dict(job=binding, logs=logs)


def _producer(request, job, receipt, attempts, snapshot):
    key = (str(old.CENTRAL_RETRIES / (job['job_id'] + '.json')), receipt['attempt'])
    cached = snapshot.producers.get(key)
    if cached is not None:
        _require(cached['execution_receipt'] == receipt, 'units disagree on producing-attempt receipt')
        return cached

    def audit(job_id, attempt):
        found = attempts.get((job_id, attempt), {})
        _require(set(found) == {'attempt_started', 'running_transition'}, 'missing central producing-attempt audit')
        return found

    producer = evidence.producer_evidence(request, job, receipt,
        logs_root=old.CENTRAL_LOGS, retries_root=old.CENTRAL_RETRIES,
        validate_execution=old.validate_execution, binding=_binding, read_json=_read_json,
        log_evidence=old._log_evidence, audit=audit)
    snapshot.walk(producer)
    return producer


def _read_terminal(store, seed, split, snapshot):
    """The exact status checks, with source/manifest checks batched at run boundaries."""
    from .recovery import COMPLETE_FILES, NONCOMPLETE_FILES
    directory = store.cell_dir(seed, split)
    _require(directory.is_dir(), 'recovery scientific output is missing')
    names = {entry.name for entry in directory.iterdir()}
    completed = names == set(COMPLETE_FILES)
    _require(completed or names == set(NONCOMPLETE_FILES), 'partial or mixed recovery artifacts')
    snapshot.layouts[str(directory)] = names
    files = COMPLETE_FILES if completed else NONCOMPLETE_FILES
    bindings = [snapshot.add(_binding(directory / name)) for name in files]
    hashes = {Path(item['path']).name: item['sha256'] for item in bindings}
    marker = _read_json(directory / files[-1])
    required = {'identity', 'artifacts'} if completed else {'identity', 'outcome', 'artifacts'}
    _require(set(marker) == required and canonical(marker['identity']) == canonical(store.identity(seed, split))
        and marker['artifacts'] == {name: hashes[name] for name in files[:-1]},
        'recovery terminal marker identity or artifact hashes changed')
    if not completed:
        _require(marker['outcome'] == 'validation_patience_limit_200', 'unsupported scientific noncompletion')
    started = store._started(seed, split)
    result, arrays = (store._read(seed, split) if completed else store._read_noncompletion(seed, split))
    del arrays
    status = 'validated' if completed else 'noncompleted'
    terminal = dict(seed=seed, split=split, outcome='complete' if completed else 'budget_exhausted', artifacts=hashes)
    return status, result, started['execution'], bindings, terminal


def _thin(result):
    return dict(fingerprints=result['fingerprints'], site_sizes=result['site_sizes'],
        label_counts=result['label_counts'], evaluation={key: result['evaluation'][key] for key in
            ('raw_x_sha256', 'permutation_sha256', 'shard_indices_sha256', 'shard_sizes')},
        fit={key: result['fit'][key] for key in
             ('fit_fingerprint', 'initial_model_fingerprint', 'local_model_fingerprints', 'validation')
             if key in result['fit']})


def _terminal_thin(result, status):
    row = (_thin(result) if status == 'validated' else
           {key: result[key] for key in ('fingerprints', 'site_sizes', 'label_counts')})
    row.update(outcome=status, numeric_fit_sha256=digest(_numeric(result['fit'])))
    return row


def validate_terminal_training_pairs(records):
    """Identical training arms must agree even when neither has performance data."""
    peers = dict(control='local_permutation', local_permutation='control',
        site_gaussian_shift='site_gaussian_local_permutation',
        site_gaussian_local_permutation='site_gaussian_shift')
    checked = 0
    for (setting, cell, variant, seed, split), result in records.items():
        peer = (setting, cell, peers[variant], seed, split)
        _require(peer in records, 'terminal training pair lacks its planned peer')
        other = records[peer]
        _require(all(result[key] == other[key] for key in
            ('outcome', 'fingerprints', 'numeric_fit_sha256', 'site_sizes', 'label_counts')),
            'paired identical training changed completion state, model, data, or numeric path')
        checked += 1
    return dict(passed=True, terminal_training_pairs=checked // 2,
                includes_scientific_noncompletion=True)


def validate_available_pairs(records):
    """Compare complete peers only and disclose every missing planned pair."""
    checked, missing = Counter(), Counter()
    for setting, recipe in protocol.SETTINGS.items():
        cells = recipe['cells']
        for seed in range(300):
            for split in protocol.SPLITS:
                group = {(cell, variant): records[setting, cell, variant, seed, split]
                    for cell in cells for variant in protocol.VARIANTS
                    if (setting, cell, variant, seed, split) in records}
                if len(group) >= 2:
                    _require(len({row['fingerprints']['test'] for row in group.values()}) == 1,
                             'complete peers have different common test sets')
                    checked['common_test_groups'] += 1
                else:
                    missing['common_test_groups'] += 1
                for variant in protocol.VARIANTS:
                    peers = [group[cell, variant] for cell in cells if (cell, variant) in group]
                    for peer in peers[1:]:
                        first = peers[0]
                        _require(all(first['fingerprints'][key] == peer['fingerprints'][key]
                                     for key in ('training', 'test'))
                            and all(first[key] == peer[key] for key in ('site_sizes', 'label_counts', 'evaluation')),
                            'complete method peers changed data or evaluation plans')
                        checked['method_data_pairs'] += 1
                    missing['method_data_pairs'] += len(cells) - 1 - max(0, len(peers) - 1)
                for cell in cells:
                    fisher = protocol.CELLS[cell]['family'] == 'legacy_fedfisher'
                    fit_keys = ('initial_model_fingerprint', 'local_model_fingerprints', 'validation') if fisher else ('fit_fingerprint',)
                    target = checked if all((cell, variant) in group for variant in protocol.VARIANTS) else missing
                    target['complete_four_variant_blocks'] += 1
                    for left, right in (('control', 'local_permutation'),
                                        ('site_gaussian_shift', 'site_gaussian_local_permutation')):
                        if (cell, left) in group and (cell, right) in group:
                            a, b = group[cell, left], group[cell, right]
                            _require(a['fingerprints'] == b['fingerprints']
                                and all(a['fit'][key] == b['fit'][key] for key in fit_keys)
                                and a['evaluation']['raw_x_sha256'] == b['evaluation']['raw_x_sha256'],
                                'complete intervention peers changed fitting or data')
                            checked['intervention_only_pairs'] += 1
                        else:
                            missing['intervention_only_pairs'] += 1
                    for left, right in (('control', 'site_gaussian_shift'),
                                        ('local_permutation', 'site_gaussian_local_permutation')):
                        if (cell, left) in group and (cell, right) in group:
                            a, b = group[cell, left], group[cell, right]
                            _require(all(a[key] == b[key] for key in ('evaluation', 'site_sizes', 'label_counts')),
                                     'complete Gaussian peers changed evaluation plan or site capacities')
                            checked['factorial_scope_pairs'] += 1
                        else:
                            missing['factorial_scope_pairs'] += 1
                    if fisher and protocol.CELLS[cell]['algorithm'] == 'fedfisher_diag':
                        peer = cell.replace('fedfisher_diag_', 'fedfisher_kfac_')
                        for variant in protocol.VARIANTS:
                            if (cell, variant) in group and (peer, variant) in group:
                                a, b = group[cell, variant], group[peer, variant]
                                _require(all(a['fit'][key] == b['fit'][key] for key in fit_keys),
                                         'complete Diag/KFAC peers changed local fitting')
                                checked['diag_kfac_local_fit_pairs'] += 1
                            else:
                                missing['diag_kfac_local_fit_pairs'] += 1
    return dict(available_pairs_passed=True, checked=dict(checked), missing=dict(missing),
        entire_planned_pair_coverage_complete=not any(missing.values()),
        performance_pair_scope='only scientifically complete peers; missing pairs remain disclosed')


def _collect(inventory_path, cohort):
    from .recovery import RecoveryStore, validate_shard_summary, shard_summary_path
    provenance = current_provenance()
    closeout = verify_closeout()
    snapshot = _EvidenceSnapshot()
    inventory, inventory_binding = _inventory(inventory_path, provenance, cohort, closeout, snapshot)
    prerequisites = dict(closeout=closeout['binding'], pilot=None)
    if cohort == 'formal':
        path = default_acceptance_path(provenance)
        prerequisite = verify_pilot_acceptance(path)
        binding = snapshot.add(_binding(path))
        _require(prerequisite['closeout'] == closeout['binding']
            and inventory.get('pilot_acceptance') == dict(binding, evidence=prerequisite),
            'formal inventory lacks the current-source fully verified recovery pilot')
        prerequisites['pilot'] = binding
    attempts = _attempt_index({row['job_id'] for row in inventory['requests']})
    by_run = defaultdict(list)
    for row in inventory['requests']:
        by_run[row['run_id']].append(row)
    units, shards, records, parity_checked = [], [], {}, set()
    references = {}
    if cohort == 'pilot':
        references = {_unit_key(row): row for row in closeout['old_pilot']['units']}
        references.update({_unit_key(row): row for row in closeout['proof']['known_success_diagnostic_controls']})
    for run_id, requests in by_run.items():
        config = requests[0]['configuration']
        store = RecoveryStore(config, provenance, prerequisites)
        store._manifest()
        manifest = snapshot.add(_binding(store.directory / 'run_manifest.env'))
        expected_names = {'run_manifest.env', 'shard_summaries'} | {
            'seed_%03d_%s' % (seed, split) for row in requests for seed in row['seeds'] for split in protocol.SPLITS}
        _require({p.name for p in store.directory.iterdir()} == expected_names,
                 'recovery run contains unplanned scientific outputs')
        snapshot.layouts[str(store.directory)] = expected_names
        summary_root = store.directory / 'shard_summaries'
        expected_summaries = {row['job_id'] + '.json' for row in requests}
        _require(summary_root.is_dir() and {p.name for p in summary_root.iterdir()} == expected_summaries,
                 'recovery run summary coverage is incomplete or contains foreign shards')
        snapshot.layouts[str(summary_root)] = expected_summaries
        for request in requests:
            job, central = _completed_job(request, snapshot)
            terminal_units = []
            for seed in request['seeds']:
                for split in protocol.SPLITS:
                    status, result, receipt, artifacts, terminal = _read_terminal(store, seed, split, snapshot)
                    producer = _producer(request, job, receipt, attempts, snapshot)
                    terminal_units.append(terminal)
                    row = dict(setting=config['setting'], cell=config['cell'], variant=config['variant'],
                        seed=seed, split=split, outcome=status, run_id=run_id, job_id=job['job_id'])
                    key = _unit_key(row)
                    _require(key not in records, 'duplicate current recovery scientific unit')
                    if cohort == 'pilot':
                        row.update(artifacts=artifacts, manifest=manifest, producer_evidence=producer,
                            timing=result['timing'], memory=result.get('memory'))
                        if key in references:
                            _require(status == 'validated', 'historical pilot/control success became noncompleted')
                            validate_numeric_parity(result, _unit_result(references[key]))
                            parity_checked.add(key)
                        paired = (_thin(result) if status == 'validated' else
                            {name: result[name] for name in ('fingerprints', 'site_sizes', 'label_counts')})
                        if status == 'noncompleted':
                            paired['fit'] = {'fit_fingerprint': result['fit']['fit_fingerprint']}
                        records[key] = dict(outcome=status, result=paired,
                            numeric_fit_sha256=digest(_numeric(result['fit'])))
                    else:
                        records[key] = _terminal_thin(result, status)
                    units.append(row)
            summary = validate_shard_summary(store, job, verified_units=terminal_units)
            summary_binding = snapshot.add(_binding(shard_summary_path(store, job)))
            writer = _producer(request, job, summary['producer_execution'], attempts, snapshot)
            shards.append(dict(job_id=job['job_id'], run_id=run_id, manifest=manifest, summary=summary_binding,
                central=central, summary_producer=writer, counts=summary['counts']))
        store._manifest()
    if cohort == 'pilot':
        checks = _pilot_terminal_checks(records, parity_checked)
        count = Counter(row['outcome'] for row in units)
        body = dict(scope='model_signal_recovery_pilot_acceptance', passed=True, formal_eligible=True,
            pilot_inventory=inventory_binding, validated_request_count=290, validated_unit_count=588,
            complete_units=count['validated'], noncompleted_units=count['noncompleted'],
            whole_pilot_scientific_success=count['noncompleted'] == 0, checks=checks, units=units)
    else:
        coverage = validate_composite_coverage(closeout['retained_units'], units)
        for row in closeout['retained_units']:
            key = _unit_key(row)
            _require(key not in records, 'new/retained scientific records overlap')
            records[key] = _terminal_thin(_unit_result(row), 'validated')
        terminal_pairs = validate_terminal_training_pairs(records)
        complete_records = {key: value for key, value in records.items() if value['outcome'] == 'validated'}
        body = dict(scope='model_signal_recovery_formal_acceptance', passed=True,
            formal_inventory=inventory_binding, pilot_acceptance=prerequisites['pilot'],
            validated_request_count=4479, validated_unit_count=167962,
            coverage=coverage, terminal_training_pairs=terminal_pairs,
            paired_checks=validate_available_pairs(complete_records),
            historical_provenance=closeout['proof']['provenance'], retained_source=closeout['proof']['formal_acceptance'],
            excluded_old_valid_units=44)
    snapshot.recheck()
    closeout['evidence_snapshot'].recheck()
    _require(current_provenance() == provenance, 'source changed during recovery acceptance')
    return dict(schema_version=1, protocol=protocol.PROTOCOL, provenance=provenance,
        closeout=closeout['binding'], inference_or_fitting_performed=False, shards=shards,
        admission_note='execution/scientific evidence only; operator reviews measured resource profile', **body)


def collect_pilot_acceptance(inventory_path):
    return _collect(inventory_path, 'pilot')


def collect_formal_acceptance(inventory_path):
    return _collect(inventory_path, 'formal')


def verify_pilot_acceptance(path):
    provenance = current_provenance()
    path = old._ordinary_path(path)
    _require(path == default_acceptance_path(provenance), 'pilot proof must use its canonical source-addressed path')
    binding, proof = _binding(path), _read_json(path)
    _require(proof.get('schema_version') == 1 and proof.get('protocol') == protocol.PROTOCOL
        and proof.get('scope') == 'model_signal_recovery_pilot_acceptance'
        and proof.get('passed') is True and proof.get('formal_eligible') is True
        and proof.get('provenance') == provenance and proof.get('closeout') == protocol.SOURCE_CLOSEOUT,
        'invalid or stale recovery pilot proof')
    fresh = collect_pilot_acceptance(proof['pilot_inventory']['path'])
    _require(canonical(fresh) == canonical(proof) and _binding(path) == binding,
             'recovery pilot proof differs from freshly verified evidence')
    return proof


def check_current_pilot_gate():
    return verify_pilot_acceptance(default_acceptance_path())
