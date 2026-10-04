"""Read the sealed, incomplete V2 epoch without claiming current-source validity.

Admission checks the immutable release and the bounded parity references.  Final
acceptance additionally streams every retained unit and rechecks its original
bytes and producing attempt.  No archived Python is imported or executed.
"""
from collections import Counter
import hashlib
import io
from pathlib import Path
import subprocess

from ..infrastructure.artifacts import canonical, digest
from . import recovery_acceptance as previous
from . import recovery_protocol as v2

PROJECT_ROOT = previous.PROJECT_ROOT
SOURCE_CLOSEOUT = dict(path='/data/del6500/FedFisher/archives/model-signal-recovery-v2-4ecb6d5-20261002/closeout.json',
    sha256='1cc83b232fdfc8fc68c772215d5fa4827a6c37ec40aaf523dcdfb968829a6a46')
REFACTOR_COMMIT = '7c6c6714f51080673c56a95ec78f3065b1c79ef4'
ORIGINAL_COMMIT = '4ecb6d585b91d8119674fcdc2de052282524c9b9'
_require, _binding, _read_json = previous._require, previous._binding, previous._read_json
_key = previous._unit_key


def _git_blobs(commit, names):
    response = subprocess.run(['git', 'cat-file', '--batch'], cwd=PROJECT_ROOT,
        input=''.join(commit + ':' + name + '\n' for name in names).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    stream = io.BytesIO(response)
    result = {}
    for name in names:
        header = stream.readline().split()
        _require(len(header) == 3 and header[1] == b'blob', 'sealed Git source blob is missing')
        result[name] = stream.read(int(header[2]))
        _require(stream.read(1) == b'\n', 'malformed Git source stream')
    _require(not stream.read(), 'unexpected Git source bytes')
    return result


def verify_science_sources():
    """Protect the reviewed refactor's actual calculation/import path bytewise."""
    names = subprocess.run(['git', 'ls-tree', '-r', '--name-only', REFACTOR_COMMIT],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode().splitlines()
    prefixes = ('algs/', 'utils/', 'effect_modifier_experiment/components/',
        'effect_modifier_experiment/causal_comparison/', 'effect_modifier_experiment/fedfisher_timing/')
    fixed = {'data.py', 'models.py', 'train_model.py', 'run_one_shot_algs.py',
        'effect_modifier_experiment/dslassocov_benchmark.py',
        'effect_modifier_experiment/dslassocov_run_contract.py',
        'effect_modifier_experiment/run_dslassocov_effect_modifier.py',
        'effect_modifier_experiment/one_shot_lasso_benchmark.py',
        'effect_modifier_experiment/model_signal_ablation/protocol.py',
        'effect_modifier_experiment/model_signal_ablation/recovery_protocol.py',
        'effect_modifier_experiment/model_signal_ablation/artifacts.py',
        'effect_modifier_experiment/model_signal_ablation/runner.py'}
    selected = sorted(name for name in names if name in fixed or
                      (name.endswith('.py') and name.startswith(prefixes)))
    _require(fixed <= set(selected), 'reviewed science source inventory is incomplete')
    blobs = _git_blobs(REFACTOR_COMMIT, selected)
    for name, payload in blobs.items():
        path = previous.old._ordinary_path(PROJECT_ROOT / name)
        _require(path.read_bytes() == payload, 'completion changed protected scientific source: ' + name)
    return dict(commit=REFACTOR_COMMIT, file_count=len(selected),
        files_sha256=digest({name: hashlib.sha256(value).hexdigest() for name, value in blobs.items()}))


class SealedReader:
    """Resolve copied metadata; models always remain in their original namespace."""
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        self.audit_index = None
        self.producers = set()
        self.audit_ranges = {}
        self.layouts = {}

    def check(self, binding, *, original=False):
        path = Path(binding['path'])
        _require(path.is_absolute() and '..' not in path.parts, 'invalid sealed evidence path')
        if self.root not in path.parents and not original:
            copied = self.root / 'evidence/originals' / str(path).lstrip('/')
            if copied.exists():
                path = copied
        path = previous.old._ordinary_path(path)
        target = dict(path=str(path), sha256=binding['sha256'])
        _require(str(path) not in self.files or self.files[str(path)] == target,
                 'conflicting sealed evidence hashes')
        if str(path) not in self.files:
            _require(_binding(path) == target, 'sealed historical evidence bytes changed: ' + str(path))
            self.files[str(path)] = target
        if 'bytes' in binding:
            _require(path.stat().st_size == binding['bytes'], 'sealed evidence size changed')
        if 'archive_relative_path' in binding:
            _require(path == self.root / binding['archive_relative_path'], 'sealed archive path mismatch')
        return path

    def read(self, binding, *, original=False):
        return _read_json(self.check(binding, original=original))

    def walk(self, value):
        if isinstance(value, dict):
            if {'retry', 'logs', 'audit', 'producer_attempt'} <= set(value):
                self.producer(value)
                return
            if {'path', 'sha256'} <= set(value) and 'offset' not in value:
                self.check(value)
            for part in value.values():
                self.walk(part)
        elif isinstance(value, list):
            for part in value:
                self.walk(part)

    def producer(self, producer):
        key = digest(producer)
        if key in self.producers:
            return
        _require(self.audit_index is not None, 'archive audit index not initialized')
        self.check(producer['retry'])
        self.walk(producer['logs'])
        receipt = producer['execution_receipt']
        _require(receipt['attempt'] == producer['producer_attempt'], 'historical producer attempt mismatch')
        _require(set(producer['audit']) == {'attempt_started', 'running_transition'}, 'incomplete producer audit')
        for event, binding in producer['audit'].items():
            item = self.audit_index.get((binding['path'], binding['offset'], binding['length']))
            _require(item is not None and item['sha256'] == binding['sha256'], 'producer audit is not in sealed archive')
            path = previous.old._ordinary_path(self.root / item['archive_relative_path'])
            with path.open('rb') as stream:
                stream.seek(item['archive_offset'])
                payload = stream.read(item['length'])
            _require(hashlib.sha256(payload).hexdigest() == binding['sha256']
                and previous.old._json_bytes(payload) == binding['record']
                and binding['record']['job_id'] == receipt['job_id'], 'sealed producer audit bytes changed')
            if event == 'attempt_started':
                _require(binding['record']['attempt'] == receipt['attempt'], 'sealed audit attempt mismatch')
            self.audit_ranges[(str(path), item['archive_offset'], item['length'])] = binding['sha256']
        self.producers.add(key)

    def recheck(self):
        for binding in self.files.values():
            _require(_binding(binding['path']) == binding, 'sealed evidence changed during verification')
        for (name, offset, length), checksum in self.audit_ranges.items():
            path = previous.old._ordinary_path(name)
            with path.open('rb') as stream:
                stream.seek(offset)
                payload = stream.read(length)
            _require(hashlib.sha256(payload).hexdigest() == checksum,
                     'sealed producer audit changed during verification')
        for name, names in self.layouts.items():
            path = Path(name)
            _require(path.resolve(strict=True) == path and not path.is_symlink()
                and {entry.name for entry in path.iterdir()} == names,
                'sealed artifact layout changed during verification')


def _validate_partition(inventory, shards, failed, pending):
    requests = {row['job_id']: row for row in inventory['requests']}
    _require(len(requests) == len(inventory['requests']) == 4479
        and inventory['task'] == v2.FORMAL_TASK, 'wrong sealed formal request cohort')
    by_id = {row['job_id']: row for row in shards}
    _require(len(by_id) == len(shards) == 4608, 'sealed formal shards duplicated or missing')
    historical = {job_id for job_id, row in by_id.items() if row['historical_source']}
    retained = {job_id for job_id, row in by_id.items() if not row['historical_source']
                and row['final_central_state'] == 'completed'}
    replaced = set(requests) - retained
    _require(len(historical) == 129 and len(retained) == 4455 and len(replaced) == 24
        and set(by_id) == set(requests) | historical
        and all(by_id[j]['final_central_state'] == 'failed' for j in replaced), 'sealed retained/replaced partition changed')
    _require(failed.get('classification') == 'audit_only_failed_shard'
        and failed.get('whole_shard_pending') is True and len(failed['shards']) == 24
        and {row['job_id'] for row in failed['shards']} == replaced, 'whole failed-shard scope changed')
    planned, replacement = {}, set()
    for job_id, row in requests.items():
        p = v2.validate_parameters(v2.FORMAL_TASK, row['parameters'])
        config = v2.run_config(v2.FORMAL_TASK, p)
        seeds = list(v2.seed_sequence(v2.FORMAL_TASK, p))
        _require(canonical(row['configuration']) == canonical(config)
            and row['configuration_sha256'] == digest(config) and row['seeds'] == seeds
            and row['splits'] == list(v2.SPLITS), 'sealed scientific configuration changed')
        for seed in seeds:
            for split in v2.SPLITS:
                key = (p['setting'], p['cell'], p['variant'], seed, split)
                _require(key not in planned, 'overlapping sealed shards')
                planned[key] = job_id
                if job_id in replaced:
                    replacement.add(key)
    failed_by_id = {row['job_id']: row for row in failed['shards']}
    for job_id in replaced:
        row, saved = requests[job_id], failed_by_id[job_id]
        _require(all(saved[key] == row[key] for key in
            ('parameters', 'configuration', 'configuration_sha256', 'run_id', 'seeds', 'splits'))
            and saved['request_sha256'] == row['sha256'] and saved['central_state'] == 'failed',
            'failed-shard recipe differs from sealed request')
    rows = pending['failed_shard_units']
    _require(len(replacement) == len(rows) == 896 and {_key(row) for row in rows} == replacement
        and all(row['job_id'] == planned[_key(row)] and row['performance_accepted'] is False for row in rows)
        and Counter(row['outcome'] for row in rows) == {'validated': 287, 'missing': 597, 'partial': 12},
        'failed-shard units were promoted, removed, or duplicated')
    _require(len(planned) == 167962
        and sum(sum(by_id[j]['counts'].values()) for j in historical) == 4838,
        'sealed historical unit denominators changed')
    return retained, historical, replaced, planned, replacement


def verify_closeout(path=None):
    """Bounded admission check. Whole-epoch artifact replay belongs to final acceptance."""
    path = SOURCE_CLOSEOUT['path'] if path is None else str(path)
    _require(path == SOURCE_CLOSEOUT['path'] and _binding(path) == SOURCE_CLOSEOUT,
             'completion requires the fixed immutable archive closeout')
    proof = _read_json(path)
    _require(proof.get('schema_version') == 1
        and proof.get('scope') == 'model_signal_recovery_v2_incomplete_epoch_archival_source_release'
        and proof.get('source_release_passed') is True and proof.get('existing_results_archived') is True
        and proof.get('allow_refactor') is True and proof.get('whole_cohort_accepted') is False
        and proof.get('whole_experiment_completed') is False and proof.get('new_execution_authorized') is False
        and proof.get('source_commit') == ORIGINAL_COMMIT, 'invalid incomplete-epoch release scope')
    reader = SealedReader(Path(path).parent)
    reader.check(SOURCE_CLOSEOUT)
    summary = reader.read(proof['full_verification'])
    _require(summary.get('verification_passed') is True
        and summary.get('scope') == 'model_signal_recovery_v2_source_closeout_verification'
        and summary['current_source'] == proof['source_provenance']
        and summary['accepted_formal_terminal_units'] == 171904
        and summary['performance_accepted_formal_complete_models'] == 171898
        and summary['pending_failed_shard_units'] == 896
        and summary['formal_caps_semantics_revalidated'] == 6
        and summary['failed_shard_completed_models_audit_only_semantics_revalidated'] == 287
        and summary['whole_cohort_accepted'] is False, 'archive full verification has wrong scientific scope')
    source = reader.read(proof['source_file_index'])
    _require(source['git_commit'] == ORIGINAL_COMMIT and len(source['files']) == 463,
             'sealed source inventory changed')
    names = [row['path'] for row in source['files']]
    _require(len(set(names)) == 463 and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),
             'invalid sealed source names')
    blobs = _git_blobs(ORIGINAL_COMMIT, names)
    _require(all(hashlib.sha256(blobs[row['path']]).hexdigest() == row['sha256'] for row in source['files']),
             'sealed source does not match actual original Git blobs')
    science = verify_science_sources()
    inventory = reader.read(summary['inventory'])
    _require(inventory.get('provenance') == proof['source_provenance'], 'sealed inventory source changed')
    shards = reader.read(proof['verification_evidence']['shards.json'])
    failed = reader.read(proof['pending']['exact_failed_shards'])
    pending = reader.read(proof['pending']['evidence'])
    retained, historical, replaced, planned, replacement = _validate_partition(inventory, shards, failed, pending)
    from . import completion_protocol as protocol
    _require({previous._request_key(row['parameters']) for row in failed['shards']}
        == {previous._request_key(row) for row in protocol.FORMAL_PARAMETERS}, 'new formal scope differs from archive')
    plan = reader.read(proof['exact_requests_producers_and_audit_plan'])
    reader.audit_index = {(row['path'], row['offset'], row['length']): row for row in plan['audit_lines']}
    _require(len(reader.audit_index) == len(plan['audit_lines']), 'duplicate sealed audit positions')
    pilot = reader.read(summary['pilot_proof'])
    _require(pilot.get('passed') is True and pilot.get('validated_unit_count') == 588
        and pilot.get('provenance') == proof['source_provenance'] and pilot.get('protocol') == v2.PROTOCOL,
        'old pilot is not the sealed successful V2 reference')
    expected = {(p['setting'], p['cell'], p['variant'], 0, split)
                for p in protocol.PILOT_PARAMETERS for split in v2.SPLITS}
    refs = {_key(row): row for row in pilot['units'] if _key(row) in expected}
    _require(len(expected) == len(refs) == 16, 'bounded pilot references are missing or duplicated')
    for row in refs.values():
        _require(row['outcome'] == 'validated', 'bounded pilot reference was not a success')
        reader.walk(row)
    reader.recheck()
    return dict(binding=dict(SOURCE_CLOSEOUT), proof=proof, old_inventory=inventory,
        retained_request_ids=sorted(retained), retained_units=171904, retained_v1_request_ids=sorted(historical),
        replacement_request_ids=sorted(replaced), replacement_keys=replacement, planned_v2=planned,
        shards=shards, pilot_references=refs, pilot_reference_binding=summary['pilot_proof'],
        source_guard=science, reader=reader, summary=summary)


def reference_result(closeout, key):
    row = closeout['pilot_references'][key]
    matches = [item for item in row['artifacts'] if Path(item['path']).name == 'result.json']
    _require(len(matches) == 1, 'pilot reference result missing')
    return closeout['reader'].read(matches[0])


def _jsonl(reader, binding):
    path = reader.check(binding)
    with path.open('rb') as stream:
        for line in stream:
            _require(line.endswith(b'\n'), 'truncated sealed ledger')
            yield previous.old._json_bytes(line)


def _retained_record(reader, row, producer):
    """Restore only hash-bound saved evidence, never a model for prediction."""
    outcome = row['outcome']
    _require(row['final_central_state'] == 'completed' and outcome in ('validated', 'noncompleted')
        and row['performance_accepted'] == (outcome == 'validated'), 'invalid retained scientific classification')
    reader.check(row['manifest'], original=True)
    artifacts = {Path(item['path']).name: item for item in row['artifacts']}
    from .recovery import COMPLETE_FILES, NONCOMPLETE_FILES
    files = COMPLETE_FILES if outcome == 'validated' else NONCOMPLETE_FILES
    _require(len(row['artifacts']) == len(artifacts) == len(files) and set(artifacts) == set(files),
             'historical unit file set changed')
    directory = previous.old._ordinary_path(artifacts[files[0]]['path']).parent
    _require({entry.name for entry in directory.iterdir()} == set(files), 'historical artifact layout changed')
    reader.layouts[str(directory)] = set(files)
    for item in artifacts.values():
        _require(Path(item['path']).parent == directory, 'historical artifacts cross units')
        reader.check(item, original=True)
    marker = reader.read(artifacts[files[-1]], original=True)
    result = reader.read(artifacts['result.json' if outcome == 'validated' else 'noncompletion.json'], original=True)
    started = reader.read(artifacts['started.json'], original=True)
    _require(marker['identity'] == result['identity'] == started['identity'] == row['identity']
        and marker['artifacts'] == {name: artifacts[name]['sha256'] for name in files[:-1]},
        'sealed terminal identity/hash mismatch')
    _require(result['status'] == ('complete' if outcome == 'validated' else 'scientific_noncompletion'),
             'historical terminal status changed')
    _require(producer is not None and producer['execution_receipt'] == started['execution']
        and producer['execution_receipt']['job_id'] == row['job_id']
        and producer['central_completed'] is True and producer['final_state'] == 'completed',
        'retained unit producer mismatch')
    reader.producer(producer)
    return previous._terminal_thin(result, outcome)


def load_retained_evidence(closeout):
    """Final-only complete hash/producer replay, returning compact pairing records."""
    reader, proof = closeout['reader'], closeout['proof']
    producers = {}
    for name in ('producers.jsonl', 'historical-producers.jsonl'):
        for row in _jsonl(reader, proof['verification_evidence'][name]):
            key = row['execution_receipt']['job_id'] + ':attempt-' + str(row['producer_attempt'])
            if name == 'producers.jsonl':
                _require(row['producer_key'] == key and key not in producers, 'duplicate historical producing attempt')
                producers[key] = row
            else:
                _require(key in producers and all(producers[key][field] == row[field] for field in
                    ('execution_receipt', 'audit', 'logs', 'retry', 'producer_attempt')),
                    'historical producing-attempt chains disagree')
    retained_ids = set(closeout['retained_request_ids']) | set(closeout['retained_v1_request_ids'])
    shards = {row['job_id']: row for row in closeout['shards']}
    layouts = reader.read(proof['verification_evidence']['layouts.json'])
    retained_runs = {str(Path('/data/del6500/FedFisher/effect_modifier_experiment/outputs') / shards[job_id]['run_id'])
                     for job_id in retained_ids}
    for name, names in layouts.items():
        if name in retained_runs or (Path(name).name == 'shard_summaries' and str(Path(name).parent) in retained_runs):
            path = Path(name)
            _require(path.resolve(strict=True) == path and not path.is_symlink()
                and {entry.name for entry in path.iterdir()} == set(names), 'retained run layout changed')
            reader.layouts[name] = set(names)
    _require(retained_runs <= set(reader.layouts), 'sealed retained run layouts are missing')
    del layouts
    for job_id in retained_ids:
        shard = shards[job_id]
        reader.walk(shard['central'])
        job = reader.read(shard['central']['job'])
        _require(job['job_id'] == job_id and job['state'] == 'completed' and job['exit_code'] == 0,
                 'retained historical shard has no completed producer')
        if shard['summary'] is not None:
            reader.check(shard['summary'], original=True)
    records, counts, owners = {}, Counter(), Counter()
    for row in _jsonl(reader, proof['verification_evidence']['unit-ledger.jsonl']):
        if row['scope'] != 'formal' or row['job_id'] not in retained_ids:
            continue
        key, outcome = _key(row), row['outcome']
        _require(key not in records and key not in closeout['replacement_keys']
            and row['final_central_state'] == 'completed' and outcome in ('validated', 'noncompleted')
            and row['performance_accepted'] == (outcome == 'validated'), 'invalid historical retained unit')
        if row['job_id'] in closeout['retained_request_ids']:
            _require(closeout['planned_v2'].get(key) == row['job_id'], 'retained V2 ownership changed')
        producer = producers.get(row['producer_key'])
        records[key] = _retained_record(reader, row, producer)
        owners[row['job_id']] += 1
        counts[outcome] += 1
    _require(len(records) == 171904 and counts == {'validated': 171898, 'noncompleted': 6}
        and set(owners) == retained_ids, 'retained terminal coverage changed')
    for job_id, count in owners.items():
        _require(sum(shards[job_id]['counts'].values()) == count, 'retained shard unit coverage mismatch')
    return records, dict(retained_units=len(records), complete_units=counts['validated'],
        noncompleted_units=counts['noncompleted'], retained_v2_requests=4455, retained_v1_requests=129,
        original_artifact_hashes_rechecked=True, archived_producing_attempts_rechecked=True,
        semantics='Inherited only from the immutable full old-source saved-evidence verification; no current-source relabeling',
        source_closeout=closeout['binding'], unit_ledger=proof['verification_evidence']['unit-ledger.jsonl'])
