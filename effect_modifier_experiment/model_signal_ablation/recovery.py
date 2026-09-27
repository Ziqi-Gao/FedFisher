"""Immutable recovery units and foreground execution of unchanged recipes.

A verified patience-budget exhaustion is an explicit scientific noncompletion.
It has neither a successful completion marker nor evaluation/performance values.
Only this new task epoch may continue after that narrowly validated outcome.
"""

from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import resource

import numpy as np

from ..causal_comparison.noncompletion import (
    ScientificNoncompletion, validate_noncompletion_result,
)
from ..fedfisher_timing import artifacts as lifecycle
from ..infrastructure.artifacts import canonical, create_bytes, read_json, safe_path
from ..infrastructure.evidence import ordinary_path
from . import recovery_protocol as protocol
from .artifacts import ArtifactStore, MEMORY_SCOPE, current_provenance, validate_execution
from .runner import compute_cell


COMPLETE_FILES = ('started.json', 'model.npz', 'result.json', 'done.json')
NONCOMPLETE_FILES = ('started.json', 'diagnostic_model.npz', 'noncompletion.json',
                     'noncompletion_marker.json')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_prerequisites(prerequisites, task):
    _require(isinstance(prerequisites, dict) and set(prerequisites) == {'closeout', 'pilot'},
             'recovery requires explicit closeout and pilot prerequisites')
    _require(isinstance(prerequisites['closeout'], dict), 'recovery closeout binding is mandatory')
    _require((prerequisites['pilot'] is None) == (task == protocol.PILOT_TASK),
             'recovery pilot prerequisite differs from task stage')
    for name, binding in prerequisites.items():
        if binding is None:
            continue
        _require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}
                 and isinstance(binding['path'], str) and lifecycle._hash(binding['sha256']),
                 'invalid recovery prerequisite binding')
        path = ordinary_path(binding['path'], exists=False)
        _require(any(root in path.parents for root in (
            Path('/data/del6500/FedFisher/effect_modifier_experiment/reports'),
            Path('/scr/del6500/FedFisher'))), 'recovery prerequisite outside project reports')


class RecoveryStore(ArtifactStore):
    """The v1 successful-result validators, plus a distinct capped outcome."""

    def __init__(self, config, provenance, prerequisites, directory=None):
        validate_prerequisites(prerequisites, config['task'])
        self.prerequisites = deepcopy(prerequisites)
        super().__init__(config, provenance, directory)

    def _validate_config(self):
        parameters = {key: self.config[key] for key in ('setting', 'cell', 'variant')}
        parameters.update(shard_index=0,
                          shard_count=1 if self.config['task'] == protocol.PILOT_TASK else 16)
        if self.config['task'] == protocol.PILOT_TASK:
            parameters['case'] = self.config['case']
        _require(canonical(self.config) == canonical(protocol.run_config(self.config['task'], parameters)),
                 'recovery configuration is not a reviewed fixed recipe')

    def identity(self, seed, split):
        identity = super().identity(seed, split)
        if self.config.get('case') == 'cap_diagnostic':
            _require(seed in (9, 25, 41), 'diagnostic unit outside fixed seed sequence')
        return identity

    def _started(self, seed, split):
        value = read_json(self.cell_dir(seed, split) / 'started.json')
        _require(set(value) == {'identity', 'execution', 'prerequisites'}
                 and canonical(value['identity']) == canonical(self.identity(seed, split))
                 and canonical(value['prerequisites']) == canonical(self.prerequisites),
                 'recovery started identity or prerequisites mismatch')
        self._validate_execution(value['execution'])
        return value

    def start(self, seed, split, execution):
        self._manifest()
        self._validate_execution(execution)
        if self.status(seed, split) != 'missing':
            raise FileExistsError('recovery scientific unit already has a terminal outcome')
        path = self.cell_dir(seed, split)
        payload = canonical(dict(identity=self.identity(seed, split), execution=execution,
                                 prerequisites=self.prerequisites)).encode()
        self._current()
        path.mkdir()
        safe_path(path)
        create_bytes(path / 'started.json', payload)
        self._current()

    def _read_noncompletion(self, seed, split):
        path = self.cell_dir(seed, split)
        result = read_json(path / 'noncompletion.json')
        with np.load(safe_path(path / 'diagnostic_model.npz'), allow_pickle=False) as archive:
            _require(len(archive.files) == len(set(archive.files)), 'duplicate diagnostic model arrays')
            arrays = {name: archive[name] for name in archive.files}
        validate_noncompletion_result(result, arrays, self.identity(seed, split), self.config)
        return result, arrays

    def status(self, seed, split):
        self._manifest()
        path = self.cell_dir(seed, split)
        if not path.exists() or not (path / 'noncompletion_marker.json').exists():
            return super().status(seed, split)
        _require(path.is_dir() and {p.name for p in path.iterdir()} == set(NONCOMPLETE_FILES),
                 'partial or mixed recovery noncompletion; preserve without overwrite')
        marker = read_json(path / 'noncompletion_marker.json')
        _require(set(marker) == {'identity', 'outcome', 'artifacts'}
                 and canonical(marker['identity']) == canonical(self.identity(seed, split))
                 and marker['outcome'] == 'validation_patience_limit_200'
                 and set(marker['artifacts']) == set(NONCOMPLETE_FILES[:-1]),
                 'noncompletion marker identity or inventory mismatch')
        for name, checksum in marker['artifacts'].items():
            artifact = safe_path(path / name)
            _require(lifecycle._hash(checksum) and artifact.is_file()
                     and hashlib.sha256(artifact.read_bytes()).hexdigest() == checksum,
                     'noncompletion artifact checksum mismatch: ' + name)
        self._started(seed, split)
        self._read_noncompletion(seed, split)
        self._current()
        return 'noncompleted'

    def noncomplete(self, seed, split, result, arrays):
        self._manifest()
        path = self.cell_dir(seed, split)
        _require(path.is_dir() and {p.name for p in path.iterdir()} == {'started.json'},
                 'partial or terminal recovery unit cannot be overwritten')
        self._started(seed, split)
        validate_noncompletion_result(result, arrays, self.identity(seed, split), self.config)
        encoded = canonical(result).encode()
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        self._current()
        create_bytes(path / 'diagnostic_model.npz', buffer.getvalue())
        create_bytes(path / 'noncompletion.json', encoded)
        self._read_noncompletion(seed, split)
        self._current()
        marker = dict(identity=self.identity(seed, split), outcome='validation_patience_limit_200',
                      artifacts={name: hashlib.sha256(safe_path(path / name).read_bytes()).hexdigest()
                                 for name in NONCOMPLETE_FILES[:-1]})
        create_bytes(path / 'noncompletion_marker.json', canonical(marker).encode())
        _require(self.status(seed, split) == 'noncompleted', 'new noncompletion did not validate')

    def fail(self, seed, split, error):
        _require(not (self.cell_dir(seed, split) / 'noncompletion_marker.json').exists(),
                 'validated noncompletion cannot be relabeled as unknown failure')
        super().fail(seed, split, error)


def terminal_unit_record(store, seed, split, *, status=None):
    """Bind an already validated outcome; an explicit status avoids repeated checks."""
    status = store.status(seed, split) if status is None else status
    _require(status in ('validated', 'noncompleted'), 'shard still contains a nonterminal scientific unit')
    files = COMPLETE_FILES if status == 'validated' else NONCOMPLETE_FILES
    path = store.cell_dir(seed, split)
    return dict(seed=seed, split=split, outcome='complete' if status == 'validated' else 'budget_exhausted',
                artifacts={name: hashlib.sha256(safe_path(path / name).read_bytes()).hexdigest()
                           for name in files})


def shard_summary_path(store, job):
    _require(isinstance(job.get('job_id'), str), 'missing recovery shard job identity')
    # Validate before allowing a job identity to participate in any write path.
    validate_execution(dict(job_id=job['job_id'], attempt=1,
        **{key: job[key] for key in ('execution_profile', 'allocation', 'estimated_runtime_seconds')}))
    return safe_path(store.directory / 'shard_summaries' / (job['job_id'] + '.json'))


def _summary_body(store, job, execution, verified_units=None):
    parameters = protocol.validate_parameters(job['task'], job['parameters'])
    _require(canonical(protocol.run_config(job['task'], parameters)) == canonical(store.config),
             'shard summary configuration mismatch')
    validate_execution(execution)
    _require(execution['job_id'] == job['job_id'], 'shard summary belongs to another producer')
    expected = [(seed, split) for seed in protocol.seed_sequence(job['task'], parameters)
                for split in protocol.SPLITS]
    units = ([terminal_unit_record(store, seed, split) for seed, split in expected]
             if verified_units is None else deepcopy(verified_units))
    _require(isinstance(units, list) and [(u.get('seed'), u.get('split')) for u in units] == expected,
             'shard terminal inventory does not cover its exact deterministic sequence')
    for unit in units:
        _require(set(unit) == {'seed', 'split', 'outcome', 'artifacts'}
                 and unit['outcome'] in ('complete', 'budget_exhausted'), 'invalid shard terminal record')
        files = COMPLETE_FILES if unit['outcome'] == 'complete' else NONCOMPLETE_FILES
        _require(isinstance(unit['artifacts'], dict) and set(unit['artifacts']) == set(files)
                 and all(lifecycle._hash(value) for value in unit['artifacts'].values()),
                 'shard terminal artifact inventory mismatch')
        receipt = store._started(unit['seed'], unit['split'])['execution']
        _require(receipt['job_id'] == job['job_id'] and receipt['attempt'] <= execution['attempt'],
                 'shard unit has a foreign or later producing attempt')
    complete = sum(unit['outcome'] == 'complete' for unit in units)
    return dict(schema_version=1, protocol=protocol.PROTOCOL, run_id=store.run_id,
                config_sha256=store.config_sha256, provenance=store.provenance,
                job_id=job['job_id'], task=job['task'], parameters=parameters,
                producer_execution=execution, prerequisites=store.prerequisites,
                units=units, counts=dict(complete=complete, budget_exhausted=len(units)-complete,
                                        unattempted=0, errors=0, planned=len(units)),
                execution_complete=True, whole_shard_scientific_success=complete == len(units))


def validate_shard_summary(store, job, verified_units=None):
    """Check saved summary against exact outcomes; never infer success from exit 0."""
    store._manifest()
    path = shard_summary_path(store, job)
    value = read_json(path)
    expected = _summary_body(store, job, value.get('producer_execution'), verified_units)
    _require(canonical(value) == canonical(expected), 'recovery shard summary differs from validated units')
    store._current()
    return value


def publish_shard_summary(store, job, execution):
    store._manifest()
    path = shard_summary_path(store, job)
    if path.exists():
        return validate_shard_summary(store, job)
    value = _summary_body(store, job, execution)
    path.parent.mkdir(exist_ok=True)
    safe_path(path.parent)
    store._current()
    create_bytes(path, canonical(value).encode())
    return validate_shard_summary(store, job)


def run_shard(job):
    """Run only centrally allocated work; no submission, resources, or workers."""
    import torch
    from . import recovery_acceptance as gate

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = protocol.validate_parameters(job['task'], job['parameters'])
    config = protocol.run_config(job['task'], parameters)
    provenance = current_provenance()
    closeout = gate.verify_closeout()
    prerequisites = dict(closeout=closeout['binding'], pilot=None)
    if job['task'] == protocol.FORMAL_TASK:
        path = gate.default_acceptance_path(provenance)
        proof = gate.verify_pilot_acceptance(path)
        _require(proof['closeout'] == prerequisites['closeout'], 'formal gate closeout mismatch')
        old_requests = closeout['old_inventory']['requests']
        candidates = [row for row in old_requests if row['parameters'] == parameters]
        _require(len(candidates) == 1 and candidates[0]['job_id'] not in closeout['retained_request_ids'],
                 'formal recovery must replace exactly a non-retained old shard')
        prerequisites['pilot'] = gate._binding(path)
    _require(current_provenance() == provenance, 'source changed during recovery admission')
    store = RecoveryStore(config, provenance, prerequisites)
    store.initialize()
    execution = {key: job[key] for key in (
        'job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in protocol.seed_sequence(job['task'], parameters):
        for split in protocol.SPLITS:
            _require(current_provenance() == provenance, 'source changed during recovery shard')
            if store.status(seed, split) in ('validated', 'noncompleted'):
                continue
            store.start(seed, split, execution)
            try:
                try:
                    result, arrays = compute_cell(config, seed, split)
                except ScientificNoncompletion as error:
                    _require(isinstance(error.result, dict), 'noncompletion lacks full scientific context')
                    result = deepcopy(error.result)
                    result['identity'] = store.identity(seed, split)
                    result['memory'] = dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                            scope=MEMORY_SCOPE)
                    _require(current_provenance() == provenance, 'source changed during capped fit')
                    store.noncomplete(seed, split, result, error.arrays)
                else:
                    result['identity'] = store.identity(seed, split)
                    _require(current_provenance() == provenance, 'source changed during recovery fit')
                    store.complete(seed, split, result, arrays)
            except Exception as error:
                store.fail(seed, split, error)
                raise
    summary = publish_shard_summary(store, job, execution)
    print(canonical(dict(job_id=job['job_id'], execution_complete=True,
                         whole_shard_scientific_success=summary['whole_shard_scientific_success'],
                         counts=summary['counts'])), flush=True)
    _require(current_provenance() == provenance, 'source changed before recovery task exit')
    return 0
