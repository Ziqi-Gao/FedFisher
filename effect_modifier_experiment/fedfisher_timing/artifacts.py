"""Immutable CPU FedFisher timing units with current-source provenance checks."""

import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np

from ..infrastructure.artifacts import (
    OUTPUT_ROOT, PROVENANCE_KEYS, canonical, create_bytes, digest, read_json, safe_path,
)
from ..infrastructure.provenance import canonical_text, compute_provenance, env_text, parse_env_file
from . import protocol
from .validation import validate_result


PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def current_provenance():
    provenance = compute_provenance(PROJECT_ROOT)
    _require(provenance['git_dirty'] == '0', 'FedFisher timing requires finalized clean source')
    return provenance


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_execution(execution):
    _require(isinstance(execution, dict) and set(execution) == {
        'job_id', 'execution_profile', 'allocation', 'attempt', 'estimated_runtime_seconds'},
        'timing execution receipt fields mismatch')
    _require(isinstance(execution['job_id'], str) and re.fullmatch(
        r'fedfisher-timing-[A-Za-z0-9_.-]{1,112}', execution['job_id']) is not None,
        'invalid timing scheduler job identity')
    _require(execution['execution_profile'] == 'cpu-1core'
             and type(execution['attempt']) is int and execution['attempt'] >= 1,
             'invalid timing execution profile or attempt')
    seconds = execution['estimated_runtime_seconds']
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
             'invalid timing runtime estimate')
    allocation = execution['allocation']
    fixed = dict(cpu_cores=1, memory_mib=8192, gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0)
    _require(isinstance(allocation, dict) and set(allocation) == {*fixed, 'exclusive_gpu'}
             and all(type(allocation[key]) is int and allocation[key] == value for key, value in fixed.items())
             and type(allocation['exclusive_gpu']) is bool,
             'invalid timing final allocation')


class ArtifactStore:
    """Read-only status and exclusive publication for deterministic seed/split cells."""

    def __init__(self, config, provenance, directory=None):
        self.config = json.loads(canonical(config))
        parameters = dict(cell=self.config['cell'], variant=self.config['variant'],
            shard_index=0, shard_count=1 if self.config['task'] == protocol.PILOT_TASK else 8)
        _require(self.config == protocol.run_config(self.config['task'], parameters),
                 'timing configuration is not a reviewed fixed recipe')
        self.provenance = dict(provenance)
        _require(set(self.provenance) == {*PROVENANCE_KEYS, 'provenance_sha256'},
                 'timing provenance field inventory mismatch')
        _require(self.provenance['git_dirty'] == '0' and isinstance(self.provenance['git_commit'], str)
                 and re.fullmatch('[0-9a-f]{40}', self.provenance['git_commit']) is not None,
                 'timing provenance requires a finalized Git commit')
        _require(all(_hash(self.provenance[key]) for key in
                     ('git_status_sha256', 'source_tree_sha256', 'provenance_sha256')),
                 'invalid timing provenance digest')
        _require(hashlib.sha256(canonical_text(self.provenance, PROVENANCE_KEYS).encode()).hexdigest()
                 == self.provenance['provenance_sha256'], 'inconsistent timing provenance digest')
        self.run_id = self.config['run_id']
        self.config_sha256 = digest(self.config)
        self.directory = safe_path(OUTPUT_ROOT/self.run_id if directory is None else directory)
        _require(self.directory.name == self.run_id, 'timing directory/run identity mismatch')
        self.manifest = dict(format_version='2', protocol=self.config['protocol'], run_id=self.run_id,
            config_json=canonical(self.config), config_sha256=self.config_sha256, **self.provenance)
        self._current()

    def _current(self):
        _require(canonical(self.config) == self.manifest['config_json']
                 and digest(self.config) == self.config_sha256,
                 'timing configuration changed after construction')
        _require(current_provenance() == self.provenance,
                 'current source provenance differs from timing scientific manifest')
        safe_path(self.directory)

    def _manifest(self):
        self._current()
        path = safe_path(self.directory/'run_manifest.env')
        _require(path.is_file() and parse_env_file(path) == self.manifest,
                 'timing run manifest/provenance/configuration mismatch')

    def initialize(self):
        self._current()
        path = safe_path(self.directory/'run_manifest.env')
        if self.directory.exists():
            _require(self.directory.is_dir(), 'timing run path is not a directory')
            _require(path.exists() or not any(self.directory.iterdir()),
                     'nonempty timing run directory lacks a manifest')
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_path(self.directory)
        if not path.exists():
            try:
                create_bytes(path, env_text(self.manifest.items()).encode())
            except FileExistsError:
                pass  # Another admitted deterministic shard may initialize the same run.
        self._manifest()

    def identity(self, seed, split):
        _require(type(seed) is int and self.config['seeds'][0] <= seed <= self.config['seeds'][1]
                 and isinstance(split, str) and split in self.config['splits'], 'invalid timing scientific unit')
        return dict(run_id=self.run_id, config_sha256=self.config_sha256,
            provenance_sha256=self.provenance['provenance_sha256'],
            seed=seed, split=split, cell=self.config['cell'], variant=self.config['variant'])

    def cell_dir(self, seed, split):
        self.identity(seed, split)
        return safe_path(self.directory/('seed_%03d_%s' % (seed, split)))

    def _started(self, seed, split):
        started = read_json(self.cell_dir(seed, split)/'started.json')
        _require(set(started) == {'identity', 'execution'} and started['identity'] == self.identity(seed, split),
                 'timing started identity mismatch')
        validate_execution(started['execution'])
        return started

    def _read(self, seed, split):
        path = self.cell_dir(seed, split)
        result = read_json(path/'result.json')
        with np.load(safe_path(path/'model.npz'), allow_pickle=False) as archive:
            _require(len(archive.files) == len(set(archive.files)), 'duplicate timing model arrays')
            arrays = {name: archive[name] for name in archive.files}
        validate_result(result, arrays, self.identity(seed, split), self.config)
        return result, arrays

    def status(self, seed, split):
        self._manifest()
        path = self.cell_dir(seed, split)
        if not path.exists():
            self._current()
            return 'missing'
        _require(path.is_dir() and {entry.name for entry in path.iterdir()} == {
            'started.json', 'model.npz', 'result.json', 'done.json'},
            'partial or invalid timing output; preserve without overwrite')
        marker = read_json(path/'done.json')
        _require(set(marker) == {'identity', 'artifacts'} and marker['identity'] == self.identity(seed, split)
                 and set(marker['artifacts']) == {'started.json', 'model.npz', 'result.json'},
                 'timing completion identity/inventory mismatch')
        for name, checksum in marker['artifacts'].items():
            artifact = safe_path(path/name)
            _require(_hash(checksum) and artifact.is_file()
                     and hashlib.sha256(artifact.read_bytes()).hexdigest() == checksum,
                     'timing artifact checksum mismatch: '+name)
        self._started(seed, split)
        self._read(seed, split)
        self._current()
        return 'validated'

    def start(self, seed, split, execution):
        self._manifest()
        validate_execution(execution)
        if self.status(seed, split) != 'missing':
            raise FileExistsError('timing cell already complete')
        path = self.cell_dir(seed, split)
        encoded = canonical(dict(identity=self.identity(seed, split), execution=execution)).encode()
        self._current()
        path.mkdir()
        safe_path(path)
        create_bytes(path/'started.json', encoded)
        self._current()

    def complete(self, seed, split, result, arrays):
        self._manifest()
        path = self.cell_dir(seed, split)
        _require(path.is_dir() and {entry.name for entry in path.iterdir()} == {'started.json'},
                 'partial or completed timing cell cannot be overwritten')
        self._started(seed, split)
        validate_result(result, arrays, self.identity(seed, split), self.config)
        encoded = canonical(result).encode()
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        self._current()
        create_bytes(path/'model.npz', buffer.getvalue())
        create_bytes(path/'result.json', encoded)
        self._read(seed, split)
        self._current()
        marker = dict(identity=self.identity(seed, split), artifacts={name: hashlib.sha256(
            safe_path(path/name).read_bytes()).hexdigest() for name in ('started.json', 'model.npz', 'result.json')})
        create_bytes(path/'done.json', canonical(marker).encode())
        _require(self.status(seed, split) == 'validated', 'new timing completion did not validate')

    def fail(self, seed, split, error):
        self._manifest()
        path = self.cell_dir(seed, split)
        self._started(seed, split)
        _require(not (path/'done.json').exists(), 'completed timing output cannot be marked failed')
        create_bytes(path/'failure.json', canonical(dict(identity=self.identity(seed, split),
            error_type=type(error).__name__, message=str(error))).encode())
        self._current()
