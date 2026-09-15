"""A distinct immutable completion contract for replay-only JSON observations."""
import hashlib
import math
import re

from ..artifacts import EffectArtifactStore, _atomic_create, read_json, safe_path
from ..wire import canonical
from .protocol import PROTOCOL
from .sources import verify_source_bindings
from .validation import validate_result


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ReplayArtifactStore(EffectArtifactStore):
    """Reuse manifest/start/failure safeguards, without creating new checkpoints."""

    def __init__(self, config, provenance, directory=None):
        super().__init__(config, provenance, directory)
        self.manifest['protocol'] = PROTOCOL

    def _semantic(self, result):
        validate_result(result, self.config, self.identity)

    def _started(self):
        started = super()._started()
        execution = started['execution']
        _require(set(execution) == {'job_id','execution_profile','allocation','attempt','estimated_runtime_seconds'},
                 'replay execution receipt fields mismatch')
        _require(isinstance(execution['job_id'], str)
                 and re.fullmatch(r'fedfisher-effects-replay-[A-Za-z0-9_.-]{1,102}', execution['job_id']) is not None
                 and execution['execution_profile'] == 'cpu-1core'
                 and type(execution['attempt']) is int and execution['attempt'] >= 1,
                 'invalid replay scheduler identity/profile/attempt')
        seconds = execution['estimated_runtime_seconds']
        _require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0,
                 'invalid replay runtime estimate')
        allocation = execution['allocation']
        fixed = dict(cpu_cores=1,memory_mib=8192,gpu_count=0,gpu_memory_mib=0,gpu_utilization_pct=0)
        _require(isinstance(allocation,dict) and set(allocation) == {*fixed,'exclusive_gpu'}
                 and all(type(allocation[k]) is int and allocation[k] == v for k,v in fixed.items())
                 and type(allocation['exclusive_gpu']) is bool, 'invalid replay final allocation')
        return started

    def status(self):
        self._current()
        if not self.directory.exists():
            return 'missing'
        self._manifest()
        if not self.unit_dir.exists():
            return 'missing'
        _require(self.unit_dir.is_dir(), 'diagnostic unit is not a directory')
        marker = safe_path(self.unit_dir/'done.json')
        _require(marker.is_file(), 'partial diagnostic unit has no completion marker')
        done = read_json(marker)
        expected = {'started.json', 'result.json'}
        _require(set(done) == {'identity', 'artifacts'} and done['identity'] == self.identity
                 and set(done['artifacts']) == expected, 'diagnostic marker identity/inventory mismatch')
        _require({p.name for p in self.unit_dir.iterdir()} == expected | {'done.json'},
                 'unexpected or partial diagnostic files')
        for name, digest in done['artifacts'].items():
            _require(hashlib.sha256(safe_path(self.unit_dir/name).read_bytes()).hexdigest() == digest,
                     'diagnostic artifact checksum mismatch: '+name)
        self._started()
        verify_source_bindings(self.config)
        self._semantic(read_json(self.unit_dir/'result.json'))
        self._current()
        return 'validated'

    def complete(self, result):
        self._manifest()
        _require(self.unit_dir.is_dir() and {p.name for p in self.unit_dir.iterdir()} == {'started.json'},
                 'partial or completed diagnostic output cannot be overwritten')
        self._started()
        self._semantic(result)
        self._current()
        _atomic_create(self.unit_dir/'result.json', canonical(result).encode())
        self._semantic(read_json(self.unit_dir/'result.json'))
        self._current()
        done = dict(identity=self.identity, artifacts={name: hashlib.sha256(
            safe_path(self.unit_dir/name).read_bytes()).hexdigest() for name in ('started.json', 'result.json')})
        _atomic_create(self.unit_dir/'done.json', canonical(done).encode())
        _require(self.status() == 'validated', 'new diagnostic completion did not validate')
