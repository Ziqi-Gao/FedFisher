"""Source-bound survival results, strict semantic validation and immutable retries."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np

from ..causal_comparison.artifacts import (
    OUTPUT_ROOT, PROVENANCE_KEYS, canonical, create_bytes, digest, read_json, safe_path)
from ..dslassocov_run_contract import canonical_text, compute_provenance, env_text, parse_env_file
from . import protocol

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def current_provenance():
    result = compute_provenance(PROJECT_ROOT)
    _require(result['git_dirty'] == '0', 'survival execution requires finalized clean source')
    return result


def array_hash(arrays):
    """Bind names, dtype, shape and actual bytes, including serialized site uploads."""
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        result.update(canonical([name, value.dtype.str, list(value.shape)]).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def rows_hash(rows):
    return array_hash({name: np.asarray(getattr(rows, name))
                       for name in ('x', 'a', 'time', 'event', 'row_ids')})


def dataset_fingerprints(dataset):
    return dict(training=digest([dict(site_id=site.site_id, rows=rows_hash(site.rows))
                                 for site in dataset.sites]),
                evaluation=rows_hash(dataset.evaluation),
                evaluation_truth=array_hash({name: np.asarray(getattr(dataset.evaluation_truth, name))
                    for name in ('propensity', 'eta0', 'eta1', 'survival0', 'survival1',
                                 'delta', 'modifier_mask')}))


def validate_execution(execution):
    _require(isinstance(execution, dict) and set(execution) == {
        'job_id', 'execution_profile', 'allocation', 'attempt', 'estimated_runtime_seconds'},
        'survival execution receipt fields mismatch')
    _require(isinstance(execution['job_id'], str) and re.fullmatch(
        r'fedfisher-survival-[A-Za-z0-9_.-]{1,109}', execution['job_id']) is not None,
        'invalid survival scheduler job identity')
    _require(execution['execution_profile'] == 'cpu-1core'
             and type(execution['attempt']) is int and execution['attempt'] >= 1,
             'invalid survival execution profile or attempt')
    seconds = execution['estimated_runtime_seconds']
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
             'invalid survival runtime estimate')
    fixed = dict(cpu_cores=1, memory_mib=8192, gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0)
    allocation = execution['allocation']
    _require(isinstance(allocation, dict) and set(allocation) == {*fixed, 'exclusive_gpu'}
             and all(type(allocation[key]) is int and allocation[key] == value for key, value in fixed.items())
             and type(allocation['exclusive_gpu']) is bool, 'invalid survival allocation')


def validate_pseudo_diagnostics(diagnostics, sizes, tau):
    _require(isinstance(diagnostics, list) and len(diagnostics) == len(sizes),
             'survival pseudo diagnostic site count mismatch')
    expected = {'site_id', 'n', 'event_count', 'censored_count', 'censoring_fraction',
                'n_at_risk_tau', 'km_survival_tau', 'pv_min', 'pv_max', 'pv_mean',
                'outside_unit_interval_count', 'all_zero', 'all_one', 'tau',
                'max_observed_time', 'tau_beyond_observed_max'}
    for index, (row, size) in enumerate(zip(diagnostics, sizes)):
        _require(isinstance(row, dict) and set(row) == expected
                 and type(row['site_id']) is int and row['site_id'] == index
                 and type(row['n']) is int and row['n'] == size and row['tau'] == tau,
                 'survival pseudo diagnostic identity mismatch')
        for name in ('event_count', 'censored_count', 'n_at_risk_tau', 'outside_unit_interval_count'):
            _require(type(row[name]) is int and 0 <= row[name] <= size,
                     'invalid survival pseudo diagnostic count')
        _require(row['event_count'] + row['censored_count'] == size
                 and row['censoring_fraction'] == row['censored_count'] / size,
                 'survival censoring counts/fraction disagree')
        for name in ('km_survival_tau', 'pv_min', 'pv_max', 'pv_mean', 'max_observed_time'):
            _require(type(row[name]) in (int, float) and math.isfinite(row[name]),
                     'invalid survival pseudo diagnostic number')
        _require(0 <= row['km_survival_tau'] <= 1 and row['max_observed_time'] >= 0
                 and row['pv_min'] <= row['pv_mean'] <= row['pv_max']
                 and math.isclose(row['pv_mean'], row['km_survival_tau'], rel_tol=1e-9, abs_tol=1e-12),
                 'survival pseudo response and KM mean disagree')
        _require(type(row['all_zero']) is bool and row['all_zero'] == (row['pv_min'] == row['pv_max'] == 0)
                 and type(row['all_one']) is bool and row['all_one'] == (row['pv_min'] == row['pv_max'] == 1)
                 and type(row['tau_beyond_observed_max']) is bool
                 and row['tau_beyond_observed_max'] == (tau > row['max_observed_time'])
                 and (row['outside_unit_interval_count'] == 0)
                 == (row['pv_min'] >= 0 and row['pv_max'] <= 1),
                 'survival pseudo response flags disagree')
        _require(not row['tau_beyond_observed_max'] or row['n_at_risk_tau'] == 0,
                 'survival risk count disagrees with observed support')


def validate_result(result, arrays, identity, config):
    """Validate saved arithmetic/recipe claims; no data generation or prediction."""
    from .evaluation import validate_evaluation
    try:
        canonical(result)
        _require(isinstance(result, dict) and set(result) == {
            'identity', 'status', 'fit', 'evaluation', 'timing', 'site_sizes',
            'fingerprints', 'memory', 'pseudo_diagnostics'}, 'survival result field inventory mismatch')
        _require(result['identity'] == identity and result['status'] == 'complete',
                 'survival result identity/status mismatch')
        dgp, method = config['dgp'], config['method']
        sizes = result['site_sizes']
        _require(isinstance(sizes, list) and len(sizes) == dgp['num_sites']
                 and all(type(n) is int and dgp['site_min_samples'] <= n <= dgp['site_max_samples']
                         for n in sizes), 'invalid survival site counts')
        validate_pseudo_diagnostics(result['pseudo_diagnostics'], sizes, dgp['tau'])
        _require(isinstance(arrays, dict) and arrays, 'missing survival model arrays')
        for name, value in arrays.items():
            _require(isinstance(name, str) and isinstance(value, np.ndarray)
                     and not value.dtype.hasobject and np.isfinite(value).all(),
                     'invalid survival model array')
        if method['family'] == 'dslasso_survival_nls':
            from .dslasso import validate_fit_diagnostics
            _require(set(arrays) == {'intercept', 'coefficients'}
                     and arrays['intercept'].shape == ()
                     and arrays['coefficients'].shape == (1 + 2 * dgp['covariate_dim'],)
                     and all(value.dtype == np.float64 for value in arrays.values()),
                     'invalid survival linear model representation')
            validate_fit_diagnostics(result['fit'], method['optimizer'], arrays, sizes)
        else:
            from .fisher import FisherConfig, validate_fit_report
            expected_uploads = ['upload_%03d' % index for index in range(len(sizes))]
            _require(set(arrays) == {'vector', *expected_uploads}
                     and arrays['vector'].dtype == np.float64 and arrays['vector'].ndim == 1,
                     'invalid survival neural model representation')
            _require(all(arrays[key].dtype == np.uint8 and arrays[key].ndim == 1
                         and arrays[key].size > 0 for key in expected_uploads),
                     'invalid serialized survival site uploads')
            uploads = tuple(arrays[key].tobytes() for key in expected_uploads)
            validate_fit_report(result['fit'], FisherConfig(**method['optimizer']), method['fisher'],
                                identity['seed'], sizes, arrays['vector'], uploads,
                                1 + 2 * dgp['covariate_dim'], config['public_initial_eta'])
        fingerprints = result['fingerprints']
        _require(isinstance(fingerprints, dict) and set(fingerprints) == {
            'training', 'evaluation', 'evaluation_truth', 'model'}
            and all(_hash(value) for value in fingerprints.values()),
            'invalid survival fingerprints')
        _require(fingerprints['model'] == array_hash(arrays), 'survival model fingerprint mismatch')
        validate_evaluation(result['evaluation'], dgp['covariate_dim'], dgp['num_test'],
                            config['evaluation']['repeats'])
        timing = result['timing']
        _require(isinstance(timing, dict) and set(timing) == {'stages', 'fit_includes_all_selection_and_refits',
                 'network_transport_measured'} and timing['fit_includes_all_selection_and_refits'] is True
                 and timing['network_transport_measured'] is False,
                 'survival timing scope mismatch')
        _require(set(timing['stages']) == set(config['timing']['stages']),
                 'survival measured timing stages mismatch')
        for stage in timing['stages'].values():
            _require(isinstance(stage, dict) and set(stage) == {'wall_seconds', 'cpu_seconds'},
                     'survival timing clock inventory mismatch')
            for value in stage.values():
                _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                         'survival timing requires observed positive durations')
        inner_fit_seconds = (result['fit']['timing_seconds']['full_fit']
                             if method['family'] == 'dslasso_survival_nls'
                             else result['fit']['total_fit_seconds'])
        _require(inner_fit_seconds <= timing['stages']['fit']['wall_seconds'] + 1e-6,
                 'outer survival fit timing omits method fitting work')
        memory = result['memory']
        _require(set(memory) == {'process_peak_rss_kib', 'scope'}
                 and type(memory['process_peak_rss_kib']) is int and memory['process_peak_rss_kib'] > 0
                 and memory['scope'] == 'Linux current-process lifetime high-water RSS; not cgroup peak or reservation',
                 'survival memory observation scope mismatch')
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('malformed survival result') from error


class ArtifactStore:
    """One scenario/method manifest and immutable seed cells per scientific run."""

    def __init__(self, config, provenance, directory=None, prerequisite=None):
        self.config = json.loads(canonical(config))
        parameters = dict(cell=self.config['cell'], scenario=self.config['scenario'])
        if self.config['task'] == protocol.FORMAL_TASK:
            parameters.update(shard_index=0, shard_count=16)
        _require(self.config == protocol.run_config(self.config['task'], parameters),
                 'survival configuration is not a reviewed fixed recipe')
        self.provenance = dict(provenance)
        _require(set(self.provenance) == {*PROVENANCE_KEYS, 'provenance_sha256'},
                 'survival provenance field inventory mismatch')
        _require(self.provenance['git_dirty'] == '0' and isinstance(self.provenance['git_commit'], str)
                 and re.fullmatch('[0-9a-f]{40}', self.provenance['git_commit']) is not None,
                 'survival provenance requires a finalized Git commit')
        _require(all(_hash(self.provenance[key]) for key in
                     ('git_status_sha256', 'source_tree_sha256', 'provenance_sha256')),
                 'invalid survival provenance digest')
        _require(hashlib.sha256(canonical_text(self.provenance, PROVENANCE_KEYS).encode()).hexdigest()
                 == self.provenance['provenance_sha256'], 'inconsistent survival provenance digest')
        if self.config['task'] == protocol.FORMAL_TASK:
            _require(isinstance(prerequisite, dict) and set(prerequisite) == {
                'pilot_acceptance_sha256', 'provenance_sha256'}
                and _hash(prerequisite['pilot_acceptance_sha256'])
                and prerequisite['provenance_sha256'] == self.provenance['provenance_sha256'],
                'formal survival store requires source-bound pilot acceptance evidence')
        else:
            _require(prerequisite is None, 'pilot stores cannot bind formal prerequisites')
        self.prerequisite = None if prerequisite is None else json.loads(canonical(prerequisite))
        self._prerequisite_json = canonical(self.prerequisite)
        self.config_sha256 = digest(self.config)
        self.run_id = self.config['run_id']
        self.directory = safe_path(OUTPUT_ROOT/self.run_id if directory is None else directory)
        _require(self.directory.name == self.run_id, 'survival directory/run identity mismatch')
        self.manifest = dict(format_version='2', protocol=protocol.PROTOCOL, run_id=self.run_id,
            config_json=canonical(self.config), config_sha256=self.config_sha256, **self.provenance)
        self._current()

    def identity(self, seed=0):
        protocol.validate_seed(self.config, seed)
        return dict(run_id=self.run_id, config_sha256=self.config_sha256,
                    provenance_sha256=self.provenance['provenance_sha256'],
                    seed=seed, cell=self.config['cell'], scenario=self.config['scenario'])

    def _current(self):
        _require(canonical(self.config) == self.manifest['config_json']
                 and digest(self.config) == self.config_sha256, 'survival configuration changed')
        _require(canonical(self.prerequisite) == self._prerequisite_json,
                 'survival prerequisite changed after construction')
        _require(current_provenance() == self.provenance,
                 'current source provenance differs from survival scientific manifest')
        safe_path(self.directory)

    def _manifest(self):
        self._current()
        path = safe_path(self.directory/'run_manifest.env')
        _require(path.is_file() and parse_env_file(path) == self.manifest,
                 'survival run manifest/provenance/configuration mismatch')

    def initialize(self):
        self._current()
        path = safe_path(self.directory/'run_manifest.env')
        if self.directory.exists():
            _require(self.directory.is_dir(), 'survival run path is not a directory')
            # Inspect contents before checking the manifest: another shard may
            # atomically publish it between these two observations.
            contents = tuple(self.directory.iterdir())
            _require(not contents or path.exists(),
                     'nonempty survival run directory lacks manifest')
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_path(self.directory)
        if not path.exists():
            # Keep the in-flight manifest outside the shared run directory.
            # Other admitted shards may observe an empty run during publication;
            # an internal temporary would falsely look like partial output.
            temporary = safe_path(self.directory.parent/(
                '.%s.manifest.%d.%s.tmp' % (self.run_id, os.getpid(), uuid.uuid4().hex)))
            try:
                with temporary.open('xb') as stream:
                    stream.write(env_text(self.manifest.items()).encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass  # Validate the other admitted shard's exact manifest below.
            finally:
                temporary.unlink(missing_ok=True)
        self._manifest()

    def cell_dir(self, seed=0):
        self.identity(seed)
        return safe_path(self.directory/('seed_%03d' % seed))

    def _started(self, seed=0):
        value = read_json(self.cell_dir(seed)/'started.json')
        fields = {'identity', 'execution'} if self.prerequisite is None else {'identity', 'execution', 'prerequisite'}
        _require(set(value) == fields and value['identity'] == self.identity(seed),
                 'survival started identity mismatch')
        if self.prerequisite is not None:
            _require(value['prerequisite'] == self.prerequisite,
                     'formal survival started pilot prerequisite mismatch')
        validate_execution(value['execution'])
        return value

    def _read(self, seed=0):
        path = self.cell_dir(seed)
        result = read_json(path/'result.json')
        with np.load(safe_path(path/'model.npz'), allow_pickle=False) as archive:
            _require(len(archive.files) == len(set(archive.files)), 'duplicate survival model arrays')
            arrays = {name: archive[name] for name in archive.files}
        validate_result(result, arrays, self.identity(seed), self.config)
        return result, arrays

    def status(self, seed=0):
        self._manifest()
        path = self.cell_dir(seed)
        if not path.exists():
            self._current()
            return 'missing'
        _require(path.is_dir() and {entry.name for entry in path.iterdir()} == {
            'started.json', 'model.npz', 'result.json', 'done.json'},
            'partial or invalid survival output; preserve without overwrite')
        marker = read_json(path/'done.json')
        _require(set(marker) == {'identity', 'artifacts'} and marker['identity'] == self.identity(seed)
                 and set(marker['artifacts']) == {'started.json', 'model.npz', 'result.json'},
                 'survival completion identity/inventory mismatch')
        for name, checksum in marker['artifacts'].items():
            artifact = safe_path(path/name)
            _require(_hash(checksum) and artifact.is_file()
                     and hashlib.sha256(artifact.read_bytes()).hexdigest() == checksum,
                     'survival artifact checksum mismatch: '+name)
        self._started(seed)
        self._read(seed)
        self._current()
        return 'validated'

    def start(self, execution, seed=0):
        self._manifest()
        validate_execution(execution)
        if self.status(seed) != 'missing':
            raise FileExistsError('survival cell already complete')
        path = self.cell_dir(seed)
        self._current()
        path.mkdir()
        safe_path(path)
        started = dict(identity=self.identity(seed), execution=execution)
        if self.prerequisite is not None:
            started['prerequisite'] = self.prerequisite
        create_bytes(path/'started.json', canonical(started).encode())
        self._current()

    def complete(self, result, arrays, seed=0):
        self._manifest()
        path = self.cell_dir(seed)
        _require(path.is_dir() and {entry.name for entry in path.iterdir()} == {'started.json'},
                 'partial or completed survival cell cannot be overwritten')
        self._started(seed)
        validate_result(result, arrays, self.identity(seed), self.config)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        self._current()
        create_bytes(path/'model.npz', buffer.getvalue())
        create_bytes(path/'result.json', canonical(result).encode())
        self._read(seed)
        self._current()
        marker = dict(identity=self.identity(seed), artifacts={name: hashlib.sha256(
            safe_path(path/name).read_bytes()).hexdigest() for name in ('started.json', 'model.npz', 'result.json')})
        create_bytes(path/'done.json', canonical(marker).encode())
        _require(self.status(seed) == 'validated', 'new survival completion did not validate')

    def fail(self, error, seed=0):
        self._manifest()
        path = self.cell_dir(seed)
        self._started(seed)
        _require(not (path/'done.json').exists(), 'completed survival output cannot be marked failed')
        create_bytes(path/'failure.json', canonical(dict(identity=self.identity(seed),
            error_type=type(error).__name__, message=str(error))).encode())
        self._current()
