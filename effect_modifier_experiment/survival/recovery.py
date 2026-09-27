"""Fixed dsLasso diagnostic/recovery lifecycle; original survival contracts stay intact.

Diagnostic completion is an observation of a prescribed numerical attempt. Only
stationary models qualify as formal results. Both use the existing immutable
survival store, with distinct identity, filenames and completion semantics.
"""
from contextlib import contextmanager
import hashlib
import io
import math
import os
import resource
import time

import numpy as np

from . import artifacts
from .artifacts import _hash, _require, array_hash, canonical, dataset_fingerprints, digest

from .recovery_protocol import (
    PROTOCOL, DIAGNOSTIC_TASK, FORMAL_TASK, TASKS, SCENARIOS, RECIPES,
    ORIGINAL_CLOSEOUT, CASE_INVENTORY, CASES, validate_parameters, seed_sequence,
    run_id, method_config, run_config, validate_seed,
    REPAIR_DIAGNOSTIC_TASK, REPAIR_FORMAL_TASK, REPAIR_CASES, REPAIR_RECIPE,
    task_epoch, is_diagnostic, cases_for_task, tasks_for_epoch, config_parameters)

MEMORY_SCOPE = 'Linux current-process lifetime high-water RSS; not cgroup peak or reservation'


class FitValidationFailure(ValueError):
    """Preserve a returned fit without treating a rejected result as complete."""

    def __init__(self, error, diagnostics, arrays, stage, evidence=None):
        super().__init__(str(error))
        self.original_error_type = type(error).__name__
        self.diagnostics = diagnostics
        self.arrays = {name: value.copy() for name, value in arrays.items()}
        self.stage = stage
        self.evidence = {} if evidence is None else evidence


def validate_reference(result, reference):
    """Compare saved scientific identities; never regenerate old data or infer."""
    for name in ('training', 'evaluation', 'evaluation_truth'):
        _require(result['fingerprints'][name] == reference['fingerprints'][name],
                 'recovery differs from accepted original ' + name + ' data')
    for name in ('site_sizes', 'pseudo_diagnostics'):
        _require(result[name] == reference[name], 'recovery differs from accepted original ' + name)


def _validate_fit(result, arrays, identity, config, require_success):
    from .dslasso import DslassoConfig
    from .recovery_method import RecoveryPolicy, validate_recovery_diagnostics
    _require(result['fit']['context'] == dict(scenario=config['scenario'], seed=identity['seed'],
                                             recipe=config['recovery_recipe']),
             'recovery optimizer context differs from scientific identity')
    p = config['dgp']['covariate_dim']
    for path in result['fit']['paths']:
        _require(np.asarray(path['means']).shape == (1 + 2 * p,)
                 and np.asarray(path['scales']).shape == (1 + 2 * p,),
                 'recovery normalization differs from declared covariate dimension')
        for index, report in enumerate([path['unpenalized_start']] + path['fits']):
            width = p + 2 if index == 0 else 2 * p + 2
            _require(np.asarray(report['terminal_beta']).shape == (width,)
                     and np.asarray(report['terminal_gradient']).shape == (width,),
                     'recovery optimizer endpoint differs from declared covariate dimension')
    validate_recovery_diagnostics(result['fit'], DslassoConfig(**config['method']['optimizer']),
        RecoveryPolicy(**config['method']['recovery_policy']), arrays, result['site_sizes'],
        require_success=require_success, required_schema=config.get('required_evidence_schema'))


def validate_result(result, arrays, identity, config):
    """Pure saved-evidence validation for diagnostics and stationary formal models."""
    try:
        if not is_diagnostic(config['task']):
            artifacts.validate_result(result, arrays, identity, config,
                dslasso_validator=lambda fit, optimizer, archive, sizes:
                    _validate_fit(result, archive, identity, config, True))
            return
        canonical(result)
        _require(isinstance(result, dict) and set(result) == {
            'identity', 'status', 'outcome', 'fit', 'timing', 'site_sizes',
            'fingerprints', 'memory', 'pseudo_diagnostics'}, 'diagnostic result field inventory mismatch')
        _require(result['identity'] == identity and result['status'] == 'diagnostic_complete'
                 and result['outcome'] in ('stationary', 'iteration_limit')
                 and result['outcome'] == result['fit']['outcome'],
                 'diagnostic identity/outcome mismatch')
        dgp, sizes = config['dgp'], result['site_sizes']
        _require(isinstance(sizes, list) and len(sizes) == dgp['num_sites']
                 and all(type(n) is int and dgp['site_min_samples'] <= n <= dgp['site_max_samples']
                         for n in sizes), 'invalid diagnostic site counts')
        artifacts.validate_pseudo_diagnostics(result['pseudo_diagnostics'], sizes, dgp['tau'])
        _require(isinstance(arrays, dict) and arrays and all(isinstance(name, str)
            and isinstance(value, np.ndarray) and value.dtype == np.float64 and np.isfinite(value).all()
            for name, value in arrays.items()), 'invalid diagnostic terminal arrays')
        if result['outcome'] == 'stationary':
            _require(set(arrays) == {'intercept', 'coefficients'} and arrays['intercept'].shape == ()
                     and arrays['coefficients'].shape == (1 + 2 * dgp['covariate_dim'],),
                     'invalid stationary diagnostic model representation')
        else:
            _require(set(arrays) == {'terminal_beta', 'terminal_gradient'}
                     and arrays['terminal_beta'].ndim == 1
                     and arrays['terminal_beta'].shape == arrays['terminal_gradient'].shape,
                     'invalid nonstationary diagnostic endpoint representation')
        _validate_fit(result, arrays, identity, config, True if task_epoch(config['task']) >= 2 else None)
        fp = result['fingerprints']
        archive_key = 'model' if result['outcome'] == 'stationary' else 'arrays'
        _require(isinstance(fp, dict) and set(fp) == {
            'training', 'evaluation', 'evaluation_truth', archive_key} and all(_hash(value) for value in fp.values())
            and fp[archive_key] == array_hash(arrays), 'invalid diagnostic fingerprints')
        timing = result['timing']
        _require(isinstance(timing, dict) and set(timing) == {
            'stages', 'fit_includes_all_selection_and_refits', 'network_transport_measured'}
            and timing['fit_includes_all_selection_and_refits'] is True
            and timing['network_transport_measured'] is False
            and set(timing['stages']) == {'data_preparation', 'fit'}, 'diagnostic timing scope mismatch')
        for stage in timing['stages'].values():
            _require(isinstance(stage, dict) and set(stage) == {'wall_seconds', 'cpu_seconds'}
                and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                        for value in stage.values()), 'invalid observed diagnostic timing')
        _require(result['fit']['timing_seconds']['full_fit'] <= timing['stages']['fit']['wall_seconds'] + 1e-6,
                 'diagnostic timing omits fitting work')
        memory = result['memory']
        _require(isinstance(memory, dict) and set(memory) == {'process_peak_rss_kib', 'scope'}
            and type(memory['process_peak_rss_kib']) is int and memory['process_peak_rss_kib'] > 0
            and memory['scope'] == MEMORY_SCOPE, 'diagnostic memory observation scope mismatch')
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('malformed survival recovery result') from error


class RecoveryStore(artifacts.ArtifactStore):
    """Reuse immutable publication; diagnostics cannot become formal done markers."""
    protocol_name = PROTOCOL

    def __init__(self, config, provenance, directory=None, prerequisite=None):
        diagnostic = is_diagnostic(config['task'])
        self.protocol_name = config['protocol']
        self.result_name = 'diagnostic_result.json' if diagnostic else 'result.json'
        self.model_name = 'fit_arrays.npz' if diagnostic else 'model.npz'
        self.done_name = 'diagnostic_done.json' if diagnostic else 'done.json'
        super().__init__(config, provenance, directory, prerequisite)
        if not diagnostic:
            self.manifest.update(prerequisite_json=canonical(self.prerequisite),
                                 prerequisite_sha256=digest(self.prerequisite))

    def _validate_config(self):
        expected = run_config(self.config['task'], config_parameters(self.config),
                              recipe=self.config['recovery_recipe'])
        _require(self.config == expected, 'survival recovery configuration is not a reviewed fixed recipe')

    def _validate_prerequisite(self, prerequisite):
        if is_diagnostic(self.config['task']):
            _require(prerequisite is None, 'diagnostic store cannot bind a formal prerequisite')
        else:
            _require(isinstance(prerequisite, dict) and set(prerequisite) == {
                'diagnostic_acceptance_sha256', 'provenance_sha256', 'selected_recipe'}
                and _hash(prerequisite['diagnostic_acceptance_sha256'])
                and prerequisite['provenance_sha256'] == self.provenance['provenance_sha256']
                and prerequisite['selected_recipe'] == self.config['recovery_recipe'],
                'formal recovery store requires same-source diagnostic acceptance and selected recipe')

    def _validate_seed(self, seed):
        return validate_seed(self.config, seed)

    def identity(self, seed=0):
        result = super().identity(seed)
        result['recipe'] = self.config['recovery_recipe']
        if is_diagnostic(self.config['task']):
            result['case'] = '%s_%03d' % (self.config['scenario'], seed)
            _require(result['case'] in cases_for_task(self.config['task']), 'unknown diagnostic case identity')
        return result

    def _validate_result(self, result, arrays, identity):
        validate_result(result, arrays, identity, self.config)

    def _failure_record(self, error, seed):
        from .recovery_method import OptimizationFailure, _finite_json
        record = super()._failure_record(error, seed)
        if isinstance(error, (OptimizationFailure, FitValidationFailure)):
            record['optimization_diagnostics'] = _finite_json(error.diagnostics)
        if isinstance(error, FitValidationFailure):
            record.update(failure_stage=error.stage, original_error_type=error.original_error_type,
                          saved_evidence=_finite_json(error.evidence), scientific_completion=False)
        return record

    def fail(self, error, seed=0):
        if not isinstance(error, FitValidationFailure):
            return super().fail(error, seed)
        self._manifest()
        self._started(seed)
        path = self.cell_dir(seed)
        _require(not (path / self.done_name).exists(), 'completed survival output cannot be marked failed')
        buffer = io.BytesIO()
        np.savez(buffer, **error.arrays)
        payload = buffer.getvalue()
        record = self._failure_record(error, seed)
        record['failure_arrays'] = dict(filename='failure_arrays.npz',
            sha256=hashlib.sha256(payload).hexdigest(), arrays_sha256=array_hash(error.arrays))
        encoded = canonical(record).encode()
        self._current()
        artifacts.create_bytes(path / 'failure_arrays.npz', payload)
        artifacts.create_bytes(path / 'failure.json', encoded)
        self._current()


def compute_cell(config, seed, reference=None):
    """Compose a fixed attempt; reduced explicit configurations are unit fixtures only."""
    from .data import SurvivalConfig, generate_survival
    from .dslasso import DslassoConfig
    from .evaluation import evaluate_model
    from .pseudo import site_pseudo_diagnostics
    from .recovery_method import (OptimizationFailure, RecoveryPolicy,
                                  fit_recovery_dslasso, validate_recovery_diagnostics)
    validate_seed(config, seed)
    diagnostic = is_diagnostic(config['task'])
    repair = task_epoch(config['task']) >= 2
    stages = {}

    @contextmanager
    def measured(stage):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            stages[stage] = dict(wall_seconds=time.perf_counter() - wall,
                                 cpu_seconds=time.process_time() - cpu)

    with measured('data_preparation'):
        dataset = generate_survival(SurvivalConfig(**config['dgp']), seed, config['scenario'])
        result = dict(site_sizes=[len(site.rows) for site in dataset.sites],
            pseudo_diagnostics=[dict(site_id=site.site_id,
                **site_pseudo_diagnostics(site.rows, config['dgp']['tau'])) for site in dataset.sites],
            fingerprints=dataset_fingerprints(dataset))
        if reference is not None:
            validate_reference(result, reference)
    optimizer = DslassoConfig(**config['method']['optimizer'])
    policy = RecoveryPolicy(**config['method']['recovery_policy'])
    with measured('fit'):
        try:
            model, fit = fit_recovery_dslasso(dataset.sites, config['dgp']['tau'], seed=seed,
                config=optimizer, policy=policy,
                context=dict(scenario=config['scenario'], seed=seed, recipe=config['recovery_recipe']))
        except OptimizationFailure as error:
            if not diagnostic or repair or error.diagnostics.get('outcome') != 'iteration_limit':
                raise
            fit, arrays = error.diagnostics, error.arrays
        else:
            arrays = dict(intercept=np.asarray(model.intercept, dtype=np.float64),
                          coefficients=np.asarray(model.coefficients, dtype=np.float64))
        try:
            validate_recovery_diagnostics(fit, optimizer, policy, arrays, result['site_sizes'],
                require_success=None if diagnostic and not repair else True,
                required_schema=config.get('required_evidence_schema'))
        except Exception as error:
            raise FitValidationFailure(error, fit, arrays, 'post_fit_validation',
                dict(result=result, timing_stages=stages)) from error
    stage = 'result_assembly'
    try:
        if diagnostic:
            result.update(status='diagnostic_complete', outcome=fit['outcome'])
            result['fingerprints']['model' if fit['outcome'] == 'stationary' else 'arrays'] = array_hash(arrays)
        else:
            stage = 'evaluation'
            with measured('evaluation'):
                result['evaluation'] = evaluate_model(model.predict_eta, dataset,
                    repeats=config['evaluation']['repeats'], seed=seed)
            stage = 'result_assembly'
            result.update(status='complete')
            result['fingerprints']['model'] = array_hash(arrays)
        result.update(fit=fit,
            timing=dict(stages=stages, fit_includes_all_selection_and_refits=True, network_transport_measured=False),
            memory=dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, scope=MEMORY_SCOPE))
    except Exception as error:
        raise FitValidationFailure(error, fit, arrays, stage,
            dict(result=result, timing_stages=stages)) from error
    return result, arrays


def run_job(job):
    """Only called after the central adapter validates its complete running manifest."""
    from .recovery_acceptance import (check_current_diagnostic_gate, check_original_closeout,
                                      original_reference)
    parameters = validate_parameters(job['task'], job['parameters'])
    epoch = task_epoch(job['task'])
    provenance = artifacts.current_provenance()
    check_original_closeout()
    if epoch >= 2:
        from .recovery_acceptance import check_repair_closeout
        check_repair_closeout()
    if epoch == 3:
        from .recovery_acceptance import check_integration_reference, _integration_reference
        historical = check_integration_reference()
        if is_diagnostic(job['task']):
            _integration_reference(historical, job['parameters']['case'])
    prerequisite, selected = None, None
    if not is_diagnostic(job['task']):
        proof = check_current_diagnostic_gate() if epoch == 1 else check_current_diagnostic_gate(epoch=epoch)
        selected = proof['selected_recipe']
        prerequisite = dict(diagnostic_acceptance_sha256=digest(proof),
            provenance_sha256=provenance['provenance_sha256'], selected_recipe=selected)
    config = run_config(job['task'], parameters, recipe=selected)
    store = RecoveryStore(config, provenance, prerequisite=prerequisite)
    store.initialize()
    execution = {key: job[key] for key in (
        'job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in seed_sequence(job['task'], parameters):
        _require(artifacts.current_provenance() == provenance, 'source changed during recovery execution')
        if store.status(seed) == 'validated':
            continue
        store.start(execution, seed)
        result, arrays = None, None
        try:
            reference = original_reference(config['scenario'], seed)
            result, arrays = compute_cell(config, seed, reference=reference)
            result['identity'] = store.identity(seed)
            _require(artifacts.current_provenance() == provenance, 'source changed during recovery execution')
            store.complete(result, arrays, seed)
        except Exception as error:
            preserved = error
            if result is not None and arrays is not None and not isinstance(error, FitValidationFailure):
                preserved = FitValidationFailure(error, result['fit'], arrays,
                                                  'artifact_publication', evidence=result)
            try:
                store.fail(preserved, seed)
            except Exception:
                pass  # Preserve the original failure when a changed source also rejects recording.
            if preserved is not error:
                raise preserved from error
            raise
        _require(store.status(seed) == 'validated', 'recovery completion failed validation')
    return 0
