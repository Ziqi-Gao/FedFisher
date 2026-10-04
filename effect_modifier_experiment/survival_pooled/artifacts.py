"""Reuse immutable survival storage with a separate result contract and namespace."""
import math
import re

import numpy as np

from ..infrastructure.artifacts import canonical
from ..survival.artifacts import ArtifactStore as SurvivalStore, array_hash
from . import protocol


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_number(value, nonnegative=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def validate_result(result, arrays, identity, config):
    """Recompute probability summaries from saved arrays without model inference."""
    from .evaluation import summarize_arrays, compare_predictions
    canonical(result)
    expected = {'identity', 'status', 'fit', 'prediction', 'comparisons', 'historical',
                'historical_inputs', 'site_sizes', 'fingerprints', 'timing', 'memory',
                'prediction_arrays_sha256'}
    require(isinstance(result, dict) and set(result) == expected
            and result['identity'] == identity and result['status'] == 'complete',
            'pooled survival result identity/inventory mismatch')
    require(isinstance(arrays, dict) and arrays and all(isinstance(v, np.ndarray)
            and not v.dtype.hasobject and np.isfinite(v).all() for v in arrays.values()),
            'invalid pooled prediction/model arrays')
    require(result['prediction_arrays_sha256'] == array_hash(arrays), 'pooled saved-array fingerprint mismatch')
    sizes, dgp = result['site_sizes'], config['dgp']
    require(isinstance(sizes, list) and len(sizes) == dgp['num_sites']
            and all(type(n) is int and dgp['site_min_samples'] <= n <= dgp['site_max_samples'] for n in sizes),
            'pooled survival site size mismatch')
    tags = set(protocol.historical_tags(config['family']))
    scopes = config['pseudo_scopes']
    all_tags = tags | {'pooled_' + scope for scope in scopes}
    require(set(result['historical']) == tags and set(result['comparisons']) == tags
            and set(result['prediction']) == all_tags and set(result['fit']) == set(scopes),
            'pooled survival comparison coverage mismatch')
    from .inputs import historical_binding, validate_selected_binding
    require(result['historical_inputs'] == historical_binding(), 'historical acceptance anchor mismatch')
    for tag, receipt in result['historical'].items():
        validate_selected_binding(receipt)
        require(receipt['identity']['seed'] == identity['seed']
                and receipt['identity']['scenario'] == config['scenario']
                and receipt['identity']['cell'] == (tag if tag == 'dslasso' else tag + '_' + config['family']),
                'historical model selected for a different scientific unit')
    truth = {key: arrays['truth__' + key] for key in ('eta0', 'eta1', 'survival0', 'survival1')}
    n = dgp['num_test']
    a = arrays['factual_a']
    require(a.shape == (n,) and a.dtype == np.float64 and np.isin(a, (0., 1.)).all()
            and all(v.shape == (n,) and v.dtype == np.float64 for v in truth.values()),
            'invalid pooled probability truth/treatment arrays')
    from ..survival.links import survival_mean
    require(np.array_equal(truth['survival0'], survival_mean(truth['eta0']))
            and np.array_equal(truth['survival1'], survival_mean(truth['eta1'])),
            'saved survival truth and link disagree')
    all_truth = {key: arrays['truth__' + key] for key in (*truth, 'propensity', 'delta', 'modifier_mask')}
    require(all_truth['propensity'].shape == (n,) and all_truth['propensity'].dtype == np.float64
            and ((all_truth['propensity'] > 0) & (all_truth['propensity'] < 1)).all()
            and all_truth['delta'].shape == (n,) and all_truth['delta'].dtype == np.float64
            and np.array_equal(truth['eta1'], truth['eta0'] + all_truth['delta'])
            and all_truth['modifier_mask'].shape == (dgp['covariate_dim'],)
            and all_truth['modifier_mask'].dtype == np.bool_
            and np.array_equal(all_truth['modifier_mask'], np.arange(dgp['covariate_dim']) < 2),
            'invalid saved evaluator-only oracle truth')
    require(array_hash(all_truth) == result['fingerprints']['evaluation_truth'],
            'saved truth differs from accepted evaluation truth fingerprint')
    require(arrays['plot_x12'].shape == (min(500, n), 2) and arrays['plot_x12'].dtype == np.float64,
            'invalid pooled deterministic plot sample')
    predictions = {}
    allowed = {'truth__' + key for key in all_truth} | {'factual_a', 'plot_x12'}
    for tag in all_tags:
        prediction = {key: arrays[tag + '__prediction__' + key] for key in ('eta0', 'eta1', 's0', 's1')}
        require(all(v.shape == (n,) and v.dtype == np.float64 for v in prediction.values()),
                'pooled probability prediction dimensions/type mismatch')
        require(np.array_equal(prediction['s0'], survival_mean(prediction['eta0']))
                and np.array_equal(prediction['s1'], survival_mean(prediction['eta1'])),
                'saved predicted survival and link disagree')
        require(result['prediction'][tag] == summarize_arrays(prediction, truth, a),
                'saved pooled probability metrics differ from prediction arrays')
        predictions[tag] = prediction
        allowed |= {tag + '__prediction__' + key for key in prediction}
    for tag in tags:
        require(set(result['comparisons'][tag]) == set(scopes), 'missing pooled probability gap')
        for scope in scopes:
            require(result['comparisons'][tag][scope] == compare_predictions(
                predictions[tag], predictions['pooled_' + scope]), 'pooled prediction gap arithmetic mismatch')
    for scope in scopes:
        fit = result['fit'][scope]
        require(fit.get('pseudo_scope') == scope and fit.get('family') == config['family'],
                'pooled fit recipe identity mismatch')
        require(fit.get('seed') == identity['seed'] and type(fit.get('seed')) is int
                and fit.get('input_dim') == 1 + 2 * dgp['covariate_dim'], 'pooled fit seed/design mismatch')
        if config['family'] != 'linear':
            require(fit.get('initial_eta') == config['public_initial_eta'], 'pooled public initialization mismatch')
        prefix = 'pooled_' + scope + '__fit__'
        fitted = {name[len(prefix):]: value for name, value in arrays.items() if name.startswith(prefix)}
        require(fitted and all(value.dtype == np.float64 for value in fitted.values()),
                'missing or invalid float64 pooled fitted model representation')
        allowed |= {prefix + name for name in fitted}
        # A model is retained with predictions; its hash must bind the actual
        # parameters, while each fit's own endpoint checks distinguish a fixed
        # neural epoch budget from first-order linear stationarity.
        from .methods import validate_pooled_fit
        validate_pooled_fit(fit, fitted, config['family'], scope, config['method'], sizes)
    require(set(arrays) == allowed, 'unexpected pooled scientific arrays')
    require(set(result['fingerprints']) == {'training', 'evaluation', 'evaluation_truth'}
            and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
                    for v in result['fingerprints'].values()),
            'invalid pooled dataset fingerprints')
    for receipt in result['historical'].values():
        from .inputs import selected_fingerprints
        fingerprints = selected_fingerprints(receipt)
        require(all(fingerprints[key] == result['fingerprints'][key] for key in result['fingerprints']),
                'saved pooled data differ from selected historical FL data')
    timing = result['timing']
    require(set(timing) == {'stages', 'network_transport_measured', 'fit_includes_all_selection_and_refits'}
            and timing['network_transport_measured'] is False
            and timing['fit_includes_all_selection_and_refits'] is True
            and set(timing['stages']) == {'historical_input_validation', 'data_preparation', 'fit', 'probability_evaluation'},
            'pooled timing scope mismatch')
    for record in timing['stages'].values():
        require(set(record) == {'wall_seconds', 'cpu_seconds'}
                and all(_finite_number(v, True) and v > 0 for v in record.values()), 'invalid pooled timing record')
    require(sum(row['total_fit_seconds'] for row in result['fit'].values())
            <= timing['stages']['fit']['wall_seconds'] + 1e-6, 'pooled fit timing excludes fitting work')
    require(result['memory'].keys() == {'process_peak_rss_kib', 'scope'}
            and type(result['memory']['process_peak_rss_kib']) is int
            and result['memory']['process_peak_rss_kib'] > 0
            and result['memory']['scope'] == 'Linux process-lifetime high-water RSS, not allocation or cgroup peak',
            'pooled memory observation scope mismatch')
    return True


class ArtifactStore(SurvivalStore):
    """Independent pooled namespace; preserve the original survival contracts."""
    protocol_name = 'survival-pooled-prediction-v1'

    def _validate_config(self):
        parameters = dict(family=self.config['family'], scenario=self.config['scenario'])
        if self.config['task'] == protocol.FORMAL_TASK:
            parameters.update(shard_index=0, shard_count=16)
        require(self.config == protocol.run_config(self.config['task'], parameters),
                'pooled survival configuration is not the fixed recipe')

    def _validate_seed(self, seed):
        return protocol.validate_seed(self.config, seed)

    def _validate_prerequisite(self, prerequisite):
        if self.config['task'] == protocol.FORMAL_TASK:
            require(isinstance(prerequisite, dict) and set(prerequisite) == {
                'pilot_acceptance_sha256', 'provenance_sha256'}
                and isinstance(prerequisite['pilot_acceptance_sha256'], str)
                and len(prerequisite['pilot_acceptance_sha256']) == 64
                and prerequisite['provenance_sha256'] == self.provenance['provenance_sha256'],
                'formal pooled survival requires same-source pilot acceptance')
        else:
            require(prerequisite is None, 'pooled pilot cannot use formal prerequisites')

    def _validate_result(self, result, arrays, identity):
        validate_result(result, arrays, identity, self.config)

    def _failure_record(self, error, seed):
        record = super()._failure_record(error, seed)
        diagnostics = getattr(error, 'diagnostics', None)
        if diagnostics is not None:
            # Failed first-order checks are scientific noncompletion. Preserve
            # their endpoints rather than losing them in an exception message.
            record['optimizer_diagnostics'] = diagnostics
        return record
