"""Immutable comparator results; acceptance reads saved evidence without inference."""
import math
import re

import numpy as np

from ..infrastructure.artifacts import canonical
from ..survival.artifacts import ArtifactStore as SurvivalStore, array_hash
from ..survival import evaluation as original
from ..survival_mlp8.discovery_evaluation import _summary
from ..survival_pooled.evaluation import summarize_arrays, compare_predictions
from ..survival_pooled.inputs import load_historical, selected_fingerprints
from . import inputs, protocol
from .fedavg import FedAvgConfig, validate_fit_report

TRUTH_FIELDS = ('propensity', 'eta0', 'eta1', 'survival0', 'survival1', 'delta', 'modifier_mask')
STAGES = ('historical_input_validation', 'data_preparation', 'fit', 'probability_evaluation', 'discovery_evaluation')
FIT_ORIGIN = dict(fedavg='new_training', dslasso='accepted_r3_model_replay')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_discovery(summary, predictions, x, truth_delta, mask, reference, repeats):
    """Bind unchanged ranking arithmetic to saved predictions and archived streams."""
    n, p = x.shape
    _summary(summary, p, n, repeats)
    _summary(reference, p, n, repeats)
    baseline = predictions['eta1'] - predictions['eta0']
    error = baseline - truth_delta
    squared = float(error @ error)
    require(summary['raw_x_sha256'] == reference['raw_x_sha256'] == original._array_hash(x)
            and summary['contrast_truth_sha256'] == reference['contrast_truth_sha256']
            == original._array_hash(truth_delta)
            and summary['permutation_sha256'] == reference['permutation_sha256']
            and summary['modifier_mask'] == mask.tolist(), 'comparator discovery inputs differ from accepted reference')
    require(np.isfinite(baseline).all() and math.isfinite(squared)
            and summary['estimated_contrast_sha256'] == original._array_hash(baseline)
            and summary['contrast_squared_error_sum'] == squared
            and summary['contrast_rmse'] == float(np.sqrt(squared / n)),
            'comparator discovery baseline differs from saved predictions')


def _duration(stage):
    require(isinstance(stage, dict) and set(stage) == {'wall_seconds', 'cpu_seconds'}
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in stage.values()),
            'invalid comparator stage timing')


def _validate(result, arrays, identity, config):
    canonical(result)
    expected = {'identity', 'status', 'fit', 'fit_origin', 'prediction', 'discovery', 'comparisons',
                'references', 'historical_inputs', 'site_sizes', 'fingerprints', 'timing', 'memory',
                'prediction_arrays_sha256'}
    require(isinstance(result, dict) and set(result) == expected and result['identity'] == identity
            and result['status'] == 'complete', 'comparator result identity/inventory mismatch')
    protocol.validate_seed(config, identity['seed'])
    require(isinstance(arrays, dict) and arrays and all(isinstance(key, str) and isinstance(value, np.ndarray)
            and not value.dtype.hasobject and np.isfinite(value).all() for key, value in arrays.items()),
            'invalid comparator saved arrays')
    require(result['prediction_arrays_sha256'] == array_hash(arrays), 'comparator array fingerprint mismatch')
    require(result['historical_inputs'] == inputs.historical_binding(), 'comparator acceptance anchor mismatch')
    tags = set(protocol.TAGS)
    require(all(isinstance(result[name], dict) and set(result[name]) == tags
                for name in ('fit', 'prediction', 'discovery', 'comparisons'))
            and result['fit_origin'] == FIT_ORIGIN, 'comparator method coverage/origin mismatch')
    references = result['references']
    require(isinstance(references, dict) and set(references) == {'dslasso', 'mlp8'},
            'comparator reference inventory mismatch')
    receipt = references['dslasso']
    require(type(receipt['identity']['seed']) is int and receipt['identity']['seed'] == identity['seed']
            and receipt['identity']['scenario'] == config['scenario']
            and receipt['identity']['cell'] == 'dslasso', 'comparator dsLasso reference identity mismatch')
    fingerprints = selected_fingerprints(receipt)  # Strict selected bytes and actual producing attempt.
    linear = load_historical('linear', config['scenario'], identity['seed'])['dslasso']
    baseline = inputs.load_mlp8_reference(config['scenario'], identity['seed'])
    require(references == dict(dslasso=linear['binding'], mlp8=baseline['binding']),
            'comparator references differ from accepted units')
    require(result['fingerprints'] == fingerprints == baseline['fingerprints']
            and all(linear['fingerprints'][key] == value for key, value in fingerprints.items())
            and set(fingerprints) == {'training', 'evaluation', 'evaluation_truth'}
            and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in fingerprints.values()),
            'comparator data differ from accepted cohorts')
    dgp, sizes = config['dgp'], result['site_sizes']
    require(isinstance(sizes, list) and len(sizes) == dgp['num_sites']
            and all(type(n) is int and dgp['site_min_samples'] <= n <= dgp['site_max_samples'] for n in sizes)
            and sizes == linear['result']['site_sizes'] == baseline['result']['site_sizes'],
            'comparator site sizes differ from accepted cohorts')
    n, p = dgp['num_test'], dgp['covariate_dim']
    truth = {key: arrays['truth__' + key] for key in TRUTH_FIELDS}
    require(array_hash(truth) == fingerprints['evaluation_truth'], 'comparator oracle fingerprint mismatch')
    shared = ('factual_a', 'discovery__x', *('truth__' + key for key in TRUTH_FIELDS))
    for name in shared:
        actual, accepted = arrays[name], baseline['arrays'][name]
        require(actual.dtype == accepted.dtype and actual.shape == accepted.shape
                and actual.tobytes() == accepted.tobytes(), 'comparator saved evaluator input differs: ' + name)
    require(arrays['discovery__x'].shape == (n, p) and arrays['discovery__x'].dtype == np.float64
            and arrays['factual_a'].shape == (n,) and arrays['factual_a'].dtype == np.float64
            and np.isin(arrays['factual_a'], (0., 1.)).all(), 'invalid comparator evaluator arrays')
    allowed = set(shared)
    probability_truth = {key: truth[key] for key in ('eta0', 'eta1', 'survival0', 'survival1')}
    for tag in protocol.TAGS:
        prediction = {key: arrays[tag + '__prediction__' + key] for key in ('eta0', 'eta1', 's0', 's1')}
        require(all(v.shape == (n,) and v.dtype == np.float64 for v in prediction.values()),
                'invalid comparator prediction shape/type')
        require(result['prediction'][tag] == summarize_arrays(prediction, probability_truth, arrays['factual_a']),
                'comparator probability metrics differ from saved arrays')
        validate_discovery(result['discovery'][tag], prediction, arrays['discovery__x'], truth['delta'],
                           truth['modifier_mask'], linear['result']['evaluation'], config['evaluation']['discovery']['repeats'])
        require(set(result['comparisons'][tag]) == set(config['pseudo_scopes']), 'missing comparator pooled gap')
        for scope in config['pseudo_scopes']:
            pooled = {key: baseline['arrays']['pooled_' + scope + '__prediction__' + key] for key in prediction}
            require(result['comparisons'][tag][scope] == compare_predictions(prediction, pooled),
                    'comparator pooled gap differs from saved predictions')
        allowed |= {tag + '__prediction__' + key for key in prediction}
        prefix = tag + '__fit__'
        fitted = {name[len(prefix):]: value for name, value in arrays.items() if name.startswith(prefix)}
        require(fitted, 'missing comparator fitted arrays')
        allowed |= {prefix + name for name in fitted}
        if tag == 'fedavg':
            fit = result['fit'][tag]
            require(type(fit['seed']) is int and fit['seed'] == identity['seed']
                    and fit['input_dim'] == 1 + 2 * p and fit['initial_eta'] == config['public_initial_eta']
                    and fit['tau'] == dgp['tau'],
                    'comparator FedAvg fit identity mismatch')
            validate_fit_report(fit, fitted, FedAvgConfig(**config['method']['optimizer']), sizes)
            require(fitted['initial_vector'].tobytes() == baseline['arrays']['fedfisher_diag__fit__initial_vector'].tobytes()
                    and all(site['training_pseudo_sha256'] == baseline['result']['fit']['fedfisher_diag']['local'][i]['training_pseudo_sha256']
                            for i, site in enumerate(fit['sites'])), 'FedAvg public initialization/pseudo-values differ from accepted MLP8')
        else:
            require(set(fitted) == {'intercept', 'coefficients'} and fitted['intercept'].shape == ()
                    and fitted['coefficients'].shape == (1 + 2 * p,)
                    and all(v.dtype == np.float64 for v in fitted.values()), 'invalid dsLasso replay model')
            require(fitted['intercept'].tobytes() == np.asarray(linear['model'].intercept, dtype=np.float64).tobytes()
                    and fitted['coefficients'].tobytes() == linear['model'].coefficients.tobytes()
                    and result['fit'][tag] == linear['result']['fit']
                    and result['discovery'][tag] == linear['result']['evaluation'], 'dsLasso replay differs from accepted model')
    require(set(arrays) == allowed, 'comparator array inventory mismatch')
    timing = result['timing']
    require(set(timing) == {'stages', 'network_transport_measured', 'fit_includes_all_selection_and_refits',
                           'fit_scope', 'reused_dslasso_fit'} and set(timing['stages']) == set(STAGES)
            and timing['network_transport_measured'] is False and timing['fit_includes_all_selection_and_refits'] is True
            and timing['fit_scope'] == 'new_fedavg_training_only'
            and timing['reused_dslasso_fit'] == linear['result']['timing']['stages']['fit'],
            'comparator timing scope/origin mismatch')
    for stage in (*timing['stages'].values(), timing['reused_dslasso_fit']):
        _duration(stage)
    require(timing['stages']['fit']['wall_seconds'] + 1e-9 >= result['fit']['fedavg']['total_fit_seconds'],
            'comparator timing omits FedAvg fitting cost')
    memory = result['memory']
    require(set(memory) == {'process_peak_rss_kib', 'scope'} and type(memory['process_peak_rss_kib']) is int
            and memory['process_peak_rss_kib'] > 0 and memory['scope'] ==
            'Linux process-lifetime high-water RSS, not allocation or cgroup peak', 'invalid comparator memory scope')
    return True


def validate_result(result, arrays, identity, config):
    """Validate hashes, provenance, saved arithmetic and numerical fit evidence only."""
    try:
        return _validate(result, arrays, identity, config)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError('malformed survival comparator result') from error


class ArtifactStore(SurvivalStore):
    protocol_name = protocol.PROTOCOL

    def _validate_config(self):
        require(self.config == protocol.run_config(self.config['task'], dict(scenario=self.config['scenario'], seed=0)),
                'survival comparator configuration is not the fixed pilot recipe')

    def _validate_prerequisite(self, prerequisite):
        require(prerequisite is None, 'comparator pilot cannot bind formal prerequisites')

    def _validate_seed(self, seed):
        return protocol.validate_seed(self.config, seed)

    def _validate_result(self, result, arrays, identity):
        validate_result(result, arrays, identity, self.config)

    def _failure_record(self, error, seed):
        record = super()._failure_record(error, seed)
        if getattr(error, 'diagnostics', None) is not None:
            record['optimizer_diagnostics'] = error.diagnostics
        return record
