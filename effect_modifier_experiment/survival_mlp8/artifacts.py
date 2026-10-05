"""Independent MLP8 manifests and saved-only scientific completion checks."""
import math
import re

import numpy as np

from ..infrastructure.artifacts import canonical, read_json
from ..survival.artifacts import ArtifactStore as SurvivalStore, array_hash
from . import protocol

FEDERATED_TAGS = ('fedfisher_diag', 'fedfisher_kfac')
METHOD_TAGS = (*FEDERATED_TAGS, 'pooled_site_local_pv', 'pooled_pooled_km_pv')
TRUTH_FIELDS = ('propensity', 'eta0', 'eta1', 'survival0', 'survival1', 'delta', 'modifier_mask')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_result(result, arrays, identity, config):
    """Check every persisted claim using arrays and immutable historical evidence."""
    from ..survival.links import survival_mean
    from ..survival_pooled.evaluation import summarize_arrays, compare_predictions
    from ..survival_pooled.inputs import historical_binding, selected_fingerprints
    from .methods import MLP8Config, validate_federated_fit, validate_pooled_fit
    canonical(result)
    expected = {'identity', 'status', 'fit', 'prediction', 'comparisons', 'reference',
                'historical_inputs', 'site_sizes', 'fingerprints', 'timing', 'memory',
                'prediction_arrays_sha256'}
    require(isinstance(result, dict) and set(result) == expected and result['identity'] == identity
            and result['status'] == 'complete', 'MLP8 result identity/inventory mismatch')
    protocol.validate_seed(config, identity['seed'])
    require(isinstance(arrays, dict) and arrays and all(isinstance(name, str)
            and isinstance(value, np.ndarray) and not value.dtype.hasobject
            and np.isfinite(value).all() for name, value in arrays.items()), 'invalid MLP8 saved arrays')
    require(result['prediction_arrays_sha256'] == array_hash(arrays), 'MLP8 array fingerprint mismatch')
    dgp, sizes = config['dgp'], result['site_sizes']
    require(isinstance(sizes, list) and len(sizes) == dgp['num_sites']
            and all(type(n) is int and dgp['site_min_samples'] <= n <= dgp['site_max_samples']
                    for n in sizes), 'MLP8 site size mismatch')
    require(set(result['fit']) == set(METHOD_TAGS) and set(result['prediction']) == set(METHOD_TAGS)
            and set(result['comparisons']) == set(FEDERATED_TAGS), 'MLP8 method coverage mismatch')
    require(result['historical_inputs'] == historical_binding(), 'MLP8 historical acceptance anchor mismatch')
    receipt = result['reference']
    require(type(receipt['identity']['seed']) is int and receipt['identity']['seed'] == identity['seed']
            and receipt['identity']['scenario'] == config['scenario']
            and receipt['identity']['cell'] == 'dslasso', 'MLP8 reference belongs to another data unit')
    historical_fingerprints = selected_fingerprints(receipt)  # Includes selected receipt/producer checks.
    require(read_json(receipt['artifacts']['result.json']['path'])['site_sizes'] == sizes,
            'MLP8 site sizes differ from historical reference')
    fingerprints = result['fingerprints']
    require(isinstance(fingerprints, dict) and set(fingerprints) == {'training', 'evaluation', 'evaluation_truth'}
            and all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                    for value in fingerprints.values()), 'invalid MLP8 dataset fingerprints')
    require(fingerprints == historical_fingerprints, 'MLP8 data differ from historical reference')

    n = dgp['num_test']
    truth = {key: arrays['truth__' + key] for key in TRUTH_FIELDS}
    require(all(value.shape == (n,) and value.dtype == np.float64
                for key, value in truth.items() if key != 'modifier_mask')
            and truth['modifier_mask'].shape == (dgp['covariate_dim'],)
            and truth['modifier_mask'].dtype == np.bool_
            and np.array_equal(truth['modifier_mask'], np.arange(dgp['covariate_dim']) < 2)
            and ((truth['propensity'] > 0) & (truth['propensity'] < 1)).all()
            and np.array_equal(truth['eta1'], truth['eta0'] + truth['delta']), 'invalid MLP8 oracle arrays')
    require(np.array_equal(truth['survival0'], survival_mean(truth['eta0']))
            and np.array_equal(truth['survival1'], survival_mean(truth['eta1'])), 'MLP8 oracle link mismatch')
    require(array_hash(truth) == fingerprints['evaluation_truth'], 'MLP8 oracle fingerprint mismatch')
    treatment = arrays['factual_a']
    require(treatment.shape == (n,) and treatment.dtype == np.float64
            and np.isin(treatment, (0., 1.)).all(), 'invalid MLP8 factual treatment')
    require(arrays['plot_x12'].shape == (min(500, n), 2) and arrays['plot_x12'].dtype == np.float64,
            'invalid MLP8 deterministic plot sample')
    probability_truth = {key: truth[key] for key in ('eta0', 'eta1', 'survival0', 'survival1')}
    allowed = {'truth__' + key for key in TRUTH_FIELDS} | {'factual_a', 'plot_x12'}
    predictions, fitted_models = {}, {}
    options = MLP8Config(**config['method']['optimizer'])
    for tag in METHOD_TAGS:
        prediction = {key: arrays[tag + '__prediction__' + key] for key in ('eta0', 'eta1', 's0', 's1')}
        require(all(value.shape == (n,) and value.dtype == np.float64 for value in prediction.values()),
                'invalid MLP8 probability prediction shape/type')
        require(np.array_equal(prediction['s0'], survival_mean(prediction['eta0']))
                and np.array_equal(prediction['s1'], survival_mean(prediction['eta1'])),
                'MLP8 probability prediction link mismatch')
        require(result['prediction'][tag] == summarize_arrays(prediction, probability_truth, treatment),
                'MLP8 probability metrics differ from saved arrays')
        predictions[tag] = prediction
        allowed |= {tag + '__prediction__' + key for key in prediction}
        report = result['fit'][tag]
        require(type(report.get('seed')) is int and report['seed'] == identity['seed']
                and type(report.get('input_dim')) is int and report['input_dim'] == 1 + 2 * dgp['covariate_dim']
                and report.get('initial_eta') == config['public_initial_eta'], 'MLP8 fit seed/design/initialization mismatch')
        prefix = tag + '__fit__'
        fitted = {name[len(prefix):]: value for name, value in arrays.items() if name.startswith(prefix)}
        require(fitted and all(value.dtype == np.float64 for value in fitted.values()),
                'invalid MLP8 fitted model representation')
        if tag in FEDERATED_TAGS:
            validate_federated_fit(report, fitted, tag.removeprefix('fedfisher_'), options, sizes)
        else:
            validate_pooled_fit(report, fitted, tag.removeprefix('pooled_'), options, sizes)
        fitted_models[tag] = fitted
        allowed |= {prefix + name for name in fitted}
        require(_positive(report['total_fit_seconds']), 'invalid MLP8 complete fit cost')

    # Curvature selection must not alter SGD random streams or local solutions.
    # Pair actual bytes, including signed zeros, rather than merely equal scores.
    for name in ('initial_vector', *('local_%d__vector' % index for index in range(len(sizes)))):
        left, right = (fitted_models[tag][name] for tag in FEDERATED_TAGS)
        require(left.shape == right.shape and left.dtype == right.dtype
                and left.tobytes() == right.tobytes(), 'MLP8 Diag/KFAC local fitting mismatch')
    require(all(result['fit']['fedfisher_diag']['local'][index]['training_pseudo_sha256']
                == result['fit']['fedfisher_kfac']['local'][index]['training_pseudo_sha256']
                for index in range(len(sizes))), 'MLP8 Diag/KFAC pseudo-response mismatch')
    initial = fitted_models['fedfisher_diag']['initial_vector']
    require(all(fitted_models[tag]['initial_vector'].tobytes() == initial.tobytes() for tag in METHOD_TAGS),
            'MLP8 pooled/federated public initialization mismatch')
    for tag in FEDERATED_TAGS:
        require(set(result['comparisons'][tag]) == set(config['pseudo_scopes']), 'missing MLP8 pooled gap')
        for scope in config['pseudo_scopes']:
            require(result['comparisons'][tag][scope] == compare_predictions(
                predictions[tag], predictions['pooled_' + scope]), 'MLP8 pooled gap arithmetic mismatch')
    require(set(arrays) == allowed, 'unexpected MLP8 scientific arrays')
    timing = result['timing']
    require(set(timing) == {'stages', 'network_transport_measured', 'fit_includes_all_selection_and_refits'}
            and timing['network_transport_measured'] is False and timing['fit_includes_all_selection_and_refits'] is True
            and set(timing['stages']) == set(config['timing']['stages']), 'MLP8 timing scope mismatch')
    for record in timing['stages'].values():
        require(set(record) == {'wall_seconds', 'cpu_seconds'} and all(_positive(value) for value in record.values()),
                'invalid MLP8 stage timing')
    require(sum(report['total_fit_seconds'] for report in result['fit'].values())
            <= timing['stages']['fit']['wall_seconds'] + 1e-6, 'MLP8 fit timing excludes fitting work')
    require(set(result['memory']) == {'process_peak_rss_kib', 'scope'}
            and type(result['memory']['process_peak_rss_kib']) is int and result['memory']['process_peak_rss_kib'] > 0
            and result['memory']['scope'] == 'Linux process-lifetime high-water RSS, not allocation or cgroup peak',
            'MLP8 memory observation scope mismatch')
    return True


def validate_result(result, arrays, identity, config):
    """Semantic acceptance performs no simulation, fitting or model inference."""
    try:
        return _validate_result(result, arrays, identity, config)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('malformed MLP8 survival result') from error


class ArtifactStore(SurvivalStore):
    """Reuse immutable publication without inheriting historical model semantics."""
    protocol_name = 'survival-mlp8-prediction-v1'

    def _validate_config(self):
        parameters = dict(scenario=self.config['scenario'])
        if self.config['task'] == protocol.FORMAL_TASK:
            parameters.update(shard_index=0, shard_count=16)
        require(self.config == protocol.run_config(self.config['task'], parameters),
                'MLP8 survival configuration is not the fixed recipe')

    def _validate_seed(self, seed):
        return protocol.validate_seed(self.config, seed)

    def _validate_prerequisite(self, prerequisite):
        if self.config['task'] == protocol.FORMAL_TASK:
            require(isinstance(prerequisite, dict) and set(prerequisite) == {
                'pilot_acceptance_sha256', 'provenance_sha256'}
                and isinstance(prerequisite['pilot_acceptance_sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', prerequisite['pilot_acceptance_sha256'])
                and prerequisite['provenance_sha256'] == self.provenance['provenance_sha256'],
                'formal MLP8 survival requires same-source pilot acceptance')
        else:
            require(prerequisite is None, 'MLP8 pilot cannot use formal prerequisites')

    def _validate_result(self, result, arrays, identity):
        try:
            validate_result(result, arrays, identity, self.config)
        except ValueError as error:
            # Returned numerical endpoints can fail stationarity/publication.
            # Keep their finite reports in failure.json, without a done marker.
            if isinstance(result, dict) and isinstance(result.get('fit'), dict):
                try:
                    canonical(result['fit'])
                except (ValueError, TypeError):
                    pass
                else:
                    if getattr(error, 'diagnostics', None) is None:
                        error.diagnostics = result['fit']
            raise

    def _failure_record(self, error, seed):
        record = super()._failure_record(error, seed)
        if getattr(error, 'diagnostics', None) is not None:
            record['optimizer_diagnostics'] = error.diagnostics
        return record
