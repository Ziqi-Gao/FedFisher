"""Additive FedAvg fitting and accepted dsLasso replay on paired survival data."""
from contextlib import contextmanager
from copy import deepcopy
import resource
import time

import numpy as np

from ..survival.artifacts import array_hash, dataset_fingerprints
from ..survival.data import SurvivalConfig, generate_survival
from ..survival.evaluation import evaluate_model
from ..survival_pooled.evaluation import probability_predictions, summarize_predictions, compare_predictions
from ..survival_pooled.inputs import load_historical
from . import inputs, protocol
from .fedavg import FedAvgConfig, fit_fedavg


def compute_cell(config, seed):
    """Train FedAvg only; preserve each historical model's actual provenance.

    Hospital rows alone enter fitting. The independently generated common test
    population is fingerprint-checked against both accepted cohorts before
    training. dsLasso's coefficients and discovery scores are reused unchanged;
    its missing arm-wise probabilities are evaluated before contrast arithmetic.
    Existing MLP8 predictions are read from accepted arrays, never inferred.
    """
    protocol.validate_seed(config, seed)
    stages = {}

    @contextmanager
    def measured(name):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            stages[name] = dict(wall_seconds=time.perf_counter() - wall,
                                cpu_seconds=time.process_time() - cpu)

    with measured('historical_input_validation'):
        anchors = inputs.historical_binding()
        linear = load_historical('linear', config['scenario'], seed)['dslasso']
        reference = inputs.load_mlp8_reference(config['scenario'], seed)
    with measured('data_preparation'):
        dataset = generate_survival(SurvivalConfig(**config['dgp']), seed, config['scenario'])
        fingerprints = dataset_fingerprints(dataset)
        sizes = [len(site.rows) for site in dataset.sites]
        for accepted in (linear, reference):
            if any(accepted['fingerprints'][key] != value for key, value in fingerprints.items()):
                raise ValueError('comparator data differ from the accepted survival data')
            if accepted['result']['site_sizes'] != sizes:
                raise ValueError('comparator site sizes differ from the accepted survival data')

    with measured('fit'):
        model, fit, fitted = fit_fedavg(dataset.sites, config['dgp']['tau'], seed,
            FedAvgConfig(**config['method']['optimizer']), config['public_initial_eta'])
        if fitted['initial_vector'].tobytes() != reference['arrays']['fedfisher_diag__fit__initial_vector'].tobytes():
            raise ValueError('FedAvg public initialization differs from accepted MLP8')
        if any(site['training_pseudo_sha256'] != reference['result']['fit']['fedfisher_diag']['local'][i]['training_pseudo_sha256']
               for i, site in enumerate(fit['sites'])):
            raise ValueError('FedAvg local pseudo-values differ from accepted MLP8')
    models = dict(fedavg=model, dslasso=linear['model'])
    arrays = {'fedavg__fit__' + key: np.asarray(value) for key, value in fitted.items()}
    arrays.update(dslasso__fit__intercept=np.asarray(linear['model'].intercept, dtype=np.float64),
                  dslasso__fit__coefficients=np.asarray(linear['model'].coefficients, dtype=np.float64).copy())
    # Persist probabilities before computing any new treatment contrast.
    with measured('probability_evaluation'):
        predictions, metrics, comparisons = {}, {}, {}
        for tag, predictor in models.items():
            predictions[tag] = probability_predictions(predictor.predict_eta, dataset)
            metrics[tag] = summarize_predictions(predictions[tag], dataset)
            arrays.update({tag + '__prediction__' + key: value for key, value in predictions[tag].items()})
            comparisons[tag] = {}
            for scope in config['pseudo_scopes']:
                control = {key: reference['arrays']['pooled_' + scope + '__prediction__' + key]
                           for key in ('eta0', 'eta1', 's0', 's1')}
                comparisons[tag][scope] = compare_predictions(predictions[tag], control)
        arrays.update({'truth__' + key: np.asarray(getattr(dataset.evaluation_truth, key))
                      for key in ('propensity', 'eta0', 'eta1', 'survival0', 'survival1', 'delta', 'modifier_mask')})
        arrays['factual_a'] = np.asarray(dataset.evaluation.a, dtype=np.float64)
        arrays['discovery__x'] = np.array(dataset.evaluation.x, dtype=np.float64, copy=True)
    from .artifacts import validate_discovery
    with measured('discovery_evaluation'):
        repeats = config['evaluation']['discovery']['repeats']
        archived = linear['result']['evaluation']
        validate_discovery(archived, predictions['dslasso'], dataset.evaluation.x,
            dataset.evaluation_truth.delta, dataset.evaluation_truth.modifier_mask, archived, repeats)
        discovery = dict(dslasso=deepcopy(archived),
            fedavg=evaluate_model(model.predict_eta, dataset, repeats=repeats, seed=seed))
    result = dict(status='complete', fit=dict(fedavg=fit, dslasso=deepcopy(linear['result']['fit'])),
        fit_origin=dict(fedavg='new_training', dslasso='accepted_r3_model_replay'),
        prediction=metrics, discovery=discovery, comparisons=comparisons,
        references=dict(dslasso=linear['binding'], mlp8=reference['binding']),
        historical_inputs=anchors, site_sizes=sizes, fingerprints=fingerprints,
        timing=dict(stages=stages, network_transport_measured=False,
                    fit_includes_all_selection_and_refits=True, fit_scope='new_fedavg_training_only',
                    reused_dslasso_fit=deepcopy(linear['result']['timing']['stages']['fit'])),
        memory=dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope='Linux process-lifetime high-water RSS, not allocation or cgroup peak'),
        prediction_arrays_sha256=array_hash(arrays))
    return result, arrays
