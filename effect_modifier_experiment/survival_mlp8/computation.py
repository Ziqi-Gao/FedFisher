"""Path-free composition of fresh MLP8 federated and pooled survival models."""
from contextlib import contextmanager
import resource
import time

import numpy as np

from ..survival.artifacts import array_hash, dataset_fingerprints
from ..survival.data import SurvivalConfig, generate_survival
from ..survival_pooled.evaluation import probability_predictions, summarize_predictions, compare_predictions
from ..survival_pooled.inputs import historical_binding, load_historical
from . import protocol
from .methods import MLP8Config, fit_federated, fit_pooled


def compute_cell(config, seed):
    """Fit four new models, checking paired data before any fitting/inference.

    The accepted linear artifact is only a data/reference witness. Its model is
    never evaluated here. Trainers receive observed hospital rows; independent
    test rows and simulation truth enter only the probability evaluator.
    """
    return _compute_cell(config, seed, protocol, discovery=False)


def compute_discovery_cell(config, seed):
    """Fit the same four MLP8 models, assess probability, then discover modifiers.

    V2 has a separate recipe and output namespace. Its five-repeat raw-covariate
    permutation assessment follows the original survival discovery evaluator;
    the accepted dsLasso fit remains a data witness and is never inferred.
    """
    from . import discovery_protocol
    return _compute_cell(config, seed, discovery_protocol, discovery=True)


def _compute_cell(config, seed, recipe, *, discovery):
    recipe.validate_seed(config, seed)
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
        anchors = historical_binding()
        reference = load_historical('linear', config['scenario'], seed)['dslasso']
    with measured('data_preparation'):
        dataset = generate_survival(SurvivalConfig(**config['dgp']), seed, config['scenario'])
        fingerprints = dataset_fingerprints(dataset)
        sizes = [len(site.rows) for site in dataset.sites]
        if any(reference['fingerprints'][key] != fingerprints[key] for key in fingerprints):
            raise ValueError('MLP8 data differ from the accepted survival data')
        if reference['result']['site_sizes'] != sizes:
            raise ValueError('MLP8 site sizes differ from the accepted survival data')

    models, fits, arrays = {}, {}, {}
    options = MLP8Config(**config['method']['optimizer'])
    with measured('fit'):
        for kind in ('diag', 'kfac'):
            tag = 'fedfisher_' + kind
            models[tag], fits[tag], fitted = fit_federated(dataset.sites, config['dgp']['tau'],
                seed, kind, options, config['public_initial_eta'])
            arrays.update({tag + '__fit__' + key: np.asarray(value) for key, value in fitted.items()})
        for scope in config['pseudo_scopes']:
            tag = 'pooled_' + scope
            models[tag], fits[tag], fitted = fit_pooled(dataset.sites, config['dgp']['tau'],
                seed, scope, options, config['public_initial_eta'])
            arrays.update({tag + '__fit__' + key: np.asarray(value) for key, value in fitted.items()})

    # Persist arm-wise predictions before subtraction. Later acceptance and
    # reporting recompute arithmetic from these arrays, without model inference.
    with measured('probability_evaluation'):
        predictions, metrics = {}, {}
        for tag, model in models.items():
            predictions[tag] = probability_predictions(model.predict_eta, dataset)
            metrics[tag] = summarize_predictions(predictions[tag], dataset)
            arrays.update({tag + '__prediction__' + key: np.asarray(value)
                           for key, value in predictions[tag].items()})
        comparisons = {tag: {scope: compare_predictions(predictions[tag], predictions['pooled_' + scope])
                              for scope in config['pseudo_scopes']}
                       for tag in ('fedfisher_diag', 'fedfisher_kfac')}
        arrays.update({'truth__' + key: np.asarray(getattr(dataset.evaluation_truth, key))
                       for key in ('propensity', 'eta0', 'eta1', 'survival0', 'survival1',
                                   'delta', 'modifier_mask')})
        arrays['factual_a'] = np.asarray(dataset.evaluation.a, dtype=np.float64)
        arrays['plot_x12'] = np.asarray(dataset.evaluation.x[:500, :2], dtype=np.float64)
    ranking = {}
    if discovery:
        from .discovery_evaluation import evaluate_discovery
        with measured('discovery_evaluation'):
            repeats = config['evaluation']['discovery']['repeats']
            ranking = {tag: evaluate_discovery(model.predict_eta, dataset, seed, repeats=repeats)
                       for tag, model in models.items()}
            # One complete common test matrix binds the original permutation
            # recipe across methods and scenarios. Acceptance reads saved
            # predictions and scores; it never re-runs neural inference.
            arrays['discovery__x'] = np.array(dataset.evaluation.x, dtype=np.float64, copy=True)
    result = dict(status='complete', fit=fits, prediction=metrics, comparisons=comparisons,
                reference=reference['binding'], historical_inputs=anchors,
                site_sizes=sizes, fingerprints=fingerprints,
                timing=dict(stages=stages, network_transport_measured=False,
                            fit_includes_all_selection_and_refits=True),
                memory=dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    scope='Linux process-lifetime high-water RSS, not allocation or cgroup peak'),
                prediction_arrays_sha256=array_hash(arrays))
    if discovery:
        result['discovery'] = ranking
    return result, arrays
