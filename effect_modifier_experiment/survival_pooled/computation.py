"""Scientific composition; the runner owns scheduling and artifact publication."""
from contextlib import contextmanager
import resource
import time

import numpy as np

from ..survival.artifacts import array_hash, dataset_fingerprints
from ..survival.data import SurvivalConfig, generate_survival
from . import protocol
from .evaluation import probability_predictions, summarize_predictions, compare_predictions
from .inputs import historical_binding, load_historical
from .methods import fit_pooled


def compute_cell(config, seed):
    """Fit two pooled controls and evaluate accepted FL models on the same data.

    Historical artifacts are validated before any data generation or inference.
    The generator exposes oracle truth solely to the evaluator. Fitting receives
    the observed site rows and prespecified settings, never the test outcomes.
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
        reference = historical_binding()
        historical = load_historical(config['family'], config['scenario'], seed)
    with measured('data_preparation'):
        dataset = generate_survival(SurvivalConfig(**config['dgp']), seed, config['scenario'])
        fingerprints = dataset_fingerprints(dataset)
        sizes = [len(site.rows) for site in dataset.sites]
        for item in historical.values():
            if any(item['fingerprints'][key] != fingerprints[key]
                   for key in ('training', 'evaluation', 'evaluation_truth')):
                raise ValueError('pooled experiment data differ from the accepted historical FL data')
            if item['result']['site_sizes'] != sizes:
                raise ValueError('pooled experiment site sizes differ from historical FL inputs')

    models = {tag: item['model'] for tag, item in historical.items()}
    fits, arrays = {}, {}
    with measured('fit'):
        for scope in config['pseudo_scopes']:
            tag = 'pooled_' + scope
            model, fits[scope], fitted = fit_pooled(dataset.sites, config['dgp']['tau'], seed,
                config['family'], scope, config['method'], config['public_initial_eta'])
            models[tag] = model
            arrays.update({tag + '__fit__' + key: np.asarray(value) for key, value in fitted.items()})

    # Persist probability predictions so reporting only reads completed results;
    # a plotting command must never silently regenerate data or run inference.
    with measured('probability_evaluation'):
        predictions, metrics, comparisons = {}, {}, {}
        for tag, model in models.items():
            predictions[tag] = probability_predictions(model.predict_eta, dataset)
            metrics[tag] = summarize_predictions(predictions[tag], dataset)
            arrays.update({tag + '__prediction__' + key: np.asarray(value)
                           for key, value in predictions[tag].items()})
        for tag in historical:
            comparisons[tag] = {scope: compare_predictions(predictions[tag], predictions['pooled_' + scope])
                                for scope in config['pseudo_scopes']}
        truth = dataset.evaluation_truth
        arrays.update({'truth__' + key: np.asarray(getattr(truth, key))
                       for key in ('propensity', 'eta0', 'eta1', 'survival0', 'survival1',
                                   'delta', 'modifier_mask')})
        arrays['factual_a'] = np.asarray(dataset.evaluation.a, dtype=np.float64)
        arrays['plot_x12'] = np.asarray(dataset.evaluation.x[:500, :2], dtype=np.float64)
    return dict(status='complete', fit=fits, prediction=metrics, comparisons=comparisons,
                historical={tag: item['binding'] for tag, item in historical.items()},
                historical_inputs=reference, site_sizes=sizes, fingerprints=fingerprints,
                timing=dict(stages=stages, network_transport_measured=False,
                            fit_includes_all_selection_and_refits=True),
                memory=dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    scope='Linux process-lifetime high-water RSS, not allocation or cgroup peak'),
                prediction_arrays_sha256=array_hash(arrays)), arrays
