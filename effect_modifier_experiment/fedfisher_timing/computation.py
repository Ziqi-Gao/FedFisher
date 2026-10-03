"""Path-free composition of the preserved standalone Fisher timing recipe."""
import resource
from ..components.fingerprints import model_arrays, model_hash
from ..components.measurement import MethodTimer
from ..causal_comparison.evaluation import evaluate
from ..dslassocov_benchmark import dataset_fingerprint, json_safe, tensor_datasets_to_raw_sites, training_fingerprint
from .data import generate_datasets


def compute_cell(config, seed, split):
    from .fit import fit_fedfisher
    timer = MethodTimer(config['cell'])
    with timer.measure('data_preparation'):
        sites, global_training, test, label_counts = generate_datasets(config, seed, split)
        raw_sites = tensor_datasets_to_raw_sites(sites, config['dgp']['covariate_dim'])
    model, fit = fit_fedfisher(sites, config['method'], seed, timer,
                              global_training=global_training)
    arrays = model_arrays(model)
    with timer.measure('evaluation'):
        evaluation = evaluate(model, sites, test, variant=config['variant'], split=split,
                              seed=seed, dgp=config['dgp'], config=config['evaluation'])
    return json_safe(dict(status='complete', fit=fit, evaluation=evaluation,
        timing=timer.to_dict(), site_sizes=[len(site.labels) for site in raw_sites],
        label_counts=label_counts,
        memory=dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope='Linux current-process lifetime high-water RSS; not cgroup peak or reservation'),
        fingerprints=dict(training=training_fingerprint(raw_sites),
        test=dataset_fingerprint(test), model=model_hash(arrays)))), arrays
