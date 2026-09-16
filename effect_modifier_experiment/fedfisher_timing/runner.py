"""Foreground scientific composition; only the central adapter dispatches it."""
import os
import resource

from ..causal_comparison.evaluation import evaluate, model_arrays, model_hash
from ..causal_comparison.measurement import MethodTimer
from ..dslassocov_benchmark import (
    dataset_fingerprint, json_safe, tensor_datasets_to_raw_sites, training_fingerprint)
from . import protocol
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


def run_shard(job):
    import torch
    from .artifacts import ArtifactStore, current_provenance
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = protocol.validate_parameters(job['task'], job['parameters'])
    config = protocol.run_config(job['task'], parameters)
    provenance = current_provenance()
    store = ArtifactStore(config, provenance)
    store.initialize()
    execution = {key: job[key] for key in (
        'job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in protocol.seed_sequence(job['task'], parameters):
        for split in protocol.SPLITS:
            if current_provenance() != provenance:
                raise ValueError('source changed during Fisher timing shard')
            if store.status(seed, split) == 'validated':
                continue
            store.start(seed, split, execution)
            try:
                result, arrays = compute_cell(config, seed, split)
                result['identity'] = store.identity(seed, split)
                if current_provenance() != provenance:
                    raise ValueError('source changed during Fisher timing fit')
                store.complete(seed, split, result, arrays)
            except Exception as error:
                store.fail(seed, split, error)
                raise
    return 0
