"""Foreground survival composition dispatched only by the central adapter."""
from contextlib import contextmanager
import os
import resource
import time

import numpy as np

from . import protocol


def compute_cell(config, seed=None):
    """Compose one recipe; tiny fixtures may call this with explicit reduced config."""
    from .artifacts import array_hash, dataset_fingerprints
    from .data import SurvivalConfig, generate_survival
    from .evaluation import evaluate_model
    from .pseudo import site_pseudo_diagnostics

    if seed is None:
        if config['task'] != protocol.PILOT_TASK:
            raise ValueError('formal survival computation requires an explicit seed')
        seed = config['seed']
    protocol.validate_seed(config, seed)
    stages = {}

    @contextmanager
    def measured(stage):
        wall, cpu = time.perf_counter(), time.process_time()
        yield
        stages[stage] = dict(wall_seconds=time.perf_counter()-wall,
                             cpu_seconds=time.process_time()-cpu)

    with measured('data_preparation'):
        dataset = generate_survival(SurvivalConfig(**config['dgp']), seed, config['scenario'])
        # This diagnostic pass is additional to each method's own fold/full-site
        # response construction; both costs are explicitly included in timing.
        pseudo_diagnostics = [dict(site_id=site.site_id,
            **site_pseudo_diagnostics(site.rows, config['dgp']['tau'])) for site in dataset.sites]
    with measured('fit'):
        if config['cell'] == 'dslasso':
            from .dslasso import DslassoConfig, fit_dslasso
            model, fit = fit_dslasso(dataset.sites, config['dgp']['tau'], seed=seed,
                                     config=DslassoConfig(**config['method']['optimizer']))
            arrays = dict(intercept=np.asarray(model.intercept, dtype=np.float64),
                          coefficients=np.asarray(model.coefficients, dtype=np.float64))
        else:
            from .fisher import FisherConfig, fit_fedfisher
            model = fit_fedfisher(dataset.sites, config['dgp']['tau'],
                method=config['method']['fisher'], seed=seed,
                config=FisherConfig(**config['method']['optimizer']),
                initial_eta=config['public_initial_eta'])
            fit = model.diagnostics
            arrays = dict(vector=np.asarray(model.vector, dtype=np.float64))
            arrays.update({'upload_%03d' % index: np.frombuffer(payload, dtype=np.uint8).copy()
                           for index, payload in enumerate(model.uploads)})
    with measured('evaluation'):
        evaluation = evaluate_model(model.predict_eta, dataset, repeats=config['evaluation']['repeats'],
                                    seed=seed)
    return dict(status='complete', fit=fit, evaluation=evaluation,
                timing=dict(stages=stages, fit_includes_all_selection_and_refits=True,
                            network_transport_measured=False),
                site_sizes=[len(site.rows) for site in dataset.sites],
                pseudo_diagnostics=pseudo_diagnostics,
                fingerprints=dict(dataset_fingerprints(dataset), model=array_hash(arrays)),
                memory=dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    scope='Linux current-process lifetime high-water RSS; not cgroup peak or reservation')), arrays


def run_job(job):
    import torch
    from .artifacts import ArtifactStore, current_provenance, digest
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = protocol.validate_parameters(job['task'], job['parameters'])
    config = protocol.run_config(job['task'], parameters)
    provenance = current_provenance()
    prerequisite = None
    if job['task'] == protocol.FORMAL_TASK:
        from .acceptance import check_current_pilot_gate
        proof = check_current_pilot_gate()
        prerequisite = dict(pilot_acceptance_sha256=digest(proof),
                            provenance_sha256=provenance['provenance_sha256'])
    store = ArtifactStore(config, provenance, prerequisite=prerequisite)
    store.initialize()
    execution = {key: job[key] for key in (
        'job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in protocol.seed_sequence(job['task'], parameters):
        if current_provenance() != provenance:
            raise ValueError('source changed during survival execution')
        if store.status(seed) == 'validated':
            continue
        store.start(execution, seed)
        try:
            result, arrays = compute_cell(config, seed)
            result['identity'] = store.identity(seed)
            if current_provenance() != provenance:
                raise ValueError('source changed during survival execution')
            store.complete(result, arrays, seed)
        except Exception as error:
            # Preserve the original exception if source mutation also rejects the failure record.
            try:
                store.fail(error, seed)
            except Exception:
                pass
            raise
        if store.status(seed) != 'validated':
            raise ValueError('survival completion failed validation')
    return 0
