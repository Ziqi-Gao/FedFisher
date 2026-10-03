"""Replay composition under the sole validated scheduler foreground adapter."""
import os
import resource

from ...components.measurement import MethodTimer
from ..artifacts import current_provenance
from ..interventions import make_permutations
from ..data_composition import generate_data
from . import protocol
from .artifacts import ReplayArtifactStore
from .sources import load_sources, validate_data_replay, verify_source_bindings


def compute_replay(config):
    """No command-line front door; production enters only through run_job."""
    timer = MethodTimer(config['recipe'])
    with timer.measure('data_preparation'):
        with timer.measure('data_preparation', 'source_integrity_and_restore'):
            sources = load_sources(config)
        with timer.measure('data_preparation', 'regeneration_and_exact_fingerprint_check'):
            original = next(iter(sources.values()))['config']
            data = generate_data(original)
            e = original['evaluation']
            plans = [make_permutations(len(site.holdout), site.train.n_features,
                e['holdout_seed_offset']+original['seed']+1000003*site.site_id, e['repeats'])
                for site in data.sites]
            replay = validate_data_replay(data, plans, sources)
    with timer.measure('evaluation'):
        if config['recipe'] == 'dr_path':
            from .dr import diagnose_dr
            diagnostic = diagnose_dr(config, data, plans, sources)
        else:
            from .fisher import diagnose_fisher
            diagnostic = diagnose_fisher(config, data, plans, sources)
    binding = verify_source_bindings(config)
    return dict(status='complete', source_bindings=binding, data_replay=replay,
                diagnostic=diagnostic, fitting_performed=False, expansion_authorized=False,
                timing=timer.to_dict(), memory=dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    scope='Linux current-process lifetime high-water RSS sampled after replay before completion validation; not cgroup peak or a reservation'),
                scope='seed-zero simulation replay; diagnostic test curves cannot select a new method')


def run_job(job):
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    config = protocol.run_config(job['task'], job['parameters'])
    verify_source_bindings(config)
    store = ReplayArtifactStore(config, current_provenance())
    store.initialize()
    if store.status() == 'validated':
        return 0
    execution = {key: job[key] for key in ('job_id', 'execution_profile', 'allocation')}
    execution.update(attempt=int(os.environ['SERVER_SCHEDULER_ATTEMPT']),
                     estimated_runtime_seconds=job['estimated_runtime_seconds'])
    store.start(execution)
    try:
        result = compute_replay(config)
        result['identity'] = store.identity
        store.complete(result)
    except Exception as error:
        store.fail(error)
        raise
    return 0
