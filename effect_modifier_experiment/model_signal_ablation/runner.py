"""Foreground composition of unchanged FedAvg, dsLasso and original Fisher fits."""

import os
import resource

from . import protocol
from .artifacts import ArtifactStore, MEMORY_SCOPE, current_provenance


def compute_cell(config, seed, split):
    if config['method']['family'] == 'legacy_fedfisher':
        from ..fedfisher_timing.runner import compute_cell as compute
    else:
        from ..causal_comparison.runner import compute_cell as compute
    result, arrays = compute(config, seed, split)
    result['memory'] = dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            scope=MEMORY_SCOPE)
    return result, arrays


def run_shard(job):
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = protocol.validate_parameters(job['task'], job['parameters'])
    config = protocol.run_config(job['task'], parameters)
    provenance = current_provenance()
    if job['task'] == protocol.FORMAL_TASK:
        from .acceptance import check_current_pilot_gate
        check_current_pilot_gate()
    store = ArtifactStore(config, provenance)
    store.initialize()
    execution = {key: job[key] for key in (
        'job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in protocol.seed_sequence(job['task'], parameters):
        for split in protocol.SPLITS:
            if current_provenance() != provenance:
                raise ValueError('source changed during model/signal shard')
            if store.status(seed, split) == 'validated':
                continue
            store.start(seed, split, execution)
            try:
                result, arrays = compute_cell(config, seed, split)
                result['identity'] = store.identity(seed, split)
                if current_provenance() != provenance:
                    raise ValueError('source changed during model/signal fitting')
                store.complete(seed, split, result, arrays)
            except Exception as error:
                store.fail(seed, split, error)
                raise
    return 0
