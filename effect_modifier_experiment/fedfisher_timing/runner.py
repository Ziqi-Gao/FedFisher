"""Foreground scientific composition; only the central adapter dispatches it."""
import os

from . import protocol
from .computation import compute_cell


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
