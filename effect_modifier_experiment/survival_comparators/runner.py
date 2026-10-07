"""Foreground scheduler lifecycle for one deterministic seed per request."""
import os

from ..survival.artifacts import current_provenance
from . import protocol
from .artifacts import ArtifactStore


def run_job(job):
    import torch
    from .computation import compute_cell
    from .inputs import validate_inputs
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = protocol.validate_parameters(job['task'], job['parameters'])
    config, source = protocol.run_config(job['task'], parameters), current_provenance()
    validate_inputs()
    store = ArtifactStore(config, source)
    store.initialize()
    execution = {key: job[key] for key in ('job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in protocol.seed_sequence(job['task'], parameters):
        if current_provenance() != source:
            raise ValueError('source changed during survival comparator execution')
        if store.status(seed) == 'validated':
            continue
        store.start(execution, seed)
        try:
            result, arrays = compute_cell(config, seed)
            result['identity'] = store.identity(seed)
            if current_provenance() != source:
                raise ValueError('source changed during survival comparator execution')
            store.complete(result, arrays, seed)
        except Exception as error:
            store.fail(error, seed)
            raise
        if store.status(seed) != 'validated':
            raise ValueError('survival comparator completion did not validate')
    return 0
