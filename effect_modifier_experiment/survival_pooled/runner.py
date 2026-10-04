"""Foreground, registered-task-only lifecycle for pooled survival validation."""
import os

from ..survival.artifacts import current_provenance
from ..infrastructure.artifacts import digest
from . import protocol
from .artifacts import ArtifactStore


def run_job(job):
    """The central adapter has validated allocation/environment before this call."""
    import torch
    from .inputs import validate_historical_inputs
    from .computation import compute_cell
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = protocol.validate_parameters(job['task'], job['parameters'])
    config = protocol.run_config(job['task'], parameters)
    source = current_provenance()
    validate_historical_inputs()  # Fail before output, fitting or inference if anchors changed.
    prerequisite = None
    if job['task'] == protocol.FORMAL_TASK:
        from .acceptance import default_acceptance_path, verify_pilot_acceptance
        proof = verify_pilot_acceptance(default_acceptance_path(source))
        prerequisite = dict(pilot_acceptance_sha256=digest(proof), provenance_sha256=source['provenance_sha256'])
    store = ArtifactStore(config, source, prerequisite=prerequisite)
    store.initialize()
    execution = {key: job[key] for key in ('job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in protocol.seed_sequence(job['task'], parameters):
        if current_provenance() != source:
            raise ValueError('source changed during pooled survival execution')
        if store.status(seed) == 'validated':
            continue
        store.start(execution, seed)
        try:
            result, arrays = compute_cell(config, seed)
            result['identity'] = store.identity(seed)
            if current_provenance() != source:
                raise ValueError('source changed during pooled survival execution')
            store.complete(result, arrays, seed)
        except Exception as error:
            store.fail(error, seed)
            raise
        if store.status(seed) != 'validated':
            raise ValueError('pooled survival completion did not validate')
    return 0
