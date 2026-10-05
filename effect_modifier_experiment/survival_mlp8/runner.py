"""Foreground lifecycle behind the centrally validated ServerScheduler adapter."""
import os

from ..infrastructure.artifacts import digest
from ..survival.artifacts import current_provenance
from . import protocol
from .artifacts import ArtifactStore, DiscoveryArtifactStore


def run_job(job):
    """Allocation and environment were validated by the sole project adapter."""
    from .computation import compute_cell
    return _run_job(job, protocol, ArtifactStore, compute_cell)


def run_discovery_job(job):
    """Use the separate V2 recipe, outputs and fresh discovery pilot gate."""
    from . import discovery_protocol
    from .computation import compute_discovery_cell
    return _run_job(job, discovery_protocol, DiscoveryArtifactStore, compute_discovery_cell,
                    discovery=True)


def _run_job(job, recipe, store_class, compute, *, discovery=False):
    import torch
    from ..survival_pooled.inputs import validate_historical_inputs
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = recipe.validate_parameters(job['task'], job['parameters'])
    config = recipe.run_config(job['task'], parameters)
    source = current_provenance()
    validate_historical_inputs()  # Reject changed reference evidence before publishing output.
    prerequisite = None
    if job['task'] == recipe.FORMAL_TASK:
        from .acceptance import default_acceptance_path, verify_pilot_acceptance
        if discovery:
            proof = verify_pilot_acceptance(default_acceptance_path(source, discovery=True), discovery=True)
        else:
            proof = verify_pilot_acceptance(default_acceptance_path(source))
        prerequisite = dict(pilot_acceptance_sha256=digest(proof),
                            provenance_sha256=source['provenance_sha256'])
    store = store_class(config, source, prerequisite=prerequisite)
    store.initialize()
    execution = {key: job[key] for key in
                 ('job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
    execution['attempt'] = int(os.environ['SERVER_SCHEDULER_ATTEMPT'])
    for seed in recipe.seed_sequence(job['task'], parameters):
        if current_provenance() != source:
            raise ValueError('source changed during MLP8 survival execution')
        if store.status(seed) == 'validated':
            continue
        store.start(execution, seed)
        try:
            result, arrays = compute(config, seed)
            result['identity'] = store.identity(seed)
            if current_provenance() != source:
                raise ValueError('source changed during MLP8 survival execution')
            store.complete(result, arrays, seed)
        except Exception as error:
            store.fail(error, seed)
            raise
        if store.status(seed) != 'validated':
            raise ValueError('MLP8 survival completion did not validate')
    return 0
