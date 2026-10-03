"""Scientific composition called only by the registered foreground adapter."""


from . import protocol
from .artifacts import ArtifactStore, OUTPUT_ROOT, make_manifest, current_provenance
from .computation import compute_cell, fit_method


def run_shard(job):
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    parameters = protocol.validate_parameters(job["task"], job["parameters"])
    config = protocol.run_config(job["task"], parameters)
    provenance = current_provenance()
    run_id = protocol.run_id(job["task"], parameters)
    store = ArtifactStore(OUTPUT_ROOT / run_id, make_manifest(run_id, config, provenance))
    store.initialize()
    for seed in protocol.seed_sequence(job["task"], parameters):
        for split in protocol.SPLITS:
            if current_provenance() != provenance:
                raise ValueError("source changed during scientific shard")
            if store.status(seed, split) == "validated":
                continue
            store.start(seed, split, dict(job_id=job["job_id"],
                execution_profile=job["execution_profile"], allocation=job["allocation"]))
            try:
                result, arrays = compute_cell(config, seed, split)
                result["identity"] = store.identity(seed, split)
                if current_provenance() != provenance:
                    raise ValueError("source changed during scientific fit")
                store.complete(seed, split, result, arrays)
            except Exception as error:
                store.fail(seed, split, error)
                raise
    return 0
