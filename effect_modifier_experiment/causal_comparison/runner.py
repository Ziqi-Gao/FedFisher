"""Scientific composition called only by the registered foreground adapter."""

from types import SimpleNamespace

from ..dslassocov_benchmark import tensor_datasets_to_raw_sites, training_fingerprint, dataset_fingerprint, json_safe
from ..run_dslassocov_effect_modifier import generate_datasets
from . import protocol
from .artifacts import ArtifactStore, OUTPUT_ROOT, make_manifest, current_provenance
from .evaluation import evaluate, model_arrays, model_hash
from .measurement import MethodTimer


def fit_method(raw_sites, site_datasets, method, seed, timer):
    family = method["family"]
    if family == "fedavg":
        from .fedavg import FedAvgConfig, fit_multiround_fedavg
        with timer.measure("train"):
            model, audit = fit_multiround_fedavg(raw_sites, method["model"],
                FedAvgConfig(**method["optimizer"]), seed)
    elif family == "rlearner":
        from .rlearner import RlearnerConfig, fit_homogeneous_rlearner
        with timer.measure("train"):
            model, audit = fit_homogeneous_rlearner(raw_sites,
                covariate_dim=(raw_sites[0].features.shape[1] - 1) // 2,
                penalty=method["penalty"], config=RlearnerConfig(**method["optimizer"]), seed=seed)
        audit["completion_eligible"] = bool(audit["converged"])
    else:
        from .legacy import fit_legacy
        model, audit = fit_legacy(raw_sites, site_datasets, method, seed, timer)
    if audit.get("completion_eligible") is not True:
        raise RuntimeError("method stopped without meeting the declared scientific completion rule: " + str(audit.get("termination_reason", audit.get("stop_reason"))))
    return model, audit


def compute_cell(config, seed, split):
    """No paths, scheduling, or test-label access inside method fitting."""
    timer = MethodTimer(config["cell"])
    with timer.measure("data_preparation"):
        sites, test, label_counts = generate_datasets(SimpleNamespace(
            **config["dgp"], seed=seed, split=split, variant=config["variant"]))
        raw_sites = tensor_datasets_to_raw_sites(sites, config["dgp"]["covariate_dim"])
    model, fit = fit_method(raw_sites, sites, config["method"], seed, timer)
    arrays = model_arrays(model)
    with timer.measure("evaluation"):
        evaluation = evaluate(model, sites, test, variant=config["variant"], split=split,
                              seed=seed, dgp=config["dgp"], config=config["evaluation"])
    return json_safe(dict(status="complete", fit=fit, evaluation=evaluation,
        timing=timer.to_dict(), site_sizes=[len(site.labels) for site in raw_sites],
        label_counts=label_counts,
        fingerprints=dict(training=training_fingerprint(raw_sites),
                          test=dataset_fingerprint(test), model=model_hash(arrays)))), arrays


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
