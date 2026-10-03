"""Path-free composition of the existing comparison data, fitting and evaluation."""
from types import SimpleNamespace
from ..components.fingerprints import model_arrays, model_hash
from ..components.measurement import MethodTimer
from ..dslassocov_benchmark import tensor_datasets_to_raw_sites, training_fingerprint, dataset_fingerprint, json_safe
from ..run_dslassocov_effect_modifier import generate_datasets
from .evaluation import evaluate
from .noncompletion import OUTCOME, ScientificNoncompletion, validate_capped_fit


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
        if (family == "fedavg" and method["optimizer"].get("stopping_mode") == "validation_patience"
                and method["optimizer"].get("max_rounds") == 200
                and audit.get("termination_reason") == "maximum_rounds_reached"):
            arrays = model_arrays(model)
            validate_capped_fit(audit, method["optimizer"], method["model"], seed, arrays)
            raise ScientificNoncompletion(audit, arrays)
        raise RuntimeError("method stopped without meeting the declared scientific completion rule: " + str(audit.get("termination_reason", audit.get("stop_reason"))))
    return model, audit

def compute_cell(config, seed, split):
    """No paths, scheduling, or test-label access inside method fitting."""
    timer = MethodTimer(config["cell"])
    with timer.measure("data_preparation"):
        sites, test, label_counts = generate_datasets(SimpleNamespace(
            **config["dgp"], seed=seed, split=split, variant=config["variant"]))
        raw_sites = tensor_datasets_to_raw_sites(sites, config["dgp"]["covariate_dim"])
    try:
        model, fit = fit_method(raw_sites, sites, config["method"], seed, timer)
    except ScientificNoncompletion as error:
        error.result = json_safe(dict(status="scientific_noncompletion", outcome=OUTCOME,
            fit=error.fit, timing=timer.to_dict(), site_sizes=[len(site.labels) for site in raw_sites],
            label_counts=label_counts,
            fingerprints=dict(training=training_fingerprint(raw_sites),
                              test=dataset_fingerprint(test), model=model_hash(error.arrays))))
        raise
    arrays = model_arrays(model)
    with timer.measure("evaluation"):
        evaluation = evaluate(model, sites, test, variant=config["variant"], split=split,
                              seed=seed, dgp=config["dgp"], config=config["evaluation"])
    return json_safe(dict(status="complete", fit=fit, evaluation=evaluation,
        timing=timer.to_dict(), site_sizes=[len(site.labels) for site in raw_sites],
        label_counts=label_counts,
        fingerprints=dict(training=training_fingerprint(raw_sites),
                          test=dataset_fingerprint(test), model=model_hash(arrays)))), arrays
