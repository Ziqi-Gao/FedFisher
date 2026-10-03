"""Standalone CPU fitting of the preserved Diag/KFAC aggregation algorithms.

This module accepts observed training sites only. It owns no paths, datasets,
resources, or evaluation set. The RNG recipe is intentionally new: Fisher and
server work cannot alter subsequent local SGD, so both algorithms share local
model functions. It does not claim bitwise reproduction of the archived GPU
pipeline, which computed both Fisher representations between successive sites.
"""

from collections import Counter
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import math
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset

from models import get_model
from run_one_shot_algs import get_fisher_diag_model, get_fisher_kfac_model
from train_model import LocalUpdate
from effect_modifier_experiment.dslassocov_benchmark import (
    dataset_fingerprint, tensor_datasets_to_raw_sites, training_fingerprint,
)
from effect_modifier_experiment.components.fingerprints import (
    array_hash, model_arrays, model_hash,
)


OPTIMIZER_FIELDS = {
    "local_epochs", "batch_size", "learning_rate", "momentum",
    "validation_size", "server_iterations", "server_learning_rate",
    "validation_interval", "quantization_bits", "rng_policy",
}
RNG_POLICY = "isolate_fisher_from_training_v1"


@contextmanager
def _isolated_cpu_rng(seed, numpy_state):
    """Keep all fitting, Fisher and compression RNG local to one method fit."""
    numpy_before, python_before = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            np.random.set_state(numpy_state)
            random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_before)
        random.setstate(python_before)


def _validate_method(method, seed):
    if (not isinstance(method, dict) or
            set(method) != {"family", "algorithm", "model", "optimizer"} or
            method["family"] != "legacy_fedfisher" or
            method["algorithm"] not in ("fedfisher_diag", "fedfisher_kfac") or
            method["model"] not in ("SyntheticMLP", "SyntheticMLPDeep", "SyntheticMLPSmall")):
        raise ValueError("invalid standalone FedFisher method")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    options = method["optimizer"]
    if not isinstance(options, dict) or set(options) != OPTIMIZER_FIELDS:
        raise ValueError("incomplete standalone FedFisher optimizer recipe")
    for name in ("local_epochs", "batch_size", "validation_size"):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if (type(options["learning_rate"]) not in (int, float) or
            not math.isfinite(options["learning_rate"]) or options["learning_rate"] <= 0):
        raise ValueError("learning_rate must be finite and positive")
    # The imported legacy implementations fix these values internally. Reject
    # requests whose reported settings would differ from what they actually run.
    fixed = dict(momentum=0.9, server_iterations=2000, server_learning_rate=0.01,
                 validation_interval=100, quantization_bits=15, rng_policy=RNG_POLICY)
    if any(options[name] != value or isinstance(options[name], bool)
           for name, value in fixed.items()):
        raise ValueError("optimizer does not match the preserved Fisher implementation")
    if any(type(options[name]) is not int for name in
           ("server_iterations", "validation_interval", "quantization_bits")):
        raise ValueError("server iteration and quantization counts must be integers")
    return dict(options)


def _check_sites(site_datasets):
    if not isinstance(site_datasets, (list, tuple)) or not site_datasets:
        raise ValueError("ordered nonempty training sites are required")
    dimension = None
    for site in site_datasets:
        if not isinstance(site, TensorDataset) or len(site.tensors) != 2:
            raise ValueError("training sites must be observed TensorDataset pairs")
        x, y = site.tensors
        if (x.device.type != "cpu" or y.device.type != "cpu" or
                x.dtype != torch.float32 or y.dtype != torch.long or
                x.ndim != 2 or len(x) == 0 or y.shape != (len(x),) or
                x.shape[1] < 3 or x.shape[1] % 2 != 1 or
                not torch.isfinite(x).all().item() or
                not torch.all((y == 0) | (y == 1)).item()):
            raise ValueError("training sites require finite CPU float32 [A,X,A*X] and binary int64 labels")
        if dimension is not None and x.shape[1] != dimension:
            raise ValueError("training sites have different feature dimensions")
        dimension = x.shape[1]
        d = (dimension - 1) // 2
        if (not torch.all((x[:, 0] == 0) | (x[:, 0] == 1)).item() or
                not torch.equal(x[:, 1 + d:], x[:, :1] * x[:, 1:1 + d])):
            raise ValueError("training treatment/interaction inputs are inconsistent")
    return dimension


def _finite_model_hash(model):
    arrays = model_arrays(model)
    if not arrays or not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("FedFisher produced nonfinite model parameters")
    return model_hash(arrays)


def _fisher_hash(fisher, kind):
    if kind == "diag":
        tensors = [("diag", fisher)]
    else:
        tensors = [(str(layer) + "/" + str(index), tensor)
                   for layer, factors in fisher.data.items()
                   for index, tensor in enumerate(factors)]
    if not tensors:
        raise FloatingPointError("FedFisher has an empty Fisher representation")
    digest = hashlib.sha256(("standalone-fisher-" + kind + "-v1\0").encode())
    for name, tensor in tensors:
        if tensor.device.type != "cpu" or not torch.isfinite(tensor).all().item():
            raise FloatingPointError("FedFisher produced nonfinite or non-CPU Fisher factors")
        if kind == "diag" and torch.any(tensor < 0).item():
            raise FloatingPointError("diagonal Fisher must be nonnegative")
        digest.update(name.encode() + b"\0" + array_hash(tensor.detach().numpy()).encode())
    return digest.hexdigest()


def fit_fedfisher(site_datasets, method, seed, timer=None, *, global_training=None):
    """Fit one complete method, measuring its own SGD, Fisher and aggregation.

    Production passes the generator's original global training tensor to retain
    its validation row order. A site-order fallback exists for tiny fixtures.
    Validation rows remain in local fitting, preserving the original in-sample
    selection policy. Independent test rows are never accepted.
    """
    options = _validate_method(method, seed)
    dimension = _check_sites(site_datasets)
    algorithm = method["algorithm"]
    kind = "diag" if algorithm == "fedfisher_diag" else "kfac"
    args = dict(bs=options["batch_size"], local_epochs=options["local_epochs"],
                eta=options["learning_rate"], device="cpu", n_c=2,
                rounds=1, num_clients=len(site_datasets), model=method["model"],
                dataset="SyntheticEffectModifier", synthetic_dim=dimension)
    local_reports, models, vectors, fishers = [], [], [], []

    with _isolated_cpu_rng(seed, np.random.RandomState(3).get_state()):
        with timer.measure("train") if timer is not None else nullcontext():
            pooled = TensorDataset(*(torch.cat([site.tensors[i] for site in site_datasets], dim=0)
                                     for i in (0, 1)))
            pooled_order = "site_index_then_local_row_fixture_fallback"
            if global_training is not None:
                if _check_sites([global_training]) != dimension or len(global_training) != len(pooled):
                    raise ValueError("global training rows do not match the local training sites")
                def row_counts(dataset):
                    x, y = (tensor.detach().numpy() for tensor in dataset.tensors)
                    return Counter((row.tobytes(), int(label)) for row, label in zip(x, y))
                if row_counts(global_training) != row_counts(pooled):
                    raise ValueError("global training rows must be exactly the local training union")
                pooled = global_training
                pooled_order = "original_generator_global_row_order"
            # Selection indices are independent of caller RNG, and match the
            # original np.random.seed(3)/choice ordering on this same tensor.
            selection_rng = np.random.RandomState(3)
            validation_indices = selection_rng.choice(len(pooled),
                min(options["validation_size"], len(pooled)), replace=False)
            validation = TensorDataset(*(tensor[validation_indices] for tensor in pooled.tensors))
            np.random.set_state(selection_rng.get_state())
            raw_sites = tensor_datasets_to_raw_sites(site_datasets, (dimension - 1) // 2)
            training_hash = training_fingerprint(raw_sites)
            sizes = np.asarray([len(site) for site in site_datasets], dtype=np.int64)
            weights = sizes.astype(np.float64) / int(sizes.sum())
            initial = get_model(method["model"], 2, bias=False,
                                synthetic_dim=dimension).to(device="cpu", dtype=torch.float32)
            initial_hash = _finite_model_hash(initial)
            parameter_count = sum(parameter.numel() for parameter in initial.parameters())
            for site_id, site in enumerate(site_datasets):
                local = LocalUpdate(args=args, dataset=site)
                vector, model, kfac, diag = local.train_and_compute_fisher(
                    deepcopy(initial), 2, timing=timer, fisher_kind=kind,
                    isolate_fisher_rng=True)
                fisher = diag if kind == "diag" else kfac
                fisher_hash = _fisher_hash(fisher, kind)
                local_hash = _finite_model_hash(model)
                local_reports.append(dict(
                    site_idx=site_id, num_examples=int(sizes[site_id]),
                    weight=float(weights[site_id]), epochs=options["local_epochs"],
                    updates=options["local_epochs"] * math.ceil(len(site) / options["batch_size"]),
                    sample_visits=options["local_epochs"] * len(site),
                    model_fingerprint=local_hash, fisher_fingerprint=fisher_hash,
                    training_fingerprint=dataset_fingerprint(site)))
                # Preserve the original site weighting before the original
                # quantization/compression and validation-selected optimizer.
                if kind == "diag":
                    fisher = fisher * weights[site_id]
                else:
                    for layer in fisher.data:
                        fisher.data[layer] = list(fisher.data[layer])
                        fisher.data[layer][0].mul_(weights[site_id])
                        fisher.data[layer][1].mul_(weights[site_id])
                _fisher_hash(fisher, kind)
                models.append(model)
                vectors.append(vector)
                fishers.append(fisher)

        with timer.measure("aggregation") if timer is not None else nullcontext():
            if kind == "diag":
                result = get_fisher_diag_model(parameter_count, len(models), weights,
                    args, initial, vectors, fishers, validation)
            else:
                result = get_fisher_kfac_model(parameter_count, len(models), weights,
                    args, initial, models, vectors, fishers, validation, site_datasets)
            result_hash = _finite_model_hash(result)

    audit = dict(
        family="legacy_fedfisher", algorithm=algorithm, model=method["model"], seed=seed,
        completion_eligible=True, termination_reason="fixed_recipe_completed",
        convergence_claim=False, device="cpu", rng_policy=RNG_POLICY,
        standalone_method=True, fisher_representation=kind,
        historical_gpu_bitwise_reproduction=False,
        local_epochs=options["local_epochs"], parameter_count=parameter_count,
        total_local_updates=sum(site["updates"] for site in local_reports),
        total_sample_visits=sum(site["sample_visits"] for site in local_reports),
        training_communication_rounds=1,
        server_iterations=options["server_iterations"],
        validation_checks=options["server_iterations"] // options["validation_interval"],
        initial_model_fingerprint=initial_hash, model_fingerprint=result_hash,
        training_fingerprint=training_hash, site_reports=local_reports,
        local_model_fingerprints=[site["model_fingerprint"] for site in local_reports],
        validation=dict(
            scope="pooled_training_subset_also_used_in_local_fits",
            independent_test_access=False, pooled_order=pooled_order,
            selection_seed=3, num_examples=len(validation),
            indices_sha256=hashlib.sha256(np.asarray(validation_indices, dtype="<i8").tobytes()).hexdigest(),
            dataset_sha256=dataset_fingerprint(validation),
            selection_rule="highest_validation_accuracy_strict_improvement_first_tie",
            checked_iteration_indices=list(range(0, options["server_iterations"],
                                                 options["validation_interval"]))),
        timing_contract=dict(training_includes_all_selection_and_refits=True,
            fit_stages=["train", "aggregation"],
            train="initialization, all local SGD and selected Fisher representation only",
            aggregation="original quantization/compression and all server iterations/validation selection",
            tuning="included_in_aggregation", refit="none_in_original_recipe",
            evaluation="independent_post_fit_stage_owned_by_experiment"),
        communication=dict(execution_model="single_process_simulation",
            actual_network_measured=False, accounted=False,
            training_communication_rounds=1,
            reason="preserved aggregation has no serialized upload boundary; KFAC compression accesses training rows"))
    return result, audit
