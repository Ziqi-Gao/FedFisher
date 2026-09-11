"""Foreground, CPU-only multiround FedAvg for outcome-model comparisons.

This module only fits supplied training sites. It neither generates data nor
accepts an independent test set. The scheduler-facing experiment owns resource
allocation, artifacts, and final counterfactual prediction evaluation.

Validation selects a round count using a fixed holdout from each training site;
the default then repeats exactly that many rounds on all training rows, starting
from the original common initialization. Selection and refit both count toward
time and communication. Fixed-total-epoch comparisons instead use all training
rows directly, without holdout selection. Momentum is reset each local round.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from time import perf_counter, process_time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from effect_modifier_experiment.dslassocov_benchmark import RawSite
from models import get_model
from run_one_shot_algs import get_fedavg_model


ALLOWED_LOCAL_EPOCHS = (1, 5, 10, 30)
SUPPORTED_MODELS = ("SyntheticMLP", "SyntheticMLPDeep")


@dataclass(frozen=True)
class FedAvgConfig:
    covariate_dim: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    validation_fraction: float = 0.2
    min_rounds: int = 5
    max_rounds: int = 200
    patience: int = 10
    min_delta: float = 1e-4
    stopping_mode: str = "validation_patience"
    total_local_epochs: int = 30
    refit_full_data: bool = True
    split_seed_offset: int = 5100037
    loader_seed_offset: int = 6200037

    def __post_init__(self):
        positive = (
            "covariate_dim", "local_epochs", "batch_size", "min_rounds",
            "max_rounds", "patience", "total_local_epochs",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("split_seed_offset", "loader_seed_offset"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("%s must be a nonnegative integer" % name)
        if self.local_epochs not in ALLOWED_LOCAL_EPOCHS:
            raise ValueError("local_epochs must be one of 1, 5, 10, 30")
        if self.min_rounds > self.max_rounds:
            raise ValueError("min_rounds cannot exceed max_rounds")
        for name in ("learning_rate", "momentum", "validation_fraction", "min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("%s must be finite numeric" % name)
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1:
            raise ValueError("invalid SGD learning rate or momentum")
        if not 0 < self.validation_fraction < 0.5 or self.min_delta < 0:
            raise ValueError("invalid validation fraction or minimum improvement")
        if self.stopping_mode not in ("validation_patience", "fixed_total_epochs"):
            raise ValueError("unsupported stopping mode")
        if type(self.refit_full_data) is not bool:
            raise ValueError("refit_full_data must be boolean")
        if self.total_local_epochs % self.local_epochs:
            raise ValueError("total_local_epochs must be divisible by local_epochs")


@dataclass(frozen=True)
class _SiteRows:
    site_idx: int
    features: torch.Tensor
    labels: torch.Tensor


def _model_fingerprint(model: nn.Module) -> str:
    vector = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    if not np.isfinite(vector).all():
        raise FloatingPointError("FedAvg produced a nonfinite model")
    digest = hashlib.sha256(b"fedfisher-multiround-model-v1\0")
    digest.update(np.asarray(vector, dtype="<f4").tobytes())
    return digest.hexdigest()


def _indices_fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _stratified_holdout(labels, fraction: float, seed: int):
    """Split observed classes, retaining each singleton in training only."""
    generator = np.random.default_rng(seed)
    train_indices, validation_indices = [], []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        if indices.size < 2:
            train_indices.extend(indices.tolist())
            continue
        count = min(indices.size - 1, max(1, int(round(fraction * indices.size))))
        validation_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())
    if not train_indices or not validation_indices:
        raise ValueError("each site needs an observed class with at least two rows for holdout")
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)


def _prepare_sites(raw_sites: Sequence[RawSite], config: FedAvgConfig, seed: int):
    if not raw_sites:
        raise ValueError("at least one training site is required")
    full_sites, fit_sites, validation_sites, report = [], [], [], []
    for position, site in enumerate(raw_sites):
        if not isinstance(site, RawSite):
            raise TypeError("training sites must be RawSite objects")
        if type(site.site_idx) is not int or site.site_idx != position:
            raise ValueError("training sites must have consecutive ordered integer site indices")
        features = np.asarray(site.features)
        labels = np.asarray(site.labels)
        if features.ndim != 2 or features.shape[1] != 1 + 2 * config.covariate_dim:
            raise ValueError("training features have the wrong input dimension")
        if not features.shape[0] or labels.shape != (features.shape[0],):
            raise ValueError("training features and labels have incompatible or empty shapes")
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise ValueError("training arrays must be finite")
        if not set(np.unique(labels).tolist()).issubset({0, 1}):
            raise ValueError("training labels must be binary")
        x = torch.tensor(features, dtype=torch.float32, device="cpu")
        y = torch.tensor(labels, dtype=torch.long, device="cpu")
        if not torch.isfinite(x).all().item():
            raise ValueError("training features overflow float32")
        full_sites.append(_SiteRows(position, x, y))
        item = {
            "site_idx": position,
            "num_examples": int(labels.size),
            "label_counts": {str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)},
        }
        if config.stopping_mode == "validation_patience":
            split_seed = config.split_seed_offset + seed + position * 100003
            fit_indices, validation_indices = _stratified_holdout(labels, config.validation_fraction, split_seed)
            fit_sites.append(_SiteRows(position, x[fit_indices], y[fit_indices]))
            validation_sites.append(_SiteRows(position, x[validation_indices], y[validation_indices]))
            item.update({
                "split_seed": split_seed,
                "fit_examples": int(fit_indices.size),
                "validation_examples": int(validation_indices.size),
                "fit_indices_sha256": _indices_fingerprint(fit_indices),
                "validation_indices_sha256": _indices_fingerprint(validation_indices),
            })
        else:
            fit_sites.append(full_sites[-1])
            item.update({"fit_examples": int(labels.size), "validation_examples": 0})
        report.append(item)
    return full_sites, fit_sites, validation_sites, report


def _make_initial_model(model_name: str, input_dim: int, seed: int, initial_state):
    # CPU initialization is isolated from the caller's RNG and never reads or
    # initializes a CUDA device or modifies CUDA_VISIBLE_DEVICES.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model = get_model(model_name, 2, bias=False, synthetic_dim=input_dim).to(device="cpu", dtype=torch.float32)
    if initial_state is not None:
        if not isinstance(initial_state, Mapping) or any(
            not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
            for value in initial_state.values()
        ):
            raise ValueError("initial state must contain float32 CPU tensors only")
        model.load_state_dict(initial_state, strict=True)
    _model_fingerprint(model)
    return model


def _loader_seed(config, seed: int, site_idx: int, round_index: int, phase: str) -> int:
    # Refit and selection receive distinct deterministic minibatch streams.
    material = "%d:%d:%d:%d:%s" % (config.loader_seed_offset, seed, site_idx, round_index, phase)
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "little")


def _train_local_model(global_model, site: _SiteRows, config: FedAvgConfig, loader_seed: int):
    model = deepcopy(global_model)
    model.train()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(loader_seed)
    # Each round starts from the latest aggregate with fresh local momentum.
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    num_examples = site.labels.numel()
    updates = 0
    for _ in range(config.local_epochs):
        order = torch.randperm(num_examples, generator=generator, device="cpu")
        for start in range(0, num_examples, config.batch_size):
            indices = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(site.features[indices]), site.labels[indices])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite local FedAvg training loss")
            loss.backward()
            optimizer.step()
            updates += 1
    return model, updates


def _validation_loss(model: nn.Module, sites: Sequence[_SiteRows]):
    model.eval()
    reports = []
    with torch.no_grad():
        for site in sites:
            total = float(nn.functional.cross_entropy(model(site.features), site.labels, reduction="sum").item())
            if not math.isfinite(total):
                raise FloatingPointError("nonfinite FedAvg validation loss")
            reports.append({"site_idx": site.site_idx, "loss_sum": total, "num_examples": int(site.labels.numel())})
    mean = sum(item["loss_sum"] for item in reports) / sum(item["num_examples"] for item in reports)
    return mean, reports


def _run_round(global_model, sites, config, seed, phase, round_index):
    source_hash = _model_fingerprint(global_model)
    vectors, site_reports = [], []
    for site in sites:
        loader_seed = _loader_seed(config, seed, site.site_idx, round_index, phase)
        wall_start, cpu_start = perf_counter(), process_time()
        local_model, updates = _train_local_model(global_model, site, config, loader_seed)
        wall_seconds, cpu_seconds = perf_counter() - wall_start, process_time() - cpu_start
        expected_updates = config.local_epochs * math.ceil(site.labels.numel() / config.batch_size)
        if updates != expected_updates:
            raise RuntimeError("local SGD did not execute the prescribed minibatch count")
        vectors.append(parameters_to_vector(local_model.parameters()).detach().clone())
        site_reports.append({
            "site_idx": site.site_idx,
            "num_examples": int(site.labels.numel()),
            "local_epochs": config.local_epochs,
            "gradient_updates": updates,
            "loader_seed": loader_seed,
            "initial_model_fingerprint": source_hash,
            "trained_model_fingerprint": _model_fingerprint(local_model),
            "training_wall_seconds": wall_seconds,
            "training_cpu_seconds": cpu_seconds,
        })
    counts = np.asarray([site.labels.numel() for site in sites], dtype=np.float64)
    weights = counts / counts.sum()
    wall_start, cpu_start = perf_counter(), process_time()
    # The aggregation is the original repository's sample-weighted FedAvg.
    with torch.device("cpu"):
        aggregated_model = get_fedavg_model(vectors[0].numel(), len(sites), weights, {"device": "cpu"}, global_model, vectors)
    aggregation_wall, aggregation_cpu = perf_counter() - wall_start, process_time() - cpu_start
    vector_bytes = int(vectors[0].numel() * vectors[0].element_size())
    return aggregated_model, {
        "phase": phase,
        "round": round_index,
        "initial_model_fingerprint": source_hash,
        "aggregate_model_fingerprint": _model_fingerprint(aggregated_model),
        "aggregation_weights": weights.tolist(),
        "sites": site_reports,
        "aggregation_wall_seconds": aggregation_wall,
        "aggregation_cpu_seconds": aggregation_cpu,
        "optimization_payload": {
            "synchronization_phases": 1,
            "model_upload_count": len(sites),
            "model_broadcast_count": len(sites),
            "site_upload_bytes": len(sites) * (vector_bytes + 16),
            "server_broadcast_bytes": len(sites) * vector_bytes,
        },
    }


def _without_timings(value):
    if isinstance(value, dict):
        return {key: _without_timings(item) for key, item in value.items() if not key.endswith("_seconds")}
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def _communication_report(history, num_sites: int, vector_elements: int):
    rounds = len(history)
    validation_rounds = sum("validation_loss" in item for item in history)
    vector_bytes = vector_elements * 4
    # Count explicit dense tensor/scalar payloads, excluding transport framing.
    model_upload_bytes = rounds * num_sites * (vector_bytes + 16)
    model_broadcast_bytes = (rounds + validation_rounds) * num_sites * vector_bytes
    validation_upload_bytes = validation_rounds * num_sites * 24
    return {
        "measurement_kind": "logical_single_process_protocol_payloads_not_network_measurements",
        "training_communication_rounds": rounds,
        "optimization_rounds": rounds,
        "selection_training_rounds": sum(item["phase"] == "selection" for item in history),
        "refit_training_rounds": sum(item["phase"] == "refit" for item in history),
        "validation_communication_rounds": validation_rounds,
        "total_synchronization_phases": rounds + validation_rounds,
        "site_model_upload_count": rounds * num_sites,
        "site_validation_upload_count": validation_rounds * num_sites,
        "server_model_broadcast_count": (rounds + validation_rounds) * num_sites,
        "model_vector_elements": vector_elements,
        "model_vector_scalar_dtype": "float32",
        "model_upload_metadata_bytes_per_site": 16,
        "validation_upload_bytes_per_site": 24,
        "site_model_upload_bytes": model_upload_bytes,
        "site_validation_upload_bytes": validation_upload_bytes,
        "total_site_upload_bytes": model_upload_bytes + validation_upload_bytes,
        "total_server_broadcast_bytes": model_broadcast_bytes,
        "total_payload_bytes": model_upload_bytes + validation_upload_bytes + model_broadcast_bytes,
        "broadcast_policy": "model_before_every_local_round_and_aggregate_before_every_validation",
        "metadata_layout": "model: int64_site_id,int64_fit_count; validation: int64_site_id,float64_loss_sum,int64_count",
        "server_aggregation": "repository_get_fedavg_model_sample_count_weighted_mean",
    }


def fit_multiround_fedavg(
    raw_sites: Sequence[RawSite],
    model_name: str,
    config: FedAvgConfig,
    seed: int,
    *,
    initial_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Fit only supplied training rows and return model plus complete audit.

    ``completion_eligible`` is false if validation never meets its stopping rule
    or the fixed budget hits the safety bound. Such a return is diagnostic only:
    callers must reject it before writing scientific completion evidence.
    ``converged`` describes validation patience, not a nonconvex optimum; fixed
    epoch runs intentionally report ``converged=None`` and explicit completion.
    """
    total_wall_start, total_cpu_start = perf_counter(), process_time()
    if not isinstance(config, FedAvgConfig):
        raise TypeError("config must be FedAvgConfig")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError("only SyntheticMLP and SyntheticMLPDeep are supported")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("the repository FedAvg aggregation requires the default float32 dtype")
    full_sites, fit_sites, validation_sites, split_audit = _prepare_sites(raw_sites, config, seed)
    initial_model = _make_initial_model(model_name, 1 + 2 * config.covariate_dim, seed, initial_state)
    initial_hash = _model_fingerprint(initial_model)
    global_model = deepcopy(initial_model)
    preprocessing_wall = perf_counter() - total_wall_start
    preprocessing_cpu = process_time() - total_cpu_start
    history = []
    best_model, best_round, best_loss = None, None, float("inf")
    stopping_reference, stale_rounds = float("inf"), 0
    termination_reason = "maximum_rounds_reached"
    completion_eligible = False
    refit_rounds = 0

    if config.stopping_mode == "fixed_total_epochs":
        prescribed_rounds = config.total_local_epochs // config.local_epochs
        for round_index in range(1, min(config.max_rounds, prescribed_rounds) + 1):
            global_model, item = _run_round(global_model, full_sites, config, seed, "fixed_budget", round_index)
            history.append(item)
        if len(history) == prescribed_rounds:
            termination_reason, completion_eligible = "fixed_total_epochs_completed", True
        selected_rounds = len(history)
        returned_phase = "fixed_budget"
        converged = None
    else:
        for round_index in range(1, config.max_rounds + 1):
            global_model, item = _run_round(global_model, fit_sites, config, seed, "selection", round_index)
            wall_start, cpu_start = perf_counter(), process_time()
            loss, site_losses = _validation_loss(global_model, validation_sites)
            item.update({
                "validation_loss": loss,
                "validation_sites": site_losses,
                "validation_wall_seconds": perf_counter() - wall_start,
                "validation_cpu_seconds": process_time() - cpu_start,
                "validation_payload": {
                    "synchronization_phases": 1,
                    "summary_upload_count": len(validation_sites),
                    "model_broadcast_count": len(validation_sites),
                    "site_upload_bytes": len(validation_sites) * 24,
                    "server_broadcast_bytes": len(validation_sites) * int(parameters_to_vector(global_model.parameters()).numel()) * 4,
                },
            })
            # Always remember the actual minimum; min_delta affects only patience.
            if loss < best_loss:
                best_model, best_round, best_loss = deepcopy(global_model), round_index, loss
            if loss < stopping_reference - config.min_delta:
                stopping_reference, stale_rounds = loss, 0
            else:
                stale_rounds += 1
            item["stale_rounds"] = stale_rounds
            history.append(item)
            if round_index >= config.min_rounds and stale_rounds >= config.patience:
                termination_reason, completion_eligible = "validation_patience_satisfied", True
                break
        selected_rounds = best_round
        converged = completion_eligible
        if completion_eligible and config.refit_full_data:
            global_model = deepcopy(initial_model)
            refit_rounds = int(best_round)
            for round_index in range(1, refit_rounds + 1):
                global_model, item = _run_round(global_model, full_sites, config, seed, "refit", round_index)
                history.append(item)
            returned_phase = "full_data_refit"
        else:
            global_model = best_model
            returned_phase = "selected_holdout_checkpoint" if completion_eligible else "diagnostic_nonconverged_checkpoint"

    global_model.eval()
    final_hash = _model_fingerprint(global_model)
    vector_elements = int(parameters_to_vector(global_model.parameters()).numel())
    communication = _communication_report(history, len(raw_sites), vector_elements)
    per_site_updates = [sum(item["sites"][idx]["gradient_updates"] for item in history) for idx in range(len(raw_sites))]
    per_site_examples_seen = [sum(item["sites"][idx]["num_examples"] * config.local_epochs for item in history) for idx in range(len(raw_sites))]
    fingerprint_material = {
        "schema_version": 1,
        "model_name": model_name,
        "seed": seed,
        "config": asdict(config),
        "site_splits": split_audit,
        "initial_model_fingerprint": initial_hash,
        "history": _without_timings(history),
        "returned_model_fingerprint": final_hash,
        "termination_reason": termination_reason,
    }
    history_hash = hashlib.sha256(json.dumps(fingerprint_material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    timing = {
        "preprocessing_wall_seconds": preprocessing_wall,
        "preprocessing_cpu_seconds": preprocessing_cpu,
        "local_training_wall_seconds": sum(site["training_wall_seconds"] for item in history for site in item["sites"]),
        "local_training_cpu_seconds": sum(site["training_cpu_seconds"] for item in history for site in item["sites"]),
        "aggregation_wall_seconds": sum(item["aggregation_wall_seconds"] for item in history),
        "aggregation_cpu_seconds": sum(item["aggregation_cpu_seconds"] for item in history),
        "validation_wall_seconds": sum(item.get("validation_wall_seconds", 0.0) for item in history),
        "validation_cpu_seconds": sum(item.get("validation_cpu_seconds", 0.0) for item in history),
        "fit_wall_seconds": perf_counter() - total_wall_start,
        "fit_cpu_seconds": process_time() - total_cpu_start,
    }
    audit = {
        "schema_version": 1,
        "method": "multiround_fedavg_outcome_model",
        "model_name": model_name,
        "seed": seed,
        "config": asdict(config),
        "device": "cpu",
        "optimizer": "SGD",
        "momentum_policy": "reset_to_zero_at_each_site_at_each_round",
        "hyperparameter_policy": "prespecified_manifest_bound_not_claimed_tuned",
        "holdout_policy": "training_only_observed_class_stratified" if validation_sites else "none_fixed_training_budget",
        "validation_aggregation": "sum_site_cross_entropy_losses_divided_by_total_holdout_rows" if validation_sites else None,
        "site_splits": split_audit,
        "initial_model_fingerprint": initial_hash,
        "model_fingerprint": final_hash,
        "fit_fingerprint": history_hash,
        "history_fingerprint": history_hash,
        "round_history": history,
        "termination_reason": termination_reason,
        "completion_eligible": completion_eligible,
        "converged": converged,
        "convergence_definition": "prespecified_validation_patience_not_stationarity" if validation_sites else "not_claimed_fixed_budget",
        "safety_limit_reached": termination_reason == "maximum_rounds_reached",
        "best_selection_round": best_round,
        "best_validation_loss": best_loss if best_round is not None else None,
        "selected_rounds": selected_rounds,
        "refit_rounds": refit_rounds,
        "returned_model_phase": returned_phase,
        "training_rounds_executed": len(history),
        "total_local_epochs_per_site": len(history) * config.local_epochs,
        "gradient_updates_per_site": per_site_updates,
        "training_examples_seen_per_site": per_site_examples_seen,
        "total_gradient_updates": sum(per_site_updates),
        "communication": communication,
        "timing": timing,
    }
    return global_model, audit


def validate_fit_report(audit, config, model_name, seed, model_arrays):
    """Reject semantic tampering even if enclosing artifact hashes were updated."""
    if isinstance(config, dict):
        config = FedAvgConfig(**config)
    if not isinstance(config, FedAvgConfig):
        raise ValueError("invalid FedAvg validation configuration")
    try:
        if (audit["config"] != asdict(config) or audit["model_name"] != model_name or audit["seed"] != seed
                or audit["method"] != "multiround_fedavg_outcome_model" or audit["device"] != "cpu"
                or audit["completion_eligible"] is not True or audit["safety_limit_reached"] is not False
                or audit["momentum_policy"] != "reset_to_zero_at_each_site_at_each_round"):
            raise ValueError("FedAvg scientific identity or completion mismatch")
        history, sites = audit["round_history"], audit["site_splits"]
        if not history or not sites:
            raise ValueError("missing FedAvg round/site history")
        vector = np.concatenate([
            np.asarray(model_arrays[name]).reshape(-1)
            for index in range(len(model_arrays) // 2)
            for name in ("net.%d.weight" % (2 * index), "net.%d.bias" % (2 * index))
        ])
        actual_hash = hashlib.sha256(b"fedfisher-multiround-model-v1\0" + np.asarray(vector, dtype="<f4").tobytes()).hexdigest()
        if actual_hash != audit["model_fingerprint"]:
            raise ValueError("FedAvg model and fit fingerprint disagree")
        if config.stopping_mode == "fixed_total_epochs":
            prescribed = config.total_local_epochs // config.local_epochs
            if (len(history) != prescribed or prescribed > config.max_rounds
                    or any(item["phase"] != "fixed_budget" for item in history)
                    or audit["converged"] is not None or audit["termination_reason"] != "fixed_total_epochs_completed"
                    or audit["selected_rounds"] != prescribed or audit["best_selection_round"] is not None
                    or audit["refit_rounds"] != 0 or audit["returned_model_phase"] != "fixed_budget"):
                raise ValueError("invalid FedAvg fixed-epoch completion")
            selection = []
            returned_hash = history[-1]["aggregate_model_fingerprint"]
        else:
            selection = [item for item in history if item["phase"] == "selection"]
            refit = [item for item in history if item["phase"] == "refit"]
            if (not config.min_rounds <= len(selection) <= config.max_rounds
                    or history != selection + refit or audit["converged"] is not True
                    or audit["termination_reason"] != "validation_patience_satisfied"):
                raise ValueError("invalid FedAvg validation-patience completion")
            reference, stale = float("inf"), 0
            for index, item in enumerate(selection, start=1):
                val_sites = item["validation_sites"]
                loss = sum(site["loss_sum"] for site in val_sites) / sum(site["num_examples"] for site in val_sites)
                if item["validation_loss"] != loss or not math.isfinite(loss):
                    raise ValueError("FedAvg validation loss does not match site summaries")
                if [site["num_examples"] for site in val_sites] != [site["validation_examples"] for site in sites]:
                    raise ValueError("FedAvg validation row counts disagree")
                if loss < reference - config.min_delta:
                    reference, stale = loss, 0
                else:
                    stale += 1
                if item["stale_rounds"] != stale:
                    raise ValueError("FedAvg patience counter mismatch")
                satisfied = index >= config.min_rounds and stale >= config.patience
                if satisfied != (index == len(selection)):
                    raise ValueError("FedAvg stopped at the wrong validation round")
            best_round = min(range(len(selection)), key=lambda index: selection[index]["validation_loss"]) + 1
            expected_refit = best_round if config.refit_full_data else 0
            if (audit["best_selection_round"] != best_round or audit["selected_rounds"] != best_round
                    or audit["best_validation_loss"] != selection[best_round - 1]["validation_loss"]
                    or len(refit) != expected_refit or audit["refit_rounds"] != expected_refit
                    or audit["returned_model_phase"] != ("full_data_refit" if config.refit_full_data else "selected_holdout_checkpoint")):
                raise ValueError("FedAvg selected checkpoint or full-data refit mismatch")
            returned_hash = refit[-1]["aggregate_model_fingerprint"] if refit else selection[best_round - 1]["aggregate_model_fingerprint"]
        if returned_hash != actual_hash:
            raise ValueError("FedAvg returned the wrong round's model")
        phase, previous_hash, round_index = None, None, 0
        for item in history:
            if item["phase"] != phase:
                phase, previous_hash, round_index = item["phase"], audit["initial_model_fingerprint"], 0
            round_index += 1
            if item["round"] != round_index or item["initial_model_fingerprint"] != previous_hash:
                raise ValueError("FedAvg failed to rebroadcast the previous aggregate")
            previous_hash = item["aggregate_model_fingerprint"]
            counts = [site["fit_examples"] if phase == "selection" else site["num_examples"] for site in sites]
            if item["aggregation_weights"] != (np.asarray(counts, dtype=np.float64) / sum(counts)).tolist():
                raise ValueError("FedAvg aggregation weights disagree with fitted sample sizes")
            if len(item["sites"]) != len(sites):
                raise ValueError("FedAvg round omits a site")
            for site_idx, (entry, count) in enumerate(zip(item["sites"], counts)):
                expected_steps = config.local_epochs * math.ceil(count / config.batch_size)
                if (entry["site_idx"] != site_idx or entry["num_examples"] != count
                        or entry["local_epochs"] != config.local_epochs or entry["gradient_updates"] != expected_steps
                        or entry["initial_model_fingerprint"] != item["initial_model_fingerprint"]
                        or entry["loader_seed"] != _loader_seed(config, seed, site_idx, round_index, phase)):
                    raise ValueError("FedAvg local update schedule or broadcast mismatch")
            expected_payload = {"synchronization_phases": 1, "model_upload_count": len(sites),
                "model_broadcast_count": len(sites), "site_upload_bytes": len(sites) * (vector.size * 4 + 16),
                "server_broadcast_bytes": len(sites) * vector.size * 4}
            if item["optimization_payload"] != expected_payload:
                raise ValueError("FedAvg per-round payload mismatch")
        if (audit["communication"] != _communication_report(history, len(sites), vector.size)
                or audit["training_rounds_executed"] != len(history)
                or audit["total_local_epochs_per_site"] != len(history) * config.local_epochs):
            raise ValueError("FedAvg total communication or epoch count mismatch")
        updates = [sum(item["sites"][index]["gradient_updates"] for item in history) for index in range(len(sites))]
        seen = [sum(item["sites"][index]["num_examples"] * config.local_epochs for item in history) for index in range(len(sites))]
        if audit["gradient_updates_per_site"] != updates or audit["total_gradient_updates"] != sum(updates) or audit["training_examples_seen_per_site"] != seen:
            raise ValueError("FedAvg sample exposure or gradient count mismatch")
        material = dict(schema_version=1, model_name=model_name, seed=seed, config=asdict(config),
            site_splits=sites, initial_model_fingerprint=audit["initial_model_fingerprint"],
            history=_without_timings(history), returned_model_fingerprint=actual_hash,
            termination_reason=audit["termination_reason"])
        expected_hash = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if audit["history_fingerprint"] != expected_hash or audit["fit_fingerprint"] != expected_hash:
            raise ValueError("FedAvg deterministic fit history digest mismatch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in audit["timing"].values()):
            raise ValueError("invalid FedAvg timing observations")
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        raise ValueError("malformed FedAvg fit report") from error
