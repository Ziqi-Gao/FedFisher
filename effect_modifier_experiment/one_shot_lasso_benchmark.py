"""One-round effect-modifier LASSO baselines.

This module contains two additive, single-process cross-silo simulations:

* site-local logistic LASSO followed by sample-count weighted coefficient
  averaging; and
* site-local MLP explicit A-by-X input group regularization followed by the repository's existing
  one-shot FedAvg parameter averaging.

All optimization and hyperparameter selection occurs before a site exports its
single immutable payload.  Server aggregation accepts payloads only and has no
training-site callback or data argument.  These are benchmark simulations, not
deployment privacy or disclosure-control implementations.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from effect_modifier_experiment.dslassocov_benchmark import (
    OptimizerConfig,
    RawSite,
)
from effect_modifier_experiment.one_shot_local_logistic import (
    SMOOTHING_RULE,
    separation_safe_local_logistic_cv,
)
from models import get_model
from run_one_shot_algs import get_fedavg_model


FAMILIES = (
    "one_shot_local_logistic_lasso_fedavg",
    "one_shot_mlp_group_lasso_fedavg",
)
VARIANT_CONTRACT = {
    "control": {"site_gaussian_shift": False, "intervention_scope": "global"},
    "site_gaussian_shift": {
        "site_gaussian_shift": True,
        "intervention_scope": "global",
    },
    "local_permutation": {
        "site_gaussian_shift": False,
        "intervention_scope": "site",
    },
    "site_gaussian_local_permutation": {
        "site_gaussian_shift": True,
        "intervention_scope": "site",
    },
}


class SealedTrainingSite:
    """Training rows that become inaccessible immediately after one upload."""

    __slots__ = (
        "site_idx",
        "_features",
        "_labels",
        "_num_examples",
        "_sealed",
        "_data_accesses",
    )

    def __init__(self, site_idx: int, features: np.ndarray, labels: np.ndarray):
        features = np.asarray(features, dtype=np.float64, order="C")
        labels = np.asarray(labels, dtype=np.int64, order="C")
        if features.ndim != 2 or labels.shape != (features.shape[0],):
            raise ValueError("site features and labels have incompatible shapes")
        if features.shape[0] < 1:
            raise ValueError("a training site cannot be empty")
        if not np.isfinite(features).all() or not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("site rows must be finite with binary labels")
        self.site_idx = int(site_idx)
        self._features = features.copy(order="C")
        self._labels = labels.copy(order="C")
        self._num_examples = int(features.shape[0])
        self._sealed = False
        self._data_accesses = 0

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def data_accesses(self) -> int:
        return self._data_accesses

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._sealed:
            raise RuntimeError("training rows are inaccessible after payload upload")
        self._data_accesses += 1
        return self._features, self._labels

    def seal(self) -> None:
        if self._sealed:
            raise RuntimeError("training site was already sealed")
        self._sealed = True


def tensor_datasets_to_sealed_sites(
    site_datasets: Sequence[TensorDataset], covariate_dim: int
) -> List[SealedTrainingSite]:
    expected_dim = 1 + 2 * int(covariate_dim)
    sites = []
    for site_idx, dataset in enumerate(site_datasets):
        if not hasattr(dataset, "tensors") or len(dataset.tensors) != 2:
            raise TypeError("training sites must be two-tensor TensorDataset objects")
        features = dataset.tensors[0].detach().cpu().numpy()
        labels = dataset.tensors[1].detach().cpu().numpy()
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                "site %d should have %d input features" % (site_idx, expected_dim)
            )
        sites.append(SealedTrainingSite(site_idx, features, labels))
    if not sites:
        raise ValueError("at least one training site is required")
    return sites


def training_fingerprint(sites: Sequence[SealedTrainingSite]) -> str:
    digest = hashlib.sha256()
    digest.update(b"fedfisher-one-shot-training-v1\0")
    for site in sites:
        features, labels = site.arrays()
        digest.update(np.asarray([site.site_idx], dtype="<i8").tobytes())
        digest.update(np.asarray(features.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(features, dtype="<f8", order="C").tobytes())
        digest.update(np.asarray(labels, dtype="<i8", order="C").tobytes())
    return digest.hexdigest()


def _immutable_array(values, dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy(order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LinearSitePayload:
    site_idx: int
    num_examples: int
    coefficients: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "site_idx", int(self.site_idx))
        object.__setattr__(self, "num_examples", int(self.num_examples))
        object.__setattr__(
            self, "coefficients", _immutable_array(self.coefficients, np.float64)
        )


@dataclass(frozen=True)
class MlpSitePayload:
    site_idx: int
    num_examples: int
    model_vector: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "site_idx", int(self.site_idx))
        object.__setattr__(self, "num_examples", int(self.num_examples))
        object.__setattr__(
            self, "model_vector", _immutable_array(self.model_vector, np.float32)
        )


@dataclass(frozen=True)
class LocalLogisticConfig:
    covariate_dim: int = 100
    cv_folds: int = 5
    lambda_count: int = 30
    lambda_min_ratio: float = 1e-3
    cv_seed_offset: int = 4000037
    scale_floor: float = 1e-12
    optimizer_max_iter: int = 2000
    optimizer_tolerance: float = 1e-6
    optimizer_initial_step: float = 1.0
    optimizer_backtrack_factor: float = 0.5
    optimizer_min_step: float = 1e-12


@dataclass(frozen=True)
class MlpGroupLassoConfig:
    covariate_dim: int = 100
    local_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    validation_fraction: float = 0.2
    lambda_ratios: Tuple[float, ...] = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    split_seed_offset: int = 5100037
    loader_seed_offset: int = 5200037


class OneShotLinearLogisticModel(nn.Module):
    def __init__(self, coefficients: Sequence[float]):
        super().__init__()
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim != 1 or coefficients.size < 2:
            raise ValueError("linear model requires intercept plus feature coefficients")
        self.intercept = nn.Parameter(
            torch.tensor(float(coefficients[0]), dtype=torch.float32),
            requires_grad=False,
        )
        self.coefficients = nn.Parameter(
            torch.as_tensor(coefficients[1:], dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, inputs):
        score = inputs.matmul(self.coefficients) + self.intercept
        return torch.stack((torch.zeros_like(score), score), dim=1)


def _validate_site_sequence(payloads, expected_dim: int, field: str, dtype) -> None:
    if not payloads:
        raise ValueError("at least one site payload is required")
    indices = [payload.site_idx for payload in payloads]
    if len(indices) != len(set(indices)) or sorted(indices) != list(range(len(indices))):
        raise ValueError("site payload indices must be unique and contiguous from zero")
    for payload in payloads:
        if payload.num_examples <= 0:
            raise ValueError("site payload sample counts must be positive")
        values = getattr(payload, field)
        if values.shape != (expected_dim,) or values.dtype != np.dtype(dtype):
            raise ValueError("site payload has the wrong vector shape or dtype")
        if values.flags.writeable or not np.isfinite(values).all():
            raise ValueError("site payload vectors must be immutable and finite")


def fit_local_logistic_lasso_site(
    site: SealedTrainingSite,
    config: LocalLogisticConfig,
    experiment_seed: int,
) -> Tuple[LinearSitePayload, Dict[str, object]]:
    features, labels = site.arrays()
    expected_dim = 1 + 2 * config.covariate_dim
    if features.shape[1] != expected_dim:
        raise ValueError("local logistic site has the wrong feature dimension")
    raw_site = RawSite(
        features=np.asarray(features, dtype=np.float64, order="C"),
        labels=np.asarray(labels, dtype=np.float64, order="C"),
        site_idx=site.site_idx,
    )
    optimizer = OptimizerConfig(
        max_iter=config.optimizer_max_iter,
        tolerance=config.optimizer_tolerance,
        initial_step=config.optimizer_initial_step,
        backtrack_factor=config.optimizer_backtrack_factor,
        min_step=config.optimizer_min_step,
    )
    fit = separation_safe_local_logistic_cv(
        raw_site,
        covariate_dim=config.covariate_dim,
        num_folds=config.cv_folds,
        lambda_count=config.lambda_count,
        lambda_min_ratio=config.lambda_min_ratio,
        cv_seed=config.cv_seed_offset + int(experiment_seed) + site.site_idx * 100003,
        scale_floor=config.scale_floor,
        optimizer=optimizer,
    )
    selected_index = int(fit["cv"]["selected_index"])
    selected_fold_reports = [
        fold["lambda_fits"][selected_index] for fold in fit["cv"]["fold_reports"]
    ]
    selected_final_report = fit["optimizer"]["selected_lambda_fit"]
    if not all(report["converged"] for report in selected_fold_reports):
        raise RuntimeError("site-local selected lambda failed a CV convergence check")
    if not selected_final_report["converged"]:
        raise RuntimeError("site-local selected lambda failed final convergence")
    coefficients = np.concatenate(
        ([fit["original_intercept"]], np.asarray(fit["original_coefficients"]))
    )
    payload = LinearSitePayload(site.site_idx, site.num_examples, coefficients)
    audit = {
        "site_idx": site.site_idx,
        "num_examples": site.num_examples,
        "fit_scope": "site_local_only",
        "cv_scope": "site_local_training_rows_only",
        "standardization_scope": "site_local_training_rows_only",
        "response_stabilization": fit["response_stabilization"],
        "selected_lambda": float(fit["cv"]["selected_lambda"]),
        "lambda_path": [float(value) for value in fit["cv"]["lambda_path"]],
        "aggregated_validation_losses": [
            float(value) for value in fit["cv"]["aggregated_validation_losses"]
        ],
        "selected_index": selected_index,
        "selected_cv_reports": selected_fold_reports,
        "selected_final_report": selected_final_report,
        "payload_elements": int(coefficients.size),
        "payload_bytes": int(coefficients.nbytes + 16),
        "pre_upload_training_communication_rounds": 0,
    }
    site.seal()
    return payload, audit


def aggregate_linear_payloads(
    payloads: Sequence[LinearSitePayload], covariate_dim: int
) -> Tuple[np.ndarray, Dict[str, object]]:
    expected_dim = 2 + 2 * int(covariate_dim)
    _validate_site_sequence(payloads, expected_dim, "coefficients", np.float64)
    counts = np.asarray([payload.num_examples for payload in payloads], dtype=np.float64)
    weights = counts / counts.sum()
    coefficients = np.zeros(expected_dim, dtype=np.float64)
    for weight, payload in zip(weights, payloads):
        coefficients += weight * payload.coefficients
    report = _communication_report(
        payloads,
        vector_elements=expected_dim,
        scalar_bytes=8,
        aggregation="strict_sample_count_weighted_arithmetic_mean",
    )
    report["weights"] = weights.tolist()
    return coefficients, report


def _stratified_holdout(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        if indices.size == 1:
            train_indices.extend(indices.tolist())
            continue
        validation_count = max(1, int(round(indices.size * validation_fraction)))
        validation_count = min(validation_count, indices.size - 1)
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())
    if not train_indices or not validation_indices:
        raise ValueError("site-local holdout requires at least two total rows")
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(
        validation_indices, dtype=np.int64
    )


def group_penalty(model: nn.Module, covariate_dim: int) -> torch.Tensor:
    weights = model.net[0].weight[:, 1 + covariate_dim : 1 + 2 * covariate_dim]
    return torch.linalg.vector_norm(weights, ord=2, dim=0).sum()


def apply_interaction_group_prox(
    model: nn.Module, covariate_dim: int, threshold: float
) -> Dict[str, object]:
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("group proximal threshold must be finite and nonnegative")
    first_weight = model.net[0].weight
    start = 1 + int(covariate_dim)
    end = 1 + 2 * int(covariate_dim)
    if first_weight.ndim != 2 or end > first_weight.shape[1]:
        raise ValueError("model first layer is incompatible with the interaction mask")
    with torch.no_grad():
        groups = first_weight[:, start:end]
        norms = torch.linalg.vector_norm(groups, ord=2, dim=0)
        if threshold == 0.0:
            factors = torch.ones_like(norms)
        else:
            factors = torch.clamp(1.0 - threshold / torch.clamp(norms, min=1e-30), min=0.0)
        groups.mul_(factors.unsqueeze(0))
        zero_groups = int(torch.count_nonzero(norms <= threshold).item())
    return {
        "penalty_start_input": start,
        "penalty_end_input_exclusive": end,
        "num_groups": int(covariate_dim),
        "zeroed_groups_this_step": zero_groups,
    }


def _clone_common_model(model_name: str, input_dim: int, common_state):
    model = get_model(model_name, 2, bias=False, synthetic_dim=input_dim).to("cpu")
    model.load_state_dict(common_state, strict=True)
    return model


def _local_gradient_scale(
    model_name: str,
    input_dim: int,
    covariate_dim: int,
    common_state,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    model = _clone_common_model(model_name, input_dim, common_state)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    model.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(x), y).backward()
    gradients = model.net[0].weight.grad[
        :, 1 + covariate_dim : 1 + 2 * covariate_dim
    ]
    value = float(torch.linalg.vector_norm(gradients, ord=2, dim=0).max().item())
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("could not derive a positive local group-gradient scale")
    return value


def _train_group_lasso_model(
    model_name: str,
    input_dim: int,
    common_state,
    features: np.ndarray,
    labels: np.ndarray,
    lambda_value: float,
    config: MlpGroupLassoConfig,
    loader_seed: int,
) -> nn.Module:
    model = _clone_common_model(model_name, input_dim, common_state)
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(loader_seed))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )
    model.train()
    for _ in range(config.local_epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            apply_interaction_group_prox(
                model,
                config.covariate_dim,
                config.learning_rate * float(lambda_value),
            )
    return model


def _validation_losses(
    model: nn.Module, features: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        losses = nn.functional.cross_entropy(
            model(torch.as_tensor(features, dtype=torch.float32)),
            torch.as_tensor(labels, dtype=torch.long),
            reduction="none",
        ).cpu().numpy()
    mean = float(np.mean(losses))
    standard_error = (
        float(np.std(losses, ddof=1) / math.sqrt(losses.size))
        if losses.size > 1
        else 0.0
    )
    return mean, standard_error


def fit_mlp_group_lasso_site(
    site: SealedTrainingSite,
    model_name: str,
    common_state: Mapping[str, torch.Tensor],
    config: MlpGroupLassoConfig,
    experiment_seed: int,
) -> Tuple[MlpSitePayload, Dict[str, object]]:
    if model_name not in {"SyntheticMLP", "SyntheticMLPDeep"}:
        raise ValueError("unsupported MLP model")
    if config.local_epochs < 1 or config.batch_size < 1:
        raise ValueError("local_epochs and batch_size must be positive")
    if config.learning_rate <= 0.0 or not 0.0 <= config.momentum < 1.0:
        raise ValueError("invalid SGD learning rate or momentum")
    ratios = tuple(float(value) for value in config.lambda_ratios)
    if not ratios or ratios[0] != 0.0 or any(
        not math.isfinite(value) or value < 0.0 for value in ratios
    ) or tuple(sorted(set(ratios))) != ratios:
        raise ValueError("lambda ratios must be unique increasing values starting at zero")
    features, labels = site.arrays()
    input_dim = 1 + 2 * config.covariate_dim
    if features.shape[1] != input_dim:
        raise ValueError("MLP site has the wrong input dimension")
    train_indices, validation_indices = _stratified_holdout(
        labels,
        config.validation_fraction,
        config.split_seed_offset + int(experiment_seed) + site.site_idx * 100003,
    )
    scale = _local_gradient_scale(
        model_name,
        input_dim,
        config.covariate_dim,
        common_state,
        features[train_indices],
        labels[train_indices],
    )
    candidates = []
    for ratio in ratios:
        lambda_value = ratio * scale
        model = _train_group_lasso_model(
            model_name,
            input_dim,
            common_state,
            features[train_indices],
            labels[train_indices],
            lambda_value,
            config,
            config.loader_seed_offset
            + int(experiment_seed) * 1009
            + site.site_idx * 100003,
        )
        validation_loss, validation_se = _validation_losses(
            model, features[validation_indices], labels[validation_indices]
        )
        group_norms = (
            model.net[0]
            .weight[:, 1 + config.covariate_dim : 1 + 2 * config.covariate_dim]
            .detach()
            .norm(dim=0)
            .cpu()
            .numpy()
        )
        candidates.append(
            {
                "ratio": ratio,
                "lambda": lambda_value,
                "validation_loss": validation_loss,
                "validation_standard_error": validation_se,
                "zero_group_count": int(np.count_nonzero(group_norms <= 1e-12)),
                "group_norm_max": float(np.max(group_norms)),
            }
        )
    minimum_index = min(
        range(len(candidates)), key=lambda idx: candidates[idx]["validation_loss"]
    )
    one_se_threshold = (
        candidates[minimum_index]["validation_loss"]
        + candidates[minimum_index]["validation_standard_error"]
    )
    eligible = [
        idx
        for idx, candidate in enumerate(candidates)
        if candidate["validation_loss"] <= one_se_threshold + 1e-12
    ]
    selected_index = max(eligible, key=lambda idx: candidates[idx]["ratio"])
    selected = candidates[selected_index]
    final_model = _train_group_lasso_model(
        model_name,
        input_dim,
        common_state,
        features,
        labels,
        selected["lambda"],
        config,
        config.loader_seed_offset
        + int(experiment_seed) * 1009
        + site.site_idx * 100003
        + 900001,
    )
    vector = parameters_to_vector(final_model.parameters()).detach().cpu().numpy()
    payload = MlpSitePayload(site.site_idx, site.num_examples, vector)
    final_group_norms = (
        final_model.net[0]
        .weight[:, 1 + config.covariate_dim : 1 + 2 * config.covariate_dim]
        .detach()
        .norm(dim=0)
        .cpu()
        .numpy()
    )
    audit = {
        "site_idx": site.site_idx,
        "num_examples": site.num_examples,
        "fit_scope": "site_local_only",
        "lambda_selection_scope": "deterministic_site_local_observed_class_stratified_holdout",
        "lambda_scale_rule": "maximum interaction-group CE gradient norm at common initialization",
        "holdout_original_label_counts": {
            "0": int(np.count_nonzero(labels == 0)),
            "1": int(np.count_nonzero(labels == 1)),
        },
        "holdout_training_examples": int(train_indices.size),
        "holdout_validation_examples": int(validation_indices.size),
        "local_gradient_scale": scale,
        "lambda_candidates": candidates,
        "selection_rule": "largest lambda within one standard error of minimum validation CE",
        "selected_index": selected_index,
        "selected_ratio": selected["ratio"],
        "selected_lambda": selected["lambda"],
        "final_refit_loader_seed": (
            config.loader_seed_offset
            + int(experiment_seed) * 1009
            + site.site_idx * 100003
            + 900001
        ),
        "final_zero_group_count": int(np.count_nonzero(final_group_norms <= 1e-12)),
        "final_group_norms": final_group_norms.tolist(),
        "payload_elements": int(vector.size),
        "payload_bytes": int(vector.nbytes + 16),
        "pre_upload_training_communication_rounds": 0,
    }
    site.seal()
    return payload, audit


def make_common_mlp_state(model_name: str, input_dim: int, seed: int):
    torch.manual_seed(int(seed))
    model = get_model(model_name, 2, bias=False, synthetic_dim=input_dim).to("cpu")
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def aggregate_mlp_payloads(
    payloads: Sequence[MlpSitePayload],
    model_name: str,
    input_dim: int,
    common_state: Mapping[str, torch.Tensor],
) -> Tuple[nn.Module, Dict[str, object]]:
    template = _clone_common_model(model_name, input_dim, common_state)
    expected_dim = int(parameters_to_vector(template.parameters()).numel())
    _validate_site_sequence(payloads, expected_dim, "model_vector", np.float32)
    counts = np.asarray([payload.num_examples for payload in payloads], dtype=np.float64)
    weights = counts / counts.sum()
    vectors = [torch.from_numpy(np.array(payload.model_vector, copy=True)) for payload in payloads]
    model = get_fedavg_model(
        expected_dim,
        len(payloads),
        weights,
        {"device": "cpu"},
        template,
        vectors,
    )
    report = _communication_report(
        payloads,
        vector_elements=expected_dim,
        scalar_bytes=4,
        aggregation="repository_get_fedavg_model_strict_sample_count_weighted_mean",
    )
    report["weights"] = weights.tolist()
    return model, report


def _communication_report(
    payloads: Sequence[object],
    vector_elements: int,
    scalar_bytes: int,
    aggregation: str,
) -> Dict[str, object]:
    per_site_bytes = int(vector_elements * scalar_bytes + 16)
    return {
        "training_communication_rounds": 1,
        "site_upload_count": len(payloads),
        "server_broadcast_count": len(payloads),
        "post_upload_training_site_accesses": 0,
        "payload_vector_elements_per_site": int(vector_elements),
        "payload_bytes_per_site": per_site_bytes,
        "total_site_upload_bytes": per_site_bytes * len(payloads),
        "payload_forbidden_fields": ["gradient", "hessian", "fisher", "training_rows", "site_callback"],
        "server_aggregation": aggregation,
    }


def evaluation_communication_report(intervention_scope: str, num_sites: int):
    if intervention_scope == "global":
        return {
            "evaluation_communication_rounds": 0,
            "model_broadcasts": 0,
            "site_score_uploads": 0,
            "training_site_accesses": 0,
        }
    if intervention_scope == "site":
        return {
            "evaluation_communication_rounds": 1,
            "model_broadcasts": int(num_sites),
            "site_score_uploads": int(num_sites),
            "training_site_accesses": 0,
            "evaluation_payload": "site-local scalar feature scores and model summary only",
        }
    raise ValueError("intervention scope must be global or site")


def payload_fingerprint(payloads: Iterable[object]) -> str:
    digest = hashlib.sha256()
    digest.update(b"fedfisher-one-shot-payload-v1\0")
    for payload in payloads:
        digest.update(np.asarray([payload.site_idx, payload.num_examples], dtype="<i8").tobytes())
        vector = getattr(payload, "coefficients", None)
        if vector is None:
            vector = payload.model_vector
        digest.update(np.asarray(vector.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(vector).tobytes())
    return digest.hexdigest()
