from __future__ import annotations

import copy
from typing import Callable, Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .training import (
    KFACLayerStats,
    TrainConfig,
    load_parameter_vector,
    parameter_vector,
    train_from_state,
    weighted_average_vectors,
)


def client_weights(client_datasets: Iterable[TensorDataset]) -> List[float]:
    sizes = [len(dataset) for dataset in client_datasets]
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("client datasets must not be empty")
    return [size / total for size in sizes]


def fedavg_vector(local_models: Iterable[nn.Module], weights: Iterable[float]) -> torch.Tensor:
    return weighted_average_vectors([parameter_vector(model).cpu() for model in local_models], weights)


def fedfisher_diag_vector(
    local_models: Iterable[nn.Module],
    fisher_diags: Iterable[torch.Tensor],
    weights: Iterable[float],
    damping: float,
    fallback_eps: float = 1e-12,
) -> torch.Tensor:
    """Diagonal Fisher weighted model merge.

    theta_j = (sum_c p_c F_cj theta_cj + damping * theta_fedavg_j)
              / (sum_c p_c F_cj + damping)

    The damping term is centered at FedAvg. If an undamped coordinate has
    effectively no Fisher mass, use FedAvg for that coordinate.
    """
    model_vectors = [parameter_vector(model).cpu() for model in local_models]
    fisher_vectors = [fisher.detach().cpu() for fisher in fisher_diags]
    weights_list = list(weights)
    if not (len(model_vectors) == len(fisher_vectors) == len(weights_list)):
        raise ValueError("local_models, fisher_diags, and weights must have the same length")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    numerator = torch.zeros_like(model_vectors[0])
    denominator = torch.zeros_like(model_vectors[0])
    for theta, fisher, weight in zip(model_vectors, fisher_vectors, weights_list):
        weighted_fisher = float(weight) * fisher
        numerator += weighted_fisher * theta
        denominator += weighted_fisher

    fedavg = weighted_average_vectors(model_vectors, weights_list)
    merged = (numerator + damping * fedavg) / (denominator + damping)
    no_information = denominator <= fallback_eps
    merged[no_information] = fedavg[no_information]
    return merged


def fedfisher_diag_adam_vector(
    local_models: Iterable[nn.Module],
    fisher_diags: Iterable[torch.Tensor],
    weights: Iterable[float],
    server_lr: float,
    steps: int,
    eval_every: int,
    validation_score_fn: Callable[[torch.Tensor], float] | None = None,
    damping: float = 0.0,
    adam_eps: float = 1e-2,
) -> Tuple[torch.Tensor, int, float]:
    """Original-style FedFisher-Diag server optimization.

    Starting from FedAvg, minimize the diagonal-Fisher quadratic objective with
    the same Adam-like update used by the reference implementation, optionally
    selecting the best iterate on a validation set.
    """
    model_vectors = [parameter_vector(model).cpu() for model in local_models]
    fisher_vectors = [fisher.detach().cpu() for fisher in fisher_diags]
    weights_list = list(weights)
    if not (len(model_vectors) == len(fisher_vectors) == len(weights_list)):
        raise ValueError("local_models, fisher_diags, and weights must have the same length")
    if not model_vectors:
        raise ValueError("at least one local model is required")
    if server_lr <= 0:
        raise ValueError("server_lr must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if adam_eps <= 0:
        raise ValueError("adam_eps must be positive")

    precision = torch.zeros_like(model_vectors[0])
    rhs = torch.zeros_like(model_vectors[0])
    for theta, fisher, weight in zip(model_vectors, fisher_vectors, weights_list):
        weighted_fisher = float(weight) * fisher
        precision += weighted_fisher
        rhs += weighted_fisher * theta

    fedavg = weighted_average_vectors(model_vectors, weights_list)
    candidate = fedavg.clone()
    momentum = torch.zeros_like(candidate)
    second_moment = torch.zeros_like(candidate)

    best_vector = candidate.clone()
    best_step = 0
    best_score = float("-inf")

    def maybe_update_best(step: int) -> None:
        nonlocal best_vector, best_step, best_score
        if validation_score_fn is None:
            best_vector = candidate.clone()
            best_step = step
            best_score = float("nan")
            return
        score = float(validation_score_fn(candidate))
        if score > best_score:
            best_score = score
            best_step = step
            best_vector = candidate.clone()

    maybe_update_best(0)
    for step in range(1, steps + 1):
        grad = precision * candidate - rhs
        if damping > 0:
            grad = grad + damping * (candidate - fedavg)
        momentum = grad + 0.9 * momentum
        second_moment = grad * grad + 0.99 * second_moment
        candidate = candidate - server_lr * momentum / (torch.sqrt(second_moment) + adam_eps)
        if step % eval_every == 0 or step == steps:
            maybe_update_best(step)

    return best_vector, best_step, best_score


def fedfisher_full_vector(
    local_models: Iterable[nn.Module],
    fisher_matrices: Iterable[torch.Tensor],
    weights: Iterable[float],
    damping: float,
) -> torch.Tensor:
    """Full Fisher merge for small models.

    The damping term is centered at FedAvg, so unidentifiable coordinates fall
    back to parameter averaging instead of being shrunk toward zero.
    """
    model_vectors = [parameter_vector(model).cpu() for model in local_models]
    fisher_list = [fisher.detach().cpu() for fisher in fisher_matrices]
    weights_list = list(weights)
    if not (len(model_vectors) == len(fisher_list) == len(weights_list)):
        raise ValueError("local_models, fisher_matrices, and weights must have the same length")
    if damping <= 0:
        raise ValueError("full Fisher damping must be positive")
    if not model_vectors:
        raise ValueError("at least one local model is required")

    num_params = model_vectors[0].numel()
    precision = torch.zeros(num_params, num_params, dtype=torch.float64)
    fedavg = weighted_average_vectors(model_vectors, weights_list).to(torch.float64)
    rhs = damping * fedavg

    for theta, fisher, weight in zip(model_vectors, fisher_list, weights_list):
        fisher64 = fisher.to(torch.float64)
        if fisher64.shape != (num_params, num_params):
            raise ValueError(f"expected Fisher shape {(num_params, num_params)}, got {tuple(fisher64.shape)}")
        weighted_fisher = float(weight) * fisher64
        precision += weighted_fisher
        rhs += weighted_fisher.matmul(theta.to(torch.float64))

    precision += damping * torch.eye(num_params, dtype=torch.float64)
    merged = torch.linalg.solve(precision, rhs)
    return merged.to(model_vectors[0].dtype)


def kfac_stats_scalar_count(stats: List[KFACLayerStats]) -> int:
    return int(sum(layer.activation_cov.numel() + layer.gradient_cov.numel() for layer in stats))


def fedfisher_kfac_vector(
    local_models: Iterable[nn.Module],
    kfac_stats: Iterable[List[KFACLayerStats]],
    weights: Iterable[float],
    damping: float,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Kronecker-factored Fisher merge for MLP-style linear layers."""
    models_list = list(local_models)
    stats_list = list(kfac_stats)
    weights_list = list(weights)
    if not (len(models_list) == len(stats_list) == len(weights_list)):
        raise ValueError("local_models, kfac_stats, and weights must have the same length")
    if not models_list:
        raise ValueError("at least one local model is required")
    if damping <= 0:
        raise ValueError("K-FAC damping must be positive")

    local_layers = [_linear_augmented_parameters(model) for model in models_list]
    num_layers = len(local_layers[0])
    if num_layers == 0:
        raise ValueError("K-FAC merge requires at least one linear layer")
    for layers in local_layers:
        if len(layers) != num_layers:
            raise ValueError("all local models must have the same number of linear layers")
    for client_stats in stats_list:
        if len(client_stats) != num_layers:
            raise ValueError("each client must provide one K-FAC stats object per linear layer")

    merged_parts = []
    for layer_idx in range(num_layers):
        layer_thetas = [layers[layer_idx].to(torch.float64) for layers in local_layers]
        fedavg_layer = _weighted_average_matrices(layer_thetas, weights_list)
        rhs = damping * fedavg_layer

        for theta, client_stats, weight in zip(layer_thetas, stats_list, weights_list):
            stats = client_stats[layer_idx]
            activation_cov = stats.activation_cov.to(torch.float64)
            gradient_cov = stats.gradient_cov.to(torch.float64)
            _validate_kfac_shapes(theta, activation_cov, gradient_cov, layer_idx)
            rhs += float(weight) * gradient_cov.matmul(theta).matmul(activation_cov)

        def matvec(candidate: torch.Tensor) -> torch.Tensor:
            out = damping * candidate
            for client_stats, weight in zip(stats_list, weights_list):
                stats = client_stats[layer_idx]
                activation_cov = stats.activation_cov.to(torch.float64)
                gradient_cov = stats.gradient_cov.to(torch.float64)
                out = out + float(weight) * gradient_cov.matmul(candidate).matmul(activation_cov)
            return out

        solution = _conjugate_gradient_matrix(
            matvec,
            rhs,
            initial=fedavg_layer,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        merged_parts.append(_flatten_augmented_parameter(solution, models_list[0], layer_idx))

    return torch.cat(merged_parts).to(parameter_vector(models_list[0]).dtype)


def fedfisher_kfac_adam_vector(
    local_models: Iterable[nn.Module],
    kfac_stats: Iterable[List[KFACLayerStats]],
    weights: Iterable[float],
    server_lr: float,
    steps: int,
    eval_every: int,
    validation_score_fn: Callable[[torch.Tensor], float] | None = None,
    damping: float = 0.0,
    adam_eps: float = 1e-2,
) -> Tuple[torch.Tensor, int, float]:
    """Original-style FedFisher K-FAC server optimization."""
    models_list = list(local_models)
    stats_list = list(kfac_stats)
    weights_list = list(weights)
    if not (len(models_list) == len(stats_list) == len(weights_list)):
        raise ValueError("local_models, kfac_stats, and weights must have the same length")
    if not models_list:
        raise ValueError("at least one local model is required")
    if server_lr <= 0:
        raise ValueError("server_lr must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if adam_eps <= 0:
        raise ValueError("adam_eps must be positive")

    local_layers = [_linear_augmented_parameters(model) for model in models_list]
    num_layers = len(local_layers[0])
    if num_layers == 0:
        raise ValueError("K-FAC merge requires at least one linear layer")
    for layers in local_layers:
        if len(layers) != num_layers:
            raise ValueError("all local models must have the same number of linear layers")
    for client_stats in stats_list:
        if len(client_stats) != num_layers:
            raise ValueError("each client must provide one K-FAC stats object per linear layer")

    fedavg_layers = []
    rhs_layers = []
    candidate_layers = []
    momentum_layers = []
    second_moment_layers = []
    for layer_idx in range(num_layers):
        layer_thetas = [layers[layer_idx].to(torch.float64) for layers in local_layers]
        fedavg_layer = _weighted_average_matrices(layer_thetas, weights_list)
        rhs = torch.zeros_like(fedavg_layer)
        for theta, client_stats, weight in zip(layer_thetas, stats_list, weights_list):
            stats = client_stats[layer_idx]
            activation_cov = stats.activation_cov.to(torch.float64)
            gradient_cov = stats.gradient_cov.to(torch.float64)
            _validate_kfac_shapes(theta, activation_cov, gradient_cov, layer_idx)
            rhs += float(weight) * gradient_cov.matmul(theta).matmul(activation_cov)
        fedavg_layers.append(fedavg_layer)
        rhs_layers.append(rhs)
        candidate_layers.append(fedavg_layer.clone())
        momentum_layers.append(torch.zeros_like(fedavg_layer))
        second_moment_layers.append(torch.zeros_like(fedavg_layer))

    def flatten_candidate() -> torch.Tensor:
        return torch.cat(
            [
                _flatten_augmented_parameter(candidate_layers[layer_idx], models_list[0], layer_idx)
                for layer_idx in range(num_layers)
            ]
        ).to(parameter_vector(models_list[0]).dtype)

    best_vector = flatten_candidate()
    best_step = 0
    best_score = float("-inf")

    def maybe_update_best(step: int) -> None:
        nonlocal best_vector, best_step, best_score
        vector = flatten_candidate()
        if validation_score_fn is None:
            best_vector = vector
            best_step = step
            best_score = float("nan")
            return
        score = float(validation_score_fn(vector))
        if score > best_score:
            best_score = score
            best_step = step
            best_vector = vector

    maybe_update_best(0)
    for step in range(1, steps + 1):
        for layer_idx in range(num_layers):
            candidate = candidate_layers[layer_idx]
            grad = torch.zeros_like(candidate)
            for client_stats, weight in zip(stats_list, weights_list):
                stats = client_stats[layer_idx]
                activation_cov = stats.activation_cov.to(torch.float64)
                gradient_cov = stats.gradient_cov.to(torch.float64)
                grad += float(weight) * gradient_cov.matmul(candidate).matmul(activation_cov)
            grad = grad - rhs_layers[layer_idx]
            if damping > 0:
                grad = grad + damping * (candidate - fedavg_layers[layer_idx])
            momentum_layers[layer_idx] = grad + 0.9 * momentum_layers[layer_idx]
            second_moment_layers[layer_idx] = grad * grad + 0.99 * second_moment_layers[layer_idx]
            candidate_layers[layer_idx] = (
                candidate
                - server_lr
                * momentum_layers[layer_idx]
                / (torch.sqrt(second_moment_layers[layer_idx]) + adam_eps)
            )
        if step % eval_every == 0 or step == steps:
            maybe_update_best(step)

    return best_vector, best_step, best_score


def _linear_layers(model: nn.Module) -> List[nn.Linear]:
    return [module for module in model.modules() if isinstance(module, nn.Linear)]


def _linear_augmented_parameters(model: nn.Module) -> List[torch.Tensor]:
    layers = []
    for layer in _linear_layers(model):
        weight = layer.weight.detach().cpu()
        if layer.bias is None:
            layers.append(weight)
        else:
            layers.append(torch.cat([weight, layer.bias.detach().cpu().reshape(-1, 1)], dim=1))
    return layers


def _flatten_augmented_parameter(augmented: torch.Tensor, reference_model: nn.Module, layer_idx: int) -> torch.Tensor:
    layer = _linear_layers(reference_model)[layer_idx]
    if layer.bias is None:
        return augmented.reshape(-1).detach().cpu()
    weight = augmented[:, : layer.in_features]
    bias = augmented[:, layer.in_features]
    return torch.cat([weight.reshape(-1), bias.reshape(-1)]).detach().cpu()


def _weighted_average_matrices(matrices: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    out = torch.zeros_like(matrices[0])
    for matrix, weight in zip(matrices, weights):
        out += float(weight) * matrix
    return out


def _validate_kfac_shapes(
    theta: torch.Tensor,
    activation_cov: torch.Tensor,
    gradient_cov: torch.Tensor,
    layer_idx: int,
) -> None:
    expected_activation = (theta.shape[1], theta.shape[1])
    expected_gradient = (theta.shape[0], theta.shape[0])
    if tuple(activation_cov.shape) != expected_activation:
        raise ValueError(
            f"layer {layer_idx} activation covariance has shape {tuple(activation_cov.shape)}, "
            f"expected {expected_activation}"
        )
    if tuple(gradient_cov.shape) != expected_gradient:
        raise ValueError(
            f"layer {layer_idx} gradient covariance has shape {tuple(gradient_cov.shape)}, "
            f"expected {expected_gradient}"
        )


def _conjugate_gradient_matrix(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial: torch.Tensor,
    max_iter: int,
    tolerance: float,
) -> torch.Tensor:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    x = initial.clone()
    residual = rhs - matvec(x)
    direction = residual.clone()
    residual_sq = torch.sum(residual * residual)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(1.0)
    target = tolerance * rhs_norm

    if torch.sqrt(residual_sq) <= target:
        return x

    for _ in range(max_iter):
        matvec_direction = matvec(direction)
        denom = torch.sum(direction * matvec_direction)
        if torch.abs(denom) <= 1e-30:
            break
        step = residual_sq / denom
        x = x + step * direction
        residual = residual - step * matvec_direction
        new_residual_sq = torch.sum(residual * residual)
        if torch.sqrt(new_residual_sq) <= target:
            break
        direction = residual + (new_residual_sq / residual_sq) * direction
        residual_sq = new_residual_sq
    return x


def materialize_model(model_factory: Callable[[], nn.Module], vector: torch.Tensor) -> nn.Module:
    model = model_factory()
    return load_parameter_vector(model, vector)


def train_multiround_fedavg(
    model_factory: Callable[[], nn.Module],
    initial_state: Dict[str, torch.Tensor],
    client_datasets: List[TensorDataset],
    train_config: TrainConfig,
    rounds: int,
    round_epochs: int,
    seed: int,
) -> nn.Module:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if round_epochs <= 0:
        raise ValueError("round_epochs must be positive")

    weights = client_weights(client_datasets)
    global_state = copy.deepcopy(initial_state)
    round_config = TrainConfig(
        epochs=round_epochs,
        batch_size=train_config.batch_size,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        optimizer=train_config.optimizer,
        device=train_config.device,
    )

    for round_idx in range(rounds):
        local_models = []
        for client_idx, dataset in enumerate(client_datasets):
            local_model = train_from_state(
                model_factory(),
                global_state,
                dataset,
                round_config,
                seed=seed * 100_000 + round_idx * 1_000 + client_idx,
            )
            local_models.append(local_model.cpu())

        averaged = fedavg_vector(local_models, weights)
        global_model = model_factory()
        load_parameter_vector(global_model, averaged)
        global_state = {key: value.detach().clone() for key, value in global_model.state_dict().items()}

    final_model = model_factory()
    final_model.load_state_dict(global_state)
    return final_model
