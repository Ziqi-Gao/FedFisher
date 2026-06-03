from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-2
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    device: str = "auto"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")


@dataclass(frozen=True)
class KFACLayerStats:
    activation_cov: torch.Tensor
    gradient_cov: torch.Tensor


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)


def train_model(
    model: nn.Module,
    dataset: TensorDataset,
    config: TrainConfig,
    seed: int,
) -> nn.Module:
    config.validate()
    device = resolve_device(config.device)
    model.to(device)
    model.train()
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    optimizer = build_optimizer(model, config)

    for _ in range(config.epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, dataset: TensorDataset, batch_size: int, device: str) -> Dict[str, float]:
    resolved = resolve_device(device)
    model.to(resolved)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(resolved)
        y_batch = y_batch.to(resolved)
        logits = model(x_batch)
        total_loss += F.cross_entropy(logits, y_batch, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.numel()
    return {
        "accuracy": correct / total,
        "loss": total_loss / total,
    }


def parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def load_parameter_vector(model: nn.Module, vector: torch.Tensor) -> nn.Module:
    torch.nn.utils.vector_to_parameters(vector.detach().clone(), model.parameters())
    return model


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def weighted_average_vectors(vectors: Iterable[torch.Tensor], weights: Iterable[float]) -> torch.Tensor:
    vectors_list = list(vectors)
    weights_list = list(weights)
    if len(vectors_list) != len(weights_list):
        raise ValueError("vectors and weights must have the same length")
    if not vectors_list:
        raise ValueError("at least one vector is required")
    out = torch.zeros_like(vectors_list[0])
    for vector, weight in zip(vectors_list, weights_list):
        out += float(weight) * vector
    return out


def _flat_from_param_dict(param_dict: Dict[str, torch.Tensor], names: List[str]) -> torch.Tensor:
    return torch.cat([param_dict[name].reshape(-1) for name in names])


def empirical_fisher_diag(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Compute the exact empirical diagonal Fisher with per-sample gradients."""
    resolved = resolve_device(device)
    model.to(resolved)
    model.eval()

    try:
        from torch.func import functional_call, grad, vmap
    except ImportError:
        return _empirical_fisher_diag_loop(model, dataset, resolved)

    params = {name: param.detach() for name, param in model.named_parameters()}
    param_names = list(params.keys())
    fisher_parts = {name: torch.zeros_like(param, device=resolved) for name, param in params.items()}

    def loss_one(param_values: Dict[str, torch.Tensor], x_one: torch.Tensor, y_one: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, param_values, (x_one.unsqueeze(0),))
        return F.cross_entropy(logits, y_one.unsqueeze(0), reduction="sum")

    grad_one = grad(loss_one)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(resolved)
        y_batch = y_batch.to(resolved)
        grads = vmap(grad_one, in_dims=(None, 0, 0))(params, x_batch, y_batch)
        for name in param_names:
            fisher_parts[name] += grads[name].pow(2).sum(dim=0)
        total += y_batch.numel()

    fisher_flat = _flat_from_param_dict(fisher_parts, param_names) / float(total)
    return fisher_flat.detach().cpu()


def empirical_fisher_full(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Compute the exact empirical Fisher matrix with per-sample gradients."""
    resolved = resolve_device(device)
    model.to(resolved)
    model.eval()

    try:
        from torch.func import functional_call, grad, vmap
    except ImportError:
        return _empirical_fisher_full_loop(model, dataset, resolved)

    params = {name: param.detach() for name, param in model.named_parameters()}
    param_names = list(params.keys())
    num_params = sum(param.numel() for param in params.values())
    fisher = torch.zeros(num_params, num_params, dtype=torch.float64, device=resolved)

    def loss_one(param_values: Dict[str, torch.Tensor], x_one: torch.Tensor, y_one: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, param_values, (x_one.unsqueeze(0),))
        return F.cross_entropy(logits, y_one.unsqueeze(0), reduction="sum")

    grad_one = grad(loss_one)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(resolved)
        y_batch = y_batch.to(resolved)
        grads = vmap(grad_one, in_dims=(None, 0, 0))(params, x_batch, y_batch)
        flat_grads = torch.cat([grads[name].reshape(y_batch.numel(), -1) for name in param_names], dim=1)
        flat_grads = flat_grads.to(torch.float64)
        fisher += flat_grads.transpose(0, 1).matmul(flat_grads)
        total += y_batch.numel()

    return fisher.div(float(total)).detach().cpu()


def empirical_kfac_stats(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: str,
) -> List[KFACLayerStats]:
    """Compute empirical K-FAC factors for every linear layer.

    For a linear layer with augmented input a and pre-activation gradient g,
    the layer Fisher block is approximated as E[g g^T] kron E[a a^T].
    """
    resolved = resolve_device(device)
    model.to(resolved)
    model.eval()
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("K-FAC stats require at least one nn.Linear layer")

    activation_covs = [
        torch.zeros(layer.in_features + (1 if layer.bias is not None else 0),
                    layer.in_features + (1 if layer.bias is not None else 0),
                    dtype=torch.float64,
                    device=resolved)
        for layer in linear_layers
    ]
    gradient_covs = [
        torch.zeros(layer.out_features, layer.out_features, dtype=torch.float64, device=resolved)
        for layer in linear_layers
    ]
    activations: Dict[int, torch.Tensor] = {}
    gradients: Dict[int, torch.Tensor] = {}
    handles = []

    def save_activation(layer_idx: int):
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            activations[layer_idx] = inputs[0].detach()

        return hook

    def save_gradient(layer_idx: int):
        def hook(
            _module: nn.Module,
            _grad_input: Tuple[torch.Tensor, ...],
            grad_output: Tuple[torch.Tensor, ...],
        ) -> None:
            gradients[layer_idx] = grad_output[0].detach()

        return hook

    for idx, layer in enumerate(linear_layers):
        handles.append(layer.register_forward_hook(save_activation(idx)))
        handles.append(layer.register_full_backward_hook(save_gradient(idx)))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    try:
        for x_batch, y_batch in loader:
            activations.clear()
            gradients.clear()
            x_batch = x_batch.to(resolved)
            y_batch = y_batch.to(resolved)
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_batch), y_batch, reduction="sum")
            loss.backward()

            batch_size_actual = y_batch.numel()
            for idx, layer in enumerate(linear_layers):
                if idx not in activations or idx not in gradients:
                    raise RuntimeError(f"Missing K-FAC hook values for linear layer {idx}")
                a = activations[idx]
                if layer.bias is not None:
                    ones = torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)
                    a = torch.cat([a, ones], dim=1)
                g = gradients[idx]
                a64 = a.to(torch.float64)
                g64 = g.to(torch.float64)
                activation_covs[idx] += a64.transpose(0, 1).matmul(a64)
                gradient_covs[idx] += g64.transpose(0, 1).matmul(g64)
            total += batch_size_actual
    finally:
        for handle in handles:
            handle.remove()

    return [
        KFACLayerStats(
            activation_cov=activation_cov.div(float(total)).detach().cpu(),
            gradient_cov=gradient_cov.div(float(total)).detach().cpu(),
        )
        for activation_cov, gradient_cov in zip(activation_covs, gradient_covs)
    ]


def _empirical_fisher_diag_loop(model: nn.Module, dataset: TensorDataset, device: torch.device) -> torch.Tensor:
    fisher = [torch.zeros_like(param, device=device) for param in model.parameters()]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total = 0
    for x_one, y_one in loader:
        x_one = x_one.to(device)
        y_one = y_one.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_one), y_one, reduction="sum")
        loss.backward()
        for idx, param in enumerate(model.parameters()):
            if param.grad is not None:
                fisher[idx] += param.grad.detach().pow(2)
        total += 1
    return torch.cat([part.reshape(-1) for part in fisher]).div(float(total)).detach().cpu()


def _empirical_fisher_full_loop(model: nn.Module, dataset: TensorDataset, device: torch.device) -> torch.Tensor:
    num_params = parameter_count(model)
    fisher = torch.zeros(num_params, num_params, dtype=torch.float64, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total = 0
    for x_one, y_one in loader:
        x_one = x_one.to(device)
        y_one = y_one.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_one), y_one, reduction="sum")
        loss.backward()
        grad_parts = []
        for param in model.parameters():
            if param.grad is None:
                grad_parts.append(torch.zeros_like(param).reshape(-1))
            else:
                grad_parts.append(param.grad.detach().reshape(-1))
        grad_flat = torch.cat(grad_parts).to(torch.float64)
        fisher += torch.outer(grad_flat, grad_flat)
        total += 1
    return fisher.div(float(total)).detach().cpu()


def train_from_state(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    dataset: TensorDataset,
    config: TrainConfig,
    seed: int,
) -> nn.Module:
    model.load_state_dict(copy.deepcopy(state_dict))
    return train_model(model, dataset, config, seed=seed)
