from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (64, 32),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden_dims(raw: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("hidden_dims must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("hidden_dims must be positive")
    return values


def build_model(model_type: str, input_dim: int, hidden_dims: Tuple[int, ...] = (64, 32)) -> nn.Module:
    if model_type == "lr":
        return LogisticRegression(input_dim=input_dim)
    if model_type == "mlp":
        return MLP(input_dim=input_dim, hidden_dims=hidden_dims)
    raise ValueError(f"Unknown model_type {model_type!r}. Choices: 'lr', 'mlp'.")
