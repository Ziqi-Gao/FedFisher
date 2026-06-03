from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset


DEFAULT_PRIORS: Dict[str, Sequence[float]] = {
    "iid": (0.5, 0.5, 0.5, 0.5, 0.5),
    "mild": (0.30, 0.40, 0.50, 0.60, 0.70),
}


@dataclass(frozen=True)
class SyntheticDataConfig:
    num_train: int = 10_000
    num_test: int = 10_000
    dim: int = 100
    signal_dim: int = 10
    signal_strength: float = 0.7
    noise_std: float = 1.0
    num_clients: int = 5
    dirichlet_alpha: float = 0.1

    def validate(self) -> None:
        if self.num_train <= 0 or self.num_test <= 0:
            raise ValueError("num_train and num_test must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if not 1 <= self.signal_dim <= self.dim:
            raise ValueError("signal_dim must be between 1 and dim")
        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")


def theoretical_bayes_accuracy(config: SyntheticDataConfig) -> float:
    """Bayes accuracy for the balanced two-Gaussian test distribution."""
    z = config.signal_strength / config.noise_std
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def make_sparse_mu(dim: int, signal_dim: int, signal_strength: float) -> np.ndarray:
    """Create a sparse mean vector with fixed L2 norm on signal coordinates."""
    if not 1 <= signal_dim <= dim:
        raise ValueError("signal_dim must be between 1 and dim")
    mu = np.zeros(dim, dtype=np.float32)
    mu[:signal_dim] = signal_strength / np.sqrt(signal_dim)
    return mu


def client_sizes(num_train: int, num_clients: int) -> List[int]:
    base = num_train // num_clients
    remainder = num_train % num_clients
    return [base + (1 if i < remainder else 0) for i in range(num_clients)]


def priors_for_split(split: str, num_clients: int) -> Sequence[float]:
    if split not in DEFAULT_PRIORS:
        raise ValueError(f"Unknown fixed-prior split {split!r}. Choices: {sorted(DEFAULT_PRIORS)}")
    priors = DEFAULT_PRIORS[split]
    if len(priors) != num_clients:
        raise ValueError(
            f"Split {split!r} defines {len(priors)} client priors, "
            f"but num_clients={num_clients}. Use num_clients=5 or add a new split."
        )
    return priors


def dirichlet_indices_by_class(
    y: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> List[np.ndarray]:
    """Split indices with the same class-wise Dirichlet scheme as the reference code."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    for _ in range(max_attempts):
        client_class_preferences = rng.dirichlet(np.repeat(alpha, num_classes), size=num_clients)
        batches: List[List[int]] = [[] for _ in range(num_clients)]

        for class_idx in range(num_classes):
            class_indices = np.where(y == class_idx)[0]
            rng.shuffle(class_indices)
            proportions = client_class_preferences[:, class_idx]
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            for client_batch, split_indices in zip(batches, np.split(class_indices, split_points)):
                client_batch.extend(split_indices.tolist())

        if all(batches):
            out = []
            for batch in batches:
                batch_array = np.array(batch, dtype=np.int64)
                rng.shuffle(batch_array)
                out.append(batch_array)
            return out

    raise RuntimeError("Dirichlet split produced an empty client repeatedly; try a larger alpha")


def sample_binary_gaussian(
    num_samples: int,
    positive_prior: float,
    mu: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample exactly rounded class counts from the sparse Gaussian model."""
    if not 0.0 <= positive_prior <= 1.0:
        raise ValueError("positive_prior must be in [0, 1]")
    num_pos = int(round(num_samples * positive_prior))
    num_pos = min(max(num_pos, 0), num_samples)
    y = np.zeros(num_samples, dtype=np.int64)
    y[:num_pos] = 1
    rng.shuffle(y)

    signs = (2 * y - 1).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(num_samples, mu.shape[0])).astype(np.float32)
    x = signs[:, None] * mu[None, :] + noise
    return x.astype(np.float32), y


def to_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())


def make_synthetic_datasets(
    config: SyntheticDataConfig,
    split: str,
    seed: int,
) -> Tuple[List[TensorDataset], TensorDataset, TensorDataset, Dict[str, object]]:
    """Create client, pooled, and balanced test datasets for one experiment seed."""
    config.validate()
    rng = np.random.default_rng(seed)
    mu = make_sparse_mu(config.dim, config.signal_dim, config.signal_strength)

    client_datasets: List[TensorDataset] = []
    pooled_x: List[np.ndarray] = []
    pooled_y: List[np.ndarray] = []
    class_counts = []

    if split == "noniid":
        x_pool, y_pool = sample_binary_gaussian(config.num_train, 0.5, mu, config.noise_std, rng)
        index_batches = dirichlet_indices_by_class(
            y_pool,
            num_clients=config.num_clients,
            num_classes=2,
            alpha=config.dirichlet_alpha,
            rng=rng,
        )
        split_strategy = f"dirichlet_alpha_{config.dirichlet_alpha:g}"
        for indices in index_batches:
            x_client = x_pool[indices]
            y_client = y_pool[indices]
            client_datasets.append(to_tensor_dataset(x_client, y_client))
            pooled_x.append(x_client)
            pooled_y.append(y_client)
            class_counts.append(
                {
                    "negative": int((y_client == 0).sum()),
                    "positive": int((y_client == 1).sum()),
                    "positive_prior": float((y_client == 1).mean()),
                }
            )
    else:
        priors = priors_for_split(split, config.num_clients)
        sizes = client_sizes(config.num_train, config.num_clients)
        split_strategy = "fixed_priors"
        for size, prior in zip(sizes, priors):
            x_client, y_client = sample_binary_gaussian(size, prior, mu, config.noise_std, rng)
            client_datasets.append(to_tensor_dataset(x_client, y_client))
            pooled_x.append(x_client)
            pooled_y.append(y_client)
            class_counts.append(
                {
                    "negative": int((y_client == 0).sum()),
                    "positive": int((y_client == 1).sum()),
                    "positive_prior": float((y_client == 1).mean()),
                }
            )

    x_pool = np.concatenate(pooled_x, axis=0)
    y_pool = np.concatenate(pooled_y, axis=0)
    pooled_dataset = to_tensor_dataset(x_pool, y_pool)

    x_test, y_test = sample_binary_gaussian(config.num_test, 0.5, mu, config.noise_std, rng)
    test_dataset = to_tensor_dataset(x_test, y_test)

    metadata: Dict[str, object] = {
        "mu_norm": float(np.linalg.norm(mu)),
        "bayes_accuracy": theoretical_bayes_accuracy(config),
        "split_strategy": split_strategy,
        "dirichlet_alpha": config.dirichlet_alpha if split == "noniid" else None,
        "client_sizes": [len(dataset) for dataset in client_datasets],
        "client_priors": [float(counts["positive_prior"]) for counts in class_counts],
        "client_class_counts": class_counts,
    }
    return client_datasets, pooled_dataset, test_dataset, metadata
