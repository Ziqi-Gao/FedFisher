"""Matched zero-penalty one-shot FedAvg reference for the new CPU path.

This reference intentionally does not claim bitwise equivalence to the legacy
GPU ``LocalUpdate`` path, whose shuffled DataLoader consumes the process-global
PyTorch RNG.  It independently implements the same full-site CPU SGD loop used
by the new benchmark, with the same explicit site loader seed and no proximal
step.  Tests require the explicit-interaction group path at lambda zero to be
exactly equal to this reference, isolating the penalty within the new path.
"""

from typing import Mapping, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from effect_modifier_experiment.one_shot_lasso_benchmark import (
    MlpGroupLassoConfig,
    MlpSitePayload,
    SealedTrainingSite,
)
from models import get_model


REFERENCE_NAME = "matched zero-penalty one-shot FedAvg on deterministic CPU path"


def final_refit_loader_seed(
    config: MlpGroupLassoConfig, experiment_seed: int, site_idx: int
) -> int:
    return (
        int(config.loader_seed_offset)
        + int(experiment_seed) * 1009
        + int(site_idx) * 100003
        + 900001
    )


def train_matched_zero_penalty_model(
    model_name: str,
    input_dim: int,
    common_state: Mapping[str, torch.Tensor],
    features: np.ndarray,
    labels: np.ndarray,
    config: MlpGroupLassoConfig,
    loader_seed: int,
) -> nn.Module:
    """Run the independent no-prox local SGD reference on one full site."""
    if model_name not in {"SyntheticMLP", "SyntheticMLPDeep"}:
        raise ValueError("unsupported MLP model")
    model = get_model(model_name, 2, bias=False, synthetic_dim=input_dim).to("cpu")
    model.load_state_dict(common_state, strict=True)
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
    return model


def fit_matched_zero_penalty_site(
    site: SealedTrainingSite,
    model_name: str,
    common_state: Mapping[str, torch.Tensor],
    config: MlpGroupLassoConfig,
    experiment_seed: int,
) -> Tuple[MlpSitePayload, dict]:
    """Fit, upload once, and seal a site under the matched reference."""
    features, labels = site.arrays()
    input_dim = 1 + 2 * int(config.covariate_dim)
    if features.shape[1] != input_dim:
        raise ValueError("matched FedAvg site has the wrong input dimension")
    loader_seed = final_refit_loader_seed(config, experiment_seed, site.site_idx)
    model = train_matched_zero_penalty_model(
        model_name,
        input_dim,
        common_state,
        features,
        labels,
        config,
        loader_seed,
    )
    vector = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    payload = MlpSitePayload(site.site_idx, site.num_examples, vector)
    audit = {
        "reference_name": REFERENCE_NAME,
        "site_idx": site.site_idx,
        "num_examples": site.num_examples,
        "device": "cpu",
        "loader_rng": "explicit_site_torch_Generator",
        "loader_seed": loader_seed,
        "local_epochs": config.local_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "proximal_steps": 0,
        "lambda": 0.0,
        "payload_elements": int(vector.size),
        "payload_bytes": int(vector.nbytes + 16),
        "pre_upload_training_communication_rounds": 0,
    }
    site.seal()
    return payload, audit
