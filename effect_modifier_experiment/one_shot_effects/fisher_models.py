"""CPU outcome models and exact binary categorical-Fisher primitives.

These primitives receive observed rows only. They do not own files, scheduling,
truth, validation sets or any hospital-to-server communication.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from models import get_model, SYNTHETIC_MLP_HIDDEN_DIMS


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode() + value.dtype.str.encode() + value.tobytes()).hexdigest()


def make_model(model_name, covariate_dim, seed):
    if model_name not in SYNTHETIC_MLP_HIDDEN_DIMS or type(covariate_dim) is not int or covariate_dim < 1:
        raise ValueError("unsupported MLP architecture or raw-X dimension")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("initialization seed must be a nonnegative integer below 2**63")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("common initialization requires the repository float32 default")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model = get_model(model_name, 2, synthetic_dim=1 + 2 * covariate_dim)
    return model.to(device="cpu", dtype=torch.float64)


def model_vector(model):
    vector = parameters_to_vector(model.parameters()).detach().cpu().numpy().copy()
    if vector.dtype != np.float64 or not np.isfinite(vector).all():
        raise ValueError("FedFisher-EM requires finite float64 CPU parameters")
    return vector


def model_from_vector(model_name, covariate_dim, vector):
    vector = np.asarray(vector)
    model = make_model(model_name, covariate_dim, 0)
    if vector.ndim != 1 or vector.size != sum(p.numel() for p in model.parameters()) or vector.dtype != np.float64 or not np.isfinite(vector).all():
        raise ValueError("parameter vector does not match the declared model")
    vector_to_parameters(torch.tensor(vector, dtype=torch.float64, device="cpu"), model.parameters())
    return model.eval()


def input_coordinates(x, a):
    x, a = np.asarray(x, dtype=np.float64), np.asarray(a, dtype=np.float64)
    if x.ndim != 2 or a.shape != (len(x),) or not np.isfinite(x).all() or not np.isin(a, [0., 1.]).all():
        raise ValueError("finite raw X and observed binary A are required")
    return np.column_stack((a, x, a[:, None] * x))


def predict_tau(model, x, batch_size=256):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or type(batch_size) is not int or batch_size < 1:
        raise ValueError("finite two-dimensional raw X and a positive batch size are required")
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            rows = x[start:start + batch_size]
            treated = torch.tensor(input_coordinates(rows, np.ones(len(rows))), dtype=torch.float64, device="cpu")
            control = torch.tensor(input_coordinates(rows, np.zeros(len(rows))), dtype=torch.float64, device="cpu")
            outputs.append((torch.softmax(model(treated), dim=1)[:, 1] - torch.softmax(model(control), dim=1)[:, 1]).numpy())
    result = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite outcome-model treatment contrast")
    return result


def tau_and_gradient(model, raw_row):
    row = np.asarray(raw_row, dtype=np.float64).reshape(1, -1)
    treated = torch.tensor(input_coordinates(row, np.ones(1)), dtype=torch.float64, device="cpu")
    control = torch.tensor(input_coordinates(row, np.zeros(1)), dtype=torch.float64, device="cpu")
    model.eval()
    tau = torch.softmax(model(treated), dim=1)[0, 1] - torch.softmax(model(control), dim=1)[0, 1]
    gradients = torch.autograd.grad(tau, tuple(model.parameters()), create_graph=False)
    vector = parameters_to_vector(gradients).detach().numpy()
    value = float(tau.detach())
    if not math.isfinite(value) or not np.isfinite(vector).all():
        raise FloatingPointError("nonfinite treatment-effect gradient")
    return value, vector


def tau_and_directional_derivatives(model, raw_x, directions):
    """Exact autograd directional derivatives, without a D-by-D matrix."""
    x, directions = np.asarray(raw_x, dtype=np.float64), np.asarray(directions, dtype=np.float64)
    if x.ndim != 2 or directions.ndim != 2 or directions.shape[0] != sum(p.numel() for p in model.parameters()):
        raise ValueError("raw rows and parameter directions have incompatible dimensions")
    if not np.isfinite(x).all() or not np.isfinite(directions).all():
        raise ValueError("finite raw rows and directions are required")
    tau, projected = np.empty(len(x)), np.empty((len(x), directions.shape[1]))
    for index, row in enumerate(x):
        tau[index], gradient = tau_and_gradient(model, row)
        projected[index] = gradient @ directions
    return tau, projected


def mean_categorical_fisher_diag(model, train):
    """F = mean_T p0*p1*(grad(logit1-logit0))**2, not empirical-label Fisher."""
    features = input_coordinates(train.x, train.a)
    if len(features) < 1:
        raise ValueError("Fisher needs nonempty training rows")
    fisher = np.zeros(sum(p.numel() for p in model.parameters()), dtype=np.float64)
    model.eval()
    for row in features:
        logits = model(torch.tensor(row[None], dtype=torch.float64, device="cpu"))[0]
        probabilities = torch.softmax(logits, dim=0).detach()
        gradients = torch.autograd.grad(logits[1] - logits[0], tuple(model.parameters()), create_graph=False)
        gradient = parameters_to_vector(gradients).detach().numpy()
        fisher += float(probabilities[0] * probabilities[1]) * gradient * gradient
    fisher /= len(features)
    if not np.isfinite(fisher).all() or np.any(fisher < 0):
        raise FloatingPointError("invalid mean categorical Fisher diagonal")
    return fisher


@dataclass
class OutcomeEffectModel:
    model_name: str
    covariate_dim: int
    model: nn.Module

    def predict_tau(self, x):
        return predict_tau(self.model, x)


def model_to_arrays(model):
    if not isinstance(model, OutcomeEffectModel):
        raise TypeError("expected an outcome-effect model")
    return {"theta": model_vector(model.model)}


def restore_model(arrays, metadata):
    if set(arrays) != {"theta"}:
        raise ValueError("FedFisher-EM predictor artifact requires only theta")
    name, dimension = metadata["model_name"], metadata["covariate_dim"]
    return OutcomeEffectModel(name, dimension, model_from_vector(name, dimension, arrays["theta"]))
