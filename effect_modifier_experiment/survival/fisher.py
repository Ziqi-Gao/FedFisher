"""One-upload neural survival adapter with a Gaussian working Fisher.

The local objective is mean((PV - exp(-exp(eta)))**2)/(2*sigma2).
PV need not lie in [0, 1].  The curvature is the expected Fisher of a
*working*, independent N(mu, sigma2) model: E_x[J_mu J_mu.T]/sigma2.
It is not a likelihood or covariance model for dependent survival pseudo-values.
Diag is its exact diagonal; KFAC replaces each layer block by G kron A.

The server minimizes sum_k (n_k/N)/2 * (w-w_k)' F_k (w-w_k)
                       + ridge/2 * ||w-w_public||**2.
This is a separately labelled survival adapter, not the legacy categorical
Fisher implementation or its validation-selected numerical solver.  There is
no pooled validation data, compression, or test-set tuning.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
import math
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


ARCHITECTURES = {"shallow": (64, 32), "deep": (256, 128, 64, 32)}
PAYLOAD_SCHEMA = "survival-working-gaussian-fisher-v1"


@dataclass(frozen=True)
class FisherConfig:
    architecture: str = "shallow"
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    working_variance: float = 1.0
    server_ridge: float = 1e-6
    server_max_iterations: int = 2000
    server_tolerance: float = 1e-8

    def validate(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Unknown survival neural architecture")
        for name in ("epochs", "batch_size", "server_max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "working_variance", "server_ridge", "server_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.momentum, bool) or not isinstance(self.momentum, (int, float)) or not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0,1)")


def _mean(eta):
    # exp(-exp(eta)) has already underflowed at eta=40 in float64.
    # Saturation avoids an otherwise undefined autograd product inf * 0.
    return torch.exp(-torch.exp(torch.clamp(eta, max=40.0)))


def _shapes(input_dim, architecture):
    dims = (input_dim,) + ARCHITECTURES[architecture] + (1,)
    return tuple((b, a) for a, b in zip(dims[:-1], dims[1:]))


def _model(input_dim, config, seed=0):
    config.validate()
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError("input_dim must be a positive integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        layers = []
        shapes = _shapes(input_dim, config.architecture)
        for i, (out_dim, in_dim) in enumerate(shapes):
            layers.append(nn.Linear(in_dim, out_dim, dtype=torch.float64))
            if i < len(shapes) - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)


def public_initial_vector(input_dim, config, seed, initial_eta):
    """Common random weights and a public baseline log cumulative-hazard bias."""
    if not math.isfinite(initial_eta):
        raise ValueError("initial_eta must be finite and specified independently of outcomes")
    model = _model(input_dim, config, seed)
    with torch.no_grad():
        model[-1].bias.fill_(initial_eta)
    return parameters_to_vector(model.parameters()).detach().numpy().copy()


def _vector(value, size=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or (size is not None and value.size != size) or not np.isfinite(value).all():
        raise ValueError("Invalid finite model vector")
    return value


def _load_vector(model, vector):
    size = sum(p.numel() for p in model.parameters())
    value = _vector(vector, size)
    vector_to_parameters(torch.from_numpy(value.copy()), model.parameters())


def _initial_hash(vector):
    return sha256(np.asarray(vector, dtype="<f8").tobytes()).hexdigest()


def _layer_matrices(vector, shapes):
    offset = 0
    for out_dim, in_dim in shapes:
        end = offset + out_dim * in_dim
        yield np.column_stack((vector[offset:end].reshape(out_dim, in_dim),
                               vector[end:end + out_dim]))
        offset = end + out_dim
    if offset != len(vector):
        raise ValueError("Vector does not match architecture")


def _pack_layer_matrices(matrices):
    parts = []
    for value in matrices:
        parts.extend((value[:, :-1].ravel(), value[:, -1]))
    return np.concatenate(parts)


def working_fisher(model, design, *, method, variance=1.0, batch_size=256):
    """Exact per-example Jacobian diagonal, or layerwise KFAC factors.

    A includes the bias coordinate. G uses d(mu)/d(preactivation), not
    residual scores and not categorical-output Fisher. Both factors are means
    over the same site; variance is included once, in G.
    """
    if method not in ("diag", "kfac") or not math.isfinite(variance) or variance <= 0:
        raise ValueError("Invalid Fisher method/variance")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Fisher batch_size must be a positive integer")
    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or len(design) < 1 or not np.isfinite(design).all():
        raise ValueError("Fisher requires a finite nonempty design")
    modules = list(model)
    if (not modules or len(modules) % 2 != 1
            or any(not isinstance(layer, nn.Linear if i % 2 == 0 else nn.ReLU) for i, layer in enumerate(modules))):
        raise ValueError("Working Fisher supports alternating Linear/ReLU with a scalar Linear output")
    layers = modules[::2]
    if layers[-1].out_features != 1 or any(layer.bias is None for layer in layers):
        raise ValueError("Working Fisher requires scalar output and layer biases")
    if any(p.dtype != torch.float64 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Working Fisher requires CPU float64 parameters")
    if design.shape[1] != layers[0].in_features:
        raise ValueError("Fisher input dimension mismatch")
    diagonals = [torch.zeros((l.out_features, l.in_features + 1), dtype=torch.float64) for l in layers]
    activations = [torch.zeros((l.in_features + 1, l.in_features + 1), dtype=torch.float64) for l in layers]
    gradients = [torch.zeros((l.out_features, l.out_features), dtype=torch.float64) for l in layers]
    with torch.no_grad():
        for start in range(0, len(design), batch_size):
            h = torch.from_numpy(design[start:start + batch_size].copy())
            augmented, preacts = [], []
            for i, layer in enumerate(layers):
                augmented.append(torch.cat((h, torch.ones((len(h), 1), dtype=h.dtype)), dim=1))
                z = layer(h)
                preacts.append(z)
                h = torch.relu(z) if i < len(layers) - 1 else z
            hazard = torch.exp(torch.clamp(h, max=40.0))
            delta = -hazard * torch.exp(-hazard)
            for i in range(len(layers) - 1, -1, -1):
                a = augmented[i]
                if method == "diag":
                    diagonals[i] += delta.square().T @ a.square()
                else:
                    activations[i] += a.T @ a
                    gradients[i] += delta.T @ delta
                if i:
                    delta = (delta @ layers[i].weight) * (preacts[i - 1] > 0)
    if method == "diag":
        result = {"diag": _pack_layer_matrices([v.numpy() / (len(design) * variance) for v in diagonals])}
    else:
        result = {}
        for i, (a, g) in enumerate(zip(activations, gradients)):
            result[f"a{i}"] = a.numpy() / len(design)
            result[f"g{i}"] = g.numpy() / (len(design) * variance)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite working Fisher")
    return result


def _serialize(vector, curvature, n, method, input_dim, config, initial):
    meta = {"schema": PAYLOAD_SCHEMA, "method": method, "input_dim": input_dim,
            "architecture": config.architecture, "working_variance": config.working_variance,
            "initial_sha256": _initial_hash(initial)}
    stream = BytesIO()
    np.savez(stream, metadata=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
             vector=vector, n=np.array(n, dtype=np.int64), **curvature)
    return stream.getvalue()


def fit_local_fisher(design, pseudo_values, *, initial_vector, config, seed, method):
    """Site-only computation. Returns one opaque upload and local diagnostics."""
    config.validate()
    if method not in ("diag", "kfac"):
        raise ValueError("method must be diag or kfac")
    design = np.asarray(design, dtype=np.float64)
    y = np.asarray(pseudo_values, dtype=np.float64)
    if design.ndim != 2 or y.shape != (len(design),) or len(y) < 1:
        raise ValueError("Invalid local design/response shape")
    if not np.isfinite(design).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite local design/response")
    model = _model(design.shape[1], config)
    _load_vector(model, initial_vector)
    x, target = torch.from_numpy(design.copy()), torch.from_numpy(y.copy())
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    started = time.perf_counter()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    updates = 0
    with torch.no_grad():
        initial_loss = float((_mean(model(x).squeeze(-1)) - target).square().mean() / (2 * config.working_variance))
    for _ in range(config.epochs):
        permutation = torch.randperm(len(x), generator=rng)
        for start in range(0, len(x), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            optimizer.zero_grad()
            loss = (_mean(model(x[indices]).squeeze(-1)) - target[indices]).square().mean() / (2 * config.working_variance)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite survival training objective")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite survival training gradient")
            optimizer.step()
            updates += 1
    vector = _vector(parameters_to_vector(model.parameters()).detach().numpy().copy())
    with torch.no_grad():
        final_loss = float((_mean(model(x).squeeze(-1)) - target).square().mean() / (2 * config.working_variance))
    if not math.isfinite(final_loss):
        raise FloatingPointError("Nonfinite final objective")
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    curvature = working_fisher(model, design, method=method, variance=config.working_variance)
    curvature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    upload = _serialize(vector, curvature, len(y), method, design.shape[1], config, initial_vector)
    serialization_seconds = time.perf_counter() - started
    return upload, {"sample_count": len(y), "optimizer_updates": updates, "epochs": config.epochs,
                    "stop_reason": "fixed_epoch_budget", "initial_loss": initial_loss,
                    "final_loss": final_loss, "training_seconds": training_seconds,
                    "curvature_seconds": curvature_seconds, "serialization_seconds": serialization_seconds,
                    "upload_bytes": len(upload), "upload_sha256": sha256(upload).hexdigest(), "uploads": 1}


def _read_upload(blob, initial_vector, config, input_dim, method):
    if not isinstance(blob, bytes):
        raise ValueError("The aggregation server accepts serialized summaries only")
    shapes = _shapes(input_dim, config.architecture)
    curvature_keys = {"diag"} if method == "diag" else {f"{letter}{i}" for i in range(len(shapes)) for letter in ("a", "g")}
    with np.load(BytesIO(blob), allow_pickle=False) as stored:
        if len(stored.files) != len(set(stored.files)) or set(stored.files) != {"metadata", "vector", "n"} | curvature_keys:
            raise ValueError("Unexpected payload fields")
        meta = json.loads(stored["metadata"].tobytes().decode())
        expected = {"schema": PAYLOAD_SCHEMA, "method": method, "input_dim": input_dim,
                    "architecture": config.architecture, "working_variance": config.working_variance,
                    "initial_sha256": _initial_hash(initial_vector)}
        if meta != expected:
            raise ValueError("Fisher payload recipe/initialization mismatch")
        vector = _vector(stored["vector"], len(initial_vector)).copy()
        count = stored["n"]
        if count.shape != () or count.dtype.kind not in "iu" or int(count) < 1:
            raise ValueError("Invalid site sample count")
        curvature = {key: stored[key].copy() for key in curvature_keys}
    if any(value.dtype != np.float64 or not np.isfinite(value).all() for value in curvature.values()):
        raise ValueError("Invalid Fisher values")
    if method == "diag":
        if curvature["diag"].shape != vector.shape or np.any(curvature["diag"] < 0):
            raise ValueError("Invalid Fisher diagonal")
    else:
        for i, (out_dim, in_dim) in enumerate(shapes):
            for key, size in ((f"a{i}", in_dim + 1), (f"g{i}", out_dim)):
                factor = curvature[key]
                if factor.shape != (size, size) or not np.allclose(factor, factor.T, rtol=1e-12, atol=1e-14):
                    raise ValueError("Invalid KFAC factor shape/symmetry")
                if np.linalg.eigvalsh(factor)[0] < -1e-10 * max(1.0, np.linalg.norm(factor)):
                    raise ValueError("Indefinite KFAC factor")
    return vector, int(count), curvature


def _curvature_mv(vector, curvature, shapes, method):
    if method == "diag":
        return vector * curvature["diag"]
    return _pack_layer_matrices([curvature[f"g{i}"] @ w @ curvature[f"a{i}"]
                                for i, w in enumerate(_layer_matrices(vector, shapes))])


def aggregate_fisher(uploads: Sequence[bytes], *, initial_vector, input_dim, config, method):
    """Pure server: only serialized counts, parameters and Fisher summaries."""
    config.validate()
    if method not in ("diag", "kfac") or not uploads:
        raise ValueError("Invalid aggregation method/uploads")
    initial = _vector(initial_vector)
    shapes = _shapes(input_dim, config.architecture)
    if sum(o * (i + 1) for o, i in shapes) != len(initial):
        raise ValueError("Initialization dimension mismatch")
    started = time.perf_counter()
    decoded = [_read_upload(blob, initial, config, input_dim, method) for blob in uploads]
    counts = np.array([item[1] for item in decoded], dtype=np.float64)
    weights = counts / counts.sum()
    ridge = config.server_ridge

    def mv(vector):
        result = ridge * vector
        for weight, (_, _, curvature) in zip(weights, decoded):
            result = result + weight * _curvature_mv(vector, curvature, shapes, method)
        return result

    rhs = ridge * initial
    for weight, (vector, _, curvature) in zip(weights, decoded):
        rhs = rhs + weight * _curvature_mv(vector, curvature, shapes, method)
    iterations = 0
    if method == "diag":
        precision = ridge + sum(weight * item[2]["diag"] for weight, item in zip(weights, decoded))
        result = rhs / precision
        status = "closed_form"
    else:
        result = sum(weight * item[0] for weight, item in zip(weights, decoded))
        residual = rhs - mv(result)
        direction = residual.copy()
        rr = float(residual @ residual)
        threshold = config.server_tolerance * max(float(np.linalg.norm(rhs)), 1.0)
        status = "residual_tolerance" if math.sqrt(rr) <= threshold else "iteration_budget"
        for _ in range(config.server_max_iterations):
            if status == "residual_tolerance":
                break
            ad = mv(direction)
            denom = float(direction @ ad)
            if not math.isfinite(denom) or denom <= 0:
                raise FloatingPointError("Nonpositive/nonfinite CG curvature")
            step = rr / denom
            result = result + step * direction
            # Explicit residual avoids claiming convergence from recurrence drift.
            residual = rhs - mv(result)
            next_rr = float(residual @ residual)
            iterations += 1
            if math.sqrt(next_rr) <= threshold:
                status = "residual_tolerance"
                break
            direction = residual + (next_rr / rr) * direction
            rr = next_rr
    result = _vector(result, len(initial))
    residual_norm = float(np.linalg.norm(mv(result) - rhs))
    threshold = config.server_tolerance * max(float(np.linalg.norm(rhs)), 1.0)
    if not math.isfinite(residual_norm):
        raise FloatingPointError("Nonfinite aggregate residual")
    objective = ridge * float(np.square(result - initial).sum()) / 2
    for weight, (vector, _, curvature) in zip(weights, decoded):
        delta = result - vector
        objective += weight * float(delta @ _curvature_mv(delta, curvature, shapes, method)) / 2
    if not math.isfinite(objective):
        raise FloatingPointError("Nonfinite aggregate objective")
    report = {"solver": "closed_form" if method == "diag" else "conjugate_gradient",
              "stop_reason": status, "converged": residual_norm <= threshold,
              "iterations": iterations, "max_iterations": config.server_max_iterations,
              "residual_norm": residual_norm, "residual_threshold": threshold,
              "objective": objective, "ridge": ridge, "sample_counts": counts.astype(int).tolist(),
              "aggregation_seconds": time.perf_counter() - started,
              "upload_bytes": sum(len(blob) for blob in uploads), "uploads": len(uploads),
              "client_to_server_upload_phases": 1, "outcome_dependent_server_broadcasts": 0}
    return result.copy(), report


@dataclass(frozen=True)
class FisherFit:
    vector: np.ndarray
    config: FisherConfig
    input_dim: int
    diagnostics: dict
    uploads: tuple
    _prediction_model: object = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        self.config.validate()
        size = sum(o * (i + 1) for o, i in _shapes(self.input_dim, self.config.architecture))
        vector = _vector(self.vector, size)
        # A bytes-backed copy cannot be made writable by setflags(write=True).
        object.__setattr__(self, "vector", np.frombuffer(vector.astype("<f8").tobytes(), dtype="<f8"))
        object.__setattr__(self, "uploads", tuple(self.uploads))

    def predict_eta(self, design):
        design = np.asarray(design, dtype=np.float64)
        if design.ndim != 2 or design.shape[1] != self.input_dim or not np.isfinite(design).all():
            raise ValueError("Invalid prediction design")
        if self._prediction_model is None:
            model = _model(self.input_dim, self.config)
            _load_vector(model, self.vector)
            model.eval()
            model.requires_grad_(False)
            object.__setattr__(self, "_prediction_model", model)
        model = self._prediction_model
        with torch.no_grad():
            prediction = model(torch.from_numpy(design.copy())).squeeze(-1).numpy()
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite prediction")
        return prediction

    def contrast(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("Raw covariates must be a matrix")
        return self.predict_eta(np.column_stack((np.ones(len(x)), x, x))) - self.predict_eta(
            np.column_stack((np.zeros(len(x)), x, np.zeros_like(x))))


def fit_fedfisher(sites, tau, *, method="diag", config=FisherConfig(), seed=0, initial_eta):
    """Compose site-local pseudo-values/fits and one server aggregation.

    initial_eta must be a public design constant, e.g. log(lambda*tau**shape),
    and never a pooled response mean. Evaluation rows/truth are not accepted.
    """
    from .pseudo import site_pseudo_response

    started = time.perf_counter()
    if not sites or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Nonempty sites and positive finite tau are required")
    if any(type(site.site_id) is not int for site in sites) or [site.site_id for site in sites] != list(range(len(sites))):
        raise ValueError("Sites must have distinct ordered IDs 0..num_sites-1")
    input_dim = sites[0].rows.design.shape[1]
    initial = public_initial_vector(input_dim, config, seed, initial_eta)
    uploads, local_reports = [], []
    for index, site in enumerate(sites):
        t0 = time.perf_counter()
        pseudo_values = site_pseudo_response(site.rows, tau)
        pseudo_seconds = time.perf_counter() - t0
        blob, report = fit_local_fisher(site.rows.design, pseudo_values, initial_vector=initial,
                                       config=config, seed=seed + 1009 * (index + 1), method=method)
        report.update({"site_id": site.site_id, "pseudo_seconds": pseudo_seconds})
        uploads.append(blob)
        local_reports.append(report)
    vector, server = aggregate_fisher(uploads, initial_vector=initial, input_dim=input_dim,
                                     config=config, method=method)
    diagnostics = {"adapter": PAYLOAD_SCHEMA, "method": method, "config": asdict(config),
                   "seed": seed, "initial_eta": initial_eta, "initial_sha256": _initial_hash(initial),
                   "working_fisher": "expected_gaussian_mean_jacobian", "working_variance": config.working_variance,
                   "local_loss": "mean_squared_error_divided_by_2_variance",
                   "pseudo_scope": "site_local", "local": local_reports, "server": server,
                   "tuning_fits": 0, "refit_fits": len(sites), "total_fit_seconds": time.perf_counter() - started}
    return FisherFit(vector, config, input_dim, diagnostics, tuple(uploads))


def validate_fit_report(diagnostics, config, method, seed, site_sizes, vector,
                        uploads, input_dim, initial_eta):
    """Validate saved fit evidence without refitting or evaluating patients.

    Solver-budget exhaustion is valid diagnostic output but is insufficient
    scientific completion evidence. Recompute stationarity and the objective
    from the actual uploaded summaries, not a saved convergence flag.
    """
    config.validate()
    if method not in ("diag", "kfac"):
        raise ValueError("Invalid Fisher method")
    initial = public_initial_vector(input_dim, config, seed, initial_eta)
    vector = _vector(vector, len(initial))
    expected = {"adapter": PAYLOAD_SCHEMA, "method": method, "config": asdict(config),
                "seed": seed, "initial_eta": initial_eta, "initial_sha256": _initial_hash(initial),
                "working_fisher": "expected_gaussian_mean_jacobian", "working_variance": config.working_variance,
                "local_loss": "mean_squared_error_divided_by_2_variance", "pseudo_scope": "site_local",
                "tuning_fits": 0, "refit_fits": len(site_sizes)}
    if set(diagnostics) != set(expected) | {"local", "server", "total_fit_seconds"} or any(diagnostics.get(k) != value for k, value in expected.items()):
        raise ValueError("Saved Fisher recipe mismatch")
    if not site_sizes or len(uploads) != len(site_sizes) or len(diagnostics.get("local", ())) != len(site_sizes):
        raise ValueError("Saved Fisher site count mismatch")
    decoded = [_read_upload(blob, initial, config, input_dim, method) for blob in uploads]
    if [item[1] for item in decoded] != list(site_sizes):
        raise ValueError("Saved upload sample count mismatch")
    timing_sum = 0.0
    for site_id, (size, blob, report) in enumerate(zip(site_sizes, uploads, diagnostics["local"])):
        target = {"sample_count": size, "optimizer_updates": config.epochs * math.ceil(size / config.batch_size),
                  "epochs": config.epochs, "stop_reason": "fixed_epoch_budget", "upload_bytes": len(blob),
                  "upload_sha256": sha256(blob).hexdigest(), "uploads": 1, "site_id": site_id}
        numeric_fields = {"initial_loss", "final_loss", "training_seconds", "curvature_seconds", "serialization_seconds", "pseudo_seconds"}
        if set(report) != set(target) | numeric_fields or any(report.get(k) != value for k, value in target.items()):
            raise ValueError("Saved local training/upload mismatch")
        for key in ("initial_loss", "final_loss", "training_seconds", "curvature_seconds", "serialization_seconds", "pseudo_seconds"):
            value = report.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid local loss/timing evidence")
            if key.endswith("_seconds"):
                timing_sum += value
    server = diagnostics["server"]
    expected_server = {"solver": "closed_form" if method == "diag" else "conjugate_gradient",
                       "stop_reason": "closed_form" if method == "diag" else "residual_tolerance",
                       "max_iterations": config.server_max_iterations, "ridge": config.server_ridge,
                       "sample_counts": list(site_sizes), "upload_bytes": sum(map(len, uploads)),
                       "uploads": len(site_sizes), "client_to_server_upload_phases": 1,
                       "outcome_dependent_server_broadcasts": 0}
    server_fields = set(expected_server) | {"converged", "iterations", "residual_norm", "residual_threshold", "objective", "aggregation_seconds"}
    if set(server) != server_fields or any(server.get(k) != value for k, value in expected_server.items()) or server.get("converged") is not True:
        raise ValueError("Saved server recipe/convergence mismatch")
    iterations = server.get("iterations")
    if type(iterations) is not int or not 0 <= iterations <= config.server_max_iterations or (method == "diag" and iterations != 0):
        raise ValueError("Invalid server iteration count")
    shapes = _shapes(input_dim, config.architecture)
    residual = config.server_ridge * (vector - initial)
    rhs = config.server_ridge * initial
    objective = config.server_ridge * float(np.square(vector - initial).sum()) / 2
    for (local, count, curvature) in decoded:
        weight = count / sum(site_sizes)
        delta = vector - local
        fd = _curvature_mv(delta, curvature, shapes, method)
        residual += weight * fd
        rhs += weight * _curvature_mv(local, curvature, shapes, method)
        objective += weight * float(delta @ fd) / 2
    norm = float(np.linalg.norm(residual))
    threshold = config.server_tolerance * max(float(np.linalg.norm(rhs)), 1.0)
    if not math.isfinite(norm) or norm > threshold:
        raise ValueError("Saved aggregate fails stationarity")
    for key, actual in (("objective", objective), ("residual_norm", norm), ("residual_threshold", threshold)):
        stored = server.get(key)
        if not isinstance(stored, (int, float)) or not math.isfinite(stored) or not math.isclose(stored, actual, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError("Saved aggregate diagnostics mismatch")
    aggregation_seconds = server.get("aggregation_seconds")
    total_seconds = diagnostics.get("total_fit_seconds")
    if (not isinstance(aggregation_seconds, (int, float)) or not math.isfinite(aggregation_seconds) or aggregation_seconds < 0
            or not isinstance(total_seconds, (int, float)) or not math.isfinite(total_seconds)
            or total_seconds + 1e-9 < timing_sum + aggregation_seconds):
        raise ValueError("Saved total timing omits fit components")
    return True
