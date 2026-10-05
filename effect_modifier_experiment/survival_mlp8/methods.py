"""One-hidden-layer survival models, with unchanged pseudo-value/link recipes.

The scalar network predicts eta = log(-log(S(tau))). Local fitting minimizes
mean((PV - exp(-exp(eta)))**2)/(2*working_variance), with untruncated IJ PVs.
The Gaussian mean-Jacobian working Fisher is not a pseudo-value likelihood.
Only opaque, serialized count/parameter/curvature summaries enter the server.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from effect_modifier_experiment.components.fingerprints import array_hash
from effect_modifier_experiment.survival.data import SurvivalSite
from effect_modifier_experiment.survival.fisher import (
    _curvature_mv, _initial_hash, _load_vector, _mean, _vector, working_fisher,
)
from effect_modifier_experiment.survival.pseudo import pseudo_survival, site_pseudo_response


METHOD_SCHEMA = "survival-mlp8-working-gaussian-v1"
PSEUDO_SCOPES = ("site_local_pv", "pooled_km_pv")
_BASE_FIELDS = set("schema family seed input_dim initial_eta initial_sha256 model_sha256 config site_sizes sample_count objective link tuning_fits".split())
_TRAIN_FIELDS = set("sample_count training_seed epochs optimizer_updates processed_rows stop_reason initial_loss final_loss training_pseudo_sha256 model_sha256 training_seconds".split())


@dataclass(frozen=True)
class MLP8Config:
    architecture: str = "mlp8"
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    working_variance: float = 1.0
    server_ridge: float = 1e-6
    server_max_iterations: int = 2000
    server_tolerance: float = 1e-8

    def validate(self):
        if self.architecture != "mlp8":
            raise ValueError("MLP8 requires exactly one hidden layer with eight neurons")
        for key in ("epochs", "batch_size", "server_max_iterations"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + " must be a positive integer")
        for key in ("learning_rate", "working_variance", "server_ridge", "server_tolerance", "momentum"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(key + " must be finite and numeric")
            if (key == "momentum" and not 0 <= value < 1) or (key != "momentum" and value <= 0):
                raise ValueError("invalid " + key)


def _shapes(input_dim):
    if type(input_dim) is not int or input_dim < 1:
        raise ValueError("input_dim must be a positive integer")
    return ((8, input_dim), (1, 8))


def _model(input_dim, config, seed=0):
    config.validate()
    _shapes(input_dim)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(input_dim, 8, dtype=torch.float64), nn.ReLU(),
                             nn.Linear(8, 1, dtype=torch.float64))


def public_initial_vector(input_dim, config, seed, initial_eta):
    """Outcome-independent public random weights and baseline eta bias."""
    if type(seed) is not int or seed < 0 or isinstance(initial_eta, bool) or not math.isfinite(initial_eta):
        raise ValueError("invalid public initialization")
    model = _model(input_dim, config, seed)
    with torch.no_grad():
        model[-1].bias.fill_(initial_eta)
    return parameters_to_vector(model.parameters()).detach().numpy().copy()


@dataclass(frozen=True)
class MLP8Fit:
    vector: np.ndarray
    config: MLP8Config
    input_dim: int
    _prediction_model: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.config.validate()
        size = sum(out * (inp + 1) for out, inp in _shapes(self.input_dim))
        value = _vector(self.vector, size)
        object.__setattr__(self, "vector", np.frombuffer(value.astype("<f8").tobytes(), dtype="<f8"))

    def predict_eta(self, design):
        design = np.asarray(design, dtype=np.float64)
        if design.ndim != 2 or design.shape[1] != self.input_dim or not np.isfinite(design).all():
            raise ValueError("invalid MLP8 prediction design")
        if self._prediction_model is None:
            model = _model(self.input_dim, self.config)
            _load_vector(model, self.vector)
            model.eval().requires_grad_(False)
            object.__setattr__(self, "_prediction_model", model)
        with torch.no_grad():
            prediction = self._prediction_model(torch.from_numpy(design.copy())).squeeze(-1).numpy()
        if not np.isfinite(prediction).all():
            raise FloatingPointError("nonfinite MLP8 prediction")
        return prediction


def _inputs(sites, tau, seed, config, initial_eta):
    config.validate()
    if (not sites or type(seed) is not int or seed < 0 or isinstance(tau, bool)
            or not math.isfinite(tau) or tau <= 0 or isinstance(initial_eta, bool)
            or not math.isfinite(initial_eta)):
        raise ValueError("invalid MLP8 training inputs")
    if (any(not isinstance(site, SurvivalSite) for site in sites)
            or [site.site_id for site in sites] != list(range(len(sites)))
            or len({site.rows.x.shape[1] for site in sites}) != 1):
        raise ValueError("training requires ordered distinct observed sites")
    identifiers = np.concatenate([site.rows.row_ids for site in sites])
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("training row identifiers must be globally unique")
    return sites[0].rows.design.shape[1]


def _responses(sites, tau, scope):
    if scope == "site_local_pv":
        return [site_pseudo_response(site.rows, tau) for site in sites]
    if scope != "pooled_km_pv":
        raise ValueError("unknown pseudo-value scope")
    value = pseudo_survival(np.concatenate([s.rows.time for s in sites]),
                            np.concatenate([s.rows.event for s in sites]), tau)
    return list(np.split(value, np.cumsum([len(s.rows) for s in sites])[:-1]))


def _train(design, response, initial, config, training_seed):
    model = _model(design.shape[1], config)
    _load_vector(model, initial)
    x, target = torch.from_numpy(design.copy()), torch.from_numpy(response.copy())
    rng = torch.Generator(device="cpu").manual_seed(training_seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    started = time.perf_counter()
    with torch.no_grad():
        initial_loss = float((_mean(model(x).squeeze(-1)) - target).square().mean() / (2 * config.working_variance))
    for _ in range(config.epochs):
        permutation = torch.randperm(len(x), generator=rng)
        for start in range(0, len(x), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            optimizer.zero_grad()
            loss = (_mean(model(x[indices]).squeeze(-1)) - target[indices]).square().mean() / (2 * config.working_variance)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite MLP8 objective")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite MLP8 gradient")
            optimizer.step()
    vector = _vector(parameters_to_vector(model.parameters()).detach().numpy().copy())
    with torch.no_grad():
        final_loss = float((_mean(model(x).squeeze(-1)) - target).square().mean() / (2 * config.working_variance))
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise FloatingPointError("nonfinite MLP8 full objective")
    report = dict(sample_count=len(response), training_seed=training_seed, epochs=config.epochs,
                  optimizer_updates=config.epochs * math.ceil(len(response) / config.batch_size),
                  processed_rows=config.epochs * len(response), stop_reason="fixed_epoch_budget",
                  initial_loss=initial_loss, final_loss=final_loss,
                  training_pseudo_sha256=array_hash(response), model_sha256=array_hash(vector),
                  training_seconds=time.perf_counter() - started)
    return model, vector, report


def _curvature_keys(kind):
    if kind not in ("diag", "kfac"):
        raise ValueError("kind must be diag or kfac")
    return ("diag",) if kind == "diag" else ("a0", "g0", "a1", "g1")


def _serialize(vector, curvature, n, kind, input_dim, config, initial):
    metadata = dict(schema=METHOD_SCHEMA, kind=kind, input_dim=input_dim,
                    config=asdict(config), initial_sha256=_initial_hash(initial))
    stream = BytesIO()
    np.savez(stream, metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
             vector=vector, n=np.array(n, dtype=np.int64),
             **{key: curvature[key] for key in _curvature_keys(kind)})
    return stream.getvalue()


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate payload metadata key")
        result[key] = value
    return result


def _same(actual, expected):
    """JSON recipe equality without treating booleans as integer counts."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _matches(report, expected):
    return all(_same(report.get(key), value) for key, value in expected.items())


def _read_upload(blob, initial, config, input_dim, kind):
    if not isinstance(blob, bytes):
        raise ValueError("server accepts serialized summaries only")
    config.validate()
    shapes = _shapes(input_dim)
    with np.load(BytesIO(blob), allow_pickle=False) as saved:
        keys = _curvature_keys(kind)
        if len(saved.files) != len(set(saved.files)) or set(saved.files) != {"metadata", "vector", "n"} | set(keys):
            raise ValueError("invalid upload fields")
        metadata = saved["metadata"]
        if metadata.dtype != np.uint8 or metadata.ndim != 1:
            raise ValueError("invalid upload metadata encoding")
        actual = json.loads(metadata.tobytes().decode(), object_pairs_hook=_unique_json)
        expected = dict(schema=METHOD_SCHEMA, kind=kind, input_dim=input_dim,
                        config=asdict(config), initial_sha256=_initial_hash(initial))
        if (not _same(actual, expected) or metadata.tobytes() != json.dumps(expected, sort_keys=True).encode()
                or saved["vector"].dtype != np.float64):
            raise ValueError("upload recipe/initialization mismatch")
        vector = _vector(saved["vector"], sum(o * (i + 1) for o, i in shapes)).copy()
        n = saved["n"]
        if n.shape != () or n.dtype != np.int64 or int(n) < 1:
            raise ValueError("invalid uploaded sample count")
        curvature = {key: saved[key].copy() for key in keys}
    if any(v.dtype != np.float64 or not np.isfinite(v).all() for v in curvature.values()):
        raise ValueError("invalid working Fisher values")
    if kind == "diag":
        if curvature["diag"].shape != vector.shape or np.any(curvature["diag"] < 0):
            raise ValueError("invalid Fisher diagonal")
    else:
        for index, (out, inp) in enumerate(shapes):
            for key, size in ((f"a{index}", inp + 1), (f"g{index}", out)):
                factor = curvature[key]
                if (factor.shape != (size, size) or not np.allclose(factor, factor.T, rtol=1e-12, atol=1e-14)
                        or np.linalg.eigvalsh(factor)[0] < -1e-10 * max(1., np.linalg.norm(factor))):
                    raise ValueError("invalid KFAC factor")
    return vector, int(n), curvature


def _server_system(decoded, initial, input_dim, config, kind):
    weights = np.array([n for _, n, _ in decoded], dtype=np.float64)
    weights /= weights.sum()
    def mv(vector):
        return config.server_ridge * vector + sum(
            weight * _curvature_mv(vector, curvature, _shapes(input_dim), kind)
            for weight, (_, _, curvature) in zip(weights, decoded))
    rhs = config.server_ridge * initial + sum(
        weight * _curvature_mv(vector, curvature, _shapes(input_dim), kind)
        for weight, (vector, _, curvature) in zip(weights, decoded))
    return weights, mv, rhs


def aggregate_fisher(uploads, *, initial_vector, input_dim, config, kind):
    """One-upload objective: weighted local quadratic plus public-vector ridge."""
    if not uploads:
        raise ValueError("nonempty serialized uploads required")
    started = time.perf_counter()
    initial = _vector(initial_vector, sum(o * (i + 1) for o, i in _shapes(input_dim)))
    decoded = [_read_upload(blob, initial, config, input_dim, kind) for blob in uploads]
    weights, mv, rhs = _server_system(decoded, initial, input_dim, config, kind)
    threshold = config.server_tolerance * max(float(np.linalg.norm(rhs)), 1.)
    iterations = 0
    if kind == "diag":
        precision = config.server_ridge + sum(w * item[2]["diag"] for w, item in zip(weights, decoded))
        vector, status = rhs / precision, "closed_form"
    else:
        vector = sum(w * item[0] for w, item in zip(weights, decoded))
        residual = rhs - mv(vector)
        direction, rr = residual.copy(), float(residual @ residual)
        status = "residual_tolerance" if math.sqrt(rr) <= threshold else "iteration_budget"
        for _ in range(config.server_max_iterations):
            if status == "residual_tolerance":
                break
            ad = mv(direction)
            denom = float(direction @ ad)
            if not math.isfinite(denom) or denom <= 0:
                raise FloatingPointError("nonpositive/nonfinite CG curvature")
            vector += (rr / denom) * direction
            residual = rhs - mv(vector)
            next_rr = float(residual @ residual)
            iterations += 1
            if math.sqrt(next_rr) <= threshold:
                status = "residual_tolerance"
                break
            direction, rr = residual + (next_rr / rr) * direction, next_rr
    vector = _vector(vector, len(initial))
    residual = float(np.linalg.norm(mv(vector) - rhs))
    objective = config.server_ridge * float(np.square(vector - initial).sum()) / 2
    for weight, (local, _, curvature) in zip(weights, decoded):
        delta = vector - local
        objective += weight * float(delta @ _curvature_mv(delta, curvature, _shapes(input_dim), kind)) / 2
    if not math.isfinite(residual) or not math.isfinite(objective):
        raise FloatingPointError("nonfinite server endpoint")
    report = dict(solver="closed_form" if kind == "diag" else "conjugate_gradient",
                  stop_reason=status, converged=residual <= threshold, iterations=iterations,
                  max_iterations=config.server_max_iterations, residual_norm=residual,
                  residual_threshold=threshold, objective=objective, ridge=config.server_ridge,
                  sample_counts=[n for _, n, _ in decoded], uploads=len(uploads),
                  upload_bytes=sum(map(len, uploads)), client_to_server_upload_phases=1,
                  outcome_dependent_server_broadcasts=0, aggregation_seconds=time.perf_counter() - started)
    return vector.copy(), report, decoded, weights


def _base_report(seed, input_dim, initial_eta, initial, config, sites, vector):
    return dict(schema=METHOD_SCHEMA, family="mlp8", seed=seed, input_dim=input_dim,
                initial_eta=float(initial_eta), initial_sha256=_initial_hash(initial),
                model_sha256=array_hash(vector), config=asdict(config),
                site_sizes=[len(s.rows) for s in sites], sample_count=sum(len(s.rows) for s in sites),
                objective="half_mean_squared_error_divided_by_working_variance",
                link="log_minus_log_survival", tuning_fits=0)


def fit_federated(sites, tau, seed, kind, config, initial_eta):
    input_dim = _inputs(sites, tau, seed, config, initial_eta)
    _curvature_keys(kind)
    started = time.perf_counter()
    initial = public_initial_vector(input_dim, config, seed, initial_eta)
    uploads, local = [], []
    for index, site in enumerate(sites):
        t0 = time.perf_counter()
        response = site_pseudo_response(site.rows, tau)
        pseudo_seconds = time.perf_counter() - t0
        model, vector, report = _train(site.rows.design, response, initial, config, seed + 1009 * (index + 1))
        t0 = time.perf_counter()
        curvature = working_fisher(model, site.rows.design, method=kind, variance=config.working_variance)
        curvature_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        blob = _serialize(vector, curvature, len(site.rows), kind, input_dim, config, initial)
        report.update(site_id=index, pseudo_seconds=pseudo_seconds, curvature_seconds=curvature_seconds,
                      serialization_seconds=time.perf_counter() - t0, uploads=1,
                      upload_bytes=len(blob), upload_sha256=sha256(blob).hexdigest())
        uploads.append(blob)
        local.append(report)
    vector, server, decoded, weights = aggregate_fisher(uploads, initial_vector=initial,
                                                      input_dim=input_dim, config=config, kind=kind)
    report = _base_report(seed, input_dim, initial_eta, initial, config, sites, vector)
    report.update(kind=kind, execution_scope="one_upload_federated_training", pseudo_scope="site_local_pv",
                  working_fisher="expected_gaussian_mean_jacobian", local=local, server=server,
                  refit_fits=len(sites), total_fit_seconds=time.perf_counter() - started)
    arrays = dict(vector=vector, initial_vector=initial, weights=weights)
    for index, (value, _, curvature) in enumerate(decoded):
        arrays[f"local_{index}__vector"] = value
        arrays.update({f"local_{index}__{key}": val for key, val in curvature.items()})
    return MLP8Fit(vector, config, input_dim), report, arrays


def fit_pooled(sites, tau, seed, scope, config, initial_eta):
    input_dim = _inputs(sites, tau, seed, config, initial_eta)
    started = time.perf_counter()
    t0 = time.perf_counter()
    response = np.concatenate(_responses(sites, tau, scope))
    pseudo_seconds = time.perf_counter() - t0
    initial = public_initial_vector(input_dim, config, seed, initial_eta)
    _, vector, training = _train(np.concatenate([s.rows.design for s in sites]), response,
                                 initial, config, seed + 1009)
    report = _base_report(seed, input_dim, initial_eta, initial, config, sites, vector)
    report.update(training, pseudo_scope=scope, execution_scope="centralized_training",
                  communication_uploads=0, fisher_computed=False, refit_fits=1,
                  pseudo_seconds=pseudo_seconds, total_fit_seconds=time.perf_counter() - started)
    return MLP8Fit(vector, config, input_dim), report, dict(vector=vector, initial_vector=initial)


def _saved_base(report, arrays, config, site_sizes):
    config.validate()
    if (not site_sizes or any(type(n) is not int or n < 1 for n in site_sizes)
            or not _same(report.get("site_sizes"), list(site_sizes))
            or not _same(report.get("sample_count"), sum(site_sizes))):
        raise ValueError("saved sample counts mismatch")
    input_dim, seed, initial_eta = report.get("input_dim"), report.get("seed"), report.get("initial_eta")
    if type(input_dim) is not int or input_dim < 5 or input_dim % 2 != 1:
        raise ValueError("invalid saved survival design width")
    initial = public_initial_vector(input_dim, config, seed, initial_eta)
    if any(np.asarray(v).dtype != np.float64 or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("saved arrays must be finite float64")
    vector = _vector(arrays["vector"], len(initial))
    expected = dict(schema=METHOD_SCHEMA, family="mlp8", config=asdict(config),
                    initial_sha256=_initial_hash(initial), model_sha256=array_hash(vector), tuning_fits=0,
                    objective="half_mean_squared_error_divided_by_working_variance", link="log_minus_log_survival")
    if not _matches(report, expected) or not np.array_equal(arrays["initial_vector"], initial):
        raise ValueError("saved model recipe/initialization mismatch")
    _nonnegative(report, ("total_fit_seconds",))
    return initial, vector, input_dim


def _nonnegative(report, keys):
    for key in keys:
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid saved " + key)


def _saved_training(report, size, training_seed, config, vector):
    expected = dict(sample_count=size, training_seed=training_seed, epochs=config.epochs,
                    optimizer_updates=config.epochs * math.ceil(size / config.batch_size),
                    processed_rows=config.epochs * size, stop_reason="fixed_epoch_budget",
                    model_sha256=array_hash(vector))
    if (not _matches(report, expected)
            or any(type(report.get(k)) is not int for k in
                   ("sample_count", "training_seed", "epochs", "optimizer_updates", "processed_rows"))):
        raise ValueError("saved training budget/model mismatch")
    fingerprint = report.get("training_pseudo_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("invalid response fingerprint")
    _nonnegative(report, ("initial_loss", "final_loss", "training_seconds", "pseudo_seconds"))


def validate_pooled_fit(report, arrays, scope, config, site_sizes):
    initial, vector, _ = _saved_base(report, arrays, config, site_sizes)
    required_report = _BASE_FIELDS | _TRAIN_FIELDS | set("pseudo_scope execution_scope communication_uploads fisher_computed refit_fits pseudo_seconds total_fit_seconds".split())
    if (set(report) != required_report or set(arrays) != {"vector", "initial_vector"} or scope not in PSEUDO_SCOPES
            or report.get("pseudo_scope") != scope or report.get("execution_scope") != "centralized_training"
            or not _same(report.get("communication_uploads"), 0) or report.get("fisher_computed") is not False
            or not _same(report.get("refit_fits"), 1)):
        raise ValueError("saved pooled scope mismatch")
    _saved_training(report, sum(site_sizes), report["seed"] + 1009, config, vector)
    if report["total_fit_seconds"] + 1e-9 < report["training_seconds"] + report["pseudo_seconds"]:
        raise ValueError("saved timing omits pooled fitting components")
    return True


def validate_federated_fit(report, arrays, kind, config, site_sizes):
    initial, vector, input_dim = _saved_base(report, arrays, config, site_sizes)
    keys = _curvature_keys(kind)
    required = {"vector", "initial_vector", "weights"} | {
        f"local_{i}__{key}" for i in range(len(site_sizes)) for key in ("vector",) + keys}
    required_report = _BASE_FIELDS | set("kind execution_scope pseudo_scope working_fisher local server refit_fits total_fit_seconds".split())
    if (set(report) != required_report or set(arrays) != required or report.get("kind") != kind or report.get("pseudo_scope") != "site_local_pv"
            or report.get("execution_scope") != "one_upload_federated_training"
            or report.get("working_fisher") != "expected_gaussian_mean_jacobian"
            or not _same(report.get("refit_fits"), len(site_sizes)) or len(report.get("local", ())) != len(site_sizes)):
        raise ValueError("saved federated scope mismatch")
    decoded, total_bytes, component_seconds = [], 0, 0.
    for index, (size, local) in enumerate(zip(site_sizes, report["local"])):
        value = arrays[f"local_{index}__vector"]
        curvature = {key: arrays[f"local_{index}__{key}"] for key in keys}
        blob = _serialize(value, curvature, size, kind, input_dim, config, initial)
        decoded.append(_read_upload(blob, initial, config, input_dim, kind))
        _saved_training(local, size, report["seed"] + 1009 * (index + 1), config, value)
        required_local = _TRAIN_FIELDS | set("site_id pseudo_seconds curvature_seconds serialization_seconds uploads upload_bytes upload_sha256".split())
        if (set(local) != required_local or not _matches(local, dict(site_id=index, uploads=1,
                upload_bytes=len(blob), upload_sha256=sha256(blob).hexdigest()))):
            raise ValueError("saved local upload mismatch")
        _nonnegative(local, ("curvature_seconds", "serialization_seconds"))
        component_seconds += sum(local[k] for k in ("training_seconds", "pseudo_seconds", "curvature_seconds", "serialization_seconds"))
        total_bytes += len(blob)
    weights, mv, rhs = _server_system(decoded, initial, input_dim, config, kind)
    if not np.array_equal(arrays["weights"], weights):
        raise ValueError("saved aggregation weights mismatch")
    server = report.get("server", {})
    expected = dict(solver="closed_form" if kind == "diag" else "conjugate_gradient",
                    stop_reason="closed_form" if kind == "diag" else "residual_tolerance", converged=True,
                    max_iterations=config.server_max_iterations, ridge=config.server_ridge,
                    sample_counts=list(site_sizes), uploads=len(site_sizes), upload_bytes=total_bytes,
                    client_to_server_upload_phases=1, outcome_dependent_server_broadcasts=0)
    required_server = set(expected) | set("iterations residual_norm residual_threshold objective aggregation_seconds".split())
    if set(server) != required_server or not _matches(server, expected) or server.get("converged") is not True:
        raise ValueError("saved server recipe/convergence mismatch")
    iterations = server.get("iterations")
    if type(iterations) is not int or not 0 <= iterations <= config.server_max_iterations or (kind == "diag" and iterations != 0):
        raise ValueError("invalid server iteration evidence")
    norm = float(np.linalg.norm(mv(vector) - rhs))
    threshold = config.server_tolerance * max(float(np.linalg.norm(rhs)), 1.)
    if not math.isfinite(norm) or norm > threshold:
        raise ValueError("saved aggregate fails stationarity")
    if kind == "diag":
        precision = config.server_ridge + sum(w * item[2]["diag"] for w, item in zip(weights, decoded))
        if not np.array_equal(vector, rhs / precision):
            raise ValueError("saved diagonal endpoint differs from closed form")
    objective = config.server_ridge * float(np.square(vector - initial).sum()) / 2
    for weight, (local, _, curvature) in zip(weights, decoded):
        delta = vector - local
        objective += weight * float(delta @ _curvature_mv(delta, curvature, _shapes(input_dim), kind)) / 2
    for key, actual in (("residual_norm", norm), ("residual_threshold", threshold), ("objective", objective)):
        _nonnegative(server, (key,))
        if not math.isclose(server[key], actual, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError("saved server endpoint diagnostics mismatch")
    _nonnegative(server, ("aggregation_seconds",))
    if report["total_fit_seconds"] + 1e-9 < component_seconds + server["aggregation_seconds"]:
        raise ValueError("saved timing omits local/server fit components")
    return True
