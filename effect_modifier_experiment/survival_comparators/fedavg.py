"""Six-round survival FedAvg using the previously selected E5/budget30 recipe.

This adapter fits supplied observed hospital records only. Every round starts
each hospital from the latest sample-weighted aggregate, with fresh SGD momentum.
The scalar MLP8 head, survival mean and untruncated site-local pseudo-values are
the existing survival implementation. Fixed-budget completion is not convergence.
The aggregation receiver accepts only serialized count/parameter summaries;
there is no dataset, independent test sample, truth, or resource selection here.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import struct
from time import perf_counter, process_time

import numpy as np

from effect_modifier_experiment.components.fingerprints import array_hash
from effect_modifier_experiment.survival.fisher import _initial_hash, _vector
from effect_modifier_experiment.survival.pseudo import site_pseudo_response
from effect_modifier_experiment.survival_mlp8.methods import (
    MLP8Config, MLP8Fit, _inputs, _matches, _nonnegative, _same,
    _saved_training, _shapes, _train, public_initial_vector,
)


METHOD_SCHEMA = "survival-mlp8-multiround-fedavg-v1"
_RECIPE = dict(architecture="mlp8", local_epochs=5, total_local_epochs=30,
               batch_size=64, learning_rate=0.01, momentum=0.9,
               working_variance=1.0, stopping_mode="fixed_total_epochs",
               loader_seed_offset=6200037)
_TRAIN_FIELDS = set("sample_count training_seed epochs optimizer_updates processed_rows stop_reason initial_loss final_loss training_pseudo_sha256 model_sha256 training_seconds".split())
_LOCAL_FIELDS = set("site_id initial_sha256 training training_wall_seconds training_cpu_seconds serialization_wall_seconds serialization_cpu_seconds uploads upload_bytes upload_sha256".split())
_ROUND_FIELDS = set("round phase initial_sha256 aggregate_sha256 broadcast_sha256 broadcast_bytes local upload_bytes aggregation_wall_seconds aggregation_cpu_seconds broadcast_serialization_wall_seconds broadcast_serialization_cpu_seconds".split())
_SITE_FIELDS = set("site_id sample_count training_pseudo_sha256 pseudo_wall_seconds pseudo_cpu_seconds".split())
_TIMING_FIELDS = set("pseudo_wall_seconds pseudo_cpu_seconds local_training_wall_seconds local_training_cpu_seconds serialization_wall_seconds serialization_cpu_seconds aggregation_wall_seconds aggregation_cpu_seconds fit_wall_seconds fit_cpu_seconds".split())
_REPORT_FIELDS = set("schema method family seed tau input_dim initial_eta initial_sha256 model_sha256 config site_sizes sample_count objective link pseudo_scope execution_scope optimizer momentum_policy tuning_fits refit_fits validation_fits stopping_reason completion_eligible converged convergence_definition training_rounds_executed total_local_epochs_per_site gradient_updates_per_site processed_rows_per_site sites round_history history_sha256 communication timing total_fit_seconds".split())


@dataclass(frozen=True)
class FedAvgConfig:
    architecture: str = "mlp8"
    local_epochs: int = 5
    total_local_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    working_variance: float = 1.0
    stopping_mode: str = "fixed_total_epochs"
    loader_seed_offset: int = 6200037

    def validate(self):
        # This is one frozen transferred recipe, not a new tuning surface.
        if not _same(asdict(self), _RECIPE):
            raise ValueError("survival FedAvg must retain the selected E5/budget30 recipe")

    @property
    def rounds(self):
        return self.total_local_epochs // self.local_epochs

    def local_config(self):
        self.validate()
        return replace(MLP8Config(), epochs=self.local_epochs,
                       batch_size=self.batch_size, learning_rate=self.learning_rate,
                       momentum=self.momentum, working_variance=self.working_variance)


def _parameter_count(input_dim):
    return sum(out * (inp + 1) for out, inp in _shapes(input_dim))


def _serialize_upload(vector, count):
    """Canonical wire layout: little-endian int64 count then float64 vector."""
    if type(count) is not int or not 1 <= count <= np.iinfo(np.int64).max:
        raise ValueError("invalid uploaded sample count")
    values = np.asarray(vector)
    if values.dtype != np.float64:
        raise ValueError("uploaded model must be float64")
    return struct.pack("<q", count) + _vector(values).astype("<f8").tobytes()


def _read_upload(blob, input_dim):
    size = _parameter_count(input_dim)
    if not isinstance(blob, bytes) or len(blob) != 8 + 8 * size:
        raise ValueError("FedAvg receiver accepts serialized count/model summaries only")
    count = struct.unpack("<q", blob[:8])[0]
    if count < 1:
        raise ValueError("invalid received sample count")
    return _vector(np.frombuffer(blob, dtype="<f8", offset=8), size).copy(), count


def aggregate_uploads(uploads, *, input_dim):
    """Sample-weighted mean; the receiver has no observed hospital records."""
    if not uploads:
        raise ValueError("FedAvg needs at least one serialized upload")
    decoded = [_read_upload(blob, input_dim) for blob in uploads]
    counts = [count for _, count in decoded]
    weights = np.asarray(counts, dtype=np.float64)
    weights /= weights.sum()
    vector = sum(weight * local for weight, (local, _) in zip(weights, decoded))
    return _vector(vector, _parameter_count(input_dim)), weights, decoded


def _loader_seed(config, seed, site_id, round_index):
    # This is the prior binary FedAvg stream, with its original fixed-budget
    # phase, transferred unchanged; only outcome/head/loss differ in survival.
    material = "%d:%d:%d:%d:fixed_budget" % (
        config.loader_seed_offset, seed, site_id, round_index)
    return int.from_bytes(sha256(material.encode("ascii")).digest()[:8], "little")


def _without_timings(value):
    if isinstance(value, dict):
        return {key: _without_timings(item) for key, item in value.items()
                if not key.endswith("_seconds")}
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def _history_hash(report):
    material = {key: report[key] for key in
                ("seed", "tau", "config", "initial_sha256", "sites", "round_history")}
    return sha256(json.dumps(_without_timings(material), sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _communication(rounds, num_sites, elements):
    upload_bytes = rounds * num_sites * (8 + 8 * elements)
    broadcast_bytes = rounds * num_sites * 8 * elements
    return dict(measurement_kind="serialized_logical_payloads_not_network_measurements",
                training_communication_rounds=rounds, client_to_server_upload_phases=rounds,
                model_broadcast_phases=rounds, public_initialization_broadcast_phases=1,
                outcome_dependent_server_broadcast_phases=rounds - 1,
                site_model_upload_count=rounds * num_sites,
                server_model_broadcast_count=rounds * num_sites,
                validation_communication_rounds=0, selection_training_rounds=0,
                refit_training_rounds=0, model_vector_elements=elements,
                model_vector_scalar_dtype="float64", model_upload_metadata_bytes_per_site=8,
                site_model_upload_bytes=upload_bytes, total_site_upload_bytes=upload_bytes,
                total_server_broadcast_bytes=broadcast_bytes,
                total_payload_bytes=upload_bytes + broadcast_bytes,
                metadata_layout="little_endian_int64_count_then_float64_parameters",
                broadcast_policy="public_initial_model_then_aggregate_before_every_next_local_round",
                server_aggregation="sample_count_weighted_parameter_mean")


def fit_fedavg(sites, tau, seed, config, initial_eta):
    """Return the common MLP8 prediction interface, complete audit and arrays."""
    if not isinstance(config, FedAvgConfig):
        raise TypeError("config must be FedAvgConfig")
    config.validate()
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("invalid FedAvg scientific seed")
    local_config = config.local_config()
    input_dim = _inputs(sites, tau, seed, local_config, initial_eta)
    started_wall, started_cpu = perf_counter(), process_time()
    initial = public_initial_vector(input_dim, local_config, seed, initial_eta)
    responses, site_reports = [], []
    for index, site in enumerate(sites):
        wall, cpu = perf_counter(), process_time()
        response = site_pseudo_response(site.rows, tau)
        site_reports.append(dict(site_id=index, sample_count=len(site.rows),
            training_pseudo_sha256=array_hash(response),
            pseudo_wall_seconds=perf_counter() - wall, pseudo_cpu_seconds=process_time() - cpu))
        responses.append(response)
    current, history = initial.copy(), []
    arrays = dict(initial_vector=initial)
    for round_index in range(1, config.rounds + 1):
        wall, cpu = perf_counter(), process_time()
        broadcast = current.astype("<f8").tobytes()
        item = dict(round=round_index, phase="fixed_budget", initial_sha256=array_hash(current),
                    broadcast_sha256=sha256(broadcast).hexdigest(),
                    broadcast_bytes=len(sites) * len(broadcast),
                    broadcast_serialization_wall_seconds=perf_counter() - wall,
                    broadcast_serialization_cpu_seconds=process_time() - cpu)
        uploads, local_reports = [], []
        for index, (site, response) in enumerate(zip(sites, responses)):
            # Fresh deserialization and optimizer for every site/round is the
            # real next-round broadcast, rather than repeating the first model.
            received = np.frombuffer(broadcast, dtype="<f8").copy()
            wall, cpu = perf_counter(), process_time()
            _, vector, training = _train(site.rows.design, response, received,
                local_config, _loader_seed(config, seed, index, round_index))
            local = dict(site_id=index, initial_sha256=array_hash(received), training=training,
                         training_wall_seconds=perf_counter() - wall,
                         training_cpu_seconds=process_time() - cpu)
            wall, cpu = perf_counter(), process_time()
            blob = _serialize_upload(vector, len(site.rows))
            local.update(serialization_wall_seconds=perf_counter() - wall,
                         serialization_cpu_seconds=process_time() - cpu,
                         uploads=1, upload_bytes=len(blob), upload_sha256=sha256(blob).hexdigest())
            uploads.append(blob)
            local_reports.append(local)
            arrays[f"round_{round_index}__local_{index}"] = vector
        wall, cpu = perf_counter(), process_time()
        current, weights, _ = aggregate_uploads(uploads, input_dim=input_dim)
        item.update(aggregate_sha256=array_hash(current), local=local_reports,
                    upload_bytes=sum(len(blob) for blob in uploads),
                    aggregation_wall_seconds=perf_counter() - wall,
                    aggregation_cpu_seconds=process_time() - cpu)
        arrays[f"round_{round_index}__aggregate"] = current.copy()
        history.append(item)
    arrays.update(vector=current, weights=weights)
    timing = dict(
        pseudo_wall_seconds=sum(s["pseudo_wall_seconds"] for s in site_reports),
        pseudo_cpu_seconds=sum(s["pseudo_cpu_seconds"] for s in site_reports),
        local_training_wall_seconds=sum(s["training_wall_seconds"] for r in history for s in r["local"]),
        local_training_cpu_seconds=sum(s["training_cpu_seconds"] for r in history for s in r["local"]),
        serialization_wall_seconds=sum(r["broadcast_serialization_wall_seconds"] + sum(
            s["serialization_wall_seconds"] for s in r["local"]) for r in history),
        serialization_cpu_seconds=sum(r["broadcast_serialization_cpu_seconds"] + sum(
            s["serialization_cpu_seconds"] for s in r["local"]) for r in history),
        aggregation_wall_seconds=sum(r["aggregation_wall_seconds"] for r in history),
        aggregation_cpu_seconds=sum(r["aggregation_cpu_seconds"] for r in history),
        fit_wall_seconds=perf_counter() - started_wall, fit_cpu_seconds=process_time() - started_cpu)
    sizes = [len(site.rows) for site in sites]
    report = dict(schema=METHOD_SCHEMA, method="multiround_fedavg_survival_model", family="mlp8",
        seed=seed, tau=float(tau), input_dim=input_dim, initial_eta=float(initial_eta),
        initial_sha256=_initial_hash(initial), model_sha256=array_hash(current), config=asdict(config),
        site_sizes=sizes, sample_count=sum(sizes), objective="half_mean_squared_error_divided_by_working_variance",
        link="log_minus_log_survival", pseudo_scope="site_local_pv", execution_scope="multiround_federated_training",
        optimizer="SGD", momentum_policy="reset_to_zero_at_each_site_at_each_round",
        tuning_fits=0, refit_fits=0, validation_fits=0, stopping_reason="fixed_total_epochs_completed",
        completion_eligible=True, converged=None, convergence_definition="not_claimed_fixed_budget",
        training_rounds_executed=config.rounds, total_local_epochs_per_site=config.total_local_epochs,
        gradient_updates_per_site=[config.total_local_epochs * math.ceil(n / config.batch_size) for n in sizes],
        processed_rows_per_site=[config.total_local_epochs * n for n in sizes], sites=site_reports,
        round_history=history, communication=_communication(config.rounds, len(sites), len(current)),
        timing=timing, total_fit_seconds=timing["fit_wall_seconds"])
    report["history_sha256"] = _history_hash(report)
    return MLP8Fit(current, local_config, input_dim), report, arrays


def validate_fit_report(report, arrays, config, site_sizes):
    """Validate saved round arithmetic/identity without fitting or inference."""
    if not isinstance(config, FedAvgConfig):
        raise ValueError("invalid FedAvg validation config")
    config.validate()
    if (not isinstance(report, dict) or set(report) != _REPORT_FIELDS or not site_sizes
            or any(type(n) is not int or n < 1 for n in site_sizes)):
        raise ValueError("invalid FedAvg report/count fields")
    local_config = config.local_config()
    input_dim, seed, eta, tau = (report[key] for key in ("input_dim", "seed", "initial_eta", "tau"))
    if (type(input_dim) is not int or input_dim < 5 or input_dim % 2 != 1
            or type(seed) is not int or not 0 <= seed < 2**63
            or isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0):
        raise ValueError("invalid saved FedAvg design/seed/time")
    initial = public_initial_vector(input_dim, local_config, seed, eta)
    required_arrays = {"initial_vector", "vector", "weights"} | {
        f"round_{r}__{key}" for r in range(1, config.rounds + 1)
        for key in ("aggregate",) + tuple(f"local_{i}" for i in range(len(site_sizes)))}
    if (set(arrays) != required_arrays or any(np.asarray(v).dtype != np.float64
            or not np.isfinite(v).all() for v in arrays.values())
            or not np.array_equal(arrays["initial_vector"], initial)):
        raise ValueError("saved FedAvg arrays/initialization mismatch")
    vector = _vector(arrays["vector"], len(initial))
    expected = dict(schema=METHOD_SCHEMA, method="multiround_fedavg_survival_model", family="mlp8",
        config=asdict(config), site_sizes=list(site_sizes), sample_count=sum(site_sizes),
        initial_sha256=_initial_hash(initial), model_sha256=array_hash(vector),
        objective="half_mean_squared_error_divided_by_working_variance", link="log_minus_log_survival",
        pseudo_scope="site_local_pv", execution_scope="multiround_federated_training", optimizer="SGD",
        momentum_policy="reset_to_zero_at_each_site_at_each_round", tuning_fits=0, refit_fits=0,
        validation_fits=0, stopping_reason="fixed_total_epochs_completed", completion_eligible=True,
        converged=None, convergence_definition="not_claimed_fixed_budget",
        training_rounds_executed=config.rounds, total_local_epochs_per_site=config.total_local_epochs,
        gradient_updates_per_site=[config.total_local_epochs * math.ceil(n / config.batch_size) for n in site_sizes],
        processed_rows_per_site=[config.total_local_epochs * n for n in site_sizes],
        communication=_communication(config.rounds, len(site_sizes), len(initial)))
    if not _matches(report, expected) or len(report["sites"]) != len(site_sizes) or len(report["round_history"]) != config.rounds:
        raise ValueError("saved FedAvg recipe/completion mismatch")
    for index, (site, size) in enumerate(zip(report["sites"], site_sizes)):
        if set(site) != _SITE_FIELDS or not _matches(site, dict(site_id=index, sample_count=size)):
            raise ValueError("saved FedAvg site identity mismatch")
        _nonnegative(site, ("pseudo_wall_seconds", "pseudo_cpu_seconds"))
    current = initial
    for round_index, item in enumerate(report["round_history"], 1):
        broadcast = current.astype("<f8").tobytes()
        if (set(item) != _ROUND_FIELDS or len(item["local"]) != len(site_sizes)
                or not _matches(item, dict(round=round_index, phase="fixed_budget", initial_sha256=array_hash(current),
                    broadcast_sha256=sha256(broadcast).hexdigest(), broadcast_bytes=len(broadcast) * len(site_sizes)))):
            raise ValueError("saved FedAvg broadcast/round mismatch")
        uploads = []
        for index, (local, size) in enumerate(zip(item["local"], site_sizes)):
            value = arrays[f"round_{round_index}__local_{index}"]
            blob = _serialize_upload(_vector(value, len(initial)), size)
            if (set(local) != _LOCAL_FIELDS or set(local["training"]) != _TRAIN_FIELDS
                    or not _matches(local, dict(site_id=index, initial_sha256=array_hash(current), uploads=1,
                        upload_bytes=len(blob), upload_sha256=sha256(blob).hexdigest()))):
                raise ValueError("saved FedAvg local upload mismatch")
            training = dict(local["training"], pseudo_seconds=0.0)
            _saved_training(training, size, _loader_seed(config, seed, index, round_index), local_config, value)
            if training["training_pseudo_sha256"] != report["sites"][index]["training_pseudo_sha256"]:
                raise ValueError("saved FedAvg pseudo-label identity differs across rounds")
            _nonnegative(local, ("training_wall_seconds", "training_cpu_seconds", "serialization_wall_seconds", "serialization_cpu_seconds"))
            if local["training_wall_seconds"] + 1e-9 < training["training_seconds"]:
                raise ValueError("saved FedAvg local timing omits training")
            uploads.append(blob)
        current, weights, _ = aggregate_uploads(uploads, input_dim=input_dim)
        if (not np.array_equal(arrays["weights"], weights)
                or not np.array_equal(arrays[f"round_{round_index}__aggregate"], current)
                or not _matches(item, dict(aggregate_sha256=array_hash(current), upload_bytes=sum(map(len, uploads))))):
            raise ValueError("saved FedAvg aggregate differs from sample-weighted uploads")
        _nonnegative(item, ("aggregation_wall_seconds", "aggregation_cpu_seconds",
                            "broadcast_serialization_wall_seconds", "broadcast_serialization_cpu_seconds"))
    if not np.array_equal(vector, current) or report["history_sha256"] != _history_hash(report):
        raise ValueError("saved FedAvg final model/history mismatch")
    timing = report["timing"]
    if set(timing) != _TIMING_FIELDS:
        raise ValueError("invalid saved FedAvg timing fields")
    _nonnegative(timing, _TIMING_FIELDS)
    _nonnegative(report, ("total_fit_seconds",))
    sums = dict(
        pseudo_wall_seconds=sum(s["pseudo_wall_seconds"] for s in report["sites"]),
        pseudo_cpu_seconds=sum(s["pseudo_cpu_seconds"] for s in report["sites"]),
        local_training_wall_seconds=sum(s["training_wall_seconds"] for r in report["round_history"] for s in r["local"]),
        local_training_cpu_seconds=sum(s["training_cpu_seconds"] for r in report["round_history"] for s in r["local"]),
        serialization_wall_seconds=sum(r["broadcast_serialization_wall_seconds"] + sum(s["serialization_wall_seconds"] for s in r["local"]) for r in report["round_history"]),
        serialization_cpu_seconds=sum(r["broadcast_serialization_cpu_seconds"] + sum(s["serialization_cpu_seconds"] for s in r["local"]) for r in report["round_history"]),
        aggregation_wall_seconds=sum(r["aggregation_wall_seconds"] for r in report["round_history"]),
        aggregation_cpu_seconds=sum(r["aggregation_cpu_seconds"] for r in report["round_history"]))
    if any(not math.isclose(timing[key], value, rel_tol=1e-10, abs_tol=1e-9) for key, value in sums.items()):
        raise ValueError("saved FedAvg component timing arithmetic mismatch")
    if (report["total_fit_seconds"] != timing["fit_wall_seconds"]
            or timing["fit_wall_seconds"] + 1e-9 < sum(v for k, v in sums.items() if k.endswith("wall_seconds"))
            or timing["fit_cpu_seconds"] + 1e-9 < sum(v for k, v in sums.items() if k.endswith("cpu_seconds"))):
        raise ValueError("saved FedAvg timing omits fitting components")
    return True
