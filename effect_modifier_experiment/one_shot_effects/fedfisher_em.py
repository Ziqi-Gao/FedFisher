"""FedFisher-EM: local training, one upload, strict weighted aggregation.

The new primary objective uses unquantized float64 mean Fisher diagonals and
target site weights once. It is intentionally distinct from the original
unweighted, quantized, validation-selected diagonal routine. Any legacy rule
below is explicitly evaluator-only and cannot be called by the strict server.
"""

from dataclasses import asdict, dataclass
import math
from time import perf_counter, process_time

import numpy as np
import torch
from torch import nn

from .fisher_models import (OutcomeEffectModel, array_hash, input_coordinates,
    make_model, mean_categorical_fisher_diag, model_from_vector, model_to_arrays,
    model_vector, restore_model)
from .fisher_summary import (build_basis, build_score_matrices, diagnostic_scores,
                            quadratic_scores)
from .interventions import permutation_hash, validate_permutations
from .types import Rows
from .wire import communication_report, pack_payload, unpack_payload


TRAINING_METHOD = "fedfisher_em_training_v1"
SCORING_METHOD = "fedfisher_em_v1"
COORDINATES = "shared_raw_[A,X,A_times_X]_no_site_standardization"


@dataclass(frozen=True)
class FedFisherEMConfig:
    local_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    rank: int = 16
    compression: str = "fisher_gradient"
    score_damping: float = 0.001
    basis_max_rows: int = 8
    basis_repeats: int = 1
    range_oversampling: int = 8
    summary_repeats: int = 5
    basis_seed_offset: int = 7100037
    direction_seed_offset: int = 7200037
    loader_seed_offset: int = 7300037

    def __post_init__(self):
        for name in ("local_epochs", "batch_size", "rank", "basis_max_rows", "basis_repeats", "summary_repeats"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("range_oversampling", "basis_seed_offset", "direction_seed_offset", "loader_seed_offset"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("%s must be a nonnegative integer" % name)
        if self.rank not in (16, 32, 64) or self.summary_repeats != 5:
            raise ValueError("the declared ranks are 16/32/64 and holdout repeats are 5")
        if self.range_oversampling > 32:
            raise ValueError("range oversampling is capped at 32 to keep the sketch far below parameter dimension")
        if self.compression not in ("fisher_gradient", "euclidean_gradient", "random_orthogonal"):
            raise ValueError("unsupported effect-gradient compression")
        for name in ("learning_rate", "momentum", "score_damping"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("%s must be finite numeric" % name)
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1 or self.score_damping <= 0:
            raise ValueError("invalid SGD settings or scoring-metric damping")


@dataclass
class LocalFisherState:
    model: nn.Module
    model_name: str
    covariate_dim: int
    theta: np.ndarray
    fisher: np.ndarray
    train: Rows
    seed: int
    site_id: int
    config: FedFisherEMConfig
    initial_fingerprint: str
    fisher_computed: bool
    report: dict


def fit_local(train, model_name, seed, config, *, site_id=0, compute_fisher=True):
    if not isinstance(train, Rows) or not isinstance(config, FedFisherEMConfig):
        raise TypeError("fit_local accepts observed training Rows and FedFisherEMConfig")
    if type(site_id) is not int or site_id < 0 or type(compute_fisher) is not bool:
        raise ValueError("invalid local site identity or Fisher-computation flag")
    started_wall, started_cpu = perf_counter(), process_time()
    model = make_model(model_name, train.x.shape[1], seed)
    initial_hash = array_hash(model_vector(model))
    features = torch.tensor(input_coordinates(train.x, train.a), dtype=torch.float64, device="cpu")
    outcomes = torch.tensor(train.y, dtype=torch.long, device="cpu")
    generator = torch.Generator(device="cpu")
    loader_seed = config.loader_seed_offset + seed + site_id * 100003
    generator.manual_seed(loader_seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    training_wall, training_cpu = perf_counter(), process_time()
    model.train()
    updates = 0
    for _ in range(config.local_epochs):
        order = torch.randperm(len(train), generator=generator, device="cpu")
        for start in range(0, len(train), config.batch_size):
            indices = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features[indices]), outcomes[indices])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite local outcome loss")
            loss.backward()
            optimizer.step()
            updates += 1
    training_wall, training_cpu = perf_counter() - training_wall, process_time() - training_cpu
    theta = model_vector(model)
    fisher_wall, fisher_cpu = perf_counter(), process_time()
    fisher = mean_categorical_fisher_diag(model, train) if compute_fisher else np.zeros_like(theta)
    fisher_wall, fisher_cpu = perf_counter() - fisher_wall, process_time() - fisher_cpu
    model.eval()
    report = dict(
        training_scope="T_only_fixed_epochs_no_V_or_H_selection_or_refit",
        fisher_status="computed_mean_true_categorical_diag" if compute_fisher else "not_computed",
        fisher_examples=len(train) if compute_fisher else 0,
        fisher_normalization="divide_by_n_T_once" if compute_fisher else None,
        fisher_uses_observed_labels=False,
        model_name=model_name, covariate_dim=train.x.shape[1], site_id=site_id, seed=seed,
        n_train=len(train), parameter_count=theta.size, local_epochs=config.local_epochs,
        gradient_updates=updates, training_examples_seen=len(train) * config.local_epochs,
        loader_seed=loader_seed, optimizer="SGD", momentum=config.momentum,
        input_coordinates=COORDINATES, precision="float64_after_common_float32_initialization",
        quantization="none", train_row_ids_sha256=array_hash(train.row_ids),
        train_x_sha256=array_hash(train.x), initial_theta_sha256=initial_hash,
        theta_sha256=array_hash(theta), fisher_sha256=array_hash(fisher),
        training_wall_seconds=training_wall, training_cpu_seconds=training_cpu,
        fisher_wall_seconds=fisher_wall, fisher_cpu_seconds=fisher_cpu,
        full_local_wall_seconds=perf_counter() - started_wall,
        full_local_cpu_seconds=process_time() - started_cpu)
    return LocalFisherState(model, model_name, train.x.shape[1], theta, fisher, train,
                            seed, site_id, config, initial_hash, compute_fisher, report)


def _local_metadata(local, n_total, site_id):
    if not isinstance(local, LocalFisherState) or local.fisher_computed is not True:
        raise ValueError("strict Fisher payload requires a computed local Fisher")
    if type(n_total) is not int or n_total < len(local.train) or site_id != local.site_id:
        raise ValueError("local sample count/site identity mismatch")
    if array_hash(model_vector(local.model)) != array_hash(local.theta):
        raise ValueError("local model changed after Fisher calculation")
    return dict(protocol="one-shot-effects-v1", site_id=site_id, n_total=n_total,
                n_train=len(local.train), model_name=local.model_name,
                covariate_dim=local.covariate_dim, input_coordinates=COORDINATES,
                seed=local.seed, config=asdict(local.config),
                initial_theta_sha256=local.initial_fingerprint, local_fit=local.report)


def build_training_payload(local, n_total, site_id):
    """Stage-2 upload: strictly no basis construction or holdout scoring."""
    return pack_payload(TRAINING_METHOD, {"theta": local.theta, "fisher": local.fisher},
                        _local_metadata(local, n_total, site_id))


def build_payload(local, holdout_x, perturbations, config, n_total, site_id):
    """Stage-3 one-time serialized upload; its server never receives H rows."""
    started_wall, started_cpu = perf_counter(), process_time()
    if config != local.config:
        raise ValueError("summary configuration differs from local training protocol")
    x = np.asarray(holdout_x, dtype=np.float64)
    validate_permutations(x, perturbations)
    if x.shape[1] != local.covariate_dim or len(x) < 1 or n_total < len(local.train) + len(x):
        raise ValueError("holdout shape/count is incompatible with the local site")
    metadata = _local_metadata(local, n_total, site_id)
    basis, basis_report = build_basis(local, config)
    matrices, summary_report = build_score_matrices(local, basis, x, perturbations, config)
    metadata.update(n_holdout=len(x), holdout_x_sha256=array_hash(x),
        perturbations_sha256=permutation_hash(perturbations), basis=basis_report,
        summary=summary_report, build_payload_wall_seconds=perf_counter() - started_wall,
        build_payload_cpu_seconds=process_time() - started_cpu)
    return pack_payload(SCORING_METHOD, {"theta": local.theta, "fisher": local.fisher,
                                       "basis": basis, "score_matrices": matrices}, metadata)


def strict_diagonal_solution(thetas, fishers, weights, *, zero_curvature_fallback=None):
    """Solve exactly the declared PSD diagonal objective, with no damping/cutoff.

    Positive directions are rescaled coordinatewise before division, preserving
    tiny but nonzero curvature instead of silently replacing it with FedAvg.
    """
    thetas, fishers, weights = (np.asarray(value, dtype=np.float64) for value in (thetas, fishers, weights))
    if (thetas.ndim != 2 or fishers.shape != thetas.shape or weights.shape != (len(thetas),)
            or not len(thetas) or not np.isfinite(thetas).all() or not np.isfinite(fishers).all()
            or not np.isfinite(weights).all() or np.any(fishers < 0) or np.any(weights <= 0)):
        raise ValueError("invalid strict diagonal objective inputs")
    fallback = (weights / weights.sum()) @ thetas if zero_curvature_fallback is None else np.asarray(zero_curvature_fallback, dtype=np.float64)
    if fallback.shape != (thetas.shape[1],) or not np.isfinite(fallback).all():
        raise ValueError("invalid declared zero-curvature fallback")
    scale = fishers.max(axis=0)
    positive = scale > 0
    relative = np.zeros_like(fishers)
    relative[:, positive] = fishers[:, positive] / scale[positive]
    weighted = weights[:, None] * relative
    denominator = weighted.sum(axis=0)
    theta = fallback.copy()
    theta[positive] = np.sum(weighted[:, positive] * thetas[:, positive], axis=0) / denominator[positive]
    residual = np.sum(weighted * (theta[None, :] - thetas), axis=0)
    relative_residual = np.max(np.abs(residual[positive]) / (denominator[positive] * np.maximum(1., np.max(np.abs(thetas[:, positive]), axis=0)))) if np.any(positive) else 0.
    objective = .5 * float(np.sum(weights[:, None] * fishers * np.square(theta[None, :] - thetas)))
    if not np.isfinite(theta).all() or not math.isfinite(objective) or relative_residual > 1e-10:
        raise FloatingPointError("strict diagonal solution failed finite/stationarity checks")
    return theta, dict(objective_value=objective,
        objective="0.5_sum_site_weight_times_mean_Fisher_times_squared_parameter_displacement",
        positive_curvature_coordinates=int(positive.sum()), zero_curvature_coordinates=int((~positive).sum()),
        zero_curvature_rule="exact_zero_only_declared_FedAvg_fallback", curvature_threshold=0.,
        aggregation_damping=0., quantization="none_unless_explicitly_labeled_legacy_diagnostic",
        arithmetic="float64_coordinatewise_curvature_scaling", stationarity_relative_max=float(relative_residual),
        stationarity_tolerance=1e-10, theta_sha256=array_hash(theta))


def _unpack_sites(payloads, config, method):
    if not isinstance(config, FedFisherEMConfig) or not isinstance(payloads, (list, tuple)) or not payloads:
        raise ValueError("server requires serialized site payloads and a fixed configuration")
    sites = [unpack_payload(blob, method) for blob in payloads]
    expected_arrays = {"theta", "fisher"} if method == TRAINING_METHOD else {"theta", "fisher", "basis", "score_matrices"}
    expected_metadata = {"protocol", "site_id", "n_total", "n_train", "model_name", "covariate_dim", "input_coordinates", "seed", "config", "initial_theta_sha256", "local_fit"}
    if method == SCORING_METHOD:
        expected_metadata |= {"n_holdout", "holdout_x_sha256", "perturbations_sha256", "basis", "summary", "build_payload_wall_seconds", "build_payload_cpu_seconds"}
    identity = None
    for index, (arrays, metadata) in enumerate(sites):
        if (set(arrays) != expected_arrays or set(metadata) != expected_metadata
                or type(metadata["site_id"]) is not int or metadata["site_id"] != index or metadata["config"] != asdict(config)):
            raise ValueError("server payload array/site/configuration identity mismatch")
        if metadata["protocol"] != "one-shot-effects-v1" or metadata["input_coordinates"] != COORDINATES:
            raise ValueError("incompatible parameter coordinates")
        current = (metadata["model_name"], metadata["covariate_dim"], metadata["seed"], metadata["initial_theta_sha256"])
        if identity is None:
            identity = current
        elif identity != current:
            raise ValueError("sites did not use common initialization/model coordinates")
        if type(metadata["n_total"]) is not int or type(metadata["n_train"]) is not int or not 0 < metadata["n_train"] <= metadata["n_total"]:
            raise ValueError("invalid site sample counts")
        theta, fisher = arrays["theta"], arrays["fisher"]
        if theta.ndim != 1 or theta.dtype != np.float64 or fisher.dtype != np.float64 or fisher.shape != theta.shape or np.any(fisher < 0):
            raise ValueError("invalid theta/Fisher vector")
        if (metadata["local_fit"]["fisher_status"] != "computed_mean_true_categorical_diag"
                or metadata["local_fit"]["fisher_sha256"] != array_hash(fisher)
                or metadata["local_fit"]["theta_sha256"] != array_hash(theta)):
            raise ValueError("Fisher training evidence mismatch")
        if method == SCORING_METHOD:
            if (type(metadata["n_holdout"]) is not int or not 0 < metadata["n_holdout"] <= metadata["n_total"] - metadata["n_train"]
                    or metadata["summary"]["repeats"] != config.summary_repeats):
                raise ValueError("invalid holdout summary sample counts or repetitions")
            basis, matrices = arrays["basis"], arrays["score_matrices"]
            if (basis.shape != (theta.size, config.rank) or matrices.shape != (metadata["covariate_dim"], config.rank + 1, config.rank + 1)
                    or basis.dtype != np.float64 or matrices.dtype != np.float64
                    or not np.allclose(basis.T @ basis, np.eye(config.rank), rtol=0, atol=1e-9)
                    or not np.allclose(matrices, matrices.transpose(0, 2, 1), rtol=0, atol=1e-10)):
                raise ValueError("invalid orthogonal basis or scoring matrices")
            if np.min(np.linalg.eigvalsh(matrices)) < -1e-10 * max(1., float(np.linalg.norm(matrices))):
                raise ValueError("scoring matrices are not positive semidefinite")
    # Constructor validates the complete architecture/vector dimension.
    model_from_vector(identity[0], identity[1], sites[0][0]["theta"])
    return sites


def _aggregate_sites(sites, payloads):
    counts = np.asarray([metadata["n_total"] for _, metadata in sites], dtype=np.float64)
    weights = counts / counts.sum()
    theta, report = strict_diagonal_solution(np.stack([arrays["theta"] for arrays, _ in sites]),
        np.stack([arrays["fisher"] for arrays, _ in sites]), weights)
    metadata = sites[0][1]
    model = OutcomeEffectModel(metadata["model_name"], metadata["covariate_dim"],
        model_from_vector(metadata["model_name"], metadata["covariate_dim"], theta))
    report.update(method="target_weighted_unquantized_float64_strict_FedFisher_Diag",
        model_name=metadata["model_name"], covariate_dim=metadata["covariate_dim"],
        site_weights=weights.tolist(), site_total_samples=counts.astype(np.int64).tolist(),
        weighting="pi_s=n_total_s/sum_n_total_applied_once_to_site_mean_Fisher",
        server_reads_raw_data=False, communication=communication_report(payloads), completion_eligible=True)
    return model, theta, weights, report


def aggregate_training_payloads(payloads, config):
    started_wall, started_cpu = perf_counter(), process_time()
    sites = _unpack_sites(payloads, config, TRAINING_METHOD)
    model, _, _, report = _aggregate_sites(sites, payloads)
    report.update(stage="aggregation_and_linearity_no_low_rank_summary", wall_seconds=perf_counter() - started_wall,
                  cpu_seconds=process_time() - started_cpu)
    return model, report


def aggregate(payloads, config):
    started_wall, started_cpu = perf_counter(), process_time()
    sites = _unpack_sites(payloads, config, SCORING_METHOD)
    model, theta, weights, report = _aggregate_sites(sites, payloads)
    site_scores, distances = [], []
    for arrays, _ in sites:
        scores, _ = quadratic_scores(theta, arrays["theta"], arrays["fisher"], arrays["basis"], arrays["score_matrices"], config)
        site_scores.append(scores)
        delta = theta - arrays["theta"]
        distances.append(dict(euclidean=float(np.linalg.norm(delta)), Fisher=math.sqrt(float(arrays["fisher"] @ np.square(delta)))))
    scores = weights @ np.stack(site_scores)
    report.update(stage="compressed_post_aggregation_scores", score_definition="target_pi_weighted_H_mean_half_squared_perturbation",
                  score_exactness="exact_for_compressed_local_linearization_not_full_nonlinear_model",
                  compression=config.compression, rank=config.rank, per_site_scores=[value.tolist() for value in site_scores],
                  site_distances=distances, score_sha256=array_hash(scores),
                  wall_seconds=perf_counter() - started_wall, cpu_seconds=process_time() - started_cpu)
    return model, scores, report


def validate_server_result(payloads, model, scores_or_None, report, config, training_only=False):
    """Recheck a persisted server result using uploaded summaries only.

    Payloads are the complete wire records received by this method, including
    any equal-budget envelope. Decoding checks padding; accounting charges it.
    No training, prediction, gradient calculation, or raw-row access occurs.
    """
    if type(training_only) is not bool or not isinstance(model, OutcomeEffectModel):
        raise ValueError("invalid server-result validation mode or model")
    try:
        sites = _unpack_sites(payloads, config, TRAINING_METHOD if training_only else SCORING_METHOD)
        counts = np.asarray([metadata["n_total"] for _, metadata in sites], dtype=np.float64)
        weights = counts / counts.sum()
        theta, objective = strict_diagonal_solution(
            np.stack([arrays["theta"] for arrays, _ in sites]),
            np.stack([arrays["fisher"] for arrays, _ in sites]), weights)
        metadata = sites[0][1]
        if (model.model_name != metadata["model_name"] or model.covariate_dim != metadata["covariate_dim"]
                or not np.array_equal(model_vector(model.model), theta)):
            raise ValueError("model does not solve the uploaded strict weighted Fisher objective")
        for key, value in objective.items():
            if report[key] != value:
                raise ValueError("strict objective evidence mismatch: " + key)
        if (report["site_weights"] != weights.tolist() or report["site_total_samples"] != counts.astype(np.int64).tolist()
                or report["model_name"] != model.model_name or report["covariate_dim"] != model.covariate_dim
                or report["completion_eligible"] is not True or report["server_reads_raw_data"] is not False
                or report["method"] != "target_weighted_unquantized_float64_strict_FedFisher_Diag"
                or report["weighting"] != "pi_s=n_total_s/sum_n_total_applied_once_to_site_mean_Fisher"):
            raise ValueError("server model/target-weight/completion binding mismatch")
        if report["communication"] != communication_report(payloads):
            raise ValueError("complete serialized payload communication mismatch")
        for key in ("wall_seconds", "cpu_seconds"):
            value = report[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid server timing observation")
        if training_only:
            if scores_or_None is not None or report["stage"] != "aggregation_and_linearity_no_low_rank_summary":
                raise ValueError("training-only result must not claim compressed scores")
            return
        site_scores, distances = [], []
        for arrays, _ in sites:
            values, _ = quadratic_scores(theta, arrays["theta"], arrays["fisher"], arrays["basis"], arrays["score_matrices"], config)
            site_scores.append(values)
            delta = theta - arrays["theta"]
            distances.append(dict(euclidean=float(np.linalg.norm(delta)), Fisher=math.sqrt(float(arrays["fisher"] @ np.square(delta)))))
        expected_scores = weights @ np.stack(site_scores)
        if (not np.array_equal(np.asarray(scores_or_None), expected_scores)
                or report["per_site_scores"] != [value.tolist() for value in site_scores]
                or report["score_sha256"] != array_hash(expected_scores) or report["site_distances"] != distances
                or report["rank"] != config.rank or report["compression"] != config.compression
                or report["stage"] != "compressed_post_aggregation_scores"
                or report["score_definition"] != "target_pi_weighted_H_mean_half_squared_perturbation"
                or report["score_exactness"] != "exact_for_compressed_local_linearization_not_full_nonlinear_model"):
            raise ValueError("compressed summary scores or approximation identity mismatch")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed FedFisher-EM server result") from error


def full_gradient_diagnostics(local, theta_final, holdout_x, perturbations):
    return diagnostic_scores(local, theta_final, holdout_x, perturbations, local.config)


def simulation_diagnostics(local, payload, theta_final, holdout_x, perturbations):
    arrays, metadata = unpack_payload(payload, SCORING_METHOD)
    if metadata["site_id"] != local.site_id or metadata["holdout_x_sha256"] != array_hash(np.asarray(holdout_x, dtype=np.float64)) or metadata["perturbations_sha256"] != permutation_hash(perturbations):
        raise ValueError("diagnostic rows/perturbations differ from uploaded summary")
    return diagnostic_scores(local, theta_final, holdout_x, perturbations, local.config,
                             arrays["basis"], arrays["score_matrices"])


@dataclass(frozen=True)
class LegacyDiagnosticConfig:
    iterations: int = 2000
    evaluate_every: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.9
    second_moment: float = 0.99
    denominator_offset: float = 0.01
    quantization_s: int = 15

    def __post_init__(self):
        for name in ("iterations", "evaluate_every", "quantization_s"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("legacy " + name + " must be a positive integer")
        if self.quantization_s > 52:
            raise ValueError("legacy quantization must fit float64 integer precision")
        for name in ("learning_rate", "momentum", "second_moment", "denominator_offset"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("legacy " + name + " must be finite")
        if (self.learning_rate <= 0 or self.denominator_offset <= 0
                or not 0 <= self.momentum < 1 or not 0 <= self.second_moment < 1):
            raise ValueError("invalid legacy optimizer configuration")


LEGACY_SCOPE = "simulation_evaluator_only_not_deployable_one_upload_server"
LEGACY_SELECTION = "after_step_k_mod_interval_zero_strict_accuracy_improvement_starting_best_zero"
LEGACY_DIFFERENCES = ("new_T_V_H_split", "CPU_float64_not_original_CUDA_float32", "pi_from_n_total",
                      "original_V_was_at_most_500_pooled_training_rows")
LEGACY_VALIDATION_LIMITS = dict(accuracy_observations_verified=False, iteration_trajectory_verified=False,
    statement="summary checks bind recorded accuracy arithmetic and selected hashes; raw V observations and optimizer trajectory are not independently verified")


def _legacy_quantize(vector, model, s):
    quantized, start = np.empty_like(vector), 0
    for parameter in model.parameters():
        stop = start + parameter.numel()
        block = vector[start:stop]
        maximum = float(np.max(np.abs(block)))
        if maximum == 0:
            raise FloatingPointError("legacy quantize would produce 0/0 on an all-zero parameter tensor; preserved failure")
        levels = 2**s - 1
        normalized = block / maximum
        quantized[start:stop] = maximum * np.round(levels * normalized) / levels
        start = stop
    return quantized


def _legacy_uploaded_objective(sites, legacy):
    """Algebra shared by the diagnostic and its raw-data-free verifier."""
    metadata = sites[0][1]
    template = model_from_vector(metadata["model_name"], metadata["covariate_dim"], sites[0][0]["theta"])
    theta = np.stack([_legacy_quantize(arrays["theta"], template, legacy.quantization_s) for arrays, _ in sites])
    fisher = np.stack([_legacy_quantize(arrays["fisher"], template, legacy.quantization_s) for arrays, _ in sites])
    n_total = np.asarray([item[1]["n_total"] for item in sites], dtype=np.float64)
    pi = n_total / n_total.sum()
    initial = pi @ theta
    strict_theta, strict_report = strict_diagonal_solution(theta, fisher, np.ones(len(sites)), zero_curvature_fallback=initial)
    strict_report.update(method="strict_legacy_uploaded_objective", quantization="per_parameter_tensor_s%d_cpu_float64_rule_replay" % legacy.quantization_s,
                         weighting="unweighted_sum_of_quantized_site_mean_Fisher", zero_curvature_rule="same_quantized_pi_FedAvg_initialization")
    return theta, fisher, n_total, pi, initial, strict_theta, strict_report


def legacy_validation_diagnostic(payloads, validation_sites, config, *, legacy_config=None, selection_weighting="target_pi_mean"):
    """Evaluator-only old iteration state/selection, plus its exact same-objective solution.

    The supplied V rows are never passed to either production server. This uses
    new-protocol CPU float64 locals, not the old CUDA environment or overlapping
    pooled training subset; these differences are explicit in the report.
    """
    started_wall, started_cpu = perf_counter(), process_time()
    legacy = LegacyDiagnosticConfig() if legacy_config is None else legacy_config
    if (not isinstance(legacy, LegacyDiagnosticConfig) or type(legacy.iterations) is not int or legacy.iterations < 1
            or type(legacy.evaluate_every) is not int or legacy.evaluate_every < 1
            or selection_weighting not in ("target_pi_mean", "raw_pool")):
        raise ValueError("invalid simulation legacy diagnostic settings")
    sites = _unpack_sites(payloads, config, TRAINING_METHOD)
    if len(validation_sites) != len(sites) or any(not isinstance(rows, Rows) for rows in validation_sites):
        raise ValueError("diagnostic needs one explicit validation Rows object per site")
    metadata = sites[0][1]
    theta, fisher, n_total, pi, initial, strict_theta, strict_report = _legacy_uploaded_objective(sites, legacy)
    validation_counts = np.asarray([len(rows) for rows in validation_sites], dtype=np.int64)
    selection_weights = pi if selection_weighting == "target_pi_mean" else validation_counts / validation_counts.sum()
    w, momentum, second = initial.copy(), np.zeros_like(initial), np.zeros_like(initial)
    F_sum, b_sum = fisher.sum(axis=0), (fisher * theta).sum(axis=0)
    best_accuracy, best_theta, selected_step, history = 0., None, None, []

    def validation_accuracy(vector):
        model = model_from_vector(metadata["model_name"], metadata["covariate_dim"], vector)
        accuracy = []
        with torch.no_grad():
            for rows in validation_sites:
                logits = model(torch.tensor(input_coordinates(rows.x, rows.a), dtype=torch.float64, device="cpu"))
                accuracy.append(float((logits.argmax(dim=1).numpy() == rows.y).mean()) * 100.)
        weighted = float(pi @ accuracy) if selection_weighting == "target_pi_mean" else float(np.average(accuracy, weights=validation_counts))
        return weighted, accuracy

    for step in range(legacy.iterations):
        gradient = F_sum * w - b_sum
        momentum = gradient + legacy.momentum * momentum
        second = gradient * gradient + legacy.second_moment * second
        w = w - legacy.learning_rate * momentum / (np.sqrt(second) + legacy.denominator_offset)
        if not np.isfinite(w).all():
            raise FloatingPointError("legacy iterative diagnostic became nonfinite")
        if step % legacy.evaluate_every == 0:
            accuracy, site_accuracies = validation_accuracy(w)
            if accuracy > best_accuracy:
                best_accuracy, best_theta, selected_step = accuracy, w.copy(), step
            history.append(dict(zero_based_step=step, accuracy=accuracy,
                                per_site_accuracy=site_accuracies, theta_sha256=array_hash(w)))
    if best_theta is None:
        raise RuntimeError("legacy strict-greater-than-zero accuracy selection never defined best_parameters")
    selected_model = OutcomeEffectModel(metadata["model_name"], metadata["covariate_dim"], model_from_vector(metadata["model_name"], metadata["covariate_dim"], best_theta))
    strict_model = OutcomeEffectModel(metadata["model_name"], metadata["covariate_dim"], model_from_vector(metadata["model_name"], metadata["covariate_dim"], strict_theta))
    report = dict(scope=LEGACY_SCOPE, legacy_configuration=asdict(legacy),
        selection_weighting=selection_weighting, selection_rule=LEGACY_SELECTION,
        selected_zero_based_step=selected_step, best_accuracy=best_accuracy, selection_history=history,
        selected_theta_sha256=array_hash(best_theta), target_weights=pi.tolist(),
        site_total_samples=n_total.astype(np.int64).tolist(), validation_site_samples=validation_counts.tolist(),
        selection_site_weights=selection_weights.tolist(), summary_validation_limits=dict(LEGACY_VALIDATION_LIMITS),
        historical_equivalence_claim=False,
        differences_from_historical_run=list(LEGACY_DIFFERENCES),
        strict_legacy_uploaded_objective=strict_report,
        selected_to_same_objective_strict_distance=float(np.linalg.norm(best_theta - strict_theta)),
        wall_seconds=perf_counter() - started_wall, cpu_seconds=process_time() - started_cpu)
    return selected_model, strict_model, report


def validate_legacy_diagnostic(payloads, selected_model, strict_old_model, report, config, legacy_config):
    """Validate recorded legacy evidence using summaries and saved parameters.

    No optimizer iterations, model inference, or raw V reads occur. Recorded
    candidate hashes bind the selected artifact but cannot prove an unretained
    optimization trajectory. Accuracy arithmetic and selection are checked;
    the original accuracy observations are not independently verified.
    """
    if (not isinstance(legacy_config, LegacyDiagnosticConfig)
            or not isinstance(selected_model, OutcomeEffectModel)
            or not isinstance(strict_old_model, OutcomeEffectModel)):
        raise ValueError("legacy validation requires configured saved models")
    try:
        sites = _unpack_sites(payloads, config, TRAINING_METHOD)
        _, _, n_total, pi, _, strict_theta, strict_report = _legacy_uploaded_objective(sites, legacy_config)
        metadata = sites[0][1]
        for model in (selected_model, strict_old_model):
            if model.model_name != metadata["model_name"] or model.covariate_dim != metadata["covariate_dim"]:
                raise ValueError("legacy diagnostic model identity mismatch")
        if (not np.array_equal(model_vector(strict_old_model.model), strict_theta)
                or report["strict_legacy_uploaded_objective"] != strict_report):
            raise ValueError("legacy strict model does not solve the uploaded quantized unweighted objective")
        if (report["legacy_configuration"] != asdict(legacy_config) or report["scope"] != LEGACY_SCOPE
                or report["selection_rule"] != LEGACY_SELECTION or report["historical_equivalence_claim"] is not False
                or report["differences_from_historical_run"] != list(LEGACY_DIFFERENCES)
                or report["summary_validation_limits"] != LEGACY_VALIDATION_LIMITS
                or report["target_weights"] != pi.tolist()
                or report["site_total_samples"] != n_total.astype(np.int64).tolist()):
            raise ValueError("legacy diagnostic configuration, scope, or target-weight binding mismatch")
        counts = report["validation_site_samples"]
        if (not isinstance(counts, list) or len(counts) != len(sites)
                or any(type(n) is not int or n < 1 or n > meta["n_total"] - meta["n_train"]
                       for n, (_, meta) in zip(counts, sites))):
            raise ValueError("invalid recorded validation sample counts")
        mode = report["selection_weighting"]
        if mode not in ("target_pi_mean", "raw_pool"):
            raise ValueError("unrecognized legacy diagnostic selection population")
        expected_weights = pi if mode == "target_pi_mean" else np.asarray(counts) / sum(counts)
        if report["selection_site_weights"] != expected_weights.tolist():
            raise ValueError("legacy accuracy weights differ from the declared selection population")
        history = report["selection_history"]
        steps = list(range(0, legacy_config.iterations, legacy_config.evaluate_every))
        if not isinstance(history, list) or len(history) != len(steps):
            raise ValueError("legacy selection history has missing or extra candidates")
        best, winner = 0., None
        for expected_step, item in zip(steps, history):
            if (set(item) != {"zero_based_step", "accuracy", "per_site_accuracy", "theta_sha256"}
                    or type(item["zero_based_step"]) is not int or item["zero_based_step"] != expected_step):
                raise ValueError("legacy evaluation steps differ from the configured state rule")
            accuracy, values, digest = item["accuracy"], item["per_site_accuracy"], item["theta_sha256"]
            if (not isinstance(values, list) or len(values) != len(sites)
                    or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100
                           for value in [accuracy, *values])
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid legacy candidate accuracy or parameter hash")
            weighted = float(pi @ values) if mode == "target_pi_mean" else float(np.average(values, weights=counts))
            if accuracy != weighted:
                raise ValueError("recorded accuracy is inconsistent with the recorded site means")
            if accuracy > best:
                best, winner = accuracy, item
        if (winner is None or type(report["selected_zero_based_step"]) is not int
                or report["selected_zero_based_step"] != winner["zero_based_step"]
                or report["best_accuracy"] != best):
            raise ValueError("legacy selection must use the first strict accuracy maximum above zero")
        selected_theta = model_vector(selected_model.model)
        if (report["selected_theta_sha256"] != array_hash(selected_theta)
                or winner["theta_sha256"] != report["selected_theta_sha256"]
                or report["selected_to_same_objective_strict_distance"] != float(np.linalg.norm(selected_theta - strict_theta))):
            raise ValueError("selected legacy model, candidate hash, or strict distance mismatch")
        for key in ("wall_seconds", "cpu_seconds"):
            value = report[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid legacy diagnostic timing")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed legacy diagnostic evidence") from error
