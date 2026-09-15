"""Numerical primitives for evaluator-only checkpoint replay, never training.

Parameter paths are diagnostic forward evaluations. Neither this module nor
its caller provides a hospital-to-server scoring or optimization interface.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..fisher_models import input_coordinates, model_vector


def at_vector(model, vector):
    """Copy a saved model; never mutate the source checkpoint."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != model_vector(model).shape or not np.isfinite(vector).all():
        raise ValueError("finite parameter vector of the saved architecture required")
    result = deepcopy(model).eval()
    vector_to_parameters(torch.tensor(vector, dtype=torch.float64), result.parameters())
    return result


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -values))


def outputs(model, x, batch_size=256, activations=False):
    """Binary log-odds for control/treatment, plus each ReLU's input signs.

Columns are l(Y=1)-l(Y=0), first A=0 and then A=1. They are not
individual, non-identifiable class logits. Counterfactual coordinates remain
consistent [A, X, A*X]. Exact zero preactivations are retained separately.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("nonempty finite raw covariates required")
    model.eval()
    values = np.empty((len(x), 2))
    signs, zeros = {}, {}
    handles, captured = [], {}
    if activations:
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                def capture(_module, args, key=name):
                    captured[key] = args[0].detach().numpy().copy()
                handles.append(module.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for arm in (0, 1):
                for start in range(0, len(x), batch_size):
                    stop = min(len(x), start + batch_size)
                    features = input_coordinates(x[start:stop], np.full(stop - start, arm))
                    logits = model(torch.tensor(features, dtype=torch.float64))
                    values[start:stop, arm] = (logits[:, 1] - logits[:, 0]).numpy()
                    for name, preactivation in captured.items():
                        if name not in signs:
                            shape = (len(x), 2, preactivation.shape[1])
                            signs[name], zeros[name] = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
                        signs[name][start:stop, arm] = preactivation > 0
                        zeros[name][start:stop, arm] = preactivation == 0
    finally:
        for handle in handles:
            handle.remove()
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite checkpoint log-odds")
    probabilities = sigmoid(values)
    return dict(logits=values, tau=probabilities[:, 1] - probabilities[:, 0],
                probabilities=probabilities, signs=signs, zeros=zeros)


def local_derivatives(model, x, directions, direct_tau=False):
    """Cache scalar directional derivatives, never a parameter-square matrix.

The path uses exact sigmoid chain-rule derivatives of the two log-odds.
The finite-difference audit additionally differentiates tau independently.
    """
    x, directions = np.asarray(x, dtype=float), np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[0] != model_vector(model).size:
        raise ValueError("aligned parameter directions required")
    if not np.isfinite(directions).all():
        raise ValueError("finite directions required")
    baseline, projected = np.empty((len(x), 3)), np.empty((len(x), 3, directions.shape[1]))
    parameters = tuple(model.parameters())
    model.eval()
    for index, row in enumerate(x):
        features = input_coordinates(np.repeat(row[None, :], 2, axis=0), np.array([0., 1.]))
        raw_logits = model(torch.tensor(features, dtype=torch.float64))
        logits = raw_logits[:, 1] - raw_logits[:, 0]
        probability = torch.sigmoid(logits)
        tau = torch.softmax(raw_logits, dim=1)[1, 1] - torch.softmax(raw_logits, dim=1)[0, 1]
        baseline[index] = [float(logits[0].detach()), float(logits[1].detach()), float(tau.detach())]
        for arm in (0, 1):
            gradients = torch.autograd.grad(logits[arm], parameters, retain_graph=arm == 0 or direct_tau)
            vector = parameters_to_vector(gradients).detach().numpy()
            projected[index, arm] = vector @ directions
        p = probability.detach().numpy()
        chain = p[1] * (1 - p[1]) * projected[index, 1] - p[0] * (1 - p[0]) * projected[index, 0]
        if direct_tau:
            gradients = torch.autograd.grad(tau, parameters)
            projected[index, 2] = parameters_to_vector(gradients).detach().numpy() @ directions
        else:
            projected[index, 2] = chain
    if not np.isfinite(baseline).all() or not np.isfinite(projected).all():
        raise FloatingPointError("nonfinite local derivative")
    return baseline, projected


def gradient_check(model, x, delta, random_direction, steps, atol, rtol):
    """Audit normalized actual/random directions and explicitly label kinks."""
    theta = model_vector(model)
    supplied = dict(actual_aggregation=np.asarray(delta), fixed_random=np.asarray(random_direction))
    directions, norms = [], {}
    for name, value in supplied.items():
        norm = float(np.linalg.norm(value))
        norms[name] = norm
        directions.append(value / norm if norm > 0 else np.zeros_like(value))
    directions = np.column_stack(directions)
    baseline, derivative = local_derivatives(model, x, directions, direct_tau=True)
    p = sigmoid(baseline[:, :2])
    chain = (p[:, 1, None] * (1 - p[:, 1, None]) * derivative[:, 1, :] -
             p[:, 0, None] * (1 - p[:, 0, None]) * derivative[:, 0, :])
    chain_error = float(np.max(np.abs(chain - derivative[:, 2, :])))
    original = outputs(model, x, activations=True)
    endpoint = np.column_stack((original['logits'], original['tau']))
    t0_max = float(np.max(np.abs(endpoint - baseline)))
    records, stable_count, failures, boundary_count = [], 0, 0, 0
    for index, name in enumerate(supplied):
        for step in steps:
            plus = outputs(at_vector(model, theta + step * directions[:, index]), x, activations=True)
            minus = outputs(at_vector(model, theta - step * directions[:, index]), x, activations=True)
            numerical = (np.column_stack((plus['logits'], plus['tau'])) -
                         np.column_stack((minus['logits'], minus['tau']))) / (2 * step)
            stable_arm = np.ones((len(x), 2), dtype=bool)
            layer_crossings = {}
            for layer, sign in original['signs'].items():
                crossing = ((sign != plus['signs'][layer]) | (sign != minus['signs'][layer]) |
                            original['zeros'][layer] | plus['zeros'][layer] | minus['zeros'][layer])
                stable_arm &= ~crossing.any(axis=2)
                layer_crossings[layer] = crossing.mean(axis=(0, 2)).tolist()
            stable = np.column_stack((stable_arm, stable_arm.all(axis=1)))
            absolute = np.abs(numerical - derivative[:, :, index])
            threshold = atol + rtol * np.abs(derivative[:, :, index])
            within = absolute <= threshold
            stable_count += int(stable.sum())
            failures += int((stable & ~within).sum())
            boundary_count += int((~stable).sum())
            records.append(dict(direction=name, step=step, direction_original_norm=norms[name],
                degenerate_zero_direction=norms[name] == 0,
                autograd=derivative[:, :, index].tolist(), finite_difference=numerical.tolist(),
                absolute_error=absolute.tolist(),
                relative_error_floor_1e_12=(absolute / np.maximum(1e-12, np.abs(derivative[:, :, index]))).tolist(),
                tolerance=threshold.tolist(), stable_activation_mask=stable.tolist(),
                within_tolerance=within.tolist(), layer_boundary_fraction_control_treated=layer_crossings))
    return dict(columns=['control_binary_log_odds', 'treated_binary_log_odds', 'probability_CATE'],
        baseline_autograd_outputs=baseline.tolist(), t0_model_outputs=endpoint.tolist(),
        t0_maximum_absolute_error=t0_max, t0_tolerance=1e-12, t0_passed=t0_max <= 1e-12,
        stable_checks=stable_count, stable_failures=failures, boundary_checks=boundary_count,
        sigmoid_chain_derivative=chain.tolist(), chain_rule_maximum_absolute_error=chain_error,
        chain_rule_tolerance=1e-12, chain_rule_passed=chain_error <= 1e-12,
        passed=t0_max <= 1e-12 and chain_error <= 1e-12 and failures == 0 and stable_count > 0, checks=records,
        boundary_rule='any_baseline_or_plus_or_minus_zero_or_sign_change_in_any_ReLU; excluded_from_code_error_gate')


def error_moments(exact, approximate, weights=None):
    exact, approximate = np.asarray(exact), np.asarray(approximate)
    if exact.shape != approximate.shape or exact.ndim != 1 or not len(exact):
        raise ValueError("aligned prediction vectors required")
    weights = np.full(len(exact), 1 / len(exact)) if weights is None else np.asarray(weights)
    if weights.shape != exact.shape or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("normalized observation weights required")
    error = approximate - exact
    mean = float(weights @ error)
    return dict(mean_error=mean, mean_squared_error=float(weights @ np.square(error)),
        rmse=float(np.sqrt(weights @ np.square(error))),
        centered_rmse=float(np.sqrt(weights @ np.square(error - mean))),
        maximum_absolute_error=float(np.max(np.abs(error))), exact_mean=float(weights @ exact),
        exact_mean_square=float(weights @ np.square(exact)), approximate_mean=float(weights @ approximate),
        approximate_mean_square=float(weights @ np.square(approximate)))


def combine_moments(records, weights):
    weights = np.asarray(weights)
    additive = ('mean_error', 'mean_squared_error', 'exact_mean', 'exact_mean_square',
                'approximate_mean', 'approximate_mean_square')
    result = {key: float(weights @ [value[key] for value in records]) for key in additive}
    result.update(rmse=float(np.sqrt(result['mean_squared_error'])),
        centered_rmse=float(np.sqrt(max(0., result['mean_squared_error'] - result['mean_error'] ** 2))),
        maximum_absolute_error=max(value['maximum_absolute_error'] for value in records))
    return result


def test_weights(evaluation):
    if evaluation.site_ids is None:
        return np.full(len(evaluation.rows), 1 / len(evaluation.rows))
    weights = np.zeros(len(evaluation.rows))
    for site, weight in enumerate(evaluation.target_weights):
        selected = evaluation.site_ids == site
        if selected.sum() < 2:
            raise ValueError("each SCM target test site needs at least two rows")
        weights[selected] = weight / selected.sum()
    return weights


def cate_metrics(prediction, evaluation):
    prediction = np.asarray(prediction, dtype=float)
    weights = test_weights(evaluation)
    if prediction.shape != weights.shape or not np.isfinite(prediction).all():
        raise ValueError("finite test CATE predictions required")
    mean = float(weights @ prediction)
    result = dict(predicted_mean=mean,
        estimated_heterogeneity=float(weights @ np.square(prediction - mean)),
        estimated_standard_deviation=float(np.sqrt(weights @ np.square(prediction - mean))),
        true_cate_available=evaluation.truth is not None,
        cate_rmse=None, prediction_mean_bias=None, true_mean=None, true_heterogeneity=None,
        true_standard_deviation=None, centered_cate_rmse=None,
        test_weighting='target_pi_times_within_site_mean' if evaluation.site_ids is not None else 'legacy_pooled_common_test')
    if evaluation.truth is not None:
        truth = evaluation.truth.tau
        moments = error_moments(truth, prediction, weights)
        true_mean = float(weights @ truth)
        variance = float(weights @ np.square(truth - true_mean))
        result.update(cate_rmse=moments['rmse'], prediction_mean_bias=moments['mean_error'],
            true_mean=true_mean, true_heterogeneity=variance, true_standard_deviation=float(np.sqrt(variance)),
            centered_cate_rmse=moments['centered_rmse'])
    return result


def outcome_metrics(model, rows, weights=None):
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            features = input_coordinates(rows.x[start:start + 256], rows.a[start:start + 256])
            value = model(torch.tensor(features, dtype=torch.float64))
            logits.append((value[:, 1] - value[:, 0]).numpy())
    logits = np.concatenate(logits)
    probability = sigmoid(logits)
    weights = np.full(len(rows), 1 / len(rows)) if weights is None else np.asarray(weights)
    if weights.shape != (len(rows),) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('normalized outcome observation weights required')
    return dict(rows=len(rows), cross_entropy=float(weights @ (np.logaddexp(0., logits) - rows.y * logits)),
                brier_score=float(weights @ np.square(probability - rows.y)))
