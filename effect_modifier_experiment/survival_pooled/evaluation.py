"""Oracle probability accuracy before calculating treatment contrasts.

Scientific inference belongs to the scheduler task. The validators and summary
functions here recompute scalar arithmetic from saved arrays only; a plotting
script can use those completed arrays without loading or applying a model.
"""

import numpy as np

from effect_modifier_experiment.survival.links import survival_mean
from .methods import array_hash


EVALUATION_SCHEMA = "survival_probability_accuracy_v1"
PREDICTION_FIELDS = ("eta0", "eta1", "s0", "s1")
PROBABILITY_BINS = tuple(np.linspace(0., 1., 11).tolist())


def _predictions(predictions):
    if not isinstance(predictions, dict) or set(predictions) != set(PREDICTION_FIELDS):
        raise ValueError("probability predictions require eta0, eta1, s0 and s1")
    arrays = {key: np.asarray(predictions[key], dtype=np.float64) for key in PREDICTION_FIELDS}
    shape = arrays["eta0"].shape
    if (len(shape) != 1 or shape[0] < 1
            or any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError("invalid probability prediction arrays")
    for arm in (0, 1):
        if not np.array_equal(arrays[f"s{arm}"], survival_mean(arrays[f"eta{arm}"])):
            raise ValueError("survival prediction does not match the declared link")
    return arrays


def probability_predictions(predict_eta, dataset, batch_size=2048):
    """Predict each intervention arm independently, before any subtraction."""
    x = np.asarray(dataset.evaluation.x, dtype=np.float64)
    if (x.ndim != 2 or not len(x) or not np.isfinite(x).all()
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("invalid probability evaluation covariates/batch size")
    predictions = {"eta0": np.empty(len(x)), "eta1": np.empty(len(x))}
    for start in range(0, len(x), batch_size):
        local = x[start:start + batch_size]
        for arm in (0, 1):
            # Changing treatment also rebuilds every explicit A*X input.
            design = np.column_stack((np.full(len(local), arm, dtype=float),
                                      local, arm * local))
            eta = np.asarray(predict_eta(design), dtype=np.float64)
            if eta.shape != (len(local),) or not np.isfinite(eta).all():
                raise ValueError("invalid scalar probability linear predictor")
            predictions[f"eta{arm}"][start:start + len(local)] = eta
    predictions.update(s0=survival_mean(predictions["eta0"]),
                       s1=survival_mean(predictions["eta1"]))
    return _predictions(predictions)


def _accuracy(estimated, truth):
    error = np.asarray(estimated, dtype=float) - np.asarray(truth, dtype=float)
    n = int(len(error))
    squared = float(error @ error)
    absolute = float(np.abs(error).sum())
    total = float(error.sum())
    return dict(n=n, squared_error_sum=squared, absolute_error_sum=absolute,
                error_sum=total, mse=squared / n, rmse=float(np.sqrt(squared / n)),
                mae=absolute / n, bias=total / n)


def _calibration(predicted, truth):
    edges = np.asarray(PROBABILITY_BINS)
    assignments = np.minimum(np.searchsorted(edges, predicted, side="right") - 1, 9)
    result = []
    for index in range(10):
        selected = assignments == index
        n = int(selected.sum())
        result.append(dict(bin=index, lower=float(edges[index]), upper=float(edges[index + 1]),
                           count=n, predicted_mean=(float(predicted[selected].mean()) if n else None),
                           true_mean=(float(truth[selected].mean()) if n else None)))
    return result


def summarize_arrays(predictions, truth, treatment):
    """Compare saved fitted means with saved DGP probabilities.

    Each intervention arm is evaluated on all common test covariates, giving
    both arms equal weight. Factual accuracy additionally selects each row's
    observed treatment; it can conceal poorer accuracy in an infrequent arm.
    """
    predicted = _predictions(predictions)
    n = len(predicted["s0"])
    if not isinstance(truth, dict) or set(truth) != {"eta0", "eta1", "survival0", "survival1"}:
        raise ValueError("invalid saved probability truth fields")
    values = {key: np.asarray(value, dtype=np.float64) for key, value in truth.items()}
    if any(value.shape != (n,) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("invalid saved evaluator-only truth arrays")
    survival = {"0": values["survival0"], "1": values["survival1"]}
    if any(not np.array_equal(survival[str(arm)], survival_mean(values[f"eta{arm}"]))
           for arm in (0, 1)):
        raise ValueError("saved oracle probabilities do not match their true predictors")
    treatment = np.asarray(treatment)
    delta = values["eta1"] - values["eta0"]
    if (treatment.shape != (n,)
            or not np.isin(treatment, (0, 1)).all() or delta.shape != (n,)
            or not np.isfinite(delta).all()
            or any(s.shape != (n,) or not np.isfinite(s).all()
                   or np.any((s < 0) | (s > 1)) for s in survival.values())):
        raise ValueError("invalid evaluator-only probability truth")
    s0, s1 = predicted["s0"], predicted["s1"]
    factual_prediction = np.where(treatment == 1, s1, s0)
    factual_truth = np.where(treatment == 1, survival["1"], survival["0"])
    arms = {str(arm): _accuracy(predicted[f"s{arm}"], survival[str(arm)]) for arm in (0, 1)}
    combined = _accuracy(np.r_[s0, s1], np.r_[survival["0"], survival["1"]])
    # The primary probability checks above do not subtract predictions. Only
    # this secondary endpoint uses the original log cumulative-hazard scale.
    contrast = _accuracy(predicted["eta1"] - predicted["eta0"], delta)
    return dict(schema=EVALUATION_SCHEMA, num_test=n,
        target="independent_common_standard_gaussian", link="log_minus_log_survival",
        primary="oracle_survival_probability_accuracy_before_contrast",
        arms=arms, combined=combined, factual=_accuracy(factual_prediction, factual_truth),
        calibration={str(arm): _calibration(predicted[f"s{arm}"], survival[str(arm)])
                     for arm in (0, 1)},
        contrast=contrast, contrast_scale="log_cumulative_hazard",
        probability_truth_sha256={arm: array_hash(values) for arm, values in survival.items()},
        factual_treatment_sha256=array_hash(treatment),
        prediction_sha256={key: array_hash(values) for key, values in predicted.items()},
        evaluation_network_measured=False, end_to_end_one_round_claim=False)


def summarize_predictions(predictions, dataset):
    """Dataset convenience wrapper; numerical evidence is defined by arrays."""
    if len(dataset.evaluation.x) != len(np.asarray(predictions["eta0"])):
        raise ValueError("probability predictions and evaluation rows disagree")
    truth = {key: getattr(dataset.evaluation_truth, key)
             for key in ("eta0", "eta1", "survival0", "survival1")}
    return summarize_arrays(predictions, truth, dataset.evaluation.a)


def compare_predictions(left, right):
    """RMS prediction gaps between paired models on the same saved test rows."""
    left, right = _predictions(left), _predictions(right)
    if left["s0"].shape != right["s0"].shape:
        raise ValueError("paired prediction arrays have different lengths")
    arms = {str(arm): _accuracy(left[f"s{arm}"], right[f"s{arm}"]) for arm in (0, 1)}
    combined = _accuracy(np.r_[left["s0"], left["s1"]], np.r_[right["s0"], right["s1"]])
    return dict(num_test=len(left["s0"]), difference_direction="left_minus_right",
                arms=arms, combined=combined,
                left_sha256={key: array_hash(value) for key, value in left.items()},
                right_sha256={key: array_hash(value) for key, value in right.items()})


def validate_summary(summary, predictions, dataset):
    """Recompute saved numerical evidence without model inference."""
    if summary != summarize_predictions(predictions, dataset):
        raise ValueError("saved probability summary does not match its arrays/truth")


def validate_summary_arrays(summary, predictions, truth, treatment):
    """Saved-only acceptance check; no generator or model is invoked."""
    if summary != summarize_arrays(predictions, truth, treatment):
        raise ValueError("saved probability summary does not match its arrays/truth")


def validate_comparison(comparison, left, right):
    if comparison != compare_predictions(left, right):
        raise ValueError("saved FL/pooled comparison does not match prediction arrays")
