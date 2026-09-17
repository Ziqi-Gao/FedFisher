"""Evaluator-only treatment-contrast recovery on the log cumulative-hazard scale."""

import hashlib
import math

import numpy as np


PRIMARY = "permute/log_cumulative_hazard_contrast"
TIE_RULE = "uniform_without_replacement_within_exact_score_boundary_ties"


def _array_hash(array):
    array = np.asarray(array, dtype="<f8", order="C")
    return hashlib.sha256(str(array.shape).encode("ascii") + b"\0" + array.tobytes()).hexdigest()


def treatment_contrast(predict_eta, x, batch_size=2048):
    """Rebuild both A and A*X after every raw-X intervention."""
    x = np.asarray(x, dtype=np.float64)
    if (x.ndim != 2 or not len(x) or not np.isfinite(x).all()
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("invalid contrast inputs")
    delta = np.empty(len(x))
    for start in range(0, len(x), batch_size):
        local = x[start:start + batch_size]
        zero, one = np.zeros(len(local)), np.ones(len(local))
        control = np.column_stack((zero, local, np.zeros_like(local)))
        treated = np.column_stack((one, local, local))
        eta0 = np.asarray(predict_eta(control), dtype=float)
        eta1 = np.asarray(predict_eta(treated), dtype=float)
        if (eta0.shape != (len(local),) or eta1.shape != eta0.shape
                or not np.isfinite(eta0).all() or not np.isfinite(eta1).all()):
            raise ValueError("invalid scalar linear-predictor output")
        delta[start:start + len(local)] = eta1 - eta0
    return delta


def signal_recovery(importance, modifier_mask):
    values = np.asarray(importance, dtype=float)
    mask = np.asarray(modifier_mask)
    if (values.ndim != 1 or not len(values) or mask.shape != values.shape
            or mask.dtype.kind != "b" or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("invalid importance or modifier mask")
    k = int(mask.sum())
    if k == 0:
        return dict(expected_topk_hits=None, expected_topk_precision=None,
                    probability_all_signals_selected=None, auroc=None,
                    boundary_tie_size=None, boundary_slots=None, tie_rule=TIE_RULE)
    cutoff = np.sort(values)[-k]
    above, tied = values > cutoff, values == cutoff
    slots = k - int(above.sum())
    tied_count = int(tied.sum())
    true_above, true_tied = int((above & mask).sum()), int((tied & mask).sum())
    hits = true_above + slots * true_tied / tied_count
    if true_above + true_tied != k or true_tied > slots:
        all_selected = 0.
    else:
        all_selected = math.comb(tied_count - true_tied, slots - true_tied) / math.comb(tied_count, slots)
    if k == len(values):
        auc = None
    else:
        differences = values[mask, None] - values[None, ~mask]
        auc = float(np.mean((differences > 0) + .5 * (differences == 0)))
    return dict(expected_topk_hits=float(hits), expected_topk_precision=float(hits / k),
                probability_all_signals_selected=float(all_selected), auroc=auc,
                boundary_tie_size=tied_count, boundary_slots=slots, tie_rule=TIE_RULE)


def evaluate_model(predict_eta, dataset, repeats=5, seed=0):
    """Synthetic common-test evaluation, not a deployed one-round EHR evaluator."""
    if type(repeats) is not int or repeats < 1 or type(seed) is not int or seed < 0:
        raise ValueError("invalid evaluation repeats or seed")
    x = dataset.evaluation.x
    baseline = treatment_contrast(predict_eta, x)
    truth = np.asarray(dataset.evaluation_truth.delta, dtype=float)
    if truth.shape != baseline.shape or not np.isfinite(truth).all():
        raise ValueError("invalid evaluator-only contrast truth")
    scores = np.zeros((repeats, x.shape[1]))
    permutation_hash = hashlib.sha256()
    for repeat in range(repeats):
        for j in range(x.shape[1]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 4291, repeat, j]))
            permutation = rng.permutation(len(x))
            permutation_hash.update(permutation.astype("<i8").tobytes())
            changed = x.copy()
            changed[:, j] = x[permutation, j]
            scores[repeat, j] = np.mean(np.abs(treatment_contrast(predict_eta, changed) - baseline))
    importance = scores.mean(axis=0)
    error = baseline - truth
    squared_error = float(error @ error)
    return dict(primary=PRIMARY, scope="independent_synthetic_common_test",
                target="standard_gaussian_covariates",
                num_test=len(x), repeats=repeats,
                modifier_mask=dataset.evaluation_truth.modifier_mask.tolist(),
                importance_by_repeat=scores.tolist(), importance=importance.tolist(),
                recovery=signal_recovery(importance, dataset.evaluation_truth.modifier_mask),
                contrast_squared_error_sum=squared_error,
                contrast_rmse=float(np.sqrt(squared_error / len(x))),
                raw_x_sha256=_array_hash(x), contrast_truth_sha256=_array_hash(truth),
                estimated_contrast_sha256=_array_hash(baseline),
                permutation_sha256=permutation_hash.hexdigest(),
                evaluation_network_measured=False,
                end_to_end_one_round_claim=False)


def validate_evaluation(result, covariate_dim, num_test, repeats):
    """Recompute saved ranking metrics; no predictions or scientific reruns."""
    if (result["primary"] != PRIMARY or result["scope"] != "independent_synthetic_common_test"
            or result["target"] != "standard_gaussian_covariates"
            or result["num_test"] != num_test or result["repeats"] != repeats
            or result["evaluation_network_measured"] is not False
            or result["end_to_end_one_round_claim"] is not False):
        raise ValueError("survival evaluation identity mismatch")
    scores = np.asarray(result["importance_by_repeat"], dtype=float)
    importance = np.asarray(result["importance"], dtype=float)
    mask = np.asarray(result["modifier_mask"])
    if (scores.shape != (repeats, covariate_dim) or importance.shape != (covariate_dim,)
            or not np.isfinite(scores).all() or np.any(scores < 0)
            or not np.array_equal(importance, scores.mean(axis=0))
            or mask.dtype.kind != "b" or mask.shape != (covariate_dim,)
            or not np.array_equal(mask, np.arange(covariate_dim) < 2)):
        raise ValueError("invalid saved survival feature importance")
    if result["recovery"] != signal_recovery(importance, mask):
        raise ValueError("saved recovery does not match importance")
    squared_error = result["contrast_squared_error_sum"]
    if (not np.isfinite(squared_error) or squared_error < 0
            or result["contrast_rmse"] != float(np.sqrt(squared_error / num_test))):
        raise ValueError("invalid contrast RMSE")
    for field in ("raw_x_sha256", "contrast_truth_sha256", "estimated_contrast_sha256",
                  "permutation_sha256"):
        value = result[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid evaluation fingerprint")
