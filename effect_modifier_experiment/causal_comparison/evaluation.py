"""Identical raw-X interventions on the probability scale for every method."""

import hashlib
import math
import numpy as np
import torch

from data import build_effect_modifier_site_evaluation_datasets
from utils.prediction_intervention import predict_treatment_contrast, evaluate_signal_recovery
from .protocol import VARIANTS, SPLITS


def signal_recovery_with_ties(importance, signal_dim, metric):
    """Primary Top-k recovery integrates uniform ties at the selection boundary.

    The legacy stable index tie break is preserved only as a named diagnostic;
    it cannot reward X1/X2 simply for appearing first in an all-zero ranking.
    Exact equality of observed importance values defines a tie.
    """
    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("recovery requires finite nonnegative feature importances")
    if type(signal_dim) is not int or not 1 <= signal_dim <= values.size:
        raise ValueError("invalid recovery signal count")
    diagnostic = evaluate_signal_recovery(
        values, signal_dim, metric, signal_indices=list(range(signal_dim)),
        candidate_indices=list(range(values.size)))
    diagnostic = {key: value if math.isfinite(value) else None for key, value in diagnostic.items()}
    cutoff = np.sort(values)[-signal_dim]
    above = values > cutoff
    tied = values == cutoff
    slots = signal_dim - int(above.sum())
    tied_count = int(tied.sum())
    true_above = int(above[:signal_dim].sum())
    true_tied = int(tied[:signal_dim].sum())
    expected_hits = true_above + slots * true_tied / tied_count
    if true_above + true_tied != signal_dim or true_tied > slots:
        all_selected = 0.0
    else:
        all_selected = math.comb(tied_count - true_tied, slots - true_tied) / math.comb(tied_count, slots)
    return {
        "expected_topk_hits": expected_hits,
        "expected_topk_precision": expected_hits / signal_dim,
        "probability_all_signals_selected": all_selected,
        "auroc": diagnostic["auroc"],
        "boundary_tie_size": tied_count,
        "boundary_slots": slots,
        "tie_rule": "uniform_without_replacement_within_exact_score_boundary_ties",
        "deterministic_index_tiebreak_diagnostic": diagnostic,
    }


def array_hash(values):
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode())
    h.update(str(values.shape).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def model_arrays(model):
    if hasattr(model, "predict_tau"):
        return {"effect_coefficients": np.asarray(model.coefficients, dtype=np.float64)}
    return {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}


def model_hash(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode() + b"\0" + array_hash(arrays[key]).encode())
    return h.hexdigest()


def contrasts(model, x, batch_size):
    x = np.asarray(x)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("treatment contrasts require nonempty finite raw-X rows")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("evaluation batch size must be a positive integer")
    if hasattr(model, "predict_tau"):
        result = {"tau_prob": np.asarray(model.predict_tau(x), dtype=np.float64)}
    else:
        result = {key: value.detach().cpu().numpy().astype(np.float64)
                  for key, value in predict_treatment_contrast(
                      model, torch.as_tensor(x, dtype=torch.float32), batch_size, "cpu").items()}
    for values in result.values():
        if values.shape != (len(x),) or not np.isfinite(values).all():
            raise ValueError("invalid treatment contrasts")
    return result


def evaluate(model, site_datasets, test_dataset, *, variant, split, seed,
             dgp, config):
    d = dgp["covariate_dim"]
    if (variant not in VARIANTS or split not in SPLITS or type(seed) is not int or seed < 0
            or type(d) is not int or d < 1 or type(dgp["signal_dim"]) is not int
            or not 1 <= dgp["signal_dim"] <= d):
        raise ValueError("invalid evaluation variant, split, seed or raw-X dimensions")
    if (type(config["repeats"]) is not int or config["repeats"] < 1
            or type(config["permutation_seed_offset"]) is not int or config["permutation_seed_offset"] < 0
            or not isinstance(config["modes"], (list, tuple))
            or any(mode not in ("permute", "zero") for mode in config["modes"])
            or "permute" not in config["modes"] or len(set(config["modes"])) != len(config["modes"])):
        raise ValueError("invalid evaluation modes, repeats or permutation seed")
    tensors = getattr(test_dataset, "tensors", ())
    if (len(tensors) != 2 or tensors[0].ndim != 2 or tensors[0].shape[1] != 1 + 2 * d
            or not len(tensors[0]) or tensors[1].shape != (len(tensors[0]),)
            or not torch.isfinite(tensors[0]).all().item()):
        raise ValueError("common test set must contain finite [A,X,A*X] rows and labels")
    x = test_dataset.tensors[0][:, 1:1 + d].detach().cpu().numpy().astype(np.float64)
    if "local_permutation" in variant:
        shards = build_effect_modifier_site_evaluation_datasets(
            test_dataset, site_datasets, split, dgp["alpha"], seed)
        indices = [np.asarray(shard.indices, dtype=np.int64) for shard in shards]
    else:
        indices = [np.arange(len(x), dtype=np.int64)]
    joined = np.concatenate(indices)
    if len(joined) != len(x) or not np.array_equal(np.sort(joined), np.arange(len(x))):
        raise ValueError("evaluation shards must partition the common test set")
    baseline = contrasts(model, x, config["batch_size"])
    scores = {mode: {scale: np.zeros(d) for scale in baseline} for mode in config["modes"]}
    permutation_hash = hashlib.sha256()
    for shard_idx, ids in enumerate(indices):
        if not len(ids):
            continue
        local_x = x[ids]
        for j in range(d):
            for mode in config["modes"]:
                repeats = config["repeats"] if mode == "permute" else 1
                for repeat in range(repeats):
                    changed = local_x.copy()
                    if mode == "permute":
                        rng = np.random.default_rng(np.random.SeedSequence([
                            config["permutation_seed_offset"], seed, shard_idx, j, repeat]))
                        permutation = rng.permutation(len(ids))
                        changed[:, j] = changed[permutation, j]
                        permutation_hash.update(permutation.astype("<i8").tobytes())
                    else:
                        changed[:, j] = 0.0
                    predictions = contrasts(model, changed, config["batch_size"])
                    for scale in scores[mode]:
                        scores[mode][scale][j] += float(np.abs(
                            predictions[scale] - baseline[scale][ids]).sum()) / len(x) / repeats
    metrics = {}
    for mode, scale_scores in scores.items():
        for scale, values in scale_scores.items():
            metrics[mode + "/" + scale] = {
                "importance": values.tolist(),
                "recovery": signal_recovery_with_ties(
                    values, dgp["signal_dim"], "tau_intervention_" + scale.removeprefix("tau_"))}
    return dict(primary="permute/tau_prob", metrics=metrics,
                raw_x_sha256=array_hash(x), permutation_sha256=permutation_hash.hexdigest(),
                shard_indices_sha256=[array_hash(ids) for ids in indices],
                shard_sizes=[len(ids) for ids in indices],
                evaluation_network_measured=False,
                effect_only=hasattr(model, "predict_tau"))
