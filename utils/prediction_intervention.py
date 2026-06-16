import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from models import get_model


PREDICTION_DETAIL_FIELDS = [
    "seed",
    "model_source",
    "model",
    "split",
    "alpha",
    "synthetic_dim",
    "synthetic_signal_dim",
    "feature_idx",
    "is_signal",
    "intervention_mode",
    "metric",
    "importance",
    "rank",
]

PREDICTION_SUMMARY_FIELDS = [
    "seed",
    "model_source",
    "model",
    "split",
    "alpha",
    "synthetic_dim",
    "synthetic_signal_dim",
    "intervention_mode",
    "metric",
    "topk_hits",
    "topk_precision",
    "mean_signal_rank",
    "median_signal_rank",
    "mean_noise_rank",
    "auroc",
]

PREDICTION_MODEL_SUMMARY_FIELDS = [
    "seed",
    "model_source",
    "model",
    "split",
    "alpha",
    "synthetic_dim",
    "synthetic_signal_dim",
    "num_examples",
    "baseline_accuracy",
    "predicted_positive_rate",
    "mean_class1_probability",
    "mean_abs_logit_score",
]

SIGNED_RANK_METRICS = {"signed_logit_change", "signed_prob_change", "margin_drop"}


def collect_tensor_dataset(dataset, device=None, batch_size=8192):
    """Return full X/y tensors from a TensorDataset-like object or Subset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs = []
    ys = []
    for x_batch, y_batch in loader:
        xs.append(x_batch)
        ys.append(y_batch)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0).long()
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y


def predict_outputs(model, x, batch_size, device):
    """Predict model outputs without using true labels."""
    was_training = model.training
    model.eval()

    logits_list = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            x_batch = x[start:end].to(device)
            logits_list.append(model(x_batch).detach().cpu())

    if was_training:
        model.train()

    logits = torch.cat(logits_list, dim=0)
    probabilities = F.softmax(logits, dim=1)
    p1 = probabilities[:, 1]
    yhat = logits.argmax(dim=1).long()
    score = logits[:, 1] - logits[:, 0]
    return {
        "logits": logits,
        "yhat": yhat,
        "p1": p1,
        "score": score,
    }


def make_intervened_X(x, feature_idx, mode, generator=None):
    """Change one input coordinate while leaving all other coordinates fixed."""
    if mode not in {"zero", "permute"}:
        raise ValueError("mode must be 'zero' or 'permute'")

    x_changed = x.clone()
    if mode == "zero":
        x_changed[:, feature_idx] = 0.0
    else:
        permutation = torch.randperm(x.shape[0], generator=generator)
        x_changed[:, feature_idx] = x[permutation, feature_idx]
    return x_changed


def _model_level_summary(outputs, y):
    yhat = outputs["yhat"]
    p1 = outputs["p1"]
    score = outputs["score"]
    return {
        "num_examples": int(y.numel()),
        "baseline_accuracy": float(yhat.eq(y).float().mean().item()),
        "predicted_positive_rate": float(yhat.float().mean().item()),
        "mean_class1_probability": float(p1.mean().item()),
        "mean_abs_logit_score": float(score.abs().mean().item()),
    }


def prediction_intervention_importance(model, dataset, args, modes, repeats, seed):
    """Measure feature effects through prediction changes on held-out inputs.

    This is a model-based feature intervention diagnostic: after the model is
    trained, one input coordinate is modified at a time and the trained model's
    own predictions are compared with its baseline predictions. True labels are
    used only for reporting accuracy and signal-recovery evaluation.
    """
    device = args.get("device", next(model.parameters()).device)
    batch_size = args.get("prediction_intervention_batch_size", args.get("bs", 1024))
    x, y = collect_tensor_dataset(dataset, device=None)
    base_outputs = predict_outputs(model, x, batch_size=batch_size, device=device)

    dim = x.shape[1]
    c = base_outputs["yhat"].float()
    original_class_sign = (2.0 * c) - 1.0
    margin_base = original_class_sign * base_outputs["score"]

    results = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for mode in modes:
        repeat_count = 1 if mode == "zero" else max(1, int(repeats))
        mode_results = {
            "abs_logit_change": np.zeros(dim, dtype=np.float64),
            "signed_logit_change": np.zeros(dim, dtype=np.float64),
            "abs_prob_change": np.zeros(dim, dtype=np.float64),
            "signed_prob_change": np.zeros(dim, dtype=np.float64),
            "flip_rate": np.zeros(dim, dtype=np.float64),
            "margin_drop": np.zeros(dim, dtype=np.float64),
        }

        for feature_idx in range(dim):
            repeated = {metric: [] for metric in mode_results.keys()}
            for _ in range(repeat_count):
                x_changed = make_intervened_X(
                    x,
                    feature_idx,
                    mode,
                    generator=generator,
                )
                changed_outputs = predict_outputs(
                    model,
                    x_changed,
                    batch_size=batch_size,
                    device=device,
                )

                logit_delta = changed_outputs["score"] - base_outputs["score"]
                prob_delta = changed_outputs["p1"] - base_outputs["p1"]
                margin_changed = original_class_sign * changed_outputs["score"]

                repeated["abs_logit_change"].append(logit_delta.abs().mean().item())
                repeated["signed_logit_change"].append(logit_delta.mean().item())
                repeated["abs_prob_change"].append(prob_delta.abs().mean().item())
                repeated["signed_prob_change"].append(prob_delta.mean().item())
                repeated["flip_rate"].append(
                    changed_outputs["yhat"].ne(base_outputs["yhat"]).float().mean().item()
                )
                repeated["margin_drop"].append((margin_base - margin_changed).mean().item())

            for metric, values in repeated.items():
                mode_results[metric][feature_idx] = float(np.mean(values))

        results[mode] = mode_results

    return {
        "scores": results,
        "model_summary": _model_level_summary(base_outputs, y),
    }


def _ranking_values(importance, metric):
    scores = np.asarray(importance, dtype=np.float64)
    if metric in SIGNED_RANK_METRICS:
        scores = np.abs(scores)
    return scores


def rank_features(importance, metric):
    scores = _ranking_values(importance, metric)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def evaluate_signal_recovery(importance, signal_dim, metric):
    """Evaluate whether top-ranked dimensions recover the known signal set."""
    scores = _ranking_values(importance, metric)
    if signal_dim < 1 or signal_dim > len(scores):
        raise ValueError("signal_dim must be in [1, len(importance)]")

    signal_mask = np.zeros(len(scores), dtype=bool)
    signal_mask[:signal_dim] = True
    ranks = rank_features(importance, metric)
    topk = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")[:signal_dim]
    topk_hits = int(signal_mask[topk].sum())

    signal_ranks = ranks[signal_mask]
    noise_ranks = ranks[~signal_mask]
    signal_scores = scores[signal_mask]
    noise_scores = scores[~signal_mask]
    if len(noise_scores) == 0:
        auroc = float("nan")
    else:
        comparisons = signal_scores[:, None] - noise_scores[None, :]
        wins = (comparisons > 0).sum()
        ties = (comparisons == 0).sum()
        auroc = float((wins + 0.5 * ties) / comparisons.size)

    return {
        "topk_hits": topk_hits,
        "topk_precision": topk_hits / signal_dim,
        "mean_signal_rank": float(np.mean(signal_ranks)),
        "median_signal_rank": float(np.median(signal_ranks)),
        "mean_noise_rank": float(np.mean(noise_ranks)) if len(noise_ranks) else float("nan"),
        "auroc": auroc,
    }


def _add_metric_rows(detail_rows, summary_rows, metadata, intervention_mode, metric, importance):
    ranks = rank_features(importance, metric)
    signal_dim = metadata["synthetic_signal_dim"]
    metrics = evaluate_signal_recovery(importance, signal_dim, metric)
    for feature_idx, score in enumerate(importance):
        row = dict(metadata)
        row.update(
            {
                "feature_idx": feature_idx,
                "is_signal": int(feature_idx < signal_dim),
                "intervention_mode": intervention_mode,
                "metric": metric,
                "importance": float(score),
                "rank": int(ranks[feature_idx]),
            }
        )
        detail_rows.append(row)

    row = dict(metadata)
    row.update({"intervention_mode": intervention_mode, "metric": metric})
    row.update(metrics)
    summary_rows.append(row)


def add_model_prediction_intervention(
    detail_rows,
    summary_rows,
    model_summary_rows,
    model_source,
    model,
    dataset_test_global,
    args,
    metadata,
    modes,
    repeats,
    seed,
):
    print("Computing prediction intervention for", model_source)
    model_metadata = dict(metadata)
    model_metadata["model_source"] = model_source
    intervention = prediction_intervention_importance(
        model,
        dataset_test_global,
        args,
        modes=modes,
        repeats=repeats,
        seed=seed,
    )

    summary_row = dict(model_metadata)
    summary_row.update(intervention["model_summary"])
    model_summary_rows.append(summary_row)

    for mode, mode_scores in intervention["scores"].items():
        for metric, importance in mode_scores.items():
            _add_metric_rows(
                detail_rows,
                summary_rows,
                model_metadata,
                mode,
                metric,
                importance,
            )


def write_prediction_intervention_outputs(
    detail_rows,
    summary_rows,
    model_summary_rows,
    detail_path,
    summary_path,
    model_summary_path,
):
    if detail_rows:
        with open(detail_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_DETAIL_FIELDS)
            writer.writeheader()
            writer.writerows(detail_rows)
    if summary_rows:
        with open(summary_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)
    if model_summary_rows:
        with open(model_summary_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_MODEL_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(model_summary_rows)


def train_pooled_prediction_model(args, dataset_train_global):
    model = get_model(
        args["model"],
        args["n_c"],
        bias=False,
        synthetic_dim=args["synthetic_dim"],
    ).to(args["device"])
    loader = DataLoader(dataset_train_global, batch_size=args["bs"], shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args["eta"], momentum=0.9)

    model.train()
    for epoch in range(args["local_epochs"]):
        batch_loss = []
        for data, target in loader:
            data, target = data.to(args["device"]), target.to(args["device"])
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        print("Pooled Prediction Epoch No. ", epoch, "Loss ", sum(batch_loss) / len(batch_loss))

    return model
