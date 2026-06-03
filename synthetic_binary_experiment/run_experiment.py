#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from synthetic_fedfisher.data import SyntheticDataConfig, make_synthetic_datasets, theoretical_bayes_accuracy
from synthetic_fedfisher.federated import (
    client_weights,
    fedavg_vector,
    fedfisher_diag_adam_vector,
    fedfisher_full_vector,
    fedfisher_kfac_adam_vector,
    kfac_stats_scalar_count,
    materialize_model,
    train_multiround_fedavg,
)
from synthetic_fedfisher.models import build_model, parse_hidden_dims
from synthetic_fedfisher.training import (
    TrainConfig,
    build_optimizer,
    clone_state_dict,
    empirical_fisher_diag,
    empirical_fisher_full,
    empirical_kfac_stats,
    evaluate,
    parameter_count,
    resolve_device,
    set_global_seed,
    train_from_state,
)


FIELDNAMES = [
    "seed",
    "model_type",
    "split",
    "method",
    "accuracy",
    "loss",
    "gap_to_pool",
    "gain_over_fedavg",
    "rounds",
    "round_epochs",
    "uplink_scalars",
    "param_count",
    "num_train",
    "num_test",
    "dim",
    "signal_dim",
    "signal_strength",
    "noise_std",
    "bayes_accuracy",
    "num_clients",
    "dirichlet_alpha",
    "local_epochs",
    "pool_epochs",
    "pool_selected_epochs",
    "pool_batch_size",
    "pool_lr",
    "pool_weight_decay",
    "pool_optimizer",
    "pool_val_fraction",
    "pool_patience",
    "pool_min_delta",
    "batch_size",
    "lr",
    "weight_decay",
    "optimizer",
    "fisher_damping",
    "fisher_server_steps",
    "fisher_server_lr",
    "fisher_server_eval_every",
    "fisher_val_size",
    "fisher_selected_step",
    "fisher_val_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic FedFisher binary experiments.")
    parser.add_argument("--output-dir", default="synthetic_binary_experiment/outputs/default")
    parser.add_argument("--num-train", type=int, default=10_000)
    parser.add_argument("--num-test", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--signal-dim", type=int, default=10)
    parser.add_argument("--signal-strength", type=float, default=0.7)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--model-types", nargs="+", choices=["lr", "mlp"], default=["lr", "mlp"])
    parser.add_argument("--splits", nargs="+", choices=["iid", "mild", "noniid"], default=["iid", "noniid"])
    parser.add_argument("--hidden-dims", default="64,32")
    parser.add_argument("--local-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--pool-epochs", type=int, default=80)
    parser.add_argument("--pool-batch-size", type=int, default=None)
    parser.add_argument("--pool-lr", type=float, default=1e-3)
    parser.add_argument("--pool-weight-decay", type=float, default=1e-2)
    parser.add_argument("--pool-optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--pool-val-fraction", type=float, default=0.2)
    parser.add_argument("--pool-patience", type=int, default=8)
    parser.add_argument("--pool-min-delta", type=float, default=1e-4)
    parser.add_argument("--fisher-damping", type=float, default=1e-6)
    parser.add_argument("--fisher-batch-size", type=int, default=128)
    parser.add_argument("--fisher-server-steps", type=int, default=2000)
    parser.add_argument("--fisher-server-lr", type=float, default=1e-2)
    parser.add_argument("--fisher-server-eval-every", type=int, default=100)
    parser.add_argument("--fisher-val-size", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-multiround", action="store_true")
    parser.add_argument("--fedavg-rounds", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--fedavg-round-epochs", type=int, default=None)
    return parser.parse_args()


def validate_pool_selection_args(val_fraction: float, patience: int, min_delta: float) -> None:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("pool_val_fraction must be in [0, 1)")
    if patience < 0:
        raise ValueError("pool_patience must be non-negative")
    if min_delta < 0:
        raise ValueError("pool_min_delta must be non-negative")


def validate_fisher_server_args(steps: int, lr: float, eval_every: int, val_size: int) -> None:
    if steps <= 0:
        raise ValueError("fisher_server_steps must be positive")
    if lr <= 0:
        raise ValueError("fisher_server_lr must be positive")
    if eval_every <= 0:
        raise ValueError("fisher_server_eval_every must be positive")
    if val_size < 0:
        raise ValueError("fisher_val_size must be non-negative")


def split_for_validation(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Dataset | None]:
    if val_fraction <= 0.0:
        return dataset, None
    val_size = int(round(len(dataset) * val_fraction))
    val_size = min(max(val_size, 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def make_validation_subset(dataset: Dataset, max_size: int, seed: int) -> Dataset | None:
    if max_size <= 0:
        return None
    size = min(max_size, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def select_best_pool_epoch(
    model_factory: Callable[[], nn.Module],
    initial_state: Dict[str, torch.Tensor],
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    config: TrainConfig,
    seed: int,
    patience: int,
    min_delta: float,
) -> Tuple[int, float]:
    if val_dataset is None:
        return config.epochs, float("nan")

    config.validate()
    model = model_factory()
    model.load_state_dict({key: value.detach().clone() for key, value in initial_state.items()})
    device = resolve_device(config.device)
    model.to(device)
    model.train()

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    optimizer = build_optimizer(model, config)

    best_epoch = 1
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_dataset, batch_size=config.batch_size, device=config.device)
        val_loss = val_metrics["loss"]
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                break
        model.train()

    return best_epoch, best_loss


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def make_row(
    *,
    seed: int,
    model_type: str,
    split: str,
    method: str,
    metrics: Dict[str, float],
    pool_acc: float,
    fedavg_acc: float,
    rounds: int,
    round_epochs: int,
    uplink_scalars: int,
    param_count_value: int,
    data_config: SyntheticDataConfig,
    train_config: TrainConfig,
    pool_train_config: TrainConfig,
    pool_selected_epochs: int,
    pool_val_fraction: float,
    pool_patience: int,
    pool_min_delta: float,
    fisher_damping: float,
    fisher_server_steps: int,
    fisher_server_lr: float,
    fisher_server_eval_every: int,
    fisher_val_size: int,
    fisher_selected_step: int,
    fisher_val_score: float,
) -> Dict[str, object]:
    return {
        "seed": seed,
        "model_type": model_type,
        "split": split,
        "method": method,
        "accuracy": f"{metrics['accuracy']:.8f}",
        "loss": f"{metrics['loss']:.8f}",
        "gap_to_pool": f"{metrics['accuracy'] - pool_acc:.8f}",
        "gain_over_fedavg": f"{metrics['accuracy'] - fedavg_acc:.8f}",
        "rounds": rounds,
        "round_epochs": round_epochs,
        "uplink_scalars": uplink_scalars,
        "param_count": param_count_value,
        "num_train": data_config.num_train,
        "num_test": data_config.num_test,
        "dim": data_config.dim,
        "signal_dim": data_config.signal_dim,
        "signal_strength": data_config.signal_strength,
        "noise_std": data_config.noise_std,
        "bayes_accuracy": f"{theoretical_bayes_accuracy(data_config):.8f}",
        "num_clients": data_config.num_clients,
        "dirichlet_alpha": data_config.dirichlet_alpha,
        "local_epochs": train_config.epochs,
        "pool_epochs": pool_train_config.epochs,
        "pool_selected_epochs": pool_selected_epochs,
        "pool_batch_size": pool_train_config.batch_size,
        "pool_lr": pool_train_config.lr,
        "pool_weight_decay": pool_train_config.weight_decay,
        "pool_optimizer": pool_train_config.optimizer,
        "pool_val_fraction": pool_val_fraction,
        "pool_patience": pool_patience,
        "pool_min_delta": pool_min_delta,
        "batch_size": train_config.batch_size,
        "lr": train_config.lr,
        "weight_decay": train_config.weight_decay,
        "optimizer": train_config.optimizer,
        "fisher_damping": fisher_damping,
        "fisher_server_steps": fisher_server_steps,
        "fisher_server_lr": fisher_server_lr,
        "fisher_server_eval_every": fisher_server_eval_every,
        "fisher_val_size": fisher_val_size,
        "fisher_selected_step": fisher_selected_step,
        "fisher_val_score": "" if fisher_val_score != fisher_val_score else f"{fisher_val_score:.8f}",
    }


def run_one_setting(
    *,
    seed: int,
    model_type: str,
    split: str,
    hidden_dims: Iterable[int],
    data_config: SyntheticDataConfig,
    train_config: TrainConfig,
    pool_train_config: TrainConfig,
    pool_val_fraction: float,
    pool_patience: int,
    pool_min_delta: float,
    fisher_batch_size: int,
    fisher_damping: float,
    fisher_server_steps: int,
    fisher_server_lr: float,
    fisher_server_eval_every: int,
    fisher_val_size: int,
    include_multiround: bool,
    fedavg_rounds: Iterable[int],
    fedavg_round_epochs: int,
) -> List[Dict[str, object]]:
    set_global_seed(seed)
    client_datasets, pooled_dataset, test_dataset, metadata = make_synthetic_datasets(
        data_config,
        split=split,
        seed=seed,
    )

    def model_factory() -> torch.nn.Module:
        return build_model(model_type, input_dim=data_config.dim, hidden_dims=tuple(hidden_dims))

    initial_model = model_factory()
    initial_state = clone_state_dict(initial_model)
    p_count = parameter_count(initial_model)
    weights = client_weights(client_datasets)

    local_models = []
    fisher_diags = []
    fisher_full_matrices = []
    kfac_stats = []
    for client_idx, client_dataset in enumerate(client_datasets):
        local_model = train_from_state(
            model_factory(),
            initial_state,
            client_dataset,
            train_config,
            seed=seed * 10_000 + client_idx,
        )
        fisher_diag = empirical_fisher_diag(
            local_model,
            client_dataset,
            batch_size=fisher_batch_size,
            device=train_config.device,
        )
        if model_type == "lr":
            fisher_full_matrices.append(
                empirical_fisher_full(
                    local_model,
                    client_dataset,
                    batch_size=fisher_batch_size,
                    device=train_config.device,
                )
            )
        if model_type == "mlp":
            kfac_stats.append(
                empirical_kfac_stats(
                    local_model,
                    client_dataset,
                    batch_size=fisher_batch_size,
                    device=train_config.device,
                )
            )
        local_models.append(local_model.cpu())
        fisher_diags.append(fisher_diag)

    pool_train_subset, pool_val_subset = split_for_validation(
        pooled_dataset,
        val_fraction=pool_val_fraction,
        seed=seed * 10_000 + 901,
    )
    pool_selected_epochs, pool_best_val_loss = select_best_pool_epoch(
        model_factory,
        initial_state,
        pool_train_subset,
        pool_val_subset,
        pool_train_config,
        seed=seed * 10_000 + 902,
        patience=pool_patience,
        min_delta=pool_min_delta,
    )
    pooled_refit_config = replace(pool_train_config, epochs=pool_selected_epochs)
    pooled_model = train_from_state(
        model_factory(),
        initial_state,
        pooled_dataset,
        pooled_refit_config,
        seed=seed * 10_000 + 900,
    )
    pool_metrics = evaluate(pooled_model, test_dataset, batch_size=train_config.batch_size, device=train_config.device)

    fedavg = materialize_model(model_factory, fedavg_vector(local_models, weights))
    fedavg_metrics = evaluate(fedavg, test_dataset, batch_size=train_config.batch_size, device=train_config.device)

    fisher_val_dataset = make_validation_subset(
        pooled_dataset,
        max_size=fisher_val_size,
        seed=seed * 10_000 + 903,
    )

    def validation_score(vector: torch.Tensor) -> float:
        if fisher_val_dataset is None:
            return float("nan")
        model = materialize_model(model_factory, vector)
        metrics = evaluate(model, fisher_val_dataset, batch_size=train_config.batch_size, device=train_config.device)
        return metrics["accuracy"]

    fedfisher_vector, fedfisher_step, fedfisher_val_score = fedfisher_diag_adam_vector(
        local_models,
        fisher_diags,
        weights,
        server_lr=fisher_server_lr,
        steps=fisher_server_steps,
        eval_every=fisher_server_eval_every,
        validation_score_fn=validation_score if fisher_val_dataset is not None else None,
        damping=fisher_damping,
    )
    fedfisher = materialize_model(
        model_factory,
        fedfisher_vector,
    )
    fedfisher_metrics = evaluate(fedfisher, test_dataset, batch_size=train_config.batch_size, device=train_config.device)

    pool_acc = pool_metrics["accuracy"]
    fedavg_acc = fedavg_metrics["accuracy"]
    full_acc = None
    kfac_acc = None
    rows = [
        make_row(
            seed=seed,
            model_type=model_type,
            split=split,
            method="pool",
            metrics=pool_metrics,
            pool_acc=pool_acc,
            fedavg_acc=fedavg_acc,
            rounds=0,
            round_epochs=pool_selected_epochs,
            uplink_scalars=0,
            param_count_value=p_count,
            data_config=data_config,
            train_config=train_config,
            pool_train_config=pool_train_config,
            pool_selected_epochs=pool_selected_epochs,
            pool_val_fraction=pool_val_fraction,
            pool_patience=pool_patience,
            pool_min_delta=pool_min_delta,
            fisher_damping=fisher_damping,
            fisher_server_steps=fisher_server_steps,
            fisher_server_lr=fisher_server_lr,
            fisher_server_eval_every=fisher_server_eval_every,
            fisher_val_size=fisher_val_size,
            fisher_selected_step=0,
            fisher_val_score=float("nan"),
        ),
        make_row(
            seed=seed,
            model_type=model_type,
            split=split,
            method="fedavg_oneshot",
            metrics=fedavg_metrics,
            pool_acc=pool_acc,
            fedavg_acc=fedavg_acc,
            rounds=1,
            round_epochs=train_config.epochs,
            uplink_scalars=data_config.num_clients * p_count,
            param_count_value=p_count,
            data_config=data_config,
            train_config=train_config,
            pool_train_config=pool_train_config,
            pool_selected_epochs=pool_selected_epochs,
            pool_val_fraction=pool_val_fraction,
            pool_patience=pool_patience,
            pool_min_delta=pool_min_delta,
            fisher_damping=fisher_damping,
            fisher_server_steps=fisher_server_steps,
            fisher_server_lr=fisher_server_lr,
            fisher_server_eval_every=fisher_server_eval_every,
            fisher_val_size=fisher_val_size,
            fisher_selected_step=0,
            fisher_val_score=float("nan"),
        ),
        make_row(
            seed=seed,
            model_type=model_type,
            split=split,
            method="fedfisher_diag",
            metrics=fedfisher_metrics,
            pool_acc=pool_acc,
            fedavg_acc=fedavg_acc,
            rounds=1,
            round_epochs=train_config.epochs,
            uplink_scalars=data_config.num_clients * 2 * p_count,
            param_count_value=p_count,
            data_config=data_config,
            train_config=train_config,
            pool_train_config=pool_train_config,
            pool_selected_epochs=pool_selected_epochs,
            pool_val_fraction=pool_val_fraction,
            pool_patience=pool_patience,
            pool_min_delta=pool_min_delta,
            fisher_damping=fisher_damping,
            fisher_server_steps=fisher_server_steps,
            fisher_server_lr=fisher_server_lr,
            fisher_server_eval_every=fisher_server_eval_every,
            fisher_val_size=fisher_val_size,
            fisher_selected_step=fedfisher_step,
            fisher_val_score=fedfisher_val_score,
        ),
    ]

    if model_type == "lr":
        fedfisher_full = materialize_model(
            model_factory,
            fedfisher_full_vector(local_models, fisher_full_matrices, weights, damping=fisher_damping),
        )
        full_metrics = evaluate(
            fedfisher_full,
            test_dataset,
            batch_size=train_config.batch_size,
            device=train_config.device,
        )
        full_acc = full_metrics["accuracy"]
        rows.append(
            make_row(
                seed=seed,
                model_type=model_type,
                split=split,
                method="fedfisher_full",
                metrics=full_metrics,
                pool_acc=pool_acc,
                fedavg_acc=fedavg_acc,
                rounds=1,
                round_epochs=train_config.epochs,
                uplink_scalars=data_config.num_clients * (p_count + p_count * p_count),
                param_count_value=p_count,
                data_config=data_config,
                train_config=train_config,
                pool_train_config=pool_train_config,
                pool_selected_epochs=pool_selected_epochs,
                pool_val_fraction=pool_val_fraction,
                pool_patience=pool_patience,
                pool_min_delta=pool_min_delta,
                fisher_damping=fisher_damping,
                fisher_server_steps=fisher_server_steps,
                fisher_server_lr=fisher_server_lr,
                fisher_server_eval_every=fisher_server_eval_every,
                fisher_val_size=fisher_val_size,
                fisher_selected_step=0,
                fisher_val_score=float("nan"),
            )
        )

    if model_type == "mlp":
        fedfisher_kfac_vector_value, kfac_step, kfac_val_score = fedfisher_kfac_adam_vector(
            local_models,
            kfac_stats,
            weights,
            server_lr=fisher_server_lr,
            steps=fisher_server_steps,
            eval_every=fisher_server_eval_every,
            validation_score_fn=validation_score if fisher_val_dataset is not None else None,
            damping=fisher_damping,
        )
        fedfisher_kfac = materialize_model(
            model_factory,
            fedfisher_kfac_vector_value,
        )
        kfac_metrics = evaluate(
            fedfisher_kfac,
            test_dataset,
            batch_size=train_config.batch_size,
            device=train_config.device,
        )
        kfac_acc = kfac_metrics["accuracy"]
        rows.append(
            make_row(
                seed=seed,
                model_type=model_type,
                split=split,
                method="fedfisher_kfac",
                metrics=kfac_metrics,
                pool_acc=pool_acc,
                fedavg_acc=fedavg_acc,
                rounds=1,
                round_epochs=train_config.epochs,
                uplink_scalars=data_config.num_clients * (p_count + kfac_stats_scalar_count(kfac_stats[0])),
                param_count_value=p_count,
                data_config=data_config,
                train_config=train_config,
                pool_train_config=pool_train_config,
                pool_selected_epochs=pool_selected_epochs,
                pool_val_fraction=pool_val_fraction,
                pool_patience=pool_patience,
                pool_min_delta=pool_min_delta,
                fisher_damping=fisher_damping,
                fisher_server_steps=fisher_server_steps,
                fisher_server_lr=fisher_server_lr,
                fisher_server_eval_every=fisher_server_eval_every,
                fisher_val_size=fisher_val_size,
                fisher_selected_step=kfac_step,
                fisher_val_score=kfac_val_score,
            )
        )

    if include_multiround:
        for rounds in fedavg_rounds:
            model = train_multiround_fedavg(
                model_factory,
                initial_state,
                client_datasets,
                train_config,
                rounds=rounds,
                round_epochs=fedavg_round_epochs,
                seed=seed,
            )
            metrics = evaluate(model, test_dataset, batch_size=train_config.batch_size, device=train_config.device)
            rows.append(
                make_row(
                    seed=seed,
                    model_type=model_type,
                    split=split,
                    method=f"fedavg_round_{rounds}",
                    metrics=metrics,
                    pool_acc=pool_acc,
                    fedavg_acc=fedavg_acc,
                    rounds=rounds,
                    round_epochs=fedavg_round_epochs,
                    uplink_scalars=data_config.num_clients * p_count * rounds,
                    param_count_value=p_count,
                    data_config=data_config,
                    train_config=train_config,
                    pool_train_config=pool_train_config,
                    pool_selected_epochs=pool_selected_epochs,
                    pool_val_fraction=pool_val_fraction,
                    pool_patience=pool_patience,
                    pool_min_delta=pool_min_delta,
                    fisher_damping=fisher_damping,
                    fisher_server_steps=fisher_server_steps,
                    fisher_server_lr=fisher_server_lr,
                    fisher_server_eval_every=fisher_server_eval_every,
                    fisher_val_size=fisher_val_size,
                    fisher_selected_step=0,
                    fisher_val_score=float("nan"),
                )
            )

    log_payload = {
        "seed": seed,
        "model_type": model_type,
        "split": split,
        "split_strategy": metadata["split_strategy"],
        "bayes_acc": round(float(metadata["bayes_accuracy"]), 4),
        "client_priors": metadata["client_priors"],
        "client_sizes": metadata["client_sizes"],
        "pool_acc": round(pool_metrics["accuracy"], 4),
        "fedavg_acc": round(fedavg_metrics["accuracy"], 4),
        "fedfisher_diag_acc": round(fedfisher_metrics["accuracy"], 4),
        "fedfisher_diag_step": fedfisher_step,
        "fedfisher_diag_val": round(fedfisher_val_score, 4)
        if fedfisher_val_score == fedfisher_val_score
        else None,
        "pool_selected_epochs": pool_selected_epochs,
    }
    if pool_val_subset is not None:
        log_payload["pool_best_val_loss"] = round(pool_best_val_loss, 6)
    if full_acc is not None:
        log_payload["fedfisher_full_acc"] = round(full_acc, 4)
    if kfac_acc is not None:
        log_payload["fedfisher_kfac_acc"] = round(kfac_acc, 4)
        log_payload["fedfisher_kfac_step"] = kfac_step
        log_payload["fedfisher_kfac_val"] = round(kfac_val_score, 4) if kfac_val_score == kfac_val_score else None
    print(json.dumps(log_payload, sort_keys=True))
    return rows


def main() -> None:
    args = parse_args()
    data_config = SyntheticDataConfig(
        num_train=args.num_train,
        num_test=args.num_test,
        dim=args.dim,
        signal_dim=args.signal_dim,
        signal_strength=args.signal_strength,
        noise_std=args.noise_std,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    train_config = TrainConfig(
        epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        device=args.device,
    )
    pool_train_config = TrainConfig(
        epochs=args.pool_epochs,
        batch_size=args.pool_batch_size or args.batch_size,
        lr=args.pool_lr if args.pool_lr is not None else args.lr,
        weight_decay=args.pool_weight_decay if args.pool_weight_decay is not None else args.weight_decay,
        optimizer=args.pool_optimizer,
        device=args.device,
    )
    data_config.validate()
    train_config.validate()
    pool_train_config.validate()
    validate_pool_selection_args(args.pool_val_fraction, args.pool_patience, args.pool_min_delta)
    validate_fisher_server_args(
        args.fisher_server_steps,
        args.fisher_server_lr,
        args.fisher_server_eval_every,
        args.fisher_val_size,
    )
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    fedavg_round_epochs = args.fedavg_round_epochs or args.local_epochs

    all_rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        for split in args.splits:
            for model_type in args.model_types:
                all_rows.extend(
                    run_one_setting(
                        seed=seed,
                        model_type=model_type,
                        split=split,
                        hidden_dims=hidden_dims,
                        data_config=data_config,
                        train_config=train_config,
                        pool_train_config=pool_train_config,
                        pool_val_fraction=args.pool_val_fraction,
                        pool_patience=args.pool_patience,
                        pool_min_delta=args.pool_min_delta,
                        fisher_batch_size=args.fisher_batch_size,
                        fisher_damping=args.fisher_damping,
                        fisher_server_steps=args.fisher_server_steps,
                        fisher_server_lr=args.fisher_server_lr,
                        fisher_server_eval_every=args.fisher_server_eval_every,
                        fisher_val_size=args.fisher_val_size,
                        include_multiround=args.include_multiround,
                        fedavg_rounds=args.fedavg_rounds,
                        fedavg_round_epochs=fedavg_round_epochs,
                    )
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "results.csv", all_rows)
    print(f"Wrote {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
