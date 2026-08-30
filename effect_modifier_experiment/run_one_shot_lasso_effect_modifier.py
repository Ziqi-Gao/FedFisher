#!/usr/bin/env python3
"""Run one isolated one-shot LASSO effect-modifier result cell on CPU."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_effect_modifier_site_evaluation_datasets, get_dataset
from effect_modifier_experiment.dslassocov_benchmark import dataset_fingerprint, json_safe
from effect_modifier_experiment.one_shot_lasso_benchmark import (
    FAMILIES,
    VARIANT_CONTRACT,
    LocalLogisticConfig,
    MlpGroupLassoConfig,
    DEFAULT_MOMENTUM_RESET_POLISH_MAX_ITER,
    MOMENTUM_RESET_POLISH_RULE,
    SMOOTHING_RULE,
    OneShotLinearLogisticModel,
    aggregate_linear_payloads,
    aggregate_mlp_payloads,
    evaluation_communication_report,
    fit_local_logistic_lasso_site,
    fit_mlp_group_lasso_site,
    make_common_mlp_state,
    payload_fingerprint,
    tensor_datasets_to_sealed_sites,
    training_fingerprint,
)
from effect_modifier_experiment.one_shot_lasso_run_contract import (
    METHOD_NAMES,
    atomic_csv_text,
    atomic_json,
    complete_cell,
    env_text,
    initialize_or_validate_run,
    sha256_file,
)
from effect_modifier_experiment.dslassocov_run_contract import atomic_create_text
from utils.prediction_intervention import (
    PREDICTION_DETAIL_FIELDS,
    PREDICTION_MODEL_SUMMARY_FIELDS,
    PREDICTION_SITE_SUMMARY_FIELDS,
    PREDICTION_SUMMARY_FIELDS,
    add_model_prediction_intervention,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANT_CONTRACT))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-end", type=int, default=None)
    parser.add_argument("--split", choices=("iid", "noniid"), required=True)
    parser.add_argument(
        "--model",
        choices=("linear", "SyntheticMLP", "SyntheticMLPDeep"),
        default="linear",
    )
    parser.add_argument("--num-sites", type=int, default=10)
    parser.add_argument("--site-min-samples", type=int, default=500)
    parser.add_argument("--site-max-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--covariate-dim", type=int, default=100)
    parser.add_argument("--signal-dim", type=int, default=2)
    parser.add_argument("--signal-strength", type=float, default=2.0)
    parser.add_argument("--main-effect-strength", type=float, default=1.0)
    parser.add_argument("--confounding-strength", type=float, default=1.0)
    parser.add_argument("--outcome-intercept", type=float, default=0.0)
    parser.add_argument("--treatment-prob", type=float, default=0.5)
    parser.add_argument("--propensity-clip", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gaussian-mean-span", type=float, default=0.75)
    parser.add_argument("--gaussian-std-low", type=float, default=0.75)
    parser.add_argument("--gaussian-std-high", type=float, default=1.25)
    parser.add_argument("--intervention-repeats", type=int, default=5)
    parser.add_argument("--intervention-batch-size", type=int, default=1024)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--lambda-count", type=int, default=30)
    parser.add_argument("--lambda-min-ratio", type=float, default=1e-3)
    parser.add_argument("--cv-seed-offset", type=int, default=4000037)
    parser.add_argument("--optimizer-max-iter", type=int, default=2000)
    parser.add_argument(
        "--optimizer-polish-max-iter",
        type=int,
        default=DEFAULT_MOMENTUM_RESET_POLISH_MAX_ITER,
    )
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-6)
    parser.add_argument("--optimizer-initial-step", type=float, default=1.0)
    parser.add_argument("--optimizer-backtrack-factor", type=float, default=0.5)
    parser.add_argument("--optimizer-min-step", type=float, default=1e-12)
    parser.add_argument("--standardization-scale-floor", type=float, default=1e-12)
    parser.add_argument("--local-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--lambda-ratios",
        type=str,
        default="0,1,3,10,30,100,300",
        help="comma-separated site-local group-gradient scale multipliers",
    )
    parser.add_argument("--split-seed-offset", type=int, default=5100037)
    parser.add_argument("--loader-seed-offset", type=int, default=5200037)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.seed_start is None:
        args.seed_start = args.seed
    if args.seed_end is None:
        args.seed_end = args.seed
    args.lambda_ratios = tuple(float(value) for value in args.lambda_ratios.split(","))
    validate_args(args)
    return args


def validate_args(args):
    if args.seed < 0 or args.num_sites < 1:
        raise ValueError("seed must be nonnegative and num_sites positive")
    if args.seed_start < 0 or args.seed_end < args.seed_start:
        raise ValueError("invalid manifest seed range")
    if not args.seed_start <= args.seed <= args.seed_end:
        raise ValueError("seed must lie within the manifest seed range")
    if args.threads < 1 or args.threads > 4:
        raise ValueError("threads must be in [1, 4]")
    if args.site_min_samples < 2 or args.site_max_samples < args.site_min_samples:
        raise ValueError("invalid site sample bounds")
    if args.test_size < args.num_sites or args.covariate_dim < 2:
        raise ValueError("test size or covariate dimension is too small")
    if args.signal_dim != 2 or args.signal_dim > args.covariate_dim:
        raise ValueError("this benchmark requires exactly two signals")
    finite = (
        args.signal_strength,
        args.main_effect_strength,
        args.confounding_strength,
        args.outcome_intercept,
        args.treatment_prob,
        args.propensity_clip,
        args.alpha,
        args.gaussian_mean_span,
        args.gaussian_std_low,
        args.gaussian_std_high,
    )
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("DGP numeric parameters must be finite")
    if not 0.0 < args.treatment_prob < 1.0 or not 0.0 < args.propensity_clip < 0.5:
        raise ValueError("invalid treatment probability or propensity clip")
    if args.gaussian_std_low <= 0.0 or args.gaussian_std_high < args.gaussian_std_low:
        raise ValueError("invalid Gaussian standard deviation bounds")
    if args.intervention_repeats < 1 or args.intervention_batch_size < 1:
        raise ValueError("invalid intervention configuration")
    if args.family == "one_shot_local_logistic_lasso_fedavg":
        if args.model != "linear":
            raise ValueError("local logistic family requires --model linear")
        if args.cv_folds < 2 or args.cv_folds > args.site_min_samples:
            raise ValueError("cv-folds must be in [2, site_min_samples]")
        if args.lambda_count < 2 or not 0.0 < args.lambda_min_ratio < 1.0:
            raise ValueError("invalid local logistic lambda path")
        if args.optimizer_max_iter < 1 or args.optimizer_tolerance <= 0.0:
            raise ValueError("invalid local logistic optimizer limits")
        if not 1 <= args.optimizer_polish_max_iter <= args.optimizer_max_iter:
            raise ValueError("optimizer polish max iter must lie in [1, optimizer max iter]")
        if not 0.0 < args.optimizer_backtrack_factor < 1.0:
            raise ValueError("optimizer backtrack factor must lie in (0,1)")
    else:
        if args.model == "linear":
            raise ValueError("MLP explicit-interaction group-regularization family requires an MLP model")
        if args.local_epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0.0:
            raise ValueError("invalid MLP local training configuration")
        if not 0.0 <= args.momentum < 1.0 or not 0.0 < args.validation_fraction < 0.5:
            raise ValueError("invalid momentum or validation fraction")
        if (
            not args.lambda_ratios
            or args.lambda_ratios[0] != 0.0
            or tuple(sorted(set(args.lambda_ratios))) != args.lambda_ratios
            or any(value < 0.0 or not math.isfinite(value) for value in args.lambda_ratios)
        ):
            raise ValueError("lambda ratios must be increasing, unique, finite, and start at zero")


def run_config(args):
    return {
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "splits": ["iid", "noniid"],
        "synthetic_alpha": args.alpha,
        "effect_modifier_num_sites": args.num_sites,
        "synthetic_num_test": args.test_size,
        "effect_modifier_covariate_dim": args.covariate_dim,
        "effect_modifier_signal_dim": args.signal_dim,
        "effect_modifier_signal_strength": args.signal_strength,
        "effect_modifier_main_effect_strength": args.main_effect_strength,
        "effect_modifier_confounding_strength": args.confounding_strength,
        "effect_modifier_intercept": args.outcome_intercept,
        "effect_modifier_treatment_prob": args.treatment_prob,
        "effect_modifier_propensity_clip": args.propensity_clip,
        "effect_modifier_site_min_samples": args.site_min_samples,
        "effect_modifier_site_max_samples": args.site_max_samples,
        "effect_modifier_site_gaussian_mean_span": args.gaussian_mean_span,
        "effect_modifier_site_gaussian_std_low": args.gaussian_std_low,
        "effect_modifier_site_gaussian_std_high": args.gaussian_std_high,
        "prediction_intervention_modes": ["permute", "zero"],
        "prediction_intervention_repeats": args.intervention_repeats,
        "prediction_intervention_batch_size": args.intervention_batch_size,
        "threads": args.threads,
    }


def method_config(args):
    common = {
        "family": args.family,
        "training_communication_rounds": 1,
        "site_uploads_per_site": 1,
        "server_weighting": "n_k_over_N_arithmetic_mean",
        "global_cv": False,
        "post_upload_training_site_access": False,
        "hessian_uploaded": False,
        "fisher_uploaded": False,
        "gradient_uploaded": False,
    }
    if args.family == "one_shot_local_logistic_lasso_fedavg":
        common.update(
            {
                "models": ["linear"],
                "loss": "site_local_soft_target_Bernoulli_logistic",
                "unpenalized_block": "intercept,A,%d_raw_X_main_effects" % args.covariate_dim,
                "penalized_block": "%d_A_by_X_interactions_only" % args.covariate_dim,
                "penalty": "L1",
                "site_local_cv_folds": args.cv_folds,
                "site_local_lambda_count": args.lambda_count,
                "site_local_lambda_min_ratio": args.lambda_min_ratio,
                "site_local_lambda_rule": "local_lambda_max_geometric_path_min_validation_loss_largest_lambda_tie",
                "site_local_standardization": "fit_within_site_and_fold_then_backtransform",
                "response_stabilization": SMOOTHING_RULE,
                "optimizer": "site_local_FISTA_adaptive_restart_backtracking",
                "optimizer_max_iter": args.optimizer_max_iter,
                "optimizer_momentum_reset_polish_rule": MOMENTUM_RESET_POLISH_RULE,
                "optimizer_polish_max_iter": args.optimizer_polish_max_iter,
                "optimizer_tolerance": args.optimizer_tolerance,
                "optimizer_initial_step": args.optimizer_initial_step,
                "optimizer_backtrack_factor": args.optimizer_backtrack_factor,
                "optimizer_min_step": args.optimizer_min_step,
                "standardization_scale_floor": args.standardization_scale_floor,
                "payload": (
                    "site_idx_int64_plus_n_k_int64_plus_%d_float64_raw_scale_coefficients"
                    % (2 + 2 * args.covariate_dim)
                ),
                "payload_bytes_per_site": (2 + 2 * args.covariate_dim) * 8 + 16,
            }
        )
    else:
        common.update(
            {
                "models": ["SyntheticMLP", "SyntheticMLPDeep"],
                "loss": "cross_entropy",
                "local_epochs": args.local_epochs,
                "batch_size": args.batch_size,
                "optimizer": "SGD_then_explicit_interaction_input_column_group_prox",
                "learning_rate": args.learning_rate,
                "momentum": args.momentum,
                "penalty": (
                    "sum_L2_norms_of_%d_first_layer_explicit_A_by_X_input_columns"
                    % args.covariate_dim
                ),
                "unpenalized_parameters": (
                    "A_and_%d_raw_X_first_layer_columns_all_biases_and_deeper_layers"
                    % args.covariate_dim
                ),
                "site_local_validation_fraction": args.validation_fraction,
                "site_local_holdout_rule": "observed_class_stratified_singletons_retained_in_training",
                "site_local_lambda_ratios": list(args.lambda_ratios),
                "site_local_lambda_scale": "maximum_interaction_group_CE_gradient_norm_at_common_initialization",
                "site_local_selection_rule": "largest_lambda_within_one_SE_of_minimum_validation_CE",
                "site_local_refit": "common_initialization_full_site",
                "site_local_refit_epochs": args.local_epochs,
                "payload": "site_idx_int64_plus_n_k_int64_plus_complete_float32_model_vector",
                "server_aggregator": "existing_run_one_shot_algs.get_fedavg_model",
                "regularization_interpretation": (
                    "explicit_A_by_X_input_columns_only_not_identifiable_modifier_selection"
                ),
                "unpenalized_nonlinear_bypass": "A_and_raw_X_can_form_interactions_in_hidden_layers",
                "matched_zero_penalty_reference": (
                    "independent_no_prox_CPU_SGD_same_common_state_site_loader_seed_and_refit_seed"
                ),
                "legacy_GPU_FedAvg_bitwise_equivalence_claim": False,
                "execution_device": "cpu",
                "loader_rng": "explicit_per_site_CPU_torch_Generator",
            }
        )
    return common


def _csv_text(rows, fieldnames):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _model_fingerprint(model):
    vector = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(b"fedfisher-one-shot-aggregate-model-v1\0")
    digest.update(np.ascontiguousarray(vector).tobytes())
    return digest.hexdigest()


def _primary_row(rows):
    candidates = [
        row
        for row in rows
        if row["intervention_mode"] == "permute"
        and row["metric"] == "tau_intervention_logit"
    ]
    if len(candidates) != 1:
        raise RuntimeError("expected exactly one primary recovery row")
    return candidates[0]


def _refuse_existing(paths):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError("refusing to overwrite existing cell artifact: %s" % path)


def run(args):
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    contract = VARIANT_CONTRACT[args.variant]
    output_dir = Path(args.output_dir)
    manifest = initialize_or_validate_run(
        output_dir,
        args.run_id,
        args.family,
        args.variant,
        run_config(args),
        method_config(args),
    )
    model_tag = args.model
    stem = "oneshot_seed_%06d_%s_%s" % (args.seed, args.split, model_tag)
    detail_path = output_dir / (stem + "_prediction_intervention.csv")
    summary_path = output_dir / (stem + "_prediction_intervention_summary.csv")
    model_summary_path = output_dir / (stem + "_prediction_intervention_model_summary.csv")
    site_summary_path = output_dir / (stem + "_prediction_intervention_site_summary.csv")
    fit_path = output_dir / (stem + "_fit.json")
    result_path = output_dir / (stem + "_result.env")
    marker_path = output_dir / "completed_cells" / (stem + ".done")
    expected_paths = [detail_path, summary_path, model_summary_path, fit_path, result_path, marker_path]
    if contract["intervention_scope"] == "site":
        expected_paths.append(site_summary_path)
    _refuse_existing(expected_paths)

    site_datasets, _, test_dataset, _ = get_dataset(
        datatype="SyntheticEffectModifier",
        n_client=args.num_sites,
        n_c=2,
        alpha=args.alpha,
        synthetic_split=args.split,
        synthetic_num_test=args.test_size,
        effect_modifier_covariate_dim=args.covariate_dim,
        effect_modifier_signal_dim=args.signal_dim,
        effect_modifier_signal_strength=args.signal_strength,
        effect_modifier_main_effect_strength=args.main_effect_strength,
        effect_modifier_confounding_strength=args.confounding_strength,
        effect_modifier_intercept=args.outcome_intercept,
        effect_modifier_treatment_prob=args.treatment_prob,
        effect_modifier_propensity_clip=args.propensity_clip,
        effect_modifier_site_min_samples=args.site_min_samples,
        effect_modifier_site_max_samples=args.site_max_samples,
        effect_modifier_site_gaussian_shift=contract["site_gaussian_shift"],
        effect_modifier_site_gaussian_mean_span=args.gaussian_mean_span,
        effect_modifier_site_gaussian_std_low=args.gaussian_std_low,
        effect_modifier_site_gaussian_std_high=args.gaussian_std_high,
        seed=args.seed,
    )
    evaluation_sites = None
    if contract["intervention_scope"] == "site":
        evaluation_sites = build_effect_modifier_site_evaluation_datasets(
            test_dataset, site_datasets, args.split, args.alpha, args.seed
        )
    sites = tensor_datasets_to_sealed_sites(site_datasets, args.covariate_dim)
    train_hash = training_fingerprint(sites)
    test_hash = dataset_fingerprint(test_dataset)
    local_audits = []
    payloads = []

    if args.family == "one_shot_local_logistic_lasso_fedavg":
        config = LocalLogisticConfig(
            covariate_dim=args.covariate_dim,
            cv_folds=args.cv_folds,
            lambda_count=args.lambda_count,
            lambda_min_ratio=args.lambda_min_ratio,
            cv_seed_offset=args.cv_seed_offset,
            scale_floor=args.standardization_scale_floor,
            optimizer_max_iter=args.optimizer_max_iter,
            optimizer_polish_max_iter=args.optimizer_polish_max_iter,
            optimizer_tolerance=args.optimizer_tolerance,
            optimizer_initial_step=args.optimizer_initial_step,
            optimizer_backtrack_factor=args.optimizer_backtrack_factor,
            optimizer_min_step=args.optimizer_min_step,
        )
        for site in sites:
            payload, audit = fit_local_logistic_lasso_site(site, config, args.seed)
            payloads.append(payload)
            local_audits.append(audit)
        aggregate_coefficients, communication = aggregate_linear_payloads(
            payloads, args.covariate_dim
        )
        model = OneShotLinearLogisticModel(aggregate_coefficients).to("cpu")
        aggregate_metadata = {
            "aggregate_coefficients": aggregate_coefficients.tolist(),
            "penalty_mask": {
                "unpenalized_parameter_indices": [0, 1]
                + list(range(2, 2 + args.covariate_dim)),
                "penalized_parameter_indices": list(
                    range(2 + args.covariate_dim, 2 + 2 * args.covariate_dim)
                ),
            },
        }
    else:
        config = MlpGroupLassoConfig(
            covariate_dim=args.covariate_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            validation_fraction=args.validation_fraction,
            lambda_ratios=args.lambda_ratios,
            split_seed_offset=args.split_seed_offset,
            loader_seed_offset=args.loader_seed_offset,
        )
        common_state = make_common_mlp_state(
            args.model, 1 + 2 * args.covariate_dim, args.seed
        )
        for site in sites:
            payload, audit = fit_mlp_group_lasso_site(
                site, args.model, common_state, config, args.seed
            )
            payloads.append(payload)
            local_audits.append(audit)
        model, communication = aggregate_mlp_payloads(
            payloads,
            args.model,
            1 + 2 * args.covariate_dim,
            common_state,
        )
        aggregate_metadata = {
            "model_parameter_count": int(parameters_to_vector(model.parameters()).numel()),
            "penalty_mask": {
                "penalized_module": "net.0.weight",
                "penalized_input_columns": list(
                    range(1 + args.covariate_dim, 1 + 2 * args.covariate_dim)
                ),
                "group_axis": "all_first_hidden_units_per_interaction_input_column",
                "all_other_parameters_unpenalized": True,
            },
        }

    if not all(site.sealed for site in sites):
        raise RuntimeError("every training site must be sealed before aggregation completes")
    communication["observed_post_upload_training_site_accesses"] = 0
    evaluation_communication = evaluation_communication_report(
        contract["intervention_scope"], args.num_sites
    )
    detail_rows = []
    summary_rows = []
    model_summary_rows = []
    site_summary_rows = []
    eval_args = {
        "device": "cpu",
        "bs": args.intervention_batch_size,
        "prediction_intervention_batch_size": args.intervention_batch_size,
        "dataset": "SyntheticEffectModifier",
        "effect_modifier_covariate_dim": args.covariate_dim,
        "effect_modifier_signal_dim": args.signal_dim,
    }
    metadata = {
        "seed": args.seed,
        "model": args.model,
        "split": args.split,
        "alpha": args.alpha,
        "synthetic_dim": 1 + 2 * args.covariate_dim,
        "synthetic_signal_dim": args.signal_dim,
    }
    add_model_prediction_intervention(
        detail_rows,
        summary_rows,
        model_summary_rows,
        args.family,
        model,
        test_dataset,
        eval_args,
        metadata,
        modes=("permute", "zero"),
        repeats=args.intervention_repeats,
        seed=args.seed + sum(ord(character) for character in args.family),
        signal_indices=list(
            range(1 + args.covariate_dim, 1 + args.covariate_dim + args.signal_dim)
        ),
        candidate_indices=list(
            range(1 + args.covariate_dim, 1 + 2 * args.covariate_dim)
        ),
        permutation_scope=contract["intervention_scope"],
        site_datasets=evaluation_sites,
        site_summary_rows=site_summary_rows,
    )
    primary = _primary_row(summary_rows)
    fit_record = {
        "format_version": 2,
        "run_id": args.run_id,
        "family": args.family,
        "method_name": METHOD_NAMES[args.family],
        "variant": args.variant,
        "seed": args.seed,
        "split": args.split,
        "model": args.model,
        "training_fingerprint": train_hash,
        "test_fingerprint": test_hash,
        "payload_fingerprint": payload_fingerprint(payloads),
        "aggregate_model_fingerprint": _model_fingerprint(model),
        "training_communication": communication,
        "evaluation_communication": evaluation_communication,
        "protocol_boundary": {
            "individual_training_rows_uploaded": False,
            "server_receives_gradient": False,
            "server_receives_hessian": False,
            "server_receives_fisher": False,
            "global_cv": False,
            "post_upload_training_site_access": False,
            "local_audit_is_simulator_metadata_not_server_payload": True,
        },
        "local_fit_audits": local_audits,
        "aggregate": aggregate_metadata,
        "primary": {
            "metric": "permute/tau_intervention_logit",
            "top2_hits": int(primary["topk_hits"]),
            "auroc": float(primary["auroc"]),
        },
    }
    atomic_csv_text(detail_path, _csv_text(detail_rows, PREDICTION_DETAIL_FIELDS))
    atomic_csv_text(summary_path, _csv_text(summary_rows, PREDICTION_SUMMARY_FIELDS))
    atomic_csv_text(
        model_summary_path,
        _csv_text(model_summary_rows, PREDICTION_MODEL_SUMMARY_FIELDS),
    )
    if contract["intervention_scope"] == "site":
        atomic_csv_text(
            site_summary_path,
            _csv_text(site_summary_rows, PREDICTION_SITE_SUMMARY_FIELDS),
        )
    atomic_json(fit_path, json_safe(fit_record))
    result_fields = [
        ("format_version", 2),
        ("run_id", args.run_id),
        ("family", args.family),
        ("variant", args.variant),
        ("seed", args.seed),
        ("split", args.split),
        ("model", args.model),
        ("provenance_sha256", manifest["provenance_sha256"]),
        ("run_config_sha256", manifest["run_config_sha256"]),
        ("method_config_sha256", manifest["method_config_sha256"]),
        ("variant_config_sha256", manifest["variant_config_sha256"]),
        ("training_fingerprint", train_hash),
        ("test_fingerprint", test_hash),
        ("payload_fingerprint", fit_record["payload_fingerprint"]),
        ("aggregate_model_fingerprint", fit_record["aggregate_model_fingerprint"]),
        ("training_communication_rounds", communication["training_communication_rounds"]),
        ("site_upload_count", communication["site_upload_count"]),
        ("post_upload_training_site_accesses", 0),
        ("evaluation_communication_rounds", evaluation_communication["evaluation_communication_rounds"]),
        ("primary_metric", "permute/tau_intervention_logit"),
        ("primary_top2_hits", int(primary["topk_hits"])),
        ("primary_auroc", float(primary["auroc"])),
    ]
    atomic_create_text(result_path, env_text(result_fields))
    artifacts = {
        "detail_csv": sha256_file(detail_path),
        "summary_csv": sha256_file(summary_path),
        "model_summary_csv": sha256_file(model_summary_path),
        "fit_json": sha256_file(fit_path),
        "result_env": sha256_file(result_path),
    }
    if contract["intervention_scope"] == "site":
        artifacts["site_summary_csv"] = sha256_file(site_summary_path)
    complete_cell(output_dir, stem, manifest, artifacts)
    print(
        "completed family=%s variant=%s seed=%d split=%s model=%s Top2=%d AUROC=%.12g training_rounds=1 evaluation_rounds=%d"
        % (
            args.family,
            args.variant,
            args.seed,
            args.split,
            args.model,
            int(primary["topk_hits"]),
            float(primary["auroc"]),
            evaluation_communication["evaluation_communication_rounds"],
        )
    )
    return fit_record


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
