"""Adapters for the two original LASSO task contracts, without local launchers."""
import os

from .errors import AdapterError


DSLASSO_RUN_IDS = {
    "control": "dslassocov_control_server_scheduler_v2r2_300seeds",
    "site_gaussian_shift": "dslassocov_site_gaussian_shift_server_scheduler_v2r2_300seeds",
    "local_permutation": "dslassocov_local_permutation_server_scheduler_v2r2_300seeds",
    "site_gaussian_local_permutation": (
        "dslassocov_site_gaussian_local_permutation_server_scheduler_v2r2_300seeds"
    ),
}

ONE_SHOT_RUN_SUFFIX = (
    "stationarity_4000iter_full_scope_server_scheduler_v2r2_300seeds"
)

DSLASSO_ENVIRONMENT = {
    "SEED_START": "0",
    "SEED_END": "299",
    "SPLITS": "iid,noniid",
    "SYNTHETIC_ALPHA": "0.1",
    "EFFECT_MODIFIER_NUM_SITES": "10",
    "SYNTHETIC_NUM_TEST": "10000",
    "EFFECT_MODIFIER_COVARIATE_DIM": "100",
    "EFFECT_MODIFIER_SIGNAL_DIM": "2",
    "EFFECT_MODIFIER_SIGNAL_STRENGTH": "2.0",
    "EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH": "1.0",
    "EFFECT_MODIFIER_CONFOUNDING_STRENGTH": "1.0",
    "EFFECT_MODIFIER_INTERCEPT": "0.0",
    "EFFECT_MODIFIER_TREATMENT_PROB": "0.5",
    "EFFECT_MODIFIER_PROPENSITY_CLIP": "0.05",
    "EFFECT_MODIFIER_SITE_MIN_SAMPLES": "500",
    "EFFECT_MODIFIER_SITE_MAX_SAMPLES": "1000",
    "EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN": "0.75",
    "EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW": "0.75",
    "EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH": "1.25",
    "PREDICTION_INTERVENTION_MODES": "permute,zero",
    "PREDICTION_INTERVENTION_REPEATS": "5",
    "PREDICTION_INTERVENTION_BATCH_SIZE": "1024",
    "CV_FOLDS": "5",
    "CV_SEED_OFFSET": "4000037",
    "LAMBDA_COUNT": "30",
    "LAMBDA_MIN_RATIO": "0.001",
    "STANDARDIZATION_SCALE_FLOOR": "1e-12",
    "OPTIMIZER_MAX_ITER": "2000",
    "OPTIMIZER_TOLERANCE": "1e-6",
    "OPTIMIZER_INITIAL_STEP": "1.0",
    "OPTIMIZER_BACKTRACK_FACTOR": "0.5",
    "OPTIMIZER_MIN_STEP": "1e-12",
    "THREADS_PER_WORKER": "1",
}



def _dslasso_runner_args(runner, run_id, variant, output_dir):
    values = DSLASSO_ENVIRONMENT
    argv = [
        "--run-id", run_id,
        "--variant", variant,
        "--output-dir", str(output_dir),
        "--seed", "0",
        "--split", "iid",
        "--alpha", values["SYNTHETIC_ALPHA"],
        "--num-sites", values["EFFECT_MODIFIER_NUM_SITES"],
        "--num-test", values["SYNTHETIC_NUM_TEST"],
        "--covariate-dim", values["EFFECT_MODIFIER_COVARIATE_DIM"],
        "--signal-dim", values["EFFECT_MODIFIER_SIGNAL_DIM"],
        "--signal-strength", values["EFFECT_MODIFIER_SIGNAL_STRENGTH"],
        "--main-effect-strength", values["EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH"],
        "--confounding-strength", values["EFFECT_MODIFIER_CONFOUNDING_STRENGTH"],
        "--outcome-intercept", values["EFFECT_MODIFIER_INTERCEPT"],
        "--treatment-prob", values["EFFECT_MODIFIER_TREATMENT_PROB"],
        "--propensity-clip", values["EFFECT_MODIFIER_PROPENSITY_CLIP"],
        "--site-min-samples", values["EFFECT_MODIFIER_SITE_MIN_SAMPLES"],
        "--site-max-samples", values["EFFECT_MODIFIER_SITE_MAX_SAMPLES"],
        "--site-gaussian-mean-span", values["EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN"],
        "--site-gaussian-std-low", values["EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW"],
        "--site-gaussian-std-high", values["EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH"],
        "--intervention-modes",
        *values["PREDICTION_INTERVENTION_MODES"].split(","),
        "--intervention-repeats", values["PREDICTION_INTERVENTION_REPEATS"],
        "--intervention-batch-size", values["PREDICTION_INTERVENTION_BATCH_SIZE"],
        "--cv-folds", values["CV_FOLDS"],
        "--cv-seed-offset", values["CV_SEED_OFFSET"],
        "--lambda-count", values["LAMBDA_COUNT"],
        "--lambda-min-ratio", values["LAMBDA_MIN_RATIO"],
        "--standardization-scale-floor", values["STANDARDIZATION_SCALE_FLOOR"],
        "--optimizer-max-iter", values["OPTIMIZER_MAX_ITER"],
        "--optimizer-tolerance", values["OPTIMIZER_TOLERANCE"],
        "--optimizer-initial-step", values["OPTIMIZER_INITIAL_STEP"],
        "--optimizer-backtrack-factor", values["OPTIMIZER_BACKTRACK_FACTOR"],
        "--optimizer-min-step", values["OPTIMIZER_MIN_STEP"],
        "--threads-per-worker", "1",
        "--cpu-workers", os.environ["CPU_WORKERS"],
    ]
    return runner.build_parser().parse_args(argv)



def _run_dslasso(job):
    from effect_modifier_experiment import manage_dslassocov_cpu_run as manager
    from effect_modifier_experiment import run_dslassocov_effect_modifier as runner

    parameters = job["parameters"]
    variant = parameters["variant"]
    shard_index = parameters["shard_index"]
    shard_count = parameters["shard_count"]
    run_id = DSLASSO_RUN_IDS[variant]
    os.environ.update(DSLASSO_ENVIRONMENT)
    os.environ.update(
        {
            "RUN_ID": run_id,
            "EFFECT_MODIFIER_VARIANT": variant,
            "CPU_WORKERS": str(shard_count),
        }
    )
    paths, unused_manifest = manager.initialize_run(
        run_id, variant, create_legacy_scheduler_dirs=False
    )
    del unused_manifest
    runner_args = _dslasso_runner_args(runner, run_id, variant, paths["output_dir"])
    for seed in range(shard_index, 300, shard_count):
        manager.validate_current_run(run_id, variant)
        status_code = manager.seed_status(run_id, variant, seed)
        if status_code == 0:
            continue
        if status_code != 3:
            raise AdapterError("dsLassoCov seed has partial or invalid output")
        for split in ("iid", "noniid"):
            runner_args.seed = seed
            runner_args.split = split
            runner.run(runner_args)
        manager.complete_seed(run_id, variant, seed)
    return 0



def _run_one_shot(job):
    from effect_modifier_experiment import manage_one_shot_lasso_cpu_run as manager
    from effect_modifier_experiment import run_one_shot_lasso_effect_modifier as runner

    parameters = job["parameters"]
    variant = parameters["variant"]
    shard_index = parameters["shard_index"]
    shard_count = parameters["shard_count"]
    for family in manager.FAMILIES:
        manager.initialize_formal(
            manager.formal_runner_args(
                family, variant, run_suffix=ONE_SHOT_RUN_SUFFIX
            )
        )
    for seed in manager.shard_seed_sequence(shard_index, shard_count):
        for family in manager.FAMILIES:
            for split in ("iid", "noniid"):
                for model in manager.MODELS[family]:
                    args = manager.formal_runner_args(
                        family,
                        variant,
                        seed,
                        split,
                        model,
                        run_suffix=ONE_SHOT_RUN_SUFFIX,
                    )
                    status_code = manager.cell_status(args)
                    if status_code == 0:
                        continue
                    if status_code != manager.MISSING_EXIT:
                        raise AdapterError("one-shot cell has partial or invalid output")
                    runner.run(args)
                    if manager.cell_status(args) != 0:
                        raise AdapterError("one-shot completion marker did not validate")
    return 0
