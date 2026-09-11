"""Fixed scientific recipes; neither requests nor methods choose resources."""

from dataclasses import asdict

PILOT_TASK = "effect-modifier-comparison-pilot"
FORMAL_TASK = "effect-modifier-comparison-shard"
TASKS = (PILOT_TASK, FORMAL_TASK)
VARIANTS = ("control", "site_gaussian_shift", "local_permutation",
            "site_gaussian_local_permutation")
SPLITS = ("iid", "noniid")
DGP = dict(num_sites=10, alpha=0.1, num_test=10000, covariate_dim=100,
           signal_dim=2, signal_strength=2.0, main_effect_strength=1.0,
           confounding_strength=1.0, outcome_intercept=0.0, treatment_prob=0.5,
           propensity_clip=0.05, site_min_samples=500, site_max_samples=1000,
           site_gaussian_mean_span=0.75, site_gaussian_std_low=0.75,
           site_gaussian_std_high=1.25)
EVALUATION = dict(repeats=5, batch_size=1024, permutation_seed_offset=6100037,
                  primary_scale="tau_prob", auxiliary_scale="tau_logit",
                  modes=["permute", "zero"])

CELLS = {"dslassocov": {"family": "dslassocov"},
         "local_logistic_lasso": {"family": "local_logistic_lasso"},
         "r_unpenalized": {"family": "rlearner", "penalty": "none"},
         "r_adaptive_lasso": {"family": "rlearner", "penalty": "adaptive_lasso"}}
for _short, _model in (("shallow", "SyntheticMLP"), ("deep", "SyntheticMLPDeep")):
    for _family in ("one_shot_fedavg", "one_shot_group_lasso"):
        CELLS[_family + "_" + _short] = dict(family=_family, model=_model)
    for _epochs in (1, 5, 10, 30):
        for _mode in ("patience", "budget30"):
            CELLS["fedavg_%s_e%d_%s" % (_short, _epochs, _mode)] = dict(
                family="fedavg", model=_model, local_epochs=_epochs,
                stopping_mode=("validation_patience" if _mode == "patience"
                               else "fixed_total_epochs"))


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError("unknown comparison task or invalid parameters")
    if set(parameters) != {"variant", "cell", "shard_index", "shard_count"}:
        raise ValueError("comparison parameters require exactly variant/cell/shard_index/shard_count")
    if (not isinstance(parameters["variant"], str) or not isinstance(parameters["cell"], str)
            or parameters["variant"] not in VARIANTS or parameters["cell"] not in CELLS):
        raise ValueError("unknown comparison variant/cell")
    index, count = parameters["shard_index"], parameters["shard_count"]
    allowed = (1,) if task == PILOT_TASK else (8, 16)
    if type(index) is not int or type(count) is not int or count not in allowed or not 0 <= index < count:
        raise ValueError("invalid deterministic comparison shard")
    return dict(parameters)


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return tuple(range(p["shard_index"], 1 if task == PILOT_TASK else 300, p["shard_count"]))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return "comparison_%s_%s_v1_%s" % (
        p["cell"], p["variant"], "pilot" if task == PILOT_TASK else "300seeds")


def method_config(cell):
    """Serialize every algorithm default into the scientific manifest."""
    if not isinstance(cell, str) or cell not in CELLS:
        raise ValueError("unknown comparison cell")
    from .fedavg import FedAvgConfig
    from .rlearner import RlearnerConfig
    from ..one_shot_lasso_benchmark import LocalLogisticConfig, MlpGroupLassoConfig
    config = dict(CELLS[cell])
    family = config["family"]
    if family == "fedavg":
        config["optimizer"] = asdict(FedAvgConfig(
            local_epochs=config["local_epochs"], stopping_mode=config["stopping_mode"]))
    elif family == "rlearner":
        config["optimizer"] = asdict(RlearnerConfig())
    elif family == "local_logistic_lasso":
        config["optimizer"] = asdict(LocalLogisticConfig())
    elif family.startswith("one_shot_"):
        config["optimizer"] = asdict(MlpGroupLassoConfig())
    else:
        config["optimizer"] = dict(cv_folds=5, lambda_count=30, lambda_min_ratio=0.001,
            cv_seed_offset=4000037, standardization_scale_floor=1e-12,
            optimizer_max_iter=2000, optimizer_tolerance=1e-6,
            optimizer_initial_step=1.0, optimizer_backtrack_factor=0.5,
            optimizer_min_step=1e-12)
    return config


def run_config(task, parameters):
    p = validate_parameters(task, parameters)
    return dict(protocol="causal-comparison-v1", task=task, cell=p["cell"],
                variant=p["variant"], seeds=[0, 0 if task == PILOT_TASK else 299],
                splits=list(SPLITS), dgp=dict(DGP), evaluation=dict(EVALUATION),
                method=method_config(p["cell"]),
                timing_contract="serial CPU simulation; full selection/refit/evaluation cost",
                estimand="probability CATE linear working model for R; outcome contrast otherwise")
