"""Complete probability-to-modifier recipe; historical V1 remains unchanged."""
from . import protocol as prediction

PROTOCOL = 'survival-mlp8-discovery-v2'
PILOT_TASK = 'effect-modifier-survival-mlp8-discovery-pilot'
FORMAL_TASK = 'effect-modifier-survival-mlp8-discovery-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
SCENARIOS = prediction.SCENARIOS
PILOT_SEEDS = prediction.PILOT_SEEDS
FORMAL_SEEDS = prediction.FORMAL_SEEDS
PSEUDO_SCOPES = prediction.PSEUDO_SCOPES
TAGS = prediction.TAGS
FEDERATED_TAGS = prediction.FEDERATED_TAGS


def _prediction_task(task):
    if task not in TASKS:
        raise ValueError('unknown MLP8 discovery task')
    return prediction.PILOT_TASK if task == PILOT_TASK else prediction.FORMAL_TASK


def validate_parameters(task, parameters):
    return prediction.validate_parameters(_prediction_task(task), parameters)


def seed_sequence(task, parameters):
    return prediction.seed_sequence(_prediction_task(task), parameters)


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'survival_mlp8_v2_%s_%s' % (p['scenario'], 'pilot' if task == PILOT_TASK else '300seeds')


def run_config(task, parameters):
    config = prediction.run_config(_prediction_task(task), parameters)
    config.update(protocol=PROTOCOL, task=task, run_id=run_id(task, parameters))
    config['evaluation'].update(modifier_permutation_performed=True,
        discovery=dict(primary='permute/log_cumulative_hazard_contrast', repeats=5,
            top2='probability_all_signals_selected', auroc=True,
            seed_rule='SeedSequence([seed,4291,repeat,covariate])',
            tie_rule='uniform_without_replacement_within_exact_score_boundary_ties'))
    config['timing']['stages'].append('discovery_evaluation')
    return config


def validate_seed(config, seed):
    return prediction.validate_seed(config, seed)
