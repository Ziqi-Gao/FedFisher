"""Fixed MLP8 recipes; parameter admission imports no numerical libraries."""
from copy import deepcopy
from dataclasses import asdict

from ..survival_pooled.protocol import HISTORICAL_REFERENCE, public_initial_eta

PROTOCOL = 'survival-mlp8-prediction-v1'
PILOT_TASK = 'effect-modifier-survival-mlp8-pilot'
FORMAL_TASK = 'effect-modifier-survival-mlp8-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
SCENARIOS = ('iid', 'covariate_shift')
PILOT_SEEDS = (0, 49)
FORMAL_SEEDS = tuple(range(300))
PSEUDO_SCOPES = ('site_local_pv', 'pooled_km_pv')
FEDERATED_TAGS = ('fedfisher_diag', 'fedfisher_kfac')
TAGS = FEDERATED_TAGS + tuple('pooled_' + s for s in PSEUDO_SCOPES)


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown MLP8 survival task or invalid parameters')
    fields = {'scenario'} if task == PILOT_TASK else {'scenario', 'shard_index', 'shard_count'}
    if set(parameters) != fields or not isinstance(parameters['scenario'], str) or parameters['scenario'] not in SCENARIOS:
        raise ValueError('MLP8 survival parameters differ from the fixed schema')
    if task == FORMAL_TASK:
        index, count = parameters['shard_index'], parameters['shard_count']
        if type(index) is not int or type(count) is not int or count not in (8, 16) or not 0 <= index < count:
            raise ValueError('invalid deterministic MLP8 survival shard')
    return dict(parameters)


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return PILOT_SEEDS if task == PILOT_TASK else tuple(range(p['shard_index'], 300, p['shard_count']))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'survival_mlp8_v1_%s_%s' % (p['scenario'], 'pilot' if task == PILOT_TASK else '300seeds')


def run_config(task, parameters):
    from ..survival.data import SurvivalConfig
    from .methods import MLP8Config
    p = validate_parameters(task, parameters)
    dgp = asdict(SurvivalConfig())
    return dict(protocol=PROTOCOL, task=task, run_id=run_id(task, p),
        cell='mlp8', family='mlp8', scenario=p['scenario'],
        seeds=list(PILOT_SEEDS if task == PILOT_TASK else FORMAL_SEEDS),
        dgp=dgp, method=dict(family='survival_mlp8_working_gaussian', optimizer=asdict(MLP8Config())),
        pseudo_scopes=list(PSEUDO_SCOPES),
        pseudo=dict(estimator='kaplan_meier_infinitesimal_jackknife', response='survival_at_tau',
            clip_response=False, site_local_pv='preserve_local_labels_and_pool_training_rows',
            pooled_km_pv='recompute_labels_from_concatenated_observed_survival_rows'),
        mean_link=dict(scale='log_cumulative_hazard', inverse='exp(-exp(eta))'),
        public_initial_eta=public_initial_eta(dgp), historical_reference=deepcopy(HISTORICAL_REFERENCE),
        evaluation=dict(primary='survival_probability_at_tau_vs_simulation_truth',
            probability_metrics=['rmse', 'mae', 'mean_bias'], probability_targets=['factual', 'control', 'treatment'],
            federated_comparison='new_MLP8_FL_vs_both_pooled_predictions',
            secondary='log_cumulative_hazard_contrast_rmse', treatment_contrast='eta(A=1,X)-eta(A=0,X)',
            evaluation_data='independent_common_standard_gaussian', modifier_permutation_performed=False,
            end_to_end_one_round_claimed=False),
        execution_scope='paired_new_federated_and_pooled_training', privacy_deployment_claimed=False,
        timing=dict(stages=['historical_input_validation', 'data_preparation', 'fit', 'probability_evaluation'],
            fit_includes=['pseudo_responses', 'local_training', 'fisher', 'serialization', 'server_aggregation', 'pooled_training'],
            historical_federated_fit_performed=False,
            excludes=['top_level_module_imports', 'queueing', 'artifact_io', 'actual_network_transport']))


def validate_seed(config, seed):
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError('seed is outside the fixed MLP8 survival run')
    return seed
