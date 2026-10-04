"""Fixed pooled survival prediction benchmark; numerical imports stay lazy.

The new benchmark first evaluates survival probabilities against the simulated
truth.  Treatment contrasts are secondary diagnostics.  Historical federated
models retain their original producer provenance and are only replayed here.
"""
from copy import deepcopy
from dataclasses import asdict
import math

PROTOCOL = 'survival-pooled-prediction-v1'
PILOT_TASK = 'effect-modifier-survival-pooled-pilot'
FORMAL_TASK = 'effect-modifier-survival-pooled-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
FAMILIES = ('linear', 'shallow', 'deep')
CELLS = FAMILIES
SCENARIOS = ('iid', 'covariate_shift')
PILOT_SEEDS = (0, 49)
FORMAL_SEEDS = tuple(range(300))
PSEUDO_SCOPES = ('site_local_pv', 'pooled_km_pv')
HISTORICAL_REFERENCE = dict(
    path='/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1/'
         'formal-acceptance-r3-20260927T163352Z/formal.acceptance.json',
    sha256='115fd9ae833f019ec5fd42ac1f8e23bbc3400d330300ed345dc1e475b1b207d5',
)
EVALUATION = dict(
    primary='survival_probability_at_tau_vs_simulation_truth',
    probability_metrics=['rmse', 'mae', 'mean_bias'],
    probability_targets=['factual', 'control', 'treatment'],
    federated_comparison='historical_FL_vs_both_pooled_predictions',
    secondary='log_cumulative_hazard_contrast_rmse',
    treatment_contrast='eta(A=1,X)-eta(A=0,X)',
    evaluation_data='independent_common_standard_gaussian',
    modifier_permutation_performed=False,
    end_to_end_one_round_claimed=False,
)


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown pooled survival task or invalid parameters')
    fields = ({'family', 'scenario'} if task == PILOT_TASK else
              {'family', 'scenario', 'shard_index', 'shard_count'})
    if set(parameters) != fields:
        raise ValueError('pooled survival parameters do not match the fixed task schema')
    if (not isinstance(parameters['family'], str) or parameters['family'] not in FAMILIES
            or not isinstance(parameters['scenario'], str)
            or parameters['scenario'] not in SCENARIOS):
        raise ValueError('unknown pooled survival model family or scenario')
    if task == FORMAL_TASK:
        index, count = parameters['shard_index'], parameters['shard_count']
        if (type(index) is not int or type(count) is not int or count not in (8, 16)
                or not 0 <= index < count):
            raise ValueError('invalid deterministic pooled survival shard')
    return dict(parameters)


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return (PILOT_SEEDS if task == PILOT_TASK else
            tuple(range(p['shard_index'], 300, p['shard_count'])))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'survival_pooled_v1_%s_%s_%s' % (
        p['family'], p['scenario'], 'pilot' if task == PILOT_TASK else '300seeds')


def historical_cells(family):
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError('unknown pooled survival model family')
    return (('dslasso',) if family == 'linear' else
            ('fedfisher_diag_' + family, 'fedfisher_kfac_' + family))


def historical_tags(family):
    """Prediction archive keys omit the architecture, which is fixed by family."""
    historical_cells(family)
    return ('dslasso',) if family == 'linear' else ('fedfisher_diag', 'fedfisher_kfac')


def method_config(family):
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError('unknown pooled survival model family')
    if family == 'linear':
        from ..survival.recovery_protocol import method_config as recovery_config
        return recovery_config('budget_and_step')
    from ..survival.fisher import FisherConfig
    return dict(family='pooled_survival_mlp',
                optimizer=asdict(FisherConfig(architecture=family)))


def public_initial_eta(dgp=None):
    if dgp is None:
        from ..survival.data import SurvivalConfig
        dgp = asdict(SurvivalConfig())
    return math.log(dgp['weibull_lambda'] * dgp['tau'] ** dgp['weibull_shape'])


def run_config(task, parameters):
    from ..survival.data import SurvivalConfig
    p = validate_parameters(task, parameters)
    dgp = asdict(SurvivalConfig())
    return dict(
        protocol=PROTOCOL, task=task, run_id=run_id(task, p),
        cell=p['family'], family=p['family'], scenario=p['scenario'],
        seeds=list(PILOT_SEEDS if task == PILOT_TASK else FORMAL_SEEDS),
        dgp=dgp, method=method_config(p['family']),
        historical_cells=list(historical_cells(p['family'])),
        historical_reference=deepcopy(HISTORICAL_REFERENCE),
        pseudo_scopes=list(PSEUDO_SCOPES),
        pseudo=dict(estimator='kaplan_meier_infinitesimal_jackknife',
                    response='survival_at_tau', clip_response=False,
                    site_local_pv='preserve_local_labels_and_pool_training_rows',
                    pooled_km_pv='recompute_labels_from_concatenated_observed_survival_rows'),
        mean_link=dict(scale='log_cumulative_hazard', inverse='exp(-exp(eta))'),
        public_initial_eta=public_initial_eta(dgp),
        evaluation=deepcopy(EVALUATION),
        execution_scope='pooled_training_and_historical_federated_model_replay',
        privacy_deployment_claimed=False,
        timing=dict(stages=['historical_input_validation', 'data_preparation',
                            'fit', 'probability_evaluation'],
                    fit_includes=['pseudo_responses', 'tuning', 'refitting'],
                    historical_federated_fit_performed=False,
                    excludes=['top_level_module_imports', 'queueing', 'artifact_io',
                              'actual_network_transport']),
    )


def validate_seed(config, seed):
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError('seed is outside the fixed pooled survival scientific run')
    return seed
