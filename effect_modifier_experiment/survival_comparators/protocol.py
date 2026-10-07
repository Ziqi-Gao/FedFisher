"""Fixed small cohort; admission does not import numerical dependencies."""
from copy import deepcopy
from dataclasses import asdict

PROTOCOL = 'survival-comparators-v1'
PILOT_TASK = 'effect-modifier-survival-comparators-pilot'
TASKS = (PILOT_TASK,)
SCENARIOS = ('iid', 'covariate_shift')
SEEDS = tuple(range(10))
TAGS = ('fedavg', 'dslasso')
PSEUDO_SCOPES = ('site_local_pv', 'pooled_km_pv')


def validate_parameters(task, parameters):
    if (task != PILOT_TASK or not isinstance(parameters, dict)
            or set(parameters) != {'scenario', 'seed'}
            or type(parameters['scenario']) is not str or parameters['scenario'] not in SCENARIOS
            or type(parameters['seed']) is not int or parameters['seed'] not in SEEDS):
        raise ValueError('survival comparator parameters require a fixed scenario and seed 0..9')
    return dict(parameters)


def seed_sequence(task, parameters):
    return (validate_parameters(task, parameters)['seed'],)


def run_id(task, parameters):
    return 'survival_comparators_v1_%s_10seeds' % validate_parameters(task, parameters)['scenario']


def run_config(task, parameters):
    from ..survival_mlp8 import discovery_protocol
    from .fedavg import FedAvgConfig
    from .inputs import historical_binding
    p = validate_parameters(task, parameters)
    original = discovery_protocol.run_config(discovery_protocol.PILOT_TASK, {'scenario': p['scenario']})
    return dict(protocol=PROTOCOL, task=task, run_id=run_id(task, p), cell='comparators',
        family='survival_comparators', scenario=p['scenario'], seeds=list(SEEDS),
        dgp=original['dgp'], public_initial_eta=original['public_initial_eta'],
        method=dict(family='survival_fedavg', optimizer=asdict(FedAvgConfig())),
        pseudo_scopes=list(PSEUDO_SCOPES), pseudo=deepcopy(original['pseudo']),
        mean_link=deepcopy(original['mean_link']), evaluation=deepcopy(original['evaluation']),
        historical_inputs=historical_binding(), models=list(TAGS),
        fit_origin=dict(fedavg='new_training', dslasso='accepted_r3_model_replay'),
        timing=dict(stages=['historical_input_validation', 'data_preparation', 'fit',
                           'probability_evaluation', 'discovery_evaluation'],
                    historical_fit_performed=False,
                    includes='new FedAvg complete fitting and evaluation; restored dsLasso fitting charged separately',
                    excludes=['module_imports', 'queueing', 'artifact_io', 'actual_network_transport']))


def validate_seed(config, seed):
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError('seed outside the fixed ten-seed survival comparison')
    return seed
