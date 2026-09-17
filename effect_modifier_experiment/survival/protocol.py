"""Fixed survival pilot and pilot-gated deterministic formal shard recipes."""
from copy import deepcopy
from dataclasses import asdict
import math

PROTOCOL = 'survival-pseudo-v1'
PILOT_TASK = 'effect-modifier-survival-pilot'
FORMAL_TASK = 'effect-modifier-survival-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
SCENARIOS = ('iid', 'covariate_shift')
CELLS = ('dslasso', 'fedfisher_diag_shallow', 'fedfisher_diag_deep',
         'fedfisher_kfac_shallow', 'fedfisher_kfac_deep')
EVALUATION = dict(repeats=5, primary='permute/log_cumulative_hazard_contrast',
                  treatment_contrast='eta(A=1,X)-eta(A=0,X)',
                  evaluation_data='independent_common_standard_gaussian',
                  end_to_end_one_round_claimed=False)


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown survival task or invalid parameters')
    fields = {'cell', 'scenario'} if task == PILOT_TASK else {'cell', 'scenario', 'shard_index', 'shard_count'}
    if set(parameters) != fields:
        raise ValueError('survival parameters do not match the fixed task schema')
    if (not isinstance(parameters['cell'], str) or parameters['cell'] not in CELLS
            or not isinstance(parameters['scenario'], str)
            or parameters['scenario'] not in SCENARIOS):
        raise ValueError('unknown survival method or scenario')
    if task == FORMAL_TASK:
        index, count = parameters['shard_index'], parameters['shard_count']
        if type(index) is not int or type(count) is not int or count not in (8, 16) or not 0 <= index < count:
            raise ValueError('invalid deterministic survival shard')
    return dict(parameters)


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return (0,) if task == PILOT_TASK else tuple(range(p['shard_index'], 300, p['shard_count']))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'survival_v1_%s_%s_%s' % (p['cell'], p['scenario'], 'pilot' if task == PILOT_TASK else '300seeds')


def method_config(cell):
    if not isinstance(cell, str) or cell not in CELLS:
        raise ValueError('unknown survival method')
    if cell == 'dslasso':
        from .dslasso import DslassoConfig
        return dict(family='dslasso_survival_nls', optimizer=asdict(DslassoConfig()))
    from .fisher import FisherConfig
    _, kind, architecture = cell.split('_')
    return dict(family='fedfisher_survival_working_gaussian', fisher=kind,
                optimizer=asdict(FisherConfig(architecture=architecture)))


def run_config(task, parameters):
    from .data import SurvivalConfig
    p = validate_parameters(task, parameters)
    dgp = asdict(SurvivalConfig())
    result = dict(protocol=PROTOCOL, task=task, run_id=run_id(task, p),
                cell=p['cell'], scenario=p['scenario'], dgp=dgp,
                method=method_config(p['cell']), evaluation=deepcopy(EVALUATION),
                pseudo=dict(estimator='site_local_kaplan_meier_infinitesimal_jackknife',
                            response='survival_at_tau', clip_response=False),
                mean_link=dict(scale='log_cumulative_hazard', inverse='exp(-exp(eta))'),
                public_initial_eta=math.log(dgp['weibull_lambda'] * dgp['tau'] ** dgp['weibull_shape']),
                execution_scope='single_process_cross_silo_simulation',
                privacy_deployment_claimed=False,
                timing=dict(stages=['data_preparation', 'fit', 'evaluation'],
                            data_preparation='simulation_and_site_response_diagnostics; diagnostic_pseudo_values_recomputed_in_fit',
                            fit_includes=['pseudo_responses', 'tuning', 'refitting',
                                          'fisher', 'serialization', 'server_aggregation'],
                            excludes=['top_level_module_imports', 'queueing', 'artifact_io',
                                      'actual_network_transport']))
    result.update(seed=0) if task == PILOT_TASK else result.update(seeds=list(range(300)))
    return result


def validate_seed(config, seed):
    allowed = [config['seed']] if config['task'] == PILOT_TASK else config['seeds']
    if type(seed) is not int or seed not in allowed:
        raise ValueError('seed is outside the survival scientific run')
    return seed
