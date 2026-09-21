"""Standard-library-only fixed task guard; scientific imports stay lazy."""
from copy import deepcopy
from dataclasses import asdict

from . import protocol

PROTOCOL = 'survival-dslasso-recovery-v1'
DIAGNOSTIC_TASK = 'effect-modifier-survival-dslasso-diagnostic'
FORMAL_TASK = 'effect-modifier-survival-dslasso-recovery-shard'
TASKS = (DIAGNOSTIC_TASK, FORMAL_TASK)
SCENARIOS = protocol.SCENARIOS
RECIPES = ('baseline', 'budget_only', 'step_only', 'budget_and_step')
ORIGINAL_CLOSEOUT = dict(
    path='/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1/'
         'formal-closeout-20260921T043146Z/closeout.json',
    sha256='5a452ab58f02409336eaa366e88dc75f3f7fdf3a133aee530bff0528bec86a08')
CASE_INVENTORY = dict(
    path='/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1/'
         'dslasso-recovery-preparation-20260920T002124Z/diagnostic-cases.json',
    sha256='7c63a9a871bb4f1a30ad51d38cb2403cf74ab2afe1c9f99a80dad89f087afee0')

_CASE_GROUPS = (
    ('covariate_shift', 'observed_failure',
     (42, 78, 91, 96, 103, 133, 136, 146, 164, 236, 237, 241, 259, 262, 297)),
    ('iid', 'observed_failure', (57, 83, 112, 191)),
    ('iid', 'slow_success_control', (157,)),
    ('iid', 'original_pilot_control', (0,)),
    ('iid', 'previously_unattempted', (73, 89)),
    ('covariate_shift', 'slow_success_control', (193,)),
    ('covariate_shift', 'original_pilot_control', (0,)),
    ('covariate_shift', 'previously_unattempted', (58, 74)),
)
CASES = {'%s_%03d' % (scenario, seed): dict(scenario=scenario, seed=seed, role=role)
         for scenario, role, seeds in _CASE_GROUPS for seed in seeds}


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown survival recovery task or invalid parameters')
    fields = {'case', 'recipe'} if task == DIAGNOSTIC_TASK else {'scenario', 'shard_index', 'shard_count'}
    if set(parameters) != fields:
        raise ValueError('survival recovery parameters do not match the fixed task schema')
    if task == DIAGNOSTIC_TASK:
        if (not isinstance(parameters['case'], str) or parameters['case'] not in CASES
                or not isinstance(parameters['recipe'], str) or parameters['recipe'] not in RECIPES):
            raise ValueError('unknown survival recovery case or recipe')
    else:
        index, count = parameters['shard_index'], parameters['shard_count']
        if (not isinstance(parameters['scenario'], str) or parameters['scenario'] not in SCENARIOS
                or type(index) is not int or type(count) is not int or count not in (8, 16)
                or not 0 <= index < count):
            raise ValueError('invalid deterministic survival recovery shard')
    return dict(parameters)


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return ((CASES[p['case']]['seed'],) if task == DIAGNOSTIC_TASK
            else tuple(range(p['shard_index'], 300, p['shard_count'])))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    if task == DIAGNOSTIC_TASK:
        return 'survival_v1_dslasso_diagnostic_r1_%s_%s' % (p['recipe'], CASES[p['case']]['scenario'])
    return 'survival_v1_dslasso_recovery_r1_%s_300seeds' % p['scenario']


def method_config(recipe):
    from .dslasso import DslassoConfig
    from .recovery_method import RecoveryPolicy
    if not isinstance(recipe, str) or recipe not in RECIPES:
        raise ValueError('unknown fixed dsLasso recovery recipe')
    config = DslassoConfig(max_iter=16000 if recipe in ('budget_only', 'budget_and_step') else 4000)
    policy = RecoveryPolicy(max_step=64.0 if recipe in ('step_only', 'budget_and_step') else 1.0)
    return dict(family='dslasso_survival_nls', optimizer=asdict(config), recovery_policy=asdict(policy))


def run_config(task, parameters, recipe=None):
    p = validate_parameters(task, parameters)
    if task == DIAGNOSTIC_TASK:
        if recipe is not None and recipe != p['recipe']:
            raise ValueError('diagnostic recipe disagrees with fixed parameters')
        recipe = p['recipe']
        scenario = CASES[p['case']]['scenario']
    else:
        if recipe not in RECIPES or recipe == 'baseline':
            raise ValueError('formal recovery requires a gate-selected candidate recipe')
        scenario = p['scenario']
    result = protocol.run_config(protocol.FORMAL_TASK,
        dict(cell='dslasso', scenario=scenario, shard_index=0, shard_count=16))
    result.update(protocol=PROTOCOL, task=task, run_id=run_id(task, p),
        method=method_config(recipe), recovery_recipe=recipe,
        original_closeout=deepcopy(ORIGINAL_CLOSEOUT), case_inventory=deepcopy(CASE_INVENTORY))
    if task == DIAGNOSTIC_TASK:
        result['seeds'] = sorted(case['seed'] for case in CASES.values() if case['scenario'] == scenario)
        result['timing']['stages'] = ['data_preparation', 'fit']
        result['diagnostic_evaluation_performed'] = False
        result['recipe_selection_scope'] = 'numerical_convergence_and_cost_only'
    return result


def validate_seed(config, seed):
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError('seed is outside the fixed survival recovery run')
    return seed


