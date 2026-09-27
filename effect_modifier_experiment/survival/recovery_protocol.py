"""Standard-library-only fixed task guard; scientific imports stay lazy."""
from copy import deepcopy
from dataclasses import asdict

from . import protocol

PROTOCOL = 'survival-dslasso-recovery-v1'
DIAGNOSTIC_TASK = 'effect-modifier-survival-dslasso-diagnostic'
FORMAL_TASK = 'effect-modifier-survival-dslasso-recovery-shard'
REPAIR_PROTOCOL = 'survival-dslasso-repair-v2'
REPAIR_DIAGNOSTIC_TASK = 'effect-modifier-survival-dslasso-repair-pilot'
REPAIR_FORMAL_TASK = 'effect-modifier-survival-dslasso-repair-shard'
TASKS = (DIAGNOSTIC_TASK, FORMAL_TASK, REPAIR_DIAGNOSTIC_TASK, REPAIR_FORMAL_TASK)
REPAIR_RECIPE = 'budget_and_step'
REPAIR_EVIDENCE_SCHEMA = 'survival_dslasso_numerical_recovery_v2'
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
REPAIR_CLOSEOUT = dict(
    path='/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1/'
         'recovery-r1-closeout-20260927T001801Z/closeout.json',
    sha256='00e4a371797524322b584fc325da14888c11f1219dd1bd9192d7d2983de50f6c')
R1_DIAGNOSTIC_ACCEPTANCE = dict(
    path='/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1/'
         'dslasso-recovery-v1/diagnostic-'
         '4d062bc5018d28696584889e1a1365860bcbc77a06936fac590e6ab4c08f7d1c.acceptance.json',
    sha256='82dc808bbe3f2b634a3705156193a379d0bf175f0821f786b47573f29b582bff')

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
REPAIR_CASES = dict(CASES, **{
    'covariate_shift_%03d' % seed: dict(scenario='covariate_shift', seed=seed,
                                      role='majorization_validation_failure')
    for seed in (49, 63, 240)})


def tasks_for_epoch(epoch):
    if type(epoch) is not int or epoch not in (1, 2):
        raise ValueError('unknown survival recovery epoch')
    return TASKS[:2] if epoch == 1 else TASKS[2:]


def task_epoch(task):
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError('unknown survival recovery task')
    return 1 if task in TASKS[:2] else 2


def is_diagnostic(task):
    return task == tasks_for_epoch(task_epoch(task))[0]


def cases_for_task(task):
    return CASES if task_epoch(task) == 1 else REPAIR_CASES


def config_parameters(config):
    """Recover a representative shard/case without adding execution choices."""
    task = config['task']
    if is_diagnostic(task):
        case = next((name for name, row in cases_for_task(task).items()
                     if row['scenario'] == config['scenario']), None)
        if case is None:
            raise ValueError('unknown survival recovery configuration scenario')
        result = dict(case=case)
        if task_epoch(task) == 1:
            result['recipe'] = config['recovery_recipe']
        return result
    return dict(scenario=config['scenario'], shard_index=0, shard_count=16)


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown survival recovery task or invalid parameters')
    diagnostic = is_diagnostic(task)
    fields = ({'case', 'recipe'} if task_epoch(task) == 1 else {'case'}) if diagnostic else {
        'scenario', 'shard_index', 'shard_count'}
    if set(parameters) != fields:
        raise ValueError('survival recovery parameters do not match the fixed task schema')
    if diagnostic:
        if (not isinstance(parameters['case'], str) or parameters['case'] not in cases_for_task(task)
                or (task_epoch(task) == 1 and (not isinstance(parameters['recipe'], str)
                    or parameters['recipe'] not in RECIPES))):
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
    return ((cases_for_task(task)[p['case']]['seed'],) if is_diagnostic(task)
            else tuple(range(p['shard_index'], 300, p['shard_count'])))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    epoch = task_epoch(task)
    if is_diagnostic(task):
        recipe = p['recipe'] if epoch == 1 else REPAIR_RECIPE
        return 'survival_v1_dslasso_diagnostic_r%d_%s_%s' % (
            epoch, recipe, cases_for_task(task)[p['case']]['scenario'])
    return 'survival_v1_dslasso_recovery_r%d_%s_300seeds' % (epoch, p['scenario'])


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
    epoch, diagnostic = task_epoch(task), is_diagnostic(task)
    if diagnostic:
        chosen = p['recipe'] if epoch == 1 else REPAIR_RECIPE
        if recipe is not None and recipe != chosen:
            raise ValueError('diagnostic recipe disagrees with fixed parameters')
        recipe = chosen
        scenario = cases_for_task(task)[p['case']]['scenario']
    else:
        if recipe not in RECIPES or recipe == 'baseline':
            raise ValueError('formal recovery requires a gate-selected candidate recipe')
        scenario = p['scenario']
    if epoch == 2 and recipe != REPAIR_RECIPE:
        raise ValueError('repair epoch retains the previously selected budget_and_step recipe')
    result = protocol.run_config(protocol.FORMAL_TASK,
        dict(cell='dslasso', scenario=scenario, shard_index=0, shard_count=16))
    result.update(protocol=PROTOCOL if epoch == 1 else REPAIR_PROTOCOL, task=task, run_id=run_id(task, p),
        method=method_config(recipe), recovery_recipe=recipe,
        original_closeout=deepcopy(ORIGINAL_CLOSEOUT), case_inventory=deepcopy(CASE_INVENTORY))
    if epoch == 2:
        result.update(required_evidence_schema=REPAIR_EVIDENCE_SCHEMA,
            repair_closeout=deepcopy(REPAIR_CLOSEOUT),
            r1_diagnostic_acceptance=deepcopy(R1_DIAGNOSTIC_ACCEPTANCE))
    if diagnostic:
        result['seeds'] = sorted(case['seed'] for case in cases_for_task(task).values()
                                 if case['scenario'] == scenario)
        result['timing']['stages'] = ['data_preparation', 'fit']
        result['diagnostic_evaluation_performed'] = False
        result['recipe_selection_scope'] = ('numerical_convergence_and_cost_only' if epoch == 1
                                            else 'fixed_r1_recipe_and_numeric_parity')
    return result


def validate_seed(config, seed):
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError('seed is outside the fixed survival recovery run')
    return seed
