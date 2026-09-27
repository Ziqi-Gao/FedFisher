"""Recovery execution identities with the original numerical recipes unchanged."""

from copy import deepcopy

from . import protocol as original

PROTOCOL = 'model-signal-recovery-v2'
PILOT_TASK = 'effect-modifier-model-signal-recovery-pilot'
FORMAL_TASK = 'effect-modifier-model-signal-recovery-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
CELLS = deepcopy(original.CELLS)
SETTINGS = deepcopy(original.SETTINGS)
VARIANTS = original.VARIANTS
SPLITS = original.SPLITS
CASES = ('baseline', 'cap_diagnostic')
DIAGNOSTIC_VARIANTS = ('site_gaussian_shift', 'site_gaussian_local_permutation')
DIAGNOSTIC_SEEDS = (9, 25, 41)
SOURCE_CLOSEOUT = {
    'path': '/data/del6500/FedFisher/effect_modifier_experiment/reports/model-signal-ablation-v1/'
            'recovery-v2-source-closeout-20260927T174642Z/closeout.json',
    'sha256': 'bafded5e626fc350c2cf69031e257c00073e12ec32437d9834d70f4ae39a7e42',
}


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown model/signal recovery task or invalid parameters')
    required = {'setting', 'cell', 'variant', 'shard_index', 'shard_count'}
    if task == PILOT_TASK:
        required.add('case')
    if set(parameters) != required:
        raise ValueError('recovery parameters do not match the closed pilot/formal contract')
    p = dict(parameters)
    base = {key: value for key, value in p.items() if key != 'case'}
    original.validate_parameters(original.PILOT_TASK if task == PILOT_TASK else original.FORMAL_TASK, base)
    if task == FORMAL_TASK:
        if p['shard_count'] != 16:
            raise ValueError('recovery formal work requires exactly 16 deterministic shards')
    else:
        if not isinstance(p['case'], str) or p['case'] not in CASES:
            raise ValueError('unknown recovery pilot case')
        if p['case'] == 'cap_diagnostic' and (
                p['setting'] != 'small_mlp' or p['cell'] != 'fedavg_small_e1_patience'
                or p['variant'] not in DIAGNOSTIC_VARIANTS):
            raise ValueError('cap diagnostic is restricted to the two observed Gaussian failures')
    return p


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    if task == PILOT_TASK:
        return DIAGNOSTIC_SEEDS if p['case'] == 'cap_diagnostic' else (0,)
    return tuple(range(p['shard_index'], 300, 16))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    suffix = '300seeds' if task == FORMAL_TASK else (
        'diagnostic' if p['case'] == 'cap_diagnostic' else 'pilot')
    return 'model_signal_%s_%s_%s_v2_%s' % (p['setting'], p['cell'], p['variant'], suffix)


def method_config(cell):
    return original.method_config(cell)


def run_config(task, parameters):
    p = validate_parameters(task, parameters)
    base = {key: value for key, value in p.items() if key != 'case'}
    config = deepcopy(original.run_config(
        original.PILOT_TASK if task == PILOT_TASK else original.FORMAL_TASK, base))
    config.update(protocol=PROTOCOL, task=task, run_id=run_id(task, p))
    if task == PILOT_TASK:
        config['case'] = p['case']
        if p['case'] == 'cap_diagnostic':
            config['seeds'] = [DIAGNOSTIC_SEEDS[0], DIAGNOSTIC_SEEDS[-1]]
    return config
