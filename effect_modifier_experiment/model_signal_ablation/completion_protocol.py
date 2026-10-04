"""Fixed replacement scope after the archived, incomplete V2 source release.

Only execution identities change. This module deliberately imports no numerical
runner, evidence reader, scheduler client, or publication implementation.
"""
from copy import deepcopy

from . import recovery_protocol as previous


PROTOCOL = 'model-signal-completion-v3'
PILOT_TASK = 'effect-modifier-model-signal-completion-pilot'
FORMAL_TASK = 'effect-modifier-model-signal-completion-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
VARIANTS = previous.VARIANTS
SPLITS = previous.SPLITS
SETTING = 'weak_signal_05'
BUDGET_CELL = 'fedavg_shallow_e1_budget30'
PATIENCE_CELL = 'fedavg_shallow_e1_patience'
CELLS = {cell: deepcopy(previous.CELLS[cell]) for cell in (BUDGET_CELL, PATIENCE_CELL)}
SETTINGS = {SETTING: dict(signal_strength=0.5, cells=tuple(CELLS))}
SOURCE_CLOSEOUT = {
    'path': '/data/del6500/FedFisher/archives/'
            'model-signal-recovery-v2-4ecb6d5-20261002/closeout.json',
    'sha256': '1cc83b232fdfc8fc68c772215d5fa4827a6c37ec40aaf523dcdfb968829a6a46',
}
PILOT_REQUEST_COUNT = 8
PILOT_UNIT_COUNT = 16
FORMAL_REQUEST_COUNT = 24
FORMAL_UNIT_COUNT = 896
RETAINED_TERMINAL_UNIT_COUNT = 171904
RETAINED_COMPLETE_UNIT_COUNT = 171898
RETAINED_CAP_UNIT_COUNT = 6
TARGET_UNIT_COUNT = 172800
_FIELDS = frozenset(('setting', 'cell', 'variant', 'shard_index', 'shard_count'))


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict) or set(parameters) != _FIELDS:
        raise ValueError('completion requires its fixed task and five scientific parameters')
    p = dict(parameters)
    if (p['setting'] != SETTING or not isinstance(p['setting'], str)
            or not isinstance(p['cell'], str) or p['cell'] not in (BUDGET_CELL, PATIENCE_CELL)
            or not isinstance(p['variant'], str) or p['variant'] not in VARIANTS):
        raise ValueError('completion setting, recipe or variant is outside the reviewed scope')
    index, count = p['shard_index'], p['shard_count']
    if type(index) is not int or type(count) is not int:
        raise ValueError('completion shard index and count must be integers')
    if task == PILOT_TASK:
        if index != 0 or count != 1:
            raise ValueError('completion pilot fixes seed zero and one shard')
    elif count != 16 or not (
            p['cell'] == BUDGET_CELL and (
                p['variant'] == 'control' and 0 <= index < 16
                or p['variant'] == 'site_gaussian_shift' and index == 0)
            or p['cell'] == PATIENCE_CELL
            and p['variant'] == 'site_gaussian_local_permutation' and 9 <= index < 16):
        raise ValueError('completion formal work is restricted to the 24 archived failed shards')
    return p


PILOT_PARAMETERS = tuple(dict(setting=SETTING, cell=cell, variant=variant,
                             shard_index=0, shard_count=1)
                         for cell in (BUDGET_CELL, PATIENCE_CELL) for variant in VARIANTS)
FORMAL_PARAMETERS = tuple(
    [dict(setting=SETTING, cell=BUDGET_CELL, variant='control',
          shard_index=index, shard_count=16) for index in range(16)]
    + [dict(setting=SETTING, cell=BUDGET_CELL, variant='site_gaussian_shift',
            shard_index=0, shard_count=16)]
    + [dict(setting=SETTING, cell=PATIENCE_CELL, variant='site_gaussian_local_permutation',
            shard_index=index, shard_count=16) for index in range(9, 16)])


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return (0,) if task == PILOT_TASK else tuple(range(p['shard_index'], 300, 16))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'model_signal_%s_%s_%s_v3_%s' % (
        p['setting'], p['cell'], p['variant'], 'pilot' if task == PILOT_TASK else '300seeds')


def method_config(cell):
    if not isinstance(cell, str) or cell not in (BUDGET_CELL, PATIENCE_CELL):
        raise ValueError('unknown completion method')
    return previous.method_config(cell)


def previous_config(task, parameters):
    """Return the unmodified V2 recipe serving as the numerical reference."""
    p = validate_parameters(task, parameters)
    old_task = previous.PILOT_TASK if task == PILOT_TASK else previous.FORMAL_TASK
    if task == PILOT_TASK:
        p['case'] = 'baseline'
    return previous.run_config(old_task, p)


def run_config(task, parameters):
    config = deepcopy(previous_config(task, parameters))
    config.pop('case', None)
    config.update(protocol=PROTOCOL, task=task, run_id=run_id(task, parameters))
    return config


def parameters_for_config(config):
    """Select a valid representative shard for a shard-independent manifest.

    In particular, the formal combined/patience namespace starts at shard 9;
    inventing shard zero to validate that namespace would reject valid work.
    Callers must still compare the complete fixed run configuration.
    """
    if not isinstance(config, dict) or config.get('task') not in TASKS:
        raise ValueError('unknown completion configuration task')
    p = {key: config.get(key) for key in ('setting', 'cell', 'variant')}
    pilot = config['task'] == PILOT_TASK
    p.update(shard_count=1 if pilot else 16,
             shard_index=9 if not pilot and p['cell'] == PATIENCE_CELL else 0)
    return validate_parameters(config['task'], p)
