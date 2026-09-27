"""Fixed additive recipes; parameter validation imports no numerical libraries."""

from copy import deepcopy

from ..causal_comparison import protocol as comparison
from ..fedfisher_timing import protocol as fisher

PROTOCOL = 'model-signal-ablation-v1'
PILOT_TASK = 'effect-modifier-model-signal-pilot'
FORMAL_TASK = 'effect-modifier-model-signal-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
VARIANTS = comparison.VARIANTS
SPLITS = comparison.SPLITS
MODELS = dict(small='SyntheticMLPSmall', shallow='SyntheticMLP', deep='SyntheticMLPDeep')

# Recipes are copies of the reviewed originals. Their task/cell allowlists stay
# closed; adding a model to the factory never expands an old experiment.
CELLS = {}
for _depth, _model in MODELS.items():
    for _epochs in (1, 5, 10, 30):
        for _mode in ('patience', 'budget30'):
            _cell = 'fedavg_%s_e%d_%s' % (_depth, _epochs, _mode)
            _base = _cell.replace('_small_', '_shallow_')
            CELLS[_cell] = dict(deepcopy(comparison.CELLS[_base]), model=_model)
    for _kind in ('diag', 'kfac'):
        CELLS['fedfisher_%s_%s' % (_kind, _depth)] = dict(
            family='legacy_fedfisher', algorithm='fedfisher_'+_kind, model=_model)
CELLS['dslassocov'] = deepcopy(comparison.CELLS['dslassocov'])

SETTINGS = {
    'small_mlp': dict(signal_strength=2.0, cells=tuple(
        cell for cell, method in CELLS.items() if method.get('model') == 'SyntheticMLPSmall')),
    'weak_signal_1': dict(signal_strength=1.0, cells=tuple(CELLS)),
    'weak_signal_05': dict(signal_strength=0.5, cells=tuple(CELLS)),
}


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown model/signal task or invalid parameters')
    if set(parameters) != {'setting', 'cell', 'variant', 'shard_index', 'shard_count'}:
        raise ValueError('model/signal parameters require setting/cell/variant/shard_index/shard_count')
    p = parameters
    if (not isinstance(p['setting'], str) or p['setting'] not in SETTINGS
            or not isinstance(p['cell'], str) or p['cell'] not in SETTINGS[p['setting']]['cells']
            or not isinstance(p['variant'], str) or p['variant'] not in VARIANTS):
        raise ValueError('unknown or incompatible model/signal setting, cell or variant')
    index, count = p['shard_index'], p['shard_count']
    allowed = (1,) if task == PILOT_TASK else (8, 16)
    if type(index) is not int or type(count) is not int or count not in allowed or not 0 <= index < count:
        raise ValueError('invalid deterministic model/signal shard')
    return dict(p)


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return tuple(range(p['shard_index'], 1 if task == PILOT_TASK else 300, p['shard_count']))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'model_signal_%s_%s_%s_v1_%s' % (
        p['setting'], p['cell'], p['variant'], 'pilot' if task == PILOT_TASK else '300seeds')


def _original(cell):
    if not isinstance(cell, str) or cell not in CELLS:
        raise ValueError('unknown model/signal cell')
    source = fisher if CELLS[cell]['family'] == 'legacy_fedfisher' else comparison
    base_cell = cell.replace('_small', '_shallow')
    return source, base_cell


def method_config(cell):
    source, base_cell = _original(cell)
    method = deepcopy(source.method_config(base_cell))
    if 'model' in method:
        method['model'] = CELLS[cell]['model']
    return method


def run_config(task, parameters):
    p = validate_parameters(task, parameters)
    source, base_cell = _original(p['cell'])
    base_parameters = dict(cell=base_cell, variant=p['variant'],
                           shard_index=p['shard_index'], shard_count=p['shard_count'])
    source_task = source.PILOT_TASK if task == PILOT_TASK else source.FORMAL_TASK
    config = deepcopy(source.run_config(source_task, base_parameters))
    config.update(protocol=PROTOCOL, task=task, run_id=run_id(task, p),
                  setting=p['setting'], cell=p['cell'], method=method_config(p['cell']))
    config['dgp']['signal_strength'] = SETTINGS[p['setting']]['signal_strength']
    return config
