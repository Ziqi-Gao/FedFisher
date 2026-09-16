"""Frozen scientific recipes for an independently admitted timing cohort."""
from copy import deepcopy

from ..causal_comparison.protocol import DGP, EVALUATION, SPLITS, VARIANTS

PROTOCOL = 'fedfisher-timing-v1'
PILOT_TASK = 'effect-modifier-fedfisher-timing-pilot'
FORMAL_TASK = 'effect-modifier-fedfisher-timing-shard'
TASKS = (PILOT_TASK, FORMAL_TASK)
CELLS = {
    'fedfisher_%s_%s' % (kind, depth): dict(
        family='legacy_fedfisher', algorithm='fedfisher_'+kind, model=model)
    for depth, model in (('shallow', 'SyntheticMLP'), ('deep', 'SyntheticMLPDeep'))
    for kind in ('diag', 'kfac')
}
OPTIMIZER = dict(local_epochs=30, batch_size=64, learning_rate=0.01, momentum=0.9,
                 validation_size=500, server_iterations=2000,
                 server_learning_rate=0.01, validation_interval=100,
                 quantization_bits=15, rng_policy='isolate_fisher_from_training_v1')
TIMING_CONTRACT = dict(
    execution='single_process_single_core_cpu',
    clocks=['perf_counter', 'process_time'],
    stages=['data_preparation', 'train', 'tuning', 'aggregation', 'evaluation'],
    standalone_selected_fisher_only=True,
    fitting_stages=['train', 'tuning', 'aggregation'],
    aggregation_includes_compression_and_validation_selection=True,
    evaluation='causal_comparison_common_probability_and_logit_raw_x_intervention',
    excludes=['queueing', 'imports', 'artifact_io', 'actual_network_transport'],
    historical_gpu_timings_reconstructed=False,
    historical_gpu_trajectory_bitwise_equivalence_claimed=False)


def validate_parameters(task, parameters):
    if task not in TASKS or not isinstance(parameters, dict):
        raise ValueError('unknown Fisher timing task or invalid parameters')
    if set(parameters) != {'cell', 'variant', 'shard_index', 'shard_count'}:
        raise ValueError('Fisher timing parameters require cell/variant/shard_index/shard_count')
    if (not isinstance(parameters['cell'], str) or parameters['cell'] not in CELLS
            or not isinstance(parameters['variant'], str) or parameters['variant'] not in VARIANTS):
        raise ValueError('unknown Fisher timing method or variant')
    index, count = parameters['shard_index'], parameters['shard_count']
    allowed = (1,) if task == PILOT_TASK else (8, 16)
    if type(index) is not int or type(count) is not int or count not in allowed or not 0 <= index < count:
        raise ValueError('invalid deterministic Fisher timing shard')
    return dict(parameters)


def seed_sequence(task, parameters):
    p = validate_parameters(task, parameters)
    return tuple(range(p['shard_index'], 1 if task == PILOT_TASK else 300, p['shard_count']))


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'fedfisher_timing_%s_%s_v1_%s' % (
        p['cell'], p['variant'], 'pilot' if task == PILOT_TASK else '300seeds')


def method_config(cell):
    if not isinstance(cell, str) or cell not in CELLS:
        raise ValueError('unknown Fisher timing cell')
    return dict(deepcopy(CELLS[cell]), optimizer=deepcopy(OPTIMIZER))


def run_config(task, parameters):
    p = validate_parameters(task, parameters)
    return dict(protocol=PROTOCOL, task=task, run_id=run_id(task, p),
        cell=p['cell'], variant=p['variant'], seeds=[0, 0 if task == PILOT_TASK else 299],
        splits=list(SPLITS), dgp=deepcopy(DGP), evaluation=deepcopy(EVALUATION),
        method=method_config(p['cell']), timing_contract=deepcopy(TIMING_CONTRACT),
        validation_contract=dict(source='original_generator_global_training_rows',
                                 size=500, numpy_random_state_seed=3,
                                 criterion='legacy_server_validation_accuracy',
                                 overlaps_local_training=True),
        historical_reference='factorial_90700c2_300seeds; independent new CPU observations')
