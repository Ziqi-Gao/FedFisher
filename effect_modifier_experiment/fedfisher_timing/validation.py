"""Pure semantic checks for saved standalone CPU FedFisher timing results."""

import hashlib
import math
import re

import numpy as np

from models import SYNTHETIC_MLP_HIDDEN_DIMS
from ..infrastructure.artifacts import canonical
from ..causal_comparison.evaluation import model_hash, signal_recovery_with_ties
from ..causal_comparison.measurement import STAGES


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value, name):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
             'invalid timing '+name+' fingerprint')


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             'invalid finite nonnegative timing '+name)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, 'invalid timing '+name+' count')


def _model(arrays, config):
    method, d = config['method'], config['dgp']['covariate_dim']
    _require(method['family'] == 'legacy_fedfisher'
             and method['algorithm'] in ('fedfisher_diag', 'fedfisher_kfac')
             and method['model'] in SYNTHETIC_MLP_HIDDEN_DIMS, 'invalid timing method identity')
    widths = (1+2*d, *SYNTHETIC_MLP_HIDDEN_DIMS[method['model']], 2)
    shapes = {}
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        shapes['net.%d.weight' % (2*index)] = (right, left)
        shapes['net.%d.bias' % (2*index)] = (right,)
    _require(isinstance(arrays, dict) and set(arrays) == set(shapes), 'timing model array inventory mismatch')
    for name, shape in shapes.items():
        value = arrays[name]
        _require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == np.dtype('float32')
                 and np.isfinite(value).all(), 'invalid timing model shape/dtype/values: '+name)


def _fit(result, arrays, identity, config):
    fit, sizes = result['fit'], result['site_sizes']
    method, options = config['method'], config['method']['optimizer']
    kind = method['algorithm'].removeprefix('fedfisher_')
    fixed = dict(family='legacy_fedfisher', algorithm=method['algorithm'], model=method['model'],
        seed=identity['seed'], completion_eligible=True, termination_reason='fixed_recipe_completed',
        convergence_claim=False, device='cpu', rng_policy=options['rng_policy'], standalone_method=True,
        fisher_representation=kind, historical_gpu_bitwise_reproduction=False,
        local_epochs=options['local_epochs'], parameter_count=sum(value.size for value in arrays.values()),
        training_communication_rounds=1, server_iterations=options['server_iterations'],
        validation_checks=options['server_iterations']//options['validation_interval'])
    for name, value in fixed.items():
        _require(type(fit[name]) is type(value) and fit[name] == value,
                 'timing fit recipe/completion differs: '+name)
    _require(fit['model_fingerprint'] == result['fingerprints']['model']
             and fit['training_fingerprint'] == result['fingerprints']['training'],
             'timing fit/model/data fingerprints disagree')
    _hash(fit['initial_model_fingerprint'], 'initial model')
    reports = fit['site_reports']
    _require(isinstance(reports, list) and len(reports) == len(sizes), 'timing local site report count differs')
    for index, (size, site) in enumerate(zip(sizes, reports)):
        expected = dict(site_idx=index, num_examples=size, epochs=options['local_epochs'],
            updates=options['local_epochs']*math.ceil(size/options['batch_size']),
            sample_visits=options['local_epochs']*size)
        for name, value in expected.items():
            _require(type(site[name]) is int and site[name] == value, 'timing local training counts disagree: '+name)
        _number(site['weight'], 'site weight')
        _require(math.isclose(site['weight'], size/sum(sizes), rel_tol=1e-12, abs_tol=1e-12),
                 'timing local aggregation weight differs from sample size')
        for name in ('model_fingerprint', 'fisher_fingerprint', 'training_fingerprint'):
            _hash(site[name], 'site '+name)
    _require(fit['local_model_fingerprints'] == [site['model_fingerprint'] for site in reports],
             'timing local model fingerprints disagree')
    for name, source in (('total_local_updates', 'updates'), ('total_sample_visits', 'sample_visits')):
        _integer(fit[name], name, 1)
        _require(fit[name] == sum(site[source] for site in reports), 'timing total local counts disagree')
    validation = fit['validation']
    _require(validation['scope'] == 'pooled_training_subset_also_used_in_local_fits'
             and validation['independent_test_access'] is False
             and validation['pooled_order'] == 'original_generator_global_row_order'
             and type(validation['selection_seed']) is int and validation['selection_seed'] == 3
             and validation['selection_rule'] == 'highest_validation_accuracy_strict_improvement_first_tie',
             'timing validation selection source/order/rule differs')
    count = min(options['validation_size'], sum(sizes))
    _require(type(validation['num_examples']) is int and validation['num_examples'] == count
             and validation['checked_iteration_indices'] == list(range(0, options['server_iterations'],
                                                                      options['validation_interval'])),
             'timing validation selection counts differ')
    indices = np.random.RandomState(3).choice(sum(sizes), count, replace=False)
    _require(validation['indices_sha256'] == hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest(),
             'timing validation sampling fingerprint differs')
    _hash(validation['dataset_sha256'], 'validation dataset')
    contract = fit['timing_contract']
    _require(contract['training_includes_all_selection_and_refits'] is True
             and contract['fit_stages'] == ['train', 'aggregation']
             and contract['tuning'] == 'included_in_aggregation'
             and contract['refit'] == 'none_in_original_recipe'
             and contract['evaluation'] == 'independent_post_fit_stage_owned_by_experiment',
             'timing fit scope omits fitting/selection or fabricates refits')
    communication = fit['communication']
    _require(communication['execution_model'] == 'single_process_simulation'
             and communication['actual_network_measured'] is False and communication['accounted'] is False
             and type(communication['training_communication_rounds']) is int
             and communication['training_communication_rounds'] == 1
             and communication['reason'] == 'preserved aggregation has no serialized upload boundary; KFAC compression accesses training rows',
             'timing communication scope misreported')


def _timing(timing, config, num_sites):
    _require(timing['schema_version'] == 1 and type(timing['schema_version']) is int
             and timing['method'] == config['cell']
             and timing['execution_model'] == 'single_process_simulation'
             and timing['actual_network_measured'] is False, 'timing method/schema/execution scope mismatch')
    _require(timing['clock'] == dict(wall='perf_counter', cpu='process_time',
        cpu_scope='current_process_only; excludes GPU execution time', explicit_synchronization=False),
        'timing must record the independent CPU clocks')
    _require(timing['scope'] == 'instrumented stages only; not an end-to-end or deployed parallel runtime',
             'timing total scope mismatch')
    stages = timing['stages']
    _require(set(stages) == set(STAGES), 'timing stage inventory mismatch')
    kind = config['method']['algorithm'].removeprefix('fedfisher_')
    bucket_keys = {'measured', 'wall_seconds', 'cpu_seconds', 'calls', 'failed_calls'}
    for name, stage in stages.items():
        _require(set(stage) == bucket_keys | {'components'}, 'timing stage fields mismatch')
        expected_calls = 0 if name == 'tuning' else 1
        _integer(stage['calls'], name)
        _require(stage['calls'] == expected_calls and type(stage['failed_calls']) is int
                 and stage['failed_calls'] == 0 and stage['measured'] is bool(expected_calls),
                 'missing or failed full standalone timing stage: '+name)
        for clock in ('wall_seconds', 'cpu_seconds'):
            _number(stage[clock], clock)
            _require(expected_calls > 0 or stage[clock] == 0, 'unmeasured timing stage has a duration')
        _require(expected_calls == 0 or stage['wall_seconds'] > 0,
                 'required standalone timing stage has no observed wall duration')
        components = stage['components']
        _require(isinstance(components, dict) and set(components) == (
            {'sgd', 'fisher_'+kind} if name == 'train' else set()),
            'timing components omit selected Fisher or include extra work')
        for component in components.values():
            _require(set(component) == bucket_keys and component['measured'] is True
                     and type(component['calls']) is int and component['calls'] == num_sites
                     and type(component['failed_calls']) is int and component['failed_calls'] == 0,
                     'timing component measurement/site count mismatch')
            for clock in ('wall_seconds', 'cpu_seconds'):
                _number(component[clock], clock)
            _require(component['wall_seconds'] > 0, 'selected SGD/Fisher timing has no observed wall duration')
        for clock in ('wall_seconds', 'cpu_seconds'):
            component_sum = sum(component[clock] for component in components.values())
            _require(component_sum <= stage[clock]+max(1e-12, stage[clock]*1e-12),
                     'nested timing components exceed their containing stage')
    sums = timing['measured_stage_sum']
    _require(set(sums) == {'wall_seconds', 'cpu_seconds'}, 'timing stage sum clock inventory mismatch')
    for clock in sums:
        _number(sums[clock], clock)
        _require(math.isclose(sums[clock], sum(stage[clock] for stage in stages.values()),
                              rel_tol=1e-12, abs_tol=1e-12),
                 'timing total must sum disjoint stages without adding nested components')


def validate_result(result, arrays, identity, config):
    """Recompute metadata invariants without generating data or running a model."""
    canonical(result)
    try:
        _require(result['status'] == 'complete' and result['identity'] == identity,
                 'timing scientific identity/status mismatch')
        memory = result['memory']
        _require(set(memory) == {'process_peak_rss_kib', 'scope'}
                 and memory['scope'] == 'Linux current-process lifetime high-water RSS; not cgroup peak or reservation',
                 'timing memory observation scope mismatch')
        _number(memory['process_peak_rss_kib'], 'peak RSS')
        _require(memory['process_peak_rss_kib'] > 0, 'timing peak RSS must be positive')
        _model(arrays, config)
        sizes, dgp = result['site_sizes'], config['dgp']
        _require(isinstance(sizes, list) and len(sizes) == dgp['num_sites']
                 and all(type(size) is int and dgp['site_min_samples'] <= size <= dgp['site_max_samples']
                         for size in sizes), 'invalid timing training site sizes')
        counts = result['label_counts']
        _require(set(counts) == {str(index) for index in range(len(sizes))},
                 'timing label counts have wrong site identities')
        for index, size in enumerate(sizes):
            local = counts[str(index)]
            _require(isinstance(local, dict) and bool(local) and set(local) <= {'0', '1'}
                     and all(type(value) is int and value >= 0 for value in local.values())
                     and sum(local.values()) == size, 'timing label counts disagree with site sizes')
        fingerprints = result['fingerprints']
        _require(set(fingerprints) == {'training', 'test', 'model'}, 'timing fingerprint inventory mismatch')
        for name, value in fingerprints.items():
            _hash(value, name)
        _require(model_hash(arrays) == fingerprints['model'], 'timing model semantic fingerprint mismatch')
        _fit(result, arrays, identity, config)
        evaluation = result['evaluation']
        _require(evaluation['primary'] == 'permute/tau_prob' and evaluation['effect_only'] is False
                 and evaluation['evaluation_network_measured'] is False,
                 'timing evaluation endpoint or scope mismatch')
        metrics = {mode+'/'+scale for mode in config['evaluation']['modes'] for scale in ('tau_prob', 'tau_logit')}
        _require(set(evaluation['metrics']) == metrics, 'timing raw-X evaluation scales/modes differ')
        for name, metric in evaluation['metrics'].items():
            importance = np.asarray(metric['importance'], dtype=np.float64)
            _require(importance.shape == (dgp['covariate_dim'],) and np.isfinite(importance).all()
                     and (importance >= 0).all(), 'invalid timing raw-X importance')
            scale = name.split('/')[1].removeprefix('tau_')
            expected = signal_recovery_with_ties(importance, dgp['signal_dim'], 'tau_intervention_'+scale)
            _require(metric['recovery'] == expected, 'timing recovery disagrees with tie-aware importance ranking')
        for name in ('raw_x_sha256', 'permutation_sha256'):
            _hash(evaluation[name], 'evaluation '+name)
        shard_sizes, shard_hashes = evaluation['shard_sizes'], evaluation['shard_indices_sha256']
        shard_count = dgp['num_sites'] if 'local_permutation' in config['variant'] else 1
        _require(len(shard_sizes) == len(shard_hashes) == shard_count
                 and all(type(size) is int and size > 0 for size in shard_sizes)
                 and sum(shard_sizes) == dgp['num_test'], 'timing evaluation shards do not cover common test set')
        for value in shard_hashes:
            _hash(value, 'evaluation shard')
        _timing(result['timing'], config, len(sizes))
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('malformed timing scientific result') from error
