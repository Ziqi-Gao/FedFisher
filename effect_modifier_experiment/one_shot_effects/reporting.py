"""Presentation-only comparison of previously validated completed records.

This module never opens model arrays, predicts, fits, generates observations,
or changes scientific artifacts. The caller must validate completion markers
and artifact hashes before supplying records; the checks here establish that
the three compression results constitute the declared paired comparison.
"""

from copy import deepcopy
import hashlib
import math

from . import protocol
from .wire import canonical


COMPRESSIONS = ('fisher_gradient', 'euclidean_gradient', 'random_orthogonal')
STAGES = ('data_preparation', 'train', 'tuning', 'aggregation', 'evaluation')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, name + ' must be an integer')
    return value


def _finite(value, name):
    _require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
             name + ' must be finite and nonnegative')
    return value


def _sha(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _scientific_config(config):
    """Remove only the identities that necessarily encode direction strategy."""
    _require(config['protocol'] == protocol.PROTOCOL, 'wrong scientific protocol')
    parameters = protocol.validate_parameters(config['task'],
        dict(recipe=config['recipe'], dataset=config['dataset']))
    _require(config['run_id'] == protocol.run_id(config['task'], parameters),
             'recipe and run identity disagree')
    definition = config['recipe_definition']
    _require(definition == protocol.RECIPES[config['recipe']]
             and definition['family'] == 'compression', 'compression recipe required')
    _require(definition['compression'] in COMPRESSIONS
             and config['fisher']['compression'] == definition['compression']
             and config['fisher']['rank'] == definition['rank'],
             'compression/rank configuration binding differs')
    normalized = deepcopy(config)
    del normalized['recipe'], normalized['run_id']
    del normalized['recipe_definition']['compression'], normalized['fisher']['compression']
    return normalized


def _checked_record(record):
    config, result = record['config'], record['result']
    scientific = _scientific_config(config)
    _require(result['status'] == 'complete', 'only complete result records may be compared')
    identity = result['identity']
    _require(all(identity[key] == config[key] for key in ('run_id', 'seed', 'recipe', 'dataset'))
             and identity['config_sha256'] == _sha(config), 'result/config identity mismatch')
    provenance = identity['provenance_sha256']
    _require(isinstance(provenance, str) and len(provenance) == 64
             and all(c in '0123456789abcdef' for c in provenance), 'invalid source provenance')
    evidence = result['data_fingerprints']
    sites, weights = evidence['sites'], evidence['target_weights']
    _require(bool(sites) and len(sites) == len(weights), 'site and target weights disagree')
    counts = [_count(site['n_total'], 'n_total', 1) for site in sites]
    _require([site['site_id'] for site in sites] == list(range(len(sites))),
             'ordered site identities required')
    _require(weights == [n / sum(counts) for n in counts],
             'target weights must use original site n_total exactly once')
    primary = result['models']['primary']
    holdout = primary['holdout']
    _require(holdout['target_weights'] == weights
             and primary['fit']['site_weights'] == weights, 'fit/evaluation target weights differ')
    _require(holdout['rows'] == [site['counts']['holdout'] for site in sites],
             'holdout rows differ from shared partitions')
    _require(len(holdout['permutation_sha256']) == len(sites), 'holdout plans missing')
    communication = result['communication']
    native = _count(communication['native_upload_bytes'], 'native upload bytes', 1)
    padding = _count(communication['padding_bytes'], 'padding bytes')
    wire = _count(communication['upload_bytes'], 'wire upload bytes', 1)
    budget = _count(communication['wire_budget_per_site'], 'wire budget per site', 1)
    _require(budget == protocol.compression_wire_budget(config),
             'actual budget differs from predeclared public-shape budget')
    _require(native + padding == wire == len(sites) * budget
             and communication['per_site_upload_bytes'] == [budget] * len(sites)
             and communication['upload_messages'] == len(sites)
             and communication['upload_rounds'] == 1, 'actual wire byte accounting differs')
    _require(communication['network_time_measured'] is False,
             'local simulation cannot report measured network latency')
    stages = result['timing']['stages']
    timing = {}
    for stage in STAGES:
        _require(stages[stage]['failed_calls'] == 0, 'failed timing stage in complete record')
        timing[stage] = {key: _finite(stages[stage][key], stage + ' ' + key)
                         for key in ('wall_seconds', 'cpu_seconds')}
    for scope, names in (
        ('method_fit', ('train', 'tuning', 'aggregation')),
        ('method_including_evaluation', ('train', 'tuning', 'aggregation', 'evaluation')),
    ):
        timing[scope] = {key: sum(timing[name][key] for name in names)
                         for key in ('wall_seconds', 'cpu_seconds')}
    linear = result['diagnostics']['linearization']
    _require('low_rank' in linear['score_agreement'] and 'low_rank' in linear['recovery'],
             'low-rank score diagnostics missing')
    row = dict(method=config['recipe_definition']['compression'], recipe=config['recipe'],
        run_id=config['run_id'], rank=config['recipe_definition']['rank'],
        native_upload_bytes=native, padding_bytes=padding, wire_upload_bytes=wire,
        wire_budget_per_site=budget,
        protocol_distribution_bytes=communication['protocol_distribution_bytes'],
        final_model_distribution_bytes=communication['final_model_distribution_bytes'],
        score_errors=deepcopy(holdout['summary_comparison']),
        tau_rmse=deepcopy(linear['tau_rmse']),
        recovery=dict(external_test=deepcopy(primary['test']['recovery']),
                      exact_holdout=deepcopy(holdout['recovery']),
                      summary_holdout=deepcopy(linear['recovery']['low_rank'])),
        cate_rmse=primary['test']['cate_rmse'], timing=timing)
    pairing = dict(scientific_config=scientific, provenance_sha256=provenance,
        data_fingerprints=evidence, model_sha256=primary['model_sha256'],
        model_kind=primary['kind'], model_metadata=primary['metadata'],
        exact_holdout=holdout['exact_squared_scores'],
        holdout_permutations=holdout['permutation_sha256'],
        external_test=primary['test'], wire_budget_per_site=budget)
    return row, pairing


def compare_compression_records(records):
    """Return one same-data/model/rank/byte-budget table for all three strategies.

    Input items are ``dict(config=..., result=...)`` loaded from already
    validated artifacts. This seed-zero pilot table does not manufacture
    standard errors or treat methods/sites as independent replications.
    """
    try:
        _require(isinstance(records, (list, tuple)) and len(records) == 3,
                 'exactly three completed compression records are required')
        checked = [_checked_record(record) for record in records]
        by_method = {row['method']: row for row, _ in checked}
        _require(set(by_method) == set(COMPRESSIONS),
                 'each compression strategy must occur exactly once')
        reference = checked[0][1]
        _require(all(canonical(pairing) == canonical(reference) for _, pairing in checked[1:]),
                 'compression comparison changes scientific settings, data, model, source, or byte budget')
        config = reference['scientific_config']
        return dict(protocol=protocol.PROTOCOL, dataset=config['dataset'], seed=config['seed'],
            model=config['recipe_definition']['model'], rank=config['recipe_definition']['rank'],
            target_weights=deepcopy(reference['data_fingerprints']['target_weights']),
            scientific_comparison_sha256=_sha(reference),
            rows=[by_method[name] for name in COMPRESSIONS],
            comparison='same rank and actual padded upload byte budget; native byte costs remain visible',
            timing_scope='single-process stage observations; data preparation separate; no network latency',
            inference_scope='descriptive seed-zero paired pilot; no standard errors or significance claims')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('malformed completed compression record: ' + str(error)) from error
