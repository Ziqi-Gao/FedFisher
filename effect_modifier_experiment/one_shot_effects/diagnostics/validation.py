"""Validate completed replay evidence without data generation or inference."""
import math
import re

from ..wire import canonical
from . import protocol
from .sources import load_sources


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_result(result, config, identity):
    canonical(result)
    _require(config == protocol.run_config(config['task'], dict(recipe=config['recipe'], dataset=config['dataset'])),
             'diagnostic configuration is not the reviewed fixed recipe')
    _require(result.get('status') == 'complete' and result.get('identity') == identity,
             'diagnostic result status/identity mismatch')
    _require(result.get('fitting_performed') is False and result.get('expansion_authorized') is False,
             'replay cannot fit or authorize expansion')
    _require(result['source_bindings'] == {key: config[key] for key in
        ('sources', 'accepted_report', 'original_source_provenance', 'original_generation_source_files')},
        'replay original-source bindings differ')
    sources = load_sources(config)
    reference = next(iter(sources.values()))['result']
    data = result['data_replay']
    _require(data['passed'] is True and data['data_fingerprints'] == reference['data_fingerprints'],
             'replay data identity differs from original')
    primary = reference['models']['primary']
    _require(data['holdout_permutation_sha256'] == primary['holdout']['permutation_sha256']
             and data['test_permutation_sha256'] == primary['test']['permutation_sha256']
             and data['test_raw_x_sha256'] == primary['test']['raw_x_sha256'],
             'replay permutation/test evidence differs from original')
    evaluator = data['evaluator_reference']
    if config['dataset'] == 'scmv2_base':
        hashes = evaluator['truth_array_sha256']
        _require(set(hashes) == {'e','mu_0','mu_1','tau','modifier_mask'}
                 and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v)
                         for v in [evaluator['site_ids_sha256'],*hashes.values()]), 'missing evaluator replay hashes')
        dimension = next(iter(sources.values()))['config']['dgp']['covariate_dim']
        expected_roles = ['modifier','modifier','prognostic_and_confounder','prognostic','treatment_only']+['noise']*(dimension-5)
        _require(evaluator['variable_roles'] == expected_roles and evaluator['scope'] ==
            'new_replay_hashes_derived_from_pinned_original_generation_code_not_archived_old_pilot_hashes',
            'evaluator reference scope or variable roles differ')
    else:
        _require(evaluator == dict(scope='legacy_has_no_CATE_truth_or_test_site_assignment',
            site_ids_sha256=None,truth_array_sha256=None,variable_roles=None), 'legacy evaluator truth fabricated')
    if config['recipe'] == 'dr_path':
        from .dr import validate_dr_diagnostic
        validate_dr_diagnostic(result['diagnostic'], config, sources)
    else:
        from .fisher_validation import validate_fisher_diagnostic
        validate_fisher_diagnostic(result['diagnostic'], config, sources)
    timing = result['timing']
    _require(timing['method'] == config['recipe'] and timing['actual_network_measured'] is False,
             'replay timing identity/scope mismatch')
    stages = timing['stages']
    _require(set(stages) == {'data_preparation', 'evaluation', 'train', 'tuning', 'aggregation'},
             'replay timing stage inventory mismatch')
    for name, stage in stages.items():
        _require(type(stage['calls']) is int and stage['calls'] >= 0
                 and stage['failed_calls'] == 0 and stage['measured'] == (stage['calls'] > 0),
                 'invalid replay timing calls')
        for key in ('wall_seconds', 'cpu_seconds'):
            _require(type(stage[key]) in (int, float) and math.isfinite(stage[key]) and stage[key] >= 0,
                     'invalid replay duration')
        if name in ('train', 'tuning', 'aggregation'):
            _require(stage['calls'] == 0 and stage['wall_seconds'] == stage['cpu_seconds'] == 0,
                     'replay must not refit, tune or rerun aggregation')
        else:
            _require(stage['measured'], 'required replay stage unmeasured')
    for key in ('wall_seconds', 'cpu_seconds'):
        _require(math.isclose(timing['measured_stage_sum'][key], sum(v[key] for v in stages.values()),
                              rel_tol=1e-12, abs_tol=1e-12), 'replay timing sum mismatch')
