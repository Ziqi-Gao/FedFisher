"""Saved-evidence arithmetic validation for the Fisher diagnostic replay.

This validator performs no inference, training, data generation or optimizer
steps. It checks recorded arithmetic and source bindings; it cannot establish
that saved activation masks or raw predictions were honestly measured.
"""
import math

import numpy as np

from ..evaluation import score_agreement, summarize_linearization
from ..fisher_models import array_hash, model_vector
from ..wire import unpack_payload
from ..fedfisher_em import TRAINING_METHOD
from .fisher import _combine_paths, _comparison
from .fisher_metrics import sigmoid


def _close(actual, expected, tolerance):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('diagnostic object keys differ')
        for key in expected:
            _close(actual[key], expected[key], tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('diagnostic sequence dimensions differ')
        for left, right in zip(actual, expected):
            _close(left, right, tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('diagnostic categorical identity differs')
    elif not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=tolerance['rtol'], abs_tol=tolerance['atol']):
        raise ValueError('diagnostic numerical arithmetic differs')


def _finite(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite diagnostic value')


def _gradient_record(record, config, site, theta, local_theta):
    n = config['sample_rows_per_site']
    site_id = site['site_id']
    if record['site_id'] != site_id:
        raise ValueError('gradient site order changed')
    sample_seed = [config['seed'], config['sample_seed_stream'], site_id]
    indices = np.sort(np.random.default_rng(np.random.SeedSequence(sample_seed)).permutation(site['counts']['holdout'])[:n])
    direction_seed = [config['seed'], config['direction_seed_stream'], site_id]
    random = np.random.default_rng(np.random.SeedSequence(direction_seed)).normal(size=theta.size)
    if (record['sample_seed_sequence'] != sample_seed or record['sample_indices'] != indices.tolist()
            or record['direction_seed_sequence'] != direction_seed or len(record['selected_row_ids']) != n
            or len(set(record['selected_row_ids'])) != n
            or record['normalized_random_direction_sha256'] != array_hash(random / np.linalg.norm(random))
            or record['selected_row_ids_sha256'] != array_hash(np.asarray(record['selected_row_ids'], dtype=np.int64))):
        raise ValueError('gradient sample/direction seed evidence differs')
    if record['columns'] != ['control_binary_log_odds', 'treated_binary_log_odds', 'probability_CATE']:
        raise ValueError('finite-difference columns differ')
    expected_pairs = [(direction, step) for direction in ('actual_aggregation', 'fixed_random')
                      for step in config['fd_steps']]
    if [(item['direction'], item['step']) for item in record['checks']] != expected_pairs:
        raise ValueError('finite-difference directions or step grid differ')
    stable_count, failures, boundary, prior_derivatives = 0, 0, 0, {}
    for item in record['checks']:
        actual, numerical = np.asarray(item['autograd']), np.asarray(item['finite_difference'])
        stable, within = np.asarray(item['stable_activation_mask']), np.asarray(item['within_tolerance'])
        if (any(value.shape != (n, 3) for value in (actual, numerical, stable, within)) or
                stable.dtype.kind != 'b' or within.dtype.kind != 'b'):
            raise ValueError('finite-difference observation/mask dimensions differ')
        if not np.array_equal(stable[:, 2], stable[:, :2].all(axis=1)):
            raise ValueError('CATE stability must require both counterfactual arms')
        if item['direction'] in prior_derivatives and not np.array_equal(actual, prior_derivatives[item['direction']]):
            raise ValueError('autograd must be cached across finite-difference step sizes')
        prior_derivatives[item['direction']] = actual
        absolute = np.abs(actual - numerical)
        threshold = config['fd_atol'] + config['fd_rtol'] * np.abs(actual)
        expected_within = absolute <= threshold
        for key, expected in [('absolute_error', absolute), ('tolerance', threshold),
            ('relative_error_floor_1e_12', absolute / np.maximum(1e-12, np.abs(actual)))]:
            _close(item[key], expected.tolist(), config['replay_tolerance'])
        if not np.array_equal(within, expected_within):
            raise ValueError('finite-difference pass mask disagrees with tolerances')
        norm = float(np.linalg.norm(theta - local_theta)) if item['direction'] == 'actual_aggregation' else float(np.linalg.norm(random))
        _close(item['direction_original_norm'], norm, config['replay_tolerance'])
        if item['degenerate_zero_direction'] is not (norm == 0):
            raise ValueError('zero direction classification differs')
        if any(np.asarray(value).shape != (2,) or np.any(np.asarray(value) < 0) or np.any(np.asarray(value) > 1)
               for value in item['layer_boundary_fraction_control_treated'].values()):
            raise ValueError('invalid ReLU boundary proportions')
        stable_count += int(stable.sum())
        failures += int((stable & ~within).sum())
        boundary += int((~stable).sum())
    baseline, t0_outputs = np.asarray(record['baseline_autograd_outputs']), np.asarray(record['t0_model_outputs'])
    if baseline.shape != (n, 3) or t0_outputs.shape != (n, 3):
        raise ValueError('t0 predictions must align with diagnostic rows')
    t0_error = float(np.max(np.abs(baseline - t0_outputs)))
    _close(record['t0_maximum_absolute_error'], t0_error, config['replay_tolerance'])
    p = sigmoid(baseline[:, :2])
    chain = np.column_stack([p[:, 1] * (1 - p[:, 1]) * prior_derivatives[name][:, 1] -
                            p[:, 0] * (1 - p[:, 0]) * prior_derivatives[name][:, 0]
                            for name in ('actual_aggregation', 'fixed_random')])
    _close(record['sigmoid_chain_derivative'], chain.tolist(), config['replay_tolerance'])
    tau_derivative = np.column_stack([prior_derivatives[name][:, 2] for name in ('actual_aggregation', 'fixed_random')])
    if chain.shape != (n, 2):
        raise ValueError('chain-rule derivative dimensions differ')
    chain_error = float(np.max(np.abs(chain - tau_derivative)))
    _close(record['chain_rule_maximum_absolute_error'], chain_error, config['replay_tolerance'])
    if record['chain_rule_tolerance'] != 1e-12 or record['chain_rule_passed'] is not (chain_error <= 1e-12):
        raise ValueError('sigmoid chain-rule consistency decision differs')
    expected = dict(stable_checks=stable_count, stable_failures=failures, boundary_checks=boundary,
        t0_passed=t0_error <= 1e-12,
        passed=t0_error <= 1e-12 and chain_error <= 1e-12 and failures == 0 and stable_count > 0)
    if record['t0_tolerance'] != 1e-12 or record['t0_maximum_absolute_error'] < 0:
        raise ValueError('invalid t0 check')
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError('gradient gate counts or decision differ')


def _path(record, config, sites, weights, dimension):
    if record['t_grid'] != config['t_grid'] or len(record['sites']) != len(sites):
        raise ValueError('parameter path task coverage differs')
    _close(record['target_weights'], weights.tolist(), config['replay_tolerance'])
    for item, site in zip(record['sites'], sites):
        if (item['site_id'] != site['site_id'] or item['sampled_original_rows'] != config['sample_rows_per_site']
                or item['derivative_passes'] != 1 or item['derivative_reused_for_all_t'] is not True
                or [point['t'] for point in item['points']] != config['t_grid']):
            raise ValueError('parameter path site/row/grid/derivative accounting differs')
        expected_rows = config['sample_rows_per_site'] * (1 + dimension * 5)
        if item['original_and_intervened_rows'] != expected_rows or item['derivative_examples'] != expected_rows:
            raise ValueError('parameter path must use all five original permutations')
        for point in item['points']:
            scores = point['score_vectors']
            if set(scores) != {'exact', 'direct_tau', 'sigmoid_of_linear_logits'}:
                raise ValueError('parameter path approximation scope changed')
            for name in scores:
                if len(scores[name]) != dimension or np.any(np.asarray(scores[name]) < 0):
                    raise ValueError('invalid half-squared path scores')
                if name != 'exact':
                    _close(point['score_agreement'][name], score_agreement(scores['exact'], scores[name]), config['replay_tolerance'])
            for scope in ('prediction_errors', 'original_rows_prediction_errors'):
                if set(point[scope]) != {'control_logit', 'treated_logit', 'direct_tau', 'sigmoid_of_linear_logits'}:
                    raise ValueError('prediction error scope differs')
                for moments in point[scope].values():
                    _close(moments['rmse'] ** 2, moments['mean_squared_error'], config['replay_tolerance'])
                    _close(moments['centered_rmse'] ** 2, max(0., moments['mean_squared_error'] - moments['mean_error'] ** 2),
                           config['replay_tolerance'])
                    if moments['maximum_absolute_error'] < 0 or moments['mean_squared_error'] < 0:
                        raise ValueError('negative error magnitude')
    _close(record['target_weighted_points'], _combine_paths(record['sites'], weights), config['replay_tolerance'])


def _endpoint(record, config, source, weights):
    original = source['result']['diagnostics']
    local = record['local_diagnostics']
    if len(local) != len(original['local_linearization']):
        raise ValueError('full-H site coverage differs')
    dimension = source['config']['dgp']['covariate_dim']
    mask = np.arange(dimension) < source['config']['dgp'].get('signal_dim', 2)
    expected_summary = summarize_linearization(local, weights, mask, source['config']['linearity_gate'])
    _close(record['linearization'], expected_summary, config['replay_tolerance'])
    comparisons = []
    for index, (actual, previous) in enumerate(zip(local, original['local_linearization'])):
        for name in ('exact', 'full_gradient'):
            comparisons.append(_comparison('site_%d.scores.%s' % (index, name), actual['scores'][name],
                                            previous['scores'][name], config['replay_tolerance']))
        for key in ('theta_euclidean_distance', 'theta_Fisher_distance',
                    'exact_tau_mean_square_over_original_and_all_interventions'):
            comparisons.append(_comparison('site_%d.%s' % (index, key), actual[key], previous[key], config['replay_tolerance']))
        key = 'tau_rmse_over_original_and_all_interventions'
        comparisons.append(_comparison('site_%d.%s' % (index, key), actual[key]['full_gradient'],
                                        previous[key]['full_gradient'], config['replay_tolerance']))
    for key in ('relative_tau_rmse', 'relative_score_linf'):
        comparisons.append(_comparison('gate.' + key, expected_summary['gate'][key],
                                        original['linearization']['gate'][key], config['replay_tolerance']))
    _close(record['replay_comparisons'], comparisons, config['replay_tolerance'])
    matches = all(item['matches'] for item in comparisons) and expected_summary['gate']['compression_eligible'] == original['linearization']['gate']['compression_eligible']
    if record['replay_matches_source'] is not matches:
        raise ValueError('full-H source reproduction decision differs')


def validate_fisher_diagnostic(report, config, sources):
    """Reject altered gates or inconsistent saved arithmetic without inference."""
    try:
        _finite(report)
        source = sources['aggregation']
        sites = source['result']['data_fingerprints']['sites']
        weights = np.asarray(source['result']['data_fingerprints']['target_weights'])
        theta = model_vector(source['models']['primary'].model)
        if len(report['gradient_checks']) != len(sites):
            raise ValueError('gradient site coverage differs')
        for record, site, blob in zip(report['gradient_checks'], sites, source['payloads']):
            arrays, _ = unpack_payload(blob, TRAINING_METHOD)
            _gradient_record(record, config, site, theta, arrays['theta'])
        gradients = report['gradient_checks']
        expected_gate = dict(passed=all(item['passed'] for item in gradients),
            t0_passed=all(item['t0_passed'] for item in gradients),
            stable_checks=sum(item['stable_checks'] for item in gradients),
            stable_failures=sum(item['stable_failures'] for item in gradients),
            boundary_checks=sum(item['boundary_checks'] for item in gradients),
            reason='all_stable_activation_finite_differences_and_t0_must_pass; boundary_rows_are_diagnostic')
        _close(report['implementation_gate'], expected_gate, config['replay_tolerance'])
        if expected_gate['passed']:
            _path(report['parameter_path'], config, sites, weights, source['config']['dgp']['covariate_dim'])
            _endpoint(report['full_H_endpoint'], config, sources['linearity'], weights)
            quality = report['model_quality']
            if report['full_H_endpoint']['replay_matches_source']:
                expected_names = {'constant_zero_CATE', 'target_weighted_prediction_average', 'parameter_FedAvg',
                                  'strict_FedFisher', 'saved_legacy_validation_selected', 'saved_strict_legacy_uploaded_objective'}
                expected_names |= {'local_site_%03d' % site['site_id'] for site in sites}
                if set(quality['test']) != expected_names:
                    raise ValueError('model-quality checkpoint coverage differs')
                scm = source['config']['dataset_definition']['family'] == 'scmv2'
                zero = quality['test']['constant_zero_CATE']
                if any(zero[key] != 0 for key in ('predicted_mean', 'estimated_heterogeneity', 'estimated_standard_deviation')):
                    raise ValueError('constant-zero CATE must have zero mean and heterogeneity')
                if scm:
                    _close(zero['cate_rmse'] ** 2, zero['true_mean'] ** 2 + zero['true_heterogeneity'], config['replay_tolerance'])
                local_means = [quality['test']['local_site_%03d' % site['site_id']]['predicted_mean'] for site in sites]
                _close(quality['test']['target_weighted_prediction_average']['predicted_mean'],
                       float(weights @ local_means), config['replay_tolerance'])
                for value in quality['test'].values():
                    if value['true_cate_available'] is not scm:
                        raise ValueError('CATE truth availability differs')
                    fields = ('cate_rmse', 'prediction_mean_bias', 'true_mean', 'true_heterogeneity',
                              'true_standard_deviation', 'centered_cate_rmse')
                    if any((value[key] is None) == scm for key in fields):
                        raise ValueError('missing SCM truth or fabricated legacy CATE truth')
                    if value['estimated_heterogeneity'] < 0 or (scm and value['cate_rmse'] < 0):
                        raise ValueError('negative CATE error/variance')
                    _close(value['estimated_standard_deviation'] ** 2, value['estimated_heterogeneity'], config['replay_tolerance'])
                    if scm:
                        _close(value['true_mean'], zero['true_mean'], config['replay_tolerance'])
                        _close(value['true_heterogeneity'], zero['true_heterogeneity'], config['replay_tolerance'])
                        _close(value['true_standard_deviation'] ** 2, value['true_heterogeneity'], config['replay_tolerance'])
                        _close(value['prediction_mean_bias'], value['predicted_mean'] - value['true_mean'], config['replay_tolerance'])
                        _close(value['cate_rmse'] ** 2, value['centered_cate_rmse'] ** 2 + value['prediction_mean_bias'] ** 2,
                               config['replay_tolerance'])
                if len(quality['local_outcome_metrics']) != len(sites):
                    raise ValueError('local outcome coverage differs')
                for record, site in zip(quality['local_outcome_metrics'], sites):
                    for short, key in (('T', 'train'), ('V', 'validation'), ('H', 'holdout')):
                        value = record[short]
                        if (value['rows'] != site['counts'][key] or value['cross_entropy'] < 0
                                or not 0 <= value['brier_score'] <= 1):
                            raise ValueError('invalid local observed-outcome metric')
                source_names = dict(primary='strict_FedFisher', matched_fedavg='parameter_FedAvg',
                    legacy_validation_selected='saved_legacy_validation_selected',
                    strict_legacy_uploaded_objective='saved_strict_legacy_uploaded_objective')
                for original_name, replay_name in source_names.items():
                    saved = source['result']['models'][original_name]['test']
                    if 'estimated_heterogeneity' in saved:
                        _close(quality['test'][replay_name]['estimated_heterogeneity'],
                               saved['estimated_heterogeneity'], config['replay_tolerance'])
                if ([item['model'] for item in quality['source_test_rmse_comparisons']] != (list(source_names) if scm else [])):
                    raise ValueError('saved source test comparison coverage differs')
                for item in quality['source_test_rmse_comparisons']:
                    expected = source['result']['models'][item['model']]['test']['cate_rmse']
                    _close(item['previous_cate_rmse'], expected, config['replay_tolerance'])
                    _close(item['replay_cate_rmse'], expected, config['replay_tolerance'])
                    _close(item['replay_cate_rmse'], quality['test'][source_names[item['model']]]['cate_rmse'], config['replay_tolerance'])
            elif quality is not None:
                raise ValueError('model interpretation must stop after endpoint reproduction failure')
        elif any(report[key] is not None for key in ('parameter_path', 'full_H_endpoint', 'model_quality')):
            raise ValueError('downstream diagnostics continued after failed implementation gate')
        if report['downstream_training_allowed_by_this_replay'] is not False:
            raise ValueError('replay cannot authorize follow-on training')
        expected_cost = dict(training=False, Fisher_recomputed=False, optimizer_updates=0,
            legacy_selection_replayed=False, model_inference=True, local_derivative_evaluations=True,
            training_communication_rounds=0, new_deployable_scoring_protocol=False)
        _close(report['cost_scope'], expected_cost, config['replay_tolerance'])
        for key in ('wall_seconds', 'cpu_seconds'):
            if report[key] < 0:
                raise ValueError('negative replay timing')
        for timing in report['phase_timing'].values():
            if set(timing) != {'wall_seconds', 'cpu_seconds'} or min(timing.values()) < 0:
                raise ValueError('invalid diagnostic phase timing')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('malformed Fisher diagnostic evidence') from error
