"""Bounded evaluator replay of immutable FedFisher pilot checkpoints.

This module never generates data, fits parameters, recomputes Fisher, selects
legacy iterates, or owns artifacts. The scheduler runner authenticates source
artifacts and regenerated data/permutation fingerprints before calling it.
"""
from time import perf_counter, process_time

import numpy as np

from .. import fedfisher_em as em
from ..evaluation import score_agreement, summarize_linearization
from ..fisher_models import array_hash, model_from_vector, model_vector, predict_tau
from ..interventions import permutation_hash, validate_permutations
from .fisher_metrics import (at_vector, cate_metrics, combine_moments, error_moments,
    gradient_check, local_derivatives, outcome_metrics, outputs, sigmoid, test_weights)


def selected_rows(site, config):
    count = config['sample_rows_per_site']
    if len(site.holdout) < count:
        raise ValueError('the fixed diagnostic sample exceeds holdout size')
    seed = [config['seed'], config['sample_seed_stream'], site.site_id]
    indices = np.sort(np.random.default_rng(np.random.SeedSequence(seed)).permutation(len(site.holdout))[:count])
    return indices, dict(sample_seed_sequence=seed, sample_indices=indices.tolist(),
        selected_row_ids=site.holdout.row_ids[indices].tolist(),
        selected_row_ids_sha256=array_hash(site.holdout.row_ids[indices]),
        selection='uniform_without_replacement_before_predictions; sorted_for_evaluation_only')


def sampled_interventions(x, plan, indices):
    """Restrict output rows of the original full-H permutation, not its donors."""
    validate_permutations(x, plan)
    indices = np.asarray(indices)
    baseline = np.asarray(x)[indices]
    blocks = [baseline]
    for variable in range(x.shape[1]):
        for repeat in range(plan.shape[1]):
            changed = baseline.copy()
            changed[:, variable] = x[plan[variable, repeat, indices], variable]
            blocks.append(changed)
    return np.concatenate(blocks)


def _scores(values, rows, variables, repeats):
    if values.shape != (rows * (1 + variables * repeats),):
        raise ValueError('predictions do not match the fixed sampled intervention layout')
    changed = values[rows:].reshape(variables, repeats, rows)
    return np.mean(np.square(values[:rows][None, None, :] - changed), axis=(1, 2)) / 2


def _path_point(t, exact, baseline, derivative, rows, variables, repeats, baseline_signs):
    linear_logits = baseline[:, :2] + t * derivative[:, :2, 0]
    direct_tau = baseline[:, 2] + t * derivative[:, 2, 0]
    linked_probability = sigmoid(linear_logits)
    linked_tau = linked_probability[:, 1] - linked_probability[:, 0]
    prediction_values = dict(direct_tau=direct_tau, sigmoid_of_linear_logits=linked_tau)
    errors = dict(control_logit=error_moments(exact['logits'][:, 0], linear_logits[:, 0]),
                  treated_logit=error_moments(exact['logits'][:, 1], linear_logits[:, 1]))
    errors.update({name: error_moments(exact['tau'], value) for name, value in prediction_values.items()})
    original_errors = dict(control_logit=error_moments(exact['logits'][:rows, 0], linear_logits[:rows, 0]),
                          treated_logit=error_moments(exact['logits'][:rows, 1], linear_logits[:rows, 1]))
    original_errors.update({name: error_moments(exact['tau'][:rows], value[:rows])
                            for name, value in prediction_values.items()})
    scores = {'exact': _scores(exact['tau'], rows, variables, repeats)}
    scores.update({name: _scores(value, rows, variables, repeats) for name, value in prediction_values.items()})
    flips = {name: dict(control=float(np.mean(sign[:, 0] != exact['signs'][name][:, 0])),
                        treated=float(np.mean(sign[:, 1] != exact['signs'][name][:, 1])),
                        neurons_per_row=sign.shape[2]) for name, sign in baseline_signs.items()}
    return dict(t=float(t), prediction_errors=errors, original_rows_prediction_errors=original_errors,
        score_vectors={name: value.tolist() for name, value in scores.items()},
        score_agreement={name: score_agreement(scores['exact'], value)
                         for name, value in scores.items() if name != 'exact'},
        relu_flip_from_local=flips,
        probability_saturation_fraction={arm: float(np.mean((exact['probabilities'][:, i] < .01) |
                                                             (exact['probabilities'][:, i] > .99)))
                                         for i, arm in enumerate(('control', 'treated'))})


def parameter_path(local, theta_final, site, plan, indices, config):
    x = sampled_interventions(site.holdout.x, plan, indices)
    delta = theta_final - local.theta
    baseline, derivative = local_derivatives(local.model, x, delta[:, None])
    local_output = outputs(local.model, x, activations=True)
    points = []
    for t in config['t_grid']:
        exact = local_output if t == 0 else outputs(at_vector(local.model, local.theta + t * delta),
                                                   x, activations=True)
        points.append(_path_point(t, exact, baseline, derivative, len(indices), x.shape[1],
                                  plan.shape[1], local_output['signs']))
    return dict(site_id=site.site_id, sampled_original_rows=len(indices),
        original_and_intervened_rows=len(x), input_sha256=array_hash(x),
        full_H_permutation_sha256=permutation_hash(plan),
        perturbation_scope='selected_output_rows_with_donors_from_original_full_H_plan',
        score_definition='half_mean_squared_tau_change_over_fixed_sample_and_five_original_permutations',
        derivative_examples=len(x), derivative_passes=1, derivative_reused_for_all_t=True,
        derivative_definition='binary_log_odds_autograd_with_exact_sigmoid_chain_rule_for_direct_tau',
        theta_euclidean_distance=float(np.linalg.norm(delta)),
        theta_Fisher_distance=float(np.sqrt(local.fisher @ np.square(delta))), points=points)


def _combine_paths(paths, weights):
    combined = []
    for index in range(len(paths[0]['points'])):
        records = [value['points'][index] for value in paths]
        scores = {name: np.asarray(weights) @ np.asarray([value['score_vectors'][name] for value in records])
                  for name in records[0]['score_vectors']}
        result = dict(t=records[0]['t'], score_vectors={key: value.tolist() for key, value in scores.items()},
            score_agreement={key: score_agreement(scores['exact'], value)
                             for key, value in scores.items() if key != 'exact'})
        for scope in ('prediction_errors', 'original_rows_prediction_errors'):
            result[scope] = {name: combine_moments([value[scope][name] for value in records], weights)
                             for name in records[0][scope]}
        result['relu_flip_from_local'] = {layer: {
            arm: float(np.asarray(weights) @ [value['relu_flip_from_local'][layer][arm] for value in records])
            for arm in ('control', 'treated')} for layer in records[0]['relu_flip_from_local']}
        result['probability_saturation_fraction'] = {arm: float(np.asarray(weights) @ [
            value['probability_saturation_fraction'][arm] for value in records]) for arm in ('control', 'treated')}
        combined.append(result)
    return combined


def restore_locals(source, data):
    config = em.FedFisherEMConfig(**source['config']['fisher'])
    unpacked = em._unpack_sites(source['payloads'], config, em.TRAINING_METHOD)
    if len(unpacked) != len(data.sites):
        raise ValueError('source payloads and regenerated sites differ')
    result = []
    for (arrays, metadata), site in zip(unpacked, data.sites):
        if (metadata['site_id'] != site.site_id or metadata['n_total'] != site.n_total or
                metadata['local_fit']['train_row_ids_sha256'] != array_hash(site.train.row_ids) or
                metadata['local_fit']['train_x_sha256'] != array_hash(site.train.x)):
            raise ValueError('local checkpoint and regenerated training fingerprint differ')
        result.append(em.LocalFisherState(
            model=model_from_vector(metadata['model_name'], metadata['covariate_dim'], arrays['theta']),
            model_name=metadata['model_name'], covariate_dim=metadata['covariate_dim'],
            theta=arrays['theta'], fisher=arrays['fisher'], train=site.train,
            seed=metadata['seed'], site_id=site.site_id, config=config,
            initial_fingerprint=metadata['initial_theta_sha256'], fisher_computed=True,
            report=metadata['local_fit']))
    return result


def _comparison(name, actual, expected, tolerance):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    same_shape = actual.shape == expected.shape
    matches = same_shape and bool(np.allclose(actual, expected, rtol=tolerance['rtol'], atol=tolerance['atol']))
    return dict(name=name, matches=matches, actual=actual.tolist(), expected=expected.tolist(),
                maximum_absolute_error=float(np.max(np.abs(actual - expected))) if same_shape else None)


def full_holdout_endpoint(locals_, theta, data, plans, source, config):
    diagnostics = [em.full_gradient_diagnostics(local, theta, site.holdout.x, plan)
                   for local, site, plan in zip(locals_, data.sites, plans)]
    truth = data.evaluation.truth
    mask = truth.modifier_mask if truth is not None else np.arange(data.sites[0].train.n_features) < source['config']['dgp'].get('signal_dim', 2)
    linearization = summarize_linearization(diagnostics, data.evaluation.target_weights, mask,
                                           source['config']['linearity_gate'])
    expected = source['result']['diagnostics']
    comparisons = []
    for index, (actual, previous) in enumerate(zip(diagnostics, expected['local_linearization'])):
        for key in ('exact', 'full_gradient'):
            comparisons.append(_comparison('site_%d.scores.%s' % (index, key), actual['scores'][key],
                                            previous['scores'][key], config['replay_tolerance']))
        for key in ('theta_euclidean_distance', 'theta_Fisher_distance',
                    'exact_tau_mean_square_over_original_and_all_interventions'):
            comparisons.append(_comparison('site_%d.%s' % (index, key), actual[key], previous[key], config['replay_tolerance']))
        key = 'tau_rmse_over_original_and_all_interventions'
        comparisons.append(_comparison('site_%d.%s' % (index, key), actual[key]['full_gradient'],
                                        previous[key]['full_gradient'], config['replay_tolerance']))
    for key in ('relative_tau_rmse', 'relative_score_linf'):
        comparisons.append(_comparison('gate.' + key, linearization['gate'][key],
                                        expected['linearization']['gate'][key], config['replay_tolerance']))
    matches = (len(diagnostics) == len(expected['local_linearization']) and all(item['matches'] for item in comparisons)
               and linearization['gate']['compression_eligible'] == expected['linearization']['gate']['compression_eligible'])
    return dict(scope='full_original_H_and_all_five_permutations_at_t_1',
        local_diagnostics=diagnostics, linearization=linearization, replay_matches_source=matches,
        replay_comparisons=comparisons, replay_tolerance=config['replay_tolerance'])


def model_quality(locals_, source, data):
    weights, evaluation = data.evaluation.target_weights, data.evaluation
    tests = {'constant_zero_CATE': cate_metrics(np.zeros(len(evaluation.rows)), evaluation)}
    local_predictions = []
    local_outcomes = []
    for local, site in zip(locals_, data.sites):
        value = predict_tau(local.model, evaluation.rows.x)
        local_predictions.append(value)
        tests['local_site_%03d' % site.site_id] = cate_metrics(value, evaluation)
        local_outcomes.append(dict(site_id=site.site_id, scope='own_site_observed_outcome_prediction',
            **{name: outcome_metrics(local.model, getattr(site, attribute))
               for name, attribute in (('T', 'train'), ('V', 'validation'), ('H', 'holdout'))}))
    prediction_average = np.asarray(weights) @ np.stack(local_predictions)
    tests['target_weighted_prediction_average'] = cate_metrics(prediction_average, evaluation)
    tests['target_weighted_prediction_average']['definition'] = 'sum_pi_s_tau_local_s(X); function_ensemble_not_parameter_average'
    names = dict(primary='strict_FedFisher', matched_fedavg='parameter_FedAvg',
                 legacy_validation_selected='saved_legacy_validation_selected',
                 strict_legacy_uploaded_objective='saved_strict_legacy_uploaded_objective')
    outcome_by_model, source_checks = {}, []
    for name, output_name in names.items():
        model = source['models'][name]
        prediction = model.predict_tau(evaluation.rows.x)
        tests[output_name] = cate_metrics(prediction, evaluation)
        outcome_by_model[output_name] = dict(test=outcome_metrics(model.model, evaluation.rows, test_weights(evaluation)),
            sites=[dict(site_id=site.site_id,
                **{split: outcome_metrics(model.model, getattr(site, attr))
                   for split, attr in (('T', 'train'), ('V', 'validation'), ('H', 'holdout'))})
                   for site in data.sites])
        if evaluation.truth is not None:
            source_checks.append(dict(model=name, previous_cate_rmse=source['result']['models'][name]['test']['cate_rmse'],
                                      replay_cate_rmse=tests[output_name]['cate_rmse']))
    tests['parameter_FedAvg']['definition'] = 'tau_of_sum_pi_s_theta_local_s; parameter_average_not_function_ensemble'
    return dict(test=tests, local_outcome_metrics=local_outcomes, saved_model_outcome_metrics=outcome_by_model,
        source_test_rmse_comparisons=source_checks,
        estimated_heterogeneity_definition='target_weighted_centered_second_moment_of_predicted_CATE',
        selection='diagnostic_only_no_epoch_or_model_selection',
        legacy_cate_truth='not_available_not_fabricated' if evaluation.truth is None else None)


def diagnose_fisher(config, data, plans, sources):
    """Replay one architecture/dataset pair from authenticated source bundles."""
    started_wall, started_cpu = perf_counter(), process_time()
    source, linearity = sources['aggregation'], sources['linearity']
    locals_ = restore_locals(source, data)
    theta = model_vector(source['models']['primary'].model)
    if not np.array_equal(theta, model_vector(linearity['models']['primary'].model)):
        raise ValueError('aggregation and linearity source primary parameters differ')
    if len(plans) != len(data.sites):
        raise ValueError('one authenticated permutation plan per site required')
    gradient_records, selected, phases = [], [], {}
    wall, cpu = perf_counter(), process_time()
    for local, site in zip(locals_, data.sites):
        indices, sample = selected_rows(site, config)
        selected.append(indices)
        seed = [config['seed'], config['direction_seed_stream'], site.site_id]
        random_direction = np.random.default_rng(np.random.SeedSequence(seed)).normal(size=local.theta.size)
        check = gradient_check(local.model, site.holdout.x[indices], theta - local.theta,
                               random_direction, config['fd_steps'], config['fd_atol'], config['fd_rtol'])
        gradient_records.append(dict(site_id=site.site_id, **sample, direction_seed_sequence=seed,
                                     normalized_random_direction_sha256=array_hash(random_direction / np.linalg.norm(random_direction)),
                                     **check))
    phases['gradient_audit'] = dict(wall_seconds=perf_counter() - wall, cpu_seconds=process_time() - cpu)
    statuses = [value['status'] for value in gradient_records]
    status = 'failed' if 'failed' in statuses else 'inconclusive' if 'inconclusive' in statuses else 'passed'
    gate = dict(status=status, passed=status == 'passed',
        t0_passed=all(value['t0_passed'] for value in gradient_records),
        coverage_complete=all(value['coverage_complete'] for value in gradient_records),
        original_pilot_passed=all(value['original_pilot']['passed'] for value in gradient_records),
        original_pilot_stable_fd_failures=sum(value['original_pilot']['stable_fd_failures'] for value in gradient_records),
        stable_checks=sum(value['stable_checks'] for value in gradient_records),
        stable_failures=sum(value['stable_failures'] for value in gradient_records),
        boundary_checks=sum(value['boundary_checks'] for value in gradient_records),
        reason='original_pilot_and_new_derivatives_must_agree_and_pass_stable_FD; each_required_direction_and_output_needs_coverage; zero_actual_direction_is_not_applicable')
    result = dict(implementation_gate=gate, gradient_checks=gradient_records,
        parameter_path=None, full_H_endpoint=None, model_quality=None,
        cost_scope=dict(training=False, Fisher_recomputed=False, optimizer_updates=0,
            legacy_selection_replayed=False, model_inference=True, local_derivative_evaluations=True,
            training_communication_rounds=0, new_deployable_scoring_protocol=False),
        alternative_approximation_scope='sigmoid_of_linear_logits_is_diagnostic_only_not_the_current_quadratic_summary')
    if gate['passed']:
        wall, cpu = perf_counter(), process_time()
        paths = [parameter_path(local, theta, site, plan, indices, config)
                 for local, site, plan, indices in zip(locals_, data.sites, plans, selected)]
        result['parameter_path'] = dict(scope='sample_curve_for_localization_not_the_full_H_expansion_gate',
            t_grid=config['t_grid'], sites=paths, target_weights=data.evaluation.target_weights.tolist(),
            target_weighted_points=_combine_paths(paths, data.evaluation.target_weights))
        phases['sampled_parameter_path'] = dict(wall_seconds=perf_counter() - wall, cpu_seconds=process_time() - cpu)
        wall, cpu = perf_counter(), process_time()
        result['full_H_endpoint'] = full_holdout_endpoint(locals_, theta, data, plans, linearity, config)
        phases['full_H_endpoint_reproduction'] = dict(wall_seconds=perf_counter() - wall, cpu_seconds=process_time() - cpu)
        if result['full_H_endpoint']['replay_matches_source']:
            wall, cpu = perf_counter(), process_time()
            result['model_quality'] = model_quality(locals_, source, data)
            phases['model_quality'] = dict(wall_seconds=perf_counter() - wall, cpu_seconds=process_time() - cpu)
    result['phase_timing'] = phases
    result['wall_seconds'], result['cpu_seconds'] = perf_counter() - started_wall, process_time() - started_cpu
    result['downstream_training_allowed_by_this_replay'] = False
    return result
