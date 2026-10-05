"""Original modifier discovery, with saved-only MLP8 evidence checks.

Inference belongs to the registered discovery task.  Acceptance checks the
original score arithmetic and binds its unpermuted contrast to the already
saved probability predictions.  The accepted historical dsLasso result is the
witness for common raw covariates, oracle contrast and permutation streams;
acceptance never generates data, predicts, or repeats a permutation analysis.
"""

import numpy as np

from ..survival import evaluation as original
from .protocol import TAGS


_FIELDS = {
    'primary', 'scope', 'target', 'num_test', 'repeats', 'modifier_mask',
    'importance_by_repeat', 'importance', 'recovery',
    'contrast_squared_error_sum', 'contrast_rmse', 'raw_x_sha256',
    'contrast_truth_sha256', 'estimated_contrast_sha256', 'permutation_sha256',
    'evaluation_network_measured', 'end_to_end_one_round_claim',
}
_RECOVERY_FIELDS = {
    'expected_topk_hits', 'expected_topk_precision',
    'probability_all_signals_selected', 'auroc', 'boundary_tie_size',
    'boundary_slots', 'tie_rule',
}
_RECIPE = dict(primary=original.PRIMARY, repeats=5,
    top2='probability_all_signals_selected', auroc=True,
    seed_rule='SeedSequence([seed,4291,repeat,covariate])',
    tie_rule=original.TIE_RULE)


def evaluate_discovery(predict_eta, dataset, seed, repeats=5):
    """Use the original raw-X permutation/eta-contrast analysis unchanged.

    Both counterfactual designs rebuild A and A*X for every permutation.  The
    score is the mean absolute change in eta(1, X)-eta(0, X), averaged over the
    original five repeat streams.  Prognostic-only terms therefore cancel on
    this scale.  Top-2 is the probability of selecting both true modifiers,
    with the original uniform rule for exact ties at the selection boundary.
    """
    return original.evaluate_model(predict_eta, dataset, repeats=repeats, seed=seed)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(arrays, name, shape, dtype=np.float64):
    value = arrays[name]
    _require(isinstance(value, np.ndarray) and value.shape == shape
             and value.dtype == dtype and np.isfinite(value).all(),
             'invalid saved discovery array: ' + name)
    return value


def _summary(summary, p, n, repeats):
    _require(isinstance(summary, dict) and set(summary) == _FIELDS,
             'discovery summary inventory mismatch')
    _require(type(summary['num_test']) is int and type(summary['repeats']) is int,
             'discovery counts must be integers')
    importance, scores, mask = (summary[name] for name in
                               ('importance', 'importance_by_repeat', 'modifier_mask'))
    _require(isinstance(importance, list) and len(importance) == p
             and isinstance(scores, list) and len(scores) == repeats
             and all(isinstance(row, list) and len(row) == p for row in scores)
             and all(type(value) in (int, float)
                     for row in [importance, *scores] for value in row)
             and isinstance(mask, list) and all(type(value) is bool for value in mask),
             'discovery scores and modifier mask must be numeric/bool JSON lists')
    recovery = summary['recovery']
    _require(isinstance(recovery, dict) and set(recovery) == _RECOVERY_FIELDS
             and all(type(recovery[name]) is int
                     for name in ('boundary_tie_size', 'boundary_slots')),
             'discovery recovery inventory or type mismatch')
    scalar_fields = ('contrast_squared_error_sum', 'contrast_rmse')
    scalar_values = [summary[name] for name in scalar_fields] + [
        recovery[name] for name in ('expected_topk_hits', 'expected_topk_precision',
                                   'probability_all_signals_selected')]
    if p > 2:
        scalar_values.append(recovery['auroc'])
    _require(all(type(value) in (int, float) for value in scalar_values),
             'discovery metrics must be numeric scalars')
    # The unchanged validator recomputes means, all ranking metrics, tie
    # handling and contrast-RMSE arithmetic from the saved JSON scores.
    original.validate_evaluation(summary, p, n, repeats)


def _validate(discovery, arrays, config, reference_result, seed):
    _require(isinstance(discovery, dict) and set(discovery) == set(TAGS),
             'discovery requires exactly the four MLP8 methods')
    _require(isinstance(arrays, dict), 'invalid saved discovery arrays')
    recipe = config['evaluation']['discovery']
    _require(recipe == _RECIPE and type(recipe['repeats']) is int
             and recipe['auroc'] is True
             and config['evaluation']['modifier_permutation_performed'] is True,
             'discovery does not use the fixed original five-repeat recipe')
    p, n = config['dgp']['covariate_dim'], config['dgp']['num_test']
    _require(type(p) is int and p >= 2 and type(n) is int and n > 0,
             'invalid discovery dimensions')
    if seed is not None:
        _require(type(seed) is int and seed >= 0 and seed in config['seeds']
                 and type(reference_result['identity']['seed']) is int
                 and reference_result['identity']['seed'] == seed,
                 'discovery reference belongs to another seed')
    reference = reference_result['evaluation']
    _summary(reference, p, n, recipe['repeats'])
    x = _array(arrays, 'discovery__x', (n, p))
    truth = _array(arrays, 'truth__delta', (n,))
    mask = _array(arrays, 'truth__modifier_mask', (p,), np.bool_)
    _require(np.array_equal(mask, np.arange(p) < 2),
             'discovery requires exactly the original first two modifiers')
    x_hash, truth_hash = original._array_hash(x), original._array_hash(truth)
    _require(x_hash == reference['raw_x_sha256']
             and truth_hash == reference['contrast_truth_sha256'],
             'discovery covariates or oracle contrast differ from the accepted reference')
    for tag in TAGS:
        summary = discovery[tag]
        _summary(summary, p, n, recipe['repeats'])
        eta0 = _array(arrays, tag + '__prediction__eta0', (n,))
        eta1 = _array(arrays, tag + '__prediction__eta1', (n,))
        baseline = eta1 - eta0
        error = baseline - truth
        squared_error = float(error @ error)
        _require(np.isfinite(baseline).all() and np.isfinite(squared_error),
                 'nonfinite discovery contrast or error')
        _require(summary['raw_x_sha256'] == x_hash
                 and summary['contrast_truth_sha256'] == truth_hash
                 and summary['permutation_sha256'] == reference['permutation_sha256']
                 and summary['modifier_mask'] == mask.tolist(),
                 'discovery methods do not share the accepted covariates, truth and permutations')
        # Use the persisted DGP delta directly, as the original evaluator
        # does.  Reconstructing delta as oracle eta1-eta0 can round differently.
        _require(summary['estimated_contrast_sha256'] == original._array_hash(baseline)
                 and summary['contrast_squared_error_sum'] == squared_error
                 and summary['contrast_rmse'] == float(np.sqrt(squared_error / n)),
                 'discovery baseline contrast or error differs from saved eta predictions')
    return True


def validate_discovery(discovery, arrays, config, reference_result, *, seed=None):
    """Validate saved discovery evidence without fitting, inference or RNG.

    The caller validates the historical result's immutable receipt before
    passing it here.  ``seed`` additionally binds its scientific identity to
    the current unit; all methods must use its accepted permutation digest.
    """
    try:
        return _validate(discovery, arrays, config, reference_result, seed)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError('malformed saved MLP8 discovery evidence') from error
