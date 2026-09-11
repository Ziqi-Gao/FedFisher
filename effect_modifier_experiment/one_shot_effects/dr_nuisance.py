"""Site-local, T-only nested logistic nuisance fits for DR pseudo-outcomes.

The existing R-learner's tested logistic CV implementation is reused without
changing that historical method. Here it fits three distinct targets: A,
Y within A=0, and Y within A=1. Empty treatment groups fail; a nonempty
constant outcome uses a training-only smoothed Bernoulli mean. A nonconstant
target with fewer rows than inner folds fails rather than changing the grid.
"""
import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from effect_modifier_experiment.causal_comparison.rlearner import (
    RlearnerConfig, _fit_nuisance_cv, _validate_config, deterministic_folds,
)
from .types import Rows


@dataclass(frozen=True)
class DRNuisanceConfig:
    crossfit_folds: int = 5
    nuisance_cv_folds: int = 5
    nuisance_lambda_count: int = 30
    nuisance_lambda_max: float = 0.3
    nuisance_lambda_min_ratio: float = 0.001
    nuisance_max_iter: int = 4000
    nuisance_tolerance: float = 1e-6
    propensity_clip: float = 0.05
    constant_pseudocount: float = 0.5

    def __post_init__(self):
        _validate_config(RlearnerConfig(**asdict(self)))


def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def _validate_rows(rows):
    if type(rows) is not Rows:
        raise TypeError("nuisance fitting accepts observed Rows only")
    x, a, y, ids = map(np.asarray, (rows.x, rows.a, rows.y, rows.row_ids))
    if (x.ndim != 2 or len(x) < 1 or a.shape != (len(x),) or y.shape != a.shape
            or ids.shape != a.shape or ids.dtype.kind not in "iu"
            or np.unique(ids).size != len(ids) or not np.isfinite(x).all()
            or not np.isin(a, [0, 1]).all() or not np.isin(y, [0, 1]).all()):
        raise ValueError("observed nonempty binary rows with unique integer IDs required")


def _triplet(x, a, y, ids, predict_x, config, seed, stages):
    """The prediction rows' A/Y and all H rows are deliberately unavailable."""
    arm_counts = [int(np.sum(a == arm)) for arm in (0, 1)]
    if min(arm_counts) == 0:
        raise ValueError("DR nuisance training requires both treatment groups; empty arm")
    legacy_config = RlearnerConfig(**asdict(config))
    predictions, reports = {}, {}
    for offset, (name, mask, target) in enumerate((
        ("propensity", np.ones(len(x), dtype=bool), a),
        ("mu0", a == 0, y[a == 0]),
        ("mu1", a == 1, y[a == 1]),
    )):
        if np.unique(target).size != 1 and len(target) < config.nuisance_cv_folds:
            raise ValueError("nonconstant nuisance target has fewer rows than inner CV folds")
        predicted, report = _fit_nuisance_cv(
            x[mask], target, predict_x, legacy_config, seed + 1009 * offset, stages)
        if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
            raise RuntimeError("invalid nuisance probability predictions")
        reports[name] = dict(report, training_row_ids_sha256=array_hash(ids[mask]))
        predictions[name] = predicted
    raw_e = predictions["propensity"]
    clipped_e = np.clip(raw_e, config.propensity_clip, 1-config.propensity_clip)
    audit = dict(training_rows=len(x), treatment_arm_counts=arm_counts,
                 training_row_ids_sha256=array_hash(ids), models=reports,
                 propensity_clip_count=int(np.count_nonzero(raw_e != clipped_e)),
                 prediction_rows=len(predict_x),
                 raw_propensity_min=float(raw_e.min()), raw_propensity_max=float(raw_e.max()),
                 converged=True)
    return clipped_e, predictions["mu0"], predictions["mu1"], audit


def aipw_pseudo_outcome(a, y, e, mu0, mu1):
    a, y, e, mu0, mu1 = [np.asarray(value, dtype=np.float64) for value in (a, y, e, mu0, mu1)]
    if (a.ndim != 1 or any(value.shape != a.shape for value in (y, e, mu0, mu1))
            or any(not np.isfinite(value).all() for value in (a, y, e, mu0, mu1))
            or not np.isin(a, [0, 1]).all() or not np.isin(y, [0, 1]).all()
            or np.any((e <= 0) | (e >= 1))
            or np.any((mu0 < 0) | (mu0 > 1) | (mu1 < 0) | (mu1 > 1))):
        raise ValueError("valid aligned observed rows and nuisance probabilities required")
    return mu1-mu0 + a/e*(y-mu1) - (1-a)/(1-e)*(y-mu0)


def fit_dr_pseudo_outcomes(train, validation, config, seed):
    """Return (T cross-fitted psi, V T-only psi, local-only JSON audit).

    No holdout, evaluator object, oracle values, or fitting callback is accepted.
    Fold assignment is a function of T row count and seed, never A or Y.
    """
    if not isinstance(config, DRNuisanceConfig) or type(seed) is not int or seed < 0:
        raise ValueError("valid nuisance config and nonnegative integer seed required")
    _validate_rows(train)
    _validate_rows(validation)
    if train.x.shape[1] != validation.x.shape[1] or np.intersect1d(train.row_ids, validation.row_ids).size:
        raise ValueError("T and V require disjoint IDs and equal raw dimensions")
    folds = deterministic_folds(len(train.x), config.crossfit_folds, seed)
    psi = np.empty(len(train.x), dtype=np.float64)
    stages, fold_reports = {}, []
    for fold in range(config.crossfit_folds):
        fitting, held = folds != fold, folds == fold
        e, mu0, mu1, report = _triplet(
            train.x[fitting], train.a[fitting], train.y[fitting], train.row_ids[fitting],
            train.x[held], config, seed + 100003*(fold+1), stages)
        psi[held] = aipw_pseudo_outcome(train.a[held], train.y[held], e, mu0, mu1)
        fold_reports.append(dict(report, fold=fold,
                                 heldout_row_ids_sha256=array_hash(train.row_ids[held])))
    e, mu0, mu1, refit = _triplet(
        train.x, train.a, train.y, train.row_ids, validation.x,
        config, seed + 100003*(config.crossfit_folds+1), stages)
    validation_psi = aipw_pseudo_outcome(validation.a, validation.y, e, mu0, mu1)
    if not np.isfinite(psi).all() or not np.isfinite(validation_psi).all():
        raise RuntimeError("nonfinite DR pseudo-outcomes")
    return psi, validation_psi, dict(
        config=asdict(config), seed=seed, stages=stages, converged=True,
        crossfit_folds=fold_reports, validation_nuisance_refit=refit,
        training_row_ids_sha256=array_hash(train.row_ids),
        validation_row_ids_sha256=array_hash(validation.row_ids),
        fold_assignments_sha256=array_hash(folds),
        training_psi_sha256=array_hash(psi), validation_psi_sha256=array_hash(validation_psi),
        treatment_arm_counts=[int(np.sum(train.a == a)) for a in (0, 1)],
        propensity_clipping="clip_predictions_to_declared_bounds_before_AIPW",
        failure_rule="empty_training_arm_or_nonconstant_target_smaller_than_inner_folds_fail",
        constant_outcome_rule="nonempty_train_only_smoothed_Bernoulli_with_declared_pseudocount",
        standardization_scope="each_inner_training_fold_then_outer_training_refit_only",
        validation_fit_scope="all_T_only", holdout_used=False)
