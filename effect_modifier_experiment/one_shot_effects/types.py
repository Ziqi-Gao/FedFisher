"""Observed method inputs and separate evaluator-only simulation truth.

The separation is an API boundary in a single-process simulation, not a
security sandbox. Methods receive Rows or SitePartitions, never ExperimentData.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _array(value, dtype, ndim, name):
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "biuf":
        raise ValueError(name + " must be a real numeric array of the required dimension")
    result = np.array(raw, dtype=dtype, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class Rows:
    """Original-coordinate X, observed binary A/Y, and observation identifiers."""

    x: np.ndarray
    a: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray

    def __post_init__(self):
        x = _array(self.x, np.float64, 2, "x")
        a = _array(self.a, np.float64, 1, "a")
        y = _array(self.y, np.float64, 1, "y")
        raw_ids = np.asarray(self.row_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise ValueError("row_ids must be a one-dimensional integer array")
        if np.any(raw_ids < 0) or np.any(raw_ids > np.iinfo(np.int64).max):
            raise ValueError("row_ids must be nonnegative int64 values")
        ids = _array(raw_ids, np.int64, 1, "row_ids")
        if min(x.shape) < 1 or any(len(v) != len(x) for v in (a, y, ids)):
            raise ValueError("observed row dimensions do not agree")
        if not np.isin(a, (0, 1)).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("observed treatment and outcome must be binary")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("row identifiers must be unique")
        for name, value in (("x", x), ("a", a), ("y", y), ("row_ids", ids)):
            object.__setattr__(self, name, value)

    def __len__(self):
        return self.x.shape[0]

    @property
    def n_samples(self):
        return len(self)

    @property
    def n_features(self):
        return self.x.shape[1]


@dataclass(frozen=True, eq=False)
class SitePartitions:
    site_id: int
    n_total: int
    train: Rows
    validation: Rows
    holdout: Rows

    def __post_init__(self):
        if type(self.site_id) is not int or self.site_id < 0:
            raise ValueError("site_id must be a nonnegative integer")
        if type(self.n_total) is not int or self.n_total < 3:
            raise ValueError("n_total must be the original positive site size")
        blocks = (self.train, self.validation, self.holdout)
        if any(type(rows) is not Rows for rows in blocks):
            raise ValueError("sites accept only observed Rows")
        if len({rows.n_features for rows in blocks}) != 1:
            raise ValueError("partition coordinates differ")
        ids = np.concatenate([rows.row_ids for rows in blocks])
        if len(ids) != self.n_total or len(np.unique(ids)) != self.n_total:
            raise ValueError("T/V/H must be disjoint and exhaust the original site")


@dataclass(frozen=True, eq=False)
class EvaluationTruth:
    """Truth aligned with independent evaluation rows; never a method input."""

    e: np.ndarray
    mu_0: np.ndarray
    mu_1: np.ndarray
    tau: np.ndarray
    modifier_mask: np.ndarray
    variable_roles: Tuple[str, ...]

    def __post_init__(self):
        arrays = {name: _array(getattr(self, name), np.float64, 1, name)
                  for name in ("e", "mu_0", "mu_1", "tau")}
        if len({len(value) for value in arrays.values()}) != 1:
            raise ValueError("evaluation truth dimensions differ")
        if any(np.any((arrays[name] < 0) | (arrays[name] > 1))
               for name in ("e", "mu_0", "mu_1")):
            raise ValueError("invalid simulation probabilities; do not clip outcome means")
        if not np.allclose(arrays["mu_1"] - arrays["mu_0"], arrays["tau"],
                           rtol=0, atol=1e-14):
            raise ValueError("CATE does not equal the potential-outcome probability contrast")
        mask = np.asarray(self.modifier_mask)
        if mask.ndim != 1 or mask.dtype.kind != "b":
            raise ValueError("modifier_mask must be a boolean vector")
        roles = tuple(self.variable_roles)
        if len(roles) != len(mask) or not all(isinstance(role, str) and role for role in roles):
            raise ValueError("variable roles must align with raw covariates")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "modifier_mask", _array(mask, np.bool_, 1, "modifier_mask"))
        object.__setattr__(self, "variable_roles", roles)


@dataclass(frozen=True, eq=False)
class EvaluationData:
    rows: Rows
    site_ids: Optional[np.ndarray]
    target_weights: np.ndarray
    truth: Optional[EvaluationTruth]

    def __post_init__(self):
        if type(self.rows) is not Rows:
            raise ValueError("evaluation requires observed Rows")
        weights = _array(self.target_weights, np.float64, 1, "target_weights")
        if len(weights) < 1 or np.any(weights <= 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-14):
            raise ValueError("target site weights must be positive and sum to one")
        object.__setattr__(self, "target_weights", weights)
        if self.site_ids is not None:
            raw = np.asarray(self.site_ids)
            if raw.ndim != 1 or raw.dtype.kind not in "iu" or len(raw) != len(self.rows):
                raise ValueError("evaluation site identifiers must align with rows")
            if np.any(raw < 0) or np.any(raw >= len(weights)):
                raise ValueError("evaluation site identifier outside target mixture")
            object.__setattr__(self, "site_ids", _array(raw, np.int64, 1, "site_ids"))
        if self.truth is not None:
            if type(self.truth) is not EvaluationTruth or len(self.truth.tau) != len(self.rows):
                raise ValueError("evaluation truth must align with independent rows")
            if len(self.truth.modifier_mask) != self.rows.n_features:
                raise ValueError("truth mask must align with raw covariates")

    @property
    def x(self):
        return self.rows.x


@dataclass(frozen=True, eq=False)
class ExperimentData:
    sites: Tuple[SitePartitions, ...]
    evaluation: EvaluationData

    def __post_init__(self):
        sites = tuple(self.sites)
        if not sites or any(type(site) is not SitePartitions for site in sites):
            raise ValueError("experiment requires observed site partitions")
        if tuple(site.site_id for site in sites) != tuple(range(len(sites))):
            raise ValueError("sites must be ordered with consecutive zero-based identifiers")
        if type(self.evaluation) is not EvaluationData:
            raise ValueError("experiment requires separate evaluator data")
        dimensions = {site.train.n_features for site in sites} | {self.evaluation.rows.n_features}
        if len(dimensions) != 1:
            raise ValueError("sites and evaluator must use the same raw coordinates")
        sizes = np.array([site.n_total for site in sites], dtype=np.float64)
        if not np.array_equal(self.evaluation.target_weights, sizes / sizes.sum()):
            raise ValueError("target weights must use original site sizes exactly once")
        identifiers = [block.row_ids for site in sites
                       for block in (site.train, site.validation, site.holdout)]
        identifiers.append(self.evaluation.rows.row_ids)
        all_ids = np.concatenate(identifiers)
        if len(np.unique(all_ids)) != len(all_ids):
            raise ValueError("site partitions and external test identifiers must not overlap")
        object.__setattr__(self, "sites", sites)
