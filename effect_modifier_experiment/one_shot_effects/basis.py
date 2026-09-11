"""Fixed, shared raw-coordinate bases for the one-shot DR effect model.

The spline knots are protocol constants, never empirical quantiles.  Six
clamped cubic B-splines form a partition of unity on [-1, 1]; dropping the
last of these in each block leaves five directions and one shared intercept.
No site-specific or server-estimated centering/scaling is applied.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BasisConfig:
    kind: str = "additive_cubic_bspline"
    covariate_dim: int = 100

    def __post_init__(self):
        if self.kind not in {"additive_cubic_bspline", "linear"}:
            raise ValueError("unknown shared basis")
        if type(self.covariate_dim) is not int or self.covariate_dim < 1:
            raise ValueError("positive integer raw covariate dimension required")


def cubic_bspline_values(u):
    """All six cubic B-splines, including both closed boundary endpoints."""
    u = np.asarray(u, dtype=np.float64)
    if not np.isfinite(u).all() or np.any((u < -1) | (u > 1)):
        raise ValueError("spline coordinates must be finite in [-1, 1]")
    knots = np.array([-1., -1., -1., -1., -1/3, 1/3, 1., 1., 1., 1.])
    result = ((u[..., None] >= knots[:-1]) & (u[..., None] < knots[1:])).astype(float)
    for degree in range(1, 4):
        updated = np.zeros(u.shape + (len(knots) - degree - 1,))
        for j in range(updated.shape[-1]):
            left = knots[j + degree] - knots[j]
            right = knots[j + degree + 1] - knots[j + 1]
            if left > 0:
                updated[..., j] += (u-knots[j]) / left * result[..., j]
            if right > 0:
                updated[..., j] += (knots[j+degree+1]-u) / right * result[..., j+1]
        result = updated
    # Half-open degree-zero intervals exclude the last endpoint; use its
    # continuous clamped limit (tanh can round to exactly 1 for finite x).
    result[u == 1] = np.array([0., 0., 0., 0., 0., 1.])
    return result


class SharedBasis:
    def __init__(self, config):
        if not isinstance(config, BasisConfig):
            raise TypeError("BasisConfig required")
        self.config = config

    @property
    def group_size(self):
        return 5 if self.config.kind == "additive_cubic_bspline" else 1

    @property
    def dimension(self):
        return 1 + self.config.covariate_dim * self.group_size

    @property
    def groups(self):
        q = self.group_size
        return tuple(slice(1+j*q, 1+(j+1)*q) for j in range(self.config.covariate_dim))

    def blocks(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.config.covariate_dim or not np.isfinite(x).all():
            raise ValueError("finite raw covariates [n,p] matching shared basis required")
        if self.config.kind == "linear":
            return x[..., None]
        return cubic_bspline_values(np.tanh(x))[..., :5]

    def transform(self, x):
        blocks = self.blocks(x)
        return np.column_stack((np.ones(blocks.shape[0]), blocks.reshape(blocks.shape[0], -1)))

    def metadata(self):
        return dict(config=asdict(self.config), dimension=self.dimension,
                    group_size=self.group_size, shared_intercept=True,
                    standardization="none_fixed_raw_coordinate_basis",
                    transform="tanh" if self.group_size == 5 else "identity",
                    degree=3 if self.group_size == 5 else 1,
                    interior_knots=[-1/3, 1/3] if self.group_size == 5 else [],
                    boundary_knots=[-1., 1.] if self.group_size == 5 else [],
                    boundary_multiplicity=4 if self.group_size == 5 else None,
                    dropped_basis_index=5 if self.group_size == 5 else None)
