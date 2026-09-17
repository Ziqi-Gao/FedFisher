"""Survival-mean link used by Ava's pseudo-observation regression.

The response is a survival pseudo-value, not a Bernoulli observation.  The
linear predictor is log(-log(S)); ordinary cloglog on event risk 1-S gives
the same predictor.  Never apply this link to individual pseudo-values.
"""

import numpy as np


def _predictor(eta):
    values = np.asarray(eta, dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("linear predictor must not contain NaN")
    return values


def survival_mean(eta):
    """Return exp(-exp(eta)), including its limiting values at +/- infinity.

    Above 40 both the survival and its derivative have already underflowed
    to zero in float64.  Capping the inner exponent only avoids overflow;
    it does not clip pseudo-observations or change a representable result.
    """
    values = _predictor(eta)
    return np.exp(-np.exp(np.minimum(values, 40.0)))


def survival_mean_derivative(eta):
    """Derivative dS/deta = -exp(eta-exp(eta)), with stable tail limits."""
    values = np.minimum(_predictor(eta), 40.0)
    return -np.exp(values - np.exp(values))


def survival_link(survival):
    """Return log(-log(S)) for a strictly interior conditional mean S."""
    values = np.asarray(survival, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
        raise ValueError("survival link requires finite means strictly between 0 and 1")
    return np.log(-np.log(values))


def pseudo_mse_loss_gradient(eta, pseudo_values):
    """Return half mean squared error and its gradient with respect to eta.

    The factor one half is a documented optimization normalization of Ava's
    sum-squared-error objective.  Finite responses outside [0,1] are valid.
    """
    values = _predictor(eta)
    response = np.asarray(pseudo_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or response.shape != values.shape:
        raise ValueError("predictors and pseudo-values must be matching nonempty vectors")
    if not np.isfinite(response).all():
        raise ValueError("pseudo-values must be finite")
    residual = survival_mean(values) - response
    return (float(0.5 * np.mean(residual ** 2)),
            residual * survival_mean_derivative(values) / values.size)
