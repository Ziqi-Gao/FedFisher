"""Site-local Kaplan-Meier infinitesimal-jackknife survival pseudo-values.

For unweighted, one-row-per-person right-censored data, this implements the
quantity returned by survival::pseudo(survfit(Surv(time,event) ~ 1),
times=tau, type="surv"): S_hat(tau) + n * dS_hat(tau)/dw_i at w=1.
It is not the leave-one-out jackknife.  No pooled curve is constructed here.

Primary references (R survival source):
https://github.com/therneau/survival/blob/master/R/pseudo.R
https://github.com/therneau/survival/blob/master/R/residuals.survfit.R
https://github.com/therneau/survival/blob/master/R/aeqSurv.R

The statistical contract is independent censoring within each site.  Delayed
entry, case weights, repeated subjects, competing risks, and informative
censoring corrections are outside this adapter's scope.
"""

import warnings

import numpy as np


def _inputs(time, event, tau, timefix):
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    if time.ndim != 1 or time.size == 0 or event.shape != time.shape:
        raise ValueError("time and event must be matching nonempty vectors")
    if not np.isfinite(time).all() or np.any(time < 0):
        raise ValueError("observed times must be finite and nonnegative")
    if not np.isin(event, (0, 1)).all():
        raise ValueError("event must use 0=censored and 1=event")
    if isinstance(tau, (bool, np.bool_)) or np.ndim(tau) != 0:
        raise ValueError("tau must be a finite nonnegative scalar")
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be a finite nonnegative scalar")
    if not isinstance(timefix, (bool, np.bool_)):
        raise ValueError("timefix must be boolean")
    if timefix:
        # R's aeqSurv uses the mean absolute UNIQUE time, rather than the
        # observation-weighted mean, when deciding its relative tolerance.
        unique = np.unique(time)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        threshold = tolerance * max(1.0, float(np.mean(np.abs(unique))))
        cuts = unique[np.r_[True, np.diff(unique) > threshold]]
        time = cuts[np.searchsorted(cuts, time, side="right") - 1]
    if tau > time.max():
        warnings.warn("tau is beyond the observed curve; extending its last value", RuntimeWarning,
                      stacklevel=3)
    return time, event.astype(bool), tau


def _event_table(time, event, tau):
    observed_events = time[event & (time <= tau)]
    times, deaths = np.unique(observed_events, return_counts=True)
    risk = len(time) - np.searchsorted(np.sort(time), times, side="left")
    return times, deaths.astype(np.float64), risk.astype(np.float64)


def km_survival(time, event, tau, *, timefix=True):
    """Unweighted right-continuous Kaplan-Meier survival at one time point."""
    time, event, tau = _inputs(time, event, tau, timefix)
    _, deaths, risk = _event_table(time, event, tau)
    return float(np.prod((risk - deaths) / risk))


def pseudo_survival(time, event, tau, *, timefix=True):
    """Return R-style IJ pseudo-values in input order, without truncation.

    Complexity is O(n log n) time and O(n) memory.  All failures at a tied
    time use the same risk set, including subjects censored at that time.
    Events at tau count.  The default near-tie correction matches R's
    survfit(timefix=TRUE); timefix=False preserves exact supplied times.
    """
    time, event, tau = _inputs(time, event, tau, timefix)
    times, deaths, risk = _event_table(time, event, tau)
    if len(times) == 0:
        return np.ones(len(time), dtype=np.float64)
    remaining = risk - deaths
    if np.any(remaining == 0):
        # The weighted KM curve is identically zero after all remaining
        # subjects fail, so its derivative is zero too.  Avoid 0 * infinity.
        return np.zeros(len(time), dtype=np.float64)
    survival = float(np.prod(remaining / risk))
    risk_integral = np.r_[0.0, np.cumsum(deaths / (risk * remaining))]
    last_event = np.searchsorted(times, time, side="right")
    influence = risk_integral[last_event]
    fails_by_tau = event & (time <= tau)
    influence[fails_by_tau] -= 1.0 / remaining[last_event[fails_by_tau] - 1]
    return survival * (1.0 + len(time) * influence)


def site_pseudo_response(rows, tau):
    """Compute a response solely from this site's supplied observed rows.

    Pass each fitting or validation subset separately when tuning so that
    pseudo-value construction never reads the other fold's outcome records.
    """
    from .data import SurvivalRows

    if not isinstance(rows, SurvivalRows):
        raise ValueError("site pseudo-response requires observed SurvivalRows")
    return pseudo_survival(rows.time, rows.event, tau)


def site_pseudo_diagnostics(rows, tau, pv=None):
    """JSON-ready observed-data diagnostics for one site's response at tau.

    Censoring fraction is observed censored_count/n, not the exponential
    censoring distribution's rate parameter.  Counts at risk and the support
    flag use the literal supplied tau and original observed times; KM/PV use
    their documented R-style near-tie correction.  A supplied cached PV vector
    must correspond to these rows/tau; its shape and finiteness are checked
    here without repeating construction.
    """
    from .data import SurvivalRows

    if not isinstance(rows, SurvivalRows):
        raise ValueError("site diagnostics require observed SurvivalRows")
    # Validate tau even when a cached response is supplied.
    survival = km_survival(rows.time, rows.event, tau)
    tau = float(tau)
    response = site_pseudo_response(rows, tau) if pv is None else np.asarray(pv, dtype=np.float64)
    if response.shape != (len(rows),) or not np.isfinite(response).all():
        raise ValueError("cached pseudo-values must be a matching finite vector")
    n = len(rows)
    event_count = int(np.count_nonzero(rows.event))
    censored_count = n - event_count
    maximum = float(np.max(rows.time))
    with np.errstate(over="ignore", invalid="ignore"):
        response_mean = float(np.mean(response))
    if not np.isfinite(response_mean):
        raise ValueError("cached pseudo-value mean must be finite")
    return dict(n=n, event_count=event_count, censored_count=censored_count,
                censoring_fraction=float(censored_count / n),
                n_at_risk_tau=int(np.count_nonzero(rows.time >= tau)),
                km_survival_tau=survival, pv_min=float(np.min(response)),
                pv_max=float(np.max(response)), pv_mean=response_mean,
                outside_unit_interval_count=int(np.count_nonzero((response < 0) | (response > 1))),
                all_zero=bool(np.all(response == 0)), all_one=bool(np.all(response == 1)),
                tau=tau, max_observed_time=maximum, tau_beyond_observed_max=bool(tau > maximum))
