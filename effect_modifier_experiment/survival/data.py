"""Independent site-first survival simulation; the binary DGP is unchanged.

Original covariate dimensions, confounding and effect-modifier coefficients
are retained.  The outcome becomes a Weibull proportional-hazards time with
independent exponential censoring.  Censor rate is a distribution parameter,
not a target fraction censored.  Truth lives outside each site's observed rows.
"""

from dataclasses import dataclass
import math

import numpy as np

from .links import survival_mean


DATASET_ID = "SyntheticEffectModifierSurvivalV1"
SCENARIOS = ("iid", "covariate_shift")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))


def _rng(seed, stream, site_id=0):
    return np.random.default_rng(np.random.SeedSequence([seed, stream, site_id]))


@dataclass(frozen=True)
class SurvivalConfig:
    num_sites: int = 10
    site_min_samples: int = 500
    site_max_samples: int = 1000
    covariate_dim: int = 100
    num_test: int = 10000
    signal_strength: float = 2.0
    main_effect_strength: float = 1.0
    confounding_strength: float = 1.0
    outcome_intercept: float = 0.0
    treatment_prob: float = 0.5
    propensity_clip: float = 0.05
    weibull_shape: float = 1.5
    weibull_lambda: float = 0.12
    censor_rate: float = 0.05
    tau: float = 3.0
    site_mean_span: float = 0.75
    site_std_low: float = 0.75
    site_std_high: float = 1.25
    dataset_id: str = DATASET_ID

    def __post_init__(self):
        integer_fields = {"num_sites": 1, "site_min_samples": 1, "site_max_samples": 1,
                          "covariate_dim": 2, "num_test": 1}
        for name, minimum in integer_fields.items():
            _integer(getattr(self, name), name, minimum)
        for name, value in self.__dict__.items():
            if name not in integer_fields and name != "dataset_id":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(name + " must be finite and numeric")
        if self.site_max_samples < self.site_min_samples:
            raise ValueError("site size bounds are reversed")
        if min(self.signal_strength, self.main_effect_strength, self.confounding_strength,
               self.site_mean_span) < 0:
            raise ValueError("strengths and site mean span must be nonnegative")
        if min(self.weibull_shape, self.weibull_lambda, self.censor_rate, self.tau,
               self.site_std_low) <= 0 or self.site_std_high < self.site_std_low:
            raise ValueError("survival scales/rate/time and site standard deviations must be positive")
        if not 0 < self.treatment_prob < 1 or not 0 < self.propensity_clip < 0.5:
            raise ValueError("invalid propensity settings")
        if self.dataset_id != DATASET_ID:
            raise ValueError("survival simulation must retain its independent dataset identity")


def _readonly(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SurvivalRows:
    x: np.ndarray
    a: np.ndarray
    time: np.ndarray
    event: np.ndarray
    row_ids: np.ndarray

    def __post_init__(self):
        x, a, time, event, ids = (np.asarray(getattr(self, name))
                                 for name in ("x", "a", "time", "event", "row_ids"))
        if x.ndim != 2 or len(x) == 0 or x.shape[1] < 2:
            raise ValueError("x must be a nonempty matrix with at least two covariates")
        if any(values.shape != (len(x),) for values in (a, time, event, ids)):
            raise ValueError("observed row arrays have inconsistent lengths")
        if not np.isfinite(x).all() or not np.isfinite(time).all() or np.any(time < 0):
            raise ValueError("covariates and nonnegative observed times must be finite")
        if not np.isin(a, (0, 1)).all() or not np.isin(event, (0, 1)).all():
            raise ValueError("treatment and event must be binary")
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("row_ids must be unique nonnegative integers")
        for name in ("x", "a", "time", "event", "row_ids"):
            dtype = np.int64 if name in ("event", "row_ids") else np.float64
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    def __len__(self):
        return len(self.a)

    @property
    def design(self):
        return np.column_stack((self.a, self.x, self.a[:, None] * self.x))

    def subset(self, index):
        return SurvivalRows(*(getattr(self, name)[index]
                              for name in ("x", "a", "time", "event", "row_ids")))


@dataclass(frozen=True)
class SurvivalSite:
    site_id: int
    rows: SurvivalRows

    def __post_init__(self):
        _integer(self.site_id, "site_id")
        if not isinstance(self.rows, SurvivalRows):
            raise ValueError("site requires observed SurvivalRows")


@dataclass(frozen=True)
class SurvivalTruth:
    propensity: np.ndarray
    eta0: np.ndarray
    eta1: np.ndarray
    survival0: np.ndarray
    survival1: np.ndarray
    delta: np.ndarray
    modifier_mask: np.ndarray


@dataclass(frozen=True)
class SurvivalDataset:
    sites: tuple
    evaluation: SurvivalRows
    evaluation_truth: SurvivalTruth
    site_weights: np.ndarray
    config: SurvivalConfig
    seed: int
    scenario: str


def survival_truth(x, config):
    """Evaluator-only PH truth; delta is log cumulative hazard (log HR here)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != config.covariate_dim or not np.isfinite(x).all():
        raise ValueError("truth covariates must match the configured dimension")
    root_two = np.sqrt(2.0)
    propensity_eta = math.log(config.treatment_prob / (1 - config.treatment_prob))
    propensity_eta = propensity_eta + config.confounding_strength * (x[:, 0] + x[:, 1]) / root_two
    propensity = np.clip(np.exp(-np.logaddexp(0.0, -propensity_eta)),
                         config.propensity_clip, 1 - config.propensity_clip)
    baseline = math.log(config.weibull_lambda) + config.weibull_shape * math.log(config.tau)
    eta0 = baseline + config.outcome_intercept + config.main_effect_strength * (x[:, 0] + x[:, 1]) / root_two
    delta = config.signal_strength * (x[:, 0] - x[:, 1]) / root_two
    mask = np.zeros(config.covariate_dim, dtype=bool)
    mask[:2] = config.signal_strength > 0
    return SurvivalTruth(propensity, eta0, eta0 + delta, survival_mean(eta0),
                         survival_mean(eta0 + delta), delta, mask)


def site_gaussian_parameters(config):
    """Same fixed mean-by-scale proposal as legacy, without label partitioning."""
    mean_count = (config.num_sites + 1) // 2
    means = (np.zeros(1) if mean_count == 1 else
             np.linspace(-config.site_mean_span, config.site_mean_span, mean_count))
    return tuple((float(mean), float(std)) for mean in means
                 for std in (config.site_std_low, config.site_std_high))[:config.num_sites]


def _observations(x, config, seed, site_id, stream, row_start):
    truth = survival_truth(x, config)
    a = _rng(seed, stream, site_id).binomial(1, truth.propensity)
    risk = (config.outcome_intercept + config.main_effect_strength * (x[:, 0] + x[:, 1]) / np.sqrt(2.0)
            + a * truth.delta)
    exponential = _rng(seed, stream + 1, site_id).exponential(size=len(x))
    if np.any(exponential <= 0):
        raise FloatingPointError("nonpositive event-time innovation")
    log_time = (np.log(exponential) - math.log(config.weibull_lambda) - risk) / config.weibull_shape
    with np.errstate(over="raise", under="ignore"):
        event_time = np.exp(log_time)
    censor_time = _rng(seed, stream + 2, site_id).exponential(1.0 / config.censor_rate, size=len(x))
    return SurvivalRows(x, a, np.minimum(event_time, censor_time), (event_time <= censor_time).astype(int),
                        np.arange(row_start, row_start + len(x), dtype=np.int64))


def generate_survival(config, seed, scenario="iid"):
    """Generate independent sites then outcomes, plus a common independent test.

    Training generation never uses outcomes to assign sites.  Evaluation uses
    the common standard Gaussian target across both scenarios, matching the
    original project's shared-test convention.  It is not relabeled as a
    site-mixture target.  Changing num_test cannot change training records.
    """
    if not isinstance(config, SurvivalConfig) or scenario not in SCENARIOS:
        raise ValueError("survival generation requires SurvivalConfig and a declared scenario")
    _integer(seed, "seed")
    sizes = _rng(seed, 970001).integers(config.site_min_samples, config.site_max_samples + 1,
                                       size=config.num_sites)
    parameters = (site_gaussian_parameters(config) if scenario == "covariate_shift"
                  else ((0.0, 1.0),) * config.num_sites)
    sites, row_start = [], 0
    for site_id, size in enumerate(sizes):
        mean, std = parameters[site_id]
        x = _rng(seed, 970003, site_id).normal(mean, std, size=(int(size), config.covariate_dim))
        rows = _observations(x, config, seed, site_id, 970010, row_start)
        sites.append(SurvivalSite(site_id, rows))
        row_start += int(size)
    test_x = _rng(seed, 970021).normal(size=(config.num_test, config.covariate_dim))
    evaluation = _observations(test_x, config, seed, 0, 970030, row_start)
    weights = _readonly(sizes / sizes.sum(), np.float64)
    return SurvivalDataset(tuple(sites), evaluation, survival_truth(test_x, config), weights,
                           config, seed, scenario)
