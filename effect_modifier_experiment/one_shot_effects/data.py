"""Independent SCMV2 generation and an unchanged-legacy data adapter.

No entry point fits a model. The common T/V/H split uses only row count,
experiment seed and site identifier. Mechanism truth is retained exclusively
in the independent evaluation object, never in a site's observed rows.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from types import SimpleNamespace

import numpy as np

from .types import EvaluationData, EvaluationTruth, ExperimentData, Rows, SitePartitions


SCENARIOS = ("base", "covariate_shift", "correlated_shift", "interaction", "null")
DATASET_ID = "SyntheticEffectModifierSCMV2"
PARTITION_SEED_STREAM = 8300009
PARTITION_RULE = "independent permutation; floor(0.70*n) T, floor(0.15*n) V, remainder H"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))


def _rng(seed, stream, site_id=0):
    _integer(seed, "seed")
    return np.random.default_rng(np.random.SeedSequence([seed, stream, site_id]))


@dataclass(frozen=True)
class SCMConfig:
    num_sites: int = 10
    site_min_samples: int = 500
    site_max_samples: int = 1000
    covariate_dim: int = 100
    num_test: int = 10000
    site_mean_span: float = 0.75
    site_mean_design: str = "scalar_linspace_v1"
    ar1_rho: float = 0.5
    dataset_id: str = DATASET_ID

    def __post_init__(self):
        for name, minimum in (("num_sites", 1), ("site_min_samples", 7),
                              ("site_max_samples", 7), ("covariate_dim", 5), ("num_test", 1)):
            _integer(getattr(self, name), name, minimum)
        if self.site_max_samples < self.site_min_samples:
            raise ValueError("site size bounds are reversed")
        if isinstance(self.site_mean_span, bool) or not isinstance(self.site_mean_span, (int, float)):
            raise ValueError("site_mean_span must be finite and nonnegative")
        if not math.isfinite(self.site_mean_span) or self.site_mean_span < 0:
            raise ValueError("site_mean_span must be finite and nonnegative")
        if self.site_mean_design != "scalar_linspace_v1" or self.ar1_rho != 0.5:
            raise ValueError("SCMV2 fixes its site mean design and AR(1) correlation")
        if self.dataset_id != DATASET_ID:
            raise ValueError("SCMV2 must retain its independent dataset identity")


@dataclass(frozen=True)
class LegacyConfig:
    """The original generator's defaults, serialized independently of SCMV2."""

    num_sites: int = 10
    alpha: float = 0.1
    num_test: int = 10000
    covariate_dim: int = 100
    signal_dim: int = 2
    signal_strength: float = 2.0
    main_effect_strength: float = 1.0
    confounding_strength: float = 1.0
    outcome_intercept: float = 0.0
    treatment_prob: float = 0.5
    propensity_clip: float = 0.05
    site_min_samples: int = 500
    site_max_samples: int = 1000
    site_gaussian_mean_span: float = 0.75
    site_gaussian_std_low: float = 0.75
    site_gaussian_std_high: float = 1.25

    def __post_init__(self):
        for name, minimum in (("num_sites", 1), ("site_min_samples", 7),
                              ("site_max_samples", 7), ("covariate_dim", 1),
                              ("signal_dim", 1), ("num_test", 1)):
            _integer(getattr(self, name), name, minimum)
        if self.site_max_samples < self.site_min_samples or self.signal_dim > self.covariate_dim:
            raise ValueError("invalid legacy dimensions")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("legacy config values must be finite numbers: " + name)
        if self.alpha <= 0 or not 0 < self.treatment_prob < 1 or not 0 < self.propensity_clip < 0.5:
            raise ValueError("invalid legacy treatment/partition settings")
        if self.site_gaussian_mean_span < 0 or not 0 < self.site_gaussian_std_low <= self.site_gaussian_std_high:
            raise ValueError("invalid legacy Gaussian settings")


def partition_rows(rows, site_id, seed):
    """Assign T/V/H without examining X, A or Y; use the same call for every method."""
    if type(rows) is not Rows:
        raise ValueError("partitioning accepts observed Rows only")
    _integer(site_id, "site_id")
    order = _rng(seed, PARTITION_SEED_STREAM, site_id).permutation(len(rows))
    n_train, n_validation = 70 * len(rows) // 100, 15 * len(rows) // 100
    if min(n_train, n_validation, len(rows) - n_train - n_validation) < 1:
        raise ValueError("site too small for nonempty 70/15/15 partitions")
    cuts = (order[:n_train], order[n_train:n_train + n_validation], order[n_train + n_validation:])
    blocks = [Rows(rows.x[index], rows.a[index], rows.y[index], rows.row_ids[index]) for index in cuts]
    return SitePartitions(site_id, len(rows), *blocks)


def fixed_site_means(config):
    """Predeclared m_s: linspace(-span,+span) broadcast to every raw coordinate.

    A one-site configuration uses zero. No realized X/A/Y enters this design.
    """
    levels = np.linspace(-config.site_mean_span, config.site_mean_span, config.num_sites)
    if config.num_sites == 1:
        levels[0] = 0.0
    return np.repeat(levels[:, None], config.covariate_dim, axis=1)


def _covariates(rng, n, config, mean, correlated):
    values = rng.normal(size=(n, config.covariate_dim))
    if correlated:
        rho = config.ar1_rho
        for j in range(1, config.covariate_dim):
            values[:, j] = rho * values[:, j - 1] + np.sqrt(1 - rho ** 2) * values[:, j]
    return values + mean


def _mechanism(x, scenario):
    sigmoid = lambda value: np.exp(-np.logaddexp(0.0, -value))
    g = 0.8 * np.tanh(x[:, 2]) + 0.5 * np.tanh(x[:, 3])
    baseline = 0.25 + 0.5 * sigmoid(g)
    propensity = np.clip(sigmoid(0.8 * x[:, 2] - 0.6 * x[:, 4]), 0.05, 0.95)
    if scenario == "null":
        tau = np.zeros(len(x))
    elif scenario == "interaction":
        tau = 0.2 * np.tanh(x[:, 0]) * np.tanh(x[:, 1])
    else:
        tau = 0.1 * np.sin(x[:, 0]) + 0.1 * np.tanh(x[:, 1])
    mu_0, mu_1 = baseline - tau / 2, baseline + tau / 2
    if not all(np.isfinite(v).all() for v in (propensity, mu_0, mu_1, tau)):
        raise ValueError("nonfinite SCM mechanism")
    if np.any(mu_0 < 0.15) or np.any(mu_0 > 0.85) or np.any(mu_1 < 0.15) or np.any(mu_1 > 0.85):
        raise ValueError("SCMV2 outcome means are invalid; clipping is forbidden")
    return propensity, mu_0, mu_1, tau


def _observations(x, seed, a_stream, y_stream, site_id, scenario, row_start):
    e, mu_0, mu_1, _ = _mechanism(x, scenario)
    a = _rng(seed, a_stream, site_id).binomial(1, e)
    y = _rng(seed, y_stream, site_id).binomial(1, np.where(a == 1, mu_1, mu_0))
    return Rows(x, a, y, np.arange(row_start, row_start + len(x), dtype=np.int64))


def generate_scm(config, seed, scenario="base"):
    if type(config) is not SCMConfig or scenario not in SCENARIOS:
        raise ValueError("SCMV2 requires SCMConfig and a predeclared scenario")
    sizes = _rng(seed, 1900001).integers(config.site_min_samples, config.site_max_samples + 1, size=config.num_sites)
    weights = sizes.astype(np.float64) / sizes.sum()
    shifted, correlated = scenario in ("covariate_shift", "correlated_shift"), scenario == "correlated_shift"
    means = fixed_site_means(config) if shifted else np.zeros((config.num_sites, config.covariate_dim))
    sites, row_start = [], 0
    for site_id, size in enumerate(sizes):
        x = _covariates(_rng(seed, 1900013, site_id), int(size), config, means[site_id], correlated)
        rows = _observations(x, seed, 1900021, 1900027, site_id, scenario, row_start)
        sites.append(partition_rows(rows, site_id, seed))
        row_start += int(size)
    # Independent mixture and innovations: changing num_test cannot alter sites.
    site_ids = _rng(seed, 1900033).choice(config.num_sites, size=config.num_test, p=weights)
    test_x = _covariates(_rng(seed, 1900039), config.num_test, config, means[site_ids], correlated)
    test_rows = _observations(test_x, seed, 1900043, 1900049, 0, scenario, row_start)
    mask = np.zeros(config.covariate_dim, dtype=bool)
    if scenario != "null":
        mask[:2] = True
    roles = ["noise"] * config.covariate_dim
    if scenario != "null":
        roles[:2] = ["modifier", "modifier"]
    roles[2:5] = ["prognostic_and_confounder", "prognostic", "treatment_only"]
    truth = EvaluationTruth(*_mechanism(test_x, scenario), mask, tuple(roles))
    return ExperimentData(tuple(sites), EvaluationData(test_rows, site_ids, weights, truth))


def generate_legacy(config, seed, variant="control", split="iid"):
    """Keep original site/test rows exactly; add only the new common T/V/H split."""
    if type(config) is not LegacyConfig:
        raise ValueError("legacy compatibility requires LegacyConfig")
    _integer(seed, "seed")
    if split not in ("iid", "noniid"):
        raise ValueError("unknown legacy split")
    from ..run_dslassocov_effect_modifier import generate_datasets
    datasets, test, _ = generate_datasets(SimpleNamespace(**asdict(config), seed=seed, variant=variant, split=split))
    sites, row_start = [], 0
    for site_id, dataset in enumerate(datasets):
        design, outcome = (tensor.detach().cpu().numpy() for tensor in dataset.tensors)
        rows = Rows(design[:, 1:1 + config.covariate_dim], design[:, 0], outcome,
                    np.arange(row_start, row_start + len(design), dtype=np.int64))
        sites.append(partition_rows(rows, site_id, seed))
        row_start += len(rows)
    design, outcome = (tensor.detach().cpu().numpy() for tensor in test.tensors)
    external = Rows(design[:, 1:1 + config.covariate_dim], design[:, 0], outcome,
                    np.arange(row_start, row_start + len(design), dtype=np.int64))
    sizes = np.array([site.n_total for site in sites], dtype=np.float64)
    # Legacy's independent pooled test definition has no target-mixture site IDs
    # and is intentionally not relabeled as an SCMV2 truth experiment.
    return ExperimentData(tuple(sites), EvaluationData(external, None, sizes / sizes.sum(), None))


def rows_fingerprint(rows):
    """Hash observed values and identifiers only, never evaluator truth."""
    if type(rows) is not Rows:
        raise ValueError("observed Rows required")
    digest = hashlib.sha256(b"one-shot-effects-observed-rows-v1\0")
    for name in ("x", "a", "y", "row_ids"):
        values = np.asarray(getattr(rows, name), dtype="<i8" if name == "row_ids" else "<f8", order="C")
        digest.update(str(values.shape).encode("ascii") + b"\0" + values.tobytes())
    return digest.hexdigest()


def partition_fingerprint(site):
    """Only assignment IDs/site sizes; independent of observed A/Y values."""
    digest = hashlib.sha256(b"one-shot-effects-partitions-v1\0")
    digest.update(np.asarray([site.site_id, site.n_total], dtype="<i8").tobytes())
    for rows in (site.train, site.validation, site.holdout):
        digest.update(np.asarray([len(rows)], dtype="<i8").tobytes())
        digest.update(np.asarray(rows.row_ids, dtype="<i8").tobytes())
    return digest.hexdigest()
