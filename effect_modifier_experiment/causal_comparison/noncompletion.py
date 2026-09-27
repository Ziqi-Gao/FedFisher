"""Validate a diagnostic FedAvg cap without claiming scientific completion.

Only the original 200-round validation-patience recipe has this explicit
outcome. Validation replays the recorded stopping and communication evidence;
it does not fit models, evaluate predictions, or authorize any task to continue.
"""

from dataclasses import asdict
import hashlib
import math

import numpy as np

from .artifacts import _hash, _finite_nonnegative, _validate_model_arrays
from .evaluation import model_hash
from .fedavg import FedAvgConfig, _communication_report, _loader_seed, _without_timings
from ..infrastructure.artifacts import canonical, digest


OUTCOME = "validation_patience_limit_200"
MEMORY_SCOPE = "Linux current-process lifetime high-water RSS; not cgroup peak or reservation"


class ScientificNoncompletion(RuntimeError):
    """A validated diagnostic outcome, never a successful scientific result."""

    def __init__(self, fit, arrays):
        super().__init__("method stopped without meeting the declared scientific completion rule: "
                         "maximum_rounds_reached")
        self.fit = fit
        self.arrays = arrays
        self.result = None  # The caller that owns data preparation adds its context.


def _same(actual, expected, label):
    # Canonical comparison distinguishes booleans, integers, and floats.
    if canonical(actual) != canonical(expected):
        raise ValueError("noncompletion %s mismatch" % label)


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("invalid noncompletion %s fields" % label)


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid noncompletion %s integer" % label)


def _recipe(config):
    if isinstance(config, dict):
        config = FedAvgConfig(**config)
    if not isinstance(config, FedAvgConfig):
        raise ValueError("invalid noncompletion FedAvg configuration")
    if type(config.local_epochs) is not int or config.local_epochs not in (1, 5, 10, 30):
        raise ValueError("noncompletion local epochs must be one of 1, 5, 10, 30")
    original = FedAvgConfig(covariate_dim=config.covariate_dim,
                            local_epochs=config.local_epochs)
    _same(asdict(config), asdict(original), "original 200-round patience recipe")
    return config


def _timing_floor(total, parts, label):
    if total < sum(parts) and not math.isclose(total, sum(parts), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("noncompletion %s excludes recorded work" % label)


def validate_capped_fit(audit, config, model_name, seed, arrays):
    """Strictly replay the original patience recipe's exhausted 200 rounds."""
    canonical(audit)  # Reject nonfinite numbers anywhere, even unused evidence.
    try:
        config = _recipe(config)
        _integer(seed, "seed")
        if seed >= 2**63:
            raise ValueError("noncompletion seed is out of range")
        _validate_model_arrays(arrays, dict(family="fedavg", model=model_name), config.covariate_dim)
        fixed = dict(schema_version=1, method="multiround_fedavg_outcome_model",
            model_name=model_name, seed=seed, config=asdict(config), device="cpu", optimizer="SGD",
            momentum_policy="reset_to_zero_at_each_site_at_each_round",
            hyperparameter_policy="prespecified_manifest_bound_not_claimed_tuned",
            holdout_policy="training_only_observed_class_stratified",
            validation_aggregation="sum_site_cross_entropy_losses_divided_by_total_holdout_rows",
            termination_reason="maximum_rounds_reached", completion_eligible=False, converged=False,
            convergence_definition="prespecified_validation_patience_not_stationarity",
            safety_limit_reached=True, refit_rounds=0,
            returned_model_phase="diagnostic_nonconverged_checkpoint",
            training_rounds_executed=200, total_local_epochs_per_site=200 * config.local_epochs)
        variable = {"site_splits", "initial_model_fingerprint", "model_fingerprint", "fit_fingerprint",
            "history_fingerprint", "round_history", "best_selection_round", "best_validation_loss",
            "selected_rounds", "gradient_updates_per_site", "training_examples_seen_per_site",
            "total_gradient_updates", "communication", "timing"}
        _fields(audit, set(fixed) | variable, "fit")
        _same({key: audit[key] for key in fixed}, fixed, "fit identity/status")
        for name in ("initial_model_fingerprint", "model_fingerprint", "fit_fingerprint", "history_fingerprint"):
            _hash(audit[name], name)
        vector = np.concatenate([arrays[name].reshape(-1)
            for index in range(len(arrays) // 2)
            for name in ("net.%d.weight" % (2 * index), "net.%d.bias" % (2 * index))])
        actual_hash = hashlib.sha256(b"fedfisher-multiround-model-v1\0" +
                                     np.asarray(vector, dtype="<f4").tobytes()).hexdigest()
        _same(audit["model_fingerprint"], actual_hash, "diagnostic model")
        sites, history = audit["site_splits"], audit["round_history"]
        if not isinstance(sites, list) or not sites or not isinstance(history, list) or len(history) != 200:
            raise ValueError("noncompletion requires complete site evidence and exactly 200 rounds")
        site_fields = {"site_idx", "num_examples", "label_counts", "split_seed", "fit_examples",
                       "validation_examples", "fit_indices_sha256", "validation_indices_sha256"}
        for index, site in enumerate(sites):
            _fields(site, site_fields, "site split")
            _same(site["site_idx"], index, "site order")
            _integer(site["num_examples"], "site size", 1)
            _fields(site["label_counts"], {"0", "1"}, "site labels")
            for count in site["label_counts"].values():
                _integer(count, "label count")
            _same(sum(site["label_counts"].values()), site["num_examples"], "label total")
            held_out = sum(min(count - 1, max(1, int(round(config.validation_fraction * count))))
                           for count in site["label_counts"].values() if count >= 2)
            if held_out < 1 or held_out >= site["num_examples"]:
                raise ValueError("noncompletion site has no valid stratified holdout")
            _same(site["validation_examples"], held_out, "stratified holdout count")
            _same(site["fit_examples"], site["num_examples"] - held_out, "training count")
            _same(site["split_seed"], config.split_seed_offset + seed + index * 100003, "split seed")
            for name in ("fit_indices_sha256", "validation_indices_sha256"):
                _hash(site[name], name)
        counts = [site["fit_examples"] for site in sites]
        weights = (np.asarray(counts, dtype=np.float64) / sum(counts)).tolist()
        nsites, vector_bytes = len(sites), int(vector.size) * 4
        round_fields = {"phase", "round", "initial_model_fingerprint", "aggregate_model_fingerprint",
            "aggregation_weights", "sites", "aggregation_wall_seconds", "aggregation_cpu_seconds",
            "optimization_payload", "validation_loss", "validation_sites", "validation_wall_seconds",
            "validation_cpu_seconds", "validation_payload", "stale_rounds"}
        local_fields = {"site_idx", "num_examples", "local_epochs", "gradient_updates", "loader_seed",
                       "initial_model_fingerprint", "trained_model_fingerprint",
                       "training_wall_seconds", "training_cpu_seconds"}
        reference, stale, previous = float("inf"), 0, audit["initial_model_fingerprint"]
        for round_index, item in enumerate(history, start=1):
            _fields(item, round_fields, "round")
            _same(item["phase"], "selection", "selection-only phase")
            _same(item["round"], round_index, "round order")
            _same(item["initial_model_fingerprint"], previous, "aggregate rebroadcast")
            _hash(item["aggregate_model_fingerprint"], "round aggregate")
            previous = item["aggregate_model_fingerprint"]
            _same(item["aggregation_weights"], weights, "sample-weighted aggregation")
            if not isinstance(item["sites"], list) or len(item["sites"]) != nsites:
                raise ValueError("noncompletion round omits a training site")
            for index, (entry, count) in enumerate(zip(item["sites"], counts)):
                _fields(entry, local_fields, "local fit")
                expected = dict(site_idx=index, num_examples=count, local_epochs=config.local_epochs,
                    gradient_updates=config.local_epochs * math.ceil(count / config.batch_size),
                    loader_seed=_loader_seed(config, seed, index, round_index, "selection"),
                    initial_model_fingerprint=item["initial_model_fingerprint"])
                _same({key: entry[key] for key in expected}, expected, "local schedule/broadcast")
                _hash(entry["trained_model_fingerprint"], "local model")
                for clock in ("wall", "cpu"):
                    _finite_nonnegative(entry["training_%s_seconds" % clock], "local " + clock)
            _same(item["optimization_payload"], dict(synchronization_phases=1,
                model_upload_count=nsites, model_broadcast_count=nsites,
                site_upload_bytes=nsites * (vector_bytes + 16),
                server_broadcast_bytes=nsites * vector_bytes), "optimization payload")
            _same(item["validation_payload"], dict(synchronization_phases=1,
                summary_upload_count=nsites, model_broadcast_count=nsites,
                site_upload_bytes=nsites * 24, server_broadcast_bytes=nsites * vector_bytes),
                "validation payload")
            val_sites = item["validation_sites"]
            if not isinstance(val_sites, list) or len(val_sites) != nsites:
                raise ValueError("noncompletion round omits a validation site")
            for index, entry in enumerate(val_sites):
                _fields(entry, {"site_idx", "loss_sum", "num_examples"}, "validation site")
                _same(entry["site_idx"], index, "validation site order")
                _same(entry["num_examples"], sites[index]["validation_examples"], "validation rows")
                _finite_nonnegative(entry["loss_sum"], "validation loss sum")
            loss = sum(entry["loss_sum"] for entry in val_sites) / sum(entry["num_examples"] for entry in val_sites)
            _same(item["validation_loss"], loss, "weighted validation loss")
            if loss < reference - config.min_delta:
                reference, stale = loss, 0
            else:
                stale += 1
            _same(item["stale_rounds"], stale, "patience counter")
            if round_index >= config.min_rounds and stale >= config.patience:
                raise ValueError("noncompletion history already satisfied validation patience")
            for stage in ("aggregation", "validation"):
                for clock in ("wall", "cpu"):
                    _finite_nonnegative(item["%s_%s_seconds" % (stage, clock)], stage + " " + clock)
        best = min(range(200), key=lambda index: history[index]["validation_loss"])
        _same(audit["best_selection_round"], best + 1, "first minimum checkpoint")
        _same(audit["selected_rounds"], best + 1, "selected rounds")
        _same(audit["best_validation_loss"], history[best]["validation_loss"], "minimum loss")
        _same(actual_hash, history[best]["aggregate_model_fingerprint"], "returned best checkpoint")
        _same(audit["communication"], _communication_report(history, nsites, int(vector.size)),
              "complete logical communication")
        updates = [200 * config.local_epochs * math.ceil(count / config.batch_size) for count in counts]
        _same(audit["gradient_updates_per_site"], updates, "gradient updates")
        _same(audit["total_gradient_updates"], sum(updates), "total updates")
        _same(audit["training_examples_seen_per_site"], [200 * config.local_epochs * count for count in counts],
              "sample exposure")
        timing = audit["timing"]
        _fields(timing, {"%s_%s_seconds" % (stage, clock)
            for stage in ("preprocessing", "local_training", "aggregation", "validation", "fit")
            for clock in ("wall", "cpu")}, "fit timing")
        for key, value in timing.items():
            _finite_nonnegative(value, key)
        for clock in ("wall", "cpu"):
            _same(timing["local_training_%s_seconds" % clock], sum(
                entry["training_%s_seconds" % clock] for item in history for entry in item["sites"]),
                "local training " + clock)
            for stage in ("aggregation", "validation"):
                _same(timing["%s_%s_seconds" % (stage, clock)], sum(
                    item["%s_%s_seconds" % (stage, clock)] for item in history), stage + " " + clock)
            _timing_floor(timing["fit_%s_seconds" % clock], [timing["%s_%s_seconds" % (stage, clock)]
                for stage in ("preprocessing", "local_training", "aggregation", "validation")], "fit " + clock)
        material = dict(schema_version=1, model_name=model_name, seed=seed, config=asdict(config),
            site_splits=sites, initial_model_fingerprint=audit["initial_model_fingerprint"],
            history=_without_timings(history), returned_model_fingerprint=actual_hash,
            termination_reason=audit["termination_reason"])
        expected_hash = digest(material)
        _same(audit["history_fingerprint"], expected_hash, "deterministic history digest")
        _same(audit["fit_fingerprint"], expected_hash, "fit digest")
    except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("malformed FedAvg noncompletion evidence") from error


def validate_noncompletion_result(result, arrays, identity, config):
    """Validate a saved diagnostic result; evaluation values are forbidden."""
    canonical(result)
    canonical(config)
    try:
        _fields(result, {"identity", "status", "outcome", "fit", "timing", "site_sizes",
                         "label_counts", "fingerprints", "memory"}, "result")
        _fields(identity, {"run_id", "config_sha256", "provenance_sha256", "seed", "split", "cell", "variant"},
                "identity")
        _same(result["identity"], identity, "result identity")
        _same(result["status"], "scientific_noncompletion", "result status")
        _same(result["outcome"], OUTCOME, "outcome")
        for key in ("config_sha256", "provenance_sha256"):
            _hash(identity[key], key)
        _same(identity["config_sha256"], digest(config), "configuration digest")
        for key in ("run_id", "cell", "variant"):
            if not isinstance(identity[key], str) or not identity[key]:
                raise ValueError("invalid noncompletion identity label")
            if key in config:
                _same(identity[key], config[key], key)
        _integer(identity["seed"], "seed")
        bounds = config["seeds"]
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
                or not 0 <= bounds[0] <= identity["seed"] <= bounds[1]
                or identity["split"] not in ("iid", "noniid")
                or identity["split"] not in config["splits"]):
            raise ValueError("noncompletion unit lies outside its manifest")
        method, dgp = config["method"], config["dgp"]
        _fields(method, {"family", "model", "local_epochs", "stopping_mode", "optimizer"}, "method")
        _same(method["family"], "fedavg", "method family")
        _same(method["stopping_mode"], "validation_patience", "method stopping mode")
        _same(method["local_epochs"], method["optimizer"]["local_epochs"], "method local epochs")
        _same(dgp["covariate_dim"], method["optimizer"]["covariate_dim"], "input dimension")
        validate_capped_fit(result["fit"], method["optimizer"], method["model"], identity["seed"], arrays)
        sizes, sites = result["site_sizes"], result["fit"]["site_splits"]
        _integer(dgp["num_sites"], "number of sites", 1)
        if not isinstance(sizes, list) or len(sizes) != dgp["num_sites"]:
            raise ValueError("noncompletion training site count mismatch")
        for size in sizes:
            _integer(size, "site size", 1)
            if not dgp["site_min_samples"] <= size <= dgp["site_max_samples"]:
                raise ValueError("noncompletion site size outside manifest")
        _same(sizes, [site["num_examples"] for site in sites], "fit/data site sizes")
        counts = result["label_counts"]
        _fields(counts, {str(index) for index in range(len(sizes))}, "data label counts")
        for index, size in enumerate(sizes):
            entry = counts[str(index)]
            if not isinstance(entry, dict) or not entry or not set(entry).issubset({"0", "1"}):
                raise ValueError("invalid noncompletion data labels")
            for count in entry.values():
                _integer(count, "data label count")
            _same(sum(entry.values()), size, "data label total")
            _same({str(label): entry.get(str(label), 0) for label in (0, 1)},
                  sites[index]["label_counts"], "fit/data labels")
        fingerprints = result["fingerprints"]
        _fields(fingerprints, {"training", "test", "model"}, "fingerprints")
        for key, value in fingerprints.items():
            _hash(value, key)
        _same(fingerprints["model"], model_hash(arrays), "saved model fingerprint")
        _validate_timer(result["timing"], config["cell"], result["fit"]["timing"])
        memory = result["memory"]
        _fields(memory, {"process_peak_rss_kib", "scope"}, "memory")
        _integer(memory["process_peak_rss_kib"], "process memory", 1)
        _same(memory["scope"], MEMORY_SCOPE, "memory scope")
    except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("malformed scientific noncompletion result") from error


def _validate_timer(timing, cell, fit_timing):
    from .measurement import MethodTimer, STAGES
    template = MethodTimer(cell).to_dict()
    _fields(timing, template, "method timing")
    for key in set(template) - {"stages", "measured_stage_sum"}:
        _same(timing[key], template[key], "timing " + key)
    stages = timing["stages"]
    _fields(stages, STAGES, "timing stages")
    for stage in STAGES:
        entry = stages[stage]
        _fields(entry, template["stages"][stage], "timing " + stage)
        measured = stage in ("data_preparation", "train")
        for key, expected in dict(measured=measured, calls=int(measured), failed_calls=0, components={}).items():
            _same(entry[key], expected, stage + " " + key)
        for clock in ("wall_seconds", "cpu_seconds"):
            _finite_nonnegative(entry[clock], stage + " " + clock)
            if not measured:
                _same(entry[clock], 0.0, "unmeasured " + stage + " " + clock)
        if measured and entry["wall_seconds"] <= 0:
            raise ValueError("noncompletion stage has no elapsed wall time")
    _same(timing["measured_stage_sum"], {clock: sum(stages[stage][clock] for stage in STAGES)
        for clock in ("wall_seconds", "cpu_seconds")}, "stage totals")
    for clock in ("wall", "cpu"):
        _timing_floor(stages["train"][clock + "_seconds"], [fit_timing["fit_" + clock + "_seconds"]],
                      "full method " + clock)
