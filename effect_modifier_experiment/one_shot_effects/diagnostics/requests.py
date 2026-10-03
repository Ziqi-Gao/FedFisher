"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re

from effect_modifier_experiment.one_shot_effects.diagnostics import protocol
from effect_modifier_experiment.one_shot_effects.diagnostics.recovery import REPLACED_CELLS

PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")

DIAGNOSTICS = PROJECT_ROOT / "effect_modifier_experiment/one_shot_effects/diagnostics"

PROPOSAL = PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/effects-replay-v1"

REVIEW_SOURCES = {
    "proposal": PROPOSAL / "registration-proposal.toml",
    "pilot_schema": PROPOSAL / "pilot-parameters.schema.json",
    "adapter": PROJECT_ROOT / "scripts/fedfisher_server_scheduler_adapter.py",
    "protocol": DIAGNOSTICS / "protocol.py",
    "source_bindings": DIAGNOSTICS / "source_bindings.json",
    "source_validator": DIAGNOSTICS / "sources.py",
    "request_builder": PROJECT_ROOT / "scripts/prepare_effects_replay_requests.py",
    "recovery_validator": DIAGNOSTICS / "recovery.py",
}

EXPECTED_CELLS = frozenset({
    ("fisher_shallow", "scmv2_base"), ("fisher_deep", "scmv2_base"),
    ("fisher_shallow", "legacy_control_iid"), ("fisher_deep", "legacy_control_iid"),
    ("dr_path", "scmv2_base"),
})

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/one_shot_effects/diagnostics/requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/effects_replay.py'
REVIEW_SOURCES['request_publication_support'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py'

REVIEW_SOURCES.update({
    'measurement': PROJECT_ROOT / 'effect_modifier_experiment/components/measurement.py',
    'fingerprints': PROJECT_ROOT / 'effect_modifier_experiment/components/fingerprints.py',
    'recovery_metrics': PROJECT_ROOT / 'effect_modifier_experiment/components/recovery_metrics.py',
    'data_composition': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_effects/data_composition.py',
    'effect_prerequisites': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_effects/prerequisites.py',
    'effect_evaluation': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_effects/evaluation.py',
    'replay_runner': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_effects/diagnostics/runner.py',
    'replay_fisher': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_effects/diagnostics/fisher.py',
    'replay_dr': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_effects/diagnostics/dr.py',
})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _generation(value):
    if type(value) is not int or value < 1:
        raise ValueError("generation must be a positive integer")
    return value


def _cohort(value):
    if value not in ("full", "deep-recovery"):
        raise ValueError("cohort must be full or deep-recovery")
    return value


def bundle_name(generation=1, cohort="full"):
    name = "pilot" if _cohort(cohort) == "full" else "deep-recovery"
    return "fedfisher-effects-replay-v1-%s-cohort-p%d" % (name, _generation(generation))


def inventory_name(generation=1, cohort="full"):
    return bundle_name(generation, cohort) + ".inventory.json"


def build_requests(generation=1, cohort="full"):
    """Build the reviewed science-only requests without touching checkpoints."""
    generation = _generation(generation)
    cohort = _cohort(cohort)
    if cohort == "deep-recovery" and generation < 3:
        raise ValueError("deep recovery requires generation 3 or later")
    plan = protocol.cohort_plan()
    if (len(plan) != 5 or {(item["recipe"], item["dataset"]) for item in plan} != EXPECTED_CELLS):
        raise ValueError("replay contract must cover exactly the five reviewed cells")
    expected = EXPECTED_CELLS if cohort == "full" else REPLACED_CELLS
    specs = []
    for parameters in plan:
        parameters = protocol.validate_parameters(protocol.PILOT_TASK, parameters)
        if (parameters["recipe"], parameters["dataset"]) not in expected:
            continue
        job_id = "%s-%s-%s" % (
            bundle_name(generation, cohort), parameters["recipe"].replace("_", "-"),
            parameters["dataset"].replace("_", "-"))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
            raise ValueError("generation makes the job identifier exceed protocol limits")
        request = dict(schema_version=2, job_id=job_id, project="FedFisher",
                       task=protocol.PILOT_TASK, parameters=parameters, priority=0)
        configuration = json.loads(_canonical(protocol.run_config(protocol.PILOT_TASK, parameters)))
        if (type(configuration.get("seed")) is not int or configuration["seed"] != 0
                or configuration.get("fitting_allowed") is not False
                or configuration.get("training_tasks") != 0
                or configuration.get("expansion_authorized") is not False):
            raise ValueError("replay scope must remain seed zero, no fitting and no expansion")
        specs.append(dict(filename=job_id + ".job-request.json", request=request,
                          request_sha256=_sha256(_encoded(request)),
                          run_id=protocol.run_id(protocol.PILOT_TASK, parameters),
                          configuration=configuration,
                          configuration_sha256=_sha256(_canonical(configuration).encode())))
    actual = {(item["request"]["parameters"]["recipe"],
               item["request"]["parameters"]["dataset"]) for item in specs}
    if len(specs) != len(expected) or actual != expected:
        raise ValueError("replay contract must cover exactly the reviewed cohort")
    if len({item["filename"] for item in specs}) != len(expected):
        raise ValueError("replay request identifiers are not unique")
    return tuple(specs)


def _source_hashes():
    records = {}
    for name, path in REVIEW_SOURCES.items():
        if PROJECT_ROOT not in path.parents or path.resolve(strict=True) != path or any(
                item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("review source must be an ordinary canonical project file")
        records[name] = dict(path=str(path), sha256=_sha256(path.read_bytes()))
    return records


def review_sources():
    return _source_hashes()
