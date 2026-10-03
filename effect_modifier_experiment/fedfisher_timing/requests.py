"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re

from effect_modifier_experiment.fedfisher_timing import protocol
from effect_modifier_experiment.infrastructure.requests import canonical as _canonical, encoded as _encoded, sha256 as _sha256, review_sources as _review_sources

PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")

REVIEW_SOURCES = {
    "proposal": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/fedfisher-timing-v1/registration-proposal.toml",
    "pilot_schema": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/fedfisher-timing-v1/pilot-parameters.schema.json",
    "adapter": PROJECT_ROOT / "scripts/fedfisher_server_scheduler_adapter.py",
    "protocol": PROJECT_ROOT / "effect_modifier_experiment/fedfisher_timing/protocol.py",
    "request_builder": PROJECT_ROOT / "scripts/prepare_fedfisher_timing_requests.py",
    "scope_plan": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/fedfisher-timing-v1/timing-plan.json",
}

EXPECTED_CELLS = frozenset({"fedfisher_diag_shallow", "fedfisher_kfac_shallow",
                            "fedfisher_diag_deep", "fedfisher_kfac_deep"})

EXPECTED_VARIANTS = frozenset({"control", "site_gaussian_shift", "local_permutation",
                              "site_gaussian_local_permutation"})

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/fedfisher_timing/requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/fedfisher_timing.py'
REVIEW_SOURCES['request_publication_support'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py'

REVIEW_SOURCES.update({
    'measurement': PROJECT_ROOT / 'effect_modifier_experiment/components/measurement.py',
    'fingerprints': PROJECT_ROOT / 'effect_modifier_experiment/components/fingerprints.py',
    'recovery_metrics': PROJECT_ROOT / 'effect_modifier_experiment/components/recovery_metrics.py',
    'timing_computation': PROJECT_ROOT / 'effect_modifier_experiment/fedfisher_timing/computation.py',
    'timing_fit': PROJECT_ROOT / 'effect_modifier_experiment/fedfisher_timing/fit.py',
    'timing_data': PROJECT_ROOT / 'effect_modifier_experiment/fedfisher_timing/data.py',
    'comparison_evaluation': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/evaluation.py',
})


def _generation(value):
    if type(value) is not int or value < 1:
        raise ValueError("generation must be a positive integer")
    return value


def _cohort(value):
    if value not in ("pilot", "formal"):
        raise ValueError("cohort must be pilot or formal")
    return value


def bundle_name(generation=1, cohort="pilot"):
    return "fedfisher-timing-v1-%s-cohort-p%d" % (_cohort(cohort), _generation(generation))


def inventory_name(generation=1, cohort="pilot"):
    return bundle_name(generation, cohort) + ".inventory.json"


def build_requests(generation=1, cohort="pilot", shard_count=None):
    """Build deterministic scientific specs in memory, without executing fits."""
    generation = _generation(generation)
    cohort = _cohort(cohort)
    pilot = cohort == "pilot"
    if pilot and shard_count is not None:
        raise ValueError("pilot shard count is fixed; shard-count is a formal option")
    shard_count = 1 if pilot else (16 if shard_count is None else shard_count)
    if not pilot and (type(shard_count) is not int or shard_count not in (8, 16)):
        raise ValueError("formal shard count must be 8 or 16")
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    if (len(protocol.CELLS) != 4 or set(protocol.CELLS) != EXPECTED_CELLS
            or len(protocol.VARIANTS) != 4 or set(protocol.VARIANTS) != EXPECTED_VARIANTS):
        raise ValueError("timing pilot must retain exactly the four methods and four legacy variants")
    specs = []
    for cell in protocol.CELLS:
        for variant in protocol.VARIANTS:
            for shard_index in range(shard_count):
                parameters = dict(variant=variant, cell=cell, shard_index=shard_index, shard_count=shard_count)
                protocol.validate_parameters(task, parameters)
                job_id = "%s-%s-%s-s%02dof%02d" % (
                    bundle_name(generation, cohort), cell.replace("_", "-"),
                    variant.replace("_", "-"), shard_index, shard_count)
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
                    raise ValueError("generation makes the job identifier exceed protocol limits")
                request = dict(schema_version=2, job_id=job_id, project="FedFisher",
                               task=task, parameters=parameters, priority=0)
                configuration = json.loads(_canonical(protocol.run_config(task, parameters)))
                specs.append(dict(filename=job_id + ".job-request.json", request=request,
                                  request_sha256=_sha256(_encoded(request)),
                                  run_id=protocol.run_id(task, parameters),
                                  configuration=configuration,
                                  configuration_sha256=_sha256(_canonical(configuration).encode()),
                                  seed_sequence=list(protocol.seed_sequence(task, parameters)),
                                  splits=list(protocol.SPLITS)))
    expected = {(cell, variant) for cell in protocol.CELLS for variant in protocol.VARIANTS}
    actual = {(item["request"]["parameters"]["cell"], item["request"]["parameters"]["variant"]) for item in specs}
    if len(specs) != 16 * shard_count or len(expected) != 16 or actual != expected:
        raise ValueError("pilot contract must cover exactly four method/architecture cells by four variants")
    if len({item["filename"] for item in specs}) != len(specs):
        raise ValueError("pilot request identifiers are not unique")
    for item in specs:
        expected_seeds = ([0] if pilot else list(range(item["request"]["parameters"]["shard_index"], 300, shard_count)))
        if item["seed_sequence"] != expected_seeds or item["splits"] != ["iid", "noniid"]:
            raise ValueError("timing scope must retain seed zero pilot or exact 300-seed partitions and both splits")
    return tuple(specs)


def _source_hashes():
    return _review_sources(REVIEW_SOURCES, PROJECT_ROOT)


def review_sources():
    return _source_hashes()
