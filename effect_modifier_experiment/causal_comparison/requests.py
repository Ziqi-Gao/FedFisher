"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re

from effect_modifier_experiment.causal_comparison import protocol

PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")

REVIEW_SOURCES = {
    "proposal": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/comparison-registration-proposal.toml",
    "pilot_schema": PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/comparison-pilot-parameters.schema.json",
    "adapter": PROJECT_ROOT / "scripts/fedfisher_server_scheduler_adapter.py",
    "protocol": PROJECT_ROOT / "effect_modifier_experiment/causal_comparison/protocol.py",
    "request_builder": PROJECT_ROOT / "scripts/prepare_comparison_pilot_requests.py",
}

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/comparison.py'
REVIEW_SOURCES['request_publication_support'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py'

REVIEW_SOURCES.update({
    'measurement': PROJECT_ROOT / 'effect_modifier_experiment/components/measurement.py',
    'fingerprints': PROJECT_ROOT / 'effect_modifier_experiment/components/fingerprints.py',
    'recovery_metrics': PROJECT_ROOT / 'effect_modifier_experiment/components/recovery_metrics.py',
    'comparison_computation': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/computation.py',
    'comparison_runner': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/runner.py',
    'comparison_evaluation': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/evaluation.py',
    'comparison_fedavg': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/fedavg.py',
    'comparison_rlearner': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/rlearner.py',
    'comparison_legacy': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/legacy.py',
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


def bundle_name(generation=1):
    return "fedfisher-comparison-v1-pilot-cohort-p%d" % _generation(generation)


def inventory_name(generation=1):
    return bundle_name(generation) + ".inventory.json"


def build_requests(generation=1):
    """Build deterministic scientific specs in memory, without executing fits."""
    generation = _generation(generation)
    specs = []
    for cell in protocol.CELLS:
        for variant in protocol.VARIANTS:
            parameters = dict(variant=variant, cell=cell, shard_index=0, shard_count=1)
            protocol.validate_parameters(protocol.PILOT_TASK, parameters)
            job_id = "fedfisher-comparison-v1-pilot-cohort-p%d-%s-%s-s00of01" % (
                generation, cell.replace("_", "-"), variant.replace("_", "-"))
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
                raise ValueError("generation makes the job identifier exceed protocol limits")
            request = dict(schema_version=2, job_id=job_id, project="FedFisher",
                           task=protocol.PILOT_TASK, parameters=parameters, priority=0)
            configuration = json.loads(_canonical(protocol.run_config(protocol.PILOT_TASK, parameters)))
            specs.append(dict(filename=job_id + ".job-request.json", request=request,
                              request_sha256=_sha256(_encoded(request)),
                              run_id=protocol.run_id(protocol.PILOT_TASK, parameters),
                              configuration=configuration,
                              configuration_sha256=_sha256(_canonical(configuration).encode()),
                              seed_sequence=list(protocol.seed_sequence(protocol.PILOT_TASK, parameters)),
                              splits=list(protocol.SPLITS)))
    expected = {(cell, variant) for cell in protocol.CELLS for variant in protocol.VARIANTS}
    actual = {(item["request"]["parameters"]["cell"], item["request"]["parameters"]["variant"]) for item in specs}
    if len(specs) != 96 or len(expected) != 96 or actual != expected:
        raise ValueError("pilot contract must cover exactly 24 cells by four variants")
    if len({item["filename"] for item in specs}) != len(specs):
        raise ValueError("pilot request identifiers are not unique")
    if any(item["seed_sequence"] != [0] or item["splits"] != ["iid", "noniid"] for item in specs):
        raise ValueError("pilot scope must remain seed zero and both splits")
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
