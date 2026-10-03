"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re


PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")

PROJECT = "FedFisher"

EPOCH = "v2r2"

VARIANTS = (
    "control",
    "site_gaussian_shift",
    "local_permutation",
    "site_gaussian_local_permutation",
)

TASKS = {
    "dslasso": {
        "task": "effect-modifier-dslassocov-shard",
        "shard_count": 4,
        "pilot_shards": {variant: 0 for variant in VARIANTS},
    },
    "one-shot": {
        "task": "effect-modifier-one-shot-lasso-shard",
        "shard_count": 16,
        "pilot_shards": {
            "control": 6,
            "site_gaussian_shift": 13,
            "local_permutation": 6,
            "site_gaussian_local_permutation": 13,
        },
    },
}

REVIEW_SOURCES = {
    'adapter': PROJECT_ROOT / 'scripts/fedfisher_server_scheduler_adapter.py',
    'request_builder': PROJECT_ROOT / 'scripts/prepare_fedfisher_scheduler_requests.py',
}

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/original.py'
REVIEW_SOURCES['request_publication_support'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py'

REVIEW_SOURCES.update({
    'dslasso_invariants': PROJECT_ROOT / 'effect_modifier_experiment/dslassocov/invariants.py',
    'one_shot_invariants': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_lasso/invariants.py',
    'dslasso_benchmark': PROJECT_ROOT / 'effect_modifier_experiment/dslassocov_benchmark.py',
    'dslasso_runner': PROJECT_ROOT / 'effect_modifier_experiment/run_dslassocov_effect_modifier.py',
    'one_shot_benchmark': PROJECT_ROOT / 'effect_modifier_experiment/one_shot_lasso_benchmark.py',
    'one_shot_runner': PROJECT_ROOT / 'effect_modifier_experiment/run_one_shot_lasso_effect_modifier.py',
})


def _job_id(task_label, variant, phase, generation, shard_index, shard_count):
    variant_label = variant.replace("_", "-")
    return (
        "fedfisher-%s-%s-%s-%s-%s-p%d-shard-%02d-of-%02d"
        % (
            task_label,
            variant_label,
            EPOCH,
            phase,
            "cohort",
            generation,
            shard_index,
            shard_count,
        )
    )


def build_requests(phase, generation=1):
    if phase not in {"pilot", "remaining"}:
        raise ValueError("phase must be pilot or remaining")
    if isinstance(generation, bool) or int(generation) < 1:
        raise ValueError("generation must be a positive integer")
    generation = int(generation)
    requests = []
    for task_label, contract in TASKS.items():
        shard_count = contract["shard_count"]
        for variant in VARIANTS:
            pilot_index = contract["pilot_shards"][variant]
            if phase == "pilot":
                shard_indices = (pilot_index,)
            else:
                shard_indices = tuple(
                    index for index in range(shard_count) if index != pilot_index
                )
            for shard_index in shard_indices:
                job_id = _job_id(
                    task_label,
                    variant,
                    phase,
                    generation,
                    shard_index,
                    shard_count,
                )
                payload = {
                    "schema_version": 2,
                    "job_id": job_id,
                    "project": PROJECT,
                    "task": contract["task"],
                    "parameters": {
                        "variant": variant,
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                    },
                    "priority": 0,
                }
                requests.append((job_id + ".job-request.json", payload))
    return tuple(requests)


def _encoded(payload):
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def review_sources():
    from effect_modifier_experiment.infrastructure.requests import review_sources as bindings
    return bindings(REVIEW_SOURCES, PROJECT_ROOT)
