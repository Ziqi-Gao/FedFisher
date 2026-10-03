"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re

from effect_modifier_experiment.survival import protocol
from effect_modifier_experiment.infrastructure.requests import canonical, encoded, sha256, review_sources as source_bindings

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')

REVIEW_SOURCES = {
    'proposal': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-v1/registration-proposal.toml',
    'pilot_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-v1/pilot-parameters.schema.json',
    'formal_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-v1/shard-parameters.schema.json',
    'adapter': PROJECT_ROOT / 'scripts/fedfisher_server_scheduler_adapter.py',
    'protocol': PROJECT_ROOT / 'effect_modifier_experiment/survival/protocol.py',
    'runner': PROJECT_ROOT / 'effect_modifier_experiment/survival/runner.py',
    'artifacts': PROJECT_ROOT / 'effect_modifier_experiment/survival/artifacts.py',
    'acceptance': PROJECT_ROOT / 'effect_modifier_experiment/survival/acceptance.py',
    'request_builder': PROJECT_ROOT / 'scripts/prepare_survival_requests.py',
    'acceptance_cli': PROJECT_ROOT / 'scripts/validate_survival_pilot.py',
    'experiment_description': PROJECT_ROOT / 'effect_modifier_experiment/survival/README.md',
}

EXPECTED_CELLS = frozenset({'dslasso', 'fedfisher_diag_shallow', 'fedfisher_diag_deep',
                            'fedfisher_kfac_shallow', 'fedfisher_kfac_deep'})

EXPECTED_SCENARIOS = frozenset({'iid', 'covariate_shift'})

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/survival/requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/survival.py'
REVIEW_SOURCES['request_publication_support'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py'

REVIEW_SOURCES.update({
    'survival_data': PROJECT_ROOT / 'effect_modifier_experiment/survival/data.py',
    'survival_pseudo': PROJECT_ROOT / 'effect_modifier_experiment/survival/pseudo.py',
    'survival_links': PROJECT_ROOT / 'effect_modifier_experiment/survival/links.py',
    'survival_dslasso': PROJECT_ROOT / 'effect_modifier_experiment/survival/dslasso.py',
    'survival_fisher': PROJECT_ROOT / 'effect_modifier_experiment/survival/fisher.py',
    'survival_evaluation': PROJECT_ROOT / 'effect_modifier_experiment/survival/evaluation.py',
})


def _generation(generation):
    if type(generation) is not int or generation < 1:
        raise ValueError('generation must be a positive integer')
    return generation


def _cohort(cohort):
    if cohort not in ('pilot', 'formal'):
        raise ValueError('cohort must be pilot or formal')
    return cohort


def bundle_name(generation=1, cohort='pilot'):
    return 'fedfisher-survival-v1-%s-cohort-p%d' % (_cohort(cohort), _generation(generation))


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def build_requests(generation=1, cohort='pilot', shard_count=None):
    """Construct only in-memory specs; writing formal files has a separate gate."""
    _generation(generation)
    pilot = _cohort(cohort) == 'pilot'
    if pilot and shard_count is not None:
        raise ValueError('pilot has no caller-selected shards')
    count = 1 if pilot else (16 if shard_count is None else shard_count)
    if not pilot and (type(count) is not int or count not in (8, 16)):
        raise ValueError('formal shard count must be 8 or 16')
    if (len(protocol.CELLS) != 5 or set(protocol.CELLS) != EXPECTED_CELLS
            or len(protocol.SCENARIOS) != 2 or set(protocol.SCENARIOS) != EXPECTED_SCENARIOS):
        raise ValueError('survival scope must retain five methods and two scenarios')
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    specs = []
    for cell in protocol.CELLS:
        for scenario in protocol.SCENARIOS:
            seen_seeds = []
            for index in range(count):
                parameters = dict(cell=cell, scenario=scenario)
                if not pilot:
                    parameters.update(shard_index=index, shard_count=count)
                protocol.validate_parameters(task, parameters)
                job_id = '%s-%s-%s' % (bundle_name(generation, cohort),
                                       cell.replace('_', '-'), scenario.replace('_', '-'))
                if not pilot:
                    job_id += '-s%02dof%02d' % (index, count)
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', job_id) is None:
                    raise ValueError('invalid or too long scheduler job identifier')
                request = dict(schema_version=2, project='FedFisher', task=task,
                               job_id=job_id, parameters=parameters, priority=0)
                configuration = json.loads(canonical(protocol.run_config(task, parameters)))
                seeds = list(protocol.seed_sequence(task, parameters))
                if seeds != ([0] if pilot else list(range(index, 300, count))):
                    raise ValueError('incorrect deterministic scientific shard coverage')
                seen_seeds.extend(seeds)
                specs.append(dict(filename=job_id + '.job-request.json', request=request,
                                  request_sha256=sha256(encoded(request)),
                                  run_id=protocol.run_id(task, parameters),
                                  configuration=configuration,
                                  configuration_sha256=sha256(canonical(configuration).encode()),
                                  seed_sequence=seeds))
            if sorted(seen_seeds) != ([0] if pilot else list(range(300))):
                raise ValueError('missing or duplicate seeds in a method/scenario cohort')
    if len(specs) != 10 * count or len({item['filename'] for item in specs}) != len(specs):
        raise ValueError('incomplete cohort or duplicate request identities')
    return tuple(specs)


def review_sources():
    return source_bindings(REVIEW_SOURCES, PROJECT_ROOT)
