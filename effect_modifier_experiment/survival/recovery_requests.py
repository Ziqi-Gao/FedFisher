"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re

from effect_modifier_experiment.survival import recovery_protocol as recovery
from effect_modifier_experiment.infrastructure.requests import canonical, encoded, sha256, review_sources as source_bindings

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')

REVIEW_SOURCES = {name: PROJECT_ROOT / path for name, path in {
    'proposal': 'effect_modifier_experiment/server_scheduler/survival-recovery-v1/registration-proposal.toml',
    'diagnostic_schema': 'effect_modifier_experiment/server_scheduler/survival-recovery-v1/diagnostic-parameters.schema.json',
    'formal_schema': 'effect_modifier_experiment/server_scheduler/survival-recovery-v1/shard-parameters.schema.json',
    'adapter': 'scripts/fedfisher_server_scheduler_adapter.py',
    'recovery': 'effect_modifier_experiment/survival/recovery.py',
    'protocol': 'effect_modifier_experiment/survival/recovery_protocol.py',
    'method': 'effect_modifier_experiment/survival/recovery_method.py',
    'dslasso': 'effect_modifier_experiment/survival/dslasso.py',
    'artifacts': 'effect_modifier_experiment/survival/artifacts.py',
    'acceptance': 'effect_modifier_experiment/survival/recovery_acceptance.py',
    'request_builder': 'scripts/prepare_survival_recovery_requests.py',
    'acceptance_cli': 'scripts/validate_survival_recovery.py',
    'description': 'effect_modifier_experiment/survival/README.md',
}.items()}

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/survival/recovery_requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/survival_recovery.py'
REVIEW_SOURCES['request_publication_support'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py'

REVIEW_SOURCES.update({
    'survival_data': PROJECT_ROOT / 'effect_modifier_experiment/survival/data.py',
    'survival_pseudo': PROJECT_ROOT / 'effect_modifier_experiment/survival/pseudo.py',
    'survival_links': PROJECT_ROOT / 'effect_modifier_experiment/survival/links.py',
    'survival_fisher': PROJECT_ROOT / 'effect_modifier_experiment/survival/fisher.py',
    'survival_evaluation': PROJECT_ROOT / 'effect_modifier_experiment/survival/evaluation.py',
})


def bundle_prefix(epoch=1):
    recovery.tasks_for_epoch(epoch)
    return 'fedfisher-survival-dslasso-' + {
        1: 'recovery-v1', 2: 'repair-v2', 3: 'integrated-v3'}[epoch]


def bundle_name(generation=1, cohort='diagnostic', epoch=1):
    if type(generation) is not int or generation < 1 or cohort not in ('diagnostic', 'formal'):
        raise ValueError('invalid recovery generation or cohort')
    return '%s-%s-p%d' % (bundle_prefix(epoch), cohort, generation)


def build_requests(generation=1, cohort='diagnostic', shard_count=None, *, recipe=None, epoch=1):
    """Pure specs only. Formal publication independently replays actual acceptance."""
    prefix = bundle_name(generation, cohort, epoch)
    diagnostic_task, formal_task = recovery.tasks_for_epoch(epoch)
    cases = recovery.cases_for_task(diagnostic_task)
    diagnostic = cohort == 'diagnostic'
    if diagnostic:
        if shard_count is not None or recipe is not None:
            raise ValueError('diagnostic has no shard or selected-recipe override')
        task = diagnostic_task
        parameters = ([dict(case=case, recipe=candidate)
                       for case in cases for candidate in recovery.RECIPES] if epoch == 1
                      else [dict(case=case) for case in cases])
    else:
        count = 16 if shard_count is None else shard_count
        if type(count) is not int or count not in (8, 16) or recipe not in recovery.RECIPES[1:]:
            raise ValueError('formal specs require a candidate recipe and 8/16 deterministic shards')
        if epoch >= 2 and recipe != recovery.REPAIR_RECIPE:
            raise ValueError('repair formal specs require the fixed previously selected recipe')
        task = formal_task
        parameters = [dict(scenario=scenario, shard_index=index, shard_count=count)
                      for scenario in ('iid', 'covariate_shift') for index in range(count)]
    specs, coverage = [], {}
    for p in parameters:
        recovery.validate_parameters(task, p)
        suffix = ((('%s-%s' % (p['case'], p['recipe'])) if epoch == 1 else p['case']) if diagnostic else
                  '%s-s%02dof%02d' % (p['scenario'], p['shard_index'], p['shard_count']))
        job_id = prefix + '-' + suffix.replace('_', '-')
        if re.fullmatch(r'fedfisher-survival-[A-Za-z0-9_.-]{1,109}', job_id) is None:
            raise ValueError('invalid recovery job identifier')
        request = dict(schema_version=2, project='FedFisher', task=task,
                       job_id=job_id, parameters=p, priority=0)
        config = json.loads(canonical(recovery.run_config(task, p, recipe=recipe)))
        seeds = list(recovery.seed_sequence(task, p))
        key = (p['case'], p.get('recipe', recovery.REPAIR_RECIPE)) if diagnostic else p['scenario']
        coverage.setdefault(key, []).extend(seeds)
        specs.append(dict(filename=job_id + '.job-request.json', request=request,
            request_sha256=sha256(encoded(request)), run_id=config['run_id'], configuration=config,
            configuration_sha256=sha256(canonical(config).encode()), seed_sequence=seeds))
    if diagnostic:
        if len(coverage) != len(parameters) or any(seeds != [cases[key[0]]['seed']]
                                     for key, seeds in coverage.items()):
            raise ValueError('diagnostic must cover every fixed case and recipe exactly once')
    elif len(coverage) != 2 or any(sorted(seeds) != list(range(300)) for seeds in coverage.values()):
        raise ValueError('formal recovery must cover exactly 600 cells')
    return tuple(specs)


def review_sources(epoch=1):
    recovery.tasks_for_epoch(epoch)
    sources = dict(REVIEW_SOURCES)
    if epoch >= 2:
        for key in ('proposal', 'diagnostic_schema', 'formal_schema'):
            directory = 'survival-repair-v2' if epoch == 2 else 'survival-integrated-v3'
            sources[key] = sources[key].parent.parent / directory / sources[key].name
    return source_bindings(sources, PROJECT_ROOT)
