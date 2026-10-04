"""Pure scientific request plans; publication and admission live separately."""
import json
from pathlib import Path
import re

from . import protocol
from ..infrastructure.requests import canonical, encoded, sha256, review_sources as source_bindings

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
REVIEW_SOURCES = {
    'proposal': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-pooled-v1/registration-proposal.toml',
    'pilot_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-pooled-v1/pilot-parameters.schema.json',
    'formal_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-pooled-v1/shard-parameters.schema.json',
    'adapter': PROJECT_ROOT / 'scripts/fedfisher_server_scheduler_adapter.py',
    'request_builder': PROJECT_ROOT / 'scripts/prepare_survival_pooled_requests.py',
    'acceptance_cli': PROJECT_ROOT / 'scripts/validate_survival_pooled.py',
    'request_publication': PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/survival_pooled.py',
    'request_publication_support': PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py',
    'report': PROJECT_ROOT / 'reporting/effect_modifier/survival_pooled.py',
    'report_cli': PROJECT_ROOT / 'scripts/make_survival_pooled_results.py',
}
for _name in ('protocol', 'requests', 'runner', 'artifacts', 'acceptance', 'inputs',
              'methods', 'computation', 'evaluation', '__init__'):
    REVIEW_SOURCES['pooled_' + _name] = PROJECT_ROOT / (
        'effect_modifier_experiment/survival_pooled/' + _name + '.py')
for _name in ('adapter', 'catalog', 'validation', 'runtime', 'dispatch', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _name] = PROJECT_ROOT / (
        'effect_modifier_experiment/scheduling/' + _name + '.py')
for _name in ('data', 'pseudo', 'links', 'dslasso', 'fisher', 'recovery_method', 'artifacts'):
    REVIEW_SOURCES['survival_' + _name] = PROJECT_ROOT / (
        'effect_modifier_experiment/survival/' + _name + '.py')


def _generation(generation):
    if type(generation) is not int or generation < 1:
        raise ValueError('generation must be a positive integer')
    return generation


def _cohort(cohort):
    if cohort not in ('pilot', 'formal'):
        raise ValueError('cohort must be pilot or formal')
    return cohort


def bundle_name(generation=1, cohort='pilot'):
    return 'fedfisher-survival-pooled-v1-%s-cohort-p%d' % (
        _cohort(cohort), _generation(generation))


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def build_requests(generation=1, cohort='pilot', shard_count=None):
    """Construct six bounded pilots or exact disjoint 300-seed formal shards."""
    _generation(generation)
    pilot = _cohort(cohort) == 'pilot'
    if pilot and shard_count is not None:
        raise ValueError('pilot has no caller-selected shards')
    count = 1 if pilot else (16 if shard_count is None else shard_count)
    if not pilot and (type(count) is not int or count not in (8, 16)):
        raise ValueError('formal shard count must be 8 or 16')
    specs = []
    for family in protocol.FAMILIES:
        for scenario in protocol.SCENARIOS:
            seen = []
            for index in range(count):
                p = dict(family=family, scenario=scenario)
                if not pilot:
                    p.update(shard_index=index, shard_count=count)
                task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
                protocol.validate_parameters(task, p)
                job_id = '%s-%s-%s' % (bundle_name(generation, cohort), family,
                                       scenario.replace('_', '-'))
                if not pilot:
                    job_id += '-s%02dof%02d' % (index, count)
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', job_id) is None:
                    raise ValueError('invalid scheduler job identifier')
                request = dict(schema_version=2, project='FedFisher', task=task,
                               job_id=job_id, parameters=p, priority=0)
                config = json.loads(canonical(protocol.run_config(task, p)))
                seeds = list(protocol.seed_sequence(task, p))
                expected = list(protocol.PILOT_SEEDS) if pilot else list(range(index, 300, count))
                if seeds != expected:
                    raise ValueError('incorrect pooled survival seed partition')
                seen.extend(seeds)
                specs.append(dict(filename=job_id + '.job-request.json', request=request,
                                  request_sha256=sha256(encoded(request)),
                                  run_id=protocol.run_id(task, p), configuration=config,
                                  configuration_sha256=sha256(canonical(config).encode()),
                                  seed_sequence=seeds))
            if sorted(seen) != (list(protocol.PILOT_SEEDS) if pilot else list(range(300))):
                raise ValueError('missing or duplicate pooled survival seeds')
    if len(specs) != 6 * count or len({x['filename'] for x in specs}) != len(specs):
        raise ValueError('incomplete or duplicate pooled survival requests')
    return tuple(specs)


def review_sources():
    return source_bindings(REVIEW_SOURCES, PROJECT_ROOT)
