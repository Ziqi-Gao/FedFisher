"""Pure deterministic MLP8 request specifications and source review bindings."""
from pathlib import Path

from . import protocol
from ..infrastructure.requests import canonical, encoded, sha256, review_sources as source_bindings
from ..scheduling.catalog import adapter_review_sources

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
REVIEW_SOURCES = adapter_review_sources()
for name in ('__init__', 'protocol', 'requests', 'methods', 'computation', 'artifacts', 'runner', 'acceptance', 'README'):
    REVIEW_SOURCES['mlp8_' + name] = PROJECT_ROOT / ('effect_modifier_experiment/survival_mlp8/' + name + ('.md' if name == 'README' else '.py'))
for name in ('data', 'pseudo', 'links', 'fisher', 'artifacts', 'acceptance'):
    REVIEW_SOURCES['survival_' + name] = PROJECT_ROOT / ('effect_modifier_experiment/survival/' + name + '.py')
for name in ('inputs', 'evaluation'):
    REVIEW_SOURCES['paired_' + name] = PROJECT_ROOT / ('effect_modifier_experiment/survival_pooled/' + name + '.py')
for name in ('artifacts', 'provenance', 'requests', 'evidence'):
    REVIEW_SOURCES['infrastructure_' + name] = PROJECT_ROOT / ('effect_modifier_experiment/infrastructure/' + name + '.py')
REVIEW_SOURCES.update({
    'proposal': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-mlp8-v1/registration-proposal.toml',
    'pilot_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-mlp8-v1/pilot-parameters.schema.json',
    'formal_schema': PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-mlp8-v1/shard-parameters.schema.json',
    'adapter': PROJECT_ROOT / 'scripts/fedfisher_server_scheduler_adapter.py',
    'request_publication': PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/survival_mlp8.py',
    'request_plan': PROJECT_ROOT / 'effect_modifier_experiment/survival_mlp8/requests.py',
    'publication_support': PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py',
    'prepare_cli': PROJECT_ROOT / 'scripts/prepare_survival_mlp8_requests.py',
    'acceptance_cli': PROJECT_ROOT / 'scripts/validate_survival_mlp8.py',
})


def bundle_name(generation=1, cohort='pilot', *, discovery=False):
    if type(generation) is not int or generation < 1 or cohort not in ('pilot', 'formal'):
        raise ValueError('positive generation and pilot/formal cohort required')
    if type(discovery) is not bool:
        raise ValueError('discovery recipe selector must be boolean')
    return 'fedfisher-survival-mlp8-v%d-%s-cohort-p%d' % (2 if discovery else 1, cohort, generation)


def inventory_name(generation=1, cohort='pilot', *, discovery=False):
    return bundle_name(generation, cohort, discovery=discovery) + '.inventory.json'


def build_requests(generation=1, cohort='pilot', shard_count=None, *, discovery=False):
    if discovery:
        from . import discovery_protocol as recipe
    else:
        recipe = protocol
    bundle = bundle_name(generation, cohort, discovery=discovery)
    pilot = cohort == 'pilot'
    if pilot and shard_count is not None:
        raise ValueError('pilot does not accept shard choices')
    count = 1 if pilot else (16 if shard_count is None else shard_count)
    if type(count) is not int or count not in ((1,) if pilot else (8, 16)):
        raise ValueError('formal shard count must be 8 or 16')
    specs = []
    for scenario in recipe.SCENARIOS:
        seen = []
        for index in range(count):
            p = dict(scenario=scenario)
            if not pilot:
                p.update(shard_index=index, shard_count=count)
            task = recipe.PILOT_TASK if pilot else recipe.FORMAL_TASK
            job_id = bundle + '-' + scenario.replace('_', '-')
            if not pilot:
                job_id += '-s%02dof%02d' % (index, count)
            if len(job_id) > 128:
                raise ValueError('generation exceeds the job identifier limit')
            request = dict(schema_version=2, project='FedFisher', task=task, job_id=job_id, parameters=p, priority=0)
            config, seeds = recipe.run_config(task, p), list(recipe.seed_sequence(task, p))
            seen.extend(seeds)
            specs.append(dict(filename=job_id + '.job-request.json', request=request,
                request_sha256=sha256(encoded(request)), run_id=config['run_id'], configuration=config,
                configuration_sha256=sha256(canonical(config).encode()), seed_sequence=seeds))
        if sorted(seen) != list(recipe.PILOT_SEEDS if pilot else recipe.FORMAL_SEEDS):
            raise ValueError('missing or duplicated MLP8 seed coverage')
    if len(specs) != 2 * count or len({s['filename'] for s in specs}) != len(specs):
        raise ValueError('invalid MLP8 request coverage')
    return tuple(specs)


def review_sources(*, discovery=False):
    if not discovery:
        return source_bindings(REVIEW_SOURCES, PROJECT_ROOT)
    sources = dict(REVIEW_SOURCES)
    for name in ('discovery_protocol', 'discovery_evaluation'):
        sources[name] = PROJECT_ROOT / ('effect_modifier_experiment/survival_mlp8/' + name + '.py')
    sources['survival_discovery'] = PROJECT_ROOT / 'effect_modifier_experiment/survival/evaluation.py'
    directory = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-mlp8-v2'
    for label, name in [('proposal', 'registration-proposal.toml'), ('pilot_schema', 'pilot-parameters.schema.json'), ('formal_schema', 'shard-parameters.schema.json')]:
        sources[label] = directory / name
    return source_bindings(sources, PROJECT_ROOT)
