"""Twenty bounded scientific requests; no 300-seed expansion exists."""
from pathlib import Path

from ..infrastructure.requests import canonical, encoded, sha256, review_sources as source_bindings
from ..scheduling.catalog import adapter_review_sources
from . import protocol

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')


def bundle_name(generation=1):
    if type(generation) is not int or generation < 1:
        raise ValueError('positive comparator request generation required')
    return 'fedfisher-survival-comparators-v1-10seed-cohort-p%d' % generation


def build_requests(generation=1):
    bundle, specs = bundle_name(generation), []
    for scenario in protocol.SCENARIOS:
        for seed in protocol.SEEDS:
            parameters = dict(scenario=scenario, seed=seed)
            job_id = '%s-%s-seed%03d' % (bundle, scenario.replace('_', '-'), seed)
            if len(job_id) > 128:
                raise ValueError('comparator request identifier too long')
            request = dict(schema_version=2, project='FedFisher', task=protocol.PILOT_TASK,
                           job_id=job_id, parameters=parameters, priority=0)
            config = protocol.run_config(protocol.PILOT_TASK, parameters)
            specs.append(dict(filename=job_id + '.job-request.json', request=request,
                request_sha256=sha256(encoded(request)), run_id=config['run_id'], configuration=config,
                configuration_sha256=sha256(canonical(config).encode()), seed_sequence=[seed]))
    return tuple(specs)


def review_sources():
    sources = adapter_review_sources()
    for name in ('__init__', 'protocol', 'inputs', 'fedavg', 'computation', 'artifacts', 'runner', 'requests', 'acceptance'):
        sources['comparators_' + name] = PROJECT_ROOT / ('effect_modifier_experiment/survival_comparators/' + name + '.py')
    sources['comparators_readme'] = PROJECT_ROOT / 'effect_modifier_experiment/survival_comparators/README.md'
    sources['proposal'] = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-comparators-v1/registration-proposal.toml'
    sources['schema'] = PROJECT_ROOT / 'effect_modifier_experiment/server_scheduler/survival-comparators-v1/pilot-parameters.schema.json'
    for name in ('prepare_survival_comparators_requests', 'validate_survival_comparators'):
        sources[name] = PROJECT_ROOT / ('scripts/' + name + '.py')
    sources['publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/survival_comparators.py'
    for package, names in (
        ('survival', ('data', 'pseudo', 'links', 'evaluation', 'artifacts', 'acceptance')),
        ('survival_mlp8', ('methods', 'artifacts', 'discovery_evaluation', 'protocol', 'discovery_protocol')),
        ('survival_pooled', ('inputs', 'evaluation', 'acceptance')),
        ('infrastructure', ('artifacts', 'provenance', 'requests', 'evidence')),
    ):
        for name in names:
            sources[package + '_' + name] = PROJECT_ROOT / ('effect_modifier_experiment/' + package + '/' + name + '.py')
    return source_bindings(sources, PROJECT_ROOT)
