"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re

from effect_modifier_experiment.model_signal_ablation import protocol
from effect_modifier_experiment.infrastructure.requests import canonical, encoded, sha256, review_sources as _review_sources

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')

REVIEW_SOURCES = {name: PROJECT_ROOT / path for name, path in {
    'proposal': 'effect_modifier_experiment/server_scheduler/model-signal-ablation-v1/registration-proposal.toml',
    'pilot_schema': 'effect_modifier_experiment/server_scheduler/model-signal-ablation-v1/pilot-parameters.schema.json',
    'shard_schema': 'effect_modifier_experiment/server_scheduler/model-signal-ablation-v1/shard-parameters.schema.json',
    'adapter': 'scripts/fedfisher_server_scheduler_adapter.py',
    'protocol': 'effect_modifier_experiment/model_signal_ablation/protocol.py',
    'request_builder': 'scripts/prepare_model_signal_requests.py',
    'acceptance': 'effect_modifier_experiment/model_signal_ablation/acceptance.py',
    'scientific_guide': 'effect_modifier_experiment/model_signal_ablation/README.md',
}.items()}

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/model_signal_ablation/requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/model_signal.py'
REVIEW_SOURCES['request_publication_support'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/__init__.py'

REVIEW_SOURCES.update({
    'measurement': PROJECT_ROOT / 'effect_modifier_experiment/components/measurement.py',
    'fingerprints': PROJECT_ROOT / 'effect_modifier_experiment/components/fingerprints.py',
    'recovery_metrics': PROJECT_ROOT / 'effect_modifier_experiment/components/recovery_metrics.py',
    'comparison_computation': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/computation.py',
    'timing_computation': PROJECT_ROOT / 'effect_modifier_experiment/fedfisher_timing/computation.py',
    'model_signal_runner': PROJECT_ROOT / 'effect_modifier_experiment/model_signal_ablation/runner.py',
    'comparison_evaluation': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/evaluation.py',
    'comparison_fedavg': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/fedavg.py',
    'comparison_legacy': PROJECT_ROOT / 'effect_modifier_experiment/causal_comparison/legacy.py',
    'timing_fit': PROJECT_ROOT / 'effect_modifier_experiment/fedfisher_timing/fit.py',
})


def bundle_name(generation=1, cohort='pilot'):
    if type(generation) is not int or generation < 1 or cohort not in ('pilot', 'formal'):
        raise ValueError('positive integer generation and pilot/formal cohort required')
    return 'fedfisher-ms-v1-%s-cohort-p%d' % (cohort, generation)


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def build_requests(generation=1, cohort='pilot', shard_count=None):
    """Pure exact-scope specifications; does not run science or write requests."""
    bundle = bundle_name(generation, cohort)
    pilot = cohort == 'pilot'
    if pilot and shard_count is not None:
        raise ValueError('pilot shard count is fixed')
    count = 1 if pilot else (16 if shard_count is None else shard_count)
    if type(count) is not int or count not in ((1,) if pilot else (8, 16)):
        raise ValueError('formal shard count must be 8 or 16')
    expected_cells = {'dslassocov'} | {
        'fedavg_%s_e%d_%s' % (model, epochs, stopping)
        for model in ('small', 'shallow', 'deep') for epochs in (1, 5, 10, 30)
        for stopping in ('patience', 'budget30')} | {
        'fedfisher_%s_%s' % (kind, model)
        for kind in ('diag', 'kfac') for model in ('small', 'shallow', 'deep')}
    expected_settings = dict(small_mlp=(2.0, {cell for cell in expected_cells if '_small' in cell}),
        weak_signal_1=(1.0, expected_cells), weak_signal_05=(0.5, expected_cells))
    if (set(protocol.CELLS) != expected_cells or set(protocol.SETTINGS) != set(expected_settings)
            or set(protocol.VARIANTS) != {'control', 'site_gaussian_shift', 'local_permutation',
                                         'site_gaussian_local_permutation'}
            or tuple(protocol.SPLITS) != ('iid', 'noniid')):
        raise ValueError('model/signal scope changed from the reviewed experiment matrix')
    for setting, (strength, cells) in expected_settings.items():
        recipe = protocol.SETTINGS[setting]
        if (type(recipe['signal_strength']) is not float or recipe['signal_strength'] != strength
                or set(recipe['cells']) != cells or len(recipe['cells']) != len(cells)):
            raise ValueError('model/signal setting changed from the reviewed experiment matrix')
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    specs = []
    for setting, recipe in protocol.SETTINGS.items():
        for cell in recipe['cells']:
            for variant in protocol.VARIANTS:
                for index in range(count):
                    parameters = dict(setting=setting, cell=cell, variant=variant,
                                      shard_index=index, shard_count=count)
                    protocol.validate_parameters(task, parameters)
                    job_id = '%s-%s-%s-%s-s%02dof%02d' % (
                        bundle, setting.replace('_', '-'), cell.replace('_', '-'),
                        variant.replace('_', '-'), index, count)
                    if re.fullmatch(r'fedfisher-ms-[A-Za-z0-9_.-]{1,115}', job_id) is None:
                        raise ValueError('generation exceeds scheduler job-identity limits')
                    request = dict(schema_version=2, job_id=job_id, project='FedFisher',
                                   task=task, parameters=parameters, priority=0)
                    config = json.loads(canonical(protocol.run_config(task, parameters)))
                    seeds = list(protocol.seed_sequence(task, parameters))
                    if seeds != ([0] if pilot else list(range(index, 300, count))):
                        raise ValueError('pilot seed zero or deterministic formal seed coverage changed')
                    specs.append(dict(filename=job_id + '.job-request.json', request=request,
                        request_sha256=sha256(encoded(request)), run_id=protocol.run_id(task, parameters),
                        configuration=config, configuration_sha256=sha256(canonical(config).encode()),
                        seed_sequence=seeds, splits=list(protocol.SPLITS)))
    if len(specs) != 288 * count or len({item['filename'] for item in specs}) != len(specs):
        raise ValueError('model/signal request coverage or job uniqueness changed')
    return tuple(specs)


def _source_hashes():
    return _review_sources(REVIEW_SOURCES, PROJECT_ROOT)


def review_sources():
    return _source_hashes()
