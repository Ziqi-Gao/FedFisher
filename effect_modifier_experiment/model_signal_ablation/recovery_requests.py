"""Experiment-owned in-memory request specifications and reviewed source metadata.

Constructing a plan never publishes requests, checks acceptance, imports a runner,
or starts computation. Publication and scientific admission gates are composed
by effect_modifier_experiment.scheduling.publication.
"""
import hashlib
import json
from pathlib import Path
import re

from effect_modifier_experiment.model_signal_ablation import recovery_protocol as protocol
from effect_modifier_experiment.model_signal_ablation import protocol as original
from effect_modifier_experiment.infrastructure.requests import canonical, encoded, sha256, review_sources as _review_sources

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')

REVIEW_SOURCES = {name: PROJECT_ROOT / path for name, path in {
    'proposal': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/registration-proposal.toml',
    'pilot_schema': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/pilot-parameters.schema.json',
    'shard_schema': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/shard-parameters.schema.json',
    'adapter': 'scripts/fedfisher_server_scheduler_adapter.py',
    'protocol': 'effect_modifier_experiment/model_signal_ablation/recovery_protocol.py',
    'runner': 'effect_modifier_experiment/model_signal_ablation/recovery.py',
    'noncompletion': 'effect_modifier_experiment/causal_comparison/noncompletion.py',
    'comparison_runner': 'effect_modifier_experiment/causal_comparison/runner.py',
    'request_builder': 'scripts/prepare_model_signal_recovery_requests.py',
    'acceptance': 'effect_modifier_experiment/model_signal_ablation/recovery_acceptance.py',
    'validator_cli': 'scripts/validate_model_signal_recovery.py',
    'scientific_guide': 'effect_modifier_experiment/server_scheduler/model-signal-recovery-v2/README.md',
}.items()}

# Review the real implementations as well as the preserved external commands.
# These are metadata paths only: planning imports no scheduler execution module.
for _module in ('catalog', 'validation', 'runtime', 'dispatch', 'adapter', 'legacy', 'errors', '__init__'):
    REVIEW_SOURCES['adapter_' + _module] = PROJECT_ROOT / ('effect_modifier_experiment/scheduling/' + _module + '.py')
REVIEW_SOURCES['request_plan'] = PROJECT_ROOT / 'effect_modifier_experiment/model_signal_ablation/recovery_requests.py'
REVIEW_SOURCES['request_publication'] = PROJECT_ROOT / 'effect_modifier_experiment/scheduling/publication/model_signal_recovery.py'
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
    return 'fedfisher-ms-v2-%s-cohort-p%d' % (cohort, generation)


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def _scope():
    expected = {'dslassocov'} | {
        'fedavg_%s_e%d_%s' % (model, epochs, stopping)
        for model in ('small', 'shallow', 'deep') for epochs in (1, 5, 10, 30)
        for stopping in ('patience', 'budget30')} | {
        'fedfisher_%s_%s' % (kind, model)
        for kind in ('diag', 'kfac') for model in ('small', 'shallow', 'deep')}
    settings = dict(small_mlp=(2.0, {cell for cell in expected if '_small' in cell}),
                    weak_signal_1=(1.0, expected), weak_signal_05=(0.5, expected))
    if (set(protocol.CELLS) != expected or set(protocol.SETTINGS) != set(settings)
            or tuple(protocol.VARIANTS) != original.VARIANTS
            or tuple(protocol.SPLITS) != ('iid', 'noniid')):
        raise ValueError('recovery experiment matrix changed')
    for name, (strength, cells) in settings.items():
        recipe = protocol.SETTINGS[name]
        if (type(recipe['signal_strength']) is not float or recipe['signal_strength'] != strength
                or set(recipe['cells']) != cells or len(recipe['cells']) != len(cells)):
            raise ValueError('recovery setting changed')


def _parameters_key(parameters):
    return tuple(parameters[key] for key in ('setting', 'cell', 'variant', 'shard_index', 'shard_count'))


def _formal_source_rows(closeout):
    if not isinstance(closeout, dict):
        raise ValueError('formal recovery requires the verified source closeout')
    retained = closeout.get('retained_request_ids')
    rows = closeout.get('old_inventory', {}).get('requests')
    if (not isinstance(retained, (list, tuple)) or len(retained) != 129
            or any(not isinstance(jid, str) for jid in retained) or len(set(retained)) != 129
            or not isinstance(rows, list) or len(rows) != 4608):
        raise ValueError('closeout must retain exactly 129 complete requests from the original 4608')
    by_key, ids = {}, set()
    for row in rows:
        p = original.validate_parameters(original.FORMAL_TASK, row['parameters'])
        key = _parameters_key(p)
        expected = json.loads(canonical(original.run_config(original.FORMAL_TASK, p)))
        if (p['shard_count'] != 16 or row['task'] != original.FORMAL_TASK
                or key in by_key or row['job_id'] in ids
                or row['configuration'] != expected or row['run_id'] != expected['run_id']
                or row['configuration_sha256'] != sha256(canonical(expected).encode())
                or row['seeds'] != list(original.seed_sequence(original.FORMAL_TASK, p))
                or row['splits'] != list(protocol.SPLITS)):
            raise ValueError('closeout original formal coverage/configuration changed')
        by_key[key] = row
        ids.add(row['job_id'])
    if not set(retained) <= ids:
        raise ValueError('retained request is outside the original cohort')
    retained_units = sum(len(row['seeds']) * len(row['splits']) for row in rows
                         if row['job_id'] in set(retained))
    if retained_units != 4838:
        raise ValueError('retained requests must cover exactly 4838 successful units')
    return by_key, set(retained)


def build_requests(generation=1, cohort='pilot', closeout=None):
    """Pure specifications; formal input must be the verified normalized closeout."""
    bundle = bundle_name(generation, cohort)
    _scope()
    pilot = cohort == 'pilot'
    old_rows, retained = ({}, set()) if pilot else _formal_source_rows(closeout)
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    parameters = []
    for setting, recipe in protocol.SETTINGS.items():
        for cell in recipe['cells']:
            for variant in protocol.VARIANTS:
                for index in range(1 if pilot else 16):
                    p = dict(setting=setting, cell=cell, variant=variant,
                             shard_index=index, shard_count=1 if pilot else 16)
                    if pilot:
                        p['case'] = 'baseline'
                    elif old_rows[_parameters_key(p)]['job_id'] in retained:
                        continue
                    parameters.append(p)
    if pilot:
        parameters.extend(dict(setting='small_mlp', cell='fedavg_small_e1_patience',
            variant=variant, shard_index=0, shard_count=1, case='cap_diagnostic')
            for variant in protocol.DIAGNOSTIC_VARIANTS)
    specs = []
    for p in parameters:
        protocol.validate_parameters(task, p)
        diagnostic = p.get('case') == 'cap_diagnostic'
        job_id = '%s-%s-%s-%s%s-s%02dof%02d' % (
            bundle, p['setting'].replace('_', '-'), p['cell'].replace('_', '-'),
            p['variant'].replace('_', '-'), '-cap-diagnostic' if diagnostic else '',
            p['shard_index'], p['shard_count'])
        if re.fullmatch(r'fedfisher-ms-v2-[A-Za-z0-9_.-]+', job_id) is None or len(job_id) > 128:
            raise ValueError('generation exceeds scheduler job-identity limits')
        request = dict(schema_version=2, job_id=job_id, project='FedFisher',
                       task=task, parameters=p, priority=0)
        config = json.loads(canonical(protocol.run_config(task, p)))
        seeds = list(protocol.seed_sequence(task, p))
        expected_seeds = ([9, 25, 41] if diagnostic else [0]) if pilot else list(range(p['shard_index'], 300, 16))
        if seeds != expected_seeds:
            raise ValueError('recovery deterministic seed coverage changed')
        if not pilot:
            old_config = old_rows[_parameters_key(p)]['configuration']
            ignored = {'protocol', 'task', 'run_id'}
            if ({k: v for k, v in config.items() if k not in ignored}
                    != {k: v for k, v in old_config.items() if k not in ignored}):
                raise ValueError('recovery changed a scientific recipe')
        specs.append(dict(filename=job_id + '.job-request.json', request=request,
            request_sha256=sha256(encoded(request)), run_id=protocol.run_id(task, p),
            configuration=config, configuration_sha256=sha256(canonical(config).encode()),
            seed_sequence=seeds, splits=list(protocol.SPLITS)))
    expected_count, expected_units = (290, 588) if pilot else (4479, 167962)
    if (len(specs) != expected_count or len({s['filename'] for s in specs}) != expected_count
            or sum(len(s['seed_sequence']) * len(s['splits']) for s in specs) != expected_units):
        raise ValueError('recovery request/unit coverage changed')
    return tuple(specs)


def review_sources():
    return _review_sources(REVIEW_SOURCES, PROJECT_ROOT)


def _source_hashes():
    return review_sources()
