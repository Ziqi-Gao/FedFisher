"""Pure replacement request plans; publication and admission live elsewhere."""
from pathlib import Path
import json
import re

from ..infrastructure.requests import canonical, encoded, sha256, review_sources as _review_sources
from . import completion_protocol as protocol
from .recovery_requests import REVIEW_SOURCES as _PREVIOUS_REVIEW_SOURCES


PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
REVIEW_SOURCES = dict(_PREVIOUS_REVIEW_SOURCES)
REVIEW_SOURCES.update({name: PROJECT_ROOT / path for name, path in {
    'proposal': 'effect_modifier_experiment/server_scheduler/model-signal-completion-v3/registration-proposal.toml',
    'pilot_schema': 'effect_modifier_experiment/server_scheduler/model-signal-completion-v3/pilot-parameters.schema.json',
    'shard_schema': 'effect_modifier_experiment/server_scheduler/model-signal-completion-v3/shard-parameters.schema.json',
    'completion_plan': 'effect_modifier_experiment/server_scheduler/model-signal-completion-v3/completion-plan.json',
    'protocol': 'effect_modifier_experiment/model_signal_ablation/completion_protocol.py',
    'runner': 'effect_modifier_experiment/model_signal_ablation/completion.py',
    'acceptance': 'effect_modifier_experiment/model_signal_ablation/completion_acceptance.py',
    'archive': 'effect_modifier_experiment/model_signal_ablation/completion_archive.py',
    'request_builder': 'scripts/prepare_model_signal_completion_requests.py',
    'validator_cli': 'scripts/validate_model_signal_completion.py',
    'shared_proof_publisher': 'scripts/validate_model_signal_recovery.py',
    'scientific_guide': 'effect_modifier_experiment/server_scheduler/model-signal-completion-v3/README.md',
    'request_plan': 'effect_modifier_experiment/model_signal_ablation/completion_requests.py',
    'request_publication': 'effect_modifier_experiment/scheduling/publication/model_signal_completion.py',
    'shared_recovery_runner': 'effect_modifier_experiment/model_signal_ablation/recovery.py',
    'previous_protocol': 'effect_modifier_experiment/model_signal_ablation/recovery_protocol.py',
    'previous_request_plan': 'effect_modifier_experiment/model_signal_ablation/recovery_requests.py',
    'previous_acceptance': 'effect_modifier_experiment/model_signal_ablation/recovery_acceptance.py',
    'original_protocol': 'effect_modifier_experiment/model_signal_ablation/protocol.py',
    'model_signal_artifacts': 'effect_modifier_experiment/model_signal_ablation/artifacts.py',
    'timing_artifacts': 'effect_modifier_experiment/fedfisher_timing/artifacts.py',
    'shared_artifacts': 'effect_modifier_experiment/infrastructure/artifacts.py',
    'shared_provenance': 'effect_modifier_experiment/infrastructure/provenance.py',
    'shared_evidence': 'effect_modifier_experiment/infrastructure/evidence.py',
    'shared_requests': 'effect_modifier_experiment/infrastructure/requests.py',
}.items()})


def bundle_name(generation=1, cohort='pilot'):
    if type(generation) is not int or generation < 1 or cohort not in ('pilot', 'formal'):
        raise ValueError('positive integer generation and pilot/formal cohort required')
    return 'fedfisher-ms-v3-%s-cohort-p%d' % (cohort, generation)


def inventory_name(generation=1, cohort='pilot'):
    return bundle_name(generation, cohort) + '.inventory.json'


def build_requests(generation=1, cohort='pilot'):
    """Return exactly the fixed pilot or failed-whole-shard replacement plan.

    No evidence file is read here. Publication/runtime independently authenticate
    the immutable archive and, for formal work, the new-source pilot proof.
    """
    bundle = bundle_name(generation, cohort)
    pilot = cohort == 'pilot'
    task = protocol.PILOT_TASK if pilot else protocol.FORMAL_TASK
    parameters = protocol.PILOT_PARAMETERS if pilot else protocol.FORMAL_PARAMETERS
    specs = []
    for item in parameters:
        p = protocol.validate_parameters(task, item)
        job_id = '%s-%s-%s-%s-s%02dof%02d' % (
            bundle, p['setting'].replace('_', '-'), p['cell'].replace('_', '-'),
            p['variant'].replace('_', '-'), p['shard_index'], p['shard_count'])
        if re.fullmatch(r'fedfisher-ms-v3-[A-Za-z0-9_.-]+', job_id) is None or len(job_id) > 128:
            raise ValueError('generation exceeds scheduler job-identity limits')
        config = json.loads(canonical(protocol.run_config(task, p)))
        old = protocol.previous_config(task, p)
        execution_fields = {'protocol', 'task', 'run_id', 'case'}
        if canonical({k: v for k, v in config.items() if k not in execution_fields}) != canonical(
                {k: v for k, v in old.items() if k not in execution_fields}):
            raise ValueError('completion changed the archived numerical recipe')
        request = dict(schema_version=2, job_id=job_id, project='FedFisher',
                       task=task, parameters=p, priority=0)
        seeds = list(protocol.seed_sequence(task, p))
        expected = [0] if pilot else list(range(p['shard_index'], 300, 16))
        if seeds != expected or tuple(protocol.SPLITS) != ('iid', 'noniid'):
            raise ValueError('completion deterministic unit coverage changed')
        specs.append(dict(filename=job_id + '.job-request.json', request=request,
            request_sha256=sha256(encoded(request)), run_id=config['run_id'], configuration=config,
            configuration_sha256=sha256(canonical(config).encode()),
            seed_sequence=seeds, splits=list(protocol.SPLITS)))
    count, units = (8, 16) if pilot else (24, 896)
    if (len(specs) != count or len({s['filename'] for s in specs}) != count
            or sum(len(s['seed_sequence']) * len(s['splits']) for s in specs) != units):
        raise ValueError('completion request or unit coverage differs from the fixed scope')
    return tuple(specs)


def review_sources():
    return _review_sources(REVIEW_SOURCES, PROJECT_ROOT)


def _source_hashes():
    return review_sources()
