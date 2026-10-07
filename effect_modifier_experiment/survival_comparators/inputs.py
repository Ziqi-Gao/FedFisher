"""Immutable historical inputs, retaining their actual source and producers.

The four accepted MLP8 baselines are read, never fitted or inferred. The r3
dsLasso reader restores its accepted coefficients for new probability inference
inside the registered task. Neither input becomes a new-source refitted model.
"""
from functools import lru_cache
import io
import json
from pathlib import Path

from ..infrastructure.evidence import file_binding, ordinary_path
from ..infrastructure.provenance import parse_env_file
from ..survival.acceptance import _recheck_producer
from ..survival_pooled import inputs as archived

MLP8_ACCEPTANCE = dict(path='/data/del6500/FedFisher/effect_modifier_experiment/reports/'
    'survival-mlp8-v1/formal-discovery-v2-0acd861cb6b481c80aad5b1cf2a1c960b3cae06960c03e49ed9904557e39783e.acceptance.json',
    sha256='6f268cce2c38225c3593daba10109e189896d92a66da1c120f72483a2a2bd930')
_PRIOR = '/data/del6500/FedFisher/effect_modifier_experiment/reports/model-signal-ablation-v1/recovery-v2/small-mlp-results-20261001T041956Z/'
PRIOR_RECIPE = dict(
    summary=dict(path=_PRIOR + 'summary.csv', sha256='5cf2f44c057c328d9612bb17ddb0f7a6e8023a8b89c80e52ad3aaba62aa1fc97'),
    bindings=dict(path=_PRIOR + 'source_bindings.json', sha256='9cade625a5fcafea6c51d89057fa32e270ad784adb19a8e70058e0f5b269ff86'),
    manifest=dict(path='/data/del6500/FedFisher/effect_modifier_experiment/outputs/'
        'model_signal_small_mlp_fedavg_small_e5_budget30_control_v2_300seeds/run_manifest.env',
        sha256='831fd8a64ae447203270ad510ad512f89e500cab03dbc08b637d3f7814683a55'))
_PROTECTED = {
    'methods.py': 'cac5022ec65d9e0d51e1d7be0e8629350578b901cf76dd40b438c3ebbfaaf99f',
    'protocol.py': '783e566bf801d0d7e64485b3a030eb38b76c76c83acfac7593e01986c3b47683',
    'discovery_protocol.py': '7464e2196290760da3ad2157567aa991c7b02c433d99acf1063939f87e1f4d8f',
    'discovery_evaluation.py': 'b8ac0789d5c4edfb3450f7da77fc6d46c408213b55c6a56c7faf43b0242bf175',
    'artifacts.py': '9488c1bf485412d56f90956968508b2dec410d27cae3fc72ed42de5f12aa8645',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def historical_binding():
    return dict(mlp8=dict(MLP8_ACCEPTANCE), dslasso=archived.historical_binding(),
        fedavg_selection=dict(recipe='fedavg_small_e5_budget30',
            rule='original signal-2 Top-2/AUROC; observed fit time breaks perfect-recovery ties',
            survival_optimality_claimed=False,
            evidence={key: dict(value) for key, value in PRIOR_RECIPE.items()}))


def _read(binding):
    return archived._check_binding(binding, read=True)


def validate_inputs():
    archived.validate_historical_inputs()
    _read(MLP8_ACCEPTANCE)
    for binding in PRIOR_RECIPE.values():
        _read(binding)
    for name, checksum in _PROTECTED.items():
        _read(dict(path=str(archived.PROJECT_ROOT / 'effect_modifier_experiment/survival_mlp8' / name),
                   sha256=checksum))
    prior = json.loads(parse_env_file(PRIOR_RECIPE['manifest']['path'])['config_json'])
    operative = prior['method']['optimizer']
    _require(all(operative[key] == expected for key, expected in dict(local_epochs=5,
        total_local_epochs=30, stopping_mode='fixed_total_epochs', batch_size=64,
        learning_rate=.01, momentum=.9, loader_seed_offset=6200037).items()),
        'prior FedAvg selected recipe changed')
    return True


@lru_cache(maxsize=1)
def _index():
    proof = json.loads(_read(MLP8_ACCEPTANCE))
    _require(proof.get('passed') is True and proof.get('inference_performed') is False
        and proof.get('scope') == 'survival_mlp8_discovery_v2_formal_scientific_acceptance'
        and proof.get('validated_unit_count') == 600 and proof.get('validated_request_count') == 32
        and proof['provenance']['git_commit'] == '106afaf64477da24c074395dd2b6811535392dc8',
        'MLP8 reference is not the complete accepted V2 cohort')
    index = {(unit['scenario'], unit['seed']): unit for unit in proof['units']}
    _require(len(index) == 600 and set(index) == {(s, seed) for s in ('iid', 'covariate_shift')
             for seed in range(300)}, 'MLP8 historical coverage changed')
    return proof, index


def load_mlp8_reference(scenario, seed):
    """Validate one historical unit without current-source resume or inference."""
    import numpy as np
    from ..survival_mlp8 import discovery_protocol
    from ..survival_mlp8.artifacts import validate_discovery_result
    _require(scenario in ('iid', 'covariate_shift') and type(seed) is int and 0 <= seed < 10,
             'only paired seeds 0..9 belong to the new comparator cohort')
    validate_inputs()
    proof, index = _index()
    unit = index[scenario, seed]
    files = {Path(b['path']).name: b for b in unit['artifacts']}
    directory = Path(files['result.json']['path']).parent
    _require(set(files) == {'started.json', 'model.npz', 'result.json', 'done.json'}
        and ordinary_path(directory, exists=False).is_dir()
        and {p.name for p in directory.iterdir()} == set(files)
        and all(Path(b['path']).parent == directory for b in files.values()),
        'MLP8 historical artifact layout changed')
    payload = {name: _read(binding) for name, binding in files.items()}
    _read(unit['manifest'])
    manifest = parse_env_file(unit['manifest']['path'])
    _require(file_binding(unit['manifest']['path']) == unit['manifest'], 'MLP8 manifest changed while parsing')
    config = discovery_protocol.run_config(discovery_protocol.FORMAL_TASK,
        dict(scenario=scenario, shard_index=0, shard_count=16))
    _require(json.loads(manifest['config_json']) == config
        and all(manifest[k] == v for k, v in proof['provenance'].items()),
        'MLP8 frozen configuration/provenance changed')
    result, started, done = (json.loads(payload[name]) for name in ('result.json', 'started.json', 'done.json'))
    from ..infrastructure.artifacts import digest
    identity = dict(run_id=config['run_id'], config_sha256=digest(config),
                    provenance_sha256=proof['provenance']['provenance_sha256'],
                    seed=seed, cell='mlp8', scenario=scenario)
    _require(result['identity'] == started['identity'] == done['identity'] == identity
        and done['artifacts'] == {name: files[name]['sha256'] for name in ('started.json', 'model.npz', 'result.json')},
        'MLP8 original completion evidence changed')
    producer = unit['producer_evidence']
    _require(started['execution'] == producer['execution_receipt'], 'MLP8 producing attempt receipt changed')
    job_id = started['execution']['job_id']
    jobs = [row for row in proof['central_jobs'] if Path(row['job']['path']).stem == job_id]
    _require(len(jobs) == 1, 'MLP8 producing job is absent from accepted central evidence')
    central = jobs[0]
    job = json.loads(_read(central['job']))
    for binding in central['logs'].values():
        _read(binding)
    _require(job.get('schema_version') == 2 and job.get('project') == 'FedFisher'
        and job.get('job_id') == job_id and job.get('task') == discovery_protocol.FORMAL_TASK
        and job.get('state') == 'completed' and type(job.get('exit_code')) is int and job['exit_code'] == 0
        and job['parameters']['scenario'] == scenario
        and seed in discovery_protocol.seed_sequence(job['task'], job['parameters'])
        and all(job[key] == started['execution'][key] for key in
                ('allocation', 'execution_profile', 'estimated_runtime_seconds')),
        'MLP8 producing central job no longer supports accepted completion')
    _recheck_producer(unit['producer_evidence'])
    with np.load(io.BytesIO(payload['model.npz']), allow_pickle=False) as saved:
        _require(len(saved.files) == len(set(saved.files)), 'duplicate historical arrays')
        arrays = {name: saved[name] for name in saved.files}
    validate_discovery_result(result, arrays, identity, config)
    _read(MLP8_ACCEPTANCE)
    return dict(result=result, arrays=arrays, config=config, fingerprints=result['fingerprints'],
                binding=dict(acceptance=dict(MLP8_ACCEPTANCE), unit=unit))
