"""Frozen, read-only inputs from the accepted survival comparison.

Historical manifests retain their actual producer provenance.  They are never
opened through a current-source ArtifactStore, changed, or promoted into the
new experiment's completion evidence.  Numerical imports are deferred until a
selected model is loaded; indexing performs no inference or data generation.
"""
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

from ..infrastructure.evidence import ordinary_path
from ..infrastructure.provenance import PROVENANCE_KEYS, canonical_text, parse_env_file

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
REPORT_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1')
ACCEPTANCE_ROOT = REPORT_ROOT / 'formal-acceptance-r3-20260927T163352Z'
ACCEPTANCE = dict(path=str(ACCEPTANCE_ROOT / 'formal.acceptance.json'),
    sha256='115fd9ae833f019ec5fd42ac1f8e23bbc3400d330300ed345dc1e475b1b207d5')
SEED_RESULTS = dict(path=str(ACCEPTANCE_ROOT / 'seed-results.csv'),
    sha256='4a1732d9a0a9f86b4f28f362b56156e6ada9e20a7247bd1c28f757ce0165d42e')
ORIGINAL_CLOSEOUT = dict(path=str(REPORT_ROOT / 'formal-closeout-20260921T043146Z/closeout.json'),
    sha256='5a452ab58f02409336eaa366e88dc75f3f7fdf3a133aee530bff0528bec86a08')
CENTRAL_JOBS = Path('/data/del6500/ServerScheduler/state/jobs')
CENTRAL_AUDIT = Path('/data/del6500/ServerScheduler/audit/events.jsonl')
SCENARIOS = ('iid', 'covariate_shift')
SEEDS = tuple(range(300))
METHODS = ('dslasso', 'fedfisher_diag_shallow', 'fedfisher_kfac_shallow',
           'fedfisher_diag_deep', 'fedfisher_kfac_deep')

# These exact source hashes were independently checked against the accepted r3
# Git blobs.  They protect restoration and saved-evidence validation while new
# comparison modules are added to the repository.  Whole-repository provenance
# may change; the historical scientific implementation may not silently change.
PROTECTED_SOURCES = {
    'data.py': 'd4b39a3bfe8d410a543b2dd1cdd782ce7182025c7eae1bae4b458c857f95574c',
    'pseudo.py': '4b6f7831d64c52f5cae6e62ef051bf004fbb15e2229bba3d13ee4b2c91f797c5',
    'links.py': '7f041b4785ea932b1940b38e184407010d5c0fdd6066c25f3058aa48ad970410',
    'fisher.py': 'b18c34a728a439705ae78733bd23f628e8c619cac9990af7a4dedaa4188b67b1',
    'evaluation.py': '614cccb90842dc4e83f7f7127efd674e6cc547abe783d42c37ebf1b848b276fd',
    'dslasso.py': '615bb6eb2dfbcad1a79397a838c347807b6f24cade7f6e7492eba02e80842e1d',
    'recovery.py': '42ebb4239416e43a6085ca4270cbea189cdc79f46cb9dcdc6affaaafe34f3963',
    'recovery_method.py': '9f6a7bdf555a93bc06e10a9e74aeef27665341dae7979d79b181083cdd55c3ef',
    'artifacts.py': '8776e113869a8f1316cbdec3797c20e28d3486cb6e5d474c70ec7580498d4503',
    'recovery_protocol.py': 'dce045c0bce704e0f23aa180e7eec2620baf0457bc48a12d964c0293e68ec250',
}
_HASH_CACHE = {}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate historical JSON key')
        result[key] = value
    return result


def _json_bytes(data):
    return json.loads(data, object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def historical_binding():
    """Pure fixed configuration; safe for scheduler schema/configuration code."""
    return dict(acceptance=dict(ACCEPTANCE), seed_results=dict(SEED_RESULTS),
                original_closeout=dict(ORIGINAL_CLOSEOUT))


def _signature(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _check_binding(binding, *, read=False):
    _require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'},
             'historical file binding fields differ')
    path = ordinary_path(binding['path'])
    before = _signature(path)
    expected = binding['sha256']
    _require(isinstance(expected, str) and len(expected) == 64
             and all(char in '0123456789abcdef' for char in expected),
             'invalid historical checksum')
    cached = _HASH_CACHE.get(str(path))
    if not read and cached == (before, expected):
        return None
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == expected,
             'historical artifact checksum mismatch: ' + str(path))
    _require(_signature(path) == before, 'historical input changed while reading')
    _HASH_CACHE[str(path)] = (before, expected)
    return data if read else None


def _check_sources():
    for name, checksum in PROTECTED_SOURCES.items():
        _check_binding(dict(path=str(PROJECT_ROOT / 'effect_modifier_experiment/survival' / name),
                            sha256=checksum), read=True)


@lru_cache(maxsize=1)
def _index():
    """Index exact 3,000 accepted references, without reading all model files."""
    accepted = _json_bytes(_check_binding(ACCEPTANCE, read=True))
    original = _json_bytes(_check_binding(ORIGINAL_CLOSEOUT, read=True))
    _require(accepted.get('scope') == 'survival_dslasso_integrated_formal_acceptance'
             and accepted.get('epoch') == 3 and accepted.get('passed') is True
             and accepted.get('validated_unit_count') == 600
             and accepted.get('archived_fedfisher_units') == 2400
             and accepted.get('selected_recipe') == 'budget_and_step'
             and accepted.get('original_closeout') == ORIGINAL_CLOSEOUT,
             'historical acceptance is not the complete accepted r3 comparison')
    _require(original.get('fedfisher_passed') is True
             and original.get('fedfisher_validated_units') == 2400
             and original.get('source_release_gate_passed') is True,
             'historical FedFisher closeout is incomplete')
    records = {}
    manifests = {row['config']['run_id']: row['manifest'] for row in original['runs']}
    central = {Path(row['job']['path']).stem: row for row in accepted['central_jobs']}
    for method, units in [('dslasso', accepted['units']), ('fedfisher', original['validated_units'])]:
        for unit in units:
            cell = 'dslasso' if method == 'dslasso' else unit['cell']
            if method == 'fedfisher' and not cell.startswith('fedfisher_'):
                continue  # Original dsLasso fits are not accepted r3 inputs.
            seed = unit['seed']
            scenario = unit['parameters']['scenario'] if method == 'dslasso' else unit['scenario']
            key = (cell, scenario, seed)
            _require(cell in METHODS and scenario in SCENARIOS and type(seed) is int
                     and 0 <= seed < 300 and key not in records,
                     'unknown or duplicate historical scientific unit')
            artifacts = (unit['artifacts'] if method == 'dslasso' else list(unit['artifacts'].values()))
            artifacts = {Path(row['path']).name: row for row in artifacts}
            _require(set(artifacts) == {'started.json', 'model.npz', 'result.json', 'done.json'},
                     'historical scientific artifact inventory differs')
            manifest = unit['manifest'] if method == 'dslasso' else manifests[unit['identity']['run_id']]
            old_job = (None if method == 'dslasso' else
                       original['central_jobs'][unit['execution_receipt']['job_id']])
            producer = unit['producer_evidence'] if method == 'dslasso' else old_job['producer']
            job_id = producer['execution_receipt']['job_id']
            job = central[job_id] if method == 'dslasso' else dict(job=old_job['job'], logs=producer['logs'])
            records[key] = dict(artifacts=artifacts, manifest=manifest, producer=producer,
                job=job, provenance=accepted['provenance'] if method == 'dslasso' else original['provenance'])
    expected = {(cell, scenario, seed) for cell in METHODS for scenario in SCENARIOS for seed in SEEDS}
    _require(set(records) == expected, 'historical input coverage is not exactly 3,000 units')
    csv_rows = list(csv.DictReader(io.StringIO(_check_binding(SEED_RESULTS, read=True).decode())))
    seen = set()
    for row in csv_rows:
        key = (row['method'], row['scenario'], int(row['seed']))
        _require(key in records and key not in seen, 'unknown or duplicate accepted summary row')
        seen.add(key)
        record = records[key]
        _require(record['artifacts']['result.json'] == dict(path=row['result_path'], sha256=row['result_sha256'])
                 and row['producer_commit'] == record['provenance']['git_commit']
                 and row['provenance_sha256'] == record['provenance']['provenance_sha256'],
                 'accepted CSV does not bind the historical result and producer')
    _require(seen == expected, 'accepted CSV coverage differs from scientific evidence')
    return records


def validate_historical_inputs():
    """Check startup anchors/source and indexed coverage; no model inference."""
    for binding in historical_binding().values():
        _check_binding(binding, read=True)
    _check_sources()
    records = _index()
    return dict(bindings=historical_binding(), scientific_unit_count=len(records),
                protected_source_count=len(PROTECTED_SOURCES), inference_performed=False)


def _producer(record, started):
    proof = record['producer']
    _require(started['execution'] == proof['execution_receipt'],
             'historical producing attempt differs from frozen evidence')
    for binding in [proof['retry'], *proof['logs'].values(), record['job']['job'],
                    *record['job']['logs'].values()]:
        _check_binding(binding)
    job = _json_bytes(_check_binding(record['job']['job'], read=True))
    _require(Path(record['job']['job']['path']).parent == CENTRAL_JOBS
             and job.get('project') == 'FedFisher' and job.get('state') == 'completed'
             and job.get('exit_code') == 0
             and job.get('job_id') == started['execution']['job_id'],
             'historical model lacks exact completed central producer evidence')
    for witness in proof['audit'].values():
        path = ordinary_path(witness['path'])
        _require(path == CENTRAL_AUDIT, 'historical audit ownership differs')
        with path.open('rb') as stream:
            stream.seek(witness['offset'])
            data = stream.read(witness['length'])
        _require(hashlib.sha256(data).hexdigest() == witness['sha256']
                 and _json_bytes(data) == witness['record'], 'historical producer audit changed')


def _selected_binding(record, identity, started):
    return dict(identity=identity, manifest=dict(record['manifest']),
        artifacts={name: dict(value) for name, value in record['artifacts'].items()},
        producer_commit=record['provenance']['git_commit'],
        producer_job_id=started['execution']['job_id'],
        producer_attempt=started['execution']['attempt'], accepted_inputs=historical_binding())


def validate_selected_binding(value):
    """Recheck a saved selected-input receipt without restoring a model.

    The current comparison's artifact validator can call this on each saved
    receipt.  Hash-bound original numeric evidence stays authoritative; this
    function checks identity, complete inventory and actual producer evidence.
    """
    validate_historical_inputs()
    _require(isinstance(value, dict) and isinstance(value.get('identity'), dict),
             'malformed saved historical input receipt')
    identity = value['identity']
    _require(type(identity.get('seed')) is int and identity.get('cell') in METHODS
             and identity.get('scenario') in SCENARIOS, 'unknown saved historical scientific identity')
    key = (identity['cell'], identity['scenario'], identity['seed'])
    _require(key in _index(), 'unknown saved historical scientific identity')
    record = _index()[key]
    for binding in [record['manifest'], *record['artifacts'].values()]:
        _check_binding(binding)
    manifest = parse_env_file(record['manifest']['path'])
    config = _json_bytes(manifest['config_json'])
    expected_identity = dict(run_id=manifest['run_id'], config_sha256=manifest['config_sha256'],
        provenance_sha256=record['provenance']['provenance_sha256'], cell=key[0], scenario=key[1], seed=key[2])
    if key[0] == 'dslasso':
        expected_identity['recipe'] = config['recovery_recipe']
    started = _json_bytes(_check_binding(record['artifacts']['started.json'], read=True))
    done = _json_bytes(_check_binding(record['artifacts']['done.json'], read=True))
    expected_dir = Path(record['artifacts']['started.json']['path']).parent
    _require({path.name for path in expected_dir.iterdir()} == set(record['artifacts'])
             and started['identity'] == identity == expected_identity
             and done == dict(identity=identity, artifacts={name: record['artifacts'][name]['sha256']
                 for name in ('started.json', 'model.npz', 'result.json')})
             and value == _selected_binding(record, identity, started),
             'saved historical input receipt differs from frozen accepted evidence')
    _producer(record, started)
    return True


def selected_fingerprints(binding):
    """Read the frozen dataset fingerprints without restoring a model.

    Comparison validation needs the actual historical training/test/oracle
    identities, rather than values supplied only by its newly saved result.
    The selected receipt and producing attempt are checked before reading the
    result bytes whose checksum is bound in the accepted historical inventory.
    """
    validate_selected_binding(binding)
    result = _json_bytes(_check_binding(binding['artifacts']['result.json'], read=True))
    _require(result.get('identity') == binding['identity'] and result.get('status') == 'complete',
             'historical fingerprint result identity/status differs')
    fingerprints = {name: result['fingerprints'][name]
                    for name in ('training', 'evaluation', 'evaluation_truth')}
    _require(all(isinstance(value, str) and len(value) == 64
                 and all(char in '0123456789abcdef' for char in value)
                 for value in fingerprints.values()), 'invalid historical dataset fingerprint')
    return fingerprints


def _load_unit(cell, scenario, seed, record):
    import numpy as np
    from ..survival import artifacts, recovery
    from ..survival.dslasso import SurvivalLinearModel
    from ..survival.fisher import FisherConfig, FisherFit

    manifest_bytes = _check_binding(record['manifest'], read=True)
    manifest = parse_env_file(record['manifest']['path'])
    # The env parser rereads the file; ensure its bytes were unchanged meanwhile.
    _require(Path(record['manifest']['path']).read_bytes() == manifest_bytes,
             'historical manifest changed while parsing')
    config = _json_bytes(manifest['config_json'])
    _require(hashlib.sha256(_canonical(config).encode()).hexdigest() == manifest['config_sha256'],
             'historical manifest configuration checksum differs')
    provenance = record['provenance']
    _require(all(manifest[key] == provenance[key] for key in (*PROVENANCE_KEYS, 'provenance_sha256'))
             and hashlib.sha256(canonical_text(provenance, PROVENANCE_KEYS).encode()).hexdigest()
             == provenance['provenance_sha256'], 'historical producer provenance differs')
    identity = dict(run_id=manifest['run_id'], config_sha256=manifest['config_sha256'],
                    provenance_sha256=provenance['provenance_sha256'], seed=seed, cell=cell, scenario=scenario)
    if cell == 'dslasso':
        _require(config.get('recovery_recipe') == 'budget_and_step', 'historical dsLasso recipe differs')
        identity['recipe'] = 'budget_and_step'
    _require(config['cell'] == cell and config['scenario'] == scenario
             and config['run_id'] == identity['run_id'] and config['seeds'] == list(range(300)),
             'historical manifest scientific identity differs')
    expected_dir = Path(record['manifest']['path']).parent / ('seed_%03d' % seed)
    _require(all(Path(binding['path']).parent == expected_dir for binding in record['artifacts'].values()),
             'historical artifacts moved outside their scientific seed directory')
    _require({path.name for path in expected_dir.iterdir()} == set(record['artifacts']),
             'partial or conflicting historical scientific directory')
    payloads = {name: _check_binding(binding, read=True) for name, binding in record['artifacts'].items()}
    result, started, done = (_json_bytes(payloads[name]) for name in ('result.json', 'started.json', 'done.json'))
    _require(result['identity'] == started['identity'] == done['identity'] == identity
             and result['status'] == 'complete'
             and done == dict(identity=identity, artifacts={name: record['artifacts'][name]['sha256']
                 for name in ('started.json', 'model.npz', 'result.json')}),
             'historical completion marker, started receipt or result identity differs')
    _producer(record, started)
    with np.load(io.BytesIO(payloads['model.npz']), allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), 'duplicate historical model arrays')
        arrays = {name: archive[name] for name in archive.files}
    if cell == 'dslasso':
        recovery.validate_result(result, arrays, identity, config)
        model = SurvivalLinearModel(float(arrays['intercept']), arrays['coefficients'].copy())
    else:
        artifacts.validate_result(result, arrays, identity, config)
        uploads = tuple(arrays['upload_%03d' % site].tobytes() for site in range(config['dgp']['num_sites']))
        model = FisherFit(arrays['vector'], FisherConfig(**config['method']['optimizer']),
                          1 + 2 * config['dgp']['covariate_dim'], result['fit'], uploads)
    binding = _selected_binding(record, identity, started)
    return dict(model=model, result=result, binding=binding, config=config,
                fingerprints=dict(result['fingerprints']))


def load_historical(family, scenario, seed):
    """Restore accepted reference models, fully checking each selected unit.

    ``family`` is linear, shallow or deep.  Returned tags are dslasso for the
    linear family and fedfisher_diag/fedfisher_kfac for the neural families.
    Restoration does not call predict_eta or generate a dataset.
    """
    _require(family in ('linear', 'shallow', 'deep') and scenario in SCENARIOS
             and type(seed) is int and 0 <= seed < 300, 'unknown historical model request')
    validate_historical_inputs()
    selected = {'dslasso': 'dslasso'} if family == 'linear' else {
        'fedfisher_' + method: 'fedfisher_' + method + '_' + family for method in ('diag', 'kfac')}
    records = _index()
    return {tag: _load_unit(cell, scenario, seed, records[cell, scenario, seed])
            for tag, cell in selected.items()}
