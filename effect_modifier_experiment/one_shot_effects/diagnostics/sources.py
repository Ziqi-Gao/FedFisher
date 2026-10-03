"""Read-only, cross-source replay of previously accepted immutable artifacts.

Original completion is anchored in an accepted, hashed report. Current replay
provenance is a separate identity; the old current-source guard is not bypassed.
"""
import hashlib
import json
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def _read_path(path):
    path = Path(path)
    roots = (Path('/data/del6500/FedFisher'), Path('/scr/del6500/FedFisher'))
    _require(path.is_absolute() and '..' not in path.parts
             and any(root in path.parents for root in roots), 'source outside FedFisher data/scratch')
    _require(path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)),
             'symlink in replay source path')
    return path


def _sha(path):
    h = hashlib.sha256()
    with _read_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_source_bindings(config):
    """Pure file/hash checks, safe for request preparation; no scientific imports."""
    reference = config['accepted_report']
    path = _read_path(reference['path'])
    _require(_sha(path) == reference['sha256'], 'original acceptance report changed')
    accepted = read_json(path)
    _require(accepted['status'] == 'scientifically_validated'
             and accepted['source_provenance'] == config['original_source_provenance'],
             'original acceptance/source provenance mismatch')
    project = Path('/home/del6500/projects/FedFisher')
    for relative, digest in config['original_generation_source_files'].items():
        code = project/relative
        _require(code.is_absolute() and '..' not in code.parts and project in code.parents
                 and code.resolve() == code and not any(p.is_symlink() for p in (code, *code.parents)),
                 'invalid frozen generation source path')
        _require(hashlib.sha256(code.read_bytes()).hexdigest() == digest,
                 'original generation code changed: '+relative)
    for binding in config['sources'].values():
        directory = _read_path(binding['directory'])
        expected_name = 'ose_v1_'+binding['recipe']+'_'+binding['dataset']+'_pilot'
        _require(directory.name == expected_name, 'original source recipe/dataset path mismatch')
        files = binding['files']
        expected = {'run_manifest.env', 'seed_000/done.json', 'seed_000/started.json',
                    'seed_000/model.npz', 'seed_000/result.json',
                    *('seed_000/site_%03d.payload' % i for i in range(10))}
        _require(set(files) == expected, 'original source file inventory mismatch')
        _require({p.name for p in directory.iterdir()} == {'run_manifest.env', 'seed_000'},
                 'unexpected original run artifacts')
        _require({p.name for p in (directory/'seed_000').iterdir()}
                 == {Path(p).name for p in expected if p.startswith('seed_000/')},
                 'unexpected or partial original unit artifacts')
        for relative, digest in files.items():
            item = _read_path(directory/relative)
            _require(_sha(item) == digest == accepted['frozen_input_hashes'].get(str(item)),
                     'original artifact hash or acceptance binding changed: '+str(item))
    return dict(accepted_report=reference, original_source_provenance=config['original_source_provenance'],
                original_generation_source_files=config['original_generation_source_files'], sources=config['sources'])


def load_sources(config):
    """Restore accepted records and validate summary arithmetic without refitting."""
    import numpy as np
    from ...dslassocov_run_contract import parse_env_file
    from .. import protocol as original_protocol
    from ..model_io import restore_model
    from ..validation import validate_result
    from ..wire import canonical, unpack_payload
    verify_source_bindings(config)
    sources = {}
    for role, binding in config['sources'].items():
        directory = _read_path(binding['directory'])
        manifest = parse_env_file(directory/'run_manifest.env')
        original_config = json.loads(manifest['config_json'])
        fixed = original_protocol.run_config(original_protocol.PILOT_TASK,
            dict(recipe=binding['recipe'], dataset=binding['dataset']))
        _require(original_config == fixed, 'original configuration changed relative to frozen recipe')
        _require(manifest['protocol'] == original_protocol.PROTOCOL and manifest['format_version'] == '2',
                 'original manifest version mismatch')
        _require(all(manifest[k] == v for k, v in config['original_source_provenance'].items()),
                 'original manifest provenance mismatch')
        config_hash = hashlib.sha256(canonical(original_config).encode()).hexdigest()
        _require(manifest['config_sha256'] == config_hash, 'original configuration hash mismatch')
        identity = dict(run_id=directory.name, recipe=binding['recipe'], dataset=binding['dataset'],
                        seed=0, config_sha256=config_hash,
                        provenance_sha256=config['original_source_provenance']['provenance_sha256'])
        unit = directory/'seed_000'
        result, done, started = (read_json(unit/name) for name in ('result.json', 'done.json', 'started.json'))
        _require(done['identity'] == started['identity'] == result['identity'] == identity,
                 'original semantic identity mismatch')
        expected_artifacts = {Path(k).name: v for k, v in binding['files'].items()
                              if k.startswith('seed_000/') and k != 'seed_000/done.json'}
        _require(done['artifacts'] == expected_artifacts, 'original done inventory mismatch')
        with np.load(_read_path(unit/'model.npz'), allow_pickle=False) as archive:
            _require(len(archive.files) == len(set(archive.files)), 'duplicate checkpoint arrays')
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        payloads = [_read_path(unit/('site_%03d.payload' % i)).read_bytes() for i in range(10)]
        validate_result(result, arrays, payloads, original_config, identity)
        sources[role] = dict(config=original_config, result=result, arrays=arrays, payloads=payloads,
                             models={name: restore_model(record, arrays) for name, record in result['models'].items()})
    records = list(sources.values())
    first = records[0]['result']
    _require(all(r['result']['data_fingerprints'] == first['data_fingerprints'] for r in records),
             'original paired data differ')
    for record in records:
        for model in record['result']['models'].values():
            _require(model['holdout']['permutation_sha256'] == first['models']['primary']['holdout']['permutation_sha256']
                     and model['test']['permutation_sha256'] == first['models']['primary']['test']['permutation_sha256'],
                     'original paired intervention plans differ')
    if set(sources) == {'aggregation', 'linearity'}:
        left, right = (sources[key] for key in ('aggregation', 'linearity'))
        _require(left['result']['models']['primary']['model_sha256']
                 == right['result']['models']['primary']['model_sha256'], 'paired aggregate models differ')
        for a, b in zip(left['payloads'], right['payloads']):
            aa, am = unpack_payload(a)
            ba, bm = unpack_payload(b)
            _require(all(np.array_equal(aa[key], ba[key]) for key in ('theta', 'fisher'))
                     and am['initial_theta_sha256'] == bm['initial_theta_sha256'],
                     'paired local parameter/Fisher/initialization mismatch')
    verify_source_bindings(config)
    return sources


def validate_data_replay(data, plans, sources):
    """Check regenerated observed rows and every original plan before inference."""
    import numpy as np
    from ..data_composition import _data_evidence
    from ..evaluation import array_hash
    from ..interventions import make_permutations, permutation_hash
    first = next(iter(sources.values()))
    original = first['config']
    evidence = _data_evidence(data)
    h_hashes = [permutation_hash(plan) for plan in plans]
    x = data.evaluation.rows.x
    e = original['evaluation']
    if data.evaluation.site_ids is None:
        test_hashes = [permutation_hash(make_permutations(len(x), x.shape[1],
                        e['test_seed_offset']+original['seed'], e['repeats']))]
    else:
        test_hashes = []
        for site_id in range(len(data.sites)):
            n = int(np.sum(data.evaluation.site_ids == site_id))
            _require(n >= 2, 'target test site lacks sufficient rows')
            test_hashes.append(permutation_hash(make_permutations(n, x.shape[1],
                e['test_seed_offset']+original['seed']+1000003*site_id, e['repeats'])))
    for source in sources.values():
        _require(evidence == source['result']['data_fingerprints'], 'regenerated T/V/H or test fingerprints differ')
        for record in source['result']['models'].values():
            _require(record['holdout']['permutation_sha256'] == h_hashes
                     and record['test']['permutation_sha256'] == test_hashes
                     and record['test']['raw_x_sha256'] == array_hash(x),
                     'regenerated H/test permutation or test covariates differ')
    evaluator = evaluator_reference(data, original)
    return dict(passed=True, data_fingerprints=evidence, holdout_permutation_sha256=h_hashes,
                test_permutation_sha256=test_hashes, test_raw_x_sha256=array_hash(x),
                evaluator_reference=evaluator,
                scope='exact equality before diagnostic inference; original plans use five full permutations')


def evaluator_reference(data, original):
    """Scheduler-only derivation of fields absent from the old observed-row hash.

    This is additional replay evidence, not a claim that the old pilot archived
    test site IDs or truth hashes. Generation source bytes are separately pinned.
    """
    import numpy as np
    from ..data import SCMConfig, generate_scm
    from ..evaluation import array_hash
    if original['dataset_definition']['family'] != 'scmv2':
        _require(data.evaluation.site_ids is None and data.evaluation.truth is None,
                 'legacy cannot acquire fabricated site IDs or CATE truth')
        return dict(scope='legacy_has_no_CATE_truth_or_test_site_assignment', site_ids_sha256=None,
                    truth_array_sha256=None, variable_roles=None)
    reference = generate_scm(SCMConfig(**original['dgp']), original['seed'],
                             original['dataset_definition']['scenario']).evaluation
    actual = data.evaluation
    _require(np.array_equal(actual.site_ids, reference.site_ids), 'test site assignment differs from frozen generator')
    _require(actual.truth is not None, 'SCMV2 evaluator truth missing')
    keys = ('e', 'mu_0', 'mu_1', 'tau', 'modifier_mask')
    _require(all(np.array_equal(getattr(actual.truth,k),getattr(reference.truth,k)) for k in keys)
             and actual.truth.variable_roles == reference.truth.variable_roles,
             'evaluator truth differs from frozen generator')
    return dict(scope='new_replay_hashes_derived_from_pinned_original_generation_code_not_archived_old_pilot_hashes',
                site_ids_sha256=array_hash(actual.site_ids),
                truth_array_sha256={k:array_hash(getattr(actual.truth,k)) for k in keys},
                variable_roles=list(actual.truth.variable_roles))
