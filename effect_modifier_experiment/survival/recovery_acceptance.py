"""Read-only recovery gates, including explicit comparison to archived v1 fits.

An iteration-limit observation may complete a diagnostic, never a formal fit.
No fitting, data regeneration, model inference or central mutation occurs here.
"""
from pathlib import Path

from ..infrastructure.artifacts import digest
from . import acceptance as old, recovery
from . import recovery_protocol as protocol
from .artifacts import current_provenance

PROOF_ROOT = old.PROOF_ROOT / 'dslasso-recovery-v1'
REPAIR_PROOF_ROOT = old.PROOF_ROOT / 'dslasso-repair-v2'
INTEGRATED_PROOF_ROOT = old.PROOF_ROOT / 'dslasso-integrated-v3'
SELECTION_ORDER = ('budget_only', 'step_only', 'budget_and_step')
SELECTION_RULE = ('all_27_cases_stationary; minimize_sum_loss_gradient_calls; '
                  'ties=budget_only,step_only,budget_and_step; no_truth_or_recovery_metrics')
PRESERVED_SCIENCE = ('data.py', 'pseudo.py', 'links.py', 'evaluation.py')
_require = old._require


def _checked(binding):
    _require(old._binding(binding['path']) == binding, 'bound recovery evidence changed')
    return old._read_json(binding['path'])


def check_original_closeout():
    """Check the immutable historical trust anchor, not current-source v1 status.

    Each launch checks its reference result separately. Full archived model
    hashes are checked once again when accepting the final mixed-version cohort.
    """
    proof = _checked(recovery.ORIGINAL_CLOSEOUT)
    _require(proof.get('scope') == 'survival_v1_formal_fedfisher_acceptance_and_dslasso_failed_cohort_closeout'
             and proof.get('source_release_gate_passed') is True
             and proof.get('fedfisher_passed') is True and proof.get('whole_cohort_passed') is False
             and proof.get('fedfisher_validated_units') == 2400
             and proof.get('dslasso_validated_units') == 420
             and proof.get('dslasso_failed_units') == 19
             and proof.get('dslasso_never_started_units') == 161,
             'original survival closeout is not the reviewed recovery anchor')
    for name in PRESERVED_SCIENCE:
        path = str(old.PROJECT_ROOT / 'effect_modifier_experiment/survival' / name)
        _require(old._binding(path) == proof['scientific_source_files'][path],
                 'recovery changed preserved data, pseudo-value, link or evaluator source')
    cases = _checked(recovery.CASE_INVENTORY)
    expected = {'%s_%03d' % (row['scenario'], row['seed']): row for row in cases['cases']}
    _require(cases['case_count'] == 27 and len(expected) == 27 and set(expected) == set(recovery.CASES),
             'diagnostic case inventory changed')
    for key, row in expected.items():
        _require(all(recovery.CASES[key][name] == row[name] for name in ('scenario', 'seed', 'role')),
                 'diagnostic case role differs from prespecified historical selection')
    return proof


def check_repair_closeout():
    """Validate the immutable r1 failure closeout before entering a new epoch."""
    proof = _checked(protocol.REPAIR_CLOSEOUT)
    diagnostic = _checked(protocol.R1_DIAGNOSTIC_ACCEPTANCE)
    _require(proof.get('scope') == 'survival_dslasso_recovery_r1_failed_cohort_closeout'
             and proof.get('source_release_gate_passed') is True
             and proof.get('whole_cohort_passed') is False
             and proof.get('counts') == dict(validated=565, failed=3, never_started=32)
             and proof.get('diagnostic_acceptance') == protocol.R1_DIAGNOSTIC_ACCEPTANCE
             and proof.get('original_closeout') == protocol.ORIGINAL_CLOSEOUT
             and proof.get('diagnostic_prerequisite_fully_replayed') is True
             and proof.get('original_archive_fully_rechecked') is True
             and diagnostic.get('provenance') == proof.get('provenance')
             and diagnostic.get('diagnostic_evidence_complete') is True
             and diagnostic.get('formal_eligible') is True
             and diagnostic.get('validated_unit_count') == 108
             and diagnostic.get('selected_recipe') == protocol.REPAIR_RECIPE,
             'repair requires the reviewed r1 source-release closeout and selected recipe')
    _checked(proof['review'])
    return proof


def check_integration_reference():
    """Bind the accepted r2 pilot as historical evidence, never current admission."""
    proof = _checked(protocol.R2_DIAGNOSTIC_ACCEPTANCE)
    _require(proof.get('schema_version') == 1 and proof.get('epoch') == 2
             and proof.get('scope') == 'survival_dslasso_repair_diagnostic_acceptance'
             and proof.get('diagnostic_evidence_complete') is True
             and proof.get('formal_eligible') is True
             and proof.get('validated_request_count') == proof.get('validated_unit_count') == 30
             and proof.get('paired_original_cases') == 30
             and proof.get('selected_recipe') == protocol.REPAIR_RECIPE
             and proof.get('original_closeout') == protocol.ORIGINAL_CLOSEOUT
             and proof.get('repair_closeout') == protocol.REPAIR_CLOSEOUT
             and proof.get('r1_diagnostic_acceptance') == protocol.R1_DIAGNOSTIC_ACCEPTANCE
             and proof.get('stationary_checks') == dict.fromkeys(protocol.REPAIR_CASES, True)
             and proof.get('r1_numeric_parity_checks') == dict.fromkeys(protocol.CASES, True),
             'integration requires the complete accepted historical r2 pilot')
    inventory = _checked(proof['inventory'])
    _require(inventory.get('provenance') == proof['provenance']
             and inventory.get('task') == protocol.REPAIR_DIAGNOSTIC_TASK
             and inventory.get('bundle_ready') is True
             and inventory.get('epoch') == 2 and inventory.get('request_count') == 30
             and inventory.get('scientific_cell_count') == 30,
             'historical r2 inventory differs from its acceptance')
    requests = inventory['requests']
    _require(len(requests) == 30 and len(proof['units']) == 30
             and {row['parameters']['case'] for row in requests} == set(protocol.REPAIR_CASES)
             and {row['parameters']['case'] for row in proof['units']} == set(protocol.REPAIR_CASES),
             'historical r2 reference coverage is incomplete')
    for row in requests:
        config = protocol.run_config(protocol.REPAIR_DIAGNOSTIC_TASK, row['parameters'])
        _require(row['configuration'] == config and row['configuration_sha256'] == digest(config)
                 and row['run_id'] == config['run_id']
                 and row['seeds'] == list(protocol.seed_sequence(protocol.REPAIR_DIAGNOSTIC_TASK, row['parameters']))
                 and old._binding(row['path']) == {key: row[key] for key in ('path', 'sha256')},
                 'historical r2 request or scientific configuration changed')
    for key, filename in (('dslasso', 'dslasso.py'), ('method', 'recovery_method.py')):
        path = old.PROJECT_ROOT / 'effect_modifier_experiment/survival' / filename
        _require(old._binding(path) == inventory['review_sources'][key],
                 'integration changed the accepted r2 optimizer implementation')
    return proof


def _old_unit(proof, cell, scenario, seed):
    found = [row for row in proof['validated_units']
             if (row['cell'], row['scenario'], row['seed']) == (cell, scenario, seed)]
    _require(len(found) == 1, 'original closeout lacks a unique paired reference')
    return found[0]


def _old_result(proof, cell, scenario, seed):
    unit = _old_unit(proof, cell, scenario, seed)
    result = _checked(unit['artifacts']['result.json'])
    _require(result['identity'] == unit['identity'], 'historical result identity changed')
    return result


def original_reference(scenario, seed):
    return _old_result(check_original_closeout(), 'fedfisher_diag_shallow', scenario, seed)


def validate_pair(result, reference, config, original_config, formal=False):
    for key in ('dgp', 'pseudo', 'mean_link', 'evaluation', 'public_initial_eta'):
        _require(config[key] == original_config[key], 'new/old scientific comparison recipe differs: ' + key)
    for key in ('training', 'evaluation', 'evaluation_truth'):
        _require(result['fingerprints'][key] == reference['fingerprints'][key],
                 'new/old survival data fingerprints differ: ' + key)
    for key in ('site_sizes', 'pseudo_diagnostics'):
        _require(result[key] == reference[key], 'new/old site or pseudo-value evidence differs')
    if formal:
        for key in ('raw_x_sha256', 'contrast_truth_sha256', 'permutation_sha256'):
            _require(result['evaluation'][key] == reference['evaluation'][key],
                     'new/old evaluator inputs or permutations differ')


def _original_projection(value, template):
    """Ignore only elapsed timings and newly added observer fields."""
    if isinstance(template, dict):
        return {key: _original_projection(value[key], part) for key, part in template.items()
                if key != 'timing_seconds'}
    if isinstance(template, list):
        _require(isinstance(value, list) and len(value) == len(template), 'baseline fit path length changed')
        return [_original_projection(a, b) for a, b in zip(value, template)]
    return value


def baseline_parity(result, reference):
    return (result['fingerprints']['model'] == reference['fingerprints']['model']
            and _original_projection(result['fit'], reference['fit'])
            == _original_projection(reference['fit'], reference['fit']))


def repair_numeric_parity(result, reference):
    """Compare every pre-existing numeric path field; new witnesses are observers."""
    ignored = {'timing_seconds', 'timing_cpu_seconds', 'diagnostic_fingerprint_seconds',
               'elapsed_wall_seconds', 'elapsed_cpu_seconds', 'recovery_schema'}

    def project(value, template):
        if isinstance(template, dict):
            return {key: project(value[key], part) for key, part in template.items() if key not in ignored}
        if isinstance(template, list):
            _require(isinstance(value, list) and len(value) == len(template), 'repair numeric path length changed')
            return [project(a, b) for a, b in zip(value, template)]
        return value

    return (result['fingerprints']['model'] == reference['fingerprints']['model']
            and project(result['fit'], reference['fit']) == project(reference['fit'], reference['fit']))


def integrated_numeric_parity(result, reference):
    """Integration changes no numeric evidence, including v2 guard witnesses."""
    ignored = {'timing_seconds', 'timing_cpu_seconds', 'diagnostic_fingerprint_seconds',
               'elapsed_wall_seconds', 'elapsed_cpu_seconds'}

    def project(value):
        if isinstance(value, dict):
            return {key: project(part) for key, part in value.items() if key not in ignored}
        if isinstance(value, list):
            return [project(part) for part in value]
        return value

    return (result['fingerprints']['model'] == reference['fingerprints']['model']
            and project(result['fit']) == project(reference['fit']))


def select_repair(observations):
    """A validation repair retains its already selected recipe; it does not retune."""
    _require(set(observations) == set(protocol.REPAIR_CASES), 'repair gate needs all 30 unique cases')
    stationary = {case: row['outcome'] == 'stationary' for case, row in observations.items()}
    parity = {case: observations[case].get('r1_numeric_parity') is True for case in protocol.CASES}
    passed = all(stationary.values()) and all(parity.values())
    return dict(selection_rule='fixed_r1_budget_and_step; all_30_stationary; '
                               'all_27_r1_models_and_numeric_paths_unchanged; no_retuning',
        stationary_checks=stationary, r1_numeric_parity_checks=parity,
        formal_eligible=passed, selected_recipe=protocol.REPAIR_RECIPE if passed else None)


def select_integration(observations):
    """Require all 30 current-source paths to reproduce the accepted r2 pilot."""
    _require(set(observations) == set(protocol.REPAIR_CASES), 'integration gate needs all 30 unique cases')
    stationary = {case: row['outcome'] == 'stationary' for case, row in observations.items()}
    parity = {case: row.get('r2_numeric_parity') is True for case, row in observations.items()}
    passed = all(stationary.values()) and all(parity.values())
    return dict(selection_rule='fixed_r2_budget_and_step; all_30_stationary; '
                               'all_30_r2_models_and_numeric_evidence_unchanged; no_retuning',
        stationary_checks=stationary, r2_numeric_parity_checks=parity,
        formal_eligible=passed, selected_recipe=protocol.REPAIR_RECIPE if passed else None)


def _repair_reference(proof, case):
    units = [row for row in proof['units'] if row['parameters'] == dict(case=case, recipe=protocol.REPAIR_RECIPE)]
    _require(len(units) == 1 and units[0]['outcome'] == 'stationary', 'repair lacks a unique stationary r1 reference')
    unit = units[0]
    bindings = [unit['manifest'], *unit['artifacts']]
    for binding in bindings:
        _require(old._binding(binding['path']) == binding, 'archived r1 diagnostic artifact changed')
    old._recheck_producer(unit['producer_evidence'])
    results = [binding for binding in unit['artifacts'] if Path(binding['path']).name == 'diagnostic_result.json']
    _require(len(results) == 1, 'repair reference lacks its diagnostic result')
    return _checked(results[0]), unit


def _integration_reference(proof, case):
    """Revalidate saved r2 artifacts under their original identity without a store."""
    import numpy as np
    from ..infrastructure.provenance import parse_env_file
    from .artifacts import canonical, validate_execution

    parameters = dict(case=case)
    units = [row for row in proof['units'] if row['parameters'] == parameters]
    inventory = _checked(proof['inventory'])
    requests = [row for row in inventory['requests'] if row['parameters'] == parameters]
    _require(len(units) == len(requests) == 1 and units[0]['outcome'] == 'stationary',
             'integration lacks a unique stationary r2 reference')
    unit, request = units[0], requests[0]
    config = protocol.run_config(protocol.REPAIR_DIAGNOSTIC_TASK, parameters)
    seed = protocol.REPAIR_CASES[case]['seed']
    provenance = proof['provenance']
    identity = dict(run_id=config['run_id'], config_sha256=digest(config),
        provenance_sha256=provenance['provenance_sha256'], seed=seed,
        cell=config['cell'], scenario=config['scenario'], recipe=config['recovery_recipe'], case=case)
    manifest = dict(format_version='2', protocol=config['protocol'], run_id=config['run_id'],
        config_json=canonical(config), config_sha256=digest(config), **provenance)
    _require(request['configuration'] == config and unit['seed'] == seed and unit['run_id'] == config['run_id'],
             'historical r2 reference configuration differs')
    expected_root = recovery.artifacts.OUTPUT_ROOT / config['run_id']
    directory = expected_root / ('seed_%03d' % seed)
    names = {'started.json', 'fit_arrays.npz', 'diagnostic_result.json', 'diagnostic_done.json'}
    bindings = {Path(row['path']).name: row for row in unit['artifacts']}
    _require(len(unit['artifacts']) == len(bindings) == 4 and set(bindings) == names
             and unit['manifest']['path'] == str(expected_root / 'run_manifest.env')
             and all(row['path'] == str(directory / name) for name, row in bindings.items()),
             'historical r2 artifact inventory differs')
    for binding in [unit['manifest'], *bindings.values()]:
        _require(old._binding(binding['path']) == binding, 'archived r2 diagnostic artifact changed')
    _require(parse_env_file(unit['manifest']['path']) == manifest,
             'historical r2 manifest differs from its original configuration and provenance')
    result = _checked(bindings['diagnostic_result.json'])
    marker, started = _checked(bindings['diagnostic_done.json']), _checked(bindings['started.json'])
    _require(marker == dict(identity=identity, artifacts={name: bindings[name]['sha256']
                for name in names - {'diagnostic_done.json'}})
             and set(started) == {'identity', 'execution'} and started['identity'] == identity,
             'historical r2 completion or started identity differs')
    validate_execution(started['execution'])
    with np.load(old._ordinary_path(bindings['fit_arrays.npz']['path']), allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), 'duplicate historical r2 model arrays')
        arrays = {name: archive[name] for name in archive.files}
    recovery.validate_result(result, arrays, identity, config)
    job, _ = _completed_job(request)
    _require(old._producer_evidence(request, job, started['execution']) == unit['producer_evidence'],
             'historical r2 producer evidence differs')
    old._recheck_producer(unit['producer_evidence'])
    return result, unit


def select_recipe(observations):
    """Pure prespecified numerical gate; performance truth is never an input."""
    expected = {(case, recipe) for case in recovery.CASES for recipe in recovery.RECIPES}
    _require(set(observations) == expected, 'diagnostic selection needs all 108 unique observations')
    baseline_checks, costs = {}, {}
    for case, context in recovery.CASES.items():
        row = observations[case, 'baseline']
        role = context['role']
        baseline_checks[case] = (row['outcome'] == 'iteration_limit' if role == 'observed_failure'
            else (row['outcome'] == 'stationary' and row.get('baseline_numeric_parity') is True)
            if role in ('slow_success_control', 'original_pilot_control')
            else row['outcome'] in ('stationary', 'iteration_limit'))
    for recipe in SELECTION_ORDER:
        rows = [observations[case, recipe] for case in recovery.CASES]
        calls = [row['loss_gradient_calls'] for row in rows]
        _require(all(type(n) is int and n > 0 for n in calls), 'invalid diagnostic gradient-call cost')
        costs[recipe] = dict(all_stationary=all(row['outcome'] == 'stationary' for row in rows),
                             loss_gradient_calls=sum(calls))
    eligible = [recipe for recipe in SELECTION_ORDER if costs[recipe]['all_stationary']]
    selected = min(eligible, key=lambda recipe: costs[recipe]['loss_gradient_calls']) if eligible else None
    baseline_passed = all(baseline_checks.values())
    if not baseline_passed:
        selected = None
    return dict(selection_rule=SELECTION_RULE, baseline_checks=baseline_checks,
                baseline_reproduction_passed=baseline_passed, candidate_costs=costs,
                formal_eligible=selected is not None, selected_recipe=selected,
                cost_note='call count is a deterministic selection cost, not a measured runtime comparison; '
                          'full/reduced designs and selected paths can have different costs per call')


def default_acceptance_path(provenance=None, cohort='diagnostic', epoch=1):
    _require(cohort in ('diagnostic', 'formal'), 'unknown recovery acceptance cohort')
    provenance = current_provenance() if provenance is None else provenance
    # Reuse the original strict clean-provenance filename validation.
    checked = old.default_acceptance_path(provenance)
    protocol.tasks_for_epoch(epoch)
    root = {1: PROOF_ROOT, 2: REPAIR_PROOF_ROOT, 3: INTEGRATED_PROOF_ROOT}[epoch]
    return old._ordinary_path(root / (cohort + '-' + checked.name[6:]), exists=False)


def _completed_job(request):
    path = old.CENTRAL_JOBS / (request['job_id'] + '.json')
    binding, job = old._binding(path), old._read_json(path)
    _require(job.get('schema_version') == 2 and job.get('project') == 'FedFisher'
             and job.get('job_id') == request['job_id'] and job.get('task') == request['task']
             and job.get('parameters') == request['parameters'] and job.get('state') == 'completed'
             and type(job.get('exit_code')) is int and job['exit_code'] == 0,
             'recovery acceptance needs exact completed central jobs with exit zero')
    logs = {kind + '_log': old._log_evidence(job[kind + '_log'], kind) for kind in ('stdout', 'stderr')}
    return job, dict(job=binding, logs=logs)


def _inventory(path, provenance, cohort, recipe=None, epoch=1):
    from scripts.prepare_survival_recovery_requests import build_requests, bundle_prefix, review_sources
    path = old._ordinary_path(path)
    _require(old.INVENTORY_ROOT in path.parents, 'recovery inventory must be in project scratch reports')
    binding, inventory = old._binding(path), old._read_json(path)
    _require(inventory.get('schema_version') == 1 and inventory.get('bundle_ready') is True
             and inventory.get('bundle') == bundle_prefix(epoch) + '-' + cohort
             and inventory.get('task') == protocol.tasks_for_epoch(epoch)[0 if cohort == 'diagnostic' else 1]
             and inventory.get('formal_publication_authorized') is (cohort == 'formal')
             and inventory.get('inventory_path') == str(path)
             and inventory.get('provenance') == provenance
             and inventory.get('project_head') == provenance['git_commit']
             and inventory.get('worktree_status') == 'clean'
             and inventory.get('review_sources') == review_sources(epoch)
             and inventory.get('original_closeout') == recovery.ORIGINAL_CLOSEOUT
             and inventory.get('case_inventory') == recovery.CASE_INVENTORY
             and inventory.get('selected_recipe') == recipe,
             'recovery inventory must bind exact reviewed clean source')
    if epoch >= 2:
        _require(inventory.get('epoch') == epoch and inventory.get('repair_closeout') == protocol.REPAIR_CLOSEOUT
                 and inventory.get('r1_diagnostic_acceptance') == protocol.R1_DIAGNOSTIC_ACCEPTANCE,
                 'repair inventory must bind the reviewed historical closeout')
    if epoch == 3:
        _require(inventory.get('r2_diagnostic_acceptance') == protocol.R2_DIAGNOSTIC_ACCEPTANCE,
                 'integration inventory must bind the accepted historical r2 pilot')
    requests = inventory['requests']
    count = None if cohort == 'diagnostic' else inventory['shard_count']
    specs = build_requests(inventory['generation'], cohort, count, recipe=recipe, epoch=epoch)
    _require(len(requests) == inventory.get('request_count') == len(specs)
             and inventory.get('scientific_cell_count') == (len(specs) if cohort == 'diagnostic' else 600),
             'recovery request or scientific coverage is incomplete')
    by_job = {item['request']['job_id']: item for item in specs}
    seen, bindings = set(), []
    for record in requests:
        job_id = record['job_id']
        _require(job_id in by_job and job_id not in seen, 'unknown or duplicate recovery request')
        seen.add(job_id)
        spec, target = by_job[job_id], old._ordinary_path(record['path'])
        _require(old.OUTBOX_ROOT in target.parents and target.name == spec['filename'],
                 'unexpected recovery outbox request path')
        request_binding = old._binding(target)
        _require(request_binding['sha256'] == record['sha256'] == spec['request_sha256']
                 and old._read_json(target) == spec['request'], 'recovery request bytes changed')
        _require(all(record[key] == spec[key] for key in ('run_id', 'configuration', 'configuration_sha256'))
                 and record['seeds'] == spec['seed_sequence']
                 and record['task'] == spec['request']['task']
                 and record['parameters'] == spec['request']['parameters'],
                 'recovery inventory configuration or seed coverage changed')
        bindings.append(request_binding)
    return inventory, binding, bindings


def _collect(inventory_path, cohort, epoch=1):
    diagnostic_task, _ = protocol.tasks_for_epoch(epoch)
    cases = protocol.cases_for_task(diagnostic_task)
    provenance, original = current_provenance(), check_original_closeout()
    repair_closeout = check_repair_closeout() if epoch >= 2 else None
    r1_diagnostic = _checked(protocol.R1_DIAGNOSTIC_ACCEPTANCE) if epoch == 2 else None
    r2_diagnostic = check_integration_reference() if epoch == 3 else None
    diagnostic = check_current_diagnostic_gate(epoch=epoch) if cohort == 'formal' else None
    recipe = None if diagnostic is None else diagnostic['selected_recipe']
    inventory, inventory_binding, request_bindings = _inventory(inventory_path, provenance, cohort, recipe, epoch)
    prerequisite = None if diagnostic is None else dict(
        diagnostic_acceptance_sha256=digest(diagnostic), provenance_sha256=provenance['provenance_sha256'],
        selected_recipe=recipe)
    if diagnostic is not None:
        expected = dict(path=str(default_acceptance_path(provenance, epoch=epoch)),
                        sha256=old._binding(default_acceptance_path(provenance, epoch=epoch))['sha256'], evidence=diagnostic)
        _require(inventory.get('pilot_acceptance') == expected, 'formal inventory diagnostic prerequisite changed')
    records, units, central_jobs, observed, paired, old_bindings = {}, [], [], {}, set(), {}
    r1_references, r2_references = [], []
    original_configs = {(row['config']['cell'], row['config']['scenario']): row['config'] for row in original['runs']}
    for request in inventory['requests']:
        job, central = _completed_job(request)
        central_jobs.append(central)
        config = request['configuration']
        store = recovery.RecoveryStore(config, provenance, prerequisite=prerequisite)
        for seed in request['seeds']:
            directory = store.cell_dir(seed)
            artifacts = [old._binding(directory / name) for name in
                         ('started.json', store.model_name, store.result_name, store.done_name)]
            manifest = old._binding(directory.parent / 'run_manifest.env')
            _require(store.status(seed) == 'validated', 'recovery output is incomplete or invalid')
            result = old._read_json(directory / store.result_name)
            receipt = old._read_json(directory / 'started.json')['execution']
            producer = old._producer_evidence(request, job, receipt)
            key = config['scenario'], seed
            reference_unit = _old_unit(original, 'fedfisher_diag_shallow', *key)
            ref_binding = reference_unit['artifacts']['result.json']
            old_bindings[ref_binding['path']] = ref_binding
            reference = _checked(ref_binding)
            validate_pair(result, reference, config, original_configs['fedfisher_diag_shallow', key[0]],
                          formal=cohort == 'formal')
            paired.add(key)
            if cohort == 'diagnostic':
                case = request['parameters']['case']
                selected = request['parameters'].get('recipe', protocol.REPAIR_RECIPE)
                record_key = (case, selected) if epoch == 1 else case
                _require(record_key not in observed, 'duplicate diagnostic case/recipe')
                observation = dict(outcome=result['outcome'], loss_gradient_calls=result['fit']['loss_gradient_calls'])
                context = cases[case]
                if epoch == 1 and selected == 'baseline' and context['role'] in ('slow_success_control', 'original_pilot_control'):
                    old_unit = _old_unit(original, 'dslasso', *key)
                    old_binding = old_unit['artifacts']['result.json']
                    old_bindings[old_binding['path']] = old_binding
                    observation['baseline_numeric_parity'] = (result['outcome'] == 'stationary'
                        and baseline_parity(result, _checked(old_binding)))
                elif epoch == 2 and case in protocol.CASES:
                    r1_result, r1_unit = _repair_reference(r1_diagnostic, case)
                    observation['r1_numeric_parity'] = (result['outcome'] == 'stationary'
                                                       and repair_numeric_parity(result, r1_result))
                    r1_references.append(r1_unit)
                elif epoch == 3:
                    r2_result, r2_unit = _integration_reference(r2_diagnostic, case)
                    observation['r2_numeric_parity'] = (result['outcome'] == 'stationary'
                        and integrated_numeric_parity(result, r2_result))
                    r2_references.append(r2_unit)
                observed[record_key] = observation
            else:
                record_key = key
            _require(record_key not in records, 'duplicate recovery scientific unit')
            records[record_key] = result
            units.append(dict(parameters=request['parameters'], seed=seed, run_id=config['run_id'],
                artifacts=artifacts, manifest=manifest, producer_evidence=producer,
                timing=result['timing'], fit_timing_seconds=result['fit']['timing_seconds'],
                communication=result['fit']['communication'], memory=result['memory'],
                outcome=result.get('outcome', 'stationary'), loss_gradient_calls=result['fit']['loss_gradient_calls']))
    expected = (108 if epoch == 1 else len(cases)) if cohort == 'diagnostic' else 600
    _require(len(units) == expected and len(paired) == (len(cases) if cohort == 'diagnostic' else 600),
             'recovery scientific coverage or original pairing is incomplete')
    if cohort == 'formal':
        _recheck_original_archive(original)
    for binding in [inventory_binding, *inventory['review_sources'].values(), *request_bindings, *old_bindings.values()]:
        _require(old._binding(binding['path']) == binding, 'evidence changed during recovery acceptance')
    for unit in units:
        for binding in [unit['manifest'], *unit['artifacts']]:
            _require(old._binding(binding['path']) == binding, 'recovery artifacts changed during acceptance')
        old._recheck_producer(unit['producer_evidence'])
    for central in central_jobs:
        for binding in [central['job'], *central['logs'].values()]:
            _require(old._binding(binding['path']) == binding, 'central recovery evidence changed')
    for unit in r1_references:
        for binding in [unit['manifest'], *unit['artifacts']]:
            _require(old._binding(binding['path']) == binding, 'r1 diagnostic changed during repair acceptance')
        old._recheck_producer(unit['producer_evidence'])
    for unit in r2_references:
        for binding in [unit['manifest'], *unit['artifacts']]:
            _require(old._binding(binding['path']) == binding, 'r2 diagnostic changed during integration acceptance')
        old._recheck_producer(unit['producer_evidence'])
    if epoch == 2:
        _require(check_repair_closeout() == repair_closeout
                 and _checked(protocol.R1_DIAGNOSTIC_ACCEPTANCE) == r1_diagnostic,
                 'repair historical evidence changed during acceptance')
    if epoch == 3:
        _require(check_repair_closeout() == repair_closeout
                 and check_integration_reference() == r2_diagnostic,
                 'integration historical evidence changed during acceptance')
    _require(current_provenance() == provenance and check_original_closeout() == original,
             'source or historical closeout changed during recovery acceptance')
    proof = dict(schema_version=1, scope='survival_dslasso_recovery_' + cohort + '_acceptance',
        provenance=provenance, inventory=inventory_binding, original_closeout=recovery.ORIGINAL_CLOSEOUT,
        case_inventory=recovery.CASE_INVENTORY, validated_request_count=len(central_jobs),
        validated_unit_count=len(units), paired_original_cases=len(paired), units=units,
        central_jobs=central_jobs, original_reference_bindings=list(old_bindings.values()),
        inference_or_fitting_performed=False,
        admission_note='numerical correctness evidence; operator reviews measured wall/CPU, memory, '
                       'communication and saturation before admission; no superiority or clinical claim')
    if cohort == 'diagnostic':
        proof.update(diagnostic_evidence_complete=True,
                     **{1: select_recipe, 2: select_repair, 3: select_integration}[epoch](observed))
    else:
        proof.update(passed=True, selected_recipe=recipe, diagnostic_prerequisite=prerequisite,
                     archived_fedfisher_units=2400, original_dslasso_successes_used_as_formal_results=False)
    if epoch == 2:
        proof.update(epoch=2, scope='survival_dslasso_repair_' + cohort + '_acceptance',
            repair_closeout=protocol.REPAIR_CLOSEOUT,
            r1_diagnostic_acceptance=protocol.R1_DIAGNOSTIC_ACCEPTANCE,
            r1_diagnostic_reference_units=r1_references)
    if epoch == 3:
        proof.update(epoch=3, scope='survival_dslasso_integrated_' + cohort + '_acceptance',
            repair_closeout=protocol.REPAIR_CLOSEOUT,
            r1_diagnostic_acceptance=protocol.R1_DIAGNOSTIC_ACCEPTANCE,
            r2_diagnostic_acceptance=protocol.R2_DIAGNOSTIC_ACCEPTANCE,
            r2_diagnostic_reference_units=r2_references)
    return proof


def _recheck_original_archive(proof):
    """Final reporting rechecks retained models and producing evidence in full."""
    inventory = _checked(proof['inventory'])
    _checked(proof['pilot_acceptance'])
    for request in inventory['requests']:
        _require(old._binding(request['path']) == {key: request[key] for key in ('path', 'sha256')},
                 'archived formal request changed')
    for row in proof['runs']:
        binding = row['manifest']
        _require(old._binding(binding['path']) == binding, 'archived run manifest changed')
    for row in proof['validated_units']:
        if row['cell'] != 'dslasso':
            for binding in row['artifacts'].values():
                _require(old._binding(binding['path']) == binding, 'archived FedFisher artifact changed')
    for central in proof['central_jobs'].values():
        # Only the accepted FedFisher cohort is mixed into the new comparison.
        if 'dslasso' not in Path(central['job']['path']).name:
            _require(old._binding(central['job']['path']) == central['job'], 'archived central job changed')
            old._recheck_producer(central['producer'])


def collect_diagnostic_acceptance(inventory_path, epoch=1):
    return _collect(inventory_path, 'diagnostic', epoch)


def collect_formal_acceptance(inventory_path, epoch=1):
    return _collect(inventory_path, 'formal', epoch)


def verify_diagnostic_acceptance(path, epoch=1):
    provenance = current_provenance()
    path = old._ordinary_path(path)
    _require(path == default_acceptance_path(provenance, epoch=epoch), 'diagnostic proof must use its canonical source path')
    binding, proof = old._binding(path), old._read_json(path)
    _require(proof.get('provenance') == provenance and proof.get('diagnostic_evidence_complete') is True,
             'diagnostic proof is incomplete or belongs to another source')
    if epoch >= 2:
        scope = 'repair' if epoch == 2 else 'integrated'
        _require(proof.get('epoch') == epoch and proof.get('scope') == 'survival_dslasso_' + scope + '_diagnostic_acceptance',
                 'repair proof cannot inherit an earlier diagnostic epoch')
    fresh = collect_diagnostic_acceptance(proof['inventory']['path'], epoch=epoch)
    _require(fresh == proof and old._binding(path) == binding, 'diagnostic proof differs from current evidence')
    return proof


def check_current_diagnostic_gate(epoch=1):
    proof = verify_diagnostic_acceptance(default_acceptance_path(epoch=epoch), epoch=epoch)
    _require(proof.get('formal_eligible') is True and proof.get('selected_recipe') in SELECTION_ORDER,
             'diagnostic evidence does not qualify a global recipe for formal expansion')
    if epoch >= 2:
        _require(proof.get('selected_recipe') == protocol.REPAIR_RECIPE,
                 'repair proof may not retune the previously selected recipe')
    return proof
