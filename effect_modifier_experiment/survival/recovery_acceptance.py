"""Read-only recovery gates, including explicit comparison to archived v1 fits.

An iteration-limit observation may complete a diagnostic, never a formal fit.
No fitting, data regeneration, model inference or central mutation occurs here.
"""
from pathlib import Path

from ..infrastructure.artifacts import digest
from . import acceptance as old, recovery
from .artifacts import current_provenance

PROOF_ROOT = old.PROOF_ROOT / 'dslasso-recovery-v1'
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


def default_acceptance_path(provenance=None, cohort='diagnostic'):
    _require(cohort in ('diagnostic', 'formal'), 'unknown recovery acceptance cohort')
    provenance = current_provenance() if provenance is None else provenance
    # Reuse the original strict clean-provenance filename validation.
    checked = old.default_acceptance_path(provenance)
    return old._ordinary_path(PROOF_ROOT / (cohort + '-' + checked.name[6:]), exists=False)


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


def _inventory(path, provenance, cohort, recipe=None):
    from scripts.prepare_survival_recovery_requests import build_requests, review_sources
    path = old._ordinary_path(path)
    _require(old.INVENTORY_ROOT in path.parents, 'recovery inventory must be in project scratch reports')
    binding, inventory = old._binding(path), old._read_json(path)
    _require(inventory.get('schema_version') == 1 and inventory.get('bundle_ready') is True
             and inventory.get('bundle') == 'fedfisher-survival-dslasso-recovery-v1-' + cohort
             and inventory.get('task') == (recovery.DIAGNOSTIC_TASK if cohort == 'diagnostic' else recovery.FORMAL_TASK)
             and inventory.get('formal_publication_authorized') is (cohort == 'formal')
             and inventory.get('inventory_path') == str(path)
             and inventory.get('provenance') == provenance
             and inventory.get('project_head') == provenance['git_commit']
             and inventory.get('worktree_status') == 'clean'
             and inventory.get('review_sources') == review_sources()
             and inventory.get('original_closeout') == recovery.ORIGINAL_CLOSEOUT
             and inventory.get('case_inventory') == recovery.CASE_INVENTORY
             and inventory.get('selected_recipe') == recipe,
             'recovery inventory must bind exact reviewed clean source')
    requests = inventory['requests']
    count = None if cohort == 'diagnostic' else inventory['shard_count']
    specs = build_requests(inventory['generation'], cohort, count, recipe=recipe)
    _require(len(requests) == inventory.get('request_count') == len(specs)
             and inventory.get('scientific_cell_count') == (108 if cohort == 'diagnostic' else 600),
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


def _collect(inventory_path, cohort):
    provenance, original = current_provenance(), check_original_closeout()
    diagnostic = check_current_diagnostic_gate() if cohort == 'formal' else None
    recipe = None if diagnostic is None else diagnostic['selected_recipe']
    inventory, inventory_binding, request_bindings = _inventory(inventory_path, provenance, cohort, recipe)
    prerequisite = None if diagnostic is None else dict(
        diagnostic_acceptance_sha256=digest(diagnostic), provenance_sha256=provenance['provenance_sha256'],
        selected_recipe=recipe)
    if diagnostic is not None:
        expected = dict(path=str(default_acceptance_path(provenance)),
                        sha256=old._binding(default_acceptance_path(provenance))['sha256'], evidence=diagnostic)
        _require(inventory.get('pilot_acceptance') == expected, 'formal inventory diagnostic prerequisite changed')
    records, units, central_jobs, observed, paired, old_bindings = {}, [], [], {}, set(), {}
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
                case, selected = request['parameters']['case'], request['parameters']['recipe']
                _require((case, selected) not in observed, 'duplicate diagnostic case/recipe')
                observation = dict(outcome=result['outcome'], loss_gradient_calls=result['fit']['loss_gradient_calls'])
                context = recovery.CASES[case]
                if selected == 'baseline' and context['role'] in ('slow_success_control', 'original_pilot_control'):
                    old_unit = _old_unit(original, 'dslasso', *key)
                    old_binding = old_unit['artifacts']['result.json']
                    old_bindings[old_binding['path']] = old_binding
                    observation['baseline_numeric_parity'] = (result['outcome'] == 'stationary'
                        and baseline_parity(result, _checked(old_binding)))
                observed[case, selected] = observation
                record_key = case, selected
            else:
                record_key = key
            _require(record_key not in records, 'duplicate recovery scientific unit')
            records[record_key] = result
            units.append(dict(parameters=request['parameters'], seed=seed, run_id=config['run_id'],
                artifacts=artifacts, manifest=manifest, producer_evidence=producer,
                timing=result['timing'], fit_timing_seconds=result['fit']['timing_seconds'],
                communication=result['fit']['communication'], memory=result['memory'],
                outcome=result.get('outcome', 'stationary'), loss_gradient_calls=result['fit']['loss_gradient_calls']))
    expected = 108 if cohort == 'diagnostic' else 600
    _require(len(units) == expected and len(paired) == (27 if cohort == 'diagnostic' else 600),
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
        proof.update(diagnostic_evidence_complete=True, **select_recipe(observed))
    else:
        proof.update(passed=True, selected_recipe=recipe, diagnostic_prerequisite=prerequisite,
                     archived_fedfisher_units=2400, original_dslasso_successes_used_as_formal_results=False)
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


def collect_diagnostic_acceptance(inventory_path):
    return _collect(inventory_path, 'diagnostic')


def collect_formal_acceptance(inventory_path):
    return _collect(inventory_path, 'formal')


def verify_diagnostic_acceptance(path):
    provenance = current_provenance()
    path = old._ordinary_path(path)
    _require(path == default_acceptance_path(provenance), 'diagnostic proof must use its canonical source path')
    binding, proof = old._binding(path), old._read_json(path)
    _require(proof.get('provenance') == provenance and proof.get('diagnostic_evidence_complete') is True,
             'diagnostic proof is incomplete or belongs to another source')
    fresh = collect_diagnostic_acceptance(proof['inventory']['path'])
    _require(fresh == proof and old._binding(path) == binding, 'diagnostic proof differs from current evidence')
    return proof


def check_current_diagnostic_gate():
    proof = verify_diagnostic_acceptance(default_acceptance_path())
    _require(proof.get('formal_eligible') is True and proof.get('selected_recipe') in SELECTION_ORDER,
             'diagnostic evidence does not qualify a global recipe for formal expansion')
    return proof
