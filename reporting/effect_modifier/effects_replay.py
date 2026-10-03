"""Foreground presentation of the five completed checkpoint replay diagnostics.

Reads validated immutable results only: no data generation, model inference,
fitting, selection or scientific completion writes. Test curves are descriptive.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
DATA_ROOT = Path('/data/del6500/FedFisher')
SCRATCH_ROOT = Path('/scr/del6500/FedFisher')
DEFAULT_OUTPUT = DATA_ROOT / 'effect_modifier_experiment/reports/effects-replay-v1-pilot-cohort-p2'
RECOVERY_OUTPUT = DATA_ROOT / 'effect_modifier_experiment/reports/effects-replay-v1-pilot-cohort-p3-recovered'
PRESENTATION_PYTHON = Path('/opt/anaconda3/bin/python')


def safe_path(path):
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts
            or not any(root in path.parents for root in (DATA_ROOT, SCRATCH_ROOT))):
        raise ValueError('report paths must be below FedFisher data or scratch')
    if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('report paths must not contain symlinks')
    return path


def configure_runtime():
    cache = safe_path(SCRATCH_ROOT / 'effects-replay-report-cache')
    cache.mkdir(parents=True, exist_ok=True)
    safe_path(cache)
    for key in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TMPDIR'):
        os.environ[key] = str(cache)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.dont_write_bytecode = True


def _bound_json(binding):
    path = safe_path(Path(binding['path']))
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != binding['sha256']:
        raise ValueError('recovery input changed after validation: ' + str(path))
    return json.loads(body)


def _retained_record(retained, inventory):
    """Read previously accepted bytes; never impersonate their execution source."""
    from effect_modifier_experiment.dslassocov_run_contract import parse_env_file
    from effect_modifier_experiment.one_shot_effects.diagnostics import protocol
    from effect_modifier_experiment.one_shot_effects.diagnostics.validation import validate_result
    from effect_modifier_experiment.one_shot_effects.wire import canonical
    request = next(item for item in inventory['requests'] if item['job_id'] == retained['job_id'])
    config, provenance = request['configuration'], inventory['provenance']
    digest = hashlib.sha256(canonical(config).encode()).hexdigest()
    if digest != request['configuration_sha256']:
        raise ValueError('retained configuration digest differs')
    identity = dict(run_id=config['run_id'], config_sha256=digest,
        provenance_sha256=provenance['provenance_sha256'], seed=config['seed'],
        recipe=config['recipe'], dataset=config['dataset'])
    bindings = {Path(item['path']).name: item for item in retained['artifacts']}
    if len(retained['artifacts']) != 4 or set(bindings) != {'run_manifest.env', 'started.json', 'result.json', 'done.json'}:
        raise ValueError('retained completed artifact inventory differs')
    path = safe_path(Path(bindings['result.json']['path']))
    if (path.parent.name != 'seed_000' or path.parent.parent.name != config['run_id']
            or any(Path(item['path']) != (path.parent.parent / name if name == 'run_manifest.env'
                   else path.parent / name) for name, item in bindings.items())
            or {item.name for item in path.parent.iterdir()} != {'started.json', 'result.json', 'done.json'}
            or {item.name for item in path.parent.parent.iterdir()} != {'run_manifest.env', 'seed_000'}):
        raise ValueError('retained scientific artifact paths/inventory differ')
    manifest_path = safe_path(Path(bindings['run_manifest.env']['path']))
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != bindings['run_manifest.env']['sha256']:
        raise ValueError('retained manifest changed after validation')
    expected_manifest = dict(format_version='2', protocol=protocol.PROTOCOL,
        run_id=config['run_id'], config_json=canonical(config), config_sha256=digest, **provenance)
    if parse_env_file(manifest_path) != expected_manifest:
        raise ValueError('retained manifest differs from frozen execution configuration/provenance')
    done, started = (_bound_json(bindings[name]) for name in ('done.json', 'started.json'))
    if (set(done) != {'identity', 'artifacts'} or done['identity'] != identity
            or done['artifacts'] != {name: bindings[name]['sha256'] for name in ('started.json', 'result.json')}
            or set(started) != {'identity', 'execution'} or started['identity'] != identity
            or started['execution']['job_id'] != retained['job_id']):
        raise ValueError('retained completion/execution identities differ')
    result = _bound_json(bindings['result.json'])
    validate_result(result, config, identity)
    return dict(**retained['parameters'], result=result, source_path=str(path),
        source_sha256=bindings['result.json']['sha256'], execution_run_id=config['run_id'],
        execution_provenance=provenance, origin='retained_p2')


def load_completed(recovery_evidence=None):
    from effect_modifier_experiment.one_shot_effects.artifacts import current_provenance, read_json
    from effect_modifier_experiment.one_shot_effects.diagnostics import protocol
    from effect_modifier_experiment.one_shot_effects.diagnostics.artifacts import ReplayArtifactStore
    provenance, records = current_provenance(), []
    envelope, retained, evidence_binding = None, {}, None
    if recovery_evidence is not None:
        from effect_modifier_experiment.one_shot_effects.diagnostics.recovery import verify_recovery_evidence
        evidence_path = safe_path(Path(recovery_evidence))
        evidence_binding = dict(path=str(evidence_path), sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest())
        envelope = verify_recovery_evidence(evidence_path)
        inventory = _bound_json(envelope['source_inventory'])
        retained = {(item['parameters']['recipe'], item['parameters']['dataset']): item
                    for item in envelope['retained']}
    for parameters in protocol.cohort_plan():
        previous = retained.get((parameters['recipe'], parameters['dataset']))
        if previous is not None:
            record = _retained_record(previous, inventory)
            record['recovery_evidence'] = evidence_binding
            records.append(record)
            continue
        config = protocol.run_config(protocol.PILOT_TASK, parameters)
        store = ReplayArtifactStore(config, provenance)
        if store.status() != 'validated':
            raise ValueError('all five replay cells must be scientifically validated: ' + config['run_id'])
        path = safe_path(store.unit_dir / 'result.json')
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != read_json(store.unit_dir / 'done.json')['artifacts']['result.json']:
            raise ValueError('completed replay result changed after validation')
        record = dict(**parameters, result=json.loads(body), source_path=str(path), source_sha256=digest,
            execution_run_id=config['run_id'], execution_provenance=provenance,
            origin='replayed_current' if envelope is not None else 'current_source')
        if evidence_binding is not None:
            record['recovery_evidence'] = evidence_binding
        records.append(record)
    if envelope is not None:
        if verify_recovery_evidence(evidence_path) != envelope or _bound_json(evidence_binding) != envelope:
            raise ValueError('recovery evidence changed during report read')
    if current_provenance() != provenance:
        raise ValueError('diagnostic source changed during report read')
    return records, provenance


def tables(records):
    result = {name: [] for name in ('status', 'fisher_gradient_coverage', 'fisher_original_pilot_audit',
        'fisher_prediction_errors', 'fisher_score_agreement',
        'fisher_activation_flips', 'model_quality', 'outcome_prediction', 'dr_path', 'dr_feature_path')}
    for record in records:
        identity = {key: record[key] for key in ('recipe', 'dataset')}
        if 'execution_provenance' in record:
            identity.update(origin=record['origin'], execution_run_id=record['execution_run_id'],
                execution_git_commit=record['execution_provenance']['git_commit'],
                execution_provenance_sha256=record['execution_provenance']['provenance_sha256'])
        diagnostic = record['result']['diagnostic']
        if record['recipe'] == 'dr_path':
            result['status'].append(dict(**identity, diagnostic_state='complete_original_selection_preserved'))
            for basis, detail in diagnostic['basis_results'].items():
                for point in detail['path']:
                    base = dict(**identity, basis=basis, index=point['index'], lambda_value=point['lambda_value'],
                                selected_by_original_V_rule=point['selected_by_original_V_rule'])
                    row = {key: value for key, value in point.items() if not isinstance(value, (list, dict))}
                    result['dr_path'].append(dict(**identity, basis=basis, **row,
                                                 **{'test_' + k: v for k, v in point['test'].items()}))
                    for group in point['feature_groups']:
                        index = group['feature_index']
                        result['dr_feature_path'].append(dict(**base, **group,
                            **{'H_' + key: detail_value[index] for key, detail_value in point['H_ranking'].items()
                               if key in ('rank_min', 'rank_max', 'average_rank')}))
            continue
        endpoint = diagnostic.get('full_H_endpoint')
        gate = diagnostic['implementation_gate']
        state = ('gradient_check_' + gate['status'] if not gate['passed']
                 else 'full_H_endpoint_mismatch' if endpoint and not endpoint['replay_matches_source']
                 else 'diagnostics_complete')
        result['status'].append(dict(**identity, diagnostic_state=state,
            implementation_gate_passed=gate['passed'], implementation_gate_status=gate['status'],
            full_H_replay_matches_source=endpoint['replay_matches_source'] if endpoint else None,
            **{key: gate.get(key) for key in ('stable_checks', 'stable_failures', 'boundary_checks',
                'coverage_complete', 'original_pilot_passed', 'original_pilot_stable_fd_failures')},
            **({'full_H_' + k: v for k, v in endpoint['linearization']['gate'].items()}
               if endpoint else {})))
        for audit in diagnostic['gradient_checks']:
            base = dict(**identity, site_id=audit['site_id'], audit_version=audit['audit_version'],
                        site_status=audit['status'])
            for direction, coverage in audit['coverage'].items():
                for column, count in coverage['stable_checks_by_output'].items():
                    result['fisher_gradient_coverage'].append(dict(**base, direction=direction,
                        output=column, required=coverage['required'], coverage_status=coverage['status'],
                        stable_checks=count, missing_output=column in coverage['missing_outputs']))
            original = audit['original_pilot']
            result['fisher_original_pilot_audit'].append(dict(**base,
                **{key: original[key] for key in ('function', 'passed', 'tau_passed',
                    'derivative_passed', 'stable_fd_checks', 'stable_fd_failures')},
                maximum_tau_absolute_error_vs_new=max(original['tau_absolute_error_vs_new']),
                maximum_tau_absolute_error_vs_model=max(original['tau_absolute_error_vs_model']),
                maximum_derivative_absolute_error_vs_new=max(
                    value for row in original['derivative_absolute_error_vs_new'] for value in row)))
        path = diagnostic.get('parameter_path')
        if path:
            paths = [('target_weighted', path['target_weighted_points'])]
            paths += [(site['site_id'], site['points']) for site in path['sites']]
            for site_id, points in paths:
                for point in points:
                    base = dict(**identity, site_id=site_id, t=point['t'])
                    for scope in ('prediction_errors', 'original_rows_prediction_errors'):
                        for method, moments in point[scope].items():
                            result['fisher_prediction_errors'].append(dict(**base, scope=scope,
                                                                           approximation=method, **moments))
                    for method, agreement in point['score_agreement'].items():
                        scalar = {k: v for k, v in agreement.items() if not isinstance(v, (list, dict))}
                        result['fisher_score_agreement'].append(dict(**base, approximation=method, **scalar))
                    for layer, flips in point['relu_flip_from_local'].items():
                        for arm in ('control', 'treated'):
                            result['fisher_activation_flips'].append(dict(**base, layer=layer, arm=arm, fraction=flips[arm]))
        quality = diagnostic.get('model_quality')
        if quality:
            for model, metrics in quality['test'].items():
                result['model_quality'].append(dict(**identity, model=model, **metrics))
            groups = [('local_site_%03d' % row['site_id'], [row]) for row in quality['local_outcome_metrics']]
            groups += [(model, values['sites']) for model, values in quality['saved_model_outcome_metrics'].items()]
            for model, sites in groups:
                for site in sites:
                    for split in ('T', 'V', 'H'):
                        result['outcome_prediction'].append(dict(**identity, model=model,
                            site_id=site['site_id'], split=split, **site[split]))
            for model, values in quality['saved_model_outcome_metrics'].items():
                result['outcome_prediction'].append(dict(**identity, model=model, site_id='target_test',
                                                         split='independent_test', **values['test']))
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with safe_path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def _save(figure, path, plt):
    path = safe_path(path)
    if path.exists():
        raise FileExistsError('refusing existing report figure')
    figure.tight_layout()
    figure.savefig(path, dpi=140, metadata={'Software': 'FedFisher completed replay reporting'})
    plt.close(figure)


def fisher_figures(record, output, plt):
    diagnostic = record['result']['diagnostic']
    path, quality = diagnostic.get('parameter_path'), diagnostic.get('model_quality')
    prefix = record['recipe'] + '_' + record['dataset']
    files = []
    if path:
        points, figure = path['target_weighted_points'], plt.figure(figsize=(14, 8))
        axes = figure.subplots(2, 4).ravel()
        t = [p['t'] for p in points]
        series = lambda method, metric: [p['prediction_errors'][method][metric] for p in points]
        for method in ('control_logit', 'treated_logit'):
            axes[0].plot(t, series(method, 'rmse'), label=method)
        for method in ('direct_tau', 'sigmoid_of_linear_logits'):
            for axis, metric in zip(axes[1:4], ('rmse', 'mean_error', 'centered_rmse')):
                axis.plot(t, series(method, metric), label=method)
            for axis, metric in zip(axes[4:7], ('maximum_absolute_error', 'spearman_rank_correlation',
                                               'top2_set_agreement_probability')):
                axis.plot(t, [p['score_agreement'][method][metric] for p in points], label=method)
        for layer in points[0]['relu_flip_from_local']:
            for arm in ('control', 'treated'):
                axes[7].plot(t, [p['relu_flip_from_local'][layer][arm] for p in points], label=layer + ' ' + arm)
        titles = ('Binary log-odds RMSE', 'CATE approximation RMSE', 'CATE mean error',
                  'CATE centered RMSE', 'H score maximum absolute error', 'H rank Spearman',
                  'H Top-2 agreement probability', 'ReLU flips from local model')
        for axis, title in zip(axes, titles):
            axis.set_title(title, fontsize=10)
            axis.set_xlabel('t')
            axis.set_xscale('symlog', linthresh=.001)
            axis.grid(alpha=.2)
            axis.legend(fontsize=6)
        figure.suptitle(prefix + ': fixed H sample; diagnostic curve, not the full-H gate', fontsize=11)
        filename = prefix + '_path.png'
        _save(figure, output / filename, plt)
        files.append(filename)
    if quality and any(value['cate_rmse'] is not None for value in quality['test'].values()):
        names = [name for name, value in quality['test'].items() if value['cate_rmse'] is not None]
        labels = {'target_weighted_prediction_average': 'Prediction average (function ensemble)',
                  'parameter_FedAvg': 'Parameter average (FedAvg)'}
        figure, axis = plt.subplots(figsize=(9, max(5, len(names) * .3)))
        axis.barh([labels.get(name, name) for name in names], [quality['test'][name]['cate_rmse'] for name in names])
        axis.invert_yaxis()
        axis.set_xlabel('Independent-test CATE RMSE; diagnostic only')
        axis.set_title(prefix)
        filename = prefix + '_cate_quality.png'
        _save(figure, output / filename, plt)
        files.append(filename)
    return files


def dr_figure(record, output, plt):
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row, (basis, detail) in enumerate(record['result']['diagnostic']['basis_results'].items()):
        points = detail['path']
        lambdas = [p['lambda_value'] for p in points]
        curves = [(key, [p[key] for p in points]) for key in
                  ('validation_risk', 'validation_risk_gap_to_minimum', 'nonzero_groups')]
        curves.append(('test CATE RMSE (diagnostic only)', [p['test']['cate_rmse'] for p in points]))
        for axis, (label, values) in zip(axes[row], curves):
            axis.plot(lambdas, values)
            axis.axvline(detail['original_selected_lambda'], color='black', linestyle='--', label='Original V selection')
            if all(value > 0 for value in lambdas):
                axis.set_xscale('log')
            axis.set_xlabel('lambda')
            axis.set_title(basis + ': ' + label, fontsize=9)
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    figure.suptitle('Saved DR paths: original selection retained; nonzero groups are not significant discoveries')
    filename = 'dr_path_scmv2_base.png'
    _save(figure, output / filename, plt)
    return [filename]


def write_report(records, provenance, output):
    """Render already-validated records; the CLI has no validation bypass."""
    output = safe_path(output)
    if output.exists():
        raise FileExistsError('refusing existing or partial report directory')
    presentation = tables(records)
    configure_runtime()
    needs_figures = any(record['recipe'] == 'dr_path' or record['result']['diagnostic'].get('parameter_path')
                        or record['result']['diagnostic'].get('model_quality') for record in records)
    plt = None
    if needs_figures:
        if importlib.util.find_spec('matplotlib') is None:
            # The science environment owns validation. A sequential, awaited
            # renderer receives JSON only and imports no model/science modules.
            if (not PRESENTATION_PYTHON.is_file()
                    or Path(sys.executable).resolve() == PRESENTATION_PYTHON.resolve()):
                raise RuntimeError('the fixed read-only Matplotlib presentation runtime is unavailable')
            code = ("import json, sys; from pathlib import Path; "
                    "from reporting.effect_modifier.effects_replay import write_report; "
                    "p=json.load(sys.stdin); "
                    "print(json.dumps(write_report(p['records'], p['provenance'], Path(p['output']))))")
            payload = json.dumps(dict(records=records, provenance=provenance, output=str(output)), allow_nan=False)
            child = subprocess.run([str(PRESENTATION_PYTHON), '-B', '-c', code], input=payload,
                text=True, stdout=subprocess.PIPE, check=True, cwd=PROJECT_ROOT, env=dict(os.environ))
            return json.loads(child.stdout)
        import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    safe_path(output)
    files = []
    for name, rows in presentation.items():
        if rows:
            write_csv(output / (name + '.csv'), rows)
            files.append(name + '.csv')
    for record in records:
        files.extend(dr_figure(record, output, plt) if record['recipe'] == 'dr_path'
                     else fisher_figures(record, output, plt))
    execution_provenances = {record['execution_provenance']['provenance_sha256']: record['execution_provenance']
                             for record in records if 'execution_provenance' in record}
    mixed = len(execution_provenances) > 1
    summary = dict(kind='presentation_only_not_scientific_completion',
        diagnostic_provenance=None if mixed else provenance, reporting_source_provenance=provenance,
        mixed_execution_epochs=mixed, execution_provenances=execution_provenances,
        plotting_runtime=dict(python_executable=sys.executable,
                              matplotlib_version=plt.matplotlib.__version__ if plt is not None else None),
        inputs=[{key: item[key] for key in ('recipe', 'dataset', 'source_path', 'source_sha256',
            'execution_run_id', 'execution_provenance', 'origin', 'recovery_evidence') if key in item}
            for item in records],
        status=presentation['status'], selection_changed=False, expansion_authorized=False,
        interpretation='single_seed_diagnostics; test_curves_do_not_select_models; no_significance_or_FDR_claim',
        files={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files})
    with (output / 'report.json').open('x') as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--recovery-evidence', type=Path,
        help='validated P2 retention evidence; combine its three accepted cells with the two current deep replays')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = RECOVERY_OUTPUT if args.recovery_evidence is not None else DEFAULT_OUTPUT
    safe_path(args.output_dir)
    configure_runtime()
    records, provenance = load_completed(args.recovery_evidence)
    summary = write_report(records, provenance, args.output_dir)
    print(json.dumps(dict(report=str(args.output_dir / 'report.json'), input_count=len(records),
                          files=len(summary['files']), expansion_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
