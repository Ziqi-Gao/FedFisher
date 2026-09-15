#!/usr/bin/env python3
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
DEFAULT_OUTPUT = DATA_ROOT / 'effect_modifier_experiment/reports/effects-replay-v1-pilot-cohort-p1'
PRESENTATION_PYTHON = Path('/opt/anaconda3/bin/python')
sys.path.insert(0, str(PROJECT_ROOT))


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


def load_completed():
    from effect_modifier_experiment.one_shot_effects.artifacts import current_provenance, read_json
    from effect_modifier_experiment.one_shot_effects.diagnostics import protocol
    from effect_modifier_experiment.one_shot_effects.diagnostics.artifacts import ReplayArtifactStore
    provenance, records = current_provenance(), []
    for parameters in protocol.cohort_plan():
        config = protocol.run_config(protocol.PILOT_TASK, parameters)
        store = ReplayArtifactStore(config, provenance)
        if store.status() != 'validated':
            raise ValueError('all five replay cells must be scientifically validated: ' + config['run_id'])
        path = safe_path(store.unit_dir / 'result.json')
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != read_json(store.unit_dir / 'done.json')['artifacts']['result.json']:
            raise ValueError('completed replay result changed after validation')
        records.append(dict(**parameters, result=json.loads(body), source_path=str(path), source_sha256=digest))
    if current_provenance() != provenance:
        raise ValueError('diagnostic source changed during report read')
    return records, provenance


def tables(records):
    result = {name: [] for name in ('status', 'fisher_prediction_errors', 'fisher_score_agreement',
        'fisher_activation_flips', 'model_quality', 'outcome_prediction', 'dr_path', 'dr_feature_path')}
    for record in records:
        identity = {key: record[key] for key in ('recipe', 'dataset')}
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
        state = ('gradient_check_failed_or_inconclusive' if not diagnostic['implementation_gate']['passed']
                 else 'full_H_endpoint_mismatch' if endpoint and not endpoint['replay_matches_source']
                 else 'diagnostics_complete')
        result['status'].append(dict(**identity, diagnostic_state=state,
            implementation_gate_passed=diagnostic['implementation_gate']['passed'],
            full_H_replay_matches_source=endpoint['replay_matches_source'] if endpoint else None,
            **{key: diagnostic['implementation_gate'].get(key) for key in
               ('stable_checks', 'stable_failures', 'boundary_checks')},
            **({'full_H_' + k: v for k, v in endpoint['linearization']['gate'].items()}
               if endpoint else {})))
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
                    "from scripts.report_effects_replay import write_report; "
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
    summary = dict(kind='presentation_only_not_scientific_completion', diagnostic_provenance=provenance,
        plotting_runtime=dict(python_executable=sys.executable,
                              matplotlib_version=plt.matplotlib.__version__ if plt is not None else None),
        inputs=[{key: item[key] for key in ('recipe', 'dataset', 'source_path', 'source_sha256')} for item in records],
        status=presentation['status'], selection_changed=False, expansion_authorized=False,
        interpretation='single_seed_diagnostics; test_curves_do_not_select_models; no_significance_or_FDR_claim',
        files={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files})
    with (output / 'report.json').open('x') as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    safe_path(args.output_dir)
    configure_runtime()
    records, provenance = load_completed()
    summary = write_report(records, provenance, args.output_dir)
    print(json.dumps(dict(report=str(args.output_dir / 'report.json'), input_count=len(records),
                          files=len(summary['files']), expansion_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
