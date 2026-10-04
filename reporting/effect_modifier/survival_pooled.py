"""Deterministic summaries and plots of already-accepted prediction artifacts."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from effect_modifier_experiment.survival.acceptance import _binding, _ordinary_path, _read_json
from effect_modifier_experiment.survival.artifacts import array_hash
from effect_modifier_experiment.survival_pooled.evaluation import summarize_arrays, compare_predictions
from effect_modifier_experiment.survival_pooled import protocol

REPORT_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-pooled-v1')
RUN_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/runs')


def load_completed(proof_path):
    """Read accepted hashes/arrays only; never restore or apply a model."""
    proof_path = _ordinary_path(proof_path)
    binding, proof = _binding(proof_path), _read_json(proof_path)
    if (proof.get('protocol') != 'survival-pooled-prediction-v1'
            or proof.get('scope') not in ('survival_pooled_pilot_scientific_acceptance',
                                        'survival_pooled_formal_scientific_acceptance')
            or proof.get('passed') is not True or proof.get('inference_performed') is not False
            or len(proof.get('units', [])) != proof.get('validated_unit_count')):
        raise ValueError('reporting requires completed pooled scientific acceptance')
    units = proof['units']
    seeds = protocol.PILOT_SEEDS if 'pilot' in proof['scope'] else protocol.FORMAL_SEEDS
    expected_units = {(family, scenario, seed) for family in protocol.FAMILIES
                      for scenario in protocol.SCENARIOS for seed in seeds}
    actual_units = [(u['family'], u['scenario'], u['seed']) for u in units]
    if len(set(actual_units)) != len(actual_units) or set(actual_units) != expected_units:
        raise ValueError('report needs exact unique accepted seed/family/scenario coverage')
    rows = []
    for unit in proof['units']:
        if len(unit['artifacts']) != 4:
            raise ValueError('duplicate or incomplete accepted artifact inventory')
        files = {Path(item['path']).name: item for item in unit['artifacts']}
        if set(files) != {'started.json', 'model.npz', 'result.json', 'done.json'}:
            raise ValueError('accepted report artifact inventory mismatch')
        for item in (unit['manifest'], *files.values()):
            if RUN_ROOT not in Path(item['path']).parents or _binding(item['path']) != item:
                raise ValueError('accepted prediction artifact changed')
        result = _read_json(files['result.json']['path'])
        done = _read_json(files['done.json']['path'])
        started = _read_json(files['started.json']['path'])
        directory = RUN_ROOT / unit['run_id'] / ('seed_%03d' % unit['seed'])
        if (any(Path(item['path']).parent != directory for item in files.values())
                or Path(unit['manifest']['path']) != directory.parent / 'run_manifest.env'
                or result['identity'] != done['identity'] or result['identity'] != started['identity']
                or result['identity']['run_id'] != unit['run_id'] or result['status'] != 'complete'
                or any(result['identity'][key] != unit[key] for key in ('seed', 'scenario'))
                or result['identity']['cell'] != unit['family']
                or done['artifacts'] != {key: files[key]['sha256'] for key in
                                          ('started.json', 'model.npz', 'result.json')}):
            raise ValueError('report completion evidence mismatch')
        with np.load(files['model.npz']['path'], allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError('duplicate prediction arrays')
            arrays = {name: archive[name] for name in archive.files}
        if array_hash(arrays) != result['prediction_arrays_sha256']:
            raise ValueError('saved prediction array checksum mismatch')
        truth = {key: arrays['truth__' + key] for key in ('eta0', 'eta1', 'survival0', 'survival1')}
        tags = set(protocol.historical_tags(unit['family'])) | {'pooled_' + scope for scope in protocol.PSEUDO_SCOPES}
        if set(result['prediction']) != tags:
            raise ValueError('incomplete or unknown reporting model coverage')
        saved_predictions = {}
        local_rows = []
        for tag, summary in result['prediction'].items():
            predictions = {key: arrays[tag + '__prediction__' + key] for key in ('eta0', 'eta1', 's0', 's1')}
            if summarize_arrays(predictions, truth, arrays['factual_a']) != summary:
                raise ValueError('report metrics differ from accepted prediction arrays')
            saved_predictions[tag] = predictions
            local_rows.append(dict(family=unit['family'], scenario=unit['scenario'], seed=unit['seed'],
                                   method=tag, summary=summary, gaps={}))
        if set(result['comparisons']) != set(protocol.historical_tags(unit['family'])):
            raise ValueError('incomplete FL-pooled comparison coverage')
        for record in local_rows:
            if record['method'] in result['comparisons']:
                gaps = result['comparisons'][record['method']]
                if set(gaps) != set(protocol.PSEUDO_SCOPES) or any(gaps[scope] != compare_predictions(
                        saved_predictions[record['method']], saved_predictions['pooled_' + scope])
                        for scope in protocol.PSEUDO_SCOPES):
                    raise ValueError('FL-pooled gaps differ from saved predictions')
                record['gaps'] = gaps
        rows.extend(local_rows)
        for item in files.values():
            if _binding(item['path']) != item:
                raise ValueError('prediction artifact changed while reporting')
    if _binding(proof_path) != binding:
        raise ValueError('acceptance proof changed while reporting')
    return proof, rows


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['family'], row['scenario'], row['method']), []).append(row)
    summary = []
    for (family, scenario, method), items in sorted(groups.items()):
        seeds = [item['seed'] for item in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError('duplicate reporting seed')
        result = dict(family=family, scenario=scenario, method=method, seeds=len(items))
        for target in ('combined', 'factual', 'contrast', 'control', 'treatment'):
            for metric in ('rmse', 'mae', 'bias'):
                values = np.array([(row['summary']['arms']['0' if target == 'control' else '1']
                    if target in ('control', 'treatment') else row['summary'][target])[metric]
                    for row in items])
                name = target + '_' + metric
                result[name + '_mean'] = float(values.mean())
                result[name + '_se'] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.
        summary.append(result)
    return summary


def _labels(tag):
    return {'dslasso': 'dsLasso', 'fedfisher_diag': 'FedFisher Diag',
        'fedfisher_kfac': 'FedFisher KFAC', 'pooled_site_local_pv': 'Pooled: site pseudo-values',
        'pooled_pooled_km_pv': 'Pooled: global pseudo-values'}[tag]


def aggregate_gaps(rows):
    groups = {}
    for row in rows:
        for scope, gap in row['gaps'].items():
            groups.setdefault((row['family'], row['scenario'], row['method'], scope), []).append(gap)
    summary = []
    for (family, scenario, method, scope), gaps in sorted(groups.items()):
        record = dict(family=family, scenario=scenario, federated_method=method,
                      pooled_pseudo_scope=scope, seeds=len(gaps))
        for target in ('combined', 'control', 'treatment'):
            values = np.array([(gap['combined'] if target == 'combined' else
                                gap['arms']['0' if target == 'control' else '1'])['rmse'] for gap in gaps])
            record[target + '_rms_gap_mean'] = float(values.mean())
            record[target + '_rms_gap_se'] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.
        summary.append(record)
    return summary


def make_figures(rows, summary, output):
    cache = Path('/scr/del6500/FedFisher/survival-pooled-reporting/matplotlib')
    if any(p.is_symlink() for p in (cache, *cache.parents)) or cache.resolve(strict=False) != cache:
        raise ValueError('plot cache must be an ordinary FedFisher scratch path')
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families = ('linear', 'shallow', 'deep')
    scenarios = ('iid', 'covariate_shift')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        for col_index, family in enumerate(families):
            axis = axes[row_index, col_index]
            records = [r for r in summary if r['family'] == family and r['scenario'] == scenario]
            for index, record in enumerate(records):
                axis.errorbar(record['combined_rmse_mean'], index, xerr=record['combined_rmse_se'],
                              fmt='o', capsize=3)
            axis.set_yticks(range(len(records)))
            axis.set_yticklabels([_labels(r['method']) for r in records], fontsize=8)
            axis.set_title(family.title() + ' / ' + scenario.replace('_', ' '))
            axis.set_xlabel('Survival probability RMSE (mean ± 1 SE)')
            axis.set_xlim(left=0)
            axis.grid(axis='x', alpha=.2)
    fig.suptitle('Prediction accuracy against simulation truth at tau = 3')
    fig.savefig(output / 'survival_probability_rmse.png', dpi=220)
    fig.savefig(output / 'survival_probability_rmse.pdf')
    plt.close(fig)
    # Pool only already-saved bin sums across seeds and both treatment arms.
    # Each arm-specific calibration remains available in result.json.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for i, scenario in enumerate(scenarios):
        for j, family in enumerate(families):
            axis = axes[i, j]
            local = [r for r in rows if r['family'] == family and r['scenario'] == scenario]
            for tag in sorted({r['method'] for r in local}):
                counts, predictions, true = np.zeros(10), np.zeros(10), np.zeros(10)
                for record in local:
                    if record['method'] != tag:
                        continue
                    for bins in record['summary']['calibration'].values():
                        for b in bins:
                            k, n = b['bin'], b['count']
                            counts[k] += n
                            if n:
                                predictions[k] += n * b['predicted_mean']
                                true[k] += n * b['true_mean']
                mask = counts > 0
                axis.plot(predictions[mask] / counts[mask], true[mask] / counts[mask], 'o-',
                          markersize=3, label=_labels(tag))
            axis.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted survival',
                     ylabel='Mean oracle survival', title=family.title() + ' / ' + scenario.replace('_', ' '))
            axis.legend(fontsize=7, loc='upper left')
    fig.suptitle('Calibration to oracle probabilities; both arms and all accepted seeds')
    fig.savefig(output / 'survival_probability_calibration.png', dpi=220)
    fig.savefig(output / 'survival_probability_calibration.pdf')
    plt.close(fig)


def write_report(proof_path, output):
    output = Path(output)
    if (not output.is_absolute() or REPORT_ROOT not in output.parents or '..' in output.parts
            or any(p.is_symlink() for p in (output, *output.parents)) or output.exists()):
        raise ValueError('report output must be a new ordinary directory below the pooled report root')
    proof, rows = load_completed(proof_path)
    summary = aggregate(rows)
    output.mkdir(parents=True)
    with (output / 'probability_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    gaps = aggregate_gaps(rows)
    with (output / 'FL_pooled_prediction_gaps.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gaps[0]))
        writer.writeheader()
        writer.writerows(gaps)
    make_figures(rows, summary, output)
    metadata = dict(schema_version=1, acceptance=_binding(proof_path),
        scientific_units=proof['validated_unit_count'], inference_performed=False,
        uncertainty='mean and one standard error of per-seed metrics; no fitting',
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())})
    (output / 'report.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    return metadata
