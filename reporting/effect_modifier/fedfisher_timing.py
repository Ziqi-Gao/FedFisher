"""Export measured CPU Fisher timing from a fully validated completed cohort."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys



def timing_row(result):
    row = {key: result['identity'][key] for key in ('cell', 'variant', 'seed', 'split', 'run_id')}
    row.update(timing_status='measured_new_cpu_rerun', execution_profile='cpu-1core',
               tuning_scope='included_in_aggregation', standalone_selected_fisher_only=True)
    stages = result['timing']['stages']
    for clock in ('wall_seconds', 'cpu_seconds'):
        row['fit_'+clock] = sum(stages[name][clock] for name in ('train', 'tuning', 'aggregation'))
        for name in ('data_preparation', 'train', 'aggregation', 'evaluation'):
            row[name+'_'+clock] = stages[name][clock]
    metric = result['evaluation']['metrics']['permute/tau_prob']['recovery']
    row.update(expected_topk_precision=metric['expected_topk_precision'],
               probability_all_signals_selected=metric['probability_all_signals_selected'],
               auroc=metric['auroc'])
    return row


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='formal')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    from effect_modifier_experiment.fedfisher_timing import protocol
    from effect_modifier_experiment.fedfisher_timing.artifacts import ArtifactStore, current_provenance
    from effect_modifier_experiment.causal_comparison.artifacts import safe_path, read_json, create_bytes
    output = safe_path(args.output_dir)
    if output.exists():
        raise FileExistsError('report directory already exists; preserve without overwrite')
    provenance = current_provenance()
    task = protocol.PILOT_TASK if args.cohort == 'pilot' else protocol.FORMAL_TASK
    count = 1 if args.cohort == 'pilot' else 16
    seeds = range(1 if args.cohort == 'pilot' else 300)
    rows, inputs = [], []
    for cell in protocol.CELLS:
        for variant in protocol.VARIANTS:
            config = protocol.run_config(task, dict(cell=cell, variant=variant,
                                                    shard_index=0, shard_count=count))
            store = ArtifactStore(config, provenance)
            for seed in seeds:
                for split in protocol.SPLITS:
                    if store.status(seed, split) != 'validated':
                        raise ValueError('cohort is incomplete; no complete report will be written')
                    path = store.cell_dir(seed, split)/'result.json'
                    body = path.read_bytes()
                    result = json.loads(body)
                    if hashlib.sha256(body).hexdigest() != read_json(path.parent/'done.json')['artifacts']['result.json']:
                        raise ValueError('result changed after validation')
                    rows.append(timing_row(result))
                    inputs.append(dict(path=str(path), sha256=hashlib.sha256(body).hexdigest()))
    if current_provenance() != provenance:
        raise ValueError('source changed during report validation')
    summaries = []
    metrics = [key for key in rows[0] if key.endswith('_seconds')] + [
        'expected_topk_precision', 'probability_all_signals_selected', 'auroc']
    for cell in protocol.CELLS:
        for variant in protocol.VARIANTS:
            for split in protocol.SPLITS:
                group = [row for row in rows if (row['cell'],row['variant'],row['split']) == (cell,variant,split)]
                summary = dict(cell=cell, variant=variant, split=split, n=len(group))
                for metric in metrics:
                    values = [row[metric] for row in group]
                    summary[metric+'_mean'] = statistics.mean(values)
                    summary[metric+'_se'] = statistics.stdev(values)/(len(values)**0.5) if len(values)>1 else None
                summaries.append(summary)
    output.mkdir(parents=True, exist_ok=False)
    safe_path(output)
    for name, table in (('timing_seed_rows.csv',rows), ('timing_summary.csv',summaries)):
        create_bytes(output/name, csv_bytes(table))
    report = dict(kind='presentation_only_not_scientific_completion', cohort=args.cohort,
        provenance=provenance, scientific_units=len(rows), inputs=inputs,
        historical_gpu_observations_modified=False,
        timing_scope='full actual CPU fit including selected Fisher/compression/validation; evaluation separate',
        files={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
               for name in ('timing_seed_rows.csv','timing_summary.csv')})
    create_bytes(output/'report.json', (json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps(dict(report=str(output/'report.json'), scientific_units=len(rows))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
