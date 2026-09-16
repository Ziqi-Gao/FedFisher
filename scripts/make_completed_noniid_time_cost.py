#!/usr/bin/env python3
"""Read accepted artifacts and plot recorded non-IID time costs, without fitting.

Only presentation tables/figures are written. No scheduler, training, inference,
dataset generation, or scientific-manifest imports or mutations are involved.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import make_completed_noniid_visualizations as base


STAGES = ('data_preparation', 'train', 'tuning', 'aggregation', 'evaluation')
CLOCKS = ('wall_seconds', 'cpu_seconds')
METRICS = tuple(f'{scope}_{clock}' for scope in ('fit', 'evaluation', 'data_preparation')
                for clock in CLOCKS)
TABLES = ('completed_noniid_time_cost_seed_rows.csv', 'completed_noniid_time_cost_summary.csv')
FIGURES = ('noniid_all_methods_time_cost', 'noniid_multiround_fedavg_time_cost',
           'noniid_logistic_lasso_time_cost')
PROVENANCE = 'completed_noniid_time_cost_provenance.json'
README = 'NONIID_TIME_COST_README.md'
require = base.require


def finite_time(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'time must be finite and nonnegative')
    return float(value)


def timing_row(result, execution):
    identity, timing, fit = result['identity'], result['timing'], result['fit']
    cell = identity['cell']
    require(identity['split'] == 'noniid' and cell in base.CELLS, 'not a requested non-IID cell')
    require(identity['run_id'] == f"comparison_{cell}_{identity['variant']}_v1_300seeds",
            'not the formal 300-seed namespace')
    require(result['status'] == 'complete' and fit['completion_eligible'] is True,
            'scientific result is not complete')
    allocation = execution['allocation']
    require(execution['execution_profile'] == 'cpu-1core' and
            allocation['cpu_cores'] == 1 and allocation['gpu_count'] == 0,
            'incomparable execution profile')
    require(timing['schema_version'] == 1 and timing['method'] == cell and
            timing['execution_model'] == 'single_process_simulation' and
            timing['actual_network_measured'] is False, 'unexpected timing contract')
    require(timing['clock']['wall'] == 'perf_counter' and
            timing['clock']['cpu'] == 'process_time', 'unexpected clocks')
    stages = timing['stages']
    require(set(stages) == set(STAGES), 'unexpected timing stages')
    for name, stage in stages.items():
        require(type(stage['measured']) is bool and stage['failed_calls'] == 0,
                'invalid or failed timing stage')
        require(type(stage['calls']) is int and stage['calls'] >= 0 and
                stage['measured'] == (stage['calls'] > 0), 'timing call count mismatch')
        for clock in CLOCKS:
            value = finite_time(stage[clock])
            require(stage['measured'] or value == 0, 'unmeasured time contains a value')
    require(all(stages[s]['measured'] for s in ('train', 'evaluation', 'data_preparation')),
            'missing required timing stage')
    # The accepted implementation includes tuning inside train, not in a
    # standalone tuning bucket. Never label its unmeasured bucket zero-cost.
    require(not stages['tuning']['measured'], 'unexpected separately timed tuning')
    separate_aggregation = cell == 'local_logistic_lasso' or cell.startswith('one_shot_fedavg_')
    require(stages['aggregation']['measured'] == separate_aggregation,
            'aggregation scope differs from the accepted implementation')
    if cell in ('dslassocov', 'local_logistic_lasso') or cell.startswith('one_shot_'):
        require(fit['timing_contract']['training_includes_all_selection_and_refits'] is True,
                'incomplete selection/refit scope')
    row = {k: identity[k] for k in ('cell', 'variant', 'seed', 'split', 'run_id')}
    row.update(execution_profile=execution['execution_profile'],
               producer_job_id=execution['job_id'], tuning_scope='included_in_fit_not_separately_measured',
               aggregation_scope='separate_bucket' if separate_aggregation else 'included_in_train')
    for clock in CLOCKS:
        measured_sum = sum(stages[s][clock] for s in STAGES if stages[s]['measured'])
        require(math.isclose(measured_sum, finite_time(timing['measured_stage_sum'][clock]),
                             rel_tol=1e-10, abs_tol=1e-8), 'timing sum mismatch')
        row[f'fit_{clock}'] = sum(stages[s][clock] for s in ('train', 'tuning', 'aggregation')
                                if stages[s]['measured'])
        for stage in ('evaluation', 'data_preparation'):
            row[f'{stage}_{clock}'] = stages[stage][clock]
    require(row['fit_wall_seconds'] > 0, 'missing positive measured fit time')
    return row


def collect():
    """Recheck the frozen acceptance ledger and extract times from those bytes.

    Verification is deliberately independent of current source provenance: this
    report reads a historical accepted cohort, never resumes or re-certifies it.
    """
    evidence_bytes = (base.AUDIT / 'old-cohort.scientific-evidence.json').read_bytes()
    require(base.sha(evidence_bytes) == base.EVIDENCE_SHA, 'accepted evidence hash changed')
    evidence = json.loads(evidence_bytes)
    require(evidence['validated_scientific_units'] == 57600 and
            len(evidence['artifacts']['manifests']) == 96, 'unexpected accepted cohort')
    all_digest, rows = hashlib.sha256(), []
    for index, saved in enumerate(evidence['artifacts']['manifests'], 1):
        manifest = base.safe_path(saved['manifest_path'], (base.OUTPUTS,), True)
        require(base.sha(manifest.read_bytes()) == saved['manifest_sha256'], 'manifest changed')
        directory = manifest.parent
        manifest_values = base.read_env(manifest)
        expected = {'run_manifest.env'} | {f'seed_{s:03d}_{split}' for s in range(300)
                                           for split in ('iid', 'noniid')}
        require({p.name for p in directory.iterdir()} == expected, f'incomplete run: {directory}')
        digest = hashlib.sha256()
        for seed in range(300):
            for split in ('iid', 'noniid'):
                unit = base.safe_path(directory / f'seed_{seed:03d}_{split}', (base.OUTPUTS,), True)
                require({p.name for p in unit.iterdir()} ==
                        {'done.json', 'started.json', 'result.json', 'model.npz'}, f'invalid unit: {unit}')
                done_bytes = base.safe_path(unit / 'done.json', (base.OUTPUTS,), True).read_bytes()
                done = json.loads(done_bytes)
                require(set(done['artifacts']) == {'started.json', 'result.json', 'model.npz'},
                        'invalid artifact inventory')
                raw, hashes = {}, {}
                for name in sorted(done['artifacts']):
                    data = base.safe_path(unit / name, (base.OUTPUTS,), True).read_bytes()
                    hashes[name] = base.sha(data)
                    if name != 'model.npz':
                        raw[name] = data
                require(hashes == done['artifacts'], f'artifact hash changed: {unit}')
                started, identity = json.loads(raw['started.json']), done['identity']
                require(identity == started['identity'] and identity['seed'] == seed and
                        identity['split'] == split and identity['run_id'] == saved['run_id'] and
                        identity['config_sha256'] == manifest_values['config_sha256'] and
                        identity['provenance_sha256'] == manifest_values['provenance_sha256'],
                        f'identity mismatch: {unit}')
                line = base.encoded(dict(relative_path=unit.name, done_sha256=base.sha(done_bytes),
                                         artifacts=hashes, identity=identity,
                                         producer_job_id=started['execution']['job_id']))
                digest.update(line)
                all_digest.update(line)
                if split == 'noniid' and identity['cell'] in base.CELLS:
                    result = json.loads(raw['result.json'])
                    require(result['identity'] == identity, 'result identity mismatch')
                    rows.append(timing_row(result, started['execution']))
        require(digest.hexdigest() == saved['unit_hash_ledger_sha256'], 'accepted run ledger changed')
        if index % 4 == 0:
            print(f'Validated {index}/96 accepted manifests; extracted {len(rows)} timing rows', flush=True)
    require(all_digest.hexdigest() == evidence['artifacts']['all_unit_hash_ledger_sha256'],
            'accepted global ledger changed')
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        require(row['split'] == 'noniid' and row['cell'] in base.CELLS and
                row['variant'] in base.VARIANTS, 'unexpected plotting row')
        groups[row['cell'], row['variant']].append(row)
    require(set(groups) == {(c, v) for c in base.CELLS for v in base.VARIANTS},
            'incomplete method/variant coverage')
    output = []
    for cell in base.CELLS:
        for variant in base.VARIANTS:
            samples = groups[cell, variant]
            require(len(samples) == 300 and {r['seed'] for r in samples} == set(range(300)),
                    'incomplete or duplicate seed coverage')
            summary = dict(cell=cell, variant=variant, split='noniid', n=300)
            for metric in METRICS:
                values = [finite_time(r[metric]) for r in samples]
                mean, se = base.mean_se(values)
                summary.update({f'{metric}_mean': mean, f'{metric}_se': se,
                                f'{metric}_median': statistics.median(values),
                                f'{metric}_min': min(values), f'{metric}_max': max(values)})
            output.append(summary)
    return output


def lookup_rows(summaries):
    return {(r['cell'], r['variant']): r for r in summaries}


def plot_overview(summaries, dest):
    canvas = base.SVG(1640, 1230)
    canvas.text(30, 45, 'Non-IID | Recorded time cost across completed methods', size=28, bold=True)
    canvas.text(30, 79, 'CPU, 1 core | 300 formal seeds per condition | Mean +/- 1 SE | No pilot or Group LASSO', size=17)
    base.variant_legend(canvas, 116)
    lookup = lookup_rows(summaries)
    top, row_height = 220, 36
    for panel, (metric, title) in enumerate((('fit_wall_seconds', 'A  Full fit (including selection and refit)'),
                                            ('evaluation_wall_seconds', 'B  Recorded evaluation workload'))):
        left, width = 505 + panel * 560, 505
        values = [r[f'{metric}_mean'] + sign * r[f'{metric}_se']
                  for r in summaries for sign in (-1, 1)]
        require(min(values) > 0, 'nonpositive interval on a log-time axis')
        lo, hi = math.log10(min(values)) - 0.18, math.log10(max(values)) + 0.18
        sx = lambda value: left + (math.log10(value) - lo) / (hi - lo) * width
        canvas.text(left, 168, title, size=18, bold=True)
        bottom = top + len(base.CELLS) * row_height
        ticks = [factor * 10 ** exponent for exponent in range(math.floor(lo), math.ceil(hi)+1)
                 for factor in (1, 2, 5) if lo <= math.log10(factor * 10 ** exponent) <= hi]
        for tick in ticks:
            x = sx(tick)
            canvas.line(x, top - 18, x, bottom, '#cbd5e1')
            canvas.text(x, bottom + 25, f'{tick:g}', size=14, anchor='middle')
        canvas.text(left + width/2, bottom + 57, 'Wall-clock seconds per seed (log scale)', size=16, anchor='middle')
        for i, cell in enumerate(base.CELLS):
            y = top + i * row_height
            if i % 2 == 0:
                canvas.rect(left, y - 15, width, row_height, '#e2e8f0', 0.35)
            if panel == 0:
                canvas.text(left - 18, y + 9, base.label(cell), size=15, anchor='end')
            for j, variant in enumerate(base.VARIANTS):
                row = lookup[cell, variant]
                mean, se = row[f'{metric}_mean'], row[f'{metric}_se']
                yy = y + (j - 1.5) * 6
                canvas.line(sx(mean-se), yy, sx(mean+se), yy, base.COLORS[j], 1.5)
                canvas.marker(sx(mean), yy, base.COLORS[j], j, 3)
    canvas.text(30, 1105, 'Full fit includes all candidate fits, validation/selection, refitting and aggregation; excludes data generation and evaluation.', size=16)
    canvas.text(30, 1137, 'Evaluation is separate: outcome methods also run logit diagnostics; R methods do not. Evaluation workloads are not identical.', size=16)
    canvas.text(30, 1169, 'Historical GPU FedFisher: N/A here (no comparable full-fit timing in this accepted CPU cohort); missing is NOT zero.', size=16, bold=True)
    canvas.text(30, 1201, 'Serial simulation measurements, not deployed parallel latency; queueing, real networking and artifact I/O are excluded.', size=16, color='#475569')
    base.save_svg(canvas, dest, FIGURES[0])


def plot_fedavg(summaries, dest):
    canvas = base.SVG(1560, 1110)
    canvas.text(30, 46, 'Non-IID | FedAvg full-fit time by local-epoch setting', size=28, bold=True)
    canvas.text(30, 78, 'CPU, 1 core | Mean +/- 1 SE over 300 formal seeds | Selection and final refit included', size=17)
    base.variant_legend(canvas, 115)
    lookup = lookup_rows(summaries)
    for col, stop in enumerate(('budget30', 'patience')):
        values = [r['fit_wall_seconds_mean'] + r['fit_wall_seconds_se'] for r in summaries
                  if r['cell'].startswith('fedavg_') and r['cell'].endswith(stop)]
        high = math.ceil(max(values) * 1.10 / 5) * 5
        for row, depth in enumerate(('shallow', 'deep')):
            title = f'{depth.capitalize()} MLP | ' + ('30 total local epochs' if stop == 'budget30' else 'Validation patience + refit')
            sx, sy = base.axes(canvas, 105+765*col, 190+425*row, 635, 310, title,
                               (0, high), [high*i/5 for i in range(6)],
                               [(i/3, str(e)) for i, e in enumerate((1, 5, 10, 30))],
                               'Local epochs per round (E)', 'Full-fit wall time (seconds)', '.1f')
            for j, variant in enumerate(base.VARIANTS):
                points = []
                for i, epoch in enumerate((1, 5, 10, 30)):
                    item = lookup[f'fedavg_{depth}_e{epoch}_{stop}', variant]
                    mean, se = item['fit_wall_seconds_mean'], item['fit_wall_seconds_se']
                    x, y = sx(i/3)+(j-1.5)*4, sy(mean)
                    points.append((x, y))
                    canvas.error(x, sy(mean-se), sy(mean+se), base.COLORS[j])
                canvas.polyline(points, base.COLORS[j], 2)
                for x, y in points:
                    canvas.marker(x, y, base.COLORS[j], j, 4.5)
    canvas.text(30, 1020, 'Shared y scale within each column. Patience includes selection training, validation and the complete final refit.', size=17)
    canvas.text(30, 1051, 'Budget 30: E = 1 / 5 / 10 / 30 corresponds to 30 / 6 / 3 / 1 training aggregation rounds.', size=16)
    canvas.text(30, 1080, 'E also changes optimizer resets and minibatch streams. No queueing or real-network time; evaluation excluded.', size=16, color='#475569')
    base.save_svg(canvas, dest, FIGURES[1])


def plot_lasso(summaries, dest):
    canvas = base.SVG(1560, 1110)
    canvas.text(30, 46, 'Non-IID | Logistic LASSO: one-shot and multiround time cost', size=27, bold=True)
    canvas.text(30, 78, 'CPU, 1 core | Mean +/- 1 SE over 300 formal seeds | Saved timing observations only', size=17)
    cells = ('dslassocov', 'local_logistic_lasso')
    colors = (base.COLORS[0], base.COLORS[2])
    for i, cell in enumerate(cells):
        x = 350 + i * 500
        canvas.rect(x, 103, 24, 14, colors[i])
        canvas.text(x+34, 116, 'Multiround dsLassoCov-style' if i == 0 else 'One-shot local Logistic LASSO', size=17)
    lookup = lookup_rows(summaries)
    for row, scope in enumerate(('fit', 'evaluation')):
        for col, clock in enumerate(CLOCKS):
            metric = f'{scope}_{clock}'
            values = [lookup[c, v][f'{metric}_mean'] + lookup[c, v][f'{metric}_se']
                      for c in cells for v in base.VARIANTS]
            step = 1 if max(values) < 8 else 5
            high = max(1, math.ceil(max(values) * 1.20 / step) * step)
            title = ('Full fit: CV + final refit + aggregation' if row == 0 else 'Evaluation (separate from fitting)')
            sx, sy = base.axes(canvas, 105+765*col, 190+425*row, 635, 310, title,
                               (0, high), [high*i/5 for i in range(6)],
                               [(i/3, v) for i, v in enumerate(base.VARIANT_LABELS)],
                               'Experiment condition', 'Wall-clock seconds' if col == 0 else 'Process CPU seconds', '.1f')
            for i, variant in enumerate(base.VARIANTS):
                for j, cell in enumerate(cells):
                    item = lookup[cell, variant]
                    mean, se = item[f'{metric}_mean'], item[f'{metric}_se']
                    x = sx(i/3)+(j-0.5)*50
                    canvas.rect(x-21, sy(mean), 42, sy(0)-sy(mean), colors[j])
                    canvas.error(x, sy(mean-se), sy(mean+se), '#334155')
                    canvas.text(x, sy(mean+se)-10, f'{mean:.1f}', size=14, anchor='middle')
    canvas.text(30, 1020, 'Full-fit time includes the complete tuning path and final refitting, not just the selected-lambda optimization.', size=17)
    canvas.text(30, 1051, 'No independent per-lambda timers were saved; these are method totals, NOT a lambda-specific timing sweep.', size=16, bold=True)
    canvas.text(30, 1080, 'Top row excludes evaluation and data generation. Both methods simulate sites serially; real network latency is not measured.', size=16, color='#475569')
    base.save_svg(canvas, dest, FIGURES[2])


def render(summaries, dest):
    base.plot_setup()
    plot_overview(summaries, dest)
    plot_fedavg(summaries, dest)
    plot_lasso(summaries, dest)


def read_cached(dest):
    report = json.loads(base.safe_path(dest / PROVENANCE, must_exist=True).read_text())
    require(report['formal_evidence_sha256'] == base.EVIDENCE_SHA, 'wrong accepted source')
    require(set(report['table_sha256']) == set(TABLES), 'missing table hashes')
    for name, expected in report['table_sha256'].items():
        require(base.sha(base.safe_path(dest / name, must_exist=True).read_bytes()) == expected,
                'cached presentation table changed')
    with (dest / TABLES[0]).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['seed'] = int(row['seed'])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return summarize(rows), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=base.DEST)
    parser.add_argument('--render-only', action='store_true', help='reuse hash-bound presentation tables')
    args = parser.parse_args()
    dest = base.safe_path(args.output_dir)
    require(dest not in (base.DATA, base.SCRATCH, base.OUTPUTS) and base.OUTPUTS not in dest.parents,
            'report cannot be written into scientific outputs or artifact roots')
    dest.mkdir(parents=True, exist_ok=True)
    for name in TABLES + (PROVENANCE, README) + tuple(f'{n}.{ext}' for n in FIGURES for ext in ('svg', 'png')):
        base.safe_path(dest / name)
    if args.render_only:
        summaries, report = read_cached(dest)
    else:
        rows = collect()
        summaries = summarize(rows)
        base.write_csv(dest / TABLES[0], rows)
        base.write_csv(dest / TABLES[1], summaries)
        report = dict(kind='deterministic-presentation-report', scientific_mutations=False,
                      formal_evidence_sha256=base.EVIDENCE_SHA,
                      actual_formal_artifact_ledger='verified_57600_units',
                      timing_rows=len(rows), method_configs=len(base.CELLS),
                      seeds_per_condition=300, split='noniid', pilot_included=False,
                      group_lasso_included=False, execution_profile='cpu-1core',
                      fit_scope='all measured train/tuning/aggregation buckets; includes selection/refit',
                      evaluation_scope='all recorded evaluations; outcome methods additionally include logit diagnostics',
                      excluded='data generation from fit; queueing, real networking, artifact I/O; historical GPU timing',
                      lambda_specific_timing='not measured; not reconstructed from communication counts',
                      table_sha256={n: base.sha((dest/n).read_bytes()) for n in TABLES})
    render(summaries, dest)
    report.update(render_status='complete', plot_script_sha256=base.sha(Path(__file__).read_bytes()),
                  renderer_script_sha256=base.sha(Path(base.__file__).read_bytes()),
                  figure_sha256={f'{n}.{ext}': base.sha((dest/f'{n}.{ext}').read_bytes())
                                 for n in FIGURES for ext in ('svg', 'png')})
    (dest / PROVENANCE).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (dest / README).write_text('''# non-IID 正式实验耗时图

- `noniid_all_methods_time_cost.png/.svg`：22 个正式 CPU 方法配置的完整拟合和评估墙钟耗时。
- `noniid_multiround_fedavg_time_cost.png/.svg`：16 个 FedAvg 参数配置的完整拟合墙钟耗时。
- `noniid_logistic_lasso_time_cost.png/.svg`：多轮与单轮 logistic LASSO 的拟合/评估墙钟和进程 CPU 时间。

每条件 300 seeds，四个条件分别呈现，均值 ± 1 SE。不含 IID、pilot 或 Group LASSO。
完整拟合成本包含模型内预处理、选参的全部候选、最终重拟合、摘要/服务器聚合。
主计时取顶层互不重叠的 train/tuning/aggregation 桶，不重复加内部细分计时。
未单列的 tuning 已包含在 train 中，不能解释成“调参耗时为零”。
数据生成与测试评估不计入拟合时间，二者在 CSV 中独立记录。评估是实际执行的全部
评估工作：结局模型还计算 logit 辅助指标，R 模型不计算，因此不能把评估时间差
完全归因于模型推理速度。墙钟为 perf_counter，CPU 为当前进程 process_time。

这批任务均为 cpu-1core 单进程串行模拟；不是部署后的站点并行总延迟，不包含排队、
真实网络、未计时的模型导出或产物 I/O。旧 GPU FedFisher 没有进入此 CPU 比较的
可比完整计时，图中明确说明 N/A，绝不补成零。无逐 lambda 计时，不能画参数耗时曲线。

`completed_noniid_time_cost_seed_rows.csv` 保留 26,400 条逐种子观测；
`completed_noniid_time_cost_summary.csv` 保留 88 行汇总，包含均值、SE、中位数、最小/最大值。
`completed_noniid_time_cost_provenance.json` 绑定验收账本、表格、绘图脚本和图片哈希。
首次汇总重新验证固定验收来源的全部 57,600 单元；不改变旧科学完成证据。

重绘入口：`scripts/make_completed_noniid_time_cost.py`，可加 `--render-only` 使用已校验表格。
在仓库根目录以前台 Python `-B` 运行即可；无需调度器注册，不训练、不推理、不重跑实验。
图表写到本目录；可再生缓存沿用 `/scr/del6500/FedFisher/completed-noniid-plot-cache/`。
旧 Top2 图和其绘图脚本均保持不变。
''', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
