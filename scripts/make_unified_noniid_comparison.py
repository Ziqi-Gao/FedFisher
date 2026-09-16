#!/usr/bin/env python3
"""One method list for completed non-IID recovery and recorded time costs.

Reporting only: read saved scores and hash-bound tables; no model execution,
scientific result mutation, scheduler registration or job submission.
"""

import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path

import make_completed_noniid_visualizations as base
import make_completed_noniid_time_cost as timing
import make_one_shot_lasso_fedfisher_comparison as legacy


FISHER_CELLS = tuple(f'fedfisher_{kind}_{depth}' for depth in ('shallow', 'deep')
                     for kind in ('diag', 'kfac'))
# Interleave by model/method family, never by experiment age.
CELLS = base.BASE_CELLS[:4] + tuple(
    cell for depth in ('shallow', 'deep') for cell in (
        f'one_shot_fedavg_{depth}', f'fedfisher_diag_{depth}', f'fedfisher_kfac_{depth}',
        *(f'fedavg_{depth}_e{e}_{stop}' for stop in ('budget30', 'patience') for e in (1, 5, 10, 30))))
METRIC = 'permute/tau_prob'
FISHER_ROWS = 'unified_noniid_fedfisher_seed_rows.csv'
SUMMARY = 'unified_noniid_methods_summary.csv'
LEDGER = 'unified_noniid_fedfisher_input_ledger.json'
PROVENANCE = 'unified_noniid_plot_provenance.json'
README = 'UNIFIED_NONIID_README.md'
FIGURE = 'noniid_all_methods_top2_and_time'
require = base.require


def saved_score_recovery(features):
    """Presentation-level ranking of SAVED scores with the common exact-tie rule."""
    require(set(features) == set(range(1, 101)), 'missing or extra raw-X scores')
    values = [features[i] for i in range(1, 101)]
    require(all(math.isfinite(v) and v >= 0 for v in values), 'invalid saved importance')
    cutoff = sorted(values)[-2]
    above = sum(v > cutoff for v in values)
    tied = sum(v == cutoff for v in values)
    slots = 2 - above
    true_above = sum(v > cutoff for v in values[:2])
    true_tied = sum(v == cutoff for v in values[:2])
    probability = (math.comb(tied - true_tied, slots - true_tied) / math.comb(tied, slots)
                   if true_above + true_tied == 2 and true_tied <= slots else 0.0)
    auc = sum((s > n) + 0.5 * (s == n) for s in values[:2] for n in values[2:]) / 196
    return probability, auc, tied


def parse_scores(data, variant, run_id, scope):
    grouped = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(data.decode())):
        if (row['model_source'] not in ('fedfisher_diag', 'fedfisher_kfac') or
                row['intervention_mode'] != 'permute' or row['metric'] != 'tau_intervention_prob'):
            continue
        seed = int(row['seed'])
        require(row['split'] == 'noniid' and seed in range(300) and
                row['model'] in ('SyntheticMLP', 'SyntheticMLPDeep') and
                row['permutation_scope'] == scope, 'invalid historical score identity')
        depth = 'shallow' if row['model'] == 'SyntheticMLP' else 'deep'
        cell = f"{row['model_source']}_{depth}"
        feature = int(row['feature_idx'])
        require(int(row['is_signal']) == int(feature in (1, 2)), 'unexpected signal labels')
        key = cell, seed
        require(feature not in grouped[key], 'duplicate saved feature score')
        grouped[key][feature] = float(row['importance'])
    require(len(grouped) == 2 and len({seed for _, seed in grouped}) == 1 and
            len({cell.rsplit('_', 1)[1] for cell, _ in grouped}) == 1,
            'each score file must contain two Fisher methods for one model/seed')
    rows = []
    for (cell, seed), features in sorted(grouped.items()):
        probability, auc, tied = saved_score_recovery(features)
        rows.append(dict(cell=cell, variant=variant, seed=seed, split='noniid', run_id=run_id,
                         metric=METRIC, top2_probability=probability, auroc=auc,
                         boundary_tie_size=tied, timing_status='not_recorded'))
    return rows


def summarize_fisher(rows):
    grouped = defaultdict(list)
    for row in rows:
        require(row['split'] == 'noniid' and row['metric'] == METRIC and
                row['cell'] in FISHER_CELLS and row['variant'] in base.VARIANTS and
                row['run_id'] == legacy.OLD_RUNS[row['variant']], 'unexpected Fisher report row')
        require(0 <= row['top2_probability'] <= 1 and 0 <= row['auroc'] <= 1,
                'invalid Fisher recovery value')
        grouped[row['cell'], row['variant']].append(row)
    require(set(grouped) == {(c, v) for c in FISHER_CELLS for v in base.VARIANTS},
            'incomplete Fisher method/variant coverage')
    summaries = []
    for (cell, variant), samples in grouped.items():
        require(len(samples) == 300 and {r['seed'] for r in samples} == set(range(300)),
                'incomplete or duplicate Fisher seed coverage')
        mean, se = base.mean_se([100 * r['top2_probability'] for r in samples])
        auc, auc_se = base.mean_se([r['auroc'] for r in samples])
        summaries.append(dict(cell=cell, variant=variant, split='noniid', n=300, metric=METRIC,
                              top2_pct=mean, top2_se_pct=se, auroc=auc, auroc_se=auc_se))
    return summaries


def collect_fisher():
    rows, artifacts, runs = [], [], []
    common = None

    def read_bound(path):
        path = base.safe_path(path, (base.OUTPUTS,), True)
        data = path.read_bytes()
        artifacts.append(dict(path=str(path), sha256=base.sha(data)))
        return data

    for variant in base.VARIANTS:
        run_id = legacy.OLD_RUNS[variant]
        directory, manifest = legacy.validate_run(run_id, variant)
        read_bound(directory / 'run_manifest.env')
        require(manifest['git_commit'] == '90700c2afcaeb7ce8f3455f49d493b06655d73a7' and
                manifest['git_dirty'] == '0', 'unexpected archived Fisher source')
        binding = tuple(manifest[k] for k in ('provenance_sha256', 'run_config_sha256'))
        require(common is None or common == binding, 'Fisher variants have different common provenance')
        common = binding
        expected_scope = 'site' if 'local_permutation' in variant else 'global'
        require(manifest['prediction_intervention_scope'] == expected_scope and
                manifest['synthetic_local_epochs'] == '30' and manifest['num_rounds'] == '1',
                'unexpected Fisher recipe')
        markers = directory / 'completed_seeds'
        require({p.name for p in markers.iterdir()} == {f'seed_{s}.done' for s in range(300)},
                'incomplete Fisher marker coverage')
        for seed in range(300):
            data = read_bound(markers / f'seed_{seed}.done')
            marker = dict(line.split('=', 1) for line in data.decode().splitlines() if '=' in line)
            require(marker['seed'] == str(seed) and marker['format_version'] == '2', 'invalid seed marker')
            require(all(marker[k] == manifest[k] for k in
                        ('run_id', 'effect_modifier_variant', 'provenance_sha256',
                         'run_config_sha256', 'variant_config_sha256')), 'Fisher marker identity changed')
        paths = sorted(directory.glob('*splitnoniid*_prediction_intervention.csv'))
        require(len(paths) == 600, 'incomplete Fisher score-file coverage')
        for index, path in enumerate(paths, 1):
            rows.extend(parse_scores(read_bound(path), variant, run_id, expected_scope))
            if index % 200 == 0:
                print(f'{variant}: read {index}/600 saved non-IID score files', flush=True)
        # These fixed historical runs predate method timers. Do not substitute
        # mixed-algorithm process timestamps for an individual method's runtime.
        timing_files = sorted(str(p.relative_to(directory)) for p in directory.rglob('*timing*') if p.is_file())
        require(not timing_files, 'new timing evidence found: review its scope before reporting N/A')
        runs.append(dict(run_id=run_id, variant=variant, score_files=len(paths), seed_markers=300,
                         timing_files=timing_files, timing_status='not_recorded',
                         original_markers_bind_score_hashes=False))
    summarize_fisher(rows)
    return rows, dict(runs=runs, artifacts=artifacts,
                      validation='archived manifest/marker identities and complete saved-score coverage; '
                      'hashes frozen at reporting time, not retroactive scheduler-era certification')


def read_cpu(dest):
    provenance = json.loads(base.safe_path(dest / 'completed_noniid_plot_provenance.json', must_exist=True).read_text())
    require(provenance['formal_evidence_sha256'] == base.EVIDENCE_SHA, 'wrong accepted Top2 source')
    name = 'completed_noniid_top2_seed_rows.csv'
    data = base.safe_path(dest / name, must_exist=True).read_bytes()
    require(base.sha(data) == provenance['table_sha256'][name], 'Top2 presentation table changed')
    raw = list(csv.DictReader(io.StringIO(data.decode())))
    for row in raw:
        row['seed'] = int(row['seed'])
        row['top2_probability'] = float(row['top2_probability'])
        row['auroc'] = float(row['auroc'])
    times, time_provenance = timing.read_cached(dest)
    return base.summarize(raw), times, {
        name: base.sha(data), **time_provenance['table_sha256']}


def merge_rows(cpu, times, fisher):
    top2 = {(r['cell'], r['variant']): r for r in cpu + fisher}
    costs = {(r['cell'], r['variant']): r for r in times}
    require(len(top2) == len(cpu) + len(fisher) == 104 and
            set(top2) == {(c, v) for c in CELLS for v in base.VARIANTS}, 'invalid unified method coverage')
    require(len(times) == len(costs) == 88 and
            set(costs) == {(c, v) for c in base.CELLS for v in base.VARIANTS}, 'invalid timing coverage')
    rows = []
    for cell in CELLS:
        for variant in base.VARIANTS:
            score = top2[cell, variant]
            require(score['metric'] == METRIC and score['split'] == 'noniid' and score['n'] == 300,
                    'unified plot requires a common probability-scale endpoint and full formal seeds')
            gpu = cell in FISHER_CELLS
            row = {k: score[k] for k in ('cell', 'variant', 'split', 'n', 'metric', 'top2_pct',
                                        'top2_se_pct', 'auroc', 'auroc_se')}
            row.update(hardware='GPU (archived recipe)' if gpu else 'CPU 1 core',
                       timing_status='not_recorded' if gpu else 'measured',
                       run_id=legacy.OLD_RUNS[variant] if gpu else f'comparison_{cell}_{variant}_v1_300seeds')
            for metric in timing.METRICS:
                for statistic in ('mean', 'se'):
                    key = f'{metric}_{statistic}'
                    row[key] = None if gpu else costs[cell, variant][key]
            rows.append(row)
    return rows


def plot_unified(rows, dest):
    canvas = base.SVG(1940, 1390)
    canvas.text(30, 45, 'Non-IID | All methods: Top-2 recovery and time cost', size=29, bold=True)
    canvas.text(30, 78, '26 method settings | 300 formal seeds per condition | No pilot or Group LASSO', size=18)
    base.variant_legend(canvas, 112)
    lookup = {(r['cell'], r['variant']): r for r in rows}
    top, step, bottom = 214, 39, 214 + 26 * 39
    canvas.text(30, 161, 'Method / model / setting', size=18, bold=True)
    canvas.text(470, 161, 'A  Probability-scale Top-2 (%)', size=20, bold=True)
    for j, label in enumerate(base.VARIANT_LABELS):
        canvas.text(524 + 108*j, 194, label, size=14, anchor='middle')
    for i, cell in enumerate(CELLS):
        y = top + i * step
        if i % 2 == 0:
            canvas.rect(20, y, 1890, step, '#e2e8f0', .35)
        label = base.label(cell) + (' [GPU]*' if cell in FISHER_CELLS else '')
        canvas.text(450, y + 25, label, size=15, anchor='end', bold=cell in FISHER_CELLS)
        for j, variant in enumerate(base.VARIANTS):
            value = lookup[cell, variant]['top2_pct']
            canvas.rect(471 + 108*j, y + 3, 106, step - 6, base.heat_color(value))
            canvas.text(524 + 108*j, y + 25, f'{value:.1f}', size=17, anchor='middle',
                        color='#ffffff' if value >= 60 else '#172033', bold=True)
    for panel, (metric, title) in enumerate((('fit_wall_seconds', 'B  Full fit: selection + refit'),
                                            ('evaluation_wall_seconds', 'C  Recorded evaluation'))):
        left, width = 960 + 500*panel, 435
        observed = [r for r in rows if r['timing_status'] == 'measured']
        intervals = [r[f'{metric}_mean'] + sign * r[f'{metric}_se'] for r in observed for sign in (-1, 1)]
        require(min(intervals) > 0, 'nonpositive interval on log-time axis')
        lo, hi = math.log10(min(intervals)) - .18, math.log10(max(intervals)) + .18
        sx = lambda v: left + (math.log10(v) - lo) / (hi - lo) * width
        canvas.text(left, 161, title, size=20, bold=True)
        canvas.text(left, 194, 'Mean +/- 1 SE | wall-clock seconds', size=15)
        ticks = [m * 10**p for p in range(math.floor(lo), math.ceil(hi) + 1) for m in (1, 2, 5)
                 if lo <= math.log10(m * 10**p) <= hi]
        for tick in ticks:
            x = sx(tick)
            canvas.line(x, top, x, bottom, '#cbd5e1')
            canvas.text(x, bottom + 24, f'{tick:g}', size=14, anchor='middle')
        canvas.text(left + width/2, bottom + 52, 'Seconds per seed (log scale)', size=16, anchor='middle')
        for i, cell in enumerate(CELLS):
            y = top + i * step + step/2
            if cell in FISHER_CELLS:
                canvas.rect(left + 62, y - 12, width - 124, 25, '#ffffff')
                canvas.text(left + width/2, y + 5, 'N/A: not recorded', size=15, anchor='middle', color='#64748b')
                continue
            for j, variant in enumerate(base.VARIANTS):
                item = lookup[cell, variant]
                mean, se = item[f'{metric}_mean'], item[f'{metric}_se']
                yy = y + (j - 1.5) * 7
                canvas.line(sx(mean-se), yy, sx(mean+se), yy, base.COLORS[j], 1.5)
                canvas.marker(sx(mean), yy, base.COLORS[j], j, 3.2)
    canvas.text(30, bottom + 25, 'Top-2 = both true modifiers selected; uniform cutoff ties.', size=15)
    canvas.text(30, bottom + 51, 'Top-2 SE and all unrounded values are retained in the CSV.', size=15)
    canvas.text(30, 1322, '* FedFisher uses saved GPU results; other rows use CPU comparison recipes. Common metric, but different fit/permutation implementations.', size=16)
    canvas.text(30, 1350, 'FedFisher method-level times were not recorded: N/A is not zero. Full fit excludes data generation/evaluation; no real-network latency is measured.', size=16)
    canvas.text(30, 1378, 'Evaluation workloads differ: outcome models also calculate logit diagnostics; R methods do not. No CPU-vs-GPU speed claim is supported.', size=16)
    base.save_svg(canvas, dest, FIGURE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=base.DEST)
    parser.add_argument('--render-only', action='store_true', help='reuse hash-bound saved Fisher report rows')
    args = parser.parse_args()
    dest = base.safe_path(args.output_dir, must_exist=True)
    require(dest not in (base.DATA, base.SCRATCH, base.OUTPUTS) and base.OUTPUTS not in dest.parents,
            'report must not write into scientific outputs or artifact roots')
    for name in (FISHER_ROWS, SUMMARY, LEDGER, PROVENANCE, README, FIGURE+'.svg', FIGURE+'.png'):
        base.safe_path(dest / name)
    cpu, times, input_hashes = read_cpu(dest)
    if args.render_only:
        report = json.loads(base.safe_path(dest / PROVENANCE, must_exist=True).read_text())
        require(report['formal_evidence_sha256'] == base.EVIDENCE_SHA, 'wrong accepted source')
        for name in (FISHER_ROWS, LEDGER):
            require(base.sha((dest / name).read_bytes()) == report['table_sha256'][name], 'Fisher report input changed')
        with (dest / FISHER_ROWS).open(newline='') as handle:
            fisher = list(csv.DictReader(handle))
        for row in fisher:
            row['seed'] = int(row['seed'])
            row['boundary_tie_size'] = int(row['boundary_tie_size'])
            for key in ('top2_probability', 'auroc'):
                row[key] = float(row[key])
    else:
        fisher, ledger = collect_fisher()
        base.write_csv(dest / FISHER_ROWS, fisher)
        (dest / LEDGER).write_text(json.dumps(ledger, indent=2) + '\n')
    rows = merge_rows(cpu, times, summarize_fisher(fisher))
    base.write_csv(dest / SUMMARY, rows)
    base.plot_setup()
    plot_unified(rows, dest)
    report = dict(kind='deterministic-presentation-report', scientific_mutations=False,
                  formal_evidence_sha256=base.EVIDENCE_SHA, cpu_input_table_sha256=input_hashes,
                  metric=METRIC, split='noniid', method_settings=len(CELLS), seeds_per_condition=300,
                  pilot_included=False, group_lasso_included=False,
                  fisher_score_rows=len(fisher), fisher_boundary_tie_rows=sum(r['boundary_tie_size'] > 1 for r in fisher),
                  missing_fisher_timing='not recorded; null/empty, never zero or inferred from batch duration',
                  comparison_limit='same scientific metric; different source/fit/permutation implementations and hardware, not a controlled timing benchmark',
                  historical_validation='manifest/marker identity and coverage; original markers lack score hashes; report-time hashes in input ledger',
                  table_sha256={n: base.sha((dest/n).read_bytes()) for n in (FISHER_ROWS, SUMMARY, LEDGER)},
                  script_sha256={p.name: base.sha(p.read_bytes()) for p in
                                 (Path(__file__), Path(base.__file__), Path(timing.__file__), Path(legacy.__file__))},
                  figure_sha256={FIGURE+'.'+ext: base.sha((dest/(FIGURE+'.'+ext)).read_bytes()) for ext in ('svg', 'png')})
    (dest / PROVENANCE).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (dest / README).write_text('''# non-IID 统一方法总览

`noniid_all_methods_top2_and_time.png/.svg` 按方法/网络深度排列，旧、新实验不再分区。
22 个 CPU 比较配置与 FedFisher diag/KFAC × shallow/deep 四个配置，共 26 个设置。
每设置/条件 300 seeds，仅 non-IID 正式实验；不含 pilot、Group LASSO。
原有单独图保留；本图是包含 FedFisher 的统一总览。

左侧统一为概率尺度 raw-X permutation treatment-contrast Top2，不是把旧 logit
指标直接拼到新概率指标。FedFisher 读取四组正式实验已经保存的概率尺度 importance，
仅做排名汇总，按新比较的规则对精确分数并列作均匀随机选取的期望处理。
未重新拟合或推理。汇总 CSV 保留均值、逐 seed 样本 SE 和运行来源；未合并不同运行的种子。
CPU 方法使用已验收、哈希绑定的报告表。旧 FedFisher 的 manifest 和 300 个完成标记
身份逐一核对，600 个 non-IID 分数文件/条件完整；旧标记本来不绑定分数文件哈希，
输入账本的哈希是在此次报告时记录，不能追溯证明旧产物从完成以来未变，也不重新认证科学完成状态。

右侧为拟合与实际评估耗时。CPU 完整拟合包含全部选参/重拟合/聚合，不包含数据生成
和独立评估。旧 FedFisher 没有逐方法时间记录，图中保留四行但标 N/A；CSV 时间为空，
不补零，也不用包含多个算法的进程时间推测单个算法。GPU 与 CPU 不是受控速度比较。
原始和新实验的拟合实现、置换随机数实现并不完全相同；这是一张统一指标的结果总览，
不是所有实现细节严格配对的实验。结局方法评估还含 logit 辅助诊断，R 方法不含，
因此评估耗时也不能直接解释成推理速度。均为模拟，不代表真实网络/多站点并行延迟。

绘图脚本：`scripts/make_unified_noniid_comparison.py`；加 `--render-only` 可使用哈希绑定
的展示表快速重绘，无需注册调度器。数据/图片在本目录，缓存位于
`/scr/del6500/FedFisher/completed-noniid-plot-cache/`；科学结果和完成证据完全不改动。
''', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
