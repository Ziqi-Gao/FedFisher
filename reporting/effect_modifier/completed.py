"""Foreground reporting only: accepted artifacts -> non-IID tables and figures.

No training, inference, data generation, task submission, or scientific writes.
The historic comparison source is frozen: verify its accepted artifact ledger
instead of importing a current-source collector or bypassing its provenance.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from collections import defaultdict


DATA = Path('/data/del6500/FedFisher')
SCRATCH = Path('/scr/del6500/FedFisher')
OUTPUTS = DATA / 'effect_modifier_experiment/outputs'
DEST = DATA / 'results/one_shot_lasso_vs_fedfisher'
AUDIT = DATA / 'effect_modifier_experiment/recovery_audit/fedfisher-effects-p1-acceptance-20260914'
EVIDENCE_SHA = 'f6bd3d74c2e740a47c6015ff46724972ebef328281a3d207da8dac80648c7989'
VARIANTS = ('control', 'local_permutation', 'site_gaussian_shift', 'site_gaussian_local_permutation')
VARIANT_LABELS = ('Control', 'Local perm.', 'Gaussian shift', 'Combined')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')
MARKERS = ('o', 's', '^', 'D')
BASE_CELLS = ('dslassocov', 'local_logistic_lasso', 'r_unpenalized',
              'r_adaptive_lasso', 'one_shot_fedavg_shallow', 'one_shot_fedavg_deep')
CELLS = BASE_CELLS + tuple(
    f'fedavg_{depth}_e{epochs}_{stop}'
    for depth in ('shallow', 'deep') for stop in ('budget30', 'patience')
    for epochs in (1, 5, 10, 30))
OLD_CELLS = ('dslassocov', 'logistic_lasso', 'fedavg_shallow', 'fedfisher_diag_shallow',
             'fedfisher_kfac_shallow', 'fedavg_deep', 'fedfisher_diag_deep', 'fedfisher_kfac_deep')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def read_env(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines()
                if '=' in line and not line.startswith('#'))


def safe_path(path, roots=(DATA, SCRATCH), must_exist=False):
    path = Path(path)
    resolved = path.resolve(strict=must_exist)
    require(any(resolved == r.resolve() or r.resolve() in resolved.parents for r in roots),
            f'path is outside allowed artifact roots: {path}')
    require(not path.is_symlink(), f'symlink is not an artifact target: {path}')
    return resolved


def mean_se(values):
    require(len(values) >= 2 and all(math.isfinite(x) for x in values), 'need finite repeated results')
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def summarize(seed_rows):
    grouped = defaultdict(list)
    for row in seed_rows:
        require(row['split'] == 'noniid', 'IID data cannot enter these figures')
        grouped[row['cell'], row['variant']].append(row)
    summaries = []
    for (cell, variant), rows in grouped.items():
        require(len(rows) == 300 and {r['seed'] for r in rows} == set(range(300)),
                f'incomplete or duplicate seed coverage: {cell}/{variant}')
        top2, error = mean_se([r['top2_probability'] * 100 for r in rows])
        auroc, auroc_error = mean_se([r['auroc'] for r in rows])
        summaries.append(dict(cohort='comparison_v1', metric='permute/tau_prob', cell=cell,
                              variant=variant, split='noniid', n=300, top2_pct=top2,
                              top2_se_pct=error, auroc=auroc, auroc_se=auroc_error))
    require({(r['cell'], r['variant']) for r in summaries} ==
            {(c, v) for c in CELLS for v in VARIANTS}, 'formal plotting matrix is incomplete')
    return summaries


def r_diagnostic(result):
    fit = result['fit']
    pure = {int(site) for site, counts in result['label_counts'].items()
            if sum(count > 0 for count in counts.values()) == 1}
    pure_var, mixed_var = [], []
    for local in fit['local_reports']:
        (pure_var if local['site_idx'] in pure else mixed_var).append(local['summary']['residual_variance'])
    coefficients = fit['shared_effect_coefficients']
    selected = fit['selected_fit']
    rec = result['evaluation']['metrics']['permute/tau_prob']['recovery']
    return dict(cell=result['identity']['cell'], variant=result['identity']['variant'],
                seed=result['identity']['seed'], split='noniid',
                top2_probability=rec['probability_all_signals_selected'], auroc=rec['auroc'],
                pure_outcome_sites=len(pure), zero_slopes=all(x == 0 for x in coefficients[1:]),
                active_slopes=sum(x != 0 for x in coefficients[1:]),
                beta_x1=coefficients[1], beta_x2=coefficients[2],
                max_abs_noise_beta=max(abs(x) for x in coefficients[3:]),
                pure_variance_median=statistics.median(pure_var) if pure_var else None,
                mixed_variance_median=statistics.median(mixed_var) if mixed_var else None,
                min_site_variance=min(pure_var + mixed_var),
                selected_lambda=fit['selected_lambda'], selected_index=fit['selected_index'],
                selected_ebic=selected.get('ebic'), zero_lambda_ebic=fit['lambda_fits'][-1].get('ebic'),
                max_kkt_ratio=max(p['kkt_residual_inf'] / p['stationarity_threshold'] for p in fit['lambda_fits']))


def collect_formal():
    evidence_bytes = (AUDIT / 'old-cohort.scientific-evidence.json').read_bytes()
    require(sha(evidence_bytes) == EVIDENCE_SHA, 'accepted evidence hash changed')
    evidence = json.loads(evidence_bytes)
    require(evidence['validated_scientific_units'] == 57600, 'unexpected accepted cohort size')
    all_digest = hashlib.sha256()
    rows, lasso, diagnostics, pairs = [], [], [], {}
    for number, saved in enumerate(evidence['artifacts']['manifests'], 1):
        manifest = safe_path(saved['manifest_path'], (OUTPUTS,), True)
        require(sha(manifest.read_bytes()) == saved['manifest_sha256'], f'manifest changed: {manifest}')
        directory = manifest.parent
        expected = {'run_manifest.env'} | {f'seed_{s:03d}_{split}' for s in range(300) for split in ('iid', 'noniid')}
        require({p.name for p in directory.iterdir()} == expected, f'incomplete run: {directory}')
        digest = hashlib.sha256()
        for seed in range(300):
            for split in ('iid', 'noniid'):
                unit = safe_path(directory / f'seed_{seed:03d}_{split}', (OUTPUTS,), True)
                require({p.name for p in unit.iterdir()} == {'done.json', 'started.json', 'result.json', 'model.npz'},
                        f'partial or unexpected unit: {unit}')
                done_bytes = safe_path(unit / 'done.json', (OUTPUTS,), True).read_bytes()
                done = json.loads(done_bytes)
                require(set(done['artifacts']) == {'started.json', 'result.json', 'model.npz'}, 'bad artifact set')
                raw, hashes = {}, {}
                for name in sorted(done['artifacts']):
                    content = safe_path(unit / name, (OUTPUTS,), True).read_bytes()
                    hashes[name] = sha(content)
                    if name != 'model.npz':
                        raw[name] = content
                require(hashes == done['artifacts'], f'artifact bytes changed: {unit}')
                started = json.loads(raw['started.json'])
                identity = done['identity']
                require(identity == started['identity'] and identity['seed'] == seed and identity['split'] == split
                        and identity['run_id'] == saved['run_id'], f'identity mismatch: {unit}')
                line = encoded(dict(relative_path=unit.name, done_sha256=sha(done_bytes), artifacts=hashes,
                                    identity=identity, producer_job_id=started['execution']['job_id']))
                digest.update(line)
                all_digest.update(line)
                cell, variant = identity['cell'], identity['variant']
                if split != 'noniid' or cell not in CELLS:
                    continue
                result = json.loads(raw['result.json'])
                require(result['identity'] == identity, f'result identity mismatch: {unit}')
                rec = result['evaluation']['metrics']['permute/tau_prob']['recovery']
                rows.append(dict(cell=cell, variant=variant, seed=seed, split=split,
                                 top2_probability=rec['probability_all_signals_selected'], auroc=rec['auroc']))
                if cell == 'dslassocov' and variant in ('control', 'site_gaussian_shift'):
                    cv = result['fit']['fit_metadata']['cv']
                    require(cv['uses_test_data'] is False and cv['lambda_count'] == 30, 'unexpected LASSO CV protocol')
                    lambdas = cv['lambda_path']
                    losses = cv['aggregated_validation_losses']
                    require(cv['selected_index'] == next(i for i, v in enumerate(losses) if v <= min(losses) + 1e-12),
                            'LASSO selection differs from its saved CV rule')
                    lasso.append(dict(variant=variant, seed=seed, lambda_path=lambdas,
                                      lambda_ratio=[x / lambdas[0] for x in lambdas], losses=losses,
                                      selected_index=cv['selected_index'], selected_lambda=cv['selected_lambda'],
                                      cv_gradient_rounds=[sum(f['lambda_fits'][i]['gradient_rounds']
                                                             for f in cv['fold_reports']) for i in range(30)]))
                if cell in ('r_unpenalized', 'r_adaptive_lasso') and variant in ('control', 'site_gaussian_shift'):
                    diagnostics.append(r_diagnostic(result))
                    fit = result['fit']
                    key = variant, seed
                    common = (fit['unpenalized_shared_coefficients'],
                              [r['summary']['payload_sha256'] for r in fit['local_reports']])
                    require(key not in pairs or pairs[key] == common, 'R methods changed pre-penalty fits')
                    pairs[key] = common
        require(digest.hexdigest() == saved['unit_hash_ledger_sha256'], f'accepted ledger changed: {directory}')
        if number % 4 == 0:
            print(f'Validated {number}/96 accepted manifests', flush=True)
    require(all_digest.hexdigest() == evidence['artifacts']['all_unit_hash_ledger_sha256'], 'accepted global ledger changed')
    require(len(rows) == 26400 and len(lasso) == 600 and len(diagnostics) == 1200, 'unexpected report coverage')
    return rows, lasso, diagnostics


def collect_historical():
    summaries = []
    source = DEST / 'comparison_summary.csv'
    with source.open(newline='') as handle:
        for r in csv.DictReader(handle):
            if r['split'] != 'noniid' or r['method'] == 'mlp_group_lasso':
                continue
            require(int(r['n']) == 300, 'historical row is not a complete formal run')
            suffix = '' if r['model'] == 'linear' else ('_shallow' if r['model'] == 'SyntheticMLP' else '_deep')
            summaries.append(dict(cohort='historical_logit', metric='permute/tau_intervention_logit',
                                  cell=r['method'] + suffix, variant=r['variant'], split='noniid', n=300,
                                  top2_pct=float(r['exact_top2_pct']), top2_se_pct=float(r['exact_top2_se_pct']),
                                  auroc=float(r['auroc_mean']), auroc_se=float(r['auroc_se'])))
    # dsLasso was not included in the old combined CSV. Read its completed,
    # marker-hashed result summaries, rather than substituting the new metric.
    for variant in VARIANTS:
        directory = OUTPUTS / f'dslassocov_{variant}_server_scheduler_v2r2_300seeds'
        rates, aucs = [], []
        for seed in range(300):
            marker = read_env(directory / 'completed_seeds' / f'seed_{seed}.done')
            result = directory / f'dslassocov_seed_{seed:06d}_noniid_result.env'
            require(sha(result.read_bytes()) == marker['noniid_result_env_sha256'], 'historical dsLasso result changed')
            values = read_env(result)
            require(values['primary_metric'] == 'permute/tau_intervention_logit' and
                    values['run_id'] == directory.name and values['seed'] == str(seed), 'old dsLasso identity mismatch')
            rates.append(100.0 * (int(values['primary_top2_hits']) == 2))
            aucs.append(float(values['primary_auroc']))
        rate, error = mean_se(rates)
        auc, auc_error = mean_se(aucs)
        summaries.append(dict(cohort='historical_logit', metric='permute/tau_intervention_logit',
                              cell='dslassocov', variant=variant, split='noniid', n=300,
                              top2_pct=rate, top2_se_pct=error, auroc=auc, auroc_se=auc_error))
    require(len(summaries) == 32 and {(r['cell'], r['variant']) for r in summaries} ==
            {(c, v) for c in OLD_CELLS for v in VARIANTS}, 'historical plotting matrix is incomplete')
    return summaries, sha(source.read_bytes())


def label(cell):
    names = {'dslassocov': 'dsLassoCov-style', 'local_logistic_lasso': 'One-shot Logistic LASSO',
             'logistic_lasso': 'One-shot Logistic LASSO', 'r_unpenalized': 'R-learner: unpenalized',
             'r_adaptive_lasso': 'R-learner: adaptive LASSO'}
    if cell in names:
        return names[cell]
    if cell.startswith('one_shot_fedavg_'):
        return 'One-shot FedAvg / ' + cell.rsplit('_', 1)[1]
    if cell.startswith('fedavg_') and '_e' in cell:
        _, depth, epoch, stop = cell.split('_')
        return f'FedAvg {depth} / E={epoch[1:]} / ' + ('budget 30' if stop == 'budget30' else 'patience')
    for prefix, name in (('fedfisher_diag_', 'FedFisher diag'), ('fedfisher_kfac_', 'FedFisher KFAC'),
                         ('fedavg_', 'One-shot FedAvg')):
        if cell.startswith(prefix):
            return name + ' / ' + cell[len(prefix):]
    raise ValueError(f'unlabeled method: {cell}')


class SVG:
    """Small SVG drawing surface, matching the existing dependency-free plots."""

    def __init__(self, width, height):
        self.width, self.height, self.parts = width, height, []
        self.rect(0, 0, width, height, '#ffffff')

    def rect(self, x, y, width, height, fill, opacity=1):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" opacity="{opacity}"/>')

    def text(self, x, y, value, size=15, anchor='start', color='#172033', bold=False, rotate=None):
        from html import escape
        transform = '' if rotate is None else f' transform="rotate({rotate} {x} {y})"'
        weight = '700' if bold else '400'
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}"{transform}>{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color='#cbd5e1', width=1):
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>')

    def polyline(self, points, color, width=2):
        text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        self.parts.append(f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')

    def band(self, points, color):
        text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        self.parts.append(f'<polygon points="{text}" fill="{color}" opacity="0.15"/>')

    def marker(self, x, y, color, kind=0, radius=4):
        if kind == 0:
            self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" stroke="white" stroke-width="0.7"/>')
        elif kind == 1:
            self.rect(x-radius, y-radius, 2*radius, 2*radius, color)
        else:
            points = ([(x, y-radius-1), (x-radius-1, y+radius), (x+radius+1, y+radius)]
                      if kind == 2 else [(x, y-radius-1), (x-radius-1, y), (x, y+radius+1), (x+radius+1, y)])
            text = ' '.join(f'{a:.2f},{b:.2f}' for a, b in points)
            self.parts.append(f'<polygon points="{text}" fill="{color}"/>')

    def error(self, x, low, high, color):
        self.line(x, low, x, high, color, 1.3)
        self.line(x-4, low, x+4, low, color, 1.3)
        self.line(x-4, high, x+4, high, color, 1.3)

    def document(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
                f'viewBox="0 0 {self.width} {self.height}"><g font-family="DejaVu Sans,Arial,sans-serif">'
                + ''.join(self.parts) + '</g></svg>\n')


def plot_setup():
    import shutil
    require(shutil.which('rsvg-convert') is not None, 'rsvg-convert is required for PNG export')
    cache = safe_path(SCRATCH / 'completed-noniid-plot-cache')
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(cache)
    os.environ['TMPDIR'] = str(safe_path(SCRATCH / 'tmp', must_exist=True))
    return None, None


def save_svg(canvas, dest, name):
    import subprocess
    svg = safe_path(dest / f'{name}.svg')
    png = safe_path(dest / f'{name}.png')
    svg.write_text(canvas.document(), encoding='utf-8')
    subprocess.run(['rsvg-convert', '-w', str(round(canvas.width * 1.5)),
                    '-o', str(png), str(svg)], check=True)


def heat_color(value):
    anchors = [(255, 247, 188), (127, 205, 187), (29, 145, 192), (34, 94, 168), (8, 29, 88)]
    position = max(0, min(100, value)) / 25
    i = min(3, int(position))
    t = position - i
    return '#' + ''.join(f'{round(anchors[i][k]*(1-t) + anchors[i+1][k]*t):02x}' for k in range(3))


def plot_all(formal, historical, dest, plt=None, np=None):
    canvas = SVG(1460, 1450)
    canvas.text(30, 45, 'Non-IID | Top-2 recovery across all completed methods', size=28, bold=True)
    canvas.text(30, 77, '300 seeds per condition | Formal experiments only | Group LASSO excluded', size=17, color='#475569')
    left, width, row_height = 500, 195, 34
    for rows, cells, top, title in (
        (formal, CELLS, 175, 'A  New formal comparison | probability-scale treatment contrast'),
        (historical, OLD_CELLS, 1050, 'B  Historical formal comparison | logit-scale treatment contrast'),
    ):
        lookup = {(r['cell'], r['variant']): r for r in rows}
        canvas.text(30, top-57, title, size=20, bold=True)
        for j, name in enumerate(VARIANT_LABELS):
            canvas.text(left+width*(j+0.5), top-15, name, size=16, anchor='middle', bold=True)
        for i, cell in enumerate(cells):
            y = top + i*row_height
            canvas.text(left-15, y+23, label(cell), size=16, anchor='end')
            for j, variant in enumerate(VARIANTS):
                value = lookup[cell, variant]['top2_pct']
                canvas.rect(left+j*width, y, width-2, row_height-2, heat_color(value))
                canvas.text(left+width*(j+0.5), y+23, f'{value:.1f}%', size=17, anchor='middle',
                            color='white' if value >= 60 else '#172033')
    # Shared color legend, with an explicit warning against pooling endpoint scales.
    for i in range(100):
        canvas.rect(1320, 285+4*i, 22, 4.1, heat_color(100-i))
    for tick in (0, 25, 50, 75, 100):
        canvas.text(1350, 290+4*(100-tick), f'{tick}%', size=14)
    canvas.text(1332, 251, 'Top-2', size=15, anchor='middle', bold=True)
    canvas.text(1332, 274, 'recovery', size=14, anchor='middle')
    canvas.text(30, 1350, 'Panels A and B use DIFFERENT effect scales; do not combine them into one ranking.', size=17, bold=True)
    canvas.text(30, 1381, 'Cells show mean recovery. Seed-level values and standard errors are supplied in CSV.', size=16, color='#475569')
    canvas.text(30, 1411, 'Top-2 means both true modifiers selected; new sparse-model results account for ties. This is not prediction accuracy.', size=15, color='#475569')
    save_svg(canvas, dest, 'noniid_all_methods_top2')


def axes(canvas, left, top, width, height, title, ylim, yticks, xticks, xlabel, ylabel, yformat='.0f'):
    low, high = ylim
    sy = lambda value: top+height-(value-low)/(high-low)*height
    sx = lambda value: left+45+value*(width-90)
    canvas.text(left, top-24, title, size=18, bold=True)
    for value in yticks:
        y = sy(value)
        canvas.line(left, y, left+width, y, '#e2e8f0')
        canvas.text(left-12, y+5, format(value, yformat), size=14, anchor='end', color='#475569')
    canvas.line(left, top, left, top+height, '#94a3b8')
    canvas.line(left, top+height, left+width, top+height, '#94a3b8')
    for position, value in xticks:
        x = sx(position)
        canvas.line(x, top+height, x, top+height+5, '#94a3b8')
        canvas.text(x, top+height+26, value, size=14, anchor='middle')
    canvas.text(left+width/2, top+height+57, xlabel, size=15, anchor='middle')
    canvas.text(left-72, top+height/2, ylabel, size=15, anchor='middle', rotate=-90)
    return sx, sy


def variant_legend(canvas, y):
    for j, (name, color) in enumerate(zip(VARIANT_LABELS, COLORS)):
        x = 375 + 235*j
        canvas.line(x, y-5, x+28, y-5, color, 2)
        canvas.marker(x+14, y-5, color, j)
        canvas.text(x+38, y, name, size=16)


def plot_fedavg(formal, dest, plt=None, np=None):
    canvas = SVG(1560, 1110)
    canvas.text(30, 46, 'Non-IID | FedAvg local-epoch and stopping-rule comparison', size=28, bold=True)
    canvas.text(30, 78, 'Probability-scale treatment contrast | Mean +/- 1 SE over 300 formal seeds', size=17, color='#475569')
    variant_legend(canvas, 115)
    lookup = {(r['cell'], r['variant']): r for r in formal}
    for row, depth in enumerate(('shallow', 'deep')):
        for col, stop in enumerate(('budget30', 'patience')):
            title = f'{depth.capitalize()} MLP | ' + ('30 total local epochs' if stop == 'budget30' else 'Validation patience + refit')
            sx, sy = axes(canvas, 105+765*col, 190+425*row, 635, 310, title, (0, 105),
                          range(0, 101, 20), [(i/3, str(e)) for i, e in enumerate((1, 5, 10, 30))],
                          'Local epochs per round (E)', 'Top-2 recovery (%)')
            for j, variant in enumerate(VARIANTS):
                points = []
                for i, epoch in enumerate((1, 5, 10, 30)):
                    r = lookup[f'fedavg_{depth}_e{epoch}_{stop}', variant]
                    x, y = sx(i/3)+(j-1.5)*4, sy(r['top2_pct'])
                    points.append((x, y))
                    canvas.error(x, sy(r['top2_pct']-r['top2_se_pct']), sy(r['top2_pct']+r['top2_se_pct']), COLORS[j])
                canvas.polyline(points, COLORS[j], 2)
                for x, y in points:
                    canvas.marker(x, y, COLORS[j], j, 4.5)
    canvas.text(30, 1020, 'Fixed budget: E = 1 / 5 / 10 / 30 means 30 / 6 / 3 / 1 training aggregation rounds.', size=17)
    canvas.text(30, 1051, 'E also changes optimizer resets and minibatch streams; this is not a communication-only intervention.', size=16, color='#475569')
    canvas.text(30, 1080, 'All four conditions and all 16 prespecified settings are shown; no test-based selection of a best setting.', size=16, color='#475569')
    save_svg(canvas, dest, 'noniid_multiround_fedavg_parameters')


def plot_lasso(lasso, formal, dest, plt=None, np=None):
    canvas = SVG(1560, 1175)
    canvas.text(30, 46, 'Non-IID | Multiround dsLassoCov-style: saved lambda-tuning diagnostics', size=26, bold=True)
    canvas.text(30, 80, '300 formal seeds per training distribution | No training or model inference in this report', size=17, color='#475569')
    training = ('control', 'site_gaussian_shift')
    colors = (COLORS[0], COLORS[2])
    for j, text in enumerate(('Control / Local perm. (shared fit)', 'Gaussian / Combined (shared fit)')):
        x = 330+535*j
        canvas.line(x, 111, x+30, 111, colors[j], 3)
        canvas.text(x+40, 117, text, size=16)
    stats = []
    for variant in training:
        rs = [r for r in lasso if r['variant'] == variant]
        require(len(rs) == 300, 'incomplete LASSO CV seeds')
        ratios = rs[0]['lambda_ratio']
        require(all(all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(r['lambda_ratio'], ratios)) for r in rs),
                'different normalized lambda grids')
        stats.append(dict(x=[(math.log10(v)+3)/3 for v in ratios],
                          losses=[mean_se([r['losses'][i] for r in rs]) for i in range(30)],
                          rounds=[mean_se([r['cv_gradient_rounds'][i] for r in rs]) for i in range(30)],
                          frequencies=[sum(r['selected_index'] == i for r in rs)/3 for i in range(30)]))
    xticks = [(0, '0.001'), (1/3, '0.01'), (2/3, '0.1'), (1, '1')]
    all_loss = [v for stat in stats for mean, se in stat['losses'] for v in (mean-se, mean+se)]
    loss_low = math.floor(min(all_loss)*20)/20
    loss_high = math.ceil(max(all_loss)*20)/20
    if loss_high == loss_low:
        loss_high += 0.05
    max_round = max(mean+se for stat in stats for mean, se in stat['rounds'])
    round_high = math.ceil(max_round/100)*100
    max_freq = max(max(stat['frequencies']) for stat in stats)
    freq_high = max(10, math.ceil(max_freq/10)*10)
    a = axes(canvas, 110, 195, 625, 300, 'A  Saved cross-validation loss',
             (loss_low, loss_high), [loss_low+(loss_high-loss_low)*i/4 for i in range(5)], xticks,
             'Relative penalty: lambda / lambda_max (log scale)', 'Validation logistic loss', '.3f')
    b = axes(canvas, 880, 195, 625, 300, 'B  Which penalty was selected?',
             (0, freq_high), [freq_high*i/4 for i in range(5)], xticks,
             'Relative penalty: lambda / lambda_max (log scale)', 'Seeds selecting grid point (%)', '.1f')
    c = axes(canvas, 110, 670, 625, 300, 'C  Gradient aggregation during CV',
             (0, round_high), [round_high*i/4 for i in range(5)], xticks,
             'Relative penalty: lambda / lambda_max (log scale)', 'Gradient rounds / lambda (5-fold sum)')
    for stat, color in zip(stats, colors):
        for key, (sx, sy) in (('losses', a), ('rounds', c)):
            points = [(sx(x), sy(mean)) for x, (mean, se) in zip(stat['x'], stat[key])]
            band = ([(sx(x), sy(mean+se)) for x, (mean, se) in zip(stat['x'], stat[key])] +
                    [(sx(x), sy(mean-se)) for x, (mean, se) in reversed(list(zip(stat['x'], stat[key])))])
            canvas.band(band, color)
            canvas.polyline(points, color, 2.5)
        sx, sy = b
        points = [(sx(x), sy(value)) for x, value in zip(stat['x'], stat['frequencies'])]
        canvas.polyline(points, color, 2)
        for x, y in points:
            canvas.marker(x, y, color, radius=3)
    sx, sy = axes(canvas, 880, 670, 625, 300, 'D  Final CV-selected model: held-out recovery',
                  (0, 110), range(0, 101, 20), [(i/3, v) for i, v in enumerate(VARIANT_LABELS)],
                  'Only the selected model has saved test Top-2', 'Top-2 recovery (%)')
    lookup = {(r['cell'], r['variant']): r for r in formal}
    for i, variant in enumerate(VARIANTS):
        row = lookup['dslassocov', variant]
        x, y = sx(i/3), sy(row['top2_pct'])
        canvas.rect(x-44, y, 88, sy(0)-y, COLORS[i])
        canvas.error(x, sy(row['top2_pct']-row['top2_se_pct']), sy(row['top2_pct']+row['top2_se_pct']), '#334155')
        canvas.text(x, y-10, f"{row['top2_pct']:.1f}%", size=16, anchor='middle', bold=True)
    canvas.text(30, 1072, 'This is NOT a fixed-lambda test Top-2 sweep: only the CV-selected model was evaluated on the test set.', size=17, bold=True)
    canvas.text(30, 1104, 'Shaded bands: mean +/- 1 SE over 300 seeds. Relative lambda aligns the recorded grid separately within each seed.', size=16, color='#475569')
    canvas.text(30, 1136, 'Panel C is NOT total communication: loss/backtracking rounds, lambda-max fitting and final refitting are excluded.', size=16, color='#475569')
    save_svg(canvas, dest, 'noniid_multiround_lasso_tuning')


def write_csv(path, rows):
    safe_path(path)
    require(rows, 'cannot write an empty table')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostics_summary(rows):
    summaries = []
    for cell in ('r_unpenalized', 'r_adaptive_lasso'):
        for variant in ('control', 'site_gaussian_shift'):
            subset = [r for r in rows if r['cell'] == cell and r['variant'] == variant]
            require(len(subset) == 300, 'incomplete R diagnostic coverage')
            for has_pure in (False, True):
                group = [r for r in subset if (r['pure_outcome_sites'] > 0) == has_pure]
                summaries.append(dict(cell=cell, variant=variant, has_pure_outcome_site=has_pure,
                                      n=len(group), all_zero_slopes=sum(r['zero_slopes'] for r in group),
                                      top2_pct=100 * statistics.mean(r['top2_probability'] for r in group) if group else None))
    return summaries


def read_render_tables(dest, report):
    """Reuse only hash-bound presentation tables; never bypass scientific validation."""
    require(report['formal_evidence_sha256'] == EVIDENCE_SHA, 'wrong accepted render source')
    for name, expected in report['table_sha256'].items():
        require(sha(safe_path(dest / name, must_exist=True).read_bytes()) == expected,
                f'cached presentation table changed: {name}')
    with (dest / 'completed_noniid_top2_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        require(row['split'] == 'noniid' and int(row['n']) == 300, 'cached row is not formal non-IID')
        row['n'] = int(row['n'])
        for key in ('top2_pct', 'top2_se_pct', 'auroc', 'auroc_se'):
            row[key] = float(row[key])
    formal = [r for r in rows if r['cohort'] == 'comparison_v1']
    historical = [r for r in rows if r['cohort'] == 'historical_logit']
    require(len(formal) == 88 and len(historical) == 32, 'cached summary coverage changed')
    grouped = defaultdict(dict)
    with (dest / 'multiround_lasso_noniid_cv_path.csv').open(newline='') as handle:
        for r in csv.DictReader(handle):
            key = r['variant'], int(r['seed'])
            index = int(r['grid_index'])
            require(index not in grouped[key], 'duplicate cached lambda index')
            grouped[key][index] = r
    lasso = []
    for (variant, seed), items in grouped.items():
        require(set(items) == set(range(30)), 'incomplete cached lambda path')
        selected = [i for i in range(30) if items[i]['selected'] == 'True']
        require(len(selected) == 1, 'invalid cached lambda selection')
        lasso.append(dict(variant=variant, seed=seed, selected_index=selected[0],
                          lambda_ratio=[float(items[i]['lambda_ratio']) for i in range(30)],
                          losses=[float(items[i]['cv_logloss']) for i in range(30)],
                          cv_gradient_rounds=[int(items[i]['cv_gradient_rounds']) for i in range(30)]))
    require(len(lasso) == 600, 'cached LASSO coverage changed')
    return formal, historical, lasso


def render(formal, historical, lasso, dest, report, skip_lasso):
    plt, np = plot_setup()
    plot_all(formal, historical, dest, plt, np)
    plot_fedavg(formal, dest, plt, np)
    names = ['noniid_all_methods_top2', 'noniid_multiround_fedavg_parameters']
    if not skip_lasso:
        plot_lasso(lasso, formal, dest, plt, np)
        names.append('noniid_multiround_lasso_tuning')
    report['render_status'] = 'complete'
    report['plot_script_sha256'] = sha(Path(__file__).read_bytes())
    report['figure_sha256'] = {f'{name}.{ext}': sha((dest / f'{name}.{ext}').read_bytes())
                               for name in names for ext in ('svg', 'png')}
    safe_path(dest / 'completed_noniid_plot_provenance.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Figures and tables: {dest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEST)
    parser.add_argument('--skip-lasso', action='store_true', help='omit the LASSO tuning figure')
    parser.add_argument('--render-only', action='store_true', help='reuse previously verified, hash-bound presentation tables')
    args = parser.parse_args()
    dest = safe_path(args.output_dir)
    require(dest != DATA and dest != SCRATCH and OUTPUTS not in dest.parents and dest != OUTPUTS,
            'figures must not be written to a scientific run or artifact root')
    dest.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        report = json.loads(safe_path(dest / 'completed_noniid_plot_provenance.json', must_exist=True).read_text())
        formal, historical, lasso = read_render_tables(dest, report)
        render(formal, historical, lasso, dest, report, args.skip_lasso)
        return
    raw, lasso, diagnostics = collect_formal()
    formal = summarize(raw)
    historical, historical_sha = collect_historical()
    write_csv(dest / 'completed_noniid_top2_summary.csv', formal + historical)
    write_csv(dest / 'completed_noniid_top2_seed_rows.csv', raw)
    write_csv(dest / 'rlearner_noniid_seed_diagnostics.csv', diagnostics)
    r_summary = diagnostics_summary(diagnostics)
    write_csv(dest / 'rlearner_noniid_diagnostic_summary.csv', r_summary)
    lambda_rows = [dict(variant=r['variant'], seed=r['seed'], grid_index=i,
                        lambda_value=r['lambda_path'][i], lambda_ratio=r['lambda_ratio'][i],
                        cv_logloss=r['losses'][i], cv_gradient_rounds=r['cv_gradient_rounds'][i],
                        selected=i == r['selected_index']) for r in lasso for i in range(30)]
    write_csv(dest / 'multiround_lasso_noniid_cv_path.csv', lambda_rows)
    report = dict(kind='deterministic-presentation-report', scientific_mutations=False,
                  formal_evidence_sha256=EVIDENCE_SHA, actual_formal_artifact_ledger='verified_57600_units',
                  historical_summary_sha256=historical_sha, formal_noniid_rows=len(raw),
                  displayed_new_cells=len(CELLS), displayed_historical_cells=len(OLD_CELLS),
                  pilot_included=False, group_lasso_included=False, r_diagnostics=r_summary,
                  historical_validation='existing 300-seed summary; dsLasso result-env hashes separately rechecked',
                  lasso_limitation='saved CV tuning path, not independently evaluated lambda/round settings',
                  standard_errors='formal: sample SD / sqrt(300); historical: retained original report SE')
    report['table_sha256'] = {name: sha((dest / name).read_bytes()) for name in (
        'completed_noniid_top2_summary.csv', 'completed_noniid_top2_seed_rows.csv',
        'rlearner_noniid_seed_diagnostics.csv', 'rlearner_noniid_diagnostic_summary.csv',
        'multiround_lasso_noniid_cv_path.csv')}
    report['render_status'] = 'not_started'
    safe_path(dest / 'completed_noniid_plot_provenance.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    render(formal, historical, lasso, dest, report, args.skip_lasso)


if __name__ == '__main__':
    main()
