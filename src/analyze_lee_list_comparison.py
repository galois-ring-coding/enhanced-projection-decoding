"""Exact-composition averages and paired mother-cluster intervals."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lee_extended import composition_probability, prepare
from lee_list_isd import Parameters, decode
from lee_z4_dcc_comparison import lee, raw_syndrome

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mother_points(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['t'], row['k0'], row['method'], row['mother'], row['a'])
        grouped[key].append(row)
    result = []
    prefixes = sorted({key[:4] for key in grouped})
    for t, k0, method, mother in prefixes:
        for group in ('sphere', 'a_positive'):
            allowed = [a for a in range(t//2+1) if group == 'sphere' or a > 0]
            mass = sum(float(composition_probability(64, t, a)) for a in allowed)
            point = dict(t=t, k0=k0, method=method, mother=mother, group=group,
                all_complete=all(row['status'] == 'complete' for a in allowed
                                 for row in grouped[t, k0, method, mother, a]))
            for metric in ('online_ms', 'completion', 'metric_tests', 'trials'):
                total = 0
                for a in allowed:
                    block = grouped[t, k0, method, mother, a]
                    if metric == 'online_ms':
                        values = [v['times']['online_ns']/1e6 for v in block]
                    elif metric == 'completion':
                        values = [v['status'] == 'complete' for v in block]
                    else:
                        values = [sum(value for key, value in v['counters'].items()
                                      if key.endswith('_'+metric)) for v in block]
                    total += float(composition_probability(64, t, a))*statistics.mean(values)/mass
                point[metric] = total
            result.append(point)
    return result


def branch_statistics(rows):
    """Infer final successful branch from the stored fixed four-trial schedule."""
    result = []
    for t, k0, method in sorted({(r['t'], r['k0'], r['method']) for r in rows}):
        block = [r for r in rows if (r['t'], r['k0'], r['method']) == (t, k0, method)]
        for phase in (('native',) if method.startswith('lee_') else ('first', 'recovery')):
            weighted = defaultdict(float)
            for a in range(t//2+1):
                cases = [r for r in block if r['a'] == a]
                mass = float(composition_probability(64, t, a))/len(cases)
                for row in cases:
                    count = row['counters'].get(phase+'_trials', 0)
                    nonzero = row['counters'].get(phase+'_nonzero_list_trials', 0)
                    if row['status'] != 'complete':
                        outcome = 'budget'
                    elif count == 0:
                        outcome = 'direct_shortcut'
                    elif nonzero > 0 and count % 4 != 0:
                        outcome = 'nonzero_list_success'
                    else:
                        outcome = 'zero_list_success'
                    weighted[outcome] += mass
            result.append(dict(t=t, k0=k0, method=method, phase=phase,
                **{name: weighted[name] for name in ('nonzero_list_success', 'zero_list_success',
                                                    'direct_shortcut', 'budget')}))
    return result


def summarize(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[point['t'], point['k0'], point['method'], point['group']].append(point)
    rng = np.random.default_rng(20261105)
    result = []
    for (t, k0, method, group), block in sorted(grouped.items()):
        values = np.asarray([p['online_ms'] for p in block])
        means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        low, high = np.quantile(means, (.025, .975))
        result.append(dict(t=t, k0=k0, method=method, group=group,
            mothers=len(block), online_ms=float(values.mean()), low_ms=float(low), high_ms=float(high),
            completion=statistics.mean(p['completion'] for p in block),
            trials=statistics.mean(p['trials'] for p in block),
            metric_tests=statistics.mean(p['metric_tests'] for p in block)))
    return result


def ratios(points, algorithms=('stern', 'mmt')):
    lookup = {(p['t'], p['k0'], p['method'], p['group'], p['mother']): p for p in points}
    result = []
    rng = np.random.default_rng(127)
    for t, k0, group in sorted({(p['t'], p['k0'], p['group']) for p in points}):
        for algorithm in algorithms:
            for variant in ('', '_family_best'):
                methods = ['ccp_'+algorithm+variant, 'lee_'+algorithm+variant]
                mothers = sorted({p['mother'] for p in points if p['t'] == t and p['k0'] == k0
                                  and p['group'] == group and p['method'] == methods[0]})
                if not mothers:
                    continue
                x = np.asarray([lookup[t, k0, methods[0], group, m]['online_ms'] for m in mothers])
                y = np.asarray([lookup[t, k0, methods[1], group, m]['online_ms'] for m in mothers])
                draws = rng.integers(0, len(mothers), size=(5000, len(mothers)))
                samples = x[draws].mean(axis=1)/y[draws].mean(axis=1)
                low, high = np.quantile(samples, (.025, .975))
                complete = all(lookup[t, k0, method, group, m]['all_complete']
                               for method in methods for m in mothers)
                result.append(dict(t=t, k0=k0, algorithm=algorithm,
                    variant='primary' if not variant else 'family_best', group=group,
                    ratio=float(x.mean()/y.mean()), low=float(low), high=float(high),
                    all_complete=complete,
                    assessment=('CCP lower' if high < 1 else 'native lower' if low > 1 else 'undetermined')
                    if complete else 'budget-limited comparison'))
    return result


def choose_families(points, folder, config):
    selected, choices = [], []
    for t in config['radii']:
        for k0 in (0, 3, 6, 9, 12):
            for prefix in ('ccp', 'lee'):
                candidates = []
                for algorithm in config['algorithms']:
                    tuning = json.loads((folder/f'tuning_{algorithm}_t{t}.json').read_text(encoding='utf-8'))
                    record = tuning[f'{k0}:{prefix}_{algorithm}']
                    loss = next(r['selection_loss_ns'] for r in record['candidates']
                                if r['parameters'] == record['family_best'])
                    candidates.append((loss, algorithm))
                loss, algorithm = min(candidates)
                choices.append(dict(t=t, k0=k0, side=prefix, algorithm=algorithm,
                                    training_loss_ns=loss))
                for point in points:
                    if (point['t'], point['k0'], point['method']) == (t, k0, f'{prefix}_{algorithm}_family_best'):
                        selected.append(dict(point, method=f'{prefix}_best'))
    return selected, choices


def plot_best(table, choices, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    choice = {(r['t'], r['k0'], r['side']): r['algorithm'] for r in choices}
    for ax, t in zip(axes, (4, 8)):
        for prefix, label, color, style in (('ccp', 'CCP + tuned binary ISD', '#1763a6', '-'),
                                           ('lee', 'Tuned native Lee ISD', '#bd3437', '--')):
            rows = sorted([r for r in table if r['t'] == t and r['group'] == 'sphere'
                           and r['method'] == prefix+'_best'], key=lambda r: r['k0'])
            x = [r['k0']/12 for r in rows]
            ax.plot(x, [r['online_ms'] for r in rows], style, color=color, marker='o', label=label)
            ax.fill_between(x, [r['low_ms'] for r in rows], [r['high_ms'] for r in rows],
                            color=color, alpha=.13)
            for r in rows:
                ax.annotate(choice[t, r['k0'], prefix][0].upper(), (r['k0']/12, r['online_ms']),
                            xytext=(3, 5 if prefix == 'ccp' else -13), textcoords='offset points',
                            color=color, fontsize=8)
        ax.set(title=f'n=64, K=12, Lee weight t={t}', xlabel='k0 / K',
               ylabel='Mean online time (ms)', xticks=[0, .25, .5, .75, 1])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Kernel and parameters selected on training mothers; S = Stern, M = MMT', fontsize=10)
    fig.tight_layout()
    fig.savefig(folder/'best_tuned_families.png', dpi=180)
    fig.savefig(folder/'best_tuned_families.pdf')
    plt.close(fig)


def plot_costs(table, folder, family=False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    suffix = '_family_best' if family else ''
    for i, t in enumerate((4, 8)):
        for j, algorithm in enumerate(('stern', 'mmt')):
            ax = axes[i, j]
            for prefix, label, color, style in (
                ('ccp_', 'CCP + binary '+algorithm.upper(), '#1763a6', '-'),
                ('lee_', 'Native Lee-'+algorithm.upper(), '#bd3437', '--')):
                rows = sorted([r for r in table if r['t'] == t and r['group'] == 'sphere'
                               and r['method'] == prefix+algorithm+suffix], key=lambda r: r['k0'])
                if rows:
                    x = [r['k0']/12 for r in rows]
                    ax.plot(x, [r['online_ms'] for r in rows], style, color=color,
                            marker='o', markersize=4, label=label)
                    ax.fill_between(x, [r['low_ms'] for r in rows], [r['high_ms'] for r in rows],
                                    color=color, alpha=.13)
            ax.set(title=f'n=64, K=12, Lee weight t={t}', xlabel='k0 / K',
                   ylabel='Mean online time (ms)', xticks=[0, .25, .5, .75, 1])
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Family-selected parameters (Prange allowed)' if family else
                 'Nonzero list parameters; every fourth trial uses zero-list weights', fontsize=11)
    fig.tight_layout()
    name = 'family_best_online' if family else 'stern_mmt_online'
    fig.savefig(folder/(name+'.png'), dpi=180)
    fig.savefig(folder/(name+'.pdf'))
    plt.close(fig)


def plot_ratios(table, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for ax, t in zip(axes, (4, 8)):
        for algorithm, color, offset in (('stern', '#1763a6', -.015), ('mmt', '#2d8659', .015)):
            rows = sorted([r for r in table if r['t'] == t and r['group'] == 'sphere'
                           and r['variant'] == 'primary' and r['algorithm'] == algorithm],
                          key=lambda r: r['k0'])
            if rows:
                ax.errorbar([r['k0']/12+offset for r in rows], [r['ratio'] for r in rows],
                    yerr=[[max(0, r['ratio']-r['low']) for r in rows],
                          [max(0, r['high']-r['ratio']) for r in rows]],
                    color=color, marker='o', capsize=3, label=algorithm.upper())
        ax.axhline(1, color='#555555', linestyle='--', linewidth=1)
        ax.set(title=f'Lee weight t={t}', xlabel='k0 / K', xticks=[0, .25, .5, .75, 1],
               ylabel='Mean CCP time / mean native Lee time')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Paired mother-cluster pointwise 95% bootstrap intervals', fontsize=10)
    fig.tight_layout()
    fig.savefig(folder/'paired_time_ratios.png', dpi=180)
    fig.savefig(folder/'paired_time_ratios.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', default='results/lee_list_isd_20261105')
    args = parser.parse_args()
    folder = ROOT/args.folder
    output = folder/'analysis'
    output.mkdir(exist_ok=True)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    for name, digest in config['hashes'].items():
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Benchmark source hash changed: '+name)
    rows = []
    for file in sorted(folder.glob('instances_*.jsonl')):
        rows.extend(json.loads(line) for line in file.read_text(encoding='utf-8').splitlines())
    keys = [(r['t'], r['k0'], r['mother'], r['a'], r['case'], r['seed_index'], r['method']) for r in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError('Duplicate instances')
    for row in rows:
        if lee(row['error']) != row['t'] or row['error'].count(2) != row['a']:
            raise AssertionError('Bad error metadata')
        if row['status'] == 'complete' and not row['correct']:
            raise AssertionError('Wrong successful call')
        if row['d_binary'] <= 16:
            raise AssertionError('Invalid distance condition')
    expected = sum(config['mothers']*5*(t//2+1)*config['samples']*config['seeds']*4
                   for t in config['radii'] for _ in config['algorithms'])
    if len(rows) != expected:
        raise AssertionError(f'Incomplete benchmark {len(rows)}/{expected}')
    # Re-run representative native/CCP calls with fresh independent entry points.
    source = Path(config['source'])
    if not source.exists():
        source = ROOT/'results/lee_sweep_20261103/subtypes_main/codes.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != config['source_sha256']:
        raise AssertionError('Archived input source hash mismatch')
    archived = json.loads(source.read_text(encoding='utf-8'))
    code_lookup = {c['code']: c for c in archived}
    replayed = 0
    for t in config['radii']:
        for method in sorted({r['method'] for r in rows if not r['method'].endswith('family_best')}):
            row = next(r for r in rows if r['t'] == t and r['method'] == method and r['k0'] == 6 and r['a'] == 1)
            public = prepare(code_lookup[row['code']]['h'])
            syndrome = raw_syndrome(public['h'], row['error'])
            check = decode(public, syndrome, t, row['search_seed'], method,
                           Parameters(**row['parameters']), seconds=30)
            if check['status'] != 'complete' or check['answer'] != row['error']:
                raise AssertionError('Independent replay failed')
            replayed += 1
    points = mother_points(rows)
    table = summarize(points)
    comparison = ratios(points)
    best_points, choices = choose_families(points, folder, config)
    best_table = summarize(best_points)
    best_ratios = ratios(best_points, ('best',))
    write_csv(output/'mother_means.csv', points)
    write_csv(output/'summary.csv', table)
    write_csv(output/'paired_ratios.csv', comparison)
    write_csv(output/'successful_branches.csv', branch_statistics(rows))
    write_csv(output/'best_selected_kernels.csv', choices)
    write_csv(output/'best_tuned_summary.csv', best_table)
    write_csv(output/'best_tuned_ratios.csv', best_ratios)
    write_csv(output/'selected_parameters.csv', [dict(t=t, key=key,
        **record['selected'], family_best=json.dumps(record['family_best']))
        for algorithm in config['algorithms'] for t in config['radii']
        for key, record in json.loads((folder/f'tuning_{algorithm}_t{t}.json').read_text(encoding='utf-8')).items()])
    statuses = dict(Counter(r['status'] for r in rows))
    audit = dict(records=len(rows), expected_records=expected, statuses=statuses,
        correct_completed=sum(r['status'] == 'complete' and r['correct'] for r in rows),
        timed_calls=sum(len(r['timing_samples']) for r in rows), independent_replays=replayed,
        unique_input_calls=len({k[:-1] for k in keys}),
        note='All raw calls include all compositions; displayed sphere averages use exact composition mass.')
    (output/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    plot_costs(table, output)
    plot_costs(table, output, family=True)
    plot_ratios(comparison, output)
    plot_best(best_table, choices, output)
    report = ['# Z4 Lee list-ISD comparison', '',
        f"n=64, K=12; {config['tune_mothers']} training mothers and {config['mothers']} distinct confirmation mothers.",
        'Subtypes k0=0,3,6,9,12; t=4 then t=8; two Stern methods then two MMT methods.',
        'Fixed K is not fixed cardinality rate: |C|=2^(K+k0). Every method comparison uses the same code.',
        f"Records: {len(rows)} counted, {audit['timed_calls']} timed; statuses: {statuses}.", '',
        '## Nonzero-list paired results', '',
        'Ratios are mean CCP online time divided by mean native Lee online time. Lower than one favors CCP.',
        'Intervals resample paired mother matrices; they are pointwise, not simultaneous region certificates.', '',
        '| t | Kernel | k0 | CCP/native | 95% interval | Assessment |',
        '|---|---|---|---|---|---|']
    for r in comparison:
        if r['group'] == 'sphere' and r['variant'] == 'primary':
            report.append(f"| {r['t']} | {r['algorithm']} | {r['k0']} | {r['ratio']:.3f} | [{r['low']:.3f}, {r['high']:.3f}] | {r['assessment']} |")
    report += ['', '## Best trained kernel in each side', '',
        'The kernel family (Stern or MMT) is also chosen on the training mothers, before using the confirmation measurements.', '',
        '| t | k0 | CCP/native | 95% interval | Assessment |',
        '|---|---|---|---|---|']
    for r in best_ratios:
        if r['group'] == 'sphere':
            report.append(f"| {r['t']} | {r['k0']} | {r['ratio']:.3f} | [{r['low']:.3f}, {r['high']:.3f}] | {r['assessment']} |")
    report += ['', '## Tuned families including the zero-weight boundary', '',
        'Use this comparison to assess the tuned algorithm families. The nonzero-list comparison above diagnoses the list construction itself.',
        'A selected zero-weight boundary is a Prange/PGE branch, not evidence that a nonzero-list Stern/MMT kernel accelerated decoding.', '',
        '| t | Kernel family | k0 | CCP/native | 95% interval | Assessment |',
        '|---|---|---|---|---|---|']
    for r in comparison:
        if r['group'] == 'sphere' and r['variant'] == 'family_best':
            report.append(f"| {r['t']} | {r['algorithm']} | {r['k0']} | {r['ratio']:.3f} | [{r['low']:.3f}, {r['high']:.3f}] | {r['assessment']} |")
    report += ['', '## Read the figures', '',
        '- stern_mmt_online: absolute online times; two radii by two kernels. Blue CCP, red native ring.',
        '- paired_time_ratios: paired intervals for the same primary comparisons.',
        '- family_best_online: training may choose zero-list weights (Prange/PGE boundary). This is a separate diagnostic.',
        '- best_tuned_families: training chooses both the kernel family and its parameters; S and M mark Stern and MMT. This is the overall tuned-family comparison.',
        '- successful_branches.csv: composition-weighted fractions of nonzero-list successes, zero-list successes and direct shortcuts in each phase.',
        '', '## Algorithms and accounting', '',
        'CCP uses complete low-digit equations from public H, then eliminates known odd positions in the high-digit system.',
        'Native Lee-Stern uses ring lists with the zero-window and all torsion equations in collision keys (Horlemann-Trautmann--Weger Algorithm 4).',
        'Native Lee-MMT follows the four-list PGE/representation construction of Li--Wang Algorithm 1; vectors and sums remain over Z4.',
        'Every fourth trial has zero base-list weights in both families. This fixed public schedule handles unknown projected weights and representation restrictions.',
        'Primary Stern parameters have positive list weight; primary MMT parameters have positive weights in both halves.',
        'A layer with insufficient radius or free-coordinate capacity falls back to feasible weights. In particular k0=0 has no binary free low-digit coordinates.',
        'Full collision buckets and all matching candidates are retained. Singular required-pivot systems are rejected and charged.',
        'Tuning uses separate mothers and exact composition weights, minimizing observed penalized budget loss. It is a finite grid, not a global optimality claim.',
        'Cold coordinate-word caches are cleared before each decode. Building words, labels, hash tables, back substitution and final verification are timed.',
        'CCP Smith preprocessing is excluded from online times and recorded separately; native methods are not charged for it.',
        'Shared live-list entry limits do not imply equal memory in bytes. memory_*.json contains separate traced allocation peaks on representative inputs; tracing times are not plotted.',
        'The archived distance-conditioned low-dimensional ensemble and fixed low weights do not establish cryptographic-scale asymptotic superiority.',
        '', '## Reproduction', '', '```powershell',
        f"python src/run_lee_list_comparison.py --output {config['output']} --mothers {config['mothers']} --tune-mothers {config['tune_mothers']} --samples {config['samples']} --seeds {config['seeds']} --repeats {config['repeats']} --tune-seconds {config['tune_seconds']} --seconds {config['seconds']}",
        f'python src/analyze_lee_list_comparison.py {config["output"]}', '```', '']
    (output/'README.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print('Results:', output)


if __name__ == '__main__':
    main()
