"""Analyze certified Z4 runs without merging stratified and uniform samples."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics

from lee_extended import composition_probability


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def interval(cluster_pairs, seed=20261102, repetitions=2000):
    pairs = list(cluster_pairs)
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sample = rng.choices(pairs, k=len(pairs))
        draws.append(sum(x[0] for x in sample)/sum(x[1] for x in sample))
    draws.sort()
    return [draws[repetitions//40], draws[repetitions-repetitions//40-1]]


def group_table(rows):
    buckets = defaultdict(list)
    for row in rows:
        pop = 'uniform' if row['population'] == 'uniform' else 'stratified'
        for a in ('all', str(row['a']), 'positive' if row['a'] else 'zero'):
            for n, k0 in (('all', 'all'), (str(row['n']), 'all'), (str(row['n']), str(row['k0']))):
                buckets[pop, a, n, k0, row['method']].append(row)
    output = []
    for (pop, a, n, k0, method), data in sorted(buckets.items()):
        completed = sum(x['status'] == 'complete' for x in data)
        output.append(dict(population=pop, a=a, n=n, k0=k0, method=method,
                           instances=len(data), clusters=len({x['cluster'] for x in data}),
                           completed=completed, success_rate=completed/len(data),
                           scope='complete' if completed == len(data) else 'budget_limited',
                           mean_online_ms=statistics.mean(x['online_ns'] for x in data)/1e6,
                           mean_first_ms=statistics.mean(x.get('first_ns', 0) for x in data)/1e6,
                           mean_recovery_ms=statistics.mean(x.get('recovery_ns', 0) for x in data)/1e6,
                           mean_cold_ms=statistics.mean(x['cold_ns'] for x in data)/1e6,
                           mean_proxy=statistics.mean(x['arithmetic_proxy'] for x in data),
                           mean_first_trials=statistics.mean(x.get('first_trials', 0) for x in data),
                           mean_recovery_trials=statistics.mean(x.get('recovery_trials', 0) for x in data),
                           mean_first_candidates=statistics.mean(x.get('first_candidates', 0) for x in data),
                           mean_second_candidates=statistics.mean(x.get('second_candidates', 0) for x in data),
                           first_singular_rate=sum(x.get('first_singular_trials', 0) for x in data)/
                           max(1, sum(x.get('first_trials', 0) for x in data)),
                           second_singular_rate=sum(x.get('second_singular_trials', 0) for x in data)/
                           max(1, sum(x.get('second_trials', 0) for x in data))))
    return output


def weighted_sphere(rows):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['population'] != 'uniform':
            buckets[row['code'], row['method']][row['a']].append(row)
    weighted = []
    for (code, method), compositions in sorted(buckets.items()):
        example = next(iter(compositions.values()))[0]
        n, t = example['n'], example['t']
        required = [a for a in range(t//2+1) if composition_probability(n, t, a)]
        if set(required) != set(compositions):
            continue
        complete = all(r['status'] == 'complete' for data in compositions.values() for r in data)
        entry = dict(code=code, cluster=example['cluster'], method=method, n=n, K=example['K'],
                     k0=example['k0'], t=t, scope='complete' if complete else 'budget_limited')
        for metric in ('online_ns', 'cold_ns', 'arithmetic_proxy', 'first_trials', 'recovery_trials'):
            entry[metric] = sum(float(composition_probability(n, t, a))*
                                statistics.mean(r.get(metric, 0) for r in compositions[a]) for a in required)
        entry['nonzero_radius_probability'] = 1-float(composition_probability(n, t, 0))
        weighted.append(entry)
    return weighted


def comparisons(weighted):
    result = []
    for n in [None]+sorted({r['n'] for r in weighted}):
        subset = [r for r in weighted if n is None or r['n'] == n]
        for baseline, ep in (('proj_ring_t', 'ep_reject'), ('proj_binary_t', 'ep_reject'),
                             ('proj_binary_all', 'ep_all'), ('ep_reject', 'ep_all')):
            selected = [r for r in subset if r['method'] in (baseline, ep)]
            if not selected or any(r['scope'] != 'complete' for r in selected):
                continue
            by_code = defaultdict(dict)
            for r in selected:
                by_code[r['code']][r['method']] = r
            if any(set(d) != {baseline, ep} for d in by_code.values()):
                continue
            for metric in ('online_ns', 'cold_ns', 'arithmetic_proxy'):
                clusters = defaultdict(lambda: [0.0, 0.0])
                worse = 0
                for d in by_code.values():
                    clusters[d[ep]['cluster']][0] += d[ep][metric]
                    clusters[d[ep]['cluster']][1] += d[baseline][metric]
                    worse += d[ep][metric] > d[baseline][metric]
                ci = interval(clusters.values())
                result.append(dict(n=n, baseline=baseline, method=ep, metric=metric,
                                   codes=len(by_code), clusters=len(clusters),
                                   ratio=sum(x[0] for x in clusters.values())/sum(x[1] for x in clusters.values()),
                                   low95=ci[0] if ci else None, high95=ci[1] if ci else None,
                                   codes_with_higher_method_cost=worse))
    return result


def radius_table(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['n'], row['population'], row['method']].append(row)
    output = []
    for (n, population, method), data in sorted(buckets.items()):
        t, radius = data[0]['t'], data[0]['initial_radius']
        output.append(dict(n=n, population=population, method=method, calls=len(data),
                            initial_radius=data[0]['initial_radius'],
                           exact_sphere_initial_radius_coverage=float(sum(
                               composition_probability(n, t, a) for a in range(t//2+1) if t-2*a <= radius)),
                           fraction_target_within_initial_radius=statistics.mean(r['s'] <= r['initial_radius'] for r in data),
                           success_rate=statistics.mean(r['status'] == 'complete' for r in data),
                           mean_budget_limited_ms=statistics.mean(r['online_ns'] for r in data)/1e6,
                           mean_radius_fallbacks=statistics.mean(r.get('first_radius_fallbacks', 0) for r in data),
                           censored=sum(r['status'] != 'complete' for r in data)))
    return output


def coverage_models(rows):
    configurations = sorted({(r['n'], r['K'], r['k0'], r['t']) for r in rows})
    result = []
    for n, k, k0, t in configurations:
        for a in range(t//2+1):
            s = t-2*a
            probability = composition_probability(n, t, a)
            if not probability:
                continue
            result.append(dict(n=n, K=k, k0=k0, t=t, a=a, s=s,
                               sphere_probability=float(probability),
                               ep_first_coverage=math.comb(n-k0, s)/math.comb(n, s),
                               projection_first_coverage=math.comb(n-k, s)/math.comb(n, s),
                               recovery_coverage=math.comb(n-k-s, a)/math.comb(n-s, a),
                               first_direct=(s == 0), recovery_direct=(a == 0)))
    return result


def analyze(folder):
    rows = [json.loads(line) for line in (folder/'instances.jsonl').read_text().splitlines()]
    groups = group_table(rows)
    weighted = weighted_sphere(rows)
    ratios = comparisons(weighted)
    write_csv(folder/'composition_groups.csv', groups)
    write_csv(folder/'sphere_weighted_by_code.csv', weighted)
    write_csv(folder/'sphere_weighted_comparisons.csv', ratios)
    write_csv(folder/'radius_groups.csv', radius_table(rows))
    write_csv(folder/'coverage_models.csv', coverage_models(rows))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row['code'], row['population'], row['case']][row['method']] = row
    invariants = dict(paired_instances=len(pairs), calls=len(rows),
                      censored=sum(r['status'] != 'complete' for r in rows),
                      incorrect_completed=sum(r['status'] == 'complete' and not r['correct'] for r in rows),
                      nonzero_radius_instances=sum(next(iter(v.values()))['a'] > 0 for v in pairs.values()),
                      radius_at_least_two_instances=sum(next(iter(v.values()))['a'] >= 2 for v in pairs.values()))
    for label, pair in pairs.items():
        for row in pair.values():
            reference = next(iter(pair.values()))
            for field in ('error', 'syndrome', 'trial_seed'):
                if row[field] != reference[field]:
                    raise AssertionError(('Unpaired input', label, field))
        baseline = pair.get('proj_ring_t')
        if baseline and baseline['status'] == 'complete':
            for name in ('proj_ring_mean_ceil', 'proj_ring_mean_floor', 'proj_ring_exact_ceil'):
                if name in pair and pair[name]['status'] == 'complete':
                    for field in ('first_trials', 'first_candidates', 'ring_trials'):
                        if pair[name].get(field, 0) != baseline.get(field, 0):
                            raise AssertionError(('Covered fixed-radius trace differs', label, name, field))
        if ('proj_binary_t' in pair and 'ep_reject' in pair and
                all(pair[m]['status'] == 'complete' for m in ('proj_binary_t', 'ep_reject'))):
            for field in ('second_trials', 'second_candidates', 'second_arithmetic_proxy'):
                if pair['proj_binary_t'].get(field, 0) != pair['ep_reject'].get(field, 0):
                    raise AssertionError(('Matched recovery differs', label, field))
        if ('proj_binary_all' in pair and 'ep_all' in pair and
                all(pair[m]['status'] == 'complete' for m in ('proj_binary_all', 'ep_all'))):
            for field in ('second_trials', 'second_candidates', 'second_arithmetic_proxy'):
                if pair['proj_binary_all'].get(field, 0) != pair['ep_all'].get(field, 0):
                    raise AssertionError(('Matched all-solutions recovery differs', label, field))
    (folder/'analysis.json').write_text(json.dumps(dict(invariants=invariants, comparisons=ratios), indent=2), encoding='utf-8')
    lines = ['# Z4 Lee Experiment Analysis', '',
             'Times are newly measured, not combined with the historical experiment. '
             'Stratified rows describe the sampling mixture; sphere-weighted estimates use every composition.', '',
             '```json', json.dumps(invariants, indent=2), '```', '',
             '| Population | a | Method | Calls | Completed | Online ms | Recovery ms | Recovery trials |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in groups:
        if row['n'] == row['k0'] == 'all' and row['a'] in ('zero', 'positive'):
            lines.append(f"| {row['population']} | {row['a']} | {row['method']} | {row['instances']} | "
                         f"{row['completed']} | {row['mean_online_ms']:.3f} | {row['mean_recovery_ms']:.3f} | "
                         f"{row['mean_recovery_trials']:.3f} |")
    lines += ['', '## Exact-Sphere Weighted Comparisons', '',
              'Each conditional composition mean is weighted by its exact finite-length sphere probability. '
              'Intervals resample mother-matrix clusters. Ratios below one favor the method column.', '',
              '| n | Baseline | Method | Metric | Ratio | 95% interval | Codes slower |',
              '|---|---|---|---|---:|---|---:|']
    for row in ratios:
        ci = (f"[{row['low95']:.3f}, {row['high95']:.3f}]" if row['low95'] is not None else 'insufficient clusters')
        lines.append(f"| {row['n'] or 'all'} | {row['baseline']} | {row['method']} | {row['metric']} | "
                     f"{row['ratio']:.3f} | {ci} | {row['codes_with_higher_method_cost']}/{row['codes']} |")
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(folder.name, json.dumps(invariants), flush=True)
    return invariants


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    args = parser.parse_args()
    for folder in args.folders:
        analyze(folder.resolve())
