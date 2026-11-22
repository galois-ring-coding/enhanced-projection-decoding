"""Audit and summarize the matched standard-rejection Lee scale experiment."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np

from analyze_lee_extended import write_csv
from analyze_lee_fixed import METRICS
from analyze_lee_scale_confirmation import weighted
from lee_extended import composition_probability, distance_certificate, prepare
from lee_fixed_kernel import decode
from lee_revision import make_code
from lee_z4_dcc_comparison import lee, raw_syndrome
from run_lee_fixed import coverage_diagnostics
from run_lee_independent import ROOT, digest, matrix_digest
from run_lee_scale_confirmation import METHODS, PROFILES, error_pool, search_seed, seed_for


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def coverage_bounds(n, k, t, k0, method, group='sphere'):
    """Coverage-only lower bounds, counting zero-weight shortcuts as zero trials."""
    if method not in METHODS or group not in ('sphere', 'a_positive'):
        raise ValueError('Unknown method or group')
    height = n-k0 if method == 'ep_reject' else n-k
    weights = [(a, float(composition_probability(n, t, a)))
               for a in range(t//2+1) if group == 'sphere' or a > 0]
    mass = sum(p for _, p in weights)
    if not mass:
        raise ValueError('Empty error group')
    first = recovery = 0.0
    for a, probability in weights:
        s = t-2*a
        if s:
            first += probability/mass*math.comb(n, s)/math.comb(height, s)
        if a:
            recovery += probability/mass*math.comb(n-s, a)/math.comb(n-k-s, a)
    return first, recovery


def validate_stopping(rows):
    counts = Counter()
    for row in rows:
        low = {j for j, value in enumerate(row['error']) if value % 2}
        rest = [j for j in range(row['n']) if j not in low]
        high = {i for i, j in enumerate(rest) if row['error'][j] == 2}
        for phase, support in (('first', low), ('recovery', high)):
            events = [event for event in row['trace'] if event['phase'] == phase]
            if not support:
                if events:
                    raise AssertionError('Zero-weight stage sampled coordinates')
                counts[phase+'_shortcuts'] += 1
                continue
            if not events:
                raise AssertionError('Missing nonzero search')
            for index, event in enumerate(events):
                covered = support.issubset(event['chosen'])
                if covered != event['covers_true_support']:
                    raise AssertionError('Stored support coverage differs')
                successful = covered and event['nullity'] == 0
                if successful != (index == len(events)-1):
                    raise AssertionError('Search did not stop at first invertible covering selection')
                if covered and not event['consistent']:
                    raise AssertionError('Covered selection is inconsistent')
                counts[phase+'_selections'] += 1
                counts[phase+'_singular_selections'] += bool(event['nullity'])
                counts[phase+'_covering_selections'] += covered
                counts[phase+'_covered_singular_rejections'] += covered and bool(event['nullity'])
            counts[phase+'_nonzero_searches'] += 1
    return dict(counts)


def audit(folder, formal=True):
    config = read_json(folder/'config.json')
    n, k, t = (config[v] for v in ('n', 'K', 't'))
    if (tuple(config['methods']) != METHODS or config['singular_policy'] != 'reject'
            or config['kernel'] != 'common'):
        raise AssertionError('Wrong decoding policy')
    if formal and (config['mothers'], config['seeds'], config['repeats']) != (16, 4, 3):
        raise AssertionError('Incomplete formal design')
    for name, expected in config['source_hashes'].items():
        if digest(ROOT/'src'/name) != expected:
            raise AssertionError('Measured source changed: '+name)
    codes = read_json(folder/'codes.json')
    by_code = {code['code']: code for code in codes}
    mothers = {code['cluster']: code for code in codes}
    pools = read_json(folder/'error_pools.json')
    if len(by_code) != len(codes) or len(codes) != config['mothers']*(k+1):
        raise AssertionError('Invalid subtype panel')
    fingerprints, distances = set(), {}
    for cluster, code in mothers.items():
        fingerprint = matrix_digest(code['ambient'])
        if fingerprint in fingerprints or fingerprint != code['ambient_sha256']:
            raise AssertionError('Repeated or modified ambient matrix')
        fingerprints.add(fingerprint)
        if code['seed'] != seed_for(config['seed'], n, code['mother_index'], 'ambient'):
            raise AssertionError('Ambient seed mismatch')
        distance = distance_certificate(code['ambient'][:k])
        if distance != code['d_binary'] or distance <= 4*t:
            raise AssertionError('Invalid distance certificate')
        distances[cluster] = distance
        base = make_code(code['ambient'], k, 0)
        for k0 in range(k+1):
            child = by_code[f'{cluster}_k0{k0}']
            expected_h = base['dual'][k:]+[[2*x % 4 for x in r] for r in base['dual'][k0:k]]
            if child['h'] != expected_h or child['ambient'] != code['ambient']:
                raise AssertionError('Subtype construction mismatch')
        if pools[cluster] != error_pool(config['seed'], n, t, code['mother_index']):
            raise AssertionError('Error pool changed')
    expected = {}
    for code in codes:
        for case in pools[code['cluster']]:
            for index in range(1 if case['population'] == 'uniform' else config['seeds']):
                for method in METHODS:
                    expected[code['code'], case['population'], case['case'], index, method] = case
    rows = [json.loads(line) for line in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    seen, pairs, outcomes = set(), defaultdict(dict), Counter()
    projection_counts = defaultdict(set)
    for row in rows:
        key = tuple(row[v] for v in ('code', 'population', 'case', 'seed_index', 'method'))
        if key in seen or key not in expected:
            raise AssertionError('Unexpected or duplicate measurement')
        seen.add(key)
        code, case = by_code[row['code']], expected[key]
        if any(row[field] != case[field] for field in ('error', 'a', 's', 'error_seed')):
            raise AssertionError('Paired input mismatch')
        if any(row[field] != code[field] for field in ('n', 'K', 't', 'k0', 'cluster', 'd_binary')):
            raise AssertionError('Code metadata mismatch')
        if lee(row['error']) != t or row['syndrome'] != raw_syndrome(code['h'], row['error']):
            raise AssertionError('Invalid error or syndrome')
        if row['trial_seed'] != search_seed(config['seed'], n, code['mother_index'],
                                           row['population'], row['case'], row['seed_index']):
            raise AssertionError('Search seed mismatch')
        if len(row['timing_samples']) != config['repeats']:
            raise AssertionError('Missing timing repeats')
        for metric in row['timing_samples'][0]:
            if row[metric] != statistics.median(sample[metric] for sample in row['timing_samples']):
                raise AssertionError('Timing median mismatch')
        outcomes[row['status']] += 1
        if row['status'] != 'complete' or not row['correct']:
            raise AssertionError('Incomplete or incorrect result; do not filter failed measurements')
        pairs[key[:-1]][row['method']] = row
        if row['method'] == 'proj_binary_t':
            projection_counts[row['cluster'], row['population'], row['case'], row['seed_index']].add(row.get('first_trials', 0))
    if seen != set(expected):
        raise AssertionError('Incomplete schedule')
    for pair in pairs.values():
        ep, projection = (pair[method] for method in METHODS)
        fields = {field for row in pair.values() for field in row if field.startswith('second_')}
        if any(ep.get(field, 0) != projection.get(field, 0) for field in fields):
            raise AssertionError('Matched recovery counters differ')
        if ([event for event in ep['trace'] if event['phase'] == 'recovery'] !=
                [event for event in projection['trace'] if event['phase'] == 'recovery']):
            raise AssertionError('Matched recovery traces differ')
        if ep['k0'] == k:
            for field in ('trace', 'first_trials', 'first_candidates', 'arithmetic_proxy'):
                if ep.get(field, 0) != projection.get(field, 0):
                    raise AssertionError('Free-code boundary differs')
    if any(len(values) != 1 for values in projection_counts.values()):
        raise AssertionError('Projection trials vary across subtypes')
    stopping = validate_stopping(rows)
    selected = [row for row in rows if row['population'] == 'a1' and row['case'] == 0
                and row['seed_index'] == 0 and (by_code[row['code']]['mother_index'] == 0 or row['k0'] == k//2)]
    prepared = {}
    for row in selected:
        if row['code'] not in prepared:
            prepared[row['code']] = prepare(by_code[row['code']]['h'])
        replay = decode(prepared[row['code']], row['syndrome'], t, row['trial_seed'],
                        row['method'], config['trials'], config['candidates'])
        replay['counters'].update(coverage_diagnostics(replay, row['error']))
        if replay['answer'] != row['error'] or replay['trace'] != row['trace']:
            raise AssertionError('Counted replay differs')
        if any(row.get(field, 0) != value for field, value in replay['counters'].items()):
            raise AssertionError('Replayed counters differ')
    check = dict(status='passed', n=n, K=k, t=t, mothers=len(mothers), codes=len(codes),
                 records=len(rows), timing_calls=sum(len(row['timing_samples']) for row in rows),
                 distances=distances, outcomes=dict(outcomes), stopping=stopping,
                 matched_recovery=True, free_boundary_identical=True, replayed_calls=len(selected))
    (folder/'audit.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    return rows, check


def summarize(points, repetitions):
    table, ratios = [], []
    for n in sorted({row['n'] for row in points}):
        data = [row for row in points if row['n'] == n]
        k, t = data[0]['K'], data[0]['t']
        clusters = sorted({row['cluster'] for row in data})
        count = len(clusters)
        lookup = {(row['cluster'], row['k0'], row['group'], row['method']): row for row in data}
        rng = np.random.default_rng(2026111923+n)
        draws = rng.multinomial(count, np.full(count, 1/count), size=repetitions)
        for group in ('sphere', 'a_positive'):
            arrays = {}
            for method in METHODS:
                for metric in METRICS:
                    arrays[method, metric] = np.array([[lookup[c, k0, group, method][metric]
                                                       for k0 in range(k+1)] for c in clusters])
                for k0 in range(k+1):
                    item = dict(n=n, K=k, t=t, k0=k0, group=group, method=method,
                                extra_fraction=(k-k0)/k, mothers=count)
                    for metric in METRICS:
                        item[metric] = float(arrays[method, metric][:, k0].mean())
                    first, recovery = coverage_bounds(n, k, t, k0, method, group)
                    item.update(coverage_first_trials_lower_bound=first,
                                coverage_recovery_trials_lower_bound=recovery)
                    for metric in ('online_ns', 'first_trials', 'recovery_trials'):
                        samples = draws @ arrays[method, metric][:, k0]/count
                        low, high = np.quantile(samples, [.025, .975])
                        item[metric+'_low95'], item[metric+'_high95'] = float(low), float(high)
                    table.append(item)
            for metric in ('online_ns', 'first_trials', 'first_ns', 'recovery_ns', 'arithmetic_proxy'):
                ep, projection = (arrays[method, metric] for method in METHODS)
                ratio = ep.mean(axis=0)/projection.mean(axis=0)
                samples = (draws @ ep)/(draws @ projection)
                width = float(np.quantile(np.max(np.abs(np.log(samples)-np.log(ratio)), axis=1), .95))
                low, high = np.quantile(samples, [.025, .975], axis=0)
                for k0 in range(k+1):
                    slo, shi = ratio[k0]*np.exp(-width), ratio[k0]*np.exp(width)
                    ratios.append(dict(n=n, K=k, t=t, k0=k0, group=group, metric=metric,
                                       ratio=float(ratio[k0]), low95=float(low[k0]), high95=float(high[k0]),
                                       simultaneous_low95=float(slo), simultaneous_high95=float(shi),
                                       result='CCP_lower' if shi < 1 else 'projection_lower' if slo > 1 else 'unresolved'))
    return table, ratios


def integer_ranges(values):
    values = sorted(values)
    runs = []
    for value in values:
        if runs and value == runs[-1][-1]+1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return ','.join(str(run[0]) if len(run) == 1 else
                    f'{run[0]},{run[1]}' if len(run) == 2 else
                    f'{run[0]},\\ldots,{run[-1]}' for run in runs)


def manuscript_results(table, ratios):
    selected = {(row['n'], row['k0'], row['group']): row for row in ratios
                if row['metric'] == 'online_ns'}
    lengths = sorted({row['n'] for row in table})
    representatives = [(n, PROFILES[n][0]//2) for n in lengths]
    reductions = [100*(1-selected[n, k0, 'sphere']['ratio']) for n, k0 in representatives]
    points = ','.join(f'({n},{k0})' for n, k0 in representatives)
    changes = ', '.join(f'${value:.1f}\\%$' for value in reductions)
    lines = [f'At the representative intermediate subtypes $(n,k_0)={points}$,',
             f'the mean online-time reductions are {changes}, respectively.',
             'The following comparisons use paired CCP/baseline time-ratio intervals with',
             '$95\\%$ simultaneous coverage over all subtypes within each length',
             'and error group.', '']
    for group, title in (('sphere', 'For uniform-sphere mean time'),
                         ('a_positive', 'For errors conditioned on $a>0$')):
        comparisons = []
        for n in lengths:
            k = PROFILES[n][0]
            categories = []
            for flag, phrase in (('CCP_lower', 'lie below one'),
                                  ('projection_lower', 'lie above one'),
                                  ('unresolved', 'contain one')):
                values = [k0 for k0 in range(k+1) if selected[n, k0, group]['result'] == flag]
                if values:
                    categories.append(phrase + ' for $k_0=' + integer_ranges(values) + '$')
            predicates = (categories[0] if len(categories) == 1 else
                          ' and '.join(categories) if len(categories) == 2 else
                          ', '.join(categories[:-1]) + ', and ' + categories[-1])
            comparisons.append(f'at $n={n}$, the intervals ' + predicates)
        lines.append(title + ', ' + '; '.join(comparisons) + '.')
        lines.append('')
    boundary = [selected[n, PROFILES[n][0], 'sphere']['ratio'] for n in lengths]
    lines += ['At the free-code boundary $k_0=K$, both low-digit systems and',
              'search counters coincide. The measured CCP/baseline time ratios are',
              ', '.join(f'${value:.3f}$' for value in boundary) +
              ' at lengths ' + ', '.join(f'${n}$' for n in lengths) + ', respectively.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/lee_scale_rejection_20261119')
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error('At least two bootstrap resamples are required')
    rows, checks, inputs = [], [], []
    for n in PROFILES:
        source = args.source/f'n{n}'
        if args.smoke and not source.exists():
            continue
        measured, check = audit(source, formal=not args.smoke)
        rows.extend(measured)
        checks.append(check)
        inputs.append(source)
        print(json.dumps(check), flush=True)
    points = weighted(rows)
    table, ratios = summarize(points, args.bootstrap)
    output = args.source/'analysis'
    output.mkdir(exist_ok=True)
    for name, data in (('weighted_by_code', points), ('panel_summary', table), ('paired_ratios', ratios)):
        write_csv(output/(name+'.csv'), data)
    rank_rows = []
    for row in table:
        for phase, prefix, covered_field in (('first', 'first_', 'first_covered_trials'),
                                             ('recovery', 'second_', None)):
            trials = row['first_trials' if phase == 'first' else 'recovery_trials']
            singular = row[prefix+'singular_trials']
            rank_rows.append(dict(n=row['n'], k0=row['k0'], method=row['method'], group=row['group'],
                                  phase=phase, mean_trials=trials, mean_singular_trials=singular,
                                  singular_fraction=singular/trials if trials else 0,
                                  mean_covered_trials=row[covered_field] if covered_field else '',
                                  mean_covered_singular_rejections=row['first_covered_singular_trials']
                                  if phase == 'first' else ''))
    write_csv(output/'rank_rejection.csv', rank_rows)
    (output/'manuscript_results.tex').write_text(manuscript_results(table, ratios), encoding='utf-8')
    (output/'validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    lines = ['# Standard-rejection Lee scale experiment', '',
             'Both methods reject every singular selection and share conditional binary recovery.',
             'All lengths were measured again; no all-solutions timings were relabeled.',
             'The first-stage coverage reciprocal is a lower bound, not an exact trial prediction.', '',
             '| n | k0 | group | CCP/baseline time | simultaneous 95% interval | result |',
             '|---:|---:|---|---:|---|---|']
    for row in ratios:
        if row['metric'] == 'online_ns':
            lines.append(f'| {row["n"]} | {row["k0"]} | {row["group"]} | {row["ratio"]:.4f} | '
                         f'[{row["simultaneous_low95"]:.4f}, {row["simultaneous_high95"]:.4f}] | {row["result"]} |')
    lines += ['', 'Intervals resample independent ambient-matrix clusters, preserving all paired subtypes,',
              'errors and search seeds. Simultaneous coverage is over all subtypes within each length,',
              'error group and metric. Uniform validation errors are excluded from composition-weighted estimates.', '',
              'Reproduce:', '', '```text', 'python src/run_lee_scale_confirmation.py',
              'python src/analyze_lee_rejection.py', 'python ../../1论文/plot_lee_comparison.py', '```', '']
    (output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    manifest = dict(bootstrap=args.bootstrap, singular_policy='reject',
                    input_hashes={str(folder.resolve()): {name: digest(folder/name) for name in
                        ('config.json', 'codes.json', 'error_pools.json', 'instances.jsonl', 'audit.json', 'environment.json')}
                        for folder in inputs},
                    analysis_sources={name: digest(ROOT/'src'/name) for name in
                        ('analyze_lee_rejection.py', 'analyze_lee_scale_confirmation.py', 'analyze_lee_fixed.py', 'analyze_lee_extended.py')},
                    output_hashes={path.name: digest(path) for path in sorted(output.iterdir())
                                   if path.is_file() and path.name != 'manifest.json'})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps([row for row in ratios if row['metric'] == 'online_ns' and row['group'] == 'sphere'
                      and row['k0'] in (0, row['K']//2, row['K'])]), flush=True)


if __name__ == '__main__':
    main()
