"""Reproduce the current paper's results without overwriting reference data."""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import statistics
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
METHODS = ('dcc_ring', 'dcc_binary', 'sc_plain', 'enhanced', 'sc_support')
NAMES = ('Proj ring', 'Proj binary', 'PI unforced', 'Enhanced Proj', 'PI supported')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def load_records(folder):
    records = [json.loads(line) for line in
               (folder / 'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    return [row for row in records if row['method'] in METHODS]


def table(folder, output):
    records = load_records(folder)
    require(records and all(row['correct'] for row in records), 'Missing or failed records')
    groups = {method: [r for r in records if r['method'] == method] for method in METHODS}
    counts = {len(rows) for rows in groups.values()}
    require(len(counts) == 1 and 0 not in counts, 'Unbalanced method counts')
    baseline = statistics.mean(r['arithmetic_proxy'] for r in groups['dcc_ring'])
    rows = []
    for method, name in zip(METHODS, NAMES):
        data = groups[method]
        rows.append(dict(method=name,
            first_trials=statistics.mean(r.get('first_trials', 0) for r in data),
            recovery_trials=statistics.mean(r.get('second_trials', 0) + r.get('ring_trials', 0) for r in data),
            weight_tests=statistics.mean(r.get('candidates', 0) for r in data),
            proxy_ratio=statistics.mean(r['arithmetic_proxy'] for r in data) / baseline,
            time_ms=statistics.mean(r['online_ns'] for r in data) / 1e6))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'table.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Method | First trials | Recovery trials | Weight tests | Proxy ratio | Time (ms) |',
             '|---|---:|---:|---:|---:|---:|']
    tex = [r'\begin{tabular}{lrrrrr}', r'\toprule',
           r'Method & First trials & Recovery trials & Weight tests & Proxy ratio & Time (ms) \\',
           r'\midrule']
    for row in rows:
        values = [row['method']] + [f'{value:.3f}' for key, value in row.items() if key != 'method']
        lines.append('| ' + ' | '.join(values) + ' |')
        tex.append(' & '.join(values) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (output / 'table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return rows


def audit(folder, output, replay_all=False):
    from lee_revision import decode, make_code, prepare
    from lee_z4_dcc_comparison import binary_distance, lee, raw_syndrome
    descriptions = json.loads((folder / 'codes.json').read_text(encoding='utf-8'))
    records = load_records(folder)
    require(records, 'No records')
    codes, prepared = {}, {}
    for info in descriptions:
        code = make_code(info['ambient'], info['K'], info['k0'])
        require(code['h'] == info['h'], 'Check matrix mismatch')
        distance = binary_distance([[x % 2 for x in row] for row in info['ambient'][:info['K']]])
        require(distance == info['d_binary'] and distance > 2 * info['t'], 'Distance certificate mismatch')
        codes[info['code']] = code
        for method in METHODS:
            prepared[info['code'], method] = prepare(code, method)
    pairs, replays = {}, 0
    for row in records:
        code = codes[row['code']]
        require(row['correct'], 'Recorded decoder failure')
        require(lee(row['error']) == row['t'], 'Lee weight mismatch')
        require(raw_syndrome(code['h'], row['error']) == row['syndrome'], 'Syndrome mismatch')
        pair = pairs.setdefault((row['code'], row['case']), {})
        require(row['method'] not in pair, 'Duplicate paired record')
        pair[row['method']] = row
        if replay_all or row['case'] == 0:
            answer, counters = decode(prepared[row['code'], row['method']],
                                      row['syndrome'], row['t'], row['trial_seed'])
            require(answer == row['error'], 'Replay error mismatch')
            require(all(row.get(key, 0) == value for key, value in counters.items()), 'Replay counter mismatch')
            replays += 1
    for pair in pairs.values():
        require(set(pair) == set(METHODS), 'Missing paired method')
        for key in ('error', 'syndrome', 'trial_seed'):
            require(all(r[key] == pair['enhanced'][key] for r in pair.values()), 'Unmatched inputs')
        for key in ('first_trials', 'first_candidates', 'second_trials', 'second_candidates', 'candidates'):
            require(pair['enhanced'].get(key, 0) == pair['sc_support'].get(key, 0), 'Search counters differ')
    rows = table(folder, output)
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    require(hashlib.sha256((ROOT / 'src' / 'lee_revision.py').read_bytes()).hexdigest()
            == summary['source_sha256'], 'Benchmark source differs from recorded source')
    stored = {r['method']: r for r in summary['groups'] if r['n'] is None}
    for method, row in zip(METHODS, rows):
        require(abs(row['first_trials'] - stored[method]['first_trials']) < 1e-10, 'Stored trial mean mismatch')
        require(abs(row['time_ms'] - stored[method]['online_ns'] / 1e6) < 1e-10, 'Stored time mean mismatch')
        require(abs(row['proxy_ratio'] - stored[method]['arithmetic_proxy'] / stored['dcc_ring']['arithmetic_proxy']) < 1e-10,
                'Stored cost mean mismatch')
    report = dict(status='passed', codes=len(codes), paired_instances=len(pairs),
                  selected_counted_calls=len(records), replayed_calls=replays,
                  selected_timed_calls=len(records) * summary['settings']['timing_repeats'],
                  scope='Exact distances, recorded inputs/counters, paired equivalence and decoder replay; no historical timing rerun')
    write_json(output / 'audit.json', report)
    print(json.dumps(report, indent=2))


def statistics_report(folder, output):
    records = load_records(folder)
    require(records and all(row['correct'] for row in records), 'Missing or failed records')
    clusters, codes = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for row in records:
        clusters[row['cluster']][row['method']].append(row)
        codes[row['code']][row['method']].append(row)
    cluster_ids = sorted(clusters)
    require(len({len(group['dcc_ring']) for group in clusters.values()}) == 1,
            'Cluster-mean aggregation requires equal cluster sizes')
    rng = random.Random(711944)
    intervals, adverse = {}, []
    for metric in ('arithmetic_proxy', 'online_ns'):
        means = {key: {method: statistics.mean(row[metric] for row in group[method])
                       for method in ('dcc_ring', 'sc_support')}
                 for key, group in clusters.items()}
        ratios = []
        for _ in range(2000):
            chosen = [rng.choice(cluster_ids) for _ in cluster_ids]
            ratios.append(sum(means[key]['dcc_ring'] for key in chosen) /
                          sum(means[key]['sc_support'] for key in chosen))
        ratios.sort()
        intervals[metric] = [ratios[50], ratios[1950]]
        for name, group in codes.items():
            baseline = statistics.mean(row[metric] for row in group['dcc_ring'])
            supported = statistics.mean(row[metric] for row in group['sc_support'])
            if supported > baseline:
                adverse.append(dict(code=name, metric=metric, dcc_over_sc=baseline / supported))
    report = dict(clusters=len(clusters), resamples=2000, seed=711944,
                  ratio='Mean Proj ring cost / mean PI supported cost',
                  cluster_bootstrap_intervals=intervals, adverse_code_means=adverse)
    write_json(output / 'statistics.json', report)
    print(json.dumps(report, indent=2))


def equivalence(output):
    from lee_z4_dcc_comparison import code_suite, error_cases, raw_syndrome
    from z4_separation import decode
    pairs = events = 0
    for code, description in code_suite(mothers=1):
        data = error_cases(code.n, description['t'], 8, description['seed'] + 99, balanced=True)
        for index, (_, error) in enumerate(data):
            syndrome = raw_syndrome(code.parity(2), error)
            for slack in (0, 1, 2):
                left = decode(code, syndrome, description['t'], index + 918, 'super', slack)
                right = decode(code, syndrome, description['t'], index + 918, 'enhanced', slack)
                require(left == right and left[0] == error, 'Independent construction mismatch')
                pairs += 1
                events += sum(map(len, left[2]))
    require((pairs, events) == (672, 4511), 'Published equivalence counts changed')
    report = dict(paired_complete_cases=pairs, exact_trace_events_checked=events, failures=0)
    write_json(output / 'equivalence.json', report)
    print(json.dumps(report, indent=2))


def models(output):
    from z4_separation import exponents
    weight = .08
    mu, kappa0, kappa, theta = 1, .1, .3, .1
    lee = exponents(.1, .3, weight * (1 - weight / 2), weight**2 / 4)
    report = dict(lee=dict(sigma=lee['sigma'], alpha=lee['alpha'],
        A0=lee['super_first'], AP=lee['dcc_first_lower_bound'], B=lee['common_second'],
        scope='Mean-composition model under the paper assumptions; not a whole-sphere expected runtime'),
        rank=dict(mu=mu, kappa0=kappa0, kappa=kappa, theta=theta,
                  epsilon=mu * (1 - kappa) - 2 * theta * (mu + 1 - 2 * theta),
                  ep_exponent_per_log2p=theta * mu * kappa0,
                  projection_exponent_per_log2p=theta * mu * kappa,
                  scope='Specified support-trapping decoder and random-code distance region'))
    write_json(output / 'models.json', report)
    print(json.dumps(report, indent=2))


def manifest(write=False):
    path = ROOT / 'reference' / 'sha256.json'
    if write:
        files = sorted(p for folder in ('src', 'tests', 'reference')
                       for p in (ROOT / folder).rglob('*')
                       if p.is_file() and p != path and '__pycache__' not in p.parts)
        write_json(path, {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    entries = json.loads(path.read_text(encoding='utf-8'))
    for name, digest in entries.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, 'Hash mismatch: ' + name)
    print('SHA-256 verified:', len(entries), 'source, test and reference files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('table', 'audit', 'statistics', 'equivalence', 'models', 'tests', 'rank', 'theory', 'benchmark', 'manifest'))
    parser.add_argument('--input', type=Path, default=ROOT / 'reference' / 'lee')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--all', action='store_true', help='Replay all five-method records during audit')
    parser.add_argument('--smoke', action='store_true', help='Small benchmark: one mother, two errors, one timing repeat')
    parser.add_argument('--write-manifest', action='store_true', help='Maintainer: regenerate hashes after intentional changes')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT / 'reference'), 'Outputs must not overwrite reference data')
    output.mkdir(parents=True, exist_ok=True)
    if args.command == 'table':
        table(args.input, output)
    elif args.command == 'audit':
        audit(args.input, output, args.all)
    elif args.command == 'statistics':
        statistics_report(args.input, output)
    elif args.command == 'equivalence':
        equivalence(output)
    elif args.command == 'models':
        models(output)
    elif args.command == 'manifest':
        manifest(args.write_manifest)
    elif args.command == 'tests':
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif args.command == 'rank':
        from rank_schur_lift import small_end_to_end
        rows = [small_end_to_end(24092026 + j, valuation=v) for v in (0, 1) for j in range(20)]
        require(all(row['correct'] for row in rows), 'Rank recovery failure')
        write_json(output / 'rank_checks.json', rows)
        print('Passed 40 complete small rank decoding/lifting cases')
    elif args.command == 'theory':
        subprocess.run([sys.executable, str(ROOT / 'src' / 'verify_theory.py')], cwd=output, check=True)
    elif args.command == 'benchmark':
        destination = output / ('lee_smoke' if args.smoke else 'lee_full')
        command = [sys.executable, str(ROOT / 'src' / 'lee_revision.py'), '--output', str(destination),
                   '--mothers', '1' if args.smoke else '8', '--samples', '2' if args.smoke else '12',
                   '--timing-repeats', '1' if args.smoke else '3']
        if args.smoke:
            command.append('--smoke')
        subprocess.run(command, cwd=ROOT, check=True)
        table(destination, output)


if __name__ == '__main__':
    main()
