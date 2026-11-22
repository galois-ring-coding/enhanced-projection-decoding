"""Audit held-out Lee list-ISD measurements and export publication tables."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from lee_extended import composition_probability, distance_certificate
from lee_list_isd import Parameters
from lee_z4_dcc_comparison import lee, raw_syndrome
from run_lee_list_confirmation import digest, matrix_digest
from run_lee_list_retuned import (ROOT, allocation, call, choose, expected_records,
                                  grid, make_public, target_id)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['target'], row['variant'], row['mother'], row['method'], row['a']].append(row)
    points = []
    cells = sorted({(r['target'], r['variant'], r['mother'], r['method']) for r in rows})
    for ident, variant, mother, method in cells:
        n, K, t, k0, algorithm = ident.split(':')
        n, K, t, k0 = map(int, (n, K, t, k0))
        point = dict(target=ident, variant=variant, mother=mother, method=method,
                     n=n, K=K, t=t, k0=k0, algorithm=algorithm, all_complete=True)
        metrics = dict(online_ms=0., trials=0., first_ms=0., recovery_ms=0.,
                       metric_tests=0., base_list_entries=0., completion=0.)
        for a in range(t//2+1):
            block = grouped[ident, variant, mother, method, a]
            if len(block) != allocation(a)[0]*allocation(a)[1]:
                raise AssertionError('Incomplete composition allocation')
            for row in block:
                factor = float(composition_probability(n, t, a))/len(block)
                point['all_complete'] &= row['status'] == 'complete'
                metrics['completion'] += factor*(row['status'] == 'complete')
                for name, source in (('online_ms', 'online_ns'), ('first_ms', 'first_ns'), ('recovery_ms', 'recovery_ns')):
                    metrics[name] += factor*row['times'].get(source, 0)/1e6
                for name in ('trials', 'metric_tests', 'base_list_entries'):
                    metrics[name] += factor*sum(v for k, v in row['counters'].items() if k.endswith('_'+name))
        point.update(metrics)
        points.append(point)
    return points


def infer(points, repetitions=20000):
    result, protocols = [], []
    for n in sorted({p['n'] for p in points}):
        panel = [p for p in points if p['n'] == n]
        cells = sorted({(p['target'], p['variant']) for p in panel})
        mothers = sorted({p['mother'] for p in panel})
        if len(mothers) < 2:
            raise ValueError('At least two held-out mothers required for inference')
        lookup = {(p['target'], p['variant'], p['mother'], p['method']): p for p in panel}
        x = np.asarray([[lookup[ident, variant, mother, 'ccp_'+ident.split(':')[-1]]['online_ms']
                         for ident, variant in cells] for mother in mothers])
        y = np.asarray([[lookup[ident, variant, mother, 'lee_'+ident.split(':')[-1]]['online_ms']
                         for ident, variant in cells] for mother in mothers])
        theta = np.log(x.mean(axis=0)/y.mean(axis=0))
        rng = np.random.default_rng(2026110505+n)
        draws = rng.integers(0, len(mothers), (repetitions, len(mothers)))
        sampled = np.log(x[draws].mean(axis=1)/y[draws].mean(axis=1))
        se = sampled.std(axis=0, ddof=1)
        critical = float(np.quantile(np.max(np.abs((sampled-theta)/np.maximum(se, 1e-12)), axis=1), .95))
        low, high = np.exp(theta-critical*se), np.exp(theta+critical*se)
        plow, phigh = np.exp(np.quantile(sampled, (.025, .975), axis=0))
        for j, (ident, variant) in enumerate(cells):
            _, K, t, k0, algorithm = ident.split(':')
            complete = all(lookup[ident, variant, mother, side+'_'+algorithm]['all_complete']
                           for mother in mothers for side in ('ccp', 'lee'))
            assessment = 'budget limited' if not complete else 'CCP faster' if high[j] < 1 else 'native faster' if low[j] > 1 else 'unresolved'
            result.append(dict(target=ident, variant=variant, n=n, K=int(K), t=int(t),
                k0=int(k0), algorithm=algorithm, mothers=len(mothers), all_complete=bool(complete),
                ccp_ms=float(x[:, j].mean()), native_ms=float(y[:, j].mean()),
                ratio=float(np.exp(theta[j])), simultaneous95_low=float(low[j]),
                simultaneous95_high=float(high[j]), point95_low=float(plow[j]),
                point95_high=float(phigh[j]), assessment=assessment))
        protocols.append(dict(n=n, comparisons=len(cells), mothers=len(mothers),
            repetitions=repetitions, seed=2026110505+n, critical_value=critical,
            scope='All targets and both parameter-selection variants at this length.'))
    return result, protocols


def audit(folder, config, mothers, tuning, rows):
    if len(rows) != expected_records(config):
        raise AssertionError('Fixed held-out sample is incomplete')
    for name, fingerprint in config['source_hashes'].items():
        if digest(ROOT/'src'/name) != fingerprint:
            raise AssertionError('Frozen source changed: '+name)
    excluded = set()
    for name in ('lee_sweep_20261103/subtypes_main/codes.json', 'lee_list_confirmation_20261105/mothers.json'):
        excluded.update(matrix_digest(row['ambient']) for row in
                        json.loads((ROOT/'results'/name).read_text(encoding='utf-8')))
    fingerprints = set()
    for mother in mothers:
        fingerprint = matrix_digest(mother['ambient'])
        if fingerprint != mother['sha256'] or fingerprint in excluded or fingerprint in fingerprints:
            raise AssertionError('Duplicate or changed mother matrix')
        fingerprints.add(fingerprint)
        distance = distance_certificate(mother['ambient'][:mother['K']])
        maximum = max(t[2] for t in config['targets'] if t[0] == mother['n'])
        if distance != mother['d_binary'] or distance <= 2*maximum:
            raise AssertionError('Distance certificate mismatch')
    training = [json.loads(s) for s in (folder/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    expected_training = sum(2*len(grid(t[-1], t[2]))*config['train_mothers']*(t[2]//2+1)
                            for t in config['targets'])
    if len(training) != expected_training:
        raise AssertionError('Incomplete training grid')
    training_keys = set()
    training_blocks = defaultdict(list)
    for row in training:
        key = tuple(row[k] for k in ('target', 'method', 'candidate', 'mother', 'a'))
        if key in training_keys:
            raise AssertionError('Duplicate training key')
        training_keys.add(key)
        training_blocks[row['target'], row['method'], row['candidate']].append(row)
        if row['status'] == 'complete' and not row['correct']:
            raise AssertionError('Wrong training answer')
    for key, value in tuning.items():
        ident, method = key.rsplit(':', 1)
        algorithm = key.split(':')[-1].split('_')[-1]
        n, _, radius, _, _ = ident.split(':')
        for candidate, score in enumerate(value['candidates']):
            block = training_blocks[ident, method, candidate]
            if len(block) != config['train_mothers']*(int(radius)//2+1):
                raise AssertionError('Incomplete training candidate')
            loss = 0.
            failures = 0
            for row in block:
                ok = row['status'] == 'complete'
                failures += not ok
                cost = row['times']['online_ns'] if ok else max(
                    row['times']['online_ns'], config['tune_seconds']*1e9)*10
                loss += float(composition_probability(int(n), int(radius), row['a']))*cost/config['train_mothers']
                if row['parameters'] != score['parameters']:
                    raise AssertionError('Training candidate parameters changed')
            if failures != score['failed_calls'] or not np.isclose(loss, score['selection_loss_ns'], rtol=1e-12):
                raise AssertionError('Training score differs from raw measurements')
        for variant in ('primary', 'family'):
            if value[variant] != choose(value['candidates'], algorithm, variant == 'primary'):
                raise AssertionError('Parameters not selected on training scores')
    paired = defaultdict(dict)
    counts = Counter()
    for row in rows:
        side = row['method'].split('_')[0]
        key = tuple(row[k] for k in ('target', 'variant', 'mother', 'a', 'case', 'seed_index'))
        if side in paired[key]:
            raise AssertionError('Duplicate held-out record')
        paired[key][side] = row
        errors, seeds = allocation(row['a'])
        if not 0 <= row['case'] < errors or not 0 <= row['seed_index'] < seeds:
            raise AssertionError('Wrong held-out stratum')
        if row['parameters'] != tuning[row['target']+':'+row['method']][row['variant']]:
            raise AssertionError('Held-out parameters differ from frozen training selection')
        if lee(row['error']) != row['t'] or row['error'].count(2) != row['a']:
            raise AssertionError('Wrong held-out error')
        if row['status'] == 'complete' and not row['correct']:
            raise AssertionError('Wrong held-out answer')
        if len(row['timing_samples']) != config['repeats']:
            raise AssertionError('Wrong timing repetition count')
        counts[row['target'], row['variant'], row['method'], row['mother']] += 1
    for pair in paired.values():
        if set(pair) != {'ccp', 'lee'}:
            raise AssertionError('Unpaired input')
        if any(pair['ccp'][k] != pair['lee'][k] for k in ('error', 'syndrome', 'error_seed', 'search_seed')):
            raise AssertionError('Paired methods use different inputs')
    for target in config['targets']:
        expected = sum(e*s for e, s in map(allocation, range(target[2]//2+1)))
        for variant in ('primary', 'family'):
            for side in ('ccp', 'lee'):
                for mother in range(config['test_mothers']):
                    if counts[target_id(target), variant, side+'_'+target[-1], mother] != expected:
                        raise AssertionError('Incomplete held-out target')
    replays = 0
    for target in config['targets']:
        mother = next(m for m in mothers if m['n'] == target[0] and m['role'] == 'test' and m['mother'] == 0)
        public, _ = make_public(mother, target[3])
        for side in ('ccp', 'lee'):
            for variant in ('primary', 'family'):
                row = paired[target_id(target), variant, 0, 0, 0, 0][side]
                if raw_syndrome(public['h'], row['error']) != row['syndrome']:
                    raise AssertionError('Replayed syndrome mismatch')
                result = call(public, row['syndrome'], row['t'], row['search_seed'], row['method'],
                              row['parameters'], config)
                if result['status'] != 'complete' or result['answer'] != row['error']:
                    raise AssertionError('Held-out replay failed')
                replays += 1
    return dict(records=len(rows), training_calls=len(training),
        training_statuses=dict(Counter(r['status'] for r in training)),
        statuses=dict(Counter(r['status'] for r in rows)), timed_calls=sum(len(r['timing_samples']) for r in rows),
        correct_completed=sum(r['status'] == 'complete' and r['correct'] for r in rows),
        distance_certificates=len(mothers), independent_replays=replays, paired_inputs=len(paired),
        source_hashes_match=True, train_test_and_previous_matrices_disjoint=True)


def interval(row):
    return f"{row['ratio']:.3f} [{row['simultaneous95_low']:.3f}, {row['simultaneous95_high']:.3f}]"


def export_tex(rows, folder):
    lookup = {(r['target'], r['variant']): r for r in rows}
    lines = [r'\begin{table}[t]',
        r'\caption{Independent parameter selection at $K=3n/16$.',
        r'Times are exact-sphere weighted mean online milliseconds for',
        r'nonzero-list parameters. Ratios divide CCP time by native Lee time.',
        r'Brackets give approximate simultaneous 95\% intervals over all six',
        r'points and both selection policies at each length. The final column',
        r'also permits zero or partially zero base weights during training.}',
        r'\label{tab:leelistretuned}',
        r'\centering\small\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{rrrrrll}', r'\toprule',
        r'$n$ & $t$ & $k_0$ & CCP & Native & Nonzero-list ratio & Full-family ratio\\', r'\midrule']
    for algorithm, title in (('stern', 'Stern'), ('mmt', 'MMT')):
        lines.append(r'\multicolumn{7}{l}{'+title+r'}\\')
        for row in sorted((r for r in rows if r['algorithm'] == algorithm and r['variant'] == 'primary'),
                          key=lambda r: (r['n'], r['t'], r['k0'])):
            other = lookup[row['target'], 'family']
            lines.append(f"{row['n']} & {row['t']} & {row['k0']} & {row['ccp_ms']:.3f} & {row['native_ms']:.3f} & {interval(row)} & {interval(other)}"+r'\\')
        if algorithm == 'stern':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (folder/'list_retuned.tex').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', default='results/lee_list_retuned_20261105')
    args = parser.parse_args()
    folder = ROOT/args.folder
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    mothers = json.loads((folder/'mothers.json').read_text(encoding='utf-8'))
    tuning = json.loads((folder/'tuning.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    checks = audit(folder, config, mothers, tuning, rows)
    points = aggregate(rows)
    results, protocols = infer(points)
    output = folder/'analysis'
    output.mkdir(exist_ok=True)
    write_csv(output/'heldout_summary.csv', results)
    write_csv(output/'mother_means.csv', points)
    selected = [dict(target=key.rsplit(':', 1)[0], method=key.rsplit(':', 1)[1],
                     variant=variant, **value[variant], candidates=len(value['candidates']))
                for key, value in tuning.items() for variant in ('primary', 'family')]
    write_csv(output/'selected_parameters.csv', selected)
    (output/'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (output/'inference.json').write_text(json.dumps(protocols, indent=2), encoding='utf-8')
    export_tex(results, output)
    print(json.dumps(dict(verification=checks, results=results), indent=2))


if __name__ == '__main__':
    main()
