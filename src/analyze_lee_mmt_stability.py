"""Audit replicated MMT tuning, diagnose the previous reversal, and compare costs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_lee_list_retuned import aggregate as original_aggregate, interval, write_csv
from lee_extended import composition_probability, distance_certificate
from lee_z4_dcc_comparison import lee, raw_syndrome
from run_lee_list_confirmation import digest, matrix_digest
from run_lee_list_retuned import allocation, call, choose, expected_records, make_public, target_id
from run_lee_mmt_stability import (BASE, ROOT, expected_training, score, shortlist, training_allocation)


def trial_count(counters):
    return sum(counters.get(phase+'_trials', 0) for phase in ('first', 'recovery', 'native'))


def aggregate(rows):
    points = original_aggregate(rows)
    counts = defaultdict(float)
    for r in rows:
        n, _, t, _, _ = r['target'].split(':')
        e, s = allocation(r['a'])
        counts[r['target'], r['variant'], r['mother'], r['method']] += (
            float(composition_probability(int(n), int(t), r['a']))*trial_count(r['counters'])/(e*s))
    for p in points:
        p['trials'] = counts[p['target'], p['variant'], p['mother'], p['method']]
    return points


def previous_diagnostics(output):
    tuning = json.loads((BASE/'tuning.json').read_text(encoding='utf-8'))
    ident = '96:18:12:14:mmt'
    data = [json.loads(s) for s in (BASE/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    data = [r for r in data if r['target'] == ident]
    rows, summary = [], {}
    for side in ('ccp', 'lee'):
        method = side+'_mmt'
        candidates = [r for r in data if r['method'] == method]
        selected = tuning[ident+':'+method]
        summary[side] = dict(all_candidate_budget_reasons=dict(Counter(
            r['reason'] for r in candidates if r['status'] != 'complete')))
        means = {}
        for variant in ('primary', 'family'):
            block = [r for r in candidates if r['parameters'] == selected[variant]]
            means[variant] = []
            for mother in sorted({r['mother'] for r in block}):
                means[variant].append(sum(float(composition_probability(96, 12, r['a']))*
                    r['times']['online_ns']/1e6 for r in block if r['mother'] == mother))
            summary[side][variant] = dict(parameters=selected[variant], calls=len(block),
                budget_calls=sum(r['status'] != 'complete' for r in block),
                sphere_weighted_ms=float(np.mean(means[variant])),
                maximum_ms=max(r['times']['online_ns'] for r in block)/1e6)
            for a in range(7):
                comp = [r for r in block if r['a'] == a]
                rows.append(dict(method=method, variant=variant, a=a,
                    sphere_mass=float(composition_probability(96, 12, a)), calls=len(comp),
                    mean_ms=float(np.mean([r['times']['online_ns']/1e6 for r in comp])),
                    budget_calls=sum(r['status'] != 'complete' for r in comp)))
        delta = np.asarray(means['family'])-np.asarray(means['primary'])
        rng = np.random.default_rng(2026110702)
        sampled = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(axis=1)
        summary[side]['training_family_minus_primary_ms'] = dict(
            mean=float(delta.mean()), pointwise95=list(map(float, np.quantile(sampled, (.025, .975)))),
            interpretation='Training-set selection diagnostic; not held-out evidence.')
    write_csv(output/'previous_reversal_compositions.csv', rows)
    (output/'previous_reversal_diagnostic.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def infer(points, repetitions=20000):
    cells = sorted({(p['target'], p['variant']) for p in points})
    lookup = {(p['target'], p['variant'], p['mother'], p['method']): p for p in points}
    theta, sampled = np.empty(len(cells)), np.empty((repetitions, len(cells)))
    costs = []
    for n in sorted({p['n'] for p in points}):
        mothers = sorted({p['mother'] for p in points if p['n'] == n})
        if len(mothers) < 2:
            raise ValueError('At least two test matrices per length required')
        rng = np.random.default_rng(2026110703+n)
        draws = rng.integers(0, len(mothers), (repetitions, len(mothers)))
        for j, (ident, variant) in enumerate(cells):
            if int(ident.split(':')[0]) != n:
                continue
            x = np.asarray([lookup[ident, variant, m, 'ccp_mmt']['online_ms'] for m in mothers])
            y = np.asarray([lookup[ident, variant, m, 'lee_mmt']['online_ms'] for m in mothers])
            theta[j] = np.log(x.mean()/y.mean())
            sampled[:, j] = np.log(x[draws].mean(axis=1)/y[draws].mean(axis=1))
            costs.append((j, x.mean(), y.mean(), mothers))
    se = sampled.std(axis=0, ddof=1)
    critical = float(np.quantile(np.max(np.abs((sampled-theta)/np.maximum(se, 1e-12)), axis=1), .95))
    low, high = np.exp(theta-critical*se), np.exp(theta+critical*se)
    point_low, point_high = np.exp(np.quantile(sampled, (.025, .975), axis=0))
    rows = []
    for j, x, y, mothers in sorted(costs):
        ident, variant = cells[j]
        n, K, t, k0, algorithm = ident.split(':')
        complete = all(lookup[ident, variant, m, side+'_mmt']['all_complete']
                       for m in mothers for side in ('ccp', 'lee'))
        rows.append(dict(target=ident, variant=variant, n=int(n), K=int(K), t=int(t), k0=int(k0),
            algorithm=algorithm, mothers=len(mothers), all_complete=bool(complete),
            ccp_ms=float(x), native_ms=float(y), ratio=float(np.exp(theta[j])),
            simultaneous95_low=float(low[j]), simultaneous95_high=float(high[j]),
            point95_low=float(point_low[j]), point95_high=float(point_high[j]),
            assessment='budget limited' if not complete else 'CCP faster' if high[j] < 1
                       else 'native faster' if low[j] > 1 else 'unresolved'))
    return rows, dict(repetitions=repetitions, comparisons=len(cells), critical_value=critical,
        scope='All six targets and both policies jointly; mother resamples independent across lengths.')


def training_diagnostics(training, tuning, config, output):
    blocks = defaultdict(list)
    for r in training:
        blocks[r['target'], r['method'], r['candidate']].append(r)
    rows, stability = [], []
    for key, value in tuning.items():
        ident, method = key.rsplit(':', 1)
        n, _, t, _, _ = ident.split(':')
        n, t = int(n), int(t)
        for candidate, stored in enumerate(value['candidates']):
            block = blocks[ident, method, candidate]
            means = [score([r for r in block if r['mother'] == m], n, t, 1,
                           config['seconds'])['selection_loss_ns']/1e6
                     for m in range(config['train_mothers'])]
            rows.append(dict(target=ident, method=method, candidate=candidate, **stored['parameters'],
                failures=stored['failed_calls'], loss_ms=stored['selection_loss_ns']/1e6,
                matrix_sd_ms=float(np.std(means, ddof=1)) if len(means)>1 else 0.,
                selected_primary=stored['parameters'] == value['primary'],
                selected_family=stored['parameters'] == value['family'],
                budget_reasons=json.dumps(dict(Counter(r['reason'] for r in block
                    if r['status'] != 'complete')), sort_keys=True)))
        if config['train_mothers'] < 2:
            continue
        for heldout in range(config['train_mothers']):
            candidates = []
            for candidate, stored in enumerate(value['candidates']):
                block = [r for r in blocks[ident, method, candidate] if r['mother'] != heldout]
                candidates.append(dict(parameters=stored['parameters'], **score(block, n, t,
                    config['train_mothers']-1, config['seconds'])))
            for variant in ('primary', 'family'):
                p = choose(candidates, 'mmt', variant == 'primary')
                stability.append(dict(target=ident, method=method, omitted_matrix=heldout,
                    variant=variant, **p, same_as_full_training=p == value[variant]))
    write_csv(output/'training_candidates.csv', rows)
    if stability:
        write_csv(output/'training_leave_one_matrix_out.csv', stability)
    old = json.loads((BASE/'tuning.json').read_text(encoding='utf-8'))
    reversal = []
    ident = '96:18:12:14:mmt'
    for method in ('ccp_mmt', 'lee_mmt'):
        for variant in ('primary', 'family'):
            params = old[ident+':'+method][variant]
            block = [r for r in training if r['target'] == ident and
                     r['method'] == method and r['parameters'] == params]
            if not block:
                continue
            for mother in range(config['train_mothers']):
                local = [r for r in block if r['mother'] == mother]
                measured = score(local, 96, 12, 1, config['seconds'])
                reversal.append(dict(target=ident, method=method, previous_variant=variant,
                    mother=mother, **params, failures=measured['failed_calls'],
                    loss_ms=measured['selection_loss_ns']/1e6))
    if reversal:
        write_csv(output/'previous_parameters_new_training.csv', reversal)


def composition_diagnostics(rows, output):
    blocks = defaultdict(list)
    for r in rows:
        blocks[r['target'], r['variant'], r['method'], r['a']].append(r)
    result = []
    for (ident, variant, method, a), block in sorted(blocks.items()):
        n, _, t, _, _ = ident.split(':')
        result.append(dict(target=ident, variant=variant, method=method, a=a,
            sphere_mass=float(composition_probability(int(n), int(t), a)), calls=len(block),
            complete=sum(r['status']=='complete' for r in block),
            online_ms=float(np.mean([r['times']['online_ns']/1e6 for r in block])),
            first_ms=float(np.mean([r['times'].get('first_ns',0)/1e6 for r in block])),
            recovery_ms=float(np.mean([r['times'].get('recovery_ns',0)/1e6 for r in block])),
            trials=float(np.mean([trial_count(r['counters']) for r in block])),
            metric_tests=float(np.mean([sum(v for k,v in r['counters'].items() if k.endswith('_metric_tests'))
                                       for r in block]))))
    write_csv(output/'test_compositions.csv', result)


def audit(folder, config, mothers, tuning, training, rows):
    for name, value in config['source_hashes'].items():
        if digest(ROOT/'src'/name) != value:
            raise AssertionError('Frozen source changed: '+name)
    if config['prior_training_hash'] != digest(BASE/'training.jsonl') or config['prior_tuning_hash'] != digest(BASE/'tuning.json'):
        raise AssertionError('Prior training archive changed')
    old = json.loads((BASE/'tuning.json').read_text(encoding='utf-8'))
    if not config['smoke']:
        for t in config['targets']:
            if config['grids'][target_id(t)] != shortlist(old, t):
                raise AssertionError('Shared shortlist differs from predeclared rule')
    excluded = set()
    for name in ('lee_sweep_20261103/subtypes_main/codes.json',
                 'lee_list_confirmation_20261105/mothers.json', 'lee_list_retuned_20261105/mothers.json'):
        excluded.update(matrix_digest(r['ambient']) for r in json.loads((ROOT/'results'/name).read_text(encoding='utf-8')))
    seen = set()
    for mother in mothers:
        fingerprint = matrix_digest(mother['ambient'])
        if fingerprint != mother['sha256'] or fingerprint in seen or fingerprint in excluded:
            raise AssertionError('Repeated or changed mother matrix')
        seen.add(fingerprint)
        d = distance_certificate(mother['ambient'][:mother['K']])
        maximum = max(t[2] for t in config['targets'] if t[0] == mother['n'])
        if d != mother['d_binary'] or d <= 2*maximum:
            raise AssertionError('Distance certificate mismatch')
    if len(training) != expected_training(config) or len(rows) != expected_records(config):
        raise AssertionError('Incomplete training or test panel')
    blocks, keys, train_pairs = defaultdict(list), set(), defaultdict(dict)
    for r in training:
        key = tuple(r[k] for k in ('target', 'method', 'candidate', 'mother', 'a', 'case', 'seed_index'))
        if key in keys:
            raise AssertionError('Duplicate training record')
        keys.add(key)
        blocks[r['target'], r['method'], r['candidate']].append(r)
        train_pairs[tuple(r[k] for k in ('target', 'candidate', 'mother', 'a', 'case', 'seed_index'))][r['method']] = r
    for pair in train_pairs.values():
        if set(pair) != {'ccp_mmt', 'lee_mmt'} or any(pair['ccp_mmt'][k] != pair['lee_mmt'][k]
                for k in ('error', 'syndrome', 'parameters', 'search_seed')):
            raise AssertionError('Training calls are not paired')
    for key, value in tuning.items():
        ident, method = key.rsplit(':', 1)
        n, _, radius, _, _ = ident.split(':')
        for candidate, stored in enumerate(value['candidates']):
            block = blocks[ident, method, candidate]
            expected = config['train_mothers']*4*(int(radius)//2+1)
            if len(block) != expected or any(r['parameters'] != stored['parameters'] for r in block):
                raise AssertionError('Incomplete training candidate')
            cells = {(r['mother'], r['a'], r['case'], r['seed_index']) for r in block}
            expected_cells = {(m, a, e, s) for m in range(config['train_mothers'])
                for a in range(int(radius)//2+1) for e in range(training_allocation(a)[0])
                for s in range(training_allocation(a)[1])}
            if cells != expected_cells:
                raise AssertionError('Training composition allocation mismatch')
            reconstructed = score(block, int(n), int(radius), config['train_mothers'], config['seconds'])
            if stored['failed_calls'] != reconstructed['failed_calls'] or not np.isclose(
                    stored['selection_loss_ns'], reconstructed['selection_loss_ns'], rtol=1e-12):
                raise AssertionError('Training losses mismatch')
        for variant in ('primary', 'family'):
            if value[variant] != choose(value['candidates'], 'mmt', variant == 'primary'):
                raise AssertionError('Selection differs from training rule')
    pairs, counts, testkeys = defaultdict(dict), Counter(), set()
    for r in rows:
        key = tuple(r[k] for k in ('target', 'variant', 'mother', 'a', 'case', 'seed_index'))
        fullkey = key+(r['method'],)
        if fullkey in testkeys:
            raise AssertionError('Duplicate held-out record')
        testkeys.add(fullkey)
        pairs[key][r['method']] = r
        counts[r['target'], r['variant'], r['mother'], r['method']] += 1
        if r['parameters'] != tuning[r['target']+':'+r['method']][r['variant']]:
            raise AssertionError('Test parameters changed')
        if lee(r['error']) != r['t'] or r['error'].count(2) != r['a']:
            raise AssertionError('Wrong error composition')
        if len(r['timing_samples']) != config['repeats']:
            raise AssertionError('Missing timing samples')
        e, s = allocation(r['a'])
        if not 0 <= r['case'] < e or not 0 <= r['seed_index'] < s:
            raise AssertionError('Held-out composition allocation mismatch')
    for r in training+rows:
        if r['status'] == 'complete' and not r['correct']:
            raise AssertionError('Wrong completed answer')
    for pair in pairs.values():
        if set(pair) != {'ccp_mmt', 'lee_mmt'} or any(pair['ccp_mmt'][k] != pair['lee_mmt'][k]
                for k in ('error', 'syndrome', 'error_seed', 'search_seed')):
            raise AssertionError('Held-out calls are not paired')
    replays = 0
    for t in config['targets']:
        ident = target_id(t)
        count = sum(e*s for e, s in map(allocation, range(t[2]//2+1)))
        for m in range(config['test_mothers']):
            for variant in ('primary', 'family'):
                for method in ('ccp_mmt', 'lee_mmt'):
                    if counts[ident, variant, m, method] != count:
                        raise AssertionError('Incomplete test stratum')
        mother = next(m for m in mothers if m['n'] == t[0] and m['role'] == 'test' and m['mother'] == 0)
        public, _ = make_public(mother, t[3])
        for variant in ('primary', 'family'):
            for method in ('ccp_mmt', 'lee_mmt'):
                r = pairs[ident, variant, 0, 0, 0, 0][method]
                if raw_syndrome(public['h'], r['error']) != r['syndrome']:
                    raise AssertionError('Replayed syndrome mismatch')
                result = call(public, r['syndrome'], r['t'], r['search_seed'], method, r['parameters'], config)
                if result['status'] != 'complete' or result['answer'] != r['error']:
                    raise AssertionError('Independent replay failed')
                replays += 1
    return dict(records=len(rows), training_records=len(training), statuses=dict(Counter(r['status'] for r in rows)),
        training_statuses=dict(Counter(r['status'] for r in training)),
        timed_calls=sum(len(r['timing_samples']) for r in rows),
        correct_completed=sum(r['status'] == 'complete' and r['correct'] for r in rows),
        distance_certificates=len(mothers), independent_replays=replays,
        fresh_train_test_matrices=True, source_hashes_match=True)


def export_tex(rows, output):
    lookup = {(r['target'], r['variant']): r for r in rows}
    lines = [r'\begin{table}[t]',
        r'\caption{MMT confirmation after replicated training on eight fresh matrices',
        r'and testing on 32 further matrices per length. Here $K=3n/16$. Times',
        r'are exact-sphere weighted mean online milliseconds for full-family',
        r'choices. Ratios divide CCP by native Lee-LMMT time. Brackets are',
        r'approximate simultaneous 95\% intervals over all six points and both',
        r'parameter policies. The final column restricts training choices to',
        r'positive base weights; the public search schedule applies in both cases.}',
        r'\label{tab:leemmtstability}', r'\centering\small\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{rrrrrll}', r'\toprule',
        r'$n$ & $t$ & $k_0$ & CCP & Native & Full-family ratio & Positive-weight ratio\\', r'\midrule']
    for r in sorted((r for r in rows if r['variant'] == 'family'), key=lambda r:(r['n'],r['t'],r['k0'])):
        lines.append(f"{r['n']} & {r['t']} & {r['k0']} & {r['ccp_ms']:.3f} & {r['native_ms']:.3f} & {interval(r)} & {interval(lookup[r['target'], 'primary'])}"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (output/'mmt_stability.tex').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', default='results/lee_mmt_stability_20261107')
    args = parser.parse_args()
    folder = ROOT/args.folder
    output = folder/'analysis'
    output.mkdir(exist_ok=True)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    mothers = json.loads((folder/'mothers.json').read_text(encoding='utf-8'))
    tuning = json.loads((folder/'tuning.json').read_text(encoding='utf-8'))
    training = [json.loads(s) for s in (folder/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [json.loads(s) for s in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    checks = audit(folder, config, mothers, tuning, training, rows)
    points = aggregate(rows)
    result, protocol = infer(points)
    diagnostic = previous_diagnostics(output)
    training_diagnostics(training, tuning, config, output)
    composition_diagnostics(rows, output)
    write_csv(output/'heldout_summary.csv', result)
    write_csv(output/'mother_means.csv', points)
    write_csv(output/'selected_parameters.csv', [dict(target=k.rsplit(':',1)[0], method=k.rsplit(':',1)[1],
        variant=v, **value[v]) for k,value in tuning.items() for v in ('primary','family')])
    (output/'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (output/'inference.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    export_tex(result, output)
    print(json.dumps(dict(verification=checks, results=result, previous_reversal=diagnostic), indent=2))


if __name__ == '__main__':
    main()
