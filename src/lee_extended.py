"""Certified Z4 Lee benchmarks; historical source/data remain unchanged.

Run from the repository root. Every decoder receives only H, syndrome, t,
and public preprocessing. Composition labels are used only by the runner.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np

from lee_revision import eliminate, make_code, selected_solutions
from lee_z4_dcc_comparison import Arithmetic, lee, raw_syndrome
from ring_constraints import smith_check

ROOT = Path(__file__).resolve().parents[1]
MAIN = ('proj_ring_t', 'proj_binary_t', 'ep_reject', 'proj_binary_all', 'ep_all')
RADIUS = ('proj_ring_t', 'proj_ring_mean_ceil', 'proj_ring_mean_floor',
          'proj_ring_exact_ceil', 'proj_ring_fallback', 'ep_reject')
ABLATION = ('ep_full_forced', 'ep_full_unforced', 'ep_full_forced_shortcut',
            'ep_full_unforced_shortcut', 'ep_reject')
PROFILES = {
    'smoke': ((16, 4, 2),),
    'main': ((40, 8, 4), (64, 12, 6), (96, 16, 8), (128, 20, 10)),
    'large': ((160, 20, 12), (192, 20, 16), (256, 20, 20)),
    'fixed_rate': ((40, 12, 3), (60, 18, 4), (80, 24, 6)),
    'control': ((40, 8, 4), (64, 12, 6)),
    'radius': ((64, 6, 12), (96, 8, 16), (128, 12, 20)),
    'ablation': ((40, 8, 4), (64, 12, 6)),
}


class BudgetExceeded(Exception):
    pass


def distance_certificate(g):
    """Enumerate all nonzero binary codewords with O(K) storage."""
    packed = [sum((x % 2) << j for j, x in enumerate(row)) for row in g]
    word, best = 0, len(g[0]) + 1
    for index in range(1, 1 << len(g)):
        word ^= packed[(index & -index).bit_length()-1]
        best = min(best, word.bit_count())
    return best


def sample_error(n, t, rng, a=None):
    if not 0 <= t <= 2*n:
        raise ValueError('Invalid Lee weight')
    if a is None:
        bits = set(rng.sample(range(2*n), t))
        gray_inverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        return [gray_inverse[(int(2*j in bits), int(2*j+1 in bits))] for j in range(n)]
    s = t-2*a
    if min(a, s) < 0 or s+a > n:
        raise ValueError('Invalid composition')
    positions = rng.sample(range(n), s+a)
    result = [0]*n
    for j in positions[:s]:
        result[j] = rng.choice((1, 3))
    for j in positions[s:]:
        result[j] = 2
    return result


def composition_probability(n, t, a):
    s = t-2*a
    if a < 0 or s < 0 or s+a > n:
        return Fraction(0)
    return Fraction(math.comb(n, s)*math.comb(n-s, a)*2**s, math.comb(2*n, t))


def initial_radius(n, t, policy):
    if policy == 't':
        return t
    average = (Fraction(t*(2*n-t), 2*n-1) if policy == 'exact_ceil'
               else Fraction(t*(2*n-t), 2*n))
    return math.floor(average) if policy == 'mean_floor' else math.ceil(average)


def prepare(h):
    """Derive both complete and local checks from public H, not its generator."""
    normal = smith_check(h, 2, 2)
    values = normal['values']
    units = values.count(0)
    if values != [0]*units+[1]*(len(values)-units):
        raise ValueError('Unexpected Z4 Smith invariants')
    normalized = [[(2**v*x) % 4 for x in row]
                  for v, row in zip(values, normal['inverse_v'])]
    n = len(h[0])
    return dict(h=h, u=normal['u'], normalized=normalized, units=units,
                h1=[[x % 2 for x in row] for row in normalized[:units]],
                h0=[[x % 2 for x in row] for row in normal['inverse_v'][:len(values)]],
                n=n, K=n-units, k0=n-len(values),
                matrix_bits=2*sum(map(len, h))+2*sum(map(len, normal['u']))+
                2*sum(map(len, normalized))+n*(units+len(values)))


def binary_search(h, target, radius, rng, op, trials, candidates,
                  known=(), forced=False, enumerate_singular=False):
    n, size = len(h[0]), len(h)
    if not any(target):
        op.event('zero_syndrome_shortcuts')
        return [0]*n
    known = set(known)
    prefix = sorted(known) if forced else []
    rest = [j for j in range(n) if j not in prefix]
    emitted = 0
    for _ in range(trials):
        op.event('trials')
        chosen = prefix+rng.sample(rest, size-len(prefix))
        if enumerate_singular:
            stream = selected_solutions(h, target, chosen, op)
        else:
            rows = eliminate(h, target, chosen, op)
            if rows is None:
                op.event('singular_trials')
                continue
            answer = [0]*n
            for j, row in zip(chosen, rows):
                answer[j] = row[-1]
            stream = (answer,)
        for answer in stream:
            if emitted >= candidates:
                raise BudgetExceeded('candidate_budget')
            emitted += 1
            op.event('candidates')
            if sum(x for j, x in enumerate(answer) if j not in known) <= radius:
                return answer
    raise BudgetExceeded('trial_budget')


def recover_binary(h, target, known, radius, rng, op, trials, candidates,
                   enumerate_singular=False):
    n, s = len(h[0]), len(known)
    rest = [j for j in range(n) if j not in known]
    rows = eliminate(h, target, known, op)
    if rows is None:
        raise ValueError('Known support columns dependent')
    reduced = [[row[j] for j in rest] for row in rows[s:]]
    beta = [row[-1] for row in rows[s:]]
    if radius == 0:
        if any(beta):
            raise ValueError('Inconsistent direct recovery')
        op.event('zero_budget_shortcuts')
        outside = [0]*len(rest)
    else:
        outside = binary_search(reduced, beta, radius, rng, op, trials, candidates,
                                enumerate_singular=enumerate_singular)
    answer = [0]*n
    for j, x in zip(rest, outside):
        answer[j] = x
    for i, j in enumerate(known):
        answer[j] = op.add(rows[i][-1], -op.dot([rows[i][k] for k in rest], outside))
    return answer


def decode(prepared, syndrome, t, seed, method, trials=4096, candidates=100000, track=True):
    first, second, ring, verify = [Arithmetic(bits, track) for bits in (1, 1, 2, 2)]
    ops = dict(first=first, second=second, ring=ring, verify=verify)
    times, status, answer, reason = {}, 'complete', None, None
    stage = 'first'
    start = time.perf_counter_ns()
    is_ep = method.startswith('ep_')
    enumeration = method.endswith('_all')
    policy = 't'
    for name in ('mean_ceil', 'mean_floor', 'exact_ceil', 'fallback'):
        if method.endswith('_'+name):
            policy = name
    radius = initial_radius(prepared['n'], t, policy)
    rng = random.Random(seed*17)
    try:
        transformed = [ring.dot(row, syndrome) for row in prepared['u']]
        if any(transformed[len(prepared['normalized']):]):
            raise ValueError('Inconsistent zero row')
        transformed = transformed[:len(prepared['normalized'])]
        units = prepared['units']
        if any(x % 2 for x in transformed[units:]):
            raise ValueError('Inconsistent torsion syndrome')
        target = [x % 2 for x in transformed[:units]]
        if is_ep:
            target += [x//2 for x in transformed[units:]]
        h0 = prepared['h0'] if is_ep else prepared['h1']
        if policy == 'fallback' and radius < t:
            early = max(1, trials//4)
            try:
                low = binary_search(h0, target, radius, rng, first, early, candidates,
                                    enumerate_singular=enumeration)
            except BudgetExceeded:
                first.event('radius_fallbacks')
                low = binary_search(h0, target, t, rng, first, trials-early, candidates,
                                    enumerate_singular=enumeration)
        else:
            low = binary_search(h0, target, radius, rng, first, trials, candidates,
                                enumerate_singular=enumeration)
        times['first_ns'] = time.perf_counter_ns()-start
        stage, start = 'recovery', time.perf_counter_ns()
        known = [j for j, x in enumerate(low) if x]
        budget = (t-len(known))//2
        if budget < 0:
            raise ValueError('Invalid recovered digit')
        recovery_rng = random.Random(seed*17+1)
        if method.startswith('proj_ring'):
            rest = [j for j in range(prepared['n']) if j not in known]
            for _ in range(trials):
                ring.event('trials')
                chosen = known+recovery_rng.sample(rest, units-len(known))
                rows = eliminate(prepared['normalized'], transformed, chosen, ring)
                if rows is None:
                    ring.event('singular_trials')
                    continue
                if any(row[-1] for row in rows[units:]):
                    ring.event('inconsistent_trials')
                    continue
                ring.event('candidates')
                candidate = [0]*prepared['n']
                for j, row in zip(chosen, rows):
                    candidate[j] = row[-1]
                if lee(candidate) <= t:
                    answer = candidate
                    break
            else:
                raise BudgetExceeded('trial_budget')
        else:
            residual = [ring.add(s, -ring.dot(row, low))
                        for row, s in zip(prepared['normalized'], transformed)]
            if any(x % 2 for x in residual) or any(residual[units:]):
                raise ValueError('Incompatible low digit')
            target1 = [x//2 for x in residual[:units]]
            full = method.startswith('ep_full_')
            shortcut = method.endswith('_shortcut')
            if full and not (shortcut and budget == 0):
                high = binary_search(prepared['h1'], target1, budget, recovery_rng, second,
                                     trials, candidates, known=known,
                                     forced='_unforced' not in method)
            else:
                high = recover_binary(prepared['h1'], target1, known, budget, recovery_rng,
                                      second, trials, candidates, enumeration)
            answer = [ring.add(x, ring.mul(2, y)) for x, y in zip(low, high)]
        times['recovery_ns'] = time.perf_counter_ns()-start
        stage, start = 'verify', time.perf_counter_ns()
        if [verify.dot(row, answer) for row in prepared['h']] != syndrome or lee(answer) > t:
            raise ValueError('Final verification failed')
        times['verify_ns'] = time.perf_counter_ns()-start
    except BudgetExceeded as exc:
        times[stage+'_ns'] = time.perf_counter_ns()-start
        status, reason = 'censored', stage+':'+str(exc)
    counts = Counter()
    for label, op in ops.items():
        counts.update(op.stats)
        counts.update({label+'_'+key: value for key, value in op.stats.items()})
    return dict(answer=answer if status == 'complete' else None, status=status,
                reason=reason, initial_radius=radius, counters=dict(counts), times=times)


def generate_codes(profile, mothers, seed, folder):
    path = folder/'codes.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    result = []
    for n, k, t in PROFILES[profile]:
        for mother in range(mothers):
            code_seed = seed+n*100003+k*997+mother
            rng = random.Random(code_seed)
            for attempt in range(10000):
                ambient = [[rng.randrange(4) for _ in range(n)] for _ in range(n)]
                try:
                    base = make_code(ambient, k, 1)
                except ValueError:
                    continue
                distance = distance_certificate(ambient[:k])
                if distance > 2*t:
                    break
            else:
                raise RuntimeError('Code generation budget exhausted')
            subtypes = {0, k} if profile == 'control' else {1, k//2, k-1}
            if profile == 'fixed_rate':
                subtypes.add(n//10)
            for k0 in sorted(subtypes):
                h = base['dual'][k:]+[[2*x % 4 for x in row] for row in base['dual'][k0:k]]
                result.append(dict(n=n, K=k, k0=k0, t=t, h=h, ambient=ambient,
                                   cluster=f'{profile}_n{n}_m{mother}',
                                   code=f'{profile}_n{n}_m{mother}_k0{k0}',
                                   seed=code_seed, d_binary=distance, attempts=attempt+1))
            print('certified', profile, n, mother, distance, flush=True)
    path.write_text(json.dumps(result), encoding='utf-8')
    return result


def case_list(info, samples, all_compositions=False):
    rng = random.Random(info['seed']+881)
    result = [('uniform', j, sample_error(info['n'], info['t'], rng)) for j in range(samples)]
    groups = (range(info['t']//2+1) if all_compositions
              else sorted({0, 1, 2, info['t']//2}))
    for a in groups:
        if not composition_probability(info['n'], info['t'], a):
            continue
        result += [(f'a{a}', j, sample_error(info['n'], info['t'], rng, a)) for j in range(samples)]
    return result


def summarize(folder):
    rows = [json.loads(line) for line in (folder/'instances.jsonl').read_text().splitlines()]
    buckets = defaultdict(list)
    for row in rows:
        population = 'uniform' if row['population'] == 'uniform' else 'stratified'
        for n, a, k0 in ((None, None, None), (row['n'], None, None),
                         (None, row['a'], None), (row['n'], row['a'], row['k0'])):
            buckets[population, n, a, k0, row['method']].append(row)
    groups = []
    for (pop, n, a, k0, method), data in buckets.items():
        item = dict(population=pop, n=n, a=a, k0=k0, method=method, calls=len(data),
                    completed=sum(r['status'] == 'complete' for r in data),
                    clusters=len({r['cluster'] for r in data}))
        for metric in ('online_ns', 'first_ns', 'recovery_ns', 'cold_ns', 'batch100_ns',
                       'arithmetic_proxy', 'first_trials', 'recovery_trials',
                       'first_singular_trials', 'second_singular_trials',
                       'first_inconsistent_trials', 'second_inconsistent_trials',
                       'first_candidates', 'second_candidates'):
            item[metric] = statistics.mean(r.get(metric, 0) for r in data)
        item['cost_scope'] = 'complete' if item['completed'] == len(data) else 'budget_limited'
        groups.append(item)
    # Bootstrap complete pairs within each population; retain all failures in groups.
    intervals = []
    for population in ('uniform', 'stratified'):
        for a in (None, 0, 1, 2):
            subset = [r for r in rows if (r['population'] == 'uniform') == (population == 'uniform')
                      and (a is None or r['a'] == a)]
            for baseline in ('proj_ring_t', 'proj_binary_t', 'proj_binary_all'):
                ep = 'ep_all' if baseline.endswith('_all') else 'ep_reject'
                data = [r for r in subset if r['method'] in (baseline, ep)]
                if not data or any(r['status'] != 'complete' for r in data):
                    continue
                by_cluster = defaultdict(lambda: defaultdict(list))
                for r in data:
                    by_cluster[r['cluster']][r['method']].append(r)
                if any(set(v) != {baseline, ep} or len(v[baseline]) != len(v[ep])
                       for v in by_cluster.values()):
                    continue
                keys = sorted(by_cluster)
                if len(keys) < 2:
                    continue
                for metric in ('online_ns', 'arithmetic_proxy'):
                    totals = {key: {m: sum(r[metric] for r in by_cluster[key][m])
                                    for m in (baseline, ep)} for key in keys}
                    rng = random.Random(20261102)
                    ratios = []
                    for _ in range(2000):
                        draw = rng.choices(keys, k=len(keys))
                        ratios.append(sum(totals[k][ep] for k in draw)/sum(totals[k][baseline] for k in draw))
                    ratios.sort()
                    intervals.append(dict(population=population, a=a, baseline=baseline, method=ep,
                                          metric=metric, clusters=len(keys),
                                          ratio=sum(v[ep] for v in totals.values())/sum(v[baseline] for v in totals.values()),
                                          interval95=[ratios[50], ratios[1949]]))
    summary = dict(calls=len(rows), completed=sum(r['status'] == 'complete' for r in rows),
                   planted_matches=sum(r.get('correct', False) for r in rows),
                   clusters=len({r['cluster'] for r in rows}), groups=groups, intervals=intervals,
                   note='Stratified pooled means use the design mixture, not uniform-sphere weights. '
                        'Censored costs are budget-limited, never complete expected costs.')
    (folder/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run(args):
    folder = Path(args.output).resolve()
    if folder == ROOT or (ROOT/'reference').resolve() in (folder, *folder.parents):
        raise ValueError('Use a separate results folder, never reference data')
    folder.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != 'output'}
    config['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['dependency_sha256'] = {
        name: hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
        for name in ('lee_revision.py', 'lee_z4_dcc_comparison.py', 'ring_constraints.py')}
    config_path = folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Resume configuration or source differs; use a new output folder')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       platform=platform.platform(), processor=platform.processor())
    (folder/'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    codes = generate_codes(args.profile, args.mothers, args.seed, folder)
    path = folder/'instances.jsonl'
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            completed.add((row['code'], row['population'], row['case'], row['method']))
    methods = RADIUS if args.profile == 'radius' else ABLATION if args.profile == 'ablation' else MAIN
    for info in codes:
        cases = case_list(info, args.samples, args.all_compositions)
        if all((info['code'], pop, case, method) in completed
               for pop, case, error in cases for method in methods):
            continue
        start = time.perf_counter_ns()
        prepared = prepare(info['h'])
        prep_ns = time.perf_counter_ns()-start
        if (prepared['K'], prepared['k0']) != (info['K'], info['k0']):
            raise AssertionError('Public preparation subtype mismatch')
        for index, (pop, case, error) in enumerate(cases):
            syndrome = raw_syndrome(info['h'], error)
            trial_seed = info['seed']*100003+index
            pending = [m for m in methods if (info['code'], pop, case, m) not in completed]
            results = {}
            for method in pending:
                result = decode(prepared, syndrome, info['t'], trial_seed, method,
                                args.trials, args.candidates)
                if result['status'] == 'complete' and result['answer'] != error:
                    raise AssertionError('Certified unique error mismatch')
                results[method] = result
            timings = defaultdict(list)
            for repeat in range(args.repeats):
                order = pending[(index+repeat) % max(1, len(pending)):]+pending[:(index+repeat) % max(1, len(pending))]
                if repeat % 2:
                    order.reverse()
                for method in order:
                    start = time.perf_counter_ns()
                    measured = decode(prepared, syndrome, info['t'], trial_seed, method,
                                      args.trials, args.candidates, track=False)
                    elapsed = time.perf_counter_ns()-start
                    if measured['status'] != results[method]['status'] or measured['answer'] != results[method]['answer']:
                        raise AssertionError('Counted and timed executions disagree')
                    timings[method].append(dict(online_ns=elapsed, **measured['times']))
            with path.open('a', encoding='utf-8') as handle:
                for method in pending:
                    result = results[method]
                    row = {k: info[k] for k in ('code', 'cluster', 'n', 'K', 'k0', 't', 'd_binary')}
                    row.update(method=method, population=pop, case=case, error=error, syndrome=syndrome,
                               trial_seed=trial_seed, a=error.count(2), s=sum(x % 2 for x in error),
                               status=result['status'], reason=result['reason'], initial_radius=result['initial_radius'],
                               correct=result['answer'] == error, setup_ns=prep_ns,
                               matrix_payload_bits=prepared['matrix_bits'], **result['counters'])
                    row['recovery_trials'] = row.get('second_trials', 0)+row.get('ring_trials', 0)
                    for metric in ('online_ns', 'first_ns', 'recovery_ns', 'verify_ns'):
                        row[metric] = statistics.median(x.get(metric, 0) for x in timings[method])
                    row['cold_ns'] = row['online_ns']+prep_ns
                    row['batch100_ns'] = row['online_ns']+prep_ns/100
                    row['timing_samples'] = timings[method]
                    handle.write(json.dumps(row)+'\n')
        print(info['code'], 'complete', flush=True)
    summary = summarize(folder)
    print(json.dumps({k: summary[k] for k in ('calls', 'completed', 'planted_matches', 'clusters')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='smoke')
    parser.add_argument('--mothers', type=int, default=2)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trials', type=int, default=4096)
    parser.add_argument('--candidates', type=int, default=100000)
    parser.add_argument('--all-compositions', action='store_true')
    parser.add_argument('--seed', type=int, default=2026110201)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.mothers, args.samples, args.repeats, args.trials, args.candidates) < 1:
        parser.error('All counts must be positive')
    run(args)


if __name__ == '__main__':
    main()
