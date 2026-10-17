"""Uniform-flag Lee ablations and exact certificates for sphere-average bounds.

Generation and the supplied common normal form are outside online timings.
All decoders get only the public syndrome, bound and code-dependent data.
No published historical outputs are overwritten.
"""
import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from lee_z4_dcc_comparison import (Arithmetic, binary_distance,
    error_cases, lee, raw_syndrome)
from z4_separation import canonical, exponents, entropy

METHODS = ('dcc_ring', 'dcc_binary', 'sc_plain', 'sc_support', 'enhanced',
           'sc_all', 'enhanced_all')


def charge(op, adds=0, mults=0, inversions=0):
    if op.track:
        op.stats['adds'] += adds
        op.stats['mults'] += mults
        op.stats['inversions'] += inversions
        op.stats['arithmetic_proxy'] += adds*op.bits+mults*op.bits**2+inversions*op.bits**3


def eliminate(h, target, selected, op):
    """Same dense scalar model as the reference, with vectorized row updates."""
    a = np.array([list(row)+[b] for row, b in zip(h, target)], dtype=np.int64) % op.mod
    height, width = a.shape
    for r, col in enumerate(selected):
        choices = np.flatnonzero(a[r:, col] % 2)
        if not len(choices):
            return None
        pivot = r+int(choices[0])
        a[[r, pivot]] = a[[pivot, r]]
        factor = pow(int(a[r, col]), -1, op.mod)
        a[r] = a[r]*factor % op.mod
        charge(op, mults=width, inversions=1)
        others = np.arange(height) != r
        a[others] = (a[others]-a[others, col, None]*a[r]) % op.mod
        charge(op, adds=(height-1)*width, mults=(height-1)*width)
    return a.tolist()


def prange(h, target, radius, rng, op):
    if len(h) == len(h[0]) and all(row == [int(j == i) for j in range(len(h))]
                                  for i, row in enumerate(h)) and any(target):
        op.event('identity_shortcuts')
        op.event('candidates')
        if sum(target) > radius:
            raise ValueError('No admissible identity solution')
        return list(target)
    return binary_search(h, target, [], radius, rng, op)


def support_digit(h, target, known, radius, rng, op):
    n, s = len(h[0]), len(known)
    rest = [j for j in range(n) if j not in known]
    a = eliminate(h, target, known, op)
    if a is None:
        raise ValueError('Dependent known columns')
    reduced = [[row[j] for j in rest] for row in a[s:]]
    beta = [row[-1] for row in a[s:]]
    if radius == 0:
        op.event('zero_budget_shortcuts')
        if any(beta):
            raise ValueError('No zero-budget completion')
        outside = [0]*len(rest)
    else:
        outside = prange(reduced, beta, radius, rng, op)
    answer = [0]*n
    for j, x in zip(rest, outside):
        answer[j] = x
    for r, j in enumerate(known):
        answer[j] = op.add(a[r][-1], -op.dot([a[r][k] for k in rest], outside))
    return answer


def inverse(a, modulus):
    n = len(a)
    rows = [list(row)+[int(i == j) for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        pivot = next((j for j in range(i, n) if rows[j][i] % 2), None)
        if pivot is None:
            raise ValueError('Noninvertible ambient matrix')
        rows[i], rows[pivot] = rows[pivot], rows[i]
        scale = pow(rows[i][i], -1, modulus)
        rows[i] = [x*scale % modulus for x in rows[i]]
        for j in range(n):
            if j != i:
                scale = rows[j][i]
                rows[j] = [(x-scale*y) % modulus for x, y in zip(rows[j], rows[i])]
    return [row[n:] for row in rows]


def make_code(ambient, k, k0):
    n = len(ambient)
    inv = inverse(ambient, 4)
    dual = [list(row) for row in zip(*inv)]
    h = dual[k:]+[[2*x % 4 for x in row] for row in dual[k0:k]]
    return dict(n=n, K=k, k0=k0, ambient=ambient, dual=dual, h=h)


def prepare(code, method):
    # The ambient normal form is common public preprocessing, not an error hint.
    n, k, k0 = code['n'], code['K'], code['k0']
    h = [list(row) for row in code['h']]
    h1 = [[x % 2 for x in row] for row in h[:n-k]]
    if method.startswith('sc_'):
        h0 = [[x % 2 for x in row] for row in code['dual'][k:]+code['dual'][k0:k]]
    elif method.startswith('enhanced'):
        h0 = h1+[[x//2 for x in row] for row in h[n-k:]]
    else:
        h0 = h1
    return dict(code=code, method=method, h=h, h0=h0, h1=h1,
                matrix_payload_bits=2*sum(map(len, h))+sum(map(len, h0))+sum(map(len, h1)))


def selected_solutions(h, target, chosen, op):
    """Stream every restricted solution, including consistent singular systems."""
    rows = np.array([[row[j] for j in chosen]+[b] for row, b in zip(h, target)], dtype=np.int64)
    rank, pivots = 0, []
    for col in range(len(chosen)):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[[rank, pivot]] = rows[[pivot, rank]]
        others = np.arange(len(rows)) != rank
        rows[others] = (rows[others]-rows[others, col, None]*rows[rank]) % 2
        count = (len(rows)-1)*(len(chosen)+1)
        charge(op, adds=count, mults=count)
        pivots.append(col)
        rank += 1
    rows = rows.tolist()
    if rank < len(h):
        op.event('singular_trials')
    if any(row[-1] for row in rows[rank:]):
        op.event('inconsistent_trials')
        return
    free = [j for j in range(len(chosen)) if j not in pivots]
    op.event('potential_list_solutions', 2**len(free))
    for mask in range(2**len(free)):
        local = [0]*len(chosen)
        for bit, j in enumerate(free):
            local[j] = mask >> bit & 1
        for row, pivot in zip(rows, pivots):
            local[pivot] = op.add(row[-1], op.dot([row[j] for j in free], [local[j] for j in free]))
        answer = [0]*len(h[0])
        for j, value in zip(chosen, local):
            answer[j] = value
        yield answer


def binary_search(h, target, known, budget, rng, op, force=False, all_solutions=False):
    n, size, known = len(h[0]), len(h), set(known)
    if not any(target):
        op.event('zero_syndrome_shortcuts')
        return [0]*n
    forced = sorted(known) if force else []
    rest = [j for j in range(n) if j not in forced]
    for _ in range(10000):
        op.event('trials')
        selected = forced+rng.sample(rest, size-len(forced))
        if all_solutions:
            candidates = selected_solutions(h, target, selected, op)
        else:
            a = eliminate(h, target, selected, op)
            if a is None:
                op.event('singular_trials')
                continue
            answer = [0]*n
            for j, row in zip(selected, a):
                answer[j] = row[-1]
            candidates = (answer,)
        for answer in candidates:
            op.event('candidates')
            if sum(value for j, value in enumerate(answer) if j not in known) <= budget:
                return answer
    raise RuntimeError('Binary trial limit reached; no ranking is permitted')


def decode(prepared, syndrome, t, seed, track=True):
    code, method = prepared['code'], prepared['method']
    n, k, k0 = code['n'], code['K'], code['k0']
    first, second, ring, verify = [Arithmetic(b, track) for b in (1, 1, 2, 2)]
    is_super = method.startswith('sc_')
    all_solutions = method.endswith('_all')
    if is_super:
        z = [0]*n
        z[k:] = syndrome[:n-k]
        z[k0:k] = [x//2 for x in syndrome[n-k:]]
        y = [ring.dot([row[j] for row in code['ambient']], z) for j in range(n)]
        target = [first.dot(row, y) for row in prepared['h0']]
    else:
        target = [x % 2 for x in syndrome[:n-k]]
        if method.startswith('enhanced'):
            target += [x//2 for x in syndrome[n-k:]]
    if all_solutions:
        low = binary_search(prepared['h0'], target, [], t, random.Random(seed*17), first,
                            all_solutions=True)
    else:
        low = prange(prepared['h0'], target, t, random.Random(seed*17), first)
    known = [j for j, value in enumerate(low) if value]
    budget = (t-len(known))//2
    if method == 'dcc_ring':
        rng = random.Random(seed*17+1)
        rest = [j for j in range(n) if j not in known]
        for _ in range(10000):
            ring.event('trials')
            chosen = known+rng.sample(rest, n-k-len(known))
            a = eliminate(prepared['h'], syndrome, chosen, ring)
            if a is None:
                ring.event('singular_trials')
                continue
            if any(row[-1] for row in a[n-k:]):
                ring.event('inconsistent_trials')
                continue
            ring.event('candidates')
            answer = [0]*n
            for j, row in zip(chosen, a):
                answer[j] = row[-1]
            if lee(answer) <= t:
                break
        else:
            raise RuntimeError('DCC trial limit reached')
    else:
        if is_super:
            word = [first.add(x % 2, -e) for x, e in zip(y, low)]
            coefficients = [first.dot(row, word) for row in code['dual'][:k0]]
            lifted = [ring.dot([row[j] for row in code['ambient'][:k0]], coefficients) for j in range(n)]
            numerator = [ring.add(ring.add(x, -e), -c) for x, e, c in zip(y, low, lifted)]
            if any(x % 2 for x in numerator):
                raise ValueError('Incompatible supercode prefix')
            residual = [x//2 for x in numerator]
            target1 = [second.dot(row, residual) for row in prepared['h1']]
        else:
            residual = [ring.add(s, -ring.dot(row, low)) for row, s in zip(prepared['h'], syndrome)]
            if any(x % 2 for x in residual) or any(residual[n-k:]):
                raise ValueError('Incompatible projected prefix')
            target1 = [x//2 for x in residual[:n-k]]
        if all_solutions:
            high = binary_search(prepared['h1'], target1, known, budget, random.Random(seed*17+1),
                                 second, force=True, all_solutions=True)
        elif method == 'sc_plain':
            high = binary_search(prepared['h1'], target1, known, budget, random.Random(seed*17+1), second)
        else:
            high = support_digit(prepared['h1'], target1, known, budget, random.Random(seed*17+1), second)
        answer = [ring.add(a, ring.mul(2, b)) for a, b in zip(low, high)]
    if [verify.dot(row, answer) for row in prepared['h']] != syndrome or lee(answer) > t:
        raise ValueError('Failed original syndrome or weight check')
    counts = Counter()
    for label, op in (('first', first), ('second', second), ('ring', ring), ('verify', verify)):
        counts.update(op.stats)
        counts.update({label+'_'+key: value for key, value in op.stats.items()})
    return answer, dict(counts)


def uniform_codes(mothers, sizes):
    for n, k, t in sizes:
        for index in range(mothers):
            seed = 2026092500+n*1000+index
            rng = random.Random(seed)
            for attempt in range(100000):
                ambient = [[rng.randrange(4) for _ in range(n)] for _ in range(n)]
                try:
                    code = make_code(ambient, k, k//2)
                except ValueError:
                    continue
                dh = binary_distance([[x % 2 for x in row] for row in ambient[:k]])
                if dh > 2*t:
                    break
            else:
                raise RuntimeError('Uniform ambient rejection exhausted')
            for k0 in sorted({1, k//2, k-1}):
                current = make_code(ambient, k, k0)
                yield current, dict(n=n, K=k, k0=k0, t=t, seed=seed, d_binary=dh,
                    cluster=f'n{n}_mother{index}', code=f'n{n}_mother{index}_k{k0}',
                    generation_attempts=attempt+1)


def certificates():
    k0, k, w = .1, .3, .08
    upper0 = w*math.log2(1/(1-k0-w))
    upper1 = w/2*math.log2(1/(1-k-w))
    lower = w*(1-w/2)*math.log2(1/(1-k))
    exact = dict(distance=25**250 < 2**175*4**40*21**210,
                 first=50**100*7**96 < 41**100*10**96,
                 second=50**50*7**96 < 31**50*10**96)
    if not all(exact.values()):
        raise AssertionError(exact)
    models = []
    for n in (100, 1000, 10000):
        t, k0n, kn = 2*n//25, n//10, 3*n//10
        normalizer = math.comb(2*n, t)
        first = second = dcc = Fraction()
        for a in range(t//2+1):
            s = t-2*a
            probability = Fraction(math.comb(n, s)*math.comb(n-s, a)*2**s, normalizer)
            first += probability*Fraction(math.comb(n, s), math.comb(n-k0n, s))
            second += probability*Fraction(math.comb(n-s, a), math.comb(n-kn-s, a))
            dcc += probability*Fraction(math.comb(n, s), math.comb(n-kn, s))
        log = lambda x: math.log2(x.numerator)-math.log2(x.denominator)
        models.append(dict(n=n, sc_coverage_sum_bits=log(first+second),
                           dcc_first_coverage_bits=log(dcc),
                           scope='Exact averaged coverage models, excludes polynomial factors; not timings'))
    grid = []
    for j in range(20001):
        alpha = w/2*j/20000
        sigma = w-2*alpha
        composition = (1-sigma-alpha, sigma, alpha)
        rate = 2*entropy(w/2)+sum(x*math.log2(x) for x in composition if x)-sigma
        values = exponents(k0, k, sigma, alpha)
        grid.append((max(values['super_first'], values['common_second'])-rate,
                     values['dcc_first_lower_bound']-rate))
    return dict(analytic_super_upper=max(upper0, upper1), analytic_dcc_lower=lower,
                certified_gap=lower-max(upper0, upper1), integer_certificates=exact,
                exploratory_grid_super_upper=max(x[0] for x in grid),
                exploratory_grid_dcc_lower=max(x[1] for x in grid),
                grid_is_not_the_proof=True, finite_models=models)


def aggregate(records):
    failures = [r for r in records if not r['correct']]
    if failures:
        return dict(status='Failures retained; aggregate ranking suppressed', failures=failures)
    groups = []
    for n in [None]+sorted({r['n'] for r in records}):
        for method in METHODS:
            selected = [r for r in records if r['method'] == method and (n is None or r['n'] == n)]
            metrics = {key: statistics.mean(r.get(key, 0) for r in selected)
                       for key in ('arithmetic_proxy', 'trials', 'candidates', 'online_ns', 'cold_ns',
                                   'batch100_ns', 'first_trials', 'second_trials', 'ring_trials',
                                   'first_arithmetic_proxy', 'second_arithmetic_proxy', 'ring_arithmetic_proxy')}
            groups.append(dict(n=n, method=method, cases=len(selected), **metrics))
    pairs = {}
    for row in records:
        pairs.setdefault((row['code'], row['case']), {})[row['method']] = row
    return dict(status='All complete decoders correct', groups=groups,
        sc_support_vs_plain_proxy_wins=sum(p['sc_support']['arithmetic_proxy'] < p['sc_plain']['arithmetic_proxy'] for p in pairs.values()),
        sc_support_vs_dcc_proxy_wins=sum(p['sc_support']['arithmetic_proxy'] < p['dcc_ring']['arithmetic_proxy'] for p in pairs.values()),
        total_pairs=len(pairs),
        equivalence_kernel_counters=all(all(p[a].get(key, 0) == p[b].get(key, 0)
            for key in ('first_trials', 'first_candidates', 'second_trials', 'second_candidates'))
            for p in pairs.values() for a, b in (('sc_support', 'enhanced'), ('sc_all', 'enhanced_all'))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mothers', type=int, default=8)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--output', default='lee_revision_results')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.mothers, args.samples, args.timing_repeats) < 1:
        parser.error('Counts must be positive')
    destination = Path(__file__).parent/args.output
    destination.mkdir(parents=True, exist_ok=True)
    sizes = ((16, 4, 2),) if args.smoke else ((16, 4, 2), (24, 6, 3), (40, 8, 4), (64, 12, 5))
    records, codes = [], []
    for code, description in uniform_codes(args.mothers, sizes):
        prepared, setups = {}, {}
        for method in METHODS:
            times = []
            for _ in range(5):
                start = time.perf_counter_ns()
                prepared[method] = prepare(code, method)
                times.append(time.perf_counter_ns()-start)
            setups[method] = statistics.median(times)
        codes.append(dict(**description, ambient=code['ambient'], h=code['h'], setup_ns=setups))
        cases = error_cases(code['n'], description['t'], args.samples, description['seed']+77)
        for index, (_, error) in enumerate(cases):
            syndrome = raw_syndrome(code['h'], error)
            seed = description['seed']*100003+index
            current = {}
            for method in METHODS:
                row = dict(**description, method=method, case=index, error=error, syndrome=syndrome, trial_seed=seed)
                try:
                    answer, counts = decode(prepared[method], syndrome, description['t'], seed)
                    row.update(counts, correct=answer == error)
                except (ValueError, RuntimeError) as exc:
                    row.update(correct=False, failure=str(exc))
                current[method] = row
            timings = {method: [] for method in METHODS}
            for repeat in range(args.timing_repeats):
                offset = (index+repeat) % len(METHODS)
                order = METHODS[offset:]+METHODS[:offset]
                if repeat % 2:
                    order = tuple(reversed(order))
                for method in order:
                    if not current[method]['correct']:
                        continue
                    start = time.perf_counter_ns()
                    answer, _ = decode(prepared[method], syndrome, description['t'], seed, track=False)
                    timings[method].append(time.perf_counter_ns()-start)
                    if answer != error:
                        raise AssertionError('Timed decoder disagrees')
            for method in METHODS:
                row = current[method]
                row['online_ns'] = statistics.median(timings[method]) if timings[method] else None
                if row['online_ns'] is not None:
                    row['cold_ns'] = row['online_ns']+setups[method]
                    row['batch100_ns'] = row['online_ns']+setups[method]/100
                records.append(row)
        print(description['code'], 'complete', flush=True)
    summary = aggregate(records)
    summary.update(certificates=certificates(), code_count=len(codes),
        independent_clusters=len({c['cluster'] for c in codes}),
        counted_calls=len(records), timed_calls=sum(r['correct'] for r in records)*args.timing_repeats,
        environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
        settings=vars(args), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Uniform ambient GL matrices conditioned on certified D1 distance; nondegenerate subtypes; shared common normal form; small complete attacks only')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (destination/'codes.json').write_text(json.dumps(codes, indent=2), encoding='utf-8')
    (destination/'instances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('status', 'code_count', 'independent_clusters', 'counted_calls')}, indent=2))
    if any(not r['correct'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
