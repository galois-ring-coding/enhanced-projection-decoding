"""Polynomial-slack Lee decoder and independent enhanced-projection control.

This verifies the theorem's kernel, not cryptographic-size wall-clock speed.
Existing zero-slack DCC timing results are kept separate and unchanged.
"""
import itertools
import json
import math
import random
from pathlib import Path

from lee_z4_dcc_comparison import (Arithmetic, code_suite, error_cases, lee,
                                 prepare, raw_syndrome, received_representative)


def canonical(h, target):
    n = len(h[0])
    rows = [sum((x % 2) << j for j, x in enumerate(row)) | ((b % 2) << n)
            for row, b in zip(h, target)]
    rank, pivots = 0, []
    for col in range(n):
        pivot = next((i for i in range(rank, len(rows)) if rows[i] >> col & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i] >> col & 1:
                rows[i] ^= rows[rank]
        pivots.append(col)
        rank += 1
    if any(row >> n & 1 for row in rows[rank:]):
        raise ValueError('Inconsistent binary system')
    return [[(row >> j) & 1 for j in range(n)] for row in rows[:rank]], \
        [(row >> n) & 1 for row in rows[:rank]], pivots


def slack_decode(h, target, known, budget, slack, seed):
    h, target, _ = canonical(h, target)
    n, rank = len(h[0]), len(h)
    known = set(known)
    rest = [j for j in range(n) if j not in known]
    extra = min(slack, n-rank)
    rng, trace = random.Random(seed), []
    for trial in range(10000):
        chosen = sorted(known)+rng.sample(rest, rank+extra-len(known))
        trace.append(('set', tuple(chosen)))
        try:
            a, b, pivots = canonical([[row[j] for j in chosen] for row in h], target)
        except ValueError:
            continue
        if len(pivots) != rank:
            continue
        free = [j for j in range(len(chosen)) if j not in pivots]
        for mask in range(2**len(free)):
            local = [0]*len(chosen)
            for bit, col in enumerate(free):
                local[col] = mask >> bit & 1
            for row, value, pivot in zip(a, b, pivots):
                local[pivot] = (value+sum(row[j]*local[j] for j in free)) % 2
            answer = [0]*n
            for col, value in zip(chosen, local):
                answer[col] = value
            trace.append(('candidate', tuple(answer)))
            if sum(value for j, value in enumerate(answer) if j not in known) <= budget:
                return answer, trace
    raise RuntimeError('Slack decoder trial limit reached')


def decode(code, syndrome, t, seed, method, slack):
    prepared = prepare(code, 'super_support')
    h, hcount = prepared['h'], prepared['unit_rows']
    if method == 'super':
        y = received_representative(prepared, syndrome, Arithmetic(2, track=False))
        h0 = prepared['h_first']
        b0 = [sum(a*b for a, b in zip(row, y)) % 2 for row in h0]
    elif method == 'enhanced':
        # Normalize the original torsion equations; no supercode residual call.
        h0 = [[x % 2 for x in row] for row in h[:hcount]]
        b0 = [x % 2 for x in syndrome[:hcount]]
        for row, value in zip(h[hcount:], syndrome[hcount:]):
            if value % 2:
                raise ValueError('Inconsistent torsion syndrome')
            h0.append([(x//2) % 2 for x in row])
            b0.append(value//2)
    else:
        raise ValueError('Unknown method')
    h0, b0, _ = canonical(h0, b0)
    low, trace0 = slack_decode(h0, b0, [], t, slack, seed*17)
    known = [j for j, value in enumerate(low) if value]
    if method == 'super':
        word = [(x-e) % 2 for x, e in zip(y, low)]
        active = prepared['active']
        lifted = [sum(word[j]*code.g[j][col] for j in active) % 4 for col in range(code.n)]
        numerator = [(x-e-c) % 4 for x, e, c in zip(y, low, lifted)]
        if any(x % 2 for x in numerator):
            raise ValueError('Nondivisible compatible residual')
        residual = [x//2 for x in numerator]
        h1 = prepared['h_second']
        b1 = [sum(a*b for a, b in zip(row, residual)) % 2 for row in h1]
    else:
        residual = [(s-v) % 4 for s, v in zip(syndrome, raw_syndrome(h, low))]
        if any(x % 2 for x in residual) or any(residual[hcount:]):
            raise ValueError('Invalid projected residue')
        h1 = [[x % 2 for x in row] for row in h[:hcount]]
        b1 = [x//2 for x in residual[:hcount]]
    h1, b1, _ = canonical(h1, b1)
    high, trace1 = slack_decode(h1, b1, known, (t-len(known))//2, slack, seed*17+1)
    answer = [x+2*y for x, y in zip(low, high)]
    if raw_syndrome(h, answer) != syndrome or lee(answer) > t:
        raise ValueError('Original Z4 verification failed')
    return answer, (h0, b0, h1, b1), (trace0, trace1)


def entropy(x):
    return 0.0 if x in (0, 1) else -x*math.log2(x)-(1-x)*math.log2(1-x)


def f(a, b):
    return a*entropy(b/a) if b else 0.0


def exponents(k0=.1, k=.3, sigma=.06, alpha=.01):
    weight = sigma+2*alpha
    sc0 = f(1, sigma)-f(1-k0, sigma)
    dcc0 = f(1, sigma)-f(1-k, sigma)
    second = f(1-sigma, alpha)-f(1-sigma-k, alpha)
    return dict(kappa0=k0, kappa=k, sigma=sigma, alpha=alpha, W=weight,
                super_first=sc0, common_second=second, super_total=max(sc0, second),
                dcc_first_lower_bound=dcc0, exponent_gap_lower_bound=dcc0-max(sc0, second),
                distance_failure_exponent=entropy(2*weight)-(1-k))


def model_rows():
    result = []
    for n in (100, 1000, 10000):
        k0, k, s, a, t = n//10, 3*n//10, 6*n//100, n//100, 8*n//100
        ell = math.ceil(4*math.log2(n))
        l0, l1 = min(ell, k0), min(ell, k)
        logcomb = lambda n, k: math.log2(math.comb(n, k))
        first = logcomb(n, s)-logcomb(n-k0+l0, s)
        second = logcomb(n-s, a)-logcomb(n-k-s+l1, a)
        original = logcomb(n, s)-logcomb(n-k, s)
        distance = math.log2(sum(math.comb(n, j) for j in range(1, 2*t+1)))+k-n+1
        result.append(dict(n=n, k0=k0, K=k, s=s, a=a, t=t, ell=ell,
            sc_first_coverage_bits=first, sc_second_coverage_bits=second,
            dcc_first_coverage_bits=original,
            sc_full_list_proxy_bits=max(first+l0, second+l1),
            distance_failure_log2_upper=min(0, distance),
            scope='Modeled support counts and full-list proxy; linear algebra not included, no attack executed'))
    return result


def main():
    pairs = traces = 0
    for code, description in code_suite(mothers=1):
        data = error_cases(code.n, description['t'], 8, description['seed']+99, balanced=True)
        for index, (_, error) in enumerate(data):
            syndrome = raw_syndrome(code.parity(2), error)
            for slack in (0, 1, 2):
                left = decode(code, syndrome, description['t'], index+918, 'super', slack)
                right = decode(code, syndrome, description['t'], index+918, 'enhanced', slack)
                if left[0] != error or right[0] != error or left[1:] != right[1:]:
                    raise AssertionError((code.name, index, slack))
                pairs += 1
                traces += len(left[2][0])+len(left[2][1])
    result = dict(paired_complete_cases=pairs, decoder_calls=2*pairs, exact_trace_events_checked=traces,
                  failures=0, rates=exponents(), finite_models=model_rows(),
                  scope='Fixed-composition theorem; complete small-code equivalence tests, not timing evidence')
    Path(__file__).with_name('z4_separation_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
