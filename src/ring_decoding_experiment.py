"""Finite experiment: Z16, n=10, Hamming metric; standard library only.

Run: python ring_decoding_experiment.py --trials 300 --seed 2026
Counts are measured work, not asymptotic complexity bounds.
"""
import argparse
import itertools
import json
import random
import time
import statistics
from collections import Counter
from pathlib import Path

N, R = 10, 4
VALUATIONS = (0, 1, 3)
# First three columns are identity; the last seven are all nonzero F2^3 columns.
G = [[int(j == i) for j in range(3)] +
     [(j >> i) & 1 for j in range(1, 8)] for i in range(3)]


def weight(x):
    return sum(a != 0 for a in x)


def parity(level):
    """Parity check for the filtration code over Z/(2^level)."""
    modulus = 2 ** level
    rows = []
    for j in range(3, N):
        row = [0] * N
        for k in range(3):
            row[k] = -G[k][j] % modulus
        row[j] = 1
        rows.append(row)
    for k, valuation in enumerate(VALUATIONS):
        exponent = max(valuation - (R - level), 0)
        if exponent:
            row = [0] * N
            row[k] = 2 ** (level - exponent)
            rows.append(row)
    return rows


def field_check(active):
    rows = []
    for j in range(N):
        if j not in active:
            row = [0] * N
            row[j] = 1
            for k in active:
                row[k] = G[k][j]
            rows.append(row)
    return rows


def syndrome(h, x, modulus):
    return [sum(a*b for a, b in zip(row, x)) % modulus for row in h]


def packed(values):
    return sum((x & 1) << i for i, x in enumerate(values))


def reduce_vector(vector, basis, stats):
    coefficients = 0
    for pivot in sorted(basis, reverse=True):
        if (vector >> pivot) & 1:
            value, representation = basis[pivot]
            vector ^= value
            coefficients ^= representation
            stats['binary_vector_xors'] += 1
            stats['kernel_packed_xors'] += 2
    return vector, coefficients


def lb(h, b, allowed, budget, rng, stats, v_max=1, *, cached_columns=None):
    """Binary LB, <=v_max errors outside a random column basis.

    Positions in allowed have zero weight cost and are included in the basis.
    This implements erasure-aware recovery without charging known support twice.
    """
    n = len(h[0])
    columns = (cached_columns if cached_columns is not None else
               [packed(row[j] for row in h) for j in range(n)])
    target = packed(b)
    allowed = set(allowed)
    for _ in range(10000):
        stats['information_sets'] += 1
        rest = [j for j in range(n) if j not in allowed]
        rng.shuffle(rest)
        basis, selected = {}, []
        for j in sorted(allowed) + rest:
            rem, coeff = reduce_vector(columns[j], basis, stats)
            if rem:
                basis[rem.bit_length()-1] = (rem, coeff ^ (1 << j))
                stats['kernel_packed_xors'] += 1
                selected.append(j)
            elif j in allowed:
                raise ValueError('Known-support columns dependent: list recovery needed')
        rem, _ = reduce_vector(target, basis, stats)
        if rem:
            raise ValueError('Inconsistent binary syndrome')
        info = [j for j in range(n) if j not in selected]
        for v in range(min(v_max, budget, len(info)) + 1):
            for subset in itertools.combinations(info, v):
                stats['candidates'] += 1
                rhs, guess = target, 0
                for j in subset:
                    rhs ^= columns[j]
                    guess ^= 1 << j
                    stats['binary_vector_xors'] += 1
                    stats['kernel_packed_xors'] += 2
                rem, correction = reduce_vector(rhs, basis, stats)
                answer = guess ^ correction
                stats['kernel_packed_xors'] += 1
                outside = sum((answer >> j) & 1 for j in range(n) if j not in allowed)
                if rem == 0 and outside <= budget:
                    return [(answer >> j) & 1 for j in range(n)]
    raise RuntimeError('LB iteration limit reached')


def projection(y, t, rng, stats, vmax):
    e = [0] * N
    for a in range(1, R+1):
        h, modulus, scale = parity(a), 2**a, 2**(a-1)
        sy, se = syndrome(h, y, modulus), syndrome(h, e, modulus)
        stats['ring_dot_products'] += 2 * len(h)
        residual = [(x-z) % modulus for x, z in zip(sy, se)]
        if any(x % scale for x in residual):
            raise ValueError('Low-level candidate does not lift')
        b = [x // scale for x in residual]
        known = {j for j in range(N) if e[j]}
        z = lb(h, b, known, t-len(known), rng, stats, vmax)
        e = [x + scale*d for x, d in zip(e, z)]
    return e


def supercode(y, t, rng, stats, vmax, informed=False):
    residual, result, known = list(y), [0]*N, set()
    for i in range(R):
        active = [j for j, v in enumerate(VALUATIONS) if v <= i]
        h = field_check(active)
        b = syndrome(h, residual, 2)
        stats['binary_dot_products'] += len(h)
        z = lb(h, b, known if informed else (),
               t-len(known) if informed else t, rng, stats, vmax)
        code_digit = [(x-d) % 2 for x, d in zip(residual, z)]
        modulus = 2**(R-i)
        lift = [sum(code_digit[k]*G[k][j] for k in active) % modulus
                for j in range(N)]
        stats['ring_dot_products'] += N
        residual = [((x-d-c) % modulus)//2
                    for x, d, c in zip(residual, z, lift)]
        result = [x+2**i*d for x, d in zip(result, z)]
        known.update(j for j, x in enumerate(result) if x)
    return result


def codewords():
    return [[sum(u[k]*2**VALUATIONS[k]*G[k][j] for k in range(3)) % 16
             for j in range(N)]
            for u in itertools.product(*(range(2**(R-v)) for v in VALUATIONS))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=300)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--exhaustive', action='store_true',
                        help='Test all 10125 weight-two errors, one random codeword each')
    parser.add_argument('--v-max', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('--output', type=Path, help='Also save the JSON report')
    args = parser.parse_args()
    words = codewords()
    distance = min(weight(c) for c in words if any(c))
    assert len({tuple(c) for c in words}) == 256 and distance == 5
    assert all(not any(syndrome(parity(4), c, 16)) for c in words)
    source = random.Random(args.seed)
    cases = []
    patterns = ([(positions, values) for positions in itertools.combinations(range(N), 2)
                 for values in itertools.product(range(1, 16), repeat=2)]
                if args.exhaustive else
                [(source.sample(range(N), 2), [source.randrange(1, 16) for _ in range(2)])
                 for _ in range(args.trials)])
    for positions, values in patterns:
        error = [0]*N
        for j, value in zip(positions, values):
            error[j] = value
        word = source.choice(words)
        cases.append(([(c+e) % 16 for c, e in zip(word, error)], error))
    args.trials = len(cases)
    methods = {
        'projection_LB_known_support': projection,
        'supercode_LB': supercode,
        'supercode_LB_known_support': lambda *a: supercode(*a, informed=True),
    }
    report = {'ring': 'Z16', 'n': N, 'subtype': [1,1,0,1],
              'code_size': len(words), 'minimum_Hamming_distance': distance,
              'error_weight': 2, 'trials': args.trials, 'seed': args.seed,
              'LB_v_max': args.v_max, 'base_generator_rows': G, 'results': {}}
    per_method = {}
    for name, method in methods.items():
        stats, elapsed = Counter(), 0.0
        records = []
        for index, (y, expected) in enumerate(cases):
            rng = random.Random(args.seed*1000003+index)
            before = stats.copy()
            start = time.perf_counter()
            answer = method(y, 2, rng, stats, args.v_max)
            elapsed += time.perf_counter()-start
            assert answer == expected, (name, index, answer, expected)
            stats['verified'] += 1
            records.append({key: stats[key] - before[key]
                            for key in ('candidates', 'information_sets',
                                        'binary_vector_xors', 'ring_dot_products',
                                        'binary_dot_products', 'kernel_packed_xors')})
        per_method[name] = records
        report['results'][name] = {'totals': dict(stats),
            'mean_candidates': stats['candidates']/args.trials,
            'mean_information_sets': stats['information_sets']/args.trials,
            'mean_binary_vector_xors': stats['binary_vector_xors']/args.trials,
            'mean_kernel_packed_xors': stats['kernel_packed_xors']/args.trials,
            'seconds': elapsed}
    report['paired_comparison'] = {}
    left = 'projection_LB_known_support'
    right = 'supercode_LB_known_support'
    for metric in ('candidates', 'binary_vector_xors', 'kernel_packed_xors',
                   'information_sets'):
        a = [row[metric] for row in per_method[left]]
        b = [row[metric] for row in per_method[right]]
        differences = [x-y for x, y in zip(a, b)]
        report['paired_comparison'][metric] = {
            'projection_mean': statistics.mean(a),
            'supercode_support_mean': statistics.mean(b),
            'mean_projection_minus_supercode': statistics.mean(differences),
            'median_projection_minus_supercode': statistics.median(differences),
            'projection_lower_count': sum(x < y for x, y in zip(a, b)),
            'supercode_lower_count': sum(y < x for x, y in zip(a, b)),
            'ties': sum(x == y for x, y in zip(a, b)),
            'standard_error_of_difference': (
                statistics.stdev(differences) / (len(differences) ** 0.5)
                if len(differences) > 1 else 0.0),
        }
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
