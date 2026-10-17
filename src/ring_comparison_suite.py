"""Controlled support-aware Z16 decoder comparison; standard library only."""
import argparse
import csv
import itertools
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ring_decoding_experiment import lb, packed, syndrome, weight


@dataclass
class Code:
    name: str
    g: list
    valuations: tuple
    r: int = 4

    def __post_init__(self):
        if self.r < 1 or not self.g or len(self.g) != len(self.valuations):
            raise ValueError('Require positive precision and one valuation per generator')
        k, n = len(self.g), len(self.g[0])
        if n < k or any(len(row) != n for row in self.g):
            raise ValueError('Generator rows must have a common length at least K')
        if any(not 0 <= v < self.r for v in self.valuations):
            raise ValueError('Generator valuations must be in [0,r)')
        if any(row[j] % 2**self.r != int(i == j)
               for i, row in enumerate(self.g) for j in range(k)):
            raise ValueError('This benchmark helper requires an unscaled mother [I_K | A]')

    @property
    def n(self):
        return len(self.g[0])

    def words(self):
        return [self.word(u) for u in itertools.product(*[
            range(2 ** (self.r-v)) for v in self.valuations])]

    def word(self, u):
        return [sum(x * 2**v * row[j] for x, v, row in
                    zip(u, self.valuations, self.g)) % 2**self.r
                for j in range(self.n)]

    def parity(self, a):
        k = len(self.g)
        rows = []
        for j in range(k, self.n):
            row = [-self.g[i][j] % 2**a for i in range(k)] + [0]*(self.n-k)
            row[j] = 1
            rows.append(row)
        for i, v in enumerate(self.valuations):
            exponent = max(v-(self.r-a), 0)
            if exponent:
                row = [0]*self.n
                row[i] = 2**(a-exponent)
                rows.append(row)
        return rows

    def field_check(self, digit):
        active = [i for i, v in enumerate(self.valuations) if v <= digit]
        rows = []
        for j in range(self.n):
            if j not in active:
                row = [0]*self.n
                row[j] = 1
                for i in active:
                    row[i] = self.g[i][j] % 2
                rows.append(row)
        return rows, active


def binary_distance(g, active):
    if not active:
        return None
    return min(weight([sum(bits[i]*g[j][c] for i, j in enumerate(active)) % 2
                       for c in range(len(g[0]))])
               for bits in itertools.product((0, 1), repeat=len(active)) if any(bits))


def examples():
    base = [[int(i == j) for j in range(3)] +
            [(j >> i) & 1 for j in range(1, 8)] for i in range(3)]
    geometries = [('simplex10', base)]
    # Independently generated systematic binary tails; reject distance < 5.
    for n, seed in ((11, 7411), (12, 7412)):
        rng = random.Random(seed)
        while True:
            g = [[int(i == j) for j in range(3)] +
                 [rng.randrange(2) for _ in range(n-3)] for i in range(3)]
            if binary_distance(g, range(3)) >= 5:
                break
        for row in g:
            for j in range(3, n):
                row[j] += 2*rng.randrange(8)
        geometries.append((f'random{n}', g))
    types = [('free', (0, 0, 0)), ('mild', (0, 0, 1)),
             ('mixed', (0, 1, 3)), ('torsion', (1, 2, 3))]
    return [Code(f'{name}_{label}', g, v) for name, g in geometries
            for label, v in types]


class Prepared:
    """Cache only code-dependent matrices and row transformations, never solutions."""
    def __init__(self, h, canonical, stats):
        m, n = len(h), len(h[0])
        rows = [packed(row) for row in h]
        transforms = [1 << i for i in range(m)]
        rank = 0
        if canonical:
            for col in range(n):
                pivot = next((i for i in range(rank, m) if rows[i] >> col & 1), None)
                if pivot is None:
                    continue
                rows[rank], rows[pivot] = rows[pivot], rows[rank]
                transforms[rank], transforms[pivot] = transforms[pivot], transforms[rank]
                for i in range(m):
                    if i != rank and rows[i] >> col & 1:
                        rows[i] ^= rows[rank]
                        transforms[i] ^= transforms[rank]
                        stats['preprocess_packed_xors'] += 2
                rank += 1
        else:
            rank = m
        self.canonical = canonical
        self.transforms = transforms
        self.rank = rank
        self.h = [[row >> j & 1 for j in range(n)] for row in rows[:rank]]
        self.columns = [packed(row[j] for row in self.h) for j in range(n)]
        stats['prepared_matrices'] += 1
        stats['packed_columns'] += n
        self.storage_bits = n*rank + (m*m if canonical else 0)

    def target(self, b, stats):
        if not self.canonical:
            return b
        target = packed(b)
        out = []
        for mask in self.transforms:
            stats['transform_parity_evaluations'] += 1
            stats['transform_scalar_xors'] += max(mask.bit_count()-1, 0)
            out.append((mask & target).bit_count() & 1)
        if any(out[self.rank:]):
            raise ValueError('Inconsistent redundant parity equations')
        return out[:self.rank]


def prepare(code, method, canonical):
    stats = Counter()
    stages = []
    for i in range(code.r):
        h, active = ((code.parity(i+1), None) if method == 'projection'
                     else code.field_check(i))
        stages.append((h, active, Prepared(h, canonical, stats)))
    stats['cache_payload_bits'] = sum(p.storage_bits +
                                     sum(len(row) for row in h)*code.r
                                     for h, _, p in stages)
    return stages, stats


def dot(h, x, modulus, stats):
    # Nominal dense modular arithmetic counts, not CPU instructions or bit costs.
    label = 'binary' if modulus == 2 else 'ring'
    stats[label+'_dot_terms'] += sum(len(row) for row in h)
    return syndrome(h, x, modulus)


def decode(code, y, method, vmax, case_seed, stages, t=2):
    stats, per_stage = Counter(), []
    e, residual, known = [0]*code.n, list(y), set()
    for i, (h, active, prepared) in enumerate(stages):
        local = Counter()
        if method == 'projection':
            scale, modulus = 2**i, 2**(i+1)
            # One syndrome of y-e instead of two independent matrix products.
            b = dot(h, [(x-z) % modulus for x, z in zip(y, e)], modulus, local)
            if any(x % scale for x in b):
                raise ValueError('Candidate does not lift')
            b = [x//scale for x in b]
        else:
            b = dot(h, residual, 2, local)
        b = prepared.target(b, local)
        # A fresh per-stage stream avoids drift from earlier retry counts.
        rng = random.Random(case_seed*17+i)
        z = lb(prepared.h, b, known, t-len(known), rng, local, vmax,
               cached_columns=prepared.columns)
        e = [x+2**i*d for x, d in zip(e, z)]
        if method == 'supercode':
            digit = [(x-d) % 2 for x, d in zip(residual, z)]
            modulus = 2**(code.r-i)
            lift = [sum(digit[k]*code.g[k][j] for k in active) % modulus
                    for j in range(code.n)]
            local['ring_dot_terms'] += code.n*len(active)
            residual = [((x-d-c) % modulus)//2
                        for x, d, c in zip(residual, z, lift)]
        known.update(j for j, x in enumerate(e) if x)
        stats.update(local)
        per_stage.append(dict(local))
    return e, stats, per_stage


def cases(code, count, seed):
    rng = random.Random(seed)
    result = []
    for index in range(count):
        e = [0]*code.n
        for j in rng.sample(range(code.n), 2):
            e[j] = rng.randrange(1, 2**code.r)
        c = code.word([rng.randrange(2**(code.r-v)) for v in code.valuations])
        result.append(([(x+z) % 2**code.r for x, z in zip(c, e)], e, seed*1000003+index))
    return result


METRICS = ('candidates', 'kernel_packed_xors', 'information_sets',
           'transform_scalar_xors', 'binary_dot_terms', 'ring_dot_terms')


def evaluate(code, samples, method, vmax, canonical, cache, keep=False):
    start = time.perf_counter()
    prepared, prep = prepare(code, method, canonical)
    setup_seconds = time.perf_counter()-start
    prep_total = prep.copy() if cache == 'warm' else Counter()
    records, totals, stage_totals = [], Counter(), [Counter() for _ in range(code.r)]
    failures = []
    start = time.perf_counter()
    for index, (y, expected, case_seed) in enumerate(samples):
        if cache == 'cold':
            prepared, fresh = prepare(code, method, canonical)
            prep_total.update({k: v for k, v in fresh.items() if k != 'cache_payload_bits'})
        try:
            answer, stats, per_stage = decode(code, y, method, vmax, case_seed, prepared)
            if answer != expected:
                raise ValueError('Incorrect decoded error')
        except (ValueError, RuntimeError) as exc:
            failures.append({'index': index, 'error': str(exc)})
            continue
        totals.update(stats)
        for total, current in zip(stage_totals, per_stage):
            total.update(current)
        records.append({k: stats[k] for k in METRICS})
    elapsed = time.perf_counter()-start
    # Never rank successful subsets after a failure.
    means = ({k: totals[k]/len(samples) for k in METRICS} if not failures else None)
    report = {'successes': len(records), 'failures': failures, 'means': means,
              'preprocessing': dict(prep_total), 'one_setup': dict(prep),
              'setup_seconds': setup_seconds, 'online_batch_seconds': elapsed,
              'stage_totals': [dict(s) for s in stage_totals]}
    return report, records if keep else None


def paired(left, right):
    out = {}
    for metric in METRICS:
        differences = [a[metric]-b[metric] for a, b in zip(left, right)]
        out[metric] = {'mean_projection_minus_supercode': statistics.mean(differences),
                       'projection_wins': sum(d < 0 for d in differences),
                       'supercode_wins': sum(d > 0 for d in differences),
                       'ties': sum(d == 0 for d in differences)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=100)
    parser.add_argument('--test', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--output', type=Path, default=Path('ring_suite_results'))
    args = parser.parse_args()
    if min(args.train, args.test) < 1:
        parser.error('Training and test sizes must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for code in examples():
        words = code.words()
        distance = min(weight(c) for c in words if any(c))
        assert len({tuple(c) for c in words}) == len(words)
        assert all(not any(syndrome(code.parity(code.r), c, 16)) for c in words)
        layer_distances = [binary_distance(code.g, [j for j, v in
                          enumerate(code.valuations) if v <= i]) for i in range(code.r)]
        assert distance >= 5 and all(d is None or d >= 5 for d in layer_distances)
        metadata = {'name': code.name, 'ring': 'Z16', 'r': code.r, 'n': code.n,
                    'subtype': [code.valuations.count(i) for i in range(code.r)],
                    'valuations': code.valuations, 'base_generator': code.g,
                    'code_size': len(words), 'distance': distance,
                    'supercode_layer_distances': layer_distances,
                    'projection_digit_distance': binary_distance(code.g, range(len(code.g))),
                    'distance_verification': 'exhaustive codeword enumeration',
                    'train_size': args.train, 'test_size_per_seed': args.test}
        runs = []
        for seed in args.seeds:
            train = cases(code, args.train, seed+1000000)
            test = cases(code, args.test, seed+2000000)
            for canonical in (False, True):
                for cache in ('cold', 'warm'):
                    result, records, training = {}, {}, {}
                    for method in ('projection', 'supercode'):
                        training[method] = {}
                        for vmax in (0, 1, 2):
                            training[method][vmax], _ = evaluate(
                                code, train, method, vmax, canonical, cache)
                            result[method, vmax], records[method, vmax] = evaluate(
                                code, test, method, vmax, canonical, cache, keep=True)
                    # Select separately for each declared metric on training only.
                    selections = {}
                    for metric in ('candidates', 'kernel_packed_xors'):
                        selections[metric] = {}
                        for method in ('projection', 'supercode'):
                            eligible = [v for v, r in training[method].items() if not r['failures']]
                            if not eligible:
                                raise RuntimeError(f'No valid training setting: {code.name} {method}')
                            selections[metric][method] = min(eligible, key=lambda v:
                                (training[method][v]['means'][metric], v))
                    comparisons = [(f'fixed_v{v}', v, v) for v in (0, 1, 2)]
                    comparisons += [(f'tuned_{metric}', s['projection'], s['supercode'])
                                    for metric, s in selections.items()]
                    comparison_reports = {}
                    for label, vp, vs in comparisons:
                        p, s = result['projection', vp], result['supercode', vs]
                        valid = not p['failures'] and not s['failures']
                        comparison_reports[label] = (paired(records['projection', vp],
                                                           records['supercode', vs]) if valid else None)
                        row = {'code': code.name, 'n': code.n, 'subtype': str(metadata['subtype']),
                               'distance': distance, 'seed': seed,
                               'matrix': 'rref' if canonical else 'raw', 'cache': cache,
                               'comparison': label, 'projection_v': vp, 'supercode_v': vs,
                               'projection_successes': p['successes'], 'supercode_successes': s['successes']}
                        if valid:
                            for metric in METRICS:
                                row['projection_'+metric] = p['means'][metric]
                                row['supercode_'+metric] = s['means'][metric]
                        summary.append(row)
                    runs.append({'seed': seed, 'canonical': canonical, 'cache': cache,
                                 'training': training, 'selections': selections,
                                 'test': {f'{m}_v{v}': r for (m, v), r in result.items()},
                                 'paired': comparison_reports})
        (args.output / (code.name+'.json')).write_text(
            json.dumps({'code': metadata, 'runs': runs}, indent=2), encoding='utf-8')
        print(f'{code.name}: n={code.n}, subtype={metadata["subtype"]}, d={distance}', flush=True)
    fields = list(dict.fromkeys(k for row in summary for k in row))
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    main()
