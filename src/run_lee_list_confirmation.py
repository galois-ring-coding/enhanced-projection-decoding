"""Independent fixed-size confirmation of eight inconclusive list-ISD pairs.

The original nonzero-list parameters and decoder implementation are frozen.
New matrices, errors and search seeds have disjoint deterministic streams.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np

from lee_extended import composition_probability, distance_certificate, prepare, sample_error
from lee_list_isd import Parameters, clear_word_cache, decode
from lee_revision import make_code
from lee_z4_dcc_comparison import lee, raw_syndrome
from run_lee_list_comparison import seed_for

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((4, 6, 'stern'), (4, 9, 'stern'), (8, 6, 'stern'), (8, 9, 'stern'),
           (4, 3, 'mmt'), (4, 6, 'mmt'), (8, 9, 'mmt'), (8, 12, 'mmt'))


def allocation(a):
    """Predeclared stratification, including every composition with nonzero mass."""
    return (4, 4) if a == 0 else (3, 3) if a == 1 else (1, 2)


def expected_records(mothers):
    return mothers*sum(2*sum(e*s for e, s in map(allocation, range(t//2+1)))
                       for t, _, _ in TARGETS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_digest(matrix):
    return hashlib.sha256(bytes(x for row in matrix for x in row)).hexdigest()


def save_immutable(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Immutable configuration mismatch: '+str(path))
    else:
        path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def frozen_parameters(base):
    result = {}
    for t, k0, algorithm in TARGETS:
        tuning = json.loads((base/f'tuning_{algorithm}_t{t}.json').read_text(encoding='utf-8'))
        for side in ('ccp', 'lee'):
            method = f'{side}_{algorithm}'
            result[f'{t}:{k0}:{method}'] = tuning[f'{k0}:{method}']['selected']
    return result


def build_mothers(folder, config):
    path = folder/'mothers.json'
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if len(cached) != config['mothers']:
            raise AssertionError('Incomplete cached mother panel')
        return cached
    archived = json.loads((ROOT/'results/lee_sweep_20261103/subtypes_main/codes.json').read_text(encoding='utf-8'))
    seen = {matrix_digest(c['ambient']) for c in archived}
    mothers = []
    for index in range(config['mothers']):
        rng = random.Random(seed_for(config['seed'], 'fresh-ambient', index))
        for attempt in range(10000):
            ambient = [[rng.randrange(4) for _ in range(64)] for _ in range(64)]
            try:
                base = make_code(ambient, 12, 0)
            except ValueError:
                continue
            distance = distance_certificate(ambient[:12])
            if distance > 16:
                break
        else:
            raise RuntimeError('Mother generation budget exhausted')
        fingerprint = matrix_digest(ambient)
        if fingerprint in seen:
            raise AssertionError('Mother duplicates a previous matrix')
        seen.add(fingerprint)
        mothers.append(dict(mother=1000+index, ambient=ambient, dual=base['dual'],
                            d_binary=distance, attempts=attempt+1, sha256=fingerprint))
        print('new certified mother', index+1, '/', config['mothers'], 'distance', distance, flush=True)
    path.write_text(json.dumps(mothers), encoding='utf-8')
    return mothers


def make_public(mother, k0):
    dual = mother['dual']
    h = dual[12:]+[[2*x % 4 for x in row] for row in dual[k0:12]]
    for i, vector in enumerate(mother['ambient'][:12]):
        generator = vector if i < k0 else [2*x % 4 for x in vector]
        if any(raw_syndrome(h, generator)):
            raise AssertionError('Generator/check inconsistency')
    public = prepare(h)
    if (public['K'], public['k0']) != (12, k0):
        raise AssertionError('Public subtype mismatch')
    return public


def run(folder, mothers, config, parameters):
    path = folder/'instances.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ('t', 'k0', 'mother', 'a', 'case', 'seed_index', 'method'))
            if key in done:
                raise AssertionError('Duplicate checkpoint')
            done.add(key)
    started = time.perf_counter()
    # Interleave all targets within a mother to avoid time-of-day/target confounding.
    for mother in mothers:
        public_cache = {}
        for t, k0, algorithm in TARGETS:
            public = public_cache.setdefault(k0, None)
            if public is None:
                public = make_public(mother, k0)
                public_cache[k0] = public
            for a in range(t//2+1):
                samples, seeds = allocation(a)
                for case in range(samples):
                    error_seed = seed_for(config['seed'], 'fresh-error', mother['mother'], t, a, case)
                    error = sample_error(64, t, random.Random(error_seed), a)
                    if lee(error) != t or error.count(2) != a:
                        raise AssertionError('Invalid error composition')
                    syndrome = raw_syndrome(public['h'], error)
                    for index in range(seeds):
                        seed = seed_for(config['seed'], 'fresh-search', mother['mother'], t, a, case, index)
                        methods = [f'ccp_{algorithm}', f'lee_{algorithm}']
                        if seed % 2:
                            methods.reverse()
                        prefix = (t, k0, mother['mother'], a, case, index)
                        pending = [m for m in methods if prefix+(m,) not in done]
                        if not pending:
                            continue
                        outputs = {}
                        for repeat in range(config['repeats']+1):
                            for method in (pending if repeat % 2 == 0 else pending[::-1]):
                                params = Parameters(**parameters[f'{t}:{k0}:{method}'])
                                clear_word_cache()
                                result = decode(public, syndrome, t, seed, method, params,
                                    track=repeat == 0, seconds=config['seconds'],
                                    trials=config['trials'], entries=config['entries'], checks=config['checks'])
                                if result['status'] == 'complete' and result['answer'] != error:
                                    raise AssertionError('Wrong confirmed error')
                                if repeat == 0:
                                    outputs[method] = dict(counted=result, samples=[], statuses=[])
                                else:
                                    outputs[method]['samples'].append(result['times'])
                                    outputs[method]['statuses'].append(result['status'])
                        with path.open('a', encoding='utf-8') as handle:
                            for method, output in outputs.items():
                                counted, timings = output['counted'], output['samples']
                                stable = all(status == counted['status'] for status in output['statuses'])
                                record = dict(n=64, K=12, t=t, k0=k0, method=method,
                                    mother=mother['mother'], code=f'fresh_m{mother["mother"]}_k0{k0}',
                                    d_binary=mother['d_binary'], a=a, case=case, seed_index=index,
                                    error_seed=error_seed, search_seed=seed, error=error, syndrome=syndrome,
                                    probability=float(composition_probability(64, t, a)),
                                    status=counted['status'] if stable else 'unstable_budget',
                                    reason=counted['reason'], correct=counted['answer'] == error,
                                    parameters=counted['parameters'], counters=counted['counters'],
                                    times={key: statistics.median(sample.get(key, 0) for sample in timings)
                                           for key in set().union(*(sample.keys() for sample in timings))},
                                    timing_samples=timings)
                                handle.write(json.dumps(record)+'\n')
                                done.add(prefix+(method,))
            print('confirmed fresh mother', mother['mother'], 'target', (t, k0, algorithm),
                  'records', len(done), 'elapsed_s', round(time.perf_counter()-started), flush=True)
    if len(done) != config['expected_records']:
        raise AssertionError('Confirmation panel incomplete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='results/lee_list_isd_20261105')
    parser.add_argument('--output', default='results/lee_list_confirmation_20261105')
    parser.add_argument('--mothers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026110502)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--trials', type=int, default=16384)
    parser.add_argument('--entries', type=int, default=200000)
    parser.add_argument('--checks', type=int, default=2000000)
    args = parser.parse_args()
    if args.mothers < 1 or args.repeats < 1:
        raise ValueError('Positive samples and repeats required')
    base = ROOT/args.base
    original = json.loads((base/'config.json').read_text(encoding='utf-8'))
    for name, expected in original['hashes'].items():
        if digest(ROOT/'src'/name) != expected:
            raise AssertionError('Original source changed: '+name)
    parameters = frozen_parameters(base)
    config = dict(vars(args), n=64, K=12, targets=[list(t) for t in TARGETS],
        allocation={str(a): list(allocation(a)) for a in range(5)},
        expected_records=expected_records(args.mothers),
        frozen_parameters=parameters, base_config_sha256=digest(base/'config.json'),
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_hashes={name: digest(ROOT/'src'/name) for name in
            ('run_lee_list_confirmation.py', 'lee_list_isd.py', 'lee_extended.py',
             'lee_revision.py', 'run_lee_list_comparison.py', 'ring_constraints.py')},
        distance_condition='d_H(D1)>16; same distribution as the original pilot',
        analysis='Primary inference uses ONLY new mothers; eight targets frozen before measurement. Paired mother bootstrap with simultaneous intervals; no significance-based stopping.',
        equivalence_margin=[1/1.1, 1.1],
        timing_policy='Cold word cache, alternating pair order, median of two uninstrumented repetitions. Counted run separate.',
        scope='Original nonzero-list parameters only; no retuning or kernel selection on new outcomes.')
    folder = ROOT/args.output
    folder.mkdir(parents=True, exist_ok=True)
    save_immutable(folder/'config.json', config)
    mothers = build_mothers(folder, config)
    for mother in mothers:
        if matrix_digest(mother['ambient']) != mother['sha256']:
            raise AssertionError('Matrix fingerprint mismatch')
        if distance_certificate(mother['ambient'][:12]) != mother['d_binary']:
            raise AssertionError('Distance certificate mismatch')
    run(folder, mothers, config, parameters)
    rows = [json.loads(line) for line in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    save_immutable(folder/'audit.json', dict(records=len(rows), expected_records=config['expected_records'],
        statuses=dict(Counter(r['status'] for r in rows)),
        correct_completed=sum(r['status'] == 'complete' and r['correct'] for r in rows),
        timed_calls=sum(len(r['timing_samples']) for r in rows), new_mothers=len(mothers),
        paired_cases=len(rows)//2))


if __name__ == '__main__':
    main()
