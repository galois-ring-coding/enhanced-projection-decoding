"""Matched standard-rejection Lee searches at K/n=3/16 and t/n=1/16."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np

from lee_extended import distance_certificate, prepare, sample_error
from lee_fixed_kernel import STANDARD_METHODS, decode
from lee_revision import make_code
from lee_z4_dcc_comparison import lee, raw_syndrome
from run_lee_fixed import coverage_diagnostics
from run_lee_independent import ROOT, digest, matrix_digest, stream_seed

METHODS = STANDARD_METHODS
PROFILES = {48: (9, 3), 64: (12, 4), 96: (18, 6)}
SOURCES = ('run_lee_scale_confirmation.py', 'run_lee_independent.py', 'run_lee_fixed.py',
           'lee_fixed_kernel.py', 'lee_extended.py', 'lee_revision.py',
           'lee_z4_dcc_comparison.py', 'ring_constraints.py')


def seed_for(seed, n, mother, label):
    return stream_seed(seed, mother, f'scale:n{n}:{label}')


def error_pool(seed, n, t, mother):
    result = []
    for population, a in [(f'a{a}', a) for a in range(t//2+1)]+[('uniform', None)]:
        for case in range(2):
            error_seed = seed_for(seed, n, mother, f'error:{population}:{case}')
            error = sample_error(n, t, random.Random(error_seed), a)
            result.append(dict(population=population, case=case, error_seed=error_seed,
                               error=error, a=error.count(2), s=sum(v % 2 for v in error)))
    return result


def search_seed(seed, n, mother, population, case, index):
    return seed_for(seed, n, mother, f'search:{population}:{case}:{index}')


def build_panel(folder, config):
    path = folder/'codes.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    n, k, t = (config[v] for v in ('n', 'K', 't'))
    result, fingerprints = [], set()
    for mother in range(config['mothers']):
        seed = seed_for(config['seed'], n, mother, 'ambient')
        rng = random.Random(seed)
        for attempt in range(10000):
            ambient = [[rng.randrange(4) for _ in range(n)] for _ in range(n)]
            try:
                base = make_code(ambient, k, 0)
            except ValueError:
                continue
            distance = distance_certificate(ambient[:k])
            if distance > 4*t:
                break
        else:
            raise RuntimeError('Mother generation budget exhausted')
        fingerprint = matrix_digest(ambient)
        if fingerprint in fingerprints:
            raise AssertionError('Repeated mother')
        fingerprints.add(fingerprint)
        cluster = f'scale_n{n}_m{mother:02d}'
        for k0 in range(k+1):
            h = base['dual'][k:]+[[2*x % 4 for x in row] for row in base['dual'][k0:k]]
            result.append(dict(n=n, K=k, t=t, k0=k0, h=h, ambient=ambient,
                code=f'{cluster}_k0{k0}', cluster=cluster, mother_index=mother,
                seed=seed, d_binary=distance, attempts=attempt+1, ambient_sha256=fingerprint))
        print(f'n={n} certified mother {mother+1}/{config["mothers"]}: d={distance}, attempts={attempt+1}', flush=True)
    path.write_text(json.dumps(result), encoding='utf-8')
    return result


def benchmark(folder, n, args):
    folder.mkdir(parents=True, exist_ok=True)
    k, t = PROFILES[n]
    config = dict(n=n, K=k, t=t, mothers=args.mothers, seed=args.seed, seeds=args.seeds,
        repeats=args.repeats, trials=args.trials, candidates=args.candidates,
        methods=list(METHODS), kernel='common', singular_policy='reject',
        distance_condition=f'd_H(D1)>{4*t}',
        errors_per_composition=2, uniform_errors_per_mother=2,
        source_hashes={name: digest(ROOT/'src'/name) for name in SOURCES},
        timing_policy='Counted warm-up; alternating pair order; median of uninstrumented repeats.',
        sampling_policy='SHA256-derived ambient/error/search streams, shared across subtypes and methods.',
        uniform_policy='Independent exact-sphere validation, one seed per error.')
    path = folder/'config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
        raise ValueError('Resume source/configuration mismatch')
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    environment = folder/'environment.json'
    if not environment.exists():
        environment.write_text(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
            platform=platform.platform(), processor=platform.processor()), indent=2), encoding='utf-8')
    codes = build_panel(folder, config)
    pools = {c['cluster']: error_pool(args.seed, n, t, c['mother_index']) for c in codes}
    pool_path = folder/'error_pools.json'
    if pool_path.exists() and json.loads(pool_path.read_text(encoding='utf-8')) != pools:
        raise AssertionError('Cached inputs changed')
    pool_path.write_text(json.dumps(pools, indent=2), encoding='utf-8')
    output, done, outcomes = folder/'instances.jsonl', set(), Counter()
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = tuple(row[v] for v in ('code', 'population', 'case', 'seed_index', 'method'))
            if key in done:
                raise AssertionError('Duplicate checkpoint row')
            done.add(key)
            outcomes[row['status']] += 1
    started = time.perf_counter()
    for code_index, code in enumerate(codes):
        prepared = None
        for case_index, case in enumerate(pools[code['cluster']]):
            error = case['error']
            if lee(error) != t:
                raise AssertionError('Wrong error weight')
            syndrome = raw_syndrome(code['h'], error)
            for index in range(1 if case['population'] == 'uniform' else args.seeds):
                pending = [m for m in METHODS if (code['code'], case['population'], case['case'], index, m) not in done]
                if not pending:
                    continue
                if prepared is None:
                    for i, vector in enumerate(code['ambient'][:k]):
                        generator = vector if i < code['k0'] else [2*x % 4 for x in vector]
                        if any(raw_syndrome(code['h'], generator)):
                            raise AssertionError('Generator/check mismatch')
                    prepared = prepare(code['h'])
                    if (prepared['K'], prepared['k0']) != (k, code['k0']):
                        raise AssertionError('Public subtype mismatch')
                trial_seed = search_seed(args.seed, n, code['mother_index'], case['population'], case['case'], index)
                results, samples = {}, defaultdict(list)
                order = pending if (code_index+case_index+index) % 2 == 0 else pending[::-1]
                for method in order:
                    result = decode(prepared, syndrome, t, trial_seed, method, args.trials, args.candidates)
                    if result['status'] == 'complete' and result['answer'] != error:
                        raise AssertionError('Incorrect completed result')
                    result['counters'].update(coverage_diagnostics(result, error))
                    results[method] = result
                for repeat in range(args.repeats):
                    order = pending if (code_index+case_index+index+repeat) % 2 == 0 else pending[::-1]
                    for method in order:
                        start = time.perf_counter_ns()
                        measured = decode(prepared, syndrome, t, trial_seed, method,
                                          args.trials, args.candidates, track=False)
                        elapsed = time.perf_counter_ns()-start
                        if (measured['status'], measured['answer']) != (results[method]['status'], results[method]['answer']):
                            raise AssertionError('Timed/counted mismatch')
                        samples[method].append(dict(online_ns=elapsed, **measured['times']))
                with output.open('a', encoding='utf-8') as handle:
                    for method in pending:
                        result = results[method]
                        row = {v: code[v] for v in ('code', 'cluster', 'n', 'K', 'k0', 't', 'd_binary')}
                        row.update(case, syndrome=syndrome, conditioning_radius=2*t,
                            method=method, kernel='common', seed_index=index, trial_seed=trial_seed,
                            status=result['status'], reason=result['reason'], correct=result['answer'] == error,
                            **result['counters'])
                        row['recovery_trials'] = row.get('second_trials', 0)
                        for metric in samples[method][0]:
                            row[metric] = statistics.median(s[metric] for s in samples[method])
                        row.update(timing_samples=samples[method], trace=result['trace'])
                        handle.write(json.dumps(row)+'\n')
                        done.add((code['code'], case['population'], case['case'], index, method))
                        outcomes[row['status']] += 1
        print(f'n={n} {code_index+1}/{len(codes)} {code["code"]}: records={len(done)}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    main = args.mothers*(k+1)*2*(t//2+1)*args.seeds
    uniform = args.mothers*(k+1)*2
    if len(done) != (main+uniform)*2:
        raise AssertionError('Incomplete schedule')
    summary = dict(status='measured', n=n, K=k, t=t, mothers=args.mothers, codes=len(codes),
        records=len(done), timing_calls=len(done)*args.repeats, main_paired_trajectories=main,
        uniform_paired_trajectories=uniform, outcomes=dict(outcomes))
    (folder/'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/lee_scale_rejection_20261119')
    parser.add_argument('--lengths', type=int, nargs='+', choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument('--mothers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026110423)
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trials', type=int, default=16384)
    parser.add_argument('--candidates', type=int, default=100000)
    args = parser.parse_args()
    if min(args.mothers, args.seeds, args.repeats, args.trials, args.candidates) < 1:
        parser.error('Counts must be positive')
    for n in args.lengths:
        benchmark(args.output.resolve()/f'n{n}', n, args)


if __name__ == '__main__':
    main()
