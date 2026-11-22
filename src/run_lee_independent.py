"""Independent-mother confirmation of matched all-solutions Lee decoding."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np

from lee_extended import distance_certificate, prepare, sample_error
from lee_fixed_kernel import decode
from lee_revision import make_code
from lee_z4_dcc_comparison import lee, raw_syndrome
from run_lee_fixed import coverage_diagnostics

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ep_all', 'proj_binary_all')
SOURCES = ('run_lee_independent.py', 'run_lee_fixed.py', 'lee_fixed_kernel.py',
           'lee_extended.py', 'lee_revision.py', 'lee_z4_dcc_comparison.py',
           'ring_constraints.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_digest(matrix):
    return hashlib.sha256(bytes(x for row in matrix for x in row)).hexdigest()


def stream_seed(seed, mother, label):
    payload = f'lee-independent:{seed}:{mother}:{label}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')


def error_pool(seed, mother):
    """Inputs are shared across subtypes; error and search streams are separate."""
    cases = []
    for population, a in [('a0', 0), ('a1', 1), ('a2', 2), ('uniform', None)]:
        for case in range(2):
            error_seed = stream_seed(seed, mother, f'error:{population}:{case}')
            error = sample_error(64, 4, random.Random(error_seed), a)
            cases.append(dict(population=population, case=case, error=error,
                              error_seed=error_seed,
                              a=error.count(2), s=sum(x % 2 for x in error)))
    return cases


def trial_seed(seed, mother, population, case, index):
    return stream_seed(seed, mother, f'search:{population}:{case}:{index}')


def build_panel(folder, args):
    path = folder/'codes.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    old = json.loads((args.archive/'codes.json').read_text(encoding='utf-8'))
    seen = {matrix_digest(c['ambient']) for c in old}
    result = []
    for mother in range(args.mothers):
        code_seed = stream_seed(args.seed, mother, 'ambient')
        rng = random.Random(code_seed)
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
            raise RuntimeError('Independent mother generation budget exhausted')
        fingerprint = matrix_digest(ambient)
        if fingerprint in seen:
            raise AssertionError('Repeated mother matrix')
        seen.add(fingerprint)
        cluster = f'independent_n64_m{mother:02d}'
        for k0 in range(13):
            h = base['dual'][12:]+[[2*x % 4 for x in row] for row in base['dual'][k0:12]]
            result.append(dict(n=64, K=12, k0=k0, t=4, h=h, ambient=ambient,
                cluster=cluster, mother_index=mother, code=f'{cluster}_k0{k0}',
                seed=code_seed, d_binary=distance, attempts=attempt+1,
                ambient_sha256=fingerprint))
        print(f'certified mother {mother+1}/{args.mothers}: distance={distance}, attempts={attempt+1}', flush=True)
    path.write_text(json.dumps(result), encoding='utf-8')
    pools = {c['cluster']: error_pool(args.seed, c['mother_index']) for c in result}
    (folder/'error_pools.json').write_text(json.dumps(pools, indent=2), encoding='utf-8')
    return result


def benchmark(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(n=64, K=12, t=4, distance_condition='d_H(D1)>16',
        mothers=args.mothers, seed=args.seed, errors_per_composition=2,
        uniform_errors_per_mother=2, seeds=args.seeds, repeats=args.repeats,
        methods=list(METHODS), kernel='common', trials=args.trials, candidates=args.candidates,
        archive=str(args.archive.resolve()),
        archive_codes_sha256=digest(args.archive/'codes.json'),
        source_hashes={name: digest(ROOT/'src'/name) for name in SOURCES},
        timing_policy='Counted warm-up, three medians by default; rotated/reversed pair order.',
        sampling_policy='Independent SHA256-derived ambient, error, and search streams; shared across subtypes.',
        uniform_policy='Two separate exact-sphere validation errors, one search seed each.')
    config_path = folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Resume source/configuration mismatch')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    environment = folder/'environment.json'
    if not environment.exists():
        environment.write_text(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
            platform=platform.platform(), processor=platform.processor()), indent=2), encoding='utf-8')
    codes = build_panel(folder, args)
    pools = json.loads((folder/'error_pools.json').read_text(encoding='utf-8'))
    output = folder/'instances.jsonl'
    done = set()
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ('code', 'population', 'case', 'seed_index', 'method'))
            if key in done:
                raise AssertionError('Duplicate checkpoint row')
            done.add(key)
    started = time.perf_counter()
    for code_index, info in enumerate(codes):
        prepared = None
        for case_index, case in enumerate(pools[info['cluster']]):
            error = case['error']
            if lee(error) != 4:
                raise AssertionError('Invalid error weight')
            syndrome = raw_syndrome(info['h'], error)
            count = 1 if case['population'] == 'uniform' else args.seeds
            for index in range(count):
                pending = [m for m in METHODS if (info['code'], case['population'], case['case'], index, m) not in done]
                if not pending:
                    continue
                if prepared is None:
                    for i, vector in enumerate(info['ambient'][:12]):
                        generator = vector if i < info['k0'] else [2*x % 4 for x in vector]
                        if any(raw_syndrome(info['h'], generator)):
                            raise AssertionError('Generator/check mismatch')
                    prepared = prepare(info['h'])
                    if (prepared['K'], prepared['k0']) != (12, info['k0']):
                        raise AssertionError('Public subtype mismatch')
                seed = trial_seed(args.seed, info['mother_index'], case['population'], case['case'], index)
                results, samples = {}, defaultdict(list)
                shift = (code_index+case_index+index) % len(pending)
                counted_order = pending[shift:]+pending[:shift]
                for method in counted_order:
                    result = decode(prepared, syndrome, 4, seed, method, args.trials,
                                    args.candidates, kernel='common')
                    if result['status'] == 'complete' and result['answer'] != error:
                        raise AssertionError('Incorrect completed result')
                    result['counters'].update(coverage_diagnostics(result, error))
                    results[method] = result
                for repeat in range(args.repeats):
                    shift = (code_index+case_index+index+repeat) % len(pending)
                    order = pending[shift:]+pending[:shift]
                    if (code_index+index+repeat) % 2:
                        order.reverse()
                    for method in order:
                        start = time.perf_counter_ns()
                        measured = decode(prepared, syndrome, 4, seed, method, args.trials,
                                          args.candidates, track=False, kernel='common')
                        elapsed = time.perf_counter_ns()-start
                        if (measured['status'], measured['answer']) != (results[method]['status'], results[method]['answer']):
                            raise AssertionError('Timed/counted trajectory mismatch')
                        samples[method].append(dict(online_ns=elapsed, **measured['times']))
                with output.open('a', encoding='utf-8') as handle:
                    for method in pending:
                        result = results[method]
                        row = {key: info[key] for key in ('code', 'cluster', 'n', 'K', 'k0', 't', 'd_binary')}
                        row.update({key: case[key] for key in ('population', 'case', 'error', 'a', 's', 'error_seed')})
                        row.update(syndrome=syndrome, conditioning_radius=8, kernel='common', method=method,
                            seed_index=index, trial_seed=seed, status=result['status'], reason=result['reason'],
                            correct=result['answer'] == error, **result['counters'])
                        row['recovery_trials'] = row.get('second_trials', 0)
                        for metric in set().union(*(s.keys() for s in samples[method])):
                            row[metric] = statistics.median(s.get(metric, 0) for s in samples[method])
                        row.update(timing_samples=samples[method], trace=result['trace'])
                        handle.write(json.dumps(row)+'\n')
                        done.add((info['code'], case['population'], case['case'], index, method))
        print(f'{code_index+1}/{len(codes)} {info["code"]}: records={len(done)}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    expected = args.mothers*13*(6*args.seeds+2)*2
    if len(done) != expected:
        raise AssertionError('Incomplete schedule')
    summary = dict(status='measured', records=len(done), codes=len(codes), mothers=args.mothers,
        unique_planted_errors=args.mothers*8, main_paired_trajectories=args.mothers*13*6*args.seeds,
        uniform_paired_trajectories=args.mothers*13*2, timing_calls=len(done)*args.repeats)
    (folder/'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/lee_independent_20261104')
    parser.add_argument('--archive', type=Path, default=ROOT/'results/lee_sweep_20261103/subtypes_main')
    parser.add_argument('--mothers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026110417)
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trials', type=int, default=16384)
    parser.add_argument('--candidates', type=int, default=100000)
    args = parser.parse_args()
    if min(args.mothers, args.seeds, args.repeats, args.trials, args.candidates) < 1:
        parser.error('Counts must be positive')
    benchmark(args)


if __name__ == '__main__':
    main()
