"""Held-out paired CCP/binary versus native Z4 Stern and MMT benchmark."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time
import tracemalloc

import numpy as np

from lee_extended import composition_probability, distance_certificate, prepare, sample_error
from lee_list_isd import Parameters, clear_word_cache, decode
from lee_z4_dcc_comparison import lee, raw_syndrome

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/lee_sweep_20261103/subtypes_main/codes.json'
K0 = (0, 3, 6, 9, 12)


def seed_for(*items):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, items)).encode()).digest()[:8], 'big')


def patterns(seed, t, mother, samples):
    result = []
    for a in range(t//2+1):
        for index in range(samples):
            rng = random.Random(seed_for(seed, 'error', t, mother, a, index))
            error = sample_error(64, t, rng, a)
            result.append(dict(a=a, case=index, error=error,
                               probability=float(composition_probability(64, t, a))))
    return result


def read_panel(args):
    archived = json.loads(SOURCE.read_text(encoding='utf-8'))
    for code in archived:
        code['mother_index'] = int(code['cluster'].rsplit('_m', 1)[1])
    rows = [c for c in archived if c['t'] == 4 and c['k0'] in K0
            and c['mother_index'] < args.mothers+args.tune_mothers]
    if len(rows) != len(K0)*(args.mothers+args.tune_mothers):
        raise ValueError('Not enough archived independent mother matrices')
    for row in rows:
        if row['d_binary'] <= 16:
            raise AssertionError('Distance certificate does not cover both radii')
    return rows


def run_one(prepared, syndrome, t, seed, method, parameters, args, track=True,
            tuning=False):
    clear_word_cache()
    return decode(prepared, syndrome, t, seed, method.removesuffix('_family_best'), parameters, track=track,
                  seconds=args.tune_seconds if tuning else args.seconds,
                  trials=args.tune_trials if tuning else args.trials,
                  entries=args.entries, checks=args.checks)


def grid(algorithm, t):
    if algorithm == 'stern':
        return [Parameters(l, p) for p in range(min(2, t//2)+1)
                for l in (0, 2, 4) if p > 0 or l == 0]
    return [Parameters(l, w1, w2, split)
            for w1, w2 in ((0, 0), (1, 0), (1, 1))
            for l in (0, 2, 4)
            for split in ((0,) if (w1, w2) == (0, 0) else (0, 2))
            if w1 > 0 or l == 0]


def tune(panel, t, algorithm, folder, args):
    path = folder/f'tuning_{algorithm}_t{t}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for k0 in K0:
        codes = [c for c in panel if c['k0'] == k0 and c['mother_index'] < args.tune_mothers]
        prepared = [(c, prepare(c['h'])) for c in codes]
        for method in (f'ccp_{algorithm}', f'lee_{algorithm}'):
            scores = []
            for parameters in grid(algorithm, t):
                cost, failures, calls, successful = 0., 0, 0, 0
                by_composition = defaultdict(list)
                for code, public in prepared:
                    cases = patterns(args.seed, t, code['mother_index'], 1)
                    # Every composition participates in tuning, with exact sphere mass.
                    for case in cases:
                        syndrome = raw_syndrome(code['h'], case['error'])
                        trial_seed = seed_for(args.seed, 'tune', t, code['mother_index'], case['a'])
                        row = run_one(public, syndrome, t, trial_seed, method,
                                      parameters, args, track=True, tuning=True)
                        if row['status'] == 'complete' and row['answer'] != case['error']:
                            raise AssertionError('Wrong tuning result')
                        ok = row['status'] == 'complete'
                        failures += not ok
                        successful += ok
                        calls += 1
                        # Penalized observed budget loss, not an expected-time estimate.
                        loss = row['times']['online_ns'] if ok else max(
                            row['times']['online_ns'], args.tune_seconds*1e9)*10
                        by_composition[case['a']].append(loss)
                cost = sum(float(composition_probability(64, t, a))*statistics.mean(values)
                           for a, values in by_composition.items())
                scores.append(dict(parameters=asdict(parameters), selection_loss_ns=cost,
                                   failed_calls=failures, calls=calls))
            best = min(scores, key=lambda r: r['selection_loss_ns'])
            nondegenerate = [r for r in scores if r['parameters']['weight'] > 0
                             and (algorithm == 'stern' or r['parameters']['weight2'] > 0)]
            best_list = min(nondegenerate, key=lambda r: r['selection_loss_ns'])
            result[f'{k0}:{method}'] = dict(selected=best_list['parameters'],
                family_best=best['parameters'], candidates=scores,
                selection_policy='Minimum exact-composition-weighted penalized budget loss on held-out training mothers.')
            print('tuned', t, k0, method, best['parameters'],
                  'list', best_list['parameters'], flush=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def benchmark(panel, t, algorithm, tuning, folder, args):
    path = folder/f'instances_{algorithm}_t{t}.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done.add((row['mother'], row['k0'], row['a'], row['case'], row['seed_index'], row['method']))
    count = len(done)
    for code in panel:
        mother, k0 = code['mother_index'], code['k0']
        if mother < args.tune_mothers:
            continue
        start = time.perf_counter_ns()
        public = prepare(code['h'])
        preprocessing_ns = time.perf_counter_ns()-start
        cases = patterns(args.seed, t, mother, args.samples)
        for case in cases:
            syndrome = raw_syndrome(code['h'], case['error'])
            for index in range(args.seeds):
                methods = [f'ccp_{algorithm}', f'lee_{algorithm}',
                           f'ccp_{algorithm}_family_best', f'lee_{algorithm}_family_best']
                if (mother+k0+case['a']+case['case']+index) % 2:
                    methods.reverse()
                trial_seed = seed_for(args.seed, 'confirm', t, mother, case['a'], case['case'], index)
                outputs = {}
                # Alternate full method order between repeats; no timed code/error cache.
                for repeat in range(args.repeats+1):
                    order = methods if repeat % 2 == 0 else methods[::-1]
                    for method in order:
                        key = (mother, k0, case['a'], case['case'], index, method)
                        if key in done:
                            continue
                        base = method.removesuffix('_family_best')
                        choice = 'family_best' if method.endswith('_family_best') else 'selected'
                        parameters = Parameters(**tuning[f'{k0}:{base}'][choice])
                        row = run_one(public, syndrome, t, trial_seed, method, parameters,
                                      args, track=repeat == 0)
                        if row['status'] == 'complete' and row['answer'] != case['error']:
                            raise AssertionError('Wrong confirmed result')
                        if repeat == 0:
                            outputs[method] = dict(counted=row, timings=[], statuses=[])
                        else:
                            outputs[method]['timings'].append(row['times'])
                            outputs[method]['statuses'].append(row['status'])
                with path.open('a', encoding='utf-8') as handle:
                    for method, output in outputs.items():
                        row = output['counted']
                        status = row['status']
                        if any(s != status for s in output['statuses']):
                            status = 'unstable_budget'
                        samples = output['timings']
                        record = dict(mother=mother, k0=k0, n=64, K=12, t=t,
                            code=code['code'], d_binary=code['d_binary'], method=method,
                            seed_index=index, search_seed=trial_seed,
                            status=status, reason=row['reason'], correct=row['answer'] == case['error'],
                            parameters=row['parameters'], counters=row['counters'],
                            preprocessing_ns=preprocessing_ns, **case,
                            times={key: statistics.median(v.get(key, 0) for v in samples)
                                   for key in set().union(*(v.keys() for v in samples))},
                            timing_samples=samples)
                        handle.write(json.dumps(record)+'\n')
                        done.add((mother, k0, case['a'], case['case'], index, method))
                        count += 1
        print('confirmed', algorithm, t, 'mother', mother, 'k0', k0, 'records', count, flush=True)


def memory_audit(panel, t, algorithm, tuning, folder, args):
    path = folder/f'memory_{algorithm}_t{t}.json'
    if path.exists():
        return
    rows = []
    mother = args.tune_mothers
    for code in panel:
        if code['mother_index'] != mother:
            continue
        public = prepare(code['h'])
        for case in patterns(args.seed, t, mother, 1):
            for method in (f'ccp_{algorithm}', f'lee_{algorithm}'):
                parameters = Parameters(**tuning[f'{code["k0"]}:{method}']['selected'])
                clear_word_cache()
                tracemalloc.start()
                result = decode(public, raw_syndrome(code['h'], case['error']), t,
                    seed_for(args.seed, 'confirm', t, mother, case['a'], case['case'], 0),
                    method, parameters, seconds=max(30, args.seconds*10), trials=args.trials,
                    entries=args.entries, checks=args.checks)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if result['status'] == 'complete' and result['answer'] != case['error']:
                    raise AssertionError('Wrong memory-audit result')
                rows.append(dict(k0=code['k0'], t=t, method=method, a=case['a'],
                    status=result['status'], peak_traced_bytes=peak,
                    peak_live_list_entries=result['counters'].get('peak_live_list_entries', 0)))
    path.write_text(json.dumps(rows, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/lee_list_isd_20261105')
    parser.add_argument('--radii', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--algorithms', nargs='+', choices=['stern', 'mmt'], default=['stern', 'mmt'])
    parser.add_argument('--mothers', type=int, default=8)
    parser.add_argument('--tune-mothers', type=int, default=2)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--tune-seconds', type=float, default=.25)
    parser.add_argument('--trials', type=int, default=4096)
    parser.add_argument('--tune-trials', type=int, default=512)
    parser.add_argument('--entries', type=int, default=200000)
    parser.add_argument('--checks', type=int, default=2000000)
    args = parser.parse_args()
    folder = ROOT/args.output
    folder.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(python=platform.python_version(), numpy=np.__version__,
        platform=platform.platform(), source=str(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        hashes={name: hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                for name in ('lee_list_isd.py', 'run_lee_list_comparison.py',
                             'lee_extended.py', 'lee_revision.py', 'ring_constraints.py')},
        policy='Held-out mothers; exact-sphere composition weighting; cold word-pattern caches; failed work retained.',
        schedule='Every fourth trial uses zero base-list weights; no oracle projected weight. Primary curves choose nonzero-weight parameters; family_best may degenerate to Prange.',
        memory_policy='Common live-list entry cap, not equal byte caps. Separate tracemalloc measurements; not used for timing.',
        preprocessing_policy='CCP Smith preparation outside online time. Native methods only read H/unit rank; not charged Smith.',
        scope='Finite parameter-grid reference implementations, not globally optimized best known ISD.')
    path = folder/'config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
        raise ValueError('Source/config mismatch: choose a new output directory')
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    panel = read_panel(args)
    # Re-certify each mother once, independently of the archived metadata.
    certificates = {}
    for code in panel:
        mother = code['mother_index']
        if mother not in certificates:
            distance = distance_certificate(code['ambient'][:12])
            if distance != code['d_binary'] or distance <= 16:
                raise AssertionError('Distance re-certification failed')
            certificates[mother] = distance
    (folder/'distance_audit.json').write_text(json.dumps(certificates, indent=2), encoding='utf-8')
    for algorithm in args.algorithms:
        for t in args.radii:
            tuning = tune(panel, t, algorithm, folder, args)
            benchmark(panel, t, algorithm, tuning, folder, args)
            memory_audit(panel, t, algorithm, tuning, folder, args)


if __name__ == '__main__':
    main()
