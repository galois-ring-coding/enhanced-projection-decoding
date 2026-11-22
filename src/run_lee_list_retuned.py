"""Independent training and held-out scale validation for Lee list decoders."""
import argparse
from collections import Counter
from dataclasses import asdict
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
from run_lee_list_confirmation import digest, matrix_digest, save_immutable

ROOT = Path(__file__).resolve().parents[1]
# (length, K, exact Lee weight, free subtype, kernel). Controls are fixed too.
TARGETS = (
    (64, 12, 4, 3, 'mmt'), (64, 12, 4, 6, 'mmt'),
    (64, 12, 8, 9, 'mmt'), (64, 12, 8, 6, 'stern'),
    (64, 12, 8, 9, 'stern'), (64, 12, 4, 6, 'stern'),
    (96, 18, 6, 5, 'mmt'), (96, 18, 6, 9, 'mmt'),
    (96, 18, 12, 14, 'mmt'), (96, 18, 12, 9, 'stern'),
    (96, 18, 12, 14, 'stern'), (96, 18, 6, 9, 'stern'),
)


def target_id(target):
    return ':'.join(map(str, target))


def allocation(a):
    return (4, 2) if a == 0 else (2, 2) if a == 1 else (1, 2)


def grid(algorithm, radius):
    if algorithm == 'stern':
        return [Parameters()] + [Parameters(window, weight)
            for weight in range(1, min(3, radius//2)+1)
            for window in (0, 2, 4, 6)]
    weights = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]
    return [Parameters()] + [Parameters(window, w1, w2, split)
        for w1, w2 in weights if 2*(w1+w2) <= radius
        for window in (0, 2, 4, 6) for split in (0, 2)]


def choose(scores, algorithm, primary):
    eligible = [r for r in scores if not primary or (
        r['parameters']['weight'] > 0 and
        (algorithm == 'stern' or r['parameters']['weight2'] > 0))]
    # Completion on every training stratum takes precedence over time.
    return min(eligible, key=lambda r: (r['failed_calls'], r['selection_loss_ns']))['parameters']


def expected_records(config):
    return config['test_mothers']*sum(4*sum(e*s for e, s in
        map(allocation, range(t//2+1))) for _, _, t, _, _ in config['targets'])


def make_public(mother, k0):
    n, K, dual = mother['n'], mother['K'], mother['dual']
    h = dual[K:]+[[2*x % 4 for x in row] for row in dual[k0:K]]
    for j, vector in enumerate(mother['ambient'][:K]):
        word = vector if j < k0 else [2*x % 4 for x in vector]
        if any(raw_syndrome(h, word)):
            raise AssertionError('Generator/check mismatch')
    begin = time.perf_counter_ns()
    public = prepare(h)
    elapsed = time.perf_counter_ns()-begin
    if (public['n'], public['K'], public['k0']) != (n, K, k0):
        raise AssertionError('Public normal form has wrong subtype')
    return public, elapsed


def build_mothers(folder, config):
    path = folder/'mothers.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    excluded = set()
    for name in ('lee_sweep_20261103/subtypes_main/codes.json',
                 'lee_list_confirmation_20261105/mothers.json'):
        archived = json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
        excluded.update(matrix_digest(row['ambient']) for row in archived)
    rows = []
    sizes = sorted({(n, K) for n, K, _, _, _ in config['targets']})
    for n, K in sizes:
        max_t = max(t for nn, _, t, _, _ in config['targets'] if nn == n)
        for role, count in (('train', config['train_mothers']), ('test', config['test_mothers'])):
            for index in range(count):
                rng = random.Random(seed_for(config['seed'], 'ambient', n, role, index))
                for attempt in range(10000):
                    ambient = [[rng.randrange(4) for _ in range(n)] for _ in range(n)]
                    try:
                        base = make_code(ambient, K, 0)
                    except ValueError:
                        continue
                    distance = distance_certificate(ambient[:K])
                    fingerprint = matrix_digest(ambient)
                    if distance > 2*max_t and fingerprint not in excluded:
                        break
                else:
                    raise RuntimeError('Mother generation budget exhausted')
                excluded.add(fingerprint)
                rows.append(dict(n=n, K=K, role=role, mother=index, ambient=ambient,
                    dual=base['dual'], sha256=fingerprint, d_binary=distance,
                    attempts=attempt+1))
                print('certified', n, role, index+1, '/', count,
                      'distance', distance, flush=True)
    save_immutable(path, rows)
    return rows


def case(mother, radius, a, index, config, role):
    error_seed = seed_for(config['seed'], role, 'error', mother['n'],
                          mother['mother'], radius, a, index)
    error = sample_error(mother['n'], radius, random.Random(error_seed), a)
    if lee(error) != radius or error.count(2) != a:
        raise AssertionError('Wrong error composition')
    return error_seed, error


def call(public, syndrome, radius, seed, method, params, config, training=False, track=False):
    clear_word_cache()
    return decode(public, syndrome, radius, seed, method, Parameters(**params), track=track,
        seconds=config['tune_seconds'] if training else config['seconds'],
        trials=config['tune_trials'] if training else config['trials'],
        entries=config['entries'], checks=config['checks'])


def tune(folder, mothers, config):
    summary = folder/'tuning.json'
    if summary.exists():
        return json.loads(summary.read_text(encoding='utf-8'))
    path = folder/'training.jsonl'
    measured = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = (row['target'], row['method'], row['candidate'], row['mother'], row['a'])
            if key in measured:
                raise AssertionError('Duplicate training checkpoint')
            measured[key] = row
    started = time.perf_counter()
    result = {}
    for target in config['targets']:
        n, K, radius, k0, algorithm = target
        ident = target_id(target)
        candidates = grid(algorithm, radius)
        training = [m for m in mothers if m['n'] == n and m['role'] == 'train']
        for mother in training:
            public, _ = make_public(mother, k0)
            for a in range(radius//2+1):
                error_seed, error = case(mother, radius, a, 0, config, 'train')
                syndrome = raw_syndrome(public['h'], error)
                seed = seed_for(config['seed'], 'train-search', n, mother['mother'], radius, a)
                # Rotate parameter order; pair native and CCP at each parameter.
                order = list(range(len(candidates)))
                random.Random(seed_for(seed, ident, 'order')).shuffle(order)
                for candidate in order:
                    sides = ('ccp', 'lee') if (candidate+mother['mother']+a) % 2 else ('lee', 'ccp')
                    for side in sides:
                        method = side+'_'+algorithm
                        key = (ident, method, candidate, mother['mother'], a)
                        if key in measured:
                            continue
                        params = asdict(candidates[candidate])
                        output = call(public, syndrome, radius, seed, method, params, config, training=True)
                        correct = output['answer'] == error
                        if output['status'] == 'complete' and not correct:
                            raise AssertionError('Wrong training answer')
                        row = dict(target=ident, method=method, candidate=candidate,
                            mother=mother['mother'], a=a, error_seed=error_seed,
                            search_seed=seed, error=error, syndrome=syndrome,
                            parameters=params, status=output['status'], reason=output['reason'],
                            correct=correct, times=output['times'])
                        with path.open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(row)+'\n')
                        measured[key] = row
            print('trained', ident, 'mother', mother['mother'], 'calls', len(measured),
                  'elapsed_s', round(time.perf_counter()-started), flush=True)
        for side in ('ccp', 'lee'):
            method = side+'_'+algorithm
            scores = []
            for candidate, params in enumerate(candidates):
                block = [measured[ident, method, candidate, m['mother'], a]
                         for m in training for a in range(radius//2+1)]
                loss, failures = 0., 0
                for row in block:
                    ok = row['status'] == 'complete'
                    failures += not ok
                    value = row['times']['online_ns'] if ok else max(
                        row['times']['online_ns'], config['tune_seconds']*1e9)*10
                    loss += float(composition_probability(n, radius, row['a']))*value/len(training)
                scores.append(dict(parameters=asdict(params), failed_calls=failures,
                                   selection_loss_ns=loss, calls=len(block)))
            result[ident+':'+method] = dict(primary=choose(scores, algorithm, True),
                family=choose(scores, algorithm, False), candidates=scores)
        print('selected', ident, {side: result[ident+':'+side+'_'+algorithm]['primary']
                                 for side in ('ccp', 'lee')}, flush=True)
    save_immutable(summary, result)
    return result


def benchmark(folder, mothers, tuning, config):
    path = folder/'instances.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = (row['target'], row['mother'], row['a'], row['case'], row['seed_index'], row['method'], row['variant'])
            if key in done:
                raise AssertionError('Duplicate test checkpoint')
            done.add(key)
    started = time.perf_counter()
    for mother in (m for m in mothers if m['role'] == 'test'):
        publics = {}
        for target in config['targets']:
            n, K, radius, k0, algorithm = target
            if n != mother['n']:
                continue
            ident = target_id(target)
            if k0 not in publics:
                publics[k0] = make_public(mother, k0)
            public, preprocessing_ns = publics[k0]
            for a in range(radius//2+1):
                errors, seeds = allocation(a)
                for index in range(errors):
                    error_seed, error = case(mother, radius, a, index, config, 'test')
                    syndrome = raw_syndrome(public['h'], error)
                    for seed_index in range(seeds):
                        seed = seed_for(config['seed'], 'test-search', n, mother['mother'],
                                        radius, a, index, seed_index)
                        pending = []
                        for side in ('ccp', 'lee'):
                            method = side+'_'+algorithm
                            for variant in ('primary', 'family'):
                                key = (ident, mother['mother'], a, index, seed_index, method, variant)
                                if key not in done:
                                    pending.append((method, variant, key))
                        if seed % 2:
                            pending.reverse()
                        outputs = {}
                        for repeat in range(config['repeats']+1):
                            for method, variant, key in (pending if repeat % 2 == 0 else pending[::-1]):
                                params = tuning[ident+':'+method][variant]
                                output = call(public, syndrome, radius, seed, method, params,
                                              config, track=repeat == 0)
                                if output['status'] == 'complete' and output['answer'] != error:
                                    raise AssertionError('Wrong held-out answer')
                                if repeat == 0:
                                    outputs[method, variant] = dict(counted=output, timings=[], statuses=[])
                                else:
                                    outputs[method, variant]['timings'].append(output['times'])
                                    outputs[method, variant]['statuses'].append(output['status'])
                        with path.open('a', encoding='utf-8') as handle:
                            for method, variant, key in pending:
                                output = outputs[method, variant]
                                counted, samples = output['counted'], output['timings']
                                stable = all(s == counted['status'] for s in output['statuses'])
                                row = dict(target=ident, n=n, K=K, t=radius, k0=k0,
                                    algorithm=algorithm, mother=mother['mother'], a=a, case=index,
                                    seed_index=seed_index, error_seed=error_seed, search_seed=seed,
                                    error=error, syndrome=syndrome, d_binary=mother['d_binary'],
                                    method=method, variant=variant, parameters=counted['parameters'],
                                    preprocessing_ns=preprocessing_ns, status=counted['status'] if stable else 'unstable_budget',
                                    correct=counted['answer'] == error, reason=counted['reason'],
                                    counters=counted['counters'], timing_samples=samples,
                                    times={k: statistics.median(s.get(k, 0) for s in samples)
                                           for k in set().union(*(s.keys() for s in samples))})
                                handle.write(json.dumps(row)+'\n')
                                done.add(key)
            print('tested', n, 'mother', mother['mother'], ident, 'records', len(done),
                  'elapsed_s', round(time.perf_counter()-started), flush=True)
    if len(done) != config['expected_records']:
        raise AssertionError('Incomplete held-out panel')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/lee_list_retuned_20261105')
    parser.add_argument('--train-mothers', type=int, default=8)
    parser.add_argument('--test-mothers', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026110504)
    parser.add_argument('--tune-seconds', type=float, default=.5)
    parser.add_argument('--tune-trials', type=int, default=1024)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--trials', type=int, default=16384)
    parser.add_argument('--entries', type=int, default=200000)
    parser.add_argument('--checks', type=int, default=2000000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.train_mothers, args.test_mothers, args.repeats) < 1:
        raise ValueError('Positive train/test sample counts and repetitions required')
    targets = ((16, 4, 2, 2, 'stern'), (16, 4, 2, 2, 'mmt')) if args.smoke else TARGETS
    # At t=2 nonzero MMT halves are infeasible, so smoke uses t=4 instead.
    if args.smoke:
        targets = ((32, 6, 4, 3, 'stern'), (32, 6, 4, 3, 'mmt'))
    config = dict(vars(args), targets=[list(t) for t in targets],
        allocation={str(a): list(allocation(a)) for a in range(7)},
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_hashes={name: digest(ROOT/'src'/name) for name in
            ('run_lee_list_retuned.py', 'lee_list_isd.py', 'lee_extended.py',
             'lee_revision.py', 'ring_constraints.py', 'run_lee_list_comparison.py')},
        grids={target_id(t): [asdict(p) for p in grid(t[-1], t[2])] for t in targets},
        policy='Independent training and test mothers; frozen targets and grids; exact Lee-sphere weighting; no outcome-based stopping.',
        distance_condition='For each length condition on d_H(D1)>2*max(t) over its targets.',
        selection='Minimize failed training calls first, then exact-sphere-weighted penalized online time; separately freeze nonzero-list and full-family parameters.',
        schedule='Every fourth selection uses zero list weights in every method; residual radius and free capacity can lower list weights.',
        timing='Cold word cache; one counted run and three uninstrumented alternating-order repetitions; median per input.',
        preprocessing='Code generation, distance certification and CCP Smith preparation excluded from online times; native methods are not charged Smith.',
        inference='Paired mother bootstrap; simultaneous 95 percent intervals across both variants and all targets within each length.')
    config['expected_records'] = expected_records(config)
    folder = ROOT/args.output
    folder.mkdir(parents=True, exist_ok=True)
    save_immutable(folder/'config.json', config)
    mothers = build_mothers(folder, config)
    tuning = tune(folder, mothers, config)
    benchmark(folder, mothers, tuning, config)
    rows = [json.loads(s) for s in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    save_immutable(folder/'audit.json', dict(records=len(rows), expected_records=config['expected_records'],
        statuses=dict(Counter(r['status'] for r in rows)),
        correct_completed=sum(r['status'] == 'complete' and r['correct'] for r in rows),
        timed_calls=sum(len(r['timing_samples']) for r in rows),
        training_mothers=config['train_mothers'], test_mothers=config['test_mothers'],
        lengths=sorted({t[0] for t in targets})))


if __name__ == '__main__':
    main()
