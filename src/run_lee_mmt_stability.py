"""Predeclared MMT retuning with replicated training and fresh test codes."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import time

import numpy as np

from lee_extended import composition_probability
from lee_z4_dcc_comparison import raw_syndrome
from run_lee_list_comparison import seed_for
from run_lee_list_confirmation import digest, matrix_digest, save_immutable
from run_lee_list_retuned import (ROOT, TARGETS as OLD_TARGETS, benchmark,
    build_mothers, call, case, choose, expected_records, make_public, target_id)

TARGETS = tuple(t for t in OLD_TARGETS if t[-1] == 'mmt')
BASE = ROOT/'results/lee_list_retuned_20261105'


def training_allocation(a):
    return 2, 2


def shortlist(tuning, target, count=2):
    """Pool the best old training candidates from both sides and both policies."""
    ident = target_id(target)
    pool = {(0, 0, 0, 0)}
    for side in ('ccp', 'lee'):
        scores = tuning[ident+':'+side+'_mmt']['candidates']
        for positive in (False, True):
            eligible = [r for r in scores if not positive or
                        r['parameters']['weight'] > 0 and r['parameters']['weight2'] > 0]
            for row in sorted(eligible, key=lambda r: (r['failed_calls'], r['selection_loss_ns']))[:count]:
                p = row['parameters']
                pool.add(tuple(p[k] for k in ('window', 'weight', 'weight2', 'split')))
    return [dict(zip(('window', 'weight', 'weight2', 'split'), p)) for p in sorted(pool)]


def score(block, n, radius, mothers, seconds):
    loss, failures = 0., 0
    for row in block:
        ok = row['status'] == 'complete'
        failures += not ok
        cost = row['times']['online_ns'] if ok else max(row['times']['online_ns'], seconds*1e9)*10
        e, s = training_allocation(row['a'])
        loss += float(composition_probability(n, radius, row['a']))*cost/(mothers*e*s)
    return dict(failed_calls=failures, selection_loss_ns=loss, calls=len(block))


def train(folder, mothers, config):
    summary = folder/'tuning.json'
    if summary.exists():
        return json.loads(summary.read_text(encoding='utf-8'))
    path = folder/'training.jsonl'
    measured = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ('target', 'method', 'candidate', 'mother', 'a', 'case', 'seed_index'))
            if key in measured:
                raise AssertionError('Duplicate training checkpoint')
            measured[key] = row
    started = time.perf_counter()
    result = {}
    for target in config['targets']:
        n, K, radius, k0, algorithm = target
        ident = target_id(target)
        candidates = config['grids'][ident]
        panel = [m for m in mothers if m['n'] == n and m['role'] == 'train']
        for mother in panel:
            public, _ = make_public(mother, k0)
            for a in range(radius//2+1):
                errors, seeds = training_allocation(a)
                for index in range(errors):
                    error_seed, error = case(mother, radius, a, index, config, 'replicated-train')
                    syndrome = raw_syndrome(public['h'], error)
                    for seed_index in range(seeds):
                        seed = seed_for(config['seed'], 'replicated-search', n, mother['mother'],
                                        radius, a, index, seed_index)
                        order = list(range(len(candidates)))
                        random.Random(seed_for(seed, ident, 'order')).shuffle(order)
                        for candidate in order:
                            sides = ('ccp', 'lee') if (candidate+seed_index+index+mother['mother']) % 2 else ('lee', 'ccp')
                            for side in sides:
                                method = side+'_mmt'
                                key = (ident, method, candidate, mother['mother'], a, index, seed_index)
                                if key in measured:
                                    continue
                                params = candidates[candidate]
                                output = call(public, syndrome, radius, seed, method, params, config, training=True)
                                correct = output['answer'] == error
                                if output['status'] == 'complete' and not correct:
                                    raise AssertionError('Wrong training answer')
                                row = dict(target=ident, method=method, candidate=candidate,
                                    mother=mother['mother'], a=a, case=index, seed_index=seed_index,
                                    error_seed=error_seed, search_seed=seed, error=error, syndrome=syndrome,
                                    parameters=params, status=output['status'], reason=output['reason'],
                                    correct=correct, times=output['times'])
                                with path.open('a', encoding='utf-8') as handle:
                                    handle.write(json.dumps(row)+'\n')
                                measured[key] = row
            print('trained', ident, 'matrix', mother['mother']+1, '/', len(panel),
                  'calls', len(measured), 'elapsed_s', round(time.perf_counter()-started), flush=True)
        for side in ('ccp', 'lee'):
            method = side+'_mmt'
            scores = []
            for candidate, params in enumerate(candidates):
                block = [measured[ident, method, candidate, m['mother'], a, i, s]
                    for m in panel for a in range(radius//2+1)
                    for i in range(training_allocation(a)[0])
                    for s in range(training_allocation(a)[1])]
                scores.append(dict(parameters=params, **score(block, n, radius, len(panel), config['seconds'])))
            result[ident+':'+method] = dict(primary=choose(scores, 'mmt', True),
                family=choose(scores, 'mmt', False), candidates=scores)
        print('frozen', ident, {s: {v: result[ident+':'+s+'_mmt'][v]
            for v in ('primary', 'family')} for s in ('ccp', 'lee')}, flush=True)
    if len(measured) != config['expected_training_records']:
        raise AssertionError('Incomplete training measurements')
    save_immutable(summary, result)
    return result


def expected_training(config):
    return sum(config['train_mothers']*len(config['grids'][target_id(t)])*2*
        sum(e*s for e, s in map(training_allocation, range(t[2]//2+1))) for t in config['targets'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/lee_mmt_stability_20261107')
    parser.add_argument('--train-mothers', type=int, default=8)
    parser.add_argument('--test-mothers', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026110701)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.train_mothers, args.test_mothers, args.repeats) < 1:
        raise ValueError('Positive sample sizes required')
    targets = ((32, 6, 4, 3, 'mmt'),) if args.smoke else TARGETS
    old = json.loads((BASE/'tuning.json').read_text(encoding='utf-8'))
    grids = {target_id(t): ([dict(window=0, weight=0, weight2=0, split=0),
        dict(window=2, weight=1, weight2=1, split=0)] if args.smoke else shortlist(old, t)) for t in targets}
    config = dict(vars(args), targets=[list(t) for t in targets], grids=grids,
        tune_seconds=15, seconds=15, tune_trials=16384, trials=16384, entries=200000, checks=2000000,
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_hashes={name: digest(ROOT/'src'/name) for name in
            ('run_lee_mmt_stability.py', 'run_lee_list_retuned.py', 'run_lee_list_confirmation.py',
             'lee_list_isd.py', 'lee_extended.py', 'lee_revision.py', 'ring_constraints.py',
             'run_lee_list_comparison.py')},
        prior_training_hash=digest(BASE/'training.jsonl'), prior_tuning_hash=digest(BASE/'tuning.json'),
        shortlist_rule='Union of top two previous training candidates per side and per policy, plus all-zero weights. Previous test timings are not used.',
        training='Eight fresh matrices per length; two errors and two search seeds per exact composition; one uninstrumented call per seed; common candidates and budgets on both sides.',
        selection='Minimize failed calls first, then exact-sphere weighted penalized mean online time; freeze primary and full-family choices before testing.',
        policy='Six MMT targets fixed in advance; 32 fresh held-out matrices per length; no optional stopping.',
        timing='Cold word cache; one counted call and three alternating-order uninstrumented repetitions; median per input.',
        inference='Joint paired mother-cluster bootstrap, independent draws by length; simultaneous approximate 95 percent intervals over all six points and both policies.',
        preprocessing='Online costs exclude generation, distance certification and cached CCP Smith preparation; native methods are not charged Smith.')
    config['expected_records'] = expected_records(config)
    config['expected_training_records'] = expected_training(config)
    folder = ROOT/args.output
    folder.mkdir(parents=True, exist_ok=True)
    save_immutable(folder/'config.json', config)
    mothers = build_mothers(folder, config)
    previous = json.loads((BASE/'mothers.json').read_text(encoding='utf-8'))
    seen = {m['sha256'] for m in previous}
    if any(matrix_digest(m['ambient']) in seen for m in mothers):
        raise AssertionError('Fresh matrix duplicates prior training or test data')
    tuning = train(folder, mothers, config)
    benchmark(folder, mothers, tuning, config)
    rows = [json.loads(s) for s in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    save_immutable(folder/'audit.json', dict(records=len(rows), expected_records=config['expected_records'],
        training_records=config['expected_training_records'], statuses=dict(Counter(r['status'] for r in rows)),
        correct_completed=sum(r['status'] == 'complete' and r['correct'] for r in rows),
        timed_calls=sum(len(r['timing_samples']) for r in rows),
        training_mothers=config['train_mothers'], test_mothers=config['test_mothers']))


if __name__ == '__main__':
    main()
