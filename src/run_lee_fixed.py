"""Reuse archived n=64,K=12,t=4 codes/errors; vary search seeds and kernels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from lee_extended import distance_certificate, prepare
from lee_fixed_kernel import LEGACY, MATCHED, decode
from lee_z4_dcc_comparison import lee, raw_syndrome

ROOT = Path(__file__).resolve().parents[1]
FOCUS = (4, 5, 6, 7, 8, 12)
SOURCES = ('lee_fixed_kernel.py', 'run_lee_fixed.py', 'lee_extended.py',
           'lee_revision.py', 'lee_z4_dcc_comparison.py', 'ring_constraints.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(row, seeds, focus_seeds):
    count = 1 if row['population'] == 'uniform' else (
        focus_seeds if row['k0'] in FOCUS else seeds)
    for index in range(count):
        trial_seed = row['trial_seed'] if index == 0 else row['trial_seed']+index*1000000000039
        yield index, trial_seed, 'common', MATCHED
        if index == 0:
            yield index, trial_seed, 'legacy', LEGACY


def coverage_diagnostics(result, error):
    counts = Counter()
    low = {j for j, value in enumerate(error) if value % 2}
    rest = [j for j in range(len(error)) if j not in low]
    high = {i for i, j in enumerate(rest) if error[j] == 2}
    for event in result.get('trace') or []:
        support = low if event['phase'] == 'first' else high
        covered = support.issubset(event['chosen'])
        event['covers_true_support'] = covered
        if covered:
            counts[event['phase']+'_covered_trials'] += 1
            if event['nullity']:
                counts[event['phase']+'_covered_singular_trials'] += 1
            if not event['consistent']:
                raise AssertionError('A covered selection must be consistent')
    return counts


def benchmark(args):
    source = args.source.resolve()
    folder = args.output.resolve()
    if folder == source or source in folder.parents:
        raise ValueError('Use a separate output directory')
    folder.mkdir(parents=True, exist_ok=True)
    audit = json.loads((source/'audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed':
        raise AssertionError('Archived panel is not audited')
    codes = [c for c in json.loads((source/'codes.json').read_text(encoding='utf-8'))
             if (c['n'], c['K'], c['t']) == (64, 12, 4)]
    if args.limit_codes:
        codes = codes[:args.limit_codes]
    names = {c['code'] for c in codes}
    originals, archived = defaultdict(list), {}
    with (source/'instances.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['code'] not in names:
                continue
            key = row['code'], row['population'], row['case']
            archived[*key, row['method']] = row
            if row['method'] == 'ep_all':
                originals[row['code']].append(row)
    config = dict(n=64, K=12, t=4, source=str(source), seeds=args.seeds,
                  focus_seeds=args.focus_seeds, focus_k0=FOCUS, repeats=args.repeats,
                  trials=args.trials, candidates=args.candidates, limit_codes=args.limit_codes,
                  common_methods=MATCHED, legacy_methods=LEGACY,
                  source_hashes={name: digest(ROOT/'src'/name) for name in SOURCES},
                  input_hashes={name: digest(source/name) for name in
                                ('codes.json', 'instances.jsonl', 'audit.json')},
                  conditioning='Inherited d_H(D1)>16; no new codes or errors.',
                  seed_policy='Original seed at index 0; add index*1000000000039 thereafter.',
                  legacy_policy='All five archived methods remeasured at seed index 0 only.',
                  uniform_policy='Archived uniform samples are validation only, seed index 0.')
    config = json.loads(json.dumps(config))
    config_path = folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Resume source/configuration mismatch')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    (folder/'environment.json').write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        processor=platform.processor()), indent=2), encoding='utf-8')
    output = folder/'instances.jsonl'
    done = set()
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ('code', 'population', 'case', 'seed_index', 'kernel', 'method'))
            if key in done:
                raise AssertionError('Duplicate checkpoint row')
            done.add(key)
    certificates = {}
    for code_index, info in enumerate(codes):
        if info['cluster'] not in certificates:
            distance = distance_certificate(info['ambient'][:12])
            if distance != info['d_binary'] or distance <= 16:
                raise AssertionError('Archived distance mismatch')
            certificates[info['cluster']] = distance
        for i, vector in enumerate(info['ambient'][:12]):
            vector = vector if i < info['k0'] else [2*x % 4 for x in vector]
            if any(raw_syndrome(info['h'], vector)):
                raise AssertionError('Archived generator/check mismatch')
        prepared = None
        for case_index, original in enumerate(originals[info['code']]):
            if lee(original['error']) != 4 or raw_syndrome(info['h'], original['error']) != original['syndrome']:
                raise AssertionError('Archived error/syndrome mismatch')
            for seed_index, trial_seed, kernel, methods in schedule(original, args.seeds, args.focus_seeds):
                pending = [m for m in methods if (info['code'], original['population'],
                           original['case'], seed_index, kernel, m) not in done]
                if not pending:
                    continue
                if prepared is None:
                    prepared = prepare(info['h'])
                    if (prepared['K'], prepared['k0']) != (12, info['k0']):
                        raise AssertionError('Public subtype mismatch')
                results, samples = {}, defaultdict(list)
                for method in pending:
                    result = decode(prepared, original['syndrome'], 4, trial_seed, method,
                                    args.trials, args.candidates, kernel=kernel)
                    if result['status'] == 'complete' and result['answer'] != original['error']:
                        raise AssertionError('Incorrect completed result')
                    result['counters'].update(coverage_diagnostics(result, original['error']))
                    if seed_index == 0:
                        old = archived[info['code'], original['population'], original['case'], method]
                        for field in ('first_trials', 'first_candidates', 'second_trials', 'second_candidates', 'ring_trials'):
                            if result['counters'].get(field, 0) != old.get(field, 0):
                                raise AssertionError(('Archived trajectory changed', field, kernel, method))
                    results[method] = result
                # Rotation and reversal spread timing order across all methods.
                for repeat in range(args.repeats):
                    shift = (code_index+case_index+seed_index+repeat) % len(pending)
                    order = pending[shift:]+pending[:shift]
                    if (code_index+seed_index+repeat) % 2:
                        order.reverse()
                    for method in order:
                        start = time.perf_counter_ns()
                        measured = decode(prepared, original['syndrome'], 4, trial_seed, method,
                                          args.trials, args.candidates, track=False, kernel=kernel)
                        elapsed = time.perf_counter_ns()-start
                        if (measured['status'], measured['answer']) != (
                                results[method]['status'], results[method]['answer']):
                            raise AssertionError('Timed/counting mismatch')
                        samples[method].append(dict(online_ns=elapsed, **measured['times']))
                with output.open('a', encoding='utf-8') as handle:
                    for method in pending:
                        result = results[method]
                        row = {key: original[key] for key in ('code', 'cluster', 'n', 'K', 'k0', 't',
                               'population', 'case', 'error', 'syndrome', 'a', 's', 'd_binary', 'conditioning_radius')}
                        row.update(method=method, kernel=kernel, seed_index=seed_index,
                                   trial_seed=trial_seed, status=result['status'], reason=result['reason'],
                                   correct=result['answer'] == original['error'], **result['counters'])
                        row['recovery_trials'] = row.get('second_trials', 0)+row.get('ring_trials', 0)
                        metrics = set().union(*(sample.keys() for sample in samples[method]))
                        for metric in sorted(metrics):
                            row[metric] = statistics.median(sample.get(metric, 0) for sample in samples[method])
                        row['timing_samples'] = samples[method]
                        row['trace'] = result.get('trace')
                        handle.write(json.dumps(row)+'\n')
                        done.add((info['code'], original['population'], original['case'], seed_index, kernel, method))
        print(f'{code_index+1}/{len(codes)} {info["code"]} records={len(done)}', flush=True)
    expected = sum(len(methods) for c in codes for row in originals[c['code']]
                   for _, _, _, methods in schedule(row, args.seeds, args.focus_seeds))
    if len(done) != expected:
        raise AssertionError('Incomplete schedule')
    summary = dict(status='measured', records=len(done), codes=len(codes),
                   mothers=len(certificates), archived_inputs=sum(map(len, originals.values())),
                   timing_calls=len(done)*args.repeats, new_codes=0, new_errors=0,
                   distance_certificates=certificates)
    (folder/'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/lee_sweep_20261103/subtypes_main')
    parser.add_argument('--output', type=Path, default=ROOT/'results/lee_fixed_20261104')
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--focus-seeds', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trials', type=int, default=16384)
    parser.add_argument('--candidates', type=int, default=100000)
    parser.add_argument('--limit-codes', type=int, default=0)
    args = parser.parse_args()
    if min(args.seeds, args.focus_seeds, args.repeats, args.trials, args.candidates) < 1:
        parser.error('Counts must be positive')
    if args.focus_seeds < args.seeds:
        parser.error('Focus seed count cannot be smaller')
    benchmark(args)


if __name__ == '__main__':
    main()
