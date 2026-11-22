"""Matched restricted-matrix kernels for the fixed Z4 Lee diagnostic panel."""
import random
import time
from collections import Counter

import numpy as np

from lee_extended import BudgetExceeded, decode as legacy_decode
from lee_revision import charge, eliminate
from lee_z4_dcc_comparison import Arithmetic, lee

STANDARD_METHODS = ('ep_reject', 'proj_binary_t')
MATCHED = (*STANDARD_METHODS, 'ep_all', 'proj_binary_all')
LEGACY = ('proj_ring_t', *MATCHED)


def restricted_rref(h, target, chosen, op):
    """Both policies finish exactly the same elimination before branching."""
    rows = np.array([[row[j] for j in chosen]+[b]
                     for row, b in zip(h, target)], dtype=np.int64)
    pivots = []
    for col in range(len(chosen)):
        rank = len(pivots)
        candidates = np.flatnonzero(rows[rank:, col])
        if not len(candidates):
            continue
        pivot = rank+int(candidates[0])
        rows[[rank, pivot]] = rows[[pivot, rank]]
        others = np.arange(len(rows)) != rank
        rows[others] = (rows[others]-rows[others, col, None]*rows[rank]) % 2
        count = (len(rows)-1)*(len(chosen)+1)
        charge(op, adds=count, mults=count)
        pivots.append(col)
    rows = rows.tolist()
    consistent = not any(row[-1] for row in rows[len(pivots):])
    return rows, pivots, consistent


def affine_solutions(rows, pivots, width, op):
    free = [j for j in range(width) if j not in pivots]
    for mask in range(1 << len(free)):
        local = [0]*width
        for bit, j in enumerate(free):
            local[j] = (mask >> bit) & 1
        for row, pivot in zip(rows, pivots):
            local[pivot] = op.add(row[-1], op.dot(
                [row[j] for j in free], [local[j] for j in free]))
        yield local


def search(h, target, radius, rng, op, trials, candidates, enumerate_singular,
           timings, phase, trace=None):
    n, size = len(h[0]), len(h)
    if not any(target):
        op.event('zero_syndrome_shortcuts')
        return [0]*n
    emitted = 0
    for _ in range(trials):
        op.event('trials')
        chosen = rng.sample(range(n), size)
        start = time.perf_counter_ns()
        rows, pivots, consistent = restricted_rref(h, target, chosen, op)
        timings[phase+'_linear_ns'] += time.perf_counter_ns()-start
        defect = size-len(pivots)
        op.event('rank_defect_'+str(defect))
        if trace is not None:
            trace.append(dict(phase=phase, chosen=chosen, rank=len(pivots),
                              nullity=defect, consistent=consistent))
        if defect:
            op.event('singular_trials')
        if not consistent:
            op.event('inconsistent_trials')
            continue
        if defect:
            op.event('consistent_singular_trials')
            if not enumerate_singular:
                continue
        else:
            op.event('full_rank_trials')
        op.event('potential_list_solutions', 1 << defect)
        for local in affine_solutions(rows, pivots, size, op):
            if emitted >= candidates:
                raise BudgetExceeded('candidate_budget')
            emitted += 1
            op.event('candidates')
            if sum(local) <= radius:
                answer = [0]*n
                for j, value in zip(chosen, local):
                    answer[j] = value
                return answer
    raise BudgetExceeded('trial_budget')


def decode(prepared, syndrome, t, seed, method, trials=16384,
           candidates=100000, track=True, kernel='common'):
    if kernel == 'legacy':
        return legacy_decode(prepared, syndrome, t, seed, method, trials, candidates, track)
    if kernel != 'common' or method not in MATCHED:
        raise ValueError('Unknown kernel or method')
    ops = {name: Arithmetic(bits, track) for name, bits in
           (('first', 1), ('second', 1), ('ring', 2), ('verify', 2))}
    first, second, ring, verify = (ops[name] for name in ('first', 'second', 'ring', 'verify'))
    timings = Counter()
    trace = [] if track else None
    enumeration = method.endswith('_all')
    answer, status, reason = None, 'complete', None
    phase, start = 'first', time.perf_counter_ns()
    try:
        transformed = [ring.dot(row, syndrome) for row in prepared['u']]
        if any(transformed[len(prepared['normalized']):]):
            raise ValueError('Inconsistent zero row')
        transformed = transformed[:len(prepared['normalized'])]
        units = prepared['units']
        if any(x % 2 for x in transformed[units:]):
            raise ValueError('Inconsistent torsion syndrome')
        target = [x % 2 for x in transformed[:units]]
        is_ep = method.startswith('ep_')
        if is_ep:
            target += [x//2 for x in transformed[units:]]
        h0 = prepared['h0'] if is_ep else prepared['h1']
        low = search(h0, target, t, random.Random(seed*17), first, trials, candidates,
                     enumeration, timings, phase, trace)
        timings['first_ns'] = time.perf_counter_ns()-start
        phase, start = 'recovery', time.perf_counter_ns()
        known = [j for j, value in enumerate(low) if value]
        radius = (t-len(known))//2
        if radius < 0:
            raise ValueError('Invalid recovered digit')
        residual = [ring.add(b, -ring.dot(row, low))
                    for row, b in zip(prepared['normalized'], transformed)]
        if any(x % 2 for x in residual) or any(residual[units:]):
            raise ValueError('Incompatible low digit')
        target1 = [x//2 for x in residual[:units]]
        rest = [j for j in range(prepared['n']) if j not in known]
        begin = time.perf_counter_ns()
        rows = eliminate(prepared['h1'], target1, known, second)
        timings['known_elimination_ns'] = time.perf_counter_ns()-begin
        if rows is None:
            raise ValueError('Dependent known support')
        reduced = [[row[j] for j in rest] for row in rows[len(known):]]
        beta = [row[-1] for row in rows[len(known):]]
        if radius == 0:
            if any(beta):
                raise ValueError('Inconsistent direct recovery')
            second.event('zero_budget_shortcuts')
            outside = [0]*len(rest)
        else:
            outside = search(reduced, beta, radius, random.Random(seed*17+1), second,
                             trials, candidates, enumeration, timings, phase, trace)
        high = [0]*prepared['n']
        for j, value in zip(rest, outside):
            high[j] = value
        for i, j in enumerate(known):
            high[j] = second.add(rows[i][-1], -second.dot(
                [rows[i][k] for k in rest], outside))
        answer = [ring.add(x, ring.mul(2, y)) for x, y in zip(low, high)]
        timings['recovery_ns'] = time.perf_counter_ns()-start
        phase, start = 'verify', time.perf_counter_ns()
        if [verify.dot(row, answer) for row in prepared['h']] != syndrome or lee(answer) > t:
            raise ValueError('Final verification failed')
        timings['verify_ns'] = time.perf_counter_ns()-start
    except BudgetExceeded as exc:
        timings[phase+'_ns'] = time.perf_counter_ns()-start
        status, reason = 'censored', phase+':'+str(exc)
    counts = Counter()
    for name, op in ops.items():
        counts.update(op.stats)
        counts.update({name+'_'+key: value for key, value in op.stats.items()})
    return dict(answer=answer if status == 'complete' else None, status=status, reason=reason,
                initial_radius=t, counters=dict(counts), times=dict(timings), trace=trace)
