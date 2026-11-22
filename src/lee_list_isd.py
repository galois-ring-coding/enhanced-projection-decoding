"""Binary CCP and native Z4 Lee list decoders.

Stern uses a zero window and disjoint halves (Horlemann-Trautmann--Weger,
Algorithm 4). MMT uses four representation lists on a PGE information set
(Li--Wang, Algorithm 1). Ring torsion checks remain in every collision key;
their nonuniform distributions are never replaced by uniform estimates.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import itertools
import math
import random
import time

import numpy as np

from lee_revision import eliminate
from lee_z4_dcc_comparison import Arithmetic, lee, raw_syndrome


@dataclass(frozen=True)
class Parameters:
    window: int = 0
    weight: int = 0
    weight2: int = 0
    split: int = 0


class SearchBudget(Exception):
    pass


class Work:
    def __init__(self, seconds=15, trials=4096, entries=200000, checks=2000000,
                 track=True):
        self.deadline = time.perf_counter()+seconds
        self.trial_limit, self.entry_limit, self.check_limit = trials, entries, checks
        self.stats = Counter()
        self.track = track
        self.checks = 0
        self.phase = 'native'

    def event(self, name, count=1):
        if self.track:
            self.stats[self.phase+'_'+name] += count

    def peak(self, entries):
        if entries > self.entry_limit:
            raise SearchBudget('list_entry_limit')
        if self.track:
            self.stats['peak_live_list_entries'] = max(
                self.stats['peak_live_list_entries'], entries)

    def poll(self):
        if time.perf_counter() > self.deadline:
            raise SearchBudget('time_limit')

    def candidate(self):
        self.checks += 1
        self.event('metric_tests')
        if self.checks > self.check_limit:
            raise SearchBudget('candidate_limit')
        if self.checks % 256 == 0:
            self.poll()


def weight(x, modulus):
    return sum(x) if modulus == 2 else lee(x)


def sparse_words(length, total, modulus):
    """Exact fixed-weight lists, without traversing the ambient alphabet cube."""
    if total == 0:
        yield (0,)*length
        return
    if modulus == 2:
        for support in itertools.combinations(range(length), total):
            word = [0]*length
            for j in support:
                word[j] = 1
            yield tuple(word)
        return
    for twos in range(total//2+1):
        odds = total-2*twos
        for support in itertools.combinations(range(length), twos+odds):
            for even in itertools.combinations(support, twos):
                even = set(even)
                odd = [j for j in support if j not in even]
                for signs in itertools.product((1, 3), repeat=odds):
                    word = [0]*length
                    for j in even:
                        word[j] = 2
                    for j, symbol in zip(odd, signs):
                        word[j] = symbol
                    yield tuple(word)


@lru_cache(maxsize=256)
def _word_tuple(length, total, modulus):
    return tuple(sparse_words(length, total, modulus))


def clear_word_cache():
    _word_tuple.cache_clear()


def _lists(matrix, indices, total, modulus, work):
    # Only small coordinate patterns are cached; syndrome labels are rebuilt.
    length = len(indices)
    bound = math.comb(length if modulus == 2 else 2*length, total) if 0 <= total <= (length if modulus == 2 else 2*length) else 0
    work.peak(bound)
    words = _word_tuple(len(indices), total, modulus)
    work.peak(len(words))
    if not words:
        return []
    block = matrix[:, indices]
    values = np.asarray(words, dtype=np.int64)
    labels = (values @ block.T) % modulus
    work.event('base_list_entries', len(words))
    work.event('label_scalar_products', len(words)*len(indices)*len(matrix))
    return [(tuple(map(int, label)), word) for label, word in zip(labels, words)]


def _sub(a, b, modulus):
    return tuple((x-y) % modulus for x, y in zip(a, b))


def _add(a, b, modulus):
    return tuple((x+y) % modulus for x, y in zip(a, b))


def stern_candidates(matrix, target, free, params, modulus, work):
    middle = len(free)//2
    left = _lists(matrix, free[:middle], params.weight, modulus, work)
    right = _lists(matrix, free[middle:], params.weight, modulus, work)
    work.peak(len(left)+len(right))
    lookup = defaultdict(list)
    for label, word in left:
        lookup[label].append(word)
    for label, word in right:
        work.event('hash_probes')
        for first in lookup.get(_sub(target, label, modulus), ()):
            work.event('collisions')
            yield first+word
        work.poll()


def mmt_candidates(matrix, target, free, params, modulus, work):
    """All four-list matches; repeated labels never overwrite their vectors."""
    middle = len(free)//2
    left = _lists(matrix, free[:middle], params.weight, modulus, work)
    right = _lists(matrix, free[middle:], params.weight2, modulus, work)
    split = min(params.split, len(target))
    cut = len(target)-split
    lookup = defaultdict(list)
    for label, word in right:
        lookup[label[cut:]].append((label, word))
    # First representation has zero trailing partial syndrome; the second
    # has the target's trailing partial syndrome, as in ring LMMT.
    merged = defaultdict(list)
    seen = set()
    for label, word in left:
        for other, tail in lookup.get(tuple((-x) % modulus for x in label[cut:]), ()):
            candidate = word+tail
            if candidate in seen:
                work.event('representation_duplicates')
                continue
            seen.add(candidate)
            merged[_add(label[:cut], other[:cut], modulus)].append(candidate)
            work.event('intermediate_list_entries')
            work.peak(len(left)+len(right)+len(seen))
        work.poll()
    for label, word in left:
        wanted = _sub(target[cut:], label[cut:], modulus)
        for other, tail in lookup.get(wanted, ()):
            combined = _add(label[:cut], other[:cut], modulus)
            work.event('hash_probes')
            for first in merged.get(_sub(target[:cut], combined, modulus), ()):
                work.event('representation_matches')
                yield _add(first, word+tail, modulus)
        work.poll()


def list_search(h, target, radius, rng, modulus, unit_rank, algorithm, params, work):
    n = len(h[0])
    if not any(target):
        work.event('zero_syndrome_shortcuts')
        return [0]*n
    if radius < 0:
        raise ValueError('Negative search radius')
    window = min(params.window, unit_rank)
    # Parameters depend on the public residual radius, never true composition.
    if algorithm == 'stern':
        local = Parameters(window, min(params.weight, radius//2))
    else:
        local = params
        if 2*(local.weight+local.weight2) > radius:
            local = Parameters(window, 0, 0, min(params.split, len(h)-unit_rank+window))
    op = Arithmetic(1 if modulus == 2 else 2, track=False)
    for trial in range(work.trial_limit):
        work.poll()
        work.event('trials')
        selected = rng.sample(range(n), unit_rank if algorithm == 'stern' else unit_rank-window)
        begin = time.perf_counter_ns()
        rows = eliminate(h, target, selected, op)
        work.event('elimination_ns', time.perf_counter_ns()-begin)
        if rows is None:
            work.event('singular_trials')
            continue
        work.event('full_rank_trials')
        pivots = selected[:unit_rank-window]
        excluded = set(selected if algorithm == 'stern' else pivots)
        free = [j for j in range(n) if j not in excluded]
        rng.shuffle(free)
        capacity = (1 if modulus == 2 else 2)*(len(free)//2)
        if algorithm == 'stern':
            effective = Parameters(window, min(local.weight, capacity))
        else:
            effective = Parameters(window, min(local.weight, capacity),
                                   min(local.weight2, capacity), local.split)
        # The true projected weight is unknown. A public fixed schedule gives
        # every supported weight a positive-probability recovery event.
        if trial % 4 == 3:
            effective = Parameters(window, 0, 0, 0)
        work.event('nonzero_list_trials' if effective.weight+effective.weight2 else 'zero_list_trials')
        rows = np.asarray(rows, dtype=np.int64)
        start = len(pivots)
        constraints, beta = rows[start:, :-1], tuple(map(int, rows[start:, -1]))
        stream = (stern_candidates if algorithm == 'stern' else mmt_candidates)(
            constraints, beta, free, effective, modulus, work)
        for partial in stream:
            work.candidate()
            if weight(partial, modulus) > radius:
                continue
            recovered = (rows[:start, -1]-rows[:start, free] @ np.asarray(partial)) % modulus
            if weight(partial, modulus)+weight(recovered, modulus) > radius:
                continue
            answer = [0]*n
            for j, value in zip(free, partial):
                answer[j] = int(value)
            for j, value in zip(pivots, recovered):
                answer[j] = int(value)
            if raw_mod_syndrome(h, answer, modulus) != list(target):
                raise AssertionError('List collision produced an invalid syndrome')
            return answer
    raise SearchBudget('trial_limit')


def raw_mod_syndrome(h, word, modulus):
    return [sum(x*y for x, y in zip(row, word)) % modulus for row in h]


def decode(prepared, syndrome, radius, seed, method, params, track=True,
           seconds=15, trials=4096, entries=200000, checks=2000000):
    """Only public H preprocessing, syndrome, radius and parameters are inputs."""
    work = Work(seconds, trials, entries, checks, track)
    begin = time.perf_counter_ns()
    times = {}
    answer, reason = None, None
    native = method.startswith('lee_')
    algorithm = method.split('_')[-1]
    if algorithm not in ('stern', 'mmt'):
        raise ValueError('Unknown list algorithm')
    try:
        if native:
            work.phase = 'native'
            answer = list_search(prepared['h'], syndrome, radius, random.Random(seed),
                                 4, prepared['units'], algorithm, params, work)
            times['native_ns'] = time.perf_counter_ns()-begin
        else:
            work.phase = 'first'
            transformed = raw_syndrome(prepared['u'], syndrome)
            units = prepared['units']
            if any(transformed[len(prepared['normalized']):]):
                raise ValueError('Inconsistent zero rows')
            transformed = transformed[:len(prepared['normalized'])]
            if any(x % 2 for x in transformed[units:]):
                raise ValueError('Inconsistent torsion syndrome')
            target = [x % 2 for x in transformed[:units]]+[x//2 for x in transformed[units:]]
            low = list_search(prepared['h0'], target, radius, random.Random(seed),
                              2, len(prepared['h0']), algorithm, params, work)
            times['first_ns'] = time.perf_counter_ns()-begin
            step = time.perf_counter_ns()
            known = [j for j, value in enumerate(low) if value]
            remaining = (radius-len(known))//2
            residual = [(x-y) % 4 for x, y in zip(transformed,
                raw_syndrome(prepared['normalized'], low))]
            if any(x % 2 for x in residual) or any(residual[units:]):
                raise ValueError('Incompatible low digit; unique-layer regime required')
            beta = [x//2 for x in residual[:units]]
            work.phase = 'recovery'
            rows = eliminate(prepared['h1'], beta, known, Arithmetic(1, False))
            if rows is None:
                raise ValueError('Known support dependent')
            rest = [j for j in range(prepared['n']) if j not in set(known)]
            reduced = [[row[j] for j in rest] for row in rows[len(known):]]
            rhs = [row[-1] for row in rows[len(known):]]
            if remaining == 0:
                if any(rhs):
                    raise ValueError('Invalid zero-budget recovery')
                work.event('zero_budget_shortcuts')
                outside = [0]*len(rest)
            else:
                outside = list_search(reduced, rhs, remaining, random.Random(seed+1),
                                      2, len(reduced), algorithm, params, work)
            high = [0]*prepared['n']
            for j, value in zip(rest, outside):
                high[j] = value
            for i, j in enumerate(known):
                high[j] = (rows[i][-1]-sum(rows[i][k]*high[k] for k in rest)) % 2
            answer = [x+2*y for x, y in zip(low, high)]
            times['recovery_ns'] = time.perf_counter_ns()-step
        step = time.perf_counter_ns()
        if raw_syndrome(prepared['h'], answer) != list(syndrome) or lee(answer) > radius:
            raise AssertionError('Final Lee verification failed')
        times['verification_ns'] = time.perf_counter_ns()-step
    except SearchBudget as failure:
        reason = str(failure)
    times['online_ns'] = time.perf_counter_ns()-begin
    return dict(status='complete' if reason is None else 'budget', reason=reason,
                answer=answer, counters=dict(work.stats), times=times,
                parameters=asdict(params))
