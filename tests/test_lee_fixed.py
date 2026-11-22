import itertools
import random
import unittest
from collections import Counter

from lee_extended import BudgetExceeded, prepare
from lee_fixed_kernel import MATCHED, affine_solutions, decode, restricted_rref, search
from lee_z4_dcc_comparison import Arithmetic, raw_syndrome
from analyze_lee_fixed import coverage_predictions, weighted
from run_lee_fixed import schedule


class FixedKernelTests(unittest.TestCase):
    def test_seed_weighting_and_schedule(self):
        rows = []
        for a, costs in ((0, (10, 30)), (1, (100, 300)), (2, (1000, 3000))):
            for index, cost in enumerate(costs):
                rows.append(dict(code='c', cluster='m', k0=6, kernel='common',
                                 method='ep_all', a=a, population=f'a{a}', seed_index=index,
                                 online_ns=cost))
        table = {(r['seed_scope'], r['group']): r for r in weighted(rows)}
        from lee_extended import composition_probability
        expected = sum(float(composition_probability(64, 4, a))*cost
                       for a, cost in ((0, 20), (1, 200), (2, 2000)))
        self.assertAlmostEqual(table['all', 'sphere']['online_ns'], expected)
        self.assertEqual(table['all', 'a_two']['online_ns'], 2000)
        self.assertEqual(table['original', 'a_two']['online_ns'], 1000)
        uniform = dict(rows[0], population='uniform', online_ns=1000000000)
        self.assertEqual(weighted(rows), weighted(rows+[uniform]))
        base = dict(population='a0', k0=6, trial_seed=123)
        planned = list(schedule(base, 4, 8))
        self.assertEqual(sum(k == 'common' for _, _, k, _ in planned), 8)
        self.assertEqual(sum(k == 'legacy' for _, _, k, _ in planned), 1)
        self.assertEqual(planned[0][1], 123)
        self.assertEqual(len(list(schedule(dict(base, population='uniform'), 4, 8))), 2)

    def test_exact_coverage_prediction_boundaries(self):
        rows = [dict(kernel='common', method=m, seed_scope='all', k0=12,
                     group='sphere', first_trials=1, recovery_trials=1)
                for m in ('ep_all', 'proj_binary_all')]
        values = coverage_predictions(rows)
        self.assertEqual(values[0]['predicted_first_trials'], values[1]['predicted_first_trials'])
        self.assertEqual(values[0]['predicted_recovery_trials'], values[1]['predicted_recovery_trials'])

    def test_restricted_affine_sets_exhaustive(self):
        rng = random.Random(231)
        for height in range(1, 5):
            for _ in range(25):
                h = [[rng.randrange(2) for _ in range(height+2)] for _ in range(height)]
                b = [rng.randrange(2) for _ in range(height)]
                chosen = rng.sample(range(height+2), height)
                rows, pivots, consistent = restricted_rref(h, b, chosen, Arithmetic(1))
                expected = {x for x in itertools.product(range(2), repeat=height)
                            if all(sum(row[j]*v for j, v in zip(chosen, x)) % 2 == target
                                   for row, target in zip(h, b))}
                actual = (set(map(tuple, affine_solutions(rows, pivots, height, Arithmetic(1))))
                          if consistent else set())
                self.assertEqual(actual, expected)

    def test_singular_covered_selection_can_recover(self):
        h = [[1, 1, 0], [0, 0, 1]]
        class Selection:
            def sample(self, population, size):
                return [0, 1]
        trace = []
        op = Arithmetic(1)
        result = search(h, [1, 0], 1, Selection(), op, 1, 10, True,
                        Counter(), 'first', trace)
        self.assertEqual(result, [1, 0, 0])
        self.assertEqual(trace[0]['nullity'], 1)
        self.assertEqual(op.stats['consistent_singular_trials'], 1)
        with self.assertRaises(BudgetExceeded) as raised:
            search(h, [1, 0], 1, Selection(), Arithmetic(1), 1, 10, False,
                   Counter(), 'first')
        self.assertEqual(str(raised.exception), 'trial_budget')

    def test_decoding_matches_legacy_and_free_boundary(self):
        # The binary repetition code has distance five, hence t=1 is unique.
        h = [[1, 0, 0, 0, 3], [0, 1, 0, 0, 3],
             [0, 0, 1, 0, 3], [0, 0, 0, 1, 3]]
        p = prepare(h)
        for error in ([1, 0, 0, 0, 0], [0, 0, 0, 0, 3]):
            syndrome = raw_syndrome(h, error)
            for seed in range(6):
                results = {}
                for method in MATCHED:
                    common = decode(p, syndrome, 1, seed, method)
                    legacy = decode(p, syndrome, 1, seed, method, kernel='legacy')
                    self.assertEqual(common['answer'], error)
                    self.assertEqual(common['answer'], legacy['answer'])
                    for key in ('first_trials', 'first_candidates', 'second_trials'):
                        self.assertEqual(common['counters'].get(key, 0),
                                         legacy['counters'].get(key, 0))
                    results[method] = common
                self.assertEqual(results['ep_all']['counters'],
                                 results['proj_binary_all']['counters'])
                self.assertEqual(results['ep_reject']['counters'],
                                 results['proj_binary_t']['counters'])

    def test_torsion_high_digit_and_conditional_recovery(self):
        n = 9
        top = [[int(j == i) if j < n-1 else 3 for j in range(n)] for i in range(n-1)]
        for torsion in (False, True):
            h = top+([[0]*(n-1)+[2]] if torsion else [])
            p = prepare(h)
            for error in ([2, 2]+[0]*7, [1, 3, 2]+[0]*6, [1, 3, 1, 3]+[0]*5):
                syndrome = raw_syndrome(h, error)
                results = {}
                for method in MATCHED:
                    current = decode(p, syndrome, 4, 431, method)
                    legacy = decode(p, syndrome, 4, 431, method, kernel='legacy')
                    self.assertEqual(current['answer'], error)
                    self.assertEqual(legacy['answer'], error)
                    for field in ('first_trials', 'first_candidates', 'second_trials', 'second_candidates'):
                        self.assertEqual(current['counters'].get(field, 0),
                                         legacy['counters'].get(field, 0))
                    results[method] = current
                for ep, proj in (('ep_all', 'proj_binary_all'), ('ep_reject', 'proj_binary_t')):
                    for key, value in results[ep]['counters'].items():
                        if key.startswith('second_'):
                            self.assertEqual(value, results[proj]['counters'].get(key, 0))


if __name__ == '__main__':
    unittest.main()
