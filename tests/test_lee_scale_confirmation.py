import itertools
import statistics
import unittest

from analyze_lee_fixed import weighted as fixed_weighted
from analyze_lee_scale_confirmation import weighted
from lee_extended import composition_probability
from lee_z4_dcc_comparison import lee
from present_lee_fixed import expected_searches
from run_lee_scale_confirmation import error_pool, search_seed, seed_for


class LeeScaleConfirmationTests(unittest.TestCase):
    def test_seed_namespaces_and_variable_compositions(self):
        seeds = set()
        for n, t in ((48, 3), (96, 6)):
            pool = error_pool(2026110423, n, t, 0)
            self.assertEqual(pool, error_pool(2026110423, n, t, 0))
            self.assertEqual(len(pool), 2*(t//2+2))
            self.assertEqual({r['a'] for r in pool if r['population'] != 'uniform'}, set(range(t//2+1)))
            for row in pool:
                self.assertEqual(len(row['error']), n)
                self.assertEqual(lee(row['error']), t)
                self.assertEqual(row['s']+2*row['a'], t)
                seeds.add(row['error_seed'])
            seeds.add(seed_for(2026110423, n, 0, 'ambient'))
            for index in range(4):
                seeds.add(search_seed(2026110423, n, 0, 'a0', 0, index))
        self.assertEqual(len(seeds), 6+10+2+8)

    @staticmethod
    def records(n, k, t):
        return [dict(n=n, K=k, t=t, code=f'n{n}', cluster=f'm{n}', k0=0,
                     method='ep_all', kernel='common', a=a, population=f'a{a}',
                     case=case, seed_index=index, online_ns=100*a+10*case+index)
                for a in range(t//2+1) for case in range(2) for index in range(4)]

    def test_composition_weighting_for_odd_radius_and_a_three(self):
        for n, k, t in ((48, 9, 3), (96, 18, 6)):
            rows = self.records(n, k, t)
            table = {r['group']: r for r in weighted(rows)}
            weights = [float(composition_probability(n, t, a)) for a in range(t//2+1)]
            self.assertAlmostEqual(table['sphere']['online_ns'], sum(w*(100*a+6.5) for a, w in enumerate(weights)))
            self.assertAlmostEqual(table['a_positive']['online_ns'],
                sum(w*(100*a+6.5) for a, w in enumerate(weights) if a)/sum(weights[1:]))
            self.assertNotAlmostEqual(table['sphere']['online_ns'], statistics.mean(r['online_ns'] for r in rows))
            uniform = dict(rows[0], population='uniform', online_ns=1e12)
            self.assertEqual(weighted(rows), weighted(rows+[uniform]))
            with self.assertRaisesRegex(AssertionError, 'Missing composition'):
                weighted([r for r in rows if r['a'] != 1])
            with self.assertRaisesRegex(AssertionError, 'Unbalanced composition'):
                weighted(rows[1:])

    def test_n64_weighting_agrees_with_archived_analysis(self):
        rows = self.records(64, 12, 4)
        old = {r['group']: r for r in fixed_weighted(rows) if r['seed_scope'] == 'all'}
        for row in weighted(rows):
            self.assertAlmostEqual(row['online_ns'], old[row['group']]['online_ns'])

    def test_coverage_prediction_on_entire_small_spheres(self):
        n, k = 7, 1
        for t in (3, 6):
            sphere = [e for e in itertools.product(range(4), repeat=n) if lee(e) == t]
            for k0 in (0, 1):
                for method in ('ep_all', 'proj_binary_all'):
                    height = n-k0 if method == 'ep_all' else n-k
                    selected = list(map(set, itertools.combinations(range(n), height)))
                    for group in ('sphere', 'a_positive'):
                        errors = [e for e in sphere if group == 'sphere' or 2 in e]
                        first, high = 0, 0
                        for error in errors:
                            low = {j for j, value in enumerate(error) if value % 2}
                            if low:
                                first += len(selected)/sum(low <= s for s in selected)
                            outside = [j for j in range(n) if j not in low]
                            support = {j for j in outside if error[j] == 2}
                            if support:
                                sets = list(map(set, itertools.combinations(outside, n-k-len(low))))
                                high += len(sets)/sum(support <= s for s in sets)
                        prediction = expected_searches(n, k, t, k0, method, group)
                        self.assertAlmostEqual(prediction[0], first/len(errors))
                        self.assertAlmostEqual(prediction[1], high/len(errors))


if __name__ == '__main__':
    unittest.main()
