"""Independent finite-set oracles for the new collision and representation kernels."""
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from lee_extended import distance_certificate, prepare
from lee_revision import make_code
from lee_list_isd import (Parameters, Work, decode, mmt_candidates,
                          raw_mod_syndrome, sparse_words, stern_candidates, weight)
from lee_z4_dcc_comparison import Arithmetic, raw_syndrome


class ListISDTests(unittest.TestCase):
    def test_weight_enumerators_against_ambient_cube(self):
        for modulus in (2, 4):
            for length in range(5):
                for total in range((1 if modulus == 2 else 2)*length+1):
                    actual = list(sparse_words(length, total, modulus))
                    expected = {v for v in itertools.product(range(modulus), repeat=length)
                                if weight(v, modulus) == total}
                    self.assertEqual(set(actual), expected)
                    self.assertEqual(len(actual), len(expected))
                    self.assertEqual(len(actual), math.comb(length if modulus == 2 else 2*length, total))

    def test_stern_collision_sets_exhaustive(self):
        rng = random.Random(3291)
        for modulus in (2, 4):
            # Even rows model actual Z4 torsion equations; keys must retain them.
            for _ in range(8):
                h = [[rng.randrange(modulus) for _ in range(4)] for _ in range(2)]
                if modulus == 4:
                    h[-1] = [2*(x % 2) for x in h[-1]]
                h = np.asarray(h, dtype=np.int64)
                for target in itertools.product(range(modulus), repeat=2):
                    expected = {v for v in itertools.product(range(modulus), repeat=4)
                                if weight(v[:2], modulus) == weight(v[2:], modulus) == 1
                                and raw_mod_syndrome(h, v, modulus) == list(target)}
                    actual = set(stern_candidates(h, target, list(range(4)),
                                                  Parameters(weight=1), modulus, Work()))
                    self.assertEqual(actual, expected)

    def test_mmt_representation_sets_exhaustive(self):
        for modulus in (2, 4):
            h = np.asarray([[1, 1, 0, 1], [0, 2 % modulus, 1, 0]], dtype=np.int64)
            base = [a+b for a in sparse_words(2, 1, modulus)
                    for b in sparse_words(2, 1, modulus)]
            for target in itertools.product(range(modulus), repeat=2):
                expected = set()
                for x in base:
                    if raw_mod_syndrome(h[-1:], x, modulus) != [0]:
                        continue
                    for y in base:
                        z = tuple((a+b) % modulus for a, b in zip(x, y))
                        if raw_mod_syndrome(h, z, modulus) == list(target):
                            expected.add(z)
                actual = set(mmt_candidates(h, target, list(range(4)),
                             Parameters(weight=1, weight2=1, split=1), modulus, Work()))
                self.assertEqual(actual, expected)

    def test_complete_decoders_on_free_and_nonfree_codes(self):
        n = 9
        top = [[int(i == j) if j < n-1 else 3 for j in range(n)] for i in range(n-1)]
        for torsion in (False, True):
            h = top+([[0]*(n-1)+[2]] if torsion else [])
            public = prepare(h)
            for error in ([1, 3, 2]+[0]*6, [2, 2]+[0]*7,
                          [1, 3, 1, 3]+[0]*5, [0]*9):
                for method in ('ccp_stern', 'lee_stern', 'ccp_mmt', 'lee_mmt'):
                    for seed in range(3):
                        parameters = (Parameters(0, 0, 0, 0) if method.endswith('mmt')
                                      else Parameters(1, 1, 1, 1))
                        result = decode(public, raw_syndrome(h, error), 4, seed,
                                        method, parameters, trials=4096)
                        self.assertEqual(result['status'], 'complete', (method, result))
                        self.assertEqual(result['answer'], error)

    def test_budget_retains_failed_work(self):
        h = [[1, 0, 0, 3], [0, 1, 0, 3], [0, 0, 1, 3]]
        result = decode(prepare(h), [1, 0, 0], 1, 0, 'lee_stern',
                        Parameters(0, 0), trials=0)
        self.assertEqual(result['status'], 'budget')
        self.assertEqual(result['reason'], 'trial_limit')
        self.assertGreater(result['times']['online_ns'], 0)

    def test_nondegenerate_mmt_complete_decoding_random_code(self):
        rng = random.Random(20261105)
        while True:
            ambient = [[rng.randrange(4) for _ in range(32)] for _ in range(32)]
            try:
                code = make_code(ambient, 4, 2)
            except ValueError:
                continue
            if distance_certificate(ambient[:4]) > 8:
                break
        public = prepare(code['h'])
        for error in ([1, 3, 1, 3]+[0]*28, [2, 2]+[0]*30):
            for method in ('ccp_mmt', 'lee_mmt'):
                result = decode(public, raw_syndrome(code['h'], error), 4, 129,
                                method, Parameters(2, 1, 1, 1), trials=4096)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['answer'], error)
                if any(x % 2 for x in error):
                    self.assertGreater(sum(v for k, v in result['counters'].items()
                                           if k.endswith('_nonzero_list_trials')), 0)

    def test_complete_binary_layers_against_exhaustive_ball(self):
        from lee_list_isd import list_search
        rng = random.Random(3901)
        for _ in range(12):
            h = [[rng.randrange(2) for _ in range(6)] for _ in range(4)]
            # Require full row rank without assuming the chosen columns form a basis.
            from lee_fixed_kernel import restricted_rref
            _, pivots, _ = restricted_rref(h, [0]*4, list(range(6)), Arithmetic(1, False))
            if len(pivots) != 4:
                continue
            for position in range(6):
                error = [int(j == position) for j in range(6)]
                syndrome = raw_mod_syndrome(h, error, 2)
                oracle = {x for x in itertools.product(range(2), repeat=6)
                          if sum(x) <= 1 and raw_mod_syndrome(h, x, 2) == syndrome}
                for algorithm in ('stern', 'mmt'):
                    result = list_search(h, syndrome, 1, random.Random(37), 2, 4,
                                         algorithm, Parameters(1, 1, 1, 1), Work())
                    self.assertIn(tuple(result), oracle)

    def test_sphere_averages_keep_failed_work(self):
        from analyze_lee_list_comparison import mother_points
        from lee_extended import composition_probability
        rows = []
        for a, cost, status in ((0, 1, 'complete'), (1, 10, 'complete'), (2, 100, 'budget')):
            for seed in range(2):
                rows.append(dict(t=4, k0=6, method='ccp_stern', mother=2, a=a,
                    status=status, times={'online_ns': cost*1000000},
                    counters={'first_trials': cost, 'first_metric_tests': 2*cost}))
        points = {p['group']: p for p in mother_points(rows)}
        masses = [float(composition_probability(64, 4, a)) for a in range(3)]
        self.assertAlmostEqual(points['sphere']['online_ms'], sum(p*c for p, c in zip(masses, (1, 10, 100))))
        self.assertAlmostEqual(points['sphere']['completion'], 1-masses[2])
        self.assertAlmostEqual(points['a_positive']['online_ms'],
                               (masses[1]*10+masses[2]*100)/(masses[1]+masses[2]))

    def test_kernel_selection_uses_training_not_confirmation(self):
        from analyze_lee_list_comparison import choose_families
        points = []
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for t in (4, 8):
                for algorithm in ('stern', 'mmt'):
                    records = {}
                    for k0 in (0, 3, 6, 9, 12):
                        for side in ('ccp', 'lee'):
                            prefer = 'stern' if side == 'ccp' else 'mmt'
                            loss = 1 if algorithm == prefer else 100
                            parameters = {'weight': 0}
                            records[f'{k0}:{side}_{algorithm}'] = dict(family_best=parameters,
                                candidates=[dict(parameters=parameters, selection_loss_ns=loss)])
                            points.append(dict(t=t, k0=k0, method=f'{side}_{algorithm}_family_best',
                                               online_ms=1000 if algorithm == prefer else .01))
                    (folder/f'tuning_{algorithm}_t{t}.json').write_text(json.dumps(records), encoding='utf-8')
            selected, choices = choose_families(points, folder,
                                               dict(radii=[4, 8], algorithms=['stern', 'mmt']))
            self.assertEqual(len(selected), 20)
            self.assertTrue(all(r['online_ms'] == 1000 for r in selected))
            self.assertTrue(all(r['algorithm'] == ('stern' if r['side'] == 'ccp' else 'mmt')
                                for r in choices))

    def test_completion_classification_does_not_compare_float_to_one(self):
        from analyze_lee_list_comparison import mother_points, ratios
        rows = [dict(t=8, k0=6, method=method, mother=2, a=a, status='complete',
                     times={'online_ns': 1000000}, counters={})
                for method in ('ccp_stern', 'lee_stern') for a in range(5)]
        points = mother_points(rows)
        self.assertTrue(all(p['all_complete'] for p in points))
        self.assertTrue(all(r['all_complete'] and r['assessment'] == 'undetermined'
                            for r in ratios(points)))


if __name__ == '__main__':
    unittest.main()
