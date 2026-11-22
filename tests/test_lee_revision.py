import itertools
import math
import random
import unittest
from fractions import Fraction

from lee_revision import (METHODS, aggregate, decode, eliminate, make_code, prepare,
                          selected_solutions, uniform_codes)
from lee_z4_dcc_comparison import Arithmetic, raw_syndrome, eliminate as scalar_eliminate
from ring_constraints import digit_system, matvec, smith_check
from z4_separation import canonical


class LeeRevisionTests(unittest.TestCase):
    def test_vectorized_elimination_preserves_results_and_reference_counts(self):
        rng = random.Random(1168)
        for bits in (1, 2):
            for _ in range(40):
                h = [[rng.randrange(2**bits) for _ in range(7)] for _ in range(5)]
                target = [rng.randrange(2**bits) for _ in h]
                selected = rng.sample(range(7), rng.randrange(1, 6))
                left, right = Arithmetic(bits), Arithmetic(bits)
                self.assertEqual(eliminate(h, target, selected, left),
                                 scalar_eliminate(h, target, selected, right))
                self.assertEqual(left.stats, right.stats)

    def test_singular_stream_matches_exhaustive_solutions(self):
        rng = random.Random(713)
        for _ in range(40):
            h = [[rng.randrange(2) for _ in range(5)] for _ in range(3)]
            target = [rng.randrange(2) for _ in h]
            chosen = [0, 2, 4]
            actual = {tuple(x) for x in selected_solutions(h, target, chosen, Arithmetic(1))}
            expected = set()
            for values in itertools.product(range(2), repeat=3):
                x = [0]*5
                for j, value in zip(chosen, values):
                    x[j] = value
                if matvec(h, x, 2) == target:
                    expected.add(tuple(x))
            self.assertEqual(actual, expected)

    def test_mean_kernel_identity_on_all_full_rank_small_checks(self):
        counts = []
        for bits in itertools.product(range(2), repeat=8):
            h = [list(bits[:4]), list(bits[4:])]
            reduced, _, _ = canonical(h, [0, 0])
            if len(reduced) != 2:
                continue
            count = sum(matvec([[row[0], row[2]] for row in h], list(x), 2) == [0, 0]
                        for x in itertools.product(range(2), repeat=2))
            counts.append(count)
        self.assertEqual(Fraction(sum(counts), len(counts)), 1+Fraction((2**2-1)**2, 2**4-1))

    def test_sphere_composition_identity_and_integer_certificates(self):
        for n in range(1, 9):
            for t in range(2*n+1):
                count = sum(math.comb(n, t-2*a)*math.comb(n-(t-2*a), a)*2**(t-2*a)
                            for a in range(t//2+1) if 0 <= t-2*a <= n and a <= n-(t-2*a))
                self.assertEqual(count, math.comb(2*n, t))
        self.assertLess(25**250, 2**175*4**40*21**210)
        self.assertLess(50**100*7**96, 41**100*10**96)
        self.assertLess(50**50*7**96, 31**50*10**96)

    def test_arbitrary_coordinate_flag_and_all_decoders(self):
        for code, info in uniform_codes(1, ((10, 2, 1),)):
            for error in ([1]+[0]*9, [0, 3]+[0]*8, [0]*10):
                syndrome = raw_syndrome(code['h'], error)
                for method in METHODS+('dcc_all',):
                    actual, _ = decode(prepare(code, method), syndrome, 1, 817)
                    self.assertEqual(actual, error, method)
        for code, info in uniform_codes(1, ((16, 4, 2),)):
            for error in ([2]+[0]*15, [3, 1]+[0]*14):
                syndrome = raw_syndrome(code['h'], error)
                for method in METHODS+('dcc_all',):
                    actual, _ = decode(prepare(code, method), syndrome, 2, 318)
                    self.assertEqual(actual, error, method)

    def test_smith_system_against_all_completions(self):
        rng = random.Random(990713)
        for p, r, n in ((2, 3, 2), (3, 2, 2)):
            modulus = p**r
            ambient = list(itertools.product(range(modulus), repeat=n))
            for _ in range(12):
                h = [[rng.randrange(modulus) for _ in range(n)] for _ in range(3)]
                prepared = smith_check(h, p, r)
                for j in range(n):
                    column = [row[j] for row in prepared['v']]
                    self.assertEqual(matvec(prepared['u'], matvec(h, column, modulus), modulus),
                                     [row[j] for row in prepared['diagonal']])
                    self.assertEqual(matvec(prepared['inverse_v'], column, modulus),
                                     [int(i == j) for i in range(n)])
                error = rng.choice(ambient)
                syndrome = matvec(h, error, modulus)
                solutions = [x for x in ambient if matvec(h, x, modulus) == syndrome]
                for level in range(r):
                    for prefix in itertools.product(range(p**level), repeat=n):
                        expected = {tuple(x//p**level % p for x in e) for e in solutions
                                    if all(x % p**level == y for x, y in zip(e, prefix))}
                        try:
                            a, b = digit_system(prepared, syndrome, prefix, level)
                        except ValueError:
                            self.assertFalse(expected)
                            continue
                        actual = {z for z in itertools.product(range(p), repeat=n) if matvec(a, z, p) == b}
                        self.assertEqual(actual, expected)

    def test_failures_suppress_ablation_ranking(self):
        result = aggregate([dict(correct=False, code='failure')])
        self.assertNotIn('groups', result)


if __name__ == '__main__':
    unittest.main()
