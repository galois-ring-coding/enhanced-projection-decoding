import itertools
import unittest

from z4_separation import canonical, decode, exponents, slack_decode
from lee_z4_dcc_comparison import code_suite, raw_syndrome


class Z4Separation(unittest.TestCase):
    def test_canonical_affine_system_is_row_order_invariant(self):
        h = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
        b = [1, 0, 1]
        self.assertEqual(canonical(h, b), canonical(list(reversed(h)), list(reversed(b))))
        with self.assertRaises(ValueError):
            canonical([[0, 0]], [1])

    def test_slack_solver_on_all_small_repetition_errors(self):
        h = [[1, 1, 0], [0, 1, 1]]
        for error in itertools.product(range(2), repeat=3):
            if sum(error) > 1:
                continue
            b = [sum(a*x for a, x in zip(row, error)) % 2 for row in h]
            for slack in (0, 1):
                answer, _ = slack_decode(h, b, [], 1, slack, 109)
                self.assertEqual(answer, list(error))

    def test_equivalence_with_symbol_twos_and_signs(self):
        for code, _ in code_suite(1)[:4]:
            for error in ([2]+[0]*9, [3, 1]+[0]*8, [0]*10):
                s = raw_syndrome(code.parity(2), error)
                for slack in (0, 1, 2):
                    a = decode(code, s, 2, 113, 'super', slack)
                    b = decode(code, s, 2, 113, 'enhanced', slack)
                    self.assertEqual(a, b)
                    self.assertEqual(a[0], error)

    def test_separation_interior_and_free_code_boundary(self):
        interior = exponents()
        self.assertLess(interior['distance_failure_exponent'], 0)
        self.assertGreater(interior['exponent_gap_lower_bound'], .022)
        free = exponents(k0=.3)
        self.assertAlmostEqual(free['exponent_gap_lower_bound'], 0)
        second_bottleneck = exponents(sigma=.005, alpha=.03)
        self.assertLess(second_bottleneck['exponent_gap_lower_bound'], 0)


if __name__ == '__main__':
    unittest.main()
