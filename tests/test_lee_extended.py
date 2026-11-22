import itertools
import math
import random
import unittest

from lee_extended import (ABLATION, MAIN, RADIUS, binary_search, composition_probability,
                          decode, distance_certificate, initial_radius, prepare, sample_error)
from lee_revision import make_code
from lee_z4_dcc_comparison import Arithmetic, binary_distance, lee, raw_syndrome
from ring_constraints import digit_system, matvec, smith_check


class ExtendedTests(unittest.TestCase):
    def test_gray_distance_stream_against_full_enumeration(self):
        rng = random.Random(724)
        for _ in range(30):
            g = [[rng.randrange(2) for _ in range(12)] for _ in range(5)]
            self.assertEqual(distance_certificate(g), binary_distance(g))

    def test_compositions_and_gray_sampling(self):
        rng = random.Random(132)
        for n in range(2, 9):
            for t in range(2*n+1):
                self.assertEqual(sum(composition_probability(n, t, a) for a in range(t//2+1)), 1)
                self.assertEqual(lee(sample_error(n, t, rng)), t)
                for a in range(t//2+1):
                    if composition_probability(n, t, a):
                        e = sample_error(n, t, rng, a)
                        self.assertEqual(e.count(2), a)
                        self.assertEqual(lee(e), t)

    def test_public_prepare_with_hidden_torsion_rows(self):
        h = [[1, 1, 1], [1, 3, 1]]
        p = prepare(h)
        for error in itertools.product(range(4), repeat=3):
            s = raw_syndrome(h, error)
            transformed = matvec(p['u'], s, 4)
            target = [x % 2 for x in transformed[:p['units']]]
            target += [x//2 for x in transformed[p['units']:]]
            actual = {z for z in itertools.product(range(2), repeat=3)
                      if matvec(p['h0'], z, 2) == target}
            expected = {tuple(x % 2 for x in e) for e in itertools.product(range(4), repeat=3)
                        if raw_syndrome(h, e) == s}
            self.assertEqual(actual, expected)

    def test_singular_enumeration_recovers_covered_target(self):
        h = [[1, 0, 0], [0, 0, 1]]
        class FixedSelection:
            def sample(self, population, count):
                return [0, 1]
        op = Arithmetic(1)
        answer = binary_search(h, [1, 0], 1, FixedSelection(), op, 1, 20,
                               enumerate_singular=True)
        self.assertEqual(answer, [1, 0, 0])
        self.assertEqual(op.stats['singular_trials'], 1)

    def test_all_decoders_with_two_hidden_twos(self):
        rng = random.Random(917)
        for _ in range(1000):
            ambient = [[rng.randrange(4) for _ in range(24)] for _ in range(24)]
            try:
                code = make_code(ambient, 2, 1)
            except ValueError:
                continue
            if distance_certificate(ambient[:2]) > 8:
                break
        else:
            self.fail('No test code')
        p = prepare(code['h'])
        for a in (0, 1, 2):
            error = sample_error(24, 4, rng, a)
            syndrome = raw_syndrome(code['h'], error)
            for method in set(MAIN+ABLATION+RADIUS):
                # Floor may legitimately exclude the weight-four binary target.
                if initial_radius(24, 4, 'mean_floor') < 4-2*a and method.endswith('mean_floor'):
                    continue
                for track in (True, False):
                    result = decode(p, syndrome, 4, 482, method, trials=2000, track=track)
                    self.assertEqual(result['status'], 'complete', method)
                    self.assertEqual(result['answer'], error, method)

    def test_too_small_radius_is_retained_and_fallback_completes(self):
        # Zero code: distance is infinite and all errors are uniquely determined.
        h = [[int(i == j) for j in range(12)] for i in range(12)]
        p = prepare(h)
        error = [1]*6+[0]*6
        self.assertEqual(initial_radius(12, 6, 'mean_ceil'), 5)
        fixed = decode(p, error, 6, 17, 'proj_ring_mean_ceil', trials=8)
        self.assertEqual(fixed['status'], 'censored')
        self.assertEqual(fixed['counters']['first_trials'], 8)
        fallback = decode(p, error, 6, 17, 'proj_ring_fallback', trials=8)
        self.assertEqual(fallback['answer'], error)
        self.assertEqual(fallback['counters']['first_trials'], 3)


if __name__ == '__main__':
    unittest.main()
