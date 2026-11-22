import unittest

from analyze_lee_rejection import coverage_bounds, validate_stopping


class RejectionAnalysisTests(unittest.TestCase):
    def test_covered_singular_selection_is_rejected(self):
        row = dict(n=3, error=[1, 0, 0], trace=[
            dict(phase='first', chosen=[0, 1], nullity=1, consistent=True,
                 covers_true_support=True),
            dict(phase='first', chosen=[0, 2], nullity=0, consistent=True,
                 covers_true_support=True),
        ])
        counts = validate_stopping([row])
        self.assertEqual(counts['first_covered_singular_rejections'], 1)
        self.assertEqual(counts['first_selections'], 2)

    def test_successful_selection_cannot_be_skipped(self):
        row = dict(n=3, error=[1, 0, 0], trace=[
            dict(phase='first', chosen=[0, 1], nullity=0, consistent=True,
                 covers_true_support=True),
            dict(phase='first', chosen=[0, 2], nullity=0, consistent=True,
                 covers_true_support=True),
        ])
        with self.assertRaises(AssertionError):
            validate_stopping([row])

    def test_free_boundary_and_shared_recovery(self):
        self.assertEqual(coverage_bounds(64, 12, 4, 12, 'ep_reject'),
                         coverage_bounds(64, 12, 4, 12, 'proj_binary_t'))
        ccp = coverage_bounds(64, 12, 4, 6, 'ep_reject')
        projection = coverage_bounds(64, 12, 4, 6, 'proj_binary_t')
        self.assertLess(ccp[0], projection[0])
        self.assertEqual(ccp[1], projection[1])


if __name__ == '__main__':
    unittest.main()
