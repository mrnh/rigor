import math
import unittest

from rigor import effect_size as es


class TestCohensD(unittest.TestCase):
    def test_known_hand_computed_value(self):
        # a: mean 30 var 2.5 (n=5); b: mean 25 var 2.5 (n=5) -- same data
        # as the two-sample t-test example. Pooled sd = sqrt(2.5) since
        # variances are equal, so d = 5/sqrt(2.5) = 3.1623.
        a = [30, 29, 32, 31, 28]
        b = [25, 26, 24, 27, 23]
        d = es.cohens_d(a, b)
        self.assertAlmostEqual(d, 5 / math.sqrt(2.5), places=6)

    def test_identical_samples_give_zero(self):
        a = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(es.cohens_d(a, list(a)), 0.0, places=9)

    def test_zero_pooled_sd_with_a_difference_is_infinite(self):
        d = es.cohens_d([5, 5, 5], [3, 3, 3])
        self.assertEqual(d, math.inf)

    def test_sign_flips_with_argument_order(self):
        a, b = [10, 11, 12], [5, 6, 7]
        self.assertAlmostEqual(es.cohens_d(a, b), -es.cohens_d(b, a), places=9)


class TestHedgesG(unittest.TestCase):
    def test_converges_to_cohens_d_for_large_samples(self):
        a = list(range(100, 150))
        b = list(range(90, 140))
        d = es.cohens_d(a, b)
        g = es.hedges_g(a, b)
        self.assertAlmostEqual(d, g, delta=0.01)  # correction factor is ~0.992 at this n

    def test_smaller_in_magnitude_than_cohens_d_for_small_samples(self):
        a = [10, 12, 11]
        b = [5, 7, 6]
        d = es.cohens_d(a, b)
        g = es.hedges_g(a, b)
        self.assertLess(abs(g), abs(d))


class TestCohensH(unittest.TestCase):
    def test_zero_for_equal_proportions(self):
        self.assertAlmostEqual(es.cohens_h(0.3, 0.3), 0.0, places=9)

    def test_sign_flips_with_argument_order(self):
        self.assertAlmostEqual(es.cohens_h(0.6, 0.2), -es.cohens_h(0.2, 0.6), places=9)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            es.cohens_h(1.5, 0.2)


class TestCramersV(unittest.TestCase):
    def test_zero_statistic_gives_zero(self):
        self.assertAlmostEqual(es.cramers_v(0.0, 100, 2, 2), 0.0, places=9)

    def test_matches_hand_computation_for_2x2(self):
        # chi2=10, n=100, 2x2 table -> k-1=1 -> v = sqrt(10/100) = sqrt(0.1)
        v = es.cramers_v(10.0, 100, 2, 2)
        self.assertAlmostEqual(v, math.sqrt(0.1), places=9)

    def test_requires_at_least_2x2(self):
        with self.assertRaises(ValueError):
            es.cramers_v(5.0, 50, 1, 3)


if __name__ == "__main__":
    unittest.main()
