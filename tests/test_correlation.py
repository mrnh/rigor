import math
import unittest

from rigor import correlation as corr


class TestPearsonCorrelation(unittest.TestCase):
    def test_matches_hand_computed_example(self):
        # x=[1,2,3,4,5], y=[2,4,5,4,5]: sxy=8, sxx=10, syy=6.8
        # -> r = 8/sqrt(10*6.8) = 0.774597 -- computed independently of
        # this module.
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        result = corr.pearson_correlation(x, y)
        self.assertAlmostEqual(result.statistic, 0.7745966692414834, places=9)
        self.assertEqual(result.df, 3)

    def test_perfect_positive_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        result = corr.pearson_correlation(x, y)
        self.assertAlmostEqual(result.statistic, 1.0, places=9)
        self.assertAlmostEqual(result.p_value, 0.0, places=9)

    def test_perfect_negative_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        result = corr.pearson_correlation(x, y)
        self.assertAlmostEqual(result.statistic, -1.0, places=9)

    def test_no_linear_relationship_gives_high_p_value(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [3, 7, 2, 8, 1, 9, 4, 6]  # scrambled, no trend
        result = corr.pearson_correlation(x, y)
        self.assertGreater(result.p_value, 0.05)

    def test_confidence_interval_contains_r(self):
        x = [1, 3, 2, 5, 4, 6, 8, 7, 9, 10]
        y = [2, 4, 1, 6, 3, 7, 9, 8, 10, 12]
        result = corr.pearson_correlation(x, y)
        lo, hi = result.confidence_interval
        self.assertLess(lo, result.statistic)
        self.assertGreater(hi, result.statistic)

    def test_constant_x_raises(self):
        with self.assertRaises(ValueError):
            corr.pearson_correlation([5, 5, 5, 5], [1, 2, 3, 4])

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            corr.pearson_correlation([1, 2], [3, 4])


class TestSpearmanCorrelation(unittest.TestCase):
    def test_perfect_monotonic_but_nonlinear_relationship(self):
        # y = x^3 is perfectly monotonic but not linear -- Spearman
        # should be exactly 1.0 even though Pearson's r would be < 1.
        x = [1, 2, 3, 4, 5]
        y = [xi ** 3 for xi in x]
        result = corr.spearman_correlation(x, y)
        self.assertAlmostEqual(result.statistic, 1.0, places=9)
        # Pearson on the same (curved) data should be high but not 1.0,
        # since it's measuring linearity rather than monotonicity.
        self.assertLess(corr.pearson_correlation(x, y).statistic, 1.0)

    def test_perfect_inverse_monotonic(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        result = corr.spearman_correlation(x, y)
        self.assertAlmostEqual(result.statistic, -1.0, places=9)

    def test_ties_handled_without_crashing(self):
        x = [1, 1, 2, 2, 3]
        y = [1, 2, 2, 3, 3]
        result = corr.spearman_correlation(x, y)
        self.assertTrue(-1.0 <= result.statistic <= 1.0)


if __name__ == "__main__":
    unittest.main()
