import math
import unittest

from rigor import regression as reg


class TestSimpleLinearRegression(unittest.TestCase):
    def test_exact_line_gives_perfect_fit(self):
        x = [1, 2, 3, 4, 5]
        y = [2 * xi + 1 for xi in x]  # y = 2x + 1, no noise
        result = reg.simple_linear_regression(x, y)
        self.assertAlmostEqual(result.slope, 2.0, places=9)
        self.assertAlmostEqual(result.intercept, 1.0, places=9)
        self.assertAlmostEqual(result.r_squared, 1.0, places=9)
        self.assertEqual(result.slope_se, 0.0)
        self.assertAlmostEqual(result.slope_p_value, 0.0, places=9)

    def test_matches_hand_computed_example(self):
        # x=[1,2,3,4,5], y=[3,4,2,5,6]: mean_x=3, mean_y=4, sxx=10, sxy=7
        # -> slope=0.7, intercept = 4 - 0.7*3 = 1.9 -- computed
        # independently of this module.
        x = [1, 2, 3, 4, 5]
        y = [3, 4, 2, 5, 6]
        result = reg.simple_linear_regression(x, y)
        self.assertAlmostEqual(result.slope, 0.7, places=9)
        self.assertAlmostEqual(result.intercept, 1.9, places=9)
        self.assertEqual(result.df, 3)

    def test_zero_slope_gives_high_p_value(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [5, 5, 5, 5, 5, 5, 5, 5]
        result = reg.simple_linear_regression(x, y)
        self.assertAlmostEqual(result.slope, 0.0, places=9)
        self.assertAlmostEqual(result.slope_p_value, 1.0, places=6)

    def test_confidence_interval_contains_slope(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 5, 3, 8, 7, 10, 9, 13, 12, 15]
        result = reg.simple_linear_regression(x, y)
        lo, hi = result.slope_confidence_interval
        self.assertLess(lo, result.slope)
        self.assertGreater(hi, result.slope)

    def test_constant_x_raises(self):
        with self.assertRaises(ValueError):
            reg.simple_linear_regression([5, 5, 5], [1, 2, 3])

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            reg.simple_linear_regression([1, 2], [3, 4])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            reg.simple_linear_regression([1, 2, 3], [1, 2])


if __name__ == "__main__":
    unittest.main()
