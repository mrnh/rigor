"""Hypothesis tests checked against known textbook examples and internal
consistency (e.g. a paired t-test must exactly equal a one-sample t-test
on the differences, since that's what it is by definition)."""
import unittest

from rigor import inference as inf


class TestOneSampleTTest(unittest.TestCase):
    def test_matches_hand_computed_example(self):
        # sample [51,55,50,56,49,54,52,53], H0 mu=50.
        # mean=52.5, sample variance=42/7=6.0, se=sqrt(6/8)=0.86603,
        # t=(52.5-50)/0.86603=2.88675 -- verified by direct computation,
        # independent of this module.
        sample = [51, 55, 50, 56, 49, 54, 52, 53]
        result = inf.one_sample_t_test(sample, mu0=50)
        self.assertEqual(result.df, 7)
        self.assertAlmostEqual(result.statistic, 2.886751, places=5)
        self.assertLess(result.p_value, 0.05)
        self.assertTrue(result.reject_null(alpha=0.05))

    def test_no_difference_gives_high_p_value(self):
        sample = [10, 10, 10, 10, 10]
        result = inf.one_sample_t_test(sample, mu0=10)
        self.assertEqual(result.statistic, 0.0)
        self.assertAlmostEqual(result.p_value, 1.0, places=9)

    def test_confidence_interval_contains_sample_mean(self):
        sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = inf.one_sample_t_test(sample, mu0=0)
        lo, hi = result.confidence_interval
        self.assertLess(lo, sum(sample) / len(sample))
        self.assertGreater(hi, sum(sample) / len(sample))

    def test_too_few_observations_raises(self):
        with self.assertRaises(ValueError):
            inf.one_sample_t_test([1.0], mu0=0)

    def test_zero_variance_matching_mu0_gives_zero_statistic_not_a_crash(self):
        result = inf.one_sample_t_test([10, 10, 10, 10, 10], mu0=10)
        self.assertEqual(result.statistic, 0.0)
        self.assertAlmostEqual(result.p_value, 1.0, places=9)

    def test_zero_variance_differing_from_mu0_gives_infinite_statistic(self):
        result = inf.one_sample_t_test([10, 10, 10, 10, 10], mu0=5)
        self.assertEqual(result.statistic, float("inf"))
        self.assertAlmostEqual(result.p_value, 0.0, places=9)


class TestTwoSampleTTest(unittest.TestCase):
    def test_welch_matches_hand_computed_example(self):
        # a=[30,29,32,31,28] (mean 30, var 2.5), b=[25,26,24,27,23] (mean 25, var 2.5)
        # se=sqrt(2.5/5+2.5/5)=1.0, t=(30-25)/1.0=5.0 -- verified by direct
        # computation, independent of this module.
        a = [30, 29, 32, 31, 28]
        b = [25, 26, 24, 27, 23]
        result = inf.two_sample_t_test(a, b, equal_var=False)
        self.assertAlmostEqual(result.statistic, 5.0, places=6)
        self.assertLess(result.p_value, 0.005)

    def test_pooled_and_welch_agree_for_equal_variance_equal_n(self):
        a = [10, 12, 11, 13, 9, 14]
        b = [15, 17, 16, 18, 14, 19]
        pooled = inf.two_sample_t_test(a, b, equal_var=True)
        welch = inf.two_sample_t_test(a, b, equal_var=False)
        # Equal n and similar spread -> statistics should be very close.
        self.assertAlmostEqual(pooled.statistic, welch.statistic, places=6)

    def test_identical_samples_give_zero_statistic(self):
        a = [5, 6, 7, 8]
        result = inf.two_sample_t_test(a, list(a))
        self.assertAlmostEqual(result.statistic, 0.0, places=9)
        self.assertAlmostEqual(result.p_value, 1.0, places=6)

    def test_both_groups_zero_variance_does_not_crash(self):
        # Both groups constant (zero variance) and equal to each other.
        same = inf.two_sample_t_test([7, 7, 7], [7, 7, 7])
        self.assertEqual(same.statistic, 0.0)
        self.assertAlmostEqual(same.p_value, 1.0, places=9)
        # Both groups constant but different from each other.
        different = inf.two_sample_t_test([7, 7, 7], [3, 3, 3])
        self.assertEqual(different.statistic, float("inf"))
        self.assertAlmostEqual(different.p_value, 0.0, places=9)


class TestPairedTTest(unittest.TestCase):
    def test_equals_one_sample_t_test_on_differences(self):
        before = [120, 122, 118, 130, 125]
        after = [115, 119, 120, 124, 121]
        paired = inf.paired_t_test(before, after)
        diffs = [b - a for b, a in zip(before, after)]
        one_sample = inf.one_sample_t_test(diffs, mu0=0)
        self.assertAlmostEqual(paired.statistic, one_sample.statistic, places=9)
        self.assertAlmostEqual(paired.p_value, one_sample.p_value, places=9)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            inf.paired_t_test([1, 2, 3], [1, 2])


class TestProportionZTests(unittest.TestCase):
    def test_one_proportion_known_result(self):
        # 45/100 vs p0=0.5: z = (0.45-0.5)/sqrt(0.5*0.5/100) = -1.0
        result = inf.one_proportion_z_test(45, 100, 0.5)
        self.assertAlmostEqual(result.statistic, -1.0, places=6)

    def test_two_proportion_identical_rates_gives_zero(self):
        result = inf.two_proportion_z_test(50, 100, 50, 100)
        self.assertAlmostEqual(result.statistic, 0.0, places=9)
        self.assertAlmostEqual(result.p_value, 1.0, places=6)

    def test_two_proportion_clear_difference_is_significant(self):
        result = inf.two_proportion_z_test(80, 100, 20, 100)
        self.assertLess(result.p_value, 0.0001)

    def test_out_of_range_successes_raises(self):
        with self.assertRaises(ValueError):
            inf.one_proportion_z_test(150, 100, 0.5)


class TestChiSquared(unittest.TestCase):
    def test_goodness_of_fit_known_result(self):
        # Fair die, 60 rolls, expect 10 each; observed [5,8,9,8,10,20]
        # stat = sum((o-e)^2/e) with e=10 for all 6.
        observed = [5, 8, 9, 8, 10, 20]
        expected = [10] * 6
        result = inf.chi_square_goodness_of_fit(observed, expected)
        expected_stat = sum((o - 10) ** 2 / 10 for o in observed)
        self.assertAlmostEqual(result.statistic, expected_stat, places=9)
        self.assertEqual(result.df, 5)

    def test_perfect_fit_gives_zero_statistic(self):
        result = inf.chi_square_goodness_of_fit([10, 10, 10], [10, 10, 10])
        self.assertAlmostEqual(result.statistic, 0.0, places=9)
        self.assertAlmostEqual(result.p_value, 1.0, places=6)

    def test_independence_known_2x2_result(self):
        # Classic worked example: [[10,20],[30,40]]
        table = [[10, 20], [30, 40]]
        result = inf.chi_square_independence(table)
        self.assertEqual(result.df, 1)
        # row totals 30,70; col totals 40,60; total 100
        # expected[0][0] = 30*40/100 = 12
        expected_stat = (
            (10 - 12) ** 2 / 12 + (20 - 18) ** 2 / 18 + (30 - 28) ** 2 / 28 + (40 - 42) ** 2 / 42
        )
        self.assertAlmostEqual(result.statistic, expected_stat, places=9)


class TestOneWayAnova(unittest.TestCase):
    def test_identical_groups_give_zero_statistic(self):
        result = inf.one_way_anova([1, 2, 3], [1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(result.statistic, 0.0, places=9)
        self.assertAlmostEqual(result.p_value, 1.0, places=6)

    def test_known_textbook_result(self):
        # Classic one-way ANOVA worked example (3 groups of 3):
        g1, g2, g3 = [4, 5, 6], [7, 8, 9], [1, 2, 3]
        result = inf.one_way_anova(g1, g2, g3)
        self.assertEqual(result.df, 2)
        self.assertEqual(result.df2, 6)
        # grand mean = 5; between-group SS = 3*((5-5)^2+(8-5)^2+(2-5)^2) = 3*18=54
        # within-group SS = 3 groups * (sum((x-mean)^2)) = 3*2 = 6 (each group has variance sum=2)
        self.assertAlmostEqual(result.statistic, (54 / 2) / (6 / 6), places=6)

    def test_requires_at_least_two_groups(self):
        with self.assertRaises(ValueError):
            inf.one_way_anova([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
