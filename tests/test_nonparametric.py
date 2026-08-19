import unittest

from rigor import nonparametric as npt


class TestMannWhitneyU(unittest.TestCase):
    def test_complete_separation_gives_extreme_u_and_low_p(self):
        # every value in sample1 ranks below every value in sample2 ->
        # U for sample1 is 0, the minimum possible.
        result = npt.mann_whitney_u([1, 2, 3], [4, 5, 6])
        self.assertEqual(result.statistic, 0.0)
        self.assertLess(result.p_value, 0.2)  # small n limits how low this can go

    def test_identical_distributions_give_high_p_value(self):
        result = npt.mann_whitney_u([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertGreater(result.p_value, 0.5)

    def test_statistic_matches_hand_computation_with_ties(self):
        # combined ranks of [1,2,2,3] & [2,3,4,4]: values 1,2,2,2,3,3,4,4
        # -> ranks 1, 3,3,3 (tie of three 2's at positions 2-4), 5.5,5.5, 7.5,7.5
        s1, s2 = [1, 2, 2, 3], [2, 3, 4, 4]
        result = npt.mann_whitney_u(s1, s2)
        # sample1 ranks: 1 (rank1), 2 (rank3), 2 (rank3), 3 (rank5.5) = sum 12.5
        r1 = 12.5
        u1_expected = r1 - 4 * 5 / 2.0
        self.assertAlmostEqual(result.statistic, u1_expected, places=6)

    def test_empty_sample_raises(self):
        with self.assertRaises(ValueError):
            npt.mann_whitney_u([], [1, 2, 3])


class TestWilcoxonSignedRank(unittest.TestCase):
    def test_all_positive_differences_gives_minimal_statistic(self):
        result = npt.wilcoxon_signed_rank([5, 6, 7, 8], [1, 2, 3, 4])
        self.assertEqual(result.statistic, 0)

    def test_no_systematic_difference_gives_high_p_value(self):
        a = [10, 9, 11, 8, 12, 7, 13, 6, 14, 5]
        b = [9, 10, 8, 11, 7, 12, 6, 13, 5, 14]  # alternating signs, roughly balanced
        result = npt.wilcoxon_signed_rank(a, b)
        self.assertGreater(result.p_value, 0.05)

    def test_zero_differences_are_dropped_and_flagged(self):
        result = npt.wilcoxon_signed_rank([1, 2, 3, 4], [1, 5, 3, 8])
        self.assertTrue(any("dropped" in w for w in result.warnings))

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            npt.wilcoxon_signed_rank([1, 2, 3], [1, 2])

    def test_all_pairs_equal_raises(self):
        with self.assertRaises(ValueError):
            npt.wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])


class TestKruskalWallis(unittest.TestCase):
    def test_matches_hand_computed_example(self):
        # groups [1,2],[3,4],[5,6]: rank sums 3,7,11 over n=6, no ties
        # -> H = (12/42)*(4.5+24.5+60.5) - 21 = 4.571428... -- computed
        # independently of this module.
        result = npt.kruskal_wallis([1, 2], [3, 4], [5, 6])
        self.assertAlmostEqual(result.statistic, 4.571428571428571, places=6)
        self.assertEqual(result.df, 2)

    def test_identical_groups_give_zero_statistic(self):
        result = npt.kruskal_wallis([1, 2, 3], [1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(result.statistic, 0.0, places=9)
        self.assertAlmostEqual(result.p_value, 1.0, places=6)

    def test_well_separated_groups_are_significant(self):
        result = npt.kruskal_wallis([1, 2, 3], [10, 11, 12], [20, 21, 22])
        self.assertLess(result.p_value, 0.05)

    def test_too_few_groups_raises(self):
        with self.assertRaises(ValueError):
            npt.kruskal_wallis([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
