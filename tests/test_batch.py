import unittest

from rigor import batch, inference


class TestPairwiseGroupComparisons(unittest.TestCase):
    def test_covers_every_pair_exactly_once(self):
        groups = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
        result = batch.pairwise_group_comparisons(groups)
        pairs = {(c.group_i, c.group_j) for c in result.comparisons}
        self.assertEqual(pairs, {(0, 1), (0, 2), (1, 2)})

    def test_t_test_statistics_match_calling_two_sample_t_test_directly(self):
        groups = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
        result = batch.pairwise_group_comparisons(groups)
        direct = inference.two_sample_t_test(groups[0], groups[2])
        c = next(c for c in result.comparisons if (c.group_i, c.group_j) == (0, 2))
        self.assertAlmostEqual(c.statistic, direct.statistic, places=9)
        self.assertAlmostEqual(c.p_value, direct.p_value, places=9)

    def test_bh_correction_is_never_more_conservative_than_bonferroni(self):
        # BH's adjusted decisions should reject at least as many
        # hypotheses as Bonferroni's, for the same data -- BH is the
        # less conservative procedure by construction.
        groups = [[1, 2, 3], [10, 11, 12], [20, 21, 22], [1, 2, 3]]
        bh = batch.pairwise_group_comparisons(groups, correction="bh")
        bonf = batch.pairwise_group_comparisons(groups, correction="bonferroni")
        bh_rejects = sum(c.p_value_adjusted_significant for c in bh.comparisons)
        bonf_rejects = sum(c.p_value_adjusted_significant for c in bonf.comparisons)
        self.assertGreaterEqual(bh_rejects, bonf_rejects)

    def test_no_correction_uses_raw_alpha_and_warns(self):
        groups = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
        result = batch.pairwise_group_comparisons(groups, correction="none")
        self.assertTrue(result.warnings)
        c = next(c for c in result.comparisons if (c.group_i, c.group_j) == (0, 2))
        self.assertTrue(c.p_value_adjusted_significant)  # p is well below 0.05 uncorrected too

    def test_mann_whitney_path_uses_rank_biserial_effect_size(self):
        groups = [[1, 2, 3], [4, 5, 6]]
        result = batch.pairwise_group_comparisons(groups, test="mann_whitney")
        self.assertEqual(result.test, "mann_whitney_u")
        self.assertEqual(result.comparisons[0].effect_size_name, "rank_biserial_correlation")

    def test_labels_carried_through(self):
        groups = [[1, 2, 3], [4, 5, 6]]
        result = batch.pairwise_group_comparisons(groups, labels=["control", "treatment"])
        c = result.comparisons[0]
        self.assertEqual((c.label_i, c.label_j), ("control", "treatment"))

    def test_zero_variance_pairs_report_none_effect_size_not_inf(self):
        groups = [[5, 5, 5], [3, 3, 3]]
        result = batch.pairwise_group_comparisons(groups)
        self.assertIsNone(result.comparisons[0].effect_size)

    def test_too_few_groups_raises(self):
        with self.assertRaises(ValueError):
            batch.pairwise_group_comparisons([[1, 2, 3]])

    def test_mismatched_labels_raises(self):
        with self.assertRaises(ValueError):
            batch.pairwise_group_comparisons([[1, 2], [3, 4]], labels=["only_one"])

    def test_invalid_test_raises(self):
        with self.assertRaises(ValueError):
            batch.pairwise_group_comparisons([[1, 2], [3, 4]], test="not_a_real_test")

    def test_invalid_correction_raises(self):
        with self.assertRaises(ValueError):
            batch.pairwise_group_comparisons([[1, 2], [3, 4]], correction="not_a_real_correction")


if __name__ == "__main__":
    unittest.main()
