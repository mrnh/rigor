"""These check the decision logic's outcomes, not any statistics (there
are none here) -- each case asserts recommend_test names the tool this
package actually documents as the right one for that scenario."""
import unittest

from rigor import advisor


class TestRecommendTest(unittest.TestCase):
    def test_one_sample_continuous(self):
        r = advisor.recommend_test("continuous", n_groups=1)
        self.assertEqual(r.recommended_tool, "one_sample_t_test")

    def test_two_independent_groups_normal(self):
        r = advisor.recommend_test("continuous", n_groups=2, paired=False, small_or_skewed=False)
        self.assertEqual(r.recommended_tool, "two_sample_t_test")
        self.assertEqual(r.alternative_tool, "mann_whitney_u")
        self.assertEqual(r.effect_size_tool, "cohens_d")

    def test_two_independent_groups_skewed(self):
        r = advisor.recommend_test("continuous", n_groups=2, paired=False, small_or_skewed=True)
        self.assertEqual(r.recommended_tool, "mann_whitney_u")
        self.assertEqual(r.alternative_tool, "two_sample_t_test")

    def test_paired_normal(self):
        r = advisor.recommend_test("continuous", n_groups=2, paired=True, small_or_skewed=False)
        self.assertEqual(r.recommended_tool, "paired_t_test")

    def test_paired_skewed(self):
        r = advisor.recommend_test("continuous", n_groups=2, paired=True, small_or_skewed=True)
        self.assertEqual(r.recommended_tool, "wilcoxon_signed_rank")

    def test_three_plus_groups_normal(self):
        r = advisor.recommend_test("continuous", n_groups=4, small_or_skewed=False)
        self.assertEqual(r.recommended_tool, "one_way_anova")
        self.assertTrue(any("pairwise_group_comparisons" in step for step in r.next_steps))

    def test_three_plus_groups_skewed(self):
        r = advisor.recommend_test("continuous", n_groups=4, small_or_skewed=True)
        self.assertEqual(r.recommended_tool, "kruskal_wallis")

    def test_rank_or_ordinal_always_nonparametric_even_if_not_flagged_skewed(self):
        r = advisor.recommend_test("rank_or_ordinal", n_groups=2, small_or_skewed=False)
        self.assertEqual(r.recommended_tool, "mann_whitney_u")

    def test_one_proportion(self):
        r = advisor.recommend_test("proportion", n_groups=1)
        self.assertEqual(r.recommended_tool, "one_proportion_z_test")

    def test_two_proportions(self):
        r = advisor.recommend_test("proportion", n_groups=2)
        self.assertEqual(r.recommended_tool, "two_proportion_z_test")
        self.assertEqual(r.effect_size_tool, "cohens_h")

    def test_paired_proportions_flagged_as_a_gap(self):
        r = advisor.recommend_test("proportion", n_groups=2, paired=True)
        self.assertTrue(any("McNemar" in c for c in r.caveats))

    def test_three_plus_proportions_routes_to_chi_square(self):
        r = advisor.recommend_test("proportion", n_groups=3)
        self.assertEqual(r.recommended_tool, "chi_square_independence")

    def test_goodness_of_fit(self):
        r = advisor.recommend_test("count_or_category", two_categorical_variables=False)
        self.assertEqual(r.recommended_tool, "chi_square_goodness_of_fit")

    def test_independence(self):
        r = advisor.recommend_test("count_or_category", two_categorical_variables=True)
        self.assertEqual(r.recommended_tool, "chi_square_independence")
        self.assertTrue(any("fisher_exact_test" in step for step in r.next_steps))

    def test_association_linear(self):
        r = advisor.recommend_test("continuous", testing_association=True, small_or_skewed=False)
        self.assertEqual(r.recommended_tool, "pearson_correlation")

    def test_association_nonlinear_or_skewed(self):
        r = advisor.recommend_test("continuous", testing_association=True, small_or_skewed=True)
        self.assertEqual(r.recommended_tool, "spearman_correlation")

    def test_invalid_outcome_type_raises(self):
        with self.assertRaises(ValueError):
            advisor.recommend_test("not_a_real_type")

    def test_zero_groups_raises(self):
        with self.assertRaises(ValueError):
            advisor.recommend_test("continuous", n_groups=0)


if __name__ == "__main__":
    unittest.main()
