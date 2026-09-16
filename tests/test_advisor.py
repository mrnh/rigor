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

    def test_paired_proportions_routes_to_mcnemar(self):
        r = advisor.recommend_test("proportion", n_groups=2, paired=True)
        self.assertEqual(r.recommended_tool, "mcnemar_test")
        self.assertEqual(r.alternative_tool, "mcnemar_exact_test")

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


class TestRecommendTestCheckedRepeatedly(unittest.TestCase):
    """checked_repeatedly is off by default (every case above passes
    with the parameter simply omitted) -- these check the flag actually
    changes behavior where a sequential tool exists, and is honest about
    where one doesn't, rather than silently ignoring the flag."""

    def test_default_is_off_and_does_not_change_existing_behavior(self):
        with_default = advisor.recommend_test("continuous", n_groups=2)
        explicit_false = advisor.recommend_test("continuous", n_groups=2, checked_repeatedly=False)
        self.assertEqual(with_default, explicit_false)
        self.assertEqual(with_default.recommended_tool, "two_sample_t_test")

    def test_two_independent_groups_continuous_routes_to_sequential(self):
        r = advisor.recommend_test("continuous", n_groups=2, checked_repeatedly=True)
        self.assertEqual(r.recommended_tool, "sequential_two_sample_mean_test")
        self.assertEqual(r.alternative_tool, "two_sample_t_test")

    def test_two_independent_proportions_routes_to_sequential(self):
        r = advisor.recommend_test("proportion", n_groups=2, checked_repeatedly=True)
        self.assertEqual(r.recommended_tool, "sequential_two_proportion_test")
        self.assertEqual(r.alternative_tool, "two_proportion_z_test")

    def test_paired_continuous_has_no_sequential_alternative_yet(self):
        # No sequential paired/one-sample test is implemented -- the
        # ordinary recommendation should stand, flagged with a caveat
        # rather than silently swapped for something that doesn't exist.
        r = advisor.recommend_test("continuous", n_groups=2, paired=True, checked_repeatedly=True)
        self.assertEqual(r.recommended_tool, "paired_t_test")
        self.assertTrue(any("no always-valid sequential alternative" in c for c in r.caveats))

    def test_skewed_continuous_has_no_sequential_alternative_yet(self):
        r = advisor.recommend_test("continuous", n_groups=2, small_or_skewed=True, checked_repeatedly=True)
        self.assertEqual(r.recommended_tool, "mann_whitney_u")
        self.assertTrue(any("no always-valid sequential alternative" in c for c in r.caveats))

    def test_three_plus_groups_has_no_sequential_alternative_yet(self):
        r = advisor.recommend_test("continuous", n_groups=3, checked_repeatedly=True)
        self.assertEqual(r.recommended_tool, "one_way_anova")
        self.assertTrue(any("no always-valid sequential alternative" in c for c in r.caveats))

    def test_paired_proportions_routes_to_mcnemar_with_a_sequential_gap_caveat(self):
        # McNemar's test itself is implemented (no longer a gap), but
        # there's still no sequential/always-valid version of it.
        r = advisor.recommend_test("proportion", n_groups=2, paired=True, checked_repeatedly=True)
        self.assertEqual(r.recommended_tool, "mcnemar_test")
        self.assertTrue(any("no always-valid sequential alternative" in c for c in r.caveats))

    def test_association_is_not_mistaken_for_a_two_group_comparison(self):
        # Regression guard: testing_association reuses outcome_type to
        # mean "type of variable," not "type of comparison" -- with
        # n_groups left at its default of 2, checked_repeatedly must
        # not be misread as "two independent groups, continuous outcome"
        # and wrongly recommend sequential_two_sample_mean_test for a
        # correlation question.
        r = advisor.recommend_test("continuous", testing_association=True, checked_repeatedly=True)
        self.assertEqual(r.recommended_tool, "pearson_correlation")
        self.assertTrue(any("no always-valid sequential alternative" in c for c in r.caveats))


if __name__ == "__main__":
    unittest.main()
