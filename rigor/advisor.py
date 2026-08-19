"""Which test do I even use? -- a decision helper over the rest of
rigor.

Every test in this package already documents, in its own docstring,
when to reach for it over a sibling (two_sample_t_test's docstring
points at paired_t_test for matched subjects and at mann_whitney_u for
skewed data; one_way_anova's points at pairwise follow-ups; and so on).
That's fine for a person reading source, but an agent calling tools
blind has to already know which test it wants before it can find that
guidance -- the exact chicken-and-egg problem this module exists to
remove.

recommend_test() is pure decision logic over a handful of
already-known characteristics of the data/question (no statistics
computed here, nothing that needs numerical verification) -- it just
compiles the same decision points scattered across every other
module's docstrings into one callable answer: which tool to call, why,
what to call instead if an assumption looks shaky, and what to run
alongside it (an effect size, a power calculation, a follow-up).
"""
from dataclasses import dataclass, field
from typing import List, Optional

_OUTCOME_TYPES = ("continuous", "proportion", "count_or_category", "rank_or_ordinal")


@dataclass
class TestRecommendation:
    recommended_tool: str
    reasoning: str
    alternative_tool: Optional[str] = None
    alternative_reasoning: Optional[str] = None
    effect_size_tool: Optional[str] = None
    power_tool: Optional[str] = None
    next_steps: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


def _continuous_recommendation(n_groups: int, paired: bool, small_or_skewed: bool) -> TestRecommendation:
    if n_groups == 1:
        rec = TestRecommendation(
            recommended_tool="one_sample_t_test",
            reasoning="Comparing a single sample's mean against a hypothesized value.",
            power_tool="power_for_one_sample_t_test / sample_size_for_one_sample_t_test",
        )
        if small_or_skewed:
            rec.caveats.append(
                "No one-sample non-parametric test is implemented here yet -- if "
                "normality is a serious concern, treat the p-value cautiously."
            )
        return rec

    if n_groups == 2:
        if paired:
            if small_or_skewed:
                return TestRecommendation(
                    recommended_tool="wilcoxon_signed_rank",
                    reasoning=(
                        "Paired measurements (e.g. before/after on the same subjects) "
                        "with a small sample or non-normal differences -- ranks the "
                        "absolute differences instead of assuming they're normally "
                        "distributed."
                    ),
                    alternative_tool="paired_t_test",
                    alternative_reasoning="More powerful if the differences really are approximately normal.",
                )
            return TestRecommendation(
                recommended_tool="paired_t_test",
                reasoning="Paired measurements (e.g. before/after on the same subjects).",
                alternative_tool="wilcoxon_signed_rank",
                alternative_reasoning="Use instead if the differences look skewed, have outliers, or n is small.",
                effect_size_tool="cohens_d (on the paired differences)",
                power_tool="power_for_one_sample_t_test / sample_size_for_one_sample_t_test",
            )
        if small_or_skewed:
            return TestRecommendation(
                recommended_tool="mann_whitney_u",
                reasoning=(
                    "Two independent samples with a small sample size or visibly "
                    "skewed/outlier-heavy data -- ranks the combined data instead of "
                    "assuming normal populations."
                ),
                alternative_tool="two_sample_t_test",
                alternative_reasoning="More powerful if both groups are reasonably normal.",
                effect_size_tool="rank_biserial_correlation",
            )
        return TestRecommendation(
            recommended_tool="two_sample_t_test",
            reasoning="Two independent samples, comparing means.",
            alternative_tool="mann_whitney_u",
            alternative_reasoning="Use instead if the data is skewed, has outliers, or n is small per group.",
            effect_size_tool="cohens_d",
            power_tool="power_for_two_sample_t_test / sample_size_for_two_sample_t_test",
            next_steps=["Run levene_test first if you're unsure whether to pass equal_var=True or keep the default Welch's test."],
        )

    # 3+ groups
    if small_or_skewed:
        rec = TestRecommendation(
            recommended_tool="kruskal_wallis",
            reasoning=(
                "Three or more independent groups with a small sample size or "
                "skewed/outlier-heavy data -- ranks the combined data instead of "
                "assuming normal populations."
            ),
            alternative_tool="one_way_anova",
            alternative_reasoning="More powerful if the groups are reasonably normal.",
        )
    else:
        rec = TestRecommendation(
            recommended_tool="one_way_anova",
            reasoning="Three or more independent groups, comparing means.",
            alternative_tool="kruskal_wallis",
            alternative_reasoning="Use instead if the data is skewed, has outliers, or group sizes are small.",
            effect_size_tool="eta_squared / omega_squared",
            next_steps=["Run levene_test first to check the equal-variance assumption."],
        )
    rec.next_steps.append(
        "A significant result means at least one group differs, not which -- follow "
        "up with pairwise_group_comparisons (runs every pairwise test and corrects "
        "for multiple comparisons automatically)."
    )
    return rec


def recommend_test(
    outcome_type: str,
    n_groups: int = 2,
    paired: bool = False,
    small_or_skewed: bool = False,
    two_categorical_variables: bool = False,
    testing_association: bool = False,
) -> TestRecommendation:
    """Recommend which rigor tool fits a question, and what to reach for
    if this test's assumptions don't hold.

    outcome_type: "continuous" (means), "proportion" (rates/counts of
    successes), "count_or_category" (category counts / contingency
    tables), or "rank_or_ordinal" (data that's only meaningfully
    ordered, not measured -- always routed to a rank-based test).

    n_groups: 1 = one sample vs. a hypothesized value; 2 = two groups
    or conditions; 3+ = three or more groups. Ignored when
    testing_association is set.

    paired: for n_groups == 2 (continuous/rank_or_ordinal) or
    proportion, whether the same subjects were measured twice rather
    than two independent groups.

    small_or_skewed: sample is small, visibly skewed, or has outliers
    -- nudges toward the non-parametric alternative.

    two_categorical_variables: for outcome_type="count_or_category",
    whether this is testing association between two categorical
    variables (a contingency table) rather than observed counts against
    an expected distribution.

    testing_association: this is "does x relate to/predict y" for two
    continuous (or ranked) variables, not a group comparison -- routes
    to correlation/regression instead.
    """
    if outcome_type not in _OUTCOME_TYPES:
        raise ValueError(f"outcome_type must be one of {_OUTCOME_TYPES}")
    if n_groups < 1:
        raise ValueError("n_groups must be at least 1")

    if testing_association:
        if small_or_skewed or outcome_type == "rank_or_ordinal":
            return TestRecommendation(
                recommended_tool="spearman_correlation",
                reasoning=(
                    "Testing association between two variables where the "
                    "relationship may not be linear, or outliers/ordinal data "
                    "shouldn't dominate -- correlates ranks instead of raw values."
                ),
                alternative_tool="pearson_correlation",
                alternative_reasoning="If the relationship is expected to be linear and the data is well-behaved, this is more powerful.",
                next_steps=["Follow up with simple_linear_regression if you need the actual slope (units of y per unit of x), not just the strength of association."],
            )
        return TestRecommendation(
            recommended_tool="pearson_correlation",
            reasoning="Testing for a linear association between two continuous variables.",
            alternative_tool="spearman_correlation",
            alternative_reasoning="Use instead if the relationship might be monotonic-but-not-linear, or outliers shouldn't dominate.",
            next_steps=["Follow up with simple_linear_regression for the slope itself."],
        )

    if outcome_type == "continuous":
        return _continuous_recommendation(n_groups, paired, small_or_skewed)

    if outcome_type == "rank_or_ordinal":
        rec = _continuous_recommendation(n_groups, paired, small_or_skewed=True)
        rec.caveats.append("Ordinal data -- routed straight to the rank-based test regardless of sample size, since the values themselves aren't on an interval scale.")
        return rec

    if outcome_type == "proportion":
        if n_groups == 1:
            return TestRecommendation(
                recommended_tool="one_proportion_z_test",
                reasoning="Comparing a single observed proportion against a hypothesized value.",
            )
        if n_groups == 2:
            rec = TestRecommendation(
                recommended_tool="two_proportion_z_test",
                reasoning="Comparing two independent proportions (e.g. conversion rates between two groups).",
                effect_size_tool="cohens_h",
                power_tool="power_for_two_proportion_test / sample_size_for_two_proportion_test",
            )
            if paired:
                rec.caveats.append(
                    "These are paired/matched proportions (e.g. the same subjects "
                    "before/after) -- two_proportion_z_test assumes independent "
                    "groups and isn't quite right here. No matched-pairs proportion "
                    "test (e.g. McNemar's) is implemented yet; treat this as a known gap."
                )
            return rec
        return TestRecommendation(
            recommended_tool="chi_square_independence",
            reasoning=(
                "Comparing proportions across 3+ groups is a test of association "
                "between group membership and outcome -- build a contingency table "
                "(groups as rows, success/failure as columns)."
            ),
            effect_size_tool="cramers_v",
        )

    # count_or_category
    if two_categorical_variables:
        rec = TestRecommendation(
            recommended_tool="chi_square_independence",
            reasoning="Testing whether two categorical variables are associated (a contingency table).",
            effect_size_tool="cramers_v",
        )
        rec.next_steps.append(
            "If the table is 2x2 and any expected cell count would be below 5 "
            "(chi_square_independence will warn you), use fisher_exact_test instead "
            "-- it's exact rather than approximate."
        )
        return rec
    return TestRecommendation(
        recommended_tool="chi_square_goodness_of_fit",
        reasoning="Testing whether observed category counts match an expected/hypothesized distribution.",
    )
