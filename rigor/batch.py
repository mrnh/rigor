"""Run the same comparison across many groups/hypotheses at once, with
multiple-comparisons correction applied automatically.

corrections.py already implements Bonferroni and Benjamini-Hochberg,
but using them correctly after, say, one_way_anova means the agent
itself has to remember to make k*(k-1)/2 separate two_sample_t_test
calls, collect their p-values, and pass those to
bonferroni_correction/benjamini_hochberg_correction -- three separate
tools, orchestrated by hand, with "forgot the correction" being one of
the most common real mistakes this package exists to prevent in the
first place. pairwise_group_comparisons runs that whole pipeline in one
call: it's exactly what one_way_anova's own docstring already
recommends doing after a significant result, just done for you rather
than described to you.

No new statistics here -- every number comes from an already-tested
function in inference.py, nonparametric.py, effect_size.py, or
corrections.py. This module's only job is orchestration.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from rigor import corrections, effect_size, inference, nonparametric


@dataclass
class PairwiseComparison:
    group_i: int
    group_j: int
    label_i: Optional[str]
    label_j: Optional[str]
    statistic: float
    p_value: float
    p_value_adjusted_significant: bool
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None


@dataclass
class PairwiseComparisonResult:
    comparisons: List[PairwiseComparison]
    test: str
    correction_method: str
    alpha: float
    adjusted_alpha: Optional[float]
    citation: str
    warnings: List[str] = field(default_factory=list)


def pairwise_group_comparisons(
    groups: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    test: str = "t_test",
    equal_var: bool = False,
    correction: str = "bh",
    alpha: float = 0.05,
) -> PairwiseComparisonResult:
    """Run every pairwise comparison across 2+ groups (k*(k-1)/2 tests
    for k groups) and correct the resulting p-values for multiple
    comparisons in one call. The natural follow-up after a significant
    one_way_anova/kruskal_wallis result -- pass it the same groups -- to
    find *which* group(s) differ, not just whether any do.

    test: "t_test" runs two_sample_t_test on each pair (equal_var
    controls Welch's vs. pooled, same as that function) and reports
    cohens_d as the effect size; "mann_whitney" runs mann_whitney_u on
    each pair (the non-parametric choice, matching kruskal_wallis) and
    reports rank_biserial_correlation.

    correction: "bh" (Benjamini-Hochberg, the default -- controls false
    discovery rate, less conservative) or "bonferroni" (controls
    family-wise error rate, more conservative) or "none" (report raw
    p-values, e.g. if you're going to correct elsewhere).
    """
    k = len(groups)
    if k < 2:
        raise ValueError("need at least 2 groups")
    if labels is not None and len(labels) != k:
        raise ValueError("labels must be the same length as groups")
    if test not in ("t_test", "mann_whitney"):
        raise ValueError('test must be "t_test" or "mann_whitney"')
    if correction not in ("bonferroni", "bh", "none"):
        raise ValueError('correction must be "bonferroni", "bh", or "none"')

    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    raw_results = []
    for i, j in pairs:
        if test == "t_test":
            result = inference.two_sample_t_test(groups[i], groups[j], equal_var=equal_var)
            d = effect_size.cohens_d(groups[i], groups[j])
            es, es_name = (d if math.isfinite(d) else None), "cohens_d"
        else:
            result = nonparametric.mann_whitney_u(groups[i], groups[j])
            es = effect_size.rank_biserial_correlation(result.statistic, len(groups[i]), len(groups[j]))
            es_name = "rank_biserial_correlation"
        raw_results.append((i, j, result, es, es_name))

    p_values = [r.p_value for _, _, r, _, _ in raw_results]
    if correction == "none":
        significant = [p < alpha for p in p_values]
        adjusted_alpha = None
        citation = "No correction applied -- raw per-pair p-values against alpha."
        warnings = [f"{len(pairs)} comparisons run with no multiple-comparisons correction -- the family-wise false positive rate is inflated above alpha."]
    elif correction == "bonferroni":
        corr_result = corrections.bonferroni(p_values, alpha)
        significant = corr_result.reject
        adjusted_alpha = corr_result.adjusted_alpha
        citation = corr_result.citation
        warnings = []
    else:  # bh
        corr_result = corrections.benjamini_hochberg(p_values, alpha)
        significant = corr_result.reject
        adjusted_alpha = None
        citation = corr_result.citation
        warnings = []

    comparisons = [
        PairwiseComparison(
            group_i=i,
            group_j=j,
            label_i=labels[i] if labels else None,
            label_j=labels[j] if labels else None,
            statistic=result.statistic,
            p_value=result.p_value,
            p_value_adjusted_significant=sig,
            effect_size=es,
            effect_size_name=es_name,
        )
        for (i, j, result, es, es_name), sig in zip(raw_results, significant)
    ]

    test_name = "two_sample_t_test" if test == "t_test" else "mann_whitney_u"
    return PairwiseComparisonResult(
        comparisons=comparisons,
        test=test_name,
        correction_method=correction,
        alpha=alpha,
        adjusted_alpha=adjusted_alpha,
        citation=citation,
        warnings=warnings,
    )
