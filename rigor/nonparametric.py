"""Non-parametric alternatives to the t-test / one-way ANOVA family.

These trade a parametric test's distributional assumption (normal
populations) for a weaker one -- only that observations are ranked
meaningfully -- by working on ranks rather than raw values. Reach for
one of these instead of its parametric counterpart when that test's own
small-n warning makes a normal-theory result suspect, or when the data
is ordinal to begin with:

    mann_whitney_u        <-> two_sample_t_test (independent samples)
    wilcoxon_signed_rank  <-> paired_t_test (paired samples)
    kruskal_wallis        <-> one_way_anova (3+ independent groups)

All three statistics here use a normal or chi-squared large-sample
approximation with a tie correction (valid asymptotically); exact
permutation p-values, from enumerating every rank arrangement, differ
for very small samples -- most noticeably under ~20 observations per
group for Mann-Whitney/Wilcoxon, or groups under 5 for Kruskal-Wallis.
That's flagged in each result's warnings rather than silently applied.
"""
import math
from typing import List, Sequence, Tuple

from rigor import distributions as dist
from rigor.inference import TestResult


def _ranks_with_ties(xs: Sequence[float]) -> Tuple[List[float], List[int]]:
    """Average ('fractional') ranks, 1-indexed, ties sharing the mean
    rank. Also returns the size of each tied group, for tie-correction
    terms."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    tie_sizes = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        tie_sizes.append(j - i + 1)
        i = j + 1
    return ranks, tie_sizes


def mann_whitney_u(sample1: Sequence[float], sample2: Sequence[float]) -> TestResult:
    """H0: values from population 1 are not systematically larger or
    smaller than values from population 2 (a "stochastic equality"
    test). The non-parametric alternative to two_sample_t_test --
    combined-data ranks instead of assuming normal populations.
    ``statistic`` is U for sample1 (scipy's convention); pair with
    effect_size.rank_biserial_correlation for a standardized effect
    size."""
    n1, n2 = len(sample1), len(sample2)
    if n1 < 1 or n2 < 1:
        raise ValueError("need at least 1 observation per sample")
    combined = list(sample1) + list(sample2)
    ranks, tie_sizes = _ranks_with_ties(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    tie_term = sum(t ** 3 - t for t in tie_sizes)
    mean_u = n1 * n2 / 2.0
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0

    warnings = []
    if var_u <= 0.0:
        z = 0.0
        warnings.append("every value is tied -- the test has no power to detect a difference.")
    else:
        diff = u1 - mean_u
        cc = math.copysign(0.5, diff) if diff != 0 else 0.0
        z = (diff - cc) / math.sqrt(var_u)
    p = 2 * (1 - dist.normal_cdf(abs(z)))
    if min(n1, n2) < 20:
        warnings.append(f"min(n1, n2)={min(n1, n2)} is small; the normal approximation for the p-value is less accurate below ~20 per group.")

    return TestResult(
        name="Mann-Whitney U test",
        statistic=u1,
        p_value=p,
        citation="Mann-Whitney U test (Mann & Whitney, 1947); normal approximation with continuity and tie correction.",
        warnings=warnings,
    )


def wilcoxon_signed_rank(sample1: Sequence[float], sample2: Sequence[float]) -> TestResult:
    """H0: the median of the paired differences is zero. The
    non-parametric alternative to paired_t_test -- ranks the absolute
    differences instead of assuming normally distributed differences.
    Pairs with a zero difference are dropped (and counted in a warning),
    the standard Wilcoxon procedure. ``statistic`` is T = min(W+, W-)."""
    if len(sample1) != len(sample2):
        raise ValueError("paired samples must be the same length")
    diffs = [a - b for a, b in zip(sample1, sample2) if a != b]
    n_dropped = len(sample1) - len(diffs)
    if len(diffs) < 1:
        raise ValueError("no nonzero differences -- nothing to test")

    abs_diffs = [abs(d) for d in diffs]
    ranks, tie_sizes = _ranks_with_ties(abs_diffs)
    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    n = len(diffs)
    mean_w = n * (n + 1) / 4.0
    tie_term = sum(t ** 3 - t for t in tie_sizes)
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0

    warnings = [f"{n_dropped} pair(s) with a zero difference were dropped."] if n_dropped else []
    if var_w <= 0.0:
        z = 0.0
        warnings.append("every nonzero difference is tied -- the test has no power to detect an effect.")
    else:
        diff = w_pos - mean_w
        cc = math.copysign(0.5, diff) if diff != 0 else 0.0
        z = (diff - cc) / math.sqrt(var_w)
    p = 2 * (1 - dist.normal_cdf(abs(z)))
    if n < 20:
        warnings.append(f"n={n} nonzero difference(s) is small; the normal approximation for the p-value is less accurate below ~20.")

    return TestResult(
        name="Wilcoxon signed-rank test",
        statistic=min(w_pos, w_neg),
        p_value=p,
        citation="Wilcoxon signed-rank test (Wilcoxon, 1945); normal approximation with continuity and tie correction.",
        warnings=warnings,
    )


def kruskal_wallis(*groups: Sequence[float]) -> TestResult:
    """H0: all groups are drawn from the same distribution (equal
    medians, under the usual equal-shape assumption). The non-parametric
    alternative to one_way_anova -- ranks the combined data instead of
    assuming normal populations. A significant result means at least one
    group differs, not which one -- same caveat as one_way_anova."""
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    combined = [x for g in groups for x in g]
    n = len(combined)
    ranks, tie_sizes = _ranks_with_ties(combined)

    h = 0.0
    idx = 0
    for g in groups:
        ng = len(g)
        if ng < 1:
            raise ValueError("every group needs at least 1 observation")
        rank_sum = sum(ranks[idx:idx + ng])
        h += rank_sum ** 2 / ng
        idx += ng
    h = (12.0 / (n * (n + 1))) * h - 3 * (n + 1)

    tie_term = sum(t ** 3 - t for t in tie_sizes)
    correction = 1.0 - tie_term / (n ** 3 - n) if n > 1 else 1.0
    if correction > 0:
        h /= correction

    df = len(groups) - 1
    p = 1 - dist.chi2_cdf(h, df)
    warnings = []
    if any(len(g) < 5 for g in groups):
        warnings.append("some groups have fewer than 5 observations; the chi-squared approximation is less reliable there.")

    return TestResult(
        name="Kruskal-Wallis H test",
        statistic=h,
        df=df,
        p_value=p,
        citation="Kruskal-Wallis H test (Kruskal & Wallis, 1952); chi-squared approximation with tie correction.",
        warnings=warnings,
    )
