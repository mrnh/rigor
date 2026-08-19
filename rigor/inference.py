"""Classical hypothesis tests, built on the distributions module.

Every test returns a ``TestResult`` — statistic, degrees of freedom
(where applicable), p-value, a confidence interval where one has a
standard closed form, a human-readable citation for the formula used,
and a list of warnings about violated or untestable assumptions. The
point isn't just "here's a p-value" but "here's a p-value, here's
exactly what produced it, and here's what to be suspicious of" — the
same instinct behind ``vault`` not asking for passwords: be useful
without quietly overreaching what's actually known.
"""
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from rigor import distributions as dist


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    df: Optional[float] = None
    df2: Optional[float] = None  # second degrees-of-freedom, for F-tests (df, df2)
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_level: Optional[float] = None
    citation: str = ""
    warnings: List[str] = field(default_factory=list)

    def reject_null(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: Sequence[float], mean: Optional[float] = None) -> float:
    m = _mean(xs) if mean is None else mean
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations to estimate variance")
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _safe_ratio(diff: float, se: float) -> float:
    """diff / se, defined at se == 0: a zero difference is "no evidence of
    an effect" (t=0); a nonzero difference against zero variance is treated
    as an infinitely large statistic rather than raising ZeroDivisionError
    (which Python's float division does, unlike IEEE-754 semantics)."""
    if se == 0.0:
        if diff == 0.0:
            return 0.0
        return math.inf if diff > 0 else -math.inf
    return diff / se


def _small_sample_warning(n: int, threshold: int = 30) -> List[str]:
    if n < threshold:
        return [
            f"n={n} is small; the t-test's validity leans on the population "
            "being approximately normal rather than on the Central Limit "
            "Theorem kicking in. Check for skew/outliers if that's not a "
            "safe assumption here."
        ]
    return []


def one_sample_t_test(sample: Sequence[float], mu0: float, confidence_level: float = 0.95) -> TestResult:
    """H0: the population mean equals mu0."""
    n = len(sample)
    if n < 2:
        raise ValueError("need at least 2 observations")
    m = _mean(sample)
    s2 = _variance(sample, m)
    se = math.sqrt(s2 / n)
    df = n - 1
    t = _safe_ratio(m - mu0, se)
    p = 2 * (1 - dist.t_cdf(abs(t), df))
    tcrit = dist.t_critical(1 - confidence_level, df)
    ci = (m - tcrit * se, m + tcrit * se)
    return TestResult(
        name="one-sample t-test",
        statistic=t,
        df=df,
        p_value=p,
        confidence_interval=ci,
        confidence_level=confidence_level,
        citation="Student's t-test (Gosset, 1908); two-tailed.",
        warnings=_small_sample_warning(n),
    )


def two_sample_t_test(
    sample1: Sequence[float], sample2: Sequence[float], equal_var: bool = False, confidence_level: float = 0.95
) -> TestResult:
    """H0: the two population means are equal.

    Defaults to Welch's t-test (equal_var=False), which does not assume
    equal population variances — the safer default per Welch (1947);
    set equal_var=True for the classic pooled-variance Student's t-test.
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        raise ValueError("need at least 2 observations per sample")
    m1, m2 = _mean(sample1), _mean(sample2)
    v1, v2 = _variance(sample1, m1), _variance(sample2, m2)

    if equal_var:
        df = n1 + n2 - 2
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(pooled * (1 / n1 + 1 / n2))
        citation = "Student's two-sample t-test, pooled variance."
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        num = (v1 / n1 + v2 / n2) ** 2
        denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        # denom is 0 only when both samples have zero variance (every value
        # identical within each group) -- Welch-Satterthwaite is 0/0 there;
        # fall back to the pooled df, which is exact in that degenerate case.
        df = num / denom if denom > 0 else n1 + n2 - 2
        citation = "Welch's t-test (Welch, 1947); does not assume equal population variances."

    t = _safe_ratio(m1 - m2, se)
    p = 2 * (1 - dist.t_cdf(abs(t), df))
    tcrit = dist.t_critical(1 - confidence_level, df)
    diff = m1 - m2
    ci = (diff - tcrit * se, diff + tcrit * se)
    warnings = _small_sample_warning(min(n1, n2))
    return TestResult(
        name="two-sample t-test" + (" (pooled)" if equal_var else " (Welch)"),
        statistic=t,
        df=df,
        p_value=p,
        confidence_interval=ci,
        confidence_level=confidence_level,
        citation=citation,
        warnings=warnings,
    )


def paired_t_test(sample1: Sequence[float], sample2: Sequence[float], confidence_level: float = 0.95) -> TestResult:
    """H0: the mean of the paired differences is zero."""
    if len(sample1) != len(sample2):
        raise ValueError("paired samples must be the same length")
    diffs = [a - b for a, b in zip(sample1, sample2)]
    result = one_sample_t_test(diffs, mu0=0.0, confidence_level=confidence_level)
    result.name = "paired t-test"
    result.citation = "Paired-samples t-test (equivalent to a one-sample t-test on the differences)."
    return result


def one_proportion_z_test(successes: int, n: int, p0: float, confidence_level: float = 0.95) -> TestResult:
    """H0: the population proportion equals p0."""
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    if not 0 < p0 < 1:
        raise ValueError("p0 must be in (0, 1)")
    p_hat = successes / n
    se_null = math.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / se_null
    p_value = 2 * (1 - dist.normal_cdf(abs(z)))
    se_sample = math.sqrt(p_hat * (1 - p_hat) / n) if 0 < p_hat < 1 else 0.0
    zcrit = dist.normal_ppf(1 - (1 - confidence_level) / 2)
    ci = (p_hat - zcrit * se_sample, p_hat + zcrit * se_sample)
    warnings = []
    if n * p0 < 10 or n * (1 - p0) < 10:
        warnings.append(
            f"n*p0={n*p0:.1f} or n*(1-p0)={n*(1-p0):.1f} is below the usual "
            "rule-of-thumb of 10 for the normal approximation to be reliable."
        )
    return TestResult(
        name="one-proportion z-test",
        statistic=z,
        p_value=p_value,
        confidence_interval=ci,
        confidence_level=confidence_level,
        citation="One-sample z-test for a proportion, normal approximation.",
        warnings=warnings,
    )


def two_proportion_z_test(
    successes1: int, n1: int, successes2: int, n2: int, confidence_level: float = 0.95
) -> TestResult:
    """H0: the two population proportions are equal."""
    if not (0 <= successes1 <= n1 and 0 <= successes2 <= n2):
        raise ValueError("successes must be between 0 and n for each group")
    p1, p2 = successes1 / n1, successes2 / n2
    pooled = (successes1 + successes2) / (n1 + n2)
    se_null = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_null if se_null > 0 else 0.0
    p_value = 2 * (1 - dist.normal_cdf(abs(z)))
    se_sample = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = dist.normal_ppf(1 - (1 - confidence_level) / 2)
    diff = p1 - p2
    ci = (diff - zcrit * se_sample, diff + zcrit * se_sample)
    warnings = []
    for label, n, p in (("group 1", n1, p1), ("group 2", n2, p2)):
        if n * p < 5 or n * (1 - p) < 5:
            warnings.append(f"{label}: n*p or n*(1-p) is below 5 — normal approximation may be unreliable.")
    return TestResult(
        name="two-proportion z-test",
        statistic=z,
        p_value=p_value,
        confidence_interval=ci,
        confidence_level=confidence_level,
        citation="Two-sample z-test for proportions, pooled variance under H0.",
        warnings=warnings,
    )


def chi_square_goodness_of_fit(observed: Sequence[float], expected: Sequence[float]) -> TestResult:
    """H0: the observed category counts follow the given expected distribution."""
    if len(observed) != len(expected):
        raise ValueError("observed and expected must be the same length")
    if len(observed) < 2:
        raise ValueError("need at least 2 categories")
    stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
    df = len(observed) - 1
    p = 1 - dist.chi2_cdf(stat, df)
    warnings = []
    if any(e < 5 for e in expected):
        warnings.append("Some expected counts are below 5 — chi-squared approximation may be unreliable.")
    return TestResult(
        name="chi-squared goodness-of-fit test",
        statistic=stat,
        df=df,
        p_value=p,
        citation="Pearson's chi-squared test (Pearson, 1900).",
        warnings=warnings,
    )


def chi_square_independence(table: Sequence[Sequence[float]]) -> TestResult:
    """H0: the row and column variables of a contingency table are independent."""
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows < 2 or cols < 2:
        raise ValueError("need at least a 2x2 table")
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
    total = sum(row_totals)
    if total == 0:
        raise ValueError("table is empty")
    stat = 0.0
    low_expected = False
    for r in range(rows):
        for c in range(cols):
            expected = row_totals[r] * col_totals[c] / total
            if expected < 5:
                low_expected = True
            if expected > 0:
                stat += (table[r][c] - expected) ** 2 / expected
    df = (rows - 1) * (cols - 1)
    p = 1 - dist.chi2_cdf(stat, df)
    warnings = ["Some expected cell counts are below 5 — consider Fisher's exact test instead."] if low_expected else []
    return TestResult(
        name="chi-squared test of independence",
        statistic=stat,
        df=df,
        p_value=p,
        citation="Pearson's chi-squared test of independence (Pearson, 1900).",
        warnings=warnings,
    )


def one_way_anova(*groups: Sequence[float]) -> TestResult:
    """H0: all group population means are equal."""
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    all_values = [x for g in groups for x in g]
    n_total = len(all_values)
    grand_mean = _mean(all_values)
    k = len(groups)

    ss_between = sum(len(g) * (_mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(sum((x - _mean(g)) ** 2 for x in g) for g in groups)

    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else float("nan")
    f_stat = ms_between / ms_within if ms_within > 0 else float("inf")
    p = 1 - dist.f_cdf(f_stat, df_between, df_within)

    return TestResult(
        name="one-way ANOVA",
        statistic=f_stat,
        df=df_between,
        df2=df_within,
        p_value=p,
        citation="One-way analysis of variance (Fisher, 1925); F(df_between, df_within).",
        warnings=["df_within is small; the F-test's power will be limited."] if df_within < 10 else [],
    )


def levene_test(*groups: Sequence[float]) -> TestResult:
    """H0: all groups have equal population variances (homogeneity of
    variance). Use to decide equal_var for two_sample_t_test, or to
    sanity-check one_way_anova's equal-variance assumption -- rejecting
    H0 here means Welch's t-test (the default) or a variance-robust
    ANOVA is the safer choice over the pooled-variance version.

    Uses the Brown-Forsythe variant (deviations from each group's
    *median* rather than its mean), which is more robust to non-normal
    data than Levene's original mean-based version -- then runs
    one_way_anova on those deviations, which is exactly what the test
    reduces to."""
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    deviations = []
    for g in groups:
        if len(g) < 2:
            raise ValueError("each group needs at least 2 observations")
        med = statistics.median(g)
        deviations.append([abs(x - med) for x in g])
    result = one_way_anova(*deviations)
    return TestResult(
        name="Levene's test (Brown-Forsythe)",
        statistic=result.statistic,
        df=result.df,
        df2=result.df2,
        p_value=result.p_value,
        citation=(
            "Levene's test (Levene, 1960), Brown-Forsythe median-based variant "
            "(Brown & Forsythe, 1974); computed as a one-way ANOVA on absolute "
            "deviations from each group's median."
        ),
        warnings=result.warnings,
    )


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_pmf(a: int, row1: int, row2: int, col1: int) -> float:
    """P(top-left cell == a) for a 2x2 table with fixed margins row1/row2/col1."""
    return math.exp(_log_choose(row1, a) + _log_choose(row2, col1 - a) - _log_choose(row1 + row2, col1))


def fisher_exact_test(table: Sequence[Sequence[float]]) -> TestResult:
    """H0: the row and column variables of a 2x2 contingency table are
    independent. Exact -- computed from the hypergeometric distribution
    over every table with the same row/column totals, rather than the
    chi-squared approximation chi_square_independence uses -- so it's
    the right choice for small samples, or whenever
    chi_square_independence warns that an expected cell count is below
    5. 2x2 tables only; entries must be non-negative integers.

    ``statistic`` is the sample odds ratio ((a*d)/(b*c) for table
    [[a,b],[c,d]]); +-inf/0 when a zero cell makes it degenerate. The
    p-value is two-tailed, summing every table's probability that is no
    larger than the observed table's."""
    if len(table) != 2 or len(table[0]) != 2 or len(table[1]) != 2:
        raise ValueError("fisher_exact_test needs a 2x2 table")
    (a, b), (c, d) = table
    for v in (a, b, c, d):
        if v != int(v) or v < 0:
            raise ValueError("all cell counts must be non-negative integers")
    a, b, c, d = int(a), int(b), int(c), int(d)
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    n = row1 + row2
    if n == 0:
        raise ValueError("table is empty")

    lo, hi = max(0, col1 - row2), min(row1, col1)
    observed_p = _hypergeom_pmf(a, row1, row2, col1)
    p_value = min(1.0, sum(
        p for x in range(lo, hi + 1)
        if (p := _hypergeom_pmf(x, row1, row2, col1)) <= observed_p * (1 + 1e-7)
    ))
    if b > 0 and c > 0:
        odds_ratio = (a * d) / (b * c)
    elif a > 0 and d > 0:
        odds_ratio = math.inf
    else:
        odds_ratio = 0.0 if (a == 0 or d == 0) else float("nan")

    warnings = []
    if n > 200:
        warnings.append(f"n={n} is fairly large for an exact test -- chi_square_independence's approximation is likely fine here and much faster.")
    return TestResult(
        name="Fisher's exact test",
        statistic=odds_ratio,
        p_value=p_value,
        citation="Fisher's exact test (Fisher, 1922), two-tailed via summing hypergeometric probabilities no larger than the observed table's.",
        warnings=warnings,
    )
