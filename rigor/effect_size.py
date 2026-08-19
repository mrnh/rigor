"""Effect-size measures: how big a difference is, not just whether it's
"significant" (a p-value is a function of sample size as much as effect
size — a huge n makes a trivial difference "significant"; effect size is
the part of the answer that doesn't do that)."""
import math
from typing import Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: Sequence[float]) -> float:
    m = _mean(xs)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations")
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def cohens_d(sample1: Sequence[float], sample2: Sequence[float]) -> float:
    """Standardized mean difference using the pooled standard deviation.

    Rule-of-thumb magnitudes from Cohen (1988): ~0.2 small, ~0.5 medium,
    ~0.8 large -- guidance, not a law; always interpret against domain
    context.
    """
    n1, n2 = len(sample1), len(sample2)
    v1, v2 = _variance(sample1), _variance(sample2)
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0.0:
        diff = _mean(sample1) - _mean(sample2)
        return 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    return (_mean(sample1) - _mean(sample2)) / pooled_sd


def hedges_g(sample1: Sequence[float], sample2: Sequence[float]) -> float:
    """Cohen's d with a small-sample bias correction (Hedges, 1981).

    The correction factor approaches 1 as sample sizes grow, so hedges_g
    converges to cohens_d for large samples but is less biased for small
    ones.
    """
    n1, n2 = len(sample1), len(sample2)
    d = cohens_d(sample1, sample2)
    if not math.isfinite(d):
        return d
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1)
    return d * correction


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions (Cohen, 1988),
    via the arcsine-square-root transform, which stabilizes variance
    across the [0, 1] range better than a raw difference does."""
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("p1 and p2 must be in [0, 1]")
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return phi1 - phi2


def cramers_v(chi2_statistic: float, n: int, rows: int, cols: int) -> float:
    """Effect size for a chi-squared test of independence (Cramér, 1946),
    normalized to [0, 1] regardless of table shape."""
    if chi2_statistic < 0:
        raise ValueError("chi2_statistic must be non-negative")
    if n <= 0:
        raise ValueError("n must be positive")
    k = min(rows, cols)
    if k < 2:
        raise ValueError("need at least a 2x2 table")
    return math.sqrt(chi2_statistic / (n * (k - 1)))


def eta_squared(*groups: Sequence[float]) -> float:
    """Proportion of total variance explained by group membership, for a
    one-way ANOVA (Cohen, 1988 notation; equivalent to R^2 for a
    one-way design). Rough guidance: ~0.01 small, ~0.06 medium, ~0.14
    large. Biased upward for small samples -- see omega_squared for a
    less-biased version."""
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    all_values = [x for g in groups for x in g]
    grand_mean = _mean(all_values)
    ss_between = sum(len(g) * (_mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for x in all_values)
    if ss_total == 0.0:
        return 0.0
    return ss_between / ss_total


def omega_squared(*groups: Sequence[float]) -> float:
    """Effect size for a one-way ANOVA (Hays, 1963), less biased than
    eta_squared for small samples since it subtracts out the variance
    explained by chance alone. Can come out slightly negative when the
    true effect is ~0 -- that's expected, not a bug; clamp to 0 for
    reporting if a non-negative value is wanted."""
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    k = len(groups)
    all_values = [x for g in groups for x in g]
    n = len(all_values)
    grand_mean = _mean(all_values)
    ss_between = sum(len(g) * (_mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for x in all_values)
    ss_within = ss_total - ss_between
    df_between = k - 1
    ms_within = ss_within / (n - k) if n > k else 0.0
    denom = ss_total + ms_within
    if denom == 0.0:
        return 0.0
    return (ss_between - df_between * ms_within) / denom


def rank_biserial_correlation(u1_statistic: float, n1: int, n2: int) -> float:
    """Effect size for a Mann-Whitney U test (Wendt, 1972), in [-1, 1].
    Call with the ``statistic`` mann_whitney_u(sample1, sample2) returns
    (U for sample1, its documented convention) and the two sample sizes.
    Positive means sample1's values tend to exceed sample2's; negative
    means the reverse; 0 is no tendency either way. Rough guidance
    mirrors Cohen's d: ~0.1 small, ~0.3 medium, ~0.5 large."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")
    return (2.0 * u1_statistic) / (n1 * n2) - 1.0
