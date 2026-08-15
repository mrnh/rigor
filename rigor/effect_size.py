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
