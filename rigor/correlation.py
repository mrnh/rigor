"""Correlation between two paired variables.

Both measures below return the correlation coefficient as a hypothesis
test (H0: no association) plus a confidence interval, since "is this
significant" and "how strong is it" are two different questions an
agent usually wants together -- the same instinct behind every
TestResult in ``inference.py``.

Pearson measures *linear* association on the raw values. Spearman
measures *monotonic* association by correlating ranks instead, which
doesn't assume linearity and is far less sensitive to outliers' exact
magnitude -- the same trade non-parametric tests make relative to their
parametric counterparts (see ``rigor/nonparametric.py``).
"""
import math
from typing import List, Sequence, Tuple

from rigor import distributions as dist
from rigor.inference import TestResult


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _rank(xs: Sequence[float]) -> List[float]:
    """Average ('fractional') ranks, 1-indexed, ties sharing the mean rank."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_correlation(x: Sequence[float], y: Sequence[float], confidence_level: float = 0.95) -> TestResult:
    """H0: the population Pearson correlation coefficient between x and y
    is zero (no linear association). ``statistic`` is r itself (not the
    underlying t-statistic); df is n-2, the degrees of freedom behind
    the significance test. The confidence interval is built on the
    Fisher z-transform (Fisher, 1921), the standard way to get a CI for
    r since r's own sampling distribution isn't normal even when x and y
    are."""
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must be the same length")
    if n < 3:
        raise ValueError("need at least 3 pairs (df = n-2 must be positive)")
    mx, my = _mean(x), _mean(y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0.0 or syy == 0.0:
        raise ValueError("x or y is constant -- correlation is undefined")
    r = sxy / math.sqrt(sxx * syy)
    r = max(-1.0, min(1.0, r))  # guard float rounding pushing |r| just past 1
    df = n - 2

    if abs(r) == 1.0:
        p = 0.0
        ci: Tuple[float, float] = (r, r)
        warnings = ["|r|=1 (perfect correlation) -- the Fisher z-transform CI is degenerate; reported as (r, r)."]
    else:
        t = r * math.sqrt(df / (1 - r * r))
        p = 2 * (1 - dist.t_cdf(abs(t), df))
        warnings = []
        if n < 4:
            ci = (r, r)
            warnings.append("n<4 -- confidence interval requires n>=4 (se=1/sqrt(n-3)); reported as (r, r).")
        else:
            z = math.atanh(r)
            se_z = 1.0 / math.sqrt(n - 3)
            zcrit = dist.normal_ppf(1 - (1 - confidence_level) / 2)
            ci = (math.tanh(z - zcrit * se_z), math.tanh(z + zcrit * se_z))
    if n < 30:
        warnings.append(f"n={n} is small; the significance test and CI lean on approximately bivariate-normal data.")

    return TestResult(
        name="Pearson correlation",
        statistic=r,
        df=df,
        p_value=p,
        confidence_interval=ci,
        confidence_level=confidence_level,
        citation="Pearson product-moment correlation; significance via Student's t (df=n-2); CI via Fisher's z-transform (Fisher, 1921).",
        warnings=warnings,
    )


def spearman_correlation(x: Sequence[float], y: Sequence[float], confidence_level: float = 0.95) -> TestResult:
    """H0: no monotonic association between x and y (Spearman's rho = 0).
    Computed as the Pearson correlation of the ranks -- the standard
    definition -- so it shares Pearson's significance-test and CI
    machinery, just applied to ranked data. Use over pearson_correlation
    when the relationship may be monotonic-but-not-linear, or when
    outliers in the raw values shouldn't dominate the result."""
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must be the same length")
    result = pearson_correlation(_rank(x), _rank(y), confidence_level)
    result.name = "Spearman rank correlation"
    result.citation = (
        "Spearman's rank correlation (Spearman, 1904); significance and CI via "
        "the Pearson/Fisher-z machinery applied to ranks (Zar, 1972's t-approximation)."
    )
    if n < 10:
        result.warnings.append(f"n={n} is small for the t-approximation behind this p-value; treat it as approximate.")
    return result
