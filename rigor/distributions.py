"""Probability distributions needed for classical hypothesis testing.

Standard-library only. The normal distribution is exact (Python's
``statistics.NormalDist`` implements it via ``math.erf``). The t,
chi-squared, and F distributions have no stdlib support, so their CDFs
are built here from two numerical primitives that most statistical
software is built on: the regularized incomplete gamma function
(``gammainc_reg``) and the regularized incomplete beta function
(``betainc_reg``). Quantile functions (ppf) are then obtained by
bisection on the CDF, which is monotonic — simple and robust rather than
clever, which is what you want in code whose whole job is to be trusted.

These aren't just implemented and hoped to be right: ``tests/
test_distributions.py`` checks them against closed-form identities that
follow from the definitions of these distributions (df=1 t-distribution
is exactly Cauchy; df=2 chi-squared is exactly a scaled exponential; a
squared t-variate is exactly F(1, df)), rather than trusting that a
remembered algorithm was transcribed correctly.
"""
import math
from statistics import NormalDist

_NORMAL = NormalDist(0, 1)


def normal_cdf(x: float) -> float:
    return _NORMAL.cdf(x)


def normal_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return _NORMAL.inv_cdf(p)


# ---------------------------------------------------------------------------
# Regularized incomplete gamma function P(a, x) = γ(a, x) / Γ(a)
#
# Series expansion for x < a+1, continued fraction for x >= a+1 — the
# standard split (Numerical Recipes §6.2) that keeps both branches
# converging quickly across their respective ranges.
# ---------------------------------------------------------------------------

def _gammainc_series(a: float, x: float) -> float:
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(500):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gammainc_continued_fraction(a: float, x: float) -> float:
    """Returns Q(a, x) = 1 - P(a, x) via a modified Lentz continued fraction."""
    tiny = 1e-300
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def gammainc_reg(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x), for a > 0, x >= 0."""
    if x < 0 or a <= 0:
        raise ValueError("gammainc_reg requires a > 0 and x >= 0")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gammainc_series(a, x)
    return 1.0 - _gammainc_continued_fraction(a, x)


# ---------------------------------------------------------------------------
# Regularized incomplete beta function I_x(a, b)
#
# Continued fraction (Numerical Recipes §6.4), with the standard symmetry
# swap I_x(a,b) = 1 - I_{1-x}(b,a) used when x is past the fraction's
# fast-convergence region.
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def betainc_reg(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b), for a, b > 0 and x in [0, 1]."""
    if not 0 < a and 0 < b:
        raise ValueError("betainc_reg requires a > 0 and b > 0")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _bisect_ppf(cdf, p: float, lo: float, hi: float, tol: float = 1e-10) -> float:
    """Invert a monotonically increasing CDF by bisection."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    # Expand the bracket outward until it actually contains the root.
    while cdf(lo) > p:
        lo *= 2.0
    while cdf(hi) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Student's t distribution
# ---------------------------------------------------------------------------

def t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    ib = betainc_reg(x, df / 2.0, 0.5)
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def t_ppf(p: float, df: float) -> float:
    if p == 0.5:
        return 0.0
    return _bisect_ppf(lambda t: t_cdf(t, df), p, -1.0, 1.0)


def t_critical(alpha: float, df: float, two_tailed: bool = True) -> float:
    """The critical value t* such that P(|T| > t*) = alpha (two-tailed)
    or P(T > t*) = alpha (one-tailed)."""
    target = 1.0 - alpha / 2.0 if two_tailed else 1.0 - alpha
    return t_ppf(target, df)


# ---------------------------------------------------------------------------
# Chi-squared distribution
# ---------------------------------------------------------------------------

def chi2_cdf(x: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0:
        return 0.0
    return gammainc_reg(df / 2.0, x / 2.0)


def chi2_ppf(p: float, df: float) -> float:
    return _bisect_ppf(lambda x: chi2_cdf(x, df), p, 1e-9, max(df, 1.0))


def chi2_critical(alpha: float, df: float) -> float:
    return chi2_ppf(1.0 - alpha, df)


# ---------------------------------------------------------------------------
# F distribution
# ---------------------------------------------------------------------------

def f_cdf(x: float, d1: float, d2: float) -> float:
    if d1 <= 0 or d2 <= 0:
        raise ValueError("d1 and d2 must be positive")
    if x <= 0:
        return 0.0
    y = d1 * x / (d1 * x + d2)
    return betainc_reg(y, d1 / 2.0, d2 / 2.0)


def f_ppf(p: float, d1: float, d2: float) -> float:
    return _bisect_ppf(lambda x: f_cdf(x, d1, d2), p, 1e-9, max(d1, d2, 1.0))


def f_critical(alpha: float, d1: float, d2: float) -> float:
    return f_ppf(1.0 - alpha, d1, d2)
