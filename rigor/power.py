"""Statistical power and sample-size calculation.

The question this answers is usually "how many observations do I need,"
which is the mirror image of "given this many observations, what's my
chance of detecting a real effect." Both directions are implemented from
one formula per test family, kept as exact numerical inverses of each
other by solving `sample_size` via bisection over `power` rather than
deriving two separate closed forms that could silently drift apart.

The formulas here use the normal approximation to the sampling
distribution of the test statistic under the alternative hypothesis
(the standard approach in Cohen (1988) and most textbook treatments).
This is accurate for moderate-to-large samples; for very small samples
(single digits per group) the exact answer, computed from the
noncentral t distribution, differs slightly -- usually calling for a
handful more observations than this approximation suggests. That's
flagged in each function's docstring rather than silently glossed over.
"""
import math
from rigor.distributions import normal_cdf, normal_ppf


def _solve_n(power_fn, target_power: float, lo: float = 2.0, hi: float = 1e7) -> float:
    """Smallest n (via bisection) such that power_fn(n) >= target_power.

    power_fn must be monotonically non-decreasing in n, which holds for
    every power function in this module.
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if power_fn(hi) < target_power:
        raise ValueError(
            "required sample size exceeds the search bound (1e7) -- "
            "the effect size is likely too small to be practically detectable"
        )
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if power_fn(mid) < target_power:
            lo = mid
        else:
            hi = mid
    return hi


def power_two_sample_t_test(n: float, effect_size: float, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """Power to detect a given Cohen's d with n observations per group
    (equal group sizes), using a two-sample t-test.

    Normal approximation to the noncentral t distribution -- accurate for
    n gtr-or-eq ~20 per group; for smaller n it modestly overstates power
    (i.e. the exact required n is a bit larger than sample_size_* reports).
    """
    if n <= 1:
        raise ValueError("n must be greater than 1")
    ncp = effect_size * math.sqrt(n / 2.0)
    zcrit = normal_ppf(1 - alpha / 2) if two_tailed else normal_ppf(1 - alpha)
    if two_tailed:
        return normal_cdf(ncp - zcrit) + normal_cdf(-ncp - zcrit)
    return normal_cdf(ncp - zcrit)


def sample_size_two_sample_t_test(
    effect_size: float, alpha: float = 0.05, power: float = 0.8, two_tailed: bool = True
) -> float:
    """Observations needed *per group* to detect the given Cohen's d at
    the requested power. Round up (math.ceil) before using -- this
    returns the continuous solution, and sample sizes aren't fractional.
    """
    if effect_size == 0:
        raise ValueError("effect_size must be nonzero -- a zero effect can never reach a target power")
    effect_size = abs(effect_size)
    return _solve_n(lambda n: power_two_sample_t_test(n, effect_size, alpha, two_tailed), power)


def power_one_sample_t_test(n: float, effect_size: float, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """Power for a one-sample (or paired) t-test, given Cohen's d and n observations."""
    if n <= 1:
        raise ValueError("n must be greater than 1")
    ncp = effect_size * math.sqrt(n)
    zcrit = normal_ppf(1 - alpha / 2) if two_tailed else normal_ppf(1 - alpha)
    if two_tailed:
        return normal_cdf(ncp - zcrit) + normal_cdf(-ncp - zcrit)
    return normal_cdf(ncp - zcrit)


def sample_size_one_sample_t_test(
    effect_size: float, alpha: float = 0.05, power: float = 0.8, two_tailed: bool = True
) -> float:
    if effect_size == 0:
        raise ValueError("effect_size must be nonzero -- a zero effect can never reach a target power")
    effect_size = abs(effect_size)
    return _solve_n(lambda n: power_one_sample_t_test(n, effect_size, alpha, two_tailed), power)


def power_two_proportion_z_test(
    n: float, p1: float, p2: float, alpha: float = 0.05, two_tailed: bool = True
) -> float:
    """Power to detect a difference between two proportions with n
    observations per group (equal group sizes).

    Uses the pooled proportion for the null-hypothesis rejection boundary
    and the unpooled proportions for the true sampling variance under the
    alternative -- the standard textbook derivation (e.g. Fleiss, Levin &
    Paik). No continuity correction is applied.
    """
    if n <= 1:
        raise ValueError("n must be greater than 1")
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1 and p2 must be in (0, 1)")
    pbar = (p1 + p2) / 2.0
    se_null = math.sqrt(2 * pbar * (1 - pbar) / n)
    se_alt = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    delta = abs(p1 - p2)
    zcrit = normal_ppf(1 - alpha / 2) if two_tailed else normal_ppf(1 - alpha)
    ncp = delta / se_alt
    k = zcrit * se_null / se_alt
    if two_tailed:
        return normal_cdf(ncp - k) + normal_cdf(-ncp - k)
    return normal_cdf(ncp - k)


def sample_size_two_proportion_z_test(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, two_tailed: bool = True
) -> float:
    """Observations needed *per group* to detect a difference between two
    proportions at the requested power. Round up before using."""
    if p1 == p2:
        raise ValueError("p1 and p2 must differ -- identical proportions can never reach a target power")
    return _solve_n(lambda n: power_two_proportion_z_test(n, p1, p2, alpha, two_tailed), power)
