"""Simple (single-predictor) ordinary least squares linear regression.

Fits y = intercept + slope*x by minimizing squared residuals -- closed
form, no iteration needed for one predictor. Reports the slope's
standard error, a t-test against slope=0, a confidence interval for the
slope, and R^2, the way any regression output normally does, since
"does x predict y" and "how much of y's variance does x explain" are
different questions.

Doesn't share inference.TestResult, unlike most of this package: a
regression fit has several outputs (slope, intercept, R^2) rather than
one statistic, so it gets its own dataclass -- the same call
corrections.py made for CorrectionResult.
"""
import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from rigor import distributions as dist


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float
    n: int
    df: int
    slope_se: float
    slope_t: float
    slope_p_value: float
    slope_confidence_interval: Tuple[float, float]
    confidence_level: float
    citation: str = ""
    warnings: List[str] = field(default_factory=list)


def simple_linear_regression(x: Sequence[float], y: Sequence[float], confidence_level: float = 0.95) -> RegressionResult:
    """Fit y = intercept + slope * x by ordinary least squares.

    The reported slope_p_value tests H0: slope == 0 (x has no linear
    effect on y). Requires at least 3 points (df = n-2 must be positive)
    and x not constant.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must be the same length")
    if n < 3:
        raise ValueError("need at least 3 points (df = n-2 must be positive)")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if sxx == 0.0:
        raise ValueError("x is constant -- slope is undefined")

    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    df = n - 2

    warnings = []
    if ss_res == 0.0:
        # every point lies exactly on the line -- se is 0, t is +-inf/undefined
        slope_se = 0.0
        t = 0.0 if slope == 0.0 else math.copysign(math.inf, slope)
        p = 1.0 if slope == 0.0 else 0.0
        ci = (slope, slope)
    else:
        mse = ss_res / df
        slope_se = math.sqrt(mse / sxx)
        t = slope / slope_se
        p = 2 * (1 - dist.t_cdf(abs(t), df))
        tcrit = dist.t_critical(1 - confidence_level, df)
        ci = (slope - tcrit * slope_se, slope + tcrit * slope_se)

    if n < 30:
        warnings.append(f"n={n} is small; the slope's significance test and CI lean on approximately normal, homoscedastic residuals.")
    if ss_tot == 0.0:
        warnings.append("y is constant -- R^2 is not meaningful here (reported as 1.0 by convention).")

    return RegressionResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        n=n,
        df=df,
        slope_se=slope_se,
        slope_t=t,
        slope_p_value=p,
        slope_confidence_interval=ci,
        confidence_level=confidence_level,
        citation="Ordinary least squares simple linear regression; slope significance via Student's t (df=n-2).",
        warnings=warnings,
    )
