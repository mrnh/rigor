"""MCP server exposing rigor's statistics as tools for AI agents.

Requires the official MCP Python SDK: bundled automatically by
``pip install rigor-mcp`` (or ``pip install mcp`` alongside a plain
checkout). That's the one dependency in this project that isn't the
standard library -- appropriate here, since being an MCP server *is*
the point of this file, unlike the rest of rigor (distributions/
inference/effect_size/power/corrections), which stays dependency-free.
It's a required dependency of the distribution rather than an optional
extra deliberately: `uvx rigor-mcp`, the way most MCP clients actually
invoke this, has no way to request an extra.

Serves over stdio (the transport local MCP clients like Claude Code
expect) via, in order of preference:

    rigor-mcp                    # console script, once pip-installed
    python3 -m rigor.mcp_server  # module form, from a checkout

For interactive poking with the MCP Inspector, run it as a script rather
than a module, which means the package root has to be put on the path
by hand (the inspector imports the file directly, so `rigor` wouldn't
otherwise be importable):

    PYTHONPATH=. mcp dev rigor/mcp_server.py

Each tool below is a thin, mechanical wrapper around an already-tested
function in rigor.inference / rigor.effect_size / rigor.power /
rigor.corrections -- deliberately kept with no logic of its own beyond
converting a dataclass result to a plain dict. Smoke-tested against a
real MCP client (stdio transport, tool discovery + representative calls
across all 30 tools; see tests/test_mcp_server.py).

Every tool carries the same ToolAnnotations (_PURE): every one of them
is a stateless, deterministic calculation over its arguments -- no I/O,
no external calls, no mutation, calling twice with the same input always
gives the same answer. read_only_hint/idempotent_hint=True and
destructive_hint/open_world_hint=False are simply true statements about
all 30, not a per-tool judgment call.

Every parameter also carries an explicit Field(description=...) rather
than relying on the docstring alone: the MCP SDK does not parse a
docstring's Args section into the generated JSON schema (checked
empirically -- a Google-style Args block produces no per-parameter
schema description), but Annotated[T, Field(description=...)] does. The
prose docstrings therefore focus on purpose, usage guidance (when to
reach for this tool over a sibling), and what comes back; parameter-by-
parameter meaning lives in the schema instead of being duplicated in
both places.

One deliberate exception to "no logic of its own": cohens_d returns a
dict rather than a bare float, because rigor.effect_size.cohens_d
correctly returns +-inf for zero-variance samples, but MCP's structured
content serializes non-finite floats to JSON null, which fails a bare
*number-typed* output schema. The wrapper catches that case and reports
it as an explicit null value plus a warning instead of letting the call
fail. That's specific to bare-scalar-returning tools: every dict-
returning tool (everything using _result_dict/_correction_dict/
_regression_dict) has been confirmed over real stdio to pass a non-
finite field (e.g. fisher_exact_test's odds ratio for a zero cell, or a
t-statistic's +-inf degenerate case) through untouched as JSON's
non-standard `Infinity`/`-Infinity`, since a generic dict return doesn't
get a strict per-field number schema the way a bare float return does.
Of the bare-float-returning tools, cohens_d is the only one that can
produce a non-finite value; cohens_h, cramers_v, eta_squared,
omega_squared, and rank_biserial_correlation are all bounded and always
finite for valid inputs.
"""
import math
from typing import List

try:
    from mcp.server import MCPServer
except ImportError as exc:
    # Shouldn't happen via a normal `pip install rigor-mcp` (mcp is a
    # required dependency) -- this is a fallback for a plain checkout
    # that never had mcp installed at all.
    raise ImportError(
        "rigor's MCP server needs the official MCP Python SDK, which "
        "isn't installed. Run: pip install mcp"
    ) from exc

from typing import Annotated

from mcp.types import ToolAnnotations
from pydantic import Field

from rigor import correlation, corrections, effect_size, inference, nonparametric, power, regression

mcp = MCPServer("rigor")

# Shared by every tool below -- see the module docstring for why this is
# a true statement about all 30 rather than a per-tool judgment call.
_PURE = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

_Alpha = Annotated[float, Field(description="significance level for the test (and any confidence interval); default 0.05")]
_TargetPower = Annotated[float, Field(description="desired probability of detecting the effect if it's real; 0.8 is the conventional target")]
_Proportion = Annotated[float, Field(description="a proportion in [0, 1]")]


def _result_dict(result: inference.TestResult, alpha: float = 0.05) -> dict:
    return {
        "name": result.name,
        "statistic": result.statistic,
        "df": result.df,
        "df2": result.df2,
        "p_value": result.p_value,
        "reject_null": result.reject_null(alpha),
        "alpha": alpha,
        "confidence_interval": result.confidence_interval,
        "confidence_level": result.confidence_level,
        "citation": result.citation,
        "warnings": result.warnings,
    }


def _correction_dict(result: corrections.CorrectionResult) -> dict:
    return {
        "method": result.method,
        "alpha": result.alpha,
        "p_values": result.p_values,
        "reject": result.reject,
        "adjusted_alpha": result.adjusted_alpha,
        "citation": result.citation,
    }


def _regression_dict(result: regression.RegressionResult) -> dict:
    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.r_squared,
        "n": result.n,
        "df": result.df,
        "slope_se": result.slope_se,
        "slope_t": result.slope_t,
        "slope_p_value": result.slope_p_value,
        "slope_confidence_interval": result.slope_confidence_interval,
        "confidence_level": result.confidence_level,
        "citation": result.citation,
        "warnings": result.warnings,
    }


@mcp.tool(annotations=_PURE)
def one_sample_t_test(
    data: Annotated[List[float], Field(description="the sample; one number per observation")],
    mu0: Annotated[float, Field(description="the hypothesized population mean to test the sample against")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether a sample's mean differs from a hypothesized value mu0.
    Returns the t-statistic, degrees of freedom, two-tailed p-value, a
    confidence interval for the mean, and any assumption warnings."""
    return _result_dict(inference.one_sample_t_test(data, mu0), alpha)


@mcp.tool(annotations=_PURE)
def two_sample_t_test(
    a: Annotated[List[float], Field(description="first independent sample")],
    b: Annotated[List[float], Field(description="second independent sample")],
    equal_var: Annotated[bool, Field(description="assume equal population variances (classic pooled-variance test) instead of Welch's test")] = False,
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether two independent samples have different means. Defaults
    to Welch's t-test (does not assume equal variances); pass
    equal_var=true for the classic pooled-variance test."""
    return _result_dict(inference.two_sample_t_test(a, b, equal_var=equal_var), alpha)


@mcp.tool(annotations=_PURE)
def paired_t_test(
    a: Annotated[List[float], Field(description="first measurement of each pair, e.g. 'before'")],
    b: Annotated[List[float], Field(description="second measurement of each pair, e.g. 'after' -- same length and pairing order as a")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether the mean difference between paired observations (e.g.
    before/after measurements on the same subjects, or matched pairs) is
    zero. a[i] and b[i] must be the two measurements of the same pair --
    use two_sample_t_test instead if the two samples are independent
    (different subjects in each group). Returns the t-statistic, degrees
    of freedom (n-1), two-tailed p-value, a confidence interval for the
    mean difference, a citation, and assumption warnings."""
    return _result_dict(inference.paired_t_test(a, b), alpha)


@mcp.tool(annotations=_PURE)
def one_proportion_z_test(
    successes: Annotated[int, Field(description="number of successes observed")],
    n: Annotated[int, Field(description="total number of trials/observations")],
    p0: Annotated[float, Field(description="the hypothesized true proportion to test against, in [0, 1]")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether an observed proportion (successes out of n) differs
    from a hypothesized proportion p0 -- e.g. "is this coin fair (p0=0.5)
    given 55 heads in 100 flips?" Uses the normal approximation, which
    degrades for small n or p0 near 0 or 1; a warning is included when
    that assumption looks shaky. Returns the z-statistic, two-tailed
    p-value, a confidence interval for the true proportion, a citation,
    and warnings."""
    return _result_dict(inference.one_proportion_z_test(successes, n, p0), alpha)


@mcp.tool(annotations=_PURE)
def two_proportion_z_test(
    successes1: Annotated[int, Field(description="successes observed in group 1")],
    n1: Annotated[int, Field(description="total observations in group 1")],
    successes2: Annotated[int, Field(description="successes observed in group 2")],
    n2: Annotated[int, Field(description="total observations in group 2")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether two independent proportions differ -- the standard test
    behind comparing conversion rates between two groups (e.g. an A/B
    test). Returns the z-statistic, two-tailed p-value, a confidence
    interval for the difference in proportions, a citation, and
    warnings."""
    return _result_dict(inference.two_proportion_z_test(successes1, n1, successes2, n2), alpha)


@mcp.tool(annotations=_PURE)
def chi_square_goodness_of_fit(
    observed: Annotated[List[float], Field(description="observed count per category")],
    expected: Annotated[List[float], Field(description="expected count per category, same length and category order as observed; does not need to sum to the same total")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether observed category counts match an expected
    distribution -- e.g. "are these six days-of-week signup counts
    evenly distributed, or skewed towards weekends?" Returns the
    chi-squared statistic, degrees of freedom (len-1), p-value, a
    citation, and a warning if any expected count is below 5 (the usual
    threshold below which this approximation gets unreliable)."""
    return _result_dict(inference.chi_square_goodness_of_fit(observed, expected), alpha)


@mcp.tool(annotations=_PURE)
def chi_square_independence(
    table: Annotated[List[List[float]], Field(description="contingency table as a list of rows, each a list of raw counts (not proportions), e.g. [[treated_success, treated_failure], [control_success, control_failure]] for a 2x2 table")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether the row and column variables of a contingency table
    are independent (e.g. "does group membership relate to outcome?").
    Returns the chi-squared statistic, degrees of freedom, p-value, a
    citation, and a warning if any expected cell count is below 5
    (consider cramers_v afterwards for effect size)."""
    return _result_dict(inference.chi_square_independence(table), alpha)


@mcp.tool(annotations=_PURE)
def one_way_anova(
    groups: Annotated[List[List[float]], Field(description="one list of observations per group; at least 3 groups, each with at least 2 observations")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether three or more independent groups have different
    means -- e.g. comparing average order value across three marketing
    channels. A significant result means at least one group differs from
    the others, not which one -- follow up with pairwise
    two_sample_t_test calls (correcting for multiple comparisons via
    bonferroni_correction or benjamini_hochberg_correction) to find
    which. Returns the F-statistic, between/within degrees of freedom,
    p-value, a citation, and a warning if within-group df is small."""
    return _result_dict(inference.one_way_anova(*groups), alpha)


@mcp.tool(annotations=_PURE)
def levene_test(
    groups: Annotated[List[List[float]], Field(description="one list of observations per group; at least 2 groups, each with at least 2 observations")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether two or more groups have equal population variances
    (homogeneity of variance) -- use this to decide equal_var for
    two_sample_t_test, or to sanity-check one_way_anova's
    equal-variance assumption. Uses the Brown-Forsythe variant
    (deviations from each group's median), more robust to non-normal
    data than the original mean-based Levene's test. Returns the same
    shape as one_way_anova (it's computed as one internally, on
    absolute deviations from each group's median)."""
    return _result_dict(inference.levene_test(*groups), alpha)


@mcp.tool(annotations=_PURE)
def fisher_exact_test(
    table: Annotated[List[List[int]], Field(description="2x2 contingency table as [[a, b], [c, d]], raw non-negative integer counts")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test whether the row and column variables of a 2x2 contingency
    table are independent -- exact (via the hypergeometric distribution
    over all tables with the same margins), unlike
    chi_square_independence's chi-squared approximation. Use this
    instead whenever chi_square_independence warns an expected cell
    count is below 5, or whenever the sample is small. 2x2 tables only.
    Returns the sample odds ratio as ``statistic`` (can be inf/0 for a
    zero cell), a two-tailed p-value, a citation, and warnings."""
    return _result_dict(inference.fisher_exact_test(table), alpha)


@mcp.tool(annotations=_PURE)
def cohens_d(
    a: Annotated[List[float], Field(description="first sample")],
    b: Annotated[List[float], Field(description="second sample")],
) -> dict:
    """Standardized mean difference between two samples (pooled SD).
    Use alongside two_sample_t_test, which tells you whether a
    difference is significant but not how large it is. Rough guidance:
    ~0.2 small, ~0.5 medium, ~0.8 large -- context-dependent. Returns
    {"value": float or null, "warnings": [...]}. value is null only
    when both samples have zero variance and unequal means, where the
    effect size is mathematically infinite -- see the warning for which
    direction, and report the raw mean difference instead in that case."""
    d = effect_size.cohens_d(a, b)
    if math.isfinite(d):
        return {"value": d, "warnings": []}
    direction = "higher" if d > 0 else "lower"
    return {
        "value": None,
        "warnings": [
            f"Both samples have zero variance, and sample a's mean is "
            f"{direction} than sample b's, so the standardized effect "
            "size is mathematically infinite (pooled SD is 0). Report "
            "the raw mean difference instead of a standardized d here."
        ],
    }


@mcp.tool(annotations=_PURE)
def cohens_h(p1: _Proportion, p2: _Proportion) -> float:
    """Effect size for a difference between two proportions (Cohen,
    1988), via the arcsine-square-root transform -- more appropriate
    than a raw percentage-point difference since it stabilizes variance
    across the full [0, 1] range. p1 and p2 are interchangeable (the
    sign of the result just indicates direction); use alongside
    two_proportion_z_test, which tells you whether a difference is
    significant but not how large it is. Returns a float (can be
    negative); rough guidance: ~0.2 small, ~0.5 medium, ~0.8 large."""
    return effect_size.cohens_h(p1, p2)


@mcp.tool(annotations=_PURE)
def cramers_v(
    chi2_statistic: Annotated[float, Field(description="the chi-squared statistic from chi_square_independence on the same table")],
    n: Annotated[int, Field(description="total number of observations in the table")],
    rows: Annotated[int, Field(description="number of rows in the table")],
    cols: Annotated[int, Field(description="number of columns in the table")],
) -> float:
    """Effect size for a chi-squared test of independence (Cramer, 1946),
    normalized to [0, 1] regardless of table shape so it's comparable
    across tables of different sizes, unlike the raw chi-squared
    statistic. Call after chi_square_independence, passing its
    statistic and the same table's n/rows/cols. Returns a float in
    [0, 1]; rough guidance for a 2x2 table: ~0.1 small, ~0.3 medium,
    ~0.5 large -- the threshold shifts for larger tables."""
    return effect_size.cramers_v(chi2_statistic, n, rows, cols)


@mcp.tool(annotations=_PURE)
def eta_squared(
    groups: Annotated[List[List[float]], Field(description="one list of observations per group; at least 2 groups")],
) -> float:
    """Effect size for a one-way ANOVA: proportion of total variance
    explained by group membership. Use alongside one_way_anova, which
    tells you whether groups differ but not how much of the variance
    that accounts for. Rough guidance: ~0.01 small, ~0.06 medium, ~0.14
    large. Biased upward for small samples -- prefer omega_squared when
    that matters. Returns a float in [0, 1]."""
    return effect_size.eta_squared(*groups)


@mcp.tool(annotations=_PURE)
def omega_squared(
    groups: Annotated[List[List[float]], Field(description="one list of observations per group; at least 2 groups")],
) -> float:
    """Effect size for a one-way ANOVA, less biased than eta_squared for
    small samples since it subtracts out the variance explained by
    chance alone. Use alongside one_way_anova. Can be slightly negative
    when the true effect is near zero -- that's expected, not an error."""
    return effect_size.omega_squared(*groups)


@mcp.tool(annotations=_PURE)
def rank_biserial_correlation(
    u1_statistic: Annotated[float, Field(description="the statistic returned by mann_whitney_u (U for the first sample passed to it)")],
    n1: Annotated[int, Field(description="size of the first sample passed to mann_whitney_u")],
    n2: Annotated[int, Field(description="size of the second sample passed to mann_whitney_u")],
) -> float:
    """Effect size for a Mann-Whitney U test. Call after mann_whitney_u,
    passing its statistic and the two sample sizes. Positive means
    sample 1's values tend to exceed sample 2's; negative means the
    reverse; 0 is no tendency either way. Returns a float in [-1, 1];
    rough guidance mirrors Cohen's d: ~0.1 small, ~0.3 medium, ~0.5
    large."""
    return effect_size.rank_biserial_correlation(u1_statistic, n1, n2)


@mcp.tool(annotations=_PURE)
def pearson_correlation(
    x: Annotated[List[float], Field(description="first variable, one value per observation")],
    y: Annotated[List[float], Field(description="second variable, same length and pairing order as x")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test for a *linear* association between two paired variables --
    e.g. "does hours studied predict test score?" statistic is r itself
    (in [-1, 1]), not a t-statistic. Returns r, df (n-2), a two-tailed
    p-value (H0: r=0), a confidence interval for r via the Fisher
    z-transform, a citation, and warnings. Use spearman_correlation
    instead if the relationship may be monotonic but not linear, or if
    outliers shouldn't dominate the result. Use simple_linear_regression
    instead for the actual slope (units of y per unit of x), not just
    the strength of association."""
    return _result_dict(correlation.pearson_correlation(x, y), alpha)


@mcp.tool(annotations=_PURE)
def spearman_correlation(
    x: Annotated[List[float], Field(description="first variable, one value per observation")],
    y: Annotated[List[float], Field(description="second variable, same length and pairing order as x")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Test for a *monotonic* association between two paired variables,
    via the Pearson correlation of their ranks -- doesn't assume
    linearity and is far less sensitive to outliers' exact magnitude
    than pearson_correlation. Same return shape as pearson_correlation
    (statistic is rho itself, in [-1, 1])."""
    return _result_dict(correlation.spearman_correlation(x, y), alpha)


@mcp.tool(annotations=_PURE)
def simple_linear_regression(
    x: Annotated[List[float], Field(description="the predictor variable, one value per observation")],
    y: Annotated[List[float], Field(description="the outcome variable, same length and pairing order as x")],
    alpha: _Alpha = 0.05,
) -> dict:
    """Fit y = intercept + slope * x by ordinary least squares -- single
    predictor only. Reports the slope (change in y per unit of x), the
    intercept, R^2 (proportion of y's variance explained by x), and a
    significance test + confidence interval for the slope (H0:
    slope=0). Use pearson_correlation instead if you only need the
    strength of a linear association, not its actual units/magnitude."""
    return _regression_dict(regression.simple_linear_regression(x, y))


@mcp.tool(annotations=_PURE)
def mann_whitney_u(
    a: Annotated[List[float], Field(description="first independent sample")],
    b: Annotated[List[float], Field(description="second independent sample")],
    alpha: _Alpha = 0.05,
) -> dict:
    """The non-parametric alternative to two_sample_t_test -- use when
    that test's own small-n warning makes a normal-theory result
    suspect, or the data is ordinal/skewed. Tests whether values from
    sample a are systematically larger or smaller than values from
    sample b, by ranking the combined data rather than assuming normal
    populations. statistic is U for sample a; pair with
    rank_biserial_correlation for a standardized effect size. Returns
    the same result shape as the parametric tests (statistic, p_value,
    citation, warnings)."""
    return _result_dict(nonparametric.mann_whitney_u(a, b), alpha)


@mcp.tool(annotations=_PURE)
def wilcoxon_signed_rank(
    a: Annotated[List[float], Field(description="first measurement of each pair, e.g. 'before'")],
    b: Annotated[List[float], Field(description="second measurement of each pair, e.g. 'after' -- same length and pairing order as a")],
    alpha: _Alpha = 0.05,
) -> dict:
    """The non-parametric alternative to paired_t_test -- use when that
    test's own small-n warning makes a normal-theory result suspect.
    Tests whether the median of the paired differences is zero, by
    ranking the absolute differences rather than assuming they're
    normally distributed. Pairs with a zero difference are dropped (and
    counted in a warning), the standard procedure. statistic is T =
    min(W+, W-)."""
    return _result_dict(nonparametric.wilcoxon_signed_rank(a, b), alpha)


@mcp.tool(annotations=_PURE)
def kruskal_wallis(
    groups: Annotated[List[List[float]], Field(description="one list of observations per group; at least 2 groups")],
    alpha: _Alpha = 0.05,
) -> dict:
    """The non-parametric alternative to one_way_anova -- use when that
    test's own small-df warning makes a normal-theory result suspect.
    Tests whether all groups are drawn from the same distribution, by
    ranking the combined data rather than assuming normal populations.
    A significant result means at least one group differs, not which
    one -- same caveat as one_way_anova."""
    return _result_dict(nonparametric.kruskal_wallis(*groups), alpha)


@mcp.tool(annotations=_PURE)
def sample_size_for_two_sample_t_test(
    effect_size_d: Annotated[float, Field(description="the Cohen's d you want to be able to detect")],
    alpha: _Alpha = 0.05,
    target_power: _TargetPower = 0.8,
) -> dict:
    """How many observations per group are needed to detect a given
    Cohen's d with a two-sample t-test at the target power. Returns a
    continuous value and a rounded-up integer to actually use."""
    n = power.sample_size_two_sample_t_test(effect_size_d, alpha, target_power)
    return {"n_per_group_exact": n, "n_per_group_rounded_up": math.ceil(n)}


@mcp.tool(annotations=_PURE)
def power_for_two_sample_t_test(
    n_per_group: Annotated[float, Field(description="planned (or actual) observations per group")],
    effect_size_d: Annotated[float, Field(description="the Cohen's d you want to be able to detect")],
    alpha: _Alpha = 0.05,
) -> float:
    """Statistical power to detect a given Cohen's d with n_per_group
    observations per group, using a two-sample t-test. Power is the
    probability of correctly detecting a real effect of this size at
    the given alpha; a design with low power means a non-significant
    result would be inconclusive rather than good evidence the effect
    doesn't exist. Use sample_size_for_two_sample_t_test instead to
    solve for n given a target power. Returns a float in [alpha, 1]."""
    return power.power_two_sample_t_test(n_per_group, effect_size_d, alpha)


@mcp.tool(annotations=_PURE)
def sample_size_for_one_sample_t_test(
    effect_size_d: Annotated[float, Field(description="the Cohen's d you want to be able to detect")],
    alpha: _Alpha = 0.05,
    target_power: _TargetPower = 0.8,
) -> dict:
    """How many observations are needed to detect a given Cohen's d with
    a one-sample (or paired) t-test at the target power. Use for
    paired_t_test too -- a paired t-test is a one-sample t-test on the
    differences, so the same power formula applies. Returns a
    continuous value and a rounded-up integer to actually use."""
    n = power.sample_size_one_sample_t_test(effect_size_d, alpha, target_power)
    return {"n_exact": n, "n_rounded_up": math.ceil(n)}


@mcp.tool(annotations=_PURE)
def power_for_one_sample_t_test(
    n: Annotated[float, Field(description="planned (or actual) number of observations")],
    effect_size_d: Annotated[float, Field(description="the Cohen's d you want to be able to detect")],
    alpha: _Alpha = 0.05,
) -> float:
    """Statistical power to detect a given Cohen's d with n observations,
    using a one-sample (or paired) t-test. Use for paired_t_test too --
    it's a one-sample t-test on the differences, so the same power
    formula applies. Use sample_size_for_one_sample_t_test instead to
    solve for n given a target power. Returns a float in [alpha, 1]."""
    return power.power_one_sample_t_test(n, effect_size_d, alpha)


@mcp.tool(annotations=_PURE)
def sample_size_for_two_proportion_test(
    p1: _Proportion,
    p2: _Proportion,
    alpha: _Alpha = 0.05,
    target_power: _TargetPower = 0.8,
) -> dict:
    """How many observations per group are needed to detect a difference
    between two proportions (e.g. conversion rates) at the target power.
    p1 and p2 are interchangeable -- only their difference matters."""
    n = power.sample_size_two_proportion_z_test(p1, p2, alpha, target_power)
    return {"n_per_group_exact": n, "n_per_group_rounded_up": math.ceil(n)}


@mcp.tool(annotations=_PURE)
def power_for_two_proportion_test(
    n_per_group: Annotated[float, Field(description="planned (or actual) observations per group")],
    p1: _Proportion,
    p2: _Proportion,
    alpha: _Alpha = 0.05,
) -> float:
    """Statistical power to detect a difference between two proportions
    (e.g. two conversion rates) with n_per_group observations in each
    group, using a two-proportion z-test. p1 and p2 are interchangeable
    (only their difference matters) -- e.g. current vs. new conversion
    rate. Use sample_size_for_two_proportion_test instead to solve for n
    given a target power. Returns a float in [alpha, 1]."""
    return power.power_two_proportion_z_test(n_per_group, p1, p2, alpha)


@mcp.tool(annotations=_PURE)
def bonferroni_correction(
    p_values: Annotated[List[float], Field(description="the batch of p-values to adjust")],
    alpha: Annotated[float, Field(description="family-wise significance level to control; default 0.05")] = 0.05,
) -> dict:
    """Adjust a batch of p-values for multiple comparisons, controlling
    the family-wise error rate. Conservative; use when any false positive
    among the batch is costly."""
    return _correction_dict(corrections.bonferroni(p_values, alpha))


@mcp.tool(annotations=_PURE)
def benjamini_hochberg_correction(
    p_values: Annotated[List[float], Field(description="the batch of p-values to adjust")],
    alpha: Annotated[float, Field(description="false discovery rate to control; default 0.05")] = 0.05,
) -> dict:
    """Adjust a batch of p-values for multiple comparisons, controlling
    the false discovery rate. Less conservative than Bonferroni; the
    standard choice when testing many hypotheses at once."""
    return _correction_dict(corrections.benjamini_hochberg(p_values, alpha))


def main() -> None:
    """Entry point for the `rigor-mcp` console script (and `python3 -m
    rigor.mcp_server`) -- serves over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
