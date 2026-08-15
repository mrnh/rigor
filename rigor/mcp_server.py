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
across all 17 tools; see tests/test_mcp_server.py).

One deliberate exception to "no logic of its own": cohens_d returns a
dict rather than a bare float, because rigor.effect_size.cohens_d
correctly returns +-inf for zero-variance samples, but MCP's structured
content serializes non-finite floats to JSON null, which fails a bare
number-typed output schema. The wrapper catches that case and reports
it as an explicit null value plus a warning instead of letting the call
fail. Every non-finite-capable tool needs this same treatment; cohens_d
is the only one exposed here that can produce one -- cohens_h and
cramers_v are bounded and always finite for valid inputs.
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

from rigor import corrections, effect_size, inference, power

mcp = MCPServer("rigor")


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


@mcp.tool()
def one_sample_t_test(data: List[float], mu0: float, alpha: float = 0.05) -> dict:
    """Test whether a sample's mean differs from a hypothesized value mu0.
    Returns the t-statistic, degrees of freedom, two-tailed p-value, a
    confidence interval for the mean, and any assumption warnings."""
    return _result_dict(inference.one_sample_t_test(data, mu0), alpha)


@mcp.tool()
def two_sample_t_test(a: List[float], b: List[float], equal_var: bool = False, alpha: float = 0.05) -> dict:
    """Test whether two independent samples have different means. Defaults
    to Welch's t-test (does not assume equal variances); pass
    equal_var=true for the classic pooled-variance test."""
    return _result_dict(inference.two_sample_t_test(a, b, equal_var=equal_var), alpha)


@mcp.tool()
def paired_t_test(a: List[float], b: List[float], alpha: float = 0.05) -> dict:
    """Test whether the mean difference between paired observations (e.g.
    before/after on the same subjects) is zero."""
    return _result_dict(inference.paired_t_test(a, b), alpha)


@mcp.tool()
def one_proportion_z_test(successes: int, n: int, p0: float, alpha: float = 0.05) -> dict:
    """Test whether an observed proportion (successes out of n) differs
    from a hypothesized proportion p0."""
    return _result_dict(inference.one_proportion_z_test(successes, n, p0), alpha)


@mcp.tool()
def two_proportion_z_test(successes1: int, n1: int, successes2: int, n2: int, alpha: float = 0.05) -> dict:
    """Test whether two independent proportions differ -- the standard test
    behind comparing conversion rates between two groups (e.g. an A/B test)."""
    return _result_dict(inference.two_proportion_z_test(successes1, n1, successes2, n2), alpha)


@mcp.tool()
def chi_square_goodness_of_fit(observed: List[float], expected: List[float], alpha: float = 0.05) -> dict:
    """Test whether observed category counts match an expected distribution."""
    return _result_dict(inference.chi_square_goodness_of_fit(observed, expected), alpha)


@mcp.tool()
def chi_square_independence(table: List[List[float]], alpha: float = 0.05) -> dict:
    """Test whether the row and column variables of a contingency table
    are independent (e.g. "does group membership relate to outcome")."""
    return _result_dict(inference.chi_square_independence(table), alpha)


@mcp.tool()
def one_way_anova(groups: List[List[float]], alpha: float = 0.05) -> dict:
    """Test whether three or more independent groups have different means."""
    return _result_dict(inference.one_way_anova(*groups), alpha)


@mcp.tool()
def cohens_d(a: List[float], b: List[float]) -> dict:
    """Standardized mean difference between two samples (pooled SD).
    Rough guidance: ~0.2 small, ~0.5 medium, ~0.8 large -- context-dependent.
    Returns {"value": float or null, "warnings": [...]}. value is null
    only when both samples have zero variance and unequal means, where
    the effect size is mathematically infinite -- see the warning for
    which direction and use the raw mean difference instead."""
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


@mcp.tool()
def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions (arcsine transform)."""
    return effect_size.cohens_h(p1, p2)


@mcp.tool()
def cramers_v(chi2_statistic: float, n: int, rows: int, cols: int) -> float:
    """Effect size for a chi-squared test of independence, normalized to [0, 1]."""
    return effect_size.cramers_v(chi2_statistic, n, rows, cols)


@mcp.tool()
def sample_size_for_two_sample_t_test(
    effect_size_d: float, alpha: float = 0.05, target_power: float = 0.8
) -> dict:
    """How many observations per group are needed to detect a given
    Cohen's d with a two-sample t-test at the target power. Returns a
    continuous value and a rounded-up integer to actually use."""
    n = power.sample_size_two_sample_t_test(effect_size_d, alpha, target_power)
    return {"n_per_group_exact": n, "n_per_group_rounded_up": math.ceil(n)}


@mcp.tool()
def power_for_two_sample_t_test(
    n_per_group: float, effect_size_d: float, alpha: float = 0.05
) -> float:
    """Statistical power to detect a given Cohen's d with n observations
    per group, using a two-sample t-test."""
    return power.power_two_sample_t_test(n_per_group, effect_size_d, alpha)


@mcp.tool()
def sample_size_for_two_proportion_test(
    p1: float, p2: float, alpha: float = 0.05, target_power: float = 0.8
) -> dict:
    """How many observations per group are needed to detect a difference
    between two proportions (e.g. conversion rates) at the target power."""
    n = power.sample_size_two_proportion_z_test(p1, p2, alpha, target_power)
    return {"n_per_group_exact": n, "n_per_group_rounded_up": math.ceil(n)}


@mcp.tool()
def power_for_two_proportion_test(n_per_group: float, p1: float, p2: float, alpha: float = 0.05) -> float:
    """Statistical power to detect a difference between two proportions
    with n observations per group."""
    return power.power_two_proportion_z_test(n_per_group, p1, p2, alpha)


@mcp.tool()
def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> dict:
    """Adjust a batch of p-values for multiple comparisons, controlling
    the family-wise error rate. Conservative; use when any false positive
    among the batch is costly."""
    return _correction_dict(corrections.bonferroni(p_values, alpha))


@mcp.tool()
def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> dict:
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
