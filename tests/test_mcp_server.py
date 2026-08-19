"""End-to-end check that rigor/mcp_server.py actually works against a real
MCP client over stdio -- not just that it imports cleanly. Regression
tests for two real bugs found in practice: the MCP SDK renamed FastMCP
mid-flight (an import that type-checked fine still broke at runtime),
and cohens_d's correct +-inf return crashing the transport instead of
being reported.

Skipped if the optional `mcp` package isn't installed (it's the one
dependency in this project that isn't the standard library -- see
rigor/mcp_server.py's docstring for why).
"""
import asyncio
import json
import math
import os
import sys
import unittest

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unwrap(result):
    text = result.content[0].text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


async def _call(tool_name, arguments):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "rigor.mcp_server"], cwd=REPO_ROOT
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = _unwrap(await session.call_tool(tool_name, arguments))
            return result, {t.name for t in tools.tools}


@unittest.skipUnless(HAVE_MCP, "mcp package not installed")
class TestMCPServer(unittest.TestCase):
    def test_all_tools_are_discoverable(self):
        _, names = asyncio.run(_call(
            "one_sample_t_test", {"data": [1, 2, 3, 4, 5], "mu0": 0}
        ))
        self.assertEqual(len(names), 32)

    def test_sample_size_matches_cohen_1988_reference_case(self):
        # d=0.5, alpha=.05, power=.80 -> textbook answer n~=64 per group.
        result, _ = asyncio.run(_call(
            "sample_size_for_two_sample_t_test",
            {"effect_size_d": 0.5, "alpha": 0.05, "target_power": 0.8},
        ))
        self.assertTrue(60 <= result["n_per_group_rounded_up"] <= 68, result)

    def test_one_way_anova_round_trips_correctly(self):
        result, _ = asyncio.run(_call(
            "one_way_anova", {"groups": [[1, 2, 3], [2, 3, 4], [5, 6, 7]]}
        ))
        self.assertAlmostEqual(result["statistic"], 13.0)
        self.assertEqual(result["df"], 2)
        self.assertEqual(result["df2"], 6)
        self.assertTrue(result["reject_null"])

    def test_cohens_d_normal_case_returns_finite_value_and_no_warnings(self):
        result, _ = asyncio.run(_call(
            "cohens_d", {"a": [10, 12, 11, 13, 9], "b": [15, 14, 16, 13, 17]}
        ))
        self.assertAlmostEqual(result["value"], -2.5298221281347035)
        self.assertEqual(result["warnings"], [])

    def test_fisher_exact_matches_lady_tasting_tea_reference_case(self):
        result, _ = asyncio.run(_call("fisher_exact_test", {"table": [[3, 1], [1, 3]]}))
        self.assertAlmostEqual(result["p_value"], 0.4857142857142857)

    def test_pearson_correlation_round_trips_correctly(self):
        result, _ = asyncio.run(_call(
            "pearson_correlation", {"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]}
        ))
        self.assertAlmostEqual(result["statistic"], 0.7745966692414834)
        self.assertEqual(result["df"], 3)

    def test_simple_linear_regression_round_trips_correctly(self):
        result, _ = asyncio.run(_call(
            "simple_linear_regression", {"x": [1, 2, 3, 4, 5], "y": [3, 5, 7, 9, 11]}
        ))
        self.assertAlmostEqual(result["slope"], 2.0)
        self.assertAlmostEqual(result["intercept"], 1.0)

    def test_mann_whitney_u_complete_separation(self):
        result, _ = asyncio.run(_call("mann_whitney_u", {"a": [1, 2, 3], "b": [4, 5, 6]}))
        self.assertEqual(result["statistic"], 0.0)

    def test_kruskal_wallis_matches_hand_computed_example(self):
        result, _ = asyncio.run(_call("kruskal_wallis", {"groups": [[1, 2], [3, 4], [5, 6]]}))
        self.assertAlmostEqual(result["statistic"], 4.571428571428571, places=6)

    def test_recommend_test_round_trips_correctly(self):
        result, _ = asyncio.run(_call(
            "recommend_test", {"outcome_type": "continuous", "n_groups": 2, "small_or_skewed": True}
        ))
        self.assertEqual(result["recommended_tool"], "mann_whitney_u")

    def test_pairwise_group_comparisons_covers_every_pair(self):
        result, _ = asyncio.run(_call(
            "pairwise_group_comparisons",
            {"groups": [[1, 2, 3], [2, 3, 4], [5, 6, 7]], "labels": ["A", "B", "C"]},
        ))
        self.assertEqual(len(result["comparisons"]), 3)
        pairs = {(c["group_i"], c["group_j"]) for c in result["comparisons"]}
        self.assertEqual(pairs, {(0, 1), (0, 2), (1, 2)})

    def test_cohens_d_zero_variance_edge_case_is_handled_not_crashed(self):
        # Regression test for a real bug found in practice: cohens_d
        # correctly returns +-inf for zero-variance samples, but MCP's
        # structured content serializes non-finite floats to JSON null,
        # which used to fail the tool's own number-typed output schema
        # and blow up the call. The wrapper now reports this case
        # explicitly instead of crashing.
        result, _ = asyncio.run(_call(
            "cohens_d", {"a": [0, 0, 0, 0], "b": [1, 1, 1, 1]}
        ))
        self.assertIsNone(result["value"])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("infinite", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()
