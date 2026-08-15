"""End-to-end check that rigor/mcp_server.py actually works against a real
MCP client over stdio -- not just that it imports cleanly. Regression test
for a real bug found in practice: the MCP SDK renamed FastMCP mid-flight,
so an import that type-checked fine still broke at runtime.

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
        self.assertEqual(len(names), 17)

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

    def test_cohens_d_zero_variance_edge_case_fails_at_the_transport(self):
        # Documented, known limitation: cohens_d correctly returns +-inf for
        # zero-variance samples, but MCP's structured content serializes
        # non-finite floats to JSON null, which then fails the tool's own
        # number-typed output schema. This test pins that behavior down so
        # a future SDK upgrade that silently "fixes" or changes it gets
        # noticed rather than passing quietly.
        with self.assertRaises(Exception):
            asyncio.run(_call("cohens_d", {"a": [0, 0, 0, 0], "b": [1, 1, 1, 1]}))


if __name__ == "__main__":
    unittest.main()
