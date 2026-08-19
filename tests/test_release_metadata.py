"""Nothing enforces that server.json's version tracks pyproject.toml's --
the MCP Registry and PyPI publishing are two separate manual/CI paths
(see .github/workflows/publish.yml) with no shared source of truth.
server.json actually went stale this way once already (still said
0.1.1 after two feature releases). This doesn't fix that structurally,
but it turns "stale registry metadata" from a silent drift into a
failing test."""
import json
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_version() -> str:
    with open(os.path.join(REPO_ROOT, "pyproject.toml")) as f:
        text = f.read()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise AssertionError("couldn't find a version field in pyproject.toml")
    return match.group(1)


class TestReleaseMetadataIsInSync(unittest.TestCase):
    def _load_server_json(self) -> dict:
        with open(os.path.join(REPO_ROOT, "server.json")) as f:
            return json.load(f)

    def test_server_json_version_matches_pyproject(self):
        version = _pyproject_version()
        server = self._load_server_json()
        self.assertEqual(server["version"], version, "server.json's top-level version is stale relative to pyproject.toml")
        for pkg in server["packages"]:
            self.assertEqual(pkg["version"], version, f"server.json package {pkg.get('identifier')} version is stale relative to pyproject.toml")

    def test_server_json_description_fits_registry_limit(self):
        # The MCP Registry's schema caps description at 100 chars and
        # rejects the whole publish (a hard CI failure, not a warning)
        # if it's over -- found the hard way when a more-descriptive
        # edit here broke the publish-mcp-registry CI job.
        server = self._load_server_json()
        description = server["description"]
        self.assertLessEqual(len(description), 100, f"server.json description is {len(description)} chars, over the registry's 100-char limit: {description!r}")

    def test_server_json_name_fits_registry_limit(self):
        server = self._load_server_json()
        self.assertLessEqual(len(server["name"]), 200)


if __name__ == "__main__":
    unittest.main()
