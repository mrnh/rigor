# Runs rigor's MCP server over stdio. Not needed to use rigor-mcp
# locally (pip install rigor-mcp / uvx rigor-mcp is simpler) -- this
# exists for sandboxed build/introspection checks (e.g. Glama) that
# expect a Dockerfile rather than a bare pip install.
FROM python:3.12-slim

RUN pip install --no-cache-dir rigor-mcp

ENTRYPOINT ["rigor-mcp"]
