# rigor

<!-- mcp-name: io.github.mrnh/rigor-mcp -->

Verified statistical inference for AI agents.

LLMs are decent at reciting statistics but bad at *doing* it reliably —
a t-statistic or a required sample size is a number recalled from
training data, not computed and checked. `rigor` is the alternative:
classical hypothesis testing, effect sizes, power/sample-size
calculation, and multiple-comparisons correction, computed from scratch
and returned as a cited, assumption-checked answer.

Built as an MCP server: a scan of the current MCP ecosystem (Context7
for coding docs, several physics/engineering/chemistry/geo servers,
even Bentley's STAAD integration) found statistics/experimental design
as one of the few common agent needs nobody had covered yet.

The statistics themselves (`rigor/distributions.py`, `inference.py`,
`effect_size.py`, `power.py`, `corrections.py`) are pure standard
library, no dependencies. The package as a whole does depend on the
official `mcp` SDK, since the MCP server is a first-class part of what
it ships, not an add-on -- see [Install](#install).

## Install

```sh
pip install rigor-mcp
```

(the PyPI distribution is `rigor-mcp` since plain `rigor` was already
taken by an unrelated package; the importable package and the CLI
command are both still just `rigor`.) This gets you both console
commands, `rigor` (CLI) and `rigor-mcp` (MCP server) -- deliberately
one install, no extras to get right, since `uvx rigor-mcp` (how most
MCP clients would actually invoke this) has no way to request an
extra.

## What's in it

- **`rigor/distributions.py`** — t, chi-squared, and F distributions
  built from scratch on stdlib (regularized incomplete gamma/beta),
  verified against exact closed-form identities (t(1) = Cauchy,
  chi2(2) = scaled exponential, t² = F(1, df)) rather than trusted
  transcription.
- **`rigor/inference.py`** — one-/two-sample and paired t-tests,
  one-/two-proportion z-tests, chi-squared goodness-of-fit and
  independence, one-way ANOVA. Each returns a `TestResult`:
  statistic, degrees of freedom, two-tailed p-value, a confidence
  interval, a citation, and assumption warnings (e.g. small-n normality
  reliance, low expected cell counts).
- **`rigor/effect_size.py`** — Cohen's d, Hedges' g, Cohen's h, Cramér's V.
- **`rigor/power.py`** — power and required sample size for the
  two-sample t-test and two-proportion z-test. The two directions
  (given n, find power; given power, find n) are exact numerical
  inverses of each other by construction (bisection on the same
  underlying power function), and sanity-checked against the Cohen
  (1988) d=0.5/α=.05/power=.80 textbook reference case (n≈64).
- **`rigor/corrections.py`** — Bonferroni and Benjamini-Hochberg (FDR)
  multiple-comparisons correction.
- **`rigor/cli.py`** — a CLI over all of the above (`rigor.py` at the
  repo root is a thin shim so `python3 rigor.py ...` also works from a
  plain checkout, without installing anything).
- **`rigor/mcp_server.py`** — an MCP tool wrapper exposing all 17
  operations to any MCP client (Claude Code, Claude Desktop, etc.).
  Smoke-tested end-to-end over stdio against a real client — tool
  discovery plus representative calls checked against known reference
  values, including the full round-trip still landing the Cohen (1988)
  case at n=63.

## Usage

CLI, once installed:

```sh
rigor ttest one-sample --data 5.1,4.9,5.3,5.0,4.8,5.2 --mu0 5.0
rigor power ttest-2samp --effect-size 0.5 --power 0.8
rigor --help   # full list of subcommands (ttest, ztest, chi2, anova, effect-size, power, correct)
```

or straight from a checkout without installing anything:

```sh
python3 rigor.py ttest one-sample --data 5.1,4.9,5.3,5.0,4.8,5.2 --mu0 5.0
```

MCP server, over stdio (the transport local clients like Claude Code
expect):

```sh
pip install rigor-mcp
rigor-mcp
```

or from a checkout: `pip install mcp && python3 -m rigor.mcp_server`.

Register it with Claude Code:

```sh
claude mcp add rigor -- rigor-mcp
```

(or, from a checkout: `claude mcp add rigor -- python3 -m rigor.mcp_server`,
run from this repo's root or with an absolute module path). For
interactive poking with the MCP Inspector, run it as a script rather
than the installed command — which means the package root has to be
put on the path by hand, since the Inspector imports the file directly:

```sh
pip install "mcp[cli]"
PYTHONPATH=. mcp dev rigor/mcp_server.py
```

## A transport-level edge case, handled

`cohens_d` correctly returns `+inf`/`-inf` for zero-variance samples
(per its own documented contract), but non-finite floats serialize to
JSON `null` over MCP's structured content — which used to fail the
tool's own number-typed output schema and crash the call. The MCP
`cohens_d` tool now returns `{"value": float | null, "warnings": [...]}`
instead of a bare float, so that case is reported explicitly (null
value, a warning naming the direction) rather than blowing up. Every
other numeric tool here is bounded and always finite for valid input,
so this treatment is specific to `cohens_d`.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

70 tests: 65 exercise the statistics directly; 5 spawn `mcp_server.py`
as a real MCP client would and check results over the wire (skipped
automatically if `mcp` isn't installed).

## License

MIT — see [LICENSE](LICENSE).

[![rigor MCP server](https://glama.ai/mcp/servers/mrnh/rigor/badges/card.svg)](https://glama.ai/mcp/servers/mrnh/rigor)
