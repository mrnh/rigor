# rigor

<!-- mcp-name: io.github.mrnh/rigor-mcp -->

Verified statistical inference for AI agents.

LLMs are decent at reciting statistics but bad at *doing* it reliably —
a t-statistic or a required sample size is a number recalled from
training data, not computed and checked. `rigor` is the alternative:
classical hypothesis testing (parametric and non-parametric),
correlation and regression, effect sizes, power/sample-size
calculation, and multiple-comparisons correction, computed from scratch
and returned as a cited, assumption-checked answer -- plus a decision
helper for picking the right tool and a batch tool for running/
correcting many comparisons at once, since "which test do I even use"
and "I forgot to correct for multiple comparisons" are their own common
failure modes, distinct from getting a single formula wrong.

**A concrete case where this matters.** The one sample-size number
everyone half-remembers is Cohen (1988)'s own worked example: d=0.5,
alpha=.05, power=.80 -> n≈64 per group. It's in every textbook and
slide deck, so it's also what gets pattern-matched to when a
*similar*-looking question comes up. Ask instead for d=0.46, power=.85
-- a modest, realistic revision, not a trick:

```sh
$ rigor power ttest-2samp --effect-size 0.46 --power 0.85
Required n per group = 84.86 (round up: 85)
```

85, not "about 64" -- a third more participants to recruit than the
half-remembered number suggests, from a question that *looks* like the
famous one. The formula itself isn't hard (`power.py` runs the same
bisection search either direction, in a few lines); the failure mode
is that recalling a nearby-looking answer feels indistinguishable from
computing the right one, right up until the number's wrong.

Built as an MCP server: a scan of the current MCP ecosystem (Context7
for coding docs, several physics/engineering/chemistry/geo servers,
even Bentley's STAAD integration) found statistics/experimental design
as one of the few common agent needs nobody had covered yet.

The statistics themselves (`rigor/distributions.py`, `inference.py`,
`nonparametric.py`, `correlation.py`, `regression.py`,
`effect_size.py`, `power.py`, `corrections.py`, plus the decision/batch
helpers in `advisor.py` and `batch.py`) are pure standard library, no
dependencies. The package as a whole does depend on the official `mcp`
SDK, since the MCP server is a first-class part of what it ships, not
an add-on -- see [Install](#install).

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
  independence, Fisher's exact test (2x2, exact via the hypergeometric
  distribution — the small-sample alternative chi_square_independence's
  own low-expected-count warning points to), one-way ANOVA, and
  Levene's (Brown-Forsythe) test for equal variances. Each returns a
  `TestResult`: statistic, degrees of freedom, two-tailed p-value, a
  confidence interval, a citation, and assumption warnings (e.g. small-n
  normality reliance, low expected cell counts).
- **`rigor/nonparametric.py`** — Mann-Whitney U, Wilcoxon signed-rank,
  and Kruskal-Wallis: the non-parametric alternative to
  two_sample_t_test/paired_t_test/one_way_anova respectively, for when
  a parametric test's own assumption warnings make its result suspect.
  Rank-based, with tie correction; also returns `TestResult`.
- **`rigor/correlation.py`** — Pearson (linear) and Spearman
  (monotonic, via ranks) correlation, each returned as a `TestResult`
  (H0: no association) with a confidence interval via the Fisher
  z-transform.
- **`rigor/regression.py`** — simple (single-predictor) ordinary least
  squares regression: slope, intercept, R², and a significance test +
  CI for the slope.
- **`rigor/effect_size.py`** — Cohen's d, Hedges' g, Cohen's h, Cramér's
  V, eta²/omega² (for one_way_anova), and rank-biserial correlation
  (for mann_whitney_u).
- **`rigor/power.py`** — power and required sample size for the
  one-/two-sample t-test and two-proportion z-test (the one-sample
  formula covers paired_t_test too, since a paired t-test is a
  one-sample t-test on the differences). The two directions (given n,
  find power; given power, find n) are exact numerical inverses of each
  other by construction (bisection on the same underlying power
  function), and sanity-checked against the Cohen (1988)
  d=0.5/α=.05/power=.80 textbook reference case (n≈64).
- **`rigor/corrections.py`** — Bonferroni and Benjamini-Hochberg (FDR)
  multiple-comparisons correction.
- **`rigor/advisor.py`** — `recommend_test`: a decision helper, not a
  statistic. Answer a few characteristics of the data/question
  (continuous/proportion/categorical/ordinal, how many groups, paired,
  small-or-skewed, association-not-difference) and get back which tool
  to call, what to call instead if this test's assumptions look shaky,
  and what to run alongside it -- compiling the cross-references every
  other module's docstrings already carry into one callable answer, so
  an agent doesn't need to have already read all of them to find the
  relevant one.
- **`rigor/batch.py`** — `pairwise_group_comparisons`: runs every
  pairwise comparison across 2+ groups (`two_sample_t_test` or
  `mann_whitney_u`, your choice) and applies Bonferroni/BH correction
  to the whole batch in one call, instead of the agent orchestrating
  k*(k-1)/2 separate calls plus a correction call by hand and risking
  forgetting the correction step. The natural follow-up
  `one_way_anova`/`kruskal_wallis` already recommend in their own
  docstrings once a result comes back significant.
- **`rigor/cli.py`** — a CLI over all of the above (`rigor.py` at the
  repo root is a thin shim so `python3 rigor.py ...` also works from a
  plain checkout, without installing anything).
- **`rigor/mcp_server.py`** — an MCP tool wrapper exposing all 32
  operations to any MCP client (Claude Code, Claude Desktop, etc.).
  Smoke-tested end-to-end over stdio against a real client — tool
  discovery plus representative calls checked against known reference
  values, including the full round-trip still landing the Cohen (1988)
  case at n=63 and Fisher's original "lady tasting tea" case at
  p≈0.4857.

## Usage

CLI, once installed:

```sh
rigor ttest one-sample --data 5.1,4.9,5.3,5.0,4.8,5.2 --mu0 5.0
rigor corr pearson --x 1,2,3,4,5 --y 2,4,5,4,5
rigor regress --x 1,2,3,4,5 --y 3,5,7,9,11
rigor nonparam mann-whitney --a 1,2,3 --b 4,5,6
rigor power ttest-2samp --effect-size 0.5 --power 0.8
rigor recommend --outcome-type continuous --n-groups 3   # which test fits?
rigor posthoc --groups "1,2,3|4,5,6|7,8,9" --labels A,B,C  # pairwise + correction
rigor --help   # full list of subcommands (ttest, ztest, chi2, fisher, anova,
                # levene, nonparam, corr, regress, effect-size, power, correct,
                # recommend, posthoc)
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
value, a warning naming the direction) rather than blowing up. That
fix is specific to tools with a *bare-scalar* output schema — every
tool that returns a dict (all the `TestResult`-based ones, plus
`simple_linear_regression`) has been confirmed over real stdio to pass
a non-finite field straight through as JSON's non-standard `Infinity`,
since a generic dict return doesn't get a strict per-field number
schema. Of the bare-float tools, `cohens_d` is the only one that can
actually produce a non-finite value.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

153 tests: 140 exercise the statistics/decision logic directly; 12
spawn `mcp_server.py` as a real MCP client would and check results over
the wire (skipped automatically if `mcp` isn't installed); 1 checks
that server.json's version hasn't drifted from pyproject.toml's (the
two aren't otherwise linked -- see test_release_metadata.py).

## License

MIT — see [LICENSE](LICENSE).

[![rigor MCP server](https://glama.ai/mcp/servers/mrnh/rigor/badges/card.svg)](https://glama.ai/mcp/servers/mrnh/rigor)
