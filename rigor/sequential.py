"""Sequential (peeking-safe) hypothesis testing.

Every test elsewhere in this package assumes the classic fixed-sample
design: decide n in advance (`power.py`), collect exactly that much
data, test once. That's not how most real monitoring actually happens
-- an agent watching a live experiment (or a person watching a
dashboard) checks the running result *every time new data comes in*
and is tempted to stop as soon as it looks significant. Doing that with
an ordinary p-value is a well-known way to fool yourself: a Wiener
process crosses any fixed boundary eventually with probability 1 (the
law of the iterated logarithm), so a naive test re-run at nominal
alpha=0.05 after every new observation is, in the limit of unlimited
peeking, *guaranteed* to eventually show "significant" even when there
is no real effect -- see `naive_peeking_inflation` below for a
finite-horizon Monte Carlo demonstration of the same phenomenon, and
Armitage, McPherson & Rowe (1969), "Repeated Significance Tests on
Accumulating Data".

The fix isn't "don't peek" (nobody actually manages that) -- it's a
p-value that stays valid *no matter when, or how many times, you look*.
This module implements one: the mixture sequential probability ratio
test (mSPRT) of Robbins (1970), specialized to a normal mixing prior
per Johari, Koomen, Pekelis & Walsh (2017), "Peeking at A/B Tests: Why
It Matters and What to Do About It" (KDD) -- the approach behind
Optimizely's and Netflix's/Uber's internal always-valid experimentation
platforms.

The math. At any look, the effect estimate theta_hat (a difference in
means or proportions) is treated as approximately Normal(theta,
1/information) -- the same large-sample approximation power.py and
inference.py's z-tests already lean on. Mixing the likelihood ratio for
theta over a Normal(0, tau^2) prior instead of testing a single fixed
alternative gives a closed form (no numerical integration needed):

    Lambda = (1 + I*tau^2)^(-1/2) * exp[ I*tau^2/(1+I*tau^2) * Z^2/2 ]

where I = 1/se^2 is the current information and Z = theta_hat/se is the
ordinary z-statistic. p = min(1, 1/Lambda) is then an *always-valid*
p-value: for any stopping rule, P(exists a look where p <= alpha | H0
true) <= alpha -- exactly the guarantee a fixed-sample p-value does
*not* have under repeated peeking. Larger Lambda (smaller p) means more
evidence against H0; Lambda always shrinks toward 0 as more looks pile
up under H0 with no real effect, rather than drifting past a fixed
threshold by chance.

tau is the one free choice this method asks for: the standard
deviation of a Normal prior over "what the true effect probably is,"
in theta_hat's own units (a proportion difference or a mean
difference). It is *not* a significance threshold and does not need to
be precise -- Johari et al. show power is near-optimal when tau is set
close to the smallest effect size worth caring about (the same
minimum-detectable-effect number `power.sample_size_two_proportion_z_test`
already asks for): too small and real effects smaller than tau are
underpowered; too large and the test is slow to react. When in doubt,
reuse the effect size you'd have plugged into a fixed-sample power
calculation for the same experiment.

A concrete comparison. A fixed-sample two-proportion z-test needs a
pre-committed n; checking it early and stopping the moment p<.05 is
exactly the failure mode above. The sequential version can be checked
after every single conversion instead, with no such penalty --
sequential_two_proportion_test(55, 500, 40, 500, tau=0.05).p_value is
~0.91 (not yet significant, correctly, off a 3-point gap this early);
the same function called again after more data keeps returning a
valid p-value, at n=50 or n=50000, with no correction to apply and no
"how many times have I checked" bookkeeping required.
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from rigor import distributions as dist


@dataclass
class SequentialTestResult:
    name: str
    theta_hat: float
    se: float
    tau: float
    z: float
    mixture_likelihood_ratio: float
    p_value: float
    n1: Optional[int] = None
    n2: Optional[int] = None
    citation: str = ""
    warnings: List[str] = field(default_factory=list)

    def reject_null(self, alpha: float = 0.05) -> bool:
        """True if p_value < alpha. Unlike a fixed-sample TestResult,
        this stays a valid check *no matter how many times it's already
        been called on this same experiment as more data came in*."""
        return self.p_value < alpha


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: Sequence[float], mean: Optional[float] = None) -> float:
    m = _mean(xs) if mean is None else mean
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations to estimate variance")
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _zero_se_result(theta_hat: float) -> Tuple[float, float, float]:
    """z, Lambda, p for the degenerate se=0 case (no variance observed
    yet -- e.g. both groups still constant, or complete separation with
    every observation on one side so far).

    This is deliberately *not* the asymmetric convention
    inference._safe_ratio uses for the same zero-variance situation in a
    fixed-sample test (there, a nonzero difference against zero variance
    is treated as the strongest possible evidence, z=+-inf). That
    convention is wrong here: with a plug-in variance estimate checked
    after every single observation, se hitting exactly 0 from pure
    small-sample noise is *common*, not diagnostic. Complete separation
    at n=1 per arm (one success, one failure) happens under the null
    roughly as often as the two outcomes' base rate suggests -- verified
    by simulation at ~42% for p=0.3, not a rare fluke -- so treating it
    as certainty would let that 42% immediately "reject," gutting the
    always-valid guarantee this whole module exists to provide (this
    was caught by test_stopping_at_first_significant_look_still_bounds_
    false_positive_rate, which failed at a ~42% false-positive rate
    during development, exactly matching that fraction).

    So: report p=1 and Lambda=1 (no actionable evidence) unconditionally
    whenever se=0, regardless of theta_hat -- an early degenerate
    variance estimate is a reason to distrust the statistic, not a
    reason to declare victory. z is still reported honestly (0 for a
    genuine tie, +-inf for an observed but not-yet-trustworthy
    separation) since it's a plain descriptive fact about the data, not
    a significance claim.
    """
    z = 0.0 if theta_hat == 0.0 else (math.inf if theta_hat > 0 else -math.inf)
    return z, 1.0, 1.0


def _mixture_log_likelihood_ratio(theta_hat: float, se: float, tau: float) -> float:
    if se <= 0:
        raise ValueError("se must be positive")
    if tau <= 0:
        raise ValueError(
            "tau must be positive -- it's the mixing prior's standard "
            "deviation over the true effect, in theta_hat's own units "
            "(e.g. the smallest proportion/mean difference worth detecting)"
        )
    information = 1.0 / (se * se)
    itau2 = information * tau * tau
    z = theta_hat / se
    # log(Lambda), derived by mixing the Normal(theta, se^2) likelihood
    # ratio over a Normal(0, tau^2) prior on theta and solving the
    # resulting Gaussian integral in closed form (Robbins 1970; Johari
    # et al. 2017, eq. 3) -- kept in log space since Lambda itself can
    # be astronomically large or small far from H0.
    return -0.5 * math.log1p(itau2) + (itau2 / (1.0 + itau2)) * (z * z) / 2.0


def always_valid_p_value(theta_hat: float, se: float, tau: float) -> Tuple[float, float]:
    """The mSPRT mixture likelihood ratio Lambda and always-valid
    p-value (min(1, 1/Lambda)) for an effect estimate theta_hat with
    standard error se, under a Normal(0, tau^2) mixing prior on the true
    effect. Low-level building block behind
    sequential_two_sample_mean_test/sequential_two_proportion_test;
    called directly for any other approximately-normal estimator (e.g.
    a regression coefficient) that isn't wrapped here yet.
    """
    log_lambda = _mixture_log_likelihood_ratio(theta_hat, se, tau)
    try:
        mixture_likelihood_ratio = math.exp(log_lambda)
    except OverflowError:
        mixture_likelihood_ratio = math.inf
    # p = min(1, 1/Lambda) without computing 1/Lambda directly: Lambda<=1
    # (log_lambda<=0) means p=1 by construction, and only ever exponentiating
    # a non-positive number when Lambda>1 avoids overflow at the other end.
    p_value = 1.0 if log_lambda <= 0 else math.exp(-log_lambda)
    return mixture_likelihood_ratio, p_value


def sequential_two_sample_mean_test(
    a: Sequence[float], b: Sequence[float], tau: float, equal_var: bool = False
) -> SequentialTestResult:
    """Always-valid test of H0: the two population means are equal,
    checkable after every new observation in either group without
    inflating the false-positive rate -- the sequential-monitoring
    counterpart to inference.two_sample_t_test. tau is the mixing
    prior's standard deviation over the true mean difference, in a and
    b's own units (see the module docstring for how to pick it).
    Defaults to Welch's (unequal-variance) standard error, matching
    two_sample_t_test's own default.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("need at least 2 observations per sample")
    m1, m2 = _mean(a), _mean(b)
    v1, v2 = _variance(a, m1), _variance(b, m2)
    if equal_var:
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = math.sqrt(pooled * (1 / n1 + 1 / n2))
        citation = "Sequential (mSPRT) two-sample test, pooled variance."
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        citation = "Sequential (mSPRT) two-sample test, Welch (unpooled) variance."
    theta_hat = m2 - m1
    warnings = []
    if n1 < 30 or n2 < 30:
        warnings.append(
            f"n1={n1}, n2={n2} is small; the normal approximation this test "
            "leans on (like every large-sample z-test in this package) is "
            "least reliable at small, early-look sample sizes -- treat an "
            "early result skeptically even if it clears alpha."
        )
    if se == 0.0:
        # Both samples constant so far (e.g. every observation identical
        # within each group, or one group's constant value happens to
        # differ from the other's) -- see _zero_se_result for why this
        # conservatively reports p=1 even if theta_hat != 0.
        z, lam, p = _zero_se_result(theta_hat)
        warnings.append(
            "se=0 (no variance observed in one or both samples yet) -- reporting "
            "no actionable evidence (p=1) rather than treating this as certainty, "
            "since a degenerate variance estimate this early is exactly the kind "
            "of small-sample noise this test protects against. Revisit once more "
            "data brings se above 0."
        )
    else:
        z = theta_hat / se
        lam, p = always_valid_p_value(theta_hat, se, tau)
    return SequentialTestResult(
        name="sequential two-sample mean test (mSPRT)",
        theta_hat=theta_hat,
        se=se,
        tau=tau,
        z=z,
        mixture_likelihood_ratio=lam,
        p_value=p,
        n1=n1,
        n2=n2,
        citation=citation,
        warnings=warnings,
    )


def sequential_two_proportion_test(
    successes1: int, n1: int, successes2: int, n2: int, tau: float
) -> SequentialTestResult:
    """Always-valid test of H0: the two population proportions (e.g. two
    conversion rates) are equal, checkable after every new observation
    in either group -- the sequential-monitoring counterpart to
    inference.two_proportion_z_test. tau is the mixing prior's standard
    deviation over the true proportion difference (e.g. 0.02 for "I
    mainly care about catching a 2-point-or-larger swing").
    """
    if not (0 <= successes1 <= n1 and 0 <= successes2 <= n2):
        raise ValueError("successes must be between 0 and n for each group")
    if n1 < 1 or n2 < 1:
        raise ValueError("need at least 1 observation per group")
    p1, p2 = successes1 / n1, successes2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    theta_hat = p2 - p1
    warnings = []
    for label, n, p in (("group 1", n1, p1), ("group 2", n2, p2)):
        if n * p < 5 or n * (1 - p) < 5:
            warnings.append(f"{label}: n*p or n*(1-p) is below 5 -- normal approximation may be unreliable this early.")
    if se == 0.0:
        # Both groups at 0%/100% so far, including complete separation
        # (e.g. 0/5 vs 5/5) -- see _zero_se_result for why that's
        # reported as no actionable evidence (p=1) rather than
        # certainty: at small n, complete separation happens under the
        # null often enough by pure chance that treating it as proof
        # would blow through the always-valid guarantee.
        z, lam, p = _zero_se_result(theta_hat)
        warnings.append(
            "se=0 (both groups at 0% or 100% so far) -- reporting no actionable "
            "evidence (p=1) rather than treating complete separation as proof, "
            "since that can happen by pure chance at small n. Revisit once more "
            "data brings se above 0."
        )
    else:
        z = theta_hat / se
        lam, p = always_valid_p_value(theta_hat, se, tau)
    return SequentialTestResult(
        name="sequential two-proportion test (mSPRT)",
        theta_hat=theta_hat,
        se=se,
        tau=tau,
        z=z,
        mixture_likelihood_ratio=lam,
        p_value=p,
        n1=n1,
        n2=n2,
        citation="Sequential (mSPRT) two-proportion test.",
        warnings=warnings,
    )


def naive_peeking_inflation(n_looks: int, alpha: float = 0.05, trials: int = 20000, seed: int = 1234) -> dict:
    """Monte Carlo estimate of the actual false-positive rate from the
    failure mode this module exists to fix: repeatedly re-running an
    *ordinary* fixed-sample z-test at the same nominal alpha as data
    accumulates, and stopping the first time it clears alpha, versus the
    alpha actually intended.

    Simulates `trials` independent random walks under H0 (true effect
    zero): at look k (k=1..n_looks), the standardized statistic is
    Z_k = S_k / sqrt(k) where S_k is the cumulative sum of k iid
    Normal(0,1) draws -- the standard Brownian-motion model of a
    z-statistic recomputed as independent observations accumulate.
    Reports the fraction of simulated experiments where |Z_k| >
    z_crit(alpha) at *any* k <= n_looks, i.e. the true type-I error rate
    of "check after every batch, stop at the first p<alpha."

    Deterministic (seeded) but still a simulation, not a closed form --
    unlike the rest of this module's math, there's no known closed-form
    exact answer for a *finite* number of discrete looks (the
    continuous-monitoring limit as n_looks -> infinity does have one:
    the false-positive rate tends to 1, an instance of the law of the
    iterated logarithm). Returns the point estimate plus its own Monte
    Carlo standard error so the estimate's precision is stated rather
    than implied.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    rng = random.Random(seed)
    zcrit = dist.normal_ppf(1 - alpha / 2)
    false_positives = 0
    for _ in range(trials):
        cumulative = 0.0
        rejected = False
        for k in range(1, n_looks + 1):
            cumulative += rng.gauss(0.0, 1.0)
            if abs(cumulative / math.sqrt(k)) > zcrit:
                rejected = True
                break
        if rejected:
            false_positives += 1
    rate = false_positives / trials
    se = math.sqrt(rate * (1 - rate) / trials)
    return {
        "nominal_alpha": alpha,
        "n_looks": n_looks,
        "estimated_true_alpha": rate,
        "monte_carlo_se": se,
        "trials": trials,
        "citation": (
            "Monte Carlo simulation of the naive-repeated-testing failure "
            "mode this module's mSPRT-based tests avoid; cf. Armitage, "
            "McPherson & Rowe (1969), 'Repeated Significance Tests on "
            "Accumulating Data'."
        ),
    }
