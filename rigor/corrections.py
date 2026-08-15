"""Multiple-comparisons correction.

Run enough hypothesis tests and some will look significant by chance
alone -- at alpha=0.05, testing 20 true nulls turns up about one false
positive on average. These take a list of p-values from a batch of tests
and decide which still count as significant once that's accounted for.
"""
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class CorrectionResult:
    method: str
    alpha: float
    p_values: List[float]
    reject: List[bool]  # same order as p_values
    adjusted_alpha: float = None  # only meaningful for Bonferroni (a single threshold)
    citation: str = ""


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> CorrectionResult:
    """Controls the family-wise error rate by dividing alpha by the number
    of tests (Bonferroni, 1936). Simple and conservative -- it gets more
    conservative (more likely to miss real effects) as the number of tests
    grows, since it doesn't account for how the tests' p-values are
    actually distributed.
    """
    if not p_values:
        raise ValueError("need at least one p-value")
    m = len(p_values)
    adjusted_alpha = alpha / m
    reject = [p < adjusted_alpha for p in p_values]
    return CorrectionResult(
        method="Bonferroni",
        alpha=alpha,
        p_values=list(p_values),
        reject=reject,
        adjusted_alpha=adjusted_alpha,
        citation="Bonferroni correction (Bonferroni, 1936); controls family-wise error rate.",
    )


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> CorrectionResult:
    """Controls the false discovery rate -- the expected proportion of
    false positives *among the tests called significant* -- rather than
    the probability of any false positive at all (Benjamini & Hochberg,
    1995). Less conservative than Bonferroni; the standard choice when
    testing many hypotheses and a few false positives among the "hits"
    are tolerable.
    """
    if not p_values:
        raise ValueError("need at least one p-value")
    m = len(p_values)
    indexed = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m

    # Find the largest rank k where p_(k) <= (k/m) * alpha; reject that
    # hypothesis and everything ranked below it (BH's step-up procedure).
    threshold_rank = -1
    for rank, i in enumerate(indexed, start=1):
        if p_values[i] <= (rank / m) * alpha:
            threshold_rank = rank
    if threshold_rank >= 0:
        for rank, i in enumerate(indexed, start=1):
            if rank <= threshold_rank:
                reject[i] = True

    return CorrectionResult(
        method="Benjamini-Hochberg",
        alpha=alpha,
        p_values=list(p_values),
        reject=reject,
        citation="Benjamini-Hochberg procedure (Benjamini & Hochberg, 1995); controls false discovery rate.",
    )
