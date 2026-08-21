"""
equivalence_testing.py
==========================

Master prompt v4, Fase 10: formal H0/H1 hypothesis testing framing plus
EQUIVALENCE / NON-INFERIORITY testing (TOST -- Two One-Sided Tests) for
the WDM-only vs. privileged/oracle comparison, per the master prompt's
explicit warning: "Não afirmar equivalência simplesmente porque um teste
não encontrou significância."

The thirty-second addendum found WDM-only vs. full-oracle access were
"statistically indistinguishable" (paired t-test p=0.59) -- but FAILING
to reject H0 (no difference) is NOT the same as having evidence FOR H0
(equivalence). A proper equivalence claim requires TOST: two one-sided
tests, each checking whether the true difference is significantly
WITHIN a pre-specified equivalence margin on each side.

    Standard significance test:
        H0: difference = 0        H1: difference != 0
        Failing to reject H0 does NOT mean difference = 0.

    TOST equivalence test:
        H0_lower: difference <= -margin     H1_lower: difference > -margin
        H0_upper: difference >= +margin     H1_upper: difference < +margin
        Rejecting BOTH H0_lower and H0_upper supports difference in
        (-margin, +margin) -- a genuine equivalence claim.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TOSTResult:
    mean_difference: float
    margin: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool
    alpha: float


def tost_paired(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05) -> TOSTResult:
    """
    Two One-Sided Tests (TOST) for paired samples: tests whether the true
    mean difference (a - b) lies within (-margin, +margin) -- a genuine
    equivalence claim, not merely "no significant difference found."

    `margin` MUST be chosen based on domain-relevant reasoning, never
    picked after seeing the data to force a desired conclusion.
    """
    diff = a - b
    n = len(diff)
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    if se == 0.0:
        equivalent = bool(-margin < mean_diff < margin)
        return TOSTResult(mean_diff, margin, 0.0 if equivalent else 1.0,
                           0.0 if equivalent else 1.0, 1.0 if not equivalent else 0.0, equivalent, alpha)

    df = n - 1
    t_lower = (mean_diff - (-margin)) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df=df))
    t_upper = (mean_diff - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df=df))

    p_tost = max(p_lower, p_upper)
    equivalent = bool(p_tost < alpha)

    return TOSTResult(mean_difference=mean_diff, margin=margin, p_lower=p_lower, p_upper=p_upper,
                       p_tost=p_tost, equivalent=equivalent, alpha=alpha)
