"""
run_statistical_significance.py
===================================

Master audit acceptance criterion: "statistical significance" for the
controller comparison (Sections 20, 31 experiment 10). Runs paired
t-tests, a non-parametric sign test, effect sizes (Cohen's d), and 95%
confidence intervals on the DualHead/Predictive/Reactive vs. Blind yield
differences collected across the 3 seeds used throughout this audit
(42, 123, 7).

HONEST LIMITATION stated up front, not hidden: n=3 seeds provides very
limited statistical power. A paired t-test with n=3 has only 2 degrees of
freedom, and a sign test's minimum achievable p-value at n=3 is 0.125 --
neither can reach the conventional p<0.05 threshold from a sign test alone
even if EVERY seed agreed in direction. Results are reported with this
context, not overclaimed. A more definitive test would need substantially
more independent seeds.

Usage:
    python run_statistical_significance.py
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats


RESULTS = {
    "Blind":      [31.03, 49.62, 39.45],
    "Reactive":   [31.62, 47.16, 39.49],
    "Predictive": [31.19, 50.87, 40.00],
    "DualHead":   [42.13, 53.37, 50.54],
}
SEEDS = [42, 123, 7]


def paired_analysis(name_a: str, name_b: str, values_a: list, values_b: list) -> dict:
    a, b = np.array(values_a), np.array(values_b)
    diff = a - b
    n = len(diff)

    t_stat, p_ttest = stats.ttest_rel(a, b)
    n_positive = int((diff > 0).sum())
    sign_p = stats.binomtest(n_positive, n=n, p=0.5, alternative="greater").pvalue

    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("inf")
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    ci = stats.t.interval(0.95, df=n - 1, loc=diff.mean(), scale=se) if n > 1 and se > 0 else (diff.mean(), diff.mean())

    return {
        "Comparison": f"{name_a} vs. {name_b}", "N": n, "Mean Diff (pp)": round(float(diff.mean()), 2),
        "95% CI Low": round(float(ci[0]), 2), "95% CI High": round(float(ci[1]), 2),
        "Paired t-stat": round(float(t_stat), 3), "Paired t p-value": round(float(p_ttest), 4),
        "Sign test p-value": round(float(sign_p), 4), "Cohen's d": round(float(cohens_d), 3),
        "All seeds agree in direction": bool(n_positive == n or n_positive == 0),
    }


def main():
    print(f"Seeds used: {SEEDS} (n={len(SEEDS)})")
    print("\nIMPORTANT: n=3 provides LIMITED statistical power. A sign test's minimum")
    print("achievable p-value at n=3 is 0.125 -- it cannot reach p<0.05 on its own even")
    print("if every seed agreed in direction. Results below are reported honestly with")
    print("this context, not overclaimed as definitive significance.\n")

    rows = []
    for challenger in ["DualHead", "Predictive", "Reactive"]:
        rows.append(paired_analysis(challenger, "Blind", RESULTS[challenger], RESULTS["Blind"]))
    rows.append(paired_analysis("DualHead", "Predictive", RESULTS["DualHead"], RESULTS["Predictive"]))
    rows.append(paired_analysis("DualHead", "Reactive", RESULTS["DualHead"], RESULTS["Reactive"]))

    results_df = pd.DataFrame(rows)
    print("=" * 130)
    print(" STATISTICAL SIGNIFICANCE: PAIRED COMPARISONS ACROSS 3 SEEDS ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)

    dh_vs_blind = results_df[results_df["Comparison"] == "DualHead vs. Blind"].iloc[0]
    cohens_d_value = dh_vs_blind["Cohen's d"]
    print(f"\nDualHead vs. Blind: mean advantage {dh_vs_blind['Mean Diff (pp)']:.2f}pp "
          f"(95% CI [{dh_vs_blind['95% CI Low']:.2f}, {dh_vs_blind['95% CI High']:.2f}]pp), "
          f"Cohen's d={cohens_d_value:.2f} (conventionally a LARGE effect size), "
          f"paired t-test p={dh_vs_blind['Paired t p-value']:.4f} (not below 0.05 at n=3, "
          f"consistent with the sample-size limitation stated above).")

    results_df.to_csv("outputs/statistical_significance.csv", index=False)
    print("\nSaved: outputs/statistical_significance.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
