"""
run_multiple_comparisons_correction.py
==========================================

Master prompt v5, Secao 10: applies Holm-Bonferroni and Benjamini
-Hochberg correction to REAL p-values already computed in this project's
prior experiments (currently: the forty-sixth addendum's three pairwise
DualHead-vs-{Blind,Reactive,Predictive} paired t-tests) -- resolving a
real gap this project had never addressed: multiple related p-values
were reported side by side without any family-wise/FDR correction.

Usage:
    python run_multiple_comparisons_correction.py
"""

import argparse
import os

import pandas as pd

from multiple_comparisons import holm_bonferroni, benjamini_hochberg


def main():
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/controller_comparison_10seed_pairwise.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the forty-sixth addendum's "
                                 f"controller comparison campaign first.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} pairwise comparisons from {path}:")
    print(df[["Comparison", "Paired_t_p"]].to_string(index=False))

    labels = df["Comparison"].tolist()
    raw_p = df["Paired_t_p"].tolist()

    holm_result = holm_bonferroni(raw_p, labels=labels, alpha=0.05)
    bh_result = benjamini_hochberg(raw_p, labels=labels, alpha=0.05)

    print("\n" + "=" * 90)
    print(" HOLM-BONFERRONI CORRECTION (family-wise error rate control) ".center(90, "="))
    print("=" * 90)
    print(holm_result.to_string(index=False))

    print("\n" + "=" * 90)
    print(" BENJAMINI-HOCHBERG CORRECTION (false discovery rate control) ".center(90, "="))
    print("=" * 90)
    print(bh_result.to_string(index=False))
    print("=" * 90)

    print("\nInterpretation:")
    all_significant_holm = holm_result["significant_at_alpha"].all()
    all_significant_bh = bh_result["significant_at_alpha"].all()
    print(f"  All 3 comparisons remain significant after Holm-Bonferroni correction: {all_significant_holm}")
    print(f"  All 3 comparisons remain significant after Benjamini-Hochberg correction: {all_significant_bh}")
    if all_significant_holm and all_significant_bh:
        print("\n  -> The forty-sixth addendum's DualHead-vs-{Blind,Reactive,Predictive} finding")
        print("     survives BOTH the more conservative (Holm-Bonferroni) and less conservative")
        print("     (Benjamini-Hochberg) multiple-comparisons corrections -- the original,")
        print("     uncorrected significance was NOT an artifact of running 3 related tests")
        print("     without adjustment.")

    combined = pd.concat([holm_result.assign(source="holm"), bh_result.assign(source="bh")], ignore_index=True)
    combined.to_csv("outputs/multiple_comparisons_correction.csv", index=False)
    print("\nSaved: outputs/multiple_comparisons_correction.csv")
    return holm_result, bh_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
