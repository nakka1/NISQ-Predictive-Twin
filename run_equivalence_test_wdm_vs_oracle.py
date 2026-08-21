"""
run_equivalence_test_wdm_vs_oracle.py
=========================================

Master prompt v4, Fase 10: applies the formal TOST equivalence test to
the thirty-second addendum's real 10-seed WDM-only vs. full-oracle
comparison, resolving the honest caveat that addendum left open.

MARGIN JUSTIFICATION (domain-relevant, decided BEFORE looking at whether
it produces a "nice" result): this project's admission-control threshold
is 0.65, and the seventeenth/thirty-first addenda's controller
comparisons showed conditional MAE differences smaller than 0.005
translate to only a few percentage points of downstream yield -- so a
margin of 0.005 MAE is used here as a pre-specified, decision-relevant
equivalence threshold.

Usage:
    python run_equivalence_test_wdm_vs_oracle.py
"""

import argparse
import json
import os

import pandas as pd

from equivalence_testing import tost_paired


EQUIVALENCE_MARGIN_MAE = 0.005


def main():
    os.makedirs("outputs", exist_ok=True)
    data_path = "outputs/wdm_vs_privileged_10seeds.json"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found -- run run_wdm_vs_privileged_single_seed.py across the 10 seeds "
            "(thirty-second addendum) first to produce this file."
        )

    with open(data_path) as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} seeds from {data_path}")
    print(df.to_string(index=False))

    mae_a = df["mae_a_wdm_only"].values
    mae_e = df["mae_e_full_oracle"].values

    print(f"\nEquivalence margin: {EQUIVALENCE_MARGIN_MAE} MAE (see this script's module docstring "
          f"for the pre-specified, domain-relevant justification)")

    print("\n" + "=" * 90)
    print(" FORMAL HYPOTHESIS FRAMING ".center(90, "="))
    print("=" * 90)
    print("Standard significance test:")
    print("  H0: MAE(WDM-only) - MAE(Oracle) = 0")
    print("  H1: MAE(WDM-only) - MAE(Oracle) != 0")
    print("  (thirty-second addendum: paired t p=0.5937 -- FAILS to reject H0,")
    print("   which is NOT itself evidence FOR H0.)")
    print()
    print("TOST equivalence test (this script):")
    print(f"  H0_lower: difference <= -{EQUIVALENCE_MARGIN_MAE}   H1_lower: difference > -{EQUIVALENCE_MARGIN_MAE}")
    print(f"  H0_upper: difference >= +{EQUIVALENCE_MARGIN_MAE}   H1_upper: difference < +{EQUIVALENCE_MARGIN_MAE}")
    print("  Rejecting BOTH supports a genuine equivalence claim within the margin.")

    result = tost_paired(mae_a, mae_e, margin=EQUIVALENCE_MARGIN_MAE, alpha=0.05)

    print("\n" + "=" * 90)
    print(" TOST RESULT ".center(90, "="))
    print("=" * 90)
    print(f"Mean difference (WDM-only - Oracle): {result.mean_difference:+.5f} MAE")
    print(f"p_lower (H0_lower): {result.p_lower:.4f}")
    print(f"p_upper (H0_upper): {result.p_upper:.4f}")
    print(f"p_TOST (max of the two, the conservative combined p-value): {result.p_tost:.4f}")
    print(f"Equivalent within +/-{EQUIVALENCE_MARGIN_MAE} MAE at alpha=0.05: {result.equivalent}")

    if result.equivalent:
        print(f"\n  -> WDM-only telemetry is STATISTICALLY EQUIVALENT to full-oracle access within a")
        print(f"     pre-specified, domain-relevant margin of {EQUIVALENCE_MARGIN_MAE} MAE (p_TOST="
              f"{result.p_tost:.4f} < 0.05) -- a genuine equivalence claim backed by TOST, not merely")
        print(f"     the ABSENCE of significance in a standard two-sided test.")
    else:
        print(f"\n  -> Equivalence within +/-{EQUIVALENCE_MARGIN_MAE} MAE is NOT supported by this data")
        print(f"     (p_TOST={result.p_tost:.4f} >= 0.05) -- reported honestly. The standard test's")
        print(f"     earlier 'statistically indistinguishable' framing should NOT be read as proof of")
        print(f"     equivalence; this TOST result is the more rigorous, appropriate test.")

    output = {
        "margin": EQUIVALENCE_MARGIN_MAE, "alpha": 0.05, "mean_difference": result.mean_difference,
        "p_lower": result.p_lower, "p_upper": result.p_upper, "p_tost": result.p_tost,
        "equivalent": result.equivalent, "n_seeds": len(df),
    }
    with open("outputs/equivalence_test_wdm_vs_oracle.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: outputs/equivalence_test_wdm_vs_oracle.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
