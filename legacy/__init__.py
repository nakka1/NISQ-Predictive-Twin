"""
legacy/
=========

Historical modules superseded by the causal WDM rewrite (dataset_v3.py),
kept here (not deleted) so pre-audit experiments remain runnable and
their results remain reproducible -- per this project's repeated
commitment not to silently invalidate past results.

    legacy/dataset.py  -- QuantumNetworkDataset: the pre-causal,
        Ornstein-Uhlenbeck-based fidelity generator. Superseded by
        dataset_v3.QuantumNetworkDatasetV3 (causal, WDM/quantum-feature
        -separated, leakage-free). See legacy/README.md for the full
        history and every experiment script still consuming this module.

Do NOT add new functionality here. Do NOT import from `legacy/` in any
new code -- it exists solely to keep OLD experiment scripts (predating
this project's causal rewrite) functional.
"""
