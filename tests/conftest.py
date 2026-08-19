"""
tests/conftest.py
====================

Master prompt Fase 21 (CI/CD): auto-applies pytest markers
(unit/physics/integration/slow) based on test FILE name patterns, rather
than requiring every one of this project's 291 tests (across 32 files) to
be manually decorated one at a time. This is a pragmatic, maintainable
strategy for a large EXISTING test suite -- new test files should still
add explicit `@pytest.mark.X` decorators where the auto-classification
below wouldn't get it right, but the common case (a whole file belongs to
one category) is handled automatically here.

Usage (matches the master prompt's exact examples):
    pytest -m unit
    pytest -m physics
    pytest -m integration
    pytest -m slow
"""

import os

_PHYSICS_FILES = {
    "test_physics_regression.py", "test_physics_validation.py", "test_physics_engine.py",
    "test_purification.py", "test_quantum_channel.py", "test_fast_vs_aer_channel.py",
    "test_swapping_and_memory.py", "test_causal_v3.py", "test_environment.py",
    "test_repeater_chain.py", "test_repeater_and_orchestrator.py", "test_causal_chain.py",
    "test_closed_loop_multihop.py",
}

_INTEGRATION_FILES = {
    "test_quantum_twin_package.py", "test_audit_methodology.py", "test_dataset.py",
    "test_dual_head_causal_dataset.py", "test_wdm_vs_privileged.py",
    "test_wdm_vs_privileged_dualhead.py", "test_wdm_vs_privileged_multiseed.py",
    "test_lag_analysis_dualhead.py", "test_causal_analysis.py",
}

_SLOW_FILES = {
    "test_dual_head_causal_dataset.py", "test_wdm_vs_privileged.py",
    "test_wdm_vs_privileged_dualhead.py", "test_wdm_vs_privileged_multiseed.py",
    "test_lag_analysis_dualhead.py", "test_causal_analysis.py", "test_causal_chain.py",
    "test_closed_loop_multihop.py", "test_robust_training.py", "test_edge_ai_benchmark.py",
}


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        filename = os.path.basename(str(item.fspath))
        if filename in _PHYSICS_FILES:
            item.add_marker(pytest.mark.physics)
        elif filename in _INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        if filename in _SLOW_FILES:
            item.add_marker(pytest.mark.slow)
