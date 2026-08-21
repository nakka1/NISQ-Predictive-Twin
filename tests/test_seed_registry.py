"""
tests/test_seed_registry.py

Unit tests for seed_registry.py (master prompt v5, Secao 9).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seed_registry import SeedRegistry, compute_config_hash


def test_register_returns_record_with_required_fields():
    reg = SeedRegistry(registry_path="/tmp/_unused_registry.csv")
    record = reg.register(experiment_id="exp_1", seed=42, campaign_name="test")
    assert record.experiment_id == "exp_1"
    assert record.seed == 42
    assert record.timestamp is not None
    assert record.git_commit is not None


def test_compute_config_hash_is_deterministic():
    cfg = {"a": 1, "b": {"c": 2}}
    hash1 = compute_config_hash(cfg)
    hash2 = compute_config_hash(cfg)
    assert hash1 == hash2


def test_compute_config_hash_differs_for_different_configs():
    hash1 = compute_config_hash({"a": 1})
    hash2 = compute_config_hash({"a": 2})
    assert hash1 != hash2


def test_compute_config_hash_ignores_key_order():
    """A config dict's hash must be the SAME regardless of key insertion
    order -- verifying real content equality, not incidental dict ordering."""
    hash1 = compute_config_hash({"a": 1, "b": 2})
    hash2 = compute_config_hash({"b": 2, "a": 1})
    assert hash1 == hash2


def test_verify_seeds_unique_detects_true_duplicate():
    reg = SeedRegistry(registry_path="/tmp/_unused_registry.csv")
    reg.register(experiment_id="exp_1", seed=42, campaign_name="camp")
    reg.register(experiment_id="exp_2", seed=42, campaign_name="camp")  # duplicate seed
    assert reg.verify_seeds_unique("camp") is False


def test_verify_seeds_unique_passes_for_genuinely_distinct_seeds():
    reg = SeedRegistry(registry_path="/tmp/_unused_registry.csv")
    for seed in [42, 123, 7]:
        reg.register(experiment_id=f"exp_{seed}", seed=seed, campaign_name="camp")
    assert reg.verify_seeds_unique("camp") is True


def test_verify_seeds_unique_scoped_per_campaign():
    """The same seed used in TWO DIFFERENT campaigns must not be flagged
    as a duplicate -- uniqueness is checked WITHIN a campaign, not globally."""
    reg = SeedRegistry(registry_path="/tmp/_unused_registry.csv")
    reg.register(experiment_id="exp_a", seed=42, campaign_name="campaign_A")
    reg.register(experiment_id="exp_b", seed=42, campaign_name="campaign_B")
    assert reg.verify_seeds_unique("campaign_A") is True
    assert reg.verify_seeds_unique("campaign_B") is True


def test_save_creates_and_appends_to_registry_file(tmp_path):
    registry_path = str(tmp_path / "registry.csv")
    reg1 = SeedRegistry(registry_path=registry_path)
    reg1.register(experiment_id="exp_1", seed=42, campaign_name="camp")
    df1 = reg1.save()
    assert len(df1) == 1

    reg2 = SeedRegistry(registry_path=registry_path)
    reg2.register(experiment_id="exp_2", seed=123, campaign_name="camp")
    df2 = reg2.save()
    assert len(df2) == 2  # appended, not overwritten
