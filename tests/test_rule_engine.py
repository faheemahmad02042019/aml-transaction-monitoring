"""
Unit tests for the AML Rule Engine.

Tests all detection rules:
  - Structuring (smurfing) detection
  - Rapid fund movement detection
  - Round-tripping (circular flow) detection
  - Velocity anomaly detection
  - Threshold edge cases and boundary conditions
  - Composite scoring
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.rule_engine import AMLRuleEngine, RuleResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config() -> Config:
    """Return a default Config object for testing."""
    return Config()


@pytest.fixture
def engine(config: Config) -> AMLRuleEngine:
    """Return an AMLRuleEngine instance with default configuration."""
    return AMLRuleEngine(config)


@pytest.fixture
def base_transaction_df() -> pd.DataFrame:
    """
    Return a minimal transaction DataFrame for testing.

    Contains 10 normal, legitimate-looking transactions.
    """
    rng = np.random.RandomState(42)
    n = 10
    return pd.DataFrame({
        "transaction_id": range(n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "from_id": [f"BANK_001_{10000 + i}" for i in range(n)],
        "from_bank": ["BANK_001"] * n,
        "from_account": [str(10000 + i) for i in range(n)],
        "to_id": [f"BANK_002_{20000 + i}" for i in range(n)],
        "to_bank": ["BANK_002"] * n,
        "to_account": [str(20000 + i) for i in range(n)],
        "amount": rng.uniform(100, 5000, n).round(2),
        "amount_paid": rng.uniform(100, 5000, n).round(2),
        "amount_received": rng.uniform(100, 5000, n).round(2),
        "payment_currency": ["USD"] * n,
        "receiving_currency": ["USD"] * n,
        "is_cross_currency": [0] * n,
        "payment_format": ["Wire"] * n,
        "payment_format_normalized": ["wire"] * n,
        "is_laundering": [0] * n,
        "date": pd.date_range("2024-01-01", periods=n, freq="h").date,
        "hour": list(range(n)),
        "day_of_week": [0] * n,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Structuring Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuringDetection:
    """Tests for the structuring / smurfing detection rule."""

    def test_amount_in_structuring_range_triggers(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Amounts between $8,000 and $9,999 should trigger the structuring rule."""
        df = base_transaction_df.copy()
        df.loc[0, "amount"] = 9500.00
        df.loc[1, "amount"] = 8500.00

        result = engine.evaluate(df)

        assert result.loc[0, "rule_structuring_score"] > 0.0
        assert result.loc[1, "rule_structuring_score"] > 0.0

    def test_amount_below_range_does_not_trigger(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Amounts well below $8,000 should not trigger structuring."""
        df = base_transaction_df.copy()
        df["amount"] = 500.0

        result = engine.evaluate(df)

        for i in range(len(df)):
            assert result.loc[i, "rule_structuring_score"] == 0.0

    def test_amount_above_threshold_does_not_trigger(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Amounts above $10,000 should not trigger structuring."""
        df = base_transaction_df.copy()
        df["amount"] = 15000.0

        result = engine.evaluate(df)

        for i in range(len(df)):
            assert result.loc[i, "rule_structuring_score"] == 0.0

    def test_exact_threshold_boundary(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Test amounts at exact boundary: $8,000 and $9,999.99."""
        df = base_transaction_df.copy()
        df.loc[0, "amount"] = 8000.00  # Lower bound
        df.loc[1, "amount"] = 9999.99  # Upper bound
        df.loc[2, "amount"] = 10000.00  # At threshold (should not trigger)

        result = engine.evaluate(df)

        assert result.loc[0, "rule_structuring_score"] > 0.0
        assert result.loc[1, "rule_structuring_score"] > 0.0
        assert result.loc[2, "rule_structuring_score"] == 0.0

    def test_multiple_structuring_transactions_boost_score(self, engine: AMLRuleEngine) -> None:
        """Multiple structuring-range transactions from the same account should increase score."""
        n = 5
        df = pd.DataFrame({
            "transaction_id": range(n),
            "timestamp": pd.date_range("2024-01-01 10:00", periods=n, freq="30min"),
            "from_id": ["BANK_001_12345"] * n,
            "from_bank": ["BANK_001"] * n,
            "from_account": ["12345"] * n,
            "to_id": [f"BANK_002_{20000 + i}" for i in range(n)],
            "to_bank": ["BANK_002"] * n,
            "to_account": [str(20000 + i) for i in range(n)],
            "amount": [9200, 9500, 8800, 9100, 9700],
            "amount_paid": [9200, 9500, 8800, 9100, 9700],
            "amount_received": [9200, 9500, 8800, 9100, 9700],
            "payment_currency": ["USD"] * n,
            "receiving_currency": ["USD"] * n,
            "is_cross_currency": [0] * n,
            "payment_format": ["Wire"] * n,
            "payment_format_normalized": ["wire"] * n,
            "is_laundering": [0] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="30min").date,
            "hour": [10, 10, 11, 11, 12],
            "day_of_week": [0] * n,
        })

        result = engine.evaluate(df)

        # All should be flagged with non-zero scores
        for i in range(n):
            assert result.loc[i, "rule_structuring_score"] > 0.0

    def test_score_capped_at_one(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Structuring scores should never exceed 1.0."""
        df = base_transaction_df.copy()
        df["amount"] = 9500.00

        result = engine.evaluate(df)

        assert (result["rule_structuring_score"] <= 1.0).all()
        assert (result["rule_structuring_score"] >= 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Rapid Movement Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRapidMovement:
    """Tests for the rapid fund movement detection rule."""

    def test_rapid_inflow_outflow_triggers(self, engine: AMLRuleEngine) -> None:
        """Large inflow followed quickly by outflow should trigger."""
        # Account A receives $20,000, then sends $18,000 within 2 hours
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:30"]),
            "from_id": ["EXTERNAL_001", "ACCOUNT_A"],
            "from_bank": ["EXT", "BANK_001"],
            "from_account": ["001", "A"],
            "to_id": ["ACCOUNT_A", "EXTERNAL_002"],
            "to_bank": ["BANK_001", "EXT"],
            "to_account": ["A", "002"],
            "amount": [20000.0, 18000.0],
            "amount_paid": [20000.0, 18000.0],
            "amount_received": [20000.0, 18000.0],
            "payment_currency": ["USD", "USD"],
            "receiving_currency": ["USD", "USD"],
            "is_cross_currency": [0, 0],
            "payment_format": ["Wire", "Wire"],
            "payment_format_normalized": ["wire", "wire"],
            "is_laundering": [0, 0],
            "date": [pd.Timestamp("2024-01-01").date()] * 2,
            "hour": [10, 11],
            "day_of_week": [0, 0],
        })

        result = engine.evaluate(df)

        # At least one transaction should be flagged for rapid movement
        assert result["rule_rapid_movement_score"].max() > 0.0

    def test_small_amounts_do_not_trigger(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Small transactions should not trigger rapid movement."""
        df = base_transaction_df.copy()
        df["amount"] = 100.0  # Well below minimum

        result = engine.evaluate(df)

        assert (result["rule_rapid_movement_score"] == 0.0).all()

    def test_slow_movement_does_not_trigger(self, engine: AMLRuleEngine) -> None:
        """Inflows/outflows separated by many days should not trigger."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-15 10:00"]),
            "from_id": ["EXTERNAL_001", "ACCOUNT_A"],
            "from_bank": ["EXT", "BANK_001"],
            "from_account": ["001", "A"],
            "to_id": ["ACCOUNT_A", "EXTERNAL_002"],
            "to_bank": ["BANK_001", "EXT"],
            "to_account": ["A", "002"],
            "amount": [50000.0, 48000.0],
            "amount_paid": [50000.0, 48000.0],
            "amount_received": [50000.0, 48000.0],
            "payment_currency": ["USD", "USD"],
            "receiving_currency": ["USD", "USD"],
            "is_cross_currency": [0, 0],
            "payment_format": ["Wire", "Wire"],
            "payment_format_normalized": ["wire", "wire"],
            "is_laundering": [0, 0],
            "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-15").date()],
            "hour": [10, 10],
            "day_of_week": [0, 0],
        })

        result = engine.evaluate(df)

        # Should not trigger because 14 days apart exceeds the window
        assert (result["rule_rapid_movement_score"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Round-Tripping Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTripping:
    """Tests for the round-tripping / circular flow detection rule."""

    def test_simple_cycle_triggers(self, engine: AMLRuleEngine) -> None:
        """A->B->A cycle should be detected."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:00"]),
            "from_id": ["ACCOUNT_A", "ACCOUNT_B"],
            "from_bank": ["BANK_001", "BANK_002"],
            "from_account": ["A", "B"],
            "to_id": ["ACCOUNT_B", "ACCOUNT_A"],
            "to_bank": ["BANK_002", "BANK_001"],
            "to_account": ["B", "A"],
            "amount": [5000.0, 4800.0],
            "amount_paid": [5000.0, 4800.0],
            "amount_received": [5000.0, 4800.0],
            "payment_currency": ["USD", "USD"],
            "receiving_currency": ["USD", "USD"],
            "is_cross_currency": [0, 0],
            "payment_format": ["Wire", "Wire"],
            "payment_format_normalized": ["wire", "wire"],
            "is_laundering": [0, 0],
            "date": [pd.Timestamp("2024-01-01").date()] * 2,
            "hour": [10, 15],
            "day_of_week": [0, 0],
        })

        result = engine.evaluate(df)

        assert result["rule_round_tripping_score"].max() > 0.0

    def test_no_cycle_does_not_trigger(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Linear transactions (no cycles) should not trigger."""
        df = base_transaction_df.copy()
        # Ensure all from_id are different from all to_id (no cycles possible)
        df["from_id"] = [f"SENDER_{i}" for i in range(len(df))]
        df["to_id"] = [f"RECEIVER_{i}" for i in range(len(df))]

        result = engine.evaluate(df)

        assert (result["rule_round_tripping_score"] == 0.0).all()

    def test_small_amounts_below_minimum(self, engine: AMLRuleEngine) -> None:
        """Cycles with amounts below the minimum should not trigger."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:00"]),
            "from_id": ["ACCOUNT_A", "ACCOUNT_B"],
            "from_bank": ["BANK_001", "BANK_002"],
            "from_account": ["A", "B"],
            "to_id": ["ACCOUNT_B", "ACCOUNT_A"],
            "to_bank": ["BANK_002", "BANK_001"],
            "to_account": ["B", "A"],
            "amount": [50.0, 50.0],  # Below round_trip_min_amount of $1,000
            "amount_paid": [50.0, 50.0],
            "amount_received": [50.0, 50.0],
            "payment_currency": ["USD", "USD"],
            "receiving_currency": ["USD", "USD"],
            "is_cross_currency": [0, 0],
            "payment_format": ["Wire", "Wire"],
            "payment_format_normalized": ["wire", "wire"],
            "is_laundering": [0, 0],
            "date": [pd.Timestamp("2024-01-01").date()] * 2,
            "hour": [10, 15],
            "day_of_week": [0, 0],
        })

        result = engine.evaluate(df)

        assert (result["rule_round_tripping_score"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Velocity Rule Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVelocityRules:
    """Tests for the transaction velocity anomaly detection rule."""

    def test_high_frequency_triggers(self, engine: AMLRuleEngine) -> None:
        """Many transactions from the same account in a short window should trigger."""
        n = 10  # 10 transactions within 1 hour from same account
        df = pd.DataFrame({
            "transaction_id": range(n),
            "timestamp": pd.date_range("2024-01-01 10:00", periods=n, freq="5min"),
            "from_id": ["HIGH_FREQ_ACCT"] * n,
            "from_bank": ["BANK_001"] * n,
            "from_account": ["HF"] * n,
            "to_id": [f"RECV_{i}" for i in range(n)],
            "to_bank": ["BANK_002"] * n,
            "to_account": [str(i) for i in range(n)],
            "amount": [1000.0] * n,
            "amount_paid": [1000.0] * n,
            "amount_received": [1000.0] * n,
            "payment_currency": ["USD"] * n,
            "receiving_currency": ["USD"] * n,
            "is_cross_currency": [0] * n,
            "payment_format": ["Wire"] * n,
            "payment_format_normalized": ["wire"] * n,
            "is_laundering": [0] * n,
            "date": [pd.Timestamp("2024-01-01").date()] * n,
            "hour": [10] * n,
            "day_of_week": [0] * n,
        })

        result = engine.evaluate(df)

        # With 10 transactions in < 1h, should exceed the 1h velocity threshold (5)
        assert result["rule_velocity_score"].max() > 0.0

    def test_low_frequency_does_not_trigger(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Normal transaction frequency should not trigger velocity rules."""
        df = base_transaction_df.copy()
        # Each transaction from a different account (1 per account)
        df["from_id"] = [f"ACCT_{i}" for i in range(len(df))]

        result = engine.evaluate(df)

        assert (result["rule_velocity_score"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Composite Score and General Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeScoring:
    """Tests for composite scoring and general rule engine behavior."""

    def test_composite_score_in_valid_range(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """Composite scores must always be in [0, 1]."""
        result = engine.evaluate(base_transaction_df)

        assert (result["rule_composite_score"] >= 0.0).all()
        assert (result["rule_composite_score"] <= 1.0).all()

    def test_all_rule_score_columns_present(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """All expected rule score columns should be in the output."""
        result = engine.evaluate(base_transaction_df)

        expected_cols = [
            "rule_structuring_score",
            "rule_rapid_movement_score",
            "rule_round_tripping_score",
            "rule_geographic_risk_score",
            "rule_dormant_reactivation_score",
            "rule_velocity_score",
            "rule_composite_score",
            "rule_flagged",
            "rule_trigger_count",
            "rule_max_score",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rule_flagged_binary(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """The rule_flagged column should be binary (0 or 1)."""
        result = engine.evaluate(base_transaction_df)

        assert set(result["rule_flagged"].unique()).issubset({0, 1})

    def test_trigger_count_non_negative(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """rule_trigger_count should be non-negative."""
        result = engine.evaluate(base_transaction_df)

        assert (result["rule_trigger_count"] >= 0).all()

    def test_evaluate_does_not_modify_input(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """The evaluate method should not modify the input dataframe."""
        df_copy = base_transaction_df.copy()
        original_columns = set(df_copy.columns)

        engine.evaluate(df_copy)

        assert set(df_copy.columns) == original_columns

    def test_get_rule_statistics(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """get_rule_statistics should return a dict with trigger counts."""
        engine.evaluate(base_transaction_df)
        stats = engine.get_rule_statistics()

        assert "rule_triggers" in stats
        assert "rule_weights" in stats
        assert isinstance(stats["rule_triggers"], dict)

    def test_empty_dataframe(self, engine: AMLRuleEngine) -> None:
        """Engine should handle empty dataframes gracefully."""
        df = pd.DataFrame(columns=[
            "transaction_id", "timestamp", "from_id", "from_bank", "from_account",
            "to_id", "to_bank", "to_account", "amount", "amount_paid",
            "amount_received", "payment_currency", "receiving_currency",
            "is_cross_currency", "payment_format", "payment_format_normalized",
            "is_laundering", "date", "hour", "day_of_week",
        ])

        result = engine.evaluate(df)

        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Single Transaction Evaluation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleTransactionEvaluation:
    """Tests for real-time single transaction evaluation."""

    def test_evaluate_single_returns_results(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """evaluate_single_transaction should return a list of RuleResult."""
        # Build account history first
        engine.evaluate(base_transaction_df)

        txn = {
            "transaction_id": 999,
            "timestamp": pd.Timestamp("2024-01-01 12:00"),
            "from_id": "BANK_001_10000",
            "from_bank": "BANK_001",
            "from_account": "10000",
            "to_id": "BANK_002_20000",
            "to_bank": "BANK_002",
            "to_account": "20000",
            "amount": 9500.0,
            "amount_paid": 9500.0,
            "amount_received": 9500.0,
            "payment_currency": "USD",
            "receiving_currency": "USD",
            "is_cross_currency": 0,
            "payment_format": "Wire",
            "payment_format_normalized": "wire",
            "is_laundering": 0,
            "date": pd.Timestamp("2024-01-01").date(),
            "hour": 12,
            "day_of_week": 0,
        }

        results = engine.evaluate_single_transaction(txn)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RuleResult) for r in results)

    def test_single_transaction_structuring_amount(self, engine: AMLRuleEngine, base_transaction_df: pd.DataFrame) -> None:
        """A structuring-range amount should have a positive structuring score."""
        engine.evaluate(base_transaction_df)

        txn = {
            "transaction_id": 999,
            "timestamp": pd.Timestamp("2024-01-01 12:00"),
            "from_id": "BANK_001_10000",
            "from_bank": "BANK_001",
            "from_account": "10000",
            "to_id": "BANK_002_20000",
            "to_bank": "BANK_002",
            "to_account": "20000",
            "amount": 9500.0,
            "amount_paid": 9500.0,
            "amount_received": 9500.0,
            "payment_currency": "USD",
            "receiving_currency": "USD",
            "is_cross_currency": 0,
            "payment_format": "Wire",
            "payment_format_normalized": "wire",
            "is_laundering": 0,
            "date": pd.Timestamp("2024-01-01").date(),
            "hour": 12,
            "day_of_week": 0,
        }

        results = engine.evaluate_single_transaction(txn)

        structuring_result = next((r for r in results if r.rule_id == "structuring"), None)
        assert structuring_result is not None
        assert structuring_result.risk_score > 0.0
        assert structuring_result.triggered is True
