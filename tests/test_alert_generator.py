"""
Unit tests for AML Alert Generation and Enrichment.

Tests:
  - Ensemble scoring computation
  - Alert generation from scored transactions
  - Alert deduplication
  - Severity classification (HIGH, MEDIUM, LOW)
  - Alert enrichment with context
  - Alert queue retrieval and filtering
  - Alert status updates
  - Alert statistics
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.alert_generator import Alert, AMLAlertGenerator
from src.config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config() -> Config:
    """Return a default Config object for testing."""
    return Config()


@pytest.fixture
def generator(config: Config) -> AMLAlertGenerator:
    """Return an AMLAlertGenerator instance."""
    return AMLAlertGenerator(config)


@pytest.fixture
def scored_transaction_df() -> pd.DataFrame:
    """
    Return a transaction DataFrame with rule engine scores.

    Contains a mix of flagged and unflagged transactions.
    """
    rng = np.random.RandomState(42)
    n = 50

    df = pd.DataFrame({
        "transaction_id": range(n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "from_id": [f"ACCT_{i % 10:03d}" for i in range(n)],
        "from_bank": [f"BANK_{i % 3:03d}" for i in range(n)],
        "from_account": [str(1000 + i % 10) for i in range(n)],
        "to_id": [f"ACCT_{(i + 5) % 10:03d}" for i in range(n)],
        "to_bank": [f"BANK_{(i + 1) % 3:03d}" for i in range(n)],
        "to_account": [str(2000 + (i + 5) % 10) for i in range(n)],
        "amount": rng.lognormal(8, 1.5, n).round(2),
        "payment_currency": ["USD"] * n,
        "receiving_currency": ["USD"] * n,
        "is_cross_currency": [0] * n,
        "payment_format": rng.choice(["Wire", "ACH", "Check"], n).tolist(),
        "payment_format_normalized": rng.choice(["wire", "ach", "check"], n).tolist(),
        "is_laundering": [0] * n,
        "date": pd.date_range("2024-01-01", periods=n, freq="h").date,
        "hour": [(i % 24) for i in range(n)],
        "day_of_week": [(i % 7) for i in range(n)],
        # Rule engine scores
        "rule_structuring_score": rng.choice([0.0, 0.3, 0.6, 0.8], n, p=[0.7, 0.15, 0.1, 0.05]),
        "rule_rapid_movement_score": rng.choice([0.0, 0.4, 0.7], n, p=[0.8, 0.15, 0.05]),
        "rule_round_tripping_score": rng.choice([0.0, 0.5, 0.9], n, p=[0.85, 0.1, 0.05]),
        "rule_geographic_risk_score": rng.choice([0.0, 0.2], n, p=[0.9, 0.1]),
        "rule_dormant_reactivation_score": [0.0] * n,
        "rule_velocity_score": rng.choice([0.0, 0.3], n, p=[0.9, 0.1]),
        "rule_composite_score": np.zeros(n),
        "rule_flagged": np.zeros(n, dtype=int),
        "rule_trigger_count": np.zeros(n, dtype=int),
        "rule_max_score": np.zeros(n),
    })

    # Compute composite score
    rule_cols = [c for c in df.columns if c.startswith("rule_") and c.endswith("_score") and c != "rule_composite_score" and c != "rule_max_score"]
    df["rule_composite_score"] = df[rule_cols].mean(axis=1)
    df["rule_flagged"] = (df["rule_composite_score"] > 0.3).astype(int)
    df["rule_trigger_count"] = (df[rule_cols] > 0).sum(axis=1)
    df["rule_max_score"] = df[rule_cols].max(axis=1)

    return df


@pytest.fixture
def ml_scores(scored_transaction_df: pd.DataFrame) -> np.ndarray:
    """Generate ML probability scores for the scored transactions."""
    rng = np.random.RandomState(42)
    return rng.beta(2, 5, len(scored_transaction_df))


@pytest.fixture
def sample_alert() -> Alert:
    """Return a sample Alert object for testing."""
    return Alert(
        entity_id="ACCT_001",
        severity="HIGH",
        ensemble_score=0.85,
        triggered_rules=["structuring", "rapid_movement"],
        ml_score=0.75,
        graph_score=0.60,
        transactions=[
            {"from_id": "ACCT_001", "to_id": "ACCT_005", "amount": 9500.0,
             "timestamp": "2024-01-01 10:00", "payment_format": "Wire"},
            {"from_id": "ACCT_001", "to_id": "ACCT_007", "amount": 8800.0,
             "timestamp": "2024-01-01 11:00", "payment_format": "Wire"},
        ],
        context={
            "n_counterparties": 5,
            "suspicious_counterparties": ["ACCT_005"],
            "prior_alert_count": 1,
            "in_cycle": False,
            "community_id": 3,
            "transaction_pattern": {
                "total_amount": 18300.0,
                "mean_amount": 9150.0,
                "max_amount": 9500.0,
                "n_transactions": 2,
            },
        },
        sla_hours=24,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alert Object Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertObject:
    """Tests for the Alert data class."""

    def test_alert_creation(self, sample_alert: Alert) -> None:
        """Alert should be created with all required fields."""
        assert sample_alert.entity_id == "ACCT_001"
        assert sample_alert.severity == "HIGH"
        assert sample_alert.ensemble_score == 0.85
        assert len(sample_alert.triggered_rules) == 2
        assert sample_alert.status == "OPEN"
        assert sample_alert.disposition is None

    def test_alert_id_generated(self, sample_alert: Alert) -> None:
        """Alert should have a unique auto-generated ID."""
        assert sample_alert.alert_id is not None
        assert len(sample_alert.alert_id) > 0

    def test_sla_deadline_set(self, sample_alert: Alert) -> None:
        """SLA deadline should be set based on creation time + sla_hours."""
        expected_deadline = sample_alert.created_at + timedelta(hours=24)
        # Allow small time delta for test execution
        diff = abs((sample_alert.sla_deadline - expected_deadline).total_seconds())
        assert diff < 1.0

    def test_to_dict(self, sample_alert: Alert) -> None:
        """to_dict should return a complete serialized dictionary."""
        d = sample_alert.to_dict()

        assert d["alert_id"] == sample_alert.alert_id
        assert d["entity_id"] == "ACCT_001"
        assert d["severity"] == "HIGH"
        assert d["ensemble_score"] == 0.85
        assert d["transaction_count"] == 2
        assert d["total_amount"] == 18300.0
        assert d["status"] == "OPEN"
        assert d["disposition"] is None
        assert isinstance(d["created_at"], str)
        assert isinstance(d["sla_deadline"], str)

    def test_unique_alert_ids(self) -> None:
        """Different alerts should have different IDs."""
        alerts = [
            Alert("E1", "HIGH", 0.9, [], 0.8, 0.7, [], {}, 24)
            for _ in range(10)
        ]
        ids = [a.alert_id for a in alerts]
        assert len(set(ids)) == len(ids), "Alert IDs should be unique"


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleScoring:
    """Tests for ensemble score computation."""

    def test_ensemble_score_in_valid_range(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Ensemble scores should be in [0, 1]."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        for alert in alerts:
            assert 0.0 <= alert.ensemble_score <= 1.0

    def test_ensemble_without_ml_scores(
        self, generator: AMLAlertGenerator, scored_transaction_df: pd.DataFrame
    ) -> None:
        """Alert generation should work with only rule engine scores."""
        alerts = generator.generate_alerts(scored_transaction_df)

        # Should still generate alerts (or empty list if no scores exceed threshold)
        assert isinstance(alerts, list)

    def test_higher_scores_produce_higher_ensemble(
        self, generator: AMLAlertGenerator
    ) -> None:
        """Transactions with higher component scores should have higher ensemble scores."""
        n = 2
        df = pd.DataFrame({
            "transaction_id": range(n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "from_id": ["HIGH_RISK", "LOW_RISK"],
            "from_bank": ["BANK_001", "BANK_002"],
            "from_account": ["HR", "LR"],
            "to_id": ["RECV_1", "RECV_2"],
            "to_bank": ["BANK_003", "BANK_004"],
            "to_account": ["R1", "R2"],
            "amount": [10000.0, 100.0],
            "payment_currency": ["USD", "USD"],
            "receiving_currency": ["USD", "USD"],
            "is_cross_currency": [0, 0],
            "payment_format": ["Wire", "Wire"],
            "payment_format_normalized": ["wire", "wire"],
            "is_laundering": [0, 0],
            "date": pd.date_range("2024-01-01", periods=n, freq="h").date,
            "hour": [10, 11],
            "day_of_week": [0, 0],
            "rule_composite_score": [0.9, 0.1],
            "rule_structuring_score": [0.8, 0.0],
            "rule_rapid_movement_score": [0.7, 0.0],
        })

        ml_scores = np.array([0.95, 0.05])
        alerts = generator.generate_alerts(df, ml_scores=ml_scores)

        if len(alerts) >= 1:
            # The high-risk entity should have a higher score
            assert alerts[0].ensemble_score > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeduplication:
    """Tests for alert deduplication."""

    def test_same_entity_deduplicated(
        self, generator: AMLAlertGenerator
    ) -> None:
        """Multiple alerts for the same entity should be deduplicated."""
        n = 10
        df = pd.DataFrame({
            "transaction_id": range(n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "from_id": ["SAME_ACCT"] * n,
            "from_bank": ["BANK_001"] * n,
            "from_account": ["SA"] * n,
            "to_id": [f"RECV_{i}" for i in range(n)],
            "to_bank": ["BANK_002"] * n,
            "to_account": [str(i) for i in range(n)],
            "amount": [9500.0] * n,
            "payment_currency": ["USD"] * n,
            "receiving_currency": ["USD"] * n,
            "is_cross_currency": [0] * n,
            "payment_format": ["Wire"] * n,
            "payment_format_normalized": ["wire"] * n,
            "is_laundering": [0] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="h").date,
            "hour": list(range(n)),
            "day_of_week": [0] * n,
            "rule_composite_score": [0.8] * n,
            "rule_structuring_score": [0.8] * n,
        })

        ml_scores = np.array([0.9] * n)
        alerts = generator.generate_alerts(df, ml_scores=ml_scores)

        # Should produce at most 1 alert for SAME_ACCT (deduplicated)
        entity_ids = [a.entity_id for a in alerts]
        assert entity_ids.count("SAME_ACCT") <= 1

    def test_different_entities_not_deduplicated(
        self, generator: AMLAlertGenerator, scored_transaction_df: pd.DataFrame, ml_scores: np.ndarray
    ) -> None:
        """Alerts for different entities should not be merged."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        entity_ids = [a.entity_id for a in alerts]
        # Entity IDs should be unique after dedup
        assert len(entity_ids) == len(set(entity_ids))


# ─────────────────────────────────────────────────────────────────────────────
# Severity Classification Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityClassification:
    """Tests for alert severity classification."""

    def test_high_severity_threshold(self, config: Config) -> None:
        """Ensemble score >= 0.80 should produce HIGH severity."""
        alert = Alert("E1", "PENDING", 0.85, [], 0.8, 0.7, [], {}, 24)
        # Severity is set by the generator, but we test the threshold logic
        assert config.alert.high_severity_threshold == 0.80

    def test_medium_severity_threshold(self, config: Config) -> None:
        """Ensemble score >= 0.60 and < 0.80 should be MEDIUM."""
        assert config.alert.medium_severity_threshold == 0.60

    def test_severity_assigned_by_generator(
        self, generator: AMLAlertGenerator
    ) -> None:
        """Generator should assign correct severity based on score."""
        n = 3
        df = pd.DataFrame({
            "transaction_id": range(n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "from_id": ["HIGH_ENT", "MED_ENT", "LOW_ENT"],
            "from_bank": ["B1", "B2", "B3"],
            "from_account": ["H", "M", "L"],
            "to_id": ["R1", "R2", "R3"],
            "to_bank": ["B4", "B5", "B6"],
            "to_account": ["1", "2", "3"],
            "amount": [50000.0, 20000.0, 5000.0],
            "payment_currency": ["USD"] * n,
            "receiving_currency": ["USD"] * n,
            "is_cross_currency": [0] * n,
            "payment_format": ["Wire"] * n,
            "payment_format_normalized": ["wire"] * n,
            "is_laundering": [0] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="h").date,
            "hour": [10, 11, 12],
            "day_of_week": [0] * n,
            "rule_composite_score": [0.9, 0.7, 0.5],
            "rule_structuring_score": [0.9, 0.5, 0.3],
        })

        ml_scores = np.array([0.95, 0.65, 0.55])
        alerts = generator.generate_alerts(df, ml_scores=ml_scores)

        if alerts:
            severities = {a.entity_id: a.severity for a in alerts}
            # HIGH_ENT should be HIGH (score ~0.9)
            if "HIGH_ENT" in severities:
                assert severities["HIGH_ENT"] == "HIGH"

    def test_valid_severity_values(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """All alerts should have a valid severity."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        valid_severities = {"HIGH", "MEDIUM", "LOW"}
        for alert in alerts:
            assert alert.severity in valid_severities


# ─────────────────────────────────────────────────────────────────────────────
# Alert Enrichment Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertEnrichment:
    """Tests for alert context enrichment."""

    def test_alerts_have_context(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Generated alerts should have enrichment context."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        for alert in alerts:
            assert isinstance(alert.context, dict)
            assert "n_counterparties" in alert.context

    def test_alerts_have_transactions(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Alerts should contain associated transactions."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        for alert in alerts:
            assert isinstance(alert.transactions, list)
            # Should have at least one transaction
            assert len(alert.transactions) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Alert Queue and Management Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertQueue:
    """Tests for alert queue retrieval and management."""

    def test_get_alert_queue(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """get_alert_queue should return serialized alert dictionaries."""
        generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)
        queue = generator.get_alert_queue()

        assert isinstance(queue, list)
        if queue:
            assert isinstance(queue[0], dict)
            assert "alert_id" in queue[0]

    def test_queue_severity_filter(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Severity filter should only return matching alerts."""
        generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)
        queue = generator.get_alert_queue(severity_filter="HIGH")

        for alert_dict in queue:
            assert alert_dict["severity"] == "HIGH"

    def test_queue_limit(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Queue limit should cap the number of returned alerts."""
        generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)
        queue = generator.get_alert_queue(limit=3)

        assert len(queue) <= 3

    def test_get_alert_by_id(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Should retrieve an alert by its ID."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        if alerts:
            found = generator.get_alert_by_id(alerts[0].alert_id)
            assert found is not None
            assert found.alert_id == alerts[0].alert_id

    def test_get_nonexistent_alert(self, generator: AMLAlertGenerator) -> None:
        """Looking up a nonexistent alert ID should return None."""
        result = generator.get_alert_by_id("NONEXISTENT")
        assert result is None

    def test_update_alert_status(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Should update alert status and disposition."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        if alerts:
            alert_id = alerts[0].alert_id
            success = generator.update_alert_status(
                alert_id, "ESCALATED", disposition="SAR_FILED"
            )
            assert success is True

            updated = generator.get_alert_by_id(alert_id)
            assert updated.status == "ESCALATED"
            assert updated.disposition == "SAR_FILED"

    def test_update_invalid_status(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """Updating with an invalid status should fail."""
        alerts = generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)

        if alerts:
            success = generator.update_alert_status(alerts[0].alert_id, "INVALID_STATUS")
            assert success is False


# ─────────────────────────────────────────────────────────────────────────────
# Alert Statistics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertStatistics:
    """Tests for alert statistics and export."""

    def test_statistics_structure(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """get_statistics should return a well-structured dict."""
        generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)
        stats = generator.get_statistics()

        assert "total" in stats
        assert "by_severity" in stats
        assert "by_status" in stats
        assert "score_distribution" in stats

    def test_empty_statistics(self, generator: AMLAlertGenerator) -> None:
        """Statistics for no alerts should return total=0."""
        stats = generator.get_statistics()
        assert stats["total"] == 0

    def test_export_to_dataframe(
        self,
        generator: AMLAlertGenerator,
        scored_transaction_df: pd.DataFrame,
        ml_scores: np.ndarray,
    ) -> None:
        """export_to_dataframe should return a pandas DataFrame."""
        generator.generate_alerts(scored_transaction_df, ml_scores=ml_scores)
        df = generator.export_to_dataframe()

        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "alert_id" in df.columns
            assert "entity_id" in df.columns
            assert "severity" in df.columns

    def test_no_alerts_below_threshold(self, generator: AMLAlertGenerator) -> None:
        """Transactions below the alert threshold should not generate alerts."""
        n = 5
        df = pd.DataFrame({
            "transaction_id": range(n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "from_id": [f"ACCT_{i}" for i in range(n)],
            "from_bank": ["BANK_001"] * n,
            "from_account": [str(i) for i in range(n)],
            "to_id": [f"RECV_{i}" for i in range(n)],
            "to_bank": ["BANK_002"] * n,
            "to_account": [str(100 + i) for i in range(n)],
            "amount": [100.0] * n,
            "payment_currency": ["USD"] * n,
            "receiving_currency": ["USD"] * n,
            "is_cross_currency": [0] * n,
            "payment_format": ["Wire"] * n,
            "payment_format_normalized": ["wire"] * n,
            "is_laundering": [0] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="h").date,
            "hour": list(range(n)),
            "day_of_week": [0] * n,
            "rule_composite_score": [0.1] * n,  # Below threshold
        })

        ml_scores = np.array([0.1] * n)  # Low ML scores
        alerts = generator.generate_alerts(df, ml_scores=ml_scores)

        # All ensemble scores should be below 0.50 threshold
        assert len(alerts) == 0
