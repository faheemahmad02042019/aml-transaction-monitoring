"""
AML Alert Generation and Enrichment.

Combines outputs from the rule engine, ML model, and graph analysis
into prioritized, enriched alerts for compliance review:

  - Ensemble scoring (weighted combination of layers)
  - Alert deduplication (same entity within a time window)
  - Alert enrichment (transaction history, counterparty info, graph context)
  - Severity classification (High, Medium, Low)
  - Standardized alert format for compliance workflows
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import AlertConfig, Config

logger = logging.getLogger(__name__)


class Alert:
    """
    Represents a single AML alert.

    Attributes
    ----------
    alert_id : str
        Unique identifier for the alert.
    entity_id : str
        Account or entity under investigation.
    severity : str
        Alert severity: "HIGH", "MEDIUM", or "LOW".
    ensemble_score : float
        Combined risk score from all detection layers.
    triggered_rules : List[str]
        List of rule IDs that triggered.
    ml_score : float
        ML model suspicion probability.
    graph_score : float
        Graph analysis suspicion score.
    transactions : List[Dict]
        Relevant transactions associated with this alert.
    context : Dict[str, Any]
        Enrichment context (counterparties, graph info, history).
    created_at : datetime
        Alert creation timestamp.
    sla_deadline : datetime
        SLA review deadline based on severity.
    status : str
        Alert status: "OPEN", "IN_REVIEW", "ESCALATED", "CLOSED".
    disposition : Optional[str]
        Final disposition: "SAR_FILED", "MONITORING", "CLOSED_NO_ACTION".
    """

    def __init__(
        self,
        entity_id: str,
        severity: str,
        ensemble_score: float,
        triggered_rules: List[str],
        ml_score: float,
        graph_score: float,
        transactions: List[Dict],
        context: Dict[str, Any],
        sla_hours: int,
    ) -> None:
        self.alert_id: str = str(uuid.uuid4())[:12].upper()
        self.entity_id: str = entity_id
        self.severity: str = severity
        self.ensemble_score: float = ensemble_score
        self.triggered_rules: List[str] = triggered_rules
        self.ml_score: float = ml_score
        self.graph_score: float = graph_score
        self.transactions: List[Dict] = transactions
        self.context: Dict[str, Any] = context
        self.created_at: datetime = datetime.utcnow()
        self.sla_deadline: datetime = self.created_at + timedelta(hours=sla_hours)
        self.status: str = "OPEN"
        self.disposition: Optional[str] = None
        self.llm_summary: Optional[str] = None
        self.llm_recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "entity_id": self.entity_id,
            "severity": self.severity,
            "ensemble_score": self.ensemble_score,
            "triggered_rules": self.triggered_rules,
            "ml_score": self.ml_score,
            "graph_score": self.graph_score,
            "transaction_count": len(self.transactions),
            "total_amount": sum(t.get("amount", 0) for t in self.transactions),
            "created_at": self.created_at.isoformat(),
            "sla_deadline": self.sla_deadline.isoformat(),
            "status": self.status,
            "disposition": self.disposition,
            "context_summary": {
                "n_counterparties": self.context.get("n_counterparties", 0),
                "graph_community": self.context.get("community_id", None),
                "in_cycle": self.context.get("in_cycle", False),
                "prior_alerts": self.context.get("prior_alert_count", 0),
            },
            "llm_summary": self.llm_summary,
            "llm_recommendation": self.llm_recommendation,
        }


class AMLAlertGenerator:
    """
    Generates, deduplicates, enriches, and prioritizes AML alerts.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.alert_config: AlertConfig = config.alert
        self._alerts: List[Alert] = []
        self._alert_stats: Dict[str, int] = {
            "total_generated": 0,
            "deduplicated": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    def generate_alerts(
        self,
        df: pd.DataFrame,
        ml_scores: Optional[np.ndarray] = None,
        graph_features: Optional[pd.DataFrame] = None,
        graph_suspicious_accounts: Optional[pd.DataFrame] = None,
    ) -> List[Alert]:
        """
        Generate alerts from scored transaction data.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with rule engine scores.
        ml_scores : np.ndarray, optional
            ML model probability scores.
        graph_features : pd.DataFrame, optional
            Graph-derived features per transaction.
        graph_suspicious_accounts : pd.DataFrame, optional
            Ranked suspicious accounts from graph analysis.

        Returns
        -------
        List[Alert]
            Generated and prioritized alerts.
        """
        logger.info(f"Generating alerts from {len(df):,} transactions")

        # Step 1: Compute ensemble scores
        df = self._compute_ensemble_scores(df, ml_scores, graph_features)

        # Step 2: Identify alert-worthy transactions
        alert_candidates = df[df["ensemble_score"] >= self.alert_config.alert_threshold].copy()
        logger.info(f"  Alert candidates (ensemble >= {self.alert_config.alert_threshold}): {len(alert_candidates):,}")

        if alert_candidates.empty:
            logger.warning("No transactions meet the alert threshold.")
            return []

        # Step 3: Group by entity and aggregate
        entity_alerts = self._group_by_entity(alert_candidates, df)

        # Step 4: Enrich alerts
        enriched_alerts = self._enrich_alerts(
            entity_alerts, df, graph_features, graph_suspicious_accounts
        )

        # Step 5: Deduplicate
        deduplicated = self._deduplicate_alerts(enriched_alerts)

        # Step 6: Classify severity and assign SLA
        final_alerts = self._classify_severity(deduplicated)

        # Step 7: Sort by priority
        final_alerts.sort(key=lambda a: a.ensemble_score, reverse=True)

        self._alerts = final_alerts
        self._update_stats()

        logger.info(
            f"Alerts generated: {len(final_alerts)} | "
            f"HIGH: {self._alert_stats['high']} | "
            f"MEDIUM: {self._alert_stats['medium']} | "
            f"LOW: {self._alert_stats['low']}"
        )

        return final_alerts

    def _compute_ensemble_scores(
        self,
        df: pd.DataFrame,
        ml_scores: Optional[np.ndarray],
        graph_features: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute weighted ensemble scores from all detection layers."""
        df = df.copy()
        cfg = self.alert_config

        # Rule engine score
        rule_score = df.get("rule_composite_score", pd.Series(0, index=df.index))

        # ML score
        if ml_scores is not None:
            ml_score = pd.Series(ml_scores, index=df.index)
        else:
            ml_score = pd.Series(0, index=df.index)

        # Graph score (aggregate of graph features into a single score)
        if graph_features is not None and not graph_features.empty:
            graph_score_cols = [
                c for c in graph_features.columns
                if any(kw in c for kw in ["betweenness", "pagerank", "in_cycle"])
            ]
            if graph_score_cols:
                # Normalize and combine graph signals
                graph_score = graph_features[graph_score_cols].fillna(0).mean(axis=1)
                # Clip to [0, 1]
                graph_score = graph_score.clip(0, 1)
            else:
                graph_score = pd.Series(0, index=df.index)
        else:
            graph_score = pd.Series(0, index=df.index)

        # Weighted ensemble
        df["ensemble_score"] = (
            cfg.rule_score_weight * rule_score
            + cfg.ml_score_weight * ml_score
            + cfg.graph_score_weight * graph_score
        ).clip(0, 1)

        df["ml_score"] = ml_score
        df["graph_score"] = graph_score

        return df

    def _group_by_entity(
        self, candidates: pd.DataFrame, full_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Group alert candidates by entity (account)."""
        entity_groups = []

        for entity_id, group in candidates.groupby("from_id"):
            # Get all transactions for this entity (not just flagged ones)
            entity_txns = full_df[
                (full_df["from_id"] == entity_id) | (full_df["to_id"] == entity_id)
            ].sort_values("timestamp")

            # Identify which rules triggered
            rule_cols = [c for c in group.columns if c.startswith("rule_") and c.endswith("_score")]
            triggered_rules = []
            for col in rule_cols:
                if group[col].max() > 0:
                    rule_name = col.replace("rule_", "").replace("_score", "")
                    triggered_rules.append(rule_name)

            entity_groups.append({
                "entity_id": entity_id,
                "flagged_txns": group.to_dict("records"),
                "all_txns": entity_txns.tail(50).to_dict("records"),  # Last 50 transactions
                "max_ensemble_score": float(group["ensemble_score"].max()),
                "mean_ensemble_score": float(group["ensemble_score"].mean()),
                "max_ml_score": float(group["ml_score"].max()) if "ml_score" in group.columns else 0.0,
                "max_graph_score": float(group["graph_score"].max()) if "graph_score" in group.columns else 0.0,
                "triggered_rules": triggered_rules,
                "n_flagged_txns": len(group),
                "total_flagged_amount": float(group["amount"].sum()),
                "date_range": {
                    "start": str(group["timestamp"].min()),
                    "end": str(group["timestamp"].max()),
                },
            })

        return entity_groups

    def _enrich_alerts(
        self,
        entity_alerts: List[Dict[str, Any]],
        df: pd.DataFrame,
        graph_features: Optional[pd.DataFrame],
        graph_suspicious_accounts: Optional[pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """Enrich alerts with additional context for investigation."""
        enriched = []

        # Build counterparty map
        counterparty_map: Dict[str, set] = {}
        for _, row in df.iterrows():
            counterparty_map.setdefault(row["from_id"], set()).add(row["to_id"])
            counterparty_map.setdefault(row["to_id"], set()).add(row["from_id"])

        # Build graph suspicious account lookup
        graph_suspicion: Dict[str, float] = {}
        if graph_suspicious_accounts is not None and not graph_suspicious_accounts.empty:
            graph_suspicion = dict(zip(
                graph_suspicious_accounts["account_id"],
                graph_suspicious_accounts["graph_suspicion_score"],
            ))

        for alert_data in entity_alerts:
            entity_id = alert_data["entity_id"]

            # Counterparty info
            counterparties = counterparty_map.get(entity_id, set())
            suspicious_counterparties = [
                cp for cp in counterparties if graph_suspicion.get(cp, 0) > 0.3
            ]

            # Graph context
            graph_context = {
                "graph_suspicion_score": graph_suspicion.get(entity_id, 0),
                "in_cycle": graph_suspicion.get(entity_id, 0) > 0.5,
                "community_id": None,  # Would be populated from graph analysis
            }

            # Transaction pattern summary
            flagged_txns = alert_data["flagged_txns"]
            amounts = [t.get("amount", 0) for t in flagged_txns]

            context = {
                "n_counterparties": len(counterparties),
                "suspicious_counterparties": suspicious_counterparties[:10],
                "prior_alert_count": 0,  # Would query alert history in production
                "transaction_pattern": {
                    "total_amount": sum(amounts),
                    "mean_amount": float(np.mean(amounts)) if amounts else 0,
                    "max_amount": float(max(amounts)) if amounts else 0,
                    "min_amount": float(min(amounts)) if amounts else 0,
                    "n_transactions": len(flagged_txns),
                },
                **graph_context,
            }

            alert_data["context"] = context
            enriched.append(alert_data)

        return enriched

    def _deduplicate_alerts(
        self, alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate alerts for the same entity within a time window.

        When multiple alerts exist for the same entity within the dedup
        window, they are merged according to the configured strategy.
        """
        if not alerts:
            return []

        cfg = self.alert_config
        seen_entities: Dict[str, Dict[str, Any]] = {}
        deduplicated: List[Dict[str, Any]] = []

        for alert in alerts:
            entity_id = alert["entity_id"]

            if entity_id in seen_entities:
                existing = seen_entities[entity_id]
                # Merge strategy
                if cfg.dedup_merge_strategy == "max_score":
                    if alert["max_ensemble_score"] > existing["max_ensemble_score"]:
                        # Replace with higher-scoring alert
                        seen_entities[entity_id] = alert
                elif cfg.dedup_merge_strategy == "sum_score":
                    # Combine scores (capped at 1.0)
                    existing["max_ensemble_score"] = min(
                        1.0,
                        existing["max_ensemble_score"] + alert["max_ensemble_score"] * 0.3,
                    )
                    existing["triggered_rules"] = list(
                        set(existing["triggered_rules"]) | set(alert["triggered_rules"])
                    )
                    existing["n_flagged_txns"] += alert["n_flagged_txns"]
            else:
                seen_entities[entity_id] = alert

        n_before = len(alerts)
        deduplicated = list(seen_entities.values())
        n_deduped = n_before - len(deduplicated)

        if n_deduped > 0:
            logger.info(f"  Deduplicated {n_deduped} alerts ({n_before} -> {len(deduplicated)})")

        self._alert_stats["deduplicated"] = n_deduped

        return deduplicated

    def _classify_severity(self, alerts: List[Dict[str, Any]]) -> List[Alert]:
        """Classify alert severity and create Alert objects."""
        cfg = self.alert_config
        alert_objects: List[Alert] = []

        for alert_data in alerts:
            score = alert_data["max_ensemble_score"]

            if score >= cfg.high_severity_threshold:
                severity = "HIGH"
                sla_hours = cfg.high_severity_sla_hours
            elif score >= cfg.medium_severity_threshold:
                severity = "MEDIUM"
                sla_hours = cfg.medium_severity_sla_hours
            else:
                severity = "LOW"
                sla_hours = cfg.low_severity_sla_hours

            alert = Alert(
                entity_id=alert_data["entity_id"],
                severity=severity,
                ensemble_score=score,
                triggered_rules=alert_data["triggered_rules"],
                ml_score=alert_data.get("max_ml_score", 0),
                graph_score=alert_data.get("max_graph_score", 0),
                transactions=alert_data.get("flagged_txns", []),
                context=alert_data.get("context", {}),
                sla_hours=sla_hours,
            )

            alert_objects.append(alert)

        return alert_objects

    def _update_stats(self) -> None:
        """Update alert statistics."""
        self._alert_stats["total_generated"] = len(self._alerts)
        self._alert_stats["high"] = sum(1 for a in self._alerts if a.severity == "HIGH")
        self._alert_stats["medium"] = sum(1 for a in self._alerts if a.severity == "MEDIUM")
        self._alert_stats["low"] = sum(1 for a in self._alerts if a.severity == "LOW")

    def get_alert_queue(
        self,
        severity_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get the prioritized alert queue.

        Parameters
        ----------
        severity_filter : str, optional
            Filter by severity ("HIGH", "MEDIUM", "LOW").
        status_filter : str, optional
            Filter by status ("OPEN", "IN_REVIEW", "ESCALATED", "CLOSED").
        limit : int
            Maximum number of alerts to return.

        Returns
        -------
        List[Dict[str, Any]]
            Serialized alert dictionaries.
        """
        filtered = self._alerts

        if severity_filter:
            filtered = [a for a in filtered if a.severity == severity_filter.upper()]

        if status_filter:
            filtered = [a for a in filtered if a.status == status_filter.upper()]

        # Sort by severity (HIGH first), then by score
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        filtered.sort(key=lambda a: (severity_order.get(a.severity, 3), -a.ensemble_score))

        return [a.to_dict() for a in filtered[:limit]]

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Look up an alert by its ID."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                return alert
        return None

    def update_alert_status(
        self, alert_id: str, status: str, disposition: Optional[str] = None
    ) -> bool:
        """Update the status of an alert."""
        alert = self.get_alert_by_id(alert_id)
        if alert is None:
            logger.warning(f"Alert not found: {alert_id}")
            return False

        valid_statuses = {"OPEN", "IN_REVIEW", "ESCALATED", "CLOSED"}
        if status.upper() not in valid_statuses:
            logger.warning(f"Invalid status: {status}")
            return False

        alert.status = status.upper()
        if disposition:
            alert.disposition = disposition

        logger.info(f"Alert {alert_id} updated: status={alert.status}, disposition={alert.disposition}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        if not self._alerts:
            return {"total": 0}

        scores = [a.ensemble_score for a in self._alerts]
        return {
            "total": len(self._alerts),
            "by_severity": self._alert_stats.copy(),
            "by_status": {
                status: sum(1 for a in self._alerts if a.status == status)
                for status in ["OPEN", "IN_REVIEW", "ESCALATED", "CLOSED"]
            },
            "score_distribution": {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            },
            "rules_triggered": {
                rule: sum(1 for a in self._alerts if rule in a.triggered_rules)
                for rule in set(
                    r for a in self._alerts for r in a.triggered_rules
                )
            },
            "deduplicated": self._alert_stats.get("deduplicated", 0),
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all alerts to a Pandas DataFrame."""
        return pd.DataFrame([a.to_dict() for a in self._alerts])
