"""
Rule-Based AML Transaction Monitoring Engine (Layer 1).

Implements configurable AML detection rules aligned with FinCEN typologies:
  - Structuring (smurfing)
  - Rapid movement of funds (layering)
  - Round-tripping (circular fund flows)
  - Unusual geographic patterns (high-risk jurisdictions)
  - Dormant account reactivation
  - Velocity anomalies

Each rule produces a risk score in [0, 1] and a rule identifier. Scores are
aggregated into a composite rule-based risk score for downstream ML and
alert generation.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config, RuleEngineConfig

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of a single rule evaluation for a transaction."""

    rule_id: str
    rule_name: str
    risk_score: float  # 0.0 to 1.0
    triggered: bool
    details: Dict[str, Any]


class AMLRuleEngine:
    """
    Rule-based AML monitoring engine.

    Evaluates each transaction against a battery of AML detection rules,
    producing per-rule risk scores and a composite score.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.rule_config: RuleEngineConfig = config.rules
        self._account_history: Optional[pd.DataFrame] = None
        self._rule_stats: Dict[str, int] = {}

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all AML rules on a transaction dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with columns: timestamp, from_id, to_id, amount,
            payment_currency, payment_format, etc.

        Returns
        -------
        pd.DataFrame
            Original data augmented with rule scores and composite risk score.
        """
        logger.info(f"Evaluating AML rules on {len(df):,} transactions")
        df = df.copy()

        # Precompute account-level histories for rules that need them
        self._build_account_history(df)

        # Evaluate each rule
        rule_outputs = {
            "structuring": self._rule_structuring(df),
            "rapid_movement": self._rule_rapid_movement(df),
            "round_tripping": self._rule_round_tripping(df),
            "geographic_risk": self._rule_geographic_risk(df),
            "dormant_reactivation": self._rule_dormant_reactivation(df),
            "velocity": self._rule_velocity(df),
        }

        # Attach per-rule scores to the dataframe
        for rule_name, scores in rule_outputs.items():
            col_name = f"rule_{rule_name}_score"
            df[col_name] = scores
            triggered_count = (scores > 0).sum()
            self._rule_stats[rule_name] = int(triggered_count)
            logger.info(f"  Rule '{rule_name}': {triggered_count:,} triggers ({triggered_count / len(df) * 100:.3f}%)")

        # Compute composite rule score (weighted average)
        df["rule_composite_score"] = self._compute_composite_score(df, rule_outputs)

        # Binary rule flag at a threshold
        df["rule_flagged"] = (df["rule_composite_score"] > 0.3).astype(int)

        # Count how many rules triggered per transaction
        rule_score_cols = [f"rule_{name}_score" for name in rule_outputs]
        df["rule_trigger_count"] = (df[rule_score_cols] > 0).sum(axis=1)

        # Maximum single-rule score
        df["rule_max_score"] = df[rule_score_cols].max(axis=1)

        logger.info(
            f"Rule evaluation complete | "
            f"Flagged: {df['rule_flagged'].sum():,} ({df['rule_flagged'].mean() * 100:.2f}%) | "
            f"Mean composite: {df['rule_composite_score'].mean():.4f}"
        )

        return df

    def _build_account_history(self, df: pd.DataFrame) -> None:
        """Build per-account transaction history for rule evaluation."""
        logger.info("Building account transaction history for rule evaluation")

        # Outgoing history
        out_history = (
            df.groupby("from_id")
            .agg(
                out_txn_count=("transaction_id", "count"),
                out_total_amount=("amount", "sum"),
                out_mean_amount=("amount", "mean"),
                out_std_amount=("amount", "std"),
                out_max_amount=("amount", "max"),
                first_txn=("timestamp", "min"),
                last_txn=("timestamp", "max"),
            )
            .reset_index()
            .rename(columns={"from_id": "account_id"})
        )

        # Incoming history
        in_history = (
            df.groupby("to_id")
            .agg(
                in_txn_count=("transaction_id", "count"),
                in_total_amount=("amount", "sum"),
                in_mean_amount=("amount", "mean"),
                in_max_amount=("amount", "max"),
            )
            .reset_index()
            .rename(columns={"to_id": "account_id"})
        )

        self._account_history = out_history.merge(in_history, on="account_id", how="outer").fillna(0)

    def _rule_structuring(self, df: pd.DataFrame) -> np.ndarray:
        """
        Structuring / Smurfing Detection.

        Identifies transactions deliberately structured below the Currency
        Transaction Report (CTR) threshold of $10,000. Key indicators:
          - Transaction amounts in the $8,000-$9,999 range
          - Multiple such transactions from the same account within 24 hours
          - Amounts just below round-number thresholds

        Returns
        -------
        np.ndarray
            Risk scores in [0, 1] for each transaction.
        """
        scores = np.zeros(len(df), dtype=np.float64)
        cfg = self.rule_config

        # Flag 1: Amount in structuring range
        in_range = (df["amount"] >= cfg.structuring_lower_bound) & (df["amount"] <= cfg.structuring_upper_bound)
        scores[in_range.values] += 0.4

        # Flag 2: Amount suspiciously close to threshold (within 5%)
        close_to_threshold = (
            (df["amount"] >= cfg.ctr_threshold * 0.90) &
            (df["amount"] < cfg.ctr_threshold)
        )
        scores[close_to_threshold.values] += 0.2

        # Flag 3: Multiple transactions in structuring range from same account within window
        if in_range.any():
            structuring_txns = df[in_range].copy()
            structuring_txns = structuring_txns.sort_values(["from_id", "timestamp"])

            for account_id, group in structuring_txns.groupby("from_id"):
                if len(group) < cfg.structuring_min_count:
                    continue

                # Check for transactions within the structuring window
                timestamps = group["timestamp"].values
                for i in range(len(timestamps)):
                    window_end = timestamps[i] + np.timedelta64(cfg.structuring_window_hours, "h")
                    count_in_window = ((timestamps >= timestamps[i]) & (timestamps <= window_end)).sum()

                    if count_in_window >= cfg.structuring_min_count:
                        idx = group.index[i]
                        # Boost score based on count
                        boost = min(0.4, 0.1 * count_in_window)
                        if idx < len(scores):
                            scores[idx] += boost

        # Flag 4: Round amount just below a threshold (e.g., $9,900, $4,950)
        for threshold in [10_000, 5_000, 3_000]:
            near_threshold = (
                (df["amount"] >= threshold * 0.95) &
                (df["amount"] < threshold) &
                (df["amount"] % 100 < 10)  # Suspiciously round
            )
            scores[near_threshold.values] += 0.1

        return np.clip(scores, 0.0, 1.0)

    def _rule_rapid_movement(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rapid Movement of Funds Detection.

        Identifies accounts that receive large inflows and quickly move them
        out, characteristic of layering in money laundering.

        Returns
        -------
        np.ndarray
            Risk scores in [0, 1] for each transaction.
        """
        scores = np.zeros(len(df), dtype=np.float64)
        cfg = self.rule_config
        window = timedelta(hours=cfg.rapid_movement_window_hours)

        # Only consider transactions above minimum amount
        large_txns = df[df["amount"] >= cfg.rapid_movement_min_amount].copy()
        if large_txns.empty:
            return scores

        # For each account, find inflow-outflow patterns
        all_accounts = set(large_txns["from_id"].unique()) | set(large_txns["to_id"].unique())

        for account_id in all_accounts:
            # Incoming transactions to this account
            inflows = large_txns[large_txns["to_id"] == account_id].sort_values("timestamp")
            # Outgoing transactions from this account
            outflows = large_txns[large_txns["from_id"] == account_id].sort_values("timestamp")

            if inflows.empty or outflows.empty:
                continue

            for _, inflow in inflows.iterrows():
                # Find outflows within the rapid movement window
                window_start = inflow["timestamp"]
                window_end = window_start + window
                rapid_outflows = outflows[
                    (outflows["timestamp"] >= window_start) &
                    (outflows["timestamp"] <= window_end)
                ]

                if rapid_outflows.empty:
                    continue

                total_outflow = rapid_outflows["amount"].sum()
                outflow_ratio = total_outflow / inflow["amount"] if inflow["amount"] > 0 else 0

                if outflow_ratio >= cfg.rapid_movement_outflow_ratio:
                    # Score the inflow transaction
                    if inflow.name in df.index:
                        score = min(1.0, 0.3 + 0.3 * outflow_ratio + 0.2 * (len(rapid_outflows) > 1))
                        scores[inflow.name] = max(scores[inflow.name], score)

                    # Score the outflow transactions
                    for out_idx in rapid_outflows.index:
                        if out_idx in df.index:
                            scores[out_idx] = max(scores[out_idx], score * 0.8)

        return np.clip(scores, 0.0, 1.0)

    def _rule_round_tripping(self, df: pd.DataFrame) -> np.ndarray:
        """
        Round-Tripping Detection.

        Identifies circular fund flows where money returns to the originator
        through a chain of intermediaries.

        Returns
        -------
        np.ndarray
            Risk scores in [0, 1] for each transaction.
        """
        scores = np.zeros(len(df), dtype=np.float64)
        cfg = self.rule_config

        # Filter to significant transactions
        significant = df[df["amount"] >= cfg.round_trip_min_amount].copy()
        if significant.empty:
            return scores

        # Build simple directed adjacency for recent transactions
        window_end = significant["timestamp"].max()
        window_start = window_end - timedelta(days=cfg.round_trip_time_window_days)
        recent = significant[
            (significant["timestamp"] >= window_start) &
            (significant["timestamp"] <= window_end)
        ]

        if recent.empty:
            return scores

        # Build adjacency list
        adjacency: Dict[str, List[Tuple[str, int]]] = {}
        for idx, row in recent.iterrows():
            from_id = row["from_id"]
            to_id = row["to_id"]
            if from_id not in adjacency:
                adjacency[from_id] = []
            adjacency[from_id].append((to_id, idx))

        # DFS-based cycle detection (limited depth)
        cycles_found = 0
        visited_starts: set = set()

        for start_node in list(adjacency.keys())[:5000]:  # Limit for performance
            if start_node in visited_starts:
                continue
            visited_starts.add(start_node)

            # DFS with depth limit
            stack: List[Tuple[str, List[int], int]] = [(start_node, [], 0)]
            visited_in_path: set = set()

            while stack:
                current, path_indices, depth = stack.pop()

                if depth > cfg.round_trip_max_hops:
                    continue

                if current == start_node and depth > 1:
                    # Cycle found!
                    cycles_found += 1
                    cycle_score = min(1.0, 0.5 + 0.1 * len(path_indices))
                    for idx in path_indices:
                        if idx < len(scores):
                            scores[idx] = max(scores[idx], cycle_score)
                    continue

                if current in visited_in_path and current != start_node:
                    continue

                visited_in_path.add(current)

                for neighbor, edge_idx in adjacency.get(current, []):
                    stack.append((neighbor, path_indices + [edge_idx], depth + 1))

        if cycles_found > 0:
            logger.info(f"  Round-tripping: detected {cycles_found} potential cycles")

        return np.clip(scores, 0.0, 1.0)

    def _rule_geographic_risk(self, df: pd.DataFrame) -> np.ndarray:
        """
        Geographic Risk Scoring.

        Flags transactions involving high-risk jurisdictions as defined by
        FATF (Financial Action Task Force) black/grey lists.

        Returns
        -------
        np.ndarray
            Risk scores in [0, 1] for each transaction.
        """
        scores = np.zeros(len(df), dtype=np.float64)
        high_risk = set(self.rule_config.high_risk_jurisdictions)

        # In the IBM AML dataset, bank codes may contain jurisdiction info
        # We check if any bank code starts with a high-risk country code
        for col in ["from_bank", "to_bank"]:
            if col in df.columns:
                # Extract potential country code (first 2 characters)
                country_codes = df[col].str[:2].str.upper()
                is_high_risk = country_codes.isin(high_risk)
                scores[is_high_risk.values] += 0.3

        # Cross-border + cross-currency is additional risk
        if "is_cross_currency" in df.columns:
            cross_currency = df["is_cross_currency"] == 1
            scores[cross_currency.values] += 0.1

        # High-risk + large amount is higher risk
        if "amount" in df.columns:
            large_and_risky = (scores > 0) & (df["amount"].values > 50_000)
            scores[large_and_risky] += 0.2

        return np.clip(scores, 0.0, 1.0)

    def _rule_dormant_reactivation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Dormant Account Reactivation Detection.

        Flags transactions on accounts that were inactive for an extended
        period and suddenly show large transaction activity.

        Returns
        -------
        np.ndarray
            Risk scores in [0, 1] for each transaction.
        """
        scores = np.zeros(len(df), dtype=np.float64)
        cfg = self.rule_config

        if self._account_history is None:
            return scores

        # For each account, compute days since last transaction
        df_sorted = df.sort_values(["from_id", "timestamp"])

        for account_id, group in df_sorted.groupby("from_id"):
            if len(group) < 2:
                continue

            timestamps = group["timestamp"].values
            amounts = group["amount"].values

            for i in range(1, len(timestamps)):
                gap_days = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, "D")

                if gap_days >= cfg.dormant_days_threshold and amounts[i] >= cfg.dormant_reactivation_min_amount:
                    idx = group.index[i]
                    # Score based on dormancy duration and transaction size
                    dormancy_factor = min(1.0, gap_days / 365.0)
                    amount_factor = min(1.0, amounts[i] / 50_000.0)
                    score = 0.3 + 0.4 * dormancy_factor + 0.3 * amount_factor
                    if idx < len(scores):
                        scores[idx] = max(scores[idx], score)

        return np.clip(scores, 0.0, 1.0)

    def _rule_velocity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Velocity Rule Detection.

        Flags accounts with abnormally high transaction frequency within
        specified time windows.

        Returns
        -------
        np.ndarray
            Risk scores in [0, 1] for each transaction.
        """
        scores = np.zeros(len(df), dtype=np.float64)
        cfg = self.rule_config

        df_sorted = df.sort_values(["from_id", "timestamp"])

        for window_name, window_hours in cfg.velocity_windows.items():
            threshold = cfg.velocity_thresholds.get(window_name, 999)
            window_td = timedelta(hours=window_hours)

            for account_id, group in df_sorted.groupby("from_id"):
                if len(group) < threshold:
                    continue

                timestamps = group["timestamp"].values
                indices = group.index.values

                for i in range(len(timestamps)):
                    window_start = timestamps[i] - np.timedelta64(window_hours, "h")
                    count_in_window = ((timestamps >= window_start) & (timestamps <= timestamps[i])).sum()

                    if count_in_window >= threshold:
                        # Score proportional to how much the threshold is exceeded
                        excess_ratio = count_in_window / threshold
                        score = min(1.0, 0.3 + 0.2 * (excess_ratio - 1.0))
                        if indices[i] < len(scores):
                            scores[indices[i]] = max(scores[indices[i]], score)

        return np.clip(scores, 0.0, 1.0)

    def _compute_composite_score(
        self,
        df: pd.DataFrame,
        rule_outputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute weighted composite rule score.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data.
        rule_outputs : Dict[str, np.ndarray]
            Rule name to score array mapping.

        Returns
        -------
        np.ndarray
            Composite risk scores in [0, 1].
        """
        weights = self.rule_config.rule_weights
        total_weight = sum(weights.get(rule, 0) for rule in rule_outputs)

        if total_weight == 0:
            return np.zeros(len(df))

        composite = np.zeros(len(df), dtype=np.float64)
        for rule_name, scores in rule_outputs.items():
            weight = weights.get(rule_name, 0)
            composite += scores * (weight / total_weight)

        return np.clip(composite, 0.0, 1.0)

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about rule triggers.

        Returns
        -------
        Dict[str, Any]
            Statistics including trigger counts, rates, and overlaps.
        """
        return {
            "rule_triggers": self._rule_stats.copy(),
            "rule_weights": dict(self.rule_config.rule_weights),
        }

    def evaluate_single_transaction(
        self,
        transaction: Dict[str, Any],
        account_history: Optional[pd.DataFrame] = None,
    ) -> List[RuleResult]:
        """
        Evaluate all rules on a single transaction (for real-time scoring).

        Parameters
        ----------
        transaction : Dict[str, Any]
            Single transaction as a dictionary.
        account_history : pd.DataFrame, optional
            Historical transactions for the account.

        Returns
        -------
        List[RuleResult]
            List of rule evaluation results.
        """
        txn_df = pd.DataFrame([transaction])
        results = []

        if account_history is not None:
            self._build_account_history(account_history)

        # Evaluate each rule
        rule_functions = {
            "structuring": ("Structuring Detection", self._rule_structuring),
            "rapid_movement": ("Rapid Fund Movement", self._rule_rapid_movement),
            "geographic_risk": ("Geographic Risk", self._rule_geographic_risk),
            "velocity": ("Velocity Anomaly", self._rule_velocity),
        }

        for rule_id, (rule_name, rule_fn) in rule_functions.items():
            try:
                score_array = rule_fn(txn_df)
                score = float(score_array[0]) if len(score_array) > 0 else 0.0
                results.append(RuleResult(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    risk_score=score,
                    triggered=score > 0,
                    details={"amount": transaction.get("amount", 0)},
                ))
            except Exception as e:
                logger.warning(f"Rule '{rule_id}' evaluation failed: {e}")
                results.append(RuleResult(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    risk_score=0.0,
                    triggered=False,
                    details={"error": str(e)},
                ))

        return results
