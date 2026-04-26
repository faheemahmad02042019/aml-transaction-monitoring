"""
AML-Specific Feature Engineering.

Generates a comprehensive feature set for ML-based suspicious activity
detection, organized into five categories:

  1. Transaction-level features (amount, currency, type, timing)
  2. Account-level features (frequency, amounts, balance changes)
  3. Behavioral features (deviation from historical patterns)
  4. Network features (counterparty diversity, fan-in/fan-out)
  5. Temporal features (rolling window aggregations)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config, FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class AMLFeatureEngineer:
    """
    Generates AML-specific features for transaction-level and account-level
    classification.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.feat_config: FeatureEngineeringConfig = config.features
        self._feature_names: List[str] = []
        self._feature_stats: Dict[str, Dict] = {}

    def engineer_features(
        self,
        df: pd.DataFrame,
        rule_scores: Optional[pd.DataFrame] = None,
        graph_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate the full feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with standard columns.
        rule_scores : pd.DataFrame, optional
            Rule engine output to include as features.
        graph_features : pd.DataFrame, optional
            Graph analysis features to include.

        Returns
        -------
        pd.DataFrame
            Feature matrix aligned with the input dataframe index.
        """
        logger.info(f"Engineering features for {len(df):,} transactions")

        features = pd.DataFrame(index=df.index)

        # 1. Transaction-level features
        txn_feats = self._transaction_level_features(df)
        features = pd.concat([features, txn_feats], axis=1)
        logger.info(f"  Transaction-level: {txn_feats.shape[1]} features")

        # 2. Account-level features
        acct_feats = self._account_level_features(df)
        features = pd.concat([features, acct_feats], axis=1)
        logger.info(f"  Account-level: {acct_feats.shape[1]} features")

        # 3. Behavioral features
        behav_feats = self._behavioral_features(df)
        features = pd.concat([features, behav_feats], axis=1)
        logger.info(f"  Behavioral: {behav_feats.shape[1]} features")

        # 4. Network features
        net_feats = self._network_features(df)
        features = pd.concat([features, net_feats], axis=1)
        logger.info(f"  Network: {net_feats.shape[1]} features")

        # 5. Temporal rolling features
        temp_feats = self._temporal_features(df)
        features = pd.concat([features, temp_feats], axis=1)
        logger.info(f"  Temporal: {temp_feats.shape[1]} features")

        # 6. Risk indicator features
        risk_feats = self._risk_indicator_features(df)
        features = pd.concat([features, risk_feats], axis=1)
        logger.info(f"  Risk indicators: {risk_feats.shape[1]} features")

        # 7. Include rule scores if provided
        if rule_scores is not None:
            rule_cols = [c for c in rule_scores.columns if c.startswith("rule_")]
            if rule_cols:
                features = pd.concat([features, rule_scores[rule_cols].reindex(df.index)], axis=1)
                logger.info(f"  Rule scores: {len(rule_cols)} features")

        # 8. Include graph features if provided
        if graph_features is not None:
            graph_cols = [c for c in graph_features.columns if c.startswith("graph_")]
            if graph_cols:
                features = pd.concat([features, graph_features[graph_cols].reindex(df.index)], axis=1)
                logger.info(f"  Graph features: {len(graph_cols)} features")

        # Handle infinities and NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        self._feature_names = list(features.columns)
        logger.info(f"Total features: {features.shape[1]}")

        return features

    def _transaction_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract transaction-level features.

        Features:
          - Amount (raw, log-transformed, z-score)
          - Currency indicators
          - Payment format encoding
          - Time-of-day, day-of-week indicators
          - Weekend/business hours flags
        """
        feats = pd.DataFrame(index=df.index)

        # Amount features
        feats["feat_amount"] = df["amount"]
        feats["feat_amount_log"] = np.log1p(df["amount"])
        amount_mean = df["amount"].mean()
        amount_std = df["amount"].std()
        feats["feat_amount_zscore"] = (
            (df["amount"] - amount_mean) / amount_std if amount_std > 0 else 0
        )
        feats["feat_amount_paid"] = df.get("amount_paid", df["amount"])
        feats["feat_amount_received"] = df.get("amount_received", df["amount"])

        # Amount difference (paid vs received may indicate currency conversion spreads)
        if "amount_paid" in df.columns and "amount_received" in df.columns:
            feats["feat_amount_diff"] = df["amount_received"] - df["amount_paid"]
            feats["feat_amount_ratio"] = np.where(
                df["amount_paid"] > 0,
                df["amount_received"] / df["amount_paid"],
                1.0,
            )

        # Cross-currency flag
        if "is_cross_currency" in df.columns:
            feats["feat_cross_currency"] = df["is_cross_currency"]

        # Payment format one-hot encoding
        if "payment_format_normalized" in df.columns:
            format_dummies = pd.get_dummies(
                df["payment_format_normalized"], prefix="feat_format"
            ).astype(int)
            feats = pd.concat([feats, format_dummies], axis=1)
        elif "payment_format" in df.columns:
            format_dummies = pd.get_dummies(
                df["payment_format"], prefix="feat_format"
            ).astype(int)
            feats = pd.concat([feats, format_dummies], axis=1)

        # Time-of-day features
        if "hour" in df.columns:
            feats["feat_hour"] = df["hour"]
            feats["feat_hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            feats["feat_hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            feats["feat_is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
            feats["feat_is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
            feats["feat_is_early_morning"] = ((df["hour"] >= 4) & (df["hour"] <= 7)).astype(int)

        # Day-of-week features
        if "day_of_week" in df.columns:
            feats["feat_day_of_week"] = df["day_of_week"]
            feats["feat_day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            feats["feat_day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            feats["feat_is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        return feats

    def _account_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract account-level features.

        For each transaction, compute aggregate statistics about the
        originating account based on the full dataset.

        Features:
          - Transaction count, total amount, average amount
          - Balance changes (inflow - outflow)
          - Days active, account age
          - Unique counterparties
        """
        feats = pd.DataFrame(index=df.index)

        # Sender (from_id) aggregate features
        sender_agg = df.groupby("from_id").agg(
            sender_txn_count=("transaction_id", "count"),
            sender_total_amount=("amount", "sum"),
            sender_mean_amount=("amount", "mean"),
            sender_std_amount=("amount", "std"),
            sender_max_amount=("amount", "max"),
            sender_min_amount=("amount", "min"),
            sender_median_amount=("amount", "median"),
            sender_unique_recipients=("to_id", "nunique"),
            sender_first_txn=("timestamp", "min"),
            sender_last_txn=("timestamp", "max"),
        ).reset_index()
        sender_agg["sender_std_amount"] = sender_agg["sender_std_amount"].fillna(0)
        sender_agg["sender_days_active"] = (
            (sender_agg["sender_last_txn"] - sender_agg["sender_first_txn"]).dt.total_seconds() / 86400
        ).clip(lower=1)
        sender_agg["sender_txn_rate"] = sender_agg["sender_txn_count"] / sender_agg["sender_days_active"]

        feats = feats.join(
            df[["from_id"]].merge(sender_agg, on="from_id", how="left")
            .drop(columns=["from_id", "sender_first_txn", "sender_last_txn"])
            .set_index(df.index)
        )

        # Receiver (to_id) aggregate features
        receiver_agg = df.groupby("to_id").agg(
            receiver_txn_count=("transaction_id", "count"),
            receiver_total_amount=("amount", "sum"),
            receiver_mean_amount=("amount", "mean"),
            receiver_std_amount=("amount", "std"),
            receiver_unique_senders=("from_id", "nunique"),
        ).reset_index()
        receiver_agg["receiver_std_amount"] = receiver_agg["receiver_std_amount"].fillna(0)

        feats = feats.join(
            df[["to_id"]].merge(receiver_agg, on="to_id", how="left")
            .drop(columns=["to_id"])
            .set_index(df.index)
        )

        # Rename for clarity
        for col in feats.columns:
            if not col.startswith("feat_"):
                feats = feats.rename(columns={col: f"feat_{col}"})

        return feats.fillna(0)

    def _behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract behavioral deviation features.

        For each transaction, compute how much it deviates from the
        account's historical behavior.

        Features:
          - Amount deviation from account mean/median
          - Time-of-day deviation from account pattern
          - Sudden activity changes
        """
        feats = pd.DataFrame(index=df.index)

        # Per-account statistics for deviation calculation
        account_stats = df.groupby("from_id").agg(
            acct_mean_amount=("amount", "mean"),
            acct_std_amount=("amount", "std"),
            acct_median_amount=("amount", "median"),
            acct_mean_hour=("hour", "mean") if "hour" in df.columns else ("amount", "count"),
        ).reset_index()
        account_stats["acct_std_amount"] = account_stats["acct_std_amount"].fillna(1.0).replace(0, 1.0)

        # Merge back to get per-transaction account stats
        merged = df[["from_id", "amount"]].merge(account_stats, on="from_id", how="left")

        # Amount deviation from personal mean
        feats["feat_amount_dev_from_mean"] = (
            (merged["amount"] - merged["acct_mean_amount"]) / merged["acct_std_amount"]
        ).fillna(0)

        # Amount deviation from personal median (more robust to outliers)
        feats["feat_amount_dev_from_median"] = np.where(
            merged["acct_median_amount"] > 0,
            (merged["amount"] - merged["acct_median_amount"]) / merged["acct_median_amount"],
            0,
        )

        # Is this a personal-maximum transaction?
        account_max = df.groupby("from_id")["amount"].max().reset_index()
        account_max.columns = ["from_id", "acct_max_amount"]
        merged_max = df[["from_id", "amount"]].merge(account_max, on="from_id", how="left")
        feats["feat_is_personal_max"] = (
            merged_max["amount"] >= merged_max["acct_max_amount"] * 0.95
        ).astype(int)

        # Transaction frequency deviation
        # Compute per-account daily transaction counts
        daily_counts = (
            df.groupby(["from_id", "date"])
            .size()
            .reset_index(name="daily_count")
        )
        daily_avg = daily_counts.groupby("from_id")["daily_count"].agg(["mean", "std"]).reset_index()
        daily_avg.columns = ["from_id", "daily_avg_count", "daily_std_count"]
        daily_avg["daily_std_count"] = daily_avg["daily_std_count"].fillna(1).replace(0, 1)

        # For each transaction, get that day's count and compute deviation
        txn_day_count = df.merge(
            daily_counts, on=["from_id", "date"], how="left"
        ).merge(
            daily_avg, on="from_id", how="left"
        )

        feats["feat_daily_count_deviation"] = (
            (txn_day_count["daily_count"] - txn_day_count["daily_avg_count"])
            / txn_day_count["daily_std_count"]
        ).fillna(0).values

        # Time deviation: is the transaction at an unusual hour for this account?
        if "hour" in df.columns:
            hour_mode = df.groupby("from_id")["hour"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12)
            hour_mode = hour_mode.reset_index()
            hour_mode.columns = ["from_id", "typical_hour"]
            merged_hour = df[["from_id", "hour"]].merge(hour_mode, on="from_id", how="left")
            # Circular distance for hours
            hour_diff = np.abs(merged_hour["hour"] - merged_hour["typical_hour"])
            feats["feat_hour_deviation"] = np.minimum(hour_diff, 24 - hour_diff).fillna(0).values

        return feats

    def _network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract network-based features.

        Features:
          - Number of unique counterparties (fan-out, fan-in)
          - Fan-out to fan-in ratio
          - Counterparty concentration (Herfindahl index)
          - Is this a new counterparty relationship?
        """
        feats = pd.DataFrame(index=df.index)

        # Fan-out: unique recipients per sender
        fan_out = df.groupby("from_id")["to_id"].nunique().reset_index()
        fan_out.columns = ["from_id", "sender_fan_out"]
        merged = df[["from_id"]].merge(fan_out, on="from_id", how="left")
        feats["feat_sender_fan_out"] = merged["sender_fan_out"].fillna(0).values

        # Fan-in: unique senders per receiver
        fan_in = df.groupby("to_id")["from_id"].nunique().reset_index()
        fan_in.columns = ["to_id", "receiver_fan_in"]
        merged = df[["to_id"]].merge(fan_in, on="to_id", how="left")
        feats["feat_receiver_fan_in"] = merged["receiver_fan_in"].fillna(0).values

        # Fan-out to fan-in ratio for sender
        sender_fan_in = df.groupby("from_id").apply(
            lambda g: df[df["to_id"] == g.name]["from_id"].nunique()
            if g.name in df["to_id"].values else 0
        ).reset_index()
        sender_fan_in.columns = ["from_id", "sender_fan_in"]

        merged = df[["from_id"]].merge(fan_out, on="from_id", how="left").merge(
            sender_fan_in, on="from_id", how="left"
        )
        feats["feat_fan_out_in_ratio"] = np.where(
            merged["sender_fan_in"] > 0,
            merged["sender_fan_out"] / merged["sender_fan_in"],
            merged["sender_fan_out"],
        ).astype(float)

        # Counterparty concentration (Herfindahl-Hirschman Index)
        # Measures how concentrated the sender's transactions are among recipients
        def hhi(group: pd.DataFrame) -> float:
            """Compute HHI for transaction distribution across counterparties."""
            if len(group) == 0:
                return 0.0
            shares = group["to_id"].value_counts(normalize=True)
            return float((shares ** 2).sum())

        hhi_values = df.groupby("from_id").apply(hhi).reset_index()
        hhi_values.columns = ["from_id", "sender_hhi"]
        merged = df[["from_id"]].merge(hhi_values, on="from_id", how="left")
        feats["feat_counterparty_concentration"] = merged["sender_hhi"].fillna(0).values

        # Is this counterparty relationship new?
        # Sort by time and check if sender->receiver pair has been seen before
        df_sorted = df.sort_values("timestamp")
        pair_first_seen = df_sorted.groupby(["from_id", "to_id"])["timestamp"].first().reset_index()
        pair_first_seen.columns = ["from_id", "to_id", "pair_first_seen"]
        merged = df.merge(pair_first_seen, on=["from_id", "to_id"], how="left")
        feats["feat_is_new_counterparty"] = (
            merged["timestamp"] == merged["pair_first_seen"]
        ).astype(int).values

        # Same-bank vs cross-bank
        feats["feat_is_cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(int)

        return feats

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal rolling-window features.

        Computes aggregations over 1D, 7D, 30D, and 90D windows:
          - Transaction count
          - Total/mean/max amount
          - Unique counterparties
        """
        feats = pd.DataFrame(index=df.index)
        df_sorted = df.sort_values("timestamp").copy()

        for window in self.feat_config.rolling_windows:
            logger.info(f"  Computing {window} rolling features")

            # Set timestamp as index for rolling
            df_indexed = df_sorted.set_index("timestamp")

            # Per-account rolling aggregations
            for account_col, prefix in [("from_id", "sender"), ("to_id", "receiver")]:
                grouped = df_indexed.groupby(account_col)

                # Rolling count
                rolling_count = grouped["amount"].rolling(window, min_periods=1).count()
                rolling_count = rolling_count.reset_index(level=0, drop=True).sort_index()

                # Rolling sum
                rolling_sum = grouped["amount"].rolling(window, min_periods=1).sum()
                rolling_sum = rolling_sum.reset_index(level=0, drop=True).sort_index()

                # Rolling mean
                rolling_mean = grouped["amount"].rolling(window, min_periods=1).mean()
                rolling_mean = rolling_mean.reset_index(level=0, drop=True).sort_index()

                # Rolling max
                rolling_max = grouped["amount"].rolling(window, min_periods=1).max()
                rolling_max = rolling_max.reset_index(level=0, drop=True).sort_index()

                # Rolling std
                rolling_std = grouped["amount"].rolling(window, min_periods=1).std()
                rolling_std = rolling_std.reset_index(level=0, drop=True).sort_index()

                # Map back to original index
                window_label = window.lower().replace("d", "d")
                feats[f"feat_{prefix}_count_{window_label}"] = rolling_count.reindex(df_sorted.index).fillna(0).values
                feats[f"feat_{prefix}_sum_{window_label}"] = rolling_sum.reindex(df_sorted.index).fillna(0).values
                feats[f"feat_{prefix}_mean_{window_label}"] = rolling_mean.reindex(df_sorted.index).fillna(0).values
                feats[f"feat_{prefix}_max_{window_label}"] = rolling_max.reindex(df_sorted.index).fillna(0).values
                feats[f"feat_{prefix}_std_{window_label}"] = rolling_std.reindex(df_sorted.index).fillna(0).values

        # Reindex back to original df index
        feats = feats.reindex(df.index).fillna(0)

        return feats

    def _risk_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract explicit risk indicator features.

        Features:
          - High-risk country involvement
          - Large round-amount flag
          - Structuring pattern score (amount proximity to thresholds)
          - Night/weekend activity combined with large amounts
        """
        feats = pd.DataFrame(index=df.index)

        # Large round amount flag
        for threshold in self.feat_config.round_amount_thresholds:
            is_round = (df["amount"] % threshold == 0) & (df["amount"] >= threshold)
            feats[f"feat_round_amount_{int(threshold)}"] = is_round.astype(int)

        # Structuring proximity score (distance to $10K threshold)
        feats["feat_structuring_proximity"] = np.where(
            (df["amount"] >= 5000) & (df["amount"] < 10000),
            1.0 - (10000 - df["amount"]) / 5000,
            0.0,
        )

        # Large transaction flag
        feats["feat_is_large_txn"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
        feats["feat_is_very_large_txn"] = (df["amount"] > df["amount"].quantile(0.99)).astype(int)

        # Night + large amount combination (suspicious)
        if "hour" in df.columns:
            is_night = (df["hour"] >= 22) | (df["hour"] <= 5)
            feats["feat_night_large"] = (is_night & (df["amount"] > df["amount"].median())).astype(int)

        # Weekend + large amount
        if "day_of_week" in df.columns:
            is_weekend = df["day_of_week"] >= 5
            feats["feat_weekend_large"] = (is_weekend & (df["amount"] > df["amount"].median())).astype(int)

        # Amount entropy per sender (diversity of amounts used)
        def amount_entropy(amounts: pd.Series) -> float:
            """Compute Shannon entropy of binned amounts."""
            if len(amounts) <= 1:
                return 0.0
            hist, _ = np.histogram(amounts, bins=min(20, len(amounts)))
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))

        entropy_values = df.groupby("from_id")["amount"].apply(amount_entropy).reset_index()
        entropy_values.columns = ["from_id", "amount_entropy"]
        merged = df[["from_id"]].merge(entropy_values, on="from_id", how="left")
        feats["feat_amount_entropy"] = merged["amount_entropy"].fillna(0).values

        return feats

    @property
    def feature_names(self) -> List[str]:
        """Return list of generated feature names."""
        return self._feature_names.copy()

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Return feature names grouped by category for analysis.

        Returns
        -------
        Dict[str, List[str]]
            Feature category to feature name list mapping.
        """
        groups = {
            "transaction": [],
            "account": [],
            "behavioral": [],
            "network": [],
            "temporal": [],
            "risk_indicator": [],
            "rule_score": [],
            "graph": [],
        }

        for name in self._feature_names:
            if name.startswith("feat_amount") or name.startswith("feat_format") or name.startswith("feat_hour") or name.startswith("feat_day") or name.startswith("feat_cross") or name.startswith("feat_is_business") or name.startswith("feat_is_night") or name.startswith("feat_is_early"):
                groups["transaction"].append(name)
            elif name.startswith("feat_sender_") or name.startswith("feat_receiver_") and "count_" not in name and "sum_" not in name and "mean_" not in name and "max_" not in name and "std_" not in name:
                groups["account"].append(name)
            elif "dev" in name or "deviation" in name or "personal" in name:
                groups["behavioral"].append(name)
            elif "fan" in name or "counterpart" in name or "new_counterparty" in name or "cross_bank" in name or "hhi" in name:
                groups["network"].append(name)
            elif any(w in name for w in ["_1d", "_7d", "_30d", "_90d"]):
                groups["temporal"].append(name)
            elif any(w in name for w in ["round_amount", "structuring", "large_txn", "night_large", "weekend_large", "entropy"]):
                groups["risk_indicator"].append(name)
            elif name.startswith("rule_"):
                groups["rule_score"].append(name)
            elif name.startswith("graph_"):
                groups["graph"].append(name)

        return {k: v for k, v in groups.items() if v}
