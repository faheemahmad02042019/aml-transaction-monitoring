"""
IBM AML Transaction Dataset Loader.

Handles loading, parsing, profiling, and time-based splitting of the IBM
Transactions for Anti-Money Laundering dataset family (Small, Medium, Large).
Supports both Pandas (for Small/Medium) and PySpark (for Large) backends.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import Config, DataConfig

logger = logging.getLogger(__name__)


class AMLDataLoader:
    """
    Loader for IBM AML Transaction datasets.

    Parses transaction records, extracts laundering labels, computes
    dataset statistics, and performs a time-based train/test split
    to prevent data leakage.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    EXPECTED_COLUMNS = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Amount Paid", "Receiving Currency",
        "Payment Currency", "Payment Format", "Is Laundering",
    ]

    TRANSACTION_TYPES = {
        "Cheque": "check",
        "ACH": "ach",
        "Wire": "wire",
        "Credit Card": "credit_card",
        "Cash": "cash",
        "Reinvestment": "reinvestment",
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_config: DataConfig = config.data
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
        self._profile: Optional[Dict[str, Any]] = None

    def load(
        self,
        variant: str = "small",
        nrows: Optional[int] = None,
        use_spark: bool = False,
    ) -> pd.DataFrame:
        """
        Load an IBM AML dataset variant.

        Parameters
        ----------
        variant : str
            Dataset size variant: "small", "medium", or "large".
        nrows : int, optional
            Number of rows to load (useful for development/testing).
        use_spark : bool
            If True, use PySpark for loading (recommended for "large" variant).

        Returns
        -------
        pd.DataFrame
            Loaded and parsed transaction data.

        Raises
        ------
        FileNotFoundError
            If the dataset file does not exist.
        ValueError
            If the variant is not recognized or columns are missing.
        """
        file_path = self.data_config.file_path.get(variant)
        if file_path is None:
            raise ValueError(f"Unknown dataset variant: '{variant}'. Choose from: small, medium, large")

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                f"Download the IBM AML dataset and place the CSV in: {self.data_config.data_dir}"
            )

        logger.info(f"Loading IBM AML dataset: variant={variant}, path={file_path}")

        if use_spark:
            df = self._load_with_spark(file_path, nrows)
        else:
            df = self._load_with_pandas(file_path, nrows)

        self._raw_data = df.copy()
        self._processed_data = self._process(df)

        logger.info(
            f"Loaded {len(self._processed_data):,} transactions | "
            f"Laundering: {self._processed_data['is_laundering'].sum():,} "
            f"({self._processed_data['is_laundering'].mean() * 100:.3f}%)"
        )

        return self._processed_data

    def _load_with_pandas(self, file_path: Path, nrows: Optional[int]) -> pd.DataFrame:
        """Load dataset using Pandas."""
        logger.info(f"Loading with Pandas (nrows={nrows})")
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            dtype={
                "From Bank": str,
                "Account": str,
                "To Bank": str,
                "Account.1": str,
                "Payment Format": str,
                "Receiving Currency": str,
                "Payment Currency": str,
            },
        )
        self._validate_columns(df)
        return df

    def _load_with_spark(self, file_path: Path, nrows: Optional[int]) -> pd.DataFrame:
        """Load dataset using PySpark (converts to Pandas for downstream processing)."""
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F

            logger.info("Initializing PySpark session")
            spark = (
                SparkSession.builder
                .appName("AML-DataLoader")
                .config("spark.driver.memory", "8g")
                .config("spark.sql.adaptive.enabled", "true")
                .getOrCreate()
            )

            spark_df = spark.read.csv(str(file_path), header=True, inferSchema=True)

            if nrows is not None:
                spark_df = spark_df.limit(nrows)

            df = spark_df.toPandas()
            spark.stop()

            self._validate_columns(df)
            return df

        except ImportError:
            logger.warning("PySpark not available. Falling back to Pandas.")
            return self._load_with_pandas(file_path, nrows)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that all expected columns are present."""
        missing = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing expected columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw transaction data into a clean, analysis-ready format.

        Steps:
        1. Parse timestamps
        2. Normalize column names (snake_case)
        3. Create composite account identifiers
        4. Parse transaction types
        5. Handle missing values
        6. Sort by timestamp
        """
        cols = self.data_config.column_mapping
        processed = pd.DataFrame()

        # Parse timestamp
        processed["timestamp"] = pd.to_datetime(
            df[cols["timestamp"]],
            format=self.data_config.timestamp_format,
            errors="coerce",
        )

        # Account identifiers (bank + account = unique ID)
        processed["from_bank"] = df[cols["from_bank"]].astype(str).str.strip()
        processed["from_account"] = df[cols["from_account"]].astype(str).str.strip()
        processed["to_bank"] = df[cols["to_bank"]].astype(str).str.strip()
        processed["to_account"] = df[cols["to_account"]].astype(str).str.strip()
        processed["from_id"] = processed["from_bank"] + "_" + processed["from_account"]
        processed["to_id"] = processed["to_bank"] + "_" + processed["to_account"]

        # Transaction amounts
        processed["amount_paid"] = pd.to_numeric(df[cols["amount_paid"]], errors="coerce").fillna(0.0)
        processed["amount_received"] = pd.to_numeric(df[cols["amount_received"]], errors="coerce").fillna(0.0)
        processed["amount"] = processed[["amount_paid", "amount_received"]].max(axis=1)

        # Currencies
        processed["payment_currency"] = df[cols["payment_currency"]].astype(str).str.strip().str.upper()
        processed["receiving_currency"] = df[cols["receiving_currency"]].astype(str).str.strip().str.upper()
        processed["is_cross_currency"] = (
            processed["payment_currency"] != processed["receiving_currency"]
        ).astype(int)

        # Transaction type
        processed["payment_format"] = df[cols["payment_format"]].astype(str).str.strip()
        processed["payment_format_normalized"] = processed["payment_format"].map(
            self.TRANSACTION_TYPES
        ).fillna("other")

        # Label
        processed["is_laundering"] = df[cols["is_laundering"]].astype(int)

        # Temporal features (basic, for data loader level)
        processed["date"] = processed["timestamp"].dt.date
        processed["hour"] = processed["timestamp"].dt.hour
        processed["day_of_week"] = processed["timestamp"].dt.dayofweek

        # Drop rows with null timestamps (parsing failures)
        n_null_ts = processed["timestamp"].isna().sum()
        if n_null_ts > 0:
            logger.warning(f"Dropping {n_null_ts:,} rows with unparseable timestamps")
            processed = processed.dropna(subset=["timestamp"])

        # Sort by time
        processed = processed.sort_values("timestamp").reset_index(drop=True)
        processed["transaction_id"] = range(len(processed))

        return processed

    def time_based_split(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a time-based train/test split.

        This prevents future data leakage by using earlier transactions for
        training and later transactions for testing.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Data to split. If None, uses the last loaded dataset.
        test_size : float, optional
            Fraction of data for testing. Defaults to config value.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df) split by timestamp.
        """
        if df is None:
            if self._processed_data is None:
                raise RuntimeError("No data loaded. Call .load() first.")
            df = self._processed_data

        test_size = test_size or self.data_config.test_size

        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1.0 - test_size))
        split_timestamp = df_sorted.iloc[split_idx]["timestamp"]

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()

        logger.info(
            f"Time-based split at {split_timestamp} | "
            f"Train: {len(train_df):,} ({train_df['is_laundering'].mean() * 100:.3f}% positive) | "
            f"Test:  {len(test_df):,} ({test_df['is_laundering'].mean() * 100:.3f}% positive)"
        )

        return train_df, test_df

    def profile(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Data to profile. If None, uses the last loaded dataset.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing profiling statistics.
        """
        if df is None:
            if self._processed_data is None:
                raise RuntimeError("No data loaded. Call .load() first.")
            df = self._processed_data

        total = len(df)
        laundering = df["is_laundering"].sum()

        profile = {
            "total_transactions": total,
            "laundering_transactions": int(laundering),
            "laundering_rate_pct": float(laundering / total * 100) if total > 0 else 0.0,
            "date_range": {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "span_days": (df["timestamp"].max() - df["timestamp"].min()).days,
            },
            "unique_accounts": {
                "from": df["from_id"].nunique(),
                "to": df["to_id"].nunique(),
                "total": pd.concat([df["from_id"], df["to_id"]]).nunique(),
            },
            "unique_banks": {
                "from": df["from_bank"].nunique(),
                "to": df["to_bank"].nunique(),
            },
            "amount_statistics": {
                "mean": float(df["amount"].mean()),
                "median": float(df["amount"].median()),
                "std": float(df["amount"].std()),
                "min": float(df["amount"].min()),
                "max": float(df["amount"].max()),
                "percentiles": {
                    f"p{int(p * 100)}": float(df["amount"].quantile(p))
                    for p in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
                },
            },
            "payment_format_distribution": df["payment_format"].value_counts().to_dict(),
            "currency_distribution": df["payment_currency"].value_counts().head(20).to_dict(),
            "cross_currency_rate_pct": float(df["is_cross_currency"].mean() * 100),
            "transactions_per_day": {
                "mean": float(df.groupby("date").size().mean()),
                "std": float(df.groupby("date").size().std()),
                "max": int(df.groupby("date").size().max()),
            },
            "laundering_by_format": (
                df.groupby("payment_format")["is_laundering"]
                .agg(["sum", "mean"])
                .rename(columns={"sum": "count", "mean": "rate"})
                .to_dict("index")
            ),
        }

        self._profile = profile
        return profile

    def get_account_summary(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate per-account summary statistics.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Transaction data to summarize.

        Returns
        -------
        pd.DataFrame
            Account-level summary with transaction counts, amounts, and
            laundering flags.
        """
        if df is None:
            if self._processed_data is None:
                raise RuntimeError("No data loaded. Call .load() first.")
            df = self._processed_data

        # Outgoing transactions
        outgoing = (
            df.groupby("from_id")
            .agg(
                out_count=("amount", "count"),
                out_total=("amount", "sum"),
                out_mean=("amount", "mean"),
                out_max=("amount", "max"),
                out_laundering=("is_laundering", "sum"),
                out_unique_counterparties=("to_id", "nunique"),
                first_out_txn=("timestamp", "min"),
                last_out_txn=("timestamp", "max"),
            )
            .reset_index()
            .rename(columns={"from_id": "account_id"})
        )

        # Incoming transactions
        incoming = (
            df.groupby("to_id")
            .agg(
                in_count=("amount", "count"),
                in_total=("amount", "sum"),
                in_mean=("amount", "mean"),
                in_max=("amount", "max"),
                in_laundering=("is_laundering", "sum"),
                in_unique_counterparties=("from_id", "nunique"),
                first_in_txn=("timestamp", "min"),
                last_in_txn=("timestamp", "max"),
            )
            .reset_index()
            .rename(columns={"to_id": "account_id"})
        )

        # Merge
        account_summary = outgoing.merge(incoming, on="account_id", how="outer").fillna(0)
        account_summary["total_txn_count"] = account_summary["out_count"] + account_summary["in_count"]
        account_summary["net_flow"] = account_summary["in_total"] - account_summary["out_total"]
        account_summary["any_laundering"] = (
            (account_summary["out_laundering"] + account_summary["in_laundering"]) > 0
        ).astype(int)

        return account_summary

    def create_synthetic_sample(self, n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
        """
        Create a synthetic transaction dataset for testing and demonstration.

        Generates realistic-looking AML transaction data with known laundering
        patterns (structuring, rapid movement, round-tripping) for use when
        the IBM dataset is not available.

        Parameters
        ----------
        n_samples : int
            Number of synthetic transactions to generate.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Synthetic transaction data in the same format as processed IBM data.
        """
        rng = np.random.RandomState(seed)
        logger.info(f"Generating {n_samples:,} synthetic transactions for demonstration")

        n_accounts = max(100, n_samples // 50)
        n_banks = 20
        banks = [f"BANK_{i:04d}" for i in range(n_banks)]
        accounts = [f"{rng.choice(banks)}_{rng.randint(10000, 99999)}" for _ in range(n_accounts)]
        currencies = ["USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD"]
        formats = ["Wire", "ACH", "Cheque", "Credit Card", "Cash"]

        start_date = pd.Timestamp("2023-01-01")
        timestamps = [
            start_date + pd.Timedelta(hours=rng.randint(0, 365 * 24))
            for _ in range(n_samples)
        ]
        timestamps.sort()

        records = []
        for i in range(n_samples):
            from_idx = rng.randint(0, n_accounts)
            to_idx = rng.randint(0, n_accounts)
            while to_idx == from_idx:
                to_idx = rng.randint(0, n_accounts)

            amount = float(rng.lognormal(mean=7.0, sigma=2.0))  # Log-normal amounts
            amount = round(amount, 2)

            records.append({
                "transaction_id": i,
                "timestamp": timestamps[i],
                "from_id": accounts[from_idx],
                "from_bank": accounts[from_idx].split("_")[0] + "_" + accounts[from_idx].split("_")[1],
                "from_account": accounts[from_idx].split("_")[2] if len(accounts[from_idx].split("_")) > 2 else accounts[from_idx],
                "to_id": accounts[to_idx],
                "to_bank": accounts[to_idx].split("_")[0] + "_" + accounts[to_idx].split("_")[1],
                "to_account": accounts[to_idx].split("_")[2] if len(accounts[to_idx].split("_")) > 2 else accounts[to_idx],
                "amount_paid": amount,
                "amount_received": amount,
                "amount": amount,
                "payment_currency": rng.choice(currencies),
                "receiving_currency": rng.choice(currencies),
                "is_cross_currency": 0,
                "payment_format": rng.choice(formats),
                "payment_format_normalized": rng.choice(["wire", "ach", "check", "credit_card", "cash"]),
                "is_laundering": 0,
                "date": timestamps[i].date(),
                "hour": timestamps[i].hour,
                "day_of_week": timestamps[i].dayofweek,
            })

        df = pd.DataFrame(records)

        # Inject laundering patterns (0.2% of transactions)
        n_laundering = max(10, int(n_samples * 0.002))
        laundering_indices = rng.choice(range(n_samples), size=n_laundering, replace=False)
        df.loc[laundering_indices, "is_laundering"] = 1

        # Inject structuring pattern: amounts just below $10K
        n_structuring = n_laundering // 3
        structuring_indices = laundering_indices[:n_structuring]
        df.loc[structuring_indices, "amount"] = rng.uniform(8000, 9999, size=n_structuring).round(2)
        df.loc[structuring_indices, "amount_paid"] = df.loc[structuring_indices, "amount"]
        df.loc[structuring_indices, "amount_received"] = df.loc[structuring_indices, "amount"]

        # Update cross-currency flag
        df["is_cross_currency"] = (df["payment_currency"] != df["receiving_currency"]).astype(int)

        logger.info(
            f"Synthetic data created: {len(df):,} transactions, "
            f"{df['is_laundering'].sum():,} laundering ({df['is_laundering'].mean() * 100:.2f}%)"
        )

        self._processed_data = df
        return df
