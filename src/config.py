"""
Centralized configuration for the AML Transaction Monitoring system.

Manages data paths, model hyperparameters, rule thresholds, LLM settings,
graph analysis parameters, and alert priority thresholds. All configuration
values can be overridden via environment variables or a .env file.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    data_dir: Path = DATA_DIR
    small_file: str = "HI-Small_Trans.csv"
    medium_file: str = "HI-Medium_Trans.csv"
    large_file: str = "HI-Large_Trans.csv"
    test_size: float = 0.20
    time_split_column: str = "Timestamp"
    random_state: int = 42
    timestamp_format: str = "%Y/%m/%d %H:%M"

    # Column mapping for IBM AML dataset
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "timestamp": "Timestamp",
        "from_bank": "From Bank",
        "from_account": "Account",
        "to_bank": "To Bank",
        "to_account": "Account.1",
        "amount_received": "Amount Received",
        "amount_paid": "Amount Paid",
        "receiving_currency": "Receiving Currency",
        "payment_currency": "Payment Currency",
        "payment_format": "Payment Format",
        "is_laundering": "Is Laundering",
    })

    @property
    def file_path(self) -> Dict[str, Path]:
        """Return file paths for each dataset variant."""
        return {
            "small": self.data_dir / self.small_file,
            "medium": self.data_dir / self.medium_file,
            "large": self.data_dir / self.large_file,
        }


@dataclass
class RuleEngineConfig:
    """Configuration for rule-based AML monitoring thresholds."""

    # Structuring detection
    ctr_threshold: float = 10_000.0
    structuring_lower_bound: float = 8_000.0
    structuring_upper_bound: float = 9_999.99
    structuring_window_hours: int = 24
    structuring_min_count: int = 2

    # Rapid fund movement
    rapid_movement_window_hours: int = 4
    rapid_movement_min_amount: float = 5_000.0
    rapid_movement_outflow_ratio: float = 0.80

    # Round-tripping
    round_trip_max_hops: int = 5
    round_trip_time_window_days: int = 30
    round_trip_min_amount: float = 1_000.0

    # Geographic risk
    high_risk_jurisdictions: List[str] = field(default_factory=lambda: [
        "AF", "IR", "KP", "MM", "SY", "YE",  # FATF black list (illustrative)
        "AL", "BB", "BF", "CM", "CD", "GI", "HT", "JM", "JO", "ML",
        "MZ", "NI", "PK", "PA", "PH", "SN", "SS", "TZ", "TT", "UG",
        "VU", "VN",  # FATF grey list (illustrative)
    ])

    # Dormant account reactivation
    dormant_days_threshold: int = 90
    dormant_reactivation_min_amount: float = 5_000.0

    # Velocity rules
    velocity_windows: Dict[str, int] = field(default_factory=lambda: {
        "1h": 1,
        "6h": 6,
        "24h": 24,
        "7d": 168,
    })
    velocity_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "1h": 5,
        "6h": 15,
        "24h": 30,
        "7d": 100,
    })

    # Rule weights for ensemble scoring
    rule_weights: Dict[str, float] = field(default_factory=lambda: {
        "structuring": 0.25,
        "rapid_movement": 0.20,
        "round_tripping": 0.30,
        "geographic_risk": 0.10,
        "dormant_reactivation": 0.15,
        "velocity": 0.10,
    })


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""

    rolling_windows: List[str] = field(default_factory=lambda: [
        "1D", "7D", "30D", "90D",
    ])
    amount_percentiles: List[float] = field(default_factory=lambda: [
        0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
    ])
    time_of_day_bins: int = 6  # 4-hour bins
    round_amount_thresholds: List[float] = field(default_factory=lambda: [
        100.0, 500.0, 1_000.0, 5_000.0, 10_000.0,
    ])
    max_counterparties_for_fanout: int = 50
    behavioral_lookback_days: int = 90


@dataclass
class GraphAnalysisConfig:
    """Configuration for graph-based network analysis."""

    # Graph construction
    min_edge_weight: float = 100.0  # Minimum transaction amount for edge
    max_graph_nodes: int = 100_000  # Limit for in-memory processing
    edge_time_window_days: int = 30  # Build graph from recent N days

    # Community detection (Louvain)
    louvain_resolution: float = 1.0
    min_community_size: int = 3

    # Cycle detection
    max_cycle_length: int = 6
    cycle_detection_timeout_seconds: int = 300

    # Centrality
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 100
    top_k_central_nodes: int = 100

    # Subgraph extraction
    subgraph_max_hops: int = 3
    subgraph_max_nodes: int = 500

    # Visualization
    graph_layout: str = "spring"
    node_size_factor: float = 50.0
    edge_width_factor: float = 2.0


@dataclass
class ModelConfig:
    """Configuration for ML model training."""

    model_dir: Path = MODEL_DIR
    random_state: int = 42

    # LightGBM hyperparameters
    lgbm_params: Dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "is_unbalance": True,
        "scale_pos_weight": None,  # Computed dynamically from data
    })

    # XGBoost hyperparameters
    xgb_params: Dict = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "verbosity": 0,
        "scale_pos_weight": None,  # Computed dynamically from data
    })

    # Resampling
    use_smote: bool = True
    use_adasyn: bool = False
    smote_sampling_strategy: float = 0.1  # Target minority ratio
    smote_k_neighbors: int = 5

    # Threshold optimization
    target_recall: float = 0.80
    threshold_search_range: tuple = (0.01, 0.99)
    threshold_search_steps: int = 100

    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "aml-transaction-monitoring"


@dataclass
class LLMConfig:
    """Configuration for LLM-powered alert triage."""

    # Provider selection
    provider: str = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"

    # Anthropic
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Generation parameters
    temperature: float = 0.1  # Low temperature for factual, consistent output
    max_tokens: int = 2048
    request_timeout: int = 60

    # Triage parameters
    batch_size: int = 10
    max_retries: int = 3
    retry_delay_seconds: int = 5

    # Context limits
    max_transactions_in_context: int = 50
    max_counterparties_in_context: int = 20
    max_prior_alerts_in_context: int = 5

    # Hallucination check
    enable_hallucination_check: bool = True
    fact_verification_strictness: float = 0.90  # 90% of facts must be verified

    # Vector store for few-shot retrieval
    chromadb_collection: str = "aml_case_summaries"
    embedding_model: str = "all-MiniLM-L6-v2"
    few_shot_k: int = 3


@dataclass
class AlertConfig:
    """Configuration for alert generation and prioritization."""

    # Ensemble weights
    rule_score_weight: float = 0.30
    ml_score_weight: float = 0.45
    graph_score_weight: float = 0.25

    # Alert thresholds
    alert_threshold: float = 0.50  # Minimum ensemble score to generate alert
    high_severity_threshold: float = 0.80
    medium_severity_threshold: float = 0.60

    # Deduplication
    dedup_entity_window_hours: int = 72  # Same entity, same 72h window
    dedup_merge_strategy: str = "max_score"  # "max_score" or "sum_score"

    # SLA
    high_severity_sla_hours: int = 24
    medium_severity_sla_hours: int = 72
    low_severity_sla_hours: int = 168  # 7 days


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    rules: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    graph: GraphAnalysisConfig = field(default_factory=GraphAnalysisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"

    def setup_logging(self) -> None:
        """Configure logging for the entire application."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format=self.log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Suppress noisy third-party loggers
        for noisy_logger in ["py4j", "pyspark", "urllib3", "chromadb"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    def validate(self) -> None:
        """Validate configuration values and raise errors for invalid settings."""
        errors: List[str] = []

        if not 0.0 < self.data.test_size < 1.0:
            errors.append(f"test_size must be between 0 and 1, got {self.data.test_size}")

        if self.alert.rule_score_weight + self.alert.ml_score_weight + self.alert.graph_score_weight != 1.0:
            total = self.alert.rule_score_weight + self.alert.ml_score_weight + self.alert.graph_score_weight
            errors.append(f"Alert ensemble weights must sum to 1.0, got {total}")

        if self.llm.provider not in ("anthropic", "openai"):
            errors.append(f"LLM provider must be 'anthropic' or 'openai', got {self.llm.provider}")

        if self.llm.provider == "anthropic" and not self.llm.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set; LLM triage will fail for Anthropic provider")

        if self.llm.provider == "openai" and not self.llm.openai_api_key:
            logger.warning("OPENAI_API_KEY not set; LLM triage will fail for OpenAI provider")

        if errors:
            raise ValueError(f"Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.info("Configuration validated successfully")
