"""
End-to-End AML Transaction Monitoring Pipeline.

Orchestrates the full detection and triage workflow:

  Data Loading --> Rule Engine --> Feature Engineering --> Graph Analysis
  --> ML Scoring --> Alert Generation --> LLM Triage --> Compliance Reporting

Supports stage-by-stage execution with intermediate result saving,
PySpark integration for large-scale processing, configurable logging,
and comprehensive error handling.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


class AMLPipeline:
    """
    End-to-end AML monitoring pipeline.

    Orchestrates data loading, rule evaluation, feature engineering,
    graph analysis, ML scoring, alert generation, LLM-powered triage,
    and compliance reporting into a single configurable pipeline.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    # Stage identifiers in execution order
    STAGES = [
        "data_loading",
        "rule_engine",
        "feature_engineering",
        "graph_analysis",
        "ml_scoring",
        "alert_generation",
        "llm_triage",
        "compliance_reporting",
    ]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.outputs_dir = OUTPUTS_DIR
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._results: Dict[str, Any] = {}
        self._timings: Dict[str, float] = {}
        self._stage_status: Dict[str, str] = {s: "pending" for s in self.STAGES}

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        data_variant: str = "small",
        nrows: Optional[int] = None,
        use_spark: bool = False,
        stages: Optional[List[str]] = None,
        skip_llm: bool = False,
        save_intermediates: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full AML pipeline (or a subset of stages).

        Parameters
        ----------
        data_variant : str
            IBM AML dataset variant: 'small', 'medium', or 'large'.
        nrows : int, optional
            Limit the number of rows loaded (for development/testing).
        use_spark : bool
            Use PySpark for data loading (recommended for 'large' variant).
        stages : List[str], optional
            Specific stages to execute. If None, runs all stages.
        skip_llm : bool
            If True, skip the LLM triage stage (useful when no API key).
        save_intermediates : bool
            If True, save intermediate results after each stage.

        Returns
        -------
        Dict[str, Any]
            Pipeline results including alerts, triage results,
            compliance report, and performance metrics.
        """
        self.config.setup_logging()
        self.config.validate()

        stages_to_run = stages or self.STAGES
        if skip_llm and "llm_triage" in stages_to_run:
            stages_to_run = [s for s in stages_to_run if s != "llm_triage"]

        pipeline_start = time.time()

        logger.info("=" * 70)
        logger.info("AML TRANSACTION MONITORING PIPELINE")
        logger.info(f"  Variant: {data_variant} | Rows: {nrows or 'all'} | Spark: {use_spark}")
        logger.info(f"  Stages: {', '.join(stages_to_run)}")
        logger.info("=" * 70)

        # Store configuration parameters
        self._results["config"] = {
            "data_variant": data_variant,
            "nrows": nrows,
            "use_spark": use_spark,
            "stages": stages_to_run,
            "skip_llm": skip_llm,
            "started_at": datetime.utcnow().isoformat(),
        }

        try:
            for stage in stages_to_run:
                if stage not in self.STAGES:
                    logger.warning(f"Unknown stage '{stage}', skipping")
                    continue

                self._run_stage(stage, data_variant, nrows, use_spark, save_intermediates)

        except Exception as e:
            logger.error(f"Pipeline failed at stage: {e}", exc_info=True)
            self._results["error"] = str(e)
            self._results["failed_stage"] = next(
                (s for s, st in self._stage_status.items() if st == "running"), "unknown"
            )

        pipeline_elapsed = time.time() - pipeline_start
        self._timings["total_pipeline"] = pipeline_elapsed

        self._results["timings"] = self._timings.copy()
        self._results["stage_status"] = self._stage_status.copy()
        self._results["completed_at"] = datetime.utcnow().isoformat()

        logger.info("=" * 70)
        logger.info(f"Pipeline complete in {pipeline_elapsed:.1f}s")
        for stage, status in self._stage_status.items():
            elapsed = self._timings.get(stage, 0)
            logger.info(f"  {stage:25s} | {status:10s} | {elapsed:.1f}s")
        logger.info("=" * 70)

        return self._results

    def run_stage(self, stage_name: str, **kwargs: Any) -> Any:
        """
        Execute a single pipeline stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage to run.
        **kwargs : Any
            Additional arguments passed to the stage function.

        Returns
        -------
        Any
            Stage output.
        """
        self.config.setup_logging()

        if stage_name not in self.STAGES:
            raise ValueError(f"Unknown stage: {stage_name}. Valid stages: {self.STAGES}")

        return self._run_stage(
            stage_name,
            data_variant=kwargs.get("data_variant", "small"),
            nrows=kwargs.get("nrows"),
            use_spark=kwargs.get("use_spark", False),
            save_intermediates=kwargs.get("save_intermediates", True),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Stage Dispatcher
    # ─────────────────────────────────────────────────────────────────────

    def _run_stage(
        self,
        stage: str,
        data_variant: str,
        nrows: Optional[int],
        use_spark: bool,
        save_intermediates: bool,
    ) -> Any:
        """Dispatch and execute a pipeline stage."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STAGE: {stage.upper()}")
        logger.info(f"{'=' * 60}")

        self._stage_status[stage] = "running"
        start = time.time()

        try:
            if stage == "data_loading":
                result = self._stage_data_loading(data_variant, nrows, use_spark)
            elif stage == "rule_engine":
                result = self._stage_rule_engine()
            elif stage == "feature_engineering":
                result = self._stage_feature_engineering()
            elif stage == "graph_analysis":
                result = self._stage_graph_analysis()
            elif stage == "ml_scoring":
                result = self._stage_ml_scoring()
            elif stage == "alert_generation":
                result = self._stage_alert_generation()
            elif stage == "llm_triage":
                result = self._stage_llm_triage()
            elif stage == "compliance_reporting":
                result = self._stage_compliance_reporting()
            else:
                raise ValueError(f"No handler for stage: {stage}")

            elapsed = time.time() - start
            self._timings[stage] = elapsed
            self._stage_status[stage] = "completed"
            logger.info(f"Stage '{stage}' completed in {elapsed:.1f}s")

            # Save intermediate results
            if save_intermediates:
                self._save_intermediate(stage, result)

            return result

        except Exception as e:
            elapsed = time.time() - start
            self._timings[stage] = elapsed
            self._stage_status[stage] = "failed"
            logger.error(f"Stage '{stage}' failed after {elapsed:.1f}s: {e}", exc_info=True)
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Stage Implementations
    # ─────────────────────────────────────────────────────────────────────

    def _stage_data_loading(
        self, data_variant: str, nrows: Optional[int], use_spark: bool
    ) -> pd.DataFrame:
        """Stage 1: Load and preprocess transaction data."""
        from src.data_loader import AMLDataLoader

        loader = AMLDataLoader(self.config)

        try:
            df = loader.load(variant=data_variant, nrows=nrows, use_spark=use_spark)
        except FileNotFoundError:
            logger.warning(
                "Dataset file not found. Generating synthetic data for demonstration."
            )
            df = loader.create_synthetic_sample(n_samples=nrows or 10_000)

        # Time-based train/test split
        train_df, test_df = loader.time_based_split(df)

        # Data profile
        profile = loader.profile(df)

        self._results["data"] = {
            "full_df": df,
            "train_df": train_df,
            "test_df": test_df,
            "profile": profile,
        }

        logger.info(
            f"  Data loaded: {len(df):,} transactions | "
            f"Train: {len(train_df):,} | Test: {len(test_df):,}"
        )

        return df

    def _stage_rule_engine(self) -> pd.DataFrame:
        """Stage 2: Evaluate AML rules on transaction data."""
        from src.rule_engine import AMLRuleEngine

        data = self._results.get("data")
        if data is None:
            raise RuntimeError("Data not loaded. Run 'data_loading' stage first.")

        df = data["full_df"]
        engine = AMLRuleEngine(self.config)
        scored_df = engine.evaluate(df)

        self._results["rule_engine"] = {
            "scored_df": scored_df,
            "statistics": engine.get_rule_statistics(),
        }

        return scored_df

    def _stage_feature_engineering(self) -> pd.DataFrame:
        """Stage 3: Generate ML features from transaction data."""
        from src.feature_engineering import AMLFeatureEngineer

        rule_data = self._results.get("rule_engine")
        data = self._results.get("data")

        if data is None:
            raise RuntimeError("Data not loaded. Run 'data_loading' stage first.")

        df = data["full_df"]
        rule_scores = rule_data["scored_df"] if rule_data else None

        # Graph features may or may not be available
        graph_data = self._results.get("graph_analysis")
        graph_features = graph_data.get("features") if graph_data else None

        engineer = AMLFeatureEngineer(self.config)
        features = engineer.engineer_features(
            df, rule_scores=rule_scores, graph_features=graph_features
        )

        self._results["features"] = {
            "feature_matrix": features,
            "feature_names": engineer.feature_names,
            "feature_groups": engineer.get_feature_importance_groups(),
        }

        logger.info(f"  Feature matrix: {features.shape}")

        return features

    def _stage_graph_analysis(self) -> Dict[str, Any]:
        """Stage 4: Build transaction graph and compute graph features."""
        from src.graph_analysis import TransactionGraphAnalyzer

        data = self._results.get("data")
        if data is None:
            raise RuntimeError("Data not loaded. Run 'data_loading' stage first.")

        df = data["full_df"]
        analyzer = TransactionGraphAnalyzer(self.config)

        # Build graph
        graph = analyzer.build_graph(df)

        # Detect communities
        communities = analyzer.detect_communities()

        # Detect cycles
        cycles = analyzer.detect_cycles()

        # Compute centrality
        centrality = analyzer.compute_centrality()

        # Generate per-transaction graph features
        graph_features = analyzer.generate_graph_features(df)

        # Identify suspicious accounts
        suspicious_accounts = analyzer.identify_suspicious_accounts()

        # Graph summary
        summary = analyzer.get_graph_summary()

        self._results["graph_analysis"] = {
            "graph": graph,
            "communities": communities,
            "cycles": cycles,
            "centrality": centrality,
            "features": graph_features,
            "suspicious_accounts": suspicious_accounts,
            "summary": summary,
        }

        logger.info(f"  Graph: {summary.get('nodes', 0)} nodes, {summary.get('edges', 0)} edges")

        return self._results["graph_analysis"]

    def _stage_ml_scoring(self) -> Dict[str, Any]:
        """Stage 5: Train ML models and generate scores."""
        from src.model_training import AMLModelTrainer

        data = self._results.get("data")
        features_data = self._results.get("features")

        if data is None:
            raise RuntimeError("Data not loaded. Run 'data_loading' stage first.")

        if features_data is None:
            raise RuntimeError("Features not generated. Run 'feature_engineering' stage first.")

        train_df = data["train_df"]
        test_df = data["test_df"]
        features = features_data["feature_matrix"]
        feature_names = features_data["feature_names"]

        # Split features aligned with train/test
        train_idx = train_df.index
        test_idx = test_df.index

        X_train = features.loc[features.index.isin(train_idx)].copy()
        y_train = train_df.loc[X_train.index, "is_laundering"]
        X_test = features.loc[features.index.isin(test_idx)].copy()
        y_test = test_df.loc[X_test.index, "is_laundering"]

        # Ensure alignment
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        trainer = AMLModelTrainer(self.config)

        training_results = trainer.train(
            X_train, y_train, X_test, y_test, feature_names=feature_names
        )

        # Score entire dataset
        all_scores = trainer.predict_proba(features.fillna(0))
        predictions = trainer.predict(features.fillna(0))

        # Save model
        try:
            model_path = trainer.save_model()
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
            model_path = None

        self._results["ml_scoring"] = {
            "trainer": trainer,
            "training_results": training_results,
            "all_scores": all_scores,
            "predictions": predictions,
            "test_scores": trainer.predict_proba(X_test.fillna(0)),
            "y_test": y_test.values,
            "model_path": str(model_path) if model_path else None,
        }

        logger.info(
            f"  Best model: {training_results.get('best_model', 'N/A')} | "
            f"Threshold: {training_results.get('best_threshold', 0.5):.4f}"
        )

        return self._results["ml_scoring"]

    def _stage_alert_generation(self) -> List:
        """Stage 6: Generate and prioritize alerts."""
        from src.alert_generator import AMLAlertGenerator

        rule_data = self._results.get("rule_engine")
        ml_data = self._results.get("ml_scoring")
        graph_data = self._results.get("graph_analysis")

        if rule_data is None:
            raise RuntimeError("Rule engine not run. Run 'rule_engine' stage first.")

        scored_df = rule_data["scored_df"]
        ml_scores = ml_data["all_scores"] if ml_data else None
        graph_features = graph_data.get("features") if graph_data else None
        suspicious_accounts = graph_data.get("suspicious_accounts") if graph_data else None

        generator = AMLAlertGenerator(self.config)
        alerts = generator.generate_alerts(
            scored_df,
            ml_scores=ml_scores,
            graph_features=graph_features,
            graph_suspicious_accounts=suspicious_accounts,
        )

        self._results["alerts"] = {
            "alert_objects": alerts,
            "alert_queue": generator.get_alert_queue(),
            "statistics": generator.get_statistics(),
            "generator": generator,
        }

        logger.info(f"  Alerts generated: {len(alerts)}")

        return alerts

    def _stage_llm_triage(self) -> List[Dict[str, Any]]:
        """Stage 7: LLM-powered alert triage."""
        from src.llm_alert_triage import LLMAlertTriageSystem

        alerts_data = self._results.get("alerts")
        if alerts_data is None:
            raise RuntimeError("Alerts not generated. Run 'alert_generation' stage first.")

        alerts = alerts_data["alert_objects"]

        if not alerts:
            logger.warning("No alerts to triage")
            self._results["triage"] = {"results": [], "statistics": {}}
            return []

        triage_system = LLMAlertTriageSystem(self.config)

        try:
            triage_system.initialize()
        except Exception as e:
            logger.warning(
                f"LLM initialization failed: {e}. "
                "Generating fallback triage results."
            )
            fallback_results = [
                triage_system._create_fallback_result(alert, str(e))
                for alert in alerts
            ]
            self._results["triage"] = {
                "results": fallback_results,
                "statistics": {"total_triaged": len(fallback_results), "fallback": True},
            }
            return fallback_results

        # Triage alerts (prioritize HIGH severity)
        high_alerts = [a for a in alerts if a.severity == "HIGH"]
        medium_alerts = [a for a in alerts if a.severity == "MEDIUM"]
        prioritized = high_alerts + medium_alerts

        # Limit batch size for cost control
        max_triage = min(len(prioritized), 50)
        batch = prioritized[:max_triage]

        logger.info(f"  Triaging {len(batch)} alerts (of {len(alerts)} total)")

        triage_results = triage_system.triage_batch(batch)

        self._results["triage"] = {
            "results": triage_results,
            "statistics": triage_system.get_triage_statistics(),
            "triage_export": triage_system.export_triage_results(),
        }

        return triage_results

    def _stage_compliance_reporting(self) -> Dict[str, Any]:
        """Stage 8: Generate compliance reports."""
        from src.compliance_reporter import ComplianceReporter

        alerts_data = self._results.get("alerts")
        triage_data = self._results.get("triage")
        ml_data = self._results.get("ml_scoring")

        if alerts_data is None:
            raise RuntimeError("Alerts not generated. Run 'alert_generation' stage first.")

        alerts = alerts_data["alert_objects"]
        triage_results = triage_data.get("results") if triage_data else None
        model_metrics = ml_data.get("training_results") if ml_data else None

        reporter = ComplianceReporter(self.config)

        compliance_package = reporter.generate_full_compliance_package(
            alerts=alerts,
            triage_results=triage_results,
            model_metrics=model_metrics,
        )

        self._results["compliance_report"] = compliance_package

        logger.info(
            f"  Compliance package: {compliance_package.get('sar_draft_count', 0)} SAR drafts"
        )

        return compliance_package

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(self) -> Dict[str, Any]:
        """
        Run the full evaluation suite on pipeline results.

        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation metrics.
        """
        from src.evaluation import AMLSystemEvaluator

        evaluator = AMLSystemEvaluator(self.config)

        # Transaction-level evaluation
        ml_data = self._results.get("ml_scoring")
        if ml_data:
            evaluator.evaluate_transaction_level(
                y_true=ml_data["y_test"],
                y_prob=ml_data["test_scores"],
                threshold=ml_data["training_results"].get("best_threshold", 0.5),
                model_name=ml_data["training_results"].get("best_model", "best"),
            )

        # Alert-level evaluation
        alerts_data = self._results.get("alerts")
        if alerts_data:
            evaluator.evaluate_alert_level(alerts_data["alert_objects"])

        # LLM triage evaluation
        triage_data = self._results.get("triage")
        if triage_data and triage_data.get("results"):
            evaluator.evaluate_llm_triage(triage_data["results"])

        # Rule engine evaluation
        rule_data = self._results.get("rule_engine")
        data = self._results.get("data")
        if rule_data and data:
            evaluator.evaluate_rule_engine(
                rule_data["scored_df"],
                y_true=data["full_df"]["is_laundering"].values,
            )

        # System comparison
        if ml_data and rule_data:
            scored_df = rule_data["scored_df"]
            full_df = data["full_df"]
            y_true = full_df["is_laundering"].values
            rule_scores = scored_df["rule_composite_score"].values
            ml_scores_all = ml_data["all_scores"]

            # Compute hybrid scores
            cfg = self.config.alert
            hybrid_scores = (
                cfg.rule_score_weight * rule_scores
                + cfg.ml_score_weight * ml_scores_all
            )
            # Normalize to [0, 1]
            hybrid_max = hybrid_scores.max()
            if hybrid_max > 0:
                hybrid_scores = hybrid_scores / hybrid_max

            evaluator.compare_systems(
                y_true=y_true,
                rule_scores=rule_scores,
                ml_scores=ml_scores_all,
                hybrid_scores=hybrid_scores,
            )

        # Export and log
        evaluator.export_results()
        summary = evaluator.generate_summary()
        logger.info("\n" + summary)

        eval_results = evaluator.get_all_results()
        self._results["evaluation"] = eval_results

        return eval_results

    # ─────────────────────────────────────────────────────────────────────
    # Intermediate Result Saving
    # ─────────────────────────────────────────────────────────────────────

    def _save_intermediate(self, stage: str, result: Any) -> None:
        """Save intermediate results for a completed stage."""
        try:
            stage_dir = self.outputs_dir / "intermediates"
            stage_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            if isinstance(result, pd.DataFrame):
                path = stage_dir / f"{stage}_{timestamp}.parquet"
                result.to_parquet(path, index=False)
                logger.debug(f"Saved intermediate: {path}")
            elif isinstance(result, dict):
                # Save only JSON-serializable parts
                serializable = {}
                for k, v in result.items():
                    if isinstance(v, (str, int, float, bool, list)):
                        serializable[k] = v
                    elif isinstance(v, dict):
                        # Try to serialize nested dicts
                        try:
                            json.dumps(v, default=str)
                            serializable[k] = v
                        except (TypeError, ValueError):
                            serializable[k] = str(type(v))
                    else:
                        serializable[k] = str(type(v))

                path = stage_dir / f"{stage}_{timestamp}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, indent=2, default=str)
                logger.debug(f"Saved intermediate: {path}")

        except Exception as e:
            logger.debug(f"Could not save intermediate for '{stage}': {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────

    @property
    def results(self) -> Dict[str, Any]:
        """Return all pipeline results."""
        return self._results

    @property
    def stage_status(self) -> Dict[str, str]:
        """Return the status of each pipeline stage."""
        return self._stage_status.copy()

    @property
    def timings(self) -> Dict[str, float]:
        """Return the timing for each pipeline stage."""
        return self._timings.copy()
