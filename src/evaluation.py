"""
End-to-End Evaluation Framework for the AML Transaction Monitoring System.

Provides comprehensive evaluation at multiple levels:

  1. Transaction-level ML metrics: precision, recall, F1, AUC-ROC, AUC-PR
  2. Alert-level metrics: alert-to-SAR conversion rate, false alert rate
  3. LLM triage quality: summary accuracy, recommendation consistency
  4. Rule engine analysis: hit rates, overlap analysis
  5. System comparison: rules-only vs. ML-only vs. hybrid
  6. Results logging and export
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.alert_generator import Alert
from src.config import Config

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


class AMLSystemEvaluator:
    """
    Evaluates the AML monitoring system across all detection layers.

    Computes transaction-level ML metrics, alert-level compliance metrics,
    LLM triage quality scores, rule engine coverage analysis, and
    comparative performance of rules-only vs. ML-only vs. hybrid systems.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._evaluation_results: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────
    # 1. Transaction-Level ML Metrics
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_transaction_level(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Compute transaction-level classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels (0 = legitimate, 1 = laundering).
        y_prob : np.ndarray
            Predicted probabilities for the positive class.
        threshold : float
            Decision threshold for converting probabilities to labels.
        model_name : str
            Identifier for the model being evaluated.

        Returns
        -------
        Dict[str, Any]
            Comprehensive transaction-level metrics including AUC-ROC,
            AUC-PR, precision, recall, F1, confusion matrix, and
            recall at fixed FPR levels.
        """
        logger.info(f"Evaluating transaction-level metrics for '{model_name}' (threshold={threshold:.4f})")

        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= threshold).astype(int)

        # ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        auc_roc = float(roc_auc_score(y_true, y_prob))

        # Precision-Recall curve and AUC-PR
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        auc_pr = float(average_precision_score(y_true, y_prob))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Recall at specific FPR levels (critical for AML)
        recall_at_1pct_fpr = float(np.interp(0.01, fpr, tpr))
        recall_at_5pct_fpr = float(np.interp(0.05, fpr, tpr))
        recall_at_10pct_fpr = float(np.interp(0.10, fpr, tpr))

        # Precision at specific recall levels
        precision_at_80_recall = self._precision_at_recall(pr_precision, pr_recall, 0.80)
        precision_at_90_recall = self._precision_at_recall(pr_precision, pr_recall, 0.90)

        metrics = {
            "model_name": model_name,
            "threshold": threshold,
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "n_negative": int(len(y_true) - y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "recall_at_1pct_fpr": recall_at_1pct_fpr,
            "recall_at_5pct_fpr": recall_at_5pct_fpr,
            "recall_at_10pct_fpr": recall_at_10pct_fpr,
            "precision_at_80_recall": precision_at_80_recall,
            "precision_at_90_recall": precision_at_90_recall,
            "classification_report": classification_report(
                y_true, y_pred, target_names=["legitimate", "laundering"], output_dict=True
            ),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": pr_precision.tolist(), "recall": pr_recall.tolist()},
        }

        self._evaluation_results[f"transaction_level_{model_name}"] = metrics

        logger.info(
            f"  AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f} | "
            f"F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | FPR: {metrics['false_positive_rate']:.4f}"
        )

        return metrics

    @staticmethod
    def _precision_at_recall(
        precision_curve: np.ndarray, recall_curve: np.ndarray, target_recall: float
    ) -> float:
        """Find the precision at a specific recall level on the PR curve."""
        # PR curve has recall sorted in descending order; find the threshold
        # where recall just meets the target.
        valid = recall_curve >= target_recall
        if valid.any():
            return float(precision_curve[valid][-1])
        return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # 2. Alert-Level Metrics
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_alert_level(
        self,
        alerts: List[Alert],
        ground_truth_sars: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute alert-level compliance metrics.

        Parameters
        ----------
        alerts : List[Alert]
            Generated alerts from the system.
        ground_truth_sars : List[str], optional
            Entity IDs that are known to require SAR filing (for
            computing alert-to-SAR conversion accuracy).

        Returns
        -------
        Dict[str, Any]
            Alert-level metrics including conversion rates, false alert
            rates, severity distribution, and SLA compliance.
        """
        logger.info(f"Evaluating alert-level metrics ({len(alerts)} alerts)")

        if not alerts:
            return {"total_alerts": 0, "error": "No alerts to evaluate"}

        total = len(alerts)
        high = sum(1 for a in alerts if a.severity == "HIGH")
        medium = sum(1 for a in alerts if a.severity == "MEDIUM")
        low = sum(1 for a in alerts if a.severity == "LOW")

        escalated = sum(1 for a in alerts if a.status == "ESCALATED")
        closed = sum(1 for a in alerts if a.status == "CLOSED")
        closed_no_action = sum(
            1 for a in alerts
            if a.status == "CLOSED" and a.disposition == "CLOSED_NO_ACTION"
        )

        # Alert-to-SAR conversion rate
        sar_filed = sum(
            1 for a in alerts
            if a.disposition == "SAR_FILED" or a.status == "ESCALATED"
        )
        alert_to_sar_rate = sar_filed / total if total > 0 else 0.0

        # False alert rate (closed without action / total closed)
        false_alert_rate = closed_no_action / closed if closed > 0 else 0.0

        # Scores by severity
        scores = [a.ensemble_score for a in alerts]
        high_scores = [a.ensemble_score for a in alerts if a.severity == "HIGH"]
        medium_scores = [a.ensemble_score for a in alerts if a.severity == "MEDIUM"]
        low_scores = [a.ensemble_score for a in alerts if a.severity == "LOW"]

        # Ground-truth evaluation (if available)
        gt_metrics = {}
        if ground_truth_sars:
            gt_set = set(ground_truth_sars)
            alerted_entities = {a.entity_id for a in alerts}
            true_positives = alerted_entities & gt_set
            false_negatives = gt_set - alerted_entities
            false_positives = alerted_entities - gt_set

            gt_precision = len(true_positives) / len(alerted_entities) if alerted_entities else 0
            gt_recall = len(true_positives) / len(gt_set) if gt_set else 0
            gt_f1 = (
                2 * gt_precision * gt_recall / (gt_precision + gt_recall)
                if (gt_precision + gt_recall) > 0
                else 0
            )

            gt_metrics = {
                "ground_truth_entities": len(gt_set),
                "alerted_entities": len(alerted_entities),
                "true_positive_entities": len(true_positives),
                "false_negative_entities": len(false_negatives),
                "false_positive_entities": len(false_positives),
                "entity_precision": gt_precision,
                "entity_recall": gt_recall,
                "entity_f1": gt_f1,
            }

        now = datetime.utcnow()
        sla_breached = sum(1 for a in alerts if a.status == "OPEN" and a.sla_deadline < now)

        metrics = {
            "total_alerts": total,
            "severity_distribution": {
                "HIGH": high,
                "MEDIUM": medium,
                "LOW": low,
            },
            "alert_to_sar_conversion_rate": alert_to_sar_rate,
            "false_alert_rate": false_alert_rate,
            "escalation_count": escalated,
            "closure_count": closed,
            "closed_no_action": closed_no_action,
            "sla_breached": sla_breached,
            "sla_breach_rate": sla_breached / total if total > 0 else 0.0,
            "score_statistics": {
                "overall_mean": float(np.mean(scores)),
                "overall_std": float(np.std(scores)),
                "high_mean": float(np.mean(high_scores)) if high_scores else 0.0,
                "medium_mean": float(np.mean(medium_scores)) if medium_scores else 0.0,
                "low_mean": float(np.mean(low_scores)) if low_scores else 0.0,
            },
            "rules_distribution": self._compute_rule_distribution(alerts),
            "ground_truth_evaluation": gt_metrics,
        }

        self._evaluation_results["alert_level"] = metrics

        logger.info(
            f"  Alerts: {total} (H:{high} M:{medium} L:{low}) | "
            f"SAR rate: {alert_to_sar_rate:.2%} | "
            f"False alert rate: {false_alert_rate:.2%}"
        )

        return metrics

    @staticmethod
    def _compute_rule_distribution(alerts: List[Alert]) -> Dict[str, int]:
        """Compute how many alerts each rule contributed to."""
        distribution: Dict[str, int] = defaultdict(int)
        for alert in alerts:
            for rule in alert.triggered_rules:
                distribution[rule] += 1
        return dict(distribution)

    # ─────────────────────────────────────────────────────────────────────
    # 3. LLM Triage Quality Metrics
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_llm_triage(
        self,
        triage_results: List[Dict[str, Any]],
        expert_dispositions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of LLM-generated triage outputs.

        Parameters
        ----------
        triage_results : List[Dict[str, Any]]
            LLM triage output for each alert.
        expert_dispositions : Dict[str, str], optional
            Mapping of alert_id to expert-assigned disposition for
            measuring agreement.

        Returns
        -------
        Dict[str, Any]
            Triage quality metrics including summary accuracy,
            recommendation consistency, verification pass rate,
            and disposition agreement.
        """
        logger.info(f"Evaluating LLM triage quality ({len(triage_results)} results)")

        if not triage_results:
            return {"total_triaged": 0, "error": "No triage results to evaluate"}

        total = len(triage_results)

        # Disposition distribution
        disposition_counts: Dict[str, int] = defaultdict(int)
        for result in triage_results:
            disp = result.get("recommended_disposition", "UNKNOWN")
            disposition_counts[disp] += 1

        # Confidence distribution
        confidence_counts: Dict[str, int] = defaultdict(int)
        for result in triage_results:
            conf = result.get("confidence_level", "UNKNOWN")
            confidence_counts[conf] += 1

        # Typology distribution
        typology_counts: Dict[str, int] = defaultdict(int)
        for result in triage_results:
            typo = result.get("typology_classification", "UNKNOWN")
            typology_counts[typo] += 1

        # Fact verification analysis
        verification_scores = []
        verification_passed = 0
        for result in triage_results:
            fv = result.get("fact_verification", {})
            if fv:
                score = fv.get("verification_score", 0.0)
                verification_scores.append(score)
                if fv.get("passed", False):
                    verification_passed += 1

        mean_verification = float(np.mean(verification_scores)) if verification_scores else 0.0
        verification_pass_rate = verification_passed / total if total > 0 else 0.0

        # Summary quality heuristics
        summary_lengths = [
            len(r.get("narrative_summary", "")) for r in triage_results
        ]
        risk_indicator_counts = [
            len(r.get("risk_indicators", [])) for r in triage_results
        ]
        has_timeline = sum(1 for r in triage_results if r.get("timeline"))
        has_rationale = sum(
            1 for r in triage_results
            if r.get("disposition_rationale") and len(r["disposition_rationale"]) > 20
        )

        # Parse errors
        parse_errors = sum(1 for r in triage_results if r.get("parse_error"))

        # Expert disposition agreement
        agreement_metrics: Dict[str, Any] = {}
        if expert_dispositions:
            correct = 0
            compared = 0
            confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for result in triage_results:
                alert_id = result.get("alert_id", "")
                if alert_id in expert_dispositions:
                    predicted = result.get("recommended_disposition", "UNKNOWN")
                    actual = expert_dispositions[alert_id]
                    compared += 1
                    confusion[actual][predicted] += 1
                    if predicted == actual:
                        correct += 1

            agreement_rate = correct / compared if compared > 0 else 0.0
            agreement_metrics = {
                "compared": compared,
                "correct": correct,
                "agreement_rate": agreement_rate,
                "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
            }

        # Recommendation consistency: same entity should get similar dispositions
        entity_dispositions: Dict[str, List[str]] = defaultdict(list)
        for result in triage_results:
            entity_id = result.get("entity_id", "")
            disp = result.get("recommended_disposition", "UNKNOWN")
            entity_dispositions[entity_id].append(disp)

        consistency_scores = []
        for entity, disps in entity_dispositions.items():
            if len(disps) > 1:
                most_common = max(set(disps), key=disps.count)
                consistency = disps.count(most_common) / len(disps)
                consistency_scores.append(consistency)

        mean_consistency = float(np.mean(consistency_scores)) if consistency_scores else 1.0

        metrics = {
            "total_triaged": total,
            "disposition_distribution": dict(disposition_counts),
            "confidence_distribution": dict(confidence_counts),
            "typology_distribution": dict(typology_counts),
            "fact_verification": {
                "mean_score": mean_verification,
                "pass_rate": verification_pass_rate,
                "n_verified": len(verification_scores),
            },
            "summary_quality": {
                "mean_length_chars": float(np.mean(summary_lengths)) if summary_lengths else 0.0,
                "mean_risk_indicators": float(np.mean(risk_indicator_counts)),
                "pct_with_timeline": has_timeline / total if total > 0 else 0.0,
                "pct_with_rationale": has_rationale / total if total > 0 else 0.0,
                "parse_error_rate": parse_errors / total if total > 0 else 0.0,
            },
            "recommendation_consistency": mean_consistency,
            "expert_agreement": agreement_metrics,
        }

        self._evaluation_results["llm_triage"] = metrics

        logger.info(
            f"  Verification pass rate: {verification_pass_rate:.2%} | "
            f"Consistency: {mean_consistency:.2%} | "
            f"Parse errors: {parse_errors}/{total}"
        )

        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # 4. Rule Engine Analysis
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_rule_engine(
        self,
        df: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Analyze rule engine hit rates, overlap, and effectiveness.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with rule score columns (rule_*_score).
        y_true : np.ndarray, optional
            Ground-truth labels for computing per-rule precision/recall.

        Returns
        -------
        Dict[str, Any]
            Rule engine analysis including per-rule hit rates,
            rule overlap matrix, and per-rule precision/recall.
        """
        logger.info("Evaluating rule engine performance")

        rule_cols = [c for c in df.columns if c.startswith("rule_") and c.endswith("_score")]

        if not rule_cols:
            logger.warning("No rule score columns found in dataframe")
            return {"error": "No rule score columns found"}

        total_txns = len(df)

        # Per-rule hit rates
        rule_hit_rates: Dict[str, Dict[str, Any]] = {}
        for col in rule_cols:
            rule_name = col.replace("rule_", "").replace("_score", "")
            triggered = (df[col] > 0).sum()
            mean_score = float(df[col][df[col] > 0].mean()) if triggered > 0 else 0.0

            rule_metrics: Dict[str, Any] = {
                "triggered_count": int(triggered),
                "hit_rate": triggered / total_txns if total_txns > 0 else 0.0,
                "mean_score_when_triggered": mean_score,
            }

            # Per-rule precision and recall if ground truth is available
            if y_true is not None:
                y_true_arr = np.asarray(y_true, dtype=int)
                rule_pred = (df[col] > 0).astype(int).values
                rule_tp = int(((rule_pred == 1) & (y_true_arr == 1)).sum())
                rule_fp = int(((rule_pred == 1) & (y_true_arr == 0)).sum())
                rule_fn = int(((rule_pred == 0) & (y_true_arr == 1)).sum())

                rule_precision = rule_tp / (rule_tp + rule_fp) if (rule_tp + rule_fp) > 0 else 0.0
                rule_recall = rule_tp / (rule_tp + rule_fn) if (rule_tp + rule_fn) > 0 else 0.0
                rule_f1 = (
                    2 * rule_precision * rule_recall / (rule_precision + rule_recall)
                    if (rule_precision + rule_recall) > 0
                    else 0.0
                )

                rule_metrics.update({
                    "true_positives": rule_tp,
                    "false_positives": rule_fp,
                    "false_negatives": rule_fn,
                    "precision": rule_precision,
                    "recall": rule_recall,
                    "f1": rule_f1,
                })

            rule_hit_rates[rule_name] = rule_metrics

        # Rule overlap analysis: how often do multiple rules fire together?
        triggered_matrix = pd.DataFrame(index=df.index)
        rule_names = []
        for col in rule_cols:
            name = col.replace("rule_", "").replace("_score", "")
            triggered_matrix[name] = (df[col] > 0).astype(int)
            rule_names.append(name)

        overlap_matrix = {}
        for i, rule_a in enumerate(rule_names):
            overlap_matrix[rule_a] = {}
            for j, rule_b in enumerate(rule_names):
                both = ((triggered_matrix[rule_a] == 1) & (triggered_matrix[rule_b] == 1)).sum()
                either = ((triggered_matrix[rule_a] == 1) | (triggered_matrix[rule_b] == 1)).sum()
                jaccard = both / either if either > 0 else 0.0
                overlap_matrix[rule_a][rule_b] = {
                    "co_occurrence": int(both),
                    "jaccard_similarity": float(jaccard),
                }

        # Multi-rule triggers
        multi_rule_counts = triggered_matrix.sum(axis=1)
        multi_rule_dist = multi_rule_counts.value_counts().sort_index().to_dict()
        multi_rule_dist = {str(int(k)): int(v) for k, v in multi_rule_dist.items()}

        # Composite rule score analysis
        composite_col = "rule_composite_score"
        composite_metrics = {}
        if composite_col in df.columns:
            composite = df[composite_col]
            flagged = (composite > 0.3).sum()
            composite_metrics = {
                "mean_score": float(composite.mean()),
                "std_score": float(composite.std()),
                "flagged_count": int(flagged),
                "flagged_rate": flagged / total_txns if total_txns > 0 else 0.0,
            }

            if y_true is not None:
                y_true_arr = np.asarray(y_true, dtype=int)
                comp_pred = (composite > 0.3).astype(int).values
                comp_tp = int(((comp_pred == 1) & (y_true_arr == 1)).sum())
                comp_fp = int(((comp_pred == 1) & (y_true_arr == 0)).sum())
                comp_fn = int(((comp_pred == 0) & (y_true_arr == 1)).sum())
                comp_prec = comp_tp / (comp_tp + comp_fp) if (comp_tp + comp_fp) > 0 else 0.0
                comp_rec = comp_tp / (comp_tp + comp_fn) if (comp_tp + comp_fn) > 0 else 0.0

                composite_metrics.update({
                    "precision": comp_prec,
                    "recall": comp_rec,
                    "f1": (
                        2 * comp_prec * comp_rec / (comp_prec + comp_rec)
                        if (comp_prec + comp_rec) > 0
                        else 0.0
                    ),
                })

        metrics = {
            "total_transactions": total_txns,
            "per_rule_metrics": rule_hit_rates,
            "rule_overlap_matrix": overlap_matrix,
            "multi_rule_trigger_distribution": multi_rule_dist,
            "composite_score_analysis": composite_metrics,
        }

        self._evaluation_results["rule_engine"] = metrics

        for rule_name, rm in rule_hit_rates.items():
            logger.info(
                f"  Rule '{rule_name}': hit_rate={rm['hit_rate']:.4f}, "
                f"precision={rm.get('precision', 'N/A')}, recall={rm.get('recall', 'N/A')}"
            )

        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # 5. System Comparison: Rules-Only vs. ML-Only vs. Hybrid
    # ─────────────────────────────────────────────────────────────────────

    def compare_systems(
        self,
        y_true: np.ndarray,
        rule_scores: np.ndarray,
        ml_scores: np.ndarray,
        hybrid_scores: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compare detection performance of rules-only, ML-only, and hybrid
        system configurations.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels (0/1).
        rule_scores : np.ndarray
            Rule engine composite scores (0 to 1).
        ml_scores : np.ndarray
            ML model probability scores (0 to 1).
        hybrid_scores : np.ndarray
            Ensemble/hybrid scores combining all layers (0 to 1).
        thresholds : Dict[str, float], optional
            Decision thresholds for each system. Defaults to
            {"rules": 0.3, "ml": 0.5, "hybrid": 0.5}.

        Returns
        -------
        Dict[str, Any]
            Comparative metrics for each system configuration.
        """
        logger.info("Comparing system configurations: rules-only vs. ML-only vs. hybrid")

        thresholds = thresholds or {"rules": 0.3, "ml": 0.5, "hybrid": 0.5}

        systems = {
            "rules_only": (np.asarray(rule_scores, dtype=float), thresholds.get("rules", 0.3)),
            "ml_only": (np.asarray(ml_scores, dtype=float), thresholds.get("ml", 0.5)),
            "hybrid": (np.asarray(hybrid_scores, dtype=float), thresholds.get("hybrid", 0.5)),
        }

        y_true = np.asarray(y_true, dtype=int)
        comparison: Dict[str, Dict[str, Any]] = {}

        for system_name, (scores, threshold) in systems.items():
            y_pred = (scores >= threshold).astype(int)

            # AUC metrics
            try:
                auc_roc = float(roc_auc_score(y_true, scores))
            except ValueError:
                auc_roc = 0.0

            try:
                auc_pr = float(average_precision_score(y_true, scores))
            except ValueError:
                auc_pr = 0.0

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            # Recall at fixed FPR
            try:
                fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
                recall_at_1pct = float(np.interp(0.01, fpr_arr, tpr_arr))
                recall_at_5pct = float(np.interp(0.05, fpr_arr, tpr_arr))
            except ValueError:
                recall_at_1pct = 0.0
                recall_at_5pct = 0.0

            comparison[system_name] = {
                "threshold": threshold,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "false_positive_rate": fpr_val,
                "recall_at_1pct_fpr": recall_at_1pct,
                "recall_at_5pct_fpr": recall_at_5pct,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "alerts_generated": int(y_pred.sum()),
            }

        # Compute lift metrics (hybrid over rules-only)
        rules_auc = comparison["rules_only"]["auc_roc"]
        ml_auc = comparison["ml_only"]["auc_roc"]
        hybrid_auc = comparison["hybrid"]["auc_roc"]

        lift = {
            "hybrid_over_rules_auc_roc": hybrid_auc - rules_auc,
            "hybrid_over_ml_auc_roc": hybrid_auc - ml_auc,
            "hybrid_over_rules_auc_pr": (
                comparison["hybrid"]["auc_pr"] - comparison["rules_only"]["auc_pr"]
            ),
            "hybrid_over_ml_auc_pr": (
                comparison["hybrid"]["auc_pr"] - comparison["ml_only"]["auc_pr"]
            ),
            "hybrid_over_rules_recall_at_5pct_fpr": (
                comparison["hybrid"]["recall_at_5pct_fpr"]
                - comparison["rules_only"]["recall_at_5pct_fpr"]
            ),
        }

        result = {
            "system_metrics": comparison,
            "lift_analysis": lift,
            "best_auc_roc": max(comparison, key=lambda s: comparison[s]["auc_roc"]),
            "best_auc_pr": max(comparison, key=lambda s: comparison[s]["auc_pr"]),
            "best_f1": max(comparison, key=lambda s: comparison[s]["f1"]),
        }

        self._evaluation_results["system_comparison"] = result

        for system_name, m in comparison.items():
            logger.info(
                f"  {system_name:12s} | AUC-ROC: {m['auc_roc']:.4f} | "
                f"AUC-PR: {m['auc_pr']:.4f} | F1: {m['f1']:.4f} | "
                f"Recall@5%FPR: {m['recall_at_5pct_fpr']:.4f}"
            )

        logger.info(f"  Best AUC-ROC: {result['best_auc_roc']} | Best AUC-PR: {result['best_auc_pr']}")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # 6. Results Logging and Export
    # ─────────────────────────────────────────────────────────────────────

    def get_all_results(self) -> Dict[str, Any]:
        """
        Return all evaluation results accumulated so far.

        Returns
        -------
        Dict[str, Any]
            All stored evaluation results keyed by evaluation type.
        """
        return self._evaluation_results.copy()

    def export_results(
        self,
        output_path: Optional[Path] = None,
        format: str = "json",
    ) -> Path:
        """
        Export evaluation results to a file.

        Parameters
        ----------
        output_path : Path, optional
            Output file path. Defaults to outputs/evaluation_<timestamp>.json.
        format : str
            Export format: 'json' or 'csv'. Default is 'json'.

        Returns
        -------
        Path
            Path to the exported file.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_path = output_path or self.results_dir / f"evaluation_{timestamp}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Make results JSON-serializable
            serializable = self._make_serializable(self._evaluation_results)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, default=str)

        elif format == "csv":
            output_path = output_path or self.results_dir / f"evaluation_{timestamp}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Flatten nested dict for CSV
            flat = self._flatten_dict(self._evaluation_results)
            df = pd.DataFrame([flat])
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'json' or 'csv'.")

        logger.info(f"Evaluation results exported to {output_path}")
        return output_path

    def log_to_mlflow(self) -> None:
        """
        Log evaluation metrics to MLflow for experiment tracking.

        Logs all scalar metrics from the evaluation results.
        """
        try:
            import mlflow

            flat = self._flatten_dict(self._evaluation_results)

            for key, value in flat.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    # MLflow metric names can only contain alphanumeric, /, -, ., _
                    safe_key = key.replace(" ", "_")[:250]
                    try:
                        mlflow.log_metric(safe_key, value)
                    except Exception:
                        pass  # Skip metrics that MLflow cannot log

            logger.info("Evaluation metrics logged to MLflow")

        except ImportError:
            logger.warning("MLflow not installed. Skipping MLflow logging.")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def generate_summary(self) -> str:
        """
        Generate a human-readable summary of all evaluation results.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "=" * 70,
            "AML SYSTEM EVALUATION SUMMARY",
            f"Generated: {datetime.utcnow().isoformat()}",
            "=" * 70,
        ]

        # Transaction-level metrics
        for key, metrics in self._evaluation_results.items():
            if key.startswith("transaction_level_"):
                model_name = metrics.get("model_name", key)
                lines.append(f"\n--- Transaction-Level: {model_name} ---")
                lines.append(f"  AUC-ROC:   {metrics.get('auc_roc', 0):.4f}")
                lines.append(f"  AUC-PR:    {metrics.get('auc_pr', 0):.4f}")
                lines.append(f"  F1:        {metrics.get('f1', 0):.4f}")
                lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
                lines.append(f"  Recall:    {metrics.get('recall', 0):.4f}")
                lines.append(f"  FPR:       {metrics.get('false_positive_rate', 0):.4f}")
                lines.append(f"  Recall@1%FPR: {metrics.get('recall_at_1pct_fpr', 0):.4f}")
                lines.append(f"  Recall@5%FPR: {metrics.get('recall_at_5pct_fpr', 0):.4f}")

        # Alert-level
        alert_metrics = self._evaluation_results.get("alert_level")
        if alert_metrics:
            lines.append(f"\n--- Alert-Level Metrics ---")
            lines.append(f"  Total alerts: {alert_metrics.get('total_alerts', 0)}")
            sd = alert_metrics.get("severity_distribution", {})
            lines.append(f"  HIGH: {sd.get('HIGH', 0)} | MEDIUM: {sd.get('MEDIUM', 0)} | LOW: {sd.get('LOW', 0)}")
            lines.append(f"  Alert-to-SAR rate: {alert_metrics.get('alert_to_sar_conversion_rate', 0):.2%}")
            lines.append(f"  False alert rate:  {alert_metrics.get('false_alert_rate', 0):.2%}")

        # LLM triage
        llm_metrics = self._evaluation_results.get("llm_triage")
        if llm_metrics:
            lines.append(f"\n--- LLM Triage Quality ---")
            lines.append(f"  Total triaged: {llm_metrics.get('total_triaged', 0)}")
            fv = llm_metrics.get("fact_verification", {})
            lines.append(f"  Verification pass rate: {fv.get('pass_rate', 0):.2%}")
            lines.append(f"  Mean verification score: {fv.get('mean_score', 0):.3f}")
            lines.append(f"  Recommendation consistency: {llm_metrics.get('recommendation_consistency', 0):.2%}")
            ea = llm_metrics.get("expert_agreement", {})
            if ea:
                lines.append(f"  Expert agreement rate: {ea.get('agreement_rate', 0):.2%}")

        # System comparison
        comparison = self._evaluation_results.get("system_comparison")
        if comparison:
            lines.append(f"\n--- System Comparison ---")
            for sys_name, m in comparison.get("system_metrics", {}).items():
                lines.append(
                    f"  {sys_name:12s} | AUC-ROC: {m['auc_roc']:.4f} | "
                    f"AUC-PR: {m['auc_pr']:.4f} | F1: {m['f1']:.4f}"
                )
            lines.append(f"  Best AUC-ROC: {comparison.get('best_auc_roc', 'N/A')}")
            lines.append(f"  Best AUC-PR:  {comparison.get('best_auc_pr', 'N/A')}")

        # Rule engine
        rule_metrics = self._evaluation_results.get("rule_engine")
        if rule_metrics:
            lines.append(f"\n--- Rule Engine Analysis ---")
            for rule_name, rm in rule_metrics.get("per_rule_metrics", {}).items():
                lines.append(
                    f"  {rule_name:20s} | hit_rate: {rm['hit_rate']:.4f} | "
                    f"precision: {rm.get('precision', 'N/A')} | recall: {rm.get('recall', 'N/A')}"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Recursively flatten a nested dictionary."""
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(AMLSystemEvaluator._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, np.ndarray)):
                # Skip large arrays (e.g., ROC curves)
                if len(v) < 20:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Make an object JSON-serializable by converting numpy types."""
        if isinstance(obj, dict):
            return {k: AMLSystemEvaluator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [AMLSystemEvaluator._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
