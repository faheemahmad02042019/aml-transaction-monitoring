"""
ML Model Training for Suspicious Activity Detection.

Trains LightGBM and XGBoost classifiers with:
  - Cost-sensitive learning (asymmetric loss for AML)
  - SMOTE/ADASYN oversampling for extreme class imbalance
  - Threshold optimization targeting desired recall levels
  - MLflow experiment tracking
  - SHAP-based explainability
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
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
from sklearn.model_selection import StratifiedKFold

from src.config import Config, ModelConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


class AMLModelTrainer:
    """
    Trains and evaluates ML models for AML suspicious activity detection.

    Supports LightGBM and XGBoost with cost-sensitive learning, handles
    extreme class imbalance via SMOTE/ADASYN, and provides SHAP-based
    feature explanations.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_config: ModelConfig = config.model
        self._lgbm_model = None
        self._xgb_model = None
        self._best_model = None
        self._best_threshold: float = 0.5
        self._feature_names: List[str] = []
        self._shap_values = None
        self._training_metrics: Dict[str, Any] = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train LightGBM and XGBoost models.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels (0=legitimate, 1=laundering).
        X_val : pd.DataFrame
            Validation features.
        y_val : pd.Series
            Validation labels.
        feature_names : List[str], optional
            Feature column names.

        Returns
        -------
        Dict[str, Any]
            Training results including model references and metrics.
        """
        self._feature_names = feature_names or list(X_train.columns)

        # Compute class imbalance ratio for cost-sensitive learning
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        imbalance_ratio = n_neg / n_pos if n_pos > 0 else 1.0

        logger.info(
            f"Training data: {len(X_train):,} samples | "
            f"Positive: {n_pos:,} ({n_pos / len(y_train) * 100:.3f}%) | "
            f"Imbalance ratio: {imbalance_ratio:.1f}:1"
        )

        # Apply SMOTE if configured
        X_train_resampled, y_train_resampled = self._apply_resampling(X_train, y_train)

        # Initialize MLflow tracking
        mlflow_run = self._init_mlflow()

        # Train LightGBM
        logger.info("=" * 60)
        logger.info("Training LightGBM")
        logger.info("=" * 60)
        lgbm_results = self._train_lightgbm(
            X_train_resampled, y_train_resampled, X_val, y_val, imbalance_ratio
        )

        # Train XGBoost
        logger.info("=" * 60)
        logger.info("Training XGBoost")
        logger.info("=" * 60)
        xgb_results = self._train_xgboost(
            X_train_resampled, y_train_resampled, X_val, y_val, imbalance_ratio
        )

        # Select best model based on AUC-PR (more informative for imbalanced data)
        if lgbm_results["auc_pr"] >= xgb_results["auc_pr"]:
            self._best_model = self._lgbm_model
            best_name = "LightGBM"
            best_results = lgbm_results
        else:
            self._best_model = self._xgb_model
            best_name = "XGBoost"
            best_results = xgb_results

        logger.info(f"Best model: {best_name} (AUC-PR: {best_results['auc_pr']:.4f})")

        # Optimize decision threshold
        self._best_threshold = self._optimize_threshold(
            y_val, best_results["y_prob"]
        )

        # Compute SHAP values
        self._compute_shap(X_val)

        # Log to MLflow
        self._log_mlflow(mlflow_run, best_results, best_name)

        # Compile training results
        self._training_metrics = {
            "best_model": best_name,
            "best_threshold": self._best_threshold,
            "lgbm_metrics": lgbm_results["metrics"],
            "xgb_metrics": xgb_results["metrics"],
            "imbalance_ratio": imbalance_ratio,
            "n_train": len(X_train),
            "n_train_resampled": len(X_train_resampled),
            "n_val": len(X_val),
            "n_features": len(self._feature_names),
        }

        return self._training_metrics

    def _apply_resampling(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE or ADASYN oversampling to handle class imbalance."""
        if not self.model_config.use_smote and not self.model_config.use_adasyn:
            logger.info("Resampling disabled; using original training data")
            return X, y

        try:
            if self.model_config.use_adasyn:
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(
                    sampling_strategy=self.model_config.smote_sampling_strategy,
                    random_state=self.model_config.random_state,
                    n_neighbors=self.model_config.smote_k_neighbors,
                )
                method_name = "ADASYN"
            else:
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(
                    sampling_strategy=self.model_config.smote_sampling_strategy,
                    random_state=self.model_config.random_state,
                    k_neighbors=min(self.model_config.smote_k_neighbors, int(y.sum()) - 1),
                )
                method_name = "SMOTE"

            logger.info(f"Applying {method_name} (target ratio: {self.model_config.smote_sampling_strategy})")
            X_res, y_res = sampler.fit_resample(X, y)
            X_res = pd.DataFrame(X_res, columns=X.columns)
            y_res = pd.Series(y_res, name=y.name)

            logger.info(
                f"  Resampled: {len(X_res):,} samples | "
                f"Positive: {y_res.sum():,} ({y_res.mean() * 100:.2f}%)"
            )
            return X_res, y_res

        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping resampling.")
            return X, y
        except Exception as e:
            logger.warning(f"Resampling failed: {e}. Using original data.")
            return X, y

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        imbalance_ratio: float,
    ) -> Dict[str, Any]:
        """Train LightGBM with cost-sensitive learning."""
        import lightgbm as lgb

        params = self.model_config.lgbm_params.copy()

        # Set scale_pos_weight if not manually configured
        if params.get("scale_pos_weight") is None:
            params["scale_pos_weight"] = imbalance_ratio
            params.pop("is_unbalance", None)  # Don't use both

        n_estimators = params.pop("n_estimators", 1000)
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        self._lgbm_model = model

        # Predictions
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = self._compute_metrics(y_val, y_prob)

        logger.info(f"  LightGBM AUC-ROC: {metrics['auc_roc']:.4f} | AUC-PR: {metrics['auc_pr']:.4f}")

        return {
            "model": model,
            "y_prob": y_prob,
            "metrics": metrics,
            "auc_pr": metrics["auc_pr"],
            "feature_importance": dict(zip(self._feature_names, model.feature_importances_)),
        }

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        imbalance_ratio: float,
    ) -> Dict[str, Any]:
        """Train XGBoost with cost-sensitive learning."""
        import xgboost as xgb

        params = self.model_config.xgb_params.copy()

        if params.get("scale_pos_weight") is None:
            params["scale_pos_weight"] = imbalance_ratio

        n_estimators = params.pop("n_estimators", 1000)
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            random_state=self.model_config.random_state,
            **params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self._xgb_model = model

        # Predictions
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = self._compute_metrics(y_val, y_prob)

        logger.info(f"  XGBoost AUC-ROC: {metrics['auc_roc']:.4f} | AUC-PR: {metrics['auc_pr']:.4f}")

        return {
            "model": model,
            "y_prob": y_prob,
            "metrics": metrics,
            "auc_pr": metrics["auc_pr"],
            "feature_importance": dict(zip(
                self._feature_names,
                model.feature_importances_,
            )),
        }

    def _compute_metrics(
        self, y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        y_pred = (y_prob >= threshold).astype(int)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_roc = roc_auc_score(y_true, y_prob)

        # Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)

        # Recall at specific FPR levels
        recall_at_1pct_fpr = float(np.interp(0.01, fpr, tpr))
        recall_at_5pct_fpr = float(np.interp(0.05, fpr, tpr))

        # Standard metrics at the given threshold
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        return {
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "recall_at_1pct_fpr": recall_at_1pct_fpr,
            "recall_at_5pct_fpr": recall_at_5pct_fpr,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "threshold": threshold,
        }

    def _optimize_threshold(
        self, y_true: pd.Series, y_prob: np.ndarray
    ) -> float:
        """
        Optimize the decision threshold for the target recall level.

        In AML, missing a true positive (false negative) is much costlier
        than a false alarm, so we optimize for a high recall target.

        Parameters
        ----------
        y_true : pd.Series
            True labels.
        y_prob : np.ndarray
            Predicted probabilities.

        Returns
        -------
        float
            Optimized decision threshold.
        """
        target_recall = self.model_config.target_recall
        logger.info(f"Optimizing threshold for target recall >= {target_recall:.2f}")

        thresholds = np.linspace(
            self.model_config.threshold_search_range[0],
            self.model_config.threshold_search_range[1],
            self.model_config.threshold_search_steps,
        )

        best_threshold = 0.5
        best_f1 = 0.0

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # We want the highest F1 that meets our recall constraint
            if recall >= target_recall and f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        # If no threshold meets the recall constraint, find the one with highest recall
        if best_f1 == 0.0:
            best_recall = 0.0
            for thresh in thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                recall = recall_score(y_true, y_pred, zero_division=0)
                if recall > best_recall:
                    best_recall = recall
                    best_threshold = thresh

        # Evaluate at the optimized threshold
        metrics = self._compute_metrics(y_true, y_prob, best_threshold)
        logger.info(
            f"  Optimized threshold: {best_threshold:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"FPR: {metrics['false_positive_rate']:.4f}"
        )

        return float(best_threshold)

    def _compute_shap(self, X: pd.DataFrame) -> None:
        """Compute SHAP values for model explainability."""
        try:
            import shap

            logger.info("Computing SHAP values for model explainability")

            # Sample if dataset is large
            max_samples = 5000
            if len(X) > max_samples:
                X_sample = X.sample(max_samples, random_state=self.model_config.random_state)
            else:
                X_sample = X

            explainer = shap.TreeExplainer(self._best_model)
            self._shap_values = explainer.shap_values(X_sample)

            # Get top features
            if isinstance(self._shap_values, list):
                shap_abs = np.abs(self._shap_values[1]).mean(axis=0)
            else:
                shap_abs = np.abs(self._shap_values).mean(axis=0)

            top_features = sorted(
                zip(self._feature_names, shap_abs),
                key=lambda x: x[1],
                reverse=True,
            )[:20]

            logger.info("  Top-10 features by SHAP importance:")
            for i, (name, importance) in enumerate(top_features[:10], 1):
                logger.info(f"    {i:2d}. {name:40s} {importance:.4f}")

        except ImportError:
            logger.warning("SHAP not installed. Skipping explainability analysis.")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the best model and optimized threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1).
        """
        if self._best_model is None:
            raise RuntimeError("No model trained. Call train() first.")

        probabilities = self.predict_proba(X)
        return (probabilities >= self._best_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability scores using the best model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability of laundering (class 1).
        """
        if self._best_model is None:
            raise RuntimeError("No model trained. Call train() first.")

        return self._best_model.predict_proba(X)[:, 1]

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save the best model to disk."""
        import joblib

        if self._best_model is None:
            raise RuntimeError("No model trained. Call train() first.")

        path = path or self.model_config.model_dir / "best_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self._best_model,
                "threshold": self._best_threshold,
                "feature_names": self._feature_names,
                "training_metrics": self._training_metrics,
            },
            path,
        )

        logger.info(f"Model saved to {path}")
        return path

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load a trained model from disk."""
        import joblib

        path = path or self.model_config.model_dir / "best_model.joblib"

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self._best_model = data["model"]
        self._best_threshold = data["threshold"]
        self._feature_names = data["feature_names"]
        self._training_metrics = data.get("training_metrics", {})

        logger.info(f"Model loaded from {path}")

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform stratified K-fold cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Labels.
        n_folds : int
            Number of folds.

        Returns
        -------
        Dict[str, Any]
            Cross-validation results.
        """
        logger.info(f"Running {n_folds}-fold stratified cross-validation")

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.model_config.random_state,
        )

        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"  Fold {fold_idx}/{n_folds}")
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            self.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            y_prob = self.predict_proba(X_val_fold)
            metrics = self._compute_metrics(y_val_fold, y_prob, self._best_threshold)
            fold_metrics.append(metrics)

        # Aggregate metrics across folds
        metric_names = fold_metrics[0].keys()
        agg_metrics = {}
        for metric in metric_names:
            values = [fm[metric] for fm in fold_metrics]
            agg_metrics[f"{metric}_mean"] = float(np.mean(values))
            agg_metrics[f"{metric}_std"] = float(np.std(values))

        logger.info(
            f"  CV Results | "
            f"AUC-ROC: {agg_metrics['auc_roc_mean']:.4f} +/- {agg_metrics['auc_roc_std']:.4f} | "
            f"AUC-PR: {agg_metrics['auc_pr_mean']:.4f} +/- {agg_metrics['auc_pr_std']:.4f}"
        )

        return {
            "n_folds": n_folds,
            "fold_metrics": fold_metrics,
            "aggregate_metrics": agg_metrics,
        }

    def _init_mlflow(self) -> Optional[Any]:
        """Initialize MLflow tracking."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.model_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.model_config.mlflow_experiment_name)
            run = mlflow.start_run()
            logger.info(f"MLflow run started: {run.info.run_id}")
            return run
        except ImportError:
            logger.info("MLflow not installed. Skipping experiment tracking.")
            return None
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            return None

    def _log_mlflow(
        self, run: Optional[Any], results: Dict[str, Any], model_name: str
    ) -> None:
        """Log training results to MLflow."""
        if run is None:
            return

        try:
            import mlflow

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", len(self._feature_names))
            mlflow.log_param("threshold", self._best_threshold)
            mlflow.log_param("use_smote", self.model_config.use_smote)

            for metric_name, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)

            mlflow.end_run()
            logger.info("MLflow run logged successfully")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
