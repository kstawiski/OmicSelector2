"""Automated benchmarking framework for signature evaluation.

This module implements OmicSelector's core philosophy:
- Test multiple feature selection signatures
- Evaluate with multiple models
- Compare performance using hold-out validation
- Select best signature based on rigorous testing

The benchmarking framework prevents overfitting by:
- Using hold-out test sets (never used during selection/training)
- Cross-validation for robust estimates
- Statistical comparison of methods
- Multiple performance metrics

Examples:
    >>> from omicselector2.training.benchmarking import Benchmarker
    >>>
    >>> # Define signatures from different methods
    >>> signatures = {
    ...     "Lasso_10": ["gene1", "gene2", ...],  # 10 features
    ...     "RandomForest_20": ["gene3", "gene4", ...],  # 20 features
    ...     "Ensemble_15": ["gene5", "gene6", ...],  # 15 features
    ... }
    >>>
    >>> # Benchmark with multiple models
    >>> benchmarker = Benchmarker(cv_folds=5, random_state=42)
    >>> results = benchmarker.benchmark_signatures(
    ...     signatures=signatures,
    ...     models=["RandomForest", "XGBoost", "LogisticRegression"],
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,  # Hold-out test set
    ...     y_test=y_test
    ... )
    >>>
    >>> # Get best signature
    >>> best = benchmarker.get_best_result(results, metric="test_auc_roc")
    >>> print(f"Best: {best.signature_name} with {best.model_name}")
    >>> print(f"Test AUC: {best.test_metrics['auc_roc']:.3f}")
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from omicselector2.models.classical import (
    LogisticRegressionModel,
    RandomForestClassifier,
    XGBoostClassifier,
)
from omicselector2.models.neural import TabNetClassifier
from omicselector2.training.cross_validation import CrossValidator
from omicselector2.training.evaluator import ClassificationEvaluator


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single signature with a single model.

    Attributes:
        signature_name: Name of the feature signature.
        model_name: Name of the model used.
        cv_metrics: Cross-validation metrics (mean across folds).
        test_metrics: Hold-out test set metrics.
        feature_count: Number of features in signature.
        features: List of feature names.
        cv_scores_per_fold: Optional CV scores for each fold (for statistical tests).
    """

    signature_name: str
    model_name: str
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_count: int
    features: Optional[list[str]] = None
    cv_scores_per_fold: Optional[dict[str, list[float]]] = None


class SignatureBenchmark:
    """Benchmark a single signature with a model.

    This class evaluates one feature signature with one model using:
    - Cross-validation on training data
    - Hold-out test set evaluation
    - Multiple performance metrics

    Attributes:
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Examples:
        >>> benchmark = SignatureBenchmark(cv_folds=5, random_state=42)
        >>> result = benchmark.evaluate_signature(
        ...     signature=["gene1", "gene2", "gene3"],
        ...     model_name="RandomForest",
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
        >>> print(f"Test AUC: {result.test_metrics['auc_roc']:.3f}")
    """

    def __init__(self, cv_folds: int = 5, random_state: Optional[int] = None) -> None:
        """Initialize signature benchmark.

        Args:
            cv_folds: Number of CV folds. Default 5.
            random_state: Random seed. Default None.
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.evaluator = ClassificationEvaluator()

    def evaluate_signature(
        self,
        signature: list[str],
        model_name: Literal["RandomForest", "LogisticRegression", "XGBoost"],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        signature_name: Optional[str] = None,
    ) -> BenchmarkResult:
        """Evaluate a signature with cross-validation and test set.

        Args:
            signature: List of feature names.
            model_name: Model to use ("RandomForest", "LogisticRegression", "XGBoost").
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            signature_name: Optional name for signature. Default uses model_name_N.

        Returns:
            BenchmarkResult with CV and test metrics.
        """
        # Filter to signature features
        X_train_sig = X_train[signature]
        X_test_sig = X_test[signature]

        # Cross-validation evaluation
        cv_metrics, cv_scores_per_fold = self._cross_validate(
            X_train_sig, y_train, model_name
        )

        # Train final model on all training data
        model = self._create_model(model_name)
        model.fit(X_train_sig, y_train)

        # Evaluate on test set
        y_pred_proba = model.predict_proba(X_test_sig)
        test_metrics = self.evaluator.evaluate(
            y_test.values, y_pred_proba, probabilities=True
        )

        # Create result
        if signature_name is None:
            signature_name = f"{model_name}_{len(signature)}"

        return BenchmarkResult(
            signature_name=signature_name,
            model_name=model_name,
            cv_metrics=cv_metrics,
            test_metrics=test_metrics,
            feature_count=len(signature),
            features=signature,
            cv_scores_per_fold=cv_scores_per_fold,
        )

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, model_name: str
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Perform cross-validation.

        Args:
            X: Features.
            y: Target.
            model_name: Model to use.

        Returns:
            Tuple of (mean_metrics, scores_per_fold).
        """
        cv = CrossValidator(
            cv_type="stratified", n_splits=self.cv_folds, random_state=self.random_state
        )

        # Store scores per fold for statistical tests
        cv_scores_per_fold: dict[str, list[float]] = {
            "accuracy": [],
            "auc_roc": [],
            "f1": [],
        }

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Train model
            model = self._create_model(model_name)
            model.fit(X_train_fold, y_train_fold)

            # Evaluate
            y_pred_proba = model.predict_proba(X_val_fold)
            metrics = self.evaluator.evaluate(
                y_val_fold.values, y_pred_proba, probabilities=True
            )

            # Store scores
            cv_scores_per_fold["accuracy"].append(metrics["accuracy"])
            cv_scores_per_fold["auc_roc"].append(metrics["auc_roc"])
            cv_scores_per_fold["f1"].append(metrics["f1"])

        # Compute mean metrics
        cv_metrics = {
            metric: float(np.mean(scores))
            for metric, scores in cv_scores_per_fold.items()
        }

        return cv_metrics, cv_scores_per_fold

    def _create_model(
        self, model_name: str
    ) -> RandomForestClassifier | LogisticRegressionModel | XGBoostClassifier | TabNetClassifier:
        """Create model instance.

        Args:
            model_name: Model to create.

        Returns:
            Model instance.
        """
        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            )
        elif model_name == "LogisticRegression":
            return LogisticRegressionModel(C=1.0, random_state=self.random_state)
        elif model_name == "XGBoost":
            return XGBoostClassifier(
                n_estimators=100, random_state=self.random_state
            )
        elif model_name == "TabNet":
            return TabNetClassifier(
                max_epochs=50,  # Reasonable for benchmarking
                batch_size=256,
                patience=10,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")


class Benchmarker:
    """Orchestrate benchmarking of multiple signatures with multiple models.

    This class implements the full OmicSelector benchmarking workflow:
    1. Test multiple feature signatures
    2. Evaluate each with multiple models
    3. Use cross-validation + hold-out test
    4. Rank and compare results
    5. Select best signature

    Attributes:
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Examples:
        >>> benchmarker = Benchmarker(cv_folds=5, random_state=42)
        >>>
        >>> signatures = {
        ...     "Lasso_10": lasso_genes,
        ...     "RF_20": rf_genes,
        ...     "Ensemble_15": ensemble_genes
        ... }
        >>>
        >>> results = benchmarker.benchmark_signatures(
        ...     signatures=signatures,
        ...     models=["RandomForest", "XGBoost"],
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
        >>>
        >>> # Get summary
        >>> summary = benchmarker.get_results_summary(results)
        >>> print(summary)
        >>>
        >>> # Get best
        >>> best = benchmarker.get_best_result(results, metric="test_auc_roc")
    """

    def __init__(self, cv_folds: int = 5, random_state: Optional[int] = None) -> None:
        """Initialize benchmarker.

        Args:
            cv_folds: Number of CV folds. Default 5.
            random_state: Random seed. Default None.
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.signature_benchmark = SignatureBenchmark(
            cv_folds=cv_folds, random_state=random_state
        )

    def benchmark_signatures(
        self,
        signatures: dict[str, list[str]],
        models: list[Literal["RandomForest", "LogisticRegression", "XGBoost", "TabNet"]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> list[BenchmarkResult]:
        """Benchmark multiple signatures with multiple models.

        Args:
            signatures: Dict mapping signature names to feature lists.
            models: List of model names to test.
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.

        Returns:
            List of BenchmarkResult objects (one per signature-model combination).
        """
        results = []

        for sig_name, features in signatures.items():
            for model_name in models:
                result = self.signature_benchmark.evaluate_signature(
                    signature=features,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    signature_name=sig_name,
                )
                results.append(result)

        return results

    def get_best_result(
        self,
        results: list[BenchmarkResult],
        metric: Literal[
            "test_accuracy", "test_auc_roc", "test_f1", "cv_accuracy", "cv_auc_roc"
        ] = "test_auc_roc",
    ) -> BenchmarkResult:
        """Get best result based on specified metric.

        Args:
            results: List of benchmark results.
            metric: Metric to use for ranking. Default "test_auc_roc".

        Returns:
            Best BenchmarkResult.
        """
        if metric.startswith("test_"):
            metric_key = metric.replace("test_", "")
            scores = [r.test_metrics[metric_key] for r in results]
        elif metric.startswith("cv_"):
            metric_key = metric.replace("cv_", "")
            scores = [r.cv_metrics[metric_key] for r in results]
        else:
            raise ValueError(f"Invalid metric: {metric}")

        best_idx = np.argmax(scores)
        return results[best_idx]

    def get_results_summary(self, results: list[BenchmarkResult]) -> pd.DataFrame:
        """Create summary table of all results.

        Args:
            results: List of benchmark results.

        Returns:
            DataFrame with columns:
            - signature_name
            - model_name
            - feature_count
            - cv_accuracy, cv_auc_roc, cv_f1
            - test_accuracy, test_auc_roc, test_f1
        """
        rows = []

        for result in results:
            row = {
                "signature_name": result.signature_name,
                "model_name": result.model_name,
                "feature_count": result.feature_count,
                "cv_accuracy": result.cv_metrics.get("accuracy", np.nan),
                "cv_auc_roc": result.cv_metrics.get("auc_roc", np.nan),
                "cv_f1": result.cv_metrics.get("f1", np.nan),
                "test_accuracy": result.test_metrics.get("accuracy", np.nan),
                "test_auc_roc": result.test_metrics.get("auc_roc", np.nan),
                "test_f1": result.test_metrics.get("f1", np.nan),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by test_auc_roc descending
        df = df.sort_values("test_auc_roc", ascending=False).reset_index(drop=True)

        return df

    def compare_results(
        self,
        result1: BenchmarkResult,
        result2: BenchmarkResult,
        metric: Literal["accuracy", "auc_roc", "f1"] = "auc_roc",
    ) -> dict[str, Any]:
        """Statistically compare two results using paired t-test on CV folds.

        Args:
            result1: First result.
            result2: Second result.
            metric: Metric to compare. Default "auc_roc".

        Returns:
            Dict with keys:
            - mean_diff: Difference in means
            - p_value: P-value from paired t-test
            - significant: Whether difference is significant (p < 0.05)
        """
        if result1.cv_scores_per_fold is None or result2.cv_scores_per_fold is None:
            raise ValueError("Results must have cv_scores_per_fold for comparison")

        scores1 = result1.cv_scores_per_fold[metric]
        scores2 = result2.cv_scores_per_fold[metric]

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        mean_diff = np.mean(scores1) - np.mean(scores2)

        return {
            "mean_diff": float(mean_diff),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "metric": metric,
        }
