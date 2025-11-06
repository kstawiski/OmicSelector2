"""Tests for model evaluation metrics.

The evaluator provides comprehensive metrics for model performance assessment:
- Classification metrics (accuracy, precision, recall, F1, AUC-ROC, AUC-PR)
- Regression metrics (MSE, RMSE, MAE, R², Pearson correlation)
- Survival analysis metrics (C-index, Integrated Brier Score)
- Calibration metrics
- Confusion matrix

Test coverage:
- Binary classification metrics
- Multiclass classification metrics
- Regression metrics
- Survival metrics
- Edge cases and error handling
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.training.evaluator import (  # noqa: E402
    ClassificationEvaluator,
    RegressionEvaluator,
    SurvivalEvaluator,
)


@pytest.fixture
def binary_classification_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample binary classification predictions.

    Returns:
        Tuple of (y_true, y_pred, y_score).
    """
    np.random.seed(42)
    n_samples = 100

    # True labels
    y_true = np.random.binomial(1, 0.5, n_samples)

    # Predicted probabilities (slightly correlated with true labels)
    y_score = np.random.beta(2, 2, n_samples)
    y_score = np.where(y_true == 1, y_score * 0.7 + 0.3, y_score * 0.7)

    # Predicted classes (threshold at 0.5)
    y_pred = (y_score >= 0.5).astype(int)

    return y_true, y_pred, y_score


@pytest.fixture
def multiclass_classification_predictions() -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Generate sample multiclass classification predictions.

    Returns:
        Tuple of (y_true, y_pred, y_score).
    """
    np.random.seed(42)
    n_samples = 150
    n_classes = 3

    # True labels
    y_true = np.random.choice(n_classes, n_samples)

    # Predicted probabilities
    y_score = np.random.dirichlet(np.ones(n_classes), n_samples)

    # Predicted classes (argmax of probabilities)
    y_pred = np.argmax(y_score, axis=1)

    return y_true, y_pred, y_score


@pytest.fixture
def regression_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample regression predictions.

    Returns:
        Tuple of (y_true, y_pred).
    """
    np.random.seed(42)
    n_samples = 100

    # True values
    y_true = np.random.randn(n_samples) * 10 + 50

    # Predicted values (with some error)
    noise = np.random.randn(n_samples) * 3
    y_pred = y_true + noise

    return y_true, y_pred


@pytest.fixture
def survival_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample survival data.

    Returns:
        Tuple of (event_times, event_observed, risk_scores).
    """
    np.random.seed(42)
    n_samples = 100

    # Event times (survival times)
    event_times = np.random.exponential(scale=100, size=n_samples)

    # Event observed (1=event occurred, 0=censored)
    event_observed = np.random.binomial(1, 0.7, n_samples)

    # Risk scores (higher = higher risk)
    risk_scores = np.random.randn(n_samples)

    return event_times, event_observed, risk_scores


class TestClassificationEvaluator:
    """Test suite for ClassificationEvaluator."""

    def test_import(self) -> None:
        """Test that ClassificationEvaluator can be imported."""
        from omicselector2.training.evaluator import ClassificationEvaluator

        assert ClassificationEvaluator is not None

    def test_binary_accuracy(
        self, binary_classification_predictions: tuple
    ) -> None:
        """Test accuracy calculation for binary classification."""
        y_true, y_pred, _ = binary_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

        # Manually calculate accuracy
        expected_accuracy = np.mean(y_true == y_pred)
        assert abs(metrics["accuracy"] - expected_accuracy) < 1e-6

    def test_binary_precision_recall_f1(
        self, binary_classification_predictions: tuple
    ) -> None:
        """Test precision, recall, and F1 for binary classification."""
        y_true, y_pred, _ = binary_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_binary_auc_roc(
        self, binary_classification_predictions: tuple
    ) -> None:
        """Test AUC-ROC calculation for binary classification."""
        y_true, _, y_score = binary_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_score, probabilities=True)

        assert "auc_roc" in metrics
        assert 0 <= metrics["auc_roc"] <= 1

        # AUC should be better than random (0.5)
        assert metrics["auc_roc"] > 0.5

    def test_binary_auc_pr(
        self, binary_classification_predictions: tuple
    ) -> None:
        """Test AUC-PR calculation for binary classification."""
        y_true, _, y_score = binary_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_score, probabilities=True)

        assert "auc_pr" in metrics
        assert 0 <= metrics["auc_pr"] <= 1

    def test_confusion_matrix(
        self, binary_classification_predictions: tuple
    ) -> None:
        """Test confusion matrix calculation."""
        y_true, y_pred, _ = binary_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]

        # Should be 2x2 for binary classification
        assert cm.shape == (2, 2)

        # Sum should equal number of samples
        assert cm.sum() == len(y_true)

    def test_multiclass_accuracy(
        self, multiclass_classification_predictions: tuple
    ) -> None:
        """Test accuracy for multiclass classification."""
        y_true, y_pred, _ = multiclass_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, multiclass=True)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_multiclass_macro_metrics(
        self, multiclass_classification_predictions: tuple
    ) -> None:
        """Test macro-averaged metrics for multiclass."""
        y_true, y_pred, _ = multiclass_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, multiclass=True, average="macro")

        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics

        assert 0 <= metrics["precision_macro"] <= 1
        assert 0 <= metrics["recall_macro"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_multiclass_auc_ovr(
        self, multiclass_classification_predictions: tuple
    ) -> None:
        """Test one-vs-rest AUC for multiclass."""
        y_true, _, y_score = multiclass_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(
            y_true, y_score, multiclass=True, probabilities=True
        )

        assert "auc_ovr" in metrics
        assert 0 <= metrics["auc_ovr"] <= 1

    def test_per_class_metrics(
        self, binary_classification_predictions: tuple
    ) -> None:
        """Test per-class metrics calculation."""
        y_true, y_pred, _ = binary_classification_predictions

        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, per_class=True)

        assert "per_class_precision" in metrics
        assert "per_class_recall" in metrics
        assert "per_class_f1" in metrics

        # Should have metrics for each class
        assert len(metrics["per_class_precision"]) == 2

    def test_invalid_input_shapes(self) -> None:
        """Test that mismatched input shapes raise ValueError."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0])  # Wrong length

        evaluator = ClassificationEvaluator()

        with pytest.raises(ValueError, match="shape mismatch"):
            evaluator.evaluate(y_true, y_pred)


class TestRegressionEvaluator:
    """Test suite for RegressionEvaluator."""

    def test_import(self) -> None:
        """Test that RegressionEvaluator can be imported."""
        from omicselector2.training.evaluator import RegressionEvaluator

        assert RegressionEvaluator is not None

    def test_mse(self, regression_predictions: tuple) -> None:
        """Test MSE calculation."""
        y_true, y_pred = regression_predictions

        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "mse" in metrics
        assert metrics["mse"] >= 0

        # Manually calculate MSE
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert abs(metrics["mse"] - expected_mse) < 1e-6

    def test_rmse(self, regression_predictions: tuple) -> None:
        """Test RMSE calculation."""
        y_true, y_pred = regression_predictions

        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "rmse" in metrics
        assert metrics["rmse"] >= 0

        # RMSE should be sqrt of MSE
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-6

    def test_mae(self, regression_predictions: tuple) -> None:
        """Test MAE calculation."""
        y_true, y_pred = regression_predictions

        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "mae" in metrics
        assert metrics["mae"] >= 0

        # Manually calculate MAE
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert abs(metrics["mae"] - expected_mae) < 1e-6

    def test_r2_score(self, regression_predictions: tuple) -> None:
        """Test R² score calculation."""
        y_true, y_pred = regression_predictions

        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "r2" in metrics

        # R² should be between -inf and 1 (1 = perfect, <0 = worse than mean)
        assert metrics["r2"] <= 1

    def test_pearson_correlation(self, regression_predictions: tuple) -> None:
        """Test Pearson correlation calculation."""
        y_true, y_pred = regression_predictions

        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "pearson_r" in metrics
        assert "pearson_p" in metrics

        # Correlation should be between -1 and 1
        assert -1 <= metrics["pearson_r"] <= 1

        # P-value should be between 0 and 1
        assert 0 <= metrics["pearson_p"] <= 1

    def test_residuals(self, regression_predictions: tuple) -> None:
        """Test residuals calculation."""
        y_true, y_pred = regression_predictions

        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, compute_residuals=True)

        assert "residuals" in metrics
        residuals = metrics["residuals"]

        assert len(residuals) == len(y_true)

        # Residuals should be true - pred
        expected_residuals = y_true - y_pred
        np.testing.assert_array_almost_equal(residuals, expected_residuals)

    def test_invalid_input_shapes(self) -> None:
        """Test that mismatched input shapes raise ValueError."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1])  # Wrong length

        evaluator = RegressionEvaluator()

        with pytest.raises(ValueError, match="shape mismatch"):
            evaluator.evaluate(y_true, y_pred)


class TestSurvivalEvaluator:
    """Test suite for SurvivalEvaluator."""

    def test_import(self) -> None:
        """Test that SurvivalEvaluator can be imported."""
        from omicselector2.training.evaluator import SurvivalEvaluator

        assert SurvivalEvaluator is not None

    def test_concordance_index(self, survival_data: tuple) -> None:
        """Test C-index calculation."""
        event_times, event_observed, risk_scores = survival_data

        evaluator = SurvivalEvaluator()
        metrics = evaluator.evaluate(event_times, event_observed, risk_scores)

        assert "c_index" in metrics

        # C-index should be between 0 and 1
        # 0.5 = random, >0.5 = better than random
        assert 0 <= metrics["c_index"] <= 1

    def test_c_index_with_ties(self) -> None:
        """Test C-index handles tied risk scores."""
        event_times = np.array([10, 20, 30, 40, 50])
        event_observed = np.array([1, 1, 1, 1, 1])
        risk_scores = np.array([1.0, 1.0, 2.0, 2.0, 3.0])  # Ties

        evaluator = SurvivalEvaluator()
        metrics = evaluator.evaluate(event_times, event_observed, risk_scores)

        assert "c_index" in metrics
        assert 0 <= metrics["c_index"] <= 1

    def test_invalid_input_shapes(self) -> None:
        """Test that mismatched input shapes raise ValueError."""
        event_times = np.array([10, 20, 30, 40])
        event_observed = np.array([1, 1, 0])  # Wrong length
        risk_scores = np.array([1.0, 2.0, 3.0, 4.0])

        evaluator = SurvivalEvaluator()

        with pytest.raises(ValueError, match="shape mismatch"):
            evaluator.evaluate(event_times, event_observed, risk_scores)

    def test_invalid_event_observed_values(self) -> None:
        """Test that invalid event_observed values raise ValueError."""
        event_times = np.array([10, 20, 30])
        event_observed = np.array([1, 2, 0])  # Invalid value (2)
        risk_scores = np.array([1.0, 2.0, 3.0])

        evaluator = SurvivalEvaluator()

        with pytest.raises(ValueError, match="event_observed.*must be 0 or 1"):
            evaluator.evaluate(event_times, event_observed, risk_scores)
