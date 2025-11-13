"""Tests for automated benchmarking framework.

The benchmarking framework is core to OmicSelector's philosophy:
- Test multiple feature selection signatures
- Evaluate with multiple models
- Compare performance across methods
- Select best signature based on hold-out validation

Test coverage:
- Signature benchmarking
- Model comparison
- Performance ranking
- Statistical significance testing
- Multi-metric evaluation
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.training.benchmarking import (  # noqa: E402
    BenchmarkResult,
    Benchmarker,
    SignatureBenchmark,
)


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate sample train/test data."""
    np.random.seed(42)
    n_samples_train = 100
    n_samples_test = 30
    n_features = 50

    X_train = pd.DataFrame(
        np.random.randn(n_samples_train, n_features),
        columns=[f"gene_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(np.random.binomial(1, 0.5, n_samples_train), name="target")

    X_test = pd.DataFrame(
        np.random.randn(n_samples_test, n_features),
        columns=[f"gene_{i}" for i in range(n_features)],
    )
    y_test = pd.Series(np.random.binomial(1, 0.5, n_samples_test), name="target")

    return X_train, y_train, X_test, y_test


class TestBenchmarkResult:
    """Test suite for BenchmarkResult data class."""

    def test_import(self) -> None:
        """Test that BenchmarkResult can be imported."""
        from omicselector2.training.benchmarking import BenchmarkResult

        assert BenchmarkResult is not None

    def test_initialization(self) -> None:
        """Test BenchmarkResult initialization."""
        result = BenchmarkResult(
            signature_name="Lasso_10",
            model_name="RandomForest",
            cv_metrics={"accuracy": 0.85, "auc": 0.90},
            test_metrics={"accuracy": 0.82, "auc": 0.88},
            feature_count=10,
        )

        assert result.signature_name == "Lasso_10"
        assert result.model_name == "RandomForest"
        assert result.cv_metrics["accuracy"] == 0.85
        assert result.test_metrics["accuracy"] == 0.82


class TestSignatureBenchmark:
    """Test suite for SignatureBenchmark."""

    def test_import(self) -> None:
        """Test that SignatureBenchmark can be imported."""
        from omicselector2.training.benchmarking import SignatureBenchmark

        assert SignatureBenchmark is not None

    def test_benchmark_single_signature(self, sample_data: tuple) -> None:
        """Test benchmarking a single signature."""
        X_train, y_train, X_test, y_test = sample_data

        # Select first 10 features as signature
        signature = X_train.columns[:10].tolist()

        benchmark = SignatureBenchmark()
        result = benchmark.evaluate_signature(
            signature=signature,
            model_name="RandomForest",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        assert result is not None
        assert result.model_name == "RandomForest"
        assert result.feature_count == 10
        assert "accuracy" in result.test_metrics
        assert "auc_roc" in result.test_metrics

    def test_benchmark_with_cross_validation(self, sample_data: tuple) -> None:
        """Test benchmarking with cross-validation."""
        X_train, y_train, X_test, y_test = sample_data

        signature = X_train.columns[:10].tolist()

        benchmark = SignatureBenchmark(cv_folds=5)
        result = benchmark.evaluate_signature(
            signature=signature,
            model_name="LogisticRegression",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        assert "accuracy" in result.cv_metrics
        assert "auc_roc" in result.cv_metrics
        # CV metrics should be dict or have mean/std
        assert isinstance(result.cv_metrics, dict)

    def test_benchmark_different_models(self, sample_data: tuple) -> None:
        """Test benchmarking with different models."""
        X_train, y_train, X_test, y_test = sample_data

        signature = X_train.columns[:10].tolist()

        benchmark = SignatureBenchmark()

        models = ["RandomForest", "LogisticRegression", "XGBoost"]
        results = []

        for model_name in models:
            result = benchmark.evaluate_signature(
                signature=signature,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            results.append(result)

        assert len(results) == 3
        assert all(r.feature_count == 10 for r in results)

    def test_benchmark_signature_sizes(self, sample_data: tuple) -> None:
        """Test benchmarking different signature sizes."""
        X_train, y_train, X_test, y_test = sample_data

        benchmark = SignatureBenchmark()

        results = []
        for n_features in [5, 10, 20]:
            signature = X_train.columns[:n_features].tolist()
            result = benchmark.evaluate_signature(
                signature=signature,
                model_name="RandomForest",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            results.append(result)

        assert len(results) == 3
        assert results[0].feature_count == 5
        assert results[1].feature_count == 10
        assert results[2].feature_count == 20


class TestBenchmarker:
    """Test suite for Benchmarker (full benchmarking orchestration)."""

    def test_import(self) -> None:
        """Test that Benchmarker can be imported."""
        from omicselector2.training.benchmarking import Benchmarker

        assert Benchmarker is not None

    def test_benchmark_multiple_signatures(self, sample_data: tuple) -> None:
        """Test benchmarking multiple signatures."""
        X_train, y_train, X_test, y_test = sample_data

        # Define signatures
        signatures = {
            "Top_5": X_train.columns[:5].tolist(),
            "Top_10": X_train.columns[:10].tolist(),
            "Top_20": X_train.columns[:20].tolist(),
        }

        benchmarker = Benchmarker(cv_folds=3)
        results = benchmarker.benchmark_signatures(
            signatures=signatures,
            models=["RandomForest", "LogisticRegression"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Should have 3 signatures × 2 models = 6 results
        assert len(results) == 6

    def test_get_best_signature(self, sample_data: tuple) -> None:
        """Test selecting best signature."""
        X_train, y_train, X_test, y_test = sample_data

        signatures = {
            "Top_5": X_train.columns[:5].tolist(),
            "Top_10": X_train.columns[:10].tolist(),
        }

        benchmarker = Benchmarker()
        results = benchmarker.benchmark_signatures(
            signatures=signatures,
            models=["RandomForest"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        best = benchmarker.get_best_result(results, metric="test_auc_roc")

        assert best is not None
        assert best.signature_name in ["Top_5", "Top_10"]

    def test_get_results_summary(self, sample_data: tuple) -> None:
        """Test generating results summary table."""
        X_train, y_train, X_test, y_test = sample_data

        signatures = {
            "Signature_A": X_train.columns[:10].tolist(),
            "Signature_B": X_train.columns[10:20].tolist(),
        }

        benchmarker = Benchmarker()
        results = benchmarker.benchmark_signatures(
            signatures=signatures,
            models=["RandomForest", "LogisticRegression"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        summary = benchmarker.get_results_summary(results)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4  # 2 signatures × 2 models
        assert "signature_name" in summary.columns
        assert "model_name" in summary.columns
        assert "test_accuracy" in summary.columns

    def test_compare_signatures_statistically(self, sample_data: tuple) -> None:
        """Test statistical comparison of signatures."""
        X_train, y_train, X_test, y_test = sample_data

        signatures = {
            "Sig_A": X_train.columns[:10].tolist(),
            "Sig_B": X_train.columns[:10].tolist(),  # Same features - should be similar
        }

        benchmarker = Benchmarker(cv_folds=5)
        results = benchmarker.benchmark_signatures(
            signatures=signatures,
            models=["RandomForest"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Compare results (should implement paired t-test or similar)
        comparison = benchmarker.compare_results(results[0], results[1], metric="accuracy")

        assert "p_value" in comparison or "significant" in comparison


class TestBenchmarkingWorkflow:
    """Test suite for complete benchmarking workflows."""

    def test_feature_selection_benchmarking_workflow(self, sample_data: tuple) -> None:
        """Test complete workflow: feature selection + benchmarking."""
        X_train, y_train, X_test, y_test = sample_data

        # Simulate feature selection results from multiple methods
        signatures = {
            "Lasso_L1": X_train.columns[:8].tolist(),
            "RandomForest_VI": X_train.columns[2:12].tolist(),
            "Ensemble": X_train.columns[:15].tolist(),
        }

        benchmarker = Benchmarker(cv_folds=3, random_state=42)

        # Benchmark all signatures with multiple models
        results = benchmarker.benchmark_signatures(
            signatures=signatures,
            models=["RandomForest", "XGBoost", "LogisticRegression"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Should have 3 signatures × 3 models = 9 results
        assert len(results) == 9

        # Get summary
        summary = benchmarker.get_results_summary(results)
        assert len(summary) == 9

        # Select best based on test AUC
        best = benchmarker.get_best_result(results, metric="test_auc_roc")
        assert best is not None

        # Check that best result has good performance
        assert best.test_metrics["auc_roc"] >= 0.0
        assert best.test_metrics["auc_roc"] <= 1.0
