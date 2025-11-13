"""Unit tests for Variance Threshold feature selector.

This module tests the VarianceThresholdSelector class which removes features
with low variance. Low-variance features carry little information and can be
safely removed in many machine learning tasks.

Variance threshold is a simple but effective baseline filter method, especially
for genomics data where many genes may have near-constant expression.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.variance_threshold import (  # noqa: E402
    VarianceThresholdSelector,
)


@pytest.fixture
def sample_data_with_low_variance() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample data with some low-variance features."""
    np.random.seed(42)
    n_samples = 200

    # Create features with different variances
    data = {
        "high_var_1": np.random.randn(n_samples) * 10,  # High variance
        "high_var_2": np.random.randn(n_samples) * 5,  # High variance
        "low_var_1": np.ones(n_samples) * 5,  # Zero variance (constant)
        "low_var_2": np.random.randn(n_samples) * 0.01,  # Very low variance
        "medium_var_1": np.random.randn(n_samples) * 1,  # Medium variance
        "medium_var_2": np.random.randn(n_samples) * 2,  # Medium variance
        "zero_var": np.zeros(n_samples),  # Zero variance
    }

    X = pd.DataFrame(data)
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))

    return X, y


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate standard classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))

    return X, y


class TestVarianceThresholdSelector:
    """Test suite for VarianceThresholdSelector."""

    def test_import(self) -> None:
        """Test that VarianceThresholdSelector can be imported."""
        assert VarianceThresholdSelector is not None

    def test_initialization(self) -> None:
        """Test VarianceThresholdSelector initialization with default parameters."""
        selector = VarianceThresholdSelector()
        assert selector is not None
        assert selector.threshold == 0.0
        assert selector.task == "classification"

    def test_fit(self, sample_data_with_low_variance: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method executes and calculates variances."""
        X, y = sample_data_with_low_variance
        selector = VarianceThresholdSelector(threshold=0.1)

        result = selector.fit(X, y)

        assert result is selector  # Should return self
        assert hasattr(selector, "variances_")
        assert hasattr(selector, "selected_features_")
        assert len(selector.variances_) == X.shape[1]

    def test_removes_zero_variance(
        self, sample_data_with_low_variance: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that zero-variance features are removed."""
        X, y = sample_data_with_low_variance
        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(X, y)

        # Features with zero variance should be removed
        assert "low_var_1" not in selector.selected_features_  # Constant
        assert "zero_var" not in selector.selected_features_  # Zero variance

    def test_threshold_filtering(
        self, sample_data_with_low_variance: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that features below threshold are removed."""
        X, y = sample_data_with_low_variance
        selector = VarianceThresholdSelector(threshold=0.5)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        # Low variance features should be removed
        assert "low_var_1" not in X_transformed.columns
        assert "low_var_2" not in X_transformed.columns
        assert "zero_var" not in X_transformed.columns

        # High variance features should remain
        assert "high_var_1" in X_transformed.columns
        assert "high_var_2" in X_transformed.columns

    def test_transform(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform returns DataFrame with high-variance features."""
        X, y = sample_classification_data
        selector = VarianceThresholdSelector(threshold=0.5)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]  # Same samples
        assert X_transformed.shape[1] <= X.shape[1]  # Fewer or equal features
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y = sample_classification_data
        selector = VarianceThresholdSelector(threshold=0.8)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] <= X.shape[1]
        assert hasattr(selector, "selected_features_")

    def test_get_support(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_support returns boolean mask."""
        X, y = sample_classification_data
        selector = VarianceThresholdSelector(threshold=0.5)
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]

    def test_get_support_indices(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_classification_data
        selector = VarianceThresholdSelector(threshold=0.5)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp

    def test_different_thresholds(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that higher threshold removes more features."""
        X, y = sample_classification_data

        selector_low = VarianceThresholdSelector(threshold=0.1)
        selector_low.fit(X, y)

        selector_high = VarianceThresholdSelector(threshold=2.0)
        selector_high.fit(X, y)

        # Higher threshold should select fewer features
        assert len(selector_high.selected_features_) <= len(selector_low.selected_features_)

    def test_negative_threshold_raises_error(self) -> None:
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            VarianceThresholdSelector(threshold=-0.1)

    def test_get_result(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_result returns FeatureSelectorResult."""
        X, y = sample_classification_data
        selector = VarianceThresholdSelector(threshold=0.5)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == len(selector.selected_features_)
        assert len(result.feature_scores) == len(selector.selected_features_)
        assert result.method_name == "VarianceThresholdSelector"
        # Scores should be variances (sorted descending)
        assert all(
            result.feature_scores[i] >= result.feature_scores[i + 1]
            for i in range(len(result.feature_scores) - 1)
        )

    def test_n_features_to_select_limit(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test n_features_to_select limits number of features."""
        X, y = sample_classification_data
        selector = VarianceThresholdSelector(threshold=0.0, n_features_to_select=10)
        selector.fit(X, y)

        assert len(selector.selected_features_) <= 10
        assert selector.transform(X).shape[1] <= 10

    def test_all_features_removed_handling(self) -> None:
        """Test handling when all features have variance below threshold."""
        # Create data where all features have very low variance
        X = pd.DataFrame(
            {
                "f1": np.ones(100) * 5,
                "f2": np.ones(100) * 3,
                "f3": np.zeros(100),
            }
        )
        y = pd.Series(np.random.binomial(1, 0.5, 100))

        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(X, y)

        # All features have zero variance, so all should be removed
        # But selector should handle gracefully
        assert len(selector.selected_features_) == 0

    def test_regression_task(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test variance threshold works for regression (variance is task-agnostic)."""
        X, y = sample_classification_data
        # Make y continuous
        y_continuous = pd.Series(np.random.randn(X.shape[0]))

        selector = VarianceThresholdSelector(threshold=0.5, task="regression")
        selector.fit(X, y_continuous)

        assert selector.task == "regression"
        assert len(selector.selected_features_) > 0

    def test_variance_calculation_correctness(self) -> None:
        """Test that calculated variances match expected values."""
        # Create data with known variances
        X = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5],  # var = 2.5
                "f2": [1, 1, 1, 1, 1],  # var = 0
                "f3": [1, 3, 5, 7, 9],  # var = 10
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(X, y)

        # Check calculated variances (unbiased estimator)
        assert np.isclose(selector.variances_[0], 2.5)
        assert np.isclose(selector.variances_[1], 0.0)
        assert np.isclose(selector.variances_[2], 10.0)
