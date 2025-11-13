"""Tests for Correlation Filter feature selector.

Correlation Filter removes redundant features by identifying and removing
highly correlated features. This is useful in genomics where many genes are
correlated, leading to redundancy and multicollinearity issues.

Test coverage:
- Basic functionality with Pearson correlation
- Different correlation methods (Pearson, Spearman, Kendall)
- Threshold-based filtering
- Handling of feature groups
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.filters.correlation import CorrelationFilter  # noqa: E402


@pytest.fixture
def sample_correlated_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample data with correlated features.

    Returns:
        Tuple of (X, y) with some highly correlated features.
    """
    np.random.seed(42)
    n_samples = 200

    # Generate base features
    base1 = np.random.randn(n_samples)
    base2 = np.random.randn(n_samples)
    base3 = np.random.randn(n_samples)

    # Create correlated features
    X = pd.DataFrame(
        {
            "feature_0": base1,
            "feature_1": base1
            + np.random.randn(n_samples) * 0.1,  # High correlation with feature_0
            "feature_2": base1
            + np.random.randn(n_samples) * 0.05,  # Very high correlation with feature_0
            "feature_3": base2,
            "feature_4": base2
            + np.random.randn(n_samples) * 0.1,  # High correlation with feature_3
            "feature_5": base3,
            "feature_6": np.random.randn(n_samples),  # Independent
            "feature_7": np.random.randn(n_samples),  # Independent
        }
    )

    # Binary target
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))

    return X, y


class TestCorrelationFilter:
    """Test suite for Correlation Filter."""

    def test_import(self) -> None:
        """Test that CorrelationFilter can be imported."""
        from omicselector2.features.filters.correlation import CorrelationFilter

        assert CorrelationFilter is not None

    def test_initialization(self) -> None:
        """Test CorrelationFilter initialization."""
        selector = CorrelationFilter(threshold=0.9, method="pearson")
        assert selector.threshold == 0.9
        assert selector.method == "pearson"

    def test_fit_removes_correlated(
        self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that highly correlated features are removed."""
        X, y = sample_correlated_data

        # With threshold=0.9, should remove feature_1 and feature_2 (correlated with feature_0)
        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        # Should have removed some features
        assert len(selector.selected_features_) < X.shape[1]
        assert len(selector.selected_features_) > 0

        # Feature_0 and feature_1/2 shouldn't all be selected (redundant)
        # At least one should be removed
        correlated_group = ["feature_0", "feature_1", "feature_2"]
        selected_from_group = [f for f in correlated_group if f in selector.selected_features_]
        assert len(selected_from_group) < len(correlated_group)

    def test_fit_with_spearman(
        self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test with Spearman correlation."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9, method="spearman")
        selector.fit(X, y)

        assert len(selector.selected_features_) > 0
        assert hasattr(selector, "correlation_matrix_")

    def test_fit_with_kendall(self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test with Kendall correlation."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9, method="kendall")
        selector.fit(X, y)

        assert len(selector.selected_features_) > 0

    def test_transform(self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == len(selector.selected_features_)
        assert list(X_transformed.columns) == selector.selected_features_

    def test_fit_transform(self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit_transform method."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9)
        X_transformed = selector.fit_transform(X, y)

        assert X_transformed.shape[1] < X.shape[1]

    def test_get_support(self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_support method."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        support = selector.get_support()

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]

    def test_get_support_indices(
        self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert len(indices) == len(selector.selected_features_)

    def test_invalid_method(self) -> None:
        """Test that invalid correlation method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            CorrelationFilter(threshold=0.9, method="invalid")

    def test_invalid_threshold(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            CorrelationFilter(threshold=1.5)

    def test_correlation_matrix_stored(
        self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that correlation matrix is stored."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        assert hasattr(selector, "correlation_matrix_")
        assert selector.correlation_matrix_.shape == (X.shape[1], X.shape[1])

    def test_removed_features_tracked(
        self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that removed features are tracked."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        assert hasattr(selector, "removed_features_")
        assert isinstance(selector.removed_features_, list)

        # Removed features should not be in selected features
        for feature in selector.removed_features_:
            assert feature not in selector.selected_features_

    def test_high_threshold_keeps_more_features(
        self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that higher threshold keeps more features."""
        X, y = sample_correlated_data

        selector_low = CorrelationFilter(threshold=0.7)
        selector_low.fit(X, y)

        selector_high = CorrelationFilter(threshold=0.95)
        selector_high.fit(X, y)

        # Higher threshold should keep more features (less strict)
        assert len(selector_high.selected_features_) >= len(selector_low.selected_features_)

    def test_get_result(self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_result returns metadata."""
        X, y = sample_correlated_data

        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        result = selector.get_result()

        assert result.method_name == "CorrelationFilter"
        assert result.metadata is not None
        assert "method" in result.metadata
        assert "threshold" in result.metadata
        assert "removed_features" in result.metadata

    def test_reproducibility(self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that correlation filter is deterministic."""
        X, y = sample_correlated_data

        selector1 = CorrelationFilter(threshold=0.9)
        selector1.fit(X, y)

        selector2 = CorrelationFilter(threshold=0.9)
        selector2.fit(X, y)

        assert selector1.selected_features_ == selector2.selected_features_

    def test_no_removal_when_no_correlation(self) -> None:
        """Test that no features removed when all independent."""
        np.random.seed(42)
        # Create independent features
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
        y = pd.Series(np.random.binomial(1, 0.5, 100))

        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, y)

        # All features should be kept
        assert len(selector.selected_features_) == X.shape[1]

    def test_works_without_labels(
        self, sample_correlated_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that correlation filter works without using labels."""
        X, _ = sample_correlated_data

        # Correlation filter is unsupervised
        selector = CorrelationFilter(threshold=0.9)
        selector.fit(X, pd.Series([0] * len(X)))  # Dummy labels

        assert len(selector.selected_features_) > 0
