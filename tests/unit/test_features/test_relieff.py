"""Unit tests for ReliefF feature selector.

This module tests the ReliefFSelector class which uses the ReliefF algorithm
for instance-based feature selection. ReliefF evaluates features by how well
they distinguish between instances that are near each other.

ReliefF is particularly good at detecting feature interactions and works well
with noisy, high-dimensional data common in genomics.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.relieff import ReliefFSelector  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data with interactions."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create target with feature interactions
    y = pd.Series(
        ((X["feature_0"] > 0) & (X["feature_1"] > 0) |
         (X["feature_2"] < -0.5)).astype(int)
    )

    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    weights = np.random.rand(n_informative) * 2
    y = pd.Series((X.iloc[:, :n_informative] * weights).sum(axis=1) +
                  np.random.randn(n_samples) * 0.5)

    return X, y


class TestReliefFSelector:
    """Test suite for ReliefFSelector."""

    def test_import(self) -> None:
        """Test that ReliefFSelector can be imported."""
        assert ReliefFSelector is not None

    def test_initialization(self) -> None:
        """Test ReliefFSelector initialization with defaults."""
        selector = ReliefFSelector()
        assert selector is not None
        assert selector.n_features_to_select == 10
        assert selector.n_neighbors == 10
        assert selector.task == "classification"

    def test_fit_classification(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method for classification task."""
        X, y = sample_classification_data
        selector = ReliefFSelector(n_features_to_select=15, n_neighbors=10)

        result = selector.fit(X, y)

        assert result is selector
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_scores_")
        assert len(selector.selected_features_) == 15

    def test_fit_regression(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method for regression task."""
        X, y = sample_regression_data
        selector = ReliefFSelector(task="regression", n_features_to_select=20)

        result = selector.fit(X, y)

        assert result is selector
        assert selector.task == "regression"
        assert len(selector.selected_features_) == 20

    def test_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y = sample_classification_data
        selector = ReliefFSelector(n_features_to_select=20)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 20
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y = sample_classification_data
        selector = ReliefFSelector(n_features_to_select=15)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 15
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support returns boolean mask."""
        X, y = sample_classification_data
        selector = ReliefFSelector(n_features_to_select=25)
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == 25

    def test_get_support_indices(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_classification_data
        selector = ReliefFSelector(n_features_to_select=30)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == 30

    def test_n_neighbors_parameter(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test different n_neighbors values."""
        X, y = sample_classification_data

        selector_few = ReliefFSelector(n_neighbors=5, n_features_to_select=10)
        selector_few.fit(X, y)

        selector_many = ReliefFSelector(n_neighbors=20, n_features_to_select=10)
        selector_many.fit(X, y)

        # Both should select features, potentially different
        assert len(selector_few.selected_features_) == 10
        assert len(selector_many.selected_features_) == 10

    def test_invalid_n_features(self) -> None:
        """Test invalid n_features_to_select raises ValueError."""
        with pytest.raises(ValueError, match="n_features_to_select must be positive"):
            ReliefFSelector(n_features_to_select=0)

    def test_invalid_n_neighbors(self) -> None:
        """Test invalid n_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="n_neighbors must be positive"):
            ReliefFSelector(n_neighbors=0)

    def test_invalid_task(self) -> None:
        """Test invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'regression' or 'classification'"):
            ReliefFSelector(task="invalid")

    def test_get_result(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_result returns FeatureSelectorResult."""
        X, y = sample_classification_data
        selector = ReliefFSelector(n_features_to_select=20)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == 20
        assert len(result.feature_scores) == 20
        assert result.n_features_selected == 20
        assert result.method_name == "ReliefFSelector"

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test reproducibility with same random_state."""
        X, y = sample_classification_data

        selector1 = ReliefFSelector(n_features_to_select=15, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = ReliefFSelector(n_features_to_select=15, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        assert features1 == features2

    def test_scores_are_sorted(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that features are sorted by score (descending)."""
        X, y = sample_classification_data
        selector = ReliefFSelector(n_features_to_select=20)
        selector.fit(X, y)

        # Scores should be in descending order
        assert all(selector.feature_scores_[i] >= selector.feature_scores_[i + 1]
                   for i in range(len(selector.feature_scores_) - 1))

    def test_handles_small_datasets(self) -> None:
        """Test ReliefF handles small datasets gracefully."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(20, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.binomial(1, 0.5, 20))

        # n_neighbors should be adjusted if larger than n_samples
        selector = ReliefFSelector(n_neighbors=10, n_features_to_select=5)
        selector.fit(X, y)

        assert len(selector.selected_features_) == 5
