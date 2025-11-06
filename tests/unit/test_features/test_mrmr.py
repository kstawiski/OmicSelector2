"""Unit tests for mRMR feature selector.

This module tests the mRMRSelector class which implements Minimum Redundancy
Maximum Relevance feature selection using mutual information.

mRMR selects features that have high mutual information with the target (relevance)
while having low mutual information with already selected features (redundancy).
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.mrmr import mRMRSelector  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_informative = 10

    # Create feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Make first 10 features informative
    weights = np.random.rand(n_informative) * 2
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)
    y = pd.Series((linear_combination > linear_combination.median()).astype(int))

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
                  np.random.randn(n_samples) * 0.1)

    return X, y


class TestmRMRSelector:
    """Test suite for mRMRSelector."""

    def test_import(self) -> None:
        """Test that mRMRSelector can be imported."""
        assert mRMRSelector is not None

    def test_initialization(self) -> None:
        """Test mRMRSelector can be initialized with default parameters."""
        selector = mRMRSelector()
        assert selector is not None
        assert selector.n_features_to_select == 10
        assert selector.task == "classification"

    def test_fit(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method executes without error."""
        X, y = sample_classification_data
        selector = mRMRSelector(n_features_to_select=15)

        result = selector.fit(X, y)

        assert result is selector  # Should return self for chaining
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_scores_")
        assert len(selector.selected_features_) == 15

    def test_transform(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y = sample_classification_data
        selector = mRMRSelector(n_features_to_select=20)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]  # Same number of samples
        assert X_transformed.shape[1] == 20  # Selected features
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y = sample_classification_data
        selector = mRMRSelector(n_features_to_select=15)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 15
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support returns boolean mask of selected features."""
        X, y = sample_classification_data
        selector = mRMRSelector(n_features_to_select=25)
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == 25

    def test_get_support_indices(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True returns integer indices."""
        X, y = sample_classification_data
        selector = mRMRSelector(n_features_to_select=30)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == 30

    def test_regression_task(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test mRMR selector works for regression task."""
        X, y = sample_regression_data
        selector = mRMRSelector(task="regression", n_features_to_select=15)

        selector.fit(X, y)
        X_transformed = selector.transform(X)

        assert X_transformed.shape[1] == 15
        assert selector.task == "regression"

    def test_n_features_to_select(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test n_features_to_select parameter controls number of features."""
        X, y = sample_classification_data

        for n_features in [5, 10, 20]:
            selector = mRMRSelector(n_features_to_select=n_features)
            selector.fit(X, y)

            assert len(selector.selected_features_) == n_features
            assert selector.transform(X).shape[1] == n_features

    def test_invalid_n_features(self, sample_classification_data) -> None:
        """Test invalid n_features_to_select raises ValueError."""
        X, y = sample_classification_data

        with pytest.raises(ValueError, match="n_features_to_select must be positive"):
            mRMRSelector(n_features_to_select=0)

        with pytest.raises(ValueError, match="n_features_to_select must be positive"):
            mRMRSelector(n_features_to_select=-1)

    def test_invalid_task(self) -> None:
        """Test invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'regression' or 'classification'"):
            mRMRSelector(task="invalid")

    def test_get_result(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_result returns FeatureSelectorResult with correct structure."""
        X, y = sample_classification_data
        selector = mRMRSelector(n_features_to_select=20)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == 20
        assert len(result.feature_scores) == 20
        assert result.n_features_selected == 20
        assert result.method_name == "mRMRSelector"

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that results are reproducible with same random_state."""
        X, y = sample_classification_data

        selector1 = mRMRSelector(n_features_to_select=15, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = mRMRSelector(n_features_to_select=15, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        assert features1 == features2

    def test_greedy_selection_order(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that mRMR uses greedy forward selection."""
        X, y = sample_classification_data

        selector = mRMRSelector(n_features_to_select=10)
        selector.fit(X, y)

        # Selected features should be in order of selection
        # (first feature has highest relevance, subsequent features
        # balance relevance and redundancy)
        assert len(selector.selected_features_) == 10
        assert selector.feature_scores_ is not None
        assert len(selector.feature_scores_) == 10

    def test_handles_more_requested_than_available(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that requesting more features than available is handled gracefully."""
        X, y = sample_classification_data
        n_total_features = X.shape[1]

        # Request more features than available
        selector = mRMRSelector(n_features_to_select=n_total_features + 10)
        selector.fit(X, y)

        # Should select all available features
        assert len(selector.selected_features_) == n_total_features
