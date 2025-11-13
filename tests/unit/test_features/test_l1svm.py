"""Unit tests for L1-SVM feature selector.

This module tests the L1SVMSelector class which uses Linear SVM with L1
regularization for embedded feature selection. L1 penalty encourages sparsity,
eliminating irrelevant features by driving their weights to exactly zero.

L1-SVM combines the classification power of SVM with automatic feature selection
through L1 regularization, making it effective for high-dimensional data.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.l1svm import L1SVMSelector  # noqa: E402


@pytest.fixture
def sample_binary_classification() -> tuple[pd.DataFrame, pd.Series]:
    """Generate binary classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    weights = np.random.rand(n_informative) * 3
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)
    y = pd.Series((linear_combination > linear_combination.median()).astype(int))

    return X, y


@pytest.fixture
def sample_multiclass_classification() -> tuple[pd.DataFrame, pd.Series]:
    """Generate multi-class classification data."""
    np.random.seed(42)
    n_samples = 300
    n_features = 100

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    linear_combination = X.iloc[:, :10].sum(axis=1)
    y = pd.Series(pd.qcut(linear_combination, q=3, labels=[0, 1, 2]))

    return X, y


class TestL1SVMSelector:
    """Test suite for L1SVMSelector."""

    def test_import(self) -> None:
        """Test that L1SVMSelector can be imported."""
        assert L1SVMSelector is not None

    def test_initialization(self) -> None:
        """Test L1SVMSelector initialization with defaults."""
        selector = L1SVMSelector()
        assert selector is not None
        assert selector.C == 1.0
        assert selector.penalty == "l1"

    def test_fit_binary(self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method for binary classification."""
        X, y = sample_binary_classification
        selector = L1SVMSelector(C=0.1, n_features_to_select=20)

        result = selector.fit(X, y)

        assert result is selector
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_scores_")
        assert hasattr(selector, "model_")
        assert len(selector.selected_features_) == 20

    def test_fit_multiclass(
        self, sample_multiclass_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method for multi-class classification."""
        X, y = sample_multiclass_classification
        selector = L1SVMSelector(C=0.1, n_features_to_select=25)

        result = selector.fit(X, y)

        assert result is selector
        assert len(selector.selected_features_) == 25

    def test_transform(self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y = sample_binary_classification
        selector = L1SVMSelector(C=0.1, n_features_to_select=30)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 30
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y = sample_binary_classification
        selector = L1SVMSelector(C=0.1, n_features_to_select=25)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 25
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support returns boolean mask."""
        X, y = sample_binary_classification
        selector = L1SVMSelector(C=0.1, n_features_to_select=40)
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == 40

    def test_get_support_indices(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_binary_classification
        selector = L1SVMSelector(C=0.1, n_features_to_select=35)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == 35

    def test_different_C_values(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test different C values affect sparsity."""
        X, y = sample_binary_classification

        # Smaller C = more regularization = fewer features
        selector_small_C = L1SVMSelector(C=0.01, n_features_to_select=50)
        selector_small_C.fit(X, y)

        # Larger C = less regularization = more features selected
        selector_large_C = L1SVMSelector(C=10.0, n_features_to_select=50)
        selector_large_C.fit(X, y)

        # Both should select features
        assert len(selector_small_C.selected_features_) == 50
        assert len(selector_large_C.selected_features_) == 50

    def test_n_features_to_select_limit(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test n_features_to_select parameter."""
        X, y = sample_binary_classification

        for n_features in [5, 10, 20, 50]:
            selector = L1SVMSelector(C=0.1, n_features_to_select=n_features)
            selector.fit(X, y)

            assert len(selector.selected_features_) == n_features
            assert selector.transform(X).shape[1] == n_features

    def test_invalid_C(self) -> None:
        """Test invalid C raises ValueError."""
        with pytest.raises(ValueError, match="C must be positive"):
            L1SVMSelector(C=0.0)

        with pytest.raises(ValueError, match="C must be positive"):
            L1SVMSelector(C=-1.0)

    def test_get_result(self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_result returns FeatureSelectorResult."""
        X, y = sample_binary_classification
        selector = L1SVMSelector(C=0.1, n_features_to_select=30)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == 30
        assert len(result.feature_scores) == 30
        assert result.n_features_selected == 30
        assert result.method_name == "L1SVMSelector"
        # Scores should be sorted (descending)
        assert all(
            result.feature_scores[i] >= result.feature_scores[i + 1]
            for i in range(len(result.feature_scores) - 1)
        )

    def test_reproducibility(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test reproducibility with same random_state."""
        X, y = sample_binary_classification

        selector1 = L1SVMSelector(C=0.1, n_features_to_select=25, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = L1SVMSelector(C=0.1, n_features_to_select=25, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        assert features1 == features2

    def test_max_iter_parameter(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test max_iter parameter is respected."""
        X, y = sample_binary_classification

        selector = L1SVMSelector(C=0.1, max_iter=100, n_features_to_select=20)
        selector.fit(X, y)

        assert selector.max_iter == 100
        assert len(selector.selected_features_) == 20

    def test_dual_parameter(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test dual parameter (must be False for L1 penalty)."""
        X, y = sample_binary_classification

        selector = L1SVMSelector(C=0.1, dual=False, n_features_to_select=20)
        selector.fit(X, y)

        assert selector.dual is False
        assert len(selector.selected_features_) == 20
