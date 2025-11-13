"""Unit tests for XGBoost feature selector.

This module tests the XGBoostSelector class which uses XGBoost's
gradient boosting for feature selection via feature importance scores.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.xgboost import XGBoostSelector  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    n_informative = 10

    # Create feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create target based on informative features
    linear_combination = X.iloc[:, :n_informative].sum(axis=1)
    y = pd.Series((linear_combination > linear_combination.median()).astype(int), name="target")

    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Target is sum of first 10 features + noise
    y = pd.Series(X.iloc[:, :n_informative].sum(axis=1) + np.random.randn(n_samples) * 0.1)

    return X, y


class TestXGBoostSelector:
    """Test suite for XGBoostSelector."""

    def test_import(self) -> None:
        """Test that XGBoostSelector can be imported."""
        assert XGBoostSelector is not None

    def test_initialization(self) -> None:
        """Test XGBoostSelector can be initialized with default parameters."""
        selector = XGBoostSelector()
        assert selector is not None
        assert selector.n_estimators == 100
        assert selector.importance_type == "gain"
        assert selector.task == "classification"

    def test_fit(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method executes without error."""
        X, y = sample_classification_data
        selector = XGBoostSelector(n_estimators=50)

        result = selector.fit(X, y)

        assert result is selector  # Should return self for chaining
        assert selector.model_ is not None
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_scores_")

    def test_transform(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y = sample_classification_data
        selector = XGBoostSelector(n_features_to_select=20)
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
        selector = XGBoostSelector(n_features_to_select=15)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 15
        assert hasattr(selector, "selected_features_")

    def test_get_support(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_support returns boolean mask of selected features."""
        X, y = sample_classification_data
        selector = XGBoostSelector(n_features_to_select=25)
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
        selector = XGBoostSelector(n_features_to_select=30)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == 30

    def test_importance_types(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test different importance types (gain, weight, cover)."""
        X, y = sample_classification_data

        for imp_type in ["gain", "weight", "cover"]:
            selector = XGBoostSelector(importance_type=imp_type, n_features_to_select=20)
            selector.fit(X, y)

            assert selector.importance_type == imp_type
            assert len(selector.selected_features_) == 20
            assert selector.feature_scores_ is not None

    def test_regression_task(self, sample_regression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test XGBoost selector works for regression task."""
        X, y = sample_regression_data
        selector = XGBoostSelector(task="regression", n_features_to_select=15)

        selector.fit(X, y)
        X_transformed = selector.transform(X)

        assert X_transformed.shape[1] == 15
        assert selector.task == "regression"

    def test_n_features_to_select(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test n_features_to_select parameter controls number of features."""
        X, y = sample_classification_data

        for n_features in [5, 10, 20, 50]:
            selector = XGBoostSelector(n_features_to_select=n_features)
            selector.fit(X, y)

            assert len(selector.selected_features_) == n_features
            assert selector.transform(X).shape[1] == n_features

    def test_invalid_importance_type(self) -> None:
        """Test invalid importance_type raises ValueError."""
        with pytest.raises(ValueError, match="importance_type must be one of"):
            XGBoostSelector(importance_type="invalid")

    def test_invalid_task(self) -> None:
        """Test invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'regression' or 'classification'"):
            XGBoostSelector(task="invalid")

    def test_get_result(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_result returns FeatureSelectorResult with correct structure."""
        X, y = sample_classification_data
        selector = XGBoostSelector(n_features_to_select=30)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == 30
        assert len(result.feature_scores) == 30
        assert result.n_features_selected == 30
        assert result.method_name == "XGBoostSelector"
        assert all(
            result.feature_scores[i] >= result.feature_scores[i + 1]
            for i in range(len(result.feature_scores) - 1)
        )

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that results are reproducible with same random_state."""
        X, y = sample_classification_data

        selector1 = XGBoostSelector(n_features_to_select=25, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = XGBoostSelector(n_features_to_select=25, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        assert features1 == features2

    def test_max_depth_parameter(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test max_depth parameter is passed to XGBoost model."""
        X, y = sample_classification_data

        selector = XGBoostSelector(max_depth=3, n_features_to_select=15)
        selector.fit(X, y)

        assert selector.max_depth == 3
        assert len(selector.selected_features_) == 15
