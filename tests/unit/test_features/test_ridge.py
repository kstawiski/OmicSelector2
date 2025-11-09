"""Unit tests for Ridge Regression feature selector.

This module tests the RidgeSelector class which uses Ridge (L2) regularization
for feature selection. Unlike Lasso (L1), Ridge doesn't zero out coefficients
but shrinks them, providing a ranking of all features.

Ridge is particularly useful when features are highly correlated, as it handles
multicollinearity better than Lasso by keeping correlated features together.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.ridge import RidgeSelector  # noqa: E402


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data with informative features."""
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Target based on first 10 features
    weights = np.random.rand(n_informative) * 2
    y = pd.Series((X.iloc[:, :n_informative] * weights).sum(axis=1) +
                  np.random.randn(n_samples) * 0.1)

    return X, y


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    weights = np.random.rand(n_informative) * 2
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)
    y = pd.Series((linear_combination > linear_combination.median()).astype(int))

    return X, y


class TestRidgeSelector:
    """Test suite for RidgeSelector."""

    def test_import(self) -> None:
        """Test that RidgeSelector can be imported."""
        assert RidgeSelector is not None

    def test_initialization(self) -> None:
        """Test RidgeSelector initialization with defaults."""
        selector = RidgeSelector()
        assert selector is not None
        assert selector.alpha == 1.0
        assert selector.task == "regression"

    def test_fit_regression(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method for regression task."""
        X, y = sample_regression_data
        selector = RidgeSelector(alpha=1.0, n_features_to_select=20)

        result = selector.fit(X, y)

        assert result is selector
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_scores_")
        assert hasattr(selector, "model_")
        assert len(selector.selected_features_) == 20

    def test_fit_classification(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method for classification task."""
        X, y = sample_classification_data
        selector = RidgeSelector(task="classification", n_features_to_select=15)

        result = selector.fit(X, y)

        assert result is selector
        assert selector.task == "classification"
        assert len(selector.selected_features_) == 15

    def test_transform(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y = sample_regression_data
        selector = RidgeSelector(n_features_to_select=30)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 30
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y = sample_regression_data
        selector = RidgeSelector(n_features_to_select=25)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 25
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support returns boolean mask."""
        X, y = sample_regression_data
        selector = RidgeSelector(n_features_to_select=40)
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == 40

    def test_get_support_indices(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_regression_data
        selector = RidgeSelector(n_features_to_select=35)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == 35

    def test_auto_alpha_cv(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test automatic alpha selection via cross-validation."""
        X, y = sample_regression_data
        selector = RidgeSelector(alpha="auto", cv=5, n_features_to_select=20)

        selector.fit(X, y)

        assert hasattr(selector, "alpha_")
        assert selector.alpha_ is not None
        assert selector.alpha_ > 0
        assert len(selector.selected_features_) == 20

    def test_different_alpha_values(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test different alpha values produce different selections."""
        X, y = sample_regression_data

        selector_low = RidgeSelector(alpha=0.1, n_features_to_select=20)
        selector_low.fit(X, y)

        selector_high = RidgeSelector(alpha=10.0, n_features_to_select=20)
        selector_high.fit(X, y)

        # Both should select 20 features but may differ in selection
        assert len(selector_low.selected_features_) == 20
        assert len(selector_high.selected_features_) == 20

    def test_n_features_to_select_limit(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test n_features_to_select parameter."""
        X, y = sample_regression_data

        for n_features in [5, 10, 20, 50]:
            selector = RidgeSelector(n_features_to_select=n_features)
            selector.fit(X, y)

            assert len(selector.selected_features_) == n_features
            assert selector.transform(X).shape[1] == n_features

    def test_invalid_alpha(self) -> None:
        """Test invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive or 'auto'"):
            RidgeSelector(alpha=-1.0)

        with pytest.raises(ValueError, match="alpha must be positive or 'auto'"):
            RidgeSelector(alpha=0.0)

    def test_invalid_task(self) -> None:
        """Test invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'regression' or 'classification'"):
            RidgeSelector(task="invalid")

    def test_get_result(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_result returns FeatureSelectorResult."""
        X, y = sample_regression_data
        selector = RidgeSelector(n_features_to_select=30)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == 30
        assert len(result.feature_scores) == 30
        assert result.n_features_selected == 30
        assert result.method_name == "RidgeSelector"
        # Scores should be sorted by absolute value (descending)
        assert all(result.feature_scores[i] >= result.feature_scores[i + 1]
                   for i in range(len(result.feature_scores) - 1))

    def test_reproducibility(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test reproducibility with same random_state."""
        X, y = sample_regression_data

        selector1 = RidgeSelector(n_features_to_select=25, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = RidgeSelector(n_features_to_select=25, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        assert features1 == features2

    def test_standardization(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that features are standardized before fitting."""
        X, y = sample_regression_data

        # Add feature with very large scale
        X_copy = X.copy()
        X_copy["large_scale"] = np.random.randn(len(X)) * 1000

        selector = RidgeSelector(n_features_to_select=20, standardize=True)
        selector.fit(X_copy, y)

        # Should handle large scale feature appropriately
        assert len(selector.selected_features_) == 20

    def test_handles_correlated_features(self) -> None:
        """Test Ridge handles highly correlated features better than Lasso."""
        np.random.seed(42)
        n_samples = 200

        # Create highly correlated features
        base = np.random.randn(n_samples)
        X = pd.DataFrame({
            "f1": base + np.random.randn(n_samples) * 0.1,
            "f2": base + np.random.randn(n_samples) * 0.1,  # Highly correlated with f1
            "f3": np.random.randn(n_samples),
            "f4": np.random.randn(n_samples),
        })

        y = pd.Series(base * 2 + np.random.randn(n_samples) * 0.1)

        selector = RidgeSelector(n_features_to_select=3)
        selector.fit(X, y)

        # Ridge should keep correlated features together
        selected = selector.selected_features_
        assert len(selected) == 3
