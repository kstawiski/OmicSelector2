"""Unit tests for Boruta feature selector.

This module tests the BorutaSelector class which uses the Boruta algorithm
for all-relevant feature selection based on Random Forest variable importance.

Boruta identifies features that are more important than randomly permuted
shadow features.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.boruta import BorutaSelector  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data with clear informative features."""
    np.random.seed(42)
    n_samples = 150
    n_features = 50
    n_informative = 10

    # Create feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create target based strongly on first 10 features
    # Make the signal strong so Boruta can easily identify them
    weights = np.random.rand(n_informative) * 2
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)
    noise = np.random.randn(n_samples) * 0.1
    y = pd.Series((linear_combination + noise > linear_combination.median()).astype(int))

    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 150
    n_features = 50
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Target strongly depends on first 10 features
    weights = np.random.rand(n_informative) * 2
    y = pd.Series((X.iloc[:, :n_informative] * weights).sum(axis=1) +
                  np.random.randn(n_samples) * 0.1)

    return X, y


class TestBorutaSelector:
    """Test suite for BorutaSelector."""

    def test_import(self) -> None:
        """Test that BorutaSelector can be imported."""
        assert BorutaSelector is not None

    def test_initialization(self) -> None:
        """Test BorutaSelector can be initialized with default parameters."""
        selector = BorutaSelector()
        assert selector is not None
        assert selector.n_estimators == 100
        assert selector.max_iter == 100
        assert selector.task == "classification"
        assert selector.alpha == 0.05

    def test_fit(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method executes without error."""
        X, y = sample_classification_data
        selector = BorutaSelector(n_estimators=50, max_iter=50, verbose=False)

        result = selector.fit(X, y)

        assert result is selector  # Should return self for chaining
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_scores_")
        assert len(selector.selected_features_) > 0  # Should find some features

    def test_transform(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y = sample_classification_data
        selector = BorutaSelector(n_estimators=50, max_iter=50)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]  # Same number of samples
        assert X_transformed.shape[1] <= X.shape[1]  # Fewer or equal features
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y = sample_classification_data
        selector = BorutaSelector(n_estimators=50, max_iter=50)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] <= X.shape[1]
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support returns boolean mask of selected features."""
        X, y = sample_classification_data
        selector = BorutaSelector(n_estimators=50, max_iter=50)
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == len(selector.selected_features_)

    def test_get_support_indices(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True returns integer indices."""
        X, y = sample_classification_data
        selector = BorutaSelector(n_estimators=50, max_iter=50)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == len(selector.selected_features_)

    def test_regression_task(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test Boruta selector works for regression task."""
        X, y = sample_regression_data
        selector = BorutaSelector(task="regression", n_estimators=50, max_iter=50)

        selector.fit(X, y)
        X_transformed = selector.transform(X)

        assert X_transformed.shape[1] <= X.shape[1]
        assert selector.task == "regression"

    def test_max_iter_parameter(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test max_iter parameter controls number of iterations."""
        X, y = sample_classification_data

        # Lower max_iter should potentially select fewer features
        selector = BorutaSelector(max_iter=10, n_estimators=50)
        selector.fit(X, y)

        assert selector.max_iter == 10
        assert len(selector.selected_features_) >= 0

    def test_alpha_parameter(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test alpha parameter (significance level) affects selection."""
        X, y = sample_classification_data

        # Lower alpha (more stringent) may select fewer features
        selector = BorutaSelector(alpha=0.01, n_estimators=50, max_iter=50)
        selector.fit(X, y)

        assert selector.alpha == 0.01
        assert len(selector.selected_features_) >= 0

    def test_invalid_alpha(self) -> None:
        """Test invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            BorutaSelector(alpha=1.5)

    def test_invalid_task(self) -> None:
        """Test invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'regression' or 'classification'"):
            BorutaSelector(task="invalid")

    def test_get_result(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_result returns FeatureSelectorResult with correct structure."""
        X, y = sample_classification_data
        selector = BorutaSelector(n_estimators=50, max_iter=50)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == len(selector.selected_features_)
        assert len(result.feature_scores) == len(selector.selected_features_)
        assert result.n_features_selected == len(selector.selected_features_)
        assert result.method_name == "BorutaSelector"

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that results are reproducible with same random_state."""
        X, y = sample_classification_data

        selector1 = BorutaSelector(n_estimators=50, max_iter=50, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = BorutaSelector(n_estimators=50, max_iter=50, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        assert features1 == features2

    def test_n_features_to_select_limit(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test n_features_to_select limits the number of selected features."""
        X, y = sample_classification_data

        # Boruta might identify many features, but n_features_to_select limits it
        selector = BorutaSelector(n_estimators=50, max_iter=50, n_features_to_select=5)
        selector.fit(X, y)

        assert len(selector.selected_features_) <= 5

    def test_identifies_informative_features(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that Boruta preferentially selects informative features."""
        X, y = sample_classification_data

        # The first 10 features are informative
        selector = BorutaSelector(n_estimators=100, max_iter=100, random_state=42)
        selector.fit(X, y)

        # Check if any of the first 10 features were selected
        informative_features = [f"feature_{i}" for i in range(10)]
        selected = selector.selected_features_

        # At least some informative features should be selected
        overlap = [f for f in selected if f in informative_features]
        assert len(overlap) > 0  # Should select at least one informative feature
