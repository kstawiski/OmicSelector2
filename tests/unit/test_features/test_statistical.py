"""Unit tests for Statistical (t-test/ANOVA/F-test) feature selector.

This module tests the StatisticalSelector class which uses:
- t-test for binary classification
- ANOVA F-test for multi-class classification
- F-test for regression

Statistical tests measure the relationship between each feature and the target,
selecting features with the strongest univariate associations.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.statistical import StatisticalSelector  # noqa: E402


@pytest.fixture
def sample_binary_classification() -> tuple[pd.DataFrame, pd.Series]:
    """Generate binary classification data with informative features."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Make first 10 features strongly associated with target
    weights = np.random.rand(n_informative) * 3
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)
    y = pd.Series((linear_combination > linear_combination.median()).astype(int))

    return X, y


@pytest.fixture
def sample_multiclass_classification() -> tuple[pd.DataFrame, pd.Series]:
    """Generate multi-class classification data."""
    np.random.seed(42)
    n_samples = 300
    n_features = 50
    n_classes = 3

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create 3-class target based on first 10 features
    linear_combination = X.iloc[:, :10].sum(axis=1)
    y = pd.Series(pd.qcut(linear_combination, q=n_classes, labels=[0, 1, 2]))

    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate regression data."""
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


class TestStatisticalSelector:
    """Test suite for StatisticalSelector."""

    def test_import(self) -> None:
        """Test that StatisticalSelector can be imported."""
        assert StatisticalSelector is not None

    def test_initialization(self) -> None:
        """Test StatisticalSelector initialization with defaults."""
        selector = StatisticalSelector()
        assert selector is not None
        assert selector.n_features_to_select == 10
        assert selector.task == "classification"

    def test_fit_binary_classification(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method on binary classification (uses t-test)."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=15)

        result = selector.fit(X, y)

        assert result is selector
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "scores_")
        assert hasattr(selector, "pvalues_")
        assert len(selector.selected_features_) == 15

    def test_fit_multiclass_classification(
        self, sample_multiclass_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method on multi-class classification (uses ANOVA)."""
        X, y = sample_multiclass_classification
        selector = StatisticalSelector(n_features_to_select=20, task="classification")

        result = selector.fit(X, y)

        assert result is selector
        assert len(selector.selected_features_) == 20
        assert len(selector.scores_) == 20
        assert len(selector.pvalues_) == 20

    def test_fit_regression(
        self, sample_regression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit method on regression (uses F-test)."""
        X, y = sample_regression_data
        selector = StatisticalSelector(n_features_to_select=15, task="regression")

        result = selector.fit(X, y)

        assert result is selector
        assert selector.task == "regression"
        assert len(selector.selected_features_) == 15

    def test_transform(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=20)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 20
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=15)

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 15
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support returns boolean mask."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=25)
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == 25

    def test_get_support_indices(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=30)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == 30

    def test_scores_are_sorted(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that features are sorted by score (descending)."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=20)
        selector.fit(X, y)

        # Scores should be in descending order
        assert all(selector.scores_[i] >= selector.scores_[i + 1]
                   for i in range(len(selector.scores_) - 1))

    def test_pvalues_available(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that p-values are calculated and available."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=15)
        selector.fit(X, y)

        assert selector.pvalues_ is not None
        assert len(selector.pvalues_) == 15
        # P-values should be between 0 and 1
        assert all(0 <= p <= 1 for p in selector.pvalues_)

    def test_invalid_n_features(self) -> None:
        """Test invalid n_features_to_select raises ValueError."""
        with pytest.raises(ValueError, match="n_features_to_select must be positive"):
            StatisticalSelector(n_features_to_select=0)

    def test_invalid_task(self) -> None:
        """Test invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'regression' or 'classification'"):
            StatisticalSelector(task="invalid")

    def test_get_result(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_result returns FeatureSelectorResult."""
        X, y = sample_binary_classification
        selector = StatisticalSelector(n_features_to_select=20)
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == 20
        assert len(result.feature_scores) == 20
        assert result.n_features_selected == 20
        assert result.method_name == "StatisticalSelector"
        # Metadata should include p-values
        assert "pvalues" in result.metadata

    def test_reproducibility(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test reproducibility (should be deterministic)."""
        X, y = sample_binary_classification

        selector1 = StatisticalSelector(n_features_to_select=15)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = StatisticalSelector(n_features_to_select=15)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        # Should be identical (deterministic method)
        assert features1 == features2

    def test_identifies_informative_features(
        self, sample_binary_classification: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that statistical test preferentially selects informative features."""
        X, y = sample_binary_classification

        # First 10 features are informative
        selector = StatisticalSelector(n_features_to_select=15)
        selector.fit(X, y)

        informative_features = [f"feature_{i}" for i in range(10)]
        selected = selector.selected_features_

        # Should select at least some informative features
        overlap = [f for f in selected if f in informative_features]
        assert len(overlap) > 0
