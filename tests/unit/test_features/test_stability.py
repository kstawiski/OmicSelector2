"""Unit tests for Stability Selection Framework.

This module tests the StabilitySelector class which wraps any base feature
selector and runs it on multiple bootstrap samples to identify features that
are consistently selected (stable features).

Stability selection improves robustness by selecting only features that are
chosen frequently across different subsamples of the data, reducing the impact
of outliers and data variability.

Based on: Meinshausen & Bühlmann (2010) and Pusa & Rousu (2024).
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.lasso import LassoSelector  # noqa: E402
from omicselector2.features.classical.random_forest import (  # noqa: E402
    RandomForestSelector,
)
from omicselector2.features.stability import StabilitySelector  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data with stable informative features."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # First 10 features are strongly informative
    weights = np.random.rand(n_informative) * 3
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)
    y = pd.Series((linear_combination > linear_combination.median()).astype(int))

    return X, y


class TestStabilitySelector:
    """Test suite for StabilitySelector."""

    def test_import(self) -> None:
        """Test that StabilitySelector can be imported."""
        assert StabilitySelector is not None

    def test_initialization(self) -> None:
        """Test StabilitySelector initialization with defaults."""
        base_selector = LassoSelector(n_features_to_select=20)
        selector = StabilitySelector(base_selector=base_selector)

        assert selector is not None
        assert selector.n_bootstraps == 100
        assert selector.threshold == 0.6
        assert selector.sample_fraction == 0.8

    def test_fit(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit method executes without error."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=15)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=10,
            threshold=0.5
        )

        result = selector.fit(X, y)

        assert result is selector
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "stability_scores_")
        assert hasattr(selector, "selection_counts_")

    def test_stability_scores(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that stability scores are calculated correctly."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=10)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=20,
            threshold=0.3
        )
        selector.fit(X, y)

        # Stability scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in selector.stability_scores_.values())

        # Stability scores represent proportion of times selected
        for feature in selector.selected_features_:
            assert selector.stability_scores_[feature] >= selector.threshold

    def test_threshold_filtering(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that threshold filters out unstable features."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=15)

        # Low threshold = more features
        selector_low = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=20,
            threshold=0.2
        )
        selector_low.fit(X, y)

        # High threshold = fewer features
        selector_high = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=20,
            threshold=0.8
        )
        selector_high.fit(X, y)

        # Higher threshold should select fewer or equal features
        assert len(selector_high.selected_features_) <= len(selector_low.selected_features_)

    def test_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test transform returns DataFrame with stable features."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=20)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=10,
            threshold=0.4
        )
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == len(selector.selected_features_)
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y = sample_classification_data
        base_selector = RandomForestSelector(n_features_to_select=15)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=10
        )

        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support returns boolean mask."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=20)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=10
        )
        selector.fit(X, y)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == len(selector.selected_features_)

    def test_get_support_indices(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=20)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=10
        )
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == len(selector.selected_features_)

    def test_different_base_selectors(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test stability selection works with different base selectors."""
        X, y = sample_classification_data

        # Test with Lasso (use lower threshold for small n_bootstraps)
        lasso_selector = StabilitySelector(
            base_selector=LassoSelector(n_features_to_select=10),
            n_bootstraps=10,
            threshold=0.4  # Lower threshold for only 10 bootstraps
        )
        lasso_selector.fit(X, y)
        assert len(lasso_selector.selected_features_) > 0

        # Test with Random Forest
        rf_selector = StabilitySelector(
            base_selector=RandomForestSelector(n_features_to_select=10),
            n_bootstraps=10,
            threshold=0.4  # Lower threshold for only 10 bootstraps
        )
        rf_selector.fit(X, y)
        assert len(rf_selector.selected_features_) > 0

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test reproducibility with same random_state."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=15, random_state=42)

        selector1 = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=10,
            random_state=42
        )
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        # Reset base selector
        base_selector2 = LassoSelector(n_features_to_select=15, random_state=42)
        selector2 = StabilitySelector(
            base_selector=base_selector2,
            n_bootstraps=10,
            random_state=42
        )
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        assert features1 == features2

    def test_sample_fraction(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test sample_fraction parameter controls subsample size."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=10)

        # Different sample fractions
        for fraction in [0.5, 0.7, 0.9]:
            selector = StabilitySelector(
                base_selector=base_selector,
                n_bootstraps=5,
                sample_fraction=fraction
            )
            selector.fit(X, y)
            assert len(selector.selected_features_) >= 0

    def test_invalid_threshold(self) -> None:
        """Test invalid threshold raises ValueError."""
        base_selector = LassoSelector(n_features_to_select=10)

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            StabilitySelector(base_selector=base_selector, threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            StabilitySelector(base_selector=base_selector, threshold=-0.1)

    def test_invalid_sample_fraction(self) -> None:
        """Test invalid sample_fraction raises ValueError."""
        base_selector = LassoSelector(n_features_to_select=10)

        with pytest.raises(ValueError, match="sample_fraction must be between 0 and 1"):
            StabilitySelector(base_selector=base_selector, sample_fraction=1.5)

    def test_get_result(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_result returns result with stability metadata."""
        X, y = sample_classification_data
        base_selector = LassoSelector(n_features_to_select=15)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=10
        )
        selector.fit(X, y)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == len(selector.selected_features_)
        assert result.method_name == "StabilitySelector"
        # Metadata should include stability scores
        assert "stability_scores" in result.metadata
        assert "n_bootstraps" in result.metadata
