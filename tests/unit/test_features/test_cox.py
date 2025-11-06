"""Unit tests for Cox Proportional Hazards feature selector.

This module tests the CoxSelector class which uses Cox Proportional Hazards
regression for survival analysis feature selection. Cox regression is the
gold standard for identifying prognostic biomarkers in cancer research.

Cox model: λ(t|X) = λ₀(t) exp(β₁X₁ + β₂X₂ + ... + βₚXₚ)

Features are ranked by coefficient magnitude (hazard ratios), with higher
absolute coefficients indicating stronger prognostic value.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set test environment variables before importing Settings
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.cox import CoxSelector  # noqa: E402


@pytest.fixture
def sample_survival_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample survival data.

    Returns:
        Tuple of (X, y_survival) where y_survival has 'time' and 'event' columns.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_informative = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create survival data: time and event (censoring indicator)
    # Informative features influence survival time
    weights = np.random.rand(n_informative) * 2
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)

    # Survival time inversely related to risk score
    base_time = 100
    time = base_time * np.exp(-linear_combination / 10) + np.random.exponential(10, n_samples)

    # Censoring (some patients lost to follow-up)
    event = np.random.binomial(1, 0.7, n_samples)  # 70% observed events

    y_survival = pd.DataFrame({
        "time": time,
        "event": event
    })

    return X, y_survival


class TestCoxSelector:
    """Test suite for CoxSelector."""

    def test_import(self) -> None:
        """Test that CoxSelector can be imported."""
        assert CoxSelector is not None

    def test_initialization(self) -> None:
        """Test CoxSelector initialization with defaults."""
        selector = CoxSelector()
        assert selector is not None
        assert selector.n_features_to_select == 10
        assert selector.penalty is None

    def test_fit(self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test fit method with survival data."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(n_features_to_select=15)

        result = selector.fit(X, y_survival)

        assert result is selector
        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "feature_scores_")
        assert hasattr(selector, "model_")
        assert len(selector.selected_features_) == 15

    def test_transform(self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test transform returns DataFrame with selected features."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(n_features_to_select=20)
        selector.fit(X, y_survival)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 20
        assert all(col in X.columns for col in X_transformed.columns)

    def test_fit_transform(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test fit_transform performs fit and transform in one call."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(n_features_to_select=15)

        X_transformed = selector.fit_transform(X, y_survival)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 15
        assert hasattr(selector, "selected_features_")

    def test_get_support(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test get_support returns boolean mask."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(n_features_to_select=25)
        selector.fit(X, y_survival)

        support = selector.get_support(indices=False)

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert support.sum() == 25

    def test_get_support_indices(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test get_support with indices=True."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(n_features_to_select=30)
        selector.fit(X, y_survival)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == np.intp
        assert len(indices) == 30

    def test_l1_penalty(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test Cox with L1 (Lasso) penalty for sparse selection."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(penalty="l1", penalizer=0.1, n_features_to_select=20)

        selector.fit(X, y_survival)

        assert selector.penalty == "l1"
        assert len(selector.selected_features_) == 20

    def test_l2_penalty(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test Cox with L2 (Ridge) penalty."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(penalty="l2", penalizer=0.1, n_features_to_select=20)

        selector.fit(X, y_survival)

        assert selector.penalty == "l2"
        assert len(selector.selected_features_) == 20

    def test_n_features_to_select_limit(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test n_features_to_select parameter."""
        X, y_survival = sample_survival_data

        for n_features in [5, 10, 20]:
            selector = CoxSelector(n_features_to_select=n_features)
            selector.fit(X, y_survival)

            assert len(selector.selected_features_) == n_features
            assert selector.transform(X).shape[1] == n_features

    def test_invalid_n_features(self) -> None:
        """Test invalid n_features_to_select raises ValueError."""
        with pytest.raises(ValueError, match="n_features_to_select must be positive"):
            CoxSelector(n_features_to_select=0)

    def test_invalid_penalty(self) -> None:
        """Test invalid penalty raises ValueError."""
        with pytest.raises(ValueError, match="penalty must be None, 'l1', or 'l2'"):
            CoxSelector(penalty="invalid")

    def test_get_result(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test get_result returns FeatureSelectorResult."""
        X, y_survival = sample_survival_data
        selector = CoxSelector(n_features_to_select=20)
        selector.fit(X, y_survival)

        result = selector.get_result()

        assert result is not None
        assert len(result.selected_features) == 20
        assert len(result.feature_scores) == 20
        assert result.n_features_selected == 20
        assert result.method_name == "CoxSelector"
        # Scores should be sorted (descending)
        assert all(result.feature_scores[i] >= result.feature_scores[i + 1]
                   for i in range(len(result.feature_scores) - 1))

    def test_requires_time_and_event_columns(
        self, sample_survival_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test that y must have 'time' and 'event' columns."""
        X, _ = sample_survival_data

        # Missing 'event' column
        y_missing_event = pd.DataFrame({"time": np.random.rand(len(X))})
        selector = CoxSelector()

        with pytest.raises(ValueError, match="must have 'time' and 'event' columns"):
            selector.fit(X, y_missing_event)

        # Missing 'time' column
        y_missing_time = pd.DataFrame({"event": np.random.binomial(1, 0.5, len(X))})

        with pytest.raises(ValueError, match="must have 'time' and 'event' columns"):
            selector.fit(X, y_missing_time)
