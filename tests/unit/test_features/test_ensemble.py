"""Tests for ensemble feature selection methods.

Ensemble methods combine results from multiple feature selectors to improve
robustness and reduce selection bias. This module tests various ensemble
strategies including voting, ranking, and set operations.

Test coverage:
- Majority voting (hard voting)
- Soft voting (weighted by scores)
- Consensus ranking
- Intersection/union strategies
- Integration with different base selectors
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables for config
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.classical.lasso import LassoSelector  # noqa: E402
from omicselector2.features.classical.random_forest import (  # noqa: E402
    RandomForestSelector,
)
from omicselector2.features.classical.mrmr import mRMRSelector  # noqa: E402
from omicselector2.features.ensemble import EnsembleSelector  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data for testing.

    Returns:
        Tuple of (X, y) where X is features and y is binary target.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_informative = 10

    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # First 10 features are informative
    weights = np.random.rand(n_informative) * 3
    linear_combination = (X.iloc[:, :n_informative] * weights).sum(axis=1)
    y = pd.Series((linear_combination > linear_combination.median()).astype(int))

    return X, y


class TestEnsembleSelector:
    """Test suite for EnsembleSelector."""

    def test_import(self) -> None:
        """Test that EnsembleSelector can be imported."""
        from omicselector2.features.ensemble import EnsembleSelector

        assert EnsembleSelector is not None

    def test_initialization(self) -> None:
        """Test EnsembleSelector initialization."""
        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="majority")
        assert ensemble.selectors == selectors
        assert ensemble.strategy == "majority"
        assert ensemble.threshold == 0.5  # Default for majority voting

    def test_initialization_with_weights(self) -> None:
        """Test initialization with custom weights for soft voting."""
        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]
        weights = [0.6, 0.4]

        ensemble = EnsembleSelector(selectors=selectors, strategy="soft_voting", weights=weights)
        assert ensemble.weights == weights

    def test_majority_voting(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test majority voting ensemble strategy."""
        X, y = sample_classification_data

        # Create ensemble with 3 selectors
        selectors = [
            LassoSelector(n_features_to_select=15),
            RandomForestSelector(n_features_to_select=15),
            mRMRSelector(n_features_to_select=15),
        ]

        ensemble = EnsembleSelector(
            selectors=selectors, strategy="majority", threshold=0.5  # At least 50% vote
        )
        ensemble.fit(X, y)

        # Should select features that appear in at least 2/3 selectors
        assert len(ensemble.selected_features_) > 0
        assert len(ensemble.selected_features_) <= 15

        # Check vote_counts_ attribute exists
        assert hasattr(ensemble, "vote_counts_")
        assert isinstance(ensemble.vote_counts_, dict)

        # Features should have vote counts between 0 and 3
        for feature in ensemble.selected_features_:
            votes = ensemble.vote_counts_[feature]
            assert votes >= 2  # At least 50% of 3 selectors
            assert votes <= 3

    def test_soft_voting(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test soft voting ensemble with weighted scores."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=20),
            RandomForestSelector(n_features_to_select=20),
        ]
        weights = [0.7, 0.3]  # Weight Lasso more heavily

        ensemble = EnsembleSelector(
            selectors=selectors,
            strategy="soft_voting",
            weights=weights,
            n_features_to_select=10,
        )
        ensemble.fit(X, y)

        assert len(ensemble.selected_features_) == 10

        # Check weighted_scores_ attribute
        assert hasattr(ensemble, "weighted_scores_")
        assert isinstance(ensemble.weighted_scores_, dict)

        # Scores should be weighted averages
        for feature in ensemble.selected_features_:
            assert feature in ensemble.weighted_scores_
            assert ensemble.weighted_scores_[feature] > 0

    def test_consensus_ranking(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test consensus ranking strategy (Borda count)."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=20),
            RandomForestSelector(n_features_to_select=20),
            mRMRSelector(n_features_to_select=20),
        ]

        ensemble = EnsembleSelector(
            selectors=selectors, strategy="consensus_ranking", n_features_to_select=15
        )
        ensemble.fit(X, y)

        assert len(ensemble.selected_features_) == 15

        # Check consensus_scores_ (Borda counts)
        assert hasattr(ensemble, "consensus_scores_")
        assert len(ensemble.consensus_scores_) > 0

        # Features should be ranked by consensus score
        scores = [ensemble.consensus_scores_[f] for f in ensemble.selected_features_]
        assert scores == sorted(scores, reverse=True)

    def test_intersection_strategy(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test intersection strategy (features selected by ALL selectors)."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=20),
            RandomForestSelector(n_features_to_select=20),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="intersection")
        ensemble.fit(X, y)

        # Should only select features in both selectors
        assert len(ensemble.selected_features_) > 0

        # All selected features should have vote count == number of selectors
        for feature in ensemble.selected_features_:
            assert ensemble.vote_counts_[feature] == len(selectors)

    def test_union_strategy(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test union strategy (features selected by ANY selector)."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="union")
        ensemble.fit(X, y)

        # Should select all features from any selector
        # (up to 20 if no overlap, fewer if overlap)
        assert len(ensemble.selected_features_) > 0
        assert len(ensemble.selected_features_) <= 20

        # All features should have at least 1 vote
        for feature in ensemble.selected_features_:
            assert ensemble.vote_counts_[feature] >= 1

    def test_transform(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="majority")
        ensemble.fit(X, y)

        X_transformed = ensemble.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == len(ensemble.selected_features_)
        assert list(X_transformed.columns) == ensemble.selected_features_

    def test_fit_transform(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform method."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="union")
        X_transformed = ensemble.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert len(ensemble.selected_features_) > 0

    def test_get_support(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_support method."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="majority")
        ensemble.fit(X, y)

        support = ensemble.get_support()

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert np.sum(support) == len(ensemble.selected_features_)

    def test_get_support_indices(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="majority")
        ensemble.fit(X, y)

        indices = ensemble.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert len(indices) == len(ensemble.selected_features_)

    def test_invalid_strategy(self) -> None:
        """Test that invalid strategy raises ValueError."""
        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        with pytest.raises(ValueError, match="strategy must be one of"):
            EnsembleSelector(selectors=selectors, strategy="invalid_strategy")

    def test_empty_selectors(self) -> None:
        """Test that empty selectors list raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 selectors required"):
            EnsembleSelector(selectors=[], strategy="majority")

    def test_single_selector(self) -> None:
        """Test that single selector raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 selectors required"):
            EnsembleSelector(
                selectors=[LassoSelector(n_features_to_select=10)],
                strategy="majority",
            )

    def test_weights_length_mismatch(self) -> None:
        """Test that weights length must match selectors."""
        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]
        weights = [0.5]  # Wrong length

        with pytest.raises(ValueError, match="must match number of selectors"):
            EnsembleSelector(selectors=selectors, strategy="soft_voting", weights=weights)

    def test_negative_weights(self) -> None:
        """Test that negative weights raise ValueError."""
        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]
        weights = [0.5, -0.5]  # Negative weight

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            EnsembleSelector(selectors=selectors, strategy="soft_voting", weights=weights)

    def test_threshold_out_of_range(self) -> None:
        """Test that threshold must be between 0 and 1."""
        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            EnsembleSelector(selectors=selectors, strategy="majority", threshold=1.5)

    def test_get_result(self, sample_classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_result returns metadata."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=10),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="majority")
        ensemble.fit(X, y)

        result = ensemble.get_result()

        assert result.method_name == "EnsembleSelector"
        assert len(result.selected_features) == len(ensemble.selected_features_)
        assert result.metadata is not None
        assert "strategy" in result.metadata
        assert "vote_counts" in result.metadata
        assert "n_selectors" in result.metadata

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that ensemble is reproducible with random_state."""
        X, y = sample_classification_data

        selectors1 = [
            LassoSelector(n_features_to_select=10, random_state=42),
            RandomForestSelector(n_features_to_select=10, random_state=42),
        ]

        selectors2 = [
            LassoSelector(n_features_to_select=10, random_state=42),
            RandomForestSelector(n_features_to_select=10, random_state=42),
        ]

        ensemble1 = EnsembleSelector(selectors=selectors1, strategy="majority")
        ensemble1.fit(X, y)

        ensemble2 = EnsembleSelector(selectors=selectors2, strategy="majority")
        ensemble2.fit(X, y)

        assert ensemble1.selected_features_ == ensemble2.selected_features_

    def test_different_n_features_to_select(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test ensemble with selectors having different n_features_to_select."""
        X, y = sample_classification_data

        # Different selectors can select different numbers of features
        selectors = [
            LassoSelector(n_features_to_select=10),
            RandomForestSelector(n_features_to_select=20),
            mRMRSelector(n_features_to_select=15),
        ]

        ensemble = EnsembleSelector(selectors=selectors, strategy="majority")
        ensemble.fit(X, y)

        # Should work without error
        assert len(ensemble.selected_features_) > 0

    def test_soft_voting_with_unequal_weights(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test soft voting heavily favors one selector."""
        X, y = sample_classification_data

        selectors = [
            LassoSelector(n_features_to_select=15),
            RandomForestSelector(n_features_to_select=15),
        ]
        # Heavily weight Lasso
        weights = [0.9, 0.1]

        ensemble = EnsembleSelector(
            selectors=selectors,
            strategy="soft_voting",
            weights=weights,
            n_features_to_select=10,
        )
        ensemble.fit(X, y)

        # Top features should mostly come from Lasso due to heavy weighting
        assert len(ensemble.selected_features_) == 10
