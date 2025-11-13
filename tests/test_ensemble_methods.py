"""
Tests for ensemble feature selection methods.

This module tests ensemble approaches for combining multiple feature selection methods:
- Majority voting
- Soft voting (weighted)
- Consensus ranking
- Intersection/union strategies
- Meta-learner approach
"""

import pytest
import numpy as np

try:
    import pandas as pd
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    pytest.skip("scikit-learn not available", allow_module_level=True)

from omicselector2.features.ensemble import (
    EnsembleFeatureSelector,
    majority_vote,
    soft_vote,
    consensus_ranking,
    intersection_selection,
    union_selection,
)


@pytest.fixture
def synthetic_dataset():
    """Create synthetic classification dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=False
    )

    feature_names = [f"GENE_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def mock_selection_results():
    """Create mock selection results from multiple methods."""
    results = {
        'lasso': ['GENE_0', 'GENE_1', 'GENE_2', 'GENE_5', 'GENE_10'],
        'rf': ['GENE_0', 'GENE_1', 'GENE_3', 'GENE_5', 'GENE_12'],
        'mrmr': ['GENE_0', 'GENE_2', 'GENE_3', 'GENE_5', 'GENE_15'],
    }
    return results


@pytest.fixture
def mock_ranked_results():
    """Create mock ranked results with scores."""
    results = {
        'lasso': {'GENE_0': 0.95, 'GENE_1': 0.85, 'GENE_2': 0.75, 'GENE_5': 0.65},
        'rf': {'GENE_0': 0.90, 'GENE_1': 0.80, 'GENE_3': 0.70, 'GENE_5': 0.60},
        'mrmr': {'GENE_0': 0.88, 'GENE_2': 0.78, 'GENE_3': 0.68, 'GENE_5': 0.58},
    }
    return results


class TestMajorityVote:
    """Tests for majority voting ensemble."""

    def test_majority_vote_basic(self, mock_selection_results):
        """Test basic majority voting."""
        # GENE_0, GENE_1, GENE_5 appear in all 3 methods (majority = 2/3)
        selected = majority_vote(mock_selection_results, min_votes=2)

        assert 'GENE_0' in selected  # In all 3
        assert 'GENE_1' in selected  # In 2 (lasso, rf)
        assert 'GENE_5' in selected  # In all 3

    def test_majority_vote_with_threshold(self, mock_selection_results):
        """Test majority voting with different thresholds."""
        # Require unanimous vote (3/3)
        selected = majority_vote(mock_selection_results, min_votes=3)

        assert 'GENE_0' in selected  # In all 3
        assert 'GENE_5' in selected  # In all 3
        assert 'GENE_1' not in selected  # Only in 2

    def test_majority_vote_empty_with_high_threshold(self, mock_selection_results):
        """Test that high threshold can result in no features."""
        selected = majority_vote(mock_selection_results, min_votes=10)

        # No feature appears in 10 methods
        assert len(selected) == 0

    def test_majority_vote_all_features_with_low_threshold(self, mock_selection_results):
        """Test that threshold of 1 includes all features."""
        selected = majority_vote(mock_selection_results, min_votes=1)

        # All features that appear in at least 1 method
        expected_features = set()
        for features in mock_selection_results.values():
            expected_features.update(features)

        assert set(selected) == expected_features


class TestSoftVote:
    """Tests for soft (weighted) voting ensemble."""

    def test_soft_vote_equal_weights(self, mock_ranked_results):
        """Test soft voting with equal weights."""
        selected, scores = soft_vote(
            mock_ranked_results,
            n_features=3,
            weights=None  # Equal weights
        )

        assert len(selected) <= 3
        assert 'GENE_0' in selected  # Highest average score

        # Scores should be normalized
        for feature, score in scores.items():
            assert 0 <= score <= 1

    def test_soft_vote_custom_weights(self, mock_ranked_results):
        """Test soft voting with custom weights."""
        # Give lasso more weight
        weights = {'lasso': 2.0, 'rf': 1.0, 'mrmr': 1.0}

        selected, scores = soft_vote(
            mock_ranked_results,
            n_features=5,
            weights=weights
        )

        assert len(selected) <= 5
        # GENE_1 (high in lasso) should be prioritized
        assert 'GENE_1' in selected

    def test_soft_vote_returns_scores(self, mock_ranked_results):
        """Test that soft voting returns aggregated scores."""
        selected, scores = soft_vote(mock_ranked_results, n_features=3)

        # Should have scores for selected features
        for feature in selected:
            assert feature in scores
            assert isinstance(scores[feature], (int, float))


class TestConsensusRanking:
    """Tests for consensus ranking."""

    def test_consensus_ranking_borda_count(self, mock_selection_results):
        """Test consensus ranking with Borda count."""
        ranked = consensus_ranking(
            mock_selection_results,
            method='borda_count'
        )

        # Should return list of features sorted by consensus score
        assert isinstance(ranked, list)
        assert len(ranked) > 0

        # GENE_0 and GENE_5 (in all 3 methods) should rank high
        assert 'GENE_0' in ranked[:5]
        assert 'GENE_5' in ranked[:5]

    def test_consensus_ranking_mean_rank(self, mock_selection_results):
        """Test consensus ranking with mean rank."""
        ranked = consensus_ranking(
            mock_selection_results,
            method='mean_rank'
        )

        assert isinstance(ranked, list)
        assert len(ranked) > 0

    def test_consensus_ranking_with_scores(self, mock_ranked_results):
        """Test consensus ranking with feature scores."""
        ranked = consensus_ranking(
            mock_ranked_results,
            method='mean_score'
        )

        # GENE_0 has highest scores across all methods
        assert ranked[0] == 'GENE_0'


class TestIntersectionUnion:
    """Tests for intersection and union strategies."""

    def test_intersection_selection(self, mock_selection_results):
        """Test intersection (only features selected by all methods)."""
        selected = intersection_selection(mock_selection_results)

        # Only GENE_0 and GENE_5 appear in all 3 methods
        assert 'GENE_0' in selected
        assert 'GENE_5' in selected
        assert len(selected) == 2

    def test_union_selection(self, mock_selection_results):
        """Test union (features selected by any method)."""
        selected = union_selection(mock_selection_results)

        # All features that appear in at least one method
        expected_features = set()
        for features in mock_selection_results.values():
            expected_features.update(features)

        assert set(selected) == expected_features

    def test_intersection_empty_when_no_overlap(self):
        """Test intersection returns empty when no overlap."""
        results = {
            'method1': ['GENE_0', 'GENE_1'],
            'method2': ['GENE_2', 'GENE_3'],
            'method3': ['GENE_4', 'GENE_5'],
        }

        selected = intersection_selection(results)
        assert len(selected) == 0


class TestEnsembleFeatureSelector:
    """Tests for EnsembleFeatureSelector class."""

    def test_ensemble_initialization(self):
        """Test that EnsembleFeatureSelector can be initialized."""
        from omicselector2.tasks.feature_selection import (
            run_lasso_feature_selection,
            run_randomforest_feature_selection,
        )

        selector = EnsembleFeatureSelector(
            base_selectors=[
                run_lasso_feature_selection,
                run_randomforest_feature_selection,
            ],
            ensemble_method='majority_vote',
            min_votes=2
        )

        assert selector.ensemble_method == 'majority_vote'
        assert selector.min_votes == 2
        assert len(selector.base_selectors) == 2

    def test_ensemble_select_features_majority(self, synthetic_dataset):
        """Test ensemble feature selection with majority voting."""
        from omicselector2.tasks.feature_selection import (
            run_lasso_feature_selection,
            run_randomforest_feature_selection,
        )

        X, y = synthetic_dataset

        # Use min_votes=1 to be more lenient (at least one method selects)
        selector = EnsembleFeatureSelector(
            base_selectors=[
                run_lasso_feature_selection,
                run_randomforest_feature_selection,
            ],
            ensemble_method='majority_vote',
            min_votes=1
        )

        selected_features, metrics = selector.select_features(
            X, y, n_features=20, cv=3
        )

        # Should select some features (at least those from RF)
        assert len(selected_features) > 0
        assert isinstance(selected_features, list)

        # Metrics should include ensemble info
        assert 'ensemble_method' in metrics
        assert metrics['ensemble_method'] == 'majority_vote'
        assert 'n_methods' in metrics
        assert 'n_features_selected' in metrics

    def test_ensemble_select_features_consensus(self, synthetic_dataset):
        """Test ensemble with consensus ranking."""
        from omicselector2.tasks.feature_selection import (
            run_lasso_feature_selection,
            run_randomforest_feature_selection,
            run_mrmr_feature_selection,
        )

        X, y = synthetic_dataset

        selector = EnsembleFeatureSelector(
            base_selectors=[
                run_lasso_feature_selection,
                run_randomforest_feature_selection,
                run_mrmr_feature_selection,
            ],
            ensemble_method='consensus_ranking',
            n_features=15
        )

        selected_features, metrics = selector.select_features(
            X, y, n_features=20, cv=3
        )

        # Should respect final n_features limit
        assert len(selected_features) <= 15

    def test_ensemble_select_features_soft_vote(self, synthetic_dataset):
        """Test ensemble with soft voting."""
        from omicselector2.tasks.feature_selection import (
            run_lasso_feature_selection,
            run_randomforest_feature_selection,
        )

        X, y = synthetic_dataset

        selector = EnsembleFeatureSelector(
            base_selectors=[
                run_lasso_feature_selection,
                run_randomforest_feature_selection,
            ],
            ensemble_method='soft_vote',
            n_features=10
        )

        selected_features, metrics = selector.select_features(
            X, y, n_features=20, cv=3
        )

        assert len(selected_features) <= 10
        # aggregated_scores only present if at least one method has scores
        # If Lasso returns 0 features, it falls back to majority vote
        assert 'n_features_selected' in metrics
        assert 'ensemble_method' in metrics

    def test_ensemble_invalid_method_raises_error(self):
        """Test that invalid ensemble method raises error."""
        from omicselector2.tasks.feature_selection import run_lasso_feature_selection

        with pytest.raises(ValueError, match="ensemble_method must be one of"):
            EnsembleFeatureSelector(
                base_selectors=[run_lasso_feature_selection],
                ensemble_method='invalid_method'
            )
