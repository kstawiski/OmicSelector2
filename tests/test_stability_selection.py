"""
Tests for stability selection framework.

This module tests the stability selection implementation:
- Bootstrap sampling
- Stability score calculation
- Threshold-based feature selection
- Aggregation across multiple methods
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

from omicselector2.features.stability import (
    StabilitySelector,
    calculate_stability_scores,
    aggregate_feature_rankings,
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
        shuffle=False,
    )

    feature_names = [f"GENE_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


class TestStabilitySelector:
    """Tests for StabilitySelector class."""

    def test_stability_selector_initialization(self):
        """Test that StabilitySelector can be initialized."""
        from omicselector2.tasks.feature_selection import run_lasso_feature_selection

        selector = StabilitySelector(
            base_selector=run_lasso_feature_selection,
            n_bootstraps=10,
            threshold=0.6,
            sample_fraction=0.8,
        )

        assert selector.n_bootstraps == 10
        assert selector.threshold == 0.6
        assert selector.sample_fraction == 0.8
        assert selector.base_selector is not None

    def test_stability_selector_selects_features(self, synthetic_dataset):
        """Test that StabilitySelector selects stable features."""
        from omicselector2.tasks.feature_selection import run_lasso_feature_selection

        X, y = synthetic_dataset

        selector = StabilitySelector(
            base_selector=run_lasso_feature_selection,
            n_bootstraps=20,
            threshold=0.6,
            sample_fraction=0.8,
        )

        stable_features, stability_scores = selector.select_stable_features(X, y, n_features=20)

        # Should select some features
        assert len(stable_features) > 0
        assert isinstance(stable_features, list)
        assert all(isinstance(f, str) for f in stable_features)

        # Should have stability scores for all features
        assert len(stability_scores) == X.shape[1]
        assert all(0 <= score <= 1 for score in stability_scores.values())

    def test_stability_scores_in_valid_range(self, synthetic_dataset):
        """Test that stability scores are between 0 and 1."""
        from omicselector2.tasks.feature_selection import run_lasso_feature_selection

        X, y = synthetic_dataset

        selector = StabilitySelector(
            base_selector=run_lasso_feature_selection, n_bootstraps=10, threshold=0.5
        )

        stable_features, stability_scores = selector.select_stable_features(X, y, n_features=10)

        # All scores should be between 0 and 1
        for feature, score in stability_scores.items():
            assert 0 <= score <= 1

    def test_stability_selector_respects_threshold(self, synthetic_dataset):
        """Test that only features above threshold are selected."""
        from omicselector2.tasks.feature_selection import run_lasso_feature_selection

        X, y = synthetic_dataset

        threshold = 0.7
        selector = StabilitySelector(
            base_selector=run_lasso_feature_selection, n_bootstraps=20, threshold=threshold
        )

        stable_features, stability_scores = selector.select_stable_features(X, y, n_features=50)

        # All selected features should have score >= threshold
        for feature in stable_features:
            assert stability_scores[feature] >= threshold

    def test_stability_selector_reproducibility(self, synthetic_dataset):
        """Test that StabilitySelector produces reproducible results with seed."""
        from omicselector2.tasks.feature_selection import run_lasso_feature_selection

        X, y = synthetic_dataset

        selector = StabilitySelector(
            base_selector=run_lasso_feature_selection,
            n_bootstraps=10,
            threshold=0.6,
            random_state=42,
        )

        # Run twice with same seed
        stable_1, scores_1 = selector.select_stable_features(X, y, n_features=10)

        selector2 = StabilitySelector(
            base_selector=run_lasso_feature_selection,
            n_bootstraps=10,
            threshold=0.6,
            random_state=42,
        )
        stable_2, scores_2 = selector2.select_stable_features(X, y, n_features=10)

        # Should get identical results
        assert stable_1 == stable_2
        assert scores_1 == scores_2


class TestStabilityScoreCalculation:
    """Tests for stability score calculation utilities."""

    def test_calculate_stability_scores_basic(self):
        """Test basic stability score calculation."""
        # Mock selection results: each bootstrap selects some features
        bootstrap_selections = [
            ["GENE_0", "GENE_1", "GENE_2"],
            ["GENE_0", "GENE_1", "GENE_3"],
            ["GENE_0", "GENE_2", "GENE_3"],
            ["GENE_0", "GENE_1", "GENE_2"],
        ]

        all_features = ["GENE_0", "GENE_1", "GENE_2", "GENE_3", "GENE_4"]

        scores = calculate_stability_scores(bootstrap_selections, all_features)

        # GENE_0 selected 4/4 times = 1.0
        assert scores["GENE_0"] == 1.0

        # GENE_1 selected 3/4 times = 0.75
        assert scores["GENE_1"] == 0.75

        # GENE_2 selected 3/4 times = 0.75
        assert scores["GENE_2"] == 0.75

        # GENE_3 selected 2/4 times = 0.5
        assert scores["GENE_3"] == 0.5

        # GENE_4 never selected = 0.0
        assert scores["GENE_4"] == 0.0

    def test_calculate_stability_scores_with_counts(self):
        """Test stability score calculation with explicit counts."""
        selection_counts = {
            "GENE_0": 10,
            "GENE_1": 7,
            "GENE_2": 5,
            "GENE_3": 0,
        }

        n_bootstraps = 10

        scores = calculate_stability_scores(
            bootstrap_selections=None,
            all_features=["GENE_0", "GENE_1", "GENE_2", "GENE_3"],
            selection_counts=selection_counts,
            n_bootstraps=n_bootstraps,
        )

        assert scores["GENE_0"] == 1.0
        assert scores["GENE_1"] == 0.7
        assert scores["GENE_2"] == 0.5
        assert scores["GENE_3"] == 0.0


class TestFeatureRankingAggregation:
    """Tests for aggregating feature rankings across methods."""

    def test_aggregate_rankings_mean(self):
        """Test mean aggregation of feature rankings."""
        # Rankings from different methods
        rankings = {
            "lasso": {"GENE_0": 0.9, "GENE_1": 0.7, "GENE_2": 0.5},
            "rf": {"GENE_0": 0.8, "GENE_1": 0.6, "GENE_2": 0.4},
            "mrmr": {"GENE_0": 0.85, "GENE_1": 0.65, "GENE_2": 0.45},
        }

        aggregated = aggregate_feature_rankings(rankings, method="mean")

        # Check mean values
        assert aggregated["GENE_0"] == pytest.approx((0.9 + 0.8 + 0.85) / 3)
        assert aggregated["GENE_1"] == pytest.approx((0.7 + 0.6 + 0.65) / 3)
        assert aggregated["GENE_2"] == pytest.approx((0.5 + 0.4 + 0.45) / 3)

    def test_aggregate_rankings_max(self):
        """Test max aggregation of feature rankings."""
        rankings = {
            "lasso": {"GENE_0": 0.9, "GENE_1": 0.7, "GENE_2": 0.5},
            "rf": {"GENE_0": 0.8, "GENE_1": 0.9, "GENE_2": 0.4},
        }

        aggregated = aggregate_feature_rankings(rankings, method="max")

        assert aggregated["GENE_0"] == 0.9
        assert aggregated["GENE_1"] == 0.9
        assert aggregated["GENE_2"] == 0.5

    def test_aggregate_rankings_min(self):
        """Test min aggregation of feature rankings."""
        rankings = {
            "lasso": {"GENE_0": 0.9, "GENE_1": 0.7, "GENE_2": 0.5},
            "rf": {"GENE_0": 0.8, "GENE_1": 0.9, "GENE_2": 0.4},
        }

        aggregated = aggregate_feature_rankings(rankings, method="min")

        assert aggregated["GENE_0"] == 0.8
        assert aggregated["GENE_1"] == 0.7
        assert aggregated["GENE_2"] == 0.4

    def test_aggregate_rankings_median(self):
        """Test median aggregation of feature rankings."""
        rankings = {
            "method1": {"GENE_0": 0.9},
            "method2": {"GENE_0": 0.5},
            "method3": {"GENE_0": 0.7},
        }

        aggregated = aggregate_feature_rankings(rankings, method="median")

        # Median of [0.9, 0.5, 0.7] = 0.7
        assert aggregated["GENE_0"] == 0.7

    def test_aggregate_rankings_with_missing_features(self):
        """Test aggregation when methods don't all select same features."""
        rankings = {
            "lasso": {"GENE_0": 0.9, "GENE_1": 0.7},
            "rf": {"GENE_0": 0.8, "GENE_2": 0.6},
        }

        aggregated = aggregate_feature_rankings(rankings, method="mean")

        # GENE_0 present in both: (0.9 + 0.8) / 2 = 0.85
        assert aggregated["GENE_0"] == pytest.approx(0.85)

        # GENE_1 only in lasso: 0.7 / 1 = 0.7
        assert aggregated["GENE_1"] == pytest.approx(0.7)

        # GENE_2 only in rf: 0.6 / 1 = 0.6
        assert aggregated["GENE_2"] == pytest.approx(0.6)
