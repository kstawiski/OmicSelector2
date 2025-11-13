"""
Integration tests for enhanced feature selection task.

Tests the feature selection task with:
- Single method mode
- Stability selection mode
- Ensemble mode
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from omicselector2.tasks.feature_selection import (
    run_lasso_feature_selection,
    run_randomforest_feature_selection,
    run_mrmr_feature_selection,
)


@pytest.fixture
def test_dataset():
    """Create test dataset for integration tests."""
    X, y = make_classification(
        n_samples=150,
        n_features=40,
        n_informative=15,
        n_redundant=10,
        n_classes=2,
        random_state=42,
        shuffle=False,
    )

    feature_names = [f"GENE_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


class TestSingleMethodMode:
    """Tests for single method feature selection."""

    def test_lasso_single_method(self, test_dataset):
        """Test basic Lasso feature selection."""
        X, y = test_dataset

        selected, metrics = run_lasso_feature_selection(X, y, cv=3, n_features=15)

        assert isinstance(selected, list)
        assert isinstance(metrics, dict)
        assert "method" in metrics
        assert metrics["method"] == "lasso"

    def test_rf_single_method(self, test_dataset):
        """Test Random Forest feature selection."""
        X, y = test_dataset

        selected, metrics = run_randomforest_feature_selection(X, y, cv=3, n_features=15)

        assert len(selected) > 0
        assert metrics["method"] == "random_forest"


class TestStabilityMode:
    """Tests for stability selection mode."""

    def test_stability_with_lasso(self, test_dataset):
        """Test stability selection with Lasso base selector."""
        from omicselector2.features.stability import StabilitySelector

        X, y = test_dataset

        selector = StabilitySelector(
            base_selector=run_lasso_feature_selection,
            n_bootstraps=10,  # Use small number for speed
            threshold=0.5,
            sample_fraction=0.8,
            random_state=42,
        )

        stable_features, scores = selector.select_stable_features(X, y, n_features=15, cv=3)

        # Should select features that appear consistently
        assert isinstance(stable_features, list)
        assert isinstance(scores, dict)
        assert len(scores) == X.shape[1]

        # Selected features should have high stability scores
        for feature in stable_features:
            assert scores[feature] >= 0.5

    def test_stability_with_random_forest(self, test_dataset):
        """Test stability selection with Random Forest."""
        from omicselector2.features.stability import StabilitySelector

        X, y = test_dataset

        selector = StabilitySelector(
            base_selector=run_randomforest_feature_selection,
            n_bootstraps=10,
            threshold=0.6,
            random_state=42,
        )

        stable_features, scores = selector.select_stable_features(X, y, n_features=15, cv=3)

        assert len(stable_features) > 0
        # All selected features should exceed threshold
        for f in stable_features:
            assert scores[f] >= 0.6

    def test_stability_reproducibility(self, test_dataset):
        """Test that stability selection is reproducible with same seed."""
        from omicselector2.features.stability import StabilitySelector

        X, y = test_dataset

        selector1 = StabilitySelector(
            base_selector=run_randomforest_feature_selection,
            n_bootstraps=10,
            threshold=0.5,
            random_state=42,
        )

        selector2 = StabilitySelector(
            base_selector=run_randomforest_feature_selection,
            n_bootstraps=10,
            threshold=0.5,
            random_state=42,
        )

        features1, _ = selector1.select_stable_features(X, y, n_features=15, cv=3)
        features2, _ = selector2.select_stable_features(X, y, n_features=15, cv=3)

        assert features1 == features2


class TestEnsembleMode:
    """Tests for ensemble feature selection mode."""

    def test_ensemble_majority_vote(self, test_dataset):
        """Test ensemble with majority voting."""
        from omicselector2.features.ensemble import EnsembleFeatureSelector

        X, y = test_dataset

        selector = EnsembleFeatureSelector(
            base_selectors=[
                run_lasso_feature_selection,
                run_randomforest_feature_selection,
                run_mrmr_feature_selection,
            ],
            ensemble_method="majority_vote",
            min_votes=2,
        )

        selected, metrics = selector.select_features(X, y, n_features=15, cv=3)

        assert len(selected) > 0
        assert metrics["ensemble_method"] == "majority_vote"
        assert metrics["n_methods"] >= 2

    def test_ensemble_consensus_ranking(self, test_dataset):
        """Test ensemble with consensus ranking."""
        from omicselector2.features.ensemble import EnsembleFeatureSelector

        X, y = test_dataset

        selector = EnsembleFeatureSelector(
            base_selectors=[
                run_randomforest_feature_selection,
                run_mrmr_feature_selection,
            ],
            ensemble_method="consensus_ranking",
            n_features=10,
        )

        selected, metrics = selector.select_features(X, y, n_features=20, cv=3)

        # Should respect final n_features limit
        assert len(selected) <= 10
        assert metrics["ensemble_method"] == "consensus_ranking"

    def test_ensemble_intersection(self, test_dataset):
        """Test ensemble with intersection strategy."""
        from omicselector2.features.ensemble import EnsembleFeatureSelector

        X, y = test_dataset

        selector = EnsembleFeatureSelector(
            base_selectors=[
                run_randomforest_feature_selection,
                run_mrmr_feature_selection,
            ],
            ensemble_method="intersection",
        )

        selected, metrics = selector.select_features(X, y, n_features=15, cv=3)

        # Intersection may result in fewer or no features
        assert isinstance(selected, list)
        assert metrics["ensemble_method"] == "intersection"


class TestIntegrationScenarios:
    """Tests for realistic integration scenarios."""

    def test_workflow_single_to_stability(self, test_dataset):
        """Test workflow: single method -> stability for validation."""
        X, y = test_dataset

        # First: Run single method
        lasso_features, lasso_metrics = run_lasso_feature_selection(X, y, cv=3, n_features=20)

        # Then: Validate with stability
        from omicselector2.features.stability import StabilitySelector

        stability_selector = StabilitySelector(
            base_selector=run_lasso_feature_selection,
            n_bootstraps=10,
            threshold=0.6,
            random_state=42,
        )

        stable_features, scores = stability_selector.select_stable_features(
            X, y, n_features=20, cv=3
        )

        # Stable features should be subset of or similar to single run
        # (not exact due to bootstrap variability)
        assert isinstance(stable_features, list)
        assert len(stable_features) <= 20

    def test_workflow_ensemble_with_stability_evaluation(self, test_dataset):
        """Test workflow: ensemble selection -> evaluate stability."""
        X, y = test_dataset

        # Step 1: Use ensemble to get consensus features
        from omicselector2.features.ensemble import EnsembleFeatureSelector

        ensemble_selector = EnsembleFeatureSelector(
            base_selectors=[
                run_randomforest_feature_selection,
                run_mrmr_feature_selection,
            ],
            ensemble_method="consensus_ranking",
            n_features=15,
        )

        ensemble_features, ensemble_metrics = ensemble_selector.select_features(
            X, y, n_features=20, cv=3
        )

        assert len(ensemble_features) <= 15
        assert len(ensemble_features) > 0

        # Step 2: Could further validate with stability (optional)
        # This demonstrates the full OmicSelector philosophy:
        # 1. Test multiple methods (ensemble)
        # 2. Validate stability (bootstrapping)
        # 3. Select robust signatures

    def test_config_based_selection(self, test_dataset):
        """Test config-driven feature selection (as used in Celery task)."""
        X, y = test_dataset

        # Simulate different config scenarios
        configs = [
            # Single method
            {"methods": ["lasso"], "n_features": 15, "cv_folds": 3},
            # Stability mode
            {
                "methods": ["random_forest"],
                "n_features": 15,
                "cv_folds": 3,
                "stability": {"n_bootstraps": 10, "threshold": 0.6, "sample_fraction": 0.8},
            },
            # Ensemble mode
            {
                "methods": ["lasso", "random_forest", "mrmr"],
                "n_features": 15,
                "cv_folds": 3,
                "ensemble": {"method": "majority_vote", "min_votes": 2},
            },
        ]

        for config in configs:
            # Simulate what Celery task would do
            methods = config.get("methods", ["lasso"])
            n_features = config.get("n_features", 100)
            cv_folds = config.get("cv_folds", 5)
            stability_config = config.get("stability", None)
            ensemble_config = config.get("ensemble", None)

            method_functions = {
                "lasso": run_lasso_feature_selection,
                "random_forest": run_randomforest_feature_selection,
                "mrmr": run_mrmr_feature_selection,
            }

            if ensemble_config and len(methods) > 1:
                # Ensemble mode
                from omicselector2.features.ensemble import EnsembleFeatureSelector

                selector_funcs = [method_functions[m] for m in methods if m in method_functions]
                ensemble_selector = EnsembleFeatureSelector(
                    base_selectors=selector_funcs,
                    ensemble_method=ensemble_config["method"],
                    min_votes=ensemble_config.get("min_votes", 2),
                )
                selected, metrics = ensemble_selector.select_features(X, y, n_features, cv_folds)

            elif stability_config:
                # Stability mode
                from omicselector2.features.stability import StabilitySelector

                method_name = methods[0]
                method_func = method_functions[method_name]
                stability_selector = StabilitySelector(
                    base_selector=method_func,
                    n_bootstraps=stability_config["n_bootstraps"],
                    threshold=stability_config["threshold"],
                    random_state=42,
                )
                selected, scores = stability_selector.select_stable_features(
                    X, y, n_features, cv_folds
                )
                metrics = {"stability_scores": scores}

            else:
                # Single method mode
                method_name = methods[0]
                method_func = method_functions[method_name]
                selected, metrics = method_func(X, y, cv=cv_folds, n_features=n_features)

            # All modes should return features
            assert isinstance(selected, list)
            assert isinstance(metrics, dict)
