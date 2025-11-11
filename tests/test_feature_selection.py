"""
Tests for feature selection algorithms.

This module tests the feature selection methods:
- Lasso feature selection
- Feature selection metrics calculation
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

from omicselector2.tasks.feature_selection import (
    run_lasso_feature_selection,
    run_randomforest_feature_selection,
    run_elasticnet_feature_selection,
)


@pytest.fixture
def synthetic_dataset():
    """Create synthetic classification dataset for testing."""
    # Generate dataset with 100 samples, 50 features (10 informative)
    X, y = make_classification(
        n_samples=100,
        n_features=50,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=False
    )

    # Convert to DataFrame with feature names
    feature_names = [f"GENE_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="outcome")

    return X_df, y_series


class TestLassoFeatureSelection:
    """Test Lasso feature selection algorithm."""

    def test_lasso_selects_features(self, synthetic_dataset):
        """Test that Lasso selects a subset of features."""
        X, y = synthetic_dataset

        selected_features, metrics = run_lasso_feature_selection(
            X, y, cv=3, n_features=20
        )

        # Should select some features
        assert len(selected_features) > 0
        assert len(selected_features) <= 20

        # All selected features should be from the original feature set
        assert all(f in X.columns for f in selected_features)

    def test_lasso_metrics_structure(self, synthetic_dataset):
        """Test that metrics have expected structure."""
        X, y = synthetic_dataset

        selected_features, metrics = run_lasso_feature_selection(
            X, y, cv=3, n_features=20
        )

        # Check metrics structure
        assert "method" in metrics
        assert metrics["method"] == "lasso"
        assert "n_features_selected" in metrics
        assert metrics["n_features_selected"] == len(selected_features)

        if len(selected_features) > 0:
            assert "optimal_alpha" in metrics
            assert "cv_auc_mean" in metrics
            assert "cv_auc_std" in metrics
            assert "cv_folds" in metrics

            # AUC should be between 0 and 1
            assert 0 <= metrics["cv_auc_mean"] <= 1
            assert metrics["cv_auc_std"] >= 0

    def test_lasso_respects_n_features_limit(self, synthetic_dataset):
        """Test that Lasso respects the n_features limit."""
        X, y = synthetic_dataset

        # Request only 5 features
        selected_features, metrics = run_lasso_feature_selection(
            X, y, cv=3, n_features=5
        )

        # Should not select more than requested
        assert len(selected_features) <= 5

    def test_lasso_with_different_cv_folds(self, synthetic_dataset):
        """Test Lasso with different cross-validation folds."""
        X, y = synthetic_dataset

        # Test with 5-fold CV
        selected_5fold, metrics_5fold = run_lasso_feature_selection(
            X, y, cv=5, n_features=10
        )

        assert metrics_5fold["cv_folds"] == 5

        # Test with 3-fold CV
        selected_3fold, metrics_3fold = run_lasso_feature_selection(
            X, y, cv=3, n_features=10
        )

        assert metrics_3fold["cv_folds"] == 3

    def test_lasso_reproducibility(self, synthetic_dataset):
        """Test that Lasso produces reproducible results."""
        X, y = synthetic_dataset

        # Run twice with same parameters
        selected_1, metrics_1 = run_lasso_feature_selection(
            X, y, cv=3, n_features=10
        )

        selected_2, metrics_2 = run_lasso_feature_selection(
            X, y, cv=3, n_features=10
        )

        # Should get same features (due to random_state=42)
        assert selected_1 == selected_2
        assert metrics_1["optimal_alpha"] == metrics_2["optimal_alpha"]

    def test_lasso_with_small_dataset(self):
        """Test Lasso with a small dataset."""
        # Create tiny dataset
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(20, 10),
            columns=[f"GENE_{i}" for i in range(10)]
        )
        y = pd.Series(np.random.randint(0, 2, 20))

        selected_features, metrics = run_lasso_feature_selection(
            X, y, cv=2, n_features=5  # Use cv=2 for small dataset
        )

        # Should still work
        assert isinstance(selected_features, list)
        assert isinstance(metrics, dict)

    def test_lasso_feature_names_preserved(self, synthetic_dataset):
        """Test that feature names are correctly preserved."""
        X, y = synthetic_dataset

        selected_features, metrics = run_lasso_feature_selection(
            X, y, cv=3, n_features=10
        )

        # All selected features should start with "GENE_"
        assert all(f.startswith("GENE_") for f in selected_features)

        # Should be actual column names from original data
        assert all(f in X.columns.tolist() for f in selected_features)


class TestFeatureSelectionEdgeCases:
    """Test edge cases in feature selection."""

    def test_lasso_with_no_informative_features(self):
        """Test Lasso when no features are informative."""
        # Create dataset with pure noise
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(50, 20),
            columns=[f"GENE_{i}" for i in range(20)]
        )
        # Random target
        y = pd.Series(np.random.randint(0, 2, 50))

        selected_features, metrics = run_lasso_feature_selection(
            X, y, cv=3, n_features=10
        )

        # Might select few or no features
        assert len(selected_features) >= 0
        assert isinstance(metrics, dict)

    def test_lasso_with_highly_correlated_features(self):
        """Test Lasso with highly correlated features."""
        np.random.seed(42)
        # Create base feature
        base = np.random.randn(100, 1)

        # Create 10 highly correlated copies
        X_array = np.hstack([base + np.random.randn(100, 1) * 0.1 for _ in range(10)])
        X = pd.DataFrame(X_array, columns=[f"GENE_{i}" for i in range(10)])

        # Target correlated with base
        y = pd.Series((base.ravel() > 0).astype(int))

        selected_features, metrics = run_lasso_feature_selection(
            X, y, cv=3, n_features=5
        )

        # Lasso should select a subset (L1 penalty encourages sparsity)
        assert len(selected_features) > 0
        assert len(selected_features) <= 5

    def test_lasso_performance_on_informative_dataset(self):
        """Test that Lasso achieves reasonable performance on informative data."""
        # Create dataset with clear signal
        X, y = make_classification(
            n_samples=200,
            n_features=30,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42,
            class_sep=2.0  # Increase class separation
        )

        X_df = pd.DataFrame(X, columns=[f"GENE_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selected_features, metrics = run_lasso_feature_selection(
            X_df, y_series, cv=5, n_features=20
        )

        # Should select features
        assert len(selected_features) > 0

        # Should achieve reasonable AUC (>0.6) on this dataset
        if len(selected_features) > 0:
            assert metrics["cv_auc_mean"] > 0.6


class TestRandomForestFeatureSelection:
    """Test Random Forest feature selection algorithm."""

    def test_rf_selects_features(self, synthetic_dataset):
        """Test that Random Forest selects a subset of features."""
        X, y = synthetic_dataset

        selected_features, metrics = run_randomforest_feature_selection(
            X, y, cv=3, n_features=20
        )

        # Should select some features
        assert len(selected_features) > 0
        assert len(selected_features) <= 20

        # All selected features should be from the original feature set
        assert all(f in X.columns for f in selected_features)

    def test_rf_metrics_structure(self, synthetic_dataset):
        """Test that metrics have expected structure."""
        X, y = synthetic_dataset

        selected_features, metrics = run_randomforest_feature_selection(
            X, y, cv=3, n_features=20
        )

        # Check metrics structure
        assert "method" in metrics
        assert metrics["method"] == "random_forest"
        assert "n_features_selected" in metrics
        assert metrics["n_features_selected"] == len(selected_features)

        if len(selected_features) > 0:
            assert "cv_auc_mean" in metrics
            assert "cv_auc_std" in metrics
            assert "cv_folds" in metrics
            assert "feature_importances" in metrics

            # AUC should be between 0 and 1
            assert 0 <= metrics["cv_auc_mean"] <= 1
            assert metrics["cv_auc_std"] >= 0

    def test_rf_respects_n_features_limit(self, synthetic_dataset):
        """Test that Random Forest respects the n_features limit."""
        X, y = synthetic_dataset

        # Request only 5 features
        selected_features, metrics = run_randomforest_feature_selection(
            X, y, cv=3, n_features=5
        )

        # Should not select more than requested
        assert len(selected_features) <= 5

    def test_rf_reproducibility(self, synthetic_dataset):
        """Test that Random Forest produces reproducible results."""
        X, y = synthetic_dataset

        # Run twice with same parameters
        selected_1, metrics_1 = run_randomforest_feature_selection(
            X, y, cv=3, n_features=10
        )

        selected_2, metrics_2 = run_randomforest_feature_selection(
            X, y, cv=3, n_features=10
        )

        # Should get same features (due to random_state=42)
        assert selected_1 == selected_2

    def test_rf_performance_on_informative_dataset(self):
        """Test that Random Forest achieves reasonable performance."""
        # Create dataset with clear signal
        X, y = make_classification(
            n_samples=200,
            n_features=30,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42,
            class_sep=2.0
        )

        X_df = pd.DataFrame(X, columns=[f"GENE_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selected_features, metrics = run_randomforest_feature_selection(
            X_df, y_series, cv=5, n_features=20
        )

        # Should select features
        assert len(selected_features) > 0

        # Should achieve reasonable AUC (>0.6) on this dataset
        if len(selected_features) > 0:
            assert metrics["cv_auc_mean"] > 0.6


class TestElasticNetFeatureSelection:
    """Test Elastic Net feature selection algorithm."""

    def test_elasticnet_selects_features(self, synthetic_dataset):
        """Test that Elastic Net selects a subset of features."""
        X, y = synthetic_dataset

        selected_features, metrics = run_elasticnet_feature_selection(
            X, y, cv=3, n_features=20
        )

        # Should select some features
        assert len(selected_features) > 0
        assert len(selected_features) <= 20

        # All selected features should be from the original feature set
        assert all(f in X.columns for f in selected_features)

    def test_elasticnet_metrics_structure(self, synthetic_dataset):
        """Test that metrics have expected structure."""
        X, y = synthetic_dataset

        selected_features, metrics = run_elasticnet_feature_selection(
            X, y, cv=3, n_features=20
        )

        # Check metrics structure
        assert "method" in metrics
        assert metrics["method"] == "elastic_net"
        assert "n_features_selected" in metrics
        assert metrics["n_features_selected"] == len(selected_features)

        if len(selected_features) > 0:
            assert "optimal_alpha" in metrics
            assert "optimal_l1_ratio" in metrics
            assert "cv_auc_mean" in metrics
            assert "cv_auc_std" in metrics
            assert "cv_folds" in metrics

            # AUC should be between 0 and 1
            assert 0 <= metrics["cv_auc_mean"] <= 1
            assert metrics["cv_auc_std"] >= 0

    def test_elasticnet_respects_n_features_limit(self, synthetic_dataset):
        """Test that Elastic Net respects the n_features limit."""
        X, y = synthetic_dataset

        # Request only 5 features
        selected_features, metrics = run_elasticnet_feature_selection(
            X, y, cv=3, n_features=5
        )

        # Should not select more than requested
        assert len(selected_features) <= 5

    def test_elasticnet_reproducibility(self, synthetic_dataset):
        """Test that Elastic Net produces reproducible results."""
        X, y = synthetic_dataset

        # Run twice with same parameters
        selected_1, metrics_1 = run_elasticnet_feature_selection(
            X, y, cv=3, n_features=10
        )

        selected_2, metrics_2 = run_elasticnet_feature_selection(
            X, y, cv=3, n_features=10
        )

        # Should get same features (due to random_state=42)
        assert selected_1 == selected_2
        assert metrics_1["optimal_alpha"] == metrics_2["optimal_alpha"]

    def test_elasticnet_performance_on_informative_dataset(self):
        """Test that Elastic Net achieves reasonable performance."""
        # Create dataset with clear signal
        X, y = make_classification(
            n_samples=200,
            n_features=30,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42,
            class_sep=2.0
        )

        X_df = pd.DataFrame(X, columns=[f"GENE_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        selected_features, metrics = run_elasticnet_feature_selection(
            X_df, y_series, cv=5, n_features=20
        )

        # Should select features
        assert len(selected_features) > 0

        # Should achieve reasonable AUC (>0.6) on this dataset
        if len(selected_features) > 0:
            assert metrics["cv_auc_mean"] > 0.6
