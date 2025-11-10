"""Tests for Random Forest Variable Importance feature selector."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data.

    Returns:
        Tuple of (X, y) for regression task.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 100

    # Create data with some informative features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create target with dependencies on first 10 features
    y = pd.Series(
        X.iloc[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.1,
        name="target"
    )

    return X, y


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data.

    Returns:
        Tuple of (X, y) for binary classification.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 100

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create binary target with non-linear dependencies
    linear_combo = X.iloc[:, :10].sum(axis=1)
    y = pd.Series(
        (linear_combo > linear_combo.median()).astype(int),
        name="target"
    )

    return X, y


class TestRandomForestSelector:
    """Test suite for RandomForestSelector."""

    def test_import(self) -> None:
        """Test that RandomForestSelector can be imported."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        assert RandomForestSelector is not None

    def test_initialization(self) -> None:
        """Test RandomForestSelector initialization with default parameters."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        selector = RandomForestSelector()

        assert selector.n_estimators == 100
        assert selector.max_depth is None
        assert selector.min_samples_split == 2
        assert selector.min_samples_leaf == 1
        assert selector.task == 'classification'
        assert selector.importance_type == 'gini'
        assert selector.n_jobs == -1

    def test_initialization_custom_parameters(self) -> None:
        """Test RandomForestSelector initialization with custom parameters."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        selector = RandomForestSelector(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            task='regression',
            importance_type='permutation',
            n_features_to_select=50,
            random_state=42
        )

        assert selector.n_estimators == 200
        assert selector.max_depth == 10
        assert selector.min_samples_split == 5
        assert selector.task == 'regression'
        assert selector.importance_type == 'permutation'
        assert selector.n_features_to_select == 50
        assert selector.random_state == 42

    def test_invalid_n_estimators_raises_error(self) -> None:
        """Test that invalid n_estimators raises ValueError."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        with pytest.raises(ValueError, match="n_estimators must be positive"):
            RandomForestSelector(n_estimators=0)

        with pytest.raises(ValueError, match="n_estimators must be positive"):
            RandomForestSelector(n_estimators=-10)

    def test_invalid_task_raises_error(self) -> None:
        """Test that invalid task raises ValueError."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        with pytest.raises(ValueError, match="task must be"):
            RandomForestSelector(task='invalid_task')

    def test_invalid_importance_type_raises_error(self) -> None:
        """Test that invalid importance_type raises ValueError."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        with pytest.raises(ValueError, match="importance_type must be"):
            RandomForestSelector(importance_type='invalid_type')

    def test_fit_classification(self, sample_classification_data: tuple) -> None:
        """Test that RandomForestSelector can be fitted for classification."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(
            n_estimators=50,
            task='classification',
            n_features_to_select=20
        )

        result = selector.fit(X, y)

        # Check that selector returns self
        assert result is selector

        # Check that attributes are set
        assert selector.selected_features_ is not None
        assert selector.feature_scores_ is not None
        assert selector.support_mask_ is not None
        assert selector.n_features_in_ == X.shape[1]
        assert selector.model_ is not None
        assert len(selector.selected_features_) <= 20

    def test_fit_regression(self, sample_regression_data: tuple) -> None:
        """Test that RandomForestSelector can be fitted for regression."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_regression_data
        selector = RandomForestSelector(
            n_estimators=50,
            task='regression',
            n_features_to_select=20
        )

        result = selector.fit(X, y)

        # Check that selector returns self
        assert result is selector

        # Check that attributes are set
        assert selector.selected_features_ is not None
        assert selector.feature_scores_ is not None
        assert selector.support_mask_ is not None
        assert selector.n_features_in_ == X.shape[1]
        assert selector.model_ is not None

    def test_fit_with_gini_importance(self, sample_classification_data: tuple) -> None:
        """Test fitting with Gini importance (default)."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(
            n_estimators=50,
            importance_type='gini',
            n_features_to_select=15
        )

        selector.fit(X, y)

        assert len(selector.selected_features_) == 15
        assert all(score >= 0 for score in selector.feature_scores_)

    @pytest.mark.slow
    def test_fit_with_permutation_importance(self, sample_classification_data: tuple) -> None:
        """Test fitting with permutation importance."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(
            n_estimators=30,  # Fewer estimators for speed
            importance_type='permutation',
            n_features_to_select=15
        )

        selector.fit(X, y)

        assert len(selector.selected_features_) == 15
        # Permutation importance can be negative
        assert selector.feature_scores_ is not None

    def test_transform(self, sample_classification_data: tuple) -> None:
        """Test transform method."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(n_estimators=50, n_features_to_select=20)

        selector.fit(X, y)
        X_transformed = selector.transform(X)

        # Check transformed shape
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 20

        # Check that all selected features are in original data
        for col in X_transformed.columns:
            assert col in X.columns

    def test_fit_transform(self, sample_classification_data: tuple) -> None:
        """Test fit_transform method."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(n_estimators=50, n_features_to_select=20)

        X_transformed = selector.fit_transform(X, y)

        # Check transformed shape
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 20

        # Check that selector is fitted
        assert selector.selected_features_ is not None

    def test_get_support_mask(self, sample_classification_data: tuple) -> None:
        """Test get_support returns boolean mask."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(n_estimators=50, n_features_to_select=15)

        selector.fit(X, y)
        mask = selector.get_support(indices=False)

        # Check mask properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape[0] == X.shape[1]
        assert mask.sum() == 15

    def test_get_support_indices(self, sample_classification_data: tuple) -> None:
        """Test get_support returns feature indices."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(n_estimators=50, n_features_to_select=15)

        selector.fit(X, y)
        indices = selector.get_support(indices=True)

        # Check indices properties
        assert isinstance(indices, np.ndarray)
        assert indices.dtype in [np.int32, np.int64]
        assert len(indices) == 15
        assert all(0 <= idx < X.shape[1] for idx in indices)

    def test_n_features_to_select_limit(self, sample_classification_data: tuple) -> None:
        """Test that n_features_to_select limits the number of selected features."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        n_select = 10

        selector = RandomForestSelector(n_estimators=50, n_features_to_select=n_select)
        selector.fit(X, y)

        # Check that exactly n_select features are selected
        assert len(selector.selected_features_) == n_select
        assert selector.get_support().sum() == n_select

    def test_auto_select_all_nonzero(self, sample_classification_data: tuple) -> None:
        """Test that without n_features_to_select, all non-zero importance features are selected."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(n_estimators=50)  # No n_features_to_select

        selector.fit(X, y)

        # All features with non-zero importance should be selected
        assert len(selector.selected_features_) > 0
        assert all(score > 0 for score in selector.feature_scores_)

    def test_reproducibility(self, sample_classification_data: tuple) -> None:
        """Test that results are reproducible with same random_state."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data

        selector1 = RandomForestSelector(n_estimators=50, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_
        scores1 = selector1.feature_scores_

        selector2 = RandomForestSelector(n_estimators=50, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_
        scores2 = selector2.feature_scores_

        # Check that same features are selected with same scores
        assert features1 == features2
        np.testing.assert_array_almost_equal(scores1, scores2)

    def test_get_feature_importance(self, sample_classification_data: tuple) -> None:
        """Test get_feature_importance method returns full importance array."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(
            n_estimators=50,
            importance_type='gini',
            n_features_to_select=15
        )

        selector.fit(X, y)
        importances = selector.get_feature_importance()

        # Check importances properties
        assert len(importances) == X.shape[1]  # All features
        assert all(score >= 0 for score in importances)
        assert np.sum(np.abs(importances)) > 0  # At least some non-zero

    def test_get_feature_importance_before_fit_raises_error(self) -> None:
        """Test that get_feature_importance before fit raises ValueError."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        selector = RandomForestSelector()

        with pytest.raises(ValueError, match="not fitted"):
            selector.get_feature_importance()

    def test_transform_before_fit_raises_error(self) -> None:
        """Test that transform before fit raises ValueError."""
        from omicselector2.features.classical.random_forest import RandomForestSelector
        import pandas as pd

        selector = RandomForestSelector()
        X = pd.DataFrame(np.random.randn(10, 5))

        with pytest.raises(ValueError, match="not fitted"):
            selector.transform(X)

    def test_get_support_before_fit_raises_error(self) -> None:
        """Test that get_support before fit raises ValueError."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        selector = RandomForestSelector()

        with pytest.raises(ValueError, match="not fitted"):
            selector.get_support()

    def test_identifies_informative_features(self, sample_regression_data: tuple) -> None:
        """Test that Random Forest identifies informative features."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_regression_data
        selector = RandomForestSelector(
            n_estimators=100,
            task='regression',
            n_features_to_select=20
        )

        selector.fit(X, y)

        # The first 10 features are informative (by design)
        # Check that many of these are in top selections
        informative_features = set([f"feature_{i}" for i in range(10)])
        selected_set = set(selector.selected_features_[:15])  # Top 15

        # At least half of top 15 should be informative
        overlap = len(selected_set.intersection(informative_features))
        assert overlap >= 5, f"Expected >= 5 informative features in top 15, got {overlap}"

    def test_get_result(self, sample_classification_data: tuple) -> None:
        """Test get_result method returns FeatureSelectorResult."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data
        selector = RandomForestSelector(n_estimators=50, n_features_to_select=20)

        selector.fit(X, y)
        result = selector.get_result()

        # Check result properties
        assert result is not None
        assert len(result.selected_features) == 20
        assert len(result.feature_scores) == 20
        assert all(score >= 0 for score in result.feature_scores)

    def test_repr(self) -> None:
        """Test string representation."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        selector1 = RandomForestSelector(n_estimators=100, task='classification')
        repr1 = repr(selector1)

        assert 'RandomForestSelector' in repr1
        assert 'n_estimators=100' in repr1
        assert "task='classification'" in repr1

        selector2 = RandomForestSelector(
            n_estimators=200,
            task='regression',
            n_features_to_select=50
        )
        repr2 = repr(selector2)

        assert 'n_estimators=200' in repr2
        assert "task='regression'" in repr2
        assert 'n_features_to_select=50' in repr2

    def test_max_depth_parameter(self, sample_classification_data: tuple) -> None:
        """Test that max_depth parameter affects model."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data

        # Shallow trees
        selector_shallow = RandomForestSelector(
            n_estimators=50,
            max_depth=3,
            n_features_to_select=15
        )
        selector_shallow.fit(X, y)

        # Deep trees
        selector_deep = RandomForestSelector(
            n_estimators=50,
            max_depth=None,  # No limit
            n_features_to_select=15
        )
        selector_deep.fit(X, y)

        # Both should work and select features
        assert len(selector_shallow.selected_features_) == 15
        assert len(selector_deep.selected_features_) == 15

    def test_min_samples_split_parameter(self, sample_classification_data: tuple) -> None:
        """Test that min_samples_split parameter affects model."""
        from omicselector2.features.classical.random_forest import RandomForestSelector

        X, y = sample_classification_data

        selector = RandomForestSelector(
            n_estimators=50,
            min_samples_split=10,
            n_features_to_select=15
        )
        selector.fit(X, y)

        assert len(selector.selected_features_) == 15
        assert selector.model_.min_samples_split == 10
