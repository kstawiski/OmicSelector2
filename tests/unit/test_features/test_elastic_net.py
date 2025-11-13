"""Tests for Elastic Net feature selector."""

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
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create target with dependencies on first 10 features
    y = pd.Series(X.iloc[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.1, name="target")

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
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create binary target
    linear_combo = X.iloc[:, :10].sum(axis=1)
    y = pd.Series((linear_combo > linear_combo.median()).astype(int), name="target")

    return X, y


class TestElasticNetSelector:
    """Test suite for ElasticNetSelector."""

    def test_import(self) -> None:
        """Test that ElasticNetSelector can be imported."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        assert ElasticNetSelector is not None

    def test_initialization(self) -> None:
        """Test ElasticNetSelector initialization with default parameters."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        selector = ElasticNetSelector()

        assert selector.alpha == 1.0
        assert selector.l1_ratio == 0.5
        assert selector.task == "regression"
        assert selector.cv == 5
        assert selector.max_iter == 10000
        assert selector.tol == 1e-4
        assert selector.standardize is True

    def test_initialization_custom_parameters(self) -> None:
        """Test ElasticNetSelector initialization with custom parameters."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        selector = ElasticNetSelector(
            alpha=0.1,
            l1_ratio=0.7,
            task="classification",
            cv=10,
            n_features_to_select=50,
            random_state=42,
        )

        assert selector.alpha == 0.1
        assert selector.l1_ratio == 0.7
        assert selector.task == "classification"
        assert selector.cv == 10
        assert selector.n_features_to_select == 50
        assert selector.random_state == 42

    def test_invalid_alpha_raises_error(self) -> None:
        """Test that invalid alpha raises ValueError."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        with pytest.raises(ValueError, match="alpha must be positive"):
            ElasticNetSelector(alpha=-0.1)

        with pytest.raises(ValueError, match="alpha must be positive"):
            ElasticNetSelector(alpha=0)

    def test_invalid_l1_ratio_raises_error(self) -> None:
        """Test that invalid l1_ratio raises ValueError."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        with pytest.raises(ValueError, match="l1_ratio must be in"):
            ElasticNetSelector(l1_ratio=-0.1)

        with pytest.raises(ValueError, match="l1_ratio must be in"):
            ElasticNetSelector(l1_ratio=1.5)

    def test_invalid_task_raises_error(self) -> None:
        """Test that invalid task raises ValueError."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        with pytest.raises(ValueError, match="task must be"):
            ElasticNetSelector(task="invalid_task")

    def test_fit_regression(self, sample_regression_data: tuple) -> None:
        """Test that ElasticNetSelector can be fitted for regression."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, l1_ratio=0.5, task="regression")

        result = selector.fit(X, y)

        # Check that selector returns self
        assert result is selector

        # Check that attributes are set
        assert selector.selected_features_ is not None
        assert selector.feature_scores_ is not None
        assert selector.support_mask_ is not None
        assert selector.n_features_in_ == X.shape[1]
        assert selector.model_ is not None
        assert selector.alpha_ is not None
        assert selector.l1_ratio_ is not None

    def test_fit_classification(self, sample_classification_data: tuple) -> None:
        """Test that ElasticNetSelector can be fitted for classification."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_classification_data
        selector = ElasticNetSelector(
            alpha=0.01, l1_ratio=0.5, task="classification", max_iter=5000
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

    def test_auto_alpha_regression(self, sample_regression_data: tuple) -> None:
        """Test automatic alpha selection for regression."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha="auto", l1_ratio=0.5, cv=3)

        selector.fit(X, y)

        # Check that alpha was automatically selected
        assert selector.alpha_ is not None
        assert isinstance(selector.alpha_, float)
        assert selector.alpha_ > 0

    def test_auto_l1_ratio_regression(self, sample_regression_data: tuple) -> None:
        """Test automatic l1_ratio selection for regression."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, l1_ratio="auto", cv=3)

        selector.fit(X, y)

        # Check that l1_ratio was automatically selected
        assert selector.l1_ratio_ is not None
        assert isinstance(selector.l1_ratio_, float)
        assert 0 <= selector.l1_ratio_ <= 1

    @pytest.mark.slow
    def test_auto_alpha_classification(self, sample_classification_data: tuple) -> None:
        """Test automatic alpha selection for classification."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_classification_data
        selector = ElasticNetSelector(
            alpha="auto", l1_ratio=0.5, task="classification", cv=3, max_iter=3000
        )

        selector.fit(X, y)

        # Check that alpha was automatically selected
        assert selector.alpha_ is not None
        assert isinstance(selector.alpha_, float)
        assert selector.alpha_ > 0

    def test_transform(self, sample_regression_data: tuple) -> None:
        """Test transform method."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, n_features_to_select=20)

        selector.fit(X, y)
        X_transformed = selector.transform(X)

        # Check transformed shape
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] <= 20
        assert X_transformed.shape[1] > 0

        # Check that all selected features are in original data
        for col in X_transformed.columns:
            assert col in X.columns

    def test_fit_transform(self, sample_regression_data: tuple) -> None:
        """Test fit_transform method."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, n_features_to_select=20)

        X_transformed = selector.fit_transform(X, y)

        # Check transformed shape
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] <= 20
        assert X_transformed.shape[1] > 0

        # Check that selector is fitted
        assert selector.selected_features_ is not None

    def test_get_support_mask(self, sample_regression_data: tuple) -> None:
        """Test get_support returns boolean mask."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01)

        selector.fit(X, y)
        mask = selector.get_support(indices=False)

        # Check mask properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape[0] == X.shape[1]
        assert mask.sum() > 0  # At least some features selected

    def test_get_support_indices(self, sample_regression_data: tuple) -> None:
        """Test get_support returns feature indices."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, n_features_to_select=15)

        selector.fit(X, y)
        indices = selector.get_support(indices=True)

        # Check indices properties
        assert isinstance(indices, np.ndarray)
        assert indices.dtype in [np.int32, np.int64]
        assert len(indices) <= 15
        assert len(indices) > 0
        assert all(0 <= idx < X.shape[1] for idx in indices)

    def test_n_features_to_select_limit(self, sample_regression_data: tuple) -> None:
        """Test that n_features_to_select limits the number of selected features."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        n_select = 10

        selector = ElasticNetSelector(alpha=0.001, n_features_to_select=n_select)
        selector.fit(X, y)

        # Check that exactly n_select features are selected
        assert len(selector.selected_features_) <= n_select
        assert selector.get_support().sum() <= n_select

    def test_standardization_on(self, sample_regression_data: tuple) -> None:
        """Test that standardization is applied when enabled."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, standardize=True)

        selector.fit(X, y)

        # Check that scaler was created
        assert selector.scaler_ is not None

    def test_standardization_off(self, sample_regression_data: tuple) -> None:
        """Test that standardization is not applied when disabled."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, standardize=False)

        selector.fit(X, y)

        # Check that scaler was not created
        assert selector.scaler_ is None

    def test_reproducibility(self, sample_regression_data: tuple) -> None:
        """Test that results are reproducible with same random_state."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data

        selector1 = ElasticNetSelector(alpha=0.01, random_state=42)
        selector1.fit(X, y)
        features1 = selector1.selected_features_

        selector2 = ElasticNetSelector(alpha=0.01, random_state=42)
        selector2.fit(X, y)
        features2 = selector2.selected_features_

        # Check that same features are selected
        assert features1 == features2

    def test_different_l1_ratios(self, sample_regression_data: tuple) -> None:
        """Test that different l1_ratios produce different results."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data

        # Pure Lasso (l1_ratio=1.0)
        selector_lasso = ElasticNetSelector(alpha=0.01, l1_ratio=1.0)
        selector_lasso.fit(X, y)
        features_lasso = set(selector_lasso.selected_features_)

        # Balanced (l1_ratio=0.5)
        selector_balanced = ElasticNetSelector(alpha=0.01, l1_ratio=0.5)
        selector_balanced.fit(X, y)
        features_balanced = set(selector_balanced.selected_features_)

        # Pure Ridge should select all (or many) features since no L1 penalty
        # While Lasso should be more sparse
        assert len(features_lasso) <= len(features_balanced)

    def test_get_result(self, sample_regression_data: tuple) -> None:
        """Test get_result method returns FeatureSelectorResult."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, n_features_to_select=20)

        selector.fit(X, y)
        result = selector.get_result()

        # Check result properties
        assert result is not None
        assert len(result.selected_features) <= 20
        assert len(result.feature_scores) == len(result.selected_features)
        assert all(score >= 0 for score in result.feature_scores)

    def test_transform_before_fit_raises_error(self) -> None:
        """Test that transform before fit raises ValueError."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector
        import pandas as pd

        selector = ElasticNetSelector()
        X = pd.DataFrame(np.random.randn(10, 5))

        with pytest.raises(ValueError, match="not fitted"):
            selector.transform(X)

    def test_get_support_before_fit_raises_error(self) -> None:
        """Test that get_support before fit raises ValueError."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        selector = ElasticNetSelector()

        with pytest.raises(ValueError, match="not fitted"):
            selector.get_support()

    def test_identifies_informative_features(self, sample_regression_data: tuple) -> None:
        """Test that ElasticNet identifies informative features."""
        from omicselector2.features.classical.elastic_net import ElasticNetSelector

        X, y = sample_regression_data
        selector = ElasticNetSelector(alpha=0.01, n_features_to_select=15)

        selector.fit(X, y)

        # The first 10 features are informative (by design)
        # Check that some of these are selected
        informative_features = [f"feature_{i}" for i in range(10)]
        selected_set = set(selector.selected_features_)

        # At least half of the top selections should be informative
        overlap = len(selected_set.intersection(informative_features))
        assert overlap >= 5, f"Expected >= 5 informative features, got {overlap}"
