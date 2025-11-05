"""Tests for Lasso feature selector."""

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

    # Create binary target
    linear_combo = X.iloc[:, :10].sum(axis=1)
    y = pd.Series(
        (linear_combo > linear_combo.median()).astype(int),
        name="target"
    )

    return X, y


@pytest.mark.unit
def test_lasso_selector_can_be_imported() -> None:
    """Test that LassoSelector can be imported."""
    from omicselector2.features.classical.lasso import LassoSelector

    assert LassoSelector is not None


@pytest.mark.unit
def test_lasso_selector_initialization() -> None:
    """Test LassoSelector initialization."""
    from omicselector2.features.classical.lasso import LassoSelector

    selector = LassoSelector(alpha=0.1, n_features_to_select=50)

    assert selector.alpha == 0.1
    assert selector.n_features_to_select == 50


@pytest.mark.unit
def test_lasso_selector_fit(sample_regression_data: tuple) -> None:
    """Test that LassoSelector can be fitted."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data
    selector = LassoSelector(alpha=0.01)

    result = selector.fit(X, y)

    # Check that selector returns self
    assert result is selector

    # Check that attributes are set
    assert selector.selected_features_ is not None
    assert selector.feature_scores_ is not None
    assert selector.support_mask_ is not None
    assert selector.n_features_in_ == X.shape[1]


@pytest.mark.unit
def test_lasso_selector_transform(sample_regression_data: tuple) -> None:
    """Test LassoSelector transform method."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data
    selector = LassoSelector(alpha=0.01)
    selector.fit(X, y)

    X_transformed = selector.transform(X)

    # Check output shape
    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] <= X.shape[1]
    assert X_transformed.shape[1] == len(selector.selected_features_)


@pytest.mark.unit
def test_lasso_selector_fit_transform(sample_regression_data: tuple) -> None:
    """Test LassoSelector fit_transform method."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data
    selector = LassoSelector(alpha=0.01)

    X_transformed = selector.fit_transform(X, y)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] > 0


@pytest.mark.unit
def test_lasso_selector_get_support(sample_regression_data: tuple) -> None:
    """Test LassoSelector get_support method."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data
    selector = LassoSelector(alpha=0.01)
    selector.fit(X, y)

    # Get boolean mask
    support_mask = selector.get_support(indices=False)
    assert isinstance(support_mask, np.ndarray)
    assert support_mask.dtype == bool
    assert len(support_mask) == X.shape[1]

    # Get indices
    support_indices = selector.get_support(indices=True)
    assert isinstance(support_indices, np.ndarray)
    assert support_indices.dtype in [np.int32, np.int64]
    assert len(support_indices) == support_mask.sum()


@pytest.mark.unit
def test_lasso_selector_with_cv(sample_regression_data: tuple) -> None:
    """Test LassoSelector with cross-validation for alpha selection."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data
    selector = LassoSelector(alpha='auto', cv=5)

    selector.fit(X, y)

    # Check that optimal alpha was found
    assert hasattr(selector, 'alpha_')
    assert selector.alpha_ is not None
    assert selector.alpha_ > 0


@pytest.mark.unit
def test_lasso_selector_n_features_to_select(sample_regression_data: tuple) -> None:
    """Test LassoSelector respects n_features_to_select."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data
    n_features = 20

    selector = LassoSelector(alpha=0.01, n_features_to_select=n_features)
    selector.fit(X, y)

    assert len(selector.selected_features_) <= n_features


@pytest.mark.unit
def test_lasso_selector_classification(sample_classification_data: tuple) -> None:
    """Test LassoSelector on classification task."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_classification_data
    selector = LassoSelector(alpha=0.01, task='classification')

    selector.fit(X, y)

    assert selector.selected_features_ is not None
    assert len(selector.selected_features_) > 0


@pytest.mark.unit
def test_lasso_selector_invalid_alpha() -> None:
    """Test that invalid alpha raises error."""
    from omicselector2.features.classical.lasso import LassoSelector

    with pytest.raises(ValueError, match="alpha must be positive"):
        LassoSelector(alpha=-0.1)


@pytest.mark.unit
def test_lasso_selector_get_result(sample_regression_data: tuple) -> None:
    """Test LassoSelector get_result method."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data
    selector = LassoSelector(alpha=0.01)
    selector.fit(X, y)

    result = selector.get_result()

    assert result.method_name == "LassoSelector"
    assert len(result.selected_features) > 0
    assert len(result.feature_scores) == len(result.selected_features)
    assert result.n_features_selected == len(result.selected_features)


@pytest.mark.unit
def test_lasso_selector_reproducibility(sample_regression_data: tuple) -> None:
    """Test that LassoSelector gives reproducible results."""
    from omicselector2.features.classical.lasso import LassoSelector

    X, y = sample_regression_data

    selector1 = LassoSelector(alpha=0.01, random_state=42)
    selector1.fit(X, y)

    selector2 = LassoSelector(alpha=0.01, random_state=42)
    selector2.fit(X, y)

    assert selector1.selected_features_ == selector2.selected_features_
    np.testing.assert_array_almost_equal(
        selector1.feature_scores_,
        selector2.feature_scores_
    )
