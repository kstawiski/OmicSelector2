"""Tests for base feature selector classes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample data for testing.

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.binomial(1, 0.5, n_samples), name="target")

    return X, y


@pytest.mark.unit
def test_base_selector_can_be_imported() -> None:
    """Test that BaseFeatureSelector can be imported."""
    from omicselector2.features.base import BaseFeatureSelector

    assert BaseFeatureSelector is not None


@pytest.mark.unit
def test_base_selector_is_abstract() -> None:
    """Test that BaseFeatureSelector cannot be instantiated directly."""
    from omicselector2.features.base import BaseFeatureSelector

    with pytest.raises(TypeError, match="abstract"):
        BaseFeatureSelector()


@pytest.mark.unit
def test_base_selector_has_required_methods() -> None:
    """Test that BaseFeatureSelector defines required abstract methods."""
    from omicselector2.features.base import BaseFeatureSelector

    # Check abstract methods exist
    assert hasattr(BaseFeatureSelector, "fit")
    assert hasattr(BaseFeatureSelector, "transform")
    assert hasattr(BaseFeatureSelector, "fit_transform")
    assert hasattr(BaseFeatureSelector, "get_support")
    assert hasattr(BaseFeatureSelector, "get_feature_names_out")


@pytest.mark.unit
def test_base_selector_has_common_attributes() -> None:
    """Test that BaseFeatureSelector defines common attributes."""
    from omicselector2.features.base import BaseFeatureSelector

    # These should be class-level attributes or defined in __init__
    assert hasattr(BaseFeatureSelector, "__abstractmethods__")


@pytest.mark.unit
def test_feature_selector_result_dataclass() -> None:
    """Test FeatureSelectorResult dataclass."""
    from omicselector2.features.base import FeatureSelectorResult

    # Create a result instance
    result = FeatureSelectorResult(
        selected_features=["feature_0", "feature_1", "feature_2"],
        feature_scores=np.array([0.9, 0.8, 0.7]),
        support_mask=np.array([True, True, True, False, False]),
        n_features_selected=3,
        method_name="test_method",
    )

    assert len(result.selected_features) == 3
    assert result.n_features_selected == 3
    assert result.method_name == "test_method"
    assert len(result.feature_scores) == 3
    assert result.support_mask.sum() == 3


@pytest.mark.unit
def test_feature_selector_result_to_dataframe() -> None:
    """Test FeatureSelectorResult can be converted to DataFrame."""
    from omicselector2.features.base import FeatureSelectorResult

    result = FeatureSelectorResult(
        selected_features=["feature_0", "feature_1"],
        feature_scores=np.array([0.9, 0.8]),
        support_mask=np.array([True, True, False]),
        n_features_selected=2,
        method_name="test",
    )

    df = result.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "feature" in df.columns
    assert "score" in df.columns
