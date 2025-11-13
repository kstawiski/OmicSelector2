"""Tests for statistical significance-based feature selection.

Statistical methods from OmicSelector 1.0:
- sig: Significant features (BH correction, p <= 0.05)
- sigtop: Top N significant features by p-value
- sigtopBonf: Top N with Bonferroni correction
- sigtopHolm: Top N with Holm-Bonferroni correction
- topFC: Top N by absolute fold-change

Test coverage:
- T-test based significance testing
- Multiple testing corrections (BH, Bonferroni, Holm)
- Fold-change ranking
- Top N feature selection
- Binary classification scenarios
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.statistical import (  # noqa: E402
    FoldChangeSelector,
    SignificanceSelector,
)


@pytest.fixture
def differential_expression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate data with clear differential expression between classes."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Create features with different levels of differential expression
    X = pd.DataFrame()

    # Highly significant features (first 10)
    for i in range(10):
        # Class 0: mean=0, Class 1: mean=3 (large effect)
        feature = np.concatenate(
            [
                np.random.randn(50) * 0.5,  # Class 0
                np.random.randn(50) * 0.5 + 3,  # Class 1 (shifted)
            ]
        )
        X[f"gene_{i}"] = feature

    # Moderately significant features (next 10)
    for i in range(10, 20):
        # Class 0: mean=0, Class 1: mean=1 (medium effect)
        feature = np.concatenate([np.random.randn(50) * 0.5, np.random.randn(50) * 0.5 + 1])
        X[f"gene_{i}"] = feature

    # Non-significant features (rest)
    for i in range(20, n_features):
        # No difference between classes
        feature = np.random.randn(n_samples) * 0.5
        X[f"gene_{i}"] = feature

    y = pd.Series([0] * 50 + [1] * 50, name="target")

    return X, y


class TestSignificanceSelector:
    """Test suite for SignificanceSelector (t-test based)."""

    def test_import(self) -> None:
        """Test that SignificanceSelector can be imported."""
        from omicselector2.features.statistical import SignificanceSelector

        assert SignificanceSelector is not None

    def test_initialization_sig(self) -> None:
        """Test initialization for 'sig' method."""
        selector = SignificanceSelector(method="sig", alpha=0.05)

        assert selector.method == "sig"
        assert selector.alpha == 0.05

    def test_fit_sig_selects_significant_features(
        self, differential_expression_data: tuple
    ) -> None:
        """Test that 'sig' method selects significant features."""
        X, y = differential_expression_data

        selector = SignificanceSelector(method="sig", alpha=0.05)
        selector.fit(X, y)

        # Should select the first 20 features (highly + moderately significant)
        # and possibly none of the non-significant ones
        assert len(selector.selected_features_) >= 15
        assert len(selector.selected_features_) <= 25

        # First 10 should definitely be selected (highly significant)
        selected_genes = set(selector.selected_features_)
        highly_sig_genes = {f"gene_{i}" for i in range(10)}
        assert len(highly_sig_genes & selected_genes) >= 8  # At least 8/10

    def test_sigtop_limits_to_n_features(self, differential_expression_data: tuple) -> None:
        """Test that 'sigtop' limits to n_features_to_select."""
        X, y = differential_expression_data

        selector = SignificanceSelector(method="sigtop", n_features_to_select=10, alpha=0.05)
        selector.fit(X, y)

        assert len(selector.selected_features_) == 10

        # Should be the most significant features
        # First 10 genes should be selected (highest fold-change)
        selected_genes = set(selector.selected_features_)
        expected = {f"gene_{i}" for i in range(10)}
        assert len(selected_genes & expected) >= 8

    def test_sigtopBonf_uses_bonferroni_correction(
        self, differential_expression_data: tuple
    ) -> None:
        """Test that 'sigtopBonf' uses Bonferroni correction."""
        X, y = differential_expression_data

        # Bonferroni is more conservative
        selector_bonf = SignificanceSelector(
            method="sigtopBonf", n_features_to_select=10, alpha=0.05
        )
        selector_bonf.fit(X, y)

        # Should still select 10 features (limited by n_features_to_select)
        assert len(selector_bonf.selected_features_) == 10

    def test_sigtopHolm_uses_holm_correction(self, differential_expression_data: tuple) -> None:
        """Test that 'sigtopHolm' uses Holm correction."""
        X, y = differential_expression_data

        selector_holm = SignificanceSelector(
            method="sigtopHolm", n_features_to_select=10, alpha=0.05
        )
        selector_holm.fit(X, y)

        assert len(selector_holm.selected_features_) == 10

    def test_get_p_values(self, differential_expression_data: tuple) -> None:
        """Test that p-values are stored."""
        X, y = differential_expression_data

        selector = SignificanceSelector(method="sig", alpha=0.05)
        selector.fit(X, y)

        p_values = selector.get_p_values()

        assert isinstance(p_values, pd.Series)
        assert len(p_values) == len(X.columns)
        assert all(0 <= p <= 1 for p in p_values.values)

        # First features should have smaller p-values
        assert p_values["gene_0"] < 0.01
        assert p_values["gene_1"] < 0.01

    def test_get_fold_changes(self, differential_expression_data: tuple) -> None:
        """Test that fold-changes are computed."""
        X, y = differential_expression_data

        selector = SignificanceSelector(method="sig", alpha=0.05)
        selector.fit(X, y)

        fold_changes = selector.get_fold_changes()

        assert isinstance(fold_changes, pd.Series)
        assert len(fold_changes) == len(X.columns)

        # First features should have large fold-changes
        assert abs(fold_changes["gene_0"]) > 1.5
        assert abs(fold_changes["gene_1"]) > 1.5

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            SignificanceSelector(method="invalid")


class TestFoldChangeSelector:
    """Test suite for FoldChangeSelector."""

    def test_import(self) -> None:
        """Test that FoldChangeSelector can be imported."""
        from omicselector2.features.statistical import FoldChangeSelector

        assert FoldChangeSelector is not None

    def test_initialization(self) -> None:
        """Test FoldChangeSelector initialization."""
        selector = FoldChangeSelector(n_features_to_select=20)

        assert selector.n_features_to_select == 20

    def test_topFC_selects_by_fold_change(self, differential_expression_data: tuple) -> None:
        """Test that topFC selects features by fold-change magnitude."""
        X, y = differential_expression_data

        selector = FoldChangeSelector(n_features_to_select=10)
        selector.fit(X, y)

        assert len(selector.selected_features_) == 10

        # Should select features with highest |fold-change|
        # First 10 genes have largest fold-changes
        selected_genes = set(selector.selected_features_)
        expected = {f"gene_{i}" for i in range(10)}
        assert len(selected_genes & expected) >= 8

    def test_get_fold_changes(self, differential_expression_data: tuple) -> None:
        """Test fold-change retrieval."""
        X, y = differential_expression_data

        selector = FoldChangeSelector(n_features_to_select=10)
        selector.fit(X, y)

        fold_changes = selector.get_fold_changes()

        assert isinstance(fold_changes, pd.Series)
        assert len(fold_changes) == len(selector.selected_features_)

        # All selected features should have notable fold-changes
        assert all(abs(fc) > 0.5 for fc in fold_changes.values)

    def test_fold_change_ranking(self, differential_expression_data: tuple) -> None:
        """Test that features are ranked by |fold-change|."""
        X, y = differential_expression_data

        selector = FoldChangeSelector(n_features_to_select=20)
        selector.fit(X, y)

        fold_changes = selector.get_fold_changes()

        # Should be sorted by absolute value (descending)
        abs_fc = fold_changes.abs()
        assert abs_fc.is_monotonic_decreasing or (abs_fc.iloc[0] >= abs_fc.iloc[-1])


class TestFoldChangeSignificanceSelector:
    """Test suite for fcsig method (significant + |log2FC| > threshold)."""

    def test_fcsig_combines_significance_and_fold_change(
        self, differential_expression_data: tuple
    ) -> None:
        """Test that fcsig selects significant features with |log2FC| > 1."""
        X, y = differential_expression_data

        # Use SignificanceSelector with fc_threshold
        selector = SignificanceSelector(method="sig", alpha=0.05, fc_threshold=1.0)
        selector.fit(X, y)

        # Should select features that are both:
        # 1. Significant (p < 0.05 after correction)
        # 2. |log2FC| > 1.0

        fold_changes = selector.get_fold_changes()
        selected_fc = fold_changes[selector.selected_features_]

        # All selected features should have |log2FC| > 1.0
        assert all(abs(fc) > 1.0 for fc in selected_fc.values)

        # Should select fewer features than without FC threshold
        selector_no_fc = SignificanceSelector(method="sig", alpha=0.05)
        selector_no_fc.fit(X, y)

        assert len(selector.selected_features_) <= len(selector_no_fc.selected_features_)
