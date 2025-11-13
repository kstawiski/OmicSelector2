"""Tests for HEG (High-Expression Genes) feature selector.

HEG selects genes with highest mean expression across cells. Simple but effective
for identifying highly expressed marker genes and filtering lowly expressed genes
in single-cell RNA-seq data.

Test coverage:
- Basic functionality
- Different aggregation metrics (mean, median, sum)
- Percentile-based selection
- Integration with cell type labels
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.single_cell.heg import HEGSelector  # noqa: E402


@pytest.fixture
def sample_expression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample single-cell expression data.

    Returns:
        Tuple of (X, y) where X is expression matrix and y is cell labels.
    """
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    # Create expression matrix with different expression levels
    X_data = []

    # High expression genes (first 15)
    for _ in range(15):
        expr = np.random.gamma(5, 2, n_cells)  # High mean
        X_data.append(expr)

    # Medium expression genes (next 25)
    for _ in range(25):
        expr = np.random.gamma(2, 1, n_cells)  # Medium mean
        X_data.append(expr)

    # Low expression genes (remaining 60)
    for _ in range(60):
        expr = np.random.gamma(0.5, 0.5, n_cells)  # Low mean
        X_data.append(expr)

    X = pd.DataFrame(np.array(X_data).T, columns=[f"gene_{i}" for i in range(n_genes)])

    # Binary cell type labels
    y = pd.Series(np.random.binomial(1, 0.5, n_cells))

    return X, y


class TestHEGSelector:
    """Test suite for HEG selector."""

    def test_import(self) -> None:
        """Test that HEGSelector can be imported."""
        from omicselector2.features.single_cell.heg import HEGSelector

        assert HEGSelector is not None

    def test_initialization(self) -> None:
        """Test HEGSelector initialization."""
        selector = HEGSelector(n_features_to_select=20, metric="mean")
        assert selector.n_features_to_select == 20
        assert selector.metric == "mean"

    def test_fit_with_mean(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fitting with mean expression metric."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=20, metric="mean")
        selector.fit(X, y)

        assert len(selector.selected_features_) == 20
        assert hasattr(selector, "expression_levels_")
        assert len(selector.expression_levels_) == X.shape[1]

        # Verify high expression genes are selected
        selected_means = [selector.expression_levels_[g] for g in selector.selected_features_]
        non_selected = [g for g in X.columns if g not in selector.selected_features_]
        non_selected_means = [selector.expression_levels_[g] for g in non_selected]

        # Selected genes should have higher mean than non-selected
        assert np.mean(selected_means) > np.mean(non_selected_means)

    def test_fit_with_median(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fitting with median expression metric."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=20, metric="median")
        selector.fit(X, y)

        assert len(selector.selected_features_) == 20

    def test_fit_with_sum(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fitting with sum expression metric."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=20, metric="sum")
        selector.fit(X, y)

        assert len(selector.selected_features_) == 20

    def test_transform(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test transform method."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=15)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 15
        assert list(X_transformed.columns) == selector.selected_features_

    def test_fit_transform(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit_transform method."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=15)
        X_transformed = selector.fit_transform(X, y)

        assert X_transformed.shape[1] == 15

    def test_get_support(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_support method."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=15)
        selector.fit(X, y)

        support = selector.get_support()

        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert len(support) == X.shape[1]
        assert np.sum(support) == 15

    def test_get_support_indices(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support with indices=True."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=15)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert len(indices) == 15

    def test_invalid_metric(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            HEGSelector(n_features_to_select=10, metric="invalid")

    def test_expression_levels_sorted(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that selected genes are sorted by expression level."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=20)
        selector.fit(X, y)

        # Check that selected features are sorted by expression (descending)
        selected_expression = [selector.expression_levels_[f] for f in selector.selected_features_]
        assert selected_expression == sorted(selected_expression, reverse=True)

    def test_mean_calculation(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test mean expression calculation."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=20, metric="mean")
        selector.fit(X, y)

        # Manually calculate mean for first gene
        gene_0_expr = X.iloc[:, 0]
        expected_mean = gene_0_expr.mean()
        calculated_mean = selector.expression_levels_["gene_0"]
        assert np.abs(calculated_mean - expected_mean) < 1e-6

    def test_get_result(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test get_result returns metadata."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=15)
        selector.fit(X, y)

        result = selector.get_result()

        assert result.method_name == "HEGSelector"
        assert len(result.selected_features) == 15
        assert result.metadata is not None
        assert "metric" in result.metadata
        assert "expression_levels" in result.metadata

    def test_reproducibility(self, sample_expression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that HEG is deterministic (no randomness)."""
        X, y = sample_expression_data

        selector1 = HEGSelector(n_features_to_select=20)
        selector1.fit(X, y)

        selector2 = HEGSelector(n_features_to_select=20)
        selector2.fit(X, y)

        assert selector1.selected_features_ == selector2.selected_features_

    def test_all_features_selected_when_n_exceeds_available(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that all features are selected if n_features_to_select > n_features."""
        X, y = sample_expression_data

        selector = HEGSelector(n_features_to_select=200)  # More than 100 genes
        selector.fit(X, y)

        assert len(selector.selected_features_) == X.shape[1]

    def test_works_without_labels(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that HEG works without cell labels (unsupervised)."""
        X, _ = sample_expression_data

        # HEG doesn't use labels, so y can be dummy values
        selector = HEGSelector(n_features_to_select=20)
        selector.fit(X, pd.Series([0] * len(X)))

        assert len(selector.selected_features_) == 20

    def test_percentile_selection(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test selecting by percentile instead of absolute number."""
        X, y = sample_expression_data

        # Select top 10% by expression
        selector = HEGSelector(percentile=90)
        selector.fit(X, y)

        # Should select approximately 10 genes (10% of 100)
        assert 8 <= len(selector.selected_features_) <= 12

    def test_percentile_and_n_features_conflict(self) -> None:
        """Test that specifying both percentile and n_features raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            HEGSelector(n_features_to_select=20, percentile=90)

    def test_neither_n_features_nor_percentile(self) -> None:
        """Test that not specifying either raises error."""
        with pytest.raises(ValueError, match="Must specify either"):
            HEGSelector()
