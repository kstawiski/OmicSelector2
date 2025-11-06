"""Tests for HDG (High-Deviation Genes) feature selector.

HDG selects genes with highest coefficient of variation (CV) or variance,
commonly used in single-cell RNA-seq analysis to identify highly variable genes.

Test coverage:
- Basic functionality
- Different dispersion metrics (CV, variance, std)
- Handling of zeros and low expression
- Single-cell specific scenarios
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.features.single_cell.hdg import HDGSelector  # noqa: E402


@pytest.fixture
def sample_expression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample single-cell expression data.

    Returns:
        Tuple of (X, y) where X is expression matrix and y is cell labels.
    """
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    # Create expression matrix with different variability patterns
    X_data = []

    # High variability genes (first 10)
    for _ in range(10):
        # Bimodal expression pattern (common in scRNA-seq)
        expr = np.concatenate(
            [np.random.gamma(2, 2, n_cells // 2), np.random.gamma(0.5, 1, n_cells // 2)]
        )
        np.random.shuffle(expr)
        X_data.append(expr)

    # Medium variability genes (next 20)
    for _ in range(20):
        expr = np.random.gamma(1, 1, n_cells)
        X_data.append(expr)

    # Low variability genes (next 30)
    for _ in range(30):
        expr = np.random.gamma(0.5, 0.5, n_cells)
        X_data.append(expr)

    # Constant/near-zero genes (remaining 40)
    for _ in range(40):
        expr = np.random.gamma(0.1, 0.1, n_cells)
        X_data.append(expr)

    X = pd.DataFrame(
        np.array(X_data).T, columns=[f"gene_{i}" for i in range(n_genes)]
    )

    # Binary cell type labels
    y = pd.Series(np.random.binomial(1, 0.5, n_cells))

    return X, y


class TestHDGSelector:
    """Test suite for HDG selector."""

    def test_import(self) -> None:
        """Test that HDGSelector can be imported."""
        from omicselector2.features.single_cell.hdg import HDGSelector

        assert HDGSelector is not None

    def test_initialization(self) -> None:
        """Test HDGSelector initialization."""
        selector = HDGSelector(n_features_to_select=20, metric="cv")
        assert selector.n_features_to_select == 20
        assert selector.metric == "cv"

    def test_fit_with_cv(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fitting with coefficient of variation metric."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=20, metric="cv")
        selector.fit(X, y)

        assert len(selector.selected_features_) == 20
        assert hasattr(selector, "dispersions_")
        assert len(selector.dispersions_) == X.shape[1]

        # Verify dispersions are calculated for all genes
        assert all(gene in selector.dispersions_ for gene in X.columns)

        # Verify selected genes have higher CV than non-selected genes on average
        selected_cvs = [selector.dispersions_[g] for g in selector.selected_features_]
        non_selected = [g for g in X.columns if g not in selector.selected_features_]
        non_selected_cvs = [selector.dispersions_[g] for g in non_selected if selector.dispersions_[g] > -np.inf]

        # Mean CV of selected should be higher than mean CV of non-selected
        if len(non_selected_cvs) > 0:
            assert np.mean(selected_cvs) > np.mean(non_selected_cvs)

    def test_fit_with_variance(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fitting with variance metric."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=20, metric="variance")
        selector.fit(X, y)

        assert len(selector.selected_features_) == 20
        assert hasattr(selector, "dispersions_")

    def test_fit_with_std(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fitting with standard deviation metric."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=20, metric="std")
        selector.fit(X, y)

        assert len(selector.selected_features_) == 20

    def test_transform(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test transform method."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=15)
        selector.fit(X, y)

        X_transformed = selector.transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 15
        assert list(X_transformed.columns) == selector.selected_features_

    def test_fit_transform(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit_transform method."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=15)
        X_transformed = selector.fit_transform(X, y)

        assert X_transformed.shape[1] == 15

    def test_get_support(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_support method."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=15)
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

        selector = HDGSelector(n_features_to_select=15)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)

        assert isinstance(indices, np.ndarray)
        assert len(indices) == 15

    def test_invalid_metric(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            HDGSelector(n_features_to_select=10, metric="invalid")

    def test_dispersions_sorted(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that selected genes are sorted by dispersion."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=20)
        selector.fit(X, y)

        # Check that selected features are sorted by dispersion (descending)
        selected_dispersions = [
            selector.dispersions_[f] for f in selector.selected_features_
        ]
        assert selected_dispersions == sorted(selected_dispersions, reverse=True)

    def test_cv_calculation(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test coefficient of variation calculation."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=20, metric="cv")
        selector.fit(X, y)

        # Manually calculate CV for first gene
        gene_0_expr = X.iloc[:, 0]
        mean = gene_0_expr.mean()
        if mean > 0:
            expected_cv = gene_0_expr.std() / mean
            calculated_cv = selector.dispersions_["gene_0"]
            assert np.abs(calculated_cv - expected_cv) < 1e-6

    def test_min_mean_filter(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test minimum mean expression filter."""
        X, y = sample_expression_data

        # Set high min_mean to filter out low expression genes
        selector = HDGSelector(n_features_to_select=20, metric="cv", min_mean=2.0)
        selector.fit(X, y)

        # All selected genes should have mean >= min_mean
        for feature in selector.selected_features_:
            gene_idx = X.columns.get_loc(feature)
            gene_mean = X.iloc[:, gene_idx].mean()
            assert gene_mean >= 2.0

    def test_get_result(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test get_result returns metadata."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=15)
        selector.fit(X, y)

        result = selector.get_result()

        assert result.method_name == "HDGSelector"
        assert len(result.selected_features) == 15
        assert result.metadata is not None
        assert "metric" in result.metadata
        assert "dispersions" in result.metadata

    def test_reproducibility(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that HDG is deterministic (no randomness)."""
        X, y = sample_expression_data

        selector1 = HDGSelector(n_features_to_select=20)
        selector1.fit(X, y)

        selector2 = HDGSelector(n_features_to_select=20)
        selector2.fit(X, y)

        assert selector1.selected_features_ == selector2.selected_features_

    def test_all_features_selected_when_n_exceeds_available(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that all features are selected if n_features_to_select > n_features."""
        X, y = sample_expression_data

        selector = HDGSelector(n_features_to_select=200)  # More than 100 genes
        selector.fit(X, y)

        assert len(selector.selected_features_) == X.shape[1]

    def test_works_without_labels(
        self, sample_expression_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that HDG works without cell labels (unsupervised)."""
        X, _ = sample_expression_data

        # HDG doesn't use labels, so y can be None or ignored
        selector = HDGSelector(n_features_to_select=20)
        selector.fit(X, pd.Series([0] * len(X)))  # Dummy labels

        assert len(selector.selected_features_) == 20
