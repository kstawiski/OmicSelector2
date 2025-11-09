"""HEG (High-Expression Genes) feature selector for single-cell RNA-seq.

HEG selects genes based on their expression level across cells, measured by:
- Mean expression
- Median expression
- Sum expression

This simple yet effective approach identifies highly expressed genes that often
include marker genes and genes with important biological functions. HEG is
complementary to HDG (which selects based on variability).

Key features:
- Multiple aggregation metrics (mean, median, sum)
- Percentile-based selection
- Fast, unsupervised method
- No dependence on cell labels

Use cases:
- Identify highly expressed marker genes
- Filter lowly expressed genes before analysis
- Quick QC to check for expected cell type markers
- Complement to HVG selection

Examples:
    >>> from omicselector2.features.single_cell.heg import HEGSelector
    >>>
    >>> # Select top 500 genes by mean expression
    >>> selector = HEGSelector(n_features_to_select=500, metric="mean")
    >>> selector.fit(adata.to_df(), cell_types)
    >>>
    >>> # Select top 10% by expression
    >>> selector = HEGSelector(percentile=90, metric="mean")
    >>> X_top = selector.fit_transform(X, y)
"""

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from omicselector2.features.base import BaseFeatureSelector


class HEGSelector(BaseFeatureSelector):
    """High-Expression Genes (HEG) selector for scRNA-seq.

    Selects genes with highest expression levels across cells using aggregation
    metrics. This unsupervised method identifies highly expressed genes.

    Attributes:
        metric: Expression aggregation metric ("mean", "median", "sum").
        percentile: If specified, select genes above this percentile (0-100).
        expression_levels_: Dict mapping gene names to their expression levels.

    Examples:
        >>> # Basic usage with mean expression
        >>> selector = HEGSelector(n_features_to_select=500)
        >>> selector.fit(expression_matrix, cell_labels)
        >>>
        >>> # Get highly expressed genes
        >>> hegs = selector.selected_features_
        >>>
        >>> # Select by percentile
        >>> selector = HEGSelector(percentile=95)  # Top 5%
        >>> top_genes = selector.fit_transform(X, y)
    """

    VALID_METRICS = ["mean", "median", "sum"]

    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        percentile: Optional[float] = None,
        metric: Literal["mean", "median", "sum"] = "mean",
        verbose: bool = False,
    ) -> None:
        """Initialize HEG selector.

        Args:
            n_features_to_select: Number of genes to select. If None, uses percentile.
            percentile: Percentile threshold (0-100). Genes above this percentile
                are selected. If None, uses n_features_to_select.
            metric: Expression aggregation metric:
                - "mean": Mean expression across cells (default)
                - "median": Median expression (robust to outliers)
                - "sum": Total expression (sensitive to highly expressed genes)
            verbose: Print progress messages.

        Raises:
            ValueError: If metric is invalid or both/neither n_features_to_select
                and percentile are specified.
        """
        # Validate that exactly one of n_features_to_select or percentile is specified
        if n_features_to_select is not None and percentile is not None:
            raise ValueError(
                "Cannot specify both n_features_to_select and percentile. "
                "Choose one."
            )

        if n_features_to_select is None and percentile is None:
            raise ValueError(
                "Must specify either n_features_to_select or percentile"
            )

        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=None,  # HEG is deterministic
            verbose=verbose,
        )

        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"metric must be one of {self.VALID_METRICS}, got '{metric}'"
            )

        self.metric = metric
        self.percentile = percentile

        # Attributes set during fit
        self.expression_levels_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HEGSelector":
        """Fit HEG selector by computing gene expression levels.

        Note: This method is unsupervised - y (cell labels) is not used,
        but required for API consistency with other selectors.

        Args:
            X: Expression matrix (cells × genes).
            y: Cell labels (not used by HEG, can be dummy values).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        if self.verbose:
            print(f"Computing gene expression levels using metric='{self.metric}'...")

        # Compute expression level for each gene
        expression_levels = {}

        for gene in X.columns:
            gene_expr = X[gene].values

            # Compute expression based on metric
            if self.metric == "mean":
                expr_level = np.mean(gene_expr)
            elif self.metric == "median":
                expr_level = np.median(gene_expr)
            elif self.metric == "sum":
                expr_level = np.sum(gene_expr)

            expression_levels[gene] = expr_level

        self.expression_levels_ = expression_levels

        # Sort genes by expression level (descending)
        sorted_genes = sorted(
            expression_levels.items(), key=lambda x: x[1], reverse=True
        )

        # Select genes based on n_features_to_select or percentile
        if self.n_features_to_select is not None:
            # Select top-k genes
            n_select = min(self.n_features_to_select, len(sorted_genes))
            sorted_genes = sorted_genes[:n_select]
        else:
            # Select by percentile
            percentile_value = np.percentile(
                list(expression_levels.values()), self.percentile
            )
            sorted_genes = [
                (g, level) for g, level in sorted_genes if level >= percentile_value
            ]

        selected_features = [gene for gene, _ in sorted_genes]
        feature_scores = np.array([level for _, level in sorted_genes])

        # Set attributes
        self.selected_features_ = selected_features
        self.feature_scores_ = feature_scores

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for i, gene in enumerate(X.columns):
            if gene in selected_features:
                self.support_mask_[i] = True

        if self.verbose:
            print(
                f"Selected {len(self.selected_features_)} highly expressed genes "
                f"(metric={self.metric})"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting highly expressed genes.

        Args:
            X: Expression matrix to transform.

        Returns:
            DataFrame with only selected genes.

        Raises:
            ValueError: If selector not fitted yet.
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> NDArray:
        """Get mask or indices of selected genes.

        Args:
            indices: If True, return integer indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected genes.

        Raises:
            ValueError: If selector not fitted yet.
        """
        if self.support_mask_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

    def get_result(self) -> Any:
        """Get feature selection result with HEG metadata.

        Returns:
            FeatureSelectorResult with expression levels in metadata.
        """
        result = super().get_result()

        # Initialize metadata
        if result.metadata is None:
            result.metadata = {}

        # Add HEG-specific metadata
        result.metadata["metric"] = self.metric
        result.metadata["percentile"] = self.percentile
        result.metadata["expression_levels"] = self.expression_levels_

        return result
