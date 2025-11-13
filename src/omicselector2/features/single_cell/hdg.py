"""HDG (High-Deviation Genes) feature selector for single-cell RNA-seq.

HDG selects genes based on their variability across cells, measured by:
- Coefficient of Variation (CV) = std / mean
- Variance
- Standard deviation

This is a fundamental approach in scRNA-seq analysis for identifying highly
variable genes (HVGs) that capture biological variation while filtering out
technical noise and housekeeping genes.

Key features:
- Multiple dispersion metrics (CV, variance, std)
- Optional minimum mean expression filter
- Fast, unsupervised method
- No dependence on cell labels

Based on:
- Brennecke et al. (2013) "Accounting for technical noise in single-cell RNA-seq"
- Highly variable genes detection in Scanpy, Seurat

Examples:
    >>> from omicselector2.features.single_cell.hdg import HDGSelector
    >>>
    >>> # Select top 2000 highly variable genes by CV
    >>> selector = HDGSelector(n_features_to_select=2000, metric="cv")
    >>> selector.fit(adata.to_df(), cell_types)
    >>>
    >>> # Apply minimum mean filter to exclude low-expression genes
    >>> selector = HDGSelector(
    ...     n_features_to_select=2000,
    ...     metric="cv",
    ...     min_mean=0.01
    ... )
    >>> X_hvg = selector.fit_transform(X, y)
"""

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from omicselector2.features.base import BaseFeatureSelector


class HDGSelector(BaseFeatureSelector):
    """High-Deviation Genes (HDG) selector for scRNA-seq.

    Selects genes with highest variability across cells using dispersion metrics.
    This unsupervised method identifies genes with high biological variation.

    Attributes:
        metric: Dispersion metric ("cv", "variance", "std").
        min_mean: Minimum mean expression for gene to be considered (filters low expr).
        dispersions_: Dict mapping gene names to their dispersion values.

    Examples:
        >>> # Basic usage with coefficient of variation
        >>> selector = HDGSelector(n_features_to_select=2000)
        >>> selector.fit(expression_matrix, cell_labels)
        >>>
        >>> # Get highly variable genes
        >>> hvgs = selector.selected_features_
        >>>
        >>> # Check dispersion values
        >>> for gene in hvgs[:10]:
        ...     print(f"{gene}: CV = {selector.dispersions_[gene]:.3f}")
    """

    VALID_METRICS = ["cv", "variance", "std"]

    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        metric: Literal["cv", "variance", "std"] = "cv",
        min_mean: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """Initialize HDG selector.

        Args:
            n_features_to_select: Number of genes to select. If None, selects all.
            metric: Dispersion metric to use:
                - "cv": Coefficient of variation (std/mean) - default, robust to scale
                - "variance": Variance (std^2) - sensitive to highly expressed genes
                - "std": Standard deviation - middle ground
            min_mean: Minimum mean expression. Genes with mean < min_mean are excluded.
                Useful for filtering low-expression genes prone to technical noise.
            verbose: Print progress messages.

        Raises:
            ValueError: If metric is invalid.
        """
        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=None,  # HDG is deterministic
            verbose=verbose,
        )

        if metric not in self.VALID_METRICS:
            raise ValueError(f"metric must be one of {self.VALID_METRICS}, got '{metric}'")

        self.metric = metric
        self.min_mean = min_mean

        # Attributes set during fit
        self.dispersions_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HDGSelector":
        """Fit HDG selector by computing gene dispersions.

        Note: This method is unsupervised - y (cell labels) is not used,
        but required for API consistency with other selectors.

        Args:
            X: Expression matrix (cells × genes).
            y: Cell labels (not used by HDG, can be dummy values).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        if self.verbose:
            print(f"Computing gene dispersions using metric='{self.metric}'...")

        # Compute dispersion for each gene
        dispersions = {}

        for gene in X.columns:
            gene_expr = X[gene].values

            mean_expr = np.mean(gene_expr)

            # Skip genes below minimum mean expression
            if mean_expr < self.min_mean:
                dispersions[gene] = -np.inf  # Mark as excluded
                continue

            # Compute dispersion based on metric
            if self.metric == "cv":
                # Coefficient of variation: std / mean
                # Add small epsilon to avoid division by zero
                # Use ddof=1 for sample standard deviation (consistent with pandas)
                std_expr = np.std(gene_expr, ddof=1)
                cv = std_expr / (mean_expr + 1e-10)
                dispersions[gene] = cv

            elif self.metric == "variance":
                # Variance: std^2 (sample variance)
                variance = np.var(gene_expr, ddof=1)
                dispersions[gene] = variance

            elif self.metric == "std":
                # Standard deviation (sample std)
                std_expr = np.std(gene_expr, ddof=1)
                dispersions[gene] = std_expr

        self.dispersions_ = dispersions

        # Filter out genes that didn't pass min_mean threshold
        valid_genes = [(g, d) for g, d in dispersions.items() if d > -np.inf]

        # Sort genes by dispersion (descending)
        sorted_genes = sorted(valid_genes, key=lambda x: x[1], reverse=True)

        # Select top-k genes
        if self.n_features_to_select is not None:
            n_select = min(self.n_features_to_select, len(sorted_genes))
            sorted_genes = sorted_genes[:n_select]
        else:
            # Select all valid genes with non-zero dispersion
            sorted_genes = [(g, d) for g, d in sorted_genes if d > 0]

        selected_features = [gene for gene, _ in sorted_genes]
        feature_scores = np.array([disp for _, disp in sorted_genes])

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
                f"Selected {len(self.selected_features_)} highly variable genes "
                f"(metric={self.metric})"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting highly variable genes.

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
        """Get feature selection result with HDG metadata.

        Returns:
            FeatureSelectorResult with dispersion values in metadata.
        """
        result = super().get_result()

        # Initialize metadata
        if result.metadata is None:
            result.metadata = {}

        # Add HDG-specific metadata
        result.metadata["metric"] = self.metric
        result.metadata["min_mean"] = self.min_mean
        result.metadata["dispersions"] = self.dispersions_

        return result
