"""Correlation Filter for removing redundant features.

Correlation Filter identifies and removes highly correlated (redundant) features
to reduce multicollinearity and improve model interpretability. This is especially
useful in genomics where many genes are correlated due to:
- Co-regulation in biological pathways
- Linkage disequilibrium
- Technical correlations

The filter computes pairwise correlations and iteratively removes features that
are highly correlated with others, keeping the feature that appears first in the
feature list (or based on other criteria).

Key benefits:
- Reduces dimensionality while preserving information
- Improves model interpretability
- Reduces multicollinearity issues
- Fast computation

Based on:
- Correlation-based feature selection (CFS)
- Common preprocessing in genomics pipelines

Examples:
    >>> from omicselector2.features.filters.correlation import CorrelationFilter
    >>>
    >>> # Remove features with correlation > 0.9
    >>> selector = CorrelationFilter(threshold=0.9, method="pearson")
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Check which features were removed
    >>> print(f"Removed {len(selector.removed_features_)} redundant features")
    >>> for feature in selector.removed_features_:
    ...     print(f"  {feature}")
"""

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from omicselector2.features.base import BaseFeatureSelector


class CorrelationFilter(BaseFeatureSelector):
    """Correlation-based filter for removing redundant features.

    Removes features that are highly correlated with other features to reduce
    redundancy and multicollinearity. This unsupervised filter method computes
    pairwise correlations and removes one feature from each highly correlated pair.

    Attributes:
        threshold: Correlation threshold (0-1). Features with correlation above this
            are considered redundant.
        method: Correlation method ("pearson", "spearman", "kendall").
        correlation_matrix_: Computed correlation matrix.
        removed_features_: List of features that were removed due to high correlation.

    Examples:
        >>> # Basic usage
        >>> selector = CorrelationFilter(threshold=0.9)
        >>> selector.fit(gene_expression, phenotype)
        >>>
        >>> # Get selected features
        >>> print(f"Selected {len(selector.selected_features_)} features")
        >>>
        >>> # See correlation matrix
        >>> corr_matrix = selector.correlation_matrix_
        >>>
        >>> # Spearman correlation (robust to outliers)
        >>> selector = CorrelationFilter(threshold=0.8, method="spearman")
        >>> X_filtered = selector.fit_transform(X, y)
    """

    VALID_METHODS = ["pearson", "spearman", "kendall"]

    def __init__(
        self,
        threshold: float = 0.9,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        verbose: bool = False,
    ) -> None:
        """Initialize Correlation Filter.

        Args:
            threshold: Correlation threshold (0-1, exclusive). Features with absolute
                correlation >= threshold are considered redundant. Default 0.9.
            method: Correlation method:
                - "pearson": Linear correlation (default, fast)
                - "spearman": Rank correlation (robust to outliers, non-linear)
                - "kendall": Rank correlation (robust, slower)
            verbose: Print progress messages.

        Raises:
            ValueError: If threshold not in (0, 1) or method is invalid.
        """
        super().__init__(
            n_features_to_select=None,  # Determined by correlation threshold
            random_state=None,  # Deterministic
            verbose=verbose,
        )

        if not (0 < threshold < 1):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")

        self.threshold = threshold
        self.method = method

        # Attributes set during fit
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.removed_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelationFilter":
        """Fit correlation filter by identifying redundant features.

        Note: This method is unsupervised - y (labels) is not used,
        but required for API consistency with other selectors.

        Args:
            X: Feature matrix (samples × features).
            y: Target labels (not used by correlation filter).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        if self.verbose:
            print(f"Computing {self.method} correlation matrix " f"(threshold={self.threshold})...")

        # Compute correlation matrix
        self.correlation_matrix_ = X.corr(method=self.method)

        # Find and remove highly correlated features
        features_to_remove = set()

        # Get absolute correlation values
        corr_matrix = self.correlation_matrix_.abs()

        # Iterate through upper triangle of correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                feature_i = corr_matrix.columns[i]
                feature_j = corr_matrix.columns[j]

                # Check if correlation exceeds threshold
                if corr_matrix.iloc[i, j] >= self.threshold:
                    # Remove feature_j (keep feature_i by convention)
                    # This is arbitrary but ensures deterministic behavior
                    if feature_j not in features_to_remove:
                        features_to_remove.add(feature_j)
                        if self.verbose:
                            print(
                                f"  Removing {feature_j} (corr={corr_matrix.iloc[i, j]:.3f} "
                                f"with {feature_i})"
                            )

        self.removed_features_ = sorted(list(features_to_remove))

        # Select features that were not removed
        selected_features = [f for f in X.columns if f not in features_to_remove]

        # Set attributes
        self.selected_features_ = selected_features
        # Use 1 - max_correlation as score (lower correlation = higher score)
        feature_scores = []
        for feature in selected_features:
            # Get max correlation with other selected features
            feature_idx = X.columns.get_loc(feature)
            correlations_with_selected = [
                corr_matrix.iloc[feature_idx, X.columns.get_loc(other)]
                for other in selected_features
                if other != feature
            ]
            if correlations_with_selected:
                max_corr = max(correlations_with_selected)
                score = 1.0 - max_corr  # Higher score = less correlated
            else:
                score = 1.0  # Only feature
            feature_scores.append(score)

        self.feature_scores_ = np.array(feature_scores)

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                self.support_mask_[i] = True

        if self.verbose:
            print(
                f"Selected {len(self.selected_features_)} features, "
                f"removed {len(self.removed_features_)} redundant features"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by removing redundant features.

        Args:
            X: Feature matrix to transform.

        Returns:
            DataFrame with only non-redundant features.

        Raises:
            ValueError: If selector not fitted yet.
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> NDArray:
        """Get mask or indices of selected features.

        Args:
            indices: If True, return integer indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected features.

        Raises:
            ValueError: If selector not fitted yet.
        """
        if self.support_mask_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

    def get_result(self) -> Any:
        """Get feature selection result with correlation metadata.

        Returns:
            FeatureSelectorResult with removed features in metadata.
        """
        result = super().get_result()

        # Initialize metadata
        if result.metadata is None:
            result.metadata = {}

        # Add correlation filter-specific metadata
        result.metadata["method"] = self.method
        result.metadata["threshold"] = self.threshold
        result.metadata["removed_features"] = self.removed_features_
        result.metadata["n_removed"] = len(self.removed_features_)

        return result
