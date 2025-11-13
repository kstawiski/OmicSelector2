"""Variance Threshold feature selector.

This module implements feature selection by removing low-variance features.
Features with variance below a threshold are assumed to carry little information
and can be safely removed.

Variance threshold is a simple baseline filter method that:
- Requires no target variable (unsupervised)
- Is computationally very fast
- Removes constant and near-constant features
- Works well as preprocessing before other methods

Particularly useful for:
- Genomics data with many near-constant genes
- Removing uninformative features quickly
- Preprocessing high-dimensional data
- Quality control filtering

Examples:
    >>> from omicselector2.features.classical.variance_threshold import VarianceThresholdSelector
    >>> # Remove zero-variance features
    >>> selector = VarianceThresholdSelector(threshold=0.0)
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Remove features with variance < 1.0
    >>> selector = VarianceThresholdSelector(threshold=1.0)
    >>> X_filtered = selector.fit_transform(X, y)
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from omicselector2.features.base import BaseFeatureSelector


class VarianceThresholdSelector(BaseFeatureSelector):
    """Feature selector that removes low-variance features.

    This is a simple baseline feature selection method that removes features
    whose variance doesn't meet a threshold. By default (threshold=0.0),
    it removes all constant features.

    The variance is calculated as the unbiased variance (using ddof=1).

    Note: This is an unsupervised method - it doesn't use the target variable.
    However, for API consistency with other selectors, it still accepts y in fit().

    Attributes:
        threshold: Minimum variance required for a feature to be selected.
        variances_: Calculated variance for each feature.
        selected_features_: Names of selected features.

    Examples:
        >>> # Remove constant features
        >>> selector = VarianceThresholdSelector(threshold=0.0)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Remove features with low variance
        >>> selector = VarianceThresholdSelector(threshold=0.5)
        >>> selector.fit(X_train, y_train)
        >>> X_filtered = selector.transform(X_train)
        >>>
        >>> # Select top N features by variance
        >>> selector = VarianceThresholdSelector(threshold=0.0, n_features_to_select=100)
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        threshold: float = 0.0,
        task: Literal["regression", "classification"] = "classification",
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize Variance Threshold selector.

        Args:
            threshold: Minimum variance required (must be >= 0).
            task: Task type - included for API consistency but not used.
            n_features_to_select: Maximum number of features to select.
                If specified, selects top N features by variance.
            random_state: Not used, included for API consistency.
            verbose: Print progress messages.

        Raises:
            ValueError: If threshold is negative.
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose,
        )

        self.threshold = threshold
        self.task = task

        self.variances_: Optional[NDArray[np.float64]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VarianceThresholdSelector":
        """Fit variance threshold selector.

        Calculates variance for each feature and selects features
        with variance > threshold (strictly greater).

        Args:
            X: Feature matrix (samples × features).
            y: Target variable (not used, but included for API consistency).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        # Calculate variance for each feature (unbiased estimator, ddof=1)
        self.variances_ = X.var(axis=0, ddof=1).values

        if self.verbose:
            print(f"Calculated variances for {len(self.variances_)} features")
            print(f"Variance range: [{self.variances_.min():.4f}, {self.variances_.max():.4f}]")

        # Select features with variance > threshold
        self._select_features(X)

        if self.verbose:
            print(
                f"Selected {len(self.selected_features_)} features with variance > {self.threshold}"
            )

        return self

    def _select_features(self, X: pd.DataFrame) -> None:
        """Select features based on variance threshold."""
        # Get indices of features above threshold (strictly greater)
        above_threshold = self.variances_ > self.threshold
        selected_indices = np.where(above_threshold)[0]

        if len(selected_indices) == 0:
            # No features meet threshold - handle gracefully
            if self.verbose:
                print(f"Warning: No features have variance > {self.threshold}")
            self.selected_features_ = []
            self.feature_scores_ = np.array([])
            self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
            return

        # Get variances for selected features
        selected_variances = self.variances_[selected_indices]

        # Sort by variance (descending)
        sorted_idx = np.argsort(-selected_variances)
        selected_indices = selected_indices[sorted_idx]
        selected_variances = selected_variances[sorted_idx]

        # Apply n_features_to_select limit if specified
        if self.n_features_to_select is not None:
            n_select = min(self.n_features_to_select, len(selected_indices))
            selected_indices = selected_indices[:n_select]
            selected_variances = selected_variances[:n_select]

        # Store results
        self.selected_features_ = [X.columns[i] for i in selected_indices]
        self.feature_scores_ = selected_variances

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.support_mask_[selected_indices] = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting high-variance features.

        Args:
            X: Feature matrix to transform.

        Returns:
            DataFrame with only selected features.

        Raises:
            ValueError: If selector not fitted yet.
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if len(self.selected_features_) == 0:
            # Return empty DataFrame with same index
            return pd.DataFrame(index=X.index)

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
