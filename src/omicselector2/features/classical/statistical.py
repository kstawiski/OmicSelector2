"""Statistical feature selector (t-test / ANOVA / F-test).

This module implements univariate statistical feature selection using:
- t-test for binary classification (via ANOVA F-statistic)
- ANOVA F-test for multi-class classification
- F-test for regression

These are classical statistical filter methods that measure the strength
of the univariate relationship between each feature and the target.

The methods compute:
- F-statistic: Ratio of between-group to within-group variance
- p-value: Probability of observing the data under null hypothesis

Features are ranked by F-statistic (higher is better) and selected
based on top-k ranking.

Particularly useful for:
- Pre-filtering in high-dimensional data (genomics, transcriptomics)
- Identifying features with strong univariate associations
- Fast baseline feature selection
- Statistical hypothesis testing framework

Examples:
    >>> from omicselector2.features.classical.statistical import StatisticalSelector
    >>> # Binary or multi-class classification
    >>> selector = StatisticalSelector(n_features_to_select=100, task='classification')
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Regression
    >>> selector = StatisticalSelector(n_features_to_select=50, task='regression')
    >>> X_filtered = selector.fit_transform(X, y)
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_selection import f_classif, f_regression

from omicselector2.features.base import BaseFeatureSelector


class StatisticalSelector(BaseFeatureSelector):
    """Feature selector using univariate statistical tests.

    Uses ANOVA F-test for classification and F-test for regression to
    select features with the strongest univariate relationships to the target.

    For classification:
    - Computes F-statistic from ANOVA (between-group vs within-group variance)
    - Works for both binary (equivalent to t-test) and multi-class

    For regression:
    - Computes F-statistic from univariate linear regression
    - Measures strength of linear relationship

    Attributes:
        n_features_to_select: Number of features to select.
        task: Type of task - 'regression' or 'classification'.
        scores_: F-statistics for selected features (higher = more significant).
        pvalues_: P-values for selected features.

    Examples:
        >>> # Classification: select top 100 features
        >>> selector = StatisticalSelector(n_features_to_select=100)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Regression: select features with p < 0.05
        >>> selector = StatisticalSelector(n_features_to_select=50, task='regression')
        >>> result = selector.fit(X, y).get_result()
        >>> significant = [f for f, p in zip(result.selected_features, result.metadata['pvalues']) if p < 0.05]
    """

    def __init__(
        self,
        n_features_to_select: int = 10,
        task: Literal["regression", "classification"] = "classification",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize Statistical selector.

        Args:
            n_features_to_select: Number of features to select (must be positive).
            task: 'regression' or 'classification'.
            random_state: Not used, included for API consistency.
            verbose: Print progress messages.

        Raises:
            ValueError: If n_features_to_select <= 0.
            ValueError: If task not in ['regression', 'classification'].
        """
        if n_features_to_select <= 0:
            raise ValueError(f"n_features_to_select must be positive, got {n_features_to_select}")

        if task not in ["regression", "classification"]:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")

        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose,
        )

        self.task = task

        self.scores_: Optional[NDArray[np.float64]] = None
        self.pvalues_: Optional[NDArray[np.float64]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatisticalSelector":
        """Fit statistical selector to data.

        Computes F-statistics and p-values for all features, then selects
        top-k features by F-statistic.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        # Compute F-statistics and p-values
        if self.verbose:
            test_name = "ANOVA F-test" if self.task == "classification" else "F-test"
            print(f"Computing {test_name} for {X.shape[1]} features...")

        if self.task == "classification":
            f_scores, p_values = f_classif(X, y)
        else:
            f_scores, p_values = f_regression(X, y)

        # Handle NaN values (can occur with constant features or perfect separation)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)

        # Select top features
        self._select_features(X, f_scores, p_values)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features")
            if len(self.pvalues_) > 0:
                print(f"P-value range: [{self.pvalues_.min():.2e}, {self.pvalues_.max():.2e}]")

        return self

    def _select_features(
        self,
        X: pd.DataFrame,
        f_scores: NDArray[np.float64],
        p_values: NDArray[np.float64],
    ) -> None:
        """Select top features by F-statistic."""
        n_features = X.shape[1]
        n_to_select = min(self.n_features_to_select, n_features)

        # Get indices of top features by F-score (descending)
        top_indices = np.argsort(-f_scores)[:n_to_select]

        # Sort selected features by score (descending)
        sorted_order = np.argsort(-f_scores[top_indices])
        top_indices = top_indices[sorted_order]

        # Store results
        self.selected_features_ = [X.columns[i] for i in top_indices]
        self.scores_ = f_scores[top_indices]
        self.pvalues_ = p_values[top_indices]
        self.feature_scores_ = self.scores_  # For BaseFeatureSelector compatibility

        # Create support mask
        self.support_mask_ = np.zeros(n_features, dtype=bool)
        self.support_mask_[top_indices] = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting statistically significant features.

        Args:
            X: Feature matrix to transform.

        Returns:
            DataFrame with only selected features.

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

    def get_result(self):
        """Get feature selection result with p-values in metadata.

        Returns:
            FeatureSelectorResult with p-values included in metadata.
        """
        result = super().get_result()

        # Add p-values to metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata["pvalues"] = self.pvalues_.tolist()

        return result
