"""mRMR (Minimum Redundancy Maximum Relevance) feature selector.

This module implements mRMR feature selection using mutual information.
mRMR selects features by maximizing relevance to the target while minimizing
redundancy with already selected features.

The algorithm uses greedy forward selection:
1. Calculate mutual information between each feature and target (relevance)
2. Select feature with highest relevance
3. For each remaining feature, calculate mRMR score:
   score = MI(feature, target) - mean(MI(feature, selected_features))
4. Select feature with highest mRMR score
5. Repeat until desired number of features selected

mRMR is particularly effective for:
- High-dimensional data (genomics, transcriptomics)
- Capturing non-linear relationships
- Selecting diverse, complementary features

Examples:
    >>> from omicselector2.features.classical.mrmr import mRMRSelector
    >>> selector = mRMRSelector(n_features_to_select=20)
    >>> selector.fit(X_train, y_train)
    >>> selected_features = selector.get_feature_names_out()
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from omicselector2.features.base import BaseFeatureSelector


class mRMRSelector(BaseFeatureSelector):
    """Feature selector using mRMR (Minimum Redundancy Maximum Relevance).

    mRMR is an information-theoretic filter method that balances:
    - Relevance: High mutual information with target
    - Redundancy: Low mutual information with already selected features

    Uses greedy forward selection to iteratively add features that maximize
    the mRMR criterion: max[Relevance - Redundancy].

    Attributes:
        n_features_to_select: Number of features to select (required).
        task: Type of task - 'regression' or 'classification'.
        relevance_: Array of MI(feature, target) for all features.
        redundancy_: Matrix of MI(feature_i, feature_j) for selected features.

    Examples:
        >>> # Select 20 features for classification
        >>> selector = mRMRSelector(n_features_to_select=20)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Regression task
        >>> selector = mRMRSelector(n_features_to_select=30, task='regression')
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        n_features_to_select: int = 10,
        task: Literal["regression", "classification"] = "classification",
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """Initialize mRMR feature selector.

        Args:
            n_features_to_select: Number of features to select (must be positive).
            task: 'regression' or 'classification'.
            random_state: Random seed for MI estimation.
            verbose: Print progress messages.
            n_jobs: Number of parallel threads for MI calculation.

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
        self.n_jobs = n_jobs

        self.relevance_: Optional[NDArray[np.float64]] = None
        self.redundancy_: Optional[NDArray[np.float64]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "mRMRSelector":
        """Fit mRMR selector to data.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        n_features = X.shape[1]

        # Limit n_features_to_select if it exceeds total features
        n_to_select = min(self.n_features_to_select, n_features)

        # Calculate relevance (MI with target) for all features
        if self.verbose:
            print("Calculating feature relevance (MI with target)...")

        if self.task == "classification":
            self.relevance_ = mutual_info_classif(
                X,
                y,
                discrete_features=False,
                n_neighbors=3,
                random_state=self.random_state,
            )
        else:
            self.relevance_ = mutual_info_regression(
                X,
                y,
                discrete_features=False,
                n_neighbors=3,
                random_state=self.random_state,
            )

        # Greedy forward selection
        selected_indices = []
        remaining_indices = list(range(n_features))

        # Select first feature: highest relevance
        first_idx = int(np.argmax(self.relevance_))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        if self.verbose:
            print(f"Selected feature 1/{n_to_select}: {X.columns[first_idx]}")

        # Iteratively select remaining features
        for k in range(1, n_to_select):
            if not remaining_indices:
                break

            # Calculate mRMR scores for remaining features
            mrmr_scores = self._calculate_mrmr_scores(
                X, selected_indices, remaining_indices
            )

            # Select feature with highest mRMR score
            best_idx_in_remaining = int(np.argmax(mrmr_scores))
            best_feature_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_feature_idx)
            remaining_indices.remove(best_feature_idx)

            if self.verbose:
                print(f"Selected feature {k + 1}/{n_to_select}: {X.columns[best_feature_idx]}")

        # Store results
        self._store_results(X, selected_indices)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with mRMR")

        return self

    def _calculate_mrmr_scores(
        self,
        X: pd.DataFrame,
        selected_indices: list[int],
        remaining_indices: list[int],
    ) -> NDArray[np.float64]:
        """Calculate mRMR scores for remaining features.

        mRMR score = Relevance - Redundancy
        where Redundancy = mean(MI(feature, selected_features))

        Args:
            X: Feature matrix.
            selected_indices: Indices of already selected features.
            remaining_indices: Indices of remaining candidate features.

        Returns:
            Array of mRMR scores for remaining features.
        """
        mrmr_scores = []

        # Get selected features data
        X_selected = X.iloc[:, selected_indices].values

        for idx in remaining_indices:
            # Relevance (already calculated)
            relevance = self.relevance_[idx]

            # Calculate redundancy with already selected features
            X_feature = X.iloc[:, idx].values.reshape(-1, 1)
            redundancy_scores = []

            for selected_idx in selected_indices:
                X_selected_feature = X.iloc[:, selected_idx].values

                # Compute MI between current feature and selected feature
                mi = self._compute_mutual_information(X_feature, X_selected_feature)
                redundancy_scores.append(mi)

            # Average redundancy
            redundancy = np.mean(redundancy_scores)

            # mRMR criterion: maximize relevance - redundancy
            mrmr_score = relevance - redundancy
            mrmr_scores.append(mrmr_score)

        return np.array(mrmr_scores)

    def _compute_mutual_information(
        self, X: NDArray, y: NDArray
    ) -> float:
        """Compute mutual information between two variables.

        Args:
            X: First variable (n_samples, 1).
            y: Second variable (n_samples,).

        Returns:
            Mutual information value.
        """
        # For continuous features, use MI regression
        mi = mutual_info_regression(
            X,
            y,
            discrete_features=False,
            n_neighbors=3,
            random_state=self.random_state,
        )
        return float(mi[0])

    def _store_results(self, X: pd.DataFrame, selected_indices: list[int]) -> None:
        """Store selected features and their scores."""
        # Store feature names
        self.selected_features_ = [X.columns[i] for i in selected_indices]

        # Store relevance scores for selected features
        self.feature_scores_ = self.relevance_[selected_indices]

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.support_mask_[selected_indices] = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features.

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
