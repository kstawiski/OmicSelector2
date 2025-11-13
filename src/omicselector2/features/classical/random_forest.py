"""Random Forest Variable Importance feature selector.

This module implements feature selection using Random Forest's built-in feature importance,
which measures how much each feature decreases impurity across all trees.

Random Forest VI is robust, handles non-linear relationships, and is widely used in genomics.

Examples:
    >>> from omicselector2.features.classical.random_forest import RandomForestSelector
    >>> selector = RandomForestSelector(n_estimators=100, n_features_to_select=50)
    >>> selector.fit(X_train, y_train)
    >>> selected_features = selector.get_feature_names_out()
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from omicselector2.features.base import BaseFeatureSelector


class RandomForestSelector(BaseFeatureSelector):
    """Feature selector using Random Forest Variable Importance.

    Uses the mean decrease in impurity (Gini importance) across all trees
    to rank features. Features with higher importance contribute more to
    reducing uncertainty in predictions.

    Advantages:
    - Handles non-linear relationships
    - Robust to outliers
    - No feature scaling needed
    - Captures feature interactions
    - Interpretable importance scores

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees. None = no limit.
        min_samples_split: Minimum samples required to split a node.
        task: 'regression' or 'classification'.
        importance_type: Type of importance ('gini' or 'permutation').
        n_features_to_select: Number of top features to select.
        random_state: Random seed for reproducibility.

    Examples:
        >>> # Classification task
        >>> selector = RandomForestSelector(
        ...     n_estimators=100,
        ...     task='classification',
        ...     n_features_to_select=50
        ... )
        >>> selector.fit(X_train, y_train)
        >>> result = selector.get_result()
        >>> print(result.to_dataframe().head())
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        task: Literal["regression", "classification"] = "classification",
        importance_type: Literal["gini", "permutation"] = "gini",
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """Initialize Random Forest feature selector.

        Args:
            n_estimators: Number of trees in the forest (100-500 recommended).
            max_depth: Max tree depth. None = grow until pure leaves.
            min_samples_split: Min samples to split internal node.
            min_samples_leaf: Min samples at leaf node.
            task: 'regression' or 'classification'.
            importance_type: 'gini' (fast) or 'permutation' (more accurate).
            n_features_to_select: Number of top features to select.
            random_state: Random seed for reproducibility.
            verbose: Print progress information.
            n_jobs: Number of parallel jobs (-1 = all cores).

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(
            n_features_to_select=n_features_to_select, random_state=random_state, verbose=verbose
        )

        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}")

        if task not in ["regression", "classification"]:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")

        if importance_type not in ["gini", "permutation"]:
            raise ValueError(
                f"importance_type must be 'gini' or 'permutation', got {importance_type}"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.task = task
        self.importance_type = importance_type
        self.n_jobs = n_jobs

        self.model_: Optional[RandomForestClassifier | RandomForestRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestSelector":
        """Fit Random Forest and compute feature importance.

        Args:
            X: Feature matrix.
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        # Create and fit model
        if self.task == "classification":
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=1 if self.verbose else 0,
            )
        else:
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=1 if self.verbose else 0,
            )

        self.model_.fit(X, y)

        # Get feature importance
        if self.importance_type == "gini":
            importances = self.model_.feature_importances_
        else:
            # Permutation importance (more accurate but slower)
            from sklearn.inspection import permutation_importance

            perm_result = permutation_importance(
                self.model_, X, y, n_repeats=10, random_state=self.random_state, n_jobs=self.n_jobs
            )
            importances = perm_result.importances_mean

        # Rank features by importance
        sorted_indices = np.argsort(-importances)
        all_scores = importances[sorted_indices]
        all_features = [X.columns[i] for i in sorted_indices]

        # Select top features
        if self.n_features_to_select is not None:
            n_select = min(self.n_features_to_select, len(all_features))
        else:
            # Select features with non-zero importance
            n_select = np.sum(all_scores > 0)

        selected_indices = sorted_indices[:n_select]
        self.feature_scores_ = all_scores[:n_select]
        self.selected_features_ = all_features[:n_select]

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.support_mask_[selected_indices] = True

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with Random Forest")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features."""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> NDArray:
        """Get mask or indices of selected features."""
        if self.support_mask_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

    def get_feature_importance(self) -> NDArray[np.float64]:
        """Get importance scores for all features.

        Returns:
            Array of importance scores for all input features.

        Raises:
            ValueError: If selector has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if self.importance_type == "gini":
            return self.model_.feature_importances_
        else:
            raise ValueError("Full importance only available for gini type")

    def __repr__(self) -> str:
        """Get string representation."""
        params = [f"n_estimators={self.n_estimators}", f"task='{self.task}'"]
        if self.n_features_to_select is not None:
            params.append(f"n_features_to_select={self.n_features_to_select}")
        return f"RandomForestSelector({', '.join(params)})"
