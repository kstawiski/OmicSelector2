"""XGBoost feature selector.

This module implements feature selection using XGBoost's gradient boosting,
which provides feature importance scores through gain, weight, or cover metrics.

XGBoost is particularly effective for:
- High-dimensional data with complex interactions
- Non-linear relationships
- Mixed feature types
- Handling missing values

Examples:
    >>> from omicselector2.features.classical.xgboost import XGBoostSelector
    >>> selector = XGBoostSelector(n_estimators=100, importance_type='gain')
    >>> selector.fit(X_train, y_train)
    >>> selected_features = selector.get_feature_names_out()
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray

from omicselector2.features.base import BaseFeatureSelector


class XGBoostSelector(BaseFeatureSelector):
    """Feature selector using XGBoost gradient boosting.

    XGBoost provides feature importance through three metrics:
    - 'gain': Average gain when feature is used in splits (default)
    - 'weight': Number of times feature appears in trees
    - 'cover': Average coverage (samples affected) when feature is used

    For genomics data, 'gain' is typically most informative as it measures
    the actual contribution to model performance.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth (None = no limit).
        learning_rate: Step size shrinkage for updates.
        task: Type of task - 'regression' or 'classification'.
        importance_type: Metric for feature importance ('gain', 'weight', 'cover').
        model_: Trained XGBoost model.

    Examples:
        >>> # Default: gain-based importance
        >>> selector = XGBoostSelector(n_estimators=100)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Weight-based (frequency) importance
        >>> selector = XGBoostSelector(importance_type='weight', n_features_to_select=50)
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        learning_rate: float = 0.1,
        task: Literal["regression", "classification"] = "classification",
        importance_type: Literal["gain", "weight", "cover"] = "gain",
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """Initialize XGBoost feature selector.

        Args:
            n_estimators: Number of boosting rounds (trees).
            max_depth: Maximum tree depth. None means unlimited.
            learning_rate: Learning rate (eta) for gradient boosting.
            task: 'regression' or 'classification'.
            importance_type: Feature importance metric - 'gain', 'weight', or 'cover'.
            n_features_to_select: Max features to select. None = all non-zero.
            random_state: Random seed for reproducibility.
            verbose: Print progress messages.
            n_jobs: Number of parallel threads (-1 = all cores).

        Raises:
            ValueError: If importance_type not in ['gain', 'weight', 'cover'].
            ValueError: If task not in ['regression', 'classification'].
        """
        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose,
        )

        if importance_type not in ["gain", "weight", "cover"]:
            raise ValueError(
                f"importance_type must be one of ['gain', 'weight', 'cover'], got {importance_type}"
            )

        if task not in ["regression", "classification"]:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.task = task
        self.importance_type = importance_type
        self.n_jobs = n_jobs

        self.model_: Optional[Union[xgb.XGBClassifier, xgb.XGBRegressor]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostSelector":
        """Fit XGBoost selector to data.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        # Build XGBoost model
        if self.task == "classification":
            self.model_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=1 if self.verbose else 0,
            )
        else:
            self.model_ = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=1 if self.verbose else 0,
            )

        # Fit model
        self.model_.fit(X, y)

        # Extract feature importances
        self._select_features(X)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with XGBoost")

        return self

    def _select_features(self, X: pd.DataFrame) -> None:
        """Select features based on XGBoost importance scores."""
        # Get feature importance scores from XGBoost
        # get_score() returns dict of {feature_name: score}
        importance_dict = self.model_.get_booster().get_score(importance_type=self.importance_type)

        # Map feature names back to indices
        feature_names = X.columns.tolist()
        n_features = len(feature_names)

        # Initialize scores array (features not used have 0 importance)
        scores = np.zeros(n_features, dtype=np.float64)

        # Fill in scores for features that were used
        # XGBoost returns actual feature names when DataFrame is used
        for feature_name, score in importance_dict.items():
            if feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
                scores[feature_idx] = score

        # Create support mask for non-zero importance
        self.support_mask_ = scores > 0

        # Get indices of features with non-zero importance
        selected_indices = np.where(self.support_mask_)[0]
        self.feature_scores_ = scores[selected_indices]

        # Sort by importance (descending)
        sorted_indices = np.argsort(-self.feature_scores_)
        selected_indices = selected_indices[sorted_indices]
        self.feature_scores_ = self.feature_scores_[sorted_indices]

        # Apply n_features_to_select limit if specified
        if self.n_features_to_select is not None:
            n_select = min(self.n_features_to_select, len(selected_indices))
            selected_indices = selected_indices[:n_select]
            self.feature_scores_ = self.feature_scores_[:n_select]

            # Update support mask
            new_mask = np.zeros(n_features, dtype=bool)
            new_mask[selected_indices] = True
            self.support_mask_ = new_mask

        # Store selected feature names
        self.selected_features_ = [feature_names[i] for i in selected_indices]

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
