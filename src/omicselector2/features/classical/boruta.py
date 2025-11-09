"""Boruta all-relevant feature selector.

This module implements the Boruta algorithm for identifying all features
that are statistically relevant to the prediction task.

Boruta is a wrapper method around Random Forest that compares features
against shadow features (randomly permuted copies) using statistical tests.

The algorithm:
1. Creates shadow features by shuffling original features
2. Trains Random Forest on combined dataset (original + shadow)
3. Compares each feature's importance to max shadow importance
4. Uses binomial test to decide if feature is significant
5. Iterates until convergence or max_iter reached

Examples:
    >>> from omicselector2.features.classical.boruta import BorutaSelector
    >>> selector = BorutaSelector(n_estimators=100, max_iter=100)
    >>> selector.fit(X_train, y_train)
    >>> selected_features = selector.get_feature_names_out()
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from omicselector2.features.base import BaseFeatureSelector


class BorutaSelector(BaseFeatureSelector):
    """Feature selector using Boruta algorithm.

    Boruta identifies all statistically relevant features by comparing them
    to shadow features (randomly permuted versions). Features that consistently
    perform better than the best shadow feature are selected.

    This is an "all-relevant" feature selection method, meaning it aims to find
    ALL features that have some predictive power, not just a minimal subset.

    Attributes:
        n_estimators: Number of trees in Random Forest.
        max_iter: Maximum number of Boruta iterations.
        alpha: Significance level for statistical test (default 0.05).
        task: Type of task - 'regression' or 'classification'.
        model_: Trained Random Forest model.

    Examples:
        >>> # Default usage
        >>> selector = BorutaSelector(n_estimators=100)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # More stringent selection
        >>> selector = BorutaSelector(alpha=0.01, max_iter=200)
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_iter: int = 100,
        alpha: float = 0.05,
        task: Literal["regression", "classification"] = "classification",
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """Initialize Boruta feature selector.

        Args:
            n_estimators: Number of trees in Random Forest.
            max_depth: Maximum tree depth. None means unlimited.
            max_iter: Maximum number of Boruta iterations.
            alpha: Significance level for binomial test (0-1).
            task: 'regression' or 'classification'.
            n_features_to_select: Max features to select. None = all confirmed.
            random_state: Random seed for reproducibility.
            verbose: Print progress messages.
            n_jobs: Number of parallel threads (-1 = all cores).

        Raises:
            ValueError: If alpha not in (0, 1).
            ValueError: If task not in ['regression', 'classification'].
        """
        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose,
        )

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        if task not in ["regression", "classification"]:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.alpha = alpha
        self.task = task
        self.n_jobs = n_jobs

        self.model_ = None
        self.hits_: Optional[NDArray[np.int32]] = None  # Track hits per feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaSelector":
        """Fit Boruta selector to data.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        # Initialize tracking arrays
        n_features = X.shape[1]
        self.hits_ = np.zeros(n_features, dtype=np.int32)
        confirmed = np.zeros(n_features, dtype=bool)
        rejected = np.zeros(n_features, dtype=bool)

        # Boruta iterations
        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"Boruta iteration {iteration + 1}/{self.max_iter}")

            # Create shadow features by shuffling
            X_shadow = self._create_shadow_features(X)

            # Combine original and shadow features
            X_combined = pd.concat([X, X_shadow], axis=1)

            # Train Random Forest
            model = self._train_model(X_combined, y)

            # Get feature importances
            importances = model.feature_importances_

            # Split importances: original vs shadow
            orig_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            # Find max shadow importance
            max_shadow = np.max(shadow_importances)

            # Compare each feature to max shadow
            for i in range(n_features):
                if confirmed[i] or rejected[i]:
                    continue  # Already decided

                if orig_importances[i] > max_shadow:
                    self.hits_[i] += 1
                # If less than or equal, no hit

            # Statistical test: binomial test
            # H0: feature importance = max shadow (p=0.5)
            # H1: feature importance > max shadow (p>0.5)
            n_trials = iteration + 1
            for i in range(n_features):
                if confirmed[i] or rejected[i]:
                    continue

                # Binomial test: probability of getting this many hits by chance
                p_value = binom.sf(self.hits_[i] - 1, n_trials, 0.5)

                if p_value < self.alpha:
                    # Feature significantly better than shadow
                    confirmed[i] = True
                    if self.verbose:
                        print(f"  Confirmed: {X.columns[i]}")

                elif p_value > 1 - self.alpha:
                    # Feature significantly worse than shadow
                    rejected[i] = True
                    if self.verbose:
                        print(f"  Rejected: {X.columns[i]}")

            # Check convergence
            if np.all(confirmed | rejected):
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

        # Select confirmed features
        self._select_features(X, confirmed)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with Boruta")

        return self

    def _create_shadow_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create shadow features by randomly shuffling original features."""
        X_shadow = X.copy()
        rng = np.random.RandomState(self.random_state)

        for col in X_shadow.columns:
            X_shadow[col] = rng.permutation(X_shadow[col].values)

        # Rename shadow columns
        X_shadow.columns = [f"{col}_shadow" for col in X_shadow.columns]

        return X_shadow

    def _train_model(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest on combined features."""
        if self.task == "classification":
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        model.fit(X, y)
        return model

    def _select_features(self, X: pd.DataFrame, confirmed: NDArray[np.bool_]) -> None:
        """Select features based on confirmation status."""
        confirmed_indices = np.where(confirmed)[0]

        if len(confirmed_indices) == 0:
            # No features confirmed - select top feature by hits
            if self.verbose:
                print("Warning: No features confirmed, selecting top feature by hits")
            confirmed_indices = np.array([np.argmax(self.hits_)])

        # Get scores (hits)
        scores = self.hits_[confirmed_indices].astype(np.float64)

        # Sort by score (descending)
        sorted_indices = np.argsort(-scores)
        confirmed_indices = confirmed_indices[sorted_indices]
        scores = scores[sorted_indices]

        # Apply n_features_to_select limit if specified
        if self.n_features_to_select is not None:
            n_select = min(self.n_features_to_select, len(confirmed_indices))
            confirmed_indices = confirmed_indices[:n_select]
            scores = scores[:n_select]

        # Store results
        self.selected_features_ = [X.columns[i] for i in confirmed_indices]
        self.feature_scores_ = scores

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.support_mask_[confirmed_indices] = True

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
