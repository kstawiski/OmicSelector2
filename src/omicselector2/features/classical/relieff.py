"""ReliefF feature selector.

This module implements the ReliefF algorithm for instance-based feature selection.
ReliefF evaluates features by how well they distinguish between instances that
are near each other, with different class labels (hits) vs same class (misses).

ReliefF algorithm:
1. For each sampled instance, find k nearest neighbors from same class (hits)
2. Find k nearest neighbors from each different class (misses)
3. Update feature weights based on whether feature values separate hits and misses
4. Features that separate instances from different classes get higher weights

Key characteristics:
- Instance-based: Uses actual data instances, not just statistics
- Handles feature interactions: Can detect non-linear relationships
- Works with noisy data: Robust to outliers
- No assumptions: Non-parametric method
- Scales well: Uses k-NN for efficiency

Particularly useful for:
- Genomics data with complex gene interactions
- High-dimensional noisy data
- Detecting non-linear feature effects
- When feature interactions are important

Examples:
    >>> from omicselector2.features.classical.relieff import ReliefFSelector
    >>> # Classification with 10 nearest neighbors
    >>> selector = ReliefFSelector(n_neighbors=10, n_features_to_select=50)
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Regression task
    >>> selector = ReliefFSelector(task='regression', n_neighbors=15, n_features_to_select=100)
    >>> X_filtered = selector.fit_transform(X, y)
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from omicselector2.features.base import BaseFeatureSelector


class ReliefFSelector(BaseFeatureSelector):
    """Feature selector using ReliefF algorithm.

    ReliefF is an instance-based feature selection method that evaluates
    features based on how well they distinguish between instances that are
    near each other. It updates feature weights by comparing each instance
    to its nearest neighbors from the same class (hits) and different classes
    (misses).

    Attributes:
        n_features_to_select: Number of features to select.
        n_neighbors: Number of nearest neighbors to consider (k in ReliefF).
        task: Type of task - 'regression' or 'classification'.
        feature_scores_: ReliefF weights for selected features (higher = better).

    Examples:
        >>> # Binary classification
        >>> selector = ReliefFSelector(n_neighbors=10, n_features_to_select=50)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Multi-class classification
        >>> selector = ReliefFSelector(n_neighbors=15, n_features_to_select=100)
        >>> result = selector.fit(X, y).get_result()
        >>>
        >>> # Regression
        >>> selector = ReliefFSelector(task='regression', n_neighbors=10, n_features_to_select=50)
        >>> X_filtered = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: int = 10,
        n_neighbors: int = 10,
        task: Literal["regression", "classification"] = "classification",
        n_iterations: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize ReliefF selector.

        Args:
            n_features_to_select: Number of features to select (must be positive).
            n_neighbors: Number of nearest neighbors to consider (k in ReliefF).
            task: 'regression' or 'classification'.
            n_iterations: Number of iterations (samples to process). None = all samples.
            random_state: Random seed for reproducibility.
            verbose: Print progress messages.

        Raises:
            ValueError: If n_features_to_select <= 0.
            ValueError: If n_neighbors <= 0.
            ValueError: If task not in ['regression', 'classification'].
        """
        if n_features_to_select <= 0:
            raise ValueError(f"n_features_to_select must be positive, got {n_features_to_select}")

        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")

        if task not in ["regression", "classification"]:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")

        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose,
        )

        self.n_neighbors = n_neighbors
        self.task = task
        self.n_iterations = n_iterations

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReliefFSelector":
        """Fit ReliefF selector to data.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        X_array = X.values
        y_array = y.values

        # Adjust n_neighbors if needed for small datasets
        n_samples = X_array.shape[0]
        k = min(self.n_neighbors, n_samples - 1)

        if self.verbose and k < self.n_neighbors:
            print(f"Adjusted n_neighbors from {self.n_neighbors} to {k} (n_samples={n_samples})")

        # Determine number of iterations
        if self.n_iterations is None:
            n_iter = n_samples
        else:
            n_iter = min(self.n_iterations, n_samples)

        if self.verbose:
            print(f"Running ReliefF with {n_iter} iterations, k={k} neighbors...")

        # Run ReliefF algorithm
        if self.task == "classification":
            weights = self._relieff_classification(X_array, y_array, k, n_iter)
        else:
            weights = self._relieff_regression(X_array, y_array, k, n_iter)

        # Select top features by weight
        self._select_features(X, weights)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with ReliefF")

        return self

    def _relieff_classification(
        self,
        X: NDArray,
        y: NDArray,
        k: int,
        n_iter: int,
    ) -> NDArray[np.float64]:
        """Run ReliefF for classification."""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features, dtype=np.float64)

        # Build k-NN model
        knn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        knn.fit(X)

        # Get unique classes
        classes = np.unique(y)
        class_probs = {c: np.sum(y == c) / n_samples for c in classes}

        # Random sample of instances
        rng = np.random.RandomState(self.random_state)
        sample_indices = rng.choice(n_samples, n_iter, replace=False)

        for idx in sample_indices:
            instance = X[idx:idx+1]
            instance_class = y[idx]

            # Find k nearest neighbors
            _, indices = knn.kneighbors(instance)
            indices = indices[0][1:]  # Exclude instance itself

            # Separate hits (same class) and misses (different class)
            neighbor_classes = y[indices]
            hits_mask = neighbor_classes == instance_class

            # Update weights for hits (same class)
            if np.any(hits_mask):
                hit_indices = indices[hits_mask]
                for neighbor_idx in hit_indices:
                    diff = np.abs(X[idx] - X[neighbor_idx])
                    weights -= diff / (n_iter * k)

            # Update weights for misses (different classes)
            for c in classes:
                if c == instance_class:
                    continue

                class_mask = neighbor_classes == c
                if np.any(class_mask):
                    miss_indices = indices[class_mask]
                    prior_prob = class_probs[c] / (1 - class_probs[instance_class])

                    for neighbor_idx in miss_indices:
                        diff = np.abs(X[idx] - X[neighbor_idx])
                        weights += prior_prob * diff / (n_iter * k)

        return weights

    def _relieff_regression(
        self,
        X: NDArray,
        y: NDArray,
        k: int,
        n_iter: int,
    ) -> NDArray[np.float64]:
        """Run ReliefF for regression (RReliefF variant)."""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features, dtype=np.float64)

        # Build k-NN model
        knn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        knn.fit(X)

        # Random sample of instances
        rng = np.random.RandomState(self.random_state)
        sample_indices = rng.choice(n_samples, n_iter, replace=False)

        # Compute outcome range for normalization
        y_range = np.max(y) - np.min(y)
        if y_range == 0:
            y_range = 1.0

        for idx in sample_indices:
            instance = X[idx:idx+1]
            instance_y = y[idx]

            # Find k nearest neighbors
            distances, indices = knn.kneighbors(instance)
            indices = indices[0][1:]  # Exclude instance itself

            for neighbor_idx in indices:
                # Compute similarity in outcome space
                y_diff = np.abs(instance_y - y[neighbor_idx]) / y_range

                # Update weight: penalize if feature differs but outcomes similar
                # Reward if feature differs and outcomes differ
                feature_diff = np.abs(X[idx] - X[neighbor_idx])
                weights -= feature_diff * (1 - y_diff) / (n_iter * k)
                weights += feature_diff * y_diff / (n_iter * k)

        return weights

    def _select_features(self, X: pd.DataFrame, weights: NDArray[np.float64]) -> None:
        """Select top features by weight."""
        n_features = X.shape[1]
        n_to_select = min(self.n_features_to_select, n_features)

        # Select top features by weight
        top_indices = np.argsort(-weights)[:n_to_select]

        # Sort selected features by weight (descending)
        scores = weights[top_indices]
        sorted_order = np.argsort(-scores)
        top_indices = top_indices[sorted_order]
        scores = scores[sorted_order]

        # Store results
        self.selected_features_ = [X.columns[i] for i in top_indices]
        self.feature_scores_ = scores

        # Create support mask
        self.support_mask_ = np.zeros(n_features, dtype=bool)
        self.support_mask_[top_indices] = True

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
