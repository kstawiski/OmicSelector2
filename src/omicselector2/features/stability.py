"""Stability Selection Framework.

This module implements stability selection, a method for improving the robustness
of feature selection by using bootstrap resampling. Features that are consistently
selected across many bootstrap samples are considered "stable" and more likely to
be truly relevant.

Stability selection wraps any base feature selector and runs it on multiple
bootstrap samples of the data. Features selected frequently (above a threshold)
are considered stable and are selected in the final model.

Key benefits:
- Reduces selection bias and overfitting
- Provides confidence scores (stability scores) for each feature
- Works with any feature selection method
- Improves reproducibility across datasets

Based on:
- Meinshausen & Bühlmann (2010) "Stability selection"
- Pusa & Rousu (2024) "Stable biomarker discovery in multi-omics data"

Examples:
    >>> from omicselector2.features.stability import StabilitySelector
    >>> from omicselector2.features.classical.lasso import LassoSelector
    >>>
    >>> # Wrap Lasso with stability selection
    >>> base_selector = LassoSelector(n_features_to_select=50)
    >>> selector = StabilitySelector(
    ...     base_selector=base_selector,
    ...     n_bootstraps=100,
    ...     threshold=0.7
    ... )
    >>> selector.fit(X_train, y_train)
    >>> X_filtered = selector.transform(X_train)
"""

import copy
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone as sklearn_clone

from omicselector2.features.base import BaseFeatureSelector


class StabilitySelector(BaseFeatureSelector):
    """Stability selection wrapper for feature selectors.

    Wraps any feature selector and runs it on multiple bootstrap samples
    to identify features that are consistently selected. Only features
    selected in at least `threshold` fraction of bootstrap samples are
    retained.

    This approach improves robustness by reducing the impact of:
    - Outliers and noise in the data
    - Random initialization in some methods
    - Sampling variability

    Attributes:
        base_selector: The underlying feature selector to wrap.
        n_bootstraps: Number of bootstrap samples (default 100).
        threshold: Minimum selection frequency (0-1) for a feature to be considered stable.
        sample_fraction: Fraction of samples to use in each bootstrap (default 0.8).
        selected_features_: List of stable feature names.
        stability_scores_: Dict mapping feature names to stability scores (selection frequency).
        selection_counts_: Dict mapping feature names to number of times selected.

    Examples:
        >>> # Basic usage with Lasso
        >>> from omicselector2.features.classical.lasso import LassoSelector
        >>> base = LassoSelector(n_features_to_select=20)
        >>> selector = StabilitySelector(base_selector=base, n_bootstraps=50, threshold=0.6)
        >>> selector.fit(X, y)
        >>>
        >>> # Access stability scores
        >>> for feature, score in selector.stability_scores_.items():
        ...     if score > 0.7:
        ...         print(f"{feature}: {score:.2f}")
        >>>
        >>> # Use with Random Forest
        >>> from omicselector2.features.classical.random_forest import RandomForestSelector
        >>> base = RandomForestSelector(n_features_to_select=30)
        >>> selector = StabilitySelector(base_selector=base, n_bootstraps=100)
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        base_selector: BaseFeatureSelector,
        n_bootstraps: int = 100,
        threshold: float = 0.6,
        sample_fraction: float = 0.8,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize stability selector.

        Args:
            base_selector: Feature selector to wrap (any BaseFeatureSelector).
            n_bootstraps: Number of bootstrap samples to generate.
            threshold: Minimum selection frequency (0-1) to be considered stable.
                Must be between 0 and 1 (exclusive).
            sample_fraction: Fraction of samples to use in each bootstrap.
                Must be between 0 and 1 (exclusive).
            random_state: Random seed for reproducibility.
            verbose: Print progress messages.

        Raises:
            ValueError: If threshold not in (0, 1) or sample_fraction not in (0, 1).
        """
        if not (0 < threshold < 1):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        if not (0 < sample_fraction <= 1):
            raise ValueError(
                f"sample_fraction must be between 0 and 1, got {sample_fraction}"
            )

        super().__init__(
            n_features_to_select=None,  # Determined by stability threshold
            random_state=random_state,
            verbose=verbose,
        )

        self.base_selector = base_selector
        self.n_bootstraps = n_bootstraps
        self.threshold = threshold
        self.sample_fraction = sample_fraction

        # Attributes set during fit
        self.stability_scores_: dict[str, float] = {}
        self.selection_counts_: dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StabilitySelector":
        """Fit stability selector using bootstrap resampling.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        sample_size = int(n_samples * self.sample_fraction)

        if self.verbose:
            print(
                f"Running stability selection with {self.n_bootstraps} bootstraps "
                f"(sample_fraction={self.sample_fraction}, threshold={self.threshold})..."
            )

        # Initialize selection counts
        selection_counts = {feature: 0 for feature in X.columns}

        # Random number generator
        rng = np.random.RandomState(self.random_state)

        # Run base selector on bootstrap samples
        for i in range(self.n_bootstraps):
            # Generate bootstrap sample (without replacement)
            sample_indices = rng.choice(n_samples, sample_size, replace=False)
            X_bootstrap = X.iloc[sample_indices]
            y_bootstrap = y.iloc[sample_indices]

            # Clone base selector to avoid state contamination
            # Use sklearn's clone for reliable copying of estimators
            try:
                # Try sklearn clone first (works for sklearn-based selectors)
                bootstrap_selector = sklearn_clone(self.base_selector)
            except (TypeError, AttributeError):
                # Fall back to deepcopy for custom selectors
                bootstrap_selector = copy.deepcopy(self.base_selector)

            # Fit selector on bootstrap sample
            try:
                bootstrap_selector.fit(X_bootstrap, y_bootstrap)

                # Get selected features
                if hasattr(bootstrap_selector, "selected_features_"):
                    selected = bootstrap_selector.selected_features_
                    for feature in selected:
                        selection_counts[feature] += 1
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Bootstrap {i+1} failed: {e}")
                continue

            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Completed {i+1}/{self.n_bootstraps} bootstraps")

        # Compute stability scores
        self.selection_counts_ = selection_counts
        self.stability_scores_ = {
            feature: count / self.n_bootstraps
            for feature, count in selection_counts.items()
        }

        # Select features above threshold
        stable_features = [
            feature
            for feature, score in self.stability_scores_.items()
            if score >= self.threshold
        ]

        # Sort by stability score (descending)
        stable_features.sort(key=lambda f: self.stability_scores_[f], reverse=True)

        # Store results
        self.selected_features_ = stable_features
        self.feature_scores_ = np.array(
            [self.stability_scores_[f] for f in stable_features]
        )

        # Create support mask
        self.support_mask_ = np.zeros(n_features, dtype=bool)
        for i, feature in enumerate(X.columns):
            if feature in stable_features:
                self.support_mask_[i] = True

        if self.verbose:
            print(
                f"Selected {len(self.selected_features_)} stable features "
                f"(threshold={self.threshold})"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting stable features.

        Args:
            X: Feature matrix to transform.

        Returns:
            DataFrame with only stable features.

        Raises:
            ValueError: If selector not fitted yet.
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> NDArray:
        """Get mask or indices of stable features.

        Args:
            indices: If True, return integer indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of stable features.

        Raises:
            ValueError: If selector not fitted yet.
        """
        if self.support_mask_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

    def get_result(self) -> Any:
        """Get feature selection result with stability metadata.

        Returns:
            FeatureSelectorResult with stability scores in metadata.
        """
        result = super().get_result()

        # Initialize metadata dictionary if None
        if result.metadata is None:
            result.metadata = {}

        # Add stability-specific metadata
        result.metadata["stability_scores"] = self.stability_scores_
        result.metadata["selection_counts"] = self.selection_counts_
        result.metadata["n_bootstraps"] = self.n_bootstraps
        result.metadata["threshold"] = self.threshold
        result.metadata["sample_fraction"] = self.sample_fraction

        return result
