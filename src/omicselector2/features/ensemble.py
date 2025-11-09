"""Ensemble feature selection methods.

This module implements ensemble strategies that combine results from multiple
feature selectors to improve robustness and reduce selection bias.

Ensemble strategies work by:
1. Running multiple feature selection methods independently
2. Aggregating their results using various voting/ranking schemes
3. Selecting features with high agreement/consensus

Key benefits:
- Reduces method-specific bias
- Improves stability across different data samples
- Provides confidence scores based on agreement
- Works with any combination of base selectors

Based on:
- OmicSelector 1.0's automated benchmarking philosophy
- Ensemble feature selection literature (Saeys et al., 2008)
- Borda count voting for rank aggregation

Examples:
    >>> from omicselector2.features.ensemble import EnsembleSelector
    >>> from omicselector2.features.classical.lasso import LassoSelector
    >>> from omicselector2.features.classical.random_forest import RandomForestSelector
    >>>
    >>> # Majority voting: features selected by >=50% of methods
    >>> selectors = [
    ...     LassoSelector(n_features_to_select=20),
    ...     RandomForestSelector(n_features_to_select=20),
    ... ]
    >>> ensemble = EnsembleSelector(selectors=selectors, strategy="majority")
    >>> ensemble.fit(X_train, y_train)
    >>>
    >>> # Soft voting: weighted by feature scores
    >>> ensemble = EnsembleSelector(
    ...     selectors=selectors,
    ...     strategy="soft_voting",
    ...     weights=[0.6, 0.4],
    ...     n_features_to_select=15
    ... )
    >>> X_filtered = ensemble.fit_transform(X_train, y_train)
"""

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from omicselector2.features.base import BaseFeatureSelector


class EnsembleSelector(BaseFeatureSelector):
    """Ensemble feature selector combining multiple methods.

    Combines results from multiple feature selectors using various strategies:
    - **majority**: Select features voted for by >= threshold fraction of selectors
    - **soft_voting**: Weighted average of feature scores from all selectors
    - **consensus_ranking**: Borda count rank aggregation across selectors
    - **intersection**: Features selected by ALL selectors
    - **union**: Features selected by ANY selector

    Attributes:
        selectors: List of fitted feature selectors.
        strategy: Ensemble strategy to use.
        threshold: Vote threshold for majority strategy (0-1).
        weights: Weights for soft_voting strategy (must sum to > 0).
        vote_counts_: Dict mapping features to number of selectors that chose them.
        weighted_scores_: Dict of weighted scores (soft_voting strategy).
        consensus_scores_: Dict of Borda count scores (consensus_ranking strategy).

    Examples:
        >>> # Majority voting with 3 selectors
        >>> selectors = [
        ...     LassoSelector(n_features_to_select=20),
        ...     RandomForestSelector(n_features_to_select=20),
        ...     mRMRSelector(n_features_to_select=20),
        ... ]
        >>> ensemble = EnsembleSelector(
        ...     selectors=selectors,
        ...     strategy="majority",
        ...     threshold=0.67  # >=2/3 must vote
        ... )
        >>> ensemble.fit(X, y)
        >>>
        >>> # Get vote counts for transparency
        >>> for feature in ensemble.selected_features_[:10]:
        ...     votes = ensemble.vote_counts_[feature]
        ...     print(f"{feature}: {votes}/3 votes")
    """

    VALID_STRATEGIES = [
        "majority",
        "soft_voting",
        "consensus_ranking",
        "intersection",
        "union",
    ]

    def __init__(
        self,
        selectors: list[BaseFeatureSelector],
        strategy: Literal[
            "majority", "soft_voting", "consensus_ranking", "intersection", "union"
        ] = "majority",
        threshold: float = 0.5,
        weights: Optional[list[float]] = None,
        n_features_to_select: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize ensemble selector.

        Args:
            selectors: List of feature selectors to ensemble (minimum 2).
            strategy: Ensemble strategy (see class docstring).
            threshold: For majority strategy, minimum fraction of votes (0-1).
            weights: For soft_voting, weight for each selector. Must match length
                of selectors. If None, equal weights are used.
            n_features_to_select: Max features to select. For majority/intersection/union,
                this is applied after voting. For soft_voting/consensus_ranking, this
                selects top-k by score.
            verbose: Print progress messages.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=None,
            verbose=verbose,
        )

        # Validate selectors
        if len(selectors) < 2:
            raise ValueError(
                f"At least 2 selectors required for ensemble, got {len(selectors)}"
            )

        # Validate strategy
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.VALID_STRATEGIES}, got '{strategy}'"
            )

        # Validate threshold
        if not (0 < threshold <= 1):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        # Validate weights
        if weights is not None:
            if len(weights) != len(selectors):
                raise ValueError(
                    f"weights length ({len(weights)}) must match number of "
                    f"selectors ({len(selectors)})"
                )
            if any(w < 0 for w in weights):
                raise ValueError("Weights must be non-negative")
            if sum(weights) == 0:
                raise ValueError("Weights must sum to > 0")

        self.selectors = selectors
        self.strategy = strategy
        self.threshold = threshold
        self.weights = weights if weights is not None else [1.0] * len(selectors)

        # Attributes set during fit
        self.vote_counts_: dict[str, int] = {}
        self.weighted_scores_: dict[str, float] = {}
        self.consensus_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleSelector":
        """Fit ensemble selector.

        Fits all base selectors and aggregates their results according to
        the chosen strategy.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        if self.verbose:
            print(
                f"Fitting ensemble with {len(self.selectors)} selectors "
                f"(strategy={self.strategy})..."
            )

        # Fit all selectors
        fitted_selectors = []
        for i, selector in enumerate(self.selectors):
            if self.verbose:
                print(f"  Fitting selector {i+1}/{len(self.selectors)}...")
            selector.fit(X, y)
            fitted_selectors.append(selector)

        # Aggregate results based on strategy
        if self.strategy == "majority":
            self._majority_voting(X, fitted_selectors)
        elif self.strategy == "soft_voting":
            self._soft_voting(X, fitted_selectors)
        elif self.strategy == "consensus_ranking":
            self._consensus_ranking(X, fitted_selectors)
        elif self.strategy == "intersection":
            self._intersection(X, fitted_selectors)
        elif self.strategy == "union":
            self._union(X, fitted_selectors)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features")

        return self

    def _majority_voting(
        self, X: pd.DataFrame, fitted_selectors: list[BaseFeatureSelector]
    ) -> None:
        """Implement majority voting strategy.

        Args:
            X: Feature matrix.
            fitted_selectors: List of fitted selectors.
        """
        # Count votes for each feature
        vote_counts = {feature: 0 for feature in X.columns}

        for selector in fitted_selectors:
            selected = selector.selected_features_
            for feature in selected:
                vote_counts[feature] += 1

        self.vote_counts_ = vote_counts

        # Calculate minimum votes needed
        min_votes = int(np.ceil(len(fitted_selectors) * self.threshold))

        # Select features with enough votes
        selected_features = [
            feature for feature, votes in vote_counts.items() if votes >= min_votes
        ]

        # Sort by vote count (descending)
        selected_features.sort(key=lambda f: vote_counts[f], reverse=True)

        # Limit to n_features_to_select if specified
        if self.n_features_to_select is not None:
            selected_features = selected_features[: self.n_features_to_select]

        # Set attributes
        self.selected_features_ = selected_features
        self.feature_scores_ = np.array([vote_counts[f] for f in selected_features])

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                self.support_mask_[i] = True

    def _soft_voting(
        self, X: pd.DataFrame, fitted_selectors: list[BaseFeatureSelector]
    ) -> None:
        """Implement soft voting with weighted scores.

        Args:
            X: Feature matrix.
            fitted_selectors: List of fitted selectors.
        """
        # Normalize weights
        weight_sum = sum(self.weights)
        normalized_weights = [w / weight_sum for w in self.weights]

        # Initialize weighted scores
        weighted_scores = {feature: 0.0 for feature in X.columns}

        for selector, weight in zip(fitted_selectors, normalized_weights):
            # Get feature scores (normalized to [0, 1])
            selected_features = selector.selected_features_
            if len(selected_features) == 0:
                continue

            scores = selector.feature_scores_

            # Normalize scores to [0, 1]
            if len(scores) > 0:
                max_score = np.max(scores) if np.max(scores) > 0 else 1.0
                normalized_scores = scores / max_score

                # Add weighted scores
                for feature, score in zip(selected_features, normalized_scores):
                    weighted_scores[feature] += weight * score

        self.weighted_scores_ = weighted_scores

        # Sort features by weighted score
        sorted_features = sorted(
            weighted_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Select top-k features
        if self.n_features_to_select is not None:
            sorted_features = sorted_features[: self.n_features_to_select]
        else:
            # Select features with non-zero scores
            sorted_features = [(f, s) for f, s in sorted_features if s > 0]

        selected_features = [f for f, _ in sorted_features]
        feature_scores = np.array([s for _, s in sorted_features])

        # Set attributes
        self.selected_features_ = selected_features
        self.feature_scores_ = feature_scores

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                self.support_mask_[i] = True

    def _consensus_ranking(
        self, X: pd.DataFrame, fitted_selectors: list[BaseFeatureSelector]
    ) -> None:
        """Implement consensus ranking using Borda count.

        Borda count assigns points based on rank position:
        - Top ranked feature gets n_features points
        - Second gets n_features-1 points
        - etc.

        Args:
            X: Feature matrix.
            fitted_selectors: List of fitted selectors.
        """
        # Initialize Borda counts
        borda_counts = {feature: 0 for feature in X.columns}

        for selector in fitted_selectors:
            selected_features = selector.selected_features_
            n_selected = len(selected_features)

            if n_selected == 0:
                continue

            # Assign Borda points (highest rank = most points)
            for rank, feature in enumerate(selected_features):
                points = n_selected - rank
                borda_counts[feature] += points

        self.consensus_scores_ = borda_counts

        # Sort by Borda count
        sorted_features = sorted(
            borda_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Select top-k features
        if self.n_features_to_select is not None:
            sorted_features = sorted_features[: self.n_features_to_select]
        else:
            # Select features with non-zero counts
            sorted_features = [(f, c) for f, c in sorted_features if c > 0]

        selected_features = [f for f, _ in sorted_features]
        feature_scores = np.array([c for _, c in sorted_features], dtype=np.float64)

        # Set attributes
        self.selected_features_ = selected_features
        self.feature_scores_ = feature_scores

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                self.support_mask_[i] = True

    def _intersection(
        self, X: pd.DataFrame, fitted_selectors: list[BaseFeatureSelector]
    ) -> None:
        """Implement intersection strategy (features selected by ALL).

        Args:
            X: Feature matrix.
            fitted_selectors: List of fitted selectors.
        """
        # Get intersection of all selected features
        if len(fitted_selectors) == 0:
            self.selected_features_ = []
            self.feature_scores_ = np.array([])
            self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
            return

        # Start with first selector's features
        common_features = set(fitted_selectors[0].selected_features_)

        # Intersect with remaining selectors
        for selector in fitted_selectors[1:]:
            common_features &= set(selector.selected_features_)

        # Count votes (for sorting - all should have max votes)
        vote_counts = {feature: 0 for feature in X.columns}
        for selector in fitted_selectors:
            for feature in selector.selected_features_:
                vote_counts[feature] += 1

        self.vote_counts_ = vote_counts

        # Sort by vote count (should all be equal for intersection)
        selected_features = sorted(
            common_features, key=lambda f: vote_counts[f], reverse=True
        )

        # Limit to n_features_to_select if specified
        if self.n_features_to_select is not None:
            selected_features = selected_features[: self.n_features_to_select]

        # Set attributes
        self.selected_features_ = selected_features
        self.feature_scores_ = np.array([vote_counts[f] for f in selected_features])

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                self.support_mask_[i] = True

    def _union(
        self, X: pd.DataFrame, fitted_selectors: list[BaseFeatureSelector]
    ) -> None:
        """Implement union strategy (features selected by ANY).

        Args:
            X: Feature matrix.
            fitted_selectors: List of fitted selectors.
        """
        # Count votes for each feature
        vote_counts = {feature: 0 for feature in X.columns}

        for selector in fitted_selectors:
            selected = selector.selected_features_
            for feature in selected:
                vote_counts[feature] += 1

        self.vote_counts_ = vote_counts

        # Select all features with at least 1 vote
        selected_features = [
            feature for feature, votes in vote_counts.items() if votes >= 1
        ]

        # Sort by vote count (descending)
        selected_features.sort(key=lambda f: vote_counts[f], reverse=True)

        # Limit to n_features_to_select if specified
        if self.n_features_to_select is not None:
            selected_features = selected_features[: self.n_features_to_select]

        # Set attributes
        self.selected_features_ = selected_features
        self.feature_scores_ = np.array([vote_counts[f] for f in selected_features])

        # Create support mask
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                self.support_mask_[i] = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting ensemble-chosen features.

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

    def get_result(self) -> Any:
        """Get feature selection result with ensemble metadata.

        Returns:
            FeatureSelectorResult with ensemble-specific metadata.
        """
        result = super().get_result()

        # Initialize metadata
        if result.metadata is None:
            result.metadata = {}

        # Add ensemble-specific metadata
        result.metadata["strategy"] = self.strategy
        result.metadata["n_selectors"] = len(self.selectors)
        result.metadata["threshold"] = self.threshold

        if self.strategy in ["majority", "intersection", "union"]:
            result.metadata["vote_counts"] = self.vote_counts_
        elif self.strategy == "soft_voting":
            result.metadata["weighted_scores"] = self.weighted_scores_
            result.metadata["weights"] = self.weights
        elif self.strategy == "consensus_ranking":
            result.metadata["consensus_scores"] = self.consensus_scores_

        return result
