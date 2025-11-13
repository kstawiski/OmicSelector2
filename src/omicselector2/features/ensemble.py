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


# Utility functions for ensemble feature selection

def majority_vote(
    selection_results: dict[str, list[str]], min_votes: int = 2
) -> list[str]:
    """Select features by majority voting.

    Args:
        selection_results: Dict mapping method names to selected feature lists.
        min_votes: Minimum number of votes required for selection.

    Returns:
        List of features with >= min_votes.

    Examples:
        >>> results = {
        ...     'lasso': ['GENE_0', 'GENE_1'],
        ...     'rf': ['GENE_0', 'GENE_2'],
        ...     'mrmr': ['GENE_0', 'GENE_1'],
        ... }
        >>> majority_vote(results, min_votes=2)
        ['GENE_0', 'GENE_1']
    """
    # Count votes for each feature
    vote_counts: dict[str, int] = {}
    for features in selection_results.values():
        for feature in features:
            vote_counts[feature] = vote_counts.get(feature, 0) + 1

    # Select features with enough votes
    selected = [
        feature for feature, votes in vote_counts.items() if votes >= min_votes
    ]

    # Sort by vote count (descending)
    selected.sort(key=lambda f: vote_counts[f], reverse=True)

    return selected


def soft_vote(
    ranked_results: dict[str, dict[str, float]],
    n_features: int,
    weights: Optional[dict[str, float]] = None,
) -> tuple[list[str], dict[str, float]]:
    """Select features by soft (weighted) voting.

    Args:
        ranked_results: Dict mapping method names to feature score dicts.
        n_features: Number of features to select.
        weights: Optional weights for each method. If None, equal weights.

    Returns:
        Tuple of (selected_features, aggregated_scores).

    Examples:
        >>> results = {
        ...     'lasso': {'GENE_0': 0.9, 'GENE_1': 0.7},
        ...     'rf': {'GENE_0': 0.8, 'GENE_2': 0.6},
        ... }
        >>> selected, scores = soft_vote(results, n_features=2)
    """
    # Default to equal weights
    if weights is None:
        weights = {method: 1.0 for method in ranked_results.keys()}

    # Normalize weights
    weight_sum = sum(weights.values())
    normalized_weights = {m: w / weight_sum for m, w in weights.items()}

    # Collect all features
    all_features = set()
    for features in ranked_results.values():
        all_features.update(features.keys())

    # Calculate weighted scores
    aggregated_scores: dict[str, float] = {}

    for feature in all_features:
        weighted_score = 0.0
        for method, feature_scores in ranked_results.items():
            if feature in feature_scores:
                # Normalize score to [0, 1] within method
                method_scores = list(feature_scores.values())
                max_score = max(method_scores) if method_scores else 1.0
                normalized_score = feature_scores[feature] / max_score if max_score > 0 else 0

                # Add weighted contribution
                weighted_score += normalized_weights[method] * normalized_score

        aggregated_scores[feature] = weighted_score

    # Sort and select top-k
    sorted_features = sorted(
        aggregated_scores.items(), key=lambda x: x[1], reverse=True
    )[:n_features]

    selected_features = [f for f, _ in sorted_features]

    return selected_features, aggregated_scores


def consensus_ranking(
    selection_results: dict[str, list[str]], method: str = 'borda_count'
) -> list[str]:
    """Create consensus ranking from multiple selection results.

    Args:
        selection_results: Dict mapping method names to selected feature lists
            (ordered by importance) or feature score dicts.
        method: Ranking method. Options:
            - 'borda_count': Assign points by rank (highest rank = most points)
            - 'mean_rank': Average rank across methods
            - 'mean_score': Average normalized scores (requires score dicts)

    Returns:
        List of features sorted by consensus ranking.

    Examples:
        >>> results = {
        ...     'lasso': ['GENE_0', 'GENE_1', 'GENE_2'],
        ...     'rf': ['GENE_0', 'GENE_2', 'GENE_3'],
        ... }
        >>> consensus_ranking(results, method='borda_count')
        ['GENE_0', 'GENE_2', 'GENE_1', 'GENE_3']
    """
    if method == 'borda_count':
        # Borda count: top ranked gets n points, second gets n-1, etc.
        borda_counts: dict[str, int] = {}

        for method_name, features in selection_results.items():
            if isinstance(features, dict):
                # If dict, use sorted keys (by score)
                feature_list = sorted(features.keys(), key=lambda f: features[f], reverse=True)
            else:
                feature_list = features

            n_features = len(feature_list)
            for rank, feature in enumerate(feature_list):
                points = n_features - rank
                borda_counts[feature] = borda_counts.get(feature, 0) + points

        # Sort by Borda count
        ranked = sorted(borda_counts.keys(), key=lambda f: borda_counts[f], reverse=True)
        return ranked

    elif method == 'mean_rank':
        # Average rank across methods (lower is better)
        rank_sums: dict[str, float] = {}
        rank_counts: dict[str, int] = {}

        for method_name, features in selection_results.items():
            if isinstance(features, dict):
                feature_list = sorted(features.keys(), key=lambda f: features[f], reverse=True)
            else:
                feature_list = features

            for rank, feature in enumerate(feature_list):
                rank_sums[feature] = rank_sums.get(feature, 0.0) + rank
                rank_counts[feature] = rank_counts.get(feature, 0) + 1

        # Calculate mean rank
        mean_ranks = {
            f: rank_sums[f] / rank_counts[f] for f in rank_sums.keys()
        }

        # Sort by mean rank (lower is better)
        ranked = sorted(mean_ranks.keys(), key=lambda f: mean_ranks[f])
        return ranked

    elif method == 'mean_score':
        # Average normalized scores
        score_sums: dict[str, float] = {}
        score_counts: dict[str, int] = {}

        for method_name, features in selection_results.items():
            if not isinstance(features, dict):
                raise ValueError("mean_score requires feature score dicts")

            # Normalize scores to [0, 1]
            max_score = max(features.values()) if features else 1.0
            for feature, score in features.items():
                normalized_score = score / max_score if max_score > 0 else 0
                score_sums[feature] = score_sums.get(feature, 0.0) + normalized_score
                score_counts[feature] = score_counts.get(feature, 0) + 1

        # Calculate mean score
        mean_scores = {
            f: score_sums[f] / score_counts[f] for f in score_sums.keys()
        }

        # Sort by mean score (higher is better)
        ranked = sorted(mean_scores.keys(), key=lambda f: mean_scores[f], reverse=True)
        return ranked

    else:
        raise ValueError(f"Unknown method: {method}")


def intersection_selection(selection_results: dict[str, list[str]]) -> list[str]:
    """Select features present in ALL methods (intersection).

    Args:
        selection_results: Dict mapping method names to selected feature lists.

    Returns:
        List of features selected by all methods.

    Examples:
        >>> results = {
        ...     'lasso': ['GENE_0', 'GENE_1'],
        ...     'rf': ['GENE_0', 'GENE_2'],
        ... }
        >>> intersection_selection(results)
        ['GENE_0']
    """
    if not selection_results:
        return []

    # Start with first method's features
    common_features = set(list(selection_results.values())[0])

    # Intersect with remaining methods
    for features in list(selection_results.values())[1:]:
        common_features &= set(features)

    return sorted(common_features)


def union_selection(selection_results: dict[str, list[str]]) -> list[str]:
    """Select features present in ANY method (union).

    Args:
        selection_results: Dict mapping method names to selected feature lists.

    Returns:
        List of features selected by at least one method.

    Examples:
        >>> results = {
        ...     'lasso': ['GENE_0', 'GENE_1'],
        ...     'rf': ['GENE_0', 'GENE_2'],
        ... }
        >>> union_selection(results)
        ['GENE_0', 'GENE_1', 'GENE_2']
    """
    all_features = set()
    for features in selection_results.values():
        all_features.update(features)

    return sorted(all_features)


class EnsembleFeatureSelector:
    """Ensemble feature selector for function-based selectors.

    Combines results from multiple function-based feature selection methods
    using various ensemble strategies.

    Args:
        base_selectors: List of feature selection functions.
        ensemble_method: Ensemble strategy. Options:
            - 'majority_vote': Select features voted for by >= min_votes
            - 'soft_vote': Weighted average of feature scores
            - 'consensus_ranking': Borda count rank aggregation
            - 'intersection': Features selected by ALL methods
            - 'union': Features selected by ANY method
        min_votes: For majority_vote, minimum votes required.
        n_features: Maximum features to select (for soft_vote/consensus_ranking).
        weights: Optional weights for soft_vote.
        verbose: Print progress messages.

    Examples:
        >>> from omicselector2.tasks.feature_selection import (
        ...     run_lasso_feature_selection,
        ...     run_randomforest_feature_selection,
        ... )
        >>> selector = EnsembleFeatureSelector(
        ...     base_selectors=[
        ...         run_lasso_feature_selection,
        ...         run_randomforest_feature_selection,
        ...     ],
        ...     ensemble_method='majority_vote',
        ...     min_votes=2
        ... )
        >>> selected, metrics = selector.select_features(X, y, n_features=20, cv=5)
    """

    VALID_METHODS = [
        'majority_vote',
        'soft_vote',
        'consensus_ranking',
        'intersection',
        'union',
    ]

    def __init__(
        self,
        base_selectors: list,
        ensemble_method: str = 'majority_vote',
        min_votes: int = 2,
        n_features: Optional[int] = None,
        weights: Optional[dict[str, float]] = None,
        verbose: bool = False,
    ):
        """Initialize ensemble selector."""
        if ensemble_method not in self.VALID_METHODS:
            raise ValueError(
                f"ensemble_method must be one of {self.VALID_METHODS}, "
                f"got '{ensemble_method}'"
            )

        if len(base_selectors) < 2:
            raise ValueError(
                f"At least 2 selectors required, got {len(base_selectors)}"
            )

        self.base_selectors = base_selectors
        self.ensemble_method = ensemble_method
        self.min_votes = min_votes
        self.n_features = n_features
        self.weights = weights
        self.verbose = verbose

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, n_features: int = 100, cv: int = 5
    ) -> tuple[list[str], dict]:
        """Select features using ensemble of methods.

        Args:
            X: Feature matrix.
            y: Target variable.
            n_features: Number of features for each base selector.
            cv: Cross-validation folds for base selectors.

        Returns:
            Tuple of (selected_features, metrics).
        """
        if self.verbose:
            print(
                f"Running ensemble with {len(self.base_selectors)} methods "
                f"(strategy={self.ensemble_method})..."
            )

        # Run each base selector
        method_results = {}
        method_scores = {}

        for i, selector_func in enumerate(self.base_selectors):
            if self.verbose:
                method_name = getattr(selector_func, '__name__', f'method_{i}')
                print(f"  Running {method_name}...")

            try:
                selected, metrics = selector_func(X, y, cv=cv, n_features=n_features)
                method_name = metrics.get('method', f'method_{i}')
                method_results[method_name] = selected

                # Extract scores if available
                if 'importance_scores' in metrics:
                    method_scores[method_name] = metrics['importance_scores']
                elif 'mutual_information_scores' in metrics:
                    method_scores[method_name] = metrics['mutual_information_scores']
                elif 'relief_scores' in metrics:
                    method_scores[method_name] = metrics['relief_scores']

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Method {i} failed: {e}")
                continue

        # Apply ensemble strategy
        if self.ensemble_method == 'majority_vote':
            selected_features = majority_vote(method_results, min_votes=self.min_votes)

        elif self.ensemble_method == 'soft_vote':
            if not method_scores:
                # Fallback to majority vote if no scores available
                selected_features = majority_vote(method_results, min_votes=self.min_votes)
            else:
                n_select = self.n_features if self.n_features else n_features
                selected_features, aggregated_scores = soft_vote(
                    method_scores, n_features=n_select, weights=self.weights
                )

        elif self.ensemble_method == 'consensus_ranking':
            # Use scores if available, otherwise use rankings
            if method_scores:
                selected_features = consensus_ranking(method_scores, method='mean_score')
            else:
                selected_features = consensus_ranking(method_results, method='borda_count')

            # Limit to n_features
            if self.n_features:
                selected_features = selected_features[:self.n_features]

        elif self.ensemble_method == 'intersection':
            selected_features = intersection_selection(method_results)

        elif self.ensemble_method == 'union':
            selected_features = union_selection(method_results)
            # Limit to n_features
            if self.n_features:
                selected_features = selected_features[:self.n_features]

        else:
            raise ValueError(f"Unknown ensemble_method: {self.ensemble_method}")

        # Build metrics
        ensemble_metrics = {
            'ensemble_method': self.ensemble_method,
            'n_methods': len(method_results),
            'n_features_selected': len(selected_features),
            'method_results': method_results,
        }

        if method_scores and self.ensemble_method == 'soft_vote':
            ensemble_metrics['aggregated_scores'] = aggregated_scores

        if self.verbose:
            print(f"Ensemble selected {len(selected_features)} features")

        return selected_features, ensemble_metrics

