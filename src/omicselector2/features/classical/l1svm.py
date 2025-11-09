"""L1-SVM feature selector.

This module implements feature selection using Linear SVM with L1 regularization.
The L1 penalty encourages sparsity, automatically eliminating irrelevant features
by driving their coefficients to exactly zero.

L1-SVM objective: min C∑ᵢ loss(yᵢ, f(xᵢ)) + ||w||₁

Key characteristics:
- Embedded method: Feature selection happens during model training
- Sparsity: L1 penalty zeros out irrelevant features
- Classification-based: Uses SVM's margin-based objective
- Automatic: No separate feature ranking step needed
- Scalable: Linear SVM is efficient for high-dimensional data

Particularly useful for:
- High-dimensional classification (genomics, text)
- When SVM's margin-based approach is appropriate
- Embedded feature selection during model training
- Sparse linear models

Examples:
    >>> from omicselector2.features.classical.l1svm import L1SVMSelector
    >>> # Binary classification
    >>> selector = L1SVMSelector(C=0.1, n_features_to_select=50)
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Multi-class classification with more regularization
    >>> selector = L1SVMSelector(C=0.01, n_features_to_select=100)
    >>> X_filtered = selector.fit_transform(X, y)
"""

from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from omicselector2.features.base import BaseFeatureSelector


class L1SVMSelector(BaseFeatureSelector):
    """Feature selector using Linear SVM with L1 regularization.

    Linear SVM with L1 penalty automatically performs feature selection
    by driving irrelevant feature coefficients to exactly zero. Features
    with non-zero coefficients are selected and ranked by absolute magnitude.

    The regularization strength is controlled by C (inverse regularization):
    - Smaller C = more regularization = fewer features
    - Larger C = less regularization = more features

    Attributes:
        C: Inverse regularization strength (smaller = more regularization).
        penalty: Regularization type ('l1' or 'l2'). Default 'l1'.
        model_: Trained LinearSVC model.
        scaler_: StandardScaler for feature standardization.

    Examples:
        >>> # Strong regularization (sparse model)
        >>> selector = L1SVMSelector(C=0.01, n_features_to_select=50)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Moderate regularization
        >>> selector = L1SVMSelector(C=0.1, n_features_to_select=100)
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l1",
        dual: bool = False,
        max_iter: int = 10000,
        tol: float = 1e-4,
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        standardize: bool = True,
    ) -> None:
        """Initialize L1-SVM selector.

        Args:
            C: Inverse regularization strength (must be positive).
                Smaller values = more regularization = sparser models.
            penalty: Penalty type ('l1' or 'l2'). Default 'l1'.
            dual: Whether to solve dual problem. Must be False for L1 penalty.
            max_iter: Maximum iterations for solver.
            tol: Tolerance for stopping criteria.
            n_features_to_select: Number of features to select.
            random_state: Random seed for reproducibility.
            verbose: Print progress messages.
            standardize: Standardize features before fitting.

        Raises:
            ValueError: If C <= 0.
        """
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")

        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose,
        )

        self.C = C
        self.penalty = penalty
        self.dual = dual
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize

        self.model_: Optional[LinearSVC] = None
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "L1SVMSelector":
        """Fit L1-SVM selector to data.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable (classification labels).

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        # Standardize features if requested
        X_train = X.values
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X_train)

        # Build L1-SVM model
        if self.verbose:
            print(f"Training Linear SVM with L1 penalty (C={self.C})...")

        self.model_ = LinearSVC(
            penalty=self.penalty,
            dual=self.dual,
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self.model_.fit(X_train, y)

        # Select features based on coefficient magnitude
        self._select_features(X)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with L1-SVM")

        return self

    def _select_features(self, X: pd.DataFrame) -> None:
        """Select features based on absolute coefficient magnitude."""
        # Get coefficients
        coefficients = self.model_.coef_

        # For binary classification, coefficients are 1D
        if coefficients.ndim == 2 and coefficients.shape[0] == 1:
            coefficients = coefficients.ravel()
        elif coefficients.ndim == 2:
            # Multi-class: use average absolute coefficient across classes
            coefficients = np.abs(coefficients).mean(axis=0)

        # Rank by absolute coefficient value
        abs_coefs = np.abs(coefficients)

        # Sort all features by absolute coefficient (descending)
        sorted_indices = np.argsort(-abs_coefs)

        # Determine number of features to select
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            # Select only non-zero features
            nonzero_mask = abs_coefs > 1e-10
            n_to_select = np.sum(nonzero_mask)
            if n_to_select == 0:
                n_to_select = n_features  # Select all if none are sparse
        else:
            n_to_select = min(self.n_features_to_select, n_features)

        selected_indices = sorted_indices[:n_to_select]
        selected_scores = abs_coefs[selected_indices]

        # Store results
        self.selected_features_ = [X.columns[i] for i in selected_indices]
        self.feature_scores_ = selected_scores

        # Create support mask
        self.support_mask_ = np.zeros(n_features, dtype=bool)
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
