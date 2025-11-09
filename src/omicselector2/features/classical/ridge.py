"""Ridge Regression feature selector.

This module implements feature selection using Ridge (L2) regularization.
Unlike Lasso (L1), Ridge doesn't force coefficients to exactly zero but shrinks
them proportionally, providing a ranking of all features.

Ridge regularization: min ||y - Xw||² + α||w||²

Key characteristics:
- L2 penalty shrinks coefficients but doesn't eliminate features
- Better handles multicollinearity than Lasso
- Keeps correlated features together with similar weights
- All features get non-zero weights (need to select top-k)
- More stable than Lasso when features are highly correlated

Particularly useful for:
- Genomics data with correlated gene expression
- Situations where many features are relevant
- Stabilizing coefficient estimates
- Feature ranking rather than sparse selection

Examples:
    >>> from omicselector2.features.classical.ridge import RidgeSelector
    >>> # Regression with fixed alpha
    >>> selector = RidgeSelector(alpha=1.0, n_features_to_select=50)
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Automatic alpha selection via CV
    >>> selector = RidgeSelector(alpha='auto', cv=5, n_features_to_select=100)
    >>> X_filtered = selector.fit_transform(X, y)
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge, RidgeClassifier, RidgeCV, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from omicselector2.features.base import BaseFeatureSelector


class RidgeSelector(BaseFeatureSelector):
    """Feature selector using Ridge (L2) regularization.

    Ridge regularization applies L2 penalty to regression/classification,
    shrinking coefficients proportionally. Features are ranked by absolute
    coefficient magnitude.

    Unlike Lasso, Ridge:
    - Doesn't eliminate features (all have non-zero coefficients)
    - Better handles correlated features (keeps them together)
    - Provides stable coefficient estimates
    - Requires selecting top-k features after ranking

    Attributes:
        alpha: Regularization strength (positive float or 'auto' for CV).
        task: Type of task - 'regression' or 'classification'.
        model_: Trained Ridge model.
        alpha_: Optimal alpha if using cross-validation.

    Examples:
        >>> # Fixed alpha
        >>> selector = RidgeSelector(alpha=1.0, n_features_to_select=50)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Automatic alpha via CV
        >>> selector = RidgeSelector(alpha='auto', cv=5, n_features_to_select=100)
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        alpha: Union[float, Literal["auto"]] = 1.0,
        task: Literal["regression", "classification"] = "regression",
        cv: int = 5,
        max_iter: int = 10000,
        tol: float = 1e-4,
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        standardize: bool = True,
    ) -> None:
        """Initialize Ridge selector.

        Args:
            alpha: Regularization strength (positive float or 'auto' for CV).
            task: 'regression' or 'classification'.
            cv: Number of cross-validation folds (if alpha='auto').
            max_iter: Maximum iterations for solver.
            tol: Tolerance for stopping criteria.
            n_features_to_select: Number of features to select (required).
            random_state: Random seed for reproducibility.
            verbose: Print progress messages.
            standardize: Standardize features before fitting.

        Raises:
            ValueError: If alpha is not positive or 'auto'.
            ValueError: If task not in ['regression', 'classification'].
        """
        if isinstance(alpha, (int, float)) and alpha <= 0:
            raise ValueError(f"alpha must be positive or 'auto', got {alpha}")
        elif not isinstance(alpha, (int, float, str)):
            raise ValueError(f"alpha must be positive or 'auto', got {alpha}")
        elif isinstance(alpha, str) and alpha != "auto":
            raise ValueError(f"alpha must be positive or 'auto', got {alpha}")

        if task not in ["regression", "classification"]:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")

        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose,
        )

        self.alpha = alpha
        self.task = task
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize

        self.model_: Optional[Union[Ridge, RidgeClassifier, RidgeCV, RidgeClassifierCV]] = None
        self.scaler_: Optional[StandardScaler] = None
        self.alpha_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeSelector":
        """Fit Ridge selector to data.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable.

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

        # Build Ridge model
        if self.alpha == "auto":
            # Use cross-validation to select alpha
            if self.verbose:
                print(f"Using {self.cv}-fold CV to select optimal alpha...")

            if self.task == "classification":
                self.model_ = RidgeClassifierCV(
                    alphas=np.logspace(-3, 3, 20),
                    cv=self.cv,
                )
            else:
                self.model_ = RidgeCV(
                    alphas=np.logspace(-3, 3, 20),
                    cv=self.cv,
                )

            self.model_.fit(X_train, y)
            self.alpha_ = self.model_.alpha_

            if self.verbose:
                print(f"Selected alpha: {self.alpha_:.4f}")

        else:
            # Use fixed alpha
            if self.task == "classification":
                self.model_ = RidgeClassifier(
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
            else:
                self.model_ = Ridge(
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )

            self.model_.fit(X_train, y)
            self.alpha_ = self.alpha

        # Select features based on coefficient magnitude
        self._select_features(X)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with Ridge")

        return self

    def _select_features(self, X: pd.DataFrame) -> None:
        """Select features based on absolute coefficient magnitude."""
        # Get coefficients
        if hasattr(self.model_, "coef_"):
            coefficients = self.model_.coef_
        else:
            raise AttributeError("Model doesn't have coefficients")

        # For binary classification, coefficients may be 1D
        if coefficients.ndim == 2 and coefficients.shape[0] == 1:
            coefficients = coefficients.ravel()
        elif coefficients.ndim == 2:
            # Multi-class: use average absolute coefficient
            coefficients = np.abs(coefficients).mean(axis=0)

        # Rank by absolute coefficient value
        abs_coefs = np.abs(coefficients)

        # Determine number of features to select
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_to_select = n_features
        else:
            n_to_select = min(self.n_features_to_select, n_features)

        # Select top features
        top_indices = np.argsort(-abs_coefs)[:n_to_select]

        # Sort selected features by score (descending)
        scores = abs_coefs[top_indices]
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
