"""Lasso (L1 regularization) feature selector.

This module implements feature selection using Lasso regression, which performs
L1 regularization to drive coefficients of irrelevant features to zero.

Lasso is FDA-cleared in diagnostic panels and widely used in genomics for
identifying sparse gene signatures.

Examples:
    >>> from omicselector2.features.classical.lasso import LassoSelector
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = pd.DataFrame(np.random.randn(100, 50))
    >>> y = pd.Series(np.random.randn(100))
    >>>
    >>> # Select features with Lasso
    >>> selector = LassoSelector(alpha=0.1)
    >>> selector.fit(X, y)
    >>> selected_features = selector.get_feature_names_out()
    >>> print(f"Selected {len(selected_features)} features")
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from omicselector2.features.base import BaseFeatureSelector


class LassoSelector(BaseFeatureSelector):
    """Feature selector using Lasso (L1 regularization).

    Lasso performs L1 regularization which drives coefficients of irrelevant
    features to exactly zero, providing automatic feature selection.

    For classification tasks, uses Logistic Regression with L1 penalty.
    For regression tasks, uses Lasso regression.

    Attributes:
        alpha: Regularization strength. Larger values = more regularization.
               Use 'auto' for automatic cross-validated selection.
        task: Type of task - 'regression' or 'classification'.
        cv: Number of cross-validation folds for automatic alpha selection.
        max_iter: Maximum number of iterations for optimization.
        tol: Tolerance for optimization convergence.
        n_features_to_select: Maximum number of features to select.
        random_state: Random seed for reproducibility.
        alpha_: Actual alpha value used (set after fit if alpha='auto').

    Examples:
        >>> # Regression task
        >>> selector = LassoSelector(alpha=0.01)
        >>> selector.fit(X_train, y_train)
        >>> X_selected = selector.transform(X_test)
        >>>
        >>> # Classification with auto alpha
        >>> selector = LassoSelector(alpha='auto', task='classification', cv=5)
        >>> selector.fit(X_train, y_train)
        >>> result = selector.get_result()
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        alpha: Union[float, Literal['auto']] = 1.0,
        task: Literal['regression', 'classification'] = 'regression',
        cv: int = 5,
        max_iter: int = 10000,
        tol: float = 1e-4,
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        standardize: bool = True
    ) -> None:
        """Initialize Lasso feature selector.

        Args:
            alpha: Regularization strength. Use 'auto' for CV selection.
            task: 'regression' or 'classification'.
            cv: Number of CV folds if alpha='auto'.
            max_iter: Maximum iterations for optimization.
            tol: Convergence tolerance.
            n_features_to_select: Max features to select. None = all non-zero.
            random_state: Random seed for reproducibility.
            verbose: Print progress information.
            standardize: Whether to standardize features before fitting.

        Raises:
            ValueError: If alpha is not positive or task is invalid.
        """
        super().__init__(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            verbose=verbose
        )

        if isinstance(alpha, (int, float)) and alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        if task not in ['regression', 'classification']:
            raise ValueError(
                f"task must be 'regression' or 'classification', got {task}"
            )

        self.alpha = alpha
        self.task = task
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize

        # Will be set during fit
        self.model_: Optional[Union[Lasso, LogisticRegression]] = None
        self.scaler_: Optional[StandardScaler] = None
        self.alpha_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoSelector":
        """Fit Lasso selector to data.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target variable with shape (n_samples,).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input data is invalid.
        """
        # Validate input
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        # Standardize if requested
        X_scaled = X.copy()
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler_.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

        # Fit model based on task type
        if self.task == 'regression':
            self._fit_regression(X_scaled, y)
        else:
            self._fit_classification(X_scaled, y)

        # Select features based on non-zero coefficients
        self._select_features(X)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with Lasso")

        return self

    def _fit_regression(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Lasso for regression task.

        Args:
            X: Scaled feature matrix.
            y: Target variable.
        """
        if self.alpha == 'auto':
            # Use cross-validation to find optimal alpha
            self.model_ = LassoCV(
                cv=self.cv,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model_.fit(X, y)
            self.alpha_ = self.model_.alpha_
        else:
            # Use fixed alpha
            self.model_ = Lasso(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            self.model_.fit(X, y)
            self.alpha_ = self.alpha

    def _fit_classification(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Lasso for classification task (Logistic Regression with L1).

        Args:
            X: Scaled feature matrix.
            y: Target variable.
        """
        if self.alpha == 'auto':
            # Use cross-validation to find optimal C (inverse of alpha)
            from sklearn.model_selection import cross_val_score

            # Try multiple C values
            C_values = np.logspace(-4, 4, 20)
            best_score = -np.inf
            best_C = 1.0

            for C in C_values:
                model = LogisticRegression(
                    penalty='l1',
                    C=C,
                    solver='liblinear',
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state
                )
                scores = cross_val_score(model, X, y, cv=self.cv, n_jobs=-1)
                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_C = C

            # Fit final model with best C
            self.model_ = LogisticRegression(
                penalty='l1',
                C=best_C,
                solver='liblinear',
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            self.model_.fit(X, y)
            self.alpha_ = 1.0 / best_C
        else:
            # Use fixed alpha (convert to C)
            C = 1.0 / self.alpha
            self.model_ = LogisticRegression(
                penalty='l1',
                C=C,
                solver='liblinear',
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            self.model_.fit(X, y)
            self.alpha_ = self.alpha

    def _select_features(self, X: pd.DataFrame) -> None:
        """Select features based on non-zero coefficients.

        Args:
            X: Original feature matrix (for column names).
        """
        # Get coefficients
        if hasattr(self.model_, 'coef_'):
            # For both Lasso and LogisticRegression
            coef = self.model_.coef_
            if coef.ndim > 1:
                # LogisticRegression can have shape (n_classes, n_features)
                # Use mean absolute coefficient across classes
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
        else:
            raise ValueError("Model does not have coefficients")

        # If n_features_to_select is specified, select top-k by magnitude
        # Otherwise, select only non-zero coefficients
        if self.n_features_to_select is not None:
            # Rank all features by absolute coefficient (descending)
            sorted_indices = np.argsort(-coef)
            n_select = min(self.n_features_to_select, len(coef))
            selected_indices = sorted_indices[:n_select]

            # Create support mask
            self.support_mask_ = np.zeros(len(coef), dtype=bool)
            self.support_mask_[selected_indices] = True

            # Get feature scores
            self.feature_scores_ = coef[selected_indices]
        else:
            # Select only non-zero coefficients
            self.support_mask_ = coef > 1e-10
            selected_indices = np.where(self.support_mask_)[0]

            # Get feature scores (absolute coefficients)
            self.feature_scores_ = coef[selected_indices]

            # Sort by score (descending)
            sorted_indices = np.argsort(-self.feature_scores_)
            selected_indices = selected_indices[sorted_indices]
            self.feature_scores_ = self.feature_scores_[sorted_indices]

        # Get feature names
        self.selected_features_ = [X.columns[i] for i in selected_indices]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Transformed DataFrame with selected features only.

        Raises:
            ValueError: If selector has not been fitted yet.
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")

        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> NDArray:
        """Get mask or indices of selected features.

        Args:
            indices: If True, return feature indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected features.

        Raises:
            ValueError: If selector has not been fitted yet.
        """
        if self.support_mask_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")

        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

    def get_coefficients(self) -> NDArray[np.float64]:
        """Get Lasso coefficients for all features.

        Returns:
            Array of coefficients (including zeros for non-selected features).

        Raises:
            ValueError: If selector has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")

        coef = self.model_.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)

        return coef

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation with key parameters.
        """
        params = [f"alpha={self.alpha}", f"task='{self.task}'"]
        if self.n_features_to_select is not None:
            params.append(f"n_features_to_select={self.n_features_to_select}")

        return f"LassoSelector({', '.join(params)})"
