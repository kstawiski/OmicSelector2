"""Elastic Net (L1 + L2 regularization) feature selector.

This module implements feature selection using Elastic Net regression, which combines
L1 and L2 regularization to handle correlated features better than pure Lasso.

Elastic Net is particularly useful for genomics data where features are often correlated.

Examples:
    >>> from omicselector2.features.classical.elastic_net import ElasticNetSelector
    >>> selector = ElasticNetSelector(alpha=0.1, l1_ratio=0.5)
    >>> selector.fit(X_train, y_train)
    >>> selected_features = selector.get_feature_names_out()
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from omicselector2.features.base import BaseFeatureSelector


class ElasticNetSelector(BaseFeatureSelector):
    """Feature selector using Elastic Net (L1 + L2 regularization).

    Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties, providing a balance
    between feature selection and handling correlated features.

    l1_ratio controls the mix: 1.0 = pure Lasso, 0.0 = pure Ridge.
    Recommended values: 0.5 to 0.9 for feature selection.

    Attributes:
        alpha: Overall regularization strength.
        l1_ratio: Mix of L1 vs L2 (0 to 1). Higher = more L1.
        task: Type of task - 'regression' or 'classification'.
        cv: Number of CV folds for automatic parameter selection.
        alpha_: Actual alpha used (set after fit if alpha='auto').
        l1_ratio_: Actual l1_ratio used (set after fit if l1_ratio='auto').

    Examples:
        >>> # Balanced L1/L2 for correlated features
        >>> selector = ElasticNetSelector(alpha=0.01, l1_ratio=0.7)
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Auto-select parameters
        >>> selector = ElasticNetSelector(alpha='auto', l1_ratio='auto', cv=5)
        >>> result = selector.fit(X, y).get_result()
    """

    def __init__(
        self,
        alpha: Union[float, Literal["auto"]] = 1.0,
        l1_ratio: Union[float, Literal["auto"]] = 0.5,
        task: Literal["regression", "classification"] = "regression",
        cv: int = 5,
        max_iter: int = 10000,
        tol: float = 1e-4,
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        standardize: bool = True,
    ) -> None:
        """Initialize Elastic Net feature selector.

        Args:
            alpha: Regularization strength. Use 'auto' for CV selection.
            l1_ratio: L1 vs L2 mix (0-1). Use 'auto' for CV selection.
            task: 'regression' or 'classification'.
            cv: Number of CV folds if alpha/l1_ratio='auto'.
            max_iter: Maximum iterations for optimization.
            tol: Convergence tolerance.
            n_features_to_select: Max features to select.
            random_state: Random seed.
            verbose: Print progress.
            standardize: Standardize features before fitting.

        Raises:
            ValueError: If alpha is not positive or l1_ratio not in [0, 1].
        """
        super().__init__(
            n_features_to_select=n_features_to_select, random_state=random_state, verbose=verbose
        )

        if isinstance(alpha, (int, float)) and alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        if isinstance(l1_ratio, (int, float)) and not 0 <= l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")

        if task not in ["regression", "classification"]:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.task = task
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize

        self.model_: Optional[Union[ElasticNet, LogisticRegression]] = None
        self.scaler_: Optional[StandardScaler] = None
        self.alpha_: Optional[float] = None
        self.l1_ratio_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetSelector":
        """Fit Elastic Net selector to data.

        Args:
            X: Feature matrix.
            y: Target variable.

        Returns:
            Self for method chaining.
        """
        self._validate_input(X, y)
        self._set_feature_metadata(X)

        X_scaled = X.copy()
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_scaled = pd.DataFrame(self.scaler_.fit_transform(X), columns=X.columns, index=X.index)

        if self.task == "regression":
            self._fit_regression(X_scaled, y)
        else:
            self._fit_classification(X_scaled, y)

        self._select_features(X)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features with Elastic Net")

        return self

    def _fit_regression(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Elastic Net for regression."""
        if self.alpha == "auto" or self.l1_ratio == "auto":
            # Use cross-validation
            l1_ratios = (
                [0.1, 0.5, 0.7, 0.9, 0.95, 0.99] if self.l1_ratio == "auto" else [self.l1_ratio]
            )

            self.model_ = ElasticNetCV(
                l1_ratio=l1_ratios,
                cv=self.cv,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model_.fit(X, y)
            self.alpha_ = self.model_.alpha_
            self.l1_ratio_ = self.model_.l1_ratio_
        else:
            self.model_ = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            self.model_.fit(X, y)
            self.alpha_ = self.alpha
            self.l1_ratio_ = self.l1_ratio

    def _fit_classification(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Elastic Net for classification (Logistic with elasticnet penalty)."""
        if self.alpha == "auto":
            from sklearn.model_selection import cross_val_score

            C_values = np.logspace(-4, 4, 20)
            l1_ratio = 0.5 if self.l1_ratio == "auto" else self.l1_ratio

            best_score = -np.inf
            best_C = 1.0

            for C in C_values:
                model = LogisticRegression(
                    penalty="elasticnet",
                    C=C,
                    l1_ratio=l1_ratio,
                    solver="saga",
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                scores = cross_val_score(model, X, y, cv=self.cv, n_jobs=-1)

                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_C = C

            self.model_ = LogisticRegression(
                penalty="elasticnet",
                C=best_C,
                l1_ratio=l1_ratio,
                solver="saga",
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            self.model_.fit(X, y)
            self.alpha_ = 1.0 / best_C
            self.l1_ratio_ = l1_ratio
        else:
            C = 1.0 / self.alpha
            l1_ratio = 0.5 if self.l1_ratio == "auto" else self.l1_ratio

            self.model_ = LogisticRegression(
                penalty="elasticnet",
                C=C,
                l1_ratio=l1_ratio,
                solver="saga",
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            self.model_.fit(X, y)
            self.alpha_ = self.alpha
            self.l1_ratio_ = l1_ratio

    def _select_features(self, X: pd.DataFrame) -> None:
        """Select features based on non-zero coefficients."""
        coef = self.model_.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        else:
            coef = np.abs(coef)

        self.support_mask_ = coef > 1e-10
        selected_indices = np.where(self.support_mask_)[0]
        self.feature_scores_ = coef[selected_indices]

        sorted_indices = np.argsort(-self.feature_scores_)
        selected_indices = selected_indices[sorted_indices]
        self.feature_scores_ = self.feature_scores_[sorted_indices]

        if self.n_features_to_select is not None:
            n_select = min(self.n_features_to_select, len(selected_indices))
            selected_indices = selected_indices[:n_select]
            self.feature_scores_ = self.feature_scores_[:n_select]

            new_mask = np.zeros(len(self.support_mask_), dtype=bool)
            new_mask[selected_indices] = True
            self.support_mask_ = new_mask

        self.selected_features_ = [X.columns[i] for i in selected_indices]

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
