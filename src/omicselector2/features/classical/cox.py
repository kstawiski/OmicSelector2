"""Cox Proportional Hazards feature selector for survival analysis.

This module implements feature selection using Cox Proportional Hazards regression,
the gold standard statistical model for survival analysis in cancer research.

Cox model: λ(t|X) = λ₀(t) exp(β₁X₁ + β₂X₂ + ... + βₚXₚ)

Features are ranked by the absolute value of their regression coefficients,
which represent hazard ratios. Larger |β| indicates stronger prognostic value.

Examples:
    >>> from omicselector2.features.classical.cox import CoxSelector
    >>>
    >>> # Basic usage
    >>> selector = CoxSelector(n_features_to_select=20)
    >>> selector.fit(X_train, y_survival)  # y has 'time' and 'event' columns
    >>> X_selected = selector.transform(X_train)
    >>>
    >>> # With L1 penalty for sparse selection
    >>> selector = CoxSelector(penalty="l1", penalizer=0.1, n_features_to_select=15)
    >>> selector.fit(X_train, y_survival)
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from numpy.typing import NDArray

from omicselector2.features.base import BaseFeatureSelector


class CoxSelector(BaseFeatureSelector):
    """Cox Proportional Hazards feature selector.

    Uses Cox regression to identify features (genes) with strongest prognostic
    value for survival outcomes. Features are ranked by coefficient magnitude.

    Attributes:
        n_features_to_select: Number of top features to select. Default 10.
        penalty: Regularization penalty ('l1', 'l2', or None). Default None.
        penalizer: Regularization strength (lambda). Default 0.1.
        verbose: Print progress messages.

    Examples:
        >>> # Standard Cox regression
        >>> selector = CoxSelector(n_features_to_select=30)
        >>> selector.fit(X, y_survival)
        >>>
        >>> # Cox with L1 penalty (Lasso) for high-dimensional data
        >>> selector = CoxSelector(
        ...     penalty="l1",
        ...     penalizer=0.05,
        ...     n_features_to_select=20
        ... )
        >>> selector.fit(X, y_survival)
    """

    def __init__(
        self,
        n_features_to_select: int = 10,
        penalty: Optional[Literal["l1", "l2"]] = None,
        penalizer: float = 0.1,
        verbose: bool = False,
    ) -> None:
        """Initialize CoxSelector.

        Args:
            n_features_to_select: Number of features to select. Default 10.
            penalty: Regularization ('l1', 'l2', or None). Default None.
            penalizer: Regularization strength. Default 0.1.
            verbose: Print progress. Default False.

        Raises:
            ValueError: If n_features_to_select <= 0 or invalid penalty.
        """
        super().__init__(
            n_features_to_select=n_features_to_select, verbose=verbose
        )

        if n_features_to_select <= 0:
            raise ValueError("n_features_to_select must be positive")

        if penalty not in [None, "l1", "l2"]:
            raise ValueError("penalty must be None, 'l1', or 'l2'")

        self.penalty = penalty
        self.penalizer = penalizer

        # Attributes set during fit
        self.model_: Optional[CoxPHFitter] = None

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> "CoxSelector":
        """Fit Cox regression model and select top features.

        Args:
            X: Training features (samples × features).
            y: Survival data with 'time' and 'event' columns.
                - time: Survival/censoring time (numeric)
                - event: Event indicator (1=event, 0=censored)

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If y doesn't have 'time' and 'event' columns.
        """
        # Validate input
        if not isinstance(y, pd.DataFrame):
            raise ValueError("y must be a DataFrame with 'time' and 'event' columns")

        if "time" not in y.columns or "event" not in y.columns:
            raise ValueError("y must have 'time' and 'event' columns")

        # Set feature metadata
        self._set_feature_metadata(X)

        # Combine X and y for lifelines
        data = pd.concat([X, y[["time", "event"]]], axis=1)

        # Create Cox model
        self.model_ = CoxPHFitter(penalizer=self.penalizer)

        # Fit with appropriate penalty
        if self.penalty == "l1":
            self.model_.fit(
                data,
                duration_col="time",
                event_col="event",
                l1_ratio=1.0,  # Pure L1
            )
        elif self.penalty == "l2":
            self.model_.fit(
                data,
                duration_col="time",
                event_col="event",
                l1_ratio=0.0,  # Pure L2
            )
        else:
            self.model_.fit(
                data, duration_col="time", event_col="event"
            )

        # Extract coefficients
        coefficients = self.model_.params_

        # Rank features by absolute coefficient
        feature_importance = np.abs(coefficients.values)
        feature_names = coefficients.index.tolist()

        # Sort by importance (descending)
        sorted_indices = np.argsort(-feature_importance)

        # Select top n features
        n_select = min(self.n_features_to_select, len(feature_names))
        selected_indices = sorted_indices[:n_select]

        self.selected_features_ = [feature_names[i] for i in selected_indices]
        self.feature_scores_ = feature_importance[selected_indices]

        # Create support mask
        self.support_mask_ = np.isin(feature_names, self.selected_features_)

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features using Cox PH")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features.

        Args:
            X: Feature matrix (samples × features).

        Returns:
            Transformed DataFrame with selected features only.

        Raises:
            RuntimeError: If selector not fitted.
            ValueError: If X has different features than training data.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector must be fitted before transform()")

        # Check that X has the same features as training data
        if not all(feat in X.columns for feat in self.selected_features_):
            raise ValueError(
                "X has different features than the training data. "
                "Ensure X contains all selected features."
            )

        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> NDArray:
        """Get mask or indices of selected features.

        Args:
            indices: If True, return feature indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected features.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if self.support_mask_ is None:
            raise RuntimeError("Selector must be fitted before get_support()")

        if indices:
            return np.where(self.support_mask_)[0]
        else:
            return self.support_mask_

    def get_hazard_ratios(self) -> pd.Series:
        """Get hazard ratios (exp(coefficients)) for selected features.

        Hazard ratio > 1: Feature increases hazard (poor prognosis)
        Hazard ratio < 1: Feature decreases hazard (good prognosis)

        Returns:
            Series of hazard ratios for selected features.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if self.model_ is None:
            raise RuntimeError("Selector must be fitted before getting hazard ratios")

        # exp(β) = hazard ratio
        hazard_ratios = np.exp(self.model_.params_)

        return hazard_ratios[self.selected_features_]
