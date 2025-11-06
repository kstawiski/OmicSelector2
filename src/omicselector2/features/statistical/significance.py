"""Statistical significance-based feature selection.

Implements OmicSelector 1.0's statistical testing methods:
- sig: All significant features (BH correction, p <= alpha)
- sigtop: Top N significant features by p-value
- sigtopBonf: Top N with Bonferroni correction
- sigtopHolm: Top N with Holm-Bonferroni correction
- topFC: Top N by absolute fold-change

Based on unpaired t-test for binary classification with various
multiple testing correction methods.

Examples:
    >>> from omicselector2.features.statistical import SignificanceSelector
    >>>
    >>> # Select all significant features (BH correction)
    >>> selector = SignificanceSelector(method="sig", alpha=0.05)
    >>> selector.fit(X_train, y_train)
    >>> significant_genes = selector.selected_features_
    >>>
    >>> # Select top 20 by p-value with Bonferroni correction
    >>> selector = SignificanceSelector(
    ...     method="sigtopBonf",
    ...     n_features_to_select=20,
    ...     alpha=0.05
    ... )
    >>> selector.fit(X_train, y_train)
    >>> top_genes = selector.selected_features_
    >>>
    >>> # Get p-values and fold-changes
    >>> p_values = selector.get_p_values()
    >>> fold_changes = selector.get_fold_changes()
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from omicselector2.features.base import BaseFeatureSelector


def _multipletests_bh(pvalues: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction (scipy-only implementation).

    Args:
        pvalues: Array of p-values.
        alpha: Significance threshold.

    Returns:
        Tuple of (rejected, p_corrected):
        - rejected: Boolean array indicating rejected null hypotheses
        - p_corrected: Array of corrected p-values
    """
    n = len(pvalues)

    # Sort p-values and remember original order
    sort_idx = np.argsort(pvalues)
    pvalues_sorted = pvalues[sort_idx]

    # BH correction: p_corrected[i] = p[i] * n / (i+1)
    p_corrected_sorted = pvalues_sorted * n / np.arange(1, n + 1)

    # Ensure monotonicity (going backwards)
    for i in range(n - 2, -1, -1):
        if p_corrected_sorted[i] > p_corrected_sorted[i + 1]:
            p_corrected_sorted[i] = p_corrected_sorted[i + 1]

    # Cap at 1.0
    p_corrected_sorted = np.minimum(p_corrected_sorted, 1.0)

    # Unsort to original order
    p_corrected = np.empty(n)
    p_corrected[sort_idx] = p_corrected_sorted

    # Reject null hypotheses where corrected p-value <= alpha
    rejected = p_corrected <= alpha

    return rejected, p_corrected


def _multipletests_bonferroni(pvalues: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni correction (scipy-only implementation).

    Args:
        pvalues: Array of p-values.
        alpha: Significance threshold.

    Returns:
        Tuple of (rejected, p_corrected):
        - rejected: Boolean array indicating rejected null hypotheses
        - p_corrected: Array of corrected p-values
    """
    n = len(pvalues)

    # Bonferroni correction: p_corrected = p * n
    p_corrected = pvalues * n
    p_corrected = np.minimum(p_corrected, 1.0)  # Cap at 1.0

    # Reject null hypotheses where corrected p-value <= alpha
    rejected = p_corrected <= alpha

    return rejected, p_corrected


def _multipletests_holm(pvalues: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni correction (scipy-only implementation).

    Args:
        pvalues: Array of p-values.
        alpha: Significance threshold.

    Returns:
        Tuple of (rejected, p_corrected):
        - rejected: Boolean array indicating rejected null hypotheses
        - p_corrected: Array of corrected p-values
    """
    n = len(pvalues)

    # Sort p-values and remember original order
    sort_idx = np.argsort(pvalues)
    pvalues_sorted = pvalues[sort_idx]

    # Holm correction: p_corrected[i] = p[i] * (n - i)
    p_corrected_sorted = pvalues_sorted * (n - np.arange(n))

    # Ensure monotonicity (going forward)
    for i in range(1, n):
        if p_corrected_sorted[i] < p_corrected_sorted[i - 1]:
            p_corrected_sorted[i] = p_corrected_sorted[i - 1]

    # Cap at 1.0
    p_corrected_sorted = np.minimum(p_corrected_sorted, 1.0)

    # Unsort to original order
    p_corrected = np.empty(n)
    p_corrected[sort_idx] = p_corrected_sorted

    # Reject null hypotheses where corrected p-value <= alpha
    rejected = p_corrected <= alpha

    return rejected, p_corrected


class SignificanceSelector(BaseFeatureSelector):
    """Statistical significance-based feature selector using t-test.

    Performs unpaired t-test to identify features differentially expressed
    between two classes, with various multiple testing corrections.

    Attributes:
        method: Selection method:
            - "sig": All significant features (p <= alpha after correction)
            - "sigtop": Top N significant by p-value
            - "sigtopBonf": Top N with Bonferroni correction
            - "sigtopHolm": Top N with Holm-Bonferroni correction
        alpha: Significance threshold. Default 0.05.
        n_features_to_select: Number of features for "sigtop*" methods.
        fc_threshold: Optional fold-change threshold (log2 scale).
        verbose: Print progress messages.

    Examples:
        >>> # All significant features
        >>> selector = SignificanceSelector(method="sig", alpha=0.05)
        >>> selector.fit(X, y)
        >>>
        >>> # Top 20 with Bonferroni correction
        >>> selector = SignificanceSelector(
        ...     method="sigtopBonf",
        ...     n_features_to_select=20
        ... )
        >>> selector.fit(X, y)
        >>>
        >>> # Significant + |log2FC| > 1 (fcsig method)
        >>> selector = SignificanceSelector(
        ...     method="sig",
        ...     alpha=0.05,
        ...     fc_threshold=1.0
        ... )
        >>> selector.fit(X, y)
    """

    VALID_METHODS = ["sig", "sigtop", "sigtopBonf", "sigtopHolm"]

    def __init__(
        self,
        method: Literal["sig", "sigtop", "sigtopBonf", "sigtopHolm"] = "sig",
        alpha: float = 0.05,
        n_features_to_select: Optional[int] = None,
        fc_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize SignificanceSelector.

        Args:
            method: Selection method. Default "sig".
            alpha: Significance threshold. Default 0.05.
            n_features_to_select: Number of features for "sigtop*". Required for those methods.
            fc_threshold: Optional fold-change threshold (log2 scale). Default None.
            verbose: Print progress. Default False.

        Raises:
            ValueError: If method is invalid or n_features_to_select not provided for sigtop*.
        """
        super().__init__(verbose=verbose)

        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got '{method}'"
            )

        if method.startswith("sigtop") and n_features_to_select is None:
            raise ValueError(
                f"n_features_to_select must be provided for method '{method}'"
            )

        self.method = method
        self.alpha = alpha
        self.n_features_to_select = n_features_to_select
        self.fc_threshold = fc_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SignificanceSelector":
        """Fit selector by performing t-tests.

        Args:
            X: Training features (samples × features).
            y: Training target (binary classification).

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If y is not binary.
        """
        # Validate binary classification
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"SignificanceSelector requires binary classification, "
                f"got {len(classes)} classes"
            )

        # Compute t-tests and p-values
        self.p_values_ = self._compute_p_values(X, y)
        self.fold_changes_ = self._compute_fold_changes(X, y)

        # Apply multiple testing correction
        correction_method = self._get_correction_method()

        if correction_method == "fdr_bh":
            rejected, p_corrected = _multipletests_bh(
                self.p_values_.values, self.alpha
            )
        elif correction_method == "bonferroni":
            rejected, p_corrected = _multipletests_bonferroni(
                self.p_values_.values, self.alpha
            )
        elif correction_method == "holm":
            rejected, p_corrected = _multipletests_holm(
                self.p_values_.values, self.alpha
            )
        else:
            # Default to BH
            rejected, p_corrected = _multipletests_bh(
                self.p_values_.values, self.alpha
            )

        self.p_values_corrected_ = pd.Series(
            p_corrected, index=self.p_values_.index
        )

        # Select features based on method
        if self.method == "sig":
            # All significant features
            selected_mask = rejected

            # Apply fold-change threshold if specified
            if self.fc_threshold is not None:
                fc_mask = np.abs(self.fold_changes_.values) >= self.fc_threshold
                selected_mask = selected_mask & fc_mask

            self.selected_features_ = self.p_values_.index[selected_mask].tolist()

        elif self.method.startswith("sigtop"):
            # Top N significant features
            # Sort by corrected p-value
            sorted_indices = np.argsort(self.p_values_corrected_.values)

            # Take top N that are significant
            n_select = min(
                self.n_features_to_select,
                np.sum(rejected),
                len(X.columns)
            )

            # If not enough significant features, take top N by p-value anyway
            if n_select < self.n_features_to_select:
                n_select = min(self.n_features_to_select, len(X.columns))

            selected_indices = sorted_indices[:n_select]
            self.selected_features_ = self.p_values_.index[selected_indices].tolist()

        # Store feature scores (negative log p-value for ranking)
        self.feature_scores_ = -np.log10(
            self.p_values_corrected_[self.selected_features_].values + 1e-300
        )

        # Create support mask (boolean array indicating which features were selected)
        self.support_mask_ = np.isin(self.p_values_.index, self.selected_features_)

        self.is_fitted_ = True

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features using {self.method}")

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
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before transform()")

        # Check that X has the same features as training data
        if not all(feat in X.columns for feat in self.selected_features_):
            raise ValueError(
                "X has different features than the training data. "
                "Ensure X contains all selected features."
            )

        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features.

        Args:
            indices: If True, return feature indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected features.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before get_support()")

        if indices:
            return np.where(self.support_mask_)[0]
        else:
            return self.support_mask_

    def _compute_p_values(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute p-values using unpaired t-test.

        Args:
            X: Features.
            y: Binary target.

        Returns:
            Series of p-values for each feature.
        """
        classes = np.unique(y)
        class_0_mask = y == classes[0]
        class_1_mask = y == classes[1]

        p_values = []

        for col in X.columns:
            group_0 = X.loc[class_0_mask, col]
            group_1 = X.loc[class_1_mask, col]

            # Unpaired t-test
            t_stat, p_val = stats.ttest_ind(group_0, group_1)

            p_values.append(p_val)

        return pd.Series(p_values, index=X.columns)

    def _compute_fold_changes(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute log2 fold-changes.

        Args:
            X: Features.
            y: Binary target.

        Returns:
            Series of log2 fold-changes for each feature.
        """
        classes = np.unique(y)
        class_0_mask = y == classes[0]
        class_1_mask = y == classes[1]

        # Compute all means first to determine global shift
        means_0 = X.loc[class_0_mask].mean()
        means_1 = X.loc[class_1_mask].mean()

        # Find global minimum to apply consistent shift across all features
        global_min = min(means_0.min(), means_1.min())

        # Apply shift if needed to ensure all values are positive
        if global_min <= 0:
            shift = abs(global_min) + 1.0
        else:
            shift = 0.0

        fold_changes = []

        for col in X.columns:
            mean_0 = means_0[col] + shift
            mean_1 = means_1[col] + shift

            # Add small epsilon to avoid division by zero
            fc = np.log2((mean_1 + 1e-10) / (mean_0 + 1e-10))

            fold_changes.append(fc)

        return pd.Series(fold_changes, index=X.columns)

    def _get_correction_method(self) -> str:
        """Get correction method name.

        Returns:
            Correction method string ('fdr_bh', 'bonferroni', or 'holm').
        """
        if self.method in ["sig", "sigtop"]:
            return "fdr_bh"  # Benjamini-Hochberg (FDR)
        elif self.method == "sigtopBonf":
            return "bonferroni"
        elif self.method == "sigtopHolm":
            return "holm"
        else:
            return "fdr_bh"

    def get_p_values(self) -> pd.Series:
        """Get corrected p-values for all features.

        Returns:
            Series of corrected p-values.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before getting p-values")

        return self.p_values_corrected_

    def get_fold_changes(self) -> pd.Series:
        """Get log2 fold-changes for all features.

        Returns:
            Series of log2 fold-changes for all features.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before getting fold-changes")

        return self.fold_changes_


class FoldChangeSelector(BaseFeatureSelector):
    """Select features by absolute fold-change magnitude.

    Implements the "topFC" method from OmicSelector 1.0: selects top N
    features based on decreasing absolute value of fold-change in
    differential expression analysis.

    Attributes:
        n_features_to_select: Number of features to select.
        verbose: Print progress messages.

    Examples:
        >>> # Select top 20 by |fold-change|
        >>> selector = FoldChangeSelector(n_features_to_select=20)
        >>> selector.fit(X, y)
        >>> top_genes = selector.selected_features_
        >>>
        >>> # Get fold-changes
        >>> fold_changes = selector.get_fold_changes()
    """

    def __init__(
        self,
        n_features_to_select: int,
        verbose: bool = False,
    ) -> None:
        """Initialize FoldChangeSelector.

        Args:
            n_features_to_select: Number of features to select.
            verbose: Print progress. Default False.
        """
        super().__init__(verbose=verbose)
        self.n_features_to_select = n_features_to_select

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FoldChangeSelector":
        """Fit selector by computing fold-changes.

        Args:
            X: Training features (samples × features).
            y: Training target (binary classification).

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If y is not binary.
        """
        # Validate binary classification
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"FoldChangeSelector requires binary classification, "
                f"got {len(classes)} classes"
            )

        # Compute fold-changes
        self.fold_changes_ = self._compute_fold_changes(X, y)

        # Select top N by |fold-change|
        abs_fc = np.abs(self.fold_changes_)
        sorted_indices = np.argsort(-abs_fc.values)  # Descending

        n_select = min(self.n_features_to_select, len(X.columns))
        selected_indices = sorted_indices[:n_select]

        self.selected_features_ = self.fold_changes_.index[selected_indices].tolist()
        self.feature_scores_ = abs_fc.iloc[selected_indices].values

        # Create support mask (boolean array indicating which features were selected)
        self.support_mask_ = np.isin(self.fold_changes_.index, self.selected_features_)

        self.is_fitted_ = True

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features by |fold-change|")

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
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before transform()")

        # Check that X has the same features as training data
        if not all(feat in X.columns for feat in self.selected_features_):
            raise ValueError(
                "X has different features than the training data. "
                "Ensure X contains all selected features."
            )

        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features.

        Args:
            indices: If True, return feature indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected features.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before get_support()")

        if indices:
            return np.where(self.support_mask_)[0]
        else:
            return self.support_mask_

    def _compute_fold_changes(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute log2 fold-changes.

        Args:
            X: Features.
            y: Binary target.

        Returns:
            Series of log2 fold-changes for each feature.
        """
        classes = np.unique(y)
        class_0_mask = y == classes[0]
        class_1_mask = y == classes[1]

        # Compute all means first to determine global shift
        means_0 = X.loc[class_0_mask].mean()
        means_1 = X.loc[class_1_mask].mean()

        # Find global minimum to apply consistent shift across all features
        global_min = min(means_0.min(), means_1.min())

        # Apply shift if needed to ensure all values are positive
        if global_min <= 0:
            shift = abs(global_min) + 1.0
        else:
            shift = 0.0

        fold_changes = []

        for col in X.columns:
            mean_0 = means_0[col] + shift
            mean_1 = means_1[col] + shift

            # Add small epsilon to avoid division by zero
            fc = np.log2((mean_1 + 1e-10) / (mean_0 + 1e-10))

            fold_changes.append(fc)

        return pd.Series(fold_changes, index=X.columns)

    def get_fold_changes(self) -> pd.Series:
        """Get log2 fold-changes for selected features.

        Returns:
            Series of log2 fold-changes for selected features only.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before getting fold-changes")

        return self.fold_changes_[self.selected_features_]
