"""Base classes for feature selection methods.

This module provides abstract base classes and utilities for implementing
feature selection methods in OmicSelector2.

Examples:
    >>> from omicselector2.features.base import BaseFeatureSelector
    >>> # Implement a custom selector by inheriting from BaseFeatureSelector
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class FeatureSelectorResult:
    """Result of feature selection operation.

    This dataclass encapsulates the output of a feature selection method,
    including selected features, their scores, and metadata.

    Attributes:
        selected_features: List of selected feature names.
        feature_scores: Array of importance/relevance scores for selected features.
        support_mask: Boolean mask indicating which features were selected.
        n_features_selected: Number of features selected.
        method_name: Name of the feature selection method used.
        metadata: Optional dictionary with additional method-specific information.

    Examples:
        >>> result = FeatureSelectorResult(
        ...     selected_features=["gene_1", "gene_2"],
        ...     feature_scores=np.array([0.9, 0.8]),
        ...     support_mask=np.array([True, True, False]),
        ...     n_features_selected=2,
        ...     method_name="lasso"
        ... )
        >>> df = result.to_dataframe()
        >>> print(df)
    """

    selected_features: list[str]
    feature_scores: NDArray[np.float64]
    support_mask: NDArray[np.bool_]
    n_features_selected: int
    method_name: str
    metadata: Optional[dict] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to pandas DataFrame.

        Returns:
            DataFrame with columns 'feature' and 'score', sorted by score descending.

        Examples:
            >>> result = FeatureSelectorResult(...)
            >>> df = result.to_dataframe()
            >>> print(df.head())
        """
        df = pd.DataFrame({"feature": self.selected_features, "score": self.feature_scores})
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "selected_features": self.selected_features,
            "n_features_selected": self.n_features_selected,
            "method_name": self.method_name,
            "metadata": self.metadata or {},
        }


class BaseFeatureSelector(ABC):
    """Abstract base class for all feature selectors.

    All feature selection methods in OmicSelector2 should inherit from this class
    and implement the required abstract methods.

    The base class provides a consistent interface following scikit-learn's
    transformer pattern with additional bioinformatics-specific functionality.

    Attributes:
        n_features_to_select: Number of features to select. If None, select all relevant.
        random_state: Random seed for reproducibility.
        verbose: Whether to print progress information.

    Examples:
        >>> class MySelector(BaseFeatureSelector):
        ...     def fit(self, X, y):
        ...         # Implementation
        ...         return self
        ...     def transform(self, X):
        ...         # Implementation
        ...         return X[self.selected_features_]
        ...     def get_support(self, indices=False):
        ...         # Implementation
        ...         return self.support_mask_
    """

    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize base feature selector.

        Args:
            n_features_to_select: Number of features to select. None means auto-determine.
            random_state: Random seed for reproducibility.
            verbose: If True, print progress information.
        """
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.verbose = verbose

        # Attributes set during fit
        self.selected_features_: Optional[list[str]] = None
        self.feature_scores_: Optional[NDArray[np.float64]] = None
        self.support_mask_: Optional[NDArray[np.bool_]] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[list[str]] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseFeatureSelector":
        """Fit the feature selector to data.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target variable with shape (n_samples,).

        Returns:
            Self for method chaining.

        Raises:
            NotImplementedError: If method is not implemented in subclass.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Transformed feature matrix with selected features only.

        Raises:
            NotImplementedError: If method is not implemented in subclass.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit to data, then transform it.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target variable with shape (n_samples,).

        Returns:
            Transformed feature matrix with selected features only.
        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_support(self, indices: bool = False) -> NDArray:
        """Get mask or indices of selected features.

        Args:
            indices: If True, return feature indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected features.

        Raises:
            NotImplementedError: If method is not implemented in subclass.
        """
        pass

    def get_feature_names_out(self) -> list[str]:
        """Get names of selected features.

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If selector has not been fitted yet.

        Examples:
            >>> selector.fit(X, y)
            >>> selected = selector.get_feature_names_out()
            >>> print(f"Selected {len(selected)} features")
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        return self.selected_features_

    def get_feature_scores(self) -> NDArray[np.float64]:
        """Get importance/relevance scores for selected features.

        Returns:
            Array of feature scores.

        Raises:
            ValueError: If selector has not been fitted yet.
        """
        if self.feature_scores_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")
        return self.feature_scores_

    def get_result(self) -> FeatureSelectorResult:
        """Get complete feature selection result.

        Returns:
            FeatureSelectorResult object with all information.

        Raises:
            ValueError: If selector has not been fitted yet.
        """
        if self.selected_features_ is None or self.feature_scores_ is None:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")

        return FeatureSelectorResult(
            selected_features=self.selected_features_,
            feature_scores=self.feature_scores_,
            support_mask=self.support_mask_,
            n_features_selected=len(self.selected_features_),
            method_name=self.__class__.__name__,
        )

    def _validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Validate input data.

        Args:
            X: Feature matrix to validate.
            y: Optional target variable to validate.

        Raises:
            ValueError: If input data is invalid.
            TypeError: If input types are incorrect.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

        if y is not None and not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y)}")

        if X.empty:
            raise ValueError("X cannot be empty")

        if y is not None and len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of samples. " f"Got X: {len(X)}, y: {len(y)}"
            )

        # Check for NaN values
        if X.isnull().any().any():
            raise ValueError("X contains NaN values. Please handle missing data first.")

        if y is not None and y.isnull().any():
            raise ValueError("y contains NaN values. Please handle missing data first.")

    def _set_feature_metadata(self, X: pd.DataFrame) -> None:
        """Set metadata about features from input DataFrame.

        Args:
            X: Feature matrix.
        """
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

    def __repr__(self) -> str:
        """Get string representation of selector.

        Returns:
            String representation with key parameters.
        """
        params = []
        if self.n_features_to_select is not None:
            params.append(f"n_features_to_select={self.n_features_to_select}")
        if self.random_state is not None:
            params.append(f"random_state={self.random_state}")

        return f"{self.__class__.__name__}({', '.join(params)})"
