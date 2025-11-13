"""Base classes for all models in OmicSelector2.

This module provides abstract base classes that define the interface for all models:
- BaseModel: Core interface for all models
- BaseClassifier: Interface for classification models
- BaseRegressor: Interface for regression models

All models must implement fit() and predict() methods.
Classifiers must additionally implement predict_proba().

Features:
- Consistent API across all model types
- Metadata tracking (training time, parameters, etc.)
- Model persistence (save/load with pickle)
- Type hints for all methods
- Abstract base class enforcement

Examples:
    >>> from omicselector2.models.base import BaseClassifier
    >>>
    >>> class MyClassifier(BaseClassifier):
    ...     def fit(self, X, y):
    ...         # Training logic
    ...         self.is_fitted_ = True
    ...         return self
    ...
    ...     def predict(self, X):
    ...         # Prediction logic
    ...         return predictions
    ...
    ...     def predict_proba(self, X):
    ...         # Probability prediction logic
    ...         return probabilities
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.typing import NDArray


class BaseModel(ABC):
    """Abstract base class for all models.

    All models must inherit from this class and implement:
    - fit(X, y): Train the model
    - predict(X): Make predictions

    Attributes:
        metadata: Dictionary storing model metadata (training time, params, etc.).
        is_fitted_: Boolean indicating whether model has been fitted.

    Examples:
        >>> class MyModel(BaseModel):
        ...     def fit(self, X, y):
        ...         self.is_fitted_ = True
        ...         return self
        ...
        ...     def predict(self, X):
        ...         return np.zeros(len(X))
    """

    def __init__(self) -> None:
        """Initialize base model."""
        self.metadata: dict[str, Any] = {}
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model.

        Args:
            X: Training features (samples × features).
            y: Training target.

        Returns:
            Self (for method chaining).
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> NDArray:
        """Make predictions.

        Args:
            X: Features to predict on (samples × features).

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        pass

    def save(self, path: Path) -> None:
        """Save model to disk using pickle.

        Args:
            path: Path to save model to.

        Examples:
            >>> model.save(Path("my_model.pkl"))
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk.

        Args:
            path: Path to load model from.

        Returns:
            Loaded model instance.

        Examples:
            >>> model = MyModel.load(Path("my_model.pkl"))
        """
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    def _check_is_fitted(self) -> None:
        """Check if model has been fitted.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before calling predict()")


class BaseClassifier(BaseModel):
    """Abstract base class for classification models.

    Classifiers must implement:
    - fit(X, y): Train the classifier
    - predict(X): Predict class labels
    - predict_proba(X): Predict class probabilities

    Examples:
        >>> class MyClassifier(BaseClassifier):
        ...     def fit(self, X, y):
        ...         self.classes_ = np.unique(y)
        ...         self.is_fitted_ = True
        ...         return self
        ...
        ...     def predict(self, X):
        ...         return np.zeros(len(X), dtype=int)
        ...
        ...     def predict_proba(self, X):
        ...         n_classes = len(self.classes_)
        ...         return np.ones((len(X), n_classes)) / n_classes
    """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> NDArray:
        """Predict class probabilities.

        Args:
            X: Features to predict on (samples × features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.
            Each row sums to 1.0.

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        pass


class BaseRegressor(BaseModel):
    """Abstract base class for regression models.

    Regressors must implement:
    - fit(X, y): Train the regressor
    - predict(X): Predict continuous values

    Examples:
        >>> class MyRegressor(BaseRegressor):
        ...     def fit(self, X, y):
        ...         self.mean_ = y.mean()
        ...         self.is_fitted_ = True
        ...         return self
        ...
        ...     def predict(self, X):
        ...         return np.full(len(X), self.mean_)
    """

    pass
