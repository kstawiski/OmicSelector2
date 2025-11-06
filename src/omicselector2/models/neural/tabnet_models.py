"""TabNet models for tabular omics data.

TabNet is a deep learning architecture specifically designed for tabular data,
using sequential attention to select features at each decision step. This makes
it particularly suitable for high-dimensional omics data where feature selection
and interpretability are crucial.

Key features:
- Built-in feature selection through attention mechanisms
- Interpretable via feature importance masks
- No extensive preprocessing required
- Handles high-dimensional data effectively

Reference:
    Arik, S.Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable
    Tabular Learning. AAAI 2021.

Examples:
    >>> from omicselector2.models.neural.tabnet_models import TabNetClassifier
    >>>
    >>> # Classification
    >>> model = TabNetClassifier(max_epochs=100, batch_size=256)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>>
    >>> # Get feature importances
    >>> importance = model.get_feature_importance()
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pytorch_tabnet.tab_model import TabNetClassifier as PyTorchTabNetClassifier
from pytorch_tabnet.tab_model import TabNetRegressor as PyTorchTabNetRegressor

from omicselector2.models.base import BaseClassifier, BaseRegressor


class TabNetClassifier(BaseClassifier):
    """TabNet classifier for binary and multi-class classification.

    Wrapper around pytorch-tabnet's TabNetClassifier with OmicSelector2 interface.
    Uses attention mechanisms for interpretable feature selection.

    Attributes:
        n_d: Width of the decision prediction layer (default 8).
        n_a: Width of the attention embedding (default 8).
        n_steps: Number of steps in the architecture (default 3).
        gamma: Coefficient for feature reusage in the masks (default 1.3).
        max_epochs: Maximum training epochs (default 100).
        batch_size: Training batch size (default 256).
        patience: Early stopping patience (default 10).
        verbose: Print training progress (default False).

    Examples:
        >>> model = TabNetClassifier(
        ...     n_d=16,
        ...     n_a=16,
        ...     n_steps=5,
        ...     max_epochs=200
        ... )
        >>> model.fit(X_train, y_train)
        >>> probas = model.predict_proba(X_test)
    """

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        max_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = False,
    ) -> None:
        """Initialize TabNetClassifier.

        Args:
            n_d: Width of decision prediction layer. Default 8.
            n_a: Width of attention embedding. Default 8.
            n_steps: Number of steps in architecture. Default 3.
            gamma: Coefficient for feature reusage. Default 1.3.
            max_epochs: Maximum training epochs. Default 100.
            batch_size: Training batch size. Default 256.
            patience: Early stopping patience. Default 10.
            verbose: Print training progress. Default False.
        """
        super().__init__()

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        # Will be set during fit
        self.model_: Optional[PyTorchTabNetClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabNetClassifier":
        """Fit TabNet model to training data.

        Args:
            X: Training features (samples × features).
            y: Training target.

        Returns:
            Self (for method chaining).
        """
        start_time = time.time()

        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y.values

        # Store classes and feature names
        self.classes_ = np.unique(y_array)
        self.feature_names_ = X.columns.tolist()

        # Create TabNet model
        self.model_ = PyTorchTabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            verbose=1 if self.verbose else 0,
        )

        # Fit model
        self.model_.fit(
            X_train=X_array,
            y_train=y_array,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            patience=self.patience,
        )

        self.is_fitted_ = True

        # Store metadata
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["n_samples"] = len(X)
        self.metadata["n_features"] = X.shape[1]
        self.metadata["n_classes"] = len(self.classes_)

        return self

    def predict(self, X: pd.DataFrame) -> NDArray:
        """Predict class labels.

        Args:
            X: Features to predict.

        Returns:
            Predicted class labels.

        Raises:
            RuntimeError: If model not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict()")

        X_array = X.values.astype(np.float32)
        predictions = self.model_.predict(X_array)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> NDArray:
        """Predict class probabilities.

        Args:
            X: Features to predict.

        Returns:
            Class probabilities (samples × classes).

        Raises:
            RuntimeError: If model not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_proba()")

        X_array = X.values.astype(np.float32)
        probas = self.model_.predict_proba(X_array)

        return probas

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance via attention masks.

        TabNet provides interpretable feature importance through its
        attention mechanism, showing which features were most influential
        in the predictions.

        Returns:
            Series of feature importance scores (normalized to sum to 1).

        Raises:
            RuntimeError: If model not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before getting feature importance")

        # Get feature importances from attention masks
        importance = self.model_.feature_importances_

        # Convert to pandas Series with feature names
        importance_series = pd.Series(importance, index=self.feature_names_)

        return importance_series


class TabNetRegressor(BaseRegressor):
    """TabNet regressor for continuous outcomes.

    Wrapper around pytorch-tabnet's TabNetRegressor with OmicSelector2 interface.
    Uses attention mechanisms for interpretable feature selection.

    Attributes:
        n_d: Width of the decision prediction layer (default 8).
        n_a: Width of the attention embedding (default 8).
        n_steps: Number of steps in the architecture (default 3).
        gamma: Coefficient for feature reusage in the masks (default 1.3).
        max_epochs: Maximum training epochs (default 100).
        batch_size: Training batch size (default 256).
        patience: Early stopping patience (default 10).
        verbose: Print training progress (default False).

    Examples:
        >>> model = TabNetRegressor(max_epochs=200, batch_size=512)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        max_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = False,
    ) -> None:
        """Initialize TabNetRegressor.

        Args:
            n_d: Width of decision prediction layer. Default 8.
            n_a: Width of attention embedding. Default 8.
            n_steps: Number of steps in architecture. Default 3.
            gamma: Coefficient for feature reusage. Default 1.3.
            max_epochs: Maximum training epochs. Default 100.
            batch_size: Training batch size. Default 256.
            patience: Early stopping patience. Default 10.
            verbose: Print training progress. Default False.
        """
        super().__init__()

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        # Will be set during fit
        self.model_: Optional[PyTorchTabNetRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabNetRegressor":
        """Fit TabNet model to training data.

        Args:
            X: Training features (samples × features).
            y: Training target.

        Returns:
            Self (for method chaining).
        """
        start_time = time.time()

        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y.values.reshape(-1, 1).astype(np.float32)

        # Store feature names
        self.feature_names_ = X.columns.tolist()

        # Create TabNet model
        self.model_ = PyTorchTabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            verbose=1 if self.verbose else 0,
        )

        # Fit model
        self.model_.fit(
            X_train=X_array,
            y_train=y_array,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            patience=self.patience,
        )

        self.is_fitted_ = True

        # Store metadata
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["n_samples"] = len(X)
        self.metadata["n_features"] = X.shape[1]

        return self

    def predict(self, X: pd.DataFrame) -> NDArray:
        """Predict continuous values.

        Args:
            X: Features to predict.

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If model not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict()")

        X_array = X.values.astype(np.float32)
        predictions = self.model_.predict(X_array)

        # Flatten to 1D array
        return predictions.ravel()

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance via attention masks.

        Returns:
            Series of feature importance scores (normalized to sum to 1).

        Raises:
            RuntimeError: If model not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before getting feature importance")

        # Get feature importances from attention masks
        importance = self.model_.feature_importances_

        # Convert to pandas Series with feature names
        importance_series = pd.Series(importance, index=self.feature_names_)

        return importance_series
