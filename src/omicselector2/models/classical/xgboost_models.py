"""XGBoost models for classification and regression.

XGBoost (Extreme Gradient Boosting) is a highly efficient implementation of
gradient boosting. It's one of the best-performing algorithms for structured data.

Key advantages for biomarker discovery:
- State-of-the-art performance
- Built-in regularization (L1, L2)
- Feature importance via gain, weight, cover
- Handles missing values
- Fast training with parallel processing

Examples:
    >>> from omicselector2.models.classical import XGBoostClassifier, XGBoostRegressor
    >>>
    >>> # Classification
    >>> model = XGBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    >>>
    >>> # Get feature importance
    >>> importance = model.get_feature_importance()
    >>>
    >>> # Regression
    >>> reg = XGBoostRegressor(n_estimators=100, random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from xgboost import XGBClassifier, XGBRegressor

from omicselector2.models.base import BaseClassifier, BaseRegressor


class XGBoostClassifier(BaseClassifier):
    """XGBoost classifier.

    Wrapper around xgboost's XGBClassifier with OmicSelector2 interface.

    Attributes:
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage (eta).
        max_depth: Maximum tree depth.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of features per tree.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        random_state: Random seed for reproducibility.

    Examples:
        >>> model = XGBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        >>> model.fit(X, y)
        >>> importance = model.get_feature_importance()
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds. Default 100.
            learning_rate: Learning rate (eta). Default 0.1.
            max_depth: Maximum tree depth. Default 6.
            subsample: Subsample ratio. Default 1.0 (no subsampling).
            colsample_bytree: Feature subsample ratio. Default 1.0.
            reg_alpha: L1 regularization. Default 0.0.
            reg_lambda: L2 regularization. Default 1.0.
            random_state: Random seed. Default None.
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.model_ = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            eval_metric="logloss",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostClassifier":
        """Train XGBoost classifier.

        Args:
            X: Training features (samples × features).
            y: Training target (class labels).

        Returns:
            Self (for method chaining).
        """
        import time

        start_time = time.time()

        self.model_.fit(X, y, verbose=False)
        self.classes_ = self.model_.classes_
        self.feature_names_ = X.columns.tolist()

        self.is_fitted_ = True

        # Track metadata
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["n_samples"] = len(X)
        self.metadata["n_features"] = X.shape[1]
        self.metadata["n_classes"] = len(self.classes_)

        # best_iteration only available with early stopping
        try:
            self.metadata["best_iteration"] = self.model_.best_iteration
        except AttributeError:
            self.metadata["best_iteration"] = None

        return self

    def predict(self, X: pd.DataFrame) -> NDArray:
        """Predict class labels.

        Args:
            X: Features to predict on (samples × features).

        Returns:
            Predicted class labels.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> NDArray:
        """Predict class probabilities.

        Args:
            X: Features to predict on (samples × features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()
        return self.model_.predict_proba(X)

    def get_feature_importance(
        self, importance_type: Literal["weight", "gain", "cover"] = "gain"
    ) -> pd.Series:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance metric:
                - "weight": Number of times feature is used in splits
                - "gain": Average gain when feature is used (default)
                - "cover": Average coverage (samples affected by splits)

        Returns:
            Series with feature names as index and importance as values.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()

        importance_dict = self.model_.get_booster().get_score(
            importance_type=importance_type
        )

        # Create series with zeros for all features
        importance = pd.Series(0.0, index=self.feature_names_)

        # Update with non-zero importances
        for feature, score in importance_dict.items():
            # XGBoost uses f0, f1, ... or feature_0, feature_1, ... indexing
            if feature.startswith("f") and feature[1:].isdigit():
                idx = int(feature[1:])
                if idx < len(self.feature_names_):
                    importance.iloc[idx] = score
            elif feature.startswith("feature_"):
                idx = int(feature.split("_")[1])
                if idx < len(self.feature_names_):
                    importance.iloc[idx] = score

        # Normalize to sum to 1
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance


class XGBoostRegressor(BaseRegressor):
    """XGBoost regressor.

    Wrapper around xgboost's XGBRegressor with OmicSelector2 interface.

    Attributes:
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage (eta).
        max_depth: Maximum tree depth.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of features per tree.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        random_state: Random seed for reproducibility.

    Examples:
        >>> model = XGBoostRegressor(n_estimators=100, random_state=42)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize XGBoost regressor.

        Args:
            n_estimators: Number of boosting rounds. Default 100.
            learning_rate: Learning rate (eta). Default 0.1.
            max_depth: Maximum tree depth. Default 6.
            subsample: Subsample ratio. Default 1.0.
            colsample_bytree: Feature subsample ratio. Default 1.0.
            reg_alpha: L1 regularization. Default 0.0.
            reg_lambda: L2 regularization. Default 1.0.
            random_state: Random seed. Default None.
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.model_ = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostRegressor":
        """Train XGBoost regressor.

        Args:
            X: Training features (samples × features).
            y: Training target (continuous values).

        Returns:
            Self (for method chaining).
        """
        import time

        start_time = time.time()

        self.model_.fit(X, y, verbose=False)
        self.feature_names_ = X.columns.tolist()

        self.is_fitted_ = True

        # Track metadata
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["n_samples"] = len(X)
        self.metadata["n_features"] = X.shape[1]

        # best_iteration only available with early stopping
        try:
            self.metadata["best_iteration"] = self.model_.best_iteration
        except AttributeError:
            self.metadata["best_iteration"] = None

        return self

    def predict(self, X: pd.DataFrame) -> NDArray:
        """Predict continuous values.

        Args:
            X: Features to predict on (samples × features).

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()
        return self.model_.predict(X)

    def get_feature_importance(
        self, importance_type: Literal["weight", "gain", "cover"] = "gain"
    ) -> pd.Series:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance metric ("weight", "gain", "cover").

        Returns:
            Series with feature names as index and importance as values.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()

        importance_dict = self.model_.get_booster().get_score(
            importance_type=importance_type
        )

        # Create series with zeros for all features
        importance = pd.Series(0.0, index=self.feature_names_)

        # Update with non-zero importances
        for feature, score in importance_dict.items():
            # XGBoost uses f0, f1, ... or feature_0, feature_1, ... indexing
            if feature.startswith("f") and feature[1:].isdigit():
                idx = int(feature[1:])
                if idx < len(self.feature_names_):
                    importance.iloc[idx] = score
            elif feature.startswith("feature_"):
                idx = int(feature.split("_")[1])
                if idx < len(self.feature_names_):
                    importance.iloc[idx] = score

        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance
