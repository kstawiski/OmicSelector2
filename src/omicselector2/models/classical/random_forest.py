"""Random Forest models for classification and regression.

Random Forests are ensemble methods that build multiple decision trees and
aggregate their predictions. They are robust, interpretable, and perform well
on high-dimensional genomic data.

Key advantages for biomarker discovery:
- Built-in feature importance
- No need for feature scaling
- Handles missing values naturally
- Resistant to overfitting (via bagging)
- Works well with small sample sizes

Examples:
    >>> from omicselector2.models.classical import RandomForestClassifier
    >>>
    >>> # Binary classification
    >>> model = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    >>>
    >>> # Get feature importance
    >>> importance = model.get_feature_importance()
    >>> top_features = importance.nlargest(20)
"""

from typing import Optional

import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor

from omicselector2.models.base import BaseClassifier, BaseRegressor


class RandomForestClassifier(BaseClassifier):
    """Random Forest classifier.

    Wrapper around sklearn's RandomForestClassifier with OmicSelector2 interface.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees (None = unlimited).
        min_samples_split: Minimum samples required to split node.
        min_samples_leaf: Minimum samples required at leaf node.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 = use all cores).

    Examples:
        >>> model = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        """Initialize Random Forest classifier.

        Args:
            n_estimators: Number of trees. Default 100.
            max_depth: Maximum tree depth. Default None (unlimited).
            min_samples_split: Minimum samples to split. Default 2.
            min_samples_leaf: Minimum samples at leaf. Default 1.
            random_state: Random seed. Default None.
            n_jobs: Number of parallel jobs. Default -1 (all cores).
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model_ = SKRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestClassifier":
        """Train Random Forest classifier.

        Args:
            X: Training features (samples × features).
            y: Training target (class labels).

        Returns:
            Self (for method chaining).
        """
        import time

        start_time = time.time()

        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.feature_names_ = X.columns.tolist()

        self.is_fitted_ = True

        # Track metadata
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["n_samples"] = len(X)
        self.metadata["n_features"] = X.shape[1]
        self.metadata["n_classes"] = len(self.classes_)

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

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores.

        Returns:
            Series with feature names as index and importance as values.
            Importance scores sum to 1.0.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()

        importance = self.model_.feature_importances_
        return pd.Series(importance, index=self.feature_names_)


class RandomForestRegressor(BaseRegressor):
    """Random Forest regressor.

    Wrapper around sklearn's RandomForestRegressor with OmicSelector2 interface.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees (None = unlimited).
        min_samples_split: Minimum samples required to split node.
        min_samples_leaf: Minimum samples required at leaf node.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 = use all cores).

    Examples:
        >>> model = RandomForestRegressor(n_estimators=100, random_state=42)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        """Initialize Random Forest regressor.

        Args:
            n_estimators: Number of trees. Default 100.
            max_depth: Maximum tree depth. Default None (unlimited).
            min_samples_split: Minimum samples to split. Default 2.
            min_samples_leaf: Minimum samples at leaf. Default 1.
            random_state: Random seed. Default None.
            n_jobs: Number of parallel jobs. Default -1 (all cores).
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model_ = SKRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestRegressor":
        """Train Random Forest regressor.

        Args:
            X: Training features (samples × features).
            y: Training target (continuous values).

        Returns:
            Self (for method chaining).
        """
        import time

        start_time = time.time()

        self.model_.fit(X, y)
        self.feature_names_ = X.columns.tolist()

        self.is_fitted_ = True

        # Track metadata
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["n_samples"] = len(X)
        self.metadata["n_features"] = X.shape[1]

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

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores.

        Returns:
            Series with feature names as index and importance as values.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()

        importance = self.model_.feature_importances_
        return pd.Series(importance, index=self.feature_names_)
