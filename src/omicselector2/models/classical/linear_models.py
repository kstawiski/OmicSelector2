"""Linear models for classification: Logistic Regression and SVM.

These models are well-suited for high-dimensional genomic data:
- Logistic Regression: Interpretable coefficients, regularization via C parameter
- SVM: Effective in high dimensions, kernel trick for non-linear boundaries

Examples:
    >>> from omicselector2.models.classical import LogisticRegressionModel, SVMClassifier
    >>>
    >>> # Logistic Regression
    >>> model = LogisticRegressionModel(C=1.0, random_state=42)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>>
    >>> # Get coefficients
    >>> coef = model.get_coefficients()
    >>>
    >>> # SVM
    >>> svm = SVMClassifier(C=1.0, kernel="rbf", probability=True)
    >>> svm.fit(X_train, y_train)
    >>> predictions = svm.predict(X_test)
    >>> probabilities = svm.predict_proba(X_test)
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.svm import SVC as SKSVC

from omicselector2.models.base import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    """Logistic Regression classifier.

    Wrapper around sklearn's LogisticRegression with OmicSelector2 interface.

    Logistic regression is a linear model for binary/multi-class classification.
    The C parameter controls regularization strength (smaller values = stronger regularization).

    Attributes:
        C: Inverse regularization strength (smaller = stronger regularization).
        penalty: Regularization type ("l1", "l2", "elasticnet", "none").
        solver: Algorithm to use ("lbfgs", "liblinear", "saga", etc.).
        max_iter: Maximum iterations for solver.
        random_state: Random seed for reproducibility.

    Examples:
        >>> model = LogisticRegressionModel(C=1.0, random_state=42)
        >>> model.fit(X, y)
        >>> coef = model.get_coefficients()
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Logistic Regression classifier.

        Args:
            C: Inverse regularization strength. Default 1.0.
            penalty: Regularization type. Default "l2".
            solver: Algorithm to use. Default "lbfgs".
            max_iter: Maximum iterations. Default 1000.
            random_state: Random seed. Default None.
        """
        super().__init__()

        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        self.model_ = SKLogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        """Train Logistic Regression classifier.

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
        self.metadata["n_iter"] = (
            self.model_.n_iter_[0] if hasattr(self.model_, "n_iter_") else None
        )

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

    def get_coefficients(self) -> pd.Series:
        """Get model coefficients.

        For binary classification, returns coefficients for positive class.
        For multi-class, returns mean absolute coefficient across all classes.

        Returns:
            Series with feature names as index and coefficients as values.

        Raises:
            RuntimeError: If model hasn't been fitted.
        """
        self._check_is_fitted()

        coef = self.model_.coef_

        if coef.shape[0] == 1:
            # Binary classification
            coef = coef.ravel()
        else:
            # Multi-class: take mean absolute coefficient
            coef = np.abs(coef).mean(axis=0)

        return pd.Series(coef, index=self.feature_names_)


class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier.

    Wrapper around sklearn's SVC with OmicSelector2 interface.

    SVM finds the hyperplane that best separates classes. The kernel trick
    allows non-linear decision boundaries.

    Attributes:
        C: Regularization parameter (larger = less regularization).
        kernel: Kernel type ("linear", "rbf", "poly", "sigmoid").
        gamma: Kernel coefficient (for "rbf", "poly", "sigmoid").
        probability: Whether to enable probability estimates.
        random_state: Random seed for reproducibility.

    Examples:
        >>> model = SVMClassifier(C=1.0, kernel="rbf", probability=True)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "rbf", "poly", "sigmoid"] = "rbf",
        gamma: str = "scale",
        probability: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize SVM classifier.

        Args:
            C: Regularization parameter. Default 1.0.
            kernel: Kernel type. Default "rbf".
            gamma: Kernel coefficient. Default "scale".
            probability: Enable probability estimates. Default False.
            random_state: Random seed. Default None.
        """
        super().__init__()

        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state

        self.model_ = SKSVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SVMClassifier":
        """Train SVM classifier.

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
        self.metadata["n_support_vectors"] = len(self.model_.support_)

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

        Note: Requires probability=True during initialization.

        Args:
            X: Features to predict on (samples × features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.

        Raises:
            RuntimeError: If model hasn't been fitted.
            AttributeError: If probability=False during initialization.
        """
        self._check_is_fitted()

        if not self.probability:
            raise AttributeError("predict_proba() requires probability=True during initialization")

        return self.model_.predict_proba(X)
