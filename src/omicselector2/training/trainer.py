"""Trainer abstraction for unified model training.

This module provides a unified interface for training all types of models:
- Classical ML models (single fit call)
- Deep learning models (iterative training with epochs)
- Support for callbacks (early stopping, checkpointing, etc.)
- Cross-validation integration
- History tracking
- Reproducibility via random seeds

The Trainer orchestrates the entire training process, making it consistent
across different model types.

Examples:
    >>> from omicselector2.training.trainer import Trainer
    >>> from omicselector2.models.classical import RandomForestClassifier
    >>> from omicselector2.training.evaluator import ClassificationEvaluator
    >>> from omicselector2.training.callbacks import EarlyStopping
    >>>
    >>> # Initialize trainer
    >>> model = RandomForestClassifier(n_estimators=100)
    >>> evaluator = ClassificationEvaluator()
    >>> callbacks = [EarlyStopping(monitor="val_loss", patience=10)]
    >>>
    >>> trainer = Trainer(
    ...     model=model,
    ...     evaluator=evaluator,
    ...     callbacks=callbacks,
    ...     random_state=42
    ... )
    >>>
    >>> # Train with validation
    >>> history = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>>
    >>> # Or cross-validate
    >>> from omicselector2.training.cross_validation import CrossValidator
    >>> cv = CrossValidator(cv_type="kfold", n_splits=5)
    >>> cv_results = trainer.cross_validate(X, y, cv=cv)
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from omicselector2.models.base import BaseClassifier, BaseModel
from omicselector2.training.callbacks import Callback
from omicselector2.training.cross_validation import CrossValidator
from omicselector2.training.evaluator import (
    ClassificationEvaluator,
    RegressionEvaluator,
    SurvivalEvaluator,
)


class Trainer:
    """Unified trainer for all model types.

    The Trainer provides a consistent interface for training classical ML models
    and deep learning models. It handles:
    - Model fitting with train/validation splits
    - Callback execution
    - History tracking
    - Cross-validation
    - Reproducibility

    Attributes:
        model: The model to train (must inherit from BaseModel).
        evaluator: Evaluator for computing metrics.
        callbacks: List of callbacks to execute during training.
        random_state: Random seed for reproducibility.
        history: Dictionary of training metrics (populated after fit()).
        stopped: Flag indicating if training was stopped early.

    Examples:
        >>> # Classification with early stopping
        >>> trainer = Trainer(
        ...     model=LogisticRegressionModel(),
        ...     evaluator=ClassificationEvaluator(),
        ...     callbacks=[EarlyStopping(patience=5)],
        ...     random_state=42
        ... )
        >>> history = trainer.fit(X_train, y_train, X_val, y_val)
        >>>
        >>> # Regression without callbacks
        >>> trainer = Trainer(
        ...     model=RandomForestRegressor(),
        ...     evaluator=RegressionEvaluator(),
        ...     random_state=42
        ... )
        >>> history = trainer.fit(X, y)
    """

    def __init__(
        self,
        model: BaseModel,
        evaluator: Union[ClassificationEvaluator, RegressionEvaluator, SurvivalEvaluator],
        callbacks: Optional[list[Callback]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            model: Model to train (BaseModel subclass).
            evaluator: Evaluator for computing metrics.
            callbacks: Optional list of callbacks. Default None.
            random_state: Random seed for reproducibility. Default None.
        """
        self.model = model
        self.evaluator = evaluator
        self.callbacks = callbacks or []
        self.random_state = random_state

        # Training state
        self.history: dict[str, list[float]] = {}
        self.stopped = False

        # Set random seed if provided
        if random_state is not None:
            self._set_random_seed(random_state)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        epochs: int = 1,
    ) -> dict[str, list[float]]:
        """Train the model.

        For classical ML models (sklearn-based), epochs=1 (single fit call).
        For deep learning models, epochs can be > 1 for iterative training.

        Args:
            X: Training features (samples × features).
            y: Training target.
            X_val: Optional validation features.
            y_val: Optional validation target.
            epochs: Number of training epochs. Default 1 (for classical models).

        Returns:
            Dictionary of training history with metric names as keys and
            lists of values (one per epoch) as values.

        Examples:
            >>> # Classical model (single fit call)
            >>> history = trainer.fit(X_train, y_train)
            >>>
            >>> # With validation data
            >>> history = trainer.fit(X_train, y_train, X_val, y_val)
        """
        # Reset training state
        self.history = {}
        self.stopped = False

        # Trigger on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Check if model supports incremental training (has partial_fit method)
        # Check the actual model implementation (model_) if it exists, otherwise check self.model
        model_obj = getattr(self.model, "model_", self.model)
        has_partial_fit = hasattr(model_obj, "partial_fit")

        # For classical models without partial_fit, fit only once (epochs is ignored)
        # For models with partial_fit, the loop allows iterative training
        actual_epochs = epochs if has_partial_fit else 1

        for epoch in range(1, actual_epochs + 1):
            # Trigger on_epoch_begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)

            # Fit model on training data
            # Classical models are fit only once (outside loop when actual_epochs=1)
            # Models with partial_fit can train incrementally
            if has_partial_fit and epoch > 1:
                # Use partial_fit for incremental training
                self.model.model.partial_fit(X, y)
            else:
                # First epoch or models without partial_fit use regular fit
                self.model.fit(X, y)

            # Compute training metrics
            y_train_pred = self.model.predict(X)

            # Get probabilities for classification
            if isinstance(self.model, BaseClassifier):
                y_train_proba = self.model.predict_proba(X)
                train_metrics = self.evaluator.evaluate(
                    y_true=y, y_pred=y_train_proba, probabilities=True
                )
            else:
                train_metrics = self.evaluator.evaluate(y_true=y, y_pred=y_train_pred)

            # Add train_ prefix to metrics
            epoch_metrics = {f"train_{k}": v for k, v in train_metrics.items()}

            # Compute validation metrics if validation data provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)

                if isinstance(self.model, BaseClassifier):
                    y_val_proba = self.model.predict_proba(X_val)
                    val_metrics = self.evaluator.evaluate(
                        y_true=y_val, y_pred=y_val_proba, probabilities=True
                    )
                else:
                    val_metrics = self.evaluator.evaluate(y_true=y_val, y_pred=y_val_pred)

                # Add val_ prefix to metrics
                val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
                epoch_metrics.update(val_metrics_prefixed)

            # Update history
            for metric_name, metric_value in epoch_metrics.items():
                if metric_name not in self.history:
                    self.history[metric_name] = []
                self.history[metric_name].append(metric_value)

            # Trigger on_epoch_end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_metrics, self)

            # Check if training should stop
            if self.stopped:
                break

        # Trigger on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.history

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: CrossValidator
    ) -> dict[str, list[float]]:
        """Perform cross-validation.

        Trains and evaluates the model using cross-validation. The model is
        trained once for each fold.

        Args:
            X: Features (samples × features).
            y: Target values.
            cv: CrossValidator instance defining the splitting strategy.

        Returns:
            Dictionary of cross-validation results with metric names as keys
            and lists of values (one per fold) as values.

        Examples:
            >>> from omicselector2.training.cross_validation import CrossValidator
            >>>
            >>> cv = CrossValidator(cv_type="kfold", n_splits=5, random_state=42)
            >>> cv_results = trainer.cross_validate(X, y, cv=cv)
            >>>
            >>> print(f"Mean accuracy: {np.mean(cv_results['accuracy']):.3f}")
            >>> print(f"Std accuracy: {np.std(cv_results['accuracy']):.3f}")
        """
        cv_results: dict[str, list[float]] = {}

        # Iterate over folds
        for train_idx, test_idx in cv.split(X, y):
            # Split data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Train model on this fold
            self.model.fit(X_train, y_train)

            # Predict on test fold
            y_pred = self.model.predict(X_test)

            # Compute metrics
            if isinstance(self.model, BaseClassifier):
                y_proba = self.model.predict_proba(X_test)
                fold_metrics = self.evaluator.evaluate(
                    y_true=y_test, y_pred=y_proba, probabilities=True
                )
            else:
                fold_metrics = self.evaluator.evaluate(y_true=y_test, y_pred=y_pred)

            # Append to results
            for metric_name, metric_value in fold_metrics.items():
                if metric_name not in cv_results:
                    cv_results[metric_name] = []
                cv_results[metric_name].append(metric_value)

        return cv_results

    def stop_training(self) -> None:
        """Stop training.

        This method is called by callbacks (e.g., EarlyStopping) to signal
        that training should stop. Sets the `stopped` flag to True.

        Examples:
            >>> # Called internally by EarlyStopping callback
            >>> callback = EarlyStopping(patience=5)
            >>> # When patience is exceeded:
            >>> trainer.stop_training()  # Sets trainer.stopped = True
        """
        self.stopped = True

    def _set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Sets seeds for:
        - NumPy random number generator
        - Python's random module (via np.random.seed)

        Args:
            seed: Random seed.
        """
        np.random.seed(seed)
