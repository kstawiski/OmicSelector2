"""Tests for Trainer abstraction.

This module tests the unified training interface:
- Trainer initialization
- Basic training with classical models
- Training with validation data
- Training with callbacks
- Cross-validation integration
- History tracking
- Reproducibility
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from omicselector2.models.base import BaseClassifier, BaseRegressor
from omicselector2.training.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omicselector2.training.cross_validation import CrossValidator
from omicselector2.training.evaluator import ClassificationEvaluator, RegressionEvaluator
from omicselector2.training.trainer import Trainer


# Mock models for testing
class MockClassifier(BaseClassifier):
    """Mock classifier for testing."""

    def __init__(self, fail_on_fit: bool = False) -> None:
        super().__init__()
        self.fail_on_fit = fail_on_fit
        self.n_fit_calls = 0
        self.last_X_shape: tuple[int, int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MockClassifier":
        """Mock fit method."""
        if self.fail_on_fit:
            raise ValueError("Mock fit failure")

        self.n_fit_calls += 1
        self.last_X_shape = X.shape
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Mock predict method."""
        self._check_is_fitted()
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Mock predict_proba method."""
        self._check_is_fitted()
        n_classes = len(self.classes_)
        return np.ones((len(X), n_classes)) / n_classes


class MockRegressor(BaseRegressor):
    """Mock regressor for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.n_fit_calls = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MockRegressor":
        """Mock fit method."""
        self.n_fit_calls += 1
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Mock predict method."""
        self._check_is_fitted()
        return np.zeros(len(X))


# Fixtures
@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate classification dataset."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(np.random.binomial(1, 0.5, 100), name="target")
    return X, y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate regression dataset."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(np.random.randn(100), name="target")
    return X, y


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_trainer_initialization_with_classifier(self) -> None:
        """Test Trainer can be initialized with a classifier."""
        model = MockClassifier()
        evaluator = ClassificationEvaluator()

        trainer = Trainer(model=model, evaluator=evaluator)

        assert trainer.model == model
        assert trainer.evaluator == evaluator
        assert trainer.callbacks == []
        assert trainer.random_state is None

    def test_trainer_initialization_with_callbacks(self) -> None:
        """Test Trainer initialization with callbacks."""
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        callbacks = [EarlyStopping(), ModelCheckpoint(filepath="model.pkl")]

        trainer = Trainer(model=model, evaluator=evaluator, callbacks=callbacks)

        assert len(trainer.callbacks) == 2
        assert isinstance(trainer.callbacks[0], EarlyStopping)
        assert isinstance(trainer.callbacks[1], ModelCheckpoint)

    def test_trainer_initialization_with_random_state(self) -> None:
        """Test Trainer initialization with random state."""
        model = MockClassifier()
        evaluator = ClassificationEvaluator()

        trainer = Trainer(model=model, evaluator=evaluator, random_state=42)

        assert trainer.random_state == 42

    def test_trainer_initialization_with_regressor(self) -> None:
        """Test Trainer can be initialized with a regressor."""
        model = MockRegressor()
        evaluator = RegressionEvaluator()

        trainer = Trainer(model=model, evaluator=evaluator)

        assert trainer.model == model
        assert isinstance(trainer.evaluator, RegressionEvaluator)


class TestTrainerBasicFit:
    """Test basic Trainer.fit() functionality."""

    def test_fit_trains_model(self, classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that fit() trains the model."""
        X, y = classification_data
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        history = trainer.fit(X, y)

        assert model.is_fitted_
        assert model.n_fit_calls == 1
        assert isinstance(history, dict)

    def test_fit_returns_history(self, classification_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that fit() returns training history."""
        X, y = classification_data
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        history = trainer.fit(X, y)

        assert "train_accuracy" in history
        assert "train_f1" in history
        assert len(history["train_accuracy"]) > 0

    def test_fit_with_validation_data(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test fit() with validation data."""
        X, y = classification_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        history = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        assert "train_accuracy" in history
        assert "val_accuracy" in history
        assert "val_f1" in history

    def test_fit_with_regressor(self, regression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test fit() with regressor."""
        X, y = regression_data
        model = MockRegressor()
        evaluator = RegressionEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        history = trainer.fit(X, y)

        assert model.is_fitted_
        assert "train_mse" in history or "train_rmse" in history

    def test_fit_raises_on_model_failure(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that fit() raises when model.fit() fails."""
        X, y = classification_data
        model = MockClassifier(fail_on_fit=True)
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        with pytest.raises(ValueError, match="Mock fit failure"):
            trainer.fit(X, y)


class TestTrainerWithCallbacks:
    """Test Trainer with callbacks."""

    def test_callbacks_are_triggered(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that callbacks are triggered during training."""
        X, y = classification_data

        # Custom callback to track calls
        class CallbackTracker(Callback):
            def __init__(self) -> None:
                super().__init__()
                self.on_train_begin_called = False
                self.on_train_end_called = False

            def on_train_begin(self, trainer: Trainer) -> None:
                self.on_train_begin_called = True

            def on_train_end(self, trainer: Trainer) -> None:
                self.on_train_end_called = True

        tracker = CallbackTracker()
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator, callbacks=[tracker])

        trainer.fit(X, y)

        assert tracker.on_train_begin_called
        assert tracker.on_train_end_called

    def test_model_checkpoint_saves_model(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that ModelCheckpoint callback saves model."""
        X, y = classification_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "best_model.pkl"

            model = MockClassifier()
            evaluator = ClassificationEvaluator()
            checkpoint = ModelCheckpoint(filepath=str(filepath), monitor="val_accuracy", mode="max")
            trainer = Trainer(model=model, evaluator=evaluator, callbacks=[checkpoint])

            trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            assert filepath.exists()


class TestTrainerCrossValidation:
    """Test Trainer cross-validation integration."""

    def test_cross_validate_returns_metrics(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that cross_validate() returns metrics."""
        X, y = classification_data
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        cv = CrossValidator(cv_type="kfold", n_splits=3, random_state=42)
        cv_results = trainer.cross_validate(X, y, cv=cv)

        assert "accuracy" in cv_results
        assert "f1" in cv_results
        assert len(cv_results["accuracy"]) == 3  # 3 folds

    def test_cross_validate_trains_multiple_times(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that cross_validate() trains model multiple times."""
        X, y = classification_data
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        cv = CrossValidator(cv_type="kfold", n_splits=5, random_state=42)
        trainer.cross_validate(X, y, cv=cv)

        # Model should be fitted 5 times (one per fold)
        assert model.n_fit_calls == 5

    def test_cross_validate_with_stratified(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test cross_validate() with stratified splitting."""
        X, y = classification_data
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        cv = CrossValidator(cv_type="stratified", n_splits=3, random_state=42)
        cv_results = trainer.cross_validate(X, y, cv=cv)

        assert len(cv_results["accuracy"]) == 3


class TestTrainerReproducibility:
    """Test Trainer reproducibility."""

    def test_random_state_ensures_reproducibility(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that random_state ensures reproducible results."""
        X, y = classification_data

        # Train twice with same random state
        model1 = MockClassifier()
        trainer1 = Trainer(model=model1, evaluator=ClassificationEvaluator(), random_state=42)
        trainer1.fit(X, y)

        model2 = MockClassifier()
        trainer2 = Trainer(model=model2, evaluator=ClassificationEvaluator(), random_state=42)
        trainer2.fit(X, y)

        # Results should be identical
        assert model1.last_X_shape == model2.last_X_shape


class TestTrainerStopTraining:
    """Test Trainer stop_training() method."""

    def test_stop_training_method_exists(self) -> None:
        """Test that Trainer has stop_training() method."""
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        assert hasattr(trainer, "stop_training")
        assert callable(trainer.stop_training)

    def test_stop_training_sets_flag(self) -> None:
        """Test that stop_training() sets internal flag."""
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        trainer.stop_training()

        # Should have a stopped flag
        assert hasattr(trainer, "stopped")
        assert trainer.stopped is True


class TestTrainerHistory:
    """Test Trainer history tracking."""

    def test_history_attribute_exists(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that Trainer has history attribute after training."""
        X, y = classification_data
        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        trainer.fit(X, y)

        assert hasattr(trainer, "history")
        assert isinstance(trainer.history, dict)

    def test_history_contains_metrics(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that history contains training metrics."""
        X, y = classification_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        model = MockClassifier()
        evaluator = ClassificationEvaluator()
        trainer = Trainer(model=model, evaluator=evaluator)

        trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        assert "train_accuracy" in trainer.history
        assert "val_accuracy" in trainer.history
