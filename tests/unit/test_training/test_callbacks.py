"""Tests for training callbacks.

This module tests the callback system for training:
- Base Callback interface
- EarlyStopping callback
- ModelCheckpoint callback
- ProgressLogger callback
- MLflowCallback
- Multiple callbacks working together
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from omicselector2.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
)


# Mock trainer for testing callbacks
class MockTrainer:
    """Mock trainer for testing callbacks."""

    def __init__(self) -> None:
        self.model = MockModel()
        self.epoch = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.stopped = False

    def stop_training(self) -> None:
        """Stop training."""
        self.stopped = True


class MockModel:
    """Mock model for testing."""

    def save(self, path: Path) -> None:
        """Mock save method."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("mock_model")


class TestCallback:
    """Test base Callback class."""

    def test_callback_initialization(self) -> None:
        """Test that Callback can be instantiated."""
        callback = Callback()
        assert callback is not None

    def test_callback_has_required_methods(self) -> None:
        """Test that Callback has all required methods."""
        callback = Callback()
        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_train_end")
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_epoch_end")

    def test_callback_methods_are_callable(self) -> None:
        """Test that all callback methods can be called."""
        callback = Callback()
        trainer = MockTrainer()

        # Should not raise
        callback.on_train_begin(trainer)
        callback.on_train_end(trainer)
        callback.on_epoch_begin(epoch=1, trainer=trainer)
        callback.on_epoch_end(epoch=1, metrics={"loss": 0.5}, trainer=trainer)


class TestEarlyStopping:
    """Test EarlyStopping callback."""

    def test_early_stopping_initialization(self) -> None:
        """Test EarlyStopping initialization with default parameters."""
        callback = EarlyStopping()
        assert callback.monitor == "val_loss"
        assert callback.patience == 10
        assert callback.min_delta == 0.0
        assert callback.mode == "min"

    def test_early_stopping_custom_parameters(self) -> None:
        """Test EarlyStopping with custom parameters."""
        callback = EarlyStopping(monitor="val_accuracy", patience=5, min_delta=0.001, mode="max")
        assert callback.monitor == "val_accuracy"
        assert callback.patience == 5
        assert callback.min_delta == 0.001
        assert callback.mode == "max"

    def test_early_stopping_mode_validation(self) -> None:
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            EarlyStopping(mode="invalid")  # type: ignore

    def test_early_stopping_stops_when_no_improvement_min_mode(self) -> None:
        """Test early stopping in 'min' mode stops when metric stops decreasing."""
        callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        trainer = MockTrainer()

        callback.on_train_begin(trainer)

        # Improving epochs
        callback.on_epoch_end(epoch=1, metrics={"val_loss": 1.0}, trainer=trainer)
        assert not trainer.stopped

        callback.on_epoch_end(epoch=2, metrics={"val_loss": 0.5}, trainer=trainer)
        assert not trainer.stopped

        # No improvement for 2 epochs (patience=2)
        callback.on_epoch_end(epoch=3, metrics={"val_loss": 0.6}, trainer=trainer)
        assert not trainer.stopped

        callback.on_epoch_end(epoch=4, metrics={"val_loss": 0.7}, trainer=trainer)
        assert not trainer.stopped

        # Third epoch without improvement - should stop
        callback.on_epoch_end(epoch=5, metrics={"val_loss": 0.8}, trainer=trainer)
        assert trainer.stopped

    def test_early_stopping_stops_when_no_improvement_max_mode(self) -> None:
        """Test early stopping in 'max' mode stops when metric stops increasing."""
        callback = EarlyStopping(monitor="val_accuracy", patience=2, mode="max")
        trainer = MockTrainer()

        callback.on_train_begin(trainer)

        # Improving epochs
        callback.on_epoch_end(epoch=1, metrics={"val_accuracy": 0.5}, trainer=trainer)
        assert not trainer.stopped

        callback.on_epoch_end(epoch=2, metrics={"val_accuracy": 0.8}, trainer=trainer)
        assert not trainer.stopped

        # No improvement for 2 epochs
        callback.on_epoch_end(epoch=3, metrics={"val_accuracy": 0.7}, trainer=trainer)
        assert not trainer.stopped

        callback.on_epoch_end(epoch=4, metrics={"val_accuracy": 0.75}, trainer=trainer)
        assert not trainer.stopped

        # Third epoch without improvement - should stop
        callback.on_epoch_end(epoch=5, metrics={"val_accuracy": 0.6}, trainer=trainer)
        assert trainer.stopped

    def test_early_stopping_respects_min_delta(self) -> None:
        """Test that early stopping considers min_delta for improvement."""
        callback = EarlyStopping(monitor="val_loss", patience=2, min_delta=0.01, mode="min")
        trainer = MockTrainer()

        callback.on_train_begin(trainer)

        callback.on_epoch_end(epoch=1, metrics={"val_loss": 1.0}, trainer=trainer)
        assert not trainer.stopped

        # Improvement of 0.005 < min_delta (0.01), doesn't count as improvement
        callback.on_epoch_end(epoch=2, metrics={"val_loss": 0.995}, trainer=trainer)
        assert not trainer.stopped

        callback.on_epoch_end(epoch=3, metrics={"val_loss": 0.99}, trainer=trainer)
        assert not trainer.stopped

        # After patience epochs without sufficient improvement
        callback.on_epoch_end(epoch=4, metrics={"val_loss": 0.985}, trainer=trainer)
        assert trainer.stopped

    def test_early_stopping_resets_patience_on_improvement(self) -> None:
        """Test that patience counter resets when metric improves."""
        callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        trainer = MockTrainer()

        callback.on_train_begin(trainer)

        callback.on_epoch_end(epoch=1, metrics={"val_loss": 1.0}, trainer=trainer)
        callback.on_epoch_end(epoch=2, metrics={"val_loss": 1.1}, trainer=trainer)

        # Improvement - should reset patience
        callback.on_epoch_end(epoch=3, metrics={"val_loss": 0.8}, trainer=trainer)
        assert not trainer.stopped

        # Start counting again
        callback.on_epoch_end(epoch=4, metrics={"val_loss": 0.9}, trainer=trainer)
        callback.on_epoch_end(epoch=5, metrics={"val_loss": 1.0}, trainer=trainer)
        assert not trainer.stopped

        # Third no-improvement epoch
        callback.on_epoch_end(epoch=6, metrics={"val_loss": 1.1}, trainer=trainer)
        assert trainer.stopped

    def test_early_stopping_handles_missing_metric(self) -> None:
        """Test early stopping when monitored metric is missing."""
        callback = EarlyStopping(monitor="val_loss", patience=2)
        trainer = MockTrainer()

        callback.on_train_begin(trainer)

        # Missing metric should not cause error, just skip
        callback.on_epoch_end(epoch=1, metrics={"train_loss": 1.0}, trainer=trainer)
        assert not trainer.stopped


class TestModelCheckpoint:
    """Test ModelCheckpoint callback."""

    def test_model_checkpoint_initialization(self) -> None:
        """Test ModelCheckpoint initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            callback = ModelCheckpoint(filepath=str(filepath))

            assert callback.filepath == str(filepath)
            assert callback.monitor == "val_loss"
            assert callback.save_best_only is True
            assert callback.mode == "min"

    def test_model_checkpoint_saves_best_model_min_mode(self) -> None:
        """Test that ModelCheckpoint saves when metric improves (min mode)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "best_model.pkl"
            callback = ModelCheckpoint(filepath=str(filepath), monitor="val_loss", mode="min")
            trainer = MockTrainer()

            callback.on_train_begin(trainer)

            # First epoch - should save
            callback.on_epoch_end(epoch=1, metrics={"val_loss": 1.0}, trainer=trainer)
            assert filepath.exists()

            # Remove file to test if it gets saved again
            filepath.unlink()

            # Improvement - should save
            callback.on_epoch_end(epoch=2, metrics={"val_loss": 0.5}, trainer=trainer)
            assert filepath.exists()

            filepath.unlink()

            # No improvement - should not save
            callback.on_epoch_end(epoch=3, metrics={"val_loss": 0.6}, trainer=trainer)
            assert not filepath.exists()

    def test_model_checkpoint_saves_best_model_max_mode(self) -> None:
        """Test that ModelCheckpoint saves when metric improves (max mode)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "best_model.pkl"
            callback = ModelCheckpoint(filepath=str(filepath), monitor="val_accuracy", mode="max")
            trainer = MockTrainer()

            callback.on_train_begin(trainer)

            callback.on_epoch_end(epoch=1, metrics={"val_accuracy": 0.5}, trainer=trainer)
            assert filepath.exists()

            filepath.unlink()

            # Improvement - should save
            callback.on_epoch_end(epoch=2, metrics={"val_accuracy": 0.8}, trainer=trainer)
            assert filepath.exists()

            filepath.unlink()

            # No improvement - should not save
            callback.on_epoch_end(epoch=3, metrics={"val_accuracy": 0.7}, trainer=trainer)
            assert not filepath.exists()

    def test_model_checkpoint_save_all_mode(self) -> None:
        """Test ModelCheckpoint with save_best_only=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model_{epoch}.pkl"
            callback = ModelCheckpoint(filepath=str(filepath), save_best_only=False)
            trainer = MockTrainer()

            callback.on_train_begin(trainer)

            # Should save every epoch
            callback.on_epoch_end(epoch=1, metrics={"val_loss": 1.0}, trainer=trainer)
            callback.on_epoch_end(epoch=2, metrics={"val_loss": 1.5}, trainer=trainer)
            callback.on_epoch_end(epoch=3, metrics={"val_loss": 0.8}, trainer=trainer)

            # Check that files were created (at least one)
            saved_files = list(Path(tmpdir).glob("*.pkl"))
            assert len(saved_files) > 0

    def test_model_checkpoint_handles_missing_metric(self) -> None:
        """Test ModelCheckpoint when monitored metric is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            callback = ModelCheckpoint(filepath=str(filepath), monitor="val_loss")
            trainer = MockTrainer()

            callback.on_train_begin(trainer)

            # Missing metric should not save
            callback.on_epoch_end(epoch=1, metrics={"train_loss": 1.0}, trainer=trainer)
            assert not filepath.exists()


class TestProgressLogger:
    """Test ProgressLogger callback."""

    def test_progress_logger_initialization(self) -> None:
        """Test ProgressLogger initialization."""
        callback = ProgressLogger()
        assert callback is not None

    def test_progress_logger_logs_epochs(self, capsys: Any) -> None:
        """Test that ProgressLogger logs epoch information."""
        callback = ProgressLogger()
        trainer = MockTrainer()

        callback.on_train_begin(trainer)
        callback.on_epoch_end(
            epoch=1, metrics={"train_loss": 1.0, "val_loss": 0.8}, trainer=trainer
        )

        captured = capsys.readouterr()
        assert "Epoch 1" in captured.out or "Epoch 1" in str(captured)


class TestMultipleCallbacks:
    """Test multiple callbacks working together."""

    def test_multiple_callbacks_all_execute(self) -> None:
        """Test that multiple callbacks all get executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=3),
                ModelCheckpoint(filepath=str(filepath), monitor="val_loss"),
                ProgressLogger(),
            ]

            trainer = MockTrainer()

            # Execute all callbacks
            for callback in callbacks:
                callback.on_train_begin(trainer)

            for epoch in range(1, 4):
                metrics = {"val_loss": 1.0 - (epoch * 0.1)}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch=epoch, trainer=trainer)
                    callback.on_epoch_end(epoch=epoch, metrics=metrics, trainer=trainer)

            # Model should be saved
            assert filepath.exists()

            # Should not have stopped (still improving)
            assert not trainer.stopped

    def test_early_stopping_prevents_further_epochs(self) -> None:
        """Test that early stopping stops training when triggered."""
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=1),
        ]

        trainer = MockTrainer()

        for callback in callbacks:
            callback.on_train_begin(trainer)

        # First epoch
        metrics = {"val_loss": 1.0}
        for callback in callbacks:
            callback.on_epoch_end(epoch=1, metrics=metrics, trainer=trainer)
        assert not trainer.stopped

        # Second epoch - no improvement
        metrics = {"val_loss": 1.1}
        for callback in callbacks:
            callback.on_epoch_end(epoch=2, metrics=metrics, trainer=trainer)
        assert not trainer.stopped

        # Third epoch - should stop
        metrics = {"val_loss": 1.2}
        for callback in callbacks:
            callback.on_epoch_end(epoch=3, metrics=metrics, trainer=trainer)
        assert trainer.stopped
