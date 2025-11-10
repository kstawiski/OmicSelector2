"""Training callbacks for OmicSelector2.

This module implements a flexible callback system for training:
- Base Callback interface
- EarlyStopping: Stop training when metric stops improving
- ModelCheckpoint: Save best model during training
- ProgressLogger: Log training progress

Callbacks allow inserting custom logic at different points in the training loop:
- on_train_begin(): Called at start of training
- on_train_end(): Called at end of training
- on_epoch_begin(): Called at start of each epoch
- on_epoch_end(): Called at end of each epoch

Examples:
    >>> from omicselector2.training.callbacks import EarlyStopping, ModelCheckpoint
    >>>
    >>> callbacks = [
    ...     EarlyStopping(monitor="val_loss", patience=10),
    ...     ModelCheckpoint(filepath="best_model.pkl", monitor="val_loss")
    ... ]
    >>>
    >>> # Use with Trainer
    >>> trainer = Trainer(model=model, callbacks=callbacks)
    >>> trainer.fit(X_train, y_train, X_val, y_val)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from omicselector2.training.trainer import Trainer


class Callback:
    """Base callback class.

    All callbacks should inherit from this class and override the methods
    they need to customize.

    Methods can be overridden:
        on_train_begin(trainer): Called when training starts
        on_train_end(trainer): Called when training ends
        on_epoch_begin(epoch, trainer): Called at the start of an epoch
        on_epoch_end(epoch, metrics, trainer): Called at the end of an epoch

    Examples:
        >>> class MyCallback(Callback):
        ...     def on_epoch_end(self, epoch, metrics, trainer):
        ...         print(f"Epoch {epoch}: loss={metrics['loss']:.3f}")
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called when training begins.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called when training ends.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_epoch_begin(self, epoch: int, trainer: "Trainer") -> None:
        """Called at the beginning of an epoch.

        Args:
            epoch: Current epoch number (1-indexed).
            trainer: The trainer instance.
        """
        pass

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], trainer: "Trainer"
    ) -> None:
        """Called at the end of an epoch.

        Args:
            epoch: Current epoch number (1-indexed).
            metrics: Dictionary of metrics computed this epoch.
            trainer: The trainer instance.
        """
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    The callback monitors a metric (e.g., 'val_loss') and stops training
    when it hasn't improved for `patience` epochs.

    Attributes:
        monitor: Metric name to monitor (e.g., "val_loss", "val_accuracy").
        patience: Number of epochs with no improvement to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: "min" for metrics that should decrease (loss), "max" for metrics
            that should increase (accuracy).

    Examples:
        >>> callback = EarlyStopping(monitor="val_loss", patience=10)
        >>> # Training will stop if val_loss doesn't improve for 10 epochs

        >>> callback = EarlyStopping(
        ...     monitor="val_accuracy",
        ...     patience=5,
        ...     min_delta=0.01,
        ...     mode="max"
        ... )
        >>> # Stop if val_accuracy doesn't improve by at least 0.01 for 5 epochs
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        """Initialize EarlyStopping callback.

        Args:
            monitor: Metric to monitor. Default "val_loss".
            patience: Number of epochs to wait for improvement. Default 10.
            min_delta: Minimum change to count as improvement. Default 0.0.
            mode: "min" or "max" depending on metric. Default "min".

        Raises:
            ValueError: If mode is not "min" or "max".
        """
        super().__init__()

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # Internal state
        self.wait: int = 0
        self.best_value: Optional[float] = None
        self.stopped_epoch: int = 0

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Reset state at the beginning of training.

        Args:
            trainer: The trainer instance.
        """
        self.wait = 0
        self.best_value = None
        self.stopped_epoch = 0

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], trainer: "Trainer"
    ) -> None:
        """Check if training should stop based on monitored metric.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            trainer: The trainer instance.
        """
        # Get current value of monitored metric
        current_value = metrics.get(self.monitor)

        if current_value is None:
            # Metric not available, skip this epoch
            return

        # Initialize best value on first epoch
        if self.best_value is None:
            self.best_value = current_value
            return

        # Check if metric improved
        if self._is_improvement(current_value, self.best_value):
            # Improvement detected, reset patience
            self.best_value = current_value
            self.wait = 0
        elif self._is_better(current_value, self.best_value):
            # Metric got better but improvement < min_delta
            # Update best but don't reset patience
            self.best_value = current_value
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training()
        else:
            # Metric got worse or stayed same
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training()

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best (ignoring min_delta).

        Args:
            current: Current metric value.
            best: Best metric value seen so far.

        Returns:
            True if current is better, False otherwise.
        """
        if self.mode == "min":
            return current < best
        else:
            return current > best

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best.

        An improvement means the metric got better by at least min_delta.

        Args:
            current: Current metric value.
            best: Best metric value seen so far.

        Returns:
            True if current is an improvement (better by >= min_delta), False otherwise.
        """
        if self.mode == "min":
            # For loss (minimize), improvement is decrease >= min_delta
            return current < (best - self.min_delta)
        else:
            # For accuracy (maximize), improvement is increase >= min_delta
            return current > (best + self.min_delta)


class ModelCheckpoint(Callback):
    """Save the model when a monitored metric improves.

    The callback monitors a metric (e.g., 'val_loss') and saves the model
    when it improves.

    Attributes:
        filepath: Path to save model to. Can include formatting options:
            - {epoch}: Current epoch number
            - {monitor}: Value of monitored metric
        monitor: Metric name to monitor.
        save_best_only: If True, only save when metric improves.
            If False, save every epoch.
        mode: "min" or "max" depending on metric.

    Examples:
        >>> callback = ModelCheckpoint(
        ...     filepath="best_model.pkl",
        ...     monitor="val_loss",
        ...     save_best_only=True
        ... )
        >>> # Saves model only when val_loss improves

        >>> callback = ModelCheckpoint(
        ...     filepath="model_epoch_{epoch}.pkl",
        ...     save_best_only=False
        ... )
        >>> # Saves model every epoch
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        """Initialize ModelCheckpoint callback.

        Args:
            filepath: Path to save model. Can include {epoch} placeholder.
            monitor: Metric to monitor. Default "val_loss".
            save_best_only: Only save when metric improves. Default True.
            mode: "min" or "max" depending on metric. Default "min".
        """
        super().__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode

        # Internal state
        self.best_value: Optional[float] = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Reset state at the beginning of training.

        Args:
            trainer: The trainer instance.
        """
        self.best_value = None

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], trainer: "Trainer"
    ) -> None:
        """Save model if metric improved (or every epoch if save_best_only=False).

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            trainer: The trainer instance.
        """
        current_value = metrics.get(self.monitor)

        if self.save_best_only:
            # Only save if metric improved
            if current_value is None:
                # Metric not available, skip
                return

            if self.best_value is None:
                # First epoch, save
                self.best_value = current_value
                self._save_model(epoch, current_value, trainer)
            elif self._is_improvement(current_value, self.best_value):
                # Metric improved, save
                self.best_value = current_value
                self._save_model(epoch, current_value, trainer)
        else:
            # Save every epoch
            self._save_model(epoch, current_value, trainer)

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best.

        Args:
            current: Current metric value.
            best: Best metric value seen so far.

        Returns:
            True if current is an improvement, False otherwise.
        """
        if self.mode == "min":
            return current < best
        else:
            return current > best

    def _save_model(
        self, epoch: int, metric_value: Optional[float], trainer: "Trainer"
    ) -> None:
        """Save the model to disk.

        Args:
            epoch: Current epoch number.
            metric_value: Value of monitored metric.
            trainer: The trainer instance.
        """
        # Format filepath with epoch and metric value
        filepath = self.filepath.format(epoch=epoch, monitor=metric_value or 0.0)

        # Save model
        path = Path(filepath)
        trainer.model.save(path)


class ProgressLogger(Callback):
    """Log training progress to console.

    Prints epoch number and metrics at the end of each epoch.

    Examples:
        >>> callback = ProgressLogger()
        >>> # Prints: "Epoch 1: train_loss=0.567, val_loss=0.432, val_accuracy=0.85"
    """

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], trainer: "Trainer"
    ) -> None:
        """Print epoch progress.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            trainer: The trainer instance.
        """
        metrics_str = ", ".join(
            f"{key}={value:.4f}" for key, value in sorted(metrics.items())
        )
        print(f"Epoch {epoch}: {metrics_str}")
