"""Trainer abstraction for OmicSelector2.

This module will implement the unified training interface.
STUB - to be implemented after tests are written (TDD).
"""

from typing import Any, Optional

import pandas as pd

from omicselector2.models.base import BaseModel
from omicselector2.training.callbacks import Callback
from omicselector2.training.cross_validation import CrossValidator
from omicselector2.training.evaluator import ClassificationEvaluator, RegressionEvaluator


class Trainer:
    """Trainer abstraction - TO BE IMPLEMENTED."""

    def __init__(
        self,
        model: BaseModel,
        evaluator: Any,
        callbacks: Optional[list[Callback]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize trainer - STUB."""
        self.model = model
        self.evaluator = evaluator
        self.callbacks = callbacks or []
        self.random_state = random_state
        self.history: dict[str, list[float]] = {}
        self.stopped = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict[str, list[float]]:
        """Train model - STUB."""
        raise NotImplementedError("To be implemented")

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: CrossValidator
    ) -> dict[str, list[float]]:
        """Cross-validate model - STUB."""
        raise NotImplementedError("To be implemented")

    def stop_training(self) -> None:
        """Stop training - STUB."""
        self.stopped = True
