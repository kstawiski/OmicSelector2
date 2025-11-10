"""Hyperparameter optimization using Optuna.

This module provides automated hyperparameter tuning for classical ML models using Optuna.
It includes predefined search spaces for common models and supports custom search spaces.

Examples:
    >>> from omicselector2.training.hyperparameter import HyperparameterOptimizer
    >>>
    >>> # Use predefined search space
    >>> optimizer = HyperparameterOptimizer(
    ...     model_name="RandomForest",
    ...     n_trials=100,
    ...     cv_folds=5,
    ...     metric="accuracy",
    ...     random_state=42
    ... )
    >>> study = optimizer.optimize(X, y)
    >>> best_params = optimizer.get_best_params()
    >>> best_model = optimizer.get_best_model(X, y)
    >>>
    >>> # Use custom search space
    >>> custom_space = {
    ...     "n_estimators": (10, 100),
    ...     "max_depth": (2, 10)
    ... }
    >>> optimizer = HyperparameterOptimizer(
    ...     model_name="RandomForest",
    ...     search_space=custom_space,
    ...     n_trials=50,
    ...     random_state=42
    ... )
    >>> study = optimizer.optimize(X, y)
"""

from typing import Any, Literal, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from omicselector2.models.classical import (
    LogisticRegressionModel,
    RandomForestClassifier,
    XGBoostClassifier,
)
from omicselector2.training.cross_validation import CrossValidator
from omicselector2.training.evaluator import ClassificationEvaluator

# Predefined search spaces for common models
PREDEFINED_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "n_estimators": (50, 500),
        "max_depth": (3, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
    },
    "XGBoost": {
        "n_estimators": (50, 500),
        "max_depth": (3, 10),
        "learning_rate": (0.001, 0.3, "log"),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
    },
    "LogisticRegression": {
        "C": (0.001, 100, "log"),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    },
}


class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna.

    Uses Tree-structured Parzen Estimator (TPE) sampler for efficient optimization.
    Supports cross-validation for robust parameter selection.

    Attributes:
        model_name: Name of model to optimize ("RandomForest", "XGBoost", "LogisticRegression").
        search_space: Parameter search space. If None, uses predefined space for model_name.
        n_trials: Number of optimization trials.
        cv_folds: Number of cross-validation folds.
        metric: Metric to optimize ("accuracy", "f1", "auc_roc", etc.).
        direction: "maximize" or "minimize".
        random_state: Random seed for reproducibility.

    Examples:
        >>> optimizer = HyperparameterOptimizer(
        ...     model_name="RandomForest",
        ...     n_trials=100,
        ...     metric="accuracy",
        ...     random_state=42
        ... )
        >>> study = optimizer.optimize(X_train, y_train)
        >>> best_params = optimizer.get_best_params()
        >>> print(f"Best accuracy: {study.best_value:.3f}")
    """

    def __init__(
        self,
        model_name: str,
        search_space: Optional[dict[str, Any]] = None,
        n_trials: int = 100,
        cv_folds: int = 5,
        metric: str = "accuracy",
        direction: Literal["maximize", "minimize"] = "maximize",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize HyperparameterOptimizer.

        Args:
            model_name: Model name ("RandomForest", "XGBoost", "LogisticRegression").
            search_space: Custom search space. If None, uses predefined space.
            n_trials: Number of optimization trials. Default 100.
            cv_folds: Number of CV folds. Default 5.
            metric: Metric to optimize. Default "accuracy".
            direction: "maximize" or "minimize". Default "maximize".
            random_state: Random seed. Default None.
        """
        self.model_name = model_name
        self.search_space = search_space or PREDEFINED_SEARCH_SPACES.get(model_name, {})
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.direction = direction
        self.random_state = random_state

        # Will be populated after optimize()
        self.study_: Optional[optuna.Study] = None
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[pd.Series] = None

    def optimize(
        self, X: pd.DataFrame, y: pd.Series, timeout: Optional[int] = None
    ) -> optuna.Study:
        """Run hyperparameter optimization.

        Args:
            X: Training features.
            y: Training target.
            timeout: Optional timeout in seconds.

        Returns:
            Optuna Study object containing optimization results.

        Examples:
            >>> study = optimizer.optimize(X_train, y_train)
            >>> print(f"Best params: {study.best_params}")
            >>> print(f"Best value: {study.best_value:.3f}")
            >>> print(f"Number of trials: {len(study.trials)}")
        """
        # Store data for get_best_model()
        self.X_ = X
        self.y_ = y

        # Create Optuna study
        sampler = TPESampler(seed=self.random_state)
        self.study_ = optuna.create_study(direction=self.direction, sampler=sampler)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        self.study_.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        return self.study_

    def get_best_params(self) -> dict[str, Any]:
        """Get best hyperparameters found during optimization.

        Returns:
            Dictionary of best hyperparameters.

        Raises:
            RuntimeError: If optimize() hasn't been called yet.

        Examples:
            >>> best_params = optimizer.get_best_params()
            >>> print(best_params)
            {'n_estimators': 250, 'max_depth': 15, ...}
        """
        if self.study_ is None:
            raise RuntimeError(
                "optimize() must be called first before accessing best parameters"
            )

        return self.study_.best_params

    def get_best_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train and return model with best hyperparameters.

        Args:
            X: Training features.
            y: Training target.

        Returns:
            Trained model with best hyperparameters.

        Raises:
            RuntimeError: If optimize() hasn't been called yet.

        Examples:
            >>> best_model = optimizer.get_best_model(X_train, y_train)
            >>> predictions = best_model.predict(X_test)
        """
        if self.study_ is None:
            raise RuntimeError("optimize() must be called first before getting best model")

        best_params = self.get_best_params()

        # Create and train model with best parameters
        model = self._create_model(best_params)
        model.fit(X, y)

        return model

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial.
            X: Training features.
            y: Training target.

        Returns:
            Metric value (to be maximized or minimized).
        """
        # Sample hyperparameters
        params = self._sample_params(trial)

        # Create model with sampled parameters
        model = self._create_model(params)

        # Perform cross-validation
        cv = CrossValidator(
            cv_type="stratified", n_splits=self.cv_folds, random_state=self.random_state
        )
        evaluator = ClassificationEvaluator()

        cv_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict_proba(X_val)
            metrics = evaluator.evaluate(y_true=y_val, y_pred=y_pred, probabilities=True)

            # Get metric of interest
            score = metrics.get(self.metric, 0.0)
            cv_scores.append(score)

        # Return mean CV score
        return float(np.mean(cv_scores))

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters from search space.

        Args:
            trial: Optuna trial.

        Returns:
            Dictionary of sampled hyperparameters.
        """
        params = {}

        for param_name, param_spec in self.search_space.items():
            if isinstance(param_spec, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_spec)
            elif isinstance(param_spec, tuple):
                if len(param_spec) == 2:
                    # Integer parameter
                    low, high = param_spec
                    params[param_name] = trial.suggest_int(param_name, low, high)
                elif len(param_spec) == 3 and param_spec[2] == "log":
                    # Float parameter with log scale
                    low, high, _ = param_spec
                    params[param_name] = trial.suggest_float(
                        param_name, low, high, log=True
                    )
                else:
                    # Float parameter with linear scale
                    low, high = param_spec[:2]
                    params[param_name] = trial.suggest_float(param_name, low, high)

        return params

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create model with given parameters.

        Args:
            params: Model hyperparameters.

        Returns:
            Model instance.

        Raises:
            ValueError: If model_name is not supported.
        """
        if self.model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                random_state=self.random_state,
            )
        elif self.model_name == "XGBoost":
            return XGBoostClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                subsample=params.get("subsample", 1.0),
                colsample_bytree=params.get("colsample_bytree", 1.0),
                random_state=self.random_state,
            )
        elif self.model_name == "LogisticRegression":
            return LogisticRegressionModel(
                C=params.get("C", 1.0),
                penalty=params.get("penalty", "l2"),
                solver=params.get("solver", "lbfgs"),
                max_iter=1000,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
