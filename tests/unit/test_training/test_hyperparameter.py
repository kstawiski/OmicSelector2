"""Tests for hyperparameter optimization.

This module tests the hyperparameter optimization infrastructure:
- HyperparameterOptimizer initialization
- Optimization with predefined search spaces
- Custom search space support
- Best parameters retrieval
- Integration with Trainer
- Timeout handling
"""

import numpy as np
import pandas as pd
import pytest

from omicselector2.models.base import BaseClassifier
from omicselector2.models.classical import LogisticRegressionModel, RandomForestClassifier
from omicselector2.training.cross_validation import CrossValidator
from omicselector2.training.evaluator import ClassificationEvaluator
from omicselector2.training.hyperparameter import (
    HyperparameterOptimizer,
    PREDEFINED_SEARCH_SPACES,
)


# Fixtures
@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate classification dataset."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(np.random.binomial(1, 0.5, 100), name="target")
    return X, y


class TestHyperparameterOptimizerInitialization:
    """Test HyperparameterOptimizer initialization."""

    def test_initialization_with_predefined_space(self) -> None:
        """Test initialization with predefined search space."""
        optimizer = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=10,
            cv_folds=3,
            metric="accuracy",
            random_state=42,
        )

        assert optimizer.model_name == "RandomForest"
        assert optimizer.n_trials == 10
        assert optimizer.cv_folds == 3
        assert optimizer.metric == "accuracy"
        assert optimizer.random_state == 42

    def test_initialization_with_custom_search_space(self) -> None:
        """Test initialization with custom search space."""
        custom_space = {"C": (0.1, 10.0, "log"), "penalty": ["l1", "l2"]}

        optimizer = HyperparameterOptimizer(
            model_name="LogisticRegression",
            search_space=custom_space,
            n_trials=20,
        )

        assert optimizer.search_space == custom_space

    def test_direction_maximize_for_accuracy(self) -> None:
        """Test that direction is 'maximize' for accuracy metric."""
        optimizer = HyperparameterOptimizer(
            model_name="RandomForest", metric="accuracy", direction="maximize"
        )

        assert optimizer.direction == "maximize"

    def test_direction_minimize_for_loss(self) -> None:
        """Test that direction is 'minimize' for loss metrics."""
        optimizer = HyperparameterOptimizer(
            model_name="RandomForest", metric="log_loss", direction="minimize"
        )

        assert optimizer.direction == "minimize"


class TestPredefinedSearchSpaces:
    """Test predefined search spaces."""

    def test_random_forest_search_space_exists(self) -> None:
        """Test that RandomForest has predefined search space."""
        assert "RandomForest" in PREDEFINED_SEARCH_SPACES

    def test_logistic_regression_search_space_exists(self) -> None:
        """Test that LogisticRegression has predefined search space."""
        assert "LogisticRegression" in PREDEFINED_SEARCH_SPACES

    def test_xgboost_search_space_exists(self) -> None:
        """Test that XGBoost has predefined search space."""
        assert "XGBoost" in PREDEFINED_SEARCH_SPACES

    def test_search_space_has_valid_structure(self) -> None:
        """Test that search spaces have valid structure."""
        rf_space = PREDEFINED_SEARCH_SPACES["RandomForest"]

        # Should have parameter names as keys
        assert "n_estimators" in rf_space
        assert "max_depth" in rf_space


class TestHyperparameterOptimization:
    """Test hyperparameter optimization."""

    @pytest.mark.slow
    def test_optimize_finds_parameters(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that optimize() finds hyperparameters."""
        X, y = classification_data

        optimizer = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=3,  # Small number for speed
            cv_folds=2,
            metric="accuracy",
            random_state=42,
        )

        study = optimizer.optimize(X, y)

        # Should have completed trials
        assert len(study.trials) == 3
        assert study.best_trial is not None

    @pytest.mark.slow
    def test_optimize_with_logistic_regression(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test optimization with LogisticRegression."""
        X, y = classification_data

        optimizer = HyperparameterOptimizer(
            model_name="LogisticRegression",
            n_trials=3,
            cv_folds=2,
            metric="accuracy",
            random_state=42,
        )

        study = optimizer.optimize(X, y)

        assert len(study.trials) == 3
        assert "C" in study.best_params

    @pytest.mark.slow
    def test_get_best_params_returns_dict(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that get_best_params() returns parameter dictionary."""
        X, y = classification_data

        optimizer = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=2,
            cv_folds=2,
            metric="accuracy",
            random_state=42,
        )

        optimizer.optimize(X, y)
        best_params = optimizer.get_best_params()

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params

    @pytest.mark.slow
    def test_get_best_model_returns_trained_model(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that get_best_model() returns trained model."""
        X, y = classification_data

        optimizer = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=2,
            cv_folds=2,
            metric="accuracy",
            random_state=42,
        )

        optimizer.optimize(X, y)
        best_model = optimizer.get_best_model(X, y)

        assert best_model is not None
        assert best_model.is_fitted_

    def test_get_best_params_before_optimize_raises_error(self) -> None:
        """Test that calling get_best_params before optimize raises error."""
        optimizer = HyperparameterOptimizer(
            model_name="RandomForest", n_trials=2, cv_folds=2
        )

        with pytest.raises(RuntimeError, match="optimize.*must be called first"):
            optimizer.get_best_params()


class TestOptimizationWithTimeout:
    """Test optimization with timeout."""

    @pytest.mark.slow
    def test_optimize_respects_timeout(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that optimization stops after timeout."""
        X, y = classification_data

        optimizer = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=100,  # Many trials
            cv_folds=2,
            metric="accuracy",
            random_state=42,
        )

        # Should stop after 2 seconds even though n_trials=100
        study = optimizer.optimize(X, y, timeout=2)

        # Should have stopped before 100 trials (probably < 5 trials in 2 seconds)
        assert len(study.trials) < 100


class TestCustomSearchSpace:
    """Test custom search space functionality."""

    @pytest.mark.slow
    def test_optimize_with_custom_space(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test optimization with custom search space."""
        X, y = classification_data

        custom_space = {
            "n_estimators": (10, 50),  # Smaller range
            "max_depth": (2, 5),  # Smaller depth
        }

        optimizer = HyperparameterOptimizer(
            model_name="RandomForest",
            search_space=custom_space,
            n_trials=2,
            cv_folds=2,
            random_state=42,
        )

        study = optimizer.optimize(X, y)
        best_params = optimizer.get_best_params()

        # Parameters should be within custom ranges
        assert 10 <= best_params["n_estimators"] <= 50
        assert 2 <= best_params["max_depth"] <= 5


class TestOptimizationMetrics:
    """Test different optimization metrics."""

    @pytest.mark.slow
    def test_optimize_with_f1_metric(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test optimization with F1 score."""
        X, y = classification_data

        optimizer = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=2,
            cv_folds=2,
            metric="f1",
            direction="maximize",
            random_state=42,
        )

        study = optimizer.optimize(X, y)

        # Best value should be F1 score (between 0 and 1)
        assert 0 <= study.best_value <= 1

    @pytest.mark.slow
    def test_optimize_with_auc_metric(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test optimization with AUC-ROC."""
        X, y = classification_data

        optimizer = HyperparameterOptimizer(
            model_name="LogisticRegression",
            n_trials=2,
            cv_folds=2,
            metric="auc_roc",
            direction="maximize",
            random_state=42,
        )

        study = optimizer.optimize(X, y)

        # AUC should be between 0 and 1
        assert 0 <= study.best_value <= 1


class TestReproducibility:
    """Test reproducibility with random_state."""

    @pytest.mark.slow
    def test_same_random_state_gives_same_results(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that same random_state gives reproducible results."""
        X, y = classification_data

        optimizer1 = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=3,
            cv_folds=2,
            metric="accuracy",
            random_state=42,
        )
        study1 = optimizer1.optimize(X, y)

        optimizer2 = HyperparameterOptimizer(
            model_name="RandomForest",
            n_trials=3,
            cv_folds=2,
            metric="accuracy",
            random_state=42,
        )
        study2 = optimizer2.optimize(X, y)

        # Should get same best value (or very close)
        assert abs(study1.best_value - study2.best_value) < 0.01
