"""Tests for base model classes and interfaces.

The base model infrastructure provides:
- Abstract base classes for all models
- Consistent fit/predict interface
- Hyperparameter management
- Model persistence (save/load)
- Metadata tracking

Test coverage:
- BaseModel interface
- Model lifecycle (fit, predict, save, load)
- Hyperparameter handling
- Error cases
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.models.base import BaseModel  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(np.random.binomial(1, 0.5, n_samples), name="target")

    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(np.random.randn(n_samples) * 10 + 50, name="target")

    return X, y


class TestBaseModel:
    """Test suite for BaseModel abstract base class."""

    def test_import(self) -> None:
        """Test that BaseModel can be imported."""
        from omicselector2.models.base import BaseModel

        assert BaseModel is not None

    def test_base_model_is_abstract(self) -> None:
        """Test that BaseModel cannot be instantiated directly."""
        from omicselector2.models.base import BaseModel

        with pytest.raises(TypeError):
            BaseModel()  # type: ignore

    def test_base_model_requires_fit_method(self) -> None:
        """Test that subclasses must implement fit()."""
        from omicselector2.models.base import BaseModel

        class IncompleteModel(BaseModel):
            """Model without fit() method."""

            def predict(self, X):
                pass

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore

    def test_base_model_requires_predict_method(self) -> None:
        """Test that subclasses must implement predict()."""
        from omicselector2.models.base import BaseModel

        class IncompleteModel(BaseModel):
            """Model without predict() method."""

            def fit(self, X, y):
                pass

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore


class TestBaseClassifier:
    """Test suite for BaseClassifier."""

    def test_import(self) -> None:
        """Test that BaseClassifier can be imported."""
        from omicselector2.models.base import BaseClassifier

        assert BaseClassifier is not None

    def test_base_classifier_has_predict_proba(self) -> None:
        """Test that BaseClassifier requires predict_proba()."""
        from omicselector2.models.base import BaseClassifier

        # BaseClassifier should have predict_proba in its interface
        assert hasattr(BaseClassifier, "predict_proba")


class TestBaseRegressor:
    """Test suite for BaseRegressor."""

    def test_import(self) -> None:
        """Test that BaseRegressor can be imported."""
        from omicselector2.models.base import BaseRegressor

        assert BaseRegressor is not None


class TestModelMetadata:
    """Test suite for model metadata tracking."""

    def test_model_has_metadata_attribute(self) -> None:
        """Test that models can store metadata."""
        from omicselector2.models.base import BaseModel

        class DummyModel(BaseModel):
            def __init__(self):
                super().__init__()

            def fit(self, X, y):
                self.is_fitted_ = True
                return self

            def predict(self, X):
                return np.zeros(len(X))

        model = DummyModel()
        assert hasattr(model, "metadata")

    def test_model_tracks_training_time(self) -> None:
        """Test that models track training time."""
        from omicselector2.models.base import BaseModel

        class DummyModel(BaseModel):
            def __init__(self):
                super().__init__()

            def fit(self, X, y):
                import time

                start = time.time()
                time.sleep(0.01)  # Simulate training
                end = time.time()

                self.metadata["training_time"] = end - start
                self.is_fitted_ = True
                return self

            def predict(self, X):
                return np.zeros(len(X))

        model = DummyModel()
        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series(np.random.randint(0, 2, 10))

        model.fit(X, y)

        assert "training_time" in model.metadata
        assert model.metadata["training_time"] > 0


class TestModelPersistence:
    """Test suite for model save/load functionality."""

    def test_model_can_save(self) -> None:
        """Test that models can be saved."""
        from omicselector2.models.base import BaseModel

        class DummyModel(BaseModel):
            def __init__(self, param: int = 1):
                super().__init__()
                self.param = param

            def fit(self, X, y):
                self.coef_ = np.ones(X.shape[1])
                self.is_fitted_ = True
                return self

            def predict(self, X):
                return X.values @ self.coef_

        model = DummyModel(param=42)
        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series(np.random.randn(10))
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            assert save_path.exists()

    def test_model_can_load(self) -> None:
        """Test that models can be loaded."""
        from omicselector2.models.base import BaseModel

        class DummyModel(BaseModel):
            def __init__(self, param: int = 1):
                super().__init__()
                self.param = param

            def fit(self, X, y):
                self.coef_ = np.ones(X.shape[1]) * self.param
                self.is_fitted_ = True
                return self

            def predict(self, X):
                return X.values @ self.coef_

        # Train and save model
        model = DummyModel(param=42)
        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series(np.random.randn(10))
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            # Load model
            loaded_model = DummyModel.load(save_path)

            assert loaded_model.param == 42
            assert hasattr(loaded_model, "coef_")
            assert loaded_model.is_fitted_

    def test_loaded_model_predictions_match(self) -> None:
        """Test that loaded model produces same predictions."""
        from omicselector2.models.base import BaseModel

        class DummyModel(BaseModel):
            def __init__(self, param: int = 1):
                super().__init__()
                self.param = param

            def fit(self, X, y):
                self.coef_ = np.random.randn(X.shape[1])
                self.is_fitted_ = True
                return self

            def predict(self, X):
                return X.values @ self.coef_

        # Train and save model
        model = DummyModel(param=42)
        X_train = pd.DataFrame(np.random.randn(10, 5))
        y_train = pd.Series(np.random.randn(10))
        model.fit(X_train, y_train)

        X_test = pd.DataFrame(np.random.randn(5, 5))
        original_predictions = model.predict(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            # Load and predict
            loaded_model = DummyModel.load(save_path)
            loaded_predictions = loaded_model.predict(X_test)

            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
