"""Tests for classical machine learning models.

Classical models include:
- Random Forest (classification and regression)
- Logistic Regression
- Support Vector Machines (SVM)
- XGBoost (classification and regression)

Test coverage:
- Model initialization
- Training (fit)
- Prediction
- Probability prediction (classification)
- Hyperparameter handling
- Model persistence
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

from omicselector2.models.classical import (  # noqa: E402
    LogisticRegressionModel,
    RandomForestClassifier,
    RandomForestRegressor,
    SVMClassifier,
    XGBoostClassifier,
    XGBoostRegressor,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
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
def multiclass_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample multiclass classification data."""
    np.random.seed(42)
    n_samples = 150
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(np.random.choice([0, 1, 2], n_samples), name="target")

    return X, y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
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


class TestRandomForestClassifier:
    """Test suite for RandomForestClassifier."""

    def test_import(self) -> None:
        """Test that RandomForestClassifier can be imported."""
        from omicselector2.models.classical import RandomForestClassifier

        assert RandomForestClassifier is not None

    def test_initialization(self) -> None:
        """Test RandomForestClassifier initialization."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        assert model.n_estimators == 100
        assert model.random_state == 42

    def test_fit(self, classification_data: tuple) -> None:
        """Test fitting RandomForestClassifier."""
        X, y = classification_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "is_fitted_")
        assert model.is_fitted_

    def test_predict(self, classification_data: tuple) -> None:
        """Test predictions from RandomForestClassifier."""
        X, y = classification_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, classification_data: tuple) -> None:
        """Test probability predictions."""
        X, y = classification_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        probabilities = model.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_multiclass(self, multiclass_data: tuple) -> None:
        """Test RandomForestClassifier on multiclass problem."""
        X, y = multiclass_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert set(predictions).issubset({0, 1, 2})
        assert probabilities.shape == (len(X), 3)

    def test_feature_importance(self, classification_data: tuple) -> None:
        """Test feature importance extraction."""
        X, y = classification_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)
        assert np.isclose(importance.sum(), 1.0)

    def test_save_load(self, classification_data: tuple) -> None:
        """Test model persistence."""
        X, y = classification_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rf_model.pkl"
            model.save(save_path)

            loaded_model = RandomForestClassifier.load(save_path)

            # Check predictions match
            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)

            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestRandomForestRegressor:
    """Test suite for RandomForestRegressor."""

    def test_import(self) -> None:
        """Test that RandomForestRegressor can be imported."""
        from omicselector2.models.classical import RandomForestRegressor

        assert RandomForestRegressor is not None

    def test_fit_predict(self, regression_data: tuple) -> None:
        """Test fitting and prediction."""
        X, y = regression_data

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert predictions.dtype in [np.float32, np.float64]

    def test_feature_importance(self, regression_data: tuple) -> None:
        """Test feature importance extraction."""
        X, y = regression_data

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)


class TestLogisticRegressionModel:
    """Test suite for LogisticRegressionModel."""

    def test_import(self) -> None:
        """Test that LogisticRegressionModel can be imported."""
        from omicselector2.models.classical import LogisticRegressionModel

        assert LogisticRegressionModel is not None

    def test_initialization(self) -> None:
        """Test LogisticRegressionModel initialization."""
        model = LogisticRegressionModel(C=1.0, random_state=42)

        assert model.C == 1.0
        assert model.random_state == 42

    def test_fit_predict(self, classification_data: tuple) -> None:
        """Test fitting and prediction."""
        X, y = classification_data

        model = LogisticRegressionModel(C=1.0, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, classification_data: tuple) -> None:
        """Test probability predictions."""
        X, y = classification_data

        model = LogisticRegressionModel(C=1.0, random_state=42)
        model.fit(X, y)

        probabilities = model.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_get_coefficients(self, classification_data: tuple) -> None:
        """Test coefficient extraction."""
        X, y = classification_data

        model = LogisticRegressionModel(C=1.0, random_state=42)
        model.fit(X, y)

        coef = model.get_coefficients()

        assert len(coef) == X.shape[1]

    def test_multiclass(self, multiclass_data: tuple) -> None:
        """Test multiclass logistic regression."""
        X, y = multiclass_data

        model = LogisticRegressionModel(C=1.0, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert set(predictions).issubset({0, 1, 2})
        assert probabilities.shape == (len(X), 3)


class TestSVMClassifier:
    """Test suite for SVMClassifier."""

    def test_import(self) -> None:
        """Test that SVMClassifier can be imported."""
        from omicselector2.models.classical import SVMClassifier

        assert SVMClassifier is not None

    def test_initialization(self) -> None:
        """Test SVMClassifier initialization."""
        model = SVMClassifier(C=1.0, kernel="rbf", random_state=42)

        assert model.C == 1.0
        assert model.kernel == "rbf"
        assert model.random_state == 42

    def test_fit_predict(self, classification_data: tuple) -> None:
        """Test fitting and prediction."""
        X, y = classification_data

        model = SVMClassifier(C=1.0, kernel="rbf", probability=True, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, classification_data: tuple) -> None:
        """Test probability predictions."""
        X, y = classification_data

        model = SVMClassifier(C=1.0, kernel="rbf", probability=True, random_state=42)
        model.fit(X, y)

        probabilities = model.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=0.01)

    def test_linear_kernel(self, classification_data: tuple) -> None:
        """Test SVM with linear kernel."""
        X, y = classification_data

        model = SVMClassifier(C=1.0, kernel="linear", random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)


class TestXGBoostClassifier:
    """Test suite for XGBoostClassifier."""

    def test_import(self) -> None:
        """Test that XGBoostClassifier can be imported."""
        from omicselector2.models.classical import XGBoostClassifier

        assert XGBoostClassifier is not None

    def test_initialization(self) -> None:
        """Test XGBoostClassifier initialization."""
        model = XGBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.random_state == 42

    def test_fit_predict(self, classification_data: tuple) -> None:
        """Test fitting and prediction."""
        X, y = classification_data

        model = XGBoostClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, classification_data: tuple) -> None:
        """Test probability predictions."""
        X, y = classification_data

        model = XGBoostClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        probabilities = model.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=0.01)

    def test_feature_importance(self, classification_data: tuple) -> None:
        """Test feature importance extraction."""
        X, y = classification_data

        model = XGBoostClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)

    def test_multiclass(self, multiclass_data: tuple) -> None:
        """Test XGBoostClassifier on multiclass problem."""
        X, y = multiclass_data

        model = XGBoostClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert set(predictions).issubset({0, 1, 2})
        assert probabilities.shape == (len(X), 3)


class TestXGBoostRegressor:
    """Test suite for XGBoostRegressor."""

    def test_import(self) -> None:
        """Test that XGBoostRegressor can be imported."""
        from omicselector2.models.classical import XGBoostRegressor

        assert XGBoostRegressor is not None

    def test_fit_predict(self, regression_data: tuple) -> None:
        """Test fitting and prediction."""
        X, y = regression_data

        model = XGBoostRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert predictions.dtype in [np.float32, np.float64]

    def test_feature_importance(self, regression_data: tuple) -> None:
        """Test feature importance extraction."""
        X, y = regression_data

        model = XGBoostRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)
