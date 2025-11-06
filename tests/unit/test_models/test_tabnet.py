"""Unit tests for TabNet models.

TabNet is a deep learning architecture for tabular data that uses sequential
attention to select features at each decision step, providing both high
performance and interpretability for high-dimensional omics data.

Reference:
    Arik, S.Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable
    Tabular Learning. AAAI 2021.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Set test environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.models.neural.tabnet_models import (  # noqa: E402
    TabNetClassifier,
    TabNetRegressor,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.binomial(1, 0.5, n_samples), name="target")

    return X, y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(X.iloc[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.1, name="target")

    return X, y


class TestTabNetClassifier:
    """Test suite for TabNetClassifier."""

    def test_import(self) -> None:
        """Test that TabNetClassifier can be imported."""
        assert TabNetClassifier is not None

    def test_initialization_defaults(self) -> None:
        """Test TabNetClassifier initialization with default parameters."""
        model = TabNetClassifier()
        assert model is not None
        assert model.n_d == 8
        assert model.n_a == 8
        assert model.n_steps == 3
        assert model.gamma == 1.3
        assert model.max_epochs == 100
        assert not model.is_fitted_

    def test_initialization_custom_params(self) -> None:
        """Test TabNetClassifier with custom parameters."""
        model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            max_epochs=50,
            batch_size=128,
            verbose=True,
        )
        assert model.n_d == 16
        assert model.n_a == 16
        assert model.n_steps == 5
        assert model.gamma == 1.5
        assert model.max_epochs == 50
        assert model.batch_size == 128
        assert model.verbose

    def test_fit(self, classification_data: tuple) -> None:
        """Test fit method."""
        X, y = classification_data
        model = TabNetClassifier(max_epochs=10, verbose=False)

        result = model.fit(X, y)

        assert result is model
        assert model.is_fitted_
        assert hasattr(model, "model_")
        assert hasattr(model, "classes_")
        assert hasattr(model, "feature_names_")
        assert len(model.classes_) == 2
        assert len(model.feature_names_) == X.shape[1]

    def test_predict(self, classification_data: tuple) -> None:
        """Test predict method."""
        X, y = classification_data
        model = TabNetClassifier(max_epochs=10, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_proba(self, classification_data: tuple) -> None:
        """Test predict_proba method."""
        X, y = classification_data
        model = TabNetClassifier(max_epochs=10, verbose=False)
        model.fit(X, y)

        probas = model.predict_proba(X)

        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all((probas >= 0) & (probas <= 1))

    def test_get_feature_importance(self, classification_data: tuple) -> None:
        """Test feature importance extraction via attention masks."""
        X, y = classification_data
        model = TabNetClassifier(max_epochs=10, verbose=False)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]
        assert all(importance.index == X.columns)
        # Importance values should be non-negative
        assert all(importance >= 0)
        # Should sum to approximately 1
        assert abs(importance.sum() - 1.0) < 0.1

    def test_model_persistence(
        self, classification_data: tuple, tmp_path: Path
    ) -> None:
        """Test model save and load."""
        X, y = classification_data
        model = TabNetClassifier(max_epochs=10, verbose=False)
        model.fit(X, y)

        # Save model
        save_path = tmp_path / "tabnet_classifier.pkl"
        model.save(save_path)
        assert save_path.exists()

        # Load model
        loaded_model = TabNetClassifier.load(save_path)
        assert loaded_model.is_fitted_

        # Verify predictions match
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_predict_before_fit_raises_error(
        self, classification_data: tuple
    ) -> None:
        """Test that predict before fit raises error."""
        X, _ = classification_data
        model = TabNetClassifier()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)

    def test_metadata_tracking(self, classification_data: tuple) -> None:
        """Test that metadata is tracked during training."""
        X, y = classification_data
        model = TabNetClassifier(max_epochs=10, verbose=False)
        model.fit(X, y)

        assert "training_time" in model.metadata
        assert "n_samples" in model.metadata
        assert "n_features" in model.metadata
        assert model.metadata["n_samples"] == len(X)
        assert model.metadata["n_features"] == X.shape[1]


class TestTabNetRegressor:
    """Test suite for TabNetRegressor."""

    def test_import(self) -> None:
        """Test that TabNetRegressor can be imported."""
        assert TabNetRegressor is not None

    def test_initialization_defaults(self) -> None:
        """Test TabNetRegressor initialization with default parameters."""
        model = TabNetRegressor()
        assert model is not None
        assert model.n_d == 8
        assert model.n_a == 8
        assert model.n_steps == 3
        assert not model.is_fitted_

    def test_fit(self, regression_data: tuple) -> None:
        """Test fit method."""
        X, y = regression_data
        model = TabNetRegressor(max_epochs=10, verbose=False)

        result = model.fit(X, y)

        assert result is model
        assert model.is_fitted_
        assert hasattr(model, "model_")
        assert hasattr(model, "feature_names_")

    def test_predict(self, regression_data: tuple) -> None:
        """Test predict method."""
        X, y = regression_data
        model = TabNetRegressor(max_epochs=10, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert predictions.dtype == np.float64

    def test_get_feature_importance(self, regression_data: tuple) -> None:
        """Test feature importance extraction."""
        X, y = regression_data
        model = TabNetRegressor(max_epochs=10, verbose=False)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]
        assert all(importance >= 0)

    def test_model_persistence(self, regression_data: tuple, tmp_path: Path) -> None:
        """Test model save and load."""
        X, y = regression_data
        model = TabNetRegressor(max_epochs=10, verbose=False)
        model.fit(X, y)

        # Save model
        save_path = tmp_path / "tabnet_regressor.pkl"
        model.save(save_path)
        assert save_path.exists()

        # Load model
        loaded_model = TabNetRegressor.load(save_path)
        assert loaded_model.is_fitted_

        # Verify predictions match
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)

    def test_metadata_tracking(self, regression_data: tuple) -> None:
        """Test that metadata is tracked during training."""
        X, y = regression_data
        model = TabNetRegressor(max_epochs=10, verbose=False)
        model.fit(X, y)

        assert "training_time" in model.metadata
        assert "n_samples" in model.metadata
        assert "n_features" in model.metadata


class TestTabNetBenchmarkingIntegration:
    """Test TabNet integration with benchmarking system."""

    def test_tabnet_in_benchmarking(self, classification_data: tuple) -> None:
        """Test that TabNet can be used in benchmarking workflow."""
        from omicselector2.training.benchmarking import Benchmarker

        X, y = classification_data

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create benchmarker
        benchmarker = Benchmarker(cv_folds=3, random_state=42, verbose=False)

        # Benchmark with TabNet
        signatures = {
            "all_features": X_train.columns.tolist(),
            "top_10": X_train.columns[:10].tolist(),
        }

        models = ["RandomForest", "TabNet"]  # Include TabNet

        results = benchmarker.benchmark_signatures(
            signatures=signatures,
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Verify TabNet results are present
        tabnet_results = [r for r in results if r.model_name == "TabNet"]
        assert len(tabnet_results) == 2  # One per signature

        # Verify metrics are computed
        for result in tabnet_results:
            assert "accuracy" in result.cv_metrics
            assert "auc_roc" in result.cv_metrics
            assert "accuracy" in result.test_metrics
