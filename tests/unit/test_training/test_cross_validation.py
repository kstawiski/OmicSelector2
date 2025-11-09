"""Tests for cross-validation framework.

The cross-validation framework provides robust evaluation infrastructure:
- K-Fold cross-validation
- Stratified K-Fold (for classification)
- Train/test/validation splitting
- Reproducibility guarantees
- Integration with pandas DataFrames

Test coverage:
- Basic k-fold functionality
- Stratified splits for classification
- Train/test/validation splits
- Reproducibility
- Edge cases and error handling
"""

import os

import numpy as np
import pandas as pd
import pytest

# Set required environment variables
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-32chars!!"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from omicselector2.training.cross_validation import (  # noqa: E402
    CrossValidator,
    KFoldSplitter,
    StratifiedKFoldSplitter,
    TrainTestValSplitter,
)


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification data for testing.

    Returns:
        Tuple of (X, y) where X is features and y is binary target.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Balanced binary classification
    y = pd.Series(np.random.binomial(1, 0.5, n_samples), name="target")

    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression data for testing.

    Returns:
        Tuple of (X, y) where X is features and y is continuous target.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(np.random.randn(n_samples), name="target")

    return X, y


class TestKFoldSplitter:
    """Test suite for KFoldSplitter."""

    def test_import(self) -> None:
        """Test that KFoldSplitter can be imported."""
        from omicselector2.training.cross_validation import KFoldSplitter

        assert KFoldSplitter is not None

    def test_initialization(self) -> None:
        """Test KFoldSplitter initialization."""
        splitter = KFoldSplitter(n_splits=5)
        assert splitter.n_splits == 5
        assert splitter.shuffle is True  # Default
        assert splitter.random_state is None

    def test_split(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test basic k-fold splitting."""
        X, y = sample_classification_data

        splitter = KFoldSplitter(n_splits=5, random_state=42)
        folds = list(splitter.split(X, y))

        # Should return 5 folds
        assert len(folds) == 5

        # Each fold should have train and test indices
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

            # No overlap between train and test
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # All indices should be covered
            all_idx = np.concatenate([train_idx, test_idx])
            assert len(all_idx) == len(X)
            assert set(all_idx) == set(range(len(X)))

            # Test set should be roughly 20% (1/5)
            assert abs(len(test_idx) / len(X) - 0.2) < 0.05

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that same random_state produces same splits."""
        X, y = sample_classification_data

        splitter1 = KFoldSplitter(n_splits=5, random_state=42)
        folds1 = list(splitter1.split(X, y))

        splitter2 = KFoldSplitter(n_splits=5, random_state=42)
        folds2 = list(splitter2.split(X, y))

        # Should produce identical splits
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_different_n_splits(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test different number of splits."""
        X, y = sample_classification_data

        for n_splits in [3, 5, 10]:
            splitter = KFoldSplitter(n_splits=n_splits, random_state=42)
            folds = list(splitter.split(X, y))

            assert len(folds) == n_splits

            # Each test set should be roughly 1/n_splits
            for _, test_idx in folds:
                expected_ratio = 1.0 / n_splits
                actual_ratio = len(test_idx) / len(X)
                assert abs(actual_ratio - expected_ratio) < 0.05

    def test_invalid_n_splits(self) -> None:
        """Test that invalid n_splits raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            KFoldSplitter(n_splits=1)

        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            KFoldSplitter(n_splits=0)


class TestStratifiedKFoldSplitter:
    """Test suite for StratifiedKFoldSplitter."""

    def test_import(self) -> None:
        """Test that StratifiedKFoldSplitter can be imported."""
        from omicselector2.training.cross_validation import StratifiedKFoldSplitter

        assert StratifiedKFoldSplitter is not None

    def test_initialization(self) -> None:
        """Test StratifiedKFoldSplitter initialization."""
        splitter = StratifiedKFoldSplitter(n_splits=5)
        assert splitter.n_splits == 5

    def test_stratified_split(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that splits preserve class distribution."""
        X, y = sample_classification_data

        splitter = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        folds = list(splitter.split(X, y))

        # Original class distribution
        original_dist = y.value_counts(normalize=True).sort_index()

        # Check each fold
        for train_idx, test_idx in folds:
            train_y = y.iloc[train_idx]
            test_y = y.iloc[test_idx]

            # Train and test should preserve distribution
            train_dist = train_y.value_counts(normalize=True).sort_index()
            test_dist = test_y.value_counts(normalize=True).sort_index()

            # Check distributions are close to original (within 5%)
            for cls in original_dist.index:
                assert abs(train_dist[cls] - original_dist[cls]) < 0.05
                assert abs(test_dist[cls] - original_dist[cls]) < 0.05

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test reproducibility with random_state."""
        X, y = sample_classification_data

        splitter1 = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        folds1 = list(splitter1.split(X, y))

        splitter2 = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        folds2 = list(splitter2.split(X, y))

        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_multiclass_stratification(self) -> None:
        """Test stratification with multiple classes."""
        np.random.seed(42)
        n_samples = 300
        n_features = 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # 3-class problem with imbalanced distribution
        y = pd.Series(np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]))

        splitter = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        folds = list(splitter.split(X, y))

        original_dist = y.value_counts(normalize=True).sort_index()

        for train_idx, test_idx in folds:
            test_y = y.iloc[test_idx]
            test_dist = test_y.value_counts(normalize=True).sort_index()

            # Each class should be represented in test set
            assert len(test_dist) == 3

            # Distribution should be close to original
            for cls in original_dist.index:
                assert abs(test_dist[cls] - original_dist[cls]) < 0.1


class TestTrainTestValSplitter:
    """Test suite for TrainTestValSplitter."""

    def test_import(self) -> None:
        """Test that TrainTestValSplitter can be imported."""
        from omicselector2.training.cross_validation import TrainTestValSplitter

        assert TrainTestValSplitter is not None

    def test_initialization(self) -> None:
        """Test TrainTestValSplitter initialization."""
        splitter = TrainTestValSplitter(test_size=0.2, val_size=0.2)
        assert splitter.test_size == 0.2
        assert splitter.val_size == 0.2

    def test_train_test_val_split(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test train/test/validation splitting."""
        X, y = sample_classification_data

        splitter = TrainTestValSplitter(
            test_size=0.2, val_size=0.2, random_state=42
        )
        train_idx, test_idx, val_idx = splitter.split(X, y)

        # Check sizes
        assert len(train_idx) + len(test_idx) + len(val_idx) == len(X)

        # Check proportions (within 2% tolerance)
        assert abs(len(test_idx) / len(X) - 0.2) < 0.02
        assert abs(len(val_idx) / len(X) - 0.2) < 0.02
        assert abs(len(train_idx) / len(X) - 0.6) < 0.02

        # No overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(np.intersect1d(test_idx, val_idx)) == 0

    def test_train_test_only(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test train/test splitting without validation set."""
        X, y = sample_classification_data

        splitter = TrainTestValSplitter(test_size=0.2, val_size=0.0, random_state=42)
        train_idx, test_idx, val_idx = splitter.split(X, y)

        # Validation should be empty
        assert len(val_idx) == 0

        # Train and test should cover all data
        assert len(train_idx) + len(test_idx) == len(X)
        assert abs(len(test_idx) / len(X) - 0.2) < 0.02

    def test_stratified_split(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test stratified train/test/val splitting."""
        X, y = sample_classification_data

        splitter = TrainTestValSplitter(
            test_size=0.2, val_size=0.2, stratify=True, random_state=42
        )
        train_idx, test_idx, val_idx = splitter.split(X, y)

        # Check class distributions
        original_dist = y.value_counts(normalize=True).sort_index()

        train_dist = y.iloc[train_idx].value_counts(normalize=True).sort_index()
        test_dist = y.iloc[test_idx].value_counts(normalize=True).sort_index()
        val_dist = y.iloc[val_idx].value_counts(normalize=True).sort_index()

        # All splits should preserve distribution (within 5%)
        for cls in original_dist.index:
            assert abs(train_dist[cls] - original_dist[cls]) < 0.05
            assert abs(test_dist[cls] - original_dist[cls]) < 0.05
            assert abs(val_dist[cls] - original_dist[cls]) < 0.05

    def test_reproducibility(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test reproducibility with random_state."""
        X, y = sample_classification_data

        splitter1 = TrainTestValSplitter(
            test_size=0.2, val_size=0.2, random_state=42
        )
        train1, test1, val1 = splitter1.split(X, y)

        splitter2 = TrainTestValSplitter(
            test_size=0.2, val_size=0.2, random_state=42
        )
        train2, test2, val2 = splitter2.split(X, y)

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)
        np.testing.assert_array_equal(val1, val2)

    def test_invalid_sizes(self) -> None:
        """Test that invalid sizes raise ValueError."""
        with pytest.raises(ValueError, match="test_size.*must be between 0 and 1"):
            TrainTestValSplitter(test_size=1.5, val_size=0.2)

        with pytest.raises(ValueError, match="val_size.*must be between 0 and 1"):
            TrainTestValSplitter(test_size=0.2, val_size=1.5)

        with pytest.raises(
            ValueError, match="test_size.*val_size.*must sum to less than 1"
        ):
            TrainTestValSplitter(test_size=0.6, val_size=0.6)


class TestCrossValidator:
    """Test suite for CrossValidator."""

    def test_import(self) -> None:
        """Test that CrossValidator can be imported."""
        from omicselector2.training.cross_validation import CrossValidator

        assert CrossValidator is not None

    def test_kfold_cv(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test k-fold cross-validation."""
        X, y = sample_classification_data

        cv = CrossValidator(cv_type="kfold", n_splits=5, random_state=42)
        folds = list(cv.split(X, y))

        assert len(folds) == 5

    def test_stratified_cv(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test stratified k-fold cross-validation."""
        X, y = sample_classification_data

        cv = CrossValidator(cv_type="stratified", n_splits=5, random_state=42)
        folds = list(cv.split(X, y))

        assert len(folds) == 5

        # Check stratification
        original_dist = y.value_counts(normalize=True)
        for train_idx, test_idx in folds:
            test_dist = y.iloc[test_idx].value_counts(normalize=True)
            for cls in original_dist.index:
                assert abs(test_dist[cls] - original_dist[cls]) < 0.1

    def test_get_train_test_val(
        self, sample_classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test getting train/test/val split."""
        X, y = sample_classification_data

        cv = CrossValidator(cv_type="train_test_val", test_size=0.2, val_size=0.2)
        train_idx, test_idx, val_idx = cv.get_train_test_val(X, y)

        assert len(train_idx) + len(test_idx) + len(val_idx) == len(X)

    def test_invalid_cv_type(self) -> None:
        """Test that invalid cv_type raises ValueError."""
        with pytest.raises(ValueError, match="cv_type must be one of"):
            CrossValidator(cv_type="invalid")
