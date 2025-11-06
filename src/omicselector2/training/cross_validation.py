"""Cross-validation framework for model evaluation.

This module provides robust cross-validation infrastructure for evaluating
feature selection methods and predictive models. It implements various
splitting strategies with reproducibility guarantees.

Key features:
- K-Fold cross-validation
- Stratified K-Fold (preserves class distribution)
- Train/test/validation splitting
- Reproducible splits via random_state
- Integration with pandas DataFrames
- Efficient numpy-based indexing

Based on:
- scikit-learn's cross-validation API
- OmicSelector 1.0's validation philosophy
- Clinical ML best practices (stratification, hold-out sets)

Examples:
    >>> from omicselector2.training.cross_validation import CrossValidator
    >>>
    >>> # K-Fold cross-validation
    >>> cv = CrossValidator(cv_type="kfold", n_splits=5, random_state=42)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    ...     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    ...     # Train and evaluate model
    >>>
    >>> # Train/test/validation split
    >>> cv = CrossValidator(cv_type="train_test_val", test_size=0.2, val_size=0.2)
    >>> train_idx, test_idx, val_idx = cv.get_train_test_val(X, y)
"""

from typing import Generator, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class KFoldSplitter:
    """K-Fold cross-validation splitter.

    Splits data into k consecutive folds. Each fold is used once as a test set
    while the remaining k-1 folds form the training set.

    Attributes:
        n_splits: Number of folds (must be >= 2).
        shuffle: Whether to shuffle data before splitting.
        random_state: Random seed for reproducibility (if shuffle=True).

    Examples:
        >>> splitter = KFoldSplitter(n_splits=5, random_state=42)
        >>> for train_idx, test_idx in splitter.split(X, y):
        ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize K-Fold splitter.

        Args:
            n_splits: Number of folds (must be >= 2).
            shuffle: Whether to shuffle data before splitting. Default True.
            random_state: Random seed for reproducibility. Only used if shuffle=True.

        Raises:
            ValueError: If n_splits < 2.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Generator[tuple[NDArray, NDArray], None, None]:
        """Generate train/test indices for k-fold cross-validation.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable (optional, not used by KFold but kept for API consistency).

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        # Split indices into n_splits folds
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop


class StratifiedKFoldSplitter:
    """Stratified K-Fold cross-validation splitter.

    Splits data into k folds while preserving the class distribution in each fold.
    Essential for imbalanced classification problems.

    Attributes:
        n_splits: Number of folds (must be >= 2).
        shuffle: Whether to shuffle data before splitting.
        random_state: Random seed for reproducibility.

    Examples:
        >>> splitter = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        >>> for train_idx, test_idx in splitter.split(X, y):
        ...     # Class distribution is preserved in each fold
        ...     print(y.iloc[test_idx].value_counts(normalize=True))
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Stratified K-Fold splitter.

        Args:
            n_splits: Number of folds (must be >= 2).
            shuffle: Whether to shuffle data before splitting. Default True.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If n_splits < 2.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Generator[tuple[NDArray, NDArray], None, None]:
        """Generate stratified train/test indices for k-fold cross-validation.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable (required for stratification).

        Yields:
            Tuple of (train_indices, test_indices) for each fold.

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError("y is required for stratified splitting")

        n_samples = len(X)
        rng = np.random.RandomState(self.random_state) if self.shuffle else None

        # Get class labels and their indices
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        # Count samples per class
        class_counts = np.bincount(y_indices)

        # Create indices per class
        class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]

        # Shuffle indices within each class
        if self.shuffle and rng is not None:
            for indices in class_indices:
                rng.shuffle(indices)

        # Distribute samples into folds
        fold_indices = [[] for _ in range(self.n_splits)]

        for class_idx, indices in enumerate(class_indices):
            # Split this class's samples into n_splits folds
            n_class_samples = len(indices)
            fold_sizes = np.full(
                self.n_splits, n_class_samples // self.n_splits, dtype=int
            )
            fold_sizes[: n_class_samples % self.n_splits] += 1

            current = 0
            for fold_idx, fold_size in enumerate(fold_sizes):
                start, stop = current, current + fold_size
                fold_indices[fold_idx].extend(indices[start:stop])
                current = stop

        # Generate train/test splits
        for test_fold_idx in range(self.n_splits):
            test_idx = np.array(fold_indices[test_fold_idx])
            train_idx = np.concatenate(
                [
                    fold_indices[i]
                    for i in range(self.n_splits)
                    if i != test_fold_idx
                ]
            )
            yield train_idx, test_idx


class TrainTestValSplitter:
    """Train/test/validation splitter.

    Splits data into train, test, and validation sets with configurable sizes.
    Optionally preserves class distribution (stratification).

    Attributes:
        test_size: Proportion of data for test set (0-1).
        val_size: Proportion of data for validation set (0-1).
        stratify: Whether to preserve class distribution.
        random_state: Random seed for reproducibility.

    Examples:
        >>> splitter = TrainTestValSplitter(
        ...     test_size=0.2, val_size=0.2, stratify=True, random_state=42
        ... )
        >>> train_idx, test_idx, val_idx = splitter.split(X, y)
        >>> X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        >>> X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        >>> X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.0,
        stratify: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Train/Test/Validation splitter.

        Args:
            test_size: Proportion of data for test set (0-1). Default 0.2.
            val_size: Proportion of data for validation set (0-1). Default 0.0.
            stratify: Whether to preserve class distribution. Default False.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If test_size or val_size are out of range or sum >= 1.
        """
        if not (0 <= test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        if not (0 <= val_size < 1):
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")

        if test_size + val_size >= 1:
            raise ValueError(
                f"test_size ({test_size}) + val_size ({val_size}) must sum to less than 1"
            )

        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Split data into train/test/validation sets.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable (required if stratify=True).

        Returns:
            Tuple of (train_indices, test_indices, val_indices).

        Raises:
            ValueError: If stratify=True but y is None.
        """
        if self.stratify and y is None:
            raise ValueError("y is required when stratify=True")

        n_samples = len(X)
        indices = np.arange(n_samples)

        rng = np.random.RandomState(self.random_state)

        if self.stratify and y is not None:
            # Stratified split
            train_idx, test_idx, val_idx = self._stratified_split(
                indices, y.values, rng
            )
        else:
            # Random split
            rng.shuffle(indices)

            # Calculate split points
            n_test = int(n_samples * self.test_size)
            n_val = int(n_samples * self.val_size)

            test_idx = indices[:n_test]
            val_idx = indices[n_test : n_test + n_val]
            train_idx = indices[n_test + n_val :]

        return train_idx, test_idx, val_idx

    def _stratified_split(
        self, indices: NDArray, y: NDArray, rng: np.random.RandomState
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Perform stratified split preserving class distribution.

        Args:
            indices: Sample indices.
            y: Target labels.
            rng: Random number generator.

        Returns:
            Tuple of (train_indices, test_indices, val_indices).
        """
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        train_indices = []
        test_indices = []
        val_indices = []

        # Split each class separately
        for class_idx in range(n_classes):
            class_mask = y_indices == class_idx
            class_indices = indices[class_mask]

            # Shuffle within class
            rng.shuffle(class_indices)

            n_class = len(class_indices)
            n_test = int(n_class * self.test_size)
            n_val = int(n_class * self.val_size)

            test_indices.extend(class_indices[:n_test])
            val_indices.extend(class_indices[n_test : n_test + n_val])
            train_indices.extend(class_indices[n_test + n_val :])

        return (
            np.array(train_indices),
            np.array(test_indices),
            np.array(val_indices),
        )


class CrossValidator:
    """Unified cross-validation interface.

    Provides a single API for different cross-validation strategies:
    - K-Fold cross-validation
    - Stratified K-Fold
    - Train/test/validation splitting

    Attributes:
        cv_type: Type of cross-validation ("kfold", "stratified", "train_test_val").
        n_splits: Number of splits (for kfold/stratified).
        test_size: Test set proportion (for train_test_val).
        val_size: Validation set proportion (for train_test_val).
        stratify: Whether to preserve class distribution (for train_test_val).
        random_state: Random seed for reproducibility.

    Examples:
        >>> # K-Fold cross-validation
        >>> cv = CrossValidator(cv_type="kfold", n_splits=5, random_state=42)
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     # Train and evaluate
        ...     pass
        >>>
        >>> # Stratified K-Fold
        >>> cv = CrossValidator(cv_type="stratified", n_splits=5, random_state=42)
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     # Class distribution preserved
        ...     pass
        >>>
        >>> # Train/test/validation
        >>> cv = CrossValidator(cv_type="train_test_val", test_size=0.2, val_size=0.2)
        >>> train_idx, test_idx, val_idx = cv.get_train_test_val(X, y)
    """

    VALID_CV_TYPES = ["kfold", "stratified", "train_test_val"]

    def __init__(
        self,
        cv_type: Literal["kfold", "stratified", "train_test_val"] = "kfold",
        n_splits: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.0,
        stratify: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize CrossValidator.

        Args:
            cv_type: Type of cross-validation. Must be one of:
                - "kfold": K-Fold cross-validation
                - "stratified": Stratified K-Fold
                - "train_test_val": Train/test/validation split
            n_splits: Number of splits (for kfold/stratified). Default 5.
            test_size: Test set proportion (for train_test_val). Default 0.2.
            val_size: Validation set proportion (for train_test_val). Default 0.0.
            stratify: Preserve class distribution (for train_test_val). Default False.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If cv_type is invalid.
        """
        if cv_type not in self.VALID_CV_TYPES:
            raise ValueError(
                f"cv_type must be one of {self.VALID_CV_TYPES}, got '{cv_type}'"
            )

        self.cv_type = cv_type
        self.n_splits = n_splits
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.random_state = random_state

        # Initialize appropriate splitter
        if cv_type == "kfold":
            self._splitter: Union[
                KFoldSplitter, StratifiedKFoldSplitter, TrainTestValSplitter
            ] = KFoldSplitter(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        elif cv_type == "stratified":
            self._splitter = StratifiedKFoldSplitter(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        elif cv_type == "train_test_val":
            self._splitter = TrainTestValSplitter(
                test_size=test_size,
                val_size=val_size,
                stratify=stratify,
                random_state=random_state,
            )

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Generator[tuple[NDArray, NDArray], None, None]:
        """Generate train/test indices for cross-validation.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable (required for stratified CV).

        Yields:
            Tuple of (train_indices, test_indices) for each fold.

        Raises:
            ValueError: If cv_type is train_test_val (use get_train_test_val instead).
        """
        if self.cv_type == "train_test_val":
            raise ValueError(
                "Use get_train_test_val() method for train_test_val cv_type"
            )

        yield from self._splitter.split(X, y)

    def get_train_test_val(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Get train/test/validation split.

        Args:
            X: Feature matrix (samples × features).
            y: Target variable (required if stratify=True).

        Returns:
            Tuple of (train_indices, test_indices, val_indices).

        Raises:
            ValueError: If cv_type is not train_test_val.
        """
        if self.cv_type != "train_test_val":
            raise ValueError(
                f"get_train_test_val() only available for cv_type='train_test_val', "
                f"got '{self.cv_type}'"
            )

        return self._splitter.split(X, y)  # type: ignore
