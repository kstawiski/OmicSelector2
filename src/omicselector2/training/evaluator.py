"""Model evaluation metrics for classification, regression, and survival analysis.

This module provides comprehensive evaluation metrics for assessing model performance:
- Classification: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, confusion matrix
- Regression: MSE, RMSE, MAE, R², Pearson correlation
- Survival: C-index (concordance index), Integrated Brier Score

Key features:
- Binary and multiclass classification support
- Macro/micro/weighted averaging for multiclass
- Per-class metrics
- Survival analysis metrics (C-index)
- Comprehensive error handling
- Efficient numpy-based computation

Based on:
- scikit-learn metrics API
- lifelines survival analysis library
- Clinical ML evaluation standards

Examples:
    >>> from omicselector2.training.evaluator import ClassificationEvaluator
    >>>
    >>> # Binary classification
    >>> evaluator = ClassificationEvaluator()
    >>> metrics = evaluator.evaluate(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    >>> print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    >>>
    >>> # Multiclass classification
    >>> metrics = evaluator.evaluate(y_true, y_pred, multiclass=True, average="macro")
    >>> print(f"F1 (macro): {metrics['f1_macro']:.3f}")
"""

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class ClassificationEvaluator:
    """Evaluator for classification tasks.

    Computes standard classification metrics including accuracy, precision, recall,
    F1 score, AUC-ROC, AUC-PR, and confusion matrix.

    Supports both binary and multiclass classification with various averaging strategies.

    Examples:
        >>> evaluator = ClassificationEvaluator()
        >>>
        >>> # Binary classification
        >>> metrics = evaluator.evaluate(y_true, y_pred)
        >>> print(metrics['accuracy'], metrics['f1'])
        >>>
        >>> # With probabilities for AUC
        >>> metrics = evaluator.evaluate(y_true, y_score, probabilities=True)
        >>> print(metrics['auc_roc'])
        >>>
        >>> # Multiclass with macro averaging
        >>> metrics = evaluator.evaluate(
        ...     y_true, y_pred, multiclass=True, average="macro"
        ... )
    """

    def evaluate(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        probabilities: bool = False,
        multiclass: bool = False,
        average: Literal["micro", "macro", "weighted"] = "macro",
        per_class: bool = False,
    ) -> dict[str, Any]:
        """Evaluate classification predictions.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels (or probabilities if probabilities=True).
            probabilities: Whether y_pred contains probabilities instead of labels.
            multiclass: Whether this is multiclass classification (>2 classes).
            average: Averaging strategy for multiclass ("micro", "macro", "weighted").
            per_class: Whether to compute per-class metrics.

        Returns:
            Dictionary of metrics.

        Raises:
            ValueError: If input shapes don't match or inputs are invalid.
        """
        # Validate inputs
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if not probabilities:
            if y_true.shape[0] != y_pred.shape[0]:
                raise ValueError(
                    f"shape mismatch: y_true has {y_true.shape[0]} samples, "
                    f"y_pred has {y_pred.shape[0]} samples"
                )
        else:
            # For probabilities, y_pred might be 2D
            if y_pred.ndim == 2:
                if y_true.shape[0] != y_pred.shape[0]:
                    raise ValueError(
                        f"shape mismatch: y_true has {y_true.shape[0]} samples, "
                        f"y_pred has {y_pred.shape[0]} samples"
                    )
            else:
                if y_true.shape != y_pred.shape:
                    raise ValueError(
                        f"shape mismatch: y_true shape {y_true.shape}, "
                        f"y_pred shape {y_pred.shape}"
                    )

        metrics: dict[str, Any] = {}

        # For probability predictions, compute AUC metrics
        if probabilities:
            if multiclass:
                # Multiclass AUC (one-vs-rest)
                metrics["auc_ovr"] = self._compute_multiclass_auc(y_true, y_pred)
            else:
                # Binary AUC-ROC and AUC-PR
                if y_pred.ndim == 2:
                    y_score = y_pred[:, 1]  # Probability of positive class
                else:
                    y_score = y_pred

                metrics["auc_roc"] = self._compute_auc_roc(y_true, y_score)
                metrics["auc_pr"] = self._compute_auc_pr(y_true, y_score)

        # Convert probabilities to predicted classes if needed
        if probabilities:
            if y_pred.ndim == 2:
                y_pred_class = np.argmax(y_pred, axis=1)
            else:
                y_pred_class = (y_pred >= 0.5).astype(int)
        else:
            y_pred_class = y_pred

        # Compute standard classification metrics
        metrics["accuracy"] = self._compute_accuracy(y_true, y_pred_class)

        if multiclass:
            # Multiclass metrics
            precision, recall, f1 = self._compute_multiclass_metrics(
                y_true, y_pred_class, average
            )
            metrics[f"precision_{average}"] = precision
            metrics[f"recall_{average}"] = recall
            metrics[f"f1_{average}"] = f1
        else:
            # Binary metrics
            precision, recall, f1 = self._compute_binary_metrics(y_true, y_pred_class)
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1

        # Confusion matrix
        metrics["confusion_matrix"] = self._compute_confusion_matrix(
            y_true, y_pred_class
        )

        # Per-class metrics if requested
        if per_class:
            per_class_metrics = self._compute_per_class_metrics(y_true, y_pred_class)
            metrics["per_class_precision"] = per_class_metrics["precision"]
            metrics["per_class_recall"] = per_class_metrics["recall"]
            metrics["per_class_f1"] = per_class_metrics["f1"]

        return metrics

    def _compute_accuracy(self, y_true: NDArray, y_pred: NDArray) -> float:
        """Compute accuracy."""
        return float(np.mean(y_true == y_pred))

    def _compute_binary_metrics(
        self, y_true: NDArray, y_pred: NDArray
    ) -> tuple[float, float, float]:
        """Compute binary precision, recall, and F1."""
        # True positives, false positives, false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return float(precision), float(recall), float(f1)

    def _compute_multiclass_metrics(
        self, y_true: NDArray, y_pred: NDArray, average: str
    ) -> tuple[float, float, float]:
        """Compute multiclass precision, recall, and F1."""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        # Compute per-class metrics
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for cls in classes:
            # Binary mask for this class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            # Compute binary metrics
            precision, recall, f1 = self._compute_binary_metrics(
                y_true_binary, y_pred_binary
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(np.sum(y_true == cls))

        # Average based on strategy
        if average == "macro":
            # Unweighted mean
            precision_avg = np.mean(precisions)
            recall_avg = np.mean(recalls)
            f1_avg = np.mean(f1s)
        elif average == "weighted":
            # Weighted by support
            weights = np.array(supports) / np.sum(supports)
            precision_avg = np.sum(np.array(precisions) * weights)
            recall_avg = np.sum(np.array(recalls) * weights)
            f1_avg = np.sum(np.array(f1s) * weights)
        elif average == "micro":
            # Compute globally
            tp_total = 0
            fp_total = 0
            fn_total = 0

            for cls in classes:
                y_true_binary = (y_true == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)

                tp_total += np.sum((y_true_binary == 1) & (y_pred_binary == 1))
                fp_total += np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                fn_total += np.sum((y_true_binary == 1) & (y_pred_binary == 0))

            precision_avg = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall_avg = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1_avg = (
                2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
                if (precision_avg + recall_avg) > 0
                else 0.0
            )
        else:
            raise ValueError(f"Invalid average: {average}")

        return float(precision_avg), float(recall_avg), float(f1_avg)

    def _compute_per_class_metrics(
        self, y_true: NDArray, y_pred: NDArray
    ) -> dict[str, list[float]]:
        """Compute per-class precision, recall, and F1."""
        classes = np.unique(np.concatenate([y_true, y_pred]))

        precisions = []
        recalls = []
        f1s = []

        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            precision, recall, f1 = self._compute_binary_metrics(
                y_true_binary, y_pred_binary
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {"precision": precisions, "recall": recalls, "f1": f1s}

    def _compute_confusion_matrix(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """Compute confusion matrix."""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        # Create mapping from class labels to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Initialize confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)

        # Fill confusion matrix
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1

        return cm

    def _compute_auc_roc(self, y_true: NDArray, y_score: NDArray) -> float:
        """Compute AUC-ROC for binary classification."""
        # Sort by predicted score (descending)
        desc_score_indices = np.argsort(y_score)[::-1]
        y_score_sorted = y_score[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]

        # Compute TPR and FPR at each threshold
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        # Total positives and negatives
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5  # Undefined, return random guess

        # Compute TPR and FPR
        tpr = tps / n_pos
        fpr = fps / n_neg

        # Add (0, 0) and (1, 1) points
        tpr = np.concatenate([[0], tpr, [1]])
        fpr = np.concatenate([[0], fpr, [1]])

        # Compute AUC using trapezoidal rule
        auc = float(np.trapz(tpr, fpr))

        return auc

    def _compute_auc_pr(self, y_true: NDArray, y_score: NDArray) -> float:
        """Compute AUC-PR for binary classification."""
        # Sort by predicted score (descending)
        desc_score_indices = np.argsort(y_score)[::-1]
        y_score_sorted = y_score[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]

        # Compute precision and recall at each threshold
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        n_pos = np.sum(y_true)

        if n_pos == 0:
            return 0.0

        # Precision and recall
        precisions = tps / (tps + fps)
        precisions[np.isnan(precisions)] = 0.0

        recalls = tps / n_pos

        # Add (0, 1) and (1, 0) points
        precisions = np.concatenate([[1], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])

        # Compute AUC using trapezoidal rule
        auc = float(np.trapz(precisions, recalls))

        return auc

    def _compute_multiclass_auc(self, y_true: NDArray, y_score: NDArray) -> float:
        """Compute one-vs-rest AUC for multiclass classification."""
        classes = np.unique(y_true)
        n_classes = len(classes)

        if y_score.shape[1] != n_classes:
            raise ValueError(
                f"y_score must have {n_classes} columns for {n_classes} classes"
            )

        # Compute AUC for each class vs rest
        aucs = []
        for i, cls in enumerate(classes):
            y_true_binary = (y_true == cls).astype(int)
            y_score_binary = y_score[:, i]

            if np.sum(y_true_binary) == 0 or np.sum(y_true_binary) == len(y_true_binary):
                # Skip if all samples are one class
                continue

            auc = self._compute_auc_roc(y_true_binary, y_score_binary)
            aucs.append(auc)

        # Return macro-averaged AUC
        return float(np.mean(aucs)) if aucs else 0.5


class RegressionEvaluator:
    """Evaluator for regression tasks.

    Computes standard regression metrics including MSE, RMSE, MAE, R², and
    Pearson correlation.

    Examples:
        >>> evaluator = RegressionEvaluator()
        >>> metrics = evaluator.evaluate(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.3f}")
        >>> print(f"R²: {metrics['r2']:.3f}")
    """

    def evaluate(
        self, y_true: NDArray, y_pred: NDArray, compute_residuals: bool = False
    ) -> dict[str, Any]:
        """Evaluate regression predictions.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            compute_residuals: Whether to include residuals in output.

        Returns:
            Dictionary of metrics.

        Raises:
            ValueError: If input shapes don't match.
        """
        # Validate inputs
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch: y_true shape {y_true.shape}, "
                f"y_pred shape {y_pred.shape}"
            )

        metrics: dict[str, Any] = {}

        # Compute metrics
        metrics["mse"] = self._compute_mse(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = self._compute_mae(y_true, y_pred)
        metrics["r2"] = self._compute_r2(y_true, y_pred)

        # Pearson correlation
        pearson_r, pearson_p = self._compute_pearson(y_true, y_pred)
        metrics["pearson_r"] = pearson_r
        metrics["pearson_p"] = pearson_p

        # Residuals if requested
        if compute_residuals:
            metrics["residuals"] = y_true - y_pred

        return metrics

    def _compute_mse(self, y_true: NDArray, y_pred: NDArray) -> float:
        """Compute mean squared error."""
        return float(np.mean((y_true - y_pred) ** 2))

    def _compute_mae(self, y_true: NDArray, y_pred: NDArray) -> float:
        """Compute mean absolute error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    def _compute_r2(self, y_true: NDArray, y_pred: NDArray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return float(r2)

    def _compute_pearson(self, y_true: NDArray, y_pred: NDArray) -> tuple[float, float]:
        """Compute Pearson correlation coefficient and p-value."""
        correlation, p_value = stats.pearsonr(y_true, y_pred)
        return float(correlation), float(p_value)


class SurvivalEvaluator:
    """Evaluator for survival analysis tasks.

    Computes concordance index (C-index) for survival predictions.

    Examples:
        >>> evaluator = SurvivalEvaluator()
        >>> metrics = evaluator.evaluate(event_times, event_observed, risk_scores)
        >>> print(f"C-index: {metrics['c_index']:.3f}")
    """

    def evaluate(
        self, event_times: NDArray, event_observed: NDArray, risk_scores: NDArray
    ) -> dict[str, float]:
        """Evaluate survival predictions.

        Args:
            event_times: Time to event or censoring.
            event_observed: Whether event was observed (1) or censored (0).
            risk_scores: Predicted risk scores (higher = higher risk).

        Returns:
            Dictionary of metrics.

        Raises:
            ValueError: If input shapes don't match or inputs are invalid.
        """
        # Validate inputs
        event_times = np.asarray(event_times)
        event_observed = np.asarray(event_observed)
        risk_scores = np.asarray(risk_scores)

        if not (event_times.shape == event_observed.shape == risk_scores.shape):
            raise ValueError(
                f"shape mismatch: event_times {event_times.shape}, "
                f"event_observed {event_observed.shape}, "
                f"risk_scores {risk_scores.shape}"
            )

        # Validate event_observed values
        if not np.all((event_observed == 0) | (event_observed == 1)):
            raise ValueError("event_observed values must be 0 or 1")

        metrics: dict[str, float] = {}

        # Compute C-index
        metrics["c_index"] = self._compute_concordance_index(
            event_times, event_observed, risk_scores
        )

        return metrics

    def _compute_concordance_index(
        self, event_times: NDArray, event_observed: NDArray, risk_scores: NDArray
    ) -> float:
        """Compute concordance index (C-index).

        C-index measures the fraction of all pairs of subjects where predictions
        and outcomes are concordant.

        Args:
            event_times: Time to event or censoring.
            event_observed: Whether event was observed (1) or censored (0).
            risk_scores: Predicted risk scores.

        Returns:
            C-index value between 0 and 1 (0.5 = random, 1.0 = perfect).
        """
        n = len(event_times)

        concordant = 0
        discordant = 0
        tied_risk = 0

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Only consider pairs where we can make a comparison
                # (i.e., at least one event occurred and times differ)

                # Skip if both censored
                if event_observed[i] == 0 and event_observed[j] == 0:
                    continue

                # Determine which sample had event earlier
                if event_times[i] < event_times[j]:
                    # i had event earlier (or was censored earlier)
                    if event_observed[i] == 1:
                        # i had event, j may or may not
                        # Higher risk should predict earlier event
                        if risk_scores[i] > risk_scores[j]:
                            concordant += 1
                        elif risk_scores[i] < risk_scores[j]:
                            discordant += 1
                        else:
                            tied_risk += 1
                    # else: i was censored earlier, skip

                elif event_times[j] < event_times[i]:
                    # j had event earlier (or was censored earlier)
                    if event_observed[j] == 1:
                        # j had event, i may or may not
                        # Higher risk should predict earlier event
                        if risk_scores[j] > risk_scores[i]:
                            concordant += 1
                        elif risk_scores[j] < risk_scores[i]:
                            discordant += 1
                        else:
                            tied_risk += 1
                    # else: j was censored earlier, skip

                # else: event_times are equal, skip

        # Compute C-index
        total_pairs = concordant + discordant + tied_risk

        if total_pairs == 0:
            return 0.5  # No comparable pairs, return random guess

        # Ties count as 0.5
        c_index = (concordant + 0.5 * tied_risk) / total_pairs

        return float(c_index)
