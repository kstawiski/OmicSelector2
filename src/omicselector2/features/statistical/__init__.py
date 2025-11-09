"""Statistical significance-based feature selection methods.

Implements OmicSelector 1.0's statistical testing methods for differential
expression analysis:

- SignificanceSelector: T-test based selection with multiple testing corrections
  - sig: All significant features (BH correction)
  - sigtop: Top N significant features
  - sigtopBonf: Top N with Bonferroni correction
  - sigtopHolm: Top N with Holm-Bonferroni correction
  - fcsig: Significant + |log2FC| > threshold

- FoldChangeSelector: Select by fold-change magnitude
  - topFC: Top N by absolute fold-change

Examples:
    >>> from omicselector2.features.statistical import SignificanceSelector
    >>>
    >>> # All significant features
    >>> selector = SignificanceSelector(method="sig", alpha=0.05)
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Top 20 with Bonferroni correction
    >>> selector = SignificanceSelector(
    ...     method="sigtopBonf",
    ...     n_features_to_select=20,
    ...     alpha=0.05
    ... )
    >>> selector.fit(X_train, y_train)
    >>>
    >>> # Significant + |log2FC| > 1 (fcsig)
    >>> selector = SignificanceSelector(
    ...     method="sig",
    ...     alpha=0.05,
    ...     fc_threshold=1.0
    ... )
    >>> selector.fit(X_train, y_train)
"""

from omicselector2.features.statistical.significance import (
    FoldChangeSelector,
    SignificanceSelector,
)

__all__ = [
    "SignificanceSelector",
    "FoldChangeSelector",
]
