"""Training orchestration for OmicSelector2 models.

Provides training infrastructure:
- Training loop abstractions
- Cross-validation frameworks
- Hyperparameter optimization (Optuna)
- Early stopping and learning rate scheduling
- Model checkpointing
- Experiment tracking (MLflow integration)
"""

from omicselector2.training.benchmarking import (
    BenchmarkResult,
    Benchmarker,
    SignatureBenchmark,
)
from omicselector2.training.cross_validation import (
    CrossValidator,
    KFoldSplitter,
    StratifiedKFoldSplitter,
    TrainTestValSplitter,
)
from omicselector2.training.evaluator import (
    ClassificationEvaluator,
    RegressionEvaluator,
    SurvivalEvaluator,
)

__all__ = [
    "CrossValidator",
    "KFoldSplitter",
    "StratifiedKFoldSplitter",
    "TrainTestValSplitter",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "SurvivalEvaluator",
    "BenchmarkResult",
    "Benchmarker",
    "SignatureBenchmark",
]
