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
from omicselector2.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
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
from omicselector2.training.hyperparameter import (
    HyperparameterOptimizer,
    PREDEFINED_SEARCH_SPACES,
)
from omicselector2.training.trainer import Trainer

__all__ = [
    "Trainer",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
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
    "HyperparameterOptimizer",
    "PREDEFINED_SEARCH_SPACES",
]
