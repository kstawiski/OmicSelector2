"""Neural network models.

Deep learning models for omics data:
- Feedforward neural networks
- Autoencoders (standard, variational, sparse, concrete)
- Attention mechanisms
- Multi-layer perceptrons
- TabNet (attention-based tabular learning)
"""

from omicselector2.models.neural.tabnet_models import (
    TabNetClassifier,
    TabNetRegressor,
)

__all__ = [
    "TabNetClassifier",
    "TabNetRegressor",
]
