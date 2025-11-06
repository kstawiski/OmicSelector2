"""Machine learning and deep learning models for OmicSelector2.

This module contains model implementations organized by type:

Classical ML Models:
    - Random Forest
    - Support Vector Machines (SVM)
    - Logistic Regression
    - XGBoost
    - Gradient Boosting Machines

Neural Network Models:
    - Feedforward Neural Networks
    - Autoencoders (standard, variational, sparse)
    - Attention-based models

Graph Neural Networks:
    - GCN (Graph Convolutional Networks)
    - GAT (Graph Attention Networks)
    - GraphSAGE
    - Heterogeneous GNNs

Multi-Omics Integration:
    - MOGDx (Multi-Omic Graph Diagnosis)
    - MORE (Multi-Omics hypergraph integration)
    - MultiVI (from scvi-tools)
    - Geometric GNN with Ricci curvature

Survival Analysis:
    - Cox Proportional Hazards
    - Random Survival Forests
    - DeepSurv
"""

from omicselector2.models.base import BaseClassifier, BaseModel, BaseRegressor

__all__ = [
    "BaseModel",
    "BaseClassifier",
    "BaseRegressor",
]
