"""Feature selection methods for OmicSelector2.

This module implements various feature selection approaches:

Classical Methods:
    - Lasso, Elastic Net (L1/L2 regularization)
    - Random Forest Variable Importance (RF-VI)
    - XGBoost Feature Importance
    - Boruta, mRMR, ReliefF
    - Variance threshold, statistical filters (t-test, ANOVA)

Graph-Based Methods:
    - GNN feature importance via attention weights
    - Network-based feature selection using PPI networks
    - Community detection for pathway/module-based features

Deep Learning Methods:
    - Concrete Autoencoders (differentiable feature selection)
    - Neural Feature Selection (NFS) with learned gates
    - INVASE (Instance-wise Variable Selection)

Single-Cell Specific:
    - HDG (High-Deviation Genes)
    - HEG (High-Expression Genes)
    - FEAST (Fast Entropy-based Active Set Testing)
    - DUBStepR (Determining Useful Biomarker Suites)

Ensemble Methods:
    - Stability selection framework
    - Majority/soft voting
    - Consensus ranking
"""

__all__ = []
