#!/usr/bin/env python3.11
"""Demo script for OmicSelector2 feature selection.

This script demonstrates how to use the implemented feature selectors
with synthetic biomarker data.

Usage:
    python examples/demo_feature_selection.py
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("OmicSelector2 - Feature Selection Demo")
print("=" * 80)
print()

# ============================================================================
# 1. Generate Synthetic Omics Data
# ============================================================================

print("📊 Step 1: Generating synthetic RNA-seq-like data...")
print()

n_samples = 200
n_features = 1000
n_informative = 50  # Only 50 genes are truly predictive

# Create feature matrix
print(f"  - Samples: {n_samples}")
print(f"  - Total features (genes): {n_features}")
print(f"  - Informative features: {n_informative}")
print()

X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"GENE_{i:04d}" for i in range(n_features)]
)

# Create target based on informative features
# Simulate cancer classification: 0 = healthy, 1 = tumor
linear_combination = X.iloc[:, :n_informative].sum(axis=1)
noise = np.random.randn(n_samples) * 0.5
y = pd.Series(
    (linear_combination + noise > linear_combination.median()).astype(int),
    name="cancer_status"
)

print(f"  ✓ Generated data shape: {X.shape}")
print(f"  ✓ Target distribution: {y.value_counts().to_dict()}")
print()

# ============================================================================
# 2. Split Data
# ============================================================================

print("🔀 Step 2: Splitting into train/test sets...")
print()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"  ✓ Training samples: {X_train.shape[0]}")
print(f"  ✓ Test samples: {X_test.shape[0]}")
print()

# ============================================================================
# 3. Apply Feature Selection Methods
# ============================================================================

print("🔬 Step 3: Applying feature selection methods...")
print()

from omicselector2.features.classical.lasso import LassoSelector
from omicselector2.features.classical.elastic_net import ElasticNetSelector
from omicselector2.features.classical.random_forest import RandomForestSelector

# --- Lasso ---
print("  [1/3] Lasso (L1 Regularization)")
print("  " + "-" * 50)

lasso = LassoSelector(
    alpha=0.01,
    task='classification',
    n_features_to_select=100,
    random_state=42
)
lasso.fit(X_train, y_train)
lasso_result = lasso.get_result()

print(f"    ✓ Selected {lasso_result.n_features_selected} features")
print(f"    ✓ Top 5 features: {lasso_result.selected_features[:5]}")
print(f"    ✓ Top 5 scores: {lasso_result.feature_scores[:5].round(4).tolist()}")
print()

# --- Elastic Net ---
print("  [2/3] Elastic Net (L1 + L2 Regularization)")
print("  " + "-" * 50)

enet = ElasticNetSelector(
    alpha=0.01,
    l1_ratio=0.7,
    task='classification',
    n_features_to_select=100,
    random_state=42
)
enet.fit(X_train, y_train)
enet_result = enet.get_result()

print(f"    ✓ Selected {enet_result.n_features_selected} features")
print(f"    ✓ Top 5 features: {enet_result.selected_features[:5]}")
print(f"    ✓ L1 ratio: {enet.l1_ratio_}")
print()

# --- Random Forest ---
print("  [3/3] Random Forest Variable Importance")
print("  " + "-" * 50)

rf = RandomForestSelector(
    n_estimators=100,
    task='classification',
    n_features_to_select=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_result = rf.get_result()

print(f"    ✓ Selected {rf_result.n_features_selected} features")
print(f"    ✓ Top 5 features: {rf_result.selected_features[:5]}")
print(f"    ✓ Top 5 importances: {rf_result.feature_scores[:5].round(4).tolist()}")
print()

# ============================================================================
# 4. Compare Methods
# ============================================================================

print("📊 Step 4: Comparing feature selection methods...")
print()

# Find overlap between methods
lasso_set = set(lasso_result.selected_features)
enet_set = set(enet_result.selected_features)
rf_set = set(rf_result.selected_features)

overlap_all = lasso_set & enet_set & rf_set
overlap_lasso_enet = lasso_set & enet_set
overlap_lasso_rf = lasso_set & rf_set
overlap_enet_rf = enet_set & rf_set

print(f"  Feature overlap:")
print(f"    - Lasso ∩ Elastic Net: {len(overlap_lasso_enet)} features")
print(f"    - Lasso ∩ Random Forest: {len(overlap_lasso_rf)} features")
print(f"    - Elastic Net ∩ Random Forest: {len(overlap_enet_rf)} features")
print(f"    - All three methods: {len(overlap_all)} features")
print()

if overlap_all:
    print(f"  Consensus features (selected by all 3 methods):")
    consensus = sorted(list(overlap_all))[:10]
    for i, feat in enumerate(consensus, 1):
        print(f"    {i:2d}. {feat}")
print()

# ============================================================================
# 5. Evaluate Predictive Performance
# ============================================================================

print("🎯 Step 5: Evaluating predictive performance...")
print()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Train simple logistic regression on selected features
results_table = []

for name, selector in [
    ("Lasso", lasso),
    ("Elastic Net", enet),
    ("Random Forest", rf)
]:
    # Transform data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_selected, y_train)

    # Predictions
    y_pred = clf.predict(X_test_selected)
    y_prob = clf.predict_proba(X_test_selected)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    results_table.append({
        'Method': name,
        'N Features': len(selector.selected_features_),
        'Accuracy': f"{acc:.3f}",
        'AUC': f"{auc:.3f}",
        'F1': f"{f1:.3f}"
    })

# Display results table
results_df = pd.DataFrame(results_table)
print("  Classification Performance:")
print()
print(results_df.to_string(index=False))
print()

# ============================================================================
# 6. Export Results
# ============================================================================

print("💾 Step 6: Exporting results...")
print()

# Export selected features to DataFrame
lasso_df = lasso_result.to_dataframe()
lasso_df.to_csv('examples/lasso_features.csv', index=False)
print(f"  ✓ Exported Lasso features to examples/lasso_features.csv")

enet_df = enet_result.to_dataframe()
enet_df.to_csv('examples/elastic_net_features.csv', index=False)
print(f"  ✓ Exported Elastic Net features to examples/elastic_net_features.csv")

rf_df = rf_result.to_dataframe()
rf_df.to_csv('examples/random_forest_features.csv', index=False)
print(f"  ✓ Exported Random Forest features to examples/random_forest_features.csv")

print()

# ============================================================================
# 7. Summary
# ============================================================================

print("=" * 80)
print("✅ Demo Complete!")
print("=" * 80)
print()
print("Summary:")
print(f"  • Processed {n_features} features from {n_samples} samples")
print(f"  • Tested 3 feature selection methods")
print(f"  • Identified {len(overlap_all)} consensus features")
print(f"  • Achieved {results_table[0]['AUC']} AUC with Lasso")
print()
print("Next steps:")
print("  • Try stability selection for more robust feature selection")
print("  • Experiment with different alpha/l1_ratio parameters")
print("  • Compare with additional methods (XGBoost, Boruta, mRMR)")
print("  • Apply to real RNA-seq data")
print()
