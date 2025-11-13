#!/usr/bin/env python3
"""
Complete OmicSelector2 Pipeline Example
========================================

This script demonstrates the full biomarker discovery pipeline:
1. Generate synthetic omics data
2. Apply multiple feature selection methods
3. Use ensemble consensus
4. Train multiple models
5. Cross-validate performance
6. Benchmark signatures

Author: OmicSelector2 Team
Date: 2025
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Feature selection imports
from omicselector2.features.classical.lasso import LassoSelector
from omicselector2.features.classical.elastic_net import ElasticNetSelector
from omicselector2.features.classical.random_forest import RandomForestSelector
from omicselector2.features.classical.xgboost import XGBoostSelector
from omicselector2.features.ensemble import EnsembleSelector

# Model imports
from omicselector2.models.classical.random_forest import RandomForestClassifier
from omicselector2.models.classical.xgboost_models import XGBoostClassifier
from omicselector2.models.classical.linear_models import LogisticRegressionModel

# Training imports
from omicselector2.training.cross_validation import StratifiedKFoldSplitter
from omicselector2.training.evaluator import ClassificationEvaluator
from omicselector2.training.benchmarking import Benchmarker


def generate_synthetic_data(n_samples=300, n_features=500, n_informative=30):
    """Generate synthetic RNA-seq-like data for biomarker discovery.
    
    Args:
        n_samples: Number of samples (patients)
        n_features: Total number of features (genes)
        n_informative: Number of truly predictive features
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"📊 Generating synthetic data...")
    print(f"   Samples: {n_samples}, Features: {n_features}")
    print(f"   Informative features: {n_informative}")
    
    np.random.seed(42)
    
    # Generate feature names
    feature_names = [f"GENE_{i:04d}" for i in range(n_features)]
    
    # Generate expression data
    X = np.random.randn(n_samples, n_features)
    
    # Make first n_informative features truly predictive
    for i in range(n_informative):
        X[:n_samples // 2, i] += np.random.randn(n_samples // 2) + 2.0
        X[n_samples // 2:, i] += np.random.randn(n_samples - n_samples // 2) - 2.0
    
    # Create binary target (cancer vs healthy)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    
    # Add noise
    noise_samples = int(0.05 * n_samples)
    noise_indices = np.random.choice(n_samples, noise_samples, replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="cancer_status")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=42, stratify=y_series
    )
    
    print(f"   ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")
    
    return X_train, X_test, y_train, y_test


def main():
    """Run the complete biomarker discovery pipeline."""
    
    print("=" * 80)
    print("OmicSelector2 - Complete Pipeline Example")
    print("=" * 80)
    print()
    
    # ==============================================================================
    # STEP 1: Generate Data
    # ==============================================================================
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    
    # ==============================================================================
    # STEP 2: Feature Selection with Multiple Methods
    # ==============================================================================
    print("🔬 STEP 2: Feature Selection")
    print("-" * 80)
    
    n_features_to_select = 40
    
    # Initialize feature selectors
    selectors = {
        "Lasso": LassoSelector(
            alpha=0.01,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        ),
        "ElasticNet": ElasticNetSelector(
            alpha=0.01,
            l1_ratio=0.7,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        ),
        "RandomForest": RandomForestSelector(
            n_estimators=100,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        ),
        "XGBoost": XGBoostSelector(
            n_estimators=100,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        ),
    }
    
    # Fit all selectors
    for name, selector in selectors.items():
        print(f"   Running {name}...", end=" ")
        selector.fit(X_train, y_train)
        result = selector.get_result()
        print(f"✓ Selected {result.n_features_selected} features")
    
    print()
    
    # ==============================================================================
    # STEP 3: Ensemble Feature Selection
    # ==============================================================================
    print("🤝 STEP 3: Ensemble Feature Selection")
    print("-" * 80)
    
    ensemble = EnsembleSelector(
        selectors=list(selectors.values()),
        strategy="consensus_ranking",
        n_features_to_select=30,
    )
    ensemble.fit(X_train, y_train)
    ensemble_result = ensemble.get_result()
    
    print(f"   ✓ Consensus features: {ensemble_result.n_features_selected}")
    print(f"   ✓ Top 10 consensus genes: {ensemble_result.selected_features[:10]}")
    print()
    
    # Transform data with selected features
    X_train_selected = ensemble.transform(X_train)
    X_test_selected = ensemble.transform(X_test)
    
    # ==============================================================================
    # STEP 4: Train Multiple Models
    # ==============================================================================
    print("🤖 STEP 4: Model Training & Evaluation")
    print("-" * 80)
    
    models_dict = {}
    evaluator = ClassificationEvaluator()
    
    # Train Random Forest
    print("   Training Random Forest...", end=" ")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_selected, y_train)
    y_pred_proba = rf_model.predict_proba(X_test_selected)
    rf_metrics = evaluator.evaluate(y_test.values, y_pred_proba, probabilities=True)
    models_dict["RandomForest"] = (rf_model, rf_metrics)
    print(f"✓ Test AUC: {rf_metrics.get('auc_roc', 0):.3f}")
    
    # Train XGBoost
    print("   Training XGBoost...", end=" ")
    xgb_model = XGBoostClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_selected, y_train)
    y_pred_proba = xgb_model.predict_proba(X_test_selected)
    xgb_metrics = evaluator.evaluate(y_test.values, y_pred_proba, probabilities=True)
    models_dict["XGBoost"] = (xgb_model, xgb_metrics)
    print(f"✓ Test AUC: {xgb_metrics.get('auc_roc', 0):.3f}")
    
    # Train Logistic Regression
    print("   Training Logistic Regression...", end=" ")
    lr_model = LogisticRegressionModel(random_state=42, max_iter=1000)
    lr_model.fit(X_train_selected, y_train)
    y_pred_proba = lr_model.predict_proba(X_test_selected)
    lr_metrics = evaluator.evaluate(y_test.values, y_pred_proba, probabilities=True)
    models_dict["LogisticRegression"] = (lr_model, lr_metrics)
    print(f"✓ Test AUC: {lr_metrics.get('auc_roc', 0):.3f}")
    
    print()
    
    # ==============================================================================
    # STEP 5: Cross-Validation of Best Model
    # ==============================================================================
    print("✅ STEP 5: Cross-Validation")
    print("-" * 80)
    
    # Find best model
    best_model_name = max(
        models_dict.keys(),
        key=lambda k: models_dict[k][1].get("auc_roc", 0)
    )
    print(f"   Best model: {best_model_name}")
    
    # Perform 5-fold cross-validation
    cv_splitter = StratifiedKFoldSplitter(n_splits=5, random_state=42)
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_selected, y_train)):
        X_fold_train = X_train_selected.iloc[train_idx]
        X_fold_val = X_train_selected.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train model
        if best_model_name == "RandomForest":
            fold_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif best_model_name == "XGBoost":
            fold_model = XGBoostClassifier(n_estimators=100, random_state=42)
        else:
            fold_model = LogisticRegressionModel(max_iter=1000, random_state=42)
        
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Evaluate
        y_pred_proba = fold_model.predict_proba(X_fold_val)
        metrics = evaluator.evaluate(y_fold_val.values, y_pred_proba, probabilities=True)
        fold_metrics.append(metrics)
    
    # Aggregate CV results
    cv_auc = np.mean([m.get("auc_roc", 0) for m in fold_metrics])
    cv_auc_std = np.std([m.get("auc_roc", 0) for m in fold_metrics])
    
    print(f"   ✓ 5-Fold CV AUC: {cv_auc:.3f} ± {cv_auc_std:.3f}")
    print()
    
    # ==============================================================================
    # STEP 6: Benchmark Signatures
    # ==============================================================================
    print("📊 STEP 6: Signature Benchmarking")
    print("-" * 80)
    
    # Create signatures to benchmark
    signatures = {
        "Lasso_Top30": list(selectors["Lasso"].selected_features_[:30]),
        "RF_Top30": list(selectors["RandomForest"].selected_features_[:30]),
        "Consensus_30": list(ensemble_result.selected_features),
    }
    
    # Run benchmark
    benchmarker = Benchmarker(cv_folds=3, random_state=42)
    results = benchmarker.benchmark_signatures(
        signatures=signatures,
        models=["RandomForest", "LogisticRegression"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    
    # Display results
    print(f"   Benchmarked {len(signatures)} signatures × 2 models")
    print()
    for result in results:
        print(f"   {result.signature_name:20s} + {result.model_name:20s}")
        print(f"      CV AUC:   {result.cv_metrics.get('auc_roc', 0):.3f}")
        print(f"      Test AUC: {result.test_metrics.get('auc_roc', 0):.3f}")
    
    print()
    
    # ==============================================================================
    # FINAL SUMMARY
    # ==============================================================================
    print("=" * 80)
    print("✨ PIPELINE COMPLETE - Summary")
    print("=" * 80)
    print(f"✓ Generated synthetic omics data: {len(X_train) + len(X_test)} samples, {X_train.shape[1]} features")
    print(f"✓ Feature selection: {ensemble_result.n_features_selected} consensus features from 4 methods")
    print(f"✓ Model training: 3 classifiers trained and evaluated")
    print(f"✓ Best model: {best_model_name} with Test AUC = {models_dict[best_model_name][1].get('auc_roc', 0):.3f}")
    print(f"✓ Cross-validation: {cv_auc:.3f} ± {cv_auc_std:.3f} AUC (5-fold CV)")
    print(f"✓ Benchmarking: {len(results)} signature-model combinations tested")
    print()
    print("🎉 All pipeline steps completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
