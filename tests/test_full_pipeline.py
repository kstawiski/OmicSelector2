"""
End-to-end integration test for full OmicSelector2 pipeline.

This test demonstrates the complete workflow:
1. Generate synthetic omics data
2. Apply multiple feature selection methods
3. Use ensemble and stability selection
4. Train multiple models
5. Evaluate performance with cross-validation
6. Benchmark signatures
7. Generate comprehensive results

This serves as both a test and documentation of the full pipeline capabilities.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

# Feature selection imports
from omicselector2.features.classical.lasso import LassoSelector
from omicselector2.features.classical.elastic_net import ElasticNetSelector
from omicselector2.features.classical.random_forest import RandomForestSelector
from omicselector2.features.classical.xgboost import XGBoostSelector
from omicselector2.features.ensemble import EnsembleSelector
from omicselector2.features.stability import StabilitySelector

# Model imports
from omicselector2.models.classical.random_forest import RandomForestClassifier
from omicselector2.models.classical.xgboost_models import XGBoostClassifier
from omicselector2.models.classical.linear_models import LogisticRegressionModel

# Training imports
from omicselector2.training.cross_validation import StratifiedKFoldSplitter
from omicselector2.training.evaluator import ClassificationEvaluator
from omicselector2.training.benchmarking import SignatureBenchmark, Benchmarker


@pytest.fixture
def synthetic_omics_data():
    """
    Generate synthetic RNA-seq-like data for testing.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, metadata)
    """
    np.random.seed(42)
    
    # Parameters
    n_samples = 300
    n_features = 500
    n_informative = 30  # Truly predictive genes
    
    # Generate feature names (gene names)
    feature_names = [f"GENE_{i:04d}" for i in range(n_features)]
    
    # Generate expression data
    X = np.random.randn(n_samples, n_features)
    
    # Make first n_informative features truly predictive
    # Create two clusters with different expression patterns
    for i in range(n_informative):
        X[:n_samples // 2, i] += np.random.randn(n_samples // 2) + 2.0
        X[n_samples // 2:, i] += np.random.randn(n_samples - n_samples // 2) - 2.0
    
    # Create binary target (cancer vs healthy)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    
    # Add some noise to make it more realistic
    noise_samples = int(0.05 * n_samples)  # 5% label noise
    noise_indices = np.random.choice(n_samples, noise_samples, replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="cancer_status")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=42, stratify=y_series
    )
    
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_distribution": {
            "train": y_train.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict(),
        },
    }
    
    return X_train, X_test, y_train, y_test, feature_names, metadata


class TestFullPipeline:
    """Test suite for complete OmicSelector2 pipeline."""
    
    def test_pipeline_data_generation(self, synthetic_omics_data):
        """Test that synthetic data is generated correctly."""
        X_train, X_test, y_train, y_test, feature_names, metadata = synthetic_omics_data
        
        # Check dimensions
        assert X_train.shape[1] == metadata["n_features"]
        assert X_test.shape[1] == metadata["n_features"]
        assert len(X_train) == metadata["n_train"]
        assert len(X_test) == metadata["n_test"]
        
        # Check feature names
        assert list(X_train.columns) == feature_names
        assert list(X_test.columns) == feature_names
        
        # Check target balance
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert set(y_train.unique()) == {0, 1}
        assert set(y_test.unique()) == {0, 1}
        
        print(f"\n✓ Data generation test passed")
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Class distribution: {metadata['class_distribution']}")
    
    def test_pipeline_multiple_feature_selectors(self, synthetic_omics_data):
        """Test applying multiple feature selection methods."""
        X_train, X_test, y_train, y_test, _, _ = synthetic_omics_data
        
        n_features_to_select = 50
        
        # 1. Lasso
        lasso = LassoSelector(
            alpha=0.01,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        )
        lasso.fit(X_train, y_train)
        lasso_result = lasso.get_result()
        
        assert lasso_result.n_features_selected <= n_features_to_select
        assert len(lasso_result.selected_features) > 0
        
        # 2. Elastic Net
        enet = ElasticNetSelector(
            alpha=0.01,
            l1_ratio=0.7,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        )
        enet.fit(X_train, y_train)
        enet_result = enet.get_result()
        
        assert enet_result.n_features_selected <= n_features_to_select
        assert len(enet_result.selected_features) > 0
        
        # 3. Random Forest
        rf = RandomForestSelector(
            n_estimators=100,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_result = rf.get_result()
        
        assert rf_result.n_features_selected <= n_features_to_select
        assert len(rf_result.selected_features) > 0
        
        # 4. XGBoost
        xgb = XGBoostSelector(
            n_estimators=100,
            task="classification",
            n_features_to_select=n_features_to_select,
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        xgb_result = xgb.get_result()
        
        assert xgb_result.n_features_selected <= n_features_to_select
        assert len(xgb_result.selected_features) > 0
        
        print(f"\n✓ Multiple feature selection methods test passed")
        print(f"  Lasso selected: {lasso_result.n_features_selected} features")
        print(f"  Elastic Net selected: {enet_result.n_features_selected} features")
        print(f"  Random Forest selected: {rf_result.n_features_selected} features")
        print(f"  XGBoost selected: {xgb_result.n_features_selected} features")
        
        return {
            "lasso": lasso,
            "elastic_net": enet,
            "random_forest": rf,
            "xgboost": xgb,
        }
    
    def test_pipeline_ensemble_selection(self, synthetic_omics_data):
        """Test ensemble feature selection combining multiple methods."""
        X_train, X_test, y_train, y_test, _, _ = synthetic_omics_data
        
        n_features_to_select = 50
        
        # Create multiple selectors
        selectors = [
            LassoSelector(
                alpha=0.01,
                task="classification",
                n_features_to_select=n_features_to_select,
                random_state=42,
            ),
            ElasticNetSelector(
                alpha=0.01,
                l1_ratio=0.7,
                task="classification",
                n_features_to_select=n_features_to_select,
                random_state=42,
            ),
            RandomForestSelector(
                n_estimators=100,
                task="classification",
                n_features_to_select=n_features_to_select,
                random_state=42,
            ),
        ]
        
        # Fit all selectors
        for selector in selectors:
            selector.fit(X_train, y_train)
        
        # Test majority voting
        ensemble_majority = EnsembleSelector(
            selectors=selectors,
            strategy="majority",
            threshold=2,  # Feature must be selected by at least 2 methods
        )
        ensemble_majority.fit(X_train, y_train)
        majority_result = ensemble_majority.get_result()
        
        assert len(majority_result.selected_features) > 0
        assert majority_result.n_features_selected <= n_features_to_select
        
        # Test consensus ranking
        ensemble_consensus = EnsembleSelector(
            selectors=selectors,
            strategy="consensus_ranking",
            n_features_to_select=30,
        )
        ensemble_consensus.fit(X_train, y_train)
        consensus_result = ensemble_consensus.get_result()
        
        assert consensus_result.n_features_selected == 30
        
        print(f"\n✓ Ensemble selection test passed")
        print(f"  Majority voting selected: {majority_result.n_features_selected} features")
        print(f"  Consensus ranking selected: {consensus_result.n_features_selected} features")
        
        return ensemble_consensus
    
    def test_pipeline_stability_selection(self, synthetic_omics_data):
        """Test stability selection with bootstrap resampling."""
        X_train, X_test, y_train, y_test, _, _ = synthetic_omics_data
        
        # Base selector
        base_selector = LassoSelector(
            alpha=0.01,
            task="classification",
            n_features_to_select=50,
            random_state=42,
        )
        
        # Stability selection (reduced iterations for faster testing)
        stability = StabilitySelector(
            base_selector=base_selector,
            n_bootstraps=20,  # Reduced from 100 for faster testing
            threshold=0.5,  # Feature must be selected in 50% of bootstraps
            sample_fraction=0.8,
            random_state=42,
        )
        
        stability.fit(X_train, y_train)
        stability_result = stability.get_result()
        
        assert len(stability_result.selected_features) > 0
        assert hasattr(stability_result, "stability_scores")
        
        # Check that stability scores are between 0 and 1
        assert all(0 <= score <= 1 for score in stability_result.stability_scores)
        
        print(f"\n✓ Stability selection test passed")
        print(f"  Stable features selected: {stability_result.n_features_selected}")
        print(f"  Top 5 stability scores: {stability_result.stability_scores[:5].round(3).tolist()}")
        
        return stability
    
    def test_pipeline_model_training(self, synthetic_omics_data):
        """Test training multiple models on selected features."""
        X_train, X_test, y_train, y_test, _, _ = synthetic_omics_data
        
        # First, select features
        selector = RandomForestSelector(
            n_estimators=100,
            task="classification",
            n_features_to_select=50,
            random_state=42,
        )
        selector.fit(X_train, y_train)
        
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        
        # Split train into train/validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_selected, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        evaluator = ClassificationEvaluator()
        
        # 1. Train Random Forest (directly without Trainer)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_tr, y_tr)
        
        # Evaluate
        y_pred_proba = rf_model.predict_proba(X_test_selected)
        rf_metrics = evaluator.evaluate(y_test.values, y_pred_proba, probabilities=True)
        if "auc_roc" in rf_metrics:
            rf_metrics["auc"] = rf_metrics["auc_roc"]
        
        # 2. Train XGBoost
        xgb_model = XGBoostClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_tr, y_tr)
        y_pred_proba = xgb_model.predict_proba(X_test_selected)
        xgb_metrics = evaluator.evaluate(y_test.values, y_pred_proba, probabilities=True)
        if "auc_roc" in xgb_metrics:
            xgb_metrics["auc"] = xgb_metrics["auc_roc"]
        
        # 3. Train Logistic Regression
        lr_model = LogisticRegressionModel(random_state=42, max_iter=1000)
        lr_model.fit(X_tr, y_tr)
        y_pred_proba = lr_model.predict_proba(X_test_selected)
        lr_metrics = evaluator.evaluate(y_test.values, y_pred_proba, probabilities=True)
        if "auc_roc" in lr_metrics:
            lr_metrics["auc"] = lr_metrics["auc_roc"]
        
        assert "auc" in rf_metrics
        assert "accuracy" in rf_metrics or "auc_roc" in rf_metrics
        
        print(f"\n✓ Model training test passed")
        print(f"  Random Forest - Test AUC: {rf_metrics['auc']:.3f}")
        print(f"  XGBoost - Test AUC: {xgb_metrics['auc']:.3f}")
        print(f"  Logistic Regression - Test AUC: {lr_metrics['auc']:.3f}")
        
        return {
            "rf": (rf_model, rf_metrics),
            "xgb": (xgb_model, xgb_metrics),
            "lr": (lr_model, lr_metrics),
        }
    
    def test_pipeline_cross_validation(self, synthetic_omics_data):
        """Test cross-validation evaluation."""
        X_train, X_test, y_train, y_test, _, _ = synthetic_omics_data
        
        # Select features first
        selector = LassoSelector(
            alpha=0.01,
            task="classification",
            n_features_to_select=50,
            random_state=42,
        )
        selector.fit(X_train, y_train)
        X_train_selected = selector.transform(X_train)
        
        # Perform cross-validation manually
        cv_splitter = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        evaluator = ClassificationEvaluator()
        
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_selected, y_train)):
            # Get fold data
            X_fold_train = X_train_selected.iloc[train_idx]
            X_fold_val = X_train_selected.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_fold_val)
            metrics = evaluator.evaluate(y_fold_val.values, y_pred_proba, probabilities=True)
            if "auc_roc" in metrics:
                metrics["auc"] = metrics["auc_roc"]
            fold_metrics.append(metrics)
        
        # Aggregate results
        cv_results = {
            "mean_auc": np.mean([m.get("auc", m.get("auc_roc", 0)) for m in fold_metrics]),
            "std_auc": np.std([m.get("auc", m.get("auc_roc", 0)) for m in fold_metrics]),
            "mean_accuracy": np.mean([m.get("accuracy", 0) for m in fold_metrics]),
            "std_accuracy": np.std([m.get("accuracy", 0) for m in fold_metrics]),
            "fold_metrics": fold_metrics,
        }
        
        assert len(cv_results["fold_metrics"]) == 5
        assert cv_results["mean_auc"] > 0.5
        
        print(f"\n✓ Cross-validation test passed")
        print(f"  Mean AUC: {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
        print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.3f}")
        
        return cv_results
    
    def test_pipeline_signature_benchmarking(self, synthetic_omics_data):
        """Test benchmarking multiple signatures and models."""
        X_train, X_test, y_train, y_test, _, _ = synthetic_omics_data
        
        # Create multiple feature selection methods
        lasso = LassoSelector(
            alpha=0.01, task="classification", n_features_to_select=30, random_state=42
        )
        lasso.fit(X_train, y_train)
        
        rf_selector = RandomForestSelector(
            n_estimators=100, task="classification", n_features_to_select=30, random_state=42
        )
        rf_selector.fit(X_train, y_train)
        
        # Create signatures dictionary
        signatures = {
            "Lasso_30": list(lasso.selected_features_),
            "RF_30": list(rf_selector.selected_features_),
        }
        
        # Create benchmarker
        benchmarker = Benchmarker(cv_folds=3, random_state=42)  # Reduced for faster testing
        
        # Run benchmark with multiple models
        results = benchmarker.benchmark_signatures(
            signatures=signatures,
            models=["RandomForest", "LogisticRegression"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        
        assert len(results) > 0
        # Should have 2 signatures × 2 models = 4 results
        assert len(results) == 4
        
        # Check that all results are BenchmarkResult objects
        for result in results:
            assert hasattr(result, "signature_name")
            assert hasattr(result, "model_name")
            assert hasattr(result, "cv_metrics")
            assert hasattr(result, "test_metrics")
            assert "auc_roc" in result.test_metrics
        
        print(f"\n✓ Signature benchmarking test passed")
        print(f"  Signatures tested: {list(signatures.keys())}")
        print(f"  Models tested: RandomForest, LogisticRegression")
        
        # Print summary
        for result in results:
            print(f"\n  {result.signature_name} + {result.model_name}:")
            print(f"    CV AUC: {result.cv_metrics.get('auc_roc', 0):.3f}")
            print(f"    Test AUC: {result.test_metrics.get('auc_roc', 0):.3f}")
        
        return results
        print(f"  Models tested: {list(models.keys())}")
        
        # Print summary
        for sig_name, sig_results in results.items():
            print(f"\n  Signature: {sig_name}")
            for model_name, metrics in sig_results.items():
                print(f"    {model_name}: AUC = {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.3f}")
        
        return results
    
    def test_pipeline_end_to_end(self, synthetic_omics_data):
        """
        Complete end-to-end pipeline test demonstrating full workflow.
        
        This test simulates a real biomarker discovery workflow:
        1. Generate/load omics data
        2. Apply multiple feature selection methods
        3. Use ensemble selection
        4. Train and evaluate multiple models
        5. Cross-validate performance
        6. Benchmark signatures
        7. Select best model + signature combination
        """
        X_train, X_test, y_train, y_test, feature_names, metadata = synthetic_omics_data
        
        print("\n" + "=" * 80)
        print("FULL PIPELINE TEST - End-to-End Biomarker Discovery")
        print("=" * 80)
        
        # Step 1: Feature Selection
        print("\n📊 Step 1: Feature Selection")
        print("-" * 80)
        
        selectors_dict = {}
        for name, selector_class, params in [
            ("Lasso", LassoSelector, {"alpha": 0.01, "task": "classification"}),
            ("ElasticNet", ElasticNetSelector, {"alpha": 0.01, "l1_ratio": 0.7, "task": "classification"}),
            ("RandomForest", RandomForestSelector, {"n_estimators": 100, "task": "classification"}),
            ("XGBoost", XGBoostSelector, {"n_estimators": 100, "task": "classification"}),
        ]:
            selector = selector_class(
                n_features_to_select=40, random_state=42, **params
            )
            selector.fit(X_train, y_train)
            selectors_dict[name] = selector
            result = selector.get_result()
            print(f"  {name:15s}: {result.n_features_selected:3d} features selected")
        
        # Step 2: Ensemble Selection
        print("\n🤝 Step 2: Ensemble Selection")
        print("-" * 80)
        
        ensemble = EnsembleSelector(
            selectors=list(selectors_dict.values()),
            strategy="consensus_ranking",
            n_features_to_select=30,
        )
        ensemble.fit(X_train, y_train)
        ensemble_result = ensemble.get_result()
        print(f"  Consensus features: {ensemble_result.n_features_selected}")
        
        # Step 3: Model Training
        print("\n🤖 Step 3: Model Training")
        print("-" * 80)
        
        X_train_selected = ensemble.transform(X_train)
        X_test_selected = ensemble.transform(X_test)
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_selected, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        models_dict = {}
        evaluator = ClassificationEvaluator()
        
        for name, model_class, params in [
            ("RandomForest", RandomForestClassifier, {"n_estimators": 100}),
            ("XGBoost", XGBoostClassifier, {"n_estimators": 100}),
            ("LogisticRegression", LogisticRegressionModel, {"max_iter": 1000}),
        ]:
            model = model_class(random_state=42, **params)
            model.fit(X_tr, y_tr)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_selected)
            test_metrics = evaluator.evaluate(y_test.values, y_pred_proba, probabilities=True)
            models_dict[name] = (model, test_metrics)
            
            # Rename keys to match expected format
            if "auc_roc" in test_metrics:
                test_metrics["auc"] = test_metrics["auc_roc"]
            
            print(f"  {name:20s}: Test AUC = {test_metrics.get('auc', 0):.3f}, "
                  f"Accuracy = {test_metrics.get('accuracy', 0):.3f}")
        
        # Step 4: Cross-Validation
        print("\n✅ Step 4: Cross-Validation")
        print("-" * 80)
        
        best_model_name = max(models_dict.keys(), key=lambda k: models_dict[k][1]["auc"])
        
        # Perform cross-validation on best model
        cv_splitter = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_selected, y_train)):
            # Get fold data
            X_fold_train = X_train_selected.iloc[train_idx]
            X_fold_val = X_train_selected.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Create and train new model for this fold
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
            if "auc_roc" in metrics:
                metrics["auc"] = metrics["auc_roc"]
            fold_metrics.append(metrics)
        
        # Aggregate CV results
        cv_results = {
            "mean_auc": np.mean([m.get("auc", m.get("auc_roc", 0)) for m in fold_metrics]),
            "std_auc": np.std([m.get("auc", m.get("auc_roc", 0)) for m in fold_metrics]),
            "mean_accuracy": np.mean([m.get("accuracy", 0) for m in fold_metrics]),
            "std_accuracy": np.std([m.get("accuracy", 0) for m in fold_metrics]),
        }
        
        print(f"  Best model: {best_model_name}")
        print(f"  CV AUC: {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
        print(f"  CV Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        
        # Step 5: Final Summary
        print("\n" + "=" * 80)
        print("✨ PIPELINE COMPLETE - Summary")
        print("=" * 80)
        print(f"  Dataset: {metadata['n_samples']} samples, {metadata['n_features']} features")
        print(f"  Feature selection: {ensemble_result.n_features_selected} consensus features")
        print(f"  Best model: {best_model_name}")
        print(f"  Test performance: AUC = {models_dict[best_model_name][1]['auc']:.3f}")
        print(f"  CV performance: AUC = {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
        print("=" * 80)
        
        # Assertions to ensure pipeline completed successfully
        assert ensemble_result.n_features_selected > 0
        assert all(metrics["auc"] > 0.5 for _, metrics in models_dict.values())
        assert cv_results["mean_auc"] > 0.5
        
        return {
            "selectors": selectors_dict,
            "ensemble": ensemble,
            "models": models_dict,
            "cv_results": cv_results,
            "best_model_name": best_model_name,
        }


if __name__ == "__main__":
    """Run tests directly for demonstration."""
    import sys
    
    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
