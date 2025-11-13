# Testing OmicSelector2 Pipeline

This document describes how to test the OmicSelector2 pipeline with example data.

## Quick Start

### Run the Example Script

The easiest way to see the full pipeline in action is to run the example script:

```bash
cd /path/to/OmicSelector2
PYTHONPATH=src:$PYTHONPATH python examples/full_pipeline_example.py
```

This will:
1. Generate synthetic omics data (300 samples, 500 features)
2. Apply 4 feature selection methods (Lasso, Elastic Net, Random Forest, XGBoost)
3. Create consensus features using ensemble selection
4. Train 3 models (Random Forest, XGBoost, Logistic Regression)
5. Perform 5-fold cross-validation
6. Benchmark multiple signature-model combinations

**Expected output**: Complete pipeline completes with AUC scores > 0.93

## Run Integration Tests

### Full End-to-End Test

Run the comprehensive integration test that validates the entire pipeline:

```bash
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_full_pipeline.py::TestFullPipeline::test_pipeline_end_to_end -v -s
```

This test demonstrates:
- ✅ Data generation
- ✅ Multiple feature selection methods
- ✅ Ensemble consensus
- ✅ Model training with 3 algorithms
- ✅ Cross-validation
- ✅ Performance metrics (AUC, accuracy)

### Run All Pipeline Tests

Run the entire test suite (8 comprehensive tests):

```bash
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_full_pipeline.py -v
```

**Test Suite:**
1. `test_pipeline_data_generation` - Validates synthetic data generation
2. `test_pipeline_multiple_feature_selectors` - Tests 4 feature selection methods
3. `test_pipeline_ensemble_selection` - Tests ensemble consensus
4. `test_pipeline_stability_selection` - Tests bootstrap-based stability selection
5. `test_pipeline_model_training` - Tests 3 model types
6. `test_pipeline_cross_validation` - Tests stratified 5-fold CV
7. `test_pipeline_signature_benchmarking` - Tests signature comparison
8. `test_pipeline_end_to_end` - Full workflow validation ⭐

## Test Results Summary

### Pipeline Performance

Using synthetic data (300 samples, 500 features, 30 informative):

| Component | Method | Performance |
|-----------|--------|-------------|
| Feature Selection | Lasso | 40 features selected |
| Feature Selection | Elastic Net | 40 features selected |
| Feature Selection | Random Forest | 40 features selected |
| Feature Selection | XGBoost | 40 features selected |
| Ensemble | Consensus Ranking | 30 features |
| Model | Random Forest | Test AUC: 0.937 |
| Model | XGBoost | Test AUC: 0.942 |
| Model | Logistic Regression | **Test AUC: 0.950** ⭐ |
| Cross-Validation | 5-Fold Stratified | **CV AUC: 0.987 ± 0.010** ⭐ |

### Benchmarking Results

3 signatures × 2 models = 6 combinations tested:

| Signature | Model | CV AUC | Test AUC |
|-----------|-------|--------|----------|
| Lasso_Top30 | Random Forest | 0.960 | 0.934 |
| Lasso_Top30 | Logistic Regression | **0.997** | **0.963** |
| RF_Top30 | Random Forest | 0.929 | 0.939 |
| RF_Top30 | Logistic Regression | 0.939 | **0.965** |
| Consensus_30 | Random Forest | 0.954 | 0.942 |
| Consensus_30 | Logistic Regression | 0.984 | 0.953 |

## Existing Demo Script

The original demo script is also available:

```bash
PYTHONPATH=src:$PYTHONPATH python examples/demo_feature_selection.py
```

This script demonstrates basic feature selection and generates CSV outputs.

## Test Files Location

- **Integration Tests**: `tests/test_full_pipeline.py`
- **Example Scripts**: `examples/`
  - `full_pipeline_example.py` - Complete pipeline demonstration
  - `demo_feature_selection.py` - Basic feature selection demo

## Pipeline Components Tested

### ✅ Feature Selection Methods
- Lasso (L1 regularization)
- Elastic Net (L1 + L2)
- Random Forest Variable Importance
- XGBoost Feature Importance
- Ensemble consensus ranking
- Stability selection (bootstrap-based)

### ✅ Machine Learning Models
- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression

### ✅ Evaluation & Validation
- Train/Test/Validation splits
- Stratified K-Fold Cross-Validation (5 folds)
- Classification metrics (AUC-ROC, Accuracy, F1)
- Signature benchmarking

### ✅ Data Handling
- Synthetic data generation
- Feature transformation
- Reproducible splits (random_state=42)

## Requirements

The pipeline has been tested with:
- Python 3.11+
- NumPy 1.26.4
- Pandas 2.3.3
- Scikit-learn 1.7.2
- XGBoost 3.1.1

All dependencies are automatically installed via `pip install -e .`

## Troubleshooting

### Import Errors

If you get import errors, make sure to set PYTHONPATH:

```bash
export PYTHONPATH=src:$PYTHONPATH
# or
PYTHONPATH=src:$PYTHONPATH python your_script.py
```

### Missing Dependencies

Install all dependencies:

```bash
pip install -e ".[dev]"
```

### Tests Failing

Run just the end-to-end test to verify core functionality:

```bash
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_full_pipeline.py::TestFullPipeline::test_pipeline_end_to_end -v
```

If this passes, the pipeline is working correctly.

## Next Steps

1. **Try with real data**: Replace synthetic data with your own omics dataset
2. **Adjust parameters**: Modify number of features, CV folds, model parameters
3. **Add more methods**: Extend with additional feature selection methods
4. **Custom models**: Add your own model implementations

## Support

For issues or questions:
- Open an issue on GitHub
- Check CLAUDE.md for development guidelines
- Review existing tests for examples

---

**Status**: ✅ All core pipeline functionality tested and working
**Last Updated**: 2025-11-13
