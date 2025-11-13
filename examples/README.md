# 📚 OmicSelector2 Tutorial Notebooks for Google Colab

Welcome to the OmicSelector2 tutorial series! These Jupyter notebooks are designed to work seamlessly with Google Colab and Google Drive for persistent storage.

## 🚀 Quick Start

1. **Click any "Open in Colab" badge** below
2. **Run the first cell** to install OmicSelector2
3. **Authorize Google Drive** when prompted
4. **Follow along** with the tutorial

Your data, models, and results will be saved to `My Drive/OmicSelector2/` automatically!

---

## 📖 Tutorial Series

### Beginner Track 🟢

#### **[00 - Getting Started with Google Colab](00_getting_started_colab.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/00_getting_started_colab.ipynb)

**Time**: 10-15 minutes | **Level**: Beginner

Your first biomarker discovery analysis! Learn to:
- Install OmicSelector2 on Colab
- Set up Google Drive integration
- Run feature selection
- Train and evaluate a model
- Save results to Drive

**Start here if you're new to OmicSelector2!**

---

#### **[01 - Basic Feature Selection](01_basic_feature_selection.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/01_basic_feature_selection.ipynb)

**Time**: 20-25 minutes | **Level**: Beginner

Compare multiple feature selection methods:
- Lasso, Elastic Net, Random Forest, mRMR, Boruta
- Ensemble voting strategies
- Feature importance visualization
- Performance evaluation

---

### Intermediate Track 🟡

#### **[02 - Hyperparameter Tuning & Cross-Validation](02_hyperparameter_tuning.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/02_hyperparameter_tuning.ipynb)

**Time**: 25-30 minutes | **Level**: Intermediate

Optimize your models:
- Cross-validation strategies
- Hyperparameter optimization with Optuna
- Early stopping and checkpointing
- Model stability assessment

---

#### **[03 - Signature Benchmarking](03_signature_benchmarking.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/03_signature_benchmarking.ipynb)

**Time**: 30-35 minutes | **Level**: Intermediate

**The Core OmicSelector Philosophy**: Automated benchmarking
- Generate multiple feature signatures
- Test with different ML models
- Statistical comparison
- Select most robust signature

---

#### **[04 - Data Preprocessing & Quality Control](04_data_preprocessing.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/04_data_preprocessing.ipynb)

**Time**: 20-25 minutes | **Level**: Intermediate

Prepare your own data:
- Load data from Google Drive (CSV, Excel, h5ad)
- Quality control metrics
- Normalization methods (TPM, Log, Z-score)
- Train/test/validation splitting
- Handle imbalanced datasets
- Save processed data to Drive

---

### Advanced Track 🔴

#### **[05 - Stability Selection with Checkpointing](05_stability_selection.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/05_stability_selection.ipynb)

**Time**: 30-40 minutes | **Level**: Advanced

Robust feature selection with automatic checkpointing:
- Bootstrap-based stability selection
- **Resume from checkpoint** if Colab disconnects
- Stability paths visualization
- Threshold selection strategies
- Comparison with single-run methods

**Perfect for long-running analyses!**

---

#### **[06 - Model Interpretation & Explainability](06_model_interpretation.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/06_model_interpretation.ipynb)

**Time**: 25-30 minutes | **Level**: Advanced

Understand your models:
- Comprehensive evaluation metrics
- SHAP values for feature importance
- ROC curves, PR curves, calibration plots
- Feature interactions
- Save all interpretation results to Drive

---

#### **[07 - Complete End-to-End Workflow](07_complete_workflow.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kstawiski/OmicSelector2/blob/main/examples/07_complete_workflow.ipynb)

**Time**: 45-60 minutes | **Level**: Advanced

**Capstone Tutorial**: Full biomarker discovery pipeline
- Load real-world-like data from Drive
- Complete QC and preprocessing
- Multiple feature selection methods
- Stability selection
- Model training and optimization
- Comprehensive evaluation
- SHAP interpretation
- Export results for publication

**Complete analysis, start to finish!**

---

## 🎓 Recommended Learning Paths

### Path 1: Absolute Beginner
```
00 → 01 → 04 → 02 → 07
```
Start with installation, learn basics, preprocess data, optimize models, complete workflow

### Path 2: ML Background
```
00 → 03 → 05 → 06 → 07
```
Quick start, focus on benchmarking, stability, interpretation, complete workflow

### Path 3: Quick Evaluation
```
00 → 01 → 03
```
Get started, try methods, benchmark signatures (1-2 hours total)

### Path 4: Production Pipeline
```
04 → 05 → 06 → 07
```
Real data preprocessing → stability → interpretation → full workflow

---

## 💾 Google Drive Integration

All notebooks automatically create this structure in your Drive:

```
My Drive/
└── OmicSelector2/
    ├── data/
    │   ├── raw/              # Your original data
    │   ├── processed/        # Cleaned & normalized data
    │   │   └── splits/       # Train/val/test splits
    │   └── qc_reports/       # Quality control reports
    ├── results/
    │   ├── stability/        # Stability selection results
    │   ├── interpretation/   # SHAP values, metrics
    │   └── *.csv             # Feature lists, scores
    ├── models/               # Trained models (.pkl files)
    ├── plots/                # All visualizations
    │   ├── preprocessing/
    │   ├── stability/
    │   └── interpretation/
    └── checkpoints/          # For resumable analyses
        └── stability_selection/
```

### Benefits
- ✅ **Persistent Storage**: Survives Colab disconnections
- ✅ **Shareable**: Collaborate with your team
- ✅ **Accessible**: View from any device
- ✅ **Organized**: Structured folder hierarchy
- ✅ **Resumable**: Checkpoint long-running tasks

---

## 🔧 Prerequisites

### Software
- Google account (for Colab and Drive)
- Modern web browser

### Knowledge
- Basic Python (variables, loops, functions)
- Basic ML concepts (helpful but not required)
- Familiarity with biological data (helpful)

### Hardware
- **CPU**: Any notebook will work
- **GPU**: Automatically used if available (faster for large datasets)
- **RAM**: 12GB provided by Colab (sufficient for most analyses)

---

## 💡 Tips for Using These Notebooks

### 1. **Save Your Work**
- Colab disconnects after ~90 minutes of inactivity
- All results are saved to Drive automatically
- Use checkpointing for long analyses (>30 min)

### 2. **Share with Team**
```python
# Share your Drive folder
Right-click "OmicSelector2" folder → Share → Add people
```

### 3. **Upload Your Own Data**
```python
# Option 1: Upload via Drive browser
# Option 2: In notebook
from google.colab import files
uploaded = files.upload()
!mv your_file.csv /content/drive/MyDrive/OmicSelector2/data/raw/
```

### 4. **Restart Runtime If Needed**
```
Runtime → Restart runtime
```
Then re-run cells from top

### 5. **Monitor Resources**
```
Runtime → View resources
```
Check RAM and disk usage

---

## 🆘 Troubleshooting

### Installation Issues
```python
# If installation fails, restart runtime and try:
!pip install --upgrade pip
!pip install git+https://github.com/kstawiski/OmicSelector2.git
```

### Drive Not Mounting
```python
# Force remount
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Out of Memory
```python
# Use smaller dataset or batch size
# Or request high-RAM runtime:
# Runtime → Change runtime type → High-RAM
```

### Checkpoint Not Loading
```python
# Check if checkpoint file exists
import os
checkpoint_path = '/content/drive/MyDrive/OmicSelector2/checkpoints/...'
print(os.path.exists(checkpoint_path))
```

---

## 📚 Additional Resources

### Documentation
- [Main Documentation](https://github.com/kstawiski/OmicSelector2)
- [Development Guide (CLAUDE.md)](../CLAUDE.md)
- [Implementation Status](../IMPLEMENTATION_STATUS.md)

### Community
- [GitHub Discussions](https://github.com/kstawiski/OmicSelector2/discussions)
- [Report Issues](https://github.com/kstawiski/OmicSelector2/issues)

### Publications
- OmicSelector 1.0: [bioRxiv 2022](https://doi.org/10.1101/2022.06.01.494299)
- OmicSelector2: Coming soon

---

## 🤝 Contributing

Found a bug or have a suggestion?
1. Open an [issue](https://github.com/kstawiski/OmicSelector2/issues)
2. Submit a [pull request](https://github.com/kstawiski/OmicSelector2/pulls)
3. Share your notebook in [discussions](https://github.com/kstawiski/OmicSelector2/discussions)

---

## 📄 Citation

If you use these tutorials or OmicSelector2 in your research:

```bibtex
@software{omicselector2,
  title = {OmicSelector2: Next-generation platform for multi-omic biomarker discovery},
  author = {OmicSelector Team},
  year = {2025},
  url = {https://github.com/kstawiski/OmicSelector2}
}
```

---

## 🎉 Ready to Start?

1. **Pick a notebook** from the list above
2. **Click "Open in Colab"**
3. **Follow the tutorial**
4. **Share your results!**

---

**Happy biomarker hunting! 🧬🔬**

*Built with ❤️ for the oncology research community*
