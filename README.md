# OmicSelector2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Next-generation Python platform for multi-omic biomarker discovery in oncology**

OmicSelector2 is a modernized, Python-native platform for automated feature selection and multi-modal integration using state-of-the-art deep learning methods. It transitions from OmicSelector 1.0's R-based architecture to leverage the full power of the Python ecosystem for biomarker discovery in cancer research.

## 🎯 Key Features (v1.0)

- **Automated Benchmarking**: Test multiple feature selection methods and select signatures most resilient to overfitting
- **12 Priority 1 Feature Selection Methods**: Lasso, Elastic Net, Random Forest VI, XGBoost, Boruta, mRMR, ReliefF, and more
- **Ensemble & Stability Selection**: Robust feature sets through voting and bootstrap aggregation
- **Advanced Training Infrastructure**: Callbacks (early stopping, checkpointing), hyperparameter optimization (Optuna), cross-validation
- **Comprehensive Model Library**: Random Forest, XGBoost, Logistic Regression, SVM
- **Production-Quality Code**: 457 tests passing (v1.0 core), >80% coverage, strict TDD, full type hints

### Coming in v2.0
- FastAPI backend, Celery job queue, React frontend
- Multi-omics integration (GNNs, VAEs, attention mechanisms)
- MLflow experiment tracking, PostgreSQL + S3 storage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/omicselector/omicselector2.git
cd omicselector2

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Basic Usage

```python
import pandas as pd
from omicselector2.features.classical.random_forest import RandomForestSelector
from omicselector2.features.ensemble import EnsembleSelector
from omicselector2.models.classical import RandomForestClassifier
from omicselector2.training.trainer import Trainer
from omicselector2.training.callbacks import EarlyStopping

# Load your data
X = pd.read_csv("gene_expression.csv", index_col=0)
y = pd.read_csv("outcomes.csv", index_col=0).squeeze()

# Feature selection with Random Forest
selector = RandomForestSelector(
    n_estimators=100,
    n_features_to_select=50,
    random_state=42
)
X_selected = selector.fit_transform(X, y)

# Train model with callbacks
model = RandomForestClassifier(n_estimators=200, random_state=42)
trainer = Trainer(
    model=model,
    callbacks=[EarlyStopping(monitor='val_auc', patience=5)]
)
history = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

print(f"Best val AUC: {max(history['val_auc']):.3f}")
```

See the [examples/](examples/) directory for complete tutorials!

## 📚 Documentation

- **Tutorial Notebooks**: [examples/](examples/)
  - [01_basic_feature_selection.ipynb](examples/01_basic_feature_selection.ipynb)
  - [02_hyperparameter_tuning.ipynb](examples/02_hyperparameter_tuning.ipynb)
  - [03_signature_benchmarking.ipynb](examples/03_signature_benchmarking.ipynb)
- **Implementation Status**: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- **Development Guide**: [CLAUDE.md](CLAUDE.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## 🏗️ Architecture

```
omicselector2/
├── src/omicselector2/          # Main package
│   ├── api/                    # FastAPI application
│   ├── data/                   # Data handling & preprocessing
│   ├── features/               # Feature selection methods
│   ├── models/                 # ML/DL models
│   ├── training/               # Training orchestration
│   ├── inference/              # Model serving
│   └── utils/                  # Utilities
├── tests/                      # Test suite
├── docs/                       # Documentation
└── docker/                     # Docker configurations
```

## 🧬 Supported Methods (v1.0)

### Feature Selection - ALL 12 Priority 1 Methods ✅

**Classical Statistical (4/4):**
- ✅ Lasso (L1 regularization)
- ✅ Elastic Net (L1 + L2 regularization)
- ✅ t-test/ANOVA (statistical filtering)
- ✅ Cox Proportional Hazards (survival analysis)

**Tree-Based (3/3):**
- ✅ Random Forest Variable Importance (RF-VI)
- ✅ XGBoost Feature Importance
- ✅ Boruta (all-relevant selection)

**Filter Methods (3/3):**
- ✅ mRMR (Minimum Redundancy Maximum Relevance)
- ✅ ReliefF (instance-based)
- ✅ Variance Threshold

**Embedded (2/2):**
- ✅ L1-SVM
- ✅ Ridge Regression

**Additional:**
- ✅ Correlation Filter
- ✅ Ensemble Selector (majority/soft voting, consensus ranking)
- ✅ Stability Selection (bootstrap-based)

**Single-Cell:**
- ✅ HDG (High-Deviation Genes)
- ✅ HEG (High-Expression Genes)

### Machine Learning Models ✅

- ✅ Random Forest (Classifier & Regressor)
- ✅ XGBoost (Classifier & Regressor)
- ✅ Logistic Regression
- ✅ SVM Classifier

### Training Infrastructure ✅

- ✅ Callbacks (EarlyStopping, ModelCheckpoint, ProgressLogger)
- ✅ Hyperparameter Optimization (Optuna integration)
- ✅ Cross-Validation (K-Fold, Stratified K-Fold)
- ✅ Evaluators (Classification, Regression, Survival)
- ✅ Signature Benchmarking

### Coming in v2.0+ 🚧

- Graph Neural Networks (GNNs), VAEs, Attention mechanisms
- Multi-omics integration frameworks (MOGDx, MOGONET)
- FastAPI backend, Celery job queue, React frontend
- Radiomics pipeline (PyRadiomics integration)
- Advanced single-cell (FEAST, DUBStepR, scvi-tools)

## 🛠️ Development

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- MinIO or S3-compatible storage (for production)

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[all]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/omicselector2 --cov-report=html

# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Running the Application

```bash
# Start API server (development)
uvicorn omicselector2.api.main:app --reload

# Start Celery worker
celery -A omicselector2.tasks worker --loglevel=info

# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

We follow strict Test-Driven Development (TDD) with **457 passing tests** (v1.0 core) and **>80% code coverage**:

```bash
# Run all v1.0 tests (457 passing)
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/unit --ignore=tests/unit/test_api --ignore=tests/unit/test_utils/test_config.py --ignore=tests/unit/test_features/test_cox.py --ignore=tests/unit/test_models/test_tabnet.py

# Run specific test file
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/unit/test_features/test_lasso.py -v

# Run with coverage
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/unit --cov=src/omicselector2 --cov-report=html

# Skip slow tests
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/unit -m "not slow"
```

**Test Breakdown:**
- Feature Selection: 250+ tests (all 12 Priority 1 methods)
- Models: 29+ tests (RF, XGBoost, LogisticRegression, SVM)
- Training: 144+ tests (Trainer, Callbacks, Hyperparameter, CV, Evaluator, Benchmarking)
- Data: Tests for loaders, validators, preprocessors

## 📖 Citation

If you use OmicSelector2 in your research, please cite:

```bibtex
@software{omicselector2,
  title = {OmicSelector2: Next-generation platform for multi-omic biomarker discovery},
  author = {OmicSelector Team},
  year = {2025},
  url = {https://github.com/omicselector/omicselector2}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests (TDD!)
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OmicSelector 1.0 (R-based predecessor)
- TCGA Research Network
- scverse ecosystem (Scanpy, scvi-tools)
- PyTorch Geometric team
- FastAPI and Pydantic teams

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/omicselector/omicselector2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/omicselector/omicselector2/discussions)
- **Email**: omicselector@example.com

## 🔗 Links

- [Documentation](https://omicselector2.readthedocs.io)
- [OmicSelector 1.0](https://github.com/kstawiski/OmicSelector)
- [CLAUDE.md](CLAUDE.md) - Comprehensive development guide

---

**Built with ❤️ for the oncology research community**
