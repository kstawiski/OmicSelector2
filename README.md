# OmicSelector2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Next-generation Python platform for multi-omic biomarker discovery in oncology**

OmicSelector2 is a modernized, Python-native platform for automated feature selection and multi-modal integration using state-of-the-art deep learning methods. It transitions from OmicSelector 1.0's R-based architecture to leverage the full power of the Python ecosystem for biomarker discovery in cancer research.

## 🎯 Key Features

- **Automated Benchmarking**: Test multiple feature selection methods and select signatures most resilient to overfitting
- **Multi-Omics Integration**: Native support for scRNA-seq, bulk RNA-seq, WES, methylation, CNV, and radiomics
- **State-of-the-Art Models**: Graph Neural Networks (GNNs), Variational Autoencoders (VAEs), attention mechanisms
- **Clinical Translatability**: Interpretable models (SHAP), stability selection, rigorous hold-out validation
- **Production-Ready**: FastAPI backend, Celery job queue, MLflow experiment tracking, PostgreSQL + S3 storage

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
from omicselector2 import OmicSelector

# Initialize client
client = OmicSelector(api_url="http://localhost:8000", api_key="your-key")

# Upload data
dataset = client.upload_data(
    file="cancer_rnaseq_counts.csv",
    data_type="bulk_rna_seq"
)

# Run feature selection
job = client.create_feature_selection_job(
    dataset_id=dataset.id,
    methods=["lasso", "elastic_net", "rf_importance"],
    config={"cv_folds": 5, "stability": {"enabled": True}}
)

# Get results
results = client.get_results(job.id)
print(results.stable_features)
```

## 📚 Documentation

- **User Guide**: [docs/user_guide/](docs/user_guide/)
- **API Reference**: [docs/api/](docs/api/)
- **Developer Guide**: [docs/developer/](docs/developer/)
- **Examples**: [docs/examples/](docs/examples/)

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

## 🧬 Supported Methods

### Feature Selection (Priority 1)

**Classical:**
- Lasso, Elastic Net
- Random Forest Variable Importance (RF-VI)
- XGBoost Feature Importance
- Boruta, mRMR, ReliefF

**Advanced:**
- Stability Selection Framework
- Graph Neural Network Importance
- Concrete Autoencoders
- INVASE (Instance-wise Variable Selection)

**Single-Cell Specific:**
- HDG (High-Deviation Genes)
- FEAST (Fast Entropy-based Active Set Testing)
- DUBStepR

### Multi-Omics Integration

- **GNNs**: Heterogeneous graph neural networks, MOGDx, Geometric GNN
- **Attention**: Multi-omics self-attention, MORE framework
- **VAEs**: MultiVI, conditional VAE

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

We follow strict Test-Driven Development (TDD):

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -m unit

# Run integration tests
pytest tests/integration -m integration

# Run specific test file
pytest tests/unit/test_features/test_lasso.py

# Skip slow tests
pytest -m "not slow"
```

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
