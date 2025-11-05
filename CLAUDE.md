# CLAUDE.md: OmicSelector2 Development Guide

**Last Updated**: November 5, 2025  
**Version**: 2.0  
**Purpose**: Comprehensive guide for agentic development of OmicSelector2 - a next-generation Python platform for multi-omic biomarker discovery in oncology

---

## 🎯 Executive Summary

**Vision**: OmicSelector2 modernizes biomarker discovery by transitioning from OmicSelector 1.0's R-based automated feature selection to a Python-native platform enabling guided multi-modal integration with state-of-the-art deep learning methods. The software should be state-of-the-art. Learn knowledge from `knowledge/`.

**Core Value Proposition**: 
- **Retain**: OmicSelector 1.0's proven automated benchmarking philosophy - testing multiple feature selection methods and selecting signatures most resilient to overfitting
- **Modernize**: 100% migration from R/Shiny to Python/FastAPI/React with access to SOTA PyTorch ecosystem
- **Expand**: Native multi-modal integration (scRNA-seq, bulk RNA-seq, WES, radiomics) using graph neural networks (GNNs), variational autoencoders (VAEs), and attention mechanisms
- **Enhance**: Clinical translatability through interpretable models (SHAP), stability selection, and rigorous hold-out validation

**Target Users**: Biomedical researchers, bioinformaticians, oncology data scientists working with multi-omic data for cancer biomarker discovery

**Clinical Focus**: Urological cancers (bladder, prostate) and gastrointestinal cancers (rectal), with extensibility to pan-cancer applications

---

## 🚨 CRITICAL RULES & OPERATING PRINCIPLES

### Agentic Workflow Mandate

**STRICT COMPLIANCE REQUIRED**: All development MUST follow the **Explore → Plan → Code → Commit** workflow:

1. **Explore Phase**: 
   - Read relevant files, documentation, research papers
   - Understand requirements and constraints
   - DO NOT write code yet
   - Ask clarifying questions if needed

2. **Plan Phase**:
   - Create detailed, step-by-step implementation plan
   - Document in temporary markdown or this file
   - **AWAIT HUMAN APPROVAL** before proceeding
   - Revise plan based on feedback

3. **Code Phase**:
   - Implement ONLY after plan approval
   - Follow Test-Driven Development (TDD) strictly
   - Write minimal code to pass tests
   - Document all functions with Google-style docstrings

4. **Commit Phase**:
   - Commit failing tests first
   - Commit passing implementation separately
   - Write clear, descriptive commit messages
   - Update documentation and CHANGELOG

### Test-Driven Development (TDD) - NON-NEGOTIABLE

**RED-GREEN-REFACTOR CYCLE**:

```bash
# RED: Write failing test
pytest tests/test_feature.py::test_new_feature  # MUST FAIL
git add tests/test_feature.py
git commit -m "test: add failing test for feature X"

# GREEN: Write minimal implementation
# Edit src/module.py
pytest tests/test_feature.py::test_new_feature  # MUST PASS
git add src/module.py
git commit -m "feat: implement feature X to pass test"

# REFACTOR: Improve code quality (if needed)
# Edit src/module.py
pytest tests/test_feature.py  # MUST STILL PASS
git add src/module.py
git commit -m "refactor: improve feature X implementation"
```

**TDD RULES**:
- ✅ Write test FIRST - always
- ✅ Confirm test FAILS for correct reason
- ✅ Commit failing test BEFORE implementation
- ✅ Write MINIMAL code to pass
- ❌ DO NOT modify test during implementation
- ❌ DO NOT skip test writing
- ❌ DO NOT commit untested code

### Code Quality Standards

**Python Version**: 3.11+

**Type Hints**: MANDATORY for all functions
```python
def process_features(
    data: pd.DataFrame,
    method: str,
    n_features: int = 100
) -> tuple[list[str], float]:
    """Process features using specified method.
    
    Args:
        data: Input expression matrix (samples × features)
        method: Feature selection method name
        n_features: Number of features to select
        
    Returns:
        Tuple of (selected_features, stability_score)
        
    Raises:
        ValueError: If method is not supported
    """
    pass
```

**Documentation**: Google-style docstrings for ALL functions, classes, modules

**Tools**: 
```bash
black src/ tests/           # Code formatting
isort src/ tests/           # Import sorting  
flake8 src/ tests/          # Linting
mypy src/                   # Type checking
pytest tests/ -v --cov      # Testing with coverage
```

---

## 📚 PROJECT CONTEXT

### OmicSelector 1.0 Analysis

**Architecture** (R-based):
- **Core Function**: `OmicSelector_OmicSelector()` - orchestrates 70+ feature selection methods
- **Dependencies**: Boruta, varSelRF, caret, rpart, C5.0, randomForest, keras (R interface)
- **UI**: Shiny web application with Docker deployment
- **Data Flow**: 
  1. Data preparation: `mixed_train.csv`, `mixed_test.csv`, `mixed_validation.csv`
  2. Feature selection: Parallel execution of methods (m=1:70)
  3. Benchmarking: Test signatures with various ML models
  4. Deep learning: Extension for feedforward neural networks (up to 3 hidden layers) with autoencoders

**Key Methods (Examples)**:
- Lasso, Elastic Net, t-test/ANOVA
- Random Forest Variable Importance (RF-VI)
- Boruta, Correlation-based Feature Selection (CFS)
- Recursive Feature Elimination (RFE)
- Minimal Description Length (MDL)
- WxNet (Python script integration)

**Limitations**:
- ❌ Monolithic R architecture with massive dependency conflicts
- ❌ Difficult to scale computationally
- ❌ No access to Python-native deep learning ecosystem (PyTorch Geometric, scverse)
- ❌ Limited multi-modal integration capabilities
- ❌ Shiny UI less flexible than modern React frameworks

**Strengths to Preserve**:
- ✅ Automated benchmarking philosophy - test multiple methods, select best signature
- ✅ Hold-out validation preventing overfitting
- ✅ Comprehensive method coverage
- ✅ Integration of feature selection + model development
- ✅ User-friendly workflow for biomedical researchers

### OmicSelector2 Modernization Goals

| Component | OmicSelector 1.0 | OmicSelector2 |
|-----------|------------------|---------------|
| **Language** | R | Python 3.11+ |
| **Backend** | Shiny Server | FastAPI |
| **Frontend** | Shiny UI | React + Material-UI |
| **DL Framework** | Keras (R) | PyTorch + PyTorch Geometric |
| **Feature Selection** | 70 R packages | SOTA methods (2024-2025) + classical |
| **Multi-omics** | Concatenation | GNNs, VAEs, attention mechanisms |
| **Single-cell** | Limited | Native Scanpy integration |
| **Job Queue** | Sequential/parallel R | Celery with Redis |
| **API** | None | RESTful with OpenAPI docs |
| **Database** | File-based | PostgreSQL + S3/MinIO |
| **Visualization** | ggplot2, plotly R | CanvasXpress + Plotly.js |
| **Experiment Tracking** | Manual | MLflow |
| **Deployment** | Docker (monolithic) | Microservices + Docker Compose |

---

## 🏗️ TECHNOLOGY STACK

### Backend: FastAPI

**Why FastAPI over Flask?**
- 5-8x faster (15-20K req/s vs 2-3K req/s)
- Native async/await for concurrent biomarker jobs
- Automatic OpenAPI documentation (essential for research reproducibility)
- Pydantic validation reduces bugs in complex pipelines
- Modern, future-proof (78.9k GitHub stars)

```python
# Example API endpoint structure
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="OmicSelector2 API", version="2.0.0")

class FeatureSelectionRequest(BaseModel):
    dataset_id: str
    methods: list[str]
    cv_folds: int = 5
    n_features: int = 100

@app.post("/api/v1/feature-selection/")
async def run_feature_selection(request: FeatureSelectionRequest):
    """Initiate feature selection job."""
    # Implementation
    pass
```

### Deep Learning: PyTorch + PyTorch Geometric

**Primary Framework: PyTorch**
- Dynamic computational graphs for variable-length sequences
- Selene library for biological sequences (Nature Methods published)
- Research-friendly debugging
- Growing genomics adoption

**Graph Neural Networks: PyTorch Geometric (PyG)**
- Essential for multi-omics integration via heterogeneous graphs
- GCN, GAT, GraphSAGE implementations
- Mini-batch support for large biological networks
- Integration with NetworkX, igraph

**Secondary: scikit-learn**
- Classical ML (Random Forest, SVM, Logistic Regression)
- Feature selection baselines
- Preprocessing pipelines
- Cross-validation framework

### Database: PostgreSQL + S3-compatible Storage

**PostgreSQL 15+**:
- ACID compliance for experiment tracking
- JSONB for flexible metadata storage
- Array types for gene expression profiles
- Full-text search for annotations
- Proven in genomics (TCGA, GEO)

**File Storage (S3/MinIO)**:
- AnnData objects (.h5ad) for expression matrices
- Trained models (.pt files)
- Large visualizations
- Versioned datasets

**Redis**:
- Celery backend for job queue
- Real-time pipeline status
- Result caching

### Frontend: React + Material-UI

**React 18+**:
- Largest ecosystem for data visualization
- Best D3.js/Plotly.js compatibility
- Mature component libraries

**Material-UI (MUI) v5+**:
- Comprehensive component set
- MUI X Data Grid for large result tables
- Smaller bundle than alternatives
- Excellent documentation

### Visualization: CanvasXpress + Plotly

**CanvasXpress** (Primary):
- Purpose-built for bioinformatics
- 40+ specialized graph types (volcano plots, heatmaps, Kaplan-Meier, dendrograms)
- HTML5 Canvas for performance (thousands of data points)
- Audit trail for reproducibility

**Plotly.js** (Secondary):
- 3D visualizations (UMAP, t-SNE)
- Interactive dashboards

### Job Queue: Celery with Redis

- Most feature-rich (retries, scheduling, chaining, priorities)
- Production-grade for complex workflows
- Flower monitoring dashboard
- Proven at scale (AstraZeneca genomics, Instagram)

### Bioinformatics Libraries

**Essential**:
- **Scanpy** - Single-cell RNA-seq (scverse ecosystem)
- **PyDESeq2** - Bulk RNA-seq differential expression
- **Biopython** - General bioinformatics
- **Pysam** - SAM/BAM/VCF handling
- **lifelines** - Survival analysis
- **pyradiomics** - Radiomics feature extraction
- **statsmodels** / **pingouin** - Statistical testing

**MLOps**:
- **MLflow** - Experiment tracking, model registry
- **DVC** - Data version control
- **Weights & Biases** (optional) - Advanced experiment tracking

---

## 🧬 STATE-OF-THE-ART METHODS (2024-2025 Research)

### Multi-Omics Integration Methods

#### 1. Graph Neural Networks (GNNs) - PRIORITY 1

**Research Basis**:
Recent research demonstrates that GNN-based architectures excel at multi-omics cancer classification by modeling heterogeneous biological networks, with graph-attention networks (GAT) preferred for smaller graphs and graph-convolutional networks (GCN) for larger graphs with more information.

**Key Frameworks to Implement**:

**a) Heterogeneous Multi-Layer GNNs**:
```python
# Architecture: Combine inter-omics and intra-omic connections
class MultiOmicGNN(nn.Module):
    """Multi-layer GNN for multi-omics integration.
    
    Based on:
    - Li & Nabavi (2024) - BMC Bioinformatics
    - Comparative Analysis (2025) - IEEE Access
    """
    def __init__(self, omics_dims: dict[str, int], hidden_dim: int = 128):
        super().__init__()
        # Separate encoders for each omics
        self.rna_encoder = GATConv(omics_dims['rna'], hidden_dim)
        self.methylation_encoder = GATConv(omics_dims['methylation'], hidden_dim)
        self.cnv_encoder = GATConv(omics_dims['cnv'], hidden_dim)
        
        # Cross-omics attention
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

**b) MOGDx-Inspired Architecture**:
Multi-Omic Graph Diagnosis (MOGDx) approach uses heterogeneous graphs to integrate multi-omics data for disease classification.

```python
# Patient-specific heterogeneous graph construction
def build_patient_graph(
    rna_seq: np.ndarray,
    methylation: np.ndarray,
    cnv: np.ndarray,
    ppi_network: nx.Graph,
    pathway_db: dict
) -> Data:
    """Build heterogeneous graph for single patient.
    
    Node types:
    - Genes (RNA expression)
    - CpG sites (methylation)
    - Copy number regions
    
    Edge types:
    - PPI edges (gene-gene)
    - Regulatory edges (CpG-gene)
    - Pathway edges
    """
    pass
```

**c) Geometric GNN with Ricci Curvature**:
Geometric features like Ollivier-Ricci curvature on protein-protein interaction networks have been shown to be predictive of survival outcomes in multiple cancers.

```python
# Incorporate geometric features
class GeometricGNN(nn.Module):
    """GNN with geometric network features.
    
    Based on: Zhu et al. (2023) - Comput Biol Med
    Incorporates Ollivier-Ricci curvature
    """
    def compute_ricci_curvature(self, graph: nx.Graph) -> dict:
        """Compute Ollivier-Ricci curvature for edges."""
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
        orc = OllivierRicci(graph, alpha=0.5)
        orc.compute_ricci_curvature()
        return orc.G.edges
```

#### 2. Attention-Based Multi-Omics Integration

**MORE Framework** :
- Multi-omics hypergraph encoding module
- Multi-omics self-attention mechanism
- Demonstrated competitive advantage for Alzheimer's, breast cancer, glioblastoma

```python
class MOREModel(nn.Module):
    """Multi-Omics data-driven hypergraph integration network.
    
    Based on: Wang et al. (2025) - Briefings in Bioinformatics
    """
    def __init__(self, omics_features: dict[str, int]):
        super().__init__()
        # Hypergraph encoding for each omics
        self.hypergraph_encoders = nn.ModuleDict({
            name: HypergraphEncoder(dim)
            for name, dim in omics_features.items()
        })
        
        # Multi-omics self-attention
        self.mosa = MultiOmicsSelfAttention(hidden_dim=256, num_heads=8)
        
    def forward(self, omics_data: dict[str, torch.Tensor]):
        # Encode each omics through hypergraph
        encoded = {
            name: encoder(data)
            for name, (encoder, data) in zip(
                self.hypergraph_encoders.items(),
                omics_data.items()
            )
        }
        
        # Cross-omics attention
        integrated = self.mosa(encoded)
        return integrated
```

#### 3. Variational Autoencoders for Multi-Omics

**MultiVI (scvi-tools)**: Joint analysis of paired and unpaired multiomic data

```python
# Integration with scvi-tools for single-cell multi-omics
import scvi

# For paired scRNA-seq + scATAC-seq
scvi.model.MULTIVI.setup_anndata(
    adata,
    batch_key="batch",
    modality_key="modality",
    protein_expression_obsm_key="protein_expression"
)

model = scvi.model.MULTIVI(adata)
model.train()

# Get integrated latent representation
latent = model.get_latent_representation()
```

#### 4. Stability Selection Framework

Stability selection with canonical correlation analysis (StabilityCCA) improves variable selection stability in multi-omics biomarker discovery.

```python
class StabilitySelector:
    """Stability selection framework for multi-omics.
    
    Based on: Pusa & Rousu (2024) - PLoS One
    """
    def __init__(
        self,
        base_selector: Callable,
        n_bootstraps: int = 100,
        threshold: float = 0.6,
        sample_fraction: float = 0.8
    ):
        self.base_selector = base_selector
        self.n_bootstraps = n_bootstraps
        self.threshold = threshold
        self.sample_fraction = sample_fraction
        
    def select_stable_features(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> list[int]:
        """Select features that are stably selected across bootstraps."""
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.sample_fraction)
        
        selection_counts = np.zeros(X.shape[1])
        
        for _ in range(self.n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Run base selector
            selected = self.base_selector(X_boot, y_boot)
            selection_counts[selected] += 1
            
        # Select features above threshold
        stability_scores = selection_counts / self.n_bootstraps
        stable_features = np.where(stability_scores >= self.threshold)[0]
        
        return stable_features, stability_scores
```

### Feature Selection Methods - PRIORITY LEVELS

#### PRIORITY 1 (v1.0 - Core Methods)

**Classical Statistical**:
1. **Lasso** - L1 regularization, FDA-cleared in diagnostic panels
2. **Elastic Net** - L1+L2, handles correlated features
3. **t-test/ANOVA** - Pre-filtering, dimension reduction
4. **Cox Proportional Hazards** - Survival analysis standard

**Tree-Based**:
5. **Random Forest Variable Importance (RF-VI)** - Robust, interpretable
6. **XGBoost Feature Importance** - State-of-the-art gradient boosting
7. **Boruta** - Wrapper around RF, identifies "important" features

**Filter Methods**:
8. **mRMR** (Minimum Redundancy Maximum Relevance) - Mutual information based
9. **ReliefF** - Instance-based, handles interactions
10. **Variance Threshold** - Remove low-variance features

**Embedded Methods**:
11. **L1-SVM** - Embedded feature selection in SVM
12. **Ridge Regression** - L2 regularization for stability

#### PRIORITY 2 (v1.5 - Advanced Methods)

**Deep Learning-Based**:
13. **Concrete Autoencoders** - Differentiable feature selection
14. **Neural Feature Selection (NFS)** - Learned feature gates
15. **INVASE** (Instance-wise Variable Selection) - Interpretable selection

**Graph-Based**:
16. **GNN Feature Importance** - Attention weights from GNNs
17. **Network-Based Feature Selection** - Use PPI networks
18. **Community Detection Features** - Pathway/module-based

**Single-Cell Specific**:
19. **HDG** (High-Deviation Genes) - Single-cell variability
20. **HEG** (High-Expression Genes) - Mean expression filtering
21. **DUBStepR** - Determining Useful Biomarker Suites via Step-wise Regression
22. **FEAST** - Fast Entropy-based Active Set Testing

**Ensemble Methods**:
23. **Stability Selection** - Bootstrap aggregation
24. **MOGONET-inspired** - Multi-omics GNN ensemble

#### PRIORITY 3 (v2.0 - Cutting Edge 2024-2025)

**Novel Methods from Recent Research**:

25. **Quantum Annealing Feature Selection** : Quantum computing approaches for scRNA-seq feature selection

26. **Symbolic Regression Feature Selection** : Identifying interactions in omics data using symbolic regression for clinical biomarker discovery

27. **Radiomics-Genomics Fusion** : Deep learning feature fusion models integrating radiomics with multi-omics data

28. **Patient-Specific Radiomic Features** : Reconstructed healthy persona approach for personalized feature selection

29. **Contrastive Learning Features** - Self-supervised pre-training for feature discovery

30. **Graph Attention Explainability** : CGMega framework with attention mechanisms for cancer gene module dissection

---

## 🏛️ SYSTEM ARCHITECTURE

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (React + MUI)                        │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────────────┐   │
│  │ Data Upload   │  │  Workflow      │  │  Visualization       │   │
│  │ & Validation  │  │  Builder       │  │  Dashboard           │   │
│  └───────────────┘  └────────────────┘  └──────────────────────┘   │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────────────┐   │
│  │ Method Config │  │  Results       │  │  Model Registry      │   │
│  │ & Selection   │  │  Explorer      │  │  & Deployment        │   │
│  └───────────────┘  └────────────────┘  └──────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ REST API (OpenAPI/JSON)
┌────────────────────────▼────────────────────────────────────────────┐
│                     API Layer (FastAPI)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ Auth & Users │  │ Data Upload  │  │ Jobs & Tasks │  │ Results │ │
│  │ /api/v1/auth │  │ /api/v1/data │  │ /api/v1/jobs │  │ /api/v1 │ │
│  │              │  │              │  │              │  │ /results│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │           WebSocket (Real-time progress updates)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│              Job Queue (Celery + Redis Broker)                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ high_priority   │  │ default         │  │ low_priority     │   │
│  │ (interactive)   │  │ (batch jobs)    │  │ (background)     │   │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘   │
└──────┬─────────────────────────┬──────────────────────────────────┘
       │                         │
┌──────▼──────────┐      ┌───────▼─────────────────┐
│  CPU Workers    │      │   GPU Workers           │
│  (Classical ML, │      │   (Deep Learning:       │
│  Preprocessing, │      │   GNNs, VAEs, NNs)      │
│  Statistics)    │      │                         │
└──────┬──────────┘      └───────┬─────────────────┘
       │                         │
┌──────▼─────────────────────────▼─────────────────────────────────┐
│                    Storage & Persistence Layer                    │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  PostgreSQL 15+  │  │ S3/MinIO       │  │ Redis Cache     │  │
│  │  - Metadata      │  │ - .h5ad files  │  │ - Job status    │  │
│  │  - Experiments   │  │ - Models (.pt) │  │ - Results cache │  │
│  │  - Results       │  │ - Large outputs│  │                 │  │
│  │  - User data     │  │                │  │                 │  │
│  └──────────────────┘  └────────────────┘  └─────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              MLflow Tracking Server                          │ │
│  │              - Experiment tracking                           │ │
│  │              - Model registry                                │ │
│  │              - Artifact storage                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### Microservices Architecture

**Service Decomposition**:

1. **API Service** (`api-service/`)
   - FastAPI application
   - Authentication & authorization
   - Request routing
   - WebSocket connections
   - OpenAPI documentation

2. **Data Service** (`data-service/`)
   - Data ingestion & validation
   - Format conversion (FASTQ, BAM, VCF, CSV, h5ad)
   - Quality control
   - Preprocessing pipelines
   - Data versioning

3. **Feature Selection Service** (`feature-service/`)
   - Classical methods (Lasso, RF-VI, mRMR, etc.)
   - Stability selection framework
   - Ensemble voting
   - Feature ranking

4. **Deep Learning Service** (`dl-service/`)
   - GNN training & inference
   - VAE training
   - Attention mechanisms
   - GPU-accelerated computation

5. **Model Service** (`model-service/`)
   - Model training (classical ML)
   - Hyperparameter optimization
   - Cross-validation
   - Model evaluation

6. **Inference Service** (`inference-service/`)
   - Model serving
   - Batch prediction
   - SHAP explanations
   - API for deployed models

7. **Visualization Service** (`viz-service/`)
   - Generate plots (volcano, heatmap, KM curves)
   - CanvasXpress integration
   - Interactive dashboards

### Module Structure

```
omicselector2/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── routes/
│   │   │   ├── auth.py
│   │   │   ├── data.py
│   │   │   ├── jobs.py
│   │   │   ├── results.py
│   │   │   └── models.py
│   │   ├── dependencies.py     # FastAPI dependencies
│   │   ├── middleware.py       # Auth, logging, CORS
│   │   └── websockets.py       # Real-time updates
│   │
│   ├── data/                   # Data handling
│   │   ├── loaders.py         # File format loaders
│   │   ├── validators.py      # Data validation
│   │   ├── preprocessors.py   # Normalization, filtering
│   │   ├── schema.py          # Pydantic data models
│   │   └── converters.py      # Format conversion
│   │
│   ├── features/              # Feature selection
│   │   ├── classical/
│   │   │   ├── lasso.py
│   │   │   ├── elastic_net.py
│   │   │   ├── random_forest.py
│   │   │   ├── mrmr.py
│   │   │   └── boruta.py
│   │   ├── graph_based/
│   │   │   ├── gnn_importance.py
│   │   │   └── network_features.py
│   │   ├── deep_learning/
│   │   │   ├── concrete_ae.py
│   │   │   ├── invase.py
│   │   │   └── nfs.py
│   │   ├── single_cell/
│   │   │   ├── hdg.py
│   │   │   ├── feast.py
│   │   │   └── dubstepr.py
│   │   ├── registry.py        # Method registry
│   │   ├── stability.py       # Stability selection
│   │   └── ensemble.py        # Ensemble methods
│   │
│   ├── models/                # ML/DL models
│   │   ├── base.py            # Abstract base classes
│   │   ├── classical/
│   │   │   ├── random_forest.py
│   │   │   ├── svm.py
│   │   │   ├── logistic_regression.py
│   │   │   └── xgboost.py
│   │   ├── neural/
│   │   │   ├── feedforward.py
│   │   │   ├── autoencoders.py
│   │   │   └── attention.py
│   │   ├── graph/
│   │   │   ├── gcn.py
│   │   │   ├── gat.py
│   │   │   ├── graphsage.py
│   │   │   ├── multimodal_gnn.py
│   │   │   └── geometric_gnn.py
│   │   ├── multi_omics/
│   │   │   ├── mogdx.py
│   │   │   ├── more.py
│   │   │   └── multivi.py
│   │   └── survival.py        # Cox PH, survival models
│   │
│   ├── training/              # Training orchestration
│   │   ├── trainer.py         # Training loops
│   │   ├── evaluator.py       # Metrics, cross-validation
│   │   ├── hyperparameter.py  # Optuna integration
│   │   └── experiment.py      # MLflow tracking
│   │
│   ├── inference/             # Model serving
│   │   ├── predictor.py       # Prediction logic
│   │   ├── explainer.py       # SHAP integration
│   │   └── validator.py       # Input validation
│   │
│   ├── visualization/         # Plotting
│   │   ├── plots.py           # Volcano, heatmap, etc.
│   │   ├── canvasxpress.py    # CanvasXpress integration
│   │   └── dashboards.py      # Interactive dashboards
│   │
│   ├── tasks/                 # Celery tasks
│   │   ├── feature_selection.py
│   │   ├── model_training.py
│   │   ├── preprocessing.py
│   │   └── benchmarking.py
│   │
│   └── utils/                 # Utilities
│       ├── config.py          # Configuration management
│       ├── logging.py         # Structured logging
│       ├── security.py        # Encryption, auth helpers
│       └── metrics.py         # Performance metrics
│
├── tests/                     # Test suite
│   ├── unit/
│   │   ├── test_loaders.py
│   │   ├── test_validators.py
│   │   ├── test_features.py
│   │   └── test_models.py
│   ├── integration/
│   │   ├── test_api.py
│   │   ├── test_workflows.py
│   │   └── test_pipelines.py
│   └── fixtures/
│       ├── test_data.h5ad
│       └── mock_datasets.py
│
├── frontend/                  # React application
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/          # API client
│   │   └── utils/
│   └── public/
│
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── user_guide/            # User manuals
│   ├── developer/             # Developer docs
│   └── examples/              # Jupyter notebooks
│
├── docker/                    # Docker configurations
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── scripts/                   # Utility scripts
│   ├── setup_db.py
│   ├── seed_data.py
│   └── deploy.sh
│
├── pyproject.toml             # Project dependencies
├── pytest.ini                 # Pytest configuration
├── .env.example               # Environment variables template
├── CLAUDE.md                  # This file
└── README.md                  # Project overview
```

---

## 🔄 DEVELOPMENT WORKFLOW

### Phase 1: Core Infrastructure (Weeks 1-4)

**Epic 1.1: Project Setup & CI/CD**
- [ ] Initialize Python project with Poetry/pip-tools
- [ ] Set up Git repository with branching strategy
- [ ] Configure pre-commit hooks (black, isort, flake8, mypy)
- [ ] Set up GitHub Actions CI/CD pipeline
- [ ] Configure Docker & Docker Compose
- [ ] Set up development, staging, production environments

**Epic 1.2: Database & Storage**
- [ ] Design PostgreSQL schema
- [ ] Implement Alembic migrations
- [ ] Set up MinIO/S3 for file storage
- [ ] Implement Redis caching layer
- [ ] Write database access layer with SQLAlchemy
- [ ] Create seed data for testing

**Epic 1.3: API Foundation**
- [ ] Create FastAPI application structure
- [ ] Implement authentication (JWT)
- [ ] Set up API versioning (/api/v1/)
- [ ] Configure CORS, middleware
- [ ] Implement WebSocket connections
- [ ] Generate OpenAPI documentation

**Epic 1.4: Job Queue**
- [ ] Set up Celery with Redis broker
- [ ] Implement task routing (CPU/GPU queues)
- [ ] Configure Flower monitoring
- [ ] Implement task retry logic
- [ ] Set up result backends
- [ ] Create job status tracking

### Phase 2: Data Layer (Weeks 5-8)

**Epic 2.1: Data Ingestion**
- [ ] Implement file upload API endpoint
- [ ] Create validators for: CSV, h5ad, VCF, BAM, FASTQ
- [ ] Implement format converters
- [ ] Build data quality control pipeline
- [ ] Create data versioning system
- [ ] Implement metadata extraction

**Epic 2.2: Preprocessing Pipeline**
- [ ] Implement normalization methods (TPM, RPKM, CPM, TMM)
- [ ] Create filtering functions (variance, expression)
- [ ] Implement batch effect correction (ComBat)
- [ ] Build imputation methods (KNN, MICE)
- [ ] Create train/test/validation splitting
- [ ] Implement feature scaling

**Epic 2.3: Single-Cell Integration**
- [ ] Integrate Scanpy for scRNA-seq
- [ ] Implement QC metrics (UMI count, % mitochondrial)
- [ ] Create normalization pipeline (scran, sctransform)
- [ ] Implement dimensionality reduction (PCA, UMAP, t-SNE)
- [ ] Build clustering workflows (Leiden, Louvain)
- [ ] Create cell type annotation tools

### Phase 3: Feature Selection Core (Weeks 9-14)

**Epic 3.1: Classical Methods (Priority 1)**
- [ ] Implement Lasso (scikit-learn wrapper)
- [ ] Implement Elastic Net
- [ ] Create t-test/ANOVA filters
- [ ] Implement Random Forest VI
- [ ] Create Boruta wrapper
- [ ] Implement mRMR
- [ ] Create XGBoost feature importance
- [ ] Implement variance thresholding
- [ ] Create ReliefF implementation

**Epic 3.2: Stability Selection Framework**
- [ ] Implement bootstrap sampling
- [ ] Create stability score calculation
- [ ] Build threshold selection
- [ ] Implement feature ranking aggregation
- [ ] Create visualization of stability paths
- [ ] Add parallel execution support

**Epic 3.3: Ensemble Methods**
- [ ] Implement majority voting
- [ ] Create soft voting (weighted)
- [ ] Build consensus ranking
- [ ] Implement intersection/union strategies
- [ ] Create meta-learner approach

### Phase 4: Deep Learning Integration (Weeks 15-20)

**Epic 4.1: GNN Infrastructure**
- [ ] Set up PyTorch Geometric
- [ ] Implement graph construction from PPI networks
- [ ] Create patient-specific heterogeneous graphs
- [ ] Build data loaders for graph batching
- [ ] Implement GCN layer
- [ ] Create GAT layer
- [ ] Build GraphSAGE implementation

**Epic 4.2: Multi-Omics GNN Models**
- [ ] Implement MultiOmic GNN (Priority 1)
- [ ] Create MOGDx architecture
- [ ] Build Geometric GNN with Ricci curvature
- [ ] Implement attention-based integration
- [ ] Create MOGAT framework
- [ ] Build graph contrastive learning

**Epic 4.3: VAE & Autoencoders**
- [ ] Implement concrete autoencoders
- [ ] Create variational autoencoders
- [ ] Build MultiVI integration (scvi-tools)
- [ ] Implement conditional VAE
- [ ] Create sparse autoencoders

**Epic 4.4: Attention Mechanisms**
- [ ] Implement multi-head attention
- [ ] Create cross-modal attention
- [ ] Build MORE architecture
- [ ] Implement self-attention for omics
- [ ] Create attention visualization

### Phase 5: Model Training & Evaluation (Weeks 21-24)

**Epic 5.1: Training Pipeline**
- [ ] Implement training loop abstraction
- [ ] Create cross-validation framework
- [ ] Build hyperparameter optimization (Optuna)
- [ ] Implement early stopping
- [ ] Create learning rate scheduling
- [ ] Build model checkpointing

**Epic 5.2: MLflow Integration**
- [ ] Set up MLflow tracking server
- [ ] Implement experiment logging
- [ ] Create parameter tracking
- [ ] Build metric tracking
- [ ] Implement artifact storage
- [ ] Create model registry

**Epic 5.3: Evaluation Metrics**
- [ ] Implement classification metrics (AUC, accuracy, F1)
- [ ] Create survival analysis metrics (C-index, IBS)
- [ ] Build calibration plots
- [ ] Implement confusion matrices
- [ ] Create ROC/PR curves
- [ ] Build statistical tests

**Epic 5.4: Interpretability**
- [ ] Integrate SHAP
- [ ] Implement LIME
- [ ] Create attention visualization
- [ ] Build feature importance plots
- [ ] Implement integrated gradients
- [ ] Create decision path visualization

### Phase 6: Benchmarking System (Weeks 25-28)

**Epic 6.1: Automated Benchmarking** (Core OmicSelector philosophy)
- [ ] Implement signature testing framework
- [ ] Create performance comparison table
- [ ] Build ranking system
- [ ] Implement statistical significance tests
- [ ] Create benchmarking reports
- [ ] Add visualization dashboard

**Epic 6.2: Model Comparison**
- [ ] Implement paired t-tests
- [ ] Create DeLong test for AUC
- [ ] Build calibration comparison
- [ ] Implement cost-benefit analysis
- [ ] Create meta-analysis tools

### Phase 7: Frontend Development (Weeks 29-34)

**Epic 7.1: React Application Setup**
- [ ] Initialize React project (Vite/Create React App)
- [ ] Configure Material-UI
- [ ] Set up routing (React Router)
- [ ] Create API client (Axios)
- [ ] Implement state management (Redux/Zustand)
- [ ] Configure build pipeline

**Epic 7.2: Core UI Components**
- [ ] Create data upload interface
- [ ] Build method selection wizard
- [ ] Implement parameter configuration forms
- [ ] Create job monitoring dashboard
- [ ] Build result explorer
- [ ] Implement model registry browser

**Epic 7.3: Visualization Components**
- [ ] Integrate CanvasXpress
- [ ] Create volcano plot component
- [ ] Build heatmap visualizer
- [ ] Implement Kaplan-Meier plot
- [ ] Create UMAP/t-SNE viewer
- [ ] Build ROC curve plotter
- [ ] Implement SHAP summary plots

**Epic 7.4: Workflow Builder**
- [ ] Create drag-and-drop workflow designer
- [ ] Implement node-based pipeline editor
- [ ] Build parameter templates
- [ ] Create workflow validation
- [ ] Implement workflow versioning

### Phase 8: Deployment & Documentation (Weeks 35-36)

**Epic 8.1: Production Deployment**
- [ ] Create production Docker images
- [ ] Set up Kubernetes manifests (optional)
- [ ] Configure nginx reverse proxy
- [ ] Implement SSL/TLS
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Create backup strategies
- [ ] Implement logging aggregation

**Epic 8.2: Documentation**
- [ ] Write API documentation (OpenAPI)
- [ ] Create user guide
- [ ] Build tutorial notebooks (Jupyter)
- [ ] Write developer documentation
- [ ] Create deployment guide
- [ ] Write troubleshooting guide

**Epic 8.3: Testing & QA**
- [ ] Achieve >80% code coverage
- [ ] Perform integration testing
- [ ] Conduct user acceptance testing
- [ ] Perform security audit
- [ ] Load testing
- [ ] Create regression test suite

---

## 📊 EXAMPLE WORKFLOWS

### Workflow 1: Bulk RNA-seq Biomarker Discovery (Standard Pipeline)

**User Story**: *"As a cancer researcher, I want to identify a robust gene signature from RNA-seq data that can predict patient survival, so that I can validate these biomarkers in a clinical trial."*

```python
# Python SDK usage
from omicselector2 import OmicSelector

# 1. Initialize client
client = OmicSelector(api_url="https://api.omicselector.org", api_key="...")

# 2. Upload data
dataset = client.upload_data(
    file="cancer_rnaseq_counts.csv",
    data_type="bulk_rna_seq",
    metadata={
        "study": "TCGA-BLCA",
        "outcome": "survival_months",
        "covariates": ["age", "gender", "stage"]
    }
)

# 3. Preprocess
preprocessed = client.preprocess(
    dataset_id=dataset.id,
    config={
        "normalization": "tpm",
        "log_transform": True,
        "variance_filter": 0.01,
        "batch_correction": "combat",
        "train_test_split": {"test_size": 0.2, "validation_size": 0.2, "stratify": "outcome"}
    }
)

# 4. Feature selection with stability + ensemble
job = client.create_feature_selection_job(
    dataset_id=preprocessed.id,
    methods=[
        "lasso",
        "elastic_net",
        "rf_importance",
        "mrmr",
        "boruta",
        "xgboost"
    ],
    config={
        "cv_folds": 5,
        "stability": {
            "enabled": True,
            "n_bootstraps": 100,
            "threshold": 0.7
        },
        "ensemble": {
            "enabled": True,
            "voting": "soft",
            "min_methods": 3  # Feature must be selected by ≥3 methods
        },
        "n_features_range": [10, 50, 100]  # Test multiple signature sizes
    }
)

# 5. Monitor job
job.wait_for_completion(verbose=True)

# 6. Get results
results = client.get_results(job.id)

# Access selected features
stable_features = results.stable_features  # Features selected across bootstraps
ensemble_features = results.ensemble_features  # Features with high consensus

# Access stability scores
print(f"Top 10 stable features:")
for feature, score in results.stability_scores[:10]:
    print(f"{feature}: {score:.3f}")

# 7. Benchmark signatures
benchmark_job = client.benchmark_signatures(
    dataset_id=preprocessed.id,
    signatures={
        "Lasso_10": results.methods["lasso"]["top_10"],
        "Ensemble_20": ensemble_features[:20],
        "Stable_30": stable_features[:30]
    },
    models=[
        "random_forest",
        "logistic_regression",
        "xgboost",
        "cox_ph"
    ]
)

benchmark_job.wait_for_completion()
benchmark_results = client.get_benchmark_results(benchmark_job.id)

# View performance comparison
print(benchmark_results.summary_table)
# Output:
# | Signature     | Model              | CV AUC | Test AUC | C-index | p-value |
# |---------------|--------------------|---------:|---------:|----------:|----------:|
# | Ensemble_20   | Random Forest      | 0.87   | 0.84    | 0.71    | 0.001   |
# | Stable_30     | Cox PH             | 0.85   | 0.82    | 0.73    | 0.003   |
# | ...

# 8. Train final model with best signature
best_signature = benchmark_results.best_signature  # "Ensemble_20" + "Random Forest"

model = client.train_model(
    dataset_id=preprocessed.id,
    features=best_signature.features,
    algorithm="random_forest",
    hyperparameters="auto",  # Optuna optimization
    save=True
)

# 9. Generate interpretability report
shap_values = client.explain_model(
    model_id=model.id,
    output_format="html"
)

# Save report
shap_values.save("biomarker_report.html")

# 10. Export for publication
client.export_results(
    job_id=job.id,
    format="publication_ready",  # Includes: feature list, stability plots, performance tables
    output_dir="./publication_materials/"
)
```

### Workflow 2: Multi-Omics GNN Integration (Advanced)

**User Story**: *"As a computational biologist, I want to integrate RNA-seq, DNA methylation, and copy number data using graph neural networks to discover multi-omic biomarkers for bladder cancer subtypes."*

```python
from omicselector2 import OmicSelector, MultiOmicGNN

client = OmicSelector(api_url="...")

# 1. Upload multi-omics data
rna_data = client.upload_data("bladder_rnaseq.h5ad", data_type="bulk_rna_seq")
methyl_data = client.upload_data("bladder_methylation.csv", data_type="methylation")
cnv_data = client.upload_data("bladder_cnv.csv", data_type="copy_number")

# 2. Create multi-omics dataset
multi_omics = client.create_multiomics_dataset(
    datasets={
        "rna": rna_data.id,
        "methylation": methyl_data.id,
        "cnv": cnv_data.id
    },
    alignment="sample_id",  # How to match samples across omics
    metadata={
        "outcome": "subtype",  # Classification: Luminal, Basal, Neuronal
        "clinical": "clinical_data.csv"
    }
)

# 3. Preprocess each omics layer
preprocessed = client.preprocess_multiomics(
    dataset_id=multi_omics.id,
    config={
        "rna": {
            "normalization": "tpm",
            "log_transform": True,
            "variance_filter": 0.05
        },
        "methylation": {
            "normalization": "minmax",
            "filter_na": True
        },
        "cnv": {
            "threshold": 0.3  # Keep regions with |log2ratio| > 0.3
        }
    }
)

# 4. Build heterogeneous graph
graph_config = {
    "ppi_network": "STRING",  # Use STRING database
    "ppi_threshold": 0.7,     # High-confidence interactions only
    "regulatory_edges": True,  # Add methylation → gene regulation edges
    "pathway_db": "KEGG",     # Add pathway-based edges
    "graph_type": "heterogeneous"  # Different node/edge types
}

graph_dataset = client.build_graph(
    dataset_id=preprocessed.id,
    config=graph_config
)

# 5. Train Multi-Omics GNN
gnn_job = client.train_gnn(
    dataset_id=graph_dataset.id,
    model_type="multimodal_gnn",  # MOGDx-inspired architecture
    config={
        "architecture": {
            "gnn_layers": [
                {"type": "GATConv", "hidden_dim": 128, "heads": 8},
                {"type": "GATConv", "hidden_dim": 64, "heads": 4}
            ],
            "cross_attention": {
                "enabled": True,
                "num_heads": 8
            },
            "omics_encoders": {
                "rna": {"hidden_dim": 256},
                "methylation": {"hidden_dim": 128},
                "cnv": {"hidden_dim": 64}
            }
        },
        "training": {
            "epochs": 200,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": {"patience": 20},
            "cv_folds": 5
        },
        "task": "classification",
        "classes": ["Luminal", "Basal", "Neuronal"]
    }
)

gnn_job.wait_for_completion()

# 6. Extract biomarker features
gnn_results = client.get_gnn_results(gnn_job.id)

# Get important features via attention weights
important_genes = gnn_results.get_important_features(
    method="attention",  # Use attention weights as importance
    top_n=50
)

# Get graph modules (communities of interacting genes)
gene_modules = gnn_results.get_gene_modules(
    algorithm="louvain",
    min_size=5
)

print(f"Identified {len(gene_modules)} gene modules")
for i, module in enumerate(gene_modules[:3]):
    print(f"Module {i+1}: {len(module)} genes")
    print(f"  Top genes: {', '.join(module[:5])}")
    print(f"  Enriched pathways: {module.pathway_enrichment[:3]}")

# 7. Interpretability: Explain subtype predictions
explainer = client.explain_gnn(
    model_id=gnn_results.model_id,
    explainer_type="GNNExplainer",  # PyG GNNExplainer
    sample_ids=["TCGA-BT-A20J", "TCGA-DK-A1AB"]  # Example samples
)

# Generate subgraph explanations
explainer.visualize_subgraphs(output_dir="./gnn_explanations/")

# 8. Compare with classical approach
# For comparison, run same task with concatenated features
classical_job = client.create_feature_selection_job(
    dataset_id=preprocessed.id,
    methods=["lasso", "rf_importance"],
    config={
        "feature_concatenation": "early_fusion",  # Concatenate all omics
        "cv_folds": 5
    }
)

classical_job.wait_for_completion()

# Compare performance
comparison = client.compare_approaches(
    approach_1=gnn_job.id,
    approach_2=classical_job.id,
    metrics=["accuracy", "f1_macro", "auc_ovr"]
)

print(comparison.summary)
# Output:
# | Method            | Accuracy | F1 (macro) | AUC (OvR) |
# |-------------------|----------:|-----------:|----------:|
# | Multi-Omics GNN   | 0.89     | 0.87       | 0.93      |
# | Lasso (concat)    | 0.76     | 0.73       | 0.84      |
# | RF-VI (concat)    | 0.81     | 0.78       | 0.87      |
```

### Workflow 3: Single-Cell Feature Selection

**User Story**: *"As a single-cell researcher, I want to identify marker genes for cancer cell subpopulations in my scRNA-seq data."*

```python
# 1. Upload scRNA-seq data (h5ad format)
sc_data = client.upload_data(
    "tumor_scrna.h5ad",
    data_type="single_cell_rna_seq"
)

# 2. QC and preprocessing
qc_results = client.run_qc(
    dataset_id=sc_data.id,
    config={
        "min_genes": 200,
        "max_genes": 5000,
        "max_mito_pct": 10,
        "doublet_detection": True
    }
)

preprocessed = client.preprocess_singlecell(
    dataset_id=sc_data.id,
    config={
        "normalization": "scran",  # Size factor normalization
        "log_transform": True,
        "highly_variable_genes": 2000,
        "regress_out": ["n_counts", "pct_mito"],
        "batch_correction": "harmony"  # If multiple batches
    }
)

# 3. Clustering
clustering = client.cluster_singlecell(
    dataset_id=preprocessed.id,
    config={
        "method": "leiden",
        "resolution": 0.5,
        "n_neighbors": 15,
        "umap_params": {"min_dist": 0.3}
    }
)

# 4. Feature selection for clustering
feature_job = client.create_feature_selection_job(
    dataset_id=preprocessed.id,
    methods=[
        "hdg",        # High-deviation genes
        "feast",      # Fast Entropy-based testing
        "dubstepr"    # scRNA-seq specific
    ],
    config={
        "n_top_genes": 2000,
        "cluster_column": "leiden"
    }
)

feature_job.wait_for_completion()
results = client.get_results(feature_job.id)

# 5. Differential expression (marker identification)
de_results = client.differential_expression(
    dataset_id=preprocessed.id,
    config={
        "method": "wilcoxon",
        "cluster_column": "leiden",
        "comparison": "one_vs_rest",
        "min_pct": 0.25,
        "logfc_threshold": 0.5
    }
)

# Get top markers per cluster
for cluster_id, markers in de_results.markers_per_cluster.items():
    print(f"Cluster {cluster_id} top markers:")
    for gene in markers[:10]:
        print(f"  {gene.name}: logFC={gene.logfc:.2f}, p_adj={gene.p_adj:.2e}")

# 6. Visualization
client.create_visualization(
    dataset_id=preprocessed.id,
    plot_type="umap_marker_expression",
    config={
        "markers": de_results.top_markers(n=20),
        "cluster_labels": True,
        "output_file": "umap_markers.html"
    }
)
```

---

## 🔐 SECURITY & COMPLIANCE

### Authentication & Authorization

```python
# JWT-based authentication
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Validate JWT token and return current user."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Role-based access control
from enum import Enum

class Role(str, Enum):
    USER = "user"
    RESEARCHER = "researcher"
    ADMIN = "admin"

def require_role(required_role: Role):
    """Decorator to enforce role-based access."""
    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role and current_user.role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# Usage
@app.delete("/api/v1/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    user: User = Depends(require_role(Role.RESEARCHER))
):
    """Delete dataset (requires researcher role)."""
    pass
```

### Data Privacy

**HIPAA/GDPR Considerations**:
- ✅ Encrypt data at rest (PostgreSQL encryption, S3 encryption)
- ✅ Encrypt data in transit (TLS/SSL)
- ✅ Audit logging (all data access logged)
- ✅ Data retention policies
- ✅ User consent tracking
- ✅ Right to deletion (GDPR Article 17)
- ✅ Data export (GDPR Article 20)

```python
# Audit logging
import logging
from datetime import datetime

class AuditLogger:
    """Log all sensitive data operations."""
    
    @staticmethod
    async def log_data_access(
        user_id: str,
        dataset_id: str,
        action: str,
        ip_address: str
    ):
        """Log data access event."""
        await db.audit_logs.insert({
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "dataset_id": dataset_id,
            "action": action,
            "ip_address": ip_address
        })
        
        logger.info(
            f"DATA_ACCESS user={user_id} dataset={dataset_id} "
            f"action={action} ip={ip_address}"
        )
```

### Secrets Management

**NEVER commit secrets to Git!**

```bash
# .env file (gitignored)
DATABASE_URL=postgresql://user:pass@localhost/omicselector2
SECRET_KEY=your-secret-key-here
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MLFLOW_TRACKING_URI=http://localhost:5000
REDIS_URL=redis://localhost:6379
```

```python
# config.py - Use pydantic-settings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    aws_access_key_id: str
    aws_secret_access_key: str
    mlflow_tracking_uri: str
    redis_url: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 🧪 TESTING STRATEGY

### Test Coverage Requirements

- **Unit tests**: >80% coverage
- **Integration tests**: All API endpoints
- **End-to-end tests**: Core user workflows

### Test Structure

```python
# tests/unit/test_features/test_lasso.py
import pytest
import numpy as np
from omicselector2.features.classical.lasso import LassoSelector

class TestLassoSelector:
    """Test suite for Lasso feature selector."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 50)  # 100 samples, 50 features
        y = np.random.binomial(1, 0.5, 100)  # Binary outcome
        return X, y
    
    def test_lasso_initialization(self):
        """Test Lasso selector can be initialized."""
        selector = LassoSelector(alpha=0.1)
        assert selector.alpha == 0.1
        
    def test_lasso_feature_selection(self, sample_data):
        """Test Lasso selects features."""
        X, y = sample_data
        selector = LassoSelector(alpha=0.1)
        
        selected_features = selector.fit_transform(X, y)
        
        # Assert features were selected
        assert len(selected_features) > 0
        assert len(selected_features) < X.shape[1]
        
    def test_lasso_coefficient_ranking(self, sample_data):
        """Test Lasso provides coefficient ranking."""
        X, y = sample_data
        selector = LassoSelector(alpha=0.1)
        selector.fit(X, y)
        
        rankings = selector.get_feature_rankings()
        
        # Assert rankings are sorted by absolute coefficient
        assert len(rankings) == X.shape[1]
        assert all(rankings[i] >= rankings[i+1] for i in range(len(rankings)-1))
        
    def test_lasso_with_cv(self, sample_data):
        """Test Lasso with cross-validation for alpha selection."""
        X, y = sample_data
        selector = LassoSelector(cv=5, alpha="auto")
        
        selector.fit(X, y)
        
        # Assert optimal alpha was found
        assert selector.alpha_ is not None
        assert selector.alpha_ > 0
        
    def test_invalid_alpha_raises_error(self):
        """Test invalid alpha parameter raises ValueError."""
        with pytest.raises(ValueError):
            LassoSelector(alpha=-0.1)  # Negative alpha should fail
```

### Integration Tests

```python
# tests/integration/test_api/test_feature_selection_endpoint.py
import pytest
from fastapi.testclient import TestClient
from omicselector2.api.main import app

client = TestClient(app)

class TestFeatureSelectionAPI:
    """Integration tests for feature selection API."""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers."""
        response = client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def uploaded_dataset(self, auth_headers):
        """Upload test dataset."""
        with open("tests/fixtures/test_dataset.csv", "rb") as f:
            response = client.post(
                "/api/v1/data/upload",
                files={"file": ("test_dataset.csv", f, "text/csv")},
                headers=auth_headers
            )
        assert response.status_code == 201
        return response.json()["dataset_id"]
    
    def test_create_feature_selection_job(self, auth_headers, uploaded_dataset):
        """Test creating feature selection job via API."""
        response = client.post(
            "/api/v1/feature-selection/",
            json={
                "dataset_id": uploaded_dataset,
                "methods": ["lasso", "rf_importance"],
                "config": {
                    "cv_folds": 5,
                    "n_features": 50
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 201
        job_data = response.json()
        assert "job_id" in job_data
        assert job_data["status"] == "pending"
        
    def test_get_job_status(self, auth_headers, uploaded_dataset):
        """Test retrieving job status."""
        # Create job
        create_response = client.post(
            "/api/v1/feature-selection/",
            json={
                "dataset_id": uploaded_dataset,
                "methods": ["lasso"],
                "config": {"cv_folds": 5}
            },
            headers=auth_headers
        )
        job_id = create_response.json()["job_id"]
        
        # Get status
        status_response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers=auth_headers
        )
        
        assert status_response.status_code == 200
        assert status_response.json()["job_id"] == job_id
```

---

## 📝 API DOCUMENTATION EXAMPLES

### OpenAPI Schema Generation

FastAPI automatically generates OpenAPI (Swagger) documentation. Access at `/docs`.

```python
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="OmicSelector2 API",
    description="Next-generation platform for multi-omic biomarker discovery",
    version="2.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

class FeatureSelectionRequest(BaseModel):
    """Request model for feature selection job."""
    
    dataset_id: str = Field(
        ...,
        description="Unique identifier of the uploaded dataset",
        example="ds_abc123xyz"
    )
    methods: list[str] = Field(
        ...,
        description="List of feature selection methods to apply",
        example=["lasso", "elastic_net", "rf_importance"]
    )
    cv_folds: int = Field(
        5,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )
    stability: Optional[dict] = Field(
        None,
        description="Stability selection configuration",
        example={
            "enabled": True,
            "n_bootstraps": 100,
            "threshold": 0.7
        }
    )

@app.post(
    "/api/v1/feature-selection/",
    response_model=JobResponse,
    status_code=201,
    summary="Create feature selection job",
    description="""
    Initiates a feature selection job using specified methods.
    
    The job runs asynchronously in the background. Use the returned
    job_id to monitor progress via GET /api/v1/jobs/{job_id}.
    
    **Methods available**:
    - `lasso`: L1 regularization
    - `elastic_net`: L1 + L2 regularization
    - `rf_importance`: Random Forest variable importance
    - `mrmr`: Minimum Redundancy Maximum Relevance
    - `boruta`: Boruta feature selection
    - For full list, see /api/v1/methods/
    """,
    tags=["Feature Selection"]
)
async def create_feature_selection_job(
    request: FeatureSelectionRequest,
    user: User = Depends(get_current_user)
):
    """Create and submit feature selection job."""
    # Implementation
    pass
```

---

## 🚀 DEPLOYMENT

### Docker Configuration

**Dockerfile.api**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction

# Copy application code
COPY src/ ./src/
COPY CLAUDE.md README.md ./

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile.worker** (Celery):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies + ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction

COPY src/ ./src/
COPY CLAUDE.md ./

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Run Celery worker
CMD ["celery", "-A", "src.tasks", "worker", "--loglevel=info", "--concurrency=4"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: omicselector2
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    depends_on:
      - postgres
      - redis
      - minio
    environment:
      DATABASE_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/omicselector2
      REDIS_URL: redis://redis:6379
      MINIO_ENDPOINT: minio:9000
      SECRET_KEY: ${SECRET_KEY}
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src  # Mount for development

  worker-cpu:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    depends_on:
      - postgres
      - redis
      - minio
    environment:
      DATABASE_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/omicselector2
      REDIS_URL: redis://redis:6379
      CELERY_QUEUE: default
    deploy:
      replicas: 2

  worker-gpu:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    depends_on:
      - postgres
      - redis
      - minio
    environment:
      DATABASE_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/omicselector2
      REDIS_URL: redis://redis:6379
      CELERY_QUEUE: gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  mlflow:
    image: python:3.11-slim
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server
                 --backend-store-uri postgresql://admin:${DB_PASSWORD}@postgres:5432/mlflow
                 --default-artifact-root s3://mlflow
                 --host 0.0.0.0
                 --port 5000"
    depends_on:
      - postgres
      - minio
    ports:
      - "5000:5000"
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - api

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - frontend

volumes:
  postgres_data:
  minio_data:
```

### Production Deployment Checklist

- [ ] Set strong SECRET_KEY (256-bit random)
- [ ] Configure SSL/TLS certificates
- [ ] Set up database backups (automated)
- [ ] Configure monitoring (Prometheus + Grafana)
- [ ] Set up logging aggregation (ELK stack or CloudWatch)
- [ ] Configure rate limiting
- [ ] Set up firewall rules
- [ ] Enable CORS appropriately
- [ ] Configure S3 bucket policies
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Perform security audit
- [ ] Load testing (k6 or Locust)
- [ ] Set up alerting (PagerDuty, Slack)
- [ ] Document runbook procedures

---

## 📖 REFERENCES

### Key Publications (OmicSelector 1.0)
1. Stawiski K, Kaszkowiak M, Mikulski D, et al. OmicSelector: automatic feature selection and deep learning modeling for omic experiments. bioRxiv. 2022. doi:10.1101/2022.06.01.494299

### Multi-Omics Integration (2024-2025)
2. Li B, Nabavi S. A multimodal graph neural network framework for cancer molecular subtype classification. BMC Bioinformatics. 2024;25:27
3. Elbashir MK, Mohammed M, Ezugwu AE. Comparative Analysis of Multi-Omics Integration Using Graph Neural Networks for Cancer Classification. IEEE Access. 2025;13:37724-37736
4. Valous NA, Popp F, Zörnig I, et al. Graph machine learning for integrated multi-omics analysis. Br J Cancer. 2024;131(2):205-211
5. Wang Y, Wang Z, Yu X, et al. MORE: a multi-omics data-driven hypergraph integration network for biomedical data classification and biomarker identification. Brief Bioinform. 2025;26(1):bbae658

### Geometric & Topological Methods
6. Zhu J, Oh JH, Simhal AK, et al. Geometric graph neural networks on multi-omics data to predict cancer survival outcomes. Comput Biol Med. 2023;163:107117

### Stability Selection
7. Pusa MA, Rousu J. Stable biomarker discovery in multi-omics data via canonical correlation analysis. PLoS One. 2024

### Single-Cell Methods
8. MultiVI tutorial. scvi-tools documentation. 2024

### Interpretability
9. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS. 2017

### Claude Code Best Practices
10. Anthropic. Claude Code Best Practices. 2025. https://www.anthropic.com/engineering/claude-code-best-practices

---

## 🎓 GLOSSARY

**AnnData**: Annotated data format for storing single-cell data (.h5ad files)

**Benchmarking**: Testing multiple feature selection signatures to determine which performs best

**Celery**: Distributed task queue for Python

**Cox Proportional Hazards**: Statistical model for survival analysis

**Cross-Validation**: Resampling technique to assess model generalization

**Ensemble**: Combining multiple methods to improve robustness

**FastAPI**: Modern Python web framework for building APIs

**Feature Selection**: Process of selecting relevant features from high-dimensional data

**GNN**: Graph Neural Network - deep learning on graph-structured data

**Hold-out Validation**: Data split strategy (train/test/validation) preventing information leakage

**SHAP**: SHapley Additive exPlanations - interpretability method

**Stability Selection**: Ensemble method using subsampling to identify consistently selected features

**VAE**: Variational Autoencoder - generative model for learning latent representations

---

## ⚠️ IMPORTANT NOTES

### For Agentic Coding

1. **Always read CLAUDE.md first** before starting any task
2. **Follow TDD strictly** - write tests before implementation
3. **Use type hints** for all function signatures
4. **Document with Google-style docstrings**
5. **Commit frequently** with clear messages
6. **Ask for clarification** if requirements are unclear
7. **Think before coding** - plan in the Explore phase
8. **Test incrementally** - don't write large untested blocks

### For Human Developers

- This file is the **single source of truth** for architecture decisions
- Update this file when making significant architectural changes
- Keep the development workflow section updated with progress
- Document deviations from the plan with rationale
- Use this file for onboarding new team members

### Critical Reminders

- ❌ **NEVER** commit secrets to Git
- ❌ **NEVER** skip tests
- ❌ **NEVER** modify tests during implementation (TDD rule)
- ❌ **NEVER** deploy untested code
- ✅ **ALWAYS** validate user inputs
- ✅ **ALWAYS** log sensitive operations
- ✅ **ALWAYS** handle errors gracefully
- ✅ **ALWAYS** document API endpoints

---

## 🔄 CHANGELOG

### Version 2.0 (November 5, 2025)
- Initial comprehensive CLAUDE.md created
- Integrated OmicSelector 1.0 analysis
- Added SOTA methods from 2024-2025 research
- Incorporated Anthropic Claude Code best practices
- Defined complete architecture and development roadmap
- Added detailed TDD workflow
- Included security and compliance guidelines

---

**END OF CLAUDE.MD**

This document will evolve as OmicSelector2 develops. Keep it updated, keep it accurate, and use it as your North Star for development decisions.
