# CLAUDE.md: OmicSelector2 Development Plan

## Executive Summary

**Vision**: OmicSelector2 is a next-generation Python platform for biomarker feature selection, signature benchmarking, and model development/deployment in oncology. It modernizes OmicSelector 1.0's proven methodology with improved performance, scalability, interpretability, and user experience while maintaining scientific rigor.

**Core Philosophy**: Systematic multi-method comparison with stability-focused ensemble approaches, hold-out validation to prevent overfitting, and clinical translatability through interpretable models.

**Target Users**: Biomedical researchers, bioinformaticians, oncology data scientists working with bulk RNA-seq, single-cell RNA-seq, WES, and radiomics data for cancer biomarker discovery.

---

## Technology Stack Decisions

### Backend Framework: **FastAPI** ✓

**Rationale:**
- **Performance**: 5-8x faster than Flask (15-20K req/s vs 2-3K req/s)
- **Native async/await**: Critical for handling concurrent biomarker discovery jobs
- **Automatic OpenAPI documentation**: Essential for research collaboration and reproducibility
- **Type safety**: Pydantic validation reduces bugs in complex pipelines
- **Future-proof**: Community momentum shifted to FastAPI (78.9k stars vs Flask 68.4k as of 2024)

**Alternative considered**: Flask (simpler but synchronous, slower, lacks auto-docs)

### ML Frameworks: **PyTorch + scikit-learn** ✓

**Primary - PyTorch**:
- Dynamic computational graphs ideal for variable-length biological sequences
- **Selene library**: Purpose-built for biological sequence data (Nature Methods published)
- Research-friendly debugging for novel biomarker discovery methods
- Growing genomics adoption trend

**Essential - scikit-learn**:
- Classical ML methods (Random Forest, SVM, Logistic Regression)
- Feature selection algorithms
- Preprocessing pipelines
- Cross-validation framework
- Standard for baseline models

**Secondary - TensorFlow** (for specific deployment scenarios requiring TF Serving)

### Database Architecture: **PostgreSQL + File Storage** ✓

**Primary: PostgreSQL 15+**
- ACID compliance for experiment tracking and model versioning
- JSONB support provides NoSQL flexibility without sacrificing relational benefits
- Array types for storing gene expression profiles
- Proven in genomics (TCGA, GEO metadata)
- Full-text search for experimental annotations

**File Storage: S3/MinIO compatible**
- AnnData objects (.h5ad files) for gene expression matrices
- Trained models (.pt files)
- Large visualization outputs
- Versioned datasets

**Cache/Queue: Redis**
- Celery backend for job queue
- Real-time pipeline status
- Frequently accessed result caching

### Frontend Stack: **React + Material-UI** ✓

**Framework: React 18+**
- Largest ecosystem for data visualization integration
- Best D3.js/Plotly compatibility
- Mature component libraries
- Large talent pool

**UI Library: Material-UI (MUI) v5+**
- Comprehensive component set
- MUI X Data Grid for large result tables
- Smaller bundle than Ant Design (314kB vs 465kB)
- Excellent documentation
- Better for Western audiences

**Alternative considered**: Svelte (best performance but smaller ecosystem)

### Visualization Stack: **CanvasXpress + Plotly** ✓

**Primary: CanvasXpress**
- **Purpose-built for bioinformatics** with 40+ specialized graph types
- Volcano plots, heatmaps, Kaplan-Meier curves, dendrograms built-in
- HTML5 Canvas rendering for superior performance (thousands of data points)
- R and Python integration via packages
- Audit trail for reproducibility

**Secondary: Plotly.js**
- 3D visualizations (UMAP, t-SNE plots)
- Interactive dashboards
- Quick implementation

**Utility: D3.js modules** (d3-scale, d3-array for data transformations, custom visualizations)

### Job Queue: **Celery with Redis** ✓

**Rationale:**
- Most feature-rich (retries, scheduling, chaining, priorities)
- Production-grade for complex bioinformatics workflows
- Flower monitoring dashboard
- Proven at scale (Instagram, AstraZeneca genomics)
- Multiple broker support if needed (can switch to RabbitMQ)

**Alternative considered**: RQ (simpler but 2x slower, fewer features for complex workflows)

### Bioinformatics Python Libraries

**Essential Core**:
- **Scanpy** - Single-cell RNA-seq analysis (scverse ecosystem)
- **PyDESeq2** - Bulk RNA-seq differential expression (Python DESeq2 implementation)
- **Biopython** - General bioinformatics operations
- **Pysam** - SAM/BAM/VCF handling
- **lifelines** - Survival analysis
- **statsmodels/pingouin** - Statistical testing

**Workflow \u0026 Infrastructure**:
- **MLflow** - Experiment tracking, model registry
- **DVC** - Data version control
- **Snakemake** - Workflow automation
- **Docker** - Containerization with CREDO framework principles

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Upload    │  │   Workflow   │  │  Visualization   │   │
│  │  Interface  │  │   Builder    │  │    Dashboard     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API (OpenAPI)
┌────────────────────────▼────────────────────────────────────┐
│                 API Layer (FastAPI)                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │  Auth/Users    │  │  Data Upload   │  │    Results    │ │
│  │  /api/v1/auth  │  │ /api/v1/data   │  │ /api/v1/jobs  │ │
│  └────────────────┘  └────────────────┘  └───────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Job Queue (Celery + Redis)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ high_priority│  │   default    │  │  low_priority    │  │
│  │    queue     │  │    queue     │  │     queue        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└────────────┬────────────────┬──────────────────────────────┘
             │                │
    ┌────────▼──────┐  ┌─────▼────────────┐
    │  CPU Workers  │  │   GPU Workers    │
    │  (Feature     │  │  (Deep Learning) │
    │  Selection)   │  │                  │
    └───────┬───────┘  └────────┬─────────┘
            │                   │
┌───────────▼───────────────────▼─────────────────────────────┐
│                   Storage Layer                              │
│  ┌─────────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │   PostgreSQL    │  │  File Storage │  │    Redis     │  │
│  │   (Metadata)    │  │  (h5ad, .pt)  │  │   (Cache)    │  │
│  └─────────────────┘  └───────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Core Modules and Responsibilities

**1. Data Module** (`src/data/`)
- **loaders.py**: Ingest FASTQ, BAM, VCF, h5ad, CSV formats
- **validators.py**: Format validation, quality control checks
- **preprocessors.py**: Normalization, batch correction, filtering
- **schema.py**: Data models using Pydantic

**2. Features Module** (`src/features/`)
- **selectors.py**: Feature selection method implementations
- **registry.py**: Plugin registry for selection methods
- **stability.py**: Stability selection framework
- **ensemble.py**: Ensemble voting and aggregation

**3. Models Module** (`src/models/`)
- **base.py**: Abstract base classes for models
- **classical.py**: Random Forest, SVM, Logistic Regression
- **neural.py**: PyTorch neural networks
- **survival.py**: Cox PH, survival models

**4. Training Module** (`src/training/`)
- **trainer.py**: Training orchestration
- **evaluator.py**: Cross-validation, metrics calculation
- **hyperparameter.py**: Hyperparameter optimization
- **experiment.py**: MLflow integration

**5. Inference Module** (`src/inference/`)
- **predictor.py**: Model serving logic
- **explainer.py**: SHAP integration for interpretability
- **validator.py**: Input validation for predictions

**6. API Module** (`src/api/`)
- **routes/**: REST endpoints by resource
- **dependencies.py**: FastAPI dependencies
- **middleware.py**: Authentication, logging
- **websockets.py**: Real-time progress updates

**7. Utils Module** (`src/utils/`)
- **config.py**: Configuration management
- **logging.py**: Structured logging
- **security.py**: Encryption, authentication helpers

---

## Feature Selection Methods to Implement

### PRIORITY 1 (v1.0 - MUST INCLUDE) ✓

Based on 2022-2025 literature review with strong evidence:

**Classical Statistical (Still Relevant)**:
1. **Lasso** - L1 regularization, widely used, FDA-cleared in diagnostic panels
2. **Elastic Net** - L1+L2, handles correlated features better
3. **t-test/ANOVA** - Pre-filtering, reduce dimensionality by 90%
4. **Cox Proportional Hazards** - Standard for survival analysis

**Machine Learning**:
5. **Random Forest Variable Importance** - ⭐ TOP RECOMMENDATION (best overall performance, fast, scalable)
6. **XGBoost Feature Importance** - Modern gradient boosting, excellent results
7. **mRMR** - Minimum Redundancy Maximum Relevance, #1-2 in benchmarks (note: slow for concurrent selection)
8. **Boruta** - All-relevant feature selection, proven in cancer classification

**Ensemble Methods** (CRITICAL):
9. **Stability Selection** - ⭐ ESSENTIAL for reproducibility, subsampling-based consensus
10. **Hybrid Ensemble** - Data + function perturbation, best stability+accuracy (Hyb-Wx-GR-SU pattern)
11. **Ensemble Voting** - Combine multiple methods (RF-VI + mRMR + Lasso)

**Single-Cell Specific**:
12. **HDG/HEG** - High-Deviation/High-Expression, NEW gold standard (replaces HVG)
13. **DUBStepR** - Scalable to 1M+ cells, correlation-based
14. **FEAST** - Consensus clustering, best on 11/12 benchmarks
15. **Deviance-based Selection** - Works on raw counts, theoretically sound

**Radiomics Specific**:
16. **Correlation-based Filtering** - Remove redundant features (r\u003e0.9)
17. **Linear Combinations Filter** - QR decomposition for collinearity

**Survival Analysis**:
18. **Group Lasso for Survival** - Pseudo-variables assisted, multi-omics survival

### PRIORITY 2 (v2.0 - FUTURE) ⚠

Advanced methods requiring more development/validation:

1. **scFSNN** - Deep learning for single-cell (7/8 datasets winner)
2. **Variational Autoencoders** - Non-linear feature discovery
3. **SVM-RFE** - Effective but slow (\u003e1 day for multi-omics)
4. **Bayesian Feature Selection** - For radiomics with reliability metrics
5. **DELVE** - Dynamic selection for trajectory analysis
6. **Attention Mechanisms** - For sequence models

### PRIORITY 3 (v3.0 - RESEARCH DIRECTION) 🔮

Emerging methods to monitor:

1. Transformer-based feature selection
2. Multi-modal deep learning integration
3. Causal inference-based selection
4. Graph neural networks for pathway integration
5. Federated learning for multi-site discovery

### OBSOLETE METHODS (EXCLUDE from v1.0) ✗

Based on evidence of poor performance or being superseded:

1. **Genetic Algorithms** - Too slow (2+ days), poor performance
2. **Simple variance-based filtering** (as primary) - Superseded by HDG/HEG for single-cell
3. **ReliefF** (as primary method) - Poor with small feature sets (AUC 0.68)
4. **Forward/Backward Stepwise Selection** (univariate) - Better alternatives exist
5. **Single decision trees** - Use ensembles instead

---

## API Design Principles

### RESTful Endpoints (v1)

```python
# Data Management
POST   /api/v1/data/upload          # Upload dataset
GET    /api/v1/data/{id}            # Get dataset info
DELETE /api/v1/data/{id}            # Delete dataset
POST   /api/v1/data/{id}/validate   # Validate dataset

# Feature Selection Jobs
POST   /api/v1/jobs                 # Create new analysis job
GET    /api/v1/jobs                 # List user's jobs
GET    /api/v1/jobs/{id}            # Get job details
GET    /api/v1/jobs/{id}/status     # Get job status
DELETE /api/v1/jobs/{id}            # Cancel/delete job

# Results
GET    /api/v1/jobs/{id}/results    # Get analysis results
GET    /api/v1/jobs/{id}/features   # Get selected features
GET    /api/v1/jobs/{id}/metrics    # Get performance metrics
GET    /api/v1/jobs/{id}/shap       # Get SHAP values

# Models
GET    /api/v1/models               # List trained models
POST   /api/v1/models/{id}/predict  # Make predictions
GET    /api/v1/models/{id}/explain  # Get model explanations

# Feature Selection Methods
GET    /api/v1/methods              # List available methods
GET    /api/v1/methods/{name}       # Get method details

# Users & Auth
POST   /api/v1/auth/login           # Login
POST   /api/v1/auth/logout          # Logout
GET    /api/v1/users/me             # Get current user
```

### WebSocket Endpoints

```python
WS     /ws/jobs/{id}                # Real-time job progress
```

### Request/Response Models

```python
# Job Creation Request
{
  "dataset_id": "uuid",
  "methods": ["rf_importance", "mrmr", "lasso"],
  "config": {
    "cv_folds": 5,
    "test_size": 0.2,
    "validation_size": 0.2,
    "ensemble": {
      "enabled": true,
      "voting": "soft"
    }
  },
  "priority": "normal"  # high, normal, low
}

# Job Status Response
{
  "job_id": "uuid",
  "status": "running",  # queued, running, completed, failed
  "progress": 0.67,
  "current_step": "Cross-validation fold 7/10",
  "created_at": "2025-11-05T10:00:00Z",
  "updated_at": "2025-11-05T10:15:30Z",
  "estimated_completion": "2025-11-05T10:20:00Z"
}

# Results Response
{
  "job_id": "uuid",
  "selected_features": [
    {"name": "TP53", "rank": 1, "importance": 0.95},
    {"name": "EGFR", "rank": 2, "importance": 0.87}
  ],
  "performance": {
    "auc": 0.92,
    "accuracy": 0.88,
    "sensitivity": 0.85,
    "specificity": 0.91
  },
  "stability": {
    "method": "nogueira",
    "score": 0.83
  },
  "methods_comparison": {...}
}
```

### Versioning Strategy

- **URL versioning**: `/api/v1/`, `/api/v2/` for breaking changes
- **Model versioning**: Separate from API version, stored in model metadata
- **Backward compatibility**: Maintain v1 for at least 1 year after v2 release

---

## Database Schema Concepts

### PostgreSQL Tables

**users**
- id (UUID, PK)
- email (VARCHAR, UNIQUE)
- hashed_password (VARCHAR)
- created_at (TIMESTAMP)
- last_login (TIMESTAMP)
- role (ENUM: admin, researcher, viewer)

**datasets**
- id (UUID, PK)
- user_id (UUID, FK → users)
- name (VARCHAR)
- data_type (ENUM: bulk_rna_seq, single_cell, wes, radiomics)
- file_path (VARCHAR) - Points to S3/MinIO location
- n_samples (INT)
- n_features (INT)
- metadata (JSONB) - Flexible metadata storage
- uploaded_at (TIMESTAMP)
- validated (BOOLEAN)
- validation_report (JSONB)

**jobs**
- id (UUID, PK)
- user_id (UUID, FK → users)
- dataset_id (UUID, FK → datasets)
- job_type (ENUM: feature_selection, model_training, prediction)
- status (ENUM: queued, running, completed, failed, cancelled)
- config (JSONB) - Job configuration
- priority (ENUM: high, normal, low)
- created_at (TIMESTAMP)
- started_at (TIMESTAMP)
- completed_at (TIMESTAMP)
- celery_task_id (VARCHAR) - Link to Celery task

**feature_selection_results**
- id (UUID, PK)
- job_id (UUID, FK → jobs)
- method (VARCHAR) - e.g., "rf_importance"
- selected_features (JSONB) - Array of feature objects
- performance_metrics (JSONB)
- stability_metrics (JSONB)
- hyperparameters (JSONB)
- created_at (TIMESTAMP)

**models**
- id (UUID, PK)
- job_id (UUID, FK → jobs)
- name (VARCHAR)
- model_type (VARCHAR) - e.g., "random_forest"
- version (VARCHAR)
- file_path (VARCHAR) - Points to .pt/.pkl file
- performance (JSONB)
- hyperparameters (JSONB)
- feature_list (JSONB)
- trained_at (TIMESTAMP)

**shap_values**
- id (UUID, PK)
- model_id (UUID, FK → models)
- sample_id (VARCHAR)
- shap_values (JSONB) - Array of feature-value pairs
- base_value (FLOAT)
- prediction (FLOAT)

**audit_log**
- id (UUID, PK)
- user_id (UUID, FK → users)
- action (VARCHAR) - e.g., "data_access", "model_predict"
- resource_type (VARCHAR)
- resource_id (UUID)
- ip_address (INET)
- timestamp (TIMESTAMP)
- details (JSONB)

### Indexes

```sql
CREATE INDEX idx_jobs_user_id ON jobs(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_datasets_user_id ON datasets(user_id);
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp);
```

---

## UI/UX Workflow

### Three-Panel Layout (Galaxy-Inspired)

```
┌─────────────────────────────────────────────────────────────┐
│                      Top Navigation                          │
│  [Logo] [Home] [Datasets] [Analysis] [Models] [Docs] [User] │
└─────────────────────────────────────────────────────────────┘
┌────────────┬─────────────────────────────┬──────────────────┐
│            │                             │                  │
│  Tool      │     Main Workspace          │   History/       │
│  Panel     │                             │   Progress       │
│            │  ┌─────────────────────┐    │                  │
│ □ Upload   │  │                     │    │  Recent Jobs:    │
│ □ Preproc  │  │   Visualization     │    │  ✓ Analysis #1   │
│ □ Feature  │  │   or                │    │  ⟳ Analysis #2   │
│   Select   │  │   Configuration     │    │     67% [████▌ ] │
│ □ Train    │  │   Interface         │    │  ○ Analysis #3   │
│ □ Evaluate │  │                     │    │                  │
│ □ Deploy   │  └─────────────────────┘    │  [View All]      │
│            │                             │                  │
│ [Docs]     │  [Run Analysis] [Save]      │                  │
└────────────┴─────────────────────────────┴──────────────────┘
```

### Workflow Stages

**Stage 1: Data Upload \u0026 QC**
- Drag-and-drop upload interface
- Format auto-detection (FASTQ, BAM, VCF, h5ad, CSV)
- Real-time validation with visual feedback
- QC dashboard:
  - Missing values heatmap
  - Distribution plots
  - Quality metrics (✓ Pass, ⚠ Warning, ✗ Fail)
  - Outlier detection

**Stage 2: Preprocessing**
- Step-by-step wizard:
  1. Normalization (TPM, log transform, Z-score)
  2. Batch correction (optional)
  3. Feature filtering (variance, expression threshold)
  4. Class imbalance handling (SMOTE, ROSE)
- Preview before/after for each step
- Parameter tooltips with recommendations
- "Skip" option for advanced users

**Stage 3: Feature Selection**
- Method selection interface:
  - Quick presets: "Fast \u0026 Robust", "Comprehensive", "Deep Learning"
  - Advanced: Individual method selection with parameters
  - Ensemble configuration
- Data split configuration (train/test/validation ratios)
- Cross-validation settings
- Priority selection (high = faster queue)
- **Start Analysis** button → job submitted to Celery

**Stage 4: Progress Monitoring**
- Real-time progress bar via WebSocket
- Step-by-step status:
  ```
  Analyzing Dataset: cancer_rnaseq.csv
  ━━━━━━━━━━━━━━━━━━━━━━━━ 67%
  
  ✓ Data validation complete
  ✓ Preprocessing (normalization)
  ✓ Train/test split
  ✓ Random Forest feature selection
  ⟳ mRMR feature selection (processing...)
  ○ LASSO feature selection
  ○ Cross-validation
  ○ Stability analysis
  ○ Ensemble aggregation
  
  Estimated time: ~5 minutes
  [Run in Background] [Cancel]
  ```
- Option to close browser (email notification on completion)
- Log window for detailed progress

**Stage 5: Results Dashboard**
- **Summary Panel**:
  - Top selected features (ranked table)
  - Overall performance metrics
  - Method comparison chart
  
- **Visualization Gallery** (tabs):
  - Volcano plots (differential expression)
  - Heatmaps (selected features × samples)
  - Feature importance bar charts
  - Correlation matrices
  - SHAP summary plots
  - Stability plots across CV folds
  
- **Interactive Exploration**:
  - Hover tooltips for gene information
  - Click feature → drill down to details
  - Linked plots (selection in one updates others)
  - Download buttons (CSV, PNG, SVG)

**Stage 6: Model Training \u0026 Validation**
- Select algorithm (RF, SVM, XGBoost, Neural Net)
- Hyperparameter tuning (auto or manual)
- Training progress tracking
- Performance visualization (ROC curves, confusion matrix)
- SHAP explanations automatically generated

**Stage 7: Deployment**
- Export trained model
- Generate prediction API endpoint
- Create interactive Shiny/Dash app
- Download reproducibility report (methods, parameters, results)

### Progress Indication Patterns

**For Known Duration (\u003c5 min):**
- Determinate progress bar (0-100%)
- Time estimate ("~2 minutes remaining")
- Step counter ("Step 3 of 5")

**For Unknown Duration (\u003e5 min):**
- Indeterminate spinner with text updates
- Step descriptions ("Running cross-validation fold 7/10")
- Background job option

**For Very Long Tasks (\u003e20 min):**
- "Fire and forget" pattern
- Email/notification on completion
- Job list with status indicators
- Resume from history

---

## Implementation Phases

### PHASE 1: Foundation (Weeks 1-4)

**Sprint 0: Project Setup**
- Repository initialization with Git
- Project structure creation (see module organization)
- CI/CD pipeline (GitHub Actions)
  - Linting (Black, isort, flake8)
  - Type checking (mypy)
  - Unit tests (pytest)
- Development environment (Docker Compose)
- CLAUDE.md creation and iteration

**Milestones:**
- M1.1: Repository structure complete
- M1.2: CI/CD pipeline operational
- M1.3: Docker development environment working

**Deliverables:**
- Project repository with proper structure
- Docker Compose configuration for local development
- CI/CD pipelines for code quality
- Initial CLAUDE.md
- Development setup documentation

### PHASE 2: Data Engineering (Weeks 5-8)

**Sprint 1: Data Ingestion (Weeks 5-6)**
- File upload API endpoint (FastAPI)
- Format detection and validation
- Support for: CSV, h5ad (AnnData), VCF, BAM
- Pysam, Scanpy, pandas integration
- Streaming for large files

**Sprint 2: Data Preprocessing (Weeks 7-8)**
- Normalization pipelines (TPM, log, Z-score)
- Quality control checks
- Batch correction (optional)
- Feature filtering
- Data validation reports

**Milestones:**
- M2.1: File upload working for all formats
- M2.2: QC dashboard functional

**Deliverables:**
- Data ingestion modules
- Validation pipeline with tests
- QC visualization dashboard
- Preprocessing functions
- Documentation for data models

### PHASE 3: Feature Selection Core (Weeks 9-14)

**Sprint 3: Classical Methods (Weeks 9-10)**
- Implement: Lasso, Elastic Net, t-test/ANOVA
- scikit-learn integration
- Cross-validation framework
- Performance metrics calculation

**Sprint 4: ML Methods (Weeks 11-12)**
- Implement: RF Variable Importance, XGBoost FI, mRMR
- Method registry pattern (plugin architecture)
- Hyperparameter optimization

**Sprint 5: Ensemble \u0026 Stability (Weeks 13-14)**
- Stability Selection implementation
- Ensemble voting mechanism
- Method comparison framework
- Stability metrics (Nogueira, Kuncheva)

**Milestones:**
- M3.1: 5 core methods operational
- M3.2: Ensemble framework complete
- M3.3: Method registry extensible

**Deliverables:**
- Feature selection module with 10+ methods
- Plugin registry for easy extension
- Ensemble and stability wrappers
- Performance benchmark suite
- Method comparison visualizations

### PHASE 4: Job Queue \u0026 Backend (Weeks 15-18)

**Sprint 6: Celery Integration (Weeks 15-16)**
- Celery setup with Redis broker
- Task definitions for each feature selection method
- Priority queues (high, normal, low)
- Worker pool configuration (CPU vs GPU)
- Progress tracking with WebSocket updates

**Sprint 7: API Development (Weeks 17-18)**
- FastAPI endpoints for jobs, datasets, results
- Authentication and authorization (JWT)
- OpenAPI documentation
- Rate limiting and throttling
- Error handling and logging

**Milestones:**
- M4.1: Celery job queue operational
- M4.2: API endpoints complete with auth

**Deliverables:**
- Celery task definitions
- FastAPI application with full REST API
- Authentication system
- Flower monitoring dashboard
- API documentation (Swagger UI)

### PHASE 5: Frontend Development (Weeks 19-22)

**Sprint 8: Core UI (Weeks 19-20)**
- React application setup (Vite)
- Material-UI component library integration
- Three-panel layout (Galaxy-style)
- Data upload interface
- Navigation and routing

**Sprint 9: Workflow Interface (Weeks 21-22)**
- Multi-step wizard for feature selection
- Parameter configuration forms
- Progress tracking UI with WebSocket
- History panel for job tracking

**Milestones:**
- M5.1: Basic UI navigation working
- M5.2: Workflow interface complete

**Deliverables:**
- React application
- Upload and configuration interfaces
- Real-time progress tracking
- Job history view

### PHASE 6: Visualization \u0026 Results (Weeks 23-26)

**Sprint 10: CanvasXpress Integration (Weeks 23-24)**
- CanvasXpress React wrapper
- Volcano plots
- Heatmaps with hierarchical clustering
- Feature importance charts

**Sprint 11: Results Dashboard (Weeks 25-26)**
- Results summary panel
- Interactive visualization gallery
- SHAP integration for model interpretability
- Export functionality (CSV, PNG, PDF)

**Milestones:**
- M6.1: Core visualizations operational
- M6.2: Results dashboard complete

**Deliverables:**
- Visualization library integration
- Results dashboard with 6+ plot types
- SHAP explanations
- Export and download functionality

### PHASE 7: Model Training \u0026 Interpretability (Weeks 27-30)

**Sprint 12: Model Training (Weeks 27-28)**
- Model training pipeline (scikit-learn, PyTorch)
- Hyperparameter optimization (Optuna or Ray Tune)
- MLflow integration for experiment tracking
- Model registry

**Sprint 13: Interpretability (Weeks 29-30)**
- SHAP integration (TreeExplainer, KernelExplainer)
- Feature importance visualizations
- Waterfall and beeswarm plots
- Model comparison framework

**Milestones:**
- M7.1: Model training operational
- M7.2: SHAP explanations integrated

**Deliverables:**
- Model training module
- Experiment tracking with MLflow
- SHAP interpretability framework
- Model comparison dashboard

### PHASE 8: Testing \u0026 Validation (Weeks 31-34)

**Sprint 14: Comprehensive Testing (Weeks 31-32)**
- Unit test suite (80%+ coverage)
- Integration tests for pipelines
- End-to-end tests for critical paths
- Performance testing

**Sprint 15: Validation \u0026 Optimization (Weeks 33-34)**
- Cross-validation with real datasets
- Performance optimization
- Security audit
- Bug fixes and refinements

**Milestones:**
- M8.1: Test coverage \u003e80%
- M8.2: Performance benchmarks met

**Deliverables:**
- Comprehensive test suite
- Test coverage reports
- Performance benchmark results
- Security audit report
- Bug fixes and optimizations

### PHASE 9: Documentation \u0026 Deployment (Weeks 35-38)

**Sprint 16: Documentation (Weeks 35-36)**
- User guide with tutorials
- API documentation finalization
- Developer onboarding guide
- Video tutorials

**Sprint 17: Deployment (Weeks 37-38)**
- Production Docker images
- Kubernetes manifests (optional)
- Deployment scripts
- Monitoring setup (Prometheus + Grafana)
- Production launch

**Milestones:**
- M9.1: Documentation complete
- M9.2: Production deployment successful

**Deliverables:**
- Complete documentation suite
- Production-ready deployment
- Monitoring dashboards
- Release announcement

---

## Testing Strategy

### Testing Pyramid for ML Pipelines

**Level 0: Smoke Tests (Continuous - \u003c5 minutes)**
```python
def test_pipeline_smoke():
    """Ensure pipeline runs without crashes on sample data"""
    X, y = load_sample_data(n_samples=50)
    selector = RandomForestSelector()
    selector.fit(X, y)
    assert selector.selected_features_ is not None
```

**Level 1: Unit Tests (Continuous)**
```python
def test_lasso_feature_selection():
    """Test LASSO feature selection"""
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, 100)
    selector = LassoSelector(alpha=0.01)
    selector.fit(X, y)
    selected = selector.get_selected_features()
    assert len(selected) \u003e 0
    assert len(selected) \u003c= 20

def test_data_validation():
    """Test data validation catches errors"""
    with pytest.raises(ValidationError):
        validate_csv_format("invalid_file.txt")
```

**Level 2: Integration Tests (Continuous)**
```python
def test_feature_selection_pipeline():
    """Test complete feature selection pipeline"""
    # Load data
    dataset = load_test_dataset("cancer_rnaseq.csv")
    
    # Preprocess
    preprocessed = preprocess_data(dataset)
    assert preprocessed.isna().sum().sum() == 0
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed)
    
    # Feature selection
    selector = RandomForestSelector()
    selector.fit(X_train, y_train)
    X_selected = selector.transform(X_test)
    
    # Validate output
    assert X_selected.shape[1] \u003c X_test.shape[1]
```

**Level 3: Data Quality Tests (Periodic)**
```python
def test_no_data_drift():
    """Check for distribution changes"""
    reference_data = load_reference_distribution()
    current_data = load_current_data()
    
    ks_statistic, p_value = ks_2samp(reference_data, current_data)
    assert p_value \u003e 0.01, "Data drift detected"
```

**Level 4: Model Quality Tests (Pre-deployment)**
```python
def test_model_performance_threshold():
    """Ensure model meets minimum quality"""
    model = load_trained_model("rf_v1")
    X_test, y_test = load_test_set()
    
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    
    assert auc \u003e= 0.80, f"Model AUC {auc} below threshold"
```

**Level 5: End-to-End Tests**
```python
@pytest.mark.e2e
def test_full_workflow():
    """Test entire workflow from upload to results"""
    # Upload dataset
    response = client.post("/api/v1/data/upload", files=test_file)
    dataset_id = response.json()["id"]
    
    # Create job
    job_response = client.post("/api/v1/jobs", json={
        "dataset_id": dataset_id,
        "methods": ["rf_importance", "lasso"],
        "config": {...}
    })
    job_id = job_response.json()["id"]
    
    # Wait for completion
    wait_for_job(job_id, timeout=300)
    
    # Check results
    results = client.get(f"/api/v1/jobs/{job_id}/results").json()
    assert len(results["selected_features"]) \u003e 0
    assert results["performance"]["auc"] \u003e 0.5
```

### CI/CD Pipeline Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install black isort flake8 mypy
      - run: black --check src/
      - run: isort --check src/
      - run: flake8 src/
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/smoke -v
      - run: pytest tests/unit -v --cov=src
      - run: pytest tests/integration -v
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Coverage Targets

- **Overall code coverage**: \u003e80%
- **Critical modules** (data validation, feature selection): \u003e90%
- **API endpoints**: 100% (all endpoints tested)
- **Integration tests**: All pipeline stages
- **E2E tests**: 3-5 critical user workflows

---

## Documentation Requirements

### 1. README.md

```markdown
# OmicSelector2

Biomarker feature selection and model development platform for oncology.

## Quick Start

```bash
# Clone repository
git clone https://github.com/org/omicselector2.git
cd omicselector2

# Start with Docker Compose
docker-compose up -d

# Access at http://localhost:8000
```

## Features
- 18+ feature selection methods
- Ensemble and stability selection
- Real-time progress tracking
- Interactive visualizations
- Model interpretability with SHAP

## Citation
[Publication details]

## License
MIT License
```

### 2. API Documentation (Auto-generated)

- FastAPI automatic OpenAPI documentation at `/docs`
- Redoc alternative at `/redoc`
- Include request/response examples
- Authentication instructions

### 3. User Guide (docs/user-guide/)

**Contents:**
- Installation and setup
- Data preparation guidelines
- Workflow tutorials (step-by-step with screenshots)
- Feature selection method descriptions
- Interpretation of results
- Troubleshooting guide
- FAQ

**Format:** MkDocs with Material theme

### 4. Developer Guide (docs/developer-guide/)

**Contents:**
- Development environment setup
- Project structure explanation
- Code style guidelines
- Contributing guidelines
- Testing guidelines
- Adding new feature selection methods (plugin guide)
- Release process

### 5. Architecture Documentation (docs/architecture/)

**Contents:**
- System design overview
- Component interaction diagrams (Mermaid)
- Data flow diagrams
- Database schema
- API design decisions
- Technology stack rationale

### 6. Method Documentation (docs/methods/)

**For each feature selection method:**
- Mathematical description
- When to use (data types, sample sizes)
- Hyperparameters and tuning
- Computational complexity
- References to original papers
- Implementation notes

### 7. Reproducibility Documentation

**Essential for scientific software:**
- Exact dependency versions (requirements.lock.txt)
- Docker images with all dependencies
- Random seed documentation
- Test datasets with expected outputs
- Benchmarking scripts
- Performance comparisons with OmicSelector 1.0

---

## Comparison with OmicSelector 1.0

### Preserved Strengths ✓

1. **Methodological Rigor**
   - Hold-out validation strategy (train/test/validation split)
   - Systematic multi-method comparison
   - Overfitting-resistant design
   - Clinical translatability focus

2. **Comprehensive Feature Selection**
   - Preserved all effective methods (60+ from v1.0, excluding obsolete)
   - Added modern methods (HDG/HEG, scFSNN)
   - Enhanced ensemble approaches
   - SMOTE integration for imbalanced datasets

3. **Reproducibility**
   - Docker containerization maintained
   - Enhanced with MLflow experiment tracking
   - Git-based workflow
   - DVC for data versioning

### Improvements in v2.0 ✅

**1. Performance \u0026 Scalability**
- ❌ v1.0: Days of computation for 60+ methods × 10+ models
- ✅ v2.0: Parallel execution with Celery, resume capability, progress tracking

**2. User Experience**
- ❌ v1.0: Limited GUI, no progress tracking, results in many files
- ✅ v2.0: Modern React UI, real-time progress, centralized dashboard

**3. Installation \u0026 Setup**
- ❌ v1.0: 100+ R packages, TensorFlow/Keras conflicts, platform issues
- ✅ v2.0: Docker Compose one-command start, Python-native

**4. Code Organization**
- ❌ v1.0: Monolithic functions, file-based state, global parameters
- ✅ v2.0: Modular architecture, database-backed, proper state management

**5. Interpretability**
- ❌ v1.0: No integrated explainability tools
- ✅ v2.0: SHAP integrated, feature importance visualizations, clinical explanations

**6. Scalability**
- ❌ v1.0: Binary classification only, \u003c2500 features, single-threaded bottlenecks
- ✅ v2.0: Multi-class support (roadmap), distributed computing, optimized for large datasets

**7. Result Interpretation**
- ❌ v1.0: 600+ model results, no automated ranking, overwhelming
- ✅ v2.0: Clear rankings, ensemble consensus, statistical comparisons, guided recommendations

**8. Modern ML Methods**
- ❌ v1.0: Limited deep learning (feedforward only), no attention, no ensemble methods
- ✅ v2.0: PyTorch flexibility, SHAP integration, ensemble methods, state-of-art algorithms

**9. API \u0026 Integration**
- ❌ v1.0: R package or GUI only
- ✅ v2.0: RESTful API, programmatic access, workflow integration (Nextflow, Snakemake)

**10. Documentation**
- ❌ v1.0: Assumes R knowledge, limited troubleshooting, outdated videos
- ✅ v2.0: Comprehensive docs, interactive tutorials, API documentation, video series

### Technology Migration

| Component | v1.0 | v2.0 | Reason |
|-----------|------|------|--------|
| Language | R | Python | Better ML ecosystem, easier deployment |
| Backend | Shiny | FastAPI | Performance, async, auto-docs |
| Frontend | Shiny | React + MUI | Modern UX, better visualization |
| Database | File-based | PostgreSQL | ACID compliance, better querying |
| Job Queue | None | Celery | Proper job management, resumable |
| Viz | ggplot2, plotly | CanvasXpress, Plotly | Bioinformatics-specialized |
| DL Framework | Keras/TF | PyTorch | Research flexibility, community momentum |
| Monitoring | None | MLflow + Flower | Experiment tracking, job monitoring |

---

## Development Best Practices

### Code Style (follows Anthropic guidelines)

**Python Style:**
- Use Black formatter (line length 88)
- Import organization with isort
- Type hints throughout
- Docstrings for all public functions (Google style)

```python
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def select_features(
    X: np.ndarray, 
    y: np.ndarray, 
    n_features: int = 10
) -\u003e Tuple[np.ndarray, List[int]]:
    """Select top features using Random Forest importance.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        n_features: Number of features to select
        
    Returns:
        Tuple of (transformed feature matrix, selected feature indices)
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-n_features:]
    return X[:, indices], indices.tolist()
```

**JavaScript/React Style:**
- ES6+ syntax (arrow functions, destructuring)
- Functional components with hooks
- PropTypes or TypeScript for type safety
- Prettier formatter

### Git Workflow

**Branching:**
- `main` - Production-ready code
- `develop` - Integration branch
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `release/*` - Release preparation

**Commit Messages:**
```
type(scope): Short description

Longer explanation if needed

Fixes #123
```

Types: feat, fix, docs, style, refactor, test, chore

**Pull Request Process:**
1. Create feature branch from `develop`
2. Implement feature with tests
3. Ensure CI passes (linting, tests)
4. Request code review
5. Address review comments
6. Merge to `develop`

### Configuration Management

**Environment Variables (.env):**
```
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=omicselector2
POSTGRES_USER=user
POSTGRES_PASSWORD=changeme

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=omicselector2-data

# Security
SECRET_KEY=your-secret-key-change-this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

**Configuration File (config/settings.yaml):**
```yaml
feature_selection:
  default_methods:
    - rf_importance
    - mrmr
    - lasso
  
  cv_folds: 5
  test_size: 0.2
  validation_size: 0.2
  random_state: 42
  
  ensemble:
    enabled: true
    voting: soft
    weights: null  # Equal weights

preprocessing:
  normalization: tpm
  log_transform: true
  variance_threshold: 0.01
  handle_missing: impute  # impute, drop, fail

models:
  random_forest:
    n_estimators: 100
    max_depth: null
    min_samples_split: 2
  
  xgboost:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6
```

### Logging

```python
import logging
import structlog

# Structured logging
logger = structlog.get_logger(__name__)

def process_dataset(dataset_id: str):
    logger.info(
        "Processing dataset",
        dataset_id=dataset_id,
        n_samples=len(dataset),
        n_features=dataset.shape[1]
    )
    
    try:
        result = run_analysis(dataset)
        logger.info("Analysis complete", dataset_id=dataset_id, auc=result.auc)
        return result
    except Exception as e:
        logger.error(
            "Analysis failed",
            dataset_id=dataset_id,
            error=str(e),
            exc_info=True
        )
        raise
```

---

## Security Considerations

### NIST SP 800-171 Compliance (for NIH data)

**Required as of January 2025:**

**Access Control:**
- Role-based access control (RBAC)
- Multi-factor authentication (MFA) for sensitive data
- Least privilege principle
- Session timeout after inactivity

**Audit and Accountability:**
- Log all data access (who, what, when, where)
- Tamper-evident logs
- 7-year retention
- Regular audit reviews

**Encryption:**
- TLS 1.3 for data in transit
- AES-256 for data at rest
- Encrypted database fields for PHI/PII
- Encrypted backups

**Implementation Checklist:**
```python
# Authentication middleware
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Log access
    audit_log.info("User authenticated", user_id=user.id, ip=request.client.host)
    
    return user

# Data access logging
async def access_dataset(dataset_id: str, user: User = Depends(get_current_user)):
    # Check permissions
    if not user.has_permission("read", dataset_id):
        audit_log.warning("Unauthorized access attempt", 
                         user_id=user.id, dataset_id=dataset_id)
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Log access
    audit_log.info("Dataset accessed", 
                   user_id=user.id, 
                   dataset_id=dataset_id,
                   action="read")
    
    return load_dataset(dataset_id)
```

### Data Use Agreements (DUA)

**Tracking System:**
```python
class DUA:
    def __init__(self, user_id: str, dataset_id: str, expiry_date: datetime):
        self.user_id = user_id
        self.dataset_id = dataset_id
        self.signed_date = datetime.now()
        self.expiry_date = expiry_date
    
    def is_valid(self) -\u003e bool:
        return datetime.now() \u003c self.expiry_date
    
    def can_access(self, user_id: str, dataset_id: str) -\u003e bool:
        return (self.user_id == user_id and 
                self.dataset_id == dataset_id and 
                self.is_valid())
```

---

## Commands Reference

### Development

```bash
# Setup
git clone https://github.com/org/omicselector2.git
cd omicselector2
docker-compose up -d

# Testing
pytest tests/smoke -v                 # Smoke tests (\u003c5 min)
pytest tests/unit -v                  # Unit tests
pytest tests/integration -v           # Integration tests
pytest tests/ --cov=src --cov-report=html  # Coverage report

# Code quality
black src/ tests/                     # Format code
isort src/ tests/                     # Sort imports
flake8 src/ tests/                    # Linting
mypy src/                             # Type checking

# Database
alembic revision --autogenerate -m "Description"  # Create migration
alembic upgrade head                              # Apply migrations

# Celery
celery -A src.tasks worker --loglevel=info        # Start worker
celery -A src.tasks flower                        # Start Flower (monitoring)

# MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db  # Start MLflow UI
```

### Production

```bash
# Build Docker images
docker build -t omicselector2-api:latest -f Dockerfile.api .
docker build -t omicselector2-worker:latest -f Dockerfile.worker .
docker build -t omicselector2-frontend:latest -f Dockerfile.frontend .

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Monitoring
docker logs -f omicselector2-api      # API logs
docker logs -f omicselector2-worker   # Worker logs
```

---

## Glossary

**Key Terms:**

- **Feature Selection**: Process of selecting relevant features (genes, variants, radiomics) from high-dimensional data
- **Stability Selection**: Ensemble method using subsampling to identify consistently selected features
- **Hold-out Validation**: Data split strategy (train/test/validation) preventing information leakage
- **SHAP**: SHapley Additive exPlanations - interpretability method based on game theory
- **AnnData**: Annotated data format for single-cell data (.h5ad files)
- **Cross-validation**: Resampling technique to assess model generalization
- **Ensemble**: Combining multiple methods to improve robustness
- **Biomarker**: Measurable indicator of biological state or condition

**File Formats:**
- **h5ad**: HDF5-based format for AnnData (single-cell)
- **VCF**: Variant Call Format (genetic variants)
- **BAM**: Binary Alignment Map (aligned sequencing reads)
- **FASTQ**: Raw sequencing data with quality scores

**Methods Abbreviations:**
- **RF-VI**: Random Forest Variable Importance
- **mRMR**: Minimum Redundancy Maximum Relevance
- **LASSO**: Least Absolute Shrinkage and Selection Operator
- **HDG/HEG**: High-Deviation Genes / High-Expression Genes
- **scFSNN**: Single-cell Feature Selection Neural Network

---

## Important Notes

### Data Processing
- Always validate input data format before processing
- Use streaming for files \u003e1GB to avoid memory issues
- Implement checkpointing for long-running analyses
- Store raw data immutably, process separately

### Feature Selection
- Perform feature selection INSIDE cross-validation loop to prevent data leakage
- Use stratified splits for imbalanced datasets
- Set random seeds for reproducibility
- Always include stability analysis for biomarker discovery

### Performance
- Use Celery for any task \u003e30 seconds
- Implement caching for frequently accessed results
- Monitor memory usage with large datasets
- Use GPU workers for deep learning tasks

### Security
- Never commit credentials to Git (use .env files)
- Rotate secrets regularly
- Implement rate limiting on API endpoints
- Validate all user inputs
- Log all access to sensitive data

### Deployment
- Always test in staging environment before production
- Use health checks for all services
- Implement automated backups
- Monitor error rates and performance metrics
- Have rollback plan ready

### Troubleshooting
- Check Celery logs for job failures: `docker logs omicselector2-worker`
- Verify database connections: Check PostgreSQL logs
- Redis issues: `redis-cli ping` should return PONG
- Frontend issues: Check browser console for errors
- API issues: Check `/docs` for OpenAPI documentation

---

## References

**Key Publications:**
- OmicSelector 1.0: Stawiski K, et al. bioRxiv 2022.06.01.494299
- Feature selection benchmarks: Li et al., BMC Bioinformatics 2022
- Stability Selection: Pusa \u0026 Rousu, PLOS One 2024
- Single-cell methods: Yang et al., Genome Biology 2021
- SHAP: Lundberg \u0026 Lee, NeurIPS 2017

**Tools \u0026 Frameworks:**
- FastAPI: https://fastapi.tiangolo.com/
- PyTorch: https://pytorch.org/
- Scanpy: https://scanpy.readthedocs.io/
- CanvasXpress: https://www.canvasxpress.org/
- SHAP: https://shap.readthedocs.io/
- MLflow: https://mlflow.org/

**Standards:**
- NIST SP 800-171: https://csrc.nist.gov/publications/detail/sp/800-171/rev-2/final
- NIH GDS Policy: https://sharing.nih.gov/genomic-data-sharing-policy
- OpenAPI: https://swagger.io/specification/

---

## Appendix: Sample Workflows

### Example 1: Bulk RNA-seq Biomarker Discovery

```python
# 1. Upload and validate data
dataset = upload_dataset("cancer_rnaseq.csv", data_type="bulk_rna_seq")
validate_dataset(dataset.id)

# 2. Preprocess
config = {
    "normalization": "tpm",
    "log_transform": True,
    "variance_threshold": 0.01
}
preprocessed = preprocess_dataset(dataset.id, config)

# 3. Feature selection with ensemble
job = create_job(
    dataset_id=preprocessed.id,
    methods=["rf_importance", "mrmr", "lasso", "elastic_net"],
    config={
        "cv_folds": 5,
        "test_size": 0.2,
        "validation_size": 0.2,
        "ensemble": {"enabled": True, "voting": "soft"},
        "stability": {"enabled": True, "n_bootstraps": 100}
    }
)

# 4. Monitor progress
wait_for_completion(job.id)

# 5. Retrieve results
results = get_results(job.id)
selected_features = results["ensemble_features"]  # Top features by consensus
performance = results["performance"]  # AUC, accuracy, etc.
stability = results["stability"]  # Stability scores

# 6. Train final model
model = train_model(
    dataset_id=preprocessed.id,
    features=selected_features,
    algorithm="random_forest"
)

# 7. Generate SHAP explanations
shap_values = explain_model(model.id)
plot_shap_summary(shap_values)
```

### Example 2: Single-cell RNA-seq Cell Type Markers

```python
# 1. Upload h5ad file
dataset = upload_dataset("pbmc_10x.h5ad", data_type="single_cell_rna_seq")

# 2. Quality control
qc_results = run_qc(dataset.id, min_genes=200, max_genes=2500, max_mito_pct=5)

# 3. Feature selection for clustering
job = create_job(
    dataset_id=dataset.id,
    methods=["hdg", "dubstepr", "feast"],  # Single-cell specific
    config={
        "n_top_genes": 2000,
        "clustering_resolution": 0.5
    }
)

# 4. Get results
results = get_results(job.id)
marker_genes = results["selected_features"]

# 5. Differential expression between clusters
de_results = run_differential_expression(
    dataset.id,
    cluster_column="leiden",
    method="wilcoxon"
)
```

---

**Document Version**: 1.0  
**Last Updated**: November 5, 2025  
**Maintainer**: [Your Name/Team]

---

This CLAUDE.md provides comprehensive guidance for agentic coding of OmicSelector2, following Anthropic's best practices with clear specifications, modular design, and iterative development roadmap.