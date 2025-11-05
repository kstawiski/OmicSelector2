

# **📋 Project Blueprint: OmicSelector2**

**(This file is CLAUDE.md)**

## **🚨 CRITICAL RULES & OPERATING PRINCIPLES**

This section defines the non-negotiable rules for the agentic coding process. All actions must adhere to these directives.

1. **Agentic Workflow:** You MUST follow the **Explore, Plan, Code, Commit** workflow.1 For any new feature, you will first *plan* (in a temporary markdown file or scratchpad), await human approval of the plan, and *then* implement.  
2. **Strict TDD:** ALL code generation MUST follow Test-Driven Development (TDD).1  
   * You MUST write the pytest test first.  
   * You MUST run the test and confirm it *fails* for the correct reason.  
   * You MUST commit *only* the failing test.  
   * You MUST then write the *minimal* implementation code to pass the test.  
   * You MUST NOT modify the test file during implementation.1  
   * You MUST commit the passing code.  
   * This cycle (RED-GREEN-REFACTOR) is mandatory.2 Any deviation will be rejected.  
3. **Python Version:** All Python code MUST target Python 3.11+.  
4. **Deep Learning Framework:** The primary deep learning framework MUST be **PyTorch**.3 For Graph Neural Networks, **PyTorch Geometric (PyG)** 5 MUST be used.  
5. **Modularity:** The system MUST be built as a set of **decoupled microservices**.8 A single monolithic application is forbidden.  
6. **API-First Design:** All services MUST communicate via RESTful APIs. The backend MUST be **FastAPI**, and all endpoints MUST use Pydantic models for request/response validation.8  
7. **Documentation:** All code MUST be fully type-hinted and include Google-style docstrings. All API endpoints MUST be documented with OpenAPI-compatible descriptions.  
8. **Context Management:** You MUST use the provided CLAUDE.md files (this file) and custom slash commands (.claude/commands/) for context.1 You are expected to read and understand all files in the /docs directory as the single source of truth for architectural decisions.

## **🎯 PROJECT CONTEXT & SYSTEM ARCHITECTURE**

### **1\. Project Mandate: Modernizing Oncological Biomarker Discovery**

The objective of OmicSelector2 is to transition from the 1.0 framework's paradigm of *automated feature selection* to a new paradigm of *guided multi-modal integration*. The exponential rise in heterogeneous data—spanning genomics, transcriptomics, radiomics, and single-cell data 11—has rendered traditional "feature-wrapping" 14 insufficient. True biomarker discovery in oncology now requires methods that can *integrate* these data types to capture complex, non-linear patterns.16 OmicSelector2 will provide a robust, scalable, and user-friendly Python-based platform to discover, benchmark, and deploy these next-generation multi-omic signatures.

### **2\. OmicSelector 1.0 Analysis & OmicSelector2 Modernization Goals**

A thorough analysis of OmicSelector 1.0 14 reveals its architecture and philosophy.

* **OmicSelector 1.0 (The "What"):** It is a "mega-wrapper" 23 built in **R** 14 with a **Shiny**\-based UI.18 Its core function OmicSelector\_OmicSelector 21 orchestrates a large number of existing R feature selection packages (e.g., Boruta, varSelRF, caret, rpart, C5.0, rf) 15 and a simple deep learning module via keras.20  
* **Core Value (The "Why"):** Its primary innovation was not novel algorithms, but **automated benchmarking**.14 It runs dozens of feature selection methods on a dataset, then tests (benchmarks) the resulting "signatures" (feature sets) using various ML models to find the one most "resiliant \[to\] overfitting".14  
* **Limitations:** The R-based, monolithic architecture 23 is a significant bottleneck. It has massive dependency conflicts, is difficult to scale, and, most importantly, lacks access to the SOTA Python-native ecosystem for deep learning-based multi-omic integration (e.g., PyG, scverse).  
* **OmicSelector2 Modernization Goals:**  
  1. **Retain the Core Philosophy:** The "automated benchmarking" of signatures is a key user requirement and must be the central workflow of OmicSelector2.  
  2. **Modernize the Stack:** Migrate 100% from R/Shiny to Python/FastAPI/Dash. This is not just a language swap; it is a strategic necessity to access SOTA libraries.3  
  3. **Expand the Methods:** The "methods" being benchmarked will no longer be simple wrappers.15 They will be SOTA multi-modal integration frameworks (e.g., GNNs, VAEs) as identified in the SOTA review.25  
  4. **Handle Multi-Omic Data:** The platform must be *natively* multi-modal, capable of handling scRNA-seq, radiomics, and WES as requested by the user.

**Table 1: OmicSelector 1.0 \-\> OmicSelector2 Migration & Modernization Map**

| Feature/Component | OmicSelector 1.0 (R) | OmicSelector 2.0 (Python) | Justification & Rationale |
| :---- | :---- | :---- | :---- |
| **Core Language** | R 14 | Python 3.11+ | Access to SOTA deep learning, multi-omic (scverse), and MLOps ecosystems.\[5, 24, 28\] |
| **Backend** | Monolithic R package 14 | **FastAPI** Microservices 8 | Scalability, high-performance (ASGI), API-first design, native ML model serving.\[28, 29, 30\] |
| **Frontend UI** | R Shiny via OmicApp 18 | **Plotly Dash** 31 | Pure-Python, scalable for complex data, more flexible than Streamlit \[32, 33\], and can be mounted in FastAPI.34 |
| **Data Structure** | mixed\_train.csv 21 (Flat CSVs) | **Muon (muon)** 24 | Muon is the standard for multi-modal omics 24, holding separate AnnData objects for each modality (e.g., RNA, ATAC, Radiomics). |
| **Feature Selection** | \~60-70 wrappers (e.g., rpart, C5.0, Boruta) \[15, 21\] | **SOTA Integration Models:** GNNs (MOGDx-like), VAEs (MOVE-like), Radiomic Fusion.\[3, 26, 35\] | The paradigm has shifted from FS to *integration*.\[11, 12\] OmicSelector2 will benchmark integration strategies. |
| **Benchmarking** | "benchmarking" via ML models \[14, 18\] | **MLflow**\-tracked Benchmarking Service | OmicSelector2 formalizes this process using a dedicated microservice and MLflow 36 for robust, reproducible experiment tracking. |
| **Model Dev** | Keras 20 | **PyTorch** 3 / **PyTorch Geometric** 5 | PyTorch is the dominant framework for research and SOTA models like GNNs and VAEs.\[3, 6\] |
| **Model Deployment** | Not specified (likely manual) | **BentoML** \[28, 37\] | Provides a seamless path from a trained model in MLflow to a production-ready, containerized API endpoint.\[38, 39\] |
| **Modality Support** | miRNA-seq, RNA-seq, qPCR 20 | scRNA-seq, WES, Radiomics, Bulk RNA-seq | Direct response to user query. Requires specialized libraries like scanpy 24 and pyradiomics.40 |

### **3\. Core Technology Stack & Justification**

The OmicSelector2 stack is chosen to build a high-performance, scalable, and future-proof platform.

A successful scientific platform is not defined just by its algorithms, but by the *integration* of its data pipeline, computational framework, and user interface. The chosen stack creates a "virtuous cycle" entirely within the Python ecosystem: a user uploads data (to Dash) 31, which calls the FastAPI backend.29 FastAPI 8 validates the data and sends a job to a compute worker. The worker loads the data into a Muon object 24, processes it with PyRadiomics 40 or Scanpy 24, and then runs a PyTorch Geometric GNN model.5 The results (metrics, features) are logged to MLflow.36 The "best" model is then imported by BentoML 37 and deployed as a new API. The FastAPI backend orchestrates this entire flow. This seamlessness is impossible in the OmicSelector 1.0 R-based architecture.

**Table 2: OmicSelector2 Technology Stack Justification**

| Component | Selected Technology | Alternative(s) | Justification |
| :---- | :---- | :---- | :---- |
| **Backend API** | **FastAPI** 29 | Django \[30, 41\] | **Performance & ML:** FastAPI is built on ASGI for high-performance I/O, ideal for ML microservices. Django is monolithic and slower.\[30, 41\] FastAPI's Pydantic validation 8 is superior for API development. |
| **Frontend UI** | **Plotly Dash** 31 | Streamlit \[32, 33\] | **Scalability & Complexity:** Streamlit is for rapid prototypes.\[32\] Dash is built on Flask, designed for complex, scalable, enterprise-grade dashboards.\[31, 33\] This matches our need. Crucially, Dash (as Flask) can be mounted within FastAPI.34 |
| **Data Core** | **Muon / AnnData** 24 | Pandas DataFrames | **Multi-Omic Standard:** The scverse ecosystem (scanpy, muon) 24 is the *de facto* standard for single-cell and multi-modal omics. Muon 24 is specifically designed to hold multiple AnnData objects, making it the perfect container for our multi-omic data. |
| **Radiomics** | **PyRadiomics** 40 | Custom scripts | **Standardization & Reproducibility:** pyradiomics is the open-source standard for radiomic feature extraction. It provides 2D/3D support, filtering (Wavelet, LoG) 42, and, critically, provenance information 42 for reproducibility. |
| **Graph DL** | **PyTorch Geometric (PyG)** 5 | DGL, Stellargraph \[43, 44\] | **Ecosystem:** PyG is the most widely adopted and maintained GNN library for PyTorch.\[6, 45\] SOTA research directly references it.\[7, 43\] |
| **Integration DL** | **PyTorch** 3 | TensorFlow \[46, 47\] | **Research & Flexibility:** PyTorch is the dominant framework in the research community 3 for developing novel architectures like VAEs and GNNs, offering a more flexible "Pythonic" development cycle. |
| **Experiment Tracking** | **MLflow** 36 | Weights & Biases 36 | **Open-Source & Integration:** MLflow \[28, 36\] is the leading open-source tool. Its key advantage is its seamless integration with BentoML for the model deployment pipeline.37 |
| **Model Deployment** | **BentoML** 28 | Custom Flask/FastAPI | **Productionization:** BentoML \[28, 37\] standardizes the *entire* deployment process: it packages the model, defines the API, and builds a production-grade Docker container.\[39\] This "model serving" is a specialized MLOps task that should not be rebuilt from scratch. |

### **4\. System Architecture: A Decoupled Microservices Framework**

OmicSelector2 MUST be built on a microservice architecture 8 to manage the heterogeneous and computationally intensive tasks.

A monolithic app (like OmicSelector 1.0's Docker 20) is unmanageable. The dependencies for pyradiomics 48, scverse 24, and pytorch\_geometric 5 are complex and will conflict. A microservice architecture 8 solves this by containerizing each function. We can have a specific Docker container *just* for GNN training with the correct PyG and CUDA versions, and a separate one *just* for pyradiomics processing. This promotes scalability (we can scale the GNN worker to multiple GPU nodes) and resilience (a crash in the radiomics worker won't bring down the UI).

The greatest challenge is integrating the Python-based UI (Dash) with the Python-based API (FastAPI). Dash is built on Flask 31, a WSGI application. FastAPI is an ASGI application.10 They cannot run in the same process natively. However, FastAPI provides a WSGIMiddleware.34 This allows us to mount the *entire* Dash/Flask application *inside* the FastAPI application.34

This solution leads to a single "Gateway" service. This service is a FastAPI app. It will serve all the API endpoints (e.g., /api/v1/submit\_job). It will also mount the entire Dash application at the root path or at /ui (e.g., app.mount("/", WSGIMiddleware(dash\_app.server)) 34). This provides a single entry point, simplified Dockerization, and allows us to use FastAPI's dependencies (e.g., OAuth2 8) to secure *both* the API and the UI.

**Architecture Diagram:**

(User's Browser)  
|  
       v

|  
       v

|  
  \+--

| |  
| \+-- /api/v1/\* (Endpoints for jobs, results, auth)  
| | |  
| | \+--\> (Job Queue)  
| |  
| \+-- / (Mounted Dash App via WSGIMiddleware )  
| (UI dashboards, data upload, viz)  
|  
  \+-- \[Gunicorn\] (WSGI server for Dash, managed by FastAPI)

**Downstream Services (All separate Docker Containers):**

\[Job Queue\]  
|  
  \+--\> (pyradiomics , scanpy )  
|  
  \+--\> (PyTorch, PyG )  
|  
  \+--\> (scikit-learn, xgboost)

**MLOps Infrastructure (Separate Docker Containers):**

  ^

| (All workers log to this)  
  v

|  
  \+--\> (e.g., /predict\_biomarker\_1)

---

## **STATE-OF-THE-ART REVIEW: MODERN MULTI-OMIC FEATURE SELECTION (2024-2025)**

This section fulfills the user's request for PubMed research, synthesizing SOTA literature 50 to inform the methods we will implement in EPIC 2\.

### **1\. The Paradigm Shift: From Classic Filters to Deep Learning Integration**

The field has decisively moved beyond simple statistical filters (e.g., t-tests 18, univariate filtering 50) and classic ML wrappers (RF, SVM 50) for high-dimensional data. The consensus is that Deep Learning (DL) is the most effective tool for "automatic feature extraction and pattern recognition" 11 from complex, heterogeneous multi-omic datasets.12

The SOTA *problem* is no longer "feature selection" but "data integration." OmicSelector 1.0 14 worked by selecting features from a *single* data matrix. SOTA research 12 now focuses on integrating *multiple* matrices (e.g., genomics, transcriptomics, proteomics) *first*, to find patterns that are invisible to any single omic layer. DL models, particularly VAEs 27 and GNNs 25, are the primary tools for this integration. OmicSelector2's "feature selection" engine must therefore be an "integration engine."

### **2\. SOTA Multi-Modal Integration Frameworks (GNNs, VAEs, Transformers)**

We must implement and benchmark the following SOTA integration strategies.

#### **2.1. Graph-Based Integration (GNNs)**

This is the most powerful and promising SOTA approach for multi-omic classification and biomarker discovery.25

* **Core Concept:** Instead of viewing data as a (patients x features) matrix, this approach transforms it into a (patients x patients) graph, or *Patient Similarity Network (PSN)*.26 A Graph Neural Network (GNN) is then trained to classify the *nodes* (patients) in this graph.25  
* **Key Frameworks to Re-implement:**  
  * **MOGDx (Multi-Omic Graph Diagnosis):** This method 26 is a flexible tool for multi-omic classification.  
    * **Pipeline:**  
      1. Input: Multiple omic matrices (patient x feature).58  
      2. For each omic, construct a PSN (e.g., using K-Nearest Neighbors).43  
      3. Fuse these multiple PSNs into *one* integrated graph using **Similarity Network Fusion (SNF)**.26  
      4. Train a **Graph Convolutional Network (GCN)** 26, implemented in PyG 43, on the fused graph to classify patients.  
  * **MOGONET (Multi-Omic Graph-Oriented Network):**  
    * **Pipeline:** This approach 26 keeps the omic layers separate for longer.  
      1. Construct a PSN for *each* omic type.  
      2. Train a *separate* GCN for each omic-specific graph.  
      3. Fuse the *outputs* (predictions) of these GCNs using a "View Correlation Discovery Network (VCDN)".26

How do we get a "feature list" from a GNN? The GNN itself doesn't directly select features like Boruta. It learns patient relationships. The "biomarkers" are the *features that were most important for building the graph*. MOGDx's paper 58 states it performs differential expression or penalized regression *first* to identify important features *before* building the PSN. Therefore, OmicSelector2's GNN module will be a two-stage process: (1) a preliminary FS to build the graph, and (2) GNN-based classification, where the *preliminary feature set* becomes the "signature." We can also use GNN interpretability (like GNNExplainer 7) to rank the importance of node features.

#### **2.2. VAE-Based Latent Space Integration**

Variational Autoencoders (VAEs) are a powerful unsupervised approach to integrate multi-omic data into a single, compressed *latent space*.27

* **Core Concept:** A VAE 4 is trained to "compress" all multi-omic data for a patient into a small set of numbers (e.g., 64 numbers) in the latent space, and then "decompress" it back to the original data. This forces the latent space to capture the most important, shared biological information.  
* **Key Framework to Re-implement:**  
  * **MOVE (Multi-Omics Variational autoEncoder):** This framework 3 uses an *ensemble* of VAEs to integrate both categorical and continuous data. It is implemented in PyTorch.3 The resulting latent space can be used for patient clustering, survival analysis, and biomarker discovery (by perturbing the latent space 60).

The "features" from a VAE are the nodes of the latent space.3 These nodes represent high-level biological processes. We can test these latent features for association with clinical outcomes. Alternatively, we can find the input features (genes, proteins) that contribute *most* to the most important latent nodes.

#### **2.3. Attention-Based Modality Fusion**

This is an emerging area, borrowing from Large Language Models (LLMs) and computer vision.61

* **Core Concept:** Use attention mechanisms (e.g., Transformers) to "weigh" the importance of different modalities and features when making a prediction.25 For example, a model might learn that for a specific cancer subtype, radiomic features 63 are more important than WES data, but for another subtype, the reverse is true.  
* **Key Framework to Re-implement:**  
  * **LASSO-MOGAT:** This 2024 study 25 uses LASSO for initial feature reduction, then integrates mRNA, miRNA, and methylation data using a **Graph Attention Network (GAT)** 25, which is a GNN that uses attention. It achieved 95.9% accuracy in classifying 31 cancer types, outperforming GCNs. This is a *hybrid* GNN/Attention model and represents the absolute state-of-the-art.

### **3\. Advancements in Modality-Specific Feature Engineering**

The user's new data types (scRNA-seq, radiomics) require specialized SOTA pipelines.

* **Radiomics & Deep Feature Fusion:**  
  * **SOTA:** The *best* models *fuse* "hand-crafted" radiomic features with "deep features".35  
  * **Hand-crafted:** These are the features extracted by pyradiomics 40: shape, texture (NGTDM, GLCM), intensity, etc..40  
  * **Deep Features:** These are features extracted by running the medical image (e.g., MRI slice) through a pre-trained Convolutional Neural Network (CNN).35 The output of the second-to-last layer of the CNN is used as a feature vector.  
  * **OmicSelector2 Implementation:** We MUST implement a pipeline that does both: (1) extract pyradiomics features, (2) extract deep features using a pre-trained ResNet/VGG on the image, and (3) *concatenate* these two feature vectors 64 before feeding them into a classifier.  
* **Single-Cell & Spatial Transcriptomics:**  
  * **SOTA:** The scverse ecosystem (scanpy, muon, scvi-tools, squidpy) 24 is the standard for analysis. scvi-tools 66 provides deep generative models (like VAEs) specifically for scRNA-seq.  
  * **OmicSelector2 Implementation:** Our data-ingest module (EPIC 1\) must use scanpy 24 to read and preprocess scRNA-seq data. For feature selection *from* single-cell data, we can run scvi-tools 66 or even novel methods like QUBO 67 to identify key genes, which are then used as the "signature" in the benchmarking step.

### **4\. The Imperative of Robust Benchmarking, Validation, & Interpretability**

This is the central thesis of OmicSelector2, echoing the philosophy of 1.0. The field is now *full* of complex DL models.11 It is difficult to know which is best. Most DL models are "black boxes" 69 and suffer from a lack of validation, interpretability, and standardization.53

OmicSelector2 will be the *first* platform to systematically and fairly **benchmark** these SOTA integration methods (GNNs vs. VAEs vs. Fusion) against each other on a user's dataset.

This requires rigorous validation:

1. **Cross-Validation:** Nesting CV is essential to prevent "overfitting and spurious correlations".70  
2. **External Validation:** The platform must allow users to upload a separate validation cohort.53  
3. **Interpretability:** We must provide "explainability techniques" 53 for our models (e.g., GNNExplainer 7 for GNNs, SHAP values for tree models).

**Table 3: SOTA Multi-Omic Integration & Feature Selection Methods (OmicSelector2 Implementation Plan)**

| Method Class | SOTA Framework | Key Papers | OmicSelector2 Implementation Plan (EPIC 2\) | Data Types |
| :---- | :---- | :---- | :---- | :---- |
| **Baseline (Parity)** | Classic ML Wrappers | OmicSelector 1.0 15 | Boruta (boruta\_py), RFE (sklearn), Lasso (sklearn), RandomForest (sklearn). Run on concatenated omics. | Any (concatenated) |
| **Graph Neural Network** | **MOGDx** 26 | Ryan et al. \[57\], Wang et al. \[56\] | **Task 2.3:** Implement the MOGDx pipeline: PSN construction per omic 43, SNF fusion 43, GCN classification via PyTorch Geometric.43 | Multi-Omic |
| **Graph \+ Attention** | **LASSO-MOGAT** 25 | Alharbi et al. 25 | **Task 2.3 (Advanced):** Implement a Graph Attention Network (GAT) module in PyG.5 Use LASSO as a pre-filter. | Multi-Omic |
| **Variational Autoencoder** | **MOVE** 3 | Rasmussen et al. 3 | **Task 2.2:** Implement a multi-modal VAE in PyTorch 3 to find a joint latent space. Benchmark performance of latent features. | Multi-Omic |
| **Radiomic Fusion** | **Deep Feature Fusion** | Zhang et al. 64, Hu et al. \[35\] | **Task 2.4:** Implement a "Radiomic-Fusion" pipeline: (1) Extract pyradiomics features 40, (2) Extract deep features via pre-trained CNN \[35\], (3) Concatenate and use as signature. | Radiomics \+ Image |
| **Single-Cell** | **scvi-tools / QUBO** | scvi-tools 66, Romero et al. 67 | **Task 1.3:** Use scvi-tools VAEs 66 to identify latent-space-associated genes as a signature. | scRNA-seq |

---

## **🛠️ AGENTIC DEVELOPMENT PLAN: EPICS & TASKS**

This plan follows the Agile methodology.72 Each Task will be initiated by a human operator, and you (the agent) will execute it following the TDD workflow.1

### **EPIC 1: Core System & Data Handling Architecture**

Goal: Establish the project's skeleton, data structures, and communication pathways.  
Agentic Workflow: TDD 1 and Plan-Driven.1

* **Task 1.1: Initialize Monorepo & Agentic Tooling**  
  * **Plan:**  
    1. Create the root project directory omicselector2/.  
    2. Create CLAUDE.md (this file).  
    3. Create README.md (with project overview).  
    4. Create .gitignore (for Python, IDEs, and data).  
    5. Initialize git.  
    6. Create pyproject.toml and define core dependencies (e.g., fastapi, uvicorn, pytest, muon, scanpy, pyradiomics, torch, torch-geometric, mlflow, bentoml).  
    7. Create the agent configuration directory .claude/.  
    8. Create agent commands in .claude/commands/ (e.g., /new\_fs\_module, /new\_api\_endpoint) as defined in the final section of this document.1  
  * **Execute:** You will now perform this setup.  
* **Task 1.2: Define Core Data Structures (Muon Schemas)**  
  * **Plan:**  
    1. This task is documentation-heavy but critical. Create a new file: docs/DATA\_FORMATS.md.  
    2. In this file, define the *exact* Muon 24 object structure that OmicSelector2 will use as its central data representation.  
    3. *Example Schema to Document:*  
       * mdata: The main Muon object.  
       * mdata.obs: Patient-level metadata (e.Ifc., 'patient\_id', 'survival\_time', 'event\_status', 'cancer\_type').  
       * mdata.mod\['rna'\]: An AnnData object for bulk RNA-seq. X \= (patients x genes) log-normalized counts.  
       * mdata.mod\['scrna'\]: An AnnData object for scRNA-seq. X \= (cells x genes). obs must contain 'patient\_id' to link to mdata.obs.  
       * mdata.mod\['radiomics'\]: An AnnData object for radiomics. X \= (patients x radiomic\_features). var will contain feature provenance from pyradiomics.42  
       * mdata.mod\['wes'\]: An AnnData object for WES. X \= (patients x genes) binary mutation matrix (0/1).  
  * **Execute:** Create this docs/DATA\_FORMATS.md file now.  
* **Task 1.3: TDD Implement Data Ingestion Service**  
  * **Plan:**  
    1. Create a new microservice directory: services/data\_ingest\_service/.  
    2. This service will have functions to parse various file types and return a standardized Muon object (as defined in docs/DATA\_FORMATS.md).  
    3. **TDD (RED):** Write pytest test tests/test\_ingest\_service.py with a test test\_parse\_pyradiomics\_csv. This test will use a mock CSV (emulating a pyradiomics output 75) and assert that the correct AnnData object is created.  
    4. **TDD (GREEN):** Implement the minimal code in services/data\_ingest\_service/parser.py to pass the test.  
    5. **TDD (REFACTOR):** Refactor.  
    6. **REPEAT TDD:** Repeat this TDD cycle for:  
       * test\_parse\_10x\_h5 (using scanpy.read\_10x\_h5 24).  
       * test\_parse\_dicom\_folder (this will be complex; it must call the pyradiomics CLI 75 or library 40 to *generate* the feature CSV first, then parse it).  
       * test\_build\_muon\_object (combining multiple mock AnnData objects into a single Muon object).  
  * **Execute:** Begin with the TDD cycle for test\_parse\_pyradiomics\_csv.  
* **Task 1.4: TDD Establish FastAPI Backend (Gateway)**  
  * **Plan:**  
    1. Create the main gateway directory: services/gateway/.  
    2. Create services/gateway/main.py (your FastAPI app).  
    3. **TDD (RED):** Write tests/test\_gateway.py. Start with a simple test test\_read\_root. It should make a request to / and assert a 200 OK status.  
    4. **TDD (GREEN):** Implement the minimal @app.get("/") in main.py.  
    5. **TDD (REFACTOR):** Refactor.  
    6. **TDD (Microservice Arch):**  
       * Implement a /api/v1/ingest endpoint (placeholder).  
       * This endpoint will *not* do the work. It will use Pydantic 8 to validate input and then place a job on a (mocked) job queue.  
       * Write a test test\_submit\_ingest\_job. Mock the job queue. Assert the endpoint returns a job\_id and that the mock queue was called with the correct data.  
    7. **TDD (Dash Integration):**  
       * Create a minimal Dash app services/gateway/dash\_app.py (e.g., app.layout \= html.Div("Hello Dash") 34).  
       * In main.py, mount the Dash app using WSGIMiddleware 34 (e.g., app.mount("/", WSGIMiddleware(dash\_app.server))).  
       * Write a test in tests/test\_gateway.py (test\_dash\_app\_is\_mounted) that makes a GET request to / and checks for the "Hello Dash" content.  
  * **Execute:** Begin the TDD cycle for the FastAPI gateway.

### **EPIC 2: Feature Selection (FS) Engine**

Goal: Implement the "methods" for OmicSelector2, from simple baselines to SOTA GNNs.  
Agentic Workflow: This EPIC is algorithm-heavy. You will use the /new\_fs\_module custom command.

* **Task 2.1 (Parity): TDD Implement Baseline FS Methods**  
  * **Plan:**  
    1. This module will run on a *concatenated* AnnData object (a "flattened" Muon object) for baseline comparison.  
    2. Use custom command /new\_fs\_module module\_name="baseline\_rf" description="RandomForest-based feature importance"  
    3. **TDD (RED):** The command will generate tests/test\_baseline\_rf.py. You will add a mock AnnData object and assert that the function returns a list of feature names, sorted by importance.  
    4. **TDD (GREEN):** Implement the function in omicselector2/fs\_engine/modules/baseline\_rf.py using sklearn.ensemble.RandomForestClassifier.  
    5. **REPEAT TDD:** Repeat this process for:  
       * /new\_fs\_module module\_name="baseline\_boruta" (using boruta\_py).  
       * /new\_fs\_module module\_name="baseline\_lasso" (using sklearn.linear\_model.Lasso).  
  * **Execute:** Begin with /new\_fs\_module module\_name="baseline\_rf".  
* **Task 2.2 (SOTA): TDD Develop VAE-based Integration Module (MOVE-like)**  
  * **Plan:**  
    1. This is a complex DL task. Use /new\_fs\_module module\_name="vae\_integration".  
    2. **TDD (RED):** Write tests/test\_vae\_integration.py. The test test\_vae\_model\_trains will:  
       * Create a mock Muon object 3 (e.g., 2 modalities).  
       * Initialize the VAE model (defined in .../modules/vae\_integration.py).  
       * Run a *single* training step 46 on the mock data.  
       * Assert that the model parameters have changed and the loss is a valid number.  
    3. **TDD (GREEN):** Implement the VAE model in PyTorch.3 The model's forward method must accept the Muon object, process each modality through a separate encoder, concatenate the latent representations, and then pass them through a shared decoder for reconstruction. The loss function will be the standard VAE ELBO loss.46  
    4. **TDD (RED \- Feature Extraction):** Write a test test\_get\_latent\_features.  
    5. **TDD (GREEN):** Implement a function that takes a trained VAE and a Muon object, and returns the AnnData object of the latent space (patients x latent\_dims). This *is* the "signature" from this method.  
  * **Execute:** Begin TDD for the VAE module.  
* **Task 2.3 (SOTA): TDD Develop GNN-based Integration Module (MOGDx-like)**  
  * **Plan:**  
    1. This is the most complex SOTA module. Use /new\_fs\_module module\_name="gnn\_mogdx".  
    2. **TDD (RED \- PSN):** Write tests/test\_gnn\_mogdx.py. Start with test\_build\_psn\_from\_adata. It will take a mock AnnData object and assert that a (patients x patients) similarity matrix (e.g., from sklearn.metrics.pairwise\_distances) is returned.  
    3. **TDD (GREEN \- PSN):** Implement the PSN builder.  
    4. **TDD (RED \- SNF):** Write test\_fuse\_networks. It will take a list of mock PSNs and assert a single fused matrix is returned. (You will need to implement the SNF algorithm).  
    5. **TDD (GREEN \- SNF):** Implement SNF.43  
    6. **TDD (RED \- GCN):** Write test\_gcn\_model\_trains. This will be similar to the VAE test:  
       * Create a mock torch\_geometric.data.Data object 5 from the fused PSN.  
       * Initialize the GCN model (defined in .../modules/gnn\_mogdx.py).  
       * Run a single training step.  
       * Assert loss is valid.  
    7. **TDD (GREEN \- GCN):** Implement a simple 2-layer GCN classifier using PyG (GCNConv layers).5  
    8. The "signature" from this method is the *set of features used to build the PSN*.58 The module's main function will first run baseline\_lasso (Task 2.1) on each modality, use *those* features to build the PSNs, and then run the GNN. The "signature" it returns is the union of the LASSO features.  
  * **Execute:** Begin TDD for the GNN/MOGDx module.  
* **Task 2.4 (SOTA): TDD Develop Radiomic Feature Fusion Module**  
  * **Plan:**  
    1. Use /new\_fs\_module module\_name="radiomic\_fusion".  
    2. This module will take two inputs: an AnnData of pyradiomics features 40 and a folder of raw image files (e.g., NIfTI).  
    3. **TDD (RED):** Write test\_get\_deep\_features. It will (with mocking) pass a mock image to a function and assert a feature vector (e.g., 512-dim) is returned.  
    4. **TDD (GREEN):** Implement the function. It will load a pre-trained torchvision ResNet, run the image through it, and capture the output of the final pooling layer.  
    5. **TDD (RED):** Write test\_fuse\_radiomic\_features.  
    6. **TDD (GREEN):** Implement the function. It will take the pyradiomics AnnData and the deep features (from the previous step) and concatenate them into a single AnnData object.35 This is the "signature."  
  * **Execute:** Begin TDD for the Radiomic Fusion module.  
* **Task 2.5: TDD Create FS Pipeline Service**  
  * **Plan:**  
    1. Create services/fs\_engine\_service/. This is a FastAPI microservice that will host and run all the modules from Tasks 2.1-2.4.  
    2. **TDD (RED):** Write tests/test\_fs\_engine\_api.py. Mock the baseline\_rf module. Write a test test\_run\_baseline\_rf\_job. It will call the /api/v1/run\_fs endpoint with method="baseline\_rf" and a mock Muon object.  
    3. **TDD (GREEN):** Implement the API endpoint. It will receive the request, call the (mocked) baseline\_rf function, and return the signature (list of features).  
    4. **TDD (REFACTOR):** Un-mock and integrate the *actual* baseline\_rf module.  
    5. **REPEAT TDD:** Repeat for gnn\_mogdx, vae\_integration, etc. This service is the central "method" orchestrator.  
  * **Execute:** Begin TDD for the FS Engine service.

### **EPIC 3: Signature Benchmarking & Validation Module**

**Goal:** Re-create the *core value* of OmicSelector 1.0 14: benchmarking the signatures found in EPIC 2\.

* **Task 3.1: TDD Develop Standardized Cross-Validation Framework**  
  * **Plan:**  
    1. Create services/benchmarking\_service/.  
    2. **TDD (RED):** Write tests/test\_benchmarking.py. Start with test\_nested\_cv\_split. It will take a mock AnnData object (100 patients) and assert that the correct number of (outer\_fold, inner\_fold) index-sets are generated.  
    3. **TDD (GREEN):** Implement a robust nested cross-validation splitter 70 (e.g., StratifiedKFold inside StratifiedKFold).  
  * **Execute:** Begin TDD for the CV framework.  
* **Task 3.2: TDD Implement Multi-Metric Evaluation Service**  
  * **Plan:**  
    1. This service will contain the "benchmark" function.  
    2. **TDD (RED):** Write a test test\_run\_benchmark. This test will:  
       * Create a mock AnnData object (patients x features) and a "signature" (a list of 10 feature names).  
       * Call run\_benchmark(adata, signature, cv\_folds).  
    3. **TDD (GREEN):** Implement run\_benchmark. This function will:  
       * Take the adata and *subset* it to *only* the features in signature.  
       * Loop through the nested CV folds (from 3.1).  
       * In the *inner* loop, train a simple model (e.g., RandomForestClassifier).  
       * In the *outer* loop, test the trained model.  
       * Calculate metrics (AUC, F1, Log-rank if survival data is present 71).  
       * Return a dictionary of aggregated metrics (e.g., {'mean\_test\_auc': 0.85}).  
  * **Execute:** Begin TDD for the run\_benchmark function.  
* **Task 3.3: TDD Build "Best Signature" Selection Logic**  
  * **Plan:**  
    1. This is the final orchestration step. The Gateway API (from 1.4) will call this.  
    2. **TDD (RED):** Write test\_find\_best\_signature. It will take a list of (mocked) benchmark results (e.g., \[{'signature\_name': 'baseline\_rf', 'mean\_test\_auc': 0.8}, {'signature\_name': 'gnn\_mogdx', 'mean\_test\_auc': 0.95}\]).  
    3. **TDD (GREEN):** Implement the function. It will simply sort the list by mean\_test\_auc 71 and return the name of the top-performing signature.  
  * **Execute:** Begin TDD for the selection logic.

### **EPIC 4: Model Development & MLOps Pipeline**

**Goal:** Productionize the biomarker discovery workflow, from experiment tracking to a deployable API.

* **Task 4.1: Integrate MLflow Tracking**  
  * **Plan:** This is a refactoring task. No TDD, but integration.  
    1. Set up a docker-compose.yml for the mlflow server.36  
    2. **Refactor EPIC 2:** Go into *every* FS module (e.g., baseline\_rf, gnn\_mogdx) and add mlflow.log\_param() calls for model hyperparameters.  
    3. **Refactor EPIC 3:** Go into the run\_benchmark function (Task 3.2). Add mlflow.log\_metric() calls for all metrics (AUC, F1, etc.).37  
    4. **Refactor EPIC 3:** At the end of run\_benchmark, log the signature (feature list) as an mlflow.log\_artifact().  
  * **Execute:** Set up the MLflow server and begin refactoring EPIC 2 and 3\.  
* **Task 4.2: TDD Develop Model Training Service**  
  * **Plan:**  
    1. This is *different* from benchmarking. This trains one *final* model on the "best" signature.  
    2. **TDD (RED):** Write tests/test\_model\_trainer.py. The test test\_train\_final\_model will:  
       * Mock the mlflow client.  
       * Call train\_final\_model(adata, best\_signature).  
       * Assert that mlflow.sklearn.log\_model() (or mlflow.pytorch.log\_model()) was called.  
    3. **TDD (GREEN):** Implement the function. It subsets the data to the best\_signature, trains a model on *all* the training data (no CV), and logs the final model to MLflow.37  
  * **Execute:** Begin TDD for the final model trainer.  
* **Task 4.3: TDD Integrate BentoML Model Registry**  
  * **Plan:**  
    1. This service *pulls* from MLflow and *pushes* to BentoML.  
    2. **TDD (RED):** Write tests/test\_bento\_registry.py. The test test\_import\_model\_to\_bento will:  
       * Mock mlflow.tracking.MlflowClient and bentoml.mlflow.import\_model.38  
       * Call productionize\_model(mlflow\_run\_id).  
       * Assert that bentoml.mlflow.import\_model was called with the correct model URI from the MLflow run.38  
    3. **TDD (GREEN):** Implement productionize\_model.  
  * **Execute:** Begin TDD for the BentoML import logic.  
* **Task 4.4: Create Automated Deployment Pipeline (BentoML)**  
  * **Plan:**  
    1. This task is about configuration, not TDD.  
    2. Create services/model\_serving/.  
    3. Create services/model\_serving/bentofile.yaml. This file defines the Bento (the package).76  
    4. Create services/model\_serving/service.py. This defines the production API.37  
       * It will import the model from the local BentoML store 38: model \= bentoml.sklearn.get("best\_biomarker\_model:latest").to\_runner().  
       * It will define a FastAPI endpoint (BentoML uses FastAPI) 28: @svc.api(input=NumpyNdarray(), output=NumpyNdarray()).  
       * The endpoint function will call model.run(input\_data).  
    5. The final step (for a human) will be to run bentoml build and bentoml containerize.  
  * **Execute:** Create the bentofile.yaml and service.py files.

### **EPIC 5: Web UI & Deployment**

**Goal:** Build the user-facing Plotly Dash application and finalize the system.

* **Task 5.1: TDD Develop Core Plotly Dash Components**  
  * **Plan:**  
    1. This TDD will use pytest-dash.  
    2. **TDD (RED):** Write tests/test\_dash\_app.py. The test test\_data\_upload\_component will:  
       * Mock the Dash app layout with the upload component.  
       * Simulate a file upload.  
       * Assert that a dcc.Store component is updated with the (mocked) Muon object.  
    3. **TDD (GREEN):** Implement the dcc.Upload component and its callback. The callback will call the Data Ingest Service API (from 1.4).  
    4. **REPEAT TDD:** Repeat this TDD process for all UI components:  
       * **FS Selection:** Checkboxes that populate a dcc.Store with the job request.  
       * **"Run" Button:** A button whose callback submits the job to the Gateway's /api/v1/run\_benchmark endpoint.  
       * **Results Dashboard:** dcc.Graph components 34 that are populated by a callback that polls the job status and (once 'COMPLETE') fetches results from an API endpoint (e.g., /api/v1/get\_results/\<job\_id\>).  
       * **Exploratory Analysis:** Re-implement OmicSelector 1.0's viz 18 (PCA, Heatmaps, Volcano) using plotly components, all driven by Dash callbacks.  
  * **Execute:** Begin TDD for the Dash UI, starting with the data upload component.  
* **Task 5.2: Finalize FastAPI-Dash Integration**  
  * **Plan:** This is an integration task.  
    1. In services/gateway/main.py, replace the placeholder Dash app (from 1.4) with the *actual* Dash application (from 5.1).  
    2. Run integration tests (from 1.4) to confirm the full, complex Dash app is mounted and served correctly.  
  * **Execute:** Perform the integration.  
* **Task 5.3: TDD Implement User Authentication**  
  * **Plan:**  
    1. We will secure the *entire* platform (API and UI) using FastAPI's OAuth2 support.8  
    2. **TDD (RED):** Write tests/test\_auth.py. The test test\_get\_secured\_endpoint\_unauthenticated will:  
       * Make a request to a new (mocked) /api/v1/me endpoint.  
       * Assert a 401 Unauthorized error.  
    3. **TDD (GREEN):** Implement FastAPI's OAuth2PasswordBearer 8 and add it as a Depends() to the endpoint.  
    4. **TDD (RED/GREEN):** Write tests for the /token endpoint (to get a JWT) and update test\_get\_secured\_endpoint\_authenticated to pass the token and assert a 200 OK.  
    5. Because the Dash app is *mounted inside* FastAPI 34, we can wrap the WSGIMiddleware with an authentication dependency, effectively securing the *entire* UI with the *same* auth system as the API.  
  * **Execute:** Begin TDD for the OAuth2 authentication.  
* **Task 5.4: Finalize Docker-Compose & Kubernetes Deployment**  
  * **Plan:** This is a configuration task.  
    1. Create the final docker-compose.yml.  
    2. This file will define *all* the services: gateway (FastAPI+Dash), data-ingest-worker, fs-engine-worker, benchmarking-worker, mlflow-server, postgres-db (for MLflow), redis (for job queue).  
    3. Create docker-compose.prod.yml which points to production-built images.  
    4. Create a /kubernetes directory with Helm charts or Kustomize files for deployment to a Kubernetes cluster, with horizontal pod autoscalers for the workers.  
  * **Execute:** Create the final docker-compose.yml.

---

## **🧪 TEST-DRIVEN DEVELOPMENT (TDD) STRATEGY**

The TDD strategy 1 is the central mechanism for agentic development.

### **1\. Agentic TDD Workflow (RED-GREEN-REFACTOR)**

You MUST follow this precise workflow for every task 1:

1. **Operator:** "Agent, implement Task X."  
2. **Agent (Plan):** "I will start by writing the failing test tests/test\_X.py."  
3. **Agent (RED):**  
   * Writes("tests/test\_X.py",...)  
   * Run("pytest tests/test\_X.py")  
   * "I confirm the test fails with \[AssertionError/NameError\]."  
   * Run("git add tests/test\_X.py")  
   * Run("git commit \-m 'test(scope): add failing test for X'") 1  
4. **Agent (GREEN):**  
   * "I will now write the minimal code in src/X.py to pass this test."  
   * Writes("src/X.py",...) (MUST NOT modify tests/test\_X.py)  
   * Run("pytest tests/test\_X.py")  
   * "I confirm all tests now pass."  
   * Run("git add src/X.py")  
   * Run("git commit \-m 'feat(scope): implement X'") 1  
5. **Agent (REFACTOR):**  
   * "I will now refactor the code for clarity and performance."  
   * (Refactors code, runs tests again to confirm they still pass, commits changes).

### **2\. Test Harness Definition (Pytest)**

* **Framework:** pytest is the default test runner.  
* **Fixtures:** We will use pytest fixtures extensively.  
  * @pytest.fixture: Create a muon\_mock\_data() fixture that provides a standardized Muon object 24 for all modules to test against. This is critical for integration testing.  
  * @pytest.fixture: Create an api\_client() fixture (using fastapi.testclient.TestClient) for testing API endpoints.  
* **Mocking:** Use unittest.mock to mock external services (e.g., job queues, MLflow server, BentoML registry).38

### **3\. Core Test Domains (A 3-Level Testing Strategy)**

1. **Unit Tests (TDD):** (e.g., "Does build\_psn return a matrix of the correct shape?"). This is the primary RED-GREEN-REFACTOR loop.2  
2. **Integration Tests:** (e.g., "Does the /api/v1/run\_fs endpoint correctly call the gnn\_mogdx module and return a valid signature?"). This tests the connections *between* microservices.  
3. **Scientific Validation Tests:** (e.g., "On the TCGA-BRCA-subset mock data, does the full MOGDx pipeline (EPIC 2\) \+ Benchmarking (EPIC 3\) pipeline correctly identify the 'known\_biomarker\_signature' with a test AUC \> 0.9?"). This is a final "end-to-end" test that validates the *scientific* correctness of the platform, not just its software integrity.

---

## **📚 KEY REPOSITORY FILES & AGENT INSTRUCTIONS**

This section defines the project's "memory" and custom tools for the agent.1

### **1\. File: CLAUDE.md (This document)**

This file is the root instruction set and single source of truth for the project. You MUST re-read and adhere to the **CRITICAL RULES** at the start of every session.

### **2\. File Directory: .claude/commands/**

This directory will store custom slash commands 1 to automate complex, repetitive tasks.

* # **File: .claude/commands/new\_fs\_module.md**    **Creates the skeleton for a new Feature Selection module**     **Usage: /new\_fs\_module module\_name="my\_new\_method" description="A new FS method"**    **Plan:**

  1. Get the module\_name and description from the user ($ARGUMENTS).  
  2. Create the implementation file: omicselector2/fs\_engine/modules/{module\_name}.py  
  3. Create the test file: tests/fs\_engine/test\_{module\_name}.py  
  4. Add a placeholder docstring and function to the implementation file.  
  5. Add a failing placeholder test (e.g., assert False) to the test file.  
  6. Tell the user the skeleton is created and they should now write the real failing test.

Implement the plan.

* # **File: .claude/commands/new\_api\_endpoint.md**    **Creates the TDD skeleton for a new FastAPI endpoint**     **Usage: /new\_api\_endpoint service="gateway" method="POST" path="/api/v1/new\_thing"**    **Plan:**

  1. Get service, method, and path from $ARGUMENTS.  
  2. Open the test file: tests/test\_{service}.py.  
  3. Append a new failing test function:  
     def test\_{method}\_{path.replace('/', '\_')}\_unauthenticated():  
     response \= client.{method.lower()}("{path}")  
     assert response.status\_code \== 401  
  4. Tell the user the failing test for the new endpoint is created.

Implement the plan.

### **3\. File: /docs/DATA\_FORMATS.md**

* **Content:** The detailed Muon 24 object schema, as defined in **Task 1.2**. This is the single source of truth for all data I/O.  
* **Agent Instruction:** You MUST read this file before implementing any function that accepts or returns a data object.

### **4\. File: /docs/API\_SPEC.md**

* **Content:** A human-readable OpenAPI 3.0 specification. As you (the agent) implement new FastAPI endpoints 8 via TDD, you must *also* update this file to document the endpoint, its Pydantic request/response models, and its purpose.  
* **Agent Instruction:** After any commit that adds/modifies an API endpoint, your next step MUST be to update docs/API\_SPEC.md to reflect the change.

#### **Works cited**

1. Claude Code: Best practices for agentic coding \- Anthropic, accessed November 5, 2025, [https://www.anthropic.com/engineering/claude-code-best-practices](https://www.anthropic.com/engineering/claude-code-best-practices)  
2. Agentic TDD \- Nizar's Blog, accessed November 5, 2025, [https://nizar.se/agentic-tdd/](https://nizar.se/agentic-tdd/)  
3. RasmussenLab/MOVE: MOVE (Multi-Omics Variational autoEncoder) for integrating multi-omics data and identifying cross modal associations \- GitHub, accessed November 5, 2025, [https://github.com/RasmussenLab/MOVE](https://github.com/RasmussenLab/MOVE)  
4. Variational autoencoder implemented in tensorflow and pytorch (including inverse autoregressive flow) \- GitHub, accessed November 5, 2025, [https://github.com/jaanli/variational-autoencoder](https://github.com/jaanli/variational-autoencoder)  
5. PyTorch Geometric \- Read the Docs, accessed November 5, 2025, [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)  
6. PyG: Home, accessed November 5, 2025, [https://pyg.org/](https://pyg.org/)  
7. Deep graph convolutional network-based multi-omics integration for cancer driver gene identification | Briefings in Bioinformatics | Oxford Academic, accessed November 5, 2025, [https://academic.oup.com/bib/article/26/4/bbaf364/8214181](https://academic.oup.com/bib/article/26/4/bbaf364/8214181)  
8. FastAPI for Scalable Microservices: Best Practices & Optimisation \- Webandcrafts, accessed November 5, 2025, [https://webandcrafts.com/blog/fastapi-scalable-microservices](https://webandcrafts.com/blog/fastapi-scalable-microservices)  
9. Microservice in Python using FastAPI \- DEV Community, accessed November 5, 2025, [https://dev.to/paurakhsharma/microservice-in-python-using-fastapi-24cc](https://dev.to/paurakhsharma/microservice-in-python-using-fastapi-24cc)  
10. Building a Machine Learning Microservice with FastAPI | NVIDIA Technical Blog, accessed November 5, 2025, [https://developer.nvidia.com/blog/building-a-machine-learning-microservice-with-fastapi/](https://developer.nvidia.com/blog/building-a-machine-learning-microservice-with-fastapi/)  
11. Deep learning–driven multi-omics analysis: enhancing cancer diagnostics and therapeutics, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12392270/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12392270/)  
12. Applications of multi‐omics analysis in human diseases \- PMC \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10390758/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10390758/)  
13. Harnessing AI in Multi-Modal Omics Data Integration: Paving the Path for the Next Frontier in Precision Medicine \- PMC \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11972123/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11972123/)  
14. kstawiski/OmicSelector: OmicSelector \- a package for biomarker selection based on high-throughput experiments. version 1.0.0 from GitHub \- rdrr.io, accessed November 5, 2025, [https://rdrr.io/github/kstawiski/OmicSelector/](https://rdrr.io/github/kstawiski/OmicSelector/)  
15. OmicSelector: Available methods of feature selection and benchmarking., accessed November 5, 2025, [https://kstawiski.github.io/OmicSelector/articles/metody.html](https://kstawiski.github.io/OmicSelector/articles/metody.html)  
16. MoAGL-SA: a multi-omics adaptive integration method with graph learning and self attention for cancer subtype classification \- PMC \- NIH, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11585958/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11585958/)  
17. Integration of multi-omics approaches in exploring intra-tumoral heterogeneity \- PMC \- NIH, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12395700/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12395700/)  
18. OmicSelector \- a package for biomarker selection based on high-throughput experiments., accessed November 5, 2025, [https://kstawiski.github.io/OmicSelector/](https://kstawiski.github.io/OmicSelector/)  
19. OmicSelector: Basic Functionality Tutorial., accessed November 5, 2025, [https://kstawiski.github.io/OmicSelector/articles/Tutorial.html](https://kstawiski.github.io/OmicSelector/articles/Tutorial.html)  
20. OmicSelector \- Environment, docker-based application and R package for biomarker signiture selection (feature selection) & deep learning diagnostic tool development from high-throughput high-throughput omics experiments and other multidimensional datasets. Initially developed for miRNA-seq, RNA-seq and qPCR. \- GitHub, accessed November 5, 2025, [https://github.com/kstawiski/OmicSelector](https://github.com/kstawiski/OmicSelector)  
21. OmicSelector\_OmicSelector • OmicSelector \- GitHub Pages, accessed November 5, 2025, [https://kstawiski.github.io/OmicSelector/reference/OmicSelector\_OmicSelector.html](https://kstawiski.github.io/OmicSelector/reference/OmicSelector_OmicSelector.html)  
22. kstawiski/OmicSelector source listing \- rdrr.io, accessed November 5, 2025, [https://rdrr.io/github/kstawiski/OmicSelector/f/](https://rdrr.io/github/kstawiski/OmicSelector/f/)  
23. raw.githubusercontent.com, accessed November 5, 2025, [https://raw.githubusercontent.com/kstawiski/OmicSelector/master/vignettes/setup.R](https://raw.githubusercontent.com/kstawiski/OmicSelector/master/vignettes/setup.R)  
24. Packages \- scverse, accessed November 5, 2025, [https://scverse.org/packages/](https://scverse.org/packages/)  
25. Comparative Analysis of Multi-Omics Integration Using Graph Neural Networks for Cancer Classification \- PMC \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11928009/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11928009/)  
26. Network-based analyses of multiomics data in biomedicine \- PMC \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12117783/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12117783/)  
27. A technical review of multi-omics data integration methods: from classical statistical to deep generative approaches \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12315550/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315550/)  
28. 10 Must-Know Python Libraries for MLOps in 2025 \- MachineLearningMastery.com, accessed November 5, 2025, [https://machinelearningmastery.com/10-must-know-python-libraries-for-mlops-in-2025/](https://machinelearningmastery.com/10-must-know-python-libraries-for-mlops-in-2025/)  
29. Best Python Frameworks for Scalable Web App Development in 2025 \- Zestminds, accessed November 5, 2025, [https://www.zestminds.com/blog/best-python-frameworks-web-app-2025/](https://www.zestminds.com/blog/best-python-frameworks-web-app-2025/)  
30. Streamlit vs Dash in 2025: Comparing Data App Frameworks | Squadbase Blog, accessed November 5, 2025, [https://www.squadbase.dev/en/blog/streamlit-vs-dash-in-2025-comparing-data-app-frameworks](https://www.squadbase.dev/en/blog/streamlit-vs-dash-in-2025-comparing-data-app-frameworks)  
31. How can I use FASTAPI with Plotly Dash? \- Reddit, accessed November 5, 2025, [https://www.reddit.com/r/FastAPI/comments/gmdkhj/how\_can\_i\_use\_fastapi\_with\_plotly\_dash/](https://www.reddit.com/r/FastAPI/comments/gmdkhj/how_can_i_use_fastapi_with_plotly_dash/)  
32. \[2203.05891\] A Deep Learning Model with Radiomics Analysis Integration for Glioblastoma Post-Resection Survival Prediction \- arXiv, accessed November 5, 2025, [https://arxiv.org/abs/2203.05891](https://arxiv.org/abs/2203.05891)  
33. 27 MLOps Tools for 2025: Key Features & Benefits \- lakeFS, accessed November 5, 2025, [https://lakefs.io/blog/mlops-tools/](https://lakefs.io/blog/mlops-tools/)  
34. Building ML Pipelines with MLflow and BentoML, accessed November 5, 2025, [https://www.bentoml.com/blog/building-ml-pipelines-with-mlflow-and-bentoml](https://www.bentoml.com/blog/building-ml-pipelines-with-mlflow-and-bentoml)  
35. MLflow \- BentoML Documentation, accessed November 5, 2025, [https://docs.bentoml.com/en/latest/examples/mlflow.html](https://docs.bentoml.com/en/latest/examples/mlflow.html)  
36. Welcome to pyradiomics documentation\! — pyradiomics v3.1.0rc2.post5+g6a761c4 documentation, accessed November 5, 2025, [https://pyradiomics.readthedocs.io/](https://pyradiomics.readthedocs.io/)  
37. pyradiomics \- PyPI, accessed November 5, 2025, [https://pypi.org/project/pyradiomics/](https://pypi.org/project/pyradiomics/)  
38. Multi-Omic Graph Diagnosis (MOGDx): a data integration tool to perform classification tasks for heterogeneous diseases | Bioinformatics | Oxford Academic, accessed November 5, 2025, [https://academic.oup.com/bioinformatics/article/40/9/btae523/7739700](https://academic.oup.com/bioinformatics/article/40/9/btae523/7739700)  
39. Training a Variational Autoencoder for Anomaly Detection Using TensorFlow, accessed November 5, 2025, [https://www.analyticsvidhya.com/blog/2023/09/variational-autoencode-for-anomaly-detection-using-tensorflow/](https://www.analyticsvidhya.com/blog/2023/09/variational-autoencode-for-anomaly-detection-using-tensorflow/)  
40. Convolutional Variational Autoencoder | TensorFlow Core, accessed November 5, 2025, [https://www.tensorflow.org/tutorials/generative/cvae](https://www.tensorflow.org/tutorials/generative/cvae)  
41. Welcome to pyradiomics documentation\!, accessed November 5, 2025, [https://pyradiomics.readthedocs.io/en/1.1.0/](https://pyradiomics.readthedocs.io/en/1.1.0/)  
42. Embedding Dash Dashboards in FastAPI Framework (in less than 3 mins) | by Gerard Sho, accessed November 5, 2025, [https://medium.com/@gerardsho/embedding-dash-dashboards-in-fastapi-framework-in-less-than-3-mins-b1bec12eb3](https://medium.com/@gerardsho/embedding-dash-dashboards-in-fastapi-framework-in-less-than-3-mins-b1bec12eb3)  
43. Artificial intelligence integrates multi-omics data for precision stratification and drug resistance prediction in breast cancer \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12463597/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12463597/)  
44. AI-driven multi-omics integration for multi-scale predictive modeling of causal genotype-environment-phenotype relationships \- arXiv, accessed November 5, 2025, [https://arxiv.org/html/2407.06405v1](https://arxiv.org/html/2407.06405v1)  
45. Current Bioinformatics Tools in Precision Oncology \- PMC \- PubMed Central \- NIH, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12238682/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12238682/)  
46. Machine Learning for Multi-Omics Characterization of Blood Cancers: A Systematic Review, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12427946/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12427946/)  
47. Multimodal Data Integration for Oncology in the Era of Deep Neural Networks: A Review, accessed November 5, 2025, [https://arxiv.org/html/2303.06471v3](https://arxiv.org/html/2303.06471v3)  
48. Graph machine learning for integrated multi-omics analysis \- PubMed, accessed November 5, 2025, [https://pubmed.ncbi.nlm.nih.gov/38729996/](https://pubmed.ncbi.nlm.nih.gov/38729996/)  
49. Multi-Omic Graph Diagnosis (MOGDx) : A data integration tool to perform classification tasks for heterogeneous diseases | medRxiv, accessed November 5, 2025, [https://www.medrxiv.org/content/10.1101/2023.07.09.23292410v2.full-text](https://www.medrxiv.org/content/10.1101/2023.07.09.23292410v2.full-text)  
50. Multi-Omic Graph Diagnosis (MOGDx): a data integration tool to perform classification tasks for heterogeneous diseases \- PubMed, accessed November 5, 2025, [https://pubmed.ncbi.nlm.nih.gov/39177104/](https://pubmed.ncbi.nlm.nih.gov/39177104/)  
51. Multi-Omic Graph Diagnosis (MOGDx) : A data integration tool to perform classification tasks for heterogeneous diseases \- medRxiv, accessed November 5, 2025, [https://www.medrxiv.org/content/10.1101/2023.07.09.23292410v1.full.pdf](https://www.medrxiv.org/content/10.1101/2023.07.09.23292410v1.full.pdf)  
52. Multi-Omics Feature Selection to Identify Biomarkers for Hepatocellular Carcinoma \- PMC, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12471784/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12471784/)  
53. Welcome to MOVE's documentation\! — MOVE 1.5.0 documentation, accessed November 5, 2025, [https://move-dl.readthedocs.io/](https://move-dl.readthedocs.io/)  
54. Multimodal integration strategies for clinical application in oncology \- PMC \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12405423/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405423/)  
55. Multimodal Data Integration for Oncology in the Era of Deep Neural Networks: A Review, accessed November 5, 2025, [https://arxiv.org/html/2303.06471v2](https://arxiv.org/html/2303.06471v2)  
56. Deep learning–driven multi-omics analysis: enhancing cancer diagnostics and therapeutics | Briefings in Bioinformatics | Oxford Academic, accessed November 5, 2025, [https://academic.oup.com/bib/article/26/4/bbaf440/8242583](https://academic.oup.com/bib/article/26/4/bbaf440/8242583)  
57. Value of radiomics and deep learning feature fusion models based on dce-mri in distinguishing sinonasal squamous cell carcinoma from lymphoma \- PubMed, accessed November 5, 2025, [https://pubmed.ncbi.nlm.nih.gov/39640273/](https://pubmed.ncbi.nlm.nih.gov/39640273/)  
58. Multimodal Deep Learning Integrating Tumor Radiomics and Mediastinal Adiposity Improves Survival Prediction in Non‐Small Cell Lung Cancer: A Prognostic Modeling Study \- NIH, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12319420/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12319420/)  
59. Joint analysis of paired and unpaired multiomic data with MultiVI \- scvi-tools, accessed November 5, 2025, [https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/multimodal/MultiVI\_tutorial.html](https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/multimodal/MultiVI_tutorial.html)  
60. \[2408.08867\] Quantum Annealing for Enhanced Feature Selection in Single-Cell RNA Sequencing Data Analysis \- arXiv, accessed November 5, 2025, [https://arxiv.org/abs/2408.08867](https://arxiv.org/abs/2408.08867)  
61. Machine learning and multi-omics integration: advancing cardiovascular translational research and clinical practice \- PubMed Central, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11966820/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11966820/)  
62. Patient-specific radiomic feature selection with reconstructed healthy persona of knee MR images \- arXiv, accessed November 5, 2025, [https://arxiv.org/html/2503.13131v1](https://arxiv.org/html/2503.13131v1)  
63. Identifying interactions in omics data for clinical biomarker discovery using symbolic regression | Bioinformatics | Oxford Academic, accessed November 5, 2025, [https://academic.oup.com/bioinformatics/article/38/15/3749/6613136](https://academic.oup.com/bioinformatics/article/38/15/3749/6613136)  
64. The Hallmarks of Predictive Oncology \- PMC \- PubMed Central \- NIH, accessed November 5, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11969157/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11969157/)  
65. Agile epics: definition, examples, and templates \- Atlassian, accessed November 5, 2025, [https://www.atlassian.com/agile/project-management/epics](https://www.atlassian.com/agile/project-management/epics)  
66. How to structure machine learning work effectively | TomTom Blog, accessed November 5, 2025, [https://www.tomtom.com/newsroom/explainers-and-insights/structuring-machine-learning/](https://www.tomtom.com/newsroom/explainers-and-insights/structuring-machine-learning/)  
67. Plan-Driven Agentic Coding. How to Build and Use an Implementation… | by Tim Sylvester, accessed November 5, 2025, [https://medium.com/@TimSylvester/plan-driven-agentic-coding-9ef8b28845fa](https://medium.com/@TimSylvester/plan-driven-agentic-coding-9ef8b28845fa)  
68. Usage — pyradiomics v3.1.0rc2.post5+g6a761c4 documentation, accessed November 5, 2025, [https://pyradiomics.readthedocs.io/en/latest/usage.html](https://pyradiomics.readthedocs.io/en/latest/usage.html)  
69. End-to-End MLOps project with Open Source tools | by Edwin Vivek | Medium, accessed November 5, 2025, [https://medium.com/@nedwinvivek/end-to-end-mlops-project-with-open-source-tools-a241951e68cf](https://medium.com/@nedwinvivek/end-to-end-mlops-project-with-open-source-tools-a241951e68cf)  
70. The Ultimate CLAUDE.md Configuration: Transform Your AI Development Workflow, accessed November 5, 2025, [https://deeplearning.fr/the-ultimate-claude-md-configuration-transform-your-ai-development-workflow/](https://deeplearning.fr/the-ultimate-claude-md-configuration-transform-your-ai-development-workflow/)  
71. CLAUDE MD AI ML Projects · ruvnet/claude-flow Wiki \- GitHub, accessed November 5, 2025, [https://github.com/ruvnet/claude-flow/wiki/CLAUDE-MD-AI-ML-Projects](https://github.com/ruvnet/claude-flow/wiki/CLAUDE-MD-AI-ML-Projects)  
72. CLAUDE MD Templates · ruvnet/claude-flow Wiki \- GitHub, accessed November 5, 2025, [https://github.com/ruvnet/claude-flow/wiki/CLAUDE-MD-Templates](https://github.com/ruvnet/claude-flow/wiki/CLAUDE-MD-Templates)  
73. Highly effective CLAUDE.md for large codebasees : r/ClaudeAI \- Reddit, accessed November 5, 2025, [https://www.reddit.com/r/ClaudeAI/comments/1mgfy4t/highly\_effective\_claudemd\_for\_large\_codebasees/](https://www.reddit.com/r/ClaudeAI/comments/1mgfy4t/highly_effective_claudemd_for_large_codebasees/)