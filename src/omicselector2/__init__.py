"""OmicSelector2: Next-generation platform for multi-omic biomarker discovery.

OmicSelector2 modernizes biomarker discovery by transitioning from OmicSelector 1.0's
R-based automated feature selection to a Python-native platform enabling guided
multi-modal integration with state-of-the-art deep learning methods.

Core Features:
    - Automated benchmarking: Test multiple feature selection methods
    - Multi-omics integration: Native support for scRNA-seq, bulk RNA-seq, WES, radiomics
    - SOTA models: Graph Neural Networks (GNNs), VAEs, attention mechanisms
    - Clinical translatability: Interpretable models (SHAP), stability selection

Target Users:
    Biomedical researchers, bioinformaticians, oncology data scientists working with
    multi-omic data for cancer biomarker discovery.

Clinical Focus:
    Urological cancers (bladder, prostate) and gastrointestinal cancers (rectal),
    with extensibility to pan-cancer applications.

Examples:
    >>> from omicselector2 import __version__
    >>> print(__version__)
    '0.1.0'

"""

__version__ = "0.1.0"
__author__ = "OmicSelector Team"
__email__ = "omicselector@example.com"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
