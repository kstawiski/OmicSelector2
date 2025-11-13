"""Dataset database model.

This module defines the Dataset model for storing biomarker data information.
"""

import uuid
from enum import Enum

try:
    from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
    from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
    from sqlalchemy.orm import relationship
    from sqlalchemy.sql import func

    from omicselector2.db.database import Base

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = object  # type: ignore


class DataType(str, Enum):
    """Dataset data type enumeration.

    Attributes:
        BULK_RNA_SEQ: Bulk RNA sequencing data
        SINGLE_CELL_RNA_SEQ: Single-cell RNA sequencing data
        METHYLATION: DNA methylation data
        CNV: Copy number variation data
        PROTEOMICS: Proteomics data
        METABOLOMICS: Metabolomics data
        RADIOMICS: Radiomics features
        CLINICAL: Clinical data
    """

    BULK_RNA_SEQ = "bulk_rna_seq"
    SINGLE_CELL_RNA_SEQ = "single_cell_rna_seq"
    METHYLATION = "methylation"
    CNV = "copy_number"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    RADIOMICS = "radiomics"
    CLINICAL = "clinical"


if SQLALCHEMY_AVAILABLE:

    class Dataset(Base):  # type: ignore
        """Dataset model for storing data information.

        Attributes:
            id: Unique dataset identifier (UUID)
            name: Dataset name
            description: Dataset description
            data_type: Type of omics data
            file_path: Path to data file in S3/MinIO
            n_samples: Number of samples in dataset
            n_features: Number of features in dataset
            metadata_json: Additional metadata (JSONB) - maps to 'metadata' column
            owner_id: ID of user who owns this dataset
            created_at: Dataset creation timestamp

        Relationships:
            owner: User who owns this dataset
            jobs: Jobs using this dataset
        """

        __tablename__ = "datasets"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
        name = Column(String(255), nullable=False)
        description = Column(Text, nullable=True)
        data_type = Column(ENUM(DataType, name="data_type", create_type=True), nullable=False)
        file_path = Column(String(500), nullable=True)  # S3/MinIO path
        n_samples = Column(Integer, nullable=True)
        n_features = Column(Integer, nullable=True)
        metadata_json = Column("metadata", JSONB, nullable=True)  # Flexible metadata storage

        # Foreign keys
        owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

        # Timestamps
        created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

        # Relationships
        owner = relationship("User", back_populates="datasets")
        jobs = relationship("Job", back_populates="dataset")

        def __repr__(self) -> str:
            """String representation of Dataset.

            Returns:
                Dataset representation string
            """
            return (
                f"<Dataset(id={self.id}, name={self.name}, "
                f"data_type={self.data_type}, n_samples={self.n_samples})>"
            )

else:
    # Stub class when SQLAlchemy not available
    class Dataset:  # type: ignore
        """Stub Dataset class when SQLAlchemy is not installed."""

        def __init__(self, *args, **kwargs):  # type: ignore
            raise ImportError(
                "SQLAlchemy is required for Dataset model. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )
