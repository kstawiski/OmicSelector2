"""Result database model.

This module defines the Result model for storing job results.
"""

import uuid

try:
    from sqlalchemy import Column, DateTime, ForeignKey, String
    from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
    from sqlalchemy.orm import relationship
    from sqlalchemy.sql import func

    from omicselector2.db.database import Base

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = object  # type: ignore


if SQLALCHEMY_AVAILABLE:

    class Result(Base):  # type: ignore
        """Result model for storing job results.

        Attributes:
            id: Unique result identifier (UUID)
            job_id: ID of job that produced this result
            selected_features: List of selected features (for feature selection)
            metrics: Performance metrics (JSONB)
            artifacts_path: Path to result artifacts in S3/MinIO
            created_at: Result creation timestamp

        Relationships:
            job: Job that produced this result
        """

        __tablename__ = "results"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
        job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)

        # Result data
        selected_features = Column(ARRAY(String), nullable=True)  # For feature selection jobs
        metrics = Column(JSONB, nullable=True)  # Performance metrics
        artifacts_path = Column(String(500), nullable=True)  # S3/MinIO path

        # Timestamps
        created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

        # Relationships
        job = relationship(
            "Job",
            back_populates="result",
            foreign_keys="[Result.job_id]",
            primaryjoin="Result.job_id == Job.id",
        )

        def __repr__(self) -> str:
            """String representation of Result.

            Returns:
                Result representation string
            """
            n_features = len(self.selected_features) if self.selected_features else None
            return f"<Result(id={self.id}, job_id={self.job_id}, n_features={n_features})>"

else:
    # Stub class when SQLAlchemy not available
    class Result:  # type: ignore
        """Stub Result class when SQLAlchemy is not installed."""

        def __init__(self, *args, **kwargs):  # type: ignore
            raise ImportError(
                "SQLAlchemy is required for Result model. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )
