"""Job database model.

This module defines the Job model for tracking analysis jobs.
"""

import uuid
from enum import Enum

try:
    from sqlalchemy import Column, DateTime, ForeignKey, String, Text
    from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
    from sqlalchemy.orm import relationship
    from sqlalchemy.sql import func

    from omicselector2.db.database import Base

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = object  # type: ignore


class JobType(str, Enum):
    """Job type enumeration.

    Attributes:
        FEATURE_SELECTION: Feature selection job
        MODEL_TRAINING: Model training job
        HYPERPARAMETER_TUNING: Hyperparameter optimization job
        BENCHMARKING: Signature benchmarking job
        PREPROCESSING: Data preprocessing job
        CROSS_VALIDATION: Cross-validation job
    """

    FEATURE_SELECTION = "feature_selection"
    MODEL_TRAINING = "model_training"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    BENCHMARKING = "benchmarking"
    PREPROCESSING = "preprocessing"
    CROSS_VALIDATION = "cross_validation"


class JobStatus(str, Enum):
    """Job status enumeration.

    Attributes:
        PENDING: Job created but not started
        RUNNING: Job currently executing
        COMPLETED: Job finished successfully
        FAILED: Job failed with error
        CANCELLED: Job cancelled by user
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


if SQLALCHEMY_AVAILABLE:

    class Job(Base):  # type: ignore
        """Job model for tracking analysis jobs.

        Attributes:
            id: Unique job identifier (UUID)
            job_type: Type of job (feature selection, training, etc.)
            status: Current job status
            celery_task_id: Celery task ID for tracking
            config: Job configuration (JSONB)
            user_id: ID of user who created job
            dataset_id: ID of dataset being analyzed
            started_at: Job start timestamp
            completed_at: Job completion timestamp
            error_message: Error message if job failed
            result_id: ID of result object (if completed)
            created_at: Job creation timestamp

        Relationships:
            user: User who created this job
            dataset: Dataset being analyzed
            result: Job result (if completed)
        """

        __tablename__ = "jobs"

        id = Column(
            UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
        )
        job_type = Column(
            ENUM(JobType, name="job_type", create_type=True), nullable=False
        )
        status = Column(
            ENUM(JobStatus, name="job_status", create_type=True),
            default=JobStatus.PENDING,
            nullable=False,
            index=True,
        )
        celery_task_id = Column(String(255), unique=True, nullable=True, index=True)

        # Configuration
        config = Column(JSONB, nullable=False)  # Job parameters

        # Foreign keys
        user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
        dataset_id = Column(
            UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False
        )

        # Execution details
        started_at = Column(DateTime(timezone=True), nullable=True)
        completed_at = Column(DateTime(timezone=True), nullable=True)
        error_message = Column(Text, nullable=True)

        # Result reference
        result_id = Column(
            UUID(as_uuid=True), ForeignKey("results.id"), nullable=True
        )

        # Timestamps
        created_at = Column(
            DateTime(timezone=True), server_default=func.now(), nullable=False
        )

        # Relationships
        user = relationship("User", back_populates="jobs")
        dataset = relationship("Dataset", back_populates="jobs")
        result = relationship(
            "Result",
            back_populates="job",
            uselist=False,
            foreign_keys="[Result.job_id]",
            primaryjoin="Job.id == Result.job_id"
        )

        def __repr__(self) -> str:
            """String representation of Job.

            Returns:
                Job representation string
            """
            return (
                f"<Job(id={self.id}, type={self.job_type}, "
                f"status={self.status}, dataset_id={self.dataset_id})>"
            )

else:
    # Stub class when SQLAlchemy not available
    class Job:  # type: ignore
        """Stub Job class when SQLAlchemy is not installed."""

        def __init__(self, *args, **kwargs):  # type: ignore
            raise ImportError(
                "SQLAlchemy is required for Job model. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )
