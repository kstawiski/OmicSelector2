"""Job management routes for OmicSelector2 API.

This module provides endpoints for creating, managing, and monitoring analysis jobs.
"""

import base64
import binascii
import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, status
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore
    Depends = None  # type: ignore
    HTTPException = None  # type: ignore
    Query = None  # type: ignore
    status = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore

try:
    from sqlalchemy import and_, or_
    from sqlalchemy.orm import Session

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Session = None  # type: ignore

from omicselector2.api.dependencies import get_current_user
from omicselector2.db import (
    Dataset,
    Job,
    JobStatus,
    JobType,
    Result,
    User,
    UserRole,
    get_db,
)
from omicselector2.tasks import celery_app


# Request/Response Models
class JobCreateRequest(BaseModel):
    """Job creation request model.

    Attributes:
        job_type: Type of job to create
        dataset_id: Dataset to analyze
        config: Job-specific configuration (JSONB)
    """

    job_type: str = Field(..., description="Job type (feature_selection, model_training, etc.)")
    dataset_id: str = Field(..., description="Dataset UUID")
    config: dict = Field(default_factory=dict, description="Job configuration")


class JobResponse(BaseModel):
    """Job response model.

    Attributes:
        id: Job ID
        job_type: Job type
        status: Current job status
        dataset_id: Dataset being analyzed
        celery_task_id: Celery task ID
        config: Job configuration
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        error_message: Error message if failed
        result_id: Result ID if completed
    """

    id: str
    job_type: str
    status: str
    dataset_id: str
    celery_task_id: Optional[str]
    config: dict
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    result_id: Optional[str]


class JobListResponse(BaseModel):
    """Paginated or cursor-based job list response."""

    items: list[JobResponse]
    size: int
    total: Optional[int] = None
    page: Optional[int] = None
    has_next: bool = False
    next_cursor: Optional[str] = None


class ResultResponse(BaseModel):
    """Job result response model.

    Attributes:
        id: Result ID
        job_id: Job ID
        selected_features: List of selected features
        metrics: Performance metrics
        artifacts_path: Path to result artifacts
        created_at: Creation timestamp
    """

    id: str
    job_id: str
    selected_features: Optional[list[str]]
    metrics: Optional[dict]
    artifacts_path: Optional[str]
    created_at: str


def _encode_cursor(job: Job) -> str:
    """Create a cursor token from a job instance."""

    payload = {
        "created_at": job.created_at.isoformat(),
        "id": str(job.id),
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode(
        "utf-8"
    )


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    """Decode a cursor token into timestamp and UUID components."""

    try:
        raw = base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8")
        payload = json.loads(raw)
        created_at_str = payload["created_at"]
        job_id_str = payload["id"]
    except (binascii.Error, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Invalid cursor payload") from exc

    created_at = datetime.fromisoformat(created_at_str)
    job_id = UUID(job_id_str)
    return created_at, job_id


if not FASTAPI_AVAILABLE:
    router = None  # type: ignore
else:
    router = APIRouter()

    @router.post("/", response_model=JobResponse, status_code=201)
    async def create_job(
        request: JobCreateRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Create a new analysis job.

        Args:
            request: Job creation request
            current_user: Current authenticated user
            db: Database session

        Returns:
            Created job information

        Raises:
            HTTPException: If validation fails or job creation fails
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Validate job type
        try:
            job_type_enum = JobType(request.job_type)
        except ValueError:
            valid_types = [jt.value for jt in JobType]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job_type. Must be one of: {', '.join(valid_types)}",
            )

        # Get and validate dataset
        try:
            dataset_uuid = UUID(request.dataset_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid dataset_id format",
            )

        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )

        # Check access (owner or admin)
        if dataset.owner_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to dataset",
            )

        # Create job record
        new_job = Job(
            job_type=job_type_enum,
            status=JobStatus.PENDING,
            user_id=current_user.id,
            dataset_id=dataset_uuid,
            config=request.config,
        )

        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        # Submit Celery task
        celery_task_id = None
        try:
            if celery_app:
                # Import task dynamically to avoid circular imports
                from omicselector2.tasks.feature_selection import feature_selection_task
                from omicselector2.tasks.model_training import model_training_task

                if job_type_enum == JobType.FEATURE_SELECTION:
                    task = feature_selection_task.delay(
                        str(new_job.id), str(dataset_uuid), request.config
                    )
                elif job_type_enum == JobType.MODEL_TRAINING:
                    task = model_training_task.delay(
                        str(new_job.id), str(dataset_uuid), request.config
                    )
                else:
                    # For other job types, use a generic placeholder
                    logger.warning(f"No Celery task defined for job type: {job_type_enum}")
                    task = None

                if task:
                    celery_task_id = task.id
                    new_job.celery_task_id = celery_task_id
                    db.commit()
        except Exception as e:
            logger.error(f"Failed to submit Celery task: {str(e)}")
            new_job.status = JobStatus.FAILED
            new_job.error_message = "Failed to queue job for execution"
            new_job.completed_at = datetime.utcnow()
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to queue job for execution. Please try again later.",
            ) from e

        return JobResponse(
            id=str(new_job.id),
            job_type=new_job.job_type.value,
            status=new_job.status.value,
            dataset_id=str(new_job.dataset_id),
            celery_task_id=new_job.celery_task_id,
            config=new_job.config,
            created_at=new_job.created_at.isoformat(),
            started_at=new_job.started_at.isoformat() if new_job.started_at else None,
            completed_at=new_job.completed_at.isoformat() if new_job.completed_at else None,
            error_message=new_job.error_message,
            result_id=str(new_job.result_id) if new_job.result_id else None,
        )

    @router.get("/{job_id}", response_model=JobResponse)
    async def get_job(
        job_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Get job status and details.

        Args:
            job_id: Job UUID
            current_user: Current authenticated user
            db: Database session

        Returns:
            Job information

        Raises:
            HTTPException: If job not found or access denied
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Parse job ID
        try:
            job_uuid = UUID(job_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job_id format",
            )

        # Get job
        job = db.query(Job).filter(Job.id == job_uuid).first()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )

        # Check access (owner or admin)
        if job.user_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        return JobResponse(
            id=str(job.id),
            job_type=job.job_type.value,
            status=job.status.value,
            dataset_id=str(job.dataset_id),
            celery_task_id=job.celery_task_id,
            config=job.config,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error_message=job.error_message,
            result_id=str(job.result_id) if job.result_id else None,
        )

    @router.get("/", response_model=JobListResponse)
    async def list_jobs(
        page: int = Query(1, ge=1, description="Page number (used when cursor is not provided)"),
        size: int = Query(50, ge=1, le=100, description="Page size"),
        status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
        cursor: Optional[str] = Query(
            None,
            description=(
                "Cursor token for keyset pagination. "
                "Provides better performance on large tables."
            ),
        ),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """List jobs for current user.

        Args:
            page: Page number
            size: Page size
            status_filter: Optional status filter
            current_user: Current authenticated user
            db: Database session

        Returns:
            Paginated list of jobs

        Raises:
            HTTPException: If database unavailable
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Build query
        query = db.query(Job)

        # Filter by user (admin sees all)
        if current_user.role != UserRole.ADMIN:
            query = query.filter(Job.user_id == current_user.id)

        # Filter by status if provided
        if status_filter:
            try:
                status_enum = JobStatus(status_filter)
                query = query.filter(Job.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status. Valid values: {', '.join([s.value for s in JobStatus])}",
                )

        query = query.order_by(Job.created_at.desc(), Job.id.desc())

        has_next = False
        next_cursor = None
        total: Optional[int] = None
        page_value: Optional[int] = None

        if cursor:
            try:
                cursor_created_at, cursor_id = _decode_cursor(cursor)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid cursor value",
                )

            query = query.filter(
                or_(
                    Job.created_at < cursor_created_at,
                    and_(Job.created_at == cursor_created_at, Job.id < cursor_id),
                )
            )

            jobs = query.limit(size + 1).all()
            if len(jobs) > size:
                has_next = True
                next_cursor = _encode_cursor(jobs[-1])
                jobs = jobs[:-1]
        else:
            jobs = query.offset((page - 1) * size).limit(size + 1).all()
            if len(jobs) > size:
                has_next = True
                next_cursor = _encode_cursor(jobs[-1])
                jobs = jobs[:-1]
            page_value = page
            if not has_next and page == 1:
                total = len(jobs)

        items = [
            JobResponse(
                id=str(j.id),
                job_type=j.job_type.value,
                status=j.status.value,
                dataset_id=str(j.dataset_id),
                celery_task_id=j.celery_task_id,
                config=j.config,
                created_at=j.created_at.isoformat(),
                started_at=j.started_at.isoformat() if j.started_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
                error_message=j.error_message,
                result_id=str(j.result_id) if j.result_id else None,
            )
            for j in jobs
        ]

        return JobListResponse(
            items=items,
            total=total,
            page=page_value,
            size=size,
            has_next=has_next,
            next_cursor=next_cursor,
        )

    @router.delete("/{job_id}")
    async def cancel_job(
        job_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Cancel a running job.

        Args:
            job_id: Job UUID
            current_user: Current authenticated user
            db: Database session

        Returns:
            Success message

        Raises:
            HTTPException: If job not found, access denied, or cannot be cancelled
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Parse job ID
        try:
            job_uuid = UUID(job_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job_id format",
            )

        # Get job
        job = db.query(Job).filter(Job.id == job_uuid).first()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )

        # Check access (owner or admin)
        if job.user_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        # Check if job can be cancelled
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in {job.status.value} state",
            )

        # Revoke Celery task if exists
        if job.celery_task_id and celery_app:
            try:
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            except Exception as e:
                logger.error(f"Failed to revoke Celery task {job.celery_task_id}: {str(e)}")

        # Update job status
        job.status = JobStatus.CANCELLED
        db.commit()

        return {"message": "Job cancelled successfully", "job_id": str(job.id)}

    @router.get("/{job_id}/result", response_model=ResultResponse)
    async def get_job_result(
        job_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Get result for completed job.

        Args:
            job_id: Job UUID
            current_user: Current authenticated user
            db: Database session

        Returns:
            Job result

        Raises:
            HTTPException: If job not completed or result not found
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Parse job ID
        try:
            job_uuid = UUID(job_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job_id format",
            )

        # Get job
        job = db.query(Job).filter(Job.id == job_uuid).first()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )

        # Check access (owner or admin)
        if job.user_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        # Check if job is completed
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is not completed (current status: {job.status.value})",
            )

        # Get result
        if not job.result_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Result not found for completed job",
            )

        result = db.query(Result).filter(Result.id == job.result_id).first()

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Result not found",
            )

        return ResultResponse(
            id=str(result.id),
            job_id=str(result.job_id),
            selected_features=result.selected_features,
            metrics=result.metrics,
            artifacts_path=result.artifacts_path,
            created_at=result.created_at.isoformat(),
        )
