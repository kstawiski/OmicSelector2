"""Data management routes for OmicSelector2 API.

This module provides endpoints for uploading, managing, and accessing datasets.
"""

import uuid
from typing import Optional

try:
    from fastapi import (
        APIRouter,
        Depends,
        File,
        Form,
        HTTPException,
        UploadFile,
        status,
    )
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore
    Depends = None  # type: ignore
    File = None  # type: ignore
    Form = None  # type: ignore
    HTTPException = None  # type: ignore
    UploadFile = None  # type: ignore
    status = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore

try:
    from sqlalchemy.orm import Session

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Session = None  # type: ignore

from omicselector2.api.dependencies import get_current_user
from omicselector2.db import Dataset, DataType, User, get_db
from omicselector2.utils.storage import get_storage_client


# Response Models
class DatasetResponse(BaseModel):
    """Dataset response model.

    Attributes:
        id: Dataset ID
        name: Dataset name
        description: Dataset description
        data_type: Type of omics data
        file_path: S3 path to data file
        n_samples: Number of samples
        n_features: Number of features
        created_at: Creation timestamp
    """

    id: str
    name: str
    description: Optional[str]
    data_type: str
    file_path: Optional[str]
    n_samples: Optional[int]
    n_features: Optional[int]
    created_at: str


class DatasetListResponse(BaseModel):
    """Paginated dataset list response.

    Attributes:
        items: List of datasets
        total: Total number of datasets
        page: Current page number
        size: Page size
    """

    items: list[DatasetResponse]
    total: int
    page: int
    size: int


if not FASTAPI_AVAILABLE:
    router = None  # type: ignore
else:
    router = APIRouter()

    @router.post("/upload", response_model=DatasetResponse, status_code=201)
    async def upload_dataset(
        file: UploadFile = File(...),
        name: str = Form(...),
        data_type: str = Form(...),
        description: Optional[str] = Form(None),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Upload a new dataset.

        Args:
            file: Dataset file to upload
            name: Dataset name
            data_type: Type of omics data
            description: Optional description
            current_user: Current authenticated user
            db: Database session

        Returns:
            Created dataset information

        Raises:
            HTTPException: If upload fails or validation fails
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Validate data type
        try:
            data_type_enum = DataType(data_type)
        except ValueError:
            valid_types = [dt.value for dt in DataType]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data_type. Must be one of: {', '.join(valid_types)}",
            )

        # Generate unique file path
        dataset_id = uuid.uuid4()
        file_extension = file.filename.split(".")[-1] if file.filename else "dat"
        object_name = f"datasets/{dataset_id}/data.{file_extension}"

        # Upload file to S3
        try:
            storage_client = get_storage_client()
            file_content = await file.read()
            s3_path = storage_client.upload_file(
                file_obj=file_content,
                object_name=object_name,
                metadata={
                    "original_filename": file.filename or "unknown",
                    "content_type": file.content_type or "application/octet-stream",
                    "user_id": str(current_user.id),
                },
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file: {str(e)}",
            )

        # TODO: Extract n_samples and n_features from file
        # This would require parsing the file format (CSV, h5ad, etc.)
        n_samples = None
        n_features = None

        # Create dataset record
        new_dataset = Dataset(
            id=dataset_id,
            name=name,
            description=description,
            data_type=data_type_enum,
            file_path=s3_path,
            n_samples=n_samples,
            n_features=n_features,
            metadata={
                "original_filename": file.filename,
                "file_size_bytes": len(file_content),
                "content_type": file.content_type,
            },
            owner_id=current_user.id,
        )

        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        return DatasetResponse(
            id=str(new_dataset.id),
            name=new_dataset.name,
            description=new_dataset.description,
            data_type=new_dataset.data_type.value,
            file_path=new_dataset.file_path,
            n_samples=new_dataset.n_samples,
            n_features=new_dataset.n_features,
            created_at=new_dataset.created_at.isoformat(),
        )

    @router.get("/{dataset_id}", response_model=DatasetResponse)
    async def get_dataset(
        dataset_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Get dataset by ID.

        Args:
            dataset_id: Dataset UUID
            current_user: Current authenticated user
            db: Database session

        Returns:
            Dataset information

        Raises:
            HTTPException: If dataset not found or access denied
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )

        # Check access (owner or admin)
        if dataset.owner_id != current_user.id and current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        return DatasetResponse(
            id=str(dataset.id),
            name=dataset.name,
            description=dataset.description,
            data_type=dataset.data_type.value,
            file_path=dataset.file_path,
            n_samples=dataset.n_samples,
            n_features=dataset.n_features,
            created_at=dataset.created_at.isoformat(),
        )

    @router.get("/", response_model=DatasetListResponse)
    async def list_datasets(
        page: int = 1,
        size: int = 50,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """List datasets for current user.

        Args:
            page: Page number (default 1)
            size: Page size (default 50, max 100)
            current_user: Current authenticated user
            db: Database session

        Returns:
            Paginated list of datasets

        Raises:
            HTTPException: If database unavailable
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Validate pagination
        if page < 1:
            page = 1
        if size < 1 or size > 100:
            size = 50

        # Query datasets (owner or admin sees all)
        query = db.query(Dataset)
        if current_user.role.value != "admin":
            query = query.filter(Dataset.owner_id == current_user.id)

        # Get total count
        total = query.count()

        # Get paginated results
        datasets = query.offset((page - 1) * size).limit(size).all()

        # Convert to response models
        items = [
            DatasetResponse(
                id=str(d.id),
                name=d.name,
                description=d.description,
                data_type=d.data_type.value,
                file_path=d.file_path,
                n_samples=d.n_samples,
                n_features=d.n_features,
                created_at=d.created_at.isoformat(),
            )
            for d in datasets
        ]

        return DatasetListResponse(
            items=items,
            total=total,
            page=page,
            size=size,
        )

    @router.delete("/{dataset_id}", status_code=204)
    async def delete_dataset(
        dataset_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Delete a dataset.

        Args:
            dataset_id: Dataset UUID
            current_user: Current authenticated user
            db: Database session

        Raises:
            HTTPException: If dataset not found or access denied
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )

        # Check access (owner or admin)
        if dataset.owner_id != current_user.id and current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        # Delete file from S3
        if dataset.file_path:
            try:
                storage_client = get_storage_client()
                # Extract object name from s3://bucket/path
                object_name = dataset.file_path.split(f"s3://{storage_client.bucket_name}/")[1]
                storage_client.delete_file(object_name)
            except Exception:
                # Continue even if S3 deletion fails
                pass

        # Delete database record
        db.delete(dataset)
        db.commit()

        return None
