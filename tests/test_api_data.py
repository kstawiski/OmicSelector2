"""
Tests for data management API endpoints.

This module tests the data upload and management endpoints:
- POST /api/v1/data/upload - Upload dataset
- GET /api/v1/data/{dataset_id} - Get dataset
- GET /api/v1/data/ - List datasets
- DELETE /api/v1/data/{dataset_id} - Delete dataset
"""

import pytest
import io
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from datetime import datetime

from omicselector2.api.routes.data import router
from omicselector2.db import Dataset, DataType, User, UserRole
from omicselector2.utils.storage import StorageClient

# Create a FastAPI test client
from fastapi import FastAPI

app = FastAPI()
app.include_router(router, prefix="/api/v1/data", tags=["data"])


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock()


@pytest.fixture
def mock_storage_client():
    """Mock storage client."""
    mock_client = Mock(spec=StorageClient)
    mock_client.upload_file = Mock(return_value="s3://omicselector2/datasets/test-id/data.csv")
    mock_client.delete_file = Mock()
    return mock_client


@pytest.fixture
def test_user():
    """Create test user."""
    user = Mock(spec=User)
    user.id = uuid4()
    user.email = "test@example.com"
    user.username = "testuser"
    user.role = UserRole.USER
    user.is_active = True
    return user


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


class TestDatasetUpload:
    """Test dataset upload endpoint."""

    def test_upload_dataset_success(self, test_client, mock_db, mock_storage_client, test_user):
        """Test successful dataset upload."""
        # Create test CSV file
        csv_content = b"gene_id,sample1,sample2\nGENE1,10.5,20.3\nGENE2,15.2,25.1"
        test_file = io.BytesIO(csv_content)

        # Mock dependencies
        from omicselector2.api.dependencies import get_current_user
        from omicselector2.utils.storage import get_storage_client

        async def override_get_current_user():
            return test_user

        def override_get_storage_client():
            return mock_storage_client

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        # Mock database operations
        mock_db.add = Mock()
        mock_db.commit = Mock()

        def mock_refresh(dataset):
            dataset.id = uuid4()
            dataset.created_at = datetime.utcnow()

        mock_db.refresh = mock_refresh

        # Patch get_storage_client
        with patch("omicselector2.api.routes.data.get_storage_client", override_get_storage_client):
            response = test_client.post(
                "/api/v1/data/upload",
                data={
                    "name": "Test Dataset",
                    "data_type": "bulk_rna_seq",
                    "description": "Test description"
                },
                files={"file": ("test_data.csv", test_file, "text/csv")}
            )

        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Test Dataset"
        assert data["data_type"] == "bulk_rna_seq"
        assert data["description"] == "Test description"
        assert "id" in data
        assert "created_at" in data

    def test_upload_dataset_invalid_data_type(self, test_client, test_user):
        """Test upload with invalid data type."""
        csv_content = b"data"
        test_file = io.BytesIO(csv_content)

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user

        response = test_client.post(
            "/api/v1/data/upload",
            data={
                "name": "Test Dataset",
                "data_type": "invalid_type",  # Invalid
            },
            files={"file": ("test.csv", test_file, "text/csv")}
        )

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "data_type" in response.json()["detail"].lower()

    def test_upload_dataset_storage_failure(self, test_client, test_user, mock_db):
        """Test upload when storage fails."""
        csv_content = b"data"
        test_file = io.BytesIO(csv_content)

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        def override_get_storage_client():
            mock_client = Mock()
            mock_client.upload_file = Mock(side_effect=Exception("Storage error"))
            return mock_client

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        with patch("omicselector2.api.routes.data.get_storage_client", override_get_storage_client):
            response = test_client.post(
                "/api/v1/data/upload",
                data={
                    "name": "Test Dataset",
                    "data_type": "bulk_rna_seq",
                },
                files={"file": ("test.csv", test_file, "text/csv")}
            )

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "upload" in response.json()["detail"].lower()

    def test_upload_dataset_requires_authentication(self, test_client):
        """Test upload without authentication."""
        csv_content = b"data"
        test_file = io.BytesIO(csv_content)

        response = test_client.post(
            "/api/v1/data/upload",
            data={
                "name": "Test Dataset",
                "data_type": "bulk_rna_seq",
            },
            files={"file": ("test.csv", test_file, "text/csv")}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestGetDataset:
    """Test get dataset endpoint."""

    def test_get_dataset_success(self, test_client, mock_db, test_user):
        """Test getting dataset by ID."""
        dataset_id = str(uuid4())

        # Mock dataset
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.id = dataset_id
        mock_dataset.name = "Test Dataset"
        mock_dataset.description = "Test description"
        mock_dataset.data_type = DataType.BULK_RNA_SEQ
        mock_dataset.file_path = "s3://bucket/path"
        mock_dataset.n_samples = 100
        mock_dataset.n_features = 20000
        mock_dataset.owner_id = test_user.id
        mock_dataset.created_at = datetime.utcnow()

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/data/{dataset_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "Test Dataset"
        assert data["data_type"] == "bulk_rna_seq"

    def test_get_dataset_not_found(self, test_client, mock_db, test_user):
        """Test getting non-existent dataset."""
        dataset_id = str(uuid4())
        mock_db.query.return_value.filter.return_value.first.return_value = None

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/data/{dataset_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_get_dataset_access_denied(self, test_client, mock_db, test_user):
        """Test getting dataset owned by another user."""
        dataset_id = str(uuid4())
        other_user_id = uuid4()

        mock_dataset = Mock(spec=Dataset)
        mock_dataset.id = dataset_id
        mock_dataset.owner_id = other_user_id  # Different owner
        mock_dataset.name = "Other User Dataset"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/data/{dataset_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "access denied" in response.json()["detail"].lower()


class TestListDatasets:
    """Test list datasets endpoint."""

    def test_list_datasets_success(self, test_client, mock_db, test_user):
        """Test listing datasets."""
        # Create mock datasets
        mock_datasets = []
        for i in range(3):
            dataset = Mock(spec=Dataset)
            dataset.id = uuid4()
            dataset.name = f"Dataset {i+1}"
            dataset.description = f"Description {i+1}"
            dataset.data_type = DataType.BULK_RNA_SEQ
            dataset.file_path = f"s3://bucket/dataset{i}"
            dataset.n_samples = 100 + i
            dataset.n_features = 20000 + i
            dataset.owner_id = test_user.id
            dataset.created_at = datetime.utcnow()
            mock_datasets.append(dataset)

        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_datasets

        mock_db.query.return_value = mock_query

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get("/api/v1/data/?page=1&size=10")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        assert data["page"] == 1
        assert data["size"] == 10

    def test_list_datasets_pagination(self, test_client, mock_db, test_user):
        """Test dataset list pagination."""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 50
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_db.query.return_value = mock_query

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get("/api/v1/data/?page=2&size=20")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 50
        assert data["page"] == 2
        assert data["size"] == 20


class TestDeleteDataset:
    """Test delete dataset endpoint."""

    def test_delete_dataset_success(self, test_client, mock_db, mock_storage_client, test_user):
        """Test successful dataset deletion."""
        dataset_id = str(uuid4())

        mock_dataset = Mock(spec=Dataset)
        mock_dataset.id = dataset_id
        mock_dataset.owner_id = test_user.id
        mock_dataset.file_path = "s3://omicselector2/datasets/test/data.csv"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset
        mock_db.delete = Mock()
        mock_db.commit = Mock()

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        def override_get_storage_client():
            return mock_storage_client

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        with patch("omicselector2.api.routes.data.get_storage_client", override_get_storage_client):
            response = test_client.delete(f"/api/v1/data/{dataset_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_storage_client.delete_file.assert_called_once()
        mock_db.delete.assert_called_once_with(mock_dataset)
        mock_db.commit.assert_called_once()

    def test_delete_dataset_not_found(self, test_client, mock_db, test_user):
        """Test deleting non-existent dataset."""
        dataset_id = str(uuid4())
        mock_db.query.return_value.filter.return_value.first.return_value = None

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.delete(f"/api/v1/data/{dataset_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_dataset_access_denied(self, test_client, mock_db, test_user):
        """Test deleting dataset owned by another user."""
        dataset_id = str(uuid4())
        other_user_id = uuid4()

        mock_dataset = Mock(spec=Dataset)
        mock_dataset.id = dataset_id
        mock_dataset.owner_id = other_user_id  # Different owner

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.delete(f"/api/v1/data/{dataset_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_403_FORBIDDEN
        mock_db.delete.assert_not_called()


class TestDataEndpointsIntegration:
    """Integration tests for data endpoints."""

    def test_full_dataset_lifecycle(self, test_client, mock_db, mock_storage_client, test_user):
        """Test complete workflow: upload -> get -> list -> delete."""
        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        def override_get_storage_client():
            return mock_storage_client

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        # Setup mocks
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.delete = Mock()

        dataset_id = uuid4()

        def mock_refresh(dataset):
            dataset.id = dataset_id
            dataset.created_at = datetime.utcnow()

        mock_db.refresh = mock_refresh

        # Step 1: Upload dataset
        csv_content = b"gene,value\nGENE1,10"
        test_file = io.BytesIO(csv_content)

        with patch("omicselector2.api.routes.data.get_storage_client", override_get_storage_client):
            upload_response = test_client.post(
                "/api/v1/data/upload",
                data={"name": "Lifecycle Test", "data_type": "bulk_rna_seq"},
                files={"file": ("test.csv", test_file, "text/csv")}
            )

        assert upload_response.status_code == status.HTTP_201_CREATED
        uploaded_id = upload_response.json()["id"]

        # Step 2: Get dataset
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.id = uploaded_id
        mock_dataset.name = "Lifecycle Test"
        mock_dataset.data_type = DataType.BULK_RNA_SEQ
        mock_dataset.owner_id = test_user.id
        mock_dataset.file_path = "s3://bucket/path"
        mock_dataset.created_at = datetime.utcnow()
        mock_dataset.n_samples = None
        mock_dataset.n_features = None
        mock_dataset.description = None

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        get_response = test_client.get(f"/api/v1/data/{uploaded_id}")
        assert get_response.status_code == status.HTTP_200_OK

        # Step 3: Delete dataset
        with patch("omicselector2.api.routes.data.get_storage_client", override_get_storage_client):
            delete_response = test_client.delete(f"/api/v1/data/{uploaded_id}")

        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        app.dependency_overrides.clear()
