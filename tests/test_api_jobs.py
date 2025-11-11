"""
Tests for job management API endpoints.

This module tests the job submission and management endpoints:
- POST /api/v1/jobs/ - Create job
- GET /api/v1/jobs/{job_id} - Get job status
- GET /api/v1/jobs/ - List jobs
- DELETE /api/v1/jobs/{job_id} - Cancel job
- GET /api/v1/jobs/{job_id}/result - Get job result

Following TDD: These tests are written FIRST and MUST fail initially.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status, FastAPI
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from datetime import datetime

from omicselector2.db import Job, JobType, JobStatus, Dataset, DataType, User, UserRole
from omicselector2.api.routes.jobs import router

# Create FastAPI app with jobs router
app = FastAPI()
app.include_router(router, prefix="/api/v1/jobs", tags=["jobs"])


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock()


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
def test_dataset(test_user):
    """Create test dataset."""
    dataset = Mock(spec=Dataset)
    dataset.id = uuid4()
    dataset.name = "Test Dataset"
    dataset.data_type = DataType.BULK_RNA_SEQ
    dataset.owner_id = test_user.id
    dataset.file_path = "s3://bucket/data.csv"
    dataset.n_samples = 100
    dataset.n_features = 20000
    return dataset


class TestCreateJob:
    """Test job creation endpoint."""

    def test_create_feature_selection_job_success(self, test_client, mock_db, test_user, test_dataset):
        """Test successful feature selection job creation."""
        # This test will fail until we implement the endpoint

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        # Mock dataset query
        mock_db.query.return_value.filter.return_value.first.return_value = test_dataset

        # Mock job creation
        mock_db.add = Mock()
        mock_db.commit = Mock()

        job_id = uuid4()

        def mock_refresh(job):
            job.id = job_id
            job.created_at = datetime.utcnow()
            job.status = JobStatus.PENDING

        mock_db.refresh = mock_refresh

        # Mock Celery task
        with patch('omicselector2.api.routes.jobs.feature_selection_task') as mock_task:
            mock_task.delay.return_value.id = "celery-task-id-123"

            response = test_client.post(
                "/api/v1/jobs/",
                json={
                    "job_type": "feature_selection",
                    "dataset_id": str(test_dataset.id),
                    "config": {
                        "methods": ["lasso", "elastic_net"],
                        "n_features": 100,
                        "cv_folds": 5
                    }
                }
            )

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["job_type"] == "feature_selection"
        assert data["status"] == "pending"
        assert data["dataset_id"] == str(test_dataset.id)
        assert "id" in data
        assert "celery_task_id" in data

    def test_create_job_invalid_job_type(self, test_client, test_user):
        """Test job creation with invalid job type."""

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user

        response = test_client.post(
            "/api/v1/jobs/",
            json={
                "job_type": "invalid_type",
                "dataset_id": str(uuid4()),
                "config": {}
            }
        )

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "job_type" in response.json()["detail"].lower()

    def test_create_job_dataset_not_found(self, test_client, mock_db, test_user):
        """Test job creation with non-existent dataset."""

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        # Mock dataset not found
        mock_db.query.return_value.filter.return_value.first.return_value = None

        response = test_client.post(
            "/api/v1/jobs/",
            json={
                "job_type": "feature_selection",
                "dataset_id": str(uuid4()),
                "config": {}
            }
        )

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "dataset" in response.json()["detail"].lower()

    def test_create_job_dataset_access_denied(self, test_client, mock_db, test_user):
        """Test job creation with dataset owned by another user."""

        other_user_id = uuid4()
        dataset = Mock(spec=Dataset)
        dataset.id = uuid4()
        dataset.owner_id = other_user_id  # Different owner

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        mock_db.query.return_value.filter.return_value.first.return_value = dataset

        response = test_client.post(
            "/api/v1/jobs/",
            json={
                "job_type": "feature_selection",
                "dataset_id": str(dataset.id),
                "config": {}
            }
        )

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_create_job_requires_authentication(self, test_client):
        """Test job creation without authentication."""

        response = test_client.post(
            "/api/v1/jobs/",
            json={
                "job_type": "feature_selection",
                "dataset_id": str(uuid4()),
                "config": {}
            }
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestGetJob:
    """Test get job endpoint."""

    def test_get_job_success(self, test_client, mock_db, test_user):
        """Test getting job by ID."""

        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.job_type = JobType.FEATURE_SELECTION
        mock_job.status = JobStatus.RUNNING
        mock_job.user_id = test_user.id
        mock_job.dataset_id = uuid4()
        mock_job.config = {"methods": ["lasso"]}
        mock_job.celery_task_id = "task-123"
        mock_job.created_at = datetime.utcnow()
        mock_job.started_at = datetime.utcnow()
        mock_job.completed_at = None
        mock_job.error_message = None
        mock_job.result_id = None

        mock_db.query.return_value.filter.return_value.first.return_value = mock_job

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/jobs/{job_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(job_id)
        assert data["job_type"] == "feature_selection"
        assert data["status"] == "running"

    def test_get_job_not_found(self, test_client, mock_db, test_user):
        """Test getting non-existent job."""

        mock_db.query.return_value.filter.return_value.first.return_value = None

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/jobs/{uuid4()}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_job_access_denied(self, test_client, mock_db, test_user):
        """Test getting job created by another user."""

        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.user_id = uuid4()  # Different user

        mock_db.query.return_value.filter.return_value.first.return_value = mock_job

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/jobs/{job_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestListJobs:
    """Test list jobs endpoint."""

    def test_list_jobs_success(self, test_client, mock_db, test_user):
        """Test listing user's jobs."""

        # Create mock jobs
        mock_jobs = []
        for i in range(3):
            job = Mock(spec=Job)
            job.id = uuid4()
            job.job_type = JobType.FEATURE_SELECTION
            job.status = JobStatus.COMPLETED if i < 2 else JobStatus.RUNNING
            job.user_id = test_user.id
            job.dataset_id = uuid4()
            job.created_at = datetime.utcnow()
            job.result_id = uuid4() if i < 2 else None
            mock_jobs.append(job)

        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_jobs

        mock_db.query.return_value = mock_query

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get("/api/v1/jobs/?page=1&size=10")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        assert data["page"] == 1

    def test_list_jobs_filter_by_status(self, test_client, mock_db, test_user):
        """Test filtering jobs by status."""

        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_db.query.return_value = mock_query

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get("/api/v1/jobs/?status=running")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK


class TestCancelJob:
    """Test cancel job endpoint."""

    def test_cancel_job_success(self, test_client, mock_db, test_user):
        """Test successful job cancellation."""

        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.user_id = test_user.id
        mock_job.status = JobStatus.RUNNING
        mock_job.celery_task_id = "task-123"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_job
        mock_db.commit = Mock()

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        with patch('omicselector2.api.routes.jobs.celery_app') as mock_celery:
            mock_celery.control.revoke = Mock()

            response = test_client.delete(f"/api/v1/jobs/{job_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        assert mock_job.status == JobStatus.CANCELLED

    def test_cancel_completed_job_fails(self, test_client, mock_db, test_user):
        """Test cancelling already completed job."""

        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.user_id = test_user.id
        mock_job.status = JobStatus.COMPLETED

        mock_db.query.return_value.filter.return_value.first.return_value = mock_job

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.delete(f"/api/v1/jobs/{job_id}")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot cancel" in response.json()["detail"].lower()


class TestGetJobResult:
    """Test get job result endpoint."""

    def test_get_job_result_success(self, test_client, mock_db, test_user):
        """Test getting result for completed job."""

        job_id = uuid4()
        result_id = uuid4()

        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.user_id = test_user.id
        mock_job.status = JobStatus.COMPLETED
        mock_job.result_id = result_id

        from omicselector2.db import Result
        mock_result = Mock(spec=Result)
        mock_result.id = result_id
        mock_result.job_id = job_id
        mock_result.selected_features = ["GENE1", "GENE2", "GENE3"]
        mock_result.metrics = {"auc": 0.85, "accuracy": 0.82}
        mock_result.artifacts_path = "s3://bucket/results/"

        # Mock query chain
        def mock_query_side_effect(model):
            if model == Job:
                mock_q = Mock()
                mock_q.filter.return_value.first.return_value = mock_job
                return mock_q
            elif model == Result:
                mock_q = Mock()
                mock_q.filter.return_value.first.return_value = mock_result
                return mock_q

        mock_db.query.side_effect = mock_query_side_effect

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/jobs/{job_id}/result")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "selected_features" in data
        assert "metrics" in data

    def test_get_result_job_not_completed(self, test_client, mock_db, test_user):
        """Test getting result for job that's still running."""

        job_id = uuid4()
        mock_job = Mock(spec=Job)
        mock_job.id = job_id
        mock_job.user_id = test_user.id
        mock_job.status = JobStatus.RUNNING
        mock_job.result_id = None

        mock_db.query.return_value.filter.return_value.first.return_value = mock_job

        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return test_user

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[lambda: None] = lambda: mock_db

        response = test_client.get(f"/api/v1/jobs/{job_id}/result")

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not completed" in response.json()["detail"].lower()
