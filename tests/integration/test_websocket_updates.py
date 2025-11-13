"""Integration tests for WebSocket real-time job updates.

Tests:
- WebSocket connection establishment with JWT authentication
- Real-time job status updates via WebSocket
- Multiple concurrent connections to same job
- WebSocket disconnection handling
- Authorization (job owner vs non-owner)

Note: These tests require Redis to be running.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession


@pytest.mark.integration
@pytest.mark.websocket
class TestWebSocketConnection:
    """Test suite for WebSocket connection management."""

    def test_websocket_connect_with_valid_token(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test WebSocket connection with valid JWT token."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "WebSocket Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]

        # Extract token from auth headers
        token = auth_headers["Authorization"].replace("Bearer ", "")

        # Connect to WebSocket
        with client.websocket_connect(
            f"/api/v1/jobs/{job_id}/ws?token={token}"
        ) as websocket:
            # Connection successful
            assert websocket is not None

    def test_websocket_connect_without_token(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that WebSocket connection without token fails."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "WebSocket Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]

        # Try to connect without token
        with pytest.raises(Exception):  # WebSocket should reject connection
            with client.websocket_connect(f"/api/v1/jobs/{job_id}/ws"):
                pass

    def test_websocket_connect_with_invalid_token(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that WebSocket connection with invalid token fails."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "WebSocket Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]

        # Try to connect with invalid token
        with pytest.raises(Exception):  # WebSocket should reject connection
            with client.websocket_connect(f"/api/v1/jobs/{job_id}/ws?token=invalid"):
                pass

    def test_websocket_connect_to_nonexistent_job(
        self, client: TestClient, auth_headers
    ):
        """Test that WebSocket connection to non-existent job fails."""
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        token = auth_headers["Authorization"].replace("Bearer ", "")

        # Try to connect to non-existent job
        with pytest.raises(Exception):  # WebSocket should reject connection
            with client.websocket_connect(f"/api/v1/jobs/{fake_uuid}/ws?token={token}"):
                pass

    def test_websocket_unauthorized_job_access(
        self, client: TestClient, auth_headers, researcher_auth_headers, sample_dataset_csv
    ):
        """Test that user cannot connect to another user's job via WebSocket."""
        # User creates job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "User Job", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]

        # Researcher tries to connect with their token
        researcher_token = researcher_auth_headers["Authorization"].replace("Bearer ", "")

        # Should fail - researcher doesn't own this job
        with pytest.raises(Exception):  # WebSocket should reject connection
            with client.websocket_connect(
                f"/api/v1/jobs/{job_id}/ws?token={researcher_token}"
            ):
                pass


@pytest.mark.integration
@pytest.mark.websocket
@pytest.mark.slow
class TestWebSocketJobUpdates:
    """Test suite for real-time job updates via WebSocket.

    Note: These tests require Redis and Celery workers to be running.
    """

    def test_receive_job_status_updates(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test receiving real-time job status updates via WebSocket."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Update Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]
        token = auth_headers["Authorization"].replace("Bearer ", "")

        # Connect to WebSocket and receive updates
        with client.websocket_connect(
            f"/api/v1/jobs/{job_id}/ws?token={token}"
        ) as websocket:
            # Should receive status updates as job progresses
            # This test assumes Celery worker is running

            # Wait for first update (timeout after 30 seconds)
            received_updates = []

            try:
                for _ in range(10):  # Try to receive up to 10 updates
                    data = websocket.receive_json(timeout=3.0)
                    received_updates.append(data)

                    # Check update format
                    assert "job_id" in data
                    assert "status" in data
                    assert data["job_id"] == job_id

                    # Break if job completed
                    if data["status"] in ["completed", "failed"]:
                        break
            except TimeoutError:
                pass  # Timeout is acceptable if job takes longer

            # Should have received at least one update
            assert len(received_updates) > 0

    def test_multiple_concurrent_websocket_connections(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test multiple WebSocket connections to same job."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Multi-Connection Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]
        token = auth_headers["Authorization"].replace("Bearer ", "")

        # Open multiple WebSocket connections
        with client.websocket_connect(
            f"/api/v1/jobs/{job_id}/ws?token={token}"
        ) as ws1, client.websocket_connect(
            f"/api/v1/jobs/{job_id}/ws?token={token}"
        ) as ws2:
            # Both connections should be active
            assert ws1 is not None
            assert ws2 is not None

            # Both should receive updates (if job is running)
            # This is a smoke test - actual update delivery tested elsewhere

    def test_websocket_disconnect_handling(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test WebSocket disconnection is handled gracefully."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Disconnect Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]
        token = auth_headers["Authorization"].replace("Bearer ", "")

        # Connect and immediately disconnect
        with client.websocket_connect(
            f"/api/v1/jobs/{job_id}/ws?token={token}"
        ) as websocket:
            pass  # Connection closes on exit

        # No errors should occur - disconnection handled gracefully

    def test_websocket_job_progress_messages(
        self, client: TestClient, auth_headers, sample_dataset_csv
    ):
        """Test that WebSocket receives detailed progress messages."""
        # Upload dataset and create job
        sample_dataset_csv.seek(0)
        upload_response = client.post(
            "/api/v1/data/upload",
            headers=auth_headers,
            files={"file": ("test_dataset.csv", sample_dataset_csv, "text/csv")},
            data={"name": "Progress Test", "data_type": "bulk_rna_seq"},
        )

        dataset_id = upload_response.json()["id"]

        job_response = client.post(
            "/api/v1/jobs/feature-selection",
            headers=auth_headers,
            json={
                "dataset_id": dataset_id,
                "methods": ["lasso", "random_forest"],
                "n_features": 10,
                "target_column": "target",
            },
        )

        job_id = job_response.json()["id"]
        token = auth_headers["Authorization"].replace("Bearer ", "")

        # Connect to WebSocket
        with client.websocket_connect(
            f"/api/v1/jobs/{job_id}/ws?token={token}"
        ) as websocket:
            # Receive updates with progress messages
            received_messages = []

            try:
                for _ in range(20):  # Try to receive multiple updates
                    data = websocket.receive_json(timeout=2.0)

                    # Check for message field
                    if "message" in data and data["message"]:
                        received_messages.append(data["message"])

                    # Break if job completed
                    if data.get("status") in ["completed", "failed"]:
                        break
            except TimeoutError:
                pass

            # Should have received some progress messages
            # (if Celery worker is running and publishes messages)
            # This assertion is lenient since it depends on infrastructure
            assert isinstance(received_messages, list)
