"""
Tests for authentication API endpoints.

This module tests the authentication endpoints:
- POST /api/v1/auth/register - User registration
- POST /api/v1/auth/login - User authentication
- GET /api/v1/auth/me - Get current user info

Following TDD: These tests are written to verify the implementation.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import patch, Mock
from uuid import uuid4

from omicselector2.api.routes.auth import router
from omicselector2.db import User, UserRole
from omicselector2.utils.security import hash_password, create_access_token

# Create a FastAPI test client
from fastapi import FastAPI

app = FastAPI()
app.include_router(router, prefix="/api/v1/auth", tags=["auth"])


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock()


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


class TestUserRegistration:
    """Test user registration endpoint."""

    def test_register_new_user_success(self, test_client, mock_db):
        """Test successful user registration."""
        # Mock database to return None (no existing user)
        mock_db.query.return_value.filter.return_value.first.return_value = None

        # Mock commit and refresh
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Mock the newly created user
        new_user = Mock()
        new_user.id = uuid4()
        new_user.email = "newuser@example.com"
        new_user.username = "newuser"
        new_user.full_name = "New User"
        new_user.role = UserRole.USER

        def mock_add(user):
            # Simulate database add
            pass

        def mock_refresh_user(user):
            # Copy mock user data
            user.id = new_user.id
            user.email = new_user.email
            user.username = new_user.username
            user.full_name = new_user.full_name
            user.role = new_user.role

        mock_db.add = mock_add
        mock_db.refresh = mock_refresh_user

        # Override get_db dependency
        app.dependency_overrides[lambda: None] = lambda: mock_db

        registration_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "securepassword123",
            "full_name": "New User"
        }

        response = test_client.post("/api/v1/auth/register", json=registration_data)

        # Clean up
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "password" not in data  # Password should not be in response

    def test_register_duplicate_email(self, test_client, mock_db):
        """Test registration with existing email."""
        # Mock existing user
        existing_user = Mock()
        existing_user.email = "existing@example.com"
        existing_user.username = "otheruser"

        mock_db.query.return_value.filter.return_value.first.return_value = existing_user

        app.dependency_overrides[lambda: None] = lambda: mock_db

        registration_data = {
            "email": "existing@example.com",
            "username": "newuser",
            "password": "securepassword123"
        }

        response = test_client.post("/api/v1/auth/register", json=registration_data)

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "email" in response.json()["detail"].lower()

    def test_register_duplicate_username(self, test_client, mock_db):
        """Test registration with existing username."""
        # Mock existing user
        existing_user = Mock()
        existing_user.email = "other@example.com"
        existing_user.username = "existinguser"

        mock_db.query.return_value.filter.return_value.first.return_value = existing_user

        app.dependency_overrides[lambda: None] = lambda: mock_db

        registration_data = {
            "email": "new@example.com",
            "username": "existinguser",
            "password": "securepassword123"
        }

        response = test_client.post("/api/v1/auth/register", json=registration_data)

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "username" in response.json()["detail"].lower()

    def test_register_invalid_email(self, test_client):
        """Test registration with invalid email format."""
        registration_data = {
            "email": "invalid-email",
            "username": "newuser",
            "password": "securepassword123"
        }

        response = test_client.post("/api/v1/auth/register", json=registration_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_register_short_password(self, test_client):
        """Test registration with password too short."""
        registration_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "short"  # Less than 8 characters
        }

        response = test_client.post("/api/v1/auth/register", json=registration_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestUserLogin:
    """Test user login endpoint."""

    def test_login_success_with_username(self, test_client, mock_db):
        """Test successful login with username."""
        # Mock user
        user_id = str(uuid4())
        mock_user = Mock()
        mock_user.id = user_id
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.hashed_password = hash_password("testpassword123")
        mock_user.is_active = True

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        app.dependency_overrides[lambda: None] = lambda: mock_db

        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }

        response = test_client.post("/api/v1/auth/login", json=login_data)

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_login_success_with_email(self, test_client, mock_db):
        """Test successful login with email."""
        user_id = str(uuid4())
        mock_user = Mock()
        mock_user.id = user_id
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.hashed_password = hash_password("testpassword123")
        mock_user.is_active = True

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        app.dependency_overrides[lambda: None] = lambda: mock_db

        login_data = {
            "username": "test@example.com",  # Using email
            "password": "testpassword123"
        }

        response = test_client.post("/api/v1/auth/login", json=login_data)

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK

    def test_login_user_not_found(self, test_client, mock_db):
        """Test login with non-existent user."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        app.dependency_overrides[lambda: None] = lambda: mock_db

        login_data = {
            "username": "nonexistent",
            "password": "testpassword123"
        }

        response = test_client.post("/api/v1/auth/login", json=login_data)

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "username or password" in response.json()["detail"].lower()

    def test_login_incorrect_password(self, test_client, mock_db):
        """Test login with incorrect password."""
        mock_user = Mock()
        mock_user.id = str(uuid4())
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.hashed_password = hash_password("correctpassword")
        mock_user.is_active = True

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        app.dependency_overrides[lambda: None] = lambda: mock_db

        login_data = {
            "username": "testuser",
            "password": "wrongpassword"
        }

        response = test_client.post("/api/v1/auth/login", json=login_data)

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_inactive_user(self, test_client, mock_db):
        """Test login with inactive user account."""
        mock_user = Mock()
        mock_user.id = str(uuid4())
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.hashed_password = hash_password("testpassword123")
        mock_user.is_active = False  # Inactive

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        app.dependency_overrides[lambda: None] = lambda: mock_db

        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }

        response = test_client.post("/api/v1/auth/login", json=login_data)

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "inactive" in response.json()["detail"].lower()


class TestGetCurrentUser:
    """Test get current user endpoint."""

    def test_get_me_success(self, test_client):
        """Test getting current user info with valid token."""
        # Create a real user mock
        user_id = str(uuid4())
        token = create_access_token({"sub": user_id})

        # Override get_current_user dependency
        from omicselector2.api.dependencies import get_current_user

        mock_user = Mock()
        mock_user.id = user_id
        mock_user.email = "test@example.com"
        mock_user.username = "testuser"
        mock_user.full_name = "Test User"
        mock_user.role = UserRole.USER

        async def override_get_current_user():
            return mock_user

        app.dependency_overrides[get_current_user] = override_get_current_user

        response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )

        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert data["full_name"] == "Test User"
        assert data["role"] == "user"

    def test_get_me_no_token(self, test_client):
        """Test getting current user without authentication token."""
        response = test_client.get("/api/v1/auth/me")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_get_me_invalid_token(self, test_client):
        """Test getting current user with invalid token."""
        response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"}
        )

        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]


class TestAuthenticationIntegration:
    """Integration tests for full authentication flow."""

    def test_full_registration_login_flow(self, test_client, mock_db):
        """Test complete flow: register -> login -> access protected endpoint."""
        # Step 1: Register
        user_id = uuid4()
        new_user = Mock()
        new_user.id = user_id
        new_user.email = "integration@example.com"
        new_user.username = "integrationuser"
        new_user.full_name = "Integration Test"
        new_user.role = UserRole.USER
        new_user.hashed_password = hash_password("integrationpass123")
        new_user.is_active = True

        # Mock for registration (no existing user)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_db.add = Mock()
        mock_db.commit = Mock()

        def mock_refresh(user):
            user.id = new_user.id
            user.email = new_user.email
            user.username = new_user.username
            user.full_name = new_user.full_name
            user.role = new_user.role

        mock_db.refresh = mock_refresh

        app.dependency_overrides[lambda: None] = lambda: mock_db

        registration_response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "integration@example.com",
                "username": "integrationuser",
                "password": "integrationpass123",
                "full_name": "Integration Test"
            }
        )

        assert registration_response.status_code == status.HTTP_201_CREATED

        # Step 2: Login
        # Mock for login (user exists)
        mock_db.query.return_value.filter.return_value.first.return_value = new_user

        login_response = test_client.post(
            "/api/v1/auth/login",
            json={
                "username": "integrationuser",
                "password": "integrationpass123"
            }
        )

        assert login_response.status_code == status.HTTP_200_OK
        token = login_response.json()["access_token"]

        # Step 3: Access protected endpoint
        from omicselector2.api.dependencies import get_current_user

        async def override_get_current_user():
            return new_user

        app.dependency_overrides[get_current_user] = override_get_current_user

        me_response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )

        app.dependency_overrides.clear()

        assert me_response.status_code == status.HTTP_200_OK
        user_data = me_response.json()
        assert user_data["email"] == "integration@example.com"
        assert user_data["username"] == "integrationuser"
