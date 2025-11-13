"""Integration tests for authentication flow.

Tests:
- User registration
- User login
- Token validation
- Role-based access control (RBAC)
- Get current user
- Logout
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestAuthenticationFlow:
    """Test suite for authentication workflows."""

    def test_register_new_user(self, client: TestClient):
        """Test user registration with valid data."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "securepassword123",
                "full_name": "New User",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert data["full_name"] == "New User"
        assert data["role"] == "user"
        assert data["is_active"] is True
        assert "id" in data
        assert "hashed_password" not in data  # Should not expose password

    def test_register_duplicate_email(self, client: TestClient, test_user):
        """Test registration with duplicate email fails."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",  # Already exists
                "username": "differentuser",
                "password": "password123",
                "full_name": "Different User",
            },
        )

        assert response.status_code == 400
        assert "email already registered" in response.json()["detail"].lower()

    def test_register_duplicate_username(self, client: TestClient, test_user):
        """Test registration with duplicate username fails."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "different@example.com",
                "username": "testuser",  # Already exists
                "password": "password123",
                "full_name": "Different User",
            },
        )

        assert response.status_code == 400
        assert "username already taken" in response.json()["detail"].lower()

    def test_register_invalid_email(self, client: TestClient):
        """Test registration with invalid email format fails."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "username": "newuser",
                "password": "password123",
                "full_name": "New User",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_login_success(self, client: TestClient, test_user):
        """Test successful login with valid credentials."""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "testuser", "password": "testpassword"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 0

    def test_login_wrong_password(self, client: TestClient, test_user):
        """Test login with incorrect password fails."""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "testuser", "password": "wrongpassword"},
        )

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client: TestClient):
        """Test login with non-existent user fails."""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent", "password": "password123"},
        )

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_get_current_user(self, client: TestClient, auth_headers):
        """Test getting current user with valid token."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert data["role"] == "user"

    def test_get_current_user_no_token(self, client: TestClient):
        """Test getting current user without token fails."""
        response = client.get("/api/v1/auth/me")

        assert response.status_code == 401

    def test_get_current_user_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token fails."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"},
        )

        assert response.status_code == 401

    def test_logout(self, client: TestClient, auth_headers):
        """Test logout endpoint."""
        response = client.post("/api/v1/auth/logout", headers=auth_headers)

        assert response.status_code == 200
        assert "successfully" in response.json()["message"].lower()


@pytest.mark.integration
class TestRoleBasedAccessControl:
    """Test suite for role-based access control."""

    def test_user_role_assigned_on_registration(self, client: TestClient):
        """Test that new users are assigned USER role by default."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "roletest@example.com",
                "username": "roletest",
                "password": "password123",
                "full_name": "Role Test",
            },
        )

        assert response.status_code == 201
        assert response.json()["role"] == "user"

    def test_researcher_can_access_data_endpoints(
        self, client: TestClient, researcher_auth_headers
    ):
        """Test that researcher role can access data endpoints."""
        response = client.get("/api/v1/data/", headers=researcher_auth_headers)

        # Should return 200 (may be empty list, but accessible)
        assert response.status_code == 200

    def test_admin_can_access_all_endpoints(
        self, client: TestClient, admin_auth_headers
    ):
        """Test that admin role can access all endpoints."""
        response = client.get("/api/v1/data/", headers=admin_auth_headers)

        # Should return 200
        assert response.status_code == 200

    def test_token_contains_user_role(self, client: TestClient, test_researcher):
        """Test that JWT token contains user role information."""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "researcher", "password": "researcherpassword"},
        )

        assert response.status_code == 200
        token = response.json()["access_token"]
        assert len(token) > 0

        # Get user info to verify role
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["role"] == "researcher"


@pytest.mark.integration
class TestPasswordSecurity:
    """Test suite for password security."""

    def test_password_is_hashed(self, client: TestClient):
        """Test that passwords are hashed, not stored in plain text."""
        # Register user
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "hashtest@example.com",
                "username": "hashtest",
                "password": "mypassword123",
                "full_name": "Hash Test",
            },
        )

        # Login to get user data
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "hashtest", "password": "mypassword123"},
        )

        assert response.status_code == 200

        # Get user info - should not include password
        token = response.json()["access_token"]
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        data = response.json()
        assert "password" not in data
        assert "hashed_password" not in data

    def test_password_validation_enforces_minimum_length(self, client: TestClient):
        """Test that password validation enforces minimum length."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "shortpw@example.com",
                "username": "shortpw",
                "password": "123",  # Too short
                "full_name": "Short Password",
            },
        )

        # Should fail validation (422 or 400)
        assert response.status_code in [400, 422]


@pytest.mark.integration
class TestTokenExpiration:
    """Test suite for token expiration."""

    def test_token_has_expiration(self, client: TestClient, test_user):
        """Test that JWT tokens have expiration time."""
        from omicselector2.utils.security import decode_access_token

        # Login to get token
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "testuser", "password": "testpassword"},
        )

        token = response.json()["access_token"]

        # Decode token
        payload = decode_access_token(token)

        assert payload is not None
        assert "exp" in payload  # Expiration time
        assert "sub" in payload  # Subject (user ID)
