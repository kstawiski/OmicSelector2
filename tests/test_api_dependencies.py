"""
Tests for FastAPI dependencies (authentication, database session injection).

This module tests the dependency injection functions used in FastAPI:
- get_db_session(): Database session management
- get_current_user(): JWT token validation and user extraction
- require_role(): Role-based access control (RBAC)

Following TDD: These tests are written FIRST and MUST fail initially.
"""

import pytest
from fastapi import HTTPException, status
from unittest.mock import Mock, MagicMock, patch, create_autospec
from datetime import datetime, timedelta
from uuid import uuid4

from omicselector2.api.dependencies import (
    get_db_session,
    get_current_user,
    require_role,
    require_researcher,
    require_admin,
)
from omicselector2.db import User, UserRole
from omicselector2.utils.security import create_access_token


class TestGetDBSession:
    """Test database session dependency."""

    def test_get_db_session_yields_session(self):
        """Test that get_db_session yields a database session."""
        # This should yield a session object
        session_gen = get_db_session()

        # Get the session from generator (synchronous)
        session = next(session_gen)

        # Session should not be None
        assert session is not None

        # Clean up
        try:
            next(session_gen)
        except StopIteration:
            pass  # Expected

    def test_get_db_session_closes_on_exit(self):
        """Test that get_db_session closes the session after use."""
        session_gen = get_db_session()

        # Get session
        session = next(session_gen)

        # Mock the close method
        session.close = Mock()

        # Trigger cleanup
        try:
            next(session_gen)
        except StopIteration:
            pass

        # Close should have been called
        session.close.assert_called_once()


class TestGetCurrentUser:
    """Test get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_with_valid_token(self):
        """Test that get_current_user returns user for valid token."""
        # Create a test user
        user_id = str(uuid4())
        user_email = "test@example.com"

        # Create valid JWT token
        token_data = {"sub": user_id, "email": user_email}
        token = create_access_token(token_data)

        # Mock HTTP authorization credentials
        mock_credentials = Mock()
        mock_credentials.credentials = token

        # Mock database session
        mock_db = Mock()
        mock_user = create_autospec(User, instance=True)
        mock_user.id = user_id
        mock_user.email = user_email
        mock_user.username = "testuser"
        mock_user.role = UserRole.USER
        mock_user.is_active = True

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        # Call get_current_user directly with parameters (bypassing Depends())
        user = await get_current_user(credentials=mock_credentials, db=mock_db)

        # Should return the user
        assert user is not None
        assert user.email == user_email

    @pytest.mark.asyncio
    async def test_get_current_user_with_invalid_token(self):
        """Test that get_current_user raises HTTPException for invalid token."""
        invalid_token = "invalid.jwt.token"

        mock_credentials = Mock()
        mock_credentials.credentials = invalid_token
        mock_db = Mock()

        # Should raise HTTPException with 401 status
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=mock_credentials, db=mock_db)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert (
            "Invalid" in str(exc_info.value.detail)
            or "not validate" in str(exc_info.value.detail).lower()
        )

    @pytest.mark.asyncio
    async def test_get_current_user_with_expired_token(self):
        """Test that get_current_user raises HTTPException for expired token."""
        user_id = str(uuid4())

        # Create expired token
        token_data = {"sub": user_id}
        expired_token = create_access_token(
            token_data, expires_delta=timedelta(seconds=-1)  # Already expired
        )

        mock_credentials = Mock()
        mock_credentials.credentials = expired_token
        mock_db = Mock()

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=mock_credentials, db=mock_db)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_current_user_user_not_found_in_db(self):
        """Test that get_current_user raises HTTPException if user not in database."""
        user_id = str(uuid4())

        # Create valid token
        token_data = {"sub": user_id}
        token = create_access_token(token_data)

        mock_credentials = Mock()
        mock_credentials.credentials = token

        # Mock database - user not found
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        # Should raise HTTPException with 401
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=mock_credentials, db=mock_db)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "not found" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_get_current_user_inactive_user(self):
        """Test that get_current_user raises HTTPException for inactive user."""
        user_id = str(uuid4())

        # Create valid token
        token_data = {"sub": user_id}
        token = create_access_token(token_data)

        mock_credentials = Mock()
        mock_credentials.credentials = token

        # Mock database - inactive user
        mock_db = Mock()
        mock_user = create_autospec(User, instance=True)
        mock_user.id = user_id
        mock_user.email = "test@example.com"
        mock_user.username = "testuser"
        mock_user.role = UserRole.USER
        mock_user.is_active = False  # INACTIVE

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=mock_credentials, db=mock_db)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "inactive" in str(exc_info.value.detail).lower()


class TestRequireRole:
    """Test role-based access control dependencies."""

    def test_require_role_user_has_required_role(self):
        """Test that require_role allows access when user has the role."""
        # Create user with USER role
        user = create_autospec(User, instance=True)
        user.id = uuid4()
        user.email = "test@example.com"
        user.username = "testuser"
        user.role = UserRole.USER
        user.is_active = True

        # Require USER role
        role_checker = require_role(UserRole.USER)

        # Should return the user (no exception)
        result = role_checker(current_user=user)
        assert result == user

    def test_require_role_user_lacks_required_role(self):
        """Test that require_role raises HTTPException when user lacks role."""
        # Create user with USER role
        user = create_autospec(User, instance=True)
        user.id = uuid4()
        user.email = "test@example.com"
        user.username = "testuser"
        user.role = UserRole.USER
        user.is_active = True

        # Require ADMIN role
        role_checker = require_role(UserRole.ADMIN)

        # Should raise HTTPException with 403 Forbidden
        with pytest.raises(HTTPException) as exc_info:
            role_checker(current_user=user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "permission" in str(exc_info.value.detail).lower()

    def test_require_role_admin_has_all_access(self):
        """Test that ADMIN role has access to all endpoints."""
        # Create admin user
        admin = create_autospec(User, instance=True)
        admin.id = uuid4()
        admin.email = "admin@example.com"
        admin.username = "admin"
        admin.role = UserRole.ADMIN
        admin.is_active = True

        # Require USER role (lower than ADMIN)
        role_checker = require_role(UserRole.USER)

        # Admin should have access
        result = role_checker(current_user=admin)
        assert result == admin

    def test_require_researcher_allows_researcher(self):
        """Test require_researcher allows RESEARCHER role."""
        researcher = create_autospec(User, instance=True)
        researcher.id = uuid4()
        researcher.email = "researcher@example.com"
        researcher.username = "researcher"
        researcher.role = UserRole.RESEARCHER
        researcher.is_active = True

        # Should allow access
        result = require_researcher(current_user=researcher)
        assert result == researcher

    def test_require_researcher_blocks_user(self):
        """Test require_researcher blocks USER role."""
        user = create_autospec(User, instance=True)
        user.id = uuid4()
        user.email = "user@example.com"
        user.username = "user"
        user.role = UserRole.USER
        user.is_active = True

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            require_researcher(current_user=user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    def test_require_admin_allows_admin(self):
        """Test require_admin allows ADMIN role."""
        admin = create_autospec(User, instance=True)
        admin.id = uuid4()
        admin.email = "admin@example.com"
        admin.username = "admin"
        admin.role = UserRole.ADMIN
        admin.is_active = True

        # Should allow access
        result = require_admin(current_user=admin)
        assert result == admin

    def test_require_admin_blocks_researcher(self):
        """Test require_admin blocks RESEARCHER role."""
        researcher = create_autospec(User, instance=True)
        researcher.id = uuid4()
        researcher.email = "researcher@example.com"
        researcher.username = "researcher"
        researcher.role = UserRole.RESEARCHER
        researcher.is_active = True

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            require_admin(current_user=researcher)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


class TestDependenciesIntegration:
    """Integration tests for dependencies working together."""

    @pytest.mark.asyncio
    async def test_full_authentication_dependency_chain(self):
        """Test complete dependency chain: token -> user -> role check."""
        # Step 1: Create user and token
        user_id = str(uuid4())
        researcher_user = create_autospec(User, instance=True)
        researcher_user.id = user_id
        researcher_user.email = "researcher@example.com"
        researcher_user.username = "researcher"
        researcher_user.role = UserRole.RESEARCHER
        researcher_user.is_active = True

        token = create_access_token({"sub": user_id})

        # Step 2: Mock credentials and database
        mock_credentials = Mock()
        mock_credentials.credentials = token

        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = researcher_user

        # Step 3: Get current user from token
        user = await get_current_user(credentials=mock_credentials, db=mock_db)

        # Step 4: Check role
        result = require_researcher(current_user=user)

        # Should succeed
        assert result == researcher_user
