"""FastAPI dependencies for authentication and database access.

This module provides dependency injection functions for FastAPI routes.
"""

from typing import Optional

try:
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    Depends = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore
    HTTPBearer = None  # type: ignore
    HTTPAuthorizationCredentials = None  # type: ignore

try:
    from sqlalchemy.orm import Session

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Session = None  # type: ignore

from omicselector2.db import User, UserRole, get_db
from omicselector2.utils.security import decode_access_token

# HTTP Bearer security scheme
if FASTAPI_AVAILABLE:
    security = HTTPBearer()
else:
    security = None  # type: ignore


# Alias for consistency with test expectations
get_db_session = get_db


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),  # type: ignore
    db: Session = Depends(get_db),  # type: ignore
) -> User:
    """Get current authenticated user from JWT token.

    Args:
        credentials: HTTP authorization credentials
        db: Database session

    Returns:
        Current authenticated user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for authentication")

    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for database access")

    # Decode JWT token
    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user ID from token
    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),  # type: ignore
) -> User:
    """Get current active user.

    This is an alias for get_current_user for clarity.

    Args:
        current_user: Current user from get_current_user

    Returns:
        Current active user
    """
    return current_user


def require_role(required_role: UserRole):
    """Create dependency that requires specific user role.

    Args:
        required_role: Required role (UserRole enum)

    Returns:
        FastAPI dependency function

    Examples:
        @app.delete("/datasets/{id}")
        async def delete_dataset(
            id: str,
            user: User = Depends(require_role(UserRole.RESEARCHER))
        ):
            ...
    """

    def role_checker(
        current_user: User = Depends(get_current_user),  # type: ignore
    ) -> User:
        """Check if user has required role.

        Args:
            current_user: Current user

        Returns:
            Current user if authorized

        Raises:
            HTTPException: If user doesn't have required role
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for role checking")

        # Admin has access to everything
        if current_user.role == UserRole.ADMIN:
            return current_user

        # Check if user has required role
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}",
            )

        return current_user

    return role_checker


def require_researcher(
    current_user: User = Depends(get_current_user),  # type: ignore
) -> User:
    """Require RESEARCHER or ADMIN role.

    Args:
        current_user: Current user from authentication

    Returns:
        Current user if authorized

    Raises:
        HTTPException: If user doesn't have RESEARCHER or ADMIN role
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for role checking")

    # Admin and Researcher have access
    if current_user.role in (UserRole.ADMIN, UserRole.RESEARCHER):
        return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Insufficient permissions. Required role: RESEARCHER or ADMIN",
    )


def require_admin(
    current_user: User = Depends(get_current_user),  # type: ignore
) -> User:
    """Require ADMIN role.

    Args:
        current_user: Current user from authentication

    Returns:
        Current user if authorized

    Raises:
        HTTPException: If user doesn't have ADMIN role
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for role checking")

    # Only Admin has access
    if current_user.role == UserRole.ADMIN:
        return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Insufficient permissions. Required role: ADMIN",
    )


__all__ = [
    "get_db_session",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "require_researcher",
    "require_admin",
    "security",
]
