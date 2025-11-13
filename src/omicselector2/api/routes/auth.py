"""Authentication routes for OmicSelector2 API.

This module provides endpoints for user authentication and authorization.
"""

try:
    from fastapi import APIRouter, Depends, HTTPException, status
    from pydantic import BaseModel, EmailStr, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore
    Depends = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore
    BaseModel = object  # type: ignore
    EmailStr = str  # type: ignore
    Field = None  # type: ignore

try:
    from sqlalchemy.orm import Session

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Session = None  # type: ignore

from omicselector2.api.dependencies import get_current_user as get_auth_user
from omicselector2.db import User, UserRole, get_db
from omicselector2.utils.security import (
    create_access_token,
    hash_password,
    verify_password,
)


# Request/Response Models
class UserRegisterRequest(BaseModel):
    """User registration request model.

    Attributes:
        email: User email address
        username: Username
        password: Password (minimum 8 characters)
        full_name: Full name (optional)
    """

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: str | None = None


class UserResponse(BaseModel):
    """User response model.

    Attributes:
        id: User ID
        email: User email
        username: Username
        full_name: Full name
        role: User role
    """

    id: str
    email: str
    username: str
    full_name: str | None
    role: str


class LoginRequest(BaseModel):
    """Login request model.

    Attributes:
        username: Username or email
        password: Password
    """

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model.

    Attributes:
        access_token: JWT access token
        token_type: Token type (bearer)
        expires_in: Token expiration time in seconds
    """

    access_token: str
    token_type: str = "bearer"
    expires_in: int


if not FASTAPI_AVAILABLE:
    router = None  # type: ignore
else:
    router = APIRouter()

    @router.post("/register", response_model=UserResponse, status_code=201)
    async def register(user_data: UserRegisterRequest, db: Session = Depends(get_db)):
        """Register a new user.

        Args:
            user_data: User registration data
            db: Database session

        Returns:
            Created user information

        Raises:
            HTTPException: If username or email already exists
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Check if user already exists
        existing_user = (
            db.query(User)
            .filter((User.email == user_data.email) | (User.username == user_data.username))
            .first()
        )

        if existing_user:
            if existing_user.email == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken",
                )

        # Hash password
        hashed_password = hash_password(user_data.password)

        # Create new user
        new_user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=UserRole.USER,  # Default role
            is_active=True,
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return UserResponse(
            id=str(new_user.id),
            email=new_user.email,
            username=new_user.username,
            full_name=new_user.full_name,
            role=new_user.role.value,
        )

    @router.post("/login", response_model=TokenResponse)
    async def login(credentials: LoginRequest, db: Session = Depends(get_db)):
        """Authenticate user and return access token.

        Args:
            credentials: Login credentials
            db: Database session

        Returns:
            JWT access token

        Raises:
            HTTPException: If credentials are invalid
        """
        if not SQLALCHEMY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available",
            )

        # Find user by username or email
        user = (
            db.query(User)
            .filter((User.username == credentials.username) | (User.email == credentials.username))
            .first()
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify password
        if not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user account",
            )

        # Create access token
        access_token = create_access_token(data={"sub": str(user.id)})

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,  # 60 minutes in seconds
        )

    @router.get("/me", response_model=UserResponse)
    async def get_me(current_user: User = Depends(get_auth_user)):
        """Get current authenticated user information.

        Args:
            current_user: Current authenticated user

        Returns:
            Current user information

        Raises:
            HTTPException: If not authenticated
        """
        return UserResponse(
            id=str(current_user.id),
            email=current_user.email,
            username=current_user.username,
            full_name=current_user.full_name,
            role=current_user.role.value,
        )

    @router.post("/logout")
    async def logout():
        """Logout current user.

        Returns:
            Success message
        """
        # TODO: Implement logout
        # - Invalidate token (if using token blacklist)
        # - Return success message
        return {"message": "Logged out successfully"}
