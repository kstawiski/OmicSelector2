"""User database model.

This module defines the User model for authentication and authorization.
"""

import uuid
from datetime import datetime
from enum import Enum

try:
    from sqlalchemy import Boolean, Column, DateTime, String
    from sqlalchemy.dialects.postgresql import ENUM, UUID
    from sqlalchemy.orm import relationship
    from sqlalchemy.sql import func

    from omicselector2.db.database import Base

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = object  # type: ignore


class UserRole(str, Enum):
    """User role enumeration.

    Attributes:
        USER: Regular user with basic permissions
        RESEARCHER: Researcher with advanced features
        ADMIN: Administrator with full permissions
    """

    USER = "user"
    RESEARCHER = "researcher"
    ADMIN = "admin"


if SQLALCHEMY_AVAILABLE:

    class User(Base):  # type: ignore
        """User model for authentication and authorization.

        Attributes:
            id: Unique user identifier (UUID)
            email: User email address (unique)
            username: Username (unique)
            hashed_password: Bcrypt hashed password
            full_name: User's full name
            role: User role (USER, RESEARCHER, ADMIN)
            is_active: Whether user account is active
            created_at: Account creation timestamp
            updated_at: Last update timestamp

        Relationships:
            datasets: Datasets owned by user
            jobs: Jobs created by user
        """

        __tablename__ = "users"

        id = Column(
            UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
        )
        email = Column(String(255), unique=True, nullable=False, index=True)
        username = Column(String(100), unique=True, nullable=False, index=True)
        hashed_password = Column(String(255), nullable=False)
        full_name = Column(String(255), nullable=True)
        role = Column(
            ENUM(UserRole, name="user_role", create_type=True),
            default=UserRole.USER,
            nullable=False,
        )
        is_active = Column(Boolean, default=True, nullable=False)
        created_at = Column(
            DateTime(timezone=True), server_default=func.now(), nullable=False
        )
        updated_at = Column(DateTime(timezone=True), onupdate=func.now())

        # Relationships (will be defined when other models are created)
        # datasets = relationship("Dataset", back_populates="owner")
        # jobs = relationship("Job", back_populates="user")

        def __repr__(self) -> str:
            """String representation of User.

            Returns:
                User representation string
            """
            return f"<User(id={self.id}, username={self.username}, role={self.role})>"

else:
    # Stub class when SQLAlchemy not available
    class User:  # type: ignore
        """Stub User class when SQLAlchemy is not installed."""

        def __init__(self, *args, **kwargs):  # type: ignore
            raise ImportError(
                "SQLAlchemy is required for User model. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )
