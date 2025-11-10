"""Database connection and session management.

This module provides SQLAlchemy database connection setup and session management
for OmicSelector2.
"""

from typing import Generator

try:
    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import Session, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Provide stub implementations for when SQLAlchemy is not available
    declarative_base = None  # type: ignore
    Session = None  # type: ignore

from omicselector2.utils.config import get_settings

# Base class for all models
Base = declarative_base() if SQLALCHEMY_AVAILABLE else None  # type: ignore

# Database engine (lazy initialization)
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine.

    Returns:
        SQLAlchemy engine instance

    Raises:
        ImportError: If SQLAlchemy is not installed
    """
    global _engine

    if not SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "SQLAlchemy is required for database operations. "
            "Install with: pip install sqlalchemy psycopg2-binary"
        )

    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=5,
            max_overflow=10,
        )

    return _engine


def get_session_local():
    """Get or create SessionLocal factory.

    Returns:
        SessionLocal factory for creating database sessions

    Raises:
        ImportError: If SQLAlchemy is not installed
    """
    global _SessionLocal

    if not SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "SQLAlchemy is required for database operations. "
            "Install with: pip install sqlalchemy psycopg2-binary"
        )

    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection.

    Yields:
        Database session

    Examples:
        ```python
        @app.get("/users")
        async def get_users(db: Session = Depends(get_db)):
            users = db.query(User).all()
            return users
        ```
    """
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "SQLAlchemy is required for database operations. "
            "Install with: pip install sqlalchemy psycopg2-binary"
        )

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database by creating all tables.

    This should be called on application startup.

    Raises:
        ImportError: If SQLAlchemy is not installed
    """
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "SQLAlchemy is required for database operations. "
            "Install with: pip install sqlalchemy psycopg2-binary"
        )

    from omicselector2.db import user, dataset, job, result  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
