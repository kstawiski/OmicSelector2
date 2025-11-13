"""Security utilities for authentication and authorization.

This module provides JWT token generation, password hashing, and authentication
utilities for OmicSelector2.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from passlib.context import CryptContext

    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    CryptContext = None  # type: ignore

try:
    from jose import JWTError, jwt

    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    JWTError = Exception  # type: ignore
    jwt = None  # type: ignore

from omicselector2.utils.config import get_settings

# Password hashing context
if PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None  # type: ignore


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string

    Raises:
        ImportError: If passlib is not installed
    """
    if not PASSLIB_AVAILABLE:
        raise ImportError(
            "passlib is required for password hashing. " "Install with: pip install passlib[bcrypt]"
        )

    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to verify against

    Returns:
        True if password matches, False otherwise

    Raises:
        ImportError: If passlib is not installed
    """
    if not PASSLIB_AVAILABLE:
        raise ImportError(
            "passlib is required for password verification. "
            "Install with: pip install passlib[bcrypt]"
        )

    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token (typically {"sub": user_id})
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string

    Raises:
        ImportError: If python-jose is not installed
    """
    if not JOSE_AVAILABLE:
        raise ImportError(
            "python-jose is required for JWT tokens. "
            "Install with: pip install python-jose[cryptography]"
        )

    settings = get_settings()

    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Default to 60 minutes expiration
        expire = datetime.now(timezone.utc) + timedelta(minutes=60)

    to_encode.update({"exp": expire})

    # Use HS256 algorithm and SECRET_KEY from settings
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode a JWT access token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload, or None if invalid

    Raises:
        ImportError: If python-jose is not installed
    """
    if not JOSE_AVAILABLE:
        raise ImportError(
            "python-jose is required for JWT tokens. "
            "Install with: pip install python-jose[cryptography]"
        )

    settings = get_settings()

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        return None


def get_password_hash(password: str) -> str:
    """Alias for hash_password for backwards compatibility.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return hash_password(password)


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "get_password_hash",
]
