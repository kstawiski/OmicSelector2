"""
Tests for security utilities (password hashing, JWT tokens).

This module tests the security functions used for authentication:
- Password hashing with bcrypt
- Password verification
- JWT token generation
- JWT token validation
- Token payload extraction

Following TDD: These tests are written FIRST and MUST fail initially.
"""

import pytest
from datetime import datetime, timedelta
from omicselector2.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password_returns_string(self):
        """Test that hash_password returns a string."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert isinstance(hashed, str)
        assert len(hashed) > 0
        assert hashed != password  # Must not return plaintext

    def test_hash_password_is_different_each_time(self):
        """Test that hashing same password produces different hashes (salt)."""
        password = "test_password_123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        # Due to salt, hashes should be different
        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test that verify_password returns True for correct password."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test that verify_password returns False for incorrect password."""
        password = "test_password_123"
        wrong_password = "wrong_password_456"
        hashed = hash_password(password)

        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_with_empty_string(self):
        """Test that verify_password handles empty password gracefully."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert verify_password("", hashed) is False


class TestJWTTokens:
    """Test JWT token generation and validation."""

    def test_create_access_token_returns_string(self):
        """Test that create_access_token returns a JWT string."""
        data = {"sub": "test_user_id"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0
        # JWT tokens have 3 parts separated by dots
        assert token.count(".") == 2

    def test_create_access_token_with_expiry(self):
        """Test that create_access_token accepts custom expiry."""
        data = {"sub": "test_user_id"}
        expires_delta = timedelta(minutes=15)
        token = create_access_token(data, expires_delta=expires_delta)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_access_token_valid(self):
        """Test that decode_access_token returns payload for valid token."""
        data = {"sub": "test_user_id", "email": "test@example.com"}
        token = create_access_token(data)

        payload = decode_access_token(token)

        assert payload is not None
        assert payload["sub"] == "test_user_id"
        assert payload["email"] == "test@example.com"
        assert "exp" in payload  # Should have expiry

    def test_decode_access_token_invalid(self):
        """Test that decode_access_token returns None for invalid token."""
        invalid_token = "invalid.jwt.token"

        payload = decode_access_token(invalid_token)

        assert payload is None

    def test_decode_access_token_expired(self):
        """Test that decode_access_token returns None for expired token."""
        data = {"sub": "test_user_id"}
        # Create token that expires immediately
        expires_delta = timedelta(seconds=-1)
        token = create_access_token(data, expires_delta=expires_delta)

        payload = decode_access_token(token)

        assert payload is None

    def test_decode_access_token_tampered(self):
        """Test that decode_access_token returns None for tampered token."""
        data = {"sub": "test_user_id"}
        token = create_access_token(data)

        # Tamper with token
        tampered_token = token[:-10] + "tampered00"

        payload = decode_access_token(tampered_token)

        assert payload is None

    def test_token_contains_expiry(self):
        """Test that generated token includes expiry timestamp."""
        data = {"sub": "test_user_id"}
        token = create_access_token(data)

        payload = decode_access_token(token)

        assert "exp" in payload
        assert isinstance(payload["exp"], (int, float))
        # Expiry should be in the future
        assert payload["exp"] > datetime.utcnow().timestamp()


class TestSecurityIntegration:
    """Integration tests for security utilities."""

    def test_full_authentication_flow(self):
        """Test complete authentication flow: hash, verify, create token, decode token."""
        # Step 1: User registers with password
        password = "secure_password_123"
        hashed = hash_password(password)

        # Step 2: User logs in, password is verified
        assert verify_password(password, hashed) is True

        # Step 3: Create JWT token for authenticated user
        user_data = {"sub": "user_uuid_12345", "email": "user@example.com"}
        token = create_access_token(user_data)

        # Step 4: Decode token to get user info
        payload = decode_access_token(token)

        assert payload is not None
        assert payload["sub"] == "user_uuid_12345"
        assert payload["email"] == "user@example.com"

    def test_authentication_fails_with_wrong_password(self):
        """Test that authentication fails with wrong password."""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = hash_password(password)

        # Wrong password should not verify
        assert verify_password(wrong_password, hashed) is False

        # Should not create token for failed authentication
        # (This would be enforced in the API endpoint logic)
