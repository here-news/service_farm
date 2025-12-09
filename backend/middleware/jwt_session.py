"""
JWT session management
Copied from webapp/app/auth/session.py
"""

import os
from datetime import datetime, timedelta
from jose import jwt

# Settings from environment
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev_secret_key_change_in_production')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXPIRE_MINUTES = int(os.getenv('JWT_EXPIRE_MINUTES', '1440'))  # 24 hours


def create_access_token(user) -> str:
    """
    Create JWT access token for user

    Args:
        user: User model with user_id, email, name

    Returns:
        JWT token string
    """
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)

    payload = {
        "sub": str(user.user_id),
        "email": user.email,
        "name": user.name,
        "exp": expire,
        "iat": datetime.utcnow()
    }

    token = jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

    return token


def decode_access_token(token: str) -> dict:
    """
    Decode and validate JWT token

    Returns:
        Decoded payload dict

    Raises:
        jose.JWTError if token invalid/expired
    """
    payload = jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM]
    )

    return payload
