"""
JWT session management
"""
from datetime import datetime, timedelta
from jose import jwt
from config import get_settings

settings = get_settings()


def create_access_token(user) -> str:
    """
    Create JWT access token for user

    Args:
        user: User model with user_id, email, name

    Returns:
        JWT token string
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)

    payload = {
        "sub": str(user.user_id),
        "email": user.email,
        "name": user.name,
        "exp": expire,
        "iat": datetime.utcnow()
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
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
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm]
    )

    return payload
