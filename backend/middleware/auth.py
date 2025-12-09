"""
Authentication middleware and dependencies
Copied from webapp/app/auth/middleware.py
"""

from fastapi import Request, HTTPException
from jose import JWTError
from typing import Optional
from uuid import UUID

from .jwt_session import decode_access_token


class UserPublic:
    """Minimal user info from JWT token"""
    def __init__(self, user_id: UUID, email: str, name: str, picture_url: Optional[str] = None):
        self.user_id = user_id
        self.email = email
        self.name = name
        self.picture_url = picture_url


async def get_current_user_optional(request: Request) -> Optional[UserPublic]:
    """
    Get current user from JWT token (optional - doesn't raise if not authenticated)

    Returns:
        UserPublic if authenticated, None otherwise
    """
    # Try to get token from cookie
    token = request.cookies.get("access_token")

    if not token:
        return None

    try:
        # Decode JWT
        payload = decode_access_token(token)

        # Create UserPublic from payload
        user = UserPublic(
            user_id=UUID(payload.get("sub")),
            email=payload.get("email"),
            name=payload.get("name"),
            picture_url=None  # Not stored in JWT
        )

        return user

    except JWTError:
        return None


async def get_current_user(request: Request) -> UserPublic:
    """
    Get current user (required - raises 401 if not authenticated)

    Returns:
        UserPublic

    Raises:
        HTTPException 401 if not authenticated
    """
    user = await get_current_user_optional(request)

    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return user
