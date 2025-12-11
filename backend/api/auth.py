"""
Authentication API router with Google OAuth and JWT
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from typing import Optional

from middleware.google_oauth import get_google_oauth
from middleware.jwt_session import create_access_token
from middleware.auth import get_current_user_optional
from repositories.user_repository import UserRepository
from models.api.user import UserResponse, UserPublic
from models.domain.user import User
from config import get_settings
from repositories import get_db_pool

settings = get_settings()
router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.get("/login")
async def login(request: Request):
    """
    Initiate Google OAuth login flow

    Redirects user to Google consent screen
    """
    google = get_google_oauth()

    # Construct redirect URI dynamically from request host if not explicitly set
    if settings.google_redirect_uri:
        redirect_uri = settings.google_redirect_uri
    else:
        # Automatically construct from current request
        scheme = "https" if request.url.scheme == "https" else "http"
        host = request.headers.get("host", "localhost:7272")
        redirect_uri = f"{scheme}://{host}/api/auth/callback"

    return await google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth_callback(request: Request):
    """
    OAuth callback handler

    Google redirects here after user authorization
    Creates/updates user and issues session token
    """
    try:
        google = get_google_oauth()

        # Exchange authorization code for user info
        token = await google.authorize_access_token(request)
        user_info = token.get('userinfo')

        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info from Google")

        # Extract user data
        google_id = user_info.get('sub')
        email = user_info.get('email')
        name = user_info.get('name')
        picture = user_info.get('picture')

        if not google_id or not email:
            raise HTTPException(status_code=400, detail="Invalid user info from Google")

        # Create or update user
        pool = await get_db_pool()
        user_repo = UserRepository(pool)
        user = await user_repo.get_by_google_id(google_id)

        if not user:
            # Create new user
            user = User(
                user_id="",  # Will be generated in __post_init__
                email=email,
                google_id=google_id,
                name=name or email.split('@')[0],
                picture_url=picture
            )
            user = await user_repo.create(user)
        else:
            # Update last login
            await user_repo.update_last_login(user.user_id)

        # Create JWT token
        access_token = create_access_token(user)

        # Redirect to app with token in cookie
        response = RedirectResponse(url="/app")
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            max_age=settings.jwt_expire_minutes * 60,
            samesite="lax",
            secure=False  # Set to True in production with HTTPS
        )

        return response

    except Exception as e:
        import traceback
        print(f"OAuth callback error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.get("/logout")
async def logout():
    """
    Logout user

    Clears session cookie
    """
    response = RedirectResponse(url="/")
    response.delete_cookie(key="access_token")
    return response


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    request: Request,
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Get current authenticated user info

    Returns 401 if not authenticated
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Fetch full user details from database
    pool = await get_db_pool()
    user_repo = UserRepository(pool)
    user = await user_repo.get_by_id(str(current_user.user_id))

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse.model_validate(user)


@router.get("/status")
async def auth_status(
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Check authentication status

    Returns user info if authenticated, null if not
    """
    if current_user:
        # Get full user data from database
        pool = await get_db_pool()
        user_repo = UserRepository(pool)
        user = await user_repo.get_by_id(str(current_user.user_id))

        if user:
            return {
                "authenticated": True,
                "user": UserResponse.model_validate(user)
            }

    return {
        "authenticated": False,
        "user": None
    }
