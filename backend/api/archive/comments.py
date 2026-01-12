"""
Comments API router
"""

from fastapi import APIRouter, Depends, HTTPException, status
from repositories import db_pool
from repositories.comment_repository import CommentRepository
from repositories.event_submission_repository import EventSubmissionRepository
from middleware.auth import get_current_user_optional
from models.api.user import UserPublic
from pydantic import BaseModel
from typing import Optional, List, Dict
import re
import httpx
import uuid
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/comments", tags=["comments"])

# Get settings for service_farm URL
from config import get_settings
settings = get_settings()


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = r'https?://[^\s<>"]+'
    urls = re.findall(url_pattern, text)
    # Clean up URLs (remove trailing punctuation)
    return [url.rstrip('.,;:!?)') for url in urls]


class CommentCreate(BaseModel):
    """Request model for creating a comment"""
    story_id: str
    text: str
    parent_comment_id: Optional[str] = None
    reaction_type: Optional[str] = None  # 'support', 'refute', 'question', 'comment'


class CommentResponse(BaseModel):
    """Response model for a comment"""
    id: str
    story_id: str
    user_id: str
    user_name: str
    user_picture: Optional[str]
    user_email: str
    text: str
    parent_comment_id: Optional[str]
    reaction_type: Optional[str]
    created_at: str
    updated_at: Optional[str]


@router.post("", response_model=CommentResponse)
async def create_comment(
    comment_data: CommentCreate,
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Create a new comment on a story

    Requires authentication
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to post comments"
        )

    # Validate reaction_type
    if comment_data.reaction_type and comment_data.reaction_type not in ['support', 'refute', 'question', 'comment']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reaction_type. Must be one of: support, refute, question, comment"
        )

    try:
        # Create comment
        comment = await CommentRepository.create_comment(
            db=db,
            story_id=comment_data.story_id,
            user_id=current_user.user_id,
            text=comment_data.text,
            parent_comment_id=comment_data.parent_comment_id,
            reaction_type=comment_data.reaction_type
        )

        # Extract URLs from comment and create event submissions
        urls = extract_urls(comment_data.text)
        for url in urls:
            try:
                # Create event submission for each URL
                submission = await EventSubmissionRepository.create(
                    db=db,
                    user_id=current_user.user_id,
                    content=f"URL from comment: {comment_data.text[:100]}",
                    urls=url
                )

                # Delegate to service-farm for proper URL normalization and deduplication
                task_id = None
                preview_meta = None

                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Submit to service-farm (handles normalization and deduplication)
                    submit_response = await client.post(
                        f"{settings.service_farm_url}/submit",
                        data={
                            "url": url,
                            "user_id": current_user.user_id,
                            "response_format": "json"
                        },
                        timeout=10.0
                    )

                    if submit_response.status_code == 200:
                        result = submit_response.json()
                        task_id = result.get("task_id")
                        logger.info(f"✅ Task created via service-farm: {task_id} for {url}")
                    else:
                        logger.warning(f"⚠️  Service-farm /submit failed: {submit_response.status_code}")
                        # Don't fail - continue without task

                    # Get instant preview from service_farm
                    if task_id:
                        try:
                            preview_response = await client.get(
                                f"{settings.service_farm_url}/api/preview",
                                params={"url": url},
                                timeout=5.0
                            )
                            if preview_response.status_code == 200:
                                preview_data = preview_response.json()
                                if preview_data.get("title"):
                                    preview_meta = {
                                        "title": preview_data.get("title"),
                                        "description": preview_data.get("description"),
                                        "thumbnail_url": preview_data.get("image") or preview_data.get("thumbnail_url"),
                                        "site_name": preview_data.get("site_name"),
                                        "canonical_url": preview_data.get("url")
                                    }
                        except Exception as e:
                            logger.warning(f"Preview fetch failed for {url}: {e}")

                # Update submission with task_id, status, and preview
                if task_id:
                    submission.task_id = task_id
                    submission.status = "extracting"
                    if preview_meta:
                        submission.preview_meta = json.dumps(preview_meta)
                    await db.commit()

                logger.info(f"🔗 Created extraction task from comment URL: {url} (task_id: {task_id})")

            except Exception as e:
                logger.error(f"❌ Failed to create event submission for URL {url}: {e}")
                # Don't fail the comment if event submission fails

        # Return comment with user info
        return CommentResponse(
            id=comment.id,
            story_id=comment.story_id,
            user_id=str(comment.user_id),
            user_name=current_user.name,
            user_picture=current_user.picture_url,
            user_email=current_user.email,
            text=comment.text,
            parent_comment_id=comment.parent_comment_id,
            reaction_type=comment.reaction_type,
            created_at=comment.created_at.isoformat(),
            updated_at=comment.updated_at.isoformat() if comment.updated_at else None
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create comment: {str(e)}"
        )


@router.get("/story/{story_id}", response_model=List[CommentResponse])
async def get_story_comments(
    story_id: str,
):
    """
    Get all comments for a story

    Returns comments with user information, sorted by creation time
    """
    try:
        comments = await CommentRepository.get_comments_for_story(db, story_id)
        return comments

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch comments: {str(e)}"
        )


@router.delete("/{comment_id}")
async def delete_comment(
    comment_id: str,
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Delete a comment

    Only the comment author can delete their own comment
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    success = await CommentRepository.delete_comment(
        db=db,
        comment_id=comment_id,
        user_id=current_user.user_id
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found or unauthorized"
        )

    return {"status": "success", "message": "Comment deleted"}


@router.get("/story/{story_id}/count")
async def get_comment_count(
    story_id: str,
):
    """Get total comment count for a story"""
    try:
        count = await CommentRepository.get_comment_count(db, story_id)
        return {"story_id": story_id, "count": count}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get comment count: {str(e)}"
        )
