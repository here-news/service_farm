"""
Event submission API router
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import json
import logging
import httpx
import uuid
from datetime import datetime

from middleware.auth import get_current_user_optional
from repositories import db_pool
from repositories.event_submission_repository import EventSubmissionRepository
from repositories.user_repository import UserRepository
from models.api.user import UserPublic
from models.api.event_submission import (
    EventSubmissionCreate,
    EventSubmissionResponse,
    StoryMatch,
    PreviewMeta
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/submissions", tags=["submissions"])

# Get settings for service farm URL
from config import get_settings
settings = get_settings()


def parse_json_field(field_value: str | None, model_class=None):
    """Safely parse JSON field from database"""
    if not field_value:
        return None
    try:
        data = json.loads(field_value)
        if model_class:
            return model_class(**data)
        return data
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


@router.post("", response_model=EventSubmissionResponse, status_code=status.HTTP_201_CREATED)
async def create_event_submission(
    submission_data: EventSubmissionCreate,
    current_user: UserPublic = Depends(get_current_user_optional)
):
    """
    Submit a new event for processing

    Requires authentication
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to submit events"
        )

    # Get full user data from database for picture
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(current_user.user_id)

    # Create submission
    submission = await EventSubmissionRepository.create(
        db=db,
        user_id=current_user.user_id,
        content=submission_data.content,
        urls=submission_data.urls or ""
    )

    # Extract URL and trigger extraction on service_farm
    url = submission_data.urls.strip() if submission_data.urls else None
    if url:
        try:
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
                    logger.error(f"❌ Service-farm /submit failed: {submit_response.status_code}")
                    raise HTTPException(
                        status_code=503,
                        detail="Failed to create extraction task"
                    )

                # Get instant preview from service_farm for preview_meta
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
            submission.task_id = task_id
            submission.status = "extracting"
            if preview_meta:
                submission.preview_meta = json.dumps(preview_meta)
            await db.commit()
            await db.refresh(submission)

            logger.info(f"🚀 Extraction task created: {task_id} for {url}")

        except Exception as e:
            logger.error(f"❌ Failed to create extraction task: {e}")
            # Don't fail the request - submission is created
            submission.status = "pending"
            await db.commit()

    # Return with user info
    return EventSubmissionResponse(
        id=submission.id,
        user_id=str(submission.user_id),
        user_name=user.name if user else current_user.name,
        user_picture=user.picture_url if user else None,
        content=submission.content,
        urls=submission.urls,
        status=submission.status,
        task_id=submission.task_id,
        story_match=parse_json_field(submission.story_match, StoryMatch),
        preview_meta=parse_json_field(submission.preview_meta, PreviewMeta),
        created_at=submission.created_at
    )


@router.get("/mine", response_model=List[EventSubmissionResponse])
async def get_my_submissions(
    current_user: UserPublic = Depends(get_current_user_optional)
):
    """
    Get current user's event submissions

    Requires authentication
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Get full user data from database for picture
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(current_user.user_id)

    submissions = await EventSubmissionRepository.get_by_user(db, current_user.user_id)

    # Query extraction tasks to get latest status
    from models.api.extraction import ExtractionTask
    from sqlalchemy import select

    result_list = []
    for sub in submissions:
        # Get extraction task status if task_id exists
        current_status = sub.status
        current_story_match = parse_json_field(sub.story_match, StoryMatch)
        current_preview_meta = parse_json_field(sub.preview_meta, PreviewMeta)
        current_stage = None
        error = None
        block_reason = None
        result = None
        semantic_data = None

        if sub.task_id:
            try:
                task_result = await db.execute(
                    select(ExtractionTask).where(ExtractionTask.id == sub.task_id)
                )
                task = task_result.scalar_one_or_none()

                if task:
                    # Map extraction task status to submission status
                    if task.status == "completed":
                        current_status = "completed"
                        if task.story_match:
                            current_story_match = StoryMatch(**task.story_match) if isinstance(task.story_match, dict) else current_story_match
                    elif task.status == "failed":
                        current_status = "failed"
                        error = task.error_message
                    elif task.status == "blocked":
                        current_status = "blocked"
                        block_reason = task.block_reason
                    elif task.status == "processing":
                        current_status = "extracting"

                    # Get current stage
                    current_stage = task.current_stage

                    # Update preview_meta if available
                    if task.preview_meta and isinstance(task.preview_meta, dict):
                        current_preview_meta = PreviewMeta(**task.preview_meta)

                    # Extract result data
                    if task.result and isinstance(task.result, dict):
                        from models.api.event_submission import ExtractionResult
                        result = ExtractionResult(
                            title=task.result.get("title"),
                            author=task.result.get("author"),
                            publish_date=task.result.get("publish_date"),
                            meta_description=task.result.get("meta_description"),
                            content_text=task.result.get("content_text"),
                            screenshot_url=task.result.get("screenshot_url"),
                            word_count=task.result.get("word_count"),
                            reading_time_minutes=task.result.get("reading_time_minutes"),
                            language=task.result.get("language")
                        )

                    # Extract semantic data
                    if task.semantic_data and isinstance(task.semantic_data, dict):
                        from models.api.event_submission import SemanticData
                        semantic_data = SemanticData(
                            claims=task.semantic_data.get("claims"),
                            entities=task.semantic_data.get("entities")
                        )
            except Exception as e:
                logger.error(f"Failed to fetch extraction task {sub.task_id}: {e}")

        result_list.append(EventSubmissionResponse(
            id=sub.id,
            user_id=str(sub.user_id),
            user_name=user.name if user else current_user.name,
            user_picture=user.picture_url if user else None,
            content=sub.content,
            urls=sub.urls,
            status=current_status,
            task_id=sub.task_id,
            story_match=current_story_match,
            preview_meta=current_preview_meta,
            created_at=sub.created_at,
            current_stage=current_stage,
            error=error,
            block_reason=block_reason,
            result=result,
            semantic_data=semantic_data
        ))

    return result_list
