"""
URL Extraction API Router
Handles submission and tracking of URL extraction tasks
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy import select, update, insert
from typing import Optional
import uuid
import httpx
from datetime import datetime, timedelta

from middleware.auth import get_current_user
from repositories import db_pool
from models.api.user import UserPublic
from models.api.extraction import ExtractionTask, UserURL
from app.database.models import User
from config import get_settings

settings = get_settings()
router = APIRouter(tags=["extraction"])


@router.post("/submit")
async def submit_url(
    request: Request,
    url: str = Form(...),
    force: bool = Form(False),
    target_story_id: Optional[str] = Form(None),
    response_format: str = Form("redirect"),  # "redirect" or "json"
    user: UserPublic = Depends(get_current_user),
):
    """
    Submit URL for extraction

    Args:
        url: URL to extract
        force: Bypass deduplication checks
        target_story_id: Optional story ID to assign
        response_format: "redirect" or "json"

    Returns:
        Redirect to /task/{task_id} or JSON with task_id
    """
    # Check user credits
    SUBMISSION_COST = 10

    # Get full user record with credits
    result = await db.execute(
        select(User).where(User.user_id == user.user_id)
    )
    db_user = result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(status_code=401, detail="User not found")

    if db_user.credits_balance < SUBMISSION_COST:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. You have {db_user.credits_balance} credits, but need {SUBMISSION_COST}."
        )

    # Delegate to service-farm's /submit endpoint
    # This ensures proper URL normalization and deduplication
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            submit_response = await client.post(
                f"{settings.service_farm_url}/submit",
                data={
                    "url": url,
                    "force": "true" if force else "false",
                    "target_story_id": target_story_id or "",
                    "user_id": user.user_id,
                    "format": "json"
                },
                timeout=10.0
            )

            if submit_response.status_code != 200:
                error_detail = submit_response.json().get("detail", "Unknown error")
                raise HTTPException(
                    status_code=submit_response.status_code,
                    detail=f"Service farm error: {error_detail}"
                )

            result = submit_response.json()
            task_id = result["task_id"]

            print(f"✅ Task created via service-farm: {task_id} for {url}")

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service farm unavailable: {str(e)}"
        )

    # Deduct credits
    await db.execute(
        update(User)
        .where(User.user_id == user.user_id)
        .values(credits_balance=User.credits_balance - SUBMISSION_COST)
    )

    # Link URL to user (task already created by service-farm)
    user_url = UserURL(
        user_id=user.user_id,
        task_id=task_id,
        url=url,
        credits_spent=SUBMISSION_COST
    )
    db.add(user_url)

    await db.commit()

    # Return response
    if response_format == "json":
        return {"task_id": task_id, "status": "submitted"}
    else:
        return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.get("/api/task/{task_id}")
async def get_task_status(
    task_id: str,
):
    """
    Get task status and results

    Args:
        task_id: Task UUID

    Returns:
        Task details including status, results, and semantic data
    """
    result = await db.execute(
        select(ExtractionTask).where(ExtractionTask.id == task_id)
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Build response
    response = {
        "task_id": task.id,
        "url": task.url,
        "canonical_url": task.canonical_url,
        "status": task.status,
        "current_stage": task.current_stage,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "error_message": task.error_message,
        "block_reason": task.block_reason,
        "result": task.result,
        "semantic_data": task.semantic_data,
        "story_match": task.story_match
    }

    return response


@router.get("/api/preview")
async def get_preview(
    url: str,
):
    """
    Get instant preview for a URL

    Checks if we have cached metadata from previous extractions

    Args:
        url: URL to preview

    Returns:
        Preview metadata (title, description, image, etc.)
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Check for rogue/paywalled domains
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    rogue_domains = [
        'reuters.com', 'wsj.com', 'ft.com', 'economist.com',
        'nytimes.com', 'bloomberg.com', 'inmediahk.net',
        'scmp.com', 'washingtonpost.com'
    ]

    is_rogue = any(rogue_domain in domain for rogue_domain in rogue_domains)

    # Look for recent completed task for this URL
    cutoff_time = datetime.utcnow() - timedelta(days=30)
    result = await db.execute(
        select(ExtractionTask)
        .where(ExtractionTask.url == url)
        .where(ExtractionTask.status == "completed")
        .where(ExtractionTask.created_at >= cutoff_time)
        .order_by(ExtractionTask.created_at.desc())
        .limit(1)
    )
    task = result.scalar_one_or_none()

    if task and task.result:
        result_data = task.result

        # Extract preview fields
        preview = {
            "url": url,
            "title": result_data.get("title", ""),
            "description": result_data.get("meta_description") or result_data.get("content_text", "")[:200],
            "image": result_data.get("screenshot_url"),
            "site_name": result_data.get("site_name"),
            "author": result_data.get("author"),
            "published_date": result_data.get("publish_date"),
            "word_count": result_data.get("word_count"),
            "reading_time_minutes": result_data.get("reading_time_minutes"),
            "preview_quality": "cached",
            "is_cached": True,
            "is_rogue": is_rogue
        }

        return preview

    # No cached result - return placeholder
    return {
        "url": url,
        "title": "Preview not available",
        "description": "Submit to start extraction",
        "preview_quality": "placeholder",
        "is_cached": False,
        "is_rogue": is_rogue
    }
