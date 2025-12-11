"""
Rogue URL Extraction Endpoints - Browser Extension Integration

Endpoints for browser extension to handle URLs that block scrapers (401/403)

Architecture:
1. ExtractionWorker marks failed URLs as "rogue" → creates task
2. Browser extension polls GET /rogue/tasks for pending tasks
3. Extension extracts metadata in real browser → POST /rogue/tasks/{id}/complete
4. System uses extracted metadata for preview

Refactored: Uses PageRepository and RogueTaskRepository for all data access
"""
import os
import uuid
import json
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncpg

from repositories import PageRepository, RogueTaskRepository

router = APIRouter()

# Globals (initialized on startup)
db_pool = None
page_repo = None
rogue_task_repo = None


async def init_services():
    """Initialize database pool and repositories"""
    global db_pool, page_repo, rogue_task_repo

    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'herenews_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
            database=os.getenv('POSTGRES_DB', 'herenews'),
            min_size=2,
            max_size=10
        )

    if page_repo is None:
        page_repo = PageRepository(db_pool)

    if rogue_task_repo is None:
        rogue_task_repo = RogueTaskRepository(db_pool)

    return page_repo, rogue_task_repo


class RogueTask(BaseModel):
    """Rogue extraction task for browser extension"""
    id: str
    page_id: str
    url: str
    status: Optional[str] = None
    created_at: str


class RogueTaskMetadata(BaseModel):
    """Metadata extracted by browser extension"""
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail: Optional[str] = None
    author: Optional[str] = None
    site_name: Optional[str] = None
    published_date: Optional[str] = None
    canonical_url: Optional[str] = None
    content_text: Optional[str] = None
    word_count: int = 0


@router.get("/rogue/tasks")
async def get_rogue_tasks(limit: int = 1, status: Optional[str] = None, recent: bool = False):
    """
    Get rogue extraction tasks

    Two modes:
    1. Worker mode (default): Get pending tasks and mark as processing
    2. Recent mode (recent=true): Get recent tasks for UI display (any status)

    Query params:
        limit: Number of tasks to return (default: 1)
        status: Filter by status (pending, processing, completed, failed)
        recent: If true, return recent tasks without marking as processing

    Returns:
        List of tasks
    """
    _, rogue_task_repo = await init_services()

    if recent:
        # Recent tasks mode - for UI display
        tasks_data = await rogue_task_repo.get_recent_tasks(limit=limit, status=status)

        return [
            RogueTask(
                id=str(task['id']),
                page_id=str(task['page_id']),
                url=task['url'],
                status=task['status'],
                created_at=task['created_at'].isoformat()
            )
            for task in tasks_data
        ]

    else:
        # Worker mode - get pending and mark as processing
        tasks_data = await rogue_task_repo.get_pending_tasks(limit=limit)

        return [
            RogueTask(
                id=str(task['id']),
                page_id=str(task['page_id']),
                url=task['url'],
                status=task.get('status', 'processing'),
                created_at=task['created_at'].isoformat()
            )
            for task in tasks_data
        ]


@router.post("/rogue/tasks/{task_id}/complete")
async def complete_rogue_task(task_id: str, metadata: RogueTaskMetadata):
    """
    Complete a rogue extraction task with extracted metadata

    Browser extension calls this after extracting metadata from the page.

    Args:
        task_id: Task UUID
        metadata: Extracted metadata from browser

    Returns:
        Success status
    """
    page_repo, rogue_task_repo = await init_services()

    # Get task to verify it exists and get page_id
    task = await rogue_task_repo.get_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task['status'] not in ['pending', 'processing']:
        raise HTTPException(status_code=400, detail=f"Task already {task['status']}")

    # Mark task as completed
    await rogue_task_repo.mark_completed(task_id)

    # Update page with extracted metadata
    page_id = task['page_id']
    page = await page_repo.get_by_id(page_id)

    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    # Parse published date if provided
    pub_time = None
    if metadata.published_date:
        try:
            from dateutil import parser as date_parser
            pub_time = date_parser.parse(metadata.published_date)
        except:
            pass  # Skip if date parsing fails

    # Determine new status based on content quality
    if metadata.content_text and metadata.word_count >= 100:
        new_status = 'extraction_complete'
    else:
        # At least we have metadata, mark as preview
        new_status = 'preview'

    # Update page with extracted content
    await page_repo.update_extracted_content(
        page_id=page_id,
        content_text=metadata.content_text or page.content_text or '',
        title=metadata.title or page.title or '',
        description=metadata.description,
        author=metadata.author,
        thumbnail_url=metadata.thumbnail,
        metadata_confidence=0.7,  # Rogue extraction is less reliable
        language='en',  # Default, could be improved
        word_count=metadata.word_count,
        pub_time=pub_time
    )

    # Update status separately if needed
    if new_status != 'extraction_complete':
        await page_repo.update_status(page_id, new_status)

    return {
        "success": True,
        "task_id": task_id,
        "page_updated": True,
        "extracted_fields": {
            "title": bool(metadata.title),
            "description": bool(metadata.description),
            "author": bool(metadata.author),
            "thumbnail": bool(metadata.thumbnail),
            "content_text": metadata.word_count > 0
        }
    }


class FailTaskRequest(BaseModel):
    error_message: str


@router.post("/rogue/tasks/{task_id}/fail")
async def fail_rogue_task(task_id: str, request: FailTaskRequest):
    """
    Mark a rogue extraction task as failed

    Browser extension calls this if extraction fails (CAPTCHA, login wall, etc.)

    Args:
        task_id: Task ID
        request: JSON body with error_message

    Returns:
        Success status
    """
    _, rogue_task_repo = await init_services()

    await rogue_task_repo.mark_failed(task_id, request.error_message)

    return {"success": True, "task_id": task_id}


@router.get("/rogue/stats")
async def get_rogue_stats():
    """
    Get statistics about rogue extraction queue

    Returns counts by status for monitoring
    """
    _, rogue_task_repo = await init_services()

    # Get stats by status
    stats_by_status = await rogue_task_repo.get_stats()

    # Get stuck tasks count (processing for > 5 minutes)
    stuck_count = await rogue_task_repo.get_stuck_count(hours=5/60)  # 5 minutes in hours

    # Calculate total
    total = sum(stats_by_status.values())

    return {
        "by_status": stats_by_status,
        "stuck_tasks": stuck_count,
        "total": total
    }
