"""
Rogue URL Extraction Endpoints - Browser Extension Integration

Endpoints for browser extension to handle URLs that block scrapers (401/403)

Architecture:
1. ExtractionWorker marks failed URLs as "rogue" → creates task
2. Browser extension polls GET /rogue/tasks for pending tasks
3. Extension extracts metadata in real browser → POST /rogue/tasks/{id}/complete
4. System uses extracted metadata for preview
"""
import uuid
import json
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncpg

router = APIRouter()

# Global database pool (initialized on startup)
db_pool = None


async def init_db_pool():
    """Initialize database pool"""
    global db_pool
    if db_pool is None:
        import os
        db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'herenews_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
            database=os.getenv('POSTGRES_DB', 'herenews'),
            min_size=2,
            max_size=10
        )
    return db_pool


class RogueTask(BaseModel):
    """Rogue extraction task for browser extension"""
    id: str
    page_id: str
    url: str
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
async def get_pending_rogue_tasks(limit: int = 1):
    """
    Get pending rogue extraction tasks for browser extension to process

    Browser extension polls this endpoint every 3 seconds.
    Returns oldest pending task (FIFO).

    Query params:
        limit: Number of tasks to return (default: 1)

    Returns:
        List of pending tasks
    """
    pool = await init_db_pool()

    async with pool.acquire() as conn:
        # Get pending tasks, mark as processing
        tasks = await conn.fetch("""
            SELECT id, page_id, url, created_at
            FROM core.rogue_extraction_tasks
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT $1
        """, limit)

        if not tasks:
            return []

        # Mark as processing with timeout (60 seconds)
        task_ids = [task['id'] for task in tasks]
        await conn.execute("""
            UPDATE core.rogue_extraction_tasks
            SET status = 'processing',
                processing_started_at = NOW()
            WHERE id = ANY($1::uuid[])
        """, task_ids)

        return [
            RogueTask(
                id=str(task['id']),
                page_id=str(task['page_id']),
                url=task['url'],
                created_at=task['created_at'].isoformat()
            )
            for task in tasks
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
    pool = await init_db_pool()

    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format")

    async with pool.acquire() as conn:
        # Check task exists and is processing
        task = await conn.fetchrow("""
            SELECT page_id, status
            FROM core.rogue_extraction_tasks
            WHERE id = $1
        """, task_uuid)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task['status'] not in ['pending', 'processing']:
            raise HTTPException(status_code=400, detail=f"Task already {task['status']}")

        # Update task with metadata (convert to JSON for JSONB column)
        await conn.execute("""
            UPDATE core.rogue_extraction_tasks
            SET status = 'completed',
                metadata = $2,
                completed_at = NOW()
            WHERE id = $1
        """, task_uuid, json.dumps(metadata.dict()))

        # Update page with extracted data
        page_id = task['page_id']

        update_fields = []
        params = [page_id]
        param_idx = 2

        if metadata.title:
            update_fields.append(f"title = ${param_idx}")
            params.append(metadata.title)
            param_idx += 1

        if metadata.description:
            update_fields.append(f"description = ${param_idx}")
            params.append(metadata.description)
            param_idx += 1

        if metadata.author:
            update_fields.append(f"author = ${param_idx}")
            params.append(metadata.author)
            param_idx += 1

        if metadata.thumbnail:
            update_fields.append(f"thumbnail_url = ${param_idx}")
            params.append(metadata.thumbnail)
            param_idx += 1

        if metadata.content_text and metadata.word_count >= 100:
            update_fields.append(f"content_text = ${param_idx}")
            params.append(metadata.content_text)
            param_idx += 1
            update_fields.append(f"word_count = ${param_idx}")
            params.append(metadata.word_count)
            param_idx += 1
            update_fields.append(f"status = ${param_idx}")
            params.append('extracted')
            param_idx += 1
        else:
            # At least we have metadata, mark as preview
            if task['status'] != 'extracted':
                update_fields.append(f"status = ${param_idx}")
                params.append('preview')
                param_idx += 1

        update_fields.append("updated_at = NOW()")

        if update_fields:
            query = f"""
                UPDATE core.pages
                SET {', '.join(update_fields)}
                WHERE id = $1
            """
            await conn.execute(query, *params)

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


@router.post("/rogue/tasks/{task_id}/fail")
async def fail_rogue_task(task_id: str, error_message: str):
    """
    Mark a rogue extraction task as failed

    Browser extension calls this if extraction fails (CAPTCHA, login wall, etc.)

    Args:
        task_id: Task UUID
        error_message: Error description

    Returns:
        Success status
    """
    pool = await init_db_pool()

    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format")

    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE core.rogue_extraction_tasks
            SET status = 'failed',
                error_message = $2,
                completed_at = NOW()
            WHERE id = $1
        """, task_uuid, error_message)

        return {"success": True, "task_id": task_id}


@router.get("/rogue/stats")
async def get_rogue_stats():
    """
    Get statistics about rogue extraction queue

    Returns counts by status for monitoring
    """
    pool = await init_db_pool()

    async with pool.acquire() as conn:
        stats = await conn.fetch("""
            SELECT status, COUNT(*) as count
            FROM core.rogue_extraction_tasks
            GROUP BY status
        """)

        # Get stuck tasks (processing for > 5 minutes)
        stuck_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM core.rogue_extraction_tasks
            WHERE status = 'processing'
            AND processing_started_at < NOW() - INTERVAL '5 minutes'
        """)

        return {
            "by_status": {row['status']: row['count'] for row in stats},
            "stuck_tasks": stuck_count,
            "total": sum(row['count'] for row in stats)
        }
