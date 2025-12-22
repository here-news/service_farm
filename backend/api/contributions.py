"""
Contributions API - Handle user source submissions

POST /api/contributions/submit - Submit a new source URL
GET /api/contributions/mine - Get user's contribution history
"""
import os
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
import asyncpg
import redis.asyncio as redis

router = APIRouter()

# Connection pool
db_pool = None
redis_client = None


class ContributionSubmission(BaseModel):
    url: str
    eventId: str
    questType: Optional[str] = None
    note: Optional[str] = None


class ContributionResponse(BaseModel):
    id: str
    status: str
    message: str


async def init_db():
    """Initialize database connection"""
    global db_pool, redis_client

    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'herenews_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
            database=os.getenv('POSTGRES_DB', 'herenews'),
            min_size=1,
            max_size=5
        )

    if redis_client is None:
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        redis_client = redis.from_url(redis_url, decode_responses=True)

    return db_pool, redis_client


@router.post("/contributions/submit", response_model=ContributionResponse)
async def submit_contribution(submission: ContributionSubmission):
    """
    Submit a new source URL for processing.

    The URL will be:
    1. Validated
    2. Checked for duplicates
    3. Queued for extraction
    4. Associated with the event and quest type

    Returns a contribution ID for tracking.
    """
    db_pool, redis_client = await init_db()

    # Validate URL format
    url = submission.url.strip()
    if not url.startswith('http://') and not url.startswith('https://'):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Generate contribution ID
    contribution_id = f"ct_{uuid.uuid4().hex[:8]}"

    # Check for duplicate URL (already submitted recently)
    async with db_pool.acquire() as conn:
        # Check if URL already exists as a page
        existing = await conn.fetchrow(
            "SELECT id FROM pages WHERE url = $1 OR canonical_url = $1",
            url
        )

        if existing:
            # URL already processed - but still accept as contribution intent
            return ContributionResponse(
                id=contribution_id,
                status="duplicate",
                message="This source has already been submitted. You may still earn credits if it's linked to this event."
            )

    # Queue the URL for extraction via Redis
    job = {
        "url": url,
        "contribution_id": contribution_id,
        "event_id": submission.eventId,
        "quest_type": submission.questType,
        "note": submission.note,
        "submitted_at": datetime.utcnow().isoformat(),
        "source": "user_contribution"
    }

    await redis_client.lpush("queue:extraction:high", str(job))

    return ContributionResponse(
        id=contribution_id,
        status="queued",
        message="Your source has been submitted for processing. You'll be notified once it's verified."
    )


@router.get("/contributions/mine")
async def get_my_contributions():
    """
    Get user's contribution history.

    Placeholder - requires auth system integration.
    """
    # TODO: Implement with auth
    return {
        "contributions": [],
        "total": 0,
        "total_credits": 0
    }
