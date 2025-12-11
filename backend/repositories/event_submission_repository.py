"""
Event Submission Repository - handles user-submitted events/tips.

This is a stub implementation to allow auth to load.
Full implementation will store submissions in PostgreSQL.
"""
import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EventSubmission:
    """A user-submitted event/tip"""
    submission_id: str
    user_id: str
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    status: str = "pending"  # pending, approved, rejected, processing
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    event_id: Optional[str] = None  # If submission led to event creation


class EventSubmissionRepository:
    """Repository for event submissions"""

    def __init__(self, pool):
        """
        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def create(
        self,
        user_id: str,
        url: str,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> EventSubmission:
        """Create a new submission"""
        import uuid
        submission_id = f"sub_{uuid.uuid4().hex[:8]}"

        # TODO: Store in PostgreSQL
        logger.info(f"📝 Submission created: {submission_id} by {user_id}")

        return EventSubmission(
            submission_id=submission_id,
            user_id=user_id,
            url=url,
            title=title,
            description=description,
            created_at=datetime.utcnow()
        )

    async def get_by_id(self, submission_id: str) -> Optional[EventSubmission]:
        """Get submission by ID"""
        # TODO: Implement
        return None

    async def get_by_user(self, user_id: str, limit: int = 50) -> List[EventSubmission]:
        """Get submissions by user"""
        # TODO: Implement
        return []

    async def update_status(
        self,
        submission_id: str,
        status: str,
        event_id: Optional[str] = None
    ) -> bool:
        """Update submission status"""
        # TODO: Implement
        logger.info(f"📝 Submission {submission_id} status -> {status}")
        return True

    async def get_pending(self, limit: int = 100) -> List[EventSubmission]:
        """Get pending submissions for processing"""
        # TODO: Implement
        return []
