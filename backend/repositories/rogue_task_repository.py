"""
RogueTask Repository - PostgreSQL storage for browser-based extraction tasks

Storage strategy:
- PostgreSQL: rogue_extraction_tasks table
- Handles tasks for pages that need browser-based extraction (paywall, bot-blocked, etc.)
"""
import uuid
import logging
from typing import Optional, List
import asyncpg
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RogueTaskRepository:
    """
    Repository for RogueTask management

    Handles tasks for browser extension to extract content
    from pages that block automated scrapers
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def get_pending_tasks(self, limit: int = 1) -> List[dict]:
        """
        Get pending rogue extraction tasks and mark them as processing

        Args:
            limit: Number of tasks to return

        Returns:
            List of task dictionaries
        """
        async with self.db_pool.acquire() as conn:
            # Atomically get pending tasks and mark as processing
            tasks = await conn.fetch("""
                UPDATE core.rogue_extraction_tasks
                SET status = 'processing', processing_started_at = NOW()
                WHERE id IN (
                    SELECT id FROM core.rogue_extraction_tasks
                    WHERE status = 'pending'
                    ORDER BY created_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, page_id, url, status, created_at, processing_started_at
            """, limit)

            return [dict(task) for task in tasks]

    async def get_recent_tasks(self, limit: int = 10, status: Optional[str] = None) -> List[dict]:
        """
        Get recent rogue extraction tasks for UI display

        Args:
            limit: Number of tasks to return
            status: Optional status filter (pending, processing, completed, failed)

        Returns:
            List of task dictionaries
        """
        async with self.db_pool.acquire() as conn:
            if status:
                tasks = await conn.fetch("""
                    SELECT id, page_id, url, status, created_at, completed_at
                    FROM core.rogue_extraction_tasks
                    WHERE status = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, status, limit)
            else:
                tasks = await conn.fetch("""
                    SELECT id, page_id, url, status, created_at, completed_at
                    FROM core.rogue_extraction_tasks
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)

            return [dict(task) for task in tasks]

    async def get_by_id(self, task_id: uuid.UUID) -> Optional[dict]:
        """
        Get rogue task by ID

        Args:
            task_id: Task UUID

        Returns:
            Task dictionary or None
        """
        async with self.db_pool.acquire() as conn:
            task = await conn.fetchrow("""
                SELECT id, page_id, url, status, created_at, completed_at
                FROM core.rogue_extraction_tasks
                WHERE id = $1
            """, task_id)

            return dict(task) if task else None

    async def mark_completed(self, task_id: uuid.UUID) -> None:
        """
        Mark rogue task as completed

        Args:
            task_id: Task UUID
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.rogue_extraction_tasks
                SET status = 'completed', completed_at = NOW()
                WHERE id = $1
            """, task_id)

            logger.info(f"✅ Marked rogue task {task_id} as completed")

    async def mark_failed(self, task_id: uuid.UUID, error_message: Optional[str] = None) -> None:
        """
        Mark rogue task as failed

        Args:
            task_id: Task UUID
            error_message: Optional error message
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.rogue_extraction_tasks
                SET status = 'failed', error_message = $2, completed_at = NOW()
                WHERE id = $1
            """, task_id, error_message)

            logger.error(f"❌ Marked rogue task {task_id} as failed: {error_message or 'Unknown error'}")

    async def get_stats(self) -> dict:
        """
        Get statistics about rogue extraction tasks

        Returns:
            Dictionary with counts by status
        """
        async with self.db_pool.acquire() as conn:
            stats = await conn.fetch("""
                SELECT status, COUNT(*) as count
                FROM core.rogue_extraction_tasks
                GROUP BY status
            """)

            return {row['status']: row['count'] for row in stats}

    async def get_stuck_count(self, hours: int = 1) -> int:
        """
        Get count of stuck tasks (processing for > N hours)

        Args:
            hours: Hours threshold for stuck detection

        Returns:
            Count of stuck tasks
        """
        async with self.db_pool.acquire() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM core.rogue_extraction_tasks
                WHERE status = 'processing'
                  AND processing_started_at < NOW() - INTERVAL '%s hours'
            """ % hours)

            return count or 0

    async def reset_stuck_tasks(self, hours: int = 1) -> int:
        """
        Reset stuck tasks back to pending

        Args:
            hours: Hours threshold for stuck detection

        Returns:
            Number of tasks reset
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE core.rogue_extraction_tasks
                SET status = 'pending', processing_started_at = NULL
                WHERE status = 'processing'
                  AND processing_started_at < NOW() - INTERVAL '%s hours'
            """ % hours)

            # Extract count from "UPDATE N" result
            count = int(result.split()[-1]) if result else 0

            if count > 0:
                logger.info(f"🔄 Reset {count} stuck rogue tasks")

            return count
