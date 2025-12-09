"""
Comment Repository - PostgreSQL storage for comments

Storage: PostgreSQL (comments table)
"""
import logging
from typing import Optional, List
import asyncpg
from datetime import datetime

from models.domain.comment import Comment, ReactionType

logger = logging.getLogger(__name__)


class CommentRepository:
    """
    Repository for Comment domain model

    Handles threaded comments on events and pages.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, comment_id: str) -> Optional[Comment]:
        """
        Retrieve comment by ID.

        Args:
            comment_id: Comment ID (cm_xxxxxxxx)

        Returns:
            Comment model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, user_id, event_id, page_id, text,
                       parent_comment_id, reaction_type,
                       created_at, updated_at
                FROM comments
                WHERE id = $1
            """, comment_id)

            if not row:
                return None

            return Comment(
                id=row['id'],
                user_id=str(row['user_id']),
                event_id=row['event_id'],
                page_id=row['page_id'],
                text=row['text'],
                parent_comment_id=row['parent_comment_id'],
                reaction_type=ReactionType(row['reaction_type']) if row['reaction_type'] else None,
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def get_by_event(self, event_id: str, limit: int = 100) -> List[Comment]:
        """
        Get all comments for an event.

        Args:
            event_id: Event ID (ev_xxxxxxxx)
            limit: Maximum number of comments

        Returns:
            List of comments, sorted by creation time
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id FROM comments
                WHERE event_id = $1
                ORDER BY created_at ASC
                LIMIT $2
            """, event_id, limit)

            comments = []
            for row in rows:
                comment = await self.get_by_id(row['id'])
                if comment:
                    comments.append(comment)

            return comments

    async def get_by_page(self, page_id: str, limit: int = 100) -> List[Comment]:
        """
        Get all comments for a page.

        Args:
            page_id: Page ID (pg_xxxxxxxx)
            limit: Maximum number of comments

        Returns:
            List of comments, sorted by creation time
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id FROM comments
                WHERE page_id = $1
                ORDER BY created_at ASC
                LIMIT $2
            """, page_id, limit)

            comments = []
            for row in rows:
                comment = await self.get_by_id(row['id'])
                if comment:
                    comments.append(comment)

            return comments

    async def get_by_user(self, user_id: str, limit: int = 50) -> List[Comment]:
        """
        Get all comments by a user.

        Args:
            user_id: User UUID
            limit: Maximum number of comments

        Returns:
            List of comments, sorted by creation time (newest first)
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id FROM comments
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)

            comments = []
            for row in rows:
                comment = await self.get_by_id(row['id'])
                if comment:
                    comments.append(comment)

            return comments

    async def get_replies(self, parent_comment_id: str) -> List[Comment]:
        """
        Get all replies to a comment.

        Args:
            parent_comment_id: Parent comment ID (cm_xxxxxxxx)

        Returns:
            List of reply comments
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id FROM comments
                WHERE parent_comment_id = $1
                ORDER BY created_at ASC
            """, parent_comment_id)

            replies = []
            for row in rows:
                comment = await self.get_by_id(row['id'])
                if comment:
                    replies.append(comment)

            return replies

    async def count_by_event(self, event_id: str) -> int:
        """
        Count comments for an event.

        Args:
            event_id: Event ID (ev_xxxxxxxx)

        Returns:
            Number of comments
        """
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT COUNT(*) FROM comments WHERE event_id = $1
            """, event_id)

    # =========================================================================
    # CREATE OPERATION
    # =========================================================================

    async def create(self, comment: Comment) -> Comment:
        """
        Create a new comment.

        Args:
            comment: Comment model (id will be generated if not set)

        Returns:
            Created comment with timestamps
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO comments (
                    id, user_id, event_id, page_id, text,
                    parent_comment_id, reaction_type,
                    created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
                RETURNING created_at, updated_at
            """,
                comment.id,
                comment.user_id,
                comment.event_id,
                comment.page_id,
                comment.text,
                comment.parent_comment_id,
                comment.reaction_type.value if comment.reaction_type else None
            )

            # Update comment with database-generated timestamps
            comment.created_at = row['created_at']
            comment.updated_at = row['updated_at']

            logger.info(f"Created comment {comment.id} by user {comment.user_id}")
            return comment

    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================

    async def update_text(self, comment_id: str, new_text: str) -> None:
        """
        Update comment text (for edits).

        Args:
            comment_id: Comment ID
            new_text: Updated comment text
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE comments
                SET text = $2, updated_at = NOW()
                WHERE id = $1
            """, comment_id, new_text)

            logger.info(f"Updated comment {comment_id}")

    # =========================================================================
    # DELETE OPERATION
    # =========================================================================

    async def delete(self, comment_id: str, user_id: str) -> bool:
        """
        Delete a comment (only by owner).

        Args:
            comment_id: Comment ID
            user_id: User ID (for ownership check)

        Returns:
            True if deleted, False if not found or unauthorized
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM comments
                WHERE id = $1 AND user_id = $2
            """, comment_id, user_id)

            # Check if delete happened
            rows_deleted = int(result.split()[-1])
            if rows_deleted > 0:
                logger.info(f"Deleted comment {comment_id}")
                return True
            return False
