"""
Chat Session Repository - PostgreSQL storage for AI chat sessions

Storage: PostgreSQL (chat_sessions table)
"""
import logging
from typing import Optional, List
import asyncpg
from datetime import datetime

from models.domain.chat_session import ChatSession, ChatSessionStatus

logger = logging.getLogger(__name__)


class ChatSessionRepository:
    """
    Repository for ChatSession domain model

    Handles premium AI chat sessions for events.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve chat session by ID.

        Args:
            session_id: Session ID (cs_xxxxxxxx)

        Returns:
            ChatSession model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, event_id, user_id, message_count, cost, status,
                       unlocked_at, last_message_at, created_at
                FROM chat_sessions
                WHERE id = $1
            """, session_id)

            if not row:
                return None

            return ChatSession(
                id=row['id'],
                event_id=row['event_id'],
                user_id=str(row['user_id']),
                message_count=row['message_count'] or 0,
                cost=row['cost'] or 10,
                status=ChatSessionStatus(row['status']) if row['status'] else ChatSessionStatus.ACTIVE,
                unlocked_at=row['unlocked_at'],
                last_message_at=row['last_message_at'],
                created_at=row['created_at']
            )

    async def get_by_event_and_user(self, event_id: str, user_id: str) -> Optional[ChatSession]:
        """
        Get chat session for a specific event and user.

        Args:
            event_id: Event ID (ev_xxxxxxxx)
            user_id: User UUID

        Returns:
            ChatSession model or None if not unlocked
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id FROM chat_sessions
                WHERE event_id = $1 AND user_id = $2
            """, event_id, user_id)

            if not row:
                return None

            return await self.get_by_id(row['id'])

    async def get_by_user(self, user_id: str, limit: int = 50) -> List[ChatSession]:
        """
        Get all chat sessions for a user.

        Args:
            user_id: User UUID
            limit: Maximum number of sessions

        Returns:
            List of chat sessions, sorted by creation time (newest first)
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id FROM chat_sessions
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)

            sessions = []
            for row in rows:
                session = await self.get_by_id(row['id'])
                if session:
                    sessions.append(session)

            return sessions

    # =========================================================================
    # CREATE OPERATION
    # =========================================================================

    async def create(self, session: ChatSession, deduct_credits_from_user: str) -> ChatSession:
        """
        Create a new chat session and deduct credits from user.

        Uses a transaction to ensure atomicity:
        1. Deduct credits from user
        2. Create chat session

        Args:
            session: ChatSession model
            deduct_credits_from_user: User ID to deduct credits from

        Returns:
            Created session with timestamps

        Raises:
            ValueError: If user has insufficient credits
        """
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # 1. Atomically deduct credits
                result = await conn.execute("""
                    UPDATE users
                    SET credits_balance = credits_balance - $2
                    WHERE user_id = $1 AND credits_balance >= $2
                """, deduct_credits_from_user, session.cost)

                rows_updated = int(result.split()[-1])
                if rows_updated == 0:
                    raise ValueError(f"Insufficient credits. Need {session.cost} credits to unlock chat.")

                # 2. Create session
                row = await conn.fetchrow("""
                    INSERT INTO chat_sessions (
                        id, event_id, user_id, message_count, cost, status,
                        unlocked_at, last_message_at, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, NOW(), NULL, NOW())
                    RETURNING unlocked_at, created_at
                """,
                    session.id,
                    session.event_id,
                    session.user_id,
                    session.message_count,
                    session.cost,
                    session.status.value
                )

                # Update session with database-generated timestamps
                session.unlocked_at = row['unlocked_at']
                session.created_at = row['created_at']

                logger.info(f"Created chat session {session.id} for event {session.event_id} (user: {session.user_id})")
                return session

    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================

    async def increment_message_count(self, session_id: str, max_messages: int = 100) -> ChatSession:
        """
        Increment message count and update status if exhausted.

        Args:
            session_id: Session ID
            max_messages: Maximum allowed messages

        Returns:
            Updated session

        Raises:
            ValueError: If session is already exhausted
        """
        async with self.db_pool.acquire() as conn:
            # Atomically increment if not exhausted
            row = await conn.fetchrow("""
                UPDATE chat_sessions
                SET message_count = message_count + 1,
                    last_message_at = NOW(),
                    status = CASE
                        WHEN message_count + 1 >= $2 THEN 'exhausted'
                        ELSE status
                    END
                WHERE id = $1 AND status = 'active'
                RETURNING message_count, status, last_message_at
            """, session_id, max_messages)

            if not row:
                raise ValueError("Session is already exhausted or does not exist")

            # Fetch and return updated session
            session = await self.get_by_id(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            logger.info(f"Incremented message count for session {session_id}: {session.message_count}/{max_messages}")
            return session

    async def update_status(self, session_id: str, status: ChatSessionStatus) -> None:
        """
        Update session status.

        Args:
            session_id: Session ID
            status: New status
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE chat_sessions SET status = $2 WHERE id = $1
            """, session_id, status.value)

            logger.info(f"Updated session {session_id} status to {status.value}")

    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================

    async def get_remaining_messages(self, session_id: str, max_messages: int = 100) -> int:
        """
        Get remaining message count for a session.

        Args:
            session_id: Session ID
            max_messages: Maximum allowed messages

        Returns:
            Remaining messages
        """
        session = await self.get_by_id(session_id)
        if not session:
            return 0

        return max(0, max_messages - session.message_count)
