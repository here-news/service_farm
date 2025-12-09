"""
User Repository - PostgreSQL storage for user accounts

Storage: PostgreSQL (users table)
"""
import logging
from typing import Optional, List
import asyncpg
from datetime import datetime

from models.domain.user import User

logger = logging.getLogger(__name__)


class UserRepository:
    """
    Repository for User domain model

    Handles user account management and authentication data.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """
        Retrieve user by ID.

        Args:
            user_id: User UUID

        Returns:
            User model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT user_id, email, google_id, name, picture_url,
                       credits_balance, reputation, subscription_tier, is_active,
                       created_at, last_login
                FROM users
                WHERE user_id = $1
            """, user_id)

            if not row:
                return None

            return User(
                user_id=str(row['user_id']),
                email=row['email'],
                google_id=row['google_id'],
                name=row['name'],
                picture_url=row['picture_url'],
                credits_balance=row['credits_balance'] or 1000,
                reputation=row['reputation'] or 0,
                subscription_tier=row['subscription_tier'],
                is_active=bool(row['is_active']),
                created_at=row['created_at'],
                last_login=row['last_login']
            )

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Retrieve user by email.

        Args:
            email: User email address

        Returns:
            User model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT user_id FROM users WHERE email = $1
            """, email)

            if not row:
                return None

            return await self.get_by_id(str(row['user_id']))

    async def get_by_google_id(self, google_id: str) -> Optional[User]:
        """
        Retrieve user by Google OAuth ID.

        Args:
            google_id: Google user ID (from OAuth)

        Returns:
            User model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT user_id FROM users WHERE google_id = $1
            """, google_id)

            if not row:
                return None

            return await self.get_by_id(str(row['user_id']))

    # =========================================================================
    # CREATE OPERATION
    # =========================================================================

    async def create(self, user: User) -> User:
        """
        Create a new user.

        Args:
            user: User model (id will be generated if not set)

        Returns:
            Created user with ID
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO users (
                    user_id, email, google_id, name, picture_url,
                    credits_balance, reputation, subscription_tier, is_active,
                    created_at, last_login
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
                RETURNING user_id, created_at, last_login
            """,
                user.user_id,
                user.email,
                user.google_id,
                user.name,
                user.picture_url,
                user.credits_balance,
                user.reputation,
                user.subscription_tier,
                user.is_active
            )

            # Update user with database-generated timestamps
            user.created_at = row['created_at']
            user.last_login = row['last_login']

            logger.info(f"Created user {user.user_id} ({user.email})")
            return user

    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================

    async def update(self, user: User) -> User:
        """
        Update user data.

        Args:
            user: User model with updated fields

        Returns:
            Updated user
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users
                SET name = $2,
                    picture_url = $3,
                    credits_balance = $4,
                    reputation = $5,
                    subscription_tier = $6,
                    is_active = $7
                WHERE user_id = $1
            """,
                user.user_id,
                user.name,
                user.picture_url,
                user.credits_balance,
                user.reputation,
                user.subscription_tier,
                user.is_active
            )

            logger.info(f"Updated user {user.user_id}")
            return user

    async def update_last_login(self, user_id: str) -> None:
        """
        Update user's last login timestamp.

        Args:
            user_id: User UUID
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET last_login = NOW() WHERE user_id = $1
            """, user_id)

    async def update_credits(self, user_id: str, new_balance: int) -> None:
        """
        Update user's credit balance.

        Args:
            user_id: User UUID
            new_balance: New credit balance
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET credits_balance = $2 WHERE user_id = $1
            """, user_id, new_balance)

    async def deduct_credits(self, user_id: str, amount: int) -> bool:
        """
        Atomically deduct credits from user balance.

        Args:
            user_id: User UUID
            amount: Credits to deduct

        Returns:
            True if successful, False if insufficient credits
        """
        async with self.db_pool.acquire() as conn:
            # Use atomic UPDATE with WHERE clause to ensure sufficient balance
            result = await conn.execute("""
                UPDATE users
                SET credits_balance = credits_balance - $2
                WHERE user_id = $1 AND credits_balance >= $2
            """, user_id, amount)

            # Check if update happened (status string like "UPDATE 1" or "UPDATE 0")
            rows_updated = int(result.split()[-1])
            return rows_updated > 0

    # =========================================================================
    # LIST OPERATIONS
    # =========================================================================

    async def list_active_users(self, limit: int = 100) -> List[User]:
        """
        List active users (for admin purposes).

        Args:
            limit: Maximum number of users to return

        Returns:
            List of active users
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT user_id FROM users
                WHERE is_active = true
                ORDER BY created_at DESC
                LIMIT $1
            """, limit)

            users = []
            for row in rows:
                user = await self.get_by_id(str(row['user_id']))
                if user:
                    users.append(user)

            return users
