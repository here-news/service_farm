"""
Inquiry Repository - PostgreSQL storage for Inquiry MVP

Handles persistence of inquiries, contributions, stakes, and tasks.
"""
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
import asyncpg
import json

logger = logging.getLogger(__name__)


def _generate_id(prefix: str) -> str:
    """Generate a prefixed UUID."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class InquiryRepository:
    """Repository for Inquiry domain models."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    # =========================================================================
    # INQUIRY OPERATIONS
    # =========================================================================

    async def create_inquiry(
        self,
        title: str,
        description: str = "",
        schema_type: str = "boolean",
        schema_config: Dict = None,
        rigor_level: str = "B",
        scope_entities: List[str] = None,
        scope_keywords: List[str] = None,
        created_by: str = None,
    ) -> Dict:
        """Create a new inquiry."""
        inquiry_id = _generate_id("inq")

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO inquiries (
                    id, title, description, status, rigor_level,
                    schema_type, schema_config, scope_entities, scope_keywords,
                    created_by, created_at, updated_at
                )
                VALUES ($1, $2, $3, 'open', $4, $5, $6, $7, $8, $9, now(), now())
                RETURNING *
            """,
                inquiry_id,
                title,
                description,
                rigor_level,
                schema_type,
                json.dumps(schema_config or {}),
                scope_entities or [],
                scope_keywords or [],
                created_by
            )

            logger.info(f"Created inquiry {inquiry_id}: {title[:50]}")
            return self._row_to_inquiry(row)

    async def get_inquiry(self, inquiry_id: str) -> Optional[Dict]:
        """Get inquiry by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM inquiries WHERE id = $1
            """, inquiry_id)

            if not row:
                return None
            return self._row_to_inquiry(row)

    async def list_inquiries(
        self,
        status: str = None,
        order_by: str = "stake",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """List inquiries with filtering and sorting."""
        order_clause = {
            "stake": "total_stake DESC",
            "entropy": "entropy_bits DESC",
            "contributions": "contribution_count DESC",
            "created": "created_at DESC",
        }.get(order_by, "total_stake DESC")

        async with self.db_pool.acquire() as conn:
            if status:
                rows = await conn.fetch(f"""
                    SELECT * FROM inquiries
                    WHERE status = $1
                    ORDER BY {order_clause}
                    LIMIT $2 OFFSET $3
                """, status, limit, offset)
            else:
                rows = await conn.fetch(f"""
                    SELECT * FROM inquiries
                    ORDER BY {order_clause}
                    LIMIT $1 OFFSET $2
                """, limit, offset)

            return [self._row_to_inquiry(row) for row in rows]

    async def update_belief_state(
        self,
        inquiry_id: str,
        posterior_map: Any,
        posterior_prob: float,
        entropy_bits: float,
        normalized_entropy: float = 0.0,
        credible_interval: tuple = None
    ) -> bool:
        """Update inquiry belief state after processing contributions."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE inquiries SET
                    posterior_map = $2,
                    posterior_prob = $3,
                    entropy_bits = $4,
                    normalized_entropy = $5,
                    credible_interval = $6,
                    updated_at = now()
                WHERE id = $1
            """,
                inquiry_id,
                json.dumps(posterior_map),
                posterior_prob,
                entropy_bits,
                normalized_entropy,
                json.dumps(credible_interval) if credible_interval else None
            )
            return "UPDATE 1" in result

    async def update_status(self, inquiry_id: str, status: str) -> bool:
        """Update inquiry status."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE inquiries SET
                    status = $2,
                    resolved_at = CASE WHEN $2 = 'resolved' THEN now() ELSE resolved_at END,
                    updated_at = now()
                WHERE id = $1
            """, inquiry_id, status)
            return "UPDATE 1" in result

    # =========================================================================
    # STAKE OPERATIONS
    # =========================================================================

    async def add_stake(
        self,
        inquiry_id: str,
        user_id: str,
        amount: float
    ) -> Dict:
        """Add or update stake from a user. Returns new stake record."""
        async with self.db_pool.acquire() as conn:
            # Use upsert to handle existing stake
            row = await conn.fetchrow("""
                INSERT INTO inquiry_stakes (inquiry_id, user_id, amount)
                VALUES ($1, $2, $3)
                ON CONFLICT (inquiry_id, user_id)
                DO UPDATE SET amount = inquiry_stakes.amount + $3
                RETURNING *
            """, inquiry_id, user_id, amount)

            logger.info(f"Added stake ${amount} to inquiry {inquiry_id} by user {user_id}")
            return {
                "id": row["id"],
                "inquiry_id": row["inquiry_id"],
                "user_id": str(row["user_id"]),
                "amount": float(row["amount"]),
                "created_at": row["created_at"].isoformat()
            }

    async def get_stakes(self, inquiry_id: str) -> List[Dict]:
        """Get all stakes for an inquiry."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT s.*, u.name as user_name
                FROM inquiry_stakes s
                LEFT JOIN users u ON s.user_id = u.user_id
                WHERE s.inquiry_id = $1
                ORDER BY s.amount DESC
            """, inquiry_id)

            return [
                {
                    "user_id": str(row["user_id"]),
                    "user_name": row["user_name"] or "Anonymous",
                    "amount": float(row["amount"]),
                    "created_at": row["created_at"].isoformat()
                }
                for row in rows
            ]

    async def get_total_stake(self, inquiry_id: str) -> float:
        """Get total stake for an inquiry."""
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COALESCE(SUM(amount), 0) FROM inquiry_stakes WHERE inquiry_id = $1
            """, inquiry_id)
            return float(result)

    # =========================================================================
    # CONTRIBUTION OPERATIONS
    # =========================================================================

    async def add_contribution(
        self,
        inquiry_id: str,
        user_id: str,
        contribution_type: str,
        text: str,
        source_url: str = None,
        source_name: str = None,
        extracted_value: Any = None,
        observation_kind: str = None
    ) -> Dict:
        """Add a contribution to an inquiry."""
        contrib_id = _generate_id("contrib")

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO contributions (
                    id, inquiry_id, user_id, contribution_type,
                    text, source_url, source_name,
                    extracted_value, observation_kind,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
                RETURNING *
            """,
                contrib_id,
                inquiry_id,
                user_id,
                contribution_type,
                text,
                source_url,
                source_name,
                json.dumps(extracted_value) if extracted_value else None,
                observation_kind
            )

            logger.info(f"Added contribution {contrib_id} to inquiry {inquiry_id}")
            return self._row_to_contribution(row)

    async def get_contributions(
        self,
        inquiry_id: str,
        order_by: str = "recent",
        limit: int = 50
    ) -> List[Dict]:
        """Get contributions for an inquiry."""
        order_clause = "created_at DESC" if order_by == "recent" else "posterior_impact DESC"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT c.*, u.name as user_name, u.picture_url as user_picture
                FROM contributions c
                LEFT JOIN users u ON c.user_id = u.user_id
                WHERE c.inquiry_id = $1
                ORDER BY {order_clause}
                LIMIT $2
            """, inquiry_id, limit)

            return [self._row_to_contribution(row) for row in rows]

    async def update_contribution_impact(
        self,
        contribution_id: str,
        posterior_impact: float,
        processed: bool = True,
        claim_ids: List[str] = None
    ) -> bool:
        """Update contribution after processing."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE contributions SET
                    posterior_impact = $2,
                    processed = $3,
                    claim_ids = $4
                WHERE id = $1
            """,
                contribution_id,
                posterior_impact,
                processed,
                claim_ids or []
            )
            return "UPDATE 1" in result

    async def award_contribution_reward(
        self,
        contribution_id: str,
        reward_amount: float
    ) -> bool:
        """Award reward to a contribution."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE contributions SET reward_earned = reward_earned + $2
                WHERE id = $1
            """, contribution_id, reward_amount)
            return "UPDATE 1" in result

    # =========================================================================
    # TASK OPERATIONS
    # =========================================================================

    async def create_task(
        self,
        inquiry_id: str,
        task_type: str,
        description: str,
        bounty: float = 0.0,
        meta_claim_id: str = None
    ) -> Dict:
        """Create a task for an inquiry."""
        task_id = _generate_id("task")

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO inquiry_tasks (
                    id, inquiry_id, task_type, description, bounty, meta_claim_id
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING *
            """, task_id, inquiry_id, task_type, description, bounty, meta_claim_id)

            return self._row_to_task(row)

    async def get_tasks(
        self,
        inquiry_id: str,
        include_completed: bool = False
    ) -> List[Dict]:
        """Get tasks for an inquiry."""
        async with self.db_pool.acquire() as conn:
            if include_completed:
                rows = await conn.fetch("""
                    SELECT t.*, u.name as claimed_by_name
                    FROM inquiry_tasks t
                    LEFT JOIN users u ON t.claimed_by = u.user_id
                    WHERE t.inquiry_id = $1
                    ORDER BY t.completed, t.bounty DESC
                """, inquiry_id)
            else:
                rows = await conn.fetch("""
                    SELECT t.*, u.name as claimed_by_name
                    FROM inquiry_tasks t
                    LEFT JOIN users u ON t.claimed_by = u.user_id
                    WHERE t.inquiry_id = $1 AND t.completed = false
                    ORDER BY t.bounty DESC
                """, inquiry_id)

            return [self._row_to_task(row) for row in rows]

    async def claim_task(self, task_id: str, user_id: str) -> bool:
        """Claim a task."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE inquiry_tasks SET
                    claimed_by = $2,
                    claimed_at = now()
                WHERE id = $1 AND claimed_by IS NULL AND completed = false
            """, task_id, user_id)
            return "UPDATE 1" in result

    async def complete_task(
        self,
        task_id: str,
        contribution_id: str = None
    ) -> bool:
        """Mark task as completed."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE inquiry_tasks SET
                    completed = true,
                    completed_at = now(),
                    completion_contribution_id = $2
                WHERE id = $1
            """, task_id, contribution_id)
            return "UPDATE 1" in result

    # =========================================================================
    # CREDIT TRANSACTION OPERATIONS
    # =========================================================================

    async def record_transaction(
        self,
        user_id: str,
        amount: float,
        balance_after: float,
        transaction_type: str,
        reference_type: str = None,
        reference_id: str = None,
        description: str = None
    ) -> Dict:
        """Record a credit transaction."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO credit_transactions (
                    user_id, amount, balance_after, transaction_type,
                    reference_type, reference_id, description
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
            """,
                user_id,
                amount,
                balance_after,
                transaction_type,
                reference_type,
                reference_id,
                description
            )

            return {
                "id": row["id"],
                "user_id": str(row["user_id"]),
                "amount": float(row["amount"]),
                "balance_after": float(row["balance_after"]),
                "transaction_type": row["transaction_type"],
                "created_at": row["created_at"].isoformat()
            }

    async def get_user_transactions(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """Get credit transactions for a user."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM credit_transactions
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)

            return [
                {
                    "id": row["id"],
                    "amount": float(row["amount"]),
                    "balance_after": float(row["balance_after"]),
                    "transaction_type": row["transaction_type"],
                    "reference_type": row["reference_type"],
                    "reference_id": row["reference_id"],
                    "description": row["description"],
                    "created_at": row["created_at"].isoformat()
                }
                for row in rows
            ]

    # =========================================================================
    # TRACE OPERATIONS (for full epistemic trace)
    # =========================================================================

    async def get_trace(self, inquiry_id: str) -> Optional[Dict]:
        """Get full epistemic trace for an inquiry."""
        inquiry = await self.get_inquiry(inquiry_id)
        if not inquiry:
            return None

        contributions = await self.get_contributions(inquiry_id, limit=100)
        tasks = await self.get_tasks(inquiry_id, include_completed=True)
        stakes = await self.get_stakes(inquiry_id)

        return {
            "inquiry": inquiry,
            "belief_state": {
                "map": inquiry.get("posterior_map"),
                "map_probability": inquiry.get("posterior_prob", 0),
                "entropy_bits": inquiry.get("entropy_bits", 0),
                "normalized_entropy": inquiry.get("normalized_entropy", 0),
                "observation_count": inquiry.get("contribution_count", 0),
                "total_log_score": -inquiry.get("entropy_bits", 0) * 2
            },
            "contributions": contributions,
            "tasks": tasks,
            "stakes": stakes,
            "resolution": {
                "status": inquiry.get("status"),
                "resolvable": inquiry.get("posterior_prob", 0) >= 0.95,
                "stable_since": inquiry.get("stable_since"),
                "blocking_tasks": [t["id"] for t in tasks if not t["completed"]]
            }
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _row_to_inquiry(self, row: asyncpg.Record) -> Dict:
        """Convert database row to inquiry dict."""
        return {
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "status": row["status"],
            "rigor_level": row["rigor_level"],
            "schema_type": row["schema_type"],
            "schema_config": json.loads(row["schema_config"]) if row["schema_config"] else {},
            "scope_entities": list(row["scope_entities"] or []),
            "scope_keywords": list(row["scope_keywords"] or []),
            "posterior_map": json.loads(row["posterior_map"]) if row["posterior_map"] else None,
            "posterior_prob": float(row["posterior_prob"] or 0),
            "entropy_bits": float(row["entropy_bits"] or 0),
            "normalized_entropy": float(row["normalized_entropy"] or 0),
            "credible_interval": json.loads(row["credible_interval"]) if row["credible_interval"] else None,
            "total_stake": float(row["total_stake"] or 0),
            "distributed": float(row["distributed"] or 0),
            "contribution_count": row["contribution_count"] or 0,
            "open_tasks_count": row["open_tasks_count"] or 0,
            "created_by": str(row["created_by"]) if row["created_by"] else None,
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "resolved_at": row["resolved_at"].isoformat() if row["resolved_at"] else None,
            "stable_since": row["stable_since"].isoformat() if row["stable_since"] else None,
        }

    def _row_to_contribution(self, row: asyncpg.Record) -> Dict:
        """Convert database row to contribution dict."""
        return {
            "id": row["id"],
            "inquiry_id": row["inquiry_id"],
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "user_name": row.get("user_name") or "Anonymous",
            "user_picture": row.get("user_picture"),
            "type": row["contribution_type"],
            "text": row["text"],
            "source_url": row["source_url"],
            "source_name": row["source_name"],
            "extracted_value": json.loads(row["extracted_value"]) if row["extracted_value"] else None,
            "observation_kind": row["observation_kind"],
            "processed": row["processed"],
            "claim_ids": list(row["claim_ids"] or []),
            "posterior_impact": float(row["posterior_impact"] or 0),
            "impact": float(row["posterior_impact"] or 0),  # Alias for frontend
            "reward_earned": float(row["reward_earned"] or 0),
            "created_at": row["created_at"].isoformat(),
        }

    def _row_to_task(self, row: asyncpg.Record) -> Dict:
        """Convert database row to task dict."""
        return {
            "id": row["id"],
            "inquiry_id": row["inquiry_id"],
            "type": row["task_type"],
            "description": row["description"],
            "bounty": float(row["bounty"] or 0),
            "claimed_by": str(row["claimed_by"]) if row["claimed_by"] else None,
            "claimed_by_name": row.get("claimed_by_name"),
            "claimed_at": row["claimed_at"].isoformat() if row["claimed_at"] else None,
            "completed": row["completed"],
            "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
            "meta_claim_id": row["meta_claim_id"],
            "created_at": row["created_at"].isoformat(),
        }
