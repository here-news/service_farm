"""
Phase Repository - Dual-write to PostgreSQL and Neo4j

Storage strategy:
- PostgreSQL: Phase embedding (core.event_phases table)
- Neo4j: Phase nodes with relationships to events and claims
"""
import uuid
import logging
from typing import Optional, List
import asyncpg

from models.phase import Phase
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class PhaseRepository:
    """
    Repository for Phase domain model

    Handles dual-write to PostgreSQL (embedding) and Neo4j (graph)
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    async def create(self, phase: Phase) -> Phase:
        """
        Create phase in both PostgreSQL and Neo4j

        Args:
            phase: Phase domain model

        Returns:
            Created phase with timestamps
        """
        # Create in Neo4j (graph structure)
        await self.neo4j.create_phase(
            phase_id=str(phase.id),
            event_id=str(phase.event_id),
            name=phase.name,
            phase_type=phase.phase_type,
            start_time=phase.start_time,
            end_time=phase.end_time,
            confidence=phase.confidence,
            sequence=phase.sequence
        )

        # Store embedding in PostgreSQL
        if phase.embedding:
            embedding_str = '[' + ','.join(str(x) for x in phase.embedding) + ']'

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO core.event_phases (
                        id, event_id, name, phase_type, sequence, embedding, description
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::vector, $7)
                """,
                    phase.id,
                    phase.event_id,
                    phase.name,
                    phase.phase_type,
                    phase.sequence,
                    embedding_str,
                    phase.description
                )

                # Fetch timestamps
                row = await conn.fetchrow("""
                    SELECT created_at, updated_at FROM core.event_phases WHERE id = $1
                """, phase.id)

                phase.created_at = row['created_at']
                phase.updated_at = row['updated_at']

        logger.debug(f"📍 Created phase: {phase.name}")
        return phase

    async def get_by_id(self, phase_id: uuid.UUID) -> Optional[Phase]:
        """
        Retrieve phase from PostgreSQL by ID

        Args:
            phase_id: Phase UUID

        Returns:
            Phase model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id, event_id, name, phase_type, sequence,
                    embedding, description, created_at, updated_at
                FROM core.event_phases
                WHERE id = $1
            """, phase_id)

            if not row:
                return None

            # Parse embedding
            embedding = None
            if row['embedding']:
                embedding = self._parse_embedding(row['embedding'])

            return Phase(
                id=row['id'],
                event_id=row['event_id'],
                name=row['name'],
                phase_type=row['phase_type'],
                sequence=row['sequence'],
                embedding=embedding,
                description=row['description'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def get_by_event(self, event_id: uuid.UUID) -> List[Phase]:
        """
        Retrieve all phases for an event

        Args:
            event_id: Event UUID

        Returns:
            List of Phase models ordered by sequence
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id, event_id, name, phase_type, sequence,
                    embedding, description, created_at, updated_at
                FROM core.event_phases
                WHERE event_id = $1
                ORDER BY sequence
            """, event_id)

            phases = []
            for row in rows:
                embedding = None
                if row['embedding']:
                    embedding = self._parse_embedding(row['embedding'])

                phases.append(Phase(
                    id=row['id'],
                    event_id=row['event_id'],
                    name=row['name'],
                    phase_type=row['phase_type'],
                    sequence=row['sequence'],
                    embedding=embedding,
                    description=row['description'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))

            return phases

    def _parse_embedding(self, emb_str: str) -> Optional[List[float]]:
        """Parse embedding from PostgreSQL vector string"""
        try:
            if emb_str.startswith('[') and emb_str.endswith(']'):
                return [float(x.strip()) for x in emb_str[1:-1].split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
        return None
