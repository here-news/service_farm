"""
Surface Repository - Neo4j + PostgreSQL storage for surfaces

Storage strategy:
- Neo4j: Surface nodes, CONTAINS edges to Claims, entity links
- PostgreSQL: Surface centroid embeddings for similarity search (pgvector)

The repository abstracts storage from consumers - they work with Surface domain model.

ID format: sf_xxxxxxxx (11 chars)
"""
import logging
from typing import Optional, List, Set, Tuple, Dict
from datetime import datetime, timedelta
import asyncpg
import numpy as np

from models.domain.surface import Surface
from models.domain.claim import Claim
from services.neo4j_service import Neo4jService
from utils.id_generator import generate_id

logger = logging.getLogger(__name__)


class SurfaceRepository:
    """
    Repository for Surface domain model.

    Neo4j is primary store for surface structure.
    PostgreSQL stores centroids for similarity search.
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    # =========================================================================
    # CREATE / UPDATE OPERATIONS
    # =========================================================================

    async def save(self, surface: Surface) -> Surface:
        """
        Save surface to Neo4j + centroid to PostgreSQL.

        Args:
            surface: Surface domain model

        Returns:
            Saved surface with timestamps
        """
        now = datetime.utcnow()
        if not surface.created_at:
            surface.created_at = now
        surface.updated_at = now

        # Compute support if not set
        if surface.support == 0.0:
            surface.compute_support()

        # Neo4j: Surface node
        await self.neo4j._execute_write("""
            MERGE (s:Surface {id: $id})
            SET s.claim_count = $claim_count,
                s.source_count = $source_count,
                s.entities = $entities,
                s.anchor_entities = $anchors,
                s.sources = $sources,
                s.support = $support,
                s.time_start = $time_start,
                s.time_end = $time_end,
                s.params_version = $params_version,
                s.created_at = coalesce(s.created_at, datetime()),
                s.updated_at = datetime()
        """, {
            'id': surface.id,
            'claim_count': len(surface.claim_ids),
            'source_count': len(surface.sources),
            'entities': list(surface.entities),
            'anchors': list(surface.anchor_entities),
            'sources': list(surface.sources),
            'support': surface.support,
            'time_start': surface.time_start.isoformat() if surface.time_start else None,
            'time_end': surface.time_end.isoformat() if surface.time_end else None,
            'params_version': surface.params_version,
        })

        # Neo4j: CONTAINS edges with similarity weights
        if surface.claim_ids:
            # Build list of {claim_id, similarity} for UNWIND
            claim_data = [
                {'cid': cid, 'sim': surface.claim_similarities.get(cid, 1.0)}
                for cid in surface.claim_ids
            ]
            await self.neo4j._execute_write("""
                MATCH (s:Surface {id: $sid})
                UNWIND $claims as cd
                MATCH (c:Claim {id: cd.cid})
                MERGE (s)-[r:CONTAINS]->(c)
                SET r.similarity = cd.sim
            """, {'sid': surface.id, 'claims': claim_data})

        # PostgreSQL: Centroid for similarity search
        if surface.centroid:
            await self._store_centroid(surface)

        # PostgreSQL: Claim-surface mapping
        await self._update_claim_surface_mapping(surface)

        logger.debug(f"💾 Saved surface {surface.id} ({len(surface.claim_ids)} claims)")
        return surface

    async def _store_centroid(self, surface: Surface) -> None:
        """Store surface centroid in PostgreSQL."""
        from pgvector.asyncpg import register_vector

        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("""
                INSERT INTO content.surface_centroids
                    (surface_id, centroid, claim_count, source_count, time_start, time_end)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (surface_id) DO UPDATE SET
                    centroid = $2,
                    claim_count = $3,
                    source_count = $4,
                    time_start = $5,
                    time_end = $6,
                    updated_at = NOW()
            """,
                surface.id,
                surface.centroid,  # pgvector handles list/array after register_vector
                len(surface.claim_ids),
                len(surface.sources),
                surface.time_start,
                surface.time_end,
            )

    async def _update_claim_surface_mapping(self, surface: Surface) -> None:
        """Update claim-surface mapping in PostgreSQL."""
        if not surface.claim_ids:
            return

        async with self.db_pool.acquire() as conn:
            # Batch upsert
            await conn.executemany("""
                INSERT INTO content.claim_surfaces (claim_id, surface_id)
                VALUES ($1, $2)
                ON CONFLICT (claim_id) DO UPDATE SET surface_id = $2
            """, [(cid, surface.id) for cid in surface.claim_ids])

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, surface_id: str) -> Optional[Surface]:
        """
        Retrieve surface by ID from Neo4j.

        Args:
            surface_id: Surface ID (sf_xxxxxxxx format)

        Returns:
            Surface model or None
        """
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface {id: $id})
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            WITH s, collect(c.id) as claim_ids
            RETURN s.id as id,
                   claim_ids,
                   s.entities as entities,
                   s.anchor_entities as anchors,
                   s.sources as sources,
                   s.support as support,
                   s.time_start as time_start,
                   s.time_end as time_end,
                   s.params_version as params_version,
                   s.created_at as created_at,
                   s.updated_at as updated_at
        """, {'id': surface_id})

        if not results:
            return None

        row = results[0]
        surface = Surface(
            id=row['id'],
            claim_ids=set(row['claim_ids'] or []),
            entities=set(row['entities'] or []),
            anchor_entities=set(row['anchors'] or []),
            sources=set(row['sources'] or []),
            support=row['support'] or 0.0,
            time_start=row['time_start'],
            time_end=row['time_end'],
            params_version=row['params_version'] or 1,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

        # Load centroid from PostgreSQL
        surface.centroid = await self._get_centroid(surface_id)

        return surface

    async def get_with_claims(self, surface_id: str) -> Optional[Surface]:
        """
        Retrieve surface with hydrated claims and similarity weights.

        Args:
            surface_id: Surface ID (sf_xxxxxxxx format)

        Returns:
            Surface model with claims populated, or None
        """
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface {id: $id})
            OPTIONAL MATCH (s)-[r:CONTAINS]->(c:Claim)
            OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
            WITH s, c, p, r
            ORDER BY c.created_at
            WITH s, collect({
                id: c.id,
                text: c.text,
                event_time: c.event_time,
                reported_time: c.reported_time,
                confidence: c.confidence,
                modality: c.modality,
                page_id: p.id,
                source: p.domain,
                created_at: c.created_at,
                similarity: r.similarity
            }) as claims_data
            RETURN s.id as id,
                   s.entities as entities,
                   s.anchor_entities as anchors,
                   s.sources as sources,
                   s.support as support,
                   s.time_start as time_start,
                   s.time_end as time_end,
                   s.params_version as params_version,
                   s.created_at as created_at,
                   s.updated_at as updated_at,
                   claims_data
        """, {'id': surface_id})

        if not results:
            return None

        row = results[0]

        # Build Claim objects from data
        from models.domain.claim import Claim
        claims = []
        claim_ids = set()
        claim_similarities = {}

        for cd in row['claims_data']:
            if cd.get('id'):  # Skip nulls from OPTIONAL MATCH
                cid = cd['id']
                claim_ids.add(cid)
                claim_similarities[cid] = cd.get('similarity') or 1.0
                claims.append(Claim(
                    id=cid,
                    text=cd.get('text', ''),
                    page_id=cd.get('page_id'),
                    event_time=cd.get('event_time'),
                    reported_time=cd.get('reported_time'),
                    confidence=cd.get('confidence') or 0.8,
                    modality=cd.get('modality') or 'observation',
                    created_at=cd.get('created_at'),
                ))

        surface = Surface(
            id=row['id'],
            claim_ids=claim_ids,
            claim_similarities=claim_similarities,
            entities=set(row['entities'] or []),
            anchor_entities=set(row['anchors'] or []),
            sources=set(row['sources'] or []),
            support=row['support'] or 0.0,
            time_start=row['time_start'],
            time_end=row['time_end'],
            params_version=row['params_version'] or 1,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            claims=claims,  # Hydrated!
        )

        # Load centroid from PostgreSQL
        surface.centroid = await self._get_centroid(surface_id)

        return surface

    async def hydrate_claims(self, surface: Surface) -> Surface:
        """
        Fetch and attach claims to an existing surface with similarity weights.

        Args:
            surface: Surface to hydrate

        Returns:
            Surface with claims and claim_similarities populated
        """
        if surface.claims is not None:
            return surface  # Already hydrated

        results = await self.neo4j._execute_read("""
            MATCH (s:Surface {id: $id})-[r:CONTAINS]->(c:Claim)
            OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
            RETURN c.id as id,
                   c.text as text,
                   c.event_time as event_time,
                   c.reported_time as reported_time,
                   c.confidence as confidence,
                   c.modality as modality,
                   p.id as page_id,
                   p.domain as source,
                   c.created_at as created_at,
                   r.similarity as similarity
            ORDER BY c.created_at
        """, {'id': surface.id})

        from models.domain.claim import Claim
        surface.claims = []

        for row in results:
            cid = row['id']
            surface.claim_similarities[cid] = row.get('similarity') or 1.0
            surface.claims.append(Claim(
                id=cid,
                text=row.get('text', ''),
                page_id=row.get('page_id'),
                event_time=row.get('event_time'),
                reported_time=row.get('reported_time'),
                confidence=row.get('confidence') or 0.8,
                modality=row.get('modality') or 'observation',
                created_at=row.get('created_at'),
            ))

        logger.debug(f"✅ Hydrated {len(surface.claims)} claims for surface {surface.id}")
        return surface

    async def _get_centroid(self, surface_id: str) -> Optional[List[float]]:
        """Get surface centroid from PostgreSQL."""
        from pgvector.asyncpg import register_vector

        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            result = await conn.fetchval("""
                SELECT centroid FROM content.surface_centroids WHERE surface_id = $1
            """, surface_id)

            if result is not None:
                # pgvector returns numpy array after register_vector
                if hasattr(result, 'tolist'):
                    return result.tolist()
                elif isinstance(result, (list, tuple)):
                    return list(result)
            return None

    async def get_by_ids(self, surface_ids: List[str]) -> List[Surface]:
        """Retrieve multiple surfaces by ID."""
        if not surface_ids:
            return []

        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            WHERE s.id IN $ids
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            WITH s, collect(c.id) as claim_ids
            RETURN s.id as id,
                   claim_ids,
                   s.entities as entities,
                   s.anchor_entities as anchors,
                   s.sources as sources,
                   s.support as support,
                   s.time_start as time_start,
                   s.time_end as time_end
        """, {'ids': surface_ids})

        # Batch load centroids from PostgreSQL
        centroids = await self._get_centroids_batch(surface_ids)

        surfaces = []
        for row in results:
            surface = Surface(
                id=row['id'],
                claim_ids=set(row['claim_ids'] or []),
                entities=set(row['entities'] or []),
                anchor_entities=set(row['anchors'] or []),
                sources=set(row['sources'] or []),
                support=row['support'] or 0.0,
                time_start=row['time_start'],
                time_end=row['time_end'],
            )
            surface.centroid = centroids.get(row['id'])
            surfaces.append(surface)

        return surfaces

    async def _get_centroids_batch(self, surface_ids: List[str]) -> Dict[str, List[float]]:
        """Batch load centroids from PostgreSQL."""
        from pgvector.asyncpg import register_vector

        if not surface_ids:
            return {}

        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            results = await conn.fetch("""
                SELECT surface_id, centroid
                FROM content.surface_centroids
                WHERE surface_id = ANY($1)
            """, surface_ids)

            centroids = {}
            for r in results:
                if r['centroid'] is not None:
                    if hasattr(r['centroid'], 'tolist'):
                        centroids[r['surface_id']] = r['centroid'].tolist()
                    else:
                        centroids[r['surface_id']] = list(r['centroid'])
            return centroids

    # =========================================================================
    # CANDIDATE RETRIEVAL (for weaving)
    # =========================================================================

    async def find_candidates_by_embedding(
        self,
        embedding: List[float],
        time_window: Tuple[datetime, datetime],
        limit: int = 20,
    ) -> List[str]:
        """
        Find candidate surfaces by embedding similarity.

        Uses pgvector for efficient similarity search.

        Args:
            embedding: Query embedding vector
            time_window: (start, end) for temporal filtering
            limit: Maximum candidates to return

        Returns:
            List of surface IDs sorted by similarity
        """
        from pgvector.asyncpg import register_vector

        async with self.db_pool.acquire() as conn:
            await register_vector(conn)

            results = await conn.fetch("""
                SELECT surface_id,
                       1 - (centroid <=> $1::vector) as similarity
                FROM content.surface_centroids
                WHERE (time_end IS NULL OR time_end > $2)
                  AND (time_start IS NULL OR time_start < $3)
                ORDER BY centroid <=> $1::vector
                LIMIT $4
            """, embedding, time_window[0], time_window[1], limit)

            return [r['surface_id'] for r in results]

    async def find_candidates_by_anchor(
        self,
        anchors: Set[str],
        time_window: Tuple[datetime, datetime],
        limit: int = 20,
    ) -> List[str]:
        """
        Find surfaces sharing anchor entities.

        Uses Neo4j for entity-based lookup.

        Args:
            anchors: Set of anchor entity IDs/names
            time_window: (start, end) for temporal filtering
            limit: Maximum candidates to return

        Returns:
            List of surface IDs
        """
        if not anchors:
            return []

        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            WHERE any(a IN $anchors WHERE a IN s.anchor_entities)
              AND (s.time_end IS NULL OR s.time_end > $t_start)
              AND (s.time_start IS NULL OR s.time_start < $t_end)
            RETURN s.id as id, s.support as support
            ORDER BY s.support DESC
            LIMIT $limit
        """, {
            'anchors': list(anchors),
            't_start': time_window[0].isoformat(),
            't_end': time_window[1].isoformat(),
            'limit': limit,
        })

        return [r['id'] for r in results]

    async def find_candidates_for_claim(
        self,
        claim: Claim,
        time_window_days: int = 14,
        limit: int = 20,
    ) -> List[str]:
        """
        Find candidate surfaces for a claim.

        Combines embedding similarity + anchor overlap.

        Args:
            claim: Claim to find candidates for
            time_window_days: Days before/after claim time
            limit: Maximum candidates

        Returns:
            List of surface IDs (deduplicated, sorted by relevance)
        """
        # Time fallback chain: event_time → reported_time → created_at
        claim_time = claim.event_time or claim.reported_time or claim.created_at or datetime.utcnow()
        if isinstance(claim_time, str):
            from dateutil.parser import parse as parse_date
            claim_time = parse_date(claim_time)
        time_window = (
            claim_time - timedelta(days=time_window_days),
            claim_time + timedelta(days=time_window_days),
        )

        candidates = set()

        # By embedding similarity (if claim has embedding)
        if claim.embedding:
            by_embedding = await self.find_candidates_by_embedding(
                claim.embedding, time_window, limit
            )
            candidates.update(by_embedding)

        # By anchor entities (if claim has anchors)
        anchors = getattr(claim, 'anchor_entities', None) or set()
        if not anchors and hasattr(claim, 'entity_ids'):
            anchors = set(claim.entity_ids)

        if anchors:
            by_anchor = await self.find_candidates_by_anchor(
                anchors, time_window, limit
            )
            candidates.update(by_anchor)

        return list(candidates)[:limit]

    # =========================================================================
    # UTILITY
    # =========================================================================

    async def get_surface_for_claim(self, claim_id: str) -> Optional[str]:
        """Get surface ID for a claim."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT surface_id FROM content.claim_surfaces WHERE claim_id = $1
            """, claim_id)

    async def delete(self, surface_id: str) -> None:
        """Delete a surface from both stores."""
        # Neo4j
        await self.neo4j._execute_write("""
            MATCH (s:Surface {id: $id})
            DETACH DELETE s
        """, {'id': surface_id})

        # PostgreSQL
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM content.surface_centroids WHERE surface_id = $1",
                surface_id
            )
            await conn.execute(
                "DELETE FROM content.claim_surfaces WHERE surface_id = $1",
                surface_id
            )

        logger.debug(f"🗑️ Deleted surface {surface_id}")

    async def count_all(self) -> int:
        """Count all surfaces."""
        results = await self.neo4j._execute_read(
            "MATCH (s:Surface) RETURN count(s) as count"
        )
        return results[0]['count'] if results else 0
