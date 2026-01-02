"""
Surface Repository - Neo4j + PostgreSQL storage for L2 Surfaces

Surfaces are derived from claims + parameters. They are materialized views,
not source-of-truth. The repository handles:
- Persisting computed surfaces
- Versioning (params_version tracking)
- Invalidation on parameter changes
- Query patterns for inquiry emergence
- Centroid loading from PostgreSQL for semantic coverage

ID format: sf_xxxxxxxx (11 chars)
"""
import logging
import os
from typing import Optional, List, Dict, Set, Tuple

import asyncpg

from ..types import Surface, Relation, Parameters
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class SurfaceRepository:
    """
    Repository for Surface (L2) persistence in Neo4j + PostgreSQL.

    Surfaces are computed by IdentityLinker and persisted here.
    They are versioned by params_version and can be invalidated/recomputed.
    Centroids are stored in PostgreSQL (content.surface_centroids).
    """

    def __init__(self, neo4j_service: Neo4jService, db_pool: asyncpg.Pool = None):
        self.neo4j = neo4j_service
        self.db_pool = db_pool  # Optional: for centroid loading

    # =========================================================================
    # CENTROID LOADING (PostgreSQL)
    # =========================================================================

    async def _load_centroid(self, surface_id: str) -> Optional[List[float]]:
        """Load centroid for a single surface from PostgreSQL."""
        if not self.db_pool:
            return None

        from pgvector.asyncpg import register_vector
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            result = await conn.fetchval('''
                SELECT centroid FROM content.surface_centroids
                WHERE surface_id = $1
            ''', surface_id)

            if result is not None:
                if hasattr(result, 'tolist'):
                    return result.tolist()
                elif isinstance(result, (list, tuple)):
                    return list(result)
            return None

    async def _load_centroids_batch(self, surface_ids: List[str]) -> Dict[str, List[float]]:
        """Batch load centroids for multiple surfaces from PostgreSQL."""
        if not self.db_pool or not surface_ids:
            return {}

        from pgvector.asyncpg import register_vector
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch('''
                SELECT surface_id, centroid FROM content.surface_centroids
                WHERE surface_id = ANY($1) AND centroid IS NOT NULL
            ''', surface_ids)

            centroids = {}
            for row in rows:
                centroid = row['centroid']
                if hasattr(centroid, 'tolist'):
                    centroids[row['surface_id']] = centroid.tolist()
                elif isinstance(centroid, (list, tuple)):
                    centroids[row['surface_id']] = list(centroid)
            return centroids

    async def hydrate_centroids(self, surfaces: List[Surface]) -> None:
        """Hydrate centroids for a list of surfaces from PostgreSQL."""
        if not surfaces:
            return

        surface_ids = [s.id for s in surfaces]
        centroids = await self._load_centroids_batch(surface_ids)

        for surface in surfaces:
            if surface.id in centroids:
                surface.centroid = centroids[surface.id]

    # =========================================================================
    # CREATE / UPDATE
    # =========================================================================

    async def persist(
        self,
        surface: Surface,
        params_version: int,
        event_id: Optional[str] = None,
    ) -> Surface:
        """
        Persist a computed surface to Neo4j.

        Creates Surface node and CONTAINS relationships to claims.
        Optionally links to Event via BELONGS_TO.

        Args:
            surface: Surface domain model from IdentityLinker
            params_version: Version of parameters used for computation
            event_id: Optional event to attach surface to

        Returns:
            Persisted surface
        """
        # Extract typed surface properties if present
        question_key = None
        canonical_value = None

        # Create/update Surface node
        await self.neo4j._execute_write("""
            MERGE (s:Surface {id: $id})
            SET s.params_version = $params_version,
                s.computed_at = datetime(),
                s.entropy = $entropy,
                s.mass = $mass,
                s.claim_count = $claim_count,
                s.source_count = $source_count,
                s.sources = $sources,
                s.time_start = $time_start,
                s.time_end = $time_end,
                s.canonical_title = $canonical_title,
                s.description = $description,
                s.status = 'active'
        """, {
            'id': surface.id,
            'params_version': params_version,
            'entropy': surface.entropy,
            'mass': surface.mass,
            'claim_count': len(surface.claim_ids),
            'source_count': len(surface.sources),
            'sources': list(surface.sources),
            'time_start': surface.time_window[0].isoformat() if surface.time_window[0] else None,
            'time_end': surface.time_window[1].isoformat() if surface.time_window[1] else None,
            'canonical_title': surface.canonical_title,
            'description': surface.description,
        })

        # Link to claims via CONTAINS
        if surface.claim_ids:
            await self.neo4j._execute_write("""
                MATCH (s:Surface {id: $surface_id})
                UNWIND $claim_ids as claim_id
                MATCH (c:Claim {id: claim_id})
                MERGE (s)-[:CONTAINS]->(c)
            """, {
                'surface_id': surface.id,
                'claim_ids': list(surface.claim_ids),
            })

        # Link to entities via ABOUT
        if surface.entities:
            await self.neo4j._execute_write("""
                MATCH (s:Surface {id: $surface_id})
                UNWIND $entity_names as entity_name
                MATCH (e:Entity {canonical_name: entity_name})
                MERGE (s)-[:ABOUT]->(e)
            """, {
                'surface_id': surface.id,
                'entity_names': list(surface.entities),
            })

        # Link anchor entities via ANCHORED_BY
        if surface.anchor_entities:
            await self.neo4j._execute_write("""
                MATCH (s:Surface {id: $surface_id})
                UNWIND $entity_names as entity_name
                MATCH (e:Entity {canonical_name: entity_name})
                MERGE (s)-[:ANCHORED_BY]->(e)
            """, {
                'surface_id': surface.id,
                'entity_names': list(surface.anchor_entities),
            })

        # Link to event if provided
        if event_id:
            await self.neo4j._execute_write("""
                MATCH (s:Surface {id: $surface_id})
                MATCH (e:Event {id: $event_id})
                MERGE (s)-[:BELONGS_TO {
                    level: 'core',
                    attached_at: datetime()
                }]->(e)
            """, {
                'surface_id': surface.id,
                'event_id': event_id,
            })

        logger.debug(f"📦 Persisted surface {surface.id}: {len(surface.claim_ids)} claims")
        return surface

    async def persist_identity_edge(
        self,
        claim1_id: str,
        claim2_id: str,
        relation: Relation,
        confidence: float,
        params_version: int,
    ) -> None:
        """
        Persist an identity edge between claims.

        Args:
            claim1_id: Source claim ID
            claim2_id: Target claim ID
            relation: Type of relation (CONFIRMS, REFINES, etc.)
            confidence: Edge confidence
            params_version: Parameters version
        """
        await self.neo4j._execute_write("""
            MATCH (c1:Claim {id: $claim1_id})
            MATCH (c2:Claim {id: $claim2_id})
            MERGE (c1)-[r:IDENTITY]->(c2)
            SET r.relation = $relation,
                r.confidence = $confidence,
                r.params_version = $params_version,
                r.created_at = datetime()
        """, {
            'claim1_id': claim1_id,
            'claim2_id': claim2_id,
            'relation': relation.value,
            'confidence': confidence,
            'params_version': params_version,
        })

    async def persist_surface_relation(
        self,
        surface1_id: str,
        surface2_id: str,
        relation_type: str,
        confidence: float,
        evidence: Dict,
        params_version: int,
    ) -> None:
        """
        Persist a relation between surfaces.

        Args:
            surface1_id: Source surface ID
            surface2_id: Target surface ID
            relation_type: Type of relation (confirms, supersedes, conflicts, refines)
            confidence: Relation confidence
            evidence: Evidence dict
            params_version: Parameters version
        """
        await self.neo4j._execute_write("""
            MATCH (s1:Surface {id: $surface1_id})
            MATCH (s2:Surface {id: $surface2_id})
            MERGE (s1)-[r:SURFACE_REL]->(s2)
            SET r.type = $relation_type,
                r.confidence = $confidence,
                r.evidence = $evidence,
                r.params_version = $params_version,
                r.created_at = datetime()
        """, {
            'surface1_id': surface1_id,
            'surface2_id': surface2_id,
            'relation_type': relation_type,
            'confidence': confidence,
            'evidence': str(evidence),  # Neo4j doesn't support nested maps well
            'params_version': params_version,
        })

    # =========================================================================
    # READ
    # =========================================================================

    async def get_by_id(
        self,
        surface_id: str,
        include_claims: bool = False,
    ) -> Optional[Surface]:
        """
        Get surface by ID.

        Args:
            surface_id: Surface ID (sf_xxxxxxxx format)
            include_claims: Whether to fetch claim details

        Returns:
            Surface or None
        """
        if include_claims:
            results = await self.neo4j._execute_read("""
                MATCH (s:Surface {id: $surface_id})
                OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
                OPTIONAL MATCH (s)-[:ABOUT]->(ent:Entity)
                OPTIONAL MATCH (s)-[:ANCHORED_BY]->(anchor:Entity)
                RETURN s,
                       collect(DISTINCT c.id) as claim_ids,
                       collect(DISTINCT ent.canonical_name) as entities,
                       collect(DISTINCT anchor.canonical_name) as anchor_entities
            """, {'surface_id': surface_id})
        else:
            results = await self.neo4j._execute_read("""
                MATCH (s:Surface {id: $surface_id})
                RETURN s,
                       [] as claim_ids,
                       [] as entities,
                       [] as anchor_entities
            """, {'surface_id': surface_id})

        if not results:
            return None

        row = results[0]
        s = row['s']

        surface = Surface(
            id=s['id'],
            claim_ids=set(row['claim_ids']),
            entropy=s.get('entropy', 0.0),
            mass=s.get('mass', 0.0),
            sources=set(s.get('sources', [])),
            entities=set(row['entities']),
            anchor_entities=set(row['anchor_entities']),
            canonical_title=s.get('canonical_title'),
            description=s.get('description'),
        )

        # Load centroid from PostgreSQL if available
        surface.centroid = await self._load_centroid(surface_id)

        return surface

    async def get_by_event(
        self,
        event_id: str,
        params_version: Optional[int] = None,
        status: str = 'active',
    ) -> List[Surface]:
        """
        Get all surfaces for an event.

        Args:
            event_id: Event ID
            params_version: Optional params version filter
            status: Surface status filter (default: active)

        Returns:
            List of surfaces
        """
        query = """
            MATCH (e:Event {id: $event_id})<-[:BELONGS_TO]-(s:Surface)
            WHERE s.status = $status
        """
        params = {'event_id': event_id, 'status': status}

        if params_version is not None:
            query += " AND s.params_version = $params_version"
            params['params_version'] = params_version

        query += """
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            RETURN s,
                   collect(DISTINCT c.id) as claim_ids
            ORDER BY s.mass DESC
        """

        results = await self.neo4j._execute_read(query, params)

        surfaces = []
        for row in results:
            s = row['s']
            surfaces.append(Surface(
                id=s['id'],
                claim_ids=set(row['claim_ids']),
                entropy=s.get('entropy', 0.0),
                mass=s.get('mass', 0.0),
                sources=set(s.get('sources', [])),
                canonical_title=s.get('canonical_title'),
            ))

        # Hydrate centroids from PostgreSQL
        await self.hydrate_centroids(surfaces)

        return surfaces

    async def get_typed_surfaces(
        self,
        params_version: Optional[int] = None,
        question_key: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get surfaces with typed observations (for inquiry emergence).

        Args:
            params_version: Optional params version filter
            question_key: Optional question key filter (e.g., 'death_count')

        Returns:
            List of dicts with surface info and typed properties
        """
        query = """
            MATCH (s:Surface)
            WHERE s.status = 'active'
              AND s.question_key IS NOT NULL
        """
        params = {}

        if params_version is not None:
            query += " AND s.params_version = $params_version"
            params['params_version'] = params_version

        if question_key:
            query += " AND s.question_key = $question_key"
            params['question_key'] = question_key

        query += """
            RETURN s.id as id,
                   s.question_key as question_key,
                   s.canonical_value as canonical_value,
                   s.value_entropy as value_entropy,
                   s.claim_count as claim_count
            ORDER BY s.value_entropy DESC
        """

        results = await self.neo4j._execute_read(query, params)
        return [dict(r) for r in results]

    async def get_conflicting_surfaces(
        self,
        params_version: Optional[int] = None,
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Get pairs of conflicting surfaces.

        Args:
            params_version: Optional params version filter

        Returns:
            List of (surface1_id, surface2_id, confidence, evidence) tuples
        """
        query = """
            MATCH (s1:Surface)-[r:SURFACE_REL {type: 'conflicts'}]->(s2:Surface)
            WHERE s1.status = 'active' AND s2.status = 'active'
        """
        params = {}

        if params_version is not None:
            query += " AND r.params_version = $params_version"
            params['params_version'] = params_version

        query += """
            RETURN s1.id as surface1_id,
                   s2.id as surface2_id,
                   r.confidence as confidence,
                   r.evidence as evidence
        """

        results = await self.neo4j._execute_read(query, params)
        return [
            (r['surface1_id'], r['surface2_id'], r['confidence'], r['evidence'])
            for r in results
        ]

    async def get_claim_ids_for_surface(self, surface_id: str) -> Set[str]:
        """
        Get claim IDs contained in a surface.

        Args:
            surface_id: Surface ID

        Returns:
            Set of claim IDs
        """
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface {id: $surface_id})-[:CONTAINS]->(c:Claim)
            RETURN c.id as claim_id
        """, {'surface_id': surface_id})

        return {r['claim_id'] for r in results}

    # =========================================================================
    # VERSIONING / LIFECYCLE
    # =========================================================================

    async def invalidate_by_version(self, params_version: int) -> int:
        """
        Mark surfaces computed with given params version as stale.

        Args:
            params_version: Version to invalidate

        Returns:
            Number of surfaces invalidated
        """
        result = await self.neo4j._execute_write("""
            MATCH (s:Surface)
            WHERE s.params_version = $params_version AND s.status = 'active'
            SET s.status = 'stale'
            RETURN count(s) as count
        """, {'params_version': params_version})

        count = result[0]['count'] if result else 0
        logger.info(f"🗑️ Invalidated {count} surfaces with params_version={params_version}")
        return count

    async def invalidate_for_event(self, event_id: str) -> int:
        """
        Mark all surfaces for an event as stale (before recomputation).

        Args:
            event_id: Event ID

        Returns:
            Number of surfaces invalidated
        """
        result = await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})<-[:BELONGS_TO]-(s:Surface)
            WHERE s.status = 'active'
            SET s.status = 'stale'
            RETURN count(s) as count
        """, {'event_id': event_id})

        count = result[0]['count'] if result else 0
        logger.info(f"🗑️ Invalidated {count} surfaces for event {event_id}")
        return count

    async def delete_stale(self, older_than_days: int = 7) -> int:
        """
        Delete stale surfaces older than threshold.

        Args:
            older_than_days: Delete surfaces stale for longer than this

        Returns:
            Number of surfaces deleted
        """
        result = await self.neo4j._execute_write("""
            MATCH (s:Surface)
            WHERE s.status = 'stale'
              AND s.computed_at < datetime() - duration({days: $days})
            DETACH DELETE s
            RETURN count(s) as count
        """, {'days': older_than_days})

        count = result[0]['count'] if result else 0
        logger.info(f"🧹 Deleted {count} stale surfaces older than {older_than_days} days")
        return count

    # =========================================================================
    # STATISTICS
    # =========================================================================

    async def count_by_status(self) -> Dict[str, int]:
        """Get surface counts by status."""
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            RETURN s.status as status, count(s) as count
        """)

        return {r['status']: r['count'] for r in results}

    async def count_by_version(self) -> Dict[int, int]:
        """Get surface counts by params version."""
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            WHERE s.status = 'active'
            RETURN s.params_version as version, count(s) as count
            ORDER BY version DESC
        """)

        return {r['version']: r['count'] for r in results}
