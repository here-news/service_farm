"""
Story Repository - Unified data access for L3/L4 stories

Queries both :Incident (scale=incident) and :Case (scale=case) nodes,
returning unified Story domain objects.

This replaces separate incident/case queries with a single interface
that downstream (API, frontend) can consume without knowing the internal
distinction.
"""
from typing import Optional, List, Literal
from datetime import datetime
import logging

from services.neo4j_service import Neo4jService
from models.domain.story import Story, StoryScale

logger = logging.getLogger(__name__)


class StoryRepository:
    """
    Unified repository for Stories (L3 incidents + L4 cases).

    Queries:
    - :Incident nodes with scale='incident'
    - :Case nodes with scale='case'

    Returns Story domain objects for API consumption.
    """

    # Minimum incidents required for a valid case
    MIN_CASE_INCIDENTS = 2

    def __init__(self, neo4j: Neo4jService):
        self.neo4j = neo4j

    async def get_by_id(self, story_id: str) -> Optional[Story]:
        """
        Get a story by ID.

        Automatically detects scale from ID prefix or node labels.
        """
        # Try as Case first (has :Case label)
        rows = await self.neo4j._execute_read('''
            MATCH (s)
            WHERE s.id = $id AND (s:Case OR s:Incident)
            OPTIONAL MATCH (s)-[:CONTAINS]->(i:Incident)
            WITH s,
                 CASE WHEN s:Case THEN 'case' ELSE 'incident' END as scale,
                 count(i) as incident_count
            RETURN s.id as id,
                   scale,
                   coalesce(s.title, s.canonical_title) as title,
                   s.description as description,
                   coalesce(s.primary_entities, s.anchor_entities) as primary_entities,
                   coalesce(s.surface_count, 0) as surface_count,
                   coalesce(s.source_count, 0) as source_count,
                   coalesce(s.claim_count, 0) as claim_count,
                   incident_count,
                   s.time_start as time_start,
                   s.time_end as time_end,
                   s.scope_signature as scope_signature,
                   s.case_type as case_type,
                   s.created_at as created_at,
                   s.updated_at as updated_at
        ''', {'id': story_id})

        if not rows:
            return None

        return Story.from_neo4j_row(rows[0])

    async def list_stories(
        self,
        scale: Optional[StoryScale] = None,
        offset: int = 0,
        limit: int = 20,
        entity_filter: Optional[str] = None,
        since: Optional[str] = None,
        case_type: Optional[str] = None,
    ) -> List[Story]:
        """
        List stories with optional scale filter.

        Args:
            scale: 'incident' or 'case' (None = both)
            offset: Pagination offset
            limit: Page size
            entity_filter: Filter by entity name (substring)
            since: Filter by time_start >= date
            case_type: Filter by case_type ('developing', 'entity_storyline')

        Returns:
            List of Story objects, ordered by source_count desc
        """
        where_clauses = []
        params = {
            'offset': offset,
            'limit': limit,
            'min_incidents': self.MIN_CASE_INCIDENTS,
        }

        # Scale filter
        if scale == 'incident':
            where_clauses.append("s:Incident")
        elif scale == 'case':
            where_clauses.append("s:Case")
        else:
            where_clauses.append("(s:Incident OR s:Case)")

        # Entity filter
        if entity_filter:
            where_clauses.append("""
                (any(e IN coalesce(s.primary_entities, s.anchor_entities, []) WHERE e CONTAINS $entity))
            """)
            params['entity'] = entity_filter

        # Time filter
        if since:
            where_clauses.append("s.time_start >= $since")
            params['since'] = since

        # Case type filter (only applies to cases)
        if case_type:
            where_clauses.append("s.case_type = $case_type")
            params['case_type'] = case_type

        where_str = " AND ".join(where_clauses)

        rows = await self.neo4j._execute_read(f'''
            MATCH (s)
            WHERE {where_str}
            OPTIONAL MATCH (s)-[:CONTAINS]->(i:Incident)
            WITH s,
                 CASE WHEN s:Case THEN 'case' ELSE 'incident' END as scale,
                 count(i) as incident_count
            // For cases, require minimum incidents
            WHERE NOT s:Case OR incident_count >= $min_incidents
            RETURN s.id as id,
                   scale,
                   coalesce(s.title, s.canonical_title) as title,
                   s.description as description,
                   coalesce(s.primary_entities, s.anchor_entities) as primary_entities,
                   coalesce(s.surface_count, 0) as surface_count,
                   coalesce(s.source_count, 0) as source_count,
                   coalesce(s.claim_count, 0) as claim_count,
                   incident_count,
                   s.time_start as time_start,
                   s.time_end as time_end,
                   s.scope_signature as scope_signature,
                   s.case_type as case_type,
                   s.updated_at as updated_at
            ORDER BY s.source_count DESC, s.updated_at DESC
            SKIP $offset
            LIMIT $limit
        ''', params)

        return [Story.from_neo4j_row(row) for row in rows]

    async def count_stories(
        self,
        scale: Optional[StoryScale] = None,
        entity_filter: Optional[str] = None,
        since: Optional[str] = None,
        case_type: Optional[str] = None,
    ) -> int:
        """Count stories matching filters."""
        where_clauses = []
        params = {'min_incidents': self.MIN_CASE_INCIDENTS}

        # Scale filter
        if scale == 'incident':
            where_clauses.append("s:Incident")
        elif scale == 'case':
            where_clauses.append("s:Case")
        else:
            where_clauses.append("(s:Incident OR s:Case)")

        # Entity filter
        if entity_filter:
            where_clauses.append("""
                (any(e IN coalesce(s.primary_entities, s.anchor_entities, []) WHERE e CONTAINS $entity))
            """)
            params['entity'] = entity_filter

        # Time filter
        if since:
            where_clauses.append("s.time_start >= $since")
            params['since'] = since

        # Case type filter
        if case_type:
            where_clauses.append("s.case_type = $case_type")
            params['case_type'] = case_type

        where_str = " AND ".join(where_clauses)

        rows = await self.neo4j._execute_read(f'''
            MATCH (s)
            WHERE {where_str}
            OPTIONAL MATCH (s)-[:CONTAINS]->(i:Incident)
            WITH s, count(i) as incident_count
            WHERE NOT s:Case OR incident_count >= $min_incidents
            RETURN count(s) as total
        ''', params)

        return rows[0]['total'] if rows else 0

    async def get_surfaces_for_story(self, story_id: str) -> List[dict]:
        """
        Get surfaces for a story.

        For incidents: direct surfaces via CONTAINS
        For cases: surfaces via incidents
        """
        rows = await self.neo4j._execute_read('''
            MATCH (s {id: $id})
            WHERE s:Incident OR s:Case
            // Handle both incident->surface and case->incident->surface paths
            OPTIONAL MATCH (s)-[:CONTAINS]->(direct:Surface)
            OPTIONAL MATCH (s)-[:CONTAINS]->(i:Incident)-[:CONTAINS]->(via_incident:Surface)
            WITH s, collect(DISTINCT direct) + collect(DISTINCT via_incident) as all_surfaces
            UNWIND all_surfaces as surf
            WITH DISTINCT surf WHERE surf IS NOT NULL
            OPTIONAL MATCH (surf)-[:CONTAINS]->(c:Claim)
            WITH surf, collect(c.text)[0..3] as sample_claims
            RETURN surf.id as id,
                   surf.sources as sources,
                   surf.anchor_entities as anchors,
                   surf.time_start as time_start,
                   sample_claims
            ORDER BY surf.time_start
        ''', {'id': story_id})

        return [
            {
                "id": r['id'],
                "sources": r['sources'] or [],
                "anchors": r['anchors'] or [],
                "time_start": r['time_start'],
                "sample_claims": r['sample_claims'] or [],
            }
            for r in rows
        ]

    async def get_incidents_for_case(self, case_id: str) -> List[Story]:
        """Get incident stories for a case story."""
        rows = await self.neo4j._execute_read('''
            MATCH (c:Case {id: $id})-[:CONTAINS]->(i:Incident)
            OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
            WITH i, count(s) as surface_count
            RETURN i.id as id,
                   'incident' as scale,
                   coalesce(i.title, i.canonical_title) as title,
                   i.description as description,
                   i.anchor_entities as primary_entities,
                   surface_count,
                   0 as source_count,
                   0 as claim_count,
                   0 as incident_count,
                   i.time_start as time_start,
                   i.time_end as time_end,
                   i.id as scope_signature
            ORDER BY i.time_start
        ''', {'id': case_id})

        return [Story.from_neo4j_row(row) for row in rows]

    async def save_story(self, story: Story) -> Story:
        """
        Save a story to Neo4j.

        Creates :Incident or :Case node based on scale.
        """
        if story.scale == 'incident':
            await self._save_incident(story)
        else:
            await self._save_case(story)

        story.updated_at = datetime.utcnow()
        return story

    async def _save_incident(self, story: Story):
        """Save incident story using scope_signature for stability."""
        # MERGE by scope_signature ensures stable IDs across rebuilds
        await self.neo4j._execute_write('''
            MERGE (i:Incident:Story {scope_signature: $scope_signature})
            SET i.id = $id,
                i.scale = 'incident',
                i.title = $title,
                i.canonical_title = $title,
                i.description = $description,
                i.anchor_entities = $primary_entities,
                i.primary_entities = $primary_entities,
                i.surface_count = $surface_count,
                i.source_count = $source_count,
                i.time_start = $time_start,
                i.time_end = $time_end,
                i.updated_at = datetime()
        ''', {
            'id': story.id,
            'scope_signature': story.scope_signature or story.id,
            'title': story.title,
            'description': story.description,
            'primary_entities': story.primary_entities,
            'surface_count': story.surface_count,
            'source_count': story.source_count,
            'time_start': story.time_start.isoformat() if story.time_start else None,
            'time_end': story.time_end.isoformat() if story.time_end else None,
        })

    async def _save_case(self, story: Story):
        """Save case story using scope_signature for stability."""
        # MERGE by scope_signature ensures stable IDs across rebuilds
        await self.neo4j._execute_write('''
            MERGE (c:Case:Story {scope_signature: $scope_signature})
            SET c:Event,
                c.id = $id,
                c.scale = 'case',
                c.title = $title,
                c.canonical_title = $title,
                c.description = $description,
                c.primary_entities = $primary_entities,
                c.surface_count = $surface_count,
                c.source_count = $source_count,
                c.claim_count = $claim_count,
                c.time_start = $time_start,
                c.time_end = $time_end,
                c.updated_at = datetime()
        ''', {
            'id': story.id,
            'scope_signature': story.scope_signature or story.id,
            'title': story.title,
            'description': story.description,
            'primary_entities': story.primary_entities,
            'surface_count': story.surface_count,
            'source_count': story.source_count,
            'claim_count': story.claim_count,
            'time_start': story.time_start.isoformat() if story.time_start else None,
            'time_end': story.time_end.isoformat() if story.time_end else None,
        })

    async def delete_invalid_cases(self) -> int:
        """Delete cases with fewer than MIN_CASE_INCIDENTS incidents."""
        # Count first
        count_rows = await self.neo4j._execute_read('''
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
            WITH c, count(i) as incident_count
            WHERE incident_count < $min_incidents
            RETURN count(c) as count
        ''', {'min_incidents': self.MIN_CASE_INCIDENTS})

        count = count_rows[0]['count'] if count_rows else 0

        if count > 0:
            await self.neo4j._execute_write('''
                MATCH (c:Case)
                OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
                WITH c, count(i) as incident_count
                WHERE incident_count < $min_incidents
                DETACH DELETE c
            ''', {'min_incidents': self.MIN_CASE_INCIDENTS})
            logger.info(f"Deleted {count} invalid cases (< {self.MIN_CASE_INCIDENTS} incidents)")

        return count
