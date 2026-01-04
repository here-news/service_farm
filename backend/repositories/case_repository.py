"""
Case Repository - Data access layer for L4 Cases

Handles persistence and retrieval of Cases from Neo4j.
Cases are the user-facing "event" abstraction that groups related incidents.
"""
from typing import Optional, List, Set
from datetime import datetime
import logging

from services.neo4j_service import Neo4jService
from models.domain.case import Case

logger = logging.getLogger(__name__)


class CaseRepository:
    """
    Repository for L4 Case operations.

    Cases are stored in Neo4j with :Case:Event labels.
    The :Event label provides API compatibility with legacy event queries.
    """

    # Minimum incidents required for a valid case
    MIN_INCIDENTS = 2

    def __init__(self, neo4j: Neo4jService):
        self.neo4j = neo4j

    async def get_by_id(self, case_id: str) -> Optional[Case]:
        """Get a case by ID."""
        rows = await self.neo4j._execute_read('''
            MATCH (c:Case {id: $id})
            OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
            WITH c, collect(i.id) as incident_ids
            RETURN c.id as id,
                   c.title as title,
                   c.description as description,
                   c.primary_entities as primary_entities,
                   c.case_type as case_type,
                   c.binding_evidence as binding_evidence,
                   c.surface_count as surface_count,
                   c.source_count as source_count,
                   c.time_start as time_start,
                   c.time_end as time_end,
                   c.created_at as created_at,
                   c.updated_at as updated_at,
                   incident_ids
        ''', {'id': case_id})

        if not rows:
            return None

        return self._row_to_case(rows[0])

    async def list_valid(
        self,
        offset: int = 0,
        limit: int = 20,
        entity_filter: Optional[str] = None,
        since: Optional[str] = None,
        case_type: Optional[str] = None,
    ) -> List[Case]:
        """
        List cases that meet validity requirements (>= 2 incidents).

        Args:
            offset: Pagination offset
            limit: Page size
            entity_filter: Filter by entity name (substring match)
            since: Filter by time_start >= this date
            case_type: Filter by case_type (developing, entity_storyline, etc.)

        Returns:
            List of valid cases, ordered by source_count desc
        """
        where_clauses = []
        params = {'offset': offset, 'limit': limit, 'min_incidents': self.MIN_INCIDENTS}

        if entity_filter:
            where_clauses.append("any(e IN c.primary_entities WHERE e CONTAINS $entity)")
            params['entity'] = entity_filter

        if since:
            where_clauses.append("c.time_start >= $since")
            params['since'] = since

        if case_type:
            where_clauses.append("c.case_type = $case_type")
            params['case_type'] = case_type

        where_str = " AND ".join(where_clauses) if where_clauses else "true"

        rows = await self.neo4j._execute_read(f'''
            MATCH (c:Case)
            WHERE {where_str}
            OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
            WITH c, count(i) as incident_count, collect(i.id) as incident_ids
            WHERE incident_count >= $min_incidents
            RETURN c.id as id,
                   c.title as title,
                   c.description as description,
                   c.primary_entities as primary_entities,
                   c.case_type as case_type,
                   c.binding_evidence as binding_evidence,
                   c.surface_count as surface_count,
                   c.source_count as source_count,
                   c.time_start as time_start,
                   c.time_end as time_end,
                   c.created_at as created_at,
                   c.updated_at as updated_at,
                   incident_ids
            ORDER BY c.source_count DESC, c.updated_at DESC
            SKIP $offset
            LIMIT $limit
        ''', params)

        return [self._row_to_case(row) for row in rows]

    async def count_valid(
        self,
        entity_filter: Optional[str] = None,
        since: Optional[str] = None,
        case_type: Optional[str] = None,
    ) -> int:
        """Count valid cases (>= 2 incidents)."""
        where_clauses = []
        params = {'min_incidents': self.MIN_INCIDENTS}

        if entity_filter:
            where_clauses.append("any(e IN c.primary_entities WHERE e CONTAINS $entity)")
            params['entity'] = entity_filter

        if since:
            where_clauses.append("c.time_start >= $since")
            params['since'] = since

        if case_type:
            where_clauses.append("c.case_type = $case_type")
            params['case_type'] = case_type

        where_str = " AND ".join(where_clauses) if where_clauses else "true"

        rows = await self.neo4j._execute_read(f'''
            MATCH (c:Case)
            WHERE {where_str}
            OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
            WITH c, count(i) as incident_count
            WHERE incident_count >= $min_incidents
            RETURN count(c) as total
        ''', params)

        return rows[0]['total'] if rows else 0

    async def save(self, case: Case) -> Case:
        """Save or update a case."""
        await self.neo4j._execute_write('''
            MERGE (c:Case {id: $id})
            SET c:Event,
                c.title = $title,
                c.canonical_title = $title,
                c.description = $description,
                c.primary_entities = $primary_entities,
                c.case_type = $case_type,
                c.binding_evidence = $binding_evidence,
                c.surface_count = $surface_count,
                c.source_count = $source_count,
                c.time_start = $time_start,
                c.time_end = $time_end,
                c.updated_at = datetime()
            WITH c
            // Clear old incident links
            OPTIONAL MATCH (c)-[r:CONTAINS]->(:Incident)
            DELETE r
            WITH c
            // Create new incident links
            UNWIND $incident_ids as iid
            MATCH (i:Incident {id: iid})
            MERGE (c)-[:CONTAINS]->(i)
        ''', {
            'id': case.id,
            'title': case.title,
            'description': case.description,
            'primary_entities': case.primary_entities,
            'case_type': case.case_type,
            'binding_evidence': case.binding_evidence,
            'surface_count': case.surface_count,
            'source_count': case.source_count,
            'time_start': case.time_start.isoformat() if case.time_start else None,
            'time_end': case.time_end.isoformat() if case.time_end else None,
            'incident_ids': list(case.incident_ids),
        })

        case.updated_at = datetime.utcnow()
        return case

    async def delete(self, case_id: str) -> bool:
        """Delete a case."""
        await self.neo4j._execute_write('''
            MATCH (c:Case {id: $id})
            DETACH DELETE c
        ''', {'id': case_id})
        return True

    async def delete_invalid(self) -> int:
        """Delete all cases with fewer than MIN_INCIDENTS incidents."""
        # First count
        count_rows = await self.neo4j._execute_read('''
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
            WITH c, count(i) as incident_count
            WHERE incident_count < $min_incidents
            RETURN count(c) as count
        ''', {'min_incidents': self.MIN_INCIDENTS})

        count = count_rows[0]['count'] if count_rows else 0

        if count > 0:
            await self.neo4j._execute_write('''
                MATCH (c:Case)
                OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
                WITH c, count(i) as incident_count
                WHERE incident_count < $min_incidents
                DETACH DELETE c
            ''', {'min_incidents': self.MIN_INCIDENTS})

        return count

    async def get_surfaces_for_case(self, case_id: str) -> List[dict]:
        """Get all surfaces for a case (via incidents)."""
        rows = await self.neo4j._execute_read('''
            MATCH (c:Case {id: $id})-[:CONTAINS]->(i:Incident)-[:CONTAINS]->(s:Surface)
            OPTIONAL MATCH (s)-[:CONTAINS]->(cl:Claim)
            WITH s, collect(cl.text)[0..3] as sample_claims
            RETURN s.id as id,
                   s.sources as sources,
                   s.anchor_entities as anchors,
                   s.time_start as time_start,
                   sample_claims
            ORDER BY s.time_start
        ''', {'id': case_id})

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

    def _row_to_case(self, row: dict) -> Case:
        """Convert a Neo4j row to a Case domain object."""
        return Case(
            id=row['id'],
            incident_ids=set(row.get('incident_ids') or []),
            title=row.get('title') or "",
            description=row.get('description') or "",
            primary_entities=row.get('primary_entities') or [],
            case_type=row.get('case_type') or "developing",
            binding_evidence=row.get('binding_evidence') or [],
            surface_count=row.get('surface_count') or 0,
            source_count=row.get('source_count') or 0,
            time_start=self._parse_time(row.get('time_start')),
            time_end=self._parse_time(row.get('time_end')),
            created_at=self._parse_time(row.get('created_at')),
            updated_at=self._parse_time(row.get('updated_at')),
        )

    def _parse_time(self, val) -> Optional[datetime]:
        """Parse time from various formats."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace('Z', '+00:00'))
            except:
                return None
        return None
