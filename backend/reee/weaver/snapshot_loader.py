"""
SnapshotLoader - Load PartitionSnapshot from Neo4j.

Given scope_id and surface_time, loads:
- Existing surface by (scope_id, question_key) if any
- Candidate incidents in same scope where time overlaps sliding window
- Minimal incident fields needed by routing

Does NOT load the whole graph - just what kernel needs for routing.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Set, FrozenSet

from services.neo4j_service import Neo4jService

from ..contracts.state import (
    SurfaceKey,
    SurfaceState,
    IncidentState,
    PartitionSnapshot,
)

logger = logging.getLogger(__name__)


class SnapshotLoader:
    """Load PartitionSnapshot from Neo4j for kernel processing.

    Thin adapter - does NOT contain business logic.
    Just converts Neo4j results to kernel contracts.
    """

    # Default time window for incident candidates
    DEFAULT_TIME_WINDOW_DAYS = 14

    def __init__(self, neo4j: Neo4jService):
        self.neo4j = neo4j

    async def load_for_claim(
        self,
        scope_id: str,
        surface_time: Optional[datetime],
        question_key: Optional[str] = None,
        time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    ) -> PartitionSnapshot:
        """Load snapshot for processing a claim.

        Args:
            scope_id: Computed scope ID for the claim
            surface_time: Claim's event time (for time window)
            question_key: Optional question_key if known
            time_window_days: Days before/after for incident candidates

        Returns:
            PartitionSnapshot with surfaces and candidate incidents
        """
        # Load existing surface if question_key is known
        surfaces = []
        if question_key:
            surface = await self._load_surface(scope_id, question_key)
            if surface:
                surfaces.append(surface)

        # Load candidate incidents in time window
        incidents = await self._load_candidate_incidents(
            scope_id=scope_id,
            surface_time=surface_time,
            time_window_days=time_window_days,
        )

        return PartitionSnapshot(
            scope_id=scope_id,
            surfaces=surfaces,
            incidents=incidents,
        )

    async def load_for_scope(
        self,
        scope_id: str,
        time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    ) -> PartitionSnapshot:
        """Load full snapshot for a scope (for reconciliation).

        Args:
            scope_id: Scope to load
            time_window_days: Days back from now for incidents

        Returns:
            PartitionSnapshot with all surfaces and incidents in scope
        """
        surfaces = await self._load_all_surfaces(scope_id)
        incidents = await self._load_candidate_incidents(
            scope_id=scope_id,
            surface_time=datetime.utcnow(),
            time_window_days=time_window_days,
        )

        return PartitionSnapshot(
            scope_id=scope_id,
            surfaces=surfaces,
            incidents=incidents,
        )

    async def _load_surface(
        self,
        scope_id: str,
        question_key: str,
    ) -> Optional[SurfaceState]:
        """Load a specific surface by (scope_id, question_key).

        Uses kernel signature for lookup if available,
        falls back to scope_id + question_key match.
        """
        key = SurfaceKey(scope_id=scope_id, question_key=question_key)
        signature = key.signature

        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            WHERE s.kernel_signature = $signature
               OR (s.scope_id = $scope_id AND s.question_key = $question_key)
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            WITH s, collect(c.id) as claim_ids
            RETURN s.id as id,
                   s.kernel_signature as signature,
                   s.scope_id as scope_id,
                   s.question_key as question_key,
                   claim_ids,
                   s.entities as entities,
                   s.anchor_entities as anchors,
                   s.sources as sources,
                   s.time_start as time_start,
                   s.time_end as time_end,
                   s.params_hash as params_hash,
                   s.kernel_version as kernel_version
            LIMIT 1
        """, {
            'signature': signature,
            'scope_id': scope_id,
            'question_key': question_key,
        })

        if not results:
            return None

        row = results[0]
        return self._row_to_surface_state(row, key)

    async def _load_all_surfaces(self, scope_id: str) -> List[SurfaceState]:
        """Load all surfaces in a scope."""
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            WHERE s.scope_id = $scope_id
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            WITH s, collect(c.id) as claim_ids
            RETURN s.id as id,
                   s.kernel_signature as signature,
                   s.scope_id as scope_id,
                   s.question_key as question_key,
                   claim_ids,
                   s.entities as entities,
                   s.anchor_entities as anchors,
                   s.sources as sources,
                   s.time_start as time_start,
                   s.time_end as time_end,
                   s.params_hash as params_hash,
                   s.kernel_version as kernel_version
        """, {'scope_id': scope_id})

        surfaces = []
        for row in results:
            if row.get('question_key'):
                key = SurfaceKey(
                    scope_id=row['scope_id'] or scope_id,
                    question_key=row['question_key'],
                )
                surfaces.append(self._row_to_surface_state(row, key))

        return surfaces

    async def _load_candidate_incidents(
        self,
        scope_id: str,
        surface_time: Optional[datetime],
        time_window_days: int,
    ) -> List[IncidentState]:
        """Load incidents that could be candidates for routing.

        Includes:
        - Incidents in same scope
        - Incidents with overlapping time window
        - Incidents with unknown time (always candidates)
        """
        # Compute time bounds for query
        if surface_time:
            time_start = surface_time - timedelta(days=time_window_days)
            time_end = surface_time + timedelta(days=time_window_days)
        else:
            # No time - use wide window from now
            now = datetime.utcnow()
            time_start = now - timedelta(days=time_window_days * 2)
            time_end = now + timedelta(days=1)

        # Query incidents (Incident nodes in Neo4j)
        # Include those with overlapping time OR unknown time
        results = await self.neo4j._execute_read("""
            MATCH (i:Incident)
            WHERE i.scope_id = $scope_id
              AND (
                  i.time_start IS NULL  // Unknown time - always candidate
                  OR (i.time_end IS NULL AND i.time_start <= $time_end)
                  OR (i.time_start <= $time_end AND i.time_end >= $time_start)
              )
            OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
            WITH i, collect(s.kernel_signature) as surface_sigs
            RETURN i.id as id,
                   i.kernel_signature as signature,
                   i.scope_id as scope_id,
                   surface_sigs,
                   i.anchor_entities as anchors,
                   i.companion_entities as companions,
                   i.time_start as time_start,
                   i.time_end as time_end,
                   i.params_hash as params_hash,
                   i.kernel_version as kernel_version
        """, {
            'scope_id': scope_id,
            'time_start': time_start.isoformat(),
            'time_end': time_end.isoformat(),
        })

        incidents = []
        for row in results:
            incidents.append(self._row_to_incident_state(row))

        logger.debug(
            f"Loaded {len(incidents)} candidate incidents for scope={scope_id}, "
            f"time_window={time_start.date()}..{time_end.date()}"
        )

        return incidents

    def _row_to_surface_state(
        self,
        row: dict,
        key: SurfaceKey,
    ) -> SurfaceState:
        """Convert Neo4j row to SurfaceState."""
        return SurfaceState(
            key=key,
            claim_ids=frozenset(row.get('claim_ids') or []),
            entities=frozenset(row.get('entities') or []),
            anchor_entities=frozenset(row.get('anchors') or []),
            sources=frozenset(row.get('sources') or []),
            time_start=self._parse_datetime(row.get('time_start')),
            time_end=self._parse_datetime(row.get('time_end')),
            params_hash=row.get('params_hash') or "",
            kernel_version=row.get('kernel_version') or "",
        )

    def _row_to_incident_state(self, row: dict) -> IncidentState:
        """Convert Neo4j row to IncidentState."""
        return IncidentState(
            id=row['id'],
            signature=row.get('signature') or "",
            scope_id=row.get('scope_id') or "",
            surface_ids=frozenset(row.get('surface_sigs') or []),
            anchor_entities=frozenset(row.get('anchors') or []),
            companion_entities=frozenset(row.get('companions') or []),
            time_start=self._parse_datetime(row.get('time_start')),
            time_end=self._parse_datetime(row.get('time_end')),
            params_hash=row.get('params_hash') or "",
            kernel_version=row.get('kernel_version') or "",
        )

    def _parse_datetime(self, value) -> Optional[datetime]:
        """Parse datetime from Neo4j result."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            from dateutil.parser import parse as parse_date
            try:
                return parse_date(value)
            except Exception:
                return None
        return None
