"""
EvidenceBuilder - Build ClaimEvidence from Neo4j claim data.

Converts DB claim representation to kernel's ClaimEvidence contract.
Uses existing repositories for data access.

Optional providers can be plugged in for:
- Embedding lookup
- LLM question_key extraction
- Typed observation extraction
"""

import logging
from datetime import datetime
from typing import Optional, List, FrozenSet, Tuple, Dict, Any

from services.neo4j_service import Neo4jService

from ..contracts.evidence import ClaimEvidence, TypedObservation
from ..topo.scope import extract_primary_anchors

logger = logging.getLogger(__name__)


class EvidenceBuilder:
    """Build ClaimEvidence from Neo4j claims.

    Thin adapter - converts DB format to kernel contract.
    Does NOT contain business logic.
    """

    def __init__(self, neo4j: Neo4jService):
        self.neo4j = neo4j

    async def build_from_claim_id(self, claim_id: str) -> Optional[ClaimEvidence]:
        """Build ClaimEvidence from a claim ID.

        Loads claim + entities + page info from Neo4j.

        Args:
            claim_id: Claim ID (cl_xxxxxxxx format)

        Returns:
            ClaimEvidence or None if claim not found
        """
        results = await self.neo4j._execute_read("""
            MATCH (c:Claim {id: $claim_id})
            OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            WITH c, p, collect(e.canonical_name) as entity_names
            RETURN c.id as claim_id,
                   c.text as text,
                   c.event_time as event_time,
                   c.reported_time as reported_time,
                   c.confidence as confidence,
                   c.who_entity_ids as who_entity_ids,
                   c.where_entity_ids as where_entity_ids,
                   p.id as page_id,
                   p.domain as source_id,
                   entity_names
        """, {'claim_id': claim_id})

        if not results:
            logger.warning(f"Claim not found: {claim_id}")
            return None

        row = results[0]
        return self._row_to_evidence(row)

    async def build_batch(self, claim_ids: List[str]) -> List[ClaimEvidence]:
        """Build ClaimEvidence for multiple claims.

        More efficient than individual calls.

        Args:
            claim_ids: List of claim IDs

        Returns:
            List of ClaimEvidence (preserves order, skips not found)
        """
        if not claim_ids:
            return []

        results = await self.neo4j._execute_read("""
            UNWIND $claim_ids as cid
            MATCH (c:Claim {id: cid})
            OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            WITH c, p, collect(e.canonical_name) as entity_names
            RETURN c.id as claim_id,
                   c.text as text,
                   c.event_time as event_time,
                   c.reported_time as reported_time,
                   c.confidence as confidence,
                   c.who_entity_ids as who_entity_ids,
                   c.where_entity_ids as where_entity_ids,
                   p.id as page_id,
                   p.domain as source_id,
                   entity_names
        """, {'claim_ids': claim_ids})

        # Build dict for ordering
        evidence_by_id = {}
        for row in results:
            evidence = self._row_to_evidence(row)
            if evidence:
                evidence_by_id[evidence.claim_id] = evidence

        # Preserve order
        return [evidence_by_id[cid] for cid in claim_ids if cid in evidence_by_id]

    async def build_for_page(self, page_id: str) -> List[ClaimEvidence]:
        """Build ClaimEvidence for all claims from a page.

        Useful for processing a page's claims together.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            List of ClaimEvidence for all claims on page
        """
        results = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:EMITS]->(c:Claim)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            WITH c, p, collect(e.canonical_name) as entity_names
            ORDER BY c.created_at
            RETURN c.id as claim_id,
                   c.text as text,
                   c.event_time as event_time,
                   c.reported_time as reported_time,
                   c.confidence as confidence,
                   c.who_entity_ids as who_entity_ids,
                   c.where_entity_ids as where_entity_ids,
                   p.id as page_id,
                   p.domain as source_id,
                   entity_names
        """, {'page_id': page_id})

        return [self._row_to_evidence(row) for row in results if row.get('claim_id')]

    def _row_to_evidence(self, row: dict) -> Optional[ClaimEvidence]:
        """Convert Neo4j row to ClaimEvidence."""
        claim_id = row.get('claim_id')
        if not claim_id:
            return None

        # Parse entities
        entities = frozenset(row.get('entity_names') or [])

        # Extract anchors from entities
        # Use who/where entity IDs if available, else derive from entities
        anchors, all_hubs = extract_primary_anchors(entities)

        # Parse time
        event_time = self._parse_datetime(row.get('event_time'))
        if not event_time:
            event_time = self._parse_datetime(row.get('reported_time'))

        # Confidence from claim
        confidence = row.get('confidence') or 0.8

        return ClaimEvidence(
            claim_id=claim_id,
            text=row.get('text') or "",
            source_id=row.get('source_id') or "unknown",
            page_id=row.get('page_id'),
            entities=entities,
            anchors=anchors,
            question_key=None,  # Not extracted from DB - kernel will derive
            time=event_time,
            entity_confidence=confidence,
            question_key_confidence=0.5,  # No explicit key from DB
            time_confidence=0.7 if event_time else 0.3,
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


class EnrichedEvidenceBuilder(EvidenceBuilder):
    """EvidenceBuilder with optional enrichment providers.

    Can plug in:
    - LLM for question_key extraction
    - Embedding service for claim embeddings
    - Typed observation extractor

    These run OUTSIDE kernel - enriched data goes into ClaimEvidence.
    """

    def __init__(
        self,
        neo4j: Neo4jService,
        question_key_provider=None,  # async def(text, entities) -> (key, confidence)
        embedding_provider=None,  # async def(text) -> List[float]
        observation_extractor=None,  # async def(text) -> TypedObservation
    ):
        super().__init__(neo4j)
        self.question_key_provider = question_key_provider
        self.embedding_provider = embedding_provider
        self.observation_extractor = observation_extractor

    async def build_from_claim_id(self, claim_id: str) -> Optional[ClaimEvidence]:
        """Build enriched ClaimEvidence."""
        evidence = await super().build_from_claim_id(claim_id)
        if evidence:
            evidence = await self._enrich(evidence)
        return evidence

    async def _enrich(self, evidence: ClaimEvidence) -> ClaimEvidence:
        """Apply enrichment providers to evidence."""
        enrichments: Dict[str, Any] = {}

        # Question key from LLM
        if self.question_key_provider:
            try:
                key, confidence = await self.question_key_provider(
                    evidence.text, evidence.entities
                )
                if key:
                    enrichments['question_key'] = key
                    enrichments['question_key_confidence'] = confidence
            except Exception as e:
                logger.warning(f"Question key extraction failed: {e}")

        # Embedding
        if self.embedding_provider:
            try:
                embedding = await self.embedding_provider(evidence.text)
                if embedding:
                    enrichments['embedding'] = tuple(embedding)
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")

        # Typed observation
        if self.observation_extractor:
            try:
                observation = await self.observation_extractor(evidence.text)
                if observation:
                    enrichments['typed_observation'] = observation
            except Exception as e:
                logger.warning(f"Observation extraction failed: {e}")

        if enrichments:
            return evidence.with_enrichment(**enrichments)

        return evidence
