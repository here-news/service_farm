"""
Enriched Belief Kernel - Layer 3
=================================

Wraps the core BeliefKernel to add:
- Belief IDs for citations
- Claim ID tracking (provenance)
- Entity ID extraction
- Certainty computation
- Thematic categorization
- Entropy trajectory tracking

The core kernel is UNCHANGED. This layer enriches input/output.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from openai import AsyncOpenAI

from belief_kernel import BeliefKernel, Belief


def generate_belief_id() -> str:
    """Generate short belief ID like bl_abc123"""
    return f"bl_{uuid.uuid4().hex[:8]}"


@dataclass
class EnrichedBelief:
    """Belief with full provenance and metadata"""
    id: str
    text: str
    sources: List[str]
    claim_ids: List[str]           # Original claim IDs that support this
    entity_ids: List[str]          # Entities mentioned
    certainty: float               # Computed from source count
    category: Optional[str]        # Thematic category
    supersedes_id: Optional[str]   # Previous belief ID
    supersedes_text: Optional[str] # Previous belief text
    last_updated: str              # ISO timestamp


@dataclass
class EnrichedConflict:
    """Conflict with full context"""
    id: str
    new_claim: str
    new_claim_id: Optional[str]
    existing_belief_id: Optional[str]
    existing_belief_text: Optional[str]
    topic: Optional[str]
    reasoning: str


@dataclass
class EntropyPoint:
    """Entropy at a point in processing"""
    claim_index: int
    entropy: float
    coherence: float
    belief_count: int
    conflict_count: int


@dataclass
class EnrichedTopology:
    """Full enriched topology for UI consumption"""
    beliefs: List[EnrichedBelief]
    conflicts: List[EnrichedConflict]
    entropy_trajectory: List[EntropyPoint]
    metrics: Dict[str, Any]
    relations: Dict[str, int]
    entity_lookup: Dict[str, str]  # name -> ID


class EnrichedKernel:
    """
    Wrapper around BeliefKernel that tracks enriched metadata.

    Usage:
        kernel = EnrichedKernel()
        for claim in claims:
            await kernel.process(claim)
        topology = kernel.get_topology()
    """

    def __init__(self):
        self.kernel = BeliefKernel()

        # Enrichment tracking
        self.belief_id_map: Dict[str, str] = {}  # belief text -> ID
        self.claim_contributions: Dict[str, List[str]] = {}  # belief ID -> claim IDs
        self.entity_contributions: Dict[str, List[str]] = {}  # belief ID -> entity IDs
        self.supersession_chain: Dict[str, str] = {}  # new belief ID -> old belief ID
        self.entropy_trajectory: List[EntropyPoint] = []
        self.entity_lookup: Dict[str, str] = {}  # entity name -> ID
        self.claim_count = 0

    def _get_or_create_belief_id(self, text: str) -> str:
        """Get existing belief ID or create new one"""
        if text not in self.belief_id_map:
            self.belief_id_map[text] = generate_belief_id()
        return self.belief_id_map[text]

    def _extract_entities(self, claim: Any) -> List[str]:
        """Extract entity IDs from claim"""
        if hasattr(claim, 'entities') and claim.entities:
            return [e.id for e in claim.entities if hasattr(e, 'id')]
        return []

    def _extract_entity_lookup(self, claim: Any):
        """Build entity name -> ID lookup"""
        if hasattr(claim, 'entities') and claim.entities:
            for e in claim.entities:
                if hasattr(e, 'id') and hasattr(e, 'canonical_name'):
                    self.entity_lookup[e.canonical_name] = e.id

    def _compute_certainty(self, source_count: int) -> float:
        """
        Certainty from source count.
        Simple math, not domain heuristics: more sources = higher certainty.
        """
        return min(source_count / 4.0, 1.0)

    async def process(
        self,
        claim: Any,  # Can be Claim object or dict
        source: str,
        llm: AsyncOpenAI
    ) -> Dict:
        """
        Process claim through kernel and track enrichment.

        Args:
            claim: Claim object with id, text, entities, etc.
            source: Source name
            llm: OpenAI client

        Returns:
            Kernel result plus enrichment data
        """
        self.claim_count += 1

        # Extract claim data
        if hasattr(claim, 'text'):
            claim_text = claim.text
            claim_id = getattr(claim, 'id', None)
        else:
            claim_text = claim.get('text', str(claim))
            claim_id = claim.get('id')

        # Extract entities
        entity_ids = self._extract_entities(claim)
        self._extract_entity_lookup(claim)

        # Build rich text if temporal/modality available
        rich_text = claim_text
        if hasattr(claim, 'event_time') and claim.event_time:
            if hasattr(claim.event_time, 'strftime'):
                rich_text = f"[{claim.event_time.strftime('%Y-%m-%d %H:%M')}] {rich_text}"
        if hasattr(claim, 'modality'):
            rich_text = f"[{claim.modality}] {rich_text}"

        # Process through kernel
        result = await self.kernel.process(
            claim=rich_text,
            source=source,
            llm=llm
        )

        # Track enrichment based on relation
        relation = result.get('relation', 'COMPATIBLE')
        normalized = result.get('normalized_claim', claim_text)
        affected_idx = result.get('affected_belief_index')

        if relation == 'COMPATIBLE':
            # New belief created
            belief_id = self._get_or_create_belief_id(normalized)
            self.claim_contributions[belief_id] = [claim_id] if claim_id else []
            self.entity_contributions[belief_id] = entity_ids

        elif relation == 'REDUNDANT':
            # Added to existing belief
            if affected_idx is not None and affected_idx < len(self.kernel.beliefs):
                existing_text = self.kernel.beliefs[affected_idx].text
                belief_id = self._get_or_create_belief_id(existing_text)
                if belief_id not in self.claim_contributions:
                    self.claim_contributions[belief_id] = []
                if claim_id:
                    self.claim_contributions[belief_id].append(claim_id)
                if belief_id not in self.entity_contributions:
                    self.entity_contributions[belief_id] = []
                self.entity_contributions[belief_id].extend(entity_ids)

        elif relation in ('REFINES', 'SUPERSEDES'):
            # Replaced existing belief
            if affected_idx is not None and affected_idx < len(self.kernel.beliefs):
                old_text = self.kernel.beliefs[affected_idx].supersedes
                old_id = self.belief_id_map.get(old_text) if old_text else None

                new_id = self._get_or_create_belief_id(normalized)
                self.claim_contributions[new_id] = [claim_id] if claim_id else []
                self.entity_contributions[new_id] = entity_ids

                if old_id:
                    self.supersession_chain[new_id] = old_id
                    # Inherit old claim contributions
                    if old_id in self.claim_contributions:
                        self.claim_contributions[new_id].extend(
                            self.claim_contributions[old_id]
                        )

        # Track entropy trajectory
        self.entropy_trajectory.append(EntropyPoint(
            claim_index=self.claim_count,
            entropy=self.kernel.compute_entropy(),
            coherence=self.kernel.compute_coherence(),
            belief_count=len(self.kernel.beliefs),
            conflict_count=len(self.kernel.conflicts)
        ))

        return result

    def get_topology(self) -> EnrichedTopology:
        """
        Get full enriched topology for UI consumption.
        """
        # Build enriched beliefs
        enriched_beliefs = []
        for b in self.kernel.beliefs:
            belief_id = self._get_or_create_belief_id(b.text)

            enriched_beliefs.append(EnrichedBelief(
                id=belief_id,
                text=b.text,
                sources=b.sources,
                claim_ids=self.claim_contributions.get(belief_id, []),
                entity_ids=list(set(self.entity_contributions.get(belief_id, []))),
                certainty=self._compute_certainty(len(b.sources)),
                category=None,  # LLM organizes thematically in prose layer
                supersedes_id=self.supersession_chain.get(belief_id),
                supersedes_text=b.supersedes,
                last_updated=datetime.now().isoformat()
            ))

        # Build enriched conflicts
        enriched_conflicts = []
        for i, c in enumerate(self.kernel.conflicts):
            enriched_conflicts.append(EnrichedConflict(
                id=f"cf_{i:03d}",
                new_claim=c.get('new_claim', ''),
                new_claim_id=None,  # TODO: track in process()
                existing_belief_id=self.belief_id_map.get(c.get('existing_belief')),
                existing_belief_text=c.get('existing_belief'),
                topic=None,  # LLM infers topic in prose layer
                reasoning=c.get('reasoning', '')
            ))

        # Compute metrics
        summary = self.kernel.summary()
        metrics = {
            'coherence': self.kernel.compute_coherence(),
            'entropy': self.kernel.compute_entropy(),
            'belief_count': len(self.kernel.beliefs),
            'conflict_count': len(self.kernel.conflicts),
            'claims_processed': self.claim_count,
            'compression': self.claim_count / max(len(self.kernel.beliefs), 1),
            'high_certainty_count': sum(1 for b in enriched_beliefs if b.certainty >= 0.75)
        }

        return EnrichedTopology(
            beliefs=enriched_beliefs,
            conflicts=enriched_conflicts,
            entropy_trajectory=self.entropy_trajectory,
            metrics=metrics,
            relations=summary['relations'],
            entity_lookup=self.entity_lookup
        )

    def to_dict(self) -> Dict:
        """Convert topology to JSON-serializable dict"""
        topology = self.get_topology()
        return {
            'beliefs': [
                {
                    'id': b.id,
                    'text': b.text,
                    'sources': b.sources,
                    'claim_ids': b.claim_ids,
                    'entity_ids': b.entity_ids,
                    'certainty': b.certainty,
                    'category': b.category,
                    'supersedes_id': b.supersedes_id,
                    'supersedes_text': b.supersedes_text,
                    'last_updated': b.last_updated
                }
                for b in topology.beliefs
            ],
            'conflicts': [
                {
                    'id': c.id,
                    'new_claim': c.new_claim,
                    'existing_belief_id': c.existing_belief_id,
                    'existing_belief_text': c.existing_belief_text,
                    'topic': c.topic,
                    'reasoning': c.reasoning
                }
                for c in topology.conflicts
            ],
            'entropy_trajectory': [
                {
                    'claim_index': e.claim_index,
                    'entropy': e.entropy,
                    'coherence': e.coherence,
                    'belief_count': e.belief_count,
                    'conflict_count': e.conflict_count
                }
                for e in topology.entropy_trajectory
            ],
            'metrics': topology.metrics,
            'relations': topology.relations,
            'entity_lookup': topology.entity_lookup
        }
