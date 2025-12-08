"""
LiveEvent - Active event representation for real-time processing.

Inspired by diffusion models: events start "noisy" and refine through
iterative claim metabolism. Each new claim is a "denoising" step that
refines the event's understanding and structure.

Lifecycle:
1. Hydrate from Neo4j (or create fresh from proto-event)
2. Metabolize incoming claims (absorb, spawn sub-event, or reject)
3. Checkpoint to Neo4j on eviction or shutdown

Lazy Loading:
- Default: Load only high-confidence first-degree relationships
- On demand: Extend specific branches deeper into the graph
"""
import uuid
import math
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class MetabolismDecision(Enum):
    """What happened to a claim during metabolism"""
    ABSORB = "absorb"           # Added to core claims
    CORROBORATE = "corroborate" # Strengthens existing claim
    SPAWN_PHASE = "spawn_phase" # Triggers new temporal phase
    SPAWN_ASPECT = "spawn_aspect"  # Triggers new thematic aspect
    DELEGATE = "delegate"       # Belongs to existing sub-event
    REJECT = "reject"           # Doesn't belong to this event


@dataclass
class ClaimDigest:
    """Lightweight claim reference (avoid loading full claim objects)"""
    id: str
    text_preview: str  # First 100 chars
    confidence: float
    entity_ids: Set[str]
    event_time: Optional[datetime] = None

    @classmethod
    def from_claim(cls, claim) -> 'ClaimDigest':
        # Extract entity IDs - prefer hydrated entities, fall back to metadata
        entity_ids = set()
        if claim.entities:
            # Use hydrated Entity objects
            entity_ids = {str(e.id) for e in claim.entities}
        elif claim.entity_ids:
            # Fall back to metadata entity_ids
            entity_ids = {str(e) for e in claim.entity_ids}

        return cls(
            id=str(claim.id),
            text_preview=claim.text[:100] if claim.text else "",
            confidence=claim.confidence,
            entity_ids=entity_ids,
            event_time=claim.event_time
        )


@dataclass
class EntityDigest:
    """Lightweight entity reference"""
    id: str
    canonical_name: str
    entity_type: str
    wikidata_qid: Optional[str] = None
    confidence: float = 0.5

    @classmethod
    def from_entity(cls, entity) -> 'EntityDigest':
        return cls(
            id=str(entity.id),
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            wikidata_qid=entity.wikidata_qid,
            confidence=getattr(entity, 'confidence', 0.5)
        )


@dataclass
class MetabolismResult:
    """Result of metabolizing a batch of claims"""
    absorbed: List[ClaimDigest] = field(default_factory=list)
    corroborated: List[Tuple[ClaimDigest, ClaimDigest]] = field(default_factory=list)  # (new, existing)
    spawned_phases: List['LiveEvent'] = field(default_factory=list)
    spawned_aspects: List['LiveEvent'] = field(default_factory=list)
    delegated: List[Tuple[ClaimDigest, str]] = field(default_factory=list)  # (claim, sub_event_key)
    rejected: List[ClaimDigest] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return (len(self.absorbed) + len(self.corroborated) +
                len(self.delegated) + len(self.rejected))


@dataclass
class LiveEvent:
    """
    Active event representation that evolves through claim metabolism.

    Like diffusion models:
    - Starts "noisy" (initial claims, uncertain structure)
    - Each new claim batch "denoises" (refines understanding)
    - Structure emerges organically (sub-events, phases)

    Hydration Strategy:
    - Default: Load high-confidence first-degree relationships only
    - Lazy: Extend specific branches on demand
    - This keeps memory footprint manageable for hot events
    """

    # === Identity ===
    dedup_key: str
    canonical_name: str
    event_type: str
    id: Optional[str] = None  # Neo4j ID if persisted

    # === Core Graph (high-confidence, first-degree) ===
    core_claims: List[ClaimDigest] = field(default_factory=list)
    core_entities: Dict[str, EntityDigest] = field(default_factory=dict)  # id -> EntityDigest

    # === Pending (to be refined) ===
    pending_claims: List[ClaimDigest] = field(default_factory=list)

    # === Recursive Structure ===
    sub_events: Dict[str, 'LiveEvent'] = field(default_factory=dict)  # dedup_key -> LiveEvent
    parent_key: Optional[str] = None  # Parent event's dedup_key

    # === Temporal Bounds ===
    earliest_time: Optional[datetime] = None
    latest_time: Optional[datetime] = None

    # === Metrics ===
    confidence: float = 0.3  # Overall event confidence
    coherence: float = 0.5   # How well claims fit together
    pages_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)

    # === Lazy-loaded Extensions ===
    _extended_branches: Dict[str, dict] = field(default_factory=dict)  # entity_id -> deeper graph
    _neo4j_service: Optional[object] = field(default=None, repr=False)

    # === Thresholds (tunable) ===
    CONFIDENCE_THRESHOLD: float = 0.6  # Claims above this go to core
    CORROBORATION_SIMILARITY: float = 0.85  # Similarity to consider corroboration
    SPAWN_THRESHOLD: float = 0.4  # Below this, might spawn sub-event

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    # === Weight for Pool Eviction ===

    @property
    def weight(self) -> float:
        """
        Compute event weight for eviction decisions.
        Higher weight = more "alive", should stay in memory.
        """
        # Recency: exponential decay, half-life ~24h
        hours_old = (datetime.now() - self.last_activity).total_seconds() / 3600
        recency = math.exp(-hours_old / 24)

        # Volume: log scale
        volume = math.log1p(self.pages_count) * math.log1p(len(self.core_claims))

        # Entity importance: sum of high-confidence entities
        entity_signal = sum(
            (e.confidence or 0.5) for e in self.core_entities.values()
        ) / max(1, len(self.core_entities))

        # Structure bonus: events with sub-events are more developed
        structure_bonus = 1.0 + 0.2 * len(self.sub_events)

        return recency * (1 + volume) * (1 + entity_signal) * structure_bonus

    # === Hydration (Load from Neo4j) ===

    @classmethod
    async def hydrate(
        cls,
        neo4j_service,
        dedup_key: str,
        depth: int = 1,
        confidence_threshold: float = 0.5
    ) -> Optional['LiveEvent']:
        """
        Hydrate event from Neo4j with configurable depth.

        Args:
            neo4j_service: Neo4j service instance
            dedup_key: Event's deduplication key
            depth: How deep to load relationships (1 = first-degree only)
            confidence_threshold: Only load relationships above this confidence

        Returns:
            LiveEvent or None if not found
        """
        # Query event node
        result = await neo4j_service._execute_read("""
            MATCH (e:Event {dedup_key: $dedup_key})
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.event_type as event_type,
                   e.confidence as confidence,
                   e.coherence as coherence,
                   e.pages_count as pages_count,
                   e.earliest_time as earliest_time,
                   e.latest_time as latest_time,
                   e.updated_at as last_activity
        """, {'dedup_key': dedup_key})

        if not result:
            return None

        row = result[0]
        event = cls(
            dedup_key=dedup_key,
            id=row['id'],
            canonical_name=row['canonical_name'],
            event_type=row['event_type'] or 'UNSPECIFIED',
            confidence=row.get('confidence', 0.3),
            coherence=row.get('coherence', 0.5),
            pages_count=row.get('pages_count', 0),
            earliest_time=row.get('earliest_time'),
            latest_time=row.get('latest_time'),
            last_activity=row.get('last_activity') or datetime.now()
        )
        event._neo4j_service = neo4j_service

        # Load first-degree relationships (claims + entities)
        if depth >= 1:
            await event._load_core_graph(confidence_threshold)

        # Load sub-events
        await event._load_sub_events(depth - 1 if depth > 1 else 0)

        logger.info(f"💧 Hydrated event: {event.canonical_name} "
                   f"({len(event.core_claims)} claims, {len(event.core_entities)} entities)")

        return event

    async def _load_core_graph(self, confidence_threshold: float = 0.5):
        """Load high-confidence claims and their entities"""
        if not self._neo4j_service:
            return

        # Load claims
        claims_result = await self._neo4j_service._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            WHERE c.confidence >= $threshold
            OPTIONAL MATCH (c)-[:MENTIONS]->(entity:Entity)
            RETURN c.id as claim_id,
                   c.text as text,
                   c.confidence as confidence,
                   c.event_time as event_time,
                   collect(DISTINCT entity.id) as entity_ids
            ORDER BY c.confidence DESC
            LIMIT 100
        """, {'event_id': self.id, 'threshold': confidence_threshold})

        for row in claims_result:
            digest = ClaimDigest(
                id=row['claim_id'],
                text_preview=row['text'][:100] if row['text'] else "",
                confidence=row['confidence'] or 0.5,
                entity_ids=set(row['entity_ids'] or []),
                event_time=row.get('event_time')
            )
            self.core_claims.append(digest)

        # Load entities involved
        entities_result = await self._neo4j_service._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INVOLVES]->(entity:Entity)
            RETURN entity.id as id,
                   entity.canonical_name as canonical_name,
                   entity.entity_type as entity_type,
                   entity.wikidata_qid as wikidata_qid,
                   entity.confidence as confidence
        """, {'event_id': self.id})

        for row in entities_result:
            digest = EntityDigest(
                id=row['id'],
                canonical_name=row['canonical_name'],
                entity_type=row['entity_type'],
                wikidata_qid=row.get('wikidata_qid'),
                confidence=row.get('confidence', 0.5)
            )
            self.core_entities[digest.id] = digest

    async def _load_sub_events(self, depth: int = 0):
        """Load sub-events (phases, aspects)"""
        if not self._neo4j_service:
            return

        result = await self._neo4j_service._execute_read("""
            MATCH (e:Event {id: $event_id})-[:CONTAINS]->(sub:Event)
            RETURN sub.dedup_key as dedup_key,
                   sub.canonical_name as canonical_name,
                   sub.event_type as event_type
        """, {'event_id': self.id})

        for row in result:
            if depth > 0:
                # Recursively hydrate sub-events
                sub = await LiveEvent.hydrate(
                    self._neo4j_service,
                    row['dedup_key'],
                    depth=depth
                )
                if sub:
                    sub.parent_key = self.dedup_key
                    self.sub_events[sub.dedup_key] = sub
            else:
                # Just create stub
                sub = LiveEvent(
                    dedup_key=row['dedup_key'],
                    canonical_name=row['canonical_name'],
                    event_type=row['event_type'] or 'UNSPECIFIED',
                    parent_key=self.dedup_key
                )
                sub._neo4j_service = self._neo4j_service
                self.sub_events[sub.dedup_key] = sub

    # === On-Demand Extension ===

    async def extend_branch(self, entity_id: str, depth: int = 1) -> dict:
        """
        Lazy-load deeper relationships for a specific entity.

        Use case: User clicks on an entity, want to see its connections
        beyond the event's core graph.
        """
        if entity_id in self._extended_branches:
            return self._extended_branches[entity_id]

        if not self._neo4j_service:
            return {}

        # Load entity's relationships beyond this event
        result = await self._neo4j_service._execute_read("""
            MATCH (entity:Entity {id: $entity_id})
            OPTIONAL MATCH (entity)-[r]-(connected)
            WHERE NOT connected:Event OR connected.id <> $event_id
            RETURN type(r) as rel_type,
                   labels(connected)[0] as node_type,
                   connected.id as connected_id,
                   connected.canonical_name as connected_name
            LIMIT 50
        """, {'entity_id': entity_id, 'event_id': self.id})

        branch = {
            'entity_id': entity_id,
            'connections': [dict(row) for row in result]
        }

        self._extended_branches[entity_id] = branch
        logger.debug(f"🌿 Extended branch for entity {entity_id}: {len(result)} connections")

        return branch

    # === Metabolism (Process New Claims) ===

    async def metabolize(self, claims: List, source_credibility: float = 0.5) -> MetabolismResult:
        """
        Process new claims - the "denoising" step.

        Like diffusion denoising:
        - New claims refine our understanding
        - High-confidence claims strengthen the core
        - Outlier claims may spawn sub-events (new aspects/phases)
        - Contradictions are handled (not just rejected)

        Args:
            claims: List of Claim objects from a new page
            source_credibility: Credibility of the source (0-1)

        Returns:
            MetabolismResult describing what happened
        """
        result = MetabolismResult()

        self.pages_count += 1
        self.last_activity = datetime.now()

        for claim in claims:
            digest = ClaimDigest.from_claim(claim)

            # Adjust confidence by source credibility
            effective_confidence = digest.confidence * source_credibility

            # Decision logic
            decision = await self._decide_claim_fate(digest, effective_confidence)

            if decision == MetabolismDecision.ABSORB:
                await self._absorb_claim(digest, claim)
                result.absorbed.append(digest)

            elif decision == MetabolismDecision.CORROBORATE:
                existing = self._find_similar_claim(digest)
                if existing:
                    await self._corroborate_claim(existing, digest)
                    result.corroborated.append((digest, existing))

            elif decision == MetabolismDecision.SPAWN_PHASE:
                phase = await self._spawn_sub_event(digest, claim, "phase")
                result.spawned_phases.append(phase)

            elif decision == MetabolismDecision.SPAWN_ASPECT:
                aspect = await self._spawn_sub_event(digest, claim, "aspect")
                result.spawned_aspects.append(aspect)

            elif decision == MetabolismDecision.DELEGATE:
                sub_key = self._find_delegate_target(digest)
                if sub_key and sub_key in self.sub_events:
                    # Recursively metabolize in sub-event
                    await self.sub_events[sub_key].metabolize([claim], source_credibility)
                    result.delegated.append((digest, sub_key))

            else:  # REJECT
                result.rejected.append(digest)

        # Update event metrics after metabolism
        self._update_metrics()

        logger.info(f"🧬 Metabolized {len(claims)} claims: "
                   f"{len(result.absorbed)} absorbed, "
                   f"{len(result.corroborated)} corroborated, "
                   f"{len(result.spawned_phases)} phases, "
                   f"{len(result.rejected)} rejected")

        return result

    async def _decide_claim_fate(
        self,
        digest: ClaimDigest,
        effective_confidence: float
    ) -> MetabolismDecision:
        """
        Decide what to do with a claim.

        Decision tree:
        0. Cold start (no core yet)? → ABSORB to bootstrap
        1. High overlap with existing claim? → CORROBORATE
        2. High confidence + fits topic? → ABSORB
        3. Different time period? → SPAWN_PHASE or DELEGATE
        4. Different entity cluster? → SPAWN_ASPECT or DELEGATE
        5. Low relevance? → REJECT
        """
        # COLD START: If we have no core claims yet, absorb to bootstrap
        # This is the "noisy start" - we need data to begin refining
        if not self.core_claims:
            if effective_confidence >= 0.2:  # Very low threshold for bootstrapping
                return MetabolismDecision.ABSORB
            return MetabolismDecision.REJECT  # Only reject truly garbage claims

        # Check for corroboration (similar to existing claim)
        similar = self._find_similar_claim(digest)
        if similar:
            return MetabolismDecision.CORROBORATE

        # Check entity overlap with core
        entity_overlap = len(digest.entity_ids & set(self.core_entities.keys()))
        entity_overlap_ratio = entity_overlap / max(1, len(digest.entity_ids))

        # High overlap + good confidence = absorb
        if entity_overlap_ratio > 0.3 and effective_confidence >= self.CONFIDENCE_THRESHOLD:
            return MetabolismDecision.ABSORB

        # Medium overlap = still absorb if decent confidence
        if entity_overlap_ratio > 0.2 and effective_confidence >= 0.4:
            return MetabolismDecision.ABSORB

        # Check if it belongs to an existing sub-event
        delegate_target = self._find_delegate_target(digest)
        if delegate_target:
            return MetabolismDecision.DELEGATE

        # NOTE: Disabled phase/aspect spawning for now due to timestamp data quality issues
        # TODO: Re-enable when timestamps are reliably accurate
        #
        # # Check temporal divergence (might be a new phase)
        # # Only spawn phases if we have strong temporal signal and enough core claims
        # if len(self.core_claims) >= 20 and self._is_temporal_divergent(digest):
        #     return MetabolismDecision.SPAWN_PHASE
        #
        # # Check thematic divergence (might be a new aspect)
        # # Only spawn if we have strong core and very low overlap
        # if len(self.core_claims) >= 30 and entity_overlap_ratio < 0.05 and effective_confidence > self.SPAWN_THRESHOLD:
        #     return MetabolismDecision.SPAWN_ASPECT

        # Low relevance - but be more lenient early on
        min_claims_for_strict = 10
        reject_threshold = 0.1 if len(self.core_claims) >= min_claims_for_strict else 0.0

        if entity_overlap_ratio <= reject_threshold and effective_confidence < 0.3:
            return MetabolismDecision.REJECT

        # Default: absorb (we're still learning the event's structure)
        return MetabolismDecision.ABSORB

    def _find_similar_claim(self, digest: ClaimDigest) -> Optional[ClaimDigest]:
        """Find existing claim that's similar (for corroboration)"""
        # Simple heuristic: high entity overlap
        for existing in self.core_claims:
            overlap = len(digest.entity_ids & existing.entity_ids)
            if overlap >= 2:  # Share at least 2 entities
                return existing
        return None

    def _find_delegate_target(self, digest: ClaimDigest) -> Optional[str]:
        """Find sub-event that should handle this claim"""
        best_match = None
        best_overlap = 0

        for key, sub in self.sub_events.items():
            overlap = len(digest.entity_ids & set(sub.core_entities.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = key

        # Only delegate if good match
        if best_overlap >= 2:
            return best_match
        return None

    def _is_temporal_divergent(self, digest: ClaimDigest) -> bool:
        """Check if claim's time is significantly different from event's bounds"""
        if not digest.event_time or not self.earliest_time:
            return False

        # Ensure we have datetime objects (might come as strings from Neo4j)
        claim_time = digest.event_time
        earliest = self.earliest_time
        latest = self.latest_time

        if isinstance(claim_time, str):
            try:
                claim_time = datetime.fromisoformat(claim_time.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return False
        if isinstance(earliest, str):
            try:
                earliest = datetime.fromisoformat(earliest.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return False
        if latest and isinstance(latest, str):
            try:
                latest = datetime.fromisoformat(latest.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                latest = None

        # More than 7 days (168 hours) outside current bounds
        # Use longer threshold since news events can span multiple days
        DIVERGENCE_HOURS = 168  # 7 days
        try:
            if claim_time < earliest:
                delta = (earliest - claim_time).total_seconds() / 3600
                return delta > DIVERGENCE_HOURS
            if latest and claim_time > latest:
                delta = (claim_time - latest).total_seconds() / 3600
                return delta > DIVERGENCE_HOURS
        except (TypeError, AttributeError):
            return False

        return False

    async def _absorb_claim(self, digest: ClaimDigest, claim):
        """Add claim to core and link entities"""
        self.core_claims.append(digest)

        # Build entity lookup from hydrated claim if available
        entity_lookup = {}
        if claim.entities:
            for e in claim.entities:
                entity_lookup[str(e.id)] = e

        # Add new entities to core
        for entity_id in digest.entity_ids:
            if entity_id not in self.core_entities:
                if entity_id in entity_lookup:
                    # Use hydrated entity data
                    e = entity_lookup[entity_id]
                    self.core_entities[entity_id] = EntityDigest(
                        id=entity_id,
                        canonical_name=e.canonical_name,
                        entity_type=e.entity_type,
                        wikidata_qid=getattr(e, 'wikidata_qid', None),
                        confidence=getattr(e, 'confidence', 0.5)
                    )
                else:
                    # Create stub - will be enriched on demand
                    self.core_entities[entity_id] = EntityDigest(
                        id=entity_id,
                        canonical_name="(pending)",
                        entity_type="UNKNOWN"
                    )

        # Update temporal bounds
        if digest.event_time:
            if not self.earliest_time or digest.event_time < self.earliest_time:
                self.earliest_time = digest.event_time
            if not self.latest_time or digest.event_time > self.latest_time:
                self.latest_time = digest.event_time

    async def _corroborate_claim(self, existing: ClaimDigest, new: ClaimDigest):
        """Strengthen existing claim with corroborating evidence"""
        # Boost confidence (diminishing returns)
        boost = new.confidence * 0.1
        existing.confidence = min(0.99, existing.confidence + boost)

    async def _spawn_sub_event(
        self,
        digest: ClaimDigest,
        claim,
        sub_type: str  # "phase" or "aspect"
    ) -> 'LiveEvent':
        """Create a new sub-event from a divergent claim"""
        from services.neo4j_service import Neo4jService

        # Generate dedup_key for sub-event
        sub_name = f"{self.canonical_name} - {sub_type} {len(self.sub_events) + 1}"
        sub_dedup_key = Neo4jService._compute_event_dedup_key(sub_name, self.event_type)

        sub = LiveEvent(
            dedup_key=sub_dedup_key,
            canonical_name=sub_name,
            event_type=self.event_type,
            parent_key=self.dedup_key,
            confidence=0.3,
            pages_count=1
        )
        sub._neo4j_service = self._neo4j_service

        # Add the triggering claim
        await sub._absorb_claim(digest, claim)

        self.sub_events[sub_dedup_key] = sub

        logger.info(f"🌱 Spawned {sub_type}: {sub_name}")
        return sub

    def _update_metrics(self):
        """Update confidence and coherence after metabolism"""
        # Confidence: based on claim count and average confidence
        if self.core_claims:
            avg_claim_conf = sum(c.confidence for c in self.core_claims) / len(self.core_claims)
            volume_factor = min(1.0, math.log1p(len(self.core_claims)) / 3)
            self.confidence = avg_claim_conf * volume_factor

        # Coherence: based on graph connectivity, not pairwise density
        # A coherent event has claims connected through shared entities
        if len(self.core_claims) > 1:
            # Build entity -> claims mapping
            entity_to_claims: Dict[str, Set[int]] = {}
            for i, claim in enumerate(self.core_claims):
                for eid in claim.entity_ids:
                    if eid not in entity_to_claims:
                        entity_to_claims[eid] = set()
                    entity_to_claims[eid].add(i)

            # Find "hub" entities (mentioned in 3+ claims)
            hub_entities = [eid for eid, claims in entity_to_claims.items()
                          if len(claims) >= 3]

            # Metric 1: Hub coverage - what % of claims touch a hub entity?
            claims_with_hub = set()
            for eid in hub_entities:
                claims_with_hub.update(entity_to_claims[eid])
            hub_coverage = len(claims_with_hub) / len(self.core_claims)

            # Metric 2: Graph connectivity via union-find
            # Claims are "connected" if they share any entity
            parent = list(range(len(self.core_claims)))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Connect claims that share entities
            for eid, claim_indices in entity_to_claims.items():
                claim_list = list(claim_indices)
                for i in range(len(claim_list) - 1):
                    union(claim_list[i], claim_list[i + 1])

            # Count connected components
            components = len(set(find(i) for i in range(len(self.core_claims))))
            # Connectivity: 1.0 if single component, decreases with fragmentation
            connectivity = 1.0 / components

            # Combined coherence: weighted average
            self.coherence = 0.6 * hub_coverage + 0.4 * connectivity
        else:
            self.coherence = 0.5

    # === Checkpoint (Save to Neo4j) ===

    async def checkpoint(self):
        """
        Save current state to Neo4j before eviction.

        This is the "freeze" operation - convert live state to cold storage.
        """
        if not self._neo4j_service:
            logger.warning(f"⚠️ Cannot checkpoint {self.canonical_name}: no Neo4j service")
            return

        # Upsert event node
        await self._neo4j_service._execute_write("""
            MERGE (e:Event {dedup_key: $dedup_key})
            ON CREATE SET
                e.id = $id,
                e.canonical_name = $canonical_name,
                e.event_type = $event_type,
                e.created_at = datetime()
            SET e.confidence = $confidence,
                e.coherence = $coherence,
                e.pages_count = $pages_count,
                e.earliest_time = $earliest_time,
                e.latest_time = $latest_time,
                e.updated_at = datetime()
        """, {
            'dedup_key': self.dedup_key,
            'id': self.id,
            'canonical_name': self.canonical_name,
            'event_type': self.event_type,
            'confidence': self.confidence,
            'coherence': self.coherence,
            'pages_count': self.pages_count,
            'earliest_time': self.earliest_time,
            'latest_time': self.latest_time
        })

        # Link claims (create SUPPORTS relationships)
        for claim in self.core_claims:
            await self._neo4j_service._execute_write("""
                MATCH (e:Event {id: $event_id})
                MATCH (c:Claim {id: $claim_id})
                MERGE (e)-[r:SUPPORTS]->(c)
                ON CREATE SET r.created_at = datetime()
            """, {'event_id': self.id, 'claim_id': claim.id})

        # Link entities (create INVOLVES relationships)
        for entity_id in self.core_entities.keys():
            await self._neo4j_service._execute_write("""
                MATCH (e:Event {id: $event_id})
                MATCH (entity:Entity {id: $entity_id})
                MERGE (e)-[r:INVOLVES]->(entity)
                ON CREATE SET r.created_at = datetime()
            """, {'event_id': self.id, 'entity_id': entity_id})

        # Checkpoint sub-events and link
        for sub in self.sub_events.values():
            await sub.checkpoint()
            await self._neo4j_service._execute_write("""
                MATCH (parent:Event {id: $parent_id})
                MATCH (child:Event {id: $child_id})
                MERGE (parent)-[r:CONTAINS]->(child)
                ON CREATE SET r.created_at = datetime()
            """, {'parent_id': self.id, 'child_id': sub.id})

        logger.info(f"💾 Checkpointed: {self.canonical_name} "
                   f"({len(self.core_claims)} claims, {len(self.sub_events)} sub-events)")

    # === Factory Methods ===

    @classmethod
    def from_proto(cls, proto: dict) -> 'LiveEvent':
        """
        Create fresh LiveEvent from proto-event (extracted by KnowledgeWorker).

        This is the "noisy" initial state that will be refined.
        """
        from services.neo4j_service import Neo4jService

        title = proto.get('title', 'Unknown Event')
        event_type = proto.get('event_type', 'UNSPECIFIED').upper()

        dedup_key = Neo4jService._compute_event_dedup_key(title, event_type)

        return cls(
            dedup_key=dedup_key,
            canonical_name=title,
            event_type=event_type,
            confidence=0.3,  # Low initial confidence
            pages_count=0
        )

    async def analyze_structure(self, openai_client=None) -> dict:
        """
        Analyze and extract the internal structure of the event.

        This reveals what structure has emerged from metabolism:
        - Facets/aspects (different angles of the story)
        - Entity relationships (who connects to whom)
        - Claim clusters (groups of related facts)
        - Contradictions (where sources disagree)
        - Information gaps (what's missing)

        Returns:
            dict with structured analysis
        """
        if not self.core_claims:
            return {'error': 'No claims to analyze'}

        if openai_client is None:
            from openai import OpenAI
            import os
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Collect claim texts with their entity associations
        claims_with_entities = []
        for claim in self.core_claims[:60]:  # Top 60 claims
            entity_names = []
            for eid in claim.entity_ids:
                if eid in self.core_entities:
                    entity_names.append(self.core_entities[eid].canonical_name)
            claims_with_entities.append({
                'text': claim.text_preview,
                'confidence': claim.confidence,
                'entities': entity_names
            })

        # Collect entity info
        entities_info = [
            {'name': e.canonical_name, 'type': e.entity_type}
            for e in self.core_entities.values()
            if e.canonical_name != "(pending)"
        ]

        system_prompt = """You are an event structure analyst. Given claims about an event,
identify the STRUCTURE that emerges - not just a summary, but how the information organizes itself.

Analyze and output JSON with this structure:
{
  "facets": [
    {
      "name": "short facet name (e.g., 'The Fire', 'Casualties', 'Response')",
      "description": "what this facet covers",
      "key_claims": ["claim text 1", "claim text 2"],
      "key_entities": ["entity names involved in this facet"],
      "completeness": 0.0-1.0  // how complete is our knowledge of this facet
    }
  ],
  "entity_relationships": [
    {
      "entity1": "name",
      "relationship": "type of relationship",
      "entity2": "name",
      "evidence": "claim that shows this"
    }
  ],
  "contradictions": [
    {
      "claim_a": "one claim",
      "claim_b": "contradicting claim",
      "nature": "what the contradiction is about"
    }
  ],
  "information_gaps": [
    {
      "gap": "what information is missing",
      "importance": "high/medium/low",
      "why_needed": "why this matters"
    }
  ],
  "narrative_threads": [
    {
      "thread": "name of narrative thread",
      "progression": ["how the story develops across claims"]
    }
  ],
  "structure_score": {
    "coherence": 0.0-1.0,  // how well claims fit together
    "completeness": 0.0-1.0,  // how complete the picture is
    "reliability": 0.0-1.0  // how trustworthy based on corroboration
  }
}"""

        import json
        user_prompt = f"""Event: {self.canonical_name}
Type: {self.event_type}

Entities ({len(entities_info)}):
{json.dumps(entities_info, indent=2)}

Claims ({len(claims_with_entities)} total):
{json.dumps(claims_with_entities, indent=2)}

Analyze the structure that emerges from these claims."""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"🔍 Analyzed structure for {self.canonical_name}: "
                       f"{len(result.get('facets', []))} facets, "
                       f"{len(result.get('entity_relationships', []))} relationships")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to analyze structure: {e}")
            return {'error': str(e)}

    async def synthesize_description(self, openai_client=None) -> dict:
        """
        Synthesize a coherent, readable description from all metabolized claims.

        This is the "output" of metabolism - converting raw claims into
        a structured, consumable narrative.

        Returns:
            dict with 'summary', 'key_facts', 'timeline', 'entities_involved'
        """
        if not self.core_claims:
            return {
                'summary': f"No claims available for {self.canonical_name}",
                'key_facts': [],
                'timeline': [],
                'entities_involved': []
            }

        # Lazy import to avoid circular deps
        if openai_client is None:
            from openai import OpenAI
            import os
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Collect claim texts (sorted by confidence)
        sorted_claims = sorted(self.core_claims, key=lambda c: c.confidence, reverse=True)
        claim_texts = [c.text_preview for c in sorted_claims[:50]]  # Top 50 claims

        # Collect entity names
        entity_names = [e.canonical_name for e in self.core_entities.values()
                       if e.canonical_name != "(pending)"]

        system_prompt = """You are a news synthesizer. Given a collection of factual claims
about an event, produce a coherent, readable summary that:
1. Presents the key facts in order of importance
2. Avoids repetition
3. Maintains factual accuracy (don't invent details)
4. Notes any conflicting information
5. Is written in clear, journalistic prose

Output JSON with this structure:
{
  "summary": "A 2-3 paragraph narrative summary of the event",
  "key_facts": ["fact1", "fact2", ...],  // 5-10 most important facts
  "timeline": [{"time": "...", "what": "..."}],  // if temporal info available
  "entities_involved": [{"name": "...", "role": "..."}]  // key actors/locations
}"""

        user_prompt = f"""Event: {self.canonical_name}
Type: {self.event_type}
Confidence: {self.confidence:.2f}

Known entities: {', '.join(entity_names[:20])}

Claims ({len(claim_texts)} total):
{chr(10).join(f'- {text}' for text in claim_texts)}

Synthesize these claims into a coherent description."""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            import json
            result = json.loads(response.choices[0].message.content)
            logger.info(f"📝 Synthesized description for {self.canonical_name}")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to synthesize description: {e}")
            # Fallback: return raw claim compilation
            return {
                'summary': f"{self.canonical_name}: {' '.join(claim_texts[:5])}",
                'key_facts': claim_texts[:10],
                'timeline': [],
                'entities_involved': [{'name': n, 'role': 'mentioned'} for n in entity_names[:10]],
                'error': str(e)
            }

    async def generate_article(self, openai_client=None) -> str:
        """
        Generate a comprehensive long-form article from all metabolized knowledge.

        This is the "hologram" - the complete picture told by the LiveEvent itself,
        incorporating:
        - All claims organized by facet/aspect
        - Entity relationships and their roles
        - Timeline of events
        - Contradictions and uncertainties
        - What we know vs what remains unknown

        Returns:
            str: A complete markdown article
        """
        if not self.core_claims:
            return f"# {self.canonical_name}\n\nNo information available yet."

        if openai_client is None:
            from openai import OpenAI
            import os
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        import json

        # Gather ALL the knowledge
        # 1. Claims grouped by entity associations
        claims_data = []
        for claim in self.core_claims:
            entities = [self.core_entities[eid].canonical_name
                       for eid in claim.entity_ids
                       if eid in self.core_entities and self.core_entities[eid].canonical_name != "(pending)"]
            claims_data.append({
                'text': claim.text_preview,
                'confidence': claim.confidence,
                'entities': entities,
                'has_time': claim.event_time is not None
            })

        # 2. Entity profiles
        entity_profiles = []
        entity_claim_count = {}
        for claim in self.core_claims:
            for eid in claim.entity_ids:
                entity_claim_count[eid] = entity_claim_count.get(eid, 0) + 1

        for eid, entity in self.core_entities.items():
            if entity.canonical_name == "(pending)":
                continue
            entity_profiles.append({
                'name': entity.canonical_name,
                'type': entity.entity_type,
                'mentions': entity_claim_count.get(eid, 0),
                'is_hub': entity_claim_count.get(eid, 0) >= 3
            })

        # Sort by mentions (most important first)
        entity_profiles.sort(key=lambda e: -e['mentions'])

        # 3. Event metadata
        metadata = {
            'name': self.canonical_name,
            'type': self.event_type,
            'total_claims': len(self.core_claims),
            'total_entities': len(self.core_entities),
            'confidence': self.confidence,
            'coherence': self.coherence,
            'pages_processed': self.pages_count,
            'earliest_time': str(self.earliest_time) if self.earliest_time else None,
            'latest_time': str(self.latest_time) if self.latest_time else None
        }

        system_prompt = """You are an investigative journalist writing a comprehensive article.

Given structured data about an event (claims, entities, metadata), write a COMPLETE long-form article.

REQUIREMENTS:
1. Write 1500-2500 words in clear, professional prose
2. Organize into sections with markdown headers (##)
3. Structure should include:
   - Opening: Hook + key facts (who, what, when, where)
   - The Incident: Detailed account of what happened
   - The Human Cost: Casualties, victims, displacement
   - The Response: Emergency services, community response
   - Investigation: What authorities found, arrests, questions
   - The People Involved: Key figures and their roles
   - What Remains Unknown: Gaps, unanswered questions
   - Conclusion: Significance, context

4. USE the claims as your source material - weave them into narrative
5. HIGHLIGHT entity relationships - show how people/places/orgs connect
6. NOTE any contradictions between claims (different casualty numbers, etc.)
7. DISTINGUISH between confirmed facts (high confidence) and reports (lower confidence)
8. Include a "Key Facts" sidebar as a bulleted list
9. End with "What We're Still Learning" section for gaps

Write in third person, past tense for events, present tense for ongoing situations.
Do NOT invent details not in the claims. If something is unclear, say so.

Output the article in markdown format."""

        user_prompt = f"""EVENT DATA:

Metadata:
{json.dumps(metadata, indent=2)}

Entities (by importance):
{json.dumps(entity_profiles[:25], indent=2)}

Claims ({len(claims_data)} total):
{json.dumps(claims_data, indent=2)}

Write the comprehensive article now."""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=4000
            )

            article = response.choices[0].message.content
            logger.info(f"📰 Generated article for {self.canonical_name} "
                       f"({len(article)} chars)")
            return article

        except Exception as e:
            logger.error(f"❌ Failed to generate article: {e}")
            return f"# {self.canonical_name}\n\nError generating article: {e}"

    def __repr__(self):
        return (f"LiveEvent('{self.canonical_name}', "
                f"claims={len(self.core_claims)}, "
                f"entities={len(self.core_entities)}, "
                f"weight={self.weight:.2f})")
