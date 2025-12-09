"""
LiveEvent - Event as a living organism

A LiveEvent is an in-memory representation of an event that:
- Bootstraps from claims
- Hydrates state from storage
- Executes metabolism (examines claims, updates narrative)
- Hibernates when dormant

The pool maintains these organisms, preventing duplicates and managing lifecycle.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Set, Optional
from dataclasses import dataclass

from models.domain.event import Event
from models.domain.claim import Claim

logger = logging.getLogger(__name__)


@dataclass
class LiveEventState:
    """State of a living event"""
    event: Event
    claims: List[Claim]
    entity_ids: Set[str]
    last_claim_added: Optional[datetime]
    last_narrative_update: datetime
    created_at: datetime


class LiveEvent:
    """
    A living event organism that maintains itself.

    Lifecycle:
    1. Bootstrap - created from initial claims
    2. Hydrate - loads full state from storage
    3. Metabolism - examines new claims, updates narrative
    4. Hibernate - archives when dormant
    """

    def __init__(self, event: Event, event_service):
        self.event = event
        self.service = event_service

        # Internal state
        self.claims: List[Claim] = []
        self.entity_ids: Set[str] = set()
        self.last_claim_added: Optional[datetime] = None
        self.last_narrative_update: datetime = datetime.utcnow()
        self.created_at: datetime = datetime.utcnow()

        logger.debug(f"🌱 LiveEvent created: {event.canonical_name} ({event.id})")

    async def hydrate(self, claim_repo):
        """
        Load full state from storage.

        Called when loading existing event into pool.
        """
        # Load all claims for this event
        self.claims = await claim_repo.get_by_event(self.event.id)

        # Extract entity IDs
        for claim in self.claims:
            self.entity_ids.update(claim.entity_ids)

        # Set last claim time
        if self.claims:
            # Assume claims are sorted by created_at
            self.last_claim_added = max(c.created_at for c in self.claims if c.created_at)

        logger.info(f"💧 Hydrated event: {self.event.canonical_name} ({len(self.claims)} claims, {len(self.entity_ids)} entities)")

    async def examine(self, new_claims: List[Claim]):
        """
        Layer 3: Event's own metabolism processes claims.

        The living event examines claims and decides:
        - Accept claim into this event (same story)
        - Reject claim (unrelated)
        - Create sub-event for claim (related but distinct)

        This is the event's autonomous behavior - it processes claims
        based on its own understanding of what it is.

        Returns:
            ExaminationResult with decisions
        """
        logger.info(f"🔬 {self.event.canonical_name} examining {len(new_claims)} claims")

        # Capture coherence before examination
        old_coherence = self.event.coherence if self.event.coherence else 0.5

        # Event examines claims via its metabolism
        result = await self.service.examine_claims(self.event, new_claims)

        # Update internal state based on examination results
        if result.claims_added:
            self.claims.extend(result.claims_added)
            for claim in result.claims_added:
                self.entity_ids.update(claim.entity_ids)
            self.last_claim_added = datetime.utcnow()

            logger.info(f"✅ {self.event.canonical_name} accepted {len(result.claims_added)} claims")

            # Calculate new coherence after adding claims
            new_coherence = await self._calculate_coherence()

            # Update event coherence in storage
            await self.service.event_repo.update_coherence(self.event.id, new_coherence)
            self.event.coherence = new_coherence

            # Check coherence delta and trigger narrative regeneration if significant
            delta = new_coherence - old_coherence
            logger.info(f"📊 Coherence: {old_coherence:.3f} → {new_coherence:.3f} (Δ {delta:+.3f})")

            if delta > 0.1:
                logger.info(f"📈 Significant coherence boost (Δ={delta:.3f}) - triggering narrative regeneration")
                await self.regenerate_narrative()

        if result.claims_rejected:
            logger.info(f"❌ {self.event.canonical_name} rejected {len(result.claims_rejected)} claims")

        if result.sub_events_created:
            logger.info(f"🌿 {self.event.canonical_name} created {len(result.sub_events_created)} sub-events")

        return result

    async def _calculate_coherence(self) -> float:
        """
        Calculate event coherence using Jaynes' maximum entropy principle.

        Formula: coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity

        Hub coverage: % of claims touching entities with 3+ mentions
        Graph connectivity: 1.0 / num_connected_components (union-find)

        Based on GitHub issue #3 specification.
        """
        if not self.claims or not self.entity_ids:
            return 0.5  # Neutral coherence for empty event

        hub_coverage = await self._calculate_hub_coverage()
        graph_connectivity = await self._calculate_graph_connectivity()

        coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity

        logger.debug(f"🧮 Coherence components: hub={hub_coverage:.3f}, connectivity={graph_connectivity:.3f}")

        return coherence

    async def _calculate_hub_coverage(self) -> float:
        """
        Calculate percentage of claims that touch hub entities.

        Hub entity = entity mentioned in 3+ claims in this event.

        Returns: 0.0 to 1.0
        """
        if not self.claims:
            return 0.0

        # Count entity mentions across all claims
        entity_mention_counts = {}
        for claim in self.claims:
            for entity_id in claim.entity_ids:
                entity_mention_counts[entity_id] = entity_mention_counts.get(entity_id, 0) + 1

        # Identify hub entities (3+ mentions)
        hub_entities = {entity_id for entity_id, count in entity_mention_counts.items() if count >= 3}

        if not hub_entities:
            return 0.0

        # Count claims that touch at least one hub
        claims_touching_hubs = 0
        for claim in self.claims:
            if any(entity_id in hub_entities for entity_id in claim.entity_ids):
                claims_touching_hubs += 1

        hub_coverage = claims_touching_hubs / len(self.claims)

        logger.debug(f"🎯 Hub coverage: {claims_touching_hubs}/{len(self.claims)} claims touch {len(hub_entities)} hubs")

        return hub_coverage

    async def _calculate_graph_connectivity(self) -> float:
        """
        Calculate graph connectivity using union-find algorithm.

        Creates bipartite graph of claims and entities:
        - Nodes: claims + entities
        - Edges: claim mentions entity

        Connectivity = 1.0 / num_connected_components
        (1.0 = fully connected, lower = fragmented)

        Returns: 0.0 to 1.0
        """
        if not self.claims or not self.entity_ids:
            return 0.0

        # Build union-find structure
        parent = {}

        def find(x):
            """Find root with path compression"""
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """Union two sets"""
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        # Create edges: claim <-> entity
        for claim in self.claims:
            claim_node = f"claim:{claim.id}"
            for entity_id in claim.entity_ids:
                entity_node = f"entity:{entity_id}"
                union(claim_node, entity_node)

        # Count connected components
        components = len(set(find(node) for node in parent.keys()))

        if components == 0:
            return 0.0

        connectivity = 1.0 / components

        logger.debug(f"🕸️  Graph connectivity: {components} components → {connectivity:.3f}")

        return connectivity

    def needs_narrative_update(self) -> bool:
        """
        Check if narrative needs regeneration.

        Regenerate if:
        - Claims added in last 6 hours AND
        - Narrative not updated in last hour
        """
        if not self.last_claim_added:
            return False

        time_since_claim = (datetime.utcnow() - self.last_claim_added).total_seconds()
        time_since_update = (datetime.utcnow() - self.last_narrative_update).total_seconds()

        # Active period: claims added recently
        is_active = time_since_claim < 6 * 3600  # 6 hours

        # Stale narrative: not updated in a while
        is_stale = time_since_update > 3600  # 1 hour

        return is_active and is_stale

    async def regenerate_narrative(self):
        """
        Metabolism - update narrative based on accumulated claims.

        Generates fresh narrative from all claims and updates event.
        """
        logger.info(f"📝 Regenerating narrative for {self.event.canonical_name}")

        narrative = await self.service._generate_event_narrative(self.event, self.claims)

        # Update in storage
        await self.service.event_repo.update_narrative(self.event.id, narrative)

        # Update internal state
        self.event.summary = narrative
        self.last_narrative_update = datetime.utcnow()

        logger.info(f"✨ Narrative updated: {len(narrative)} chars")

    def idle_time_seconds(self) -> float:
        """Seconds since last claim was added"""
        if not self.last_claim_added:
            return (datetime.utcnow() - self.created_at).total_seconds()
        return (datetime.utcnow() - self.last_claim_added).total_seconds()

    def should_hibernate(self) -> bool:
        """
        Check if event should be archived.

        Hibernate if idle for 24+ hours.
        """
        return self.idle_time_seconds() > 24 * 3600

    def get_state(self) -> LiveEventState:
        """Export current state"""
        return LiveEventState(
            event=self.event,
            claims=self.claims,
            entity_ids=self.entity_ids,
            last_claim_added=self.last_claim_added,
            last_narrative_update=self.last_narrative_update,
            created_at=self.created_at
        )

    def __repr__(self):
        return f"<LiveEvent {self.event.canonical_name} ({len(self.claims)} claims, idle={self.idle_time_seconds():.0f}s)>"
