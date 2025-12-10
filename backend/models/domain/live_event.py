"""
LiveEvent - Event as a living organism

A LiveEvent is an in-memory representation of an event that:
- Bootstraps from claims
- Hydrates state from storage
- Executes metabolism (examines claims, updates narrative)
- Uses Bayesian topology for plausibility-weighted narratives
- Hibernates when dormant

The pool maintains these organisms, preventing duplicates and managing lifecycle.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Set, Optional, Dict
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

    Uses Bayesian topology (Jaynes-informed) for:
    - Plausibility scoring of claims
    - Weighted narrative generation
    - Contradiction detection and resolution
    """

    def __init__(self, event: Event, event_service, topology_service=None):
        self.event = event
        self.service = event_service
        self.topology_service = topology_service  # Optional: ClaimTopologyService

        # Internal state
        self.claims: List[Claim] = []
        self.entity_ids: Set[str] = set()

        # Source priors - stored on publisher at extraction time
        # publisher_priors: claim_id -> {'base_prior': float, 'source_type': str, 'publisher_name': str}
        self.publisher_priors: Dict[str, dict] = {}
        # page_urls: fallback for claims without stored priors
        self.page_urls: Dict[str, str] = {}

        self.last_claim_added: Optional[datetime] = None
        self.last_narrative_update: datetime = datetime.utcnow()
        self.last_topology_update: Optional[datetime] = None
        self.created_at: datetime = datetime.utcnow()

        # Cached topology result
        self._topology_result = None

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

        claim_ids = [c.id for c in self.claims]

        # Load publisher priors (stored on publisher entity at extraction time)
        self.publisher_priors = await claim_repo.get_publisher_priors_for_claims(claim_ids)

        # Also load page URLs as fallback for claims without stored priors
        self.page_urls = await claim_repo.get_page_urls_for_claims(claim_ids)

        # Set last claim time
        if self.claims:
            # Assume claims are sorted by created_at
            self.last_claim_added = max(c.created_at for c in self.claims if c.created_at)

        # Log source prior coverage
        stored_priors = sum(1 for p in self.publisher_priors.values() if p.get('base_prior'))
        logger.info(f"💧 Hydrated event: {self.event.canonical_name} ({len(self.claims)} claims, "
                   f"{stored_priors}/{len(claim_ids)} with stored priors)")

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

        Uses Bayesian topology if available, otherwise falls back to
        corroboration-guided synthesis.
        """
        logger.info(f"📝 Regenerating narrative for {self.event.canonical_name}")

        if self.topology_service and len(self.claims) >= 3:
            # Use Bayesian topology for weighted narrative
            narrative = await self._regenerate_narrative_bayesian()
        else:
            # Fallback to corroboration-guided synthesis
            narrative = await self.service._generate_event_narrative(self.event, self.claims)

        # Update in storage
        await self.service.event_repo.update_narrative(self.event.id, narrative)

        # Update internal state
        self.event.summary = narrative
        self.last_narrative_update = datetime.utcnow()

        logger.info(f"✨ Narrative updated: {len(narrative)} chars")

    async def _regenerate_narrative_bayesian(self) -> str:
        """
        Regenerate narrative using Bayesian claim topology.

        Jaynes-informed approach:
        1. Use stored publisher priors (assigned at extraction time)
        2. Compute plausibility posteriors for all claims
        3. Detect date consensus and penalize outliers
        4. Identify and report contradictions
        5. Generate narrative weighted by plausibility

        Returns:
            Narrative text weighted by claim plausibility
        """
        logger.info(f"🧬 Using Bayesian topology for {len(self.claims)} claims")

        # Run topology analysis with stored publisher priors
        # Priors are stored on publisher entities during knowledge extraction
        # page_urls is fallback for claims without stored priors
        topology = await self.topology_service.analyze(
            claims=self.claims,
            publisher_priors=self.publisher_priors,
            page_urls=self.page_urls
        )

        # Cache result for potential reuse
        self._topology_result = topology
        self.last_topology_update = datetime.utcnow()

        # Store plausibilities on claims (for graph update)
        await self._update_claim_plausibilities(topology)

        # Generate weighted narrative
        narrative = await self.topology_service.generate_weighted_narrative(
            self.event.canonical_name,
            self.claims,
            topology
        )

        logger.info(f"🧬 Bayesian narrative: pattern={topology.pattern}, "
                   f"contradictions={len(topology.contradictions)}, "
                   f"consensus_date={topology.consensus_date}")

        return narrative

    async def _update_claim_plausibilities(self, topology):
        """
        Update plausibility scores on SUPPORTS relationships in graph.

        This allows querying claims by plausibility for an event.
        """
        if not topology.claim_plausibilities:
            return

        try:
            for claim_id, result in topology.claim_plausibilities.items():
                await self.service.event_repo.update_claim_plausibility(
                    event_id=self.event.id,
                    claim_id=claim_id,
                    plausibility=result.posterior
                )
            logger.debug(f"📊 Updated plausibility for {len(topology.claim_plausibilities)} claims")
        except Exception as e:
            logger.warning(f"Failed to update claim plausibilities: {e}")

    def needs_topology_update(self) -> bool:
        """
        Check if topology analysis should be re-run.

        Re-analyze if:
        - No topology computed yet, OR
        - Claims added since last topology update, OR
        - More than 6 hours since last update
        """
        if not self.last_topology_update:
            return True

        if self.last_claim_added and self.last_claim_added > self.last_topology_update:
            return True

        hours_since = (datetime.utcnow() - self.last_topology_update).total_seconds() / 3600
        return hours_since > 6

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
