"""
LiveEventPool - Manages living event organisms

The pool:
- Maintains active events in memory
- Routes claims to appropriate events
- Bootstraps new events
- Hydrates existing events
- Executes metabolism cycles (with Bayesian topology)
- Hibernates dormant events

Prevents duplicate events across workers via careful routing logic.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

from models.domain.live_event import LiveEvent
from models.domain.event import Event
from models.domain.claim import Claim
from services.event_service import EventService
from services.claim_topology import ClaimTopologyService
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository

logger = logging.getLogger(__name__)


class LiveEventPool:
    """
    Pool of living events.

    Each event is a living organism that:
    - Examines claims
    - Updates its narrative
    - Eventually hibernates

    The pool prevents duplicates by maintaining state of active events.
    """

    def __init__(
        self,
        event_service: EventService,
        claim_repo: ClaimRepository,
        event_repo: EventRepository,
        entity_repo: EntityRepository,
        topology_service: Optional[ClaimTopologyService] = None
    ):
        self.service = event_service
        self.claim_repo = claim_repo
        self.event_repo = event_repo
        self.entity_repo = entity_repo
        self.topology_service = topology_service  # Bayesian claim analysis

        # Active events in memory
        self.active: Dict[str, LiveEvent] = {}

        logger.info(f"🌊 LiveEventPool initialized (topology={'enabled' if topology_service else 'disabled'})")

    async def route_page_claims(self, claims: List[Claim], page_embedding: Optional[List[float]]):
        """
        Route all claims from a page to appropriate living event.

        Layer 2: Find/activate appropriate event for this page's claims

        Page claims form one context (one article about one story).
        All claims should go to the same event.

        Steps:
        1. Find candidate events using page embedding + entities + time
        2. If good match: activate event and let it process all claims
        3. If no match: create new root event and bootstrap
        4. Handle sub-events created by examination

        Args:
            claims: All claims from one page (same context)
            page_embedding: Semantic embedding representing page content
        """
        logger.info(f"🎯 Routing page with {len(claims)} claims to pool")

        await self._route_page_claims(claims, page_embedding)

        logger.info(f"📊 Pool status: {len(self.active)} active events")

    async def _route_page_claims(self, claims: List[Claim], page_embedding: Optional[List[float]]):
        """
        Layer 2: Route all page claims to appropriate living event.

        This is the pool's core logic:
        - Find matching event via semantic (page) + entity + time signals
        - Activate event (load if hibernated)
        - Let event process ALL claims via its own metabolism (Layer 3)

        All claims from the same page go to the same event (one article = one context).
        """
        if not claims:
            return

        # Collect all entities from all claims
        all_entity_ids = set()
        for claim in claims:
            all_entity_ids.update(claim.entity_ids)

        # Use first claim's time as reference (all claims from same page)
        reference_time = claims[0].event_time if claims[0].event_time else None

        # Find candidate events using multi-signal matching
        # Page embedding captures semantic similarity of the entire article
        candidates = await self.event_repo.find_candidates(
            entity_ids=all_entity_ids,
            reference_time=reference_time,
            time_window_days=7,
            page_embedding=page_embedding
        )

        if candidates:
            best_event, best_score = candidates[0]

            logger.info(f"🔍 Best candidate: {best_event.canonical_name} (score: {best_score:.2f})")

            # Lowered threshold from 0.25 to 0.20 after rebalancing scoring weights
            # With 40% entity + 30% time weighting, 0.20 is a reasonable bar
            if best_score > 0.20:
                # Activate event (load from storage if needed)
                if best_event.id not in self.active:
                    await self._load_event(best_event.id)

                # Layer 3: Let living event process ALL claims via its own metabolism
                live_event = self.active[best_event.id]
                result = await live_event.examine(claims)

                # Bootstrap any sub-events created during examination
                for sub_event in result.sub_events_created:
                    await self._bootstrap_event(sub_event)

                return

        # No match - create new root event with all claims and bootstrap it
        logger.info(f"📝 Creating new root event for page ({len(claims)} claims)")
        event = await self.service.create_root_event(claims)
        await self._bootstrap_event(event)

    async def _load_event(self, event_id: str):
        """
        Load existing event into pool.

        Called when routing finds an existing event that's not in memory.
        """
        logger.info(f"💾 Loading event {event_id} into pool")

        event = await self.event_repo.get_by_id(event_id)
        if not event:
            logger.error(f"Event {event_id} not found in storage")
            return

        # Create LiveEvent with optional topology service
        live_event = LiveEvent(event, self.service, self.topology_service)

        # Hydrate state from storage
        await live_event.hydrate(self.claim_repo)

        # Add to pool
        self.active[event_id] = live_event

        logger.info(f"🌱 Loaded: {event.canonical_name} ({len(live_event.claims)} claims)")

    async def _bootstrap_event(self, event: Event):
        """
        Bootstrap new event into pool.

        Called when creating a new root event or sub-event.
        Event already has its initial claims attached.
        """
        logger.info(f"✨ Bootstrapping event: {event.canonical_name}")

        # Create LiveEvent with optional topology service
        live_event = LiveEvent(event, self.service, self.topology_service)

        # Mark as just created (has claims)
        live_event.last_claim_added = datetime.utcnow()

        # Load claims from storage to populate state
        await live_event.hydrate(self.claim_repo)

        # Add to pool
        self.active[event.id] = live_event

        logger.info(f"🌱 Bootstrapped: {event.canonical_name} ({len(live_event.claims)} claims)")

    async def metabolism_cycle(self):
        """
        Periodic maintenance - events update themselves.

        For each active event:
        1. Check if narrative needs update → regenerate
        2. Check if should hibernate → archive

        Called every hour by worker.
        """
        if not self.active:
            logger.info("😴 No active events in pool")
            return

        logger.info(f"🔄 Metabolism cycle: {len(self.active)} active events")

        for event_id in list(self.active.keys()):
            live_event = self.active[event_id]

            try:
                # Regenerate narrative if needed
                if live_event.needs_narrative_update():
                    logger.info(f"📝 Updating narrative: {live_event.event.canonical_name}")
                    await live_event.regenerate_narrative()

                # Hibernate if dormant
                if live_event.should_hibernate():
                    logger.info(f"😴 Hibernating: {live_event.event.canonical_name} (idle {live_event.idle_time_seconds()/3600:.1f}h)")
                    await self._hibernate_event(event_id)

            except Exception as e:
                logger.error(f"❌ Metabolism error for {live_event.event.canonical_name}: {e}", exc_info=True)

    async def _hibernate_event(self, event_id: str):
        """
        Hibernate event - remove from pool and mark as archived.

        Event is no longer receiving claims, can be archived.
        """
        live_event = self.active[event_id]

        # Update status in storage
        await self.event_repo.update_status(event_id, 'archived')

        # Remove from pool
        del self.active[event_id]

        logger.info(f"💤 Hibernated: {live_event.event.canonical_name}")

    def get_pool_status(self) -> dict:
        """Get current pool status for monitoring"""
        return {
            'active_count': len(self.active),
            'events': [
                {
                    'id': event.event.id,
                    'name': event.event.canonical_name,
                    'claims': len(event.claims),
                    'idle_hours': event.idle_time_seconds() / 3600,
                    'last_update': event.last_narrative_update.isoformat()
                }
                for event in self.active.values()
            ]
        }

    def __repr__(self):
        return f"<LiveEventPool: {len(self.active)} active events>"
