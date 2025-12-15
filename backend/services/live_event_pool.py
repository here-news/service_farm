"""
LiveEventPool - Manages living event organisms

The pool:
- Maintains active events in memory
- Routes claims to appropriate events (with scoring logic)
- Bootstraps new events
- Hydrates existing events
- Executes metabolism cycles (with Bayesian topology)
- Hibernates dormant events

Prevents duplicate events across workers via careful routing logic.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

from models.domain.live_event import LiveEvent
from models.domain.event import Event
from models.domain.claim import Claim
from services.event_service import EventService
from services.claim_topology import ClaimTopologyService
from services.routing_config import WEIGHT_ENTITY, WEIGHT_SEMANTIC, ROUTING_THRESHOLD, SEMANTIC_FALLBACK_THRESHOLD
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
        - Find matching event via semantic (page) + entity signals
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

        # Get candidate events (data only, no scoring)
        candidates_data = await self.event_repo.get_candidate_events(
            entity_ids=all_entity_ids,
            limit=20
        )

        if not candidates_data:
            # No entity overlap - try semantic fallback
            if page_embedding:
                logger.info(f"🔍 No entity overlap, trying semantic fallback...")
                semantic_candidates = await self.event_repo.get_candidate_events_by_embedding(
                    page_embedding=page_embedding,
                    threshold=SEMANTIC_FALLBACK_THRESHOLD,
                    limit=5
                )

                if semantic_candidates:
                    best_event, best_similarity = semantic_candidates[0]
                    logger.info(
                        f"📊 Semantic fallback found: {best_event.canonical_name} "
                        f"(similarity: {best_similarity:.2f})"
                    )

                    # Activate event and process claims
                    if best_event.id not in self.active:
                        await self._load_event(best_event.id)

                    live_event = self.active[best_event.id]
                    result = await live_event.examine(claims)

                    for sub_event in result.sub_events_created:
                        await self._bootstrap_event(sub_event)
                    return

                logger.info(f"📝 Semantic fallback found no matches (threshold={SEMANTIC_FALLBACK_THRESHOLD})")

            logger.info(f"📝 Creating new root event ({len(claims)} claims)")
            event = await self.service.create_root_event(claims)
            await self._bootstrap_event(event)
            return

        # Score candidates using routing weights
        scored_candidates = []
        for event, event_entity_ids, _source_page_ids in candidates_data:
            # Debug: Check embeddings
            has_page_emb = page_embedding is not None and len(page_embedding) > 0
            has_event_emb = event.embedding is not None and len(event.embedding) > 0

            score, entity_score, semantic_score = self._score_candidate(
                page_entity_ids=all_entity_ids,
                event_entity_ids=event_entity_ids,
                page_embedding=page_embedding,
                event_embedding=event.embedding,
                return_breakdown=True
            )
            scored_candidates.append((event, score))

            logger.info(
                f"📊 Candidate: {event.canonical_name} "
                f"(score: {score:.2f}, entity: {entity_score:.2f}, semantic: {semantic_score:.2f}, "
                f"page_emb: {has_page_emb}, event_emb: {has_event_emb})"
            )

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_event, best_score = scored_candidates[0]

        logger.info(f"🔍 Best candidate: {best_event.canonical_name} (score: {best_score:.2f})")

        if best_score > ROUTING_THRESHOLD:
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

    def _score_candidate(
        self,
        page_entity_ids: set,
        event_entity_ids: set,
        page_embedding: Optional[List[float]],
        event_embedding: Optional[List[float]],
        return_breakdown: bool = False
    ):
        """
        Score how well a page matches an existing event.

        Uses module-level WEIGHT_ENTITY and WEIGHT_SEMANTIC constants.

        Args:
            page_entity_ids: Entity IDs from incoming page's claims
            event_entity_ids: Entity IDs already linked to event
            page_embedding: Semantic embedding of incoming page content
            event_embedding: Semantic embedding of event
            return_breakdown: If True, return (score, entity_score, semantic_score)

        Returns:
            Match score 0.0-1.0 (or tuple if return_breakdown=True)
        """
        # Entity overlap (Jaccard similarity)
        if event_entity_ids:
            intersection = len(page_entity_ids & event_entity_ids)
            union = len(page_entity_ids | event_entity_ids)
            entity_score = intersection / union if union > 0 else 0.0
        else:
            entity_score = 0.0

        # Semantic similarity (cosine between page and event embeddings)
        semantic_score = 0.0
        if page_embedding and event_embedding:
            vec1 = np.array(page_embedding)
            vec2 = np.array(event_embedding)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                semantic_score = float(dot_product / (norm1 * norm2))

        # Combined score
        if semantic_score == 0.0:
            # No embedding available - rely on entity overlap alone
            score = entity_score
        else:
            score = WEIGHT_ENTITY * entity_score + WEIGHT_SEMANTIC * semantic_score

        if return_breakdown:
            return score, entity_score, semantic_score
        return score

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
        Periodic maintenance - events update themselves via on_time_tick() trigger.

        Each event's metabolism decides what action to take based on its state:
        - REGENERATE_NARRATIVE if narrative is stale
        - HIBERNATE if dormant for too long
        - EMIT_THOUGHT if anomaly detected
        - NO_OP if all is well

        Called every hour by worker.
        """
        if not self.active:
            logger.info("😴 No active events in pool")
            return

        logger.info(f"🔄 Metabolism cycle: {len(self.active)} active events")

        for event_id in list(self.active.keys()):
            live_event = self.active[event_id]

            try:
                # Trigger time-based metabolism - let the event decide what to do
                result = await live_event.on_time_tick()

                # Check if event decided to hibernate
                if hasattr(live_event, '_metabolism_state') and live_event._metabolism_state.is_hibernating:
                    logger.info(f"😴 Hibernating: {live_event.event.canonical_name} (idle {live_event.idle_time_seconds()/3600:.1f}h)")
                    await self._hibernate_event(event_id)

                # Log action taken
                if result.action_type.value != 'no_op':
                    logger.info(f"⚡ {live_event.event.canonical_name}: {result.action_type.value}")

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

    async def handle_command(self, event_id: str, command: str, params: dict = None) -> dict:
        """
        Handle a command for a living event.

        This is the pool's command router - it loads the event if needed
        and dispatches the command to the LiveEvent's handler.

        Supported commands:
        - /retopologize: Re-run full Bayesian topology analysis
        - /regenerate: Regenerate narrative
        - /hibernate: Force hibernate
        - /status: Return current status

        Args:
            event_id: Target event ID
            command: Command path (e.g., '/retopologize')
            params: Optional parameters

        Returns:
            Result dict with 'success' and optional 'data' or 'error'
        """
        params = params or {}

        logger.info(f"🎮 Pool handling command: {command} for {event_id}")

        # Load event if not in pool
        if event_id not in self.active:
            logger.info(f"📥 Loading event {event_id} for command")
            await self._load_event(event_id)

        if event_id not in self.active:
            return {'success': False, 'error': f'Event {event_id} not found'}

        live_event = self.active[event_id]

        # Dispatch command to LiveEvent
        try:
            result = await live_event.handle_command(command, params)
            return result
        except Exception as e:
            logger.error(f"❌ Command {command} failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def __repr__(self):
        return f"<LiveEventPool: {len(self.active)} active events>"
