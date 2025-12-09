"""
EventPool - Manages live events with weight-based eviction.

Single-process implementation that can be upgraded to distributed later.

Lifecycle:
1. get_or_create(dedup_key, proto) → LiveEvent
2. LiveEvent processes claims via metabolize()
3. When pool is full, lowest-weight events are evicted (checkpointed to Neo4j)

Upgrade Path:
- Phase 1 (Now): Single process, all events in memory
- Phase 2: Add Redis directory for distributed ownership
- Phase 3: Add migration for auto-scaling
"""
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime

from models.domain.live_event import LiveEvent
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class EventPool:
    """
    Pool of live events with weight-based eviction.

    Events stay "hot" (in memory) while actively processing.
    When memory pressure increases, lowest-weight events are
    checkpointed to Neo4j and evicted.
    """

    def __init__(
        self,
        neo4j_service: Neo4jService,
        max_events: int = 100,
        eviction_batch_size: int = 10
    ):
        """
        Args:
            neo4j_service: Neo4j service for persistence
            max_events: Maximum hot events (scale with available memory)
            eviction_batch_size: How many to evict when at capacity
        """
        self.neo4j = neo4j_service
        self.max_events = max_events
        self.eviction_batch_size = eviction_batch_size

        self.events: Dict[str, LiveEvent] = {}
        self._lock = asyncio.Lock()  # Protect concurrent access

        logger.info(f"🎱 EventPool initialized (max={max_events})")

    async def get_or_create(
        self,
        dedup_key: str,
        proto: Optional[dict] = None
    ) -> Optional[LiveEvent]:
        """
        Get hot event or create/hydrate one.

        Args:
            dedup_key: Event's deduplication key
            proto: Proto-event dict if creating new (from extraction)

        Returns:
            LiveEvent or None if not found and no proto provided
        """
        async with self._lock:
            # 1. Already hot?
            if dedup_key in self.events:
                event = self.events[dedup_key]
                event.last_activity = datetime.now()
                logger.debug(f"🔥 Hot event: {event.canonical_name}")
                return event

            # 2. Try to hydrate from Neo4j
            event = await LiveEvent.hydrate(self.neo4j, dedup_key)

            if event:
                await self._admit(dedup_key, event)
                logger.info(f"💧 Rehydrated: {event.canonical_name}")
                return event

            # 3. Create new from proto
            if proto and proto.get('title'):
                event = LiveEvent.from_proto(proto)
                event._neo4j_service = self.neo4j
                await self._admit(dedup_key, event)
                logger.info(f"✨ Created: {event.canonical_name}")
                return event

            logger.warning(f"⚠️ Cannot find or create event: {dedup_key}")
            return None

    async def _admit(self, dedup_key: str, event: LiveEvent):
        """Add event to pool, evicting if necessary"""
        # Check capacity
        if len(self.events) >= self.max_events:
            await self._evict_batch()

        self.events[dedup_key] = event
        event._neo4j_service = self.neo4j

    async def _evict_batch(self):
        """Evict lowest-weight events to make room"""
        if not self.events:
            return

        # Sort by weight ascending
        sorted_events = sorted(
            self.events.items(),
            key=lambda kv: kv[1].weight
        )

        # Evict lowest N
        to_evict = sorted_events[:self.eviction_batch_size]

        for dedup_key, event in to_evict:
            await self._evict_one(dedup_key, event)

        logger.info(f"🗑️ Evicted {len(to_evict)} events (pool: {len(self.events)}/{self.max_events})")

    async def _evict_one(self, dedup_key: str, event: LiveEvent):
        """Checkpoint and remove single event"""
        try:
            await event.checkpoint()
            del self.events[dedup_key]
            logger.debug(f"💾 Evicted: {event.canonical_name} (weight={event.weight:.2f})")
        except Exception as e:
            logger.error(f"❌ Failed to evict {event.canonical_name}: {e}")

    async def evict(self, dedup_key: str):
        """Manually evict a specific event"""
        async with self._lock:
            if dedup_key in self.events:
                event = self.events[dedup_key]
                await self._evict_one(dedup_key, event)

    async def checkpoint_all(self):
        """Checkpoint all events (for graceful shutdown)"""
        async with self._lock:
            logger.info(f"💾 Checkpointing {len(self.events)} events...")
            for dedup_key, event in list(self.events.items()):
                try:
                    await event.checkpoint()
                except Exception as e:
                    logger.error(f"❌ Failed to checkpoint {event.canonical_name}: {e}")

    def get_stats(self) -> dict:
        """Get pool statistics"""
        if not self.events:
            return {
                'hot_events': 0,
                'max_events': self.max_events,
                'utilization': 0.0
            }

        weights = [e.weight for e in self.events.values()]
        return {
            'hot_events': len(self.events),
            'max_events': self.max_events,
            'utilization': len(self.events) / self.max_events,
            'min_weight': min(weights),
            'max_weight': max(weights),
            'avg_weight': sum(weights) / len(weights),
            'total_claims': sum(len(e.core_claims) for e in self.events.values()),
            'total_pages': sum(e.pages_count for e in self.events.values())
        }

    def list_events(self) -> List[dict]:
        """List all hot events with their stats"""
        return [
            {
                'dedup_key': key,
                'canonical_name': event.canonical_name,
                'event_type': event.event_type,
                'weight': event.weight,
                'claims': len(event.core_claims),
                'entities': len(event.core_entities),
                'pages': event.pages_count,
                'sub_events': len(event.sub_events),
                'last_activity': event.last_activity.isoformat()
            }
            for key, event in sorted(
                self.events.items(),
                key=lambda kv: -kv[1].weight  # Sort by weight descending
            )
        ]

    def __len__(self):
        return len(self.events)

    def __contains__(self, dedup_key: str):
        return dedup_key in self.events


# Singleton instance (for single-worker setup)
_pool_instance: Optional[EventPool] = None


def get_event_pool() -> Optional[EventPool]:
    """Get the singleton EventPool instance"""
    return _pool_instance


async def init_event_pool(neo4j_service: Neo4jService, max_events: int = 100) -> EventPool:
    """Initialize the singleton EventPool"""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = EventPool(neo4j_service, max_events)
    return _pool_instance


async def shutdown_event_pool():
    """Graceful shutdown - checkpoint all events"""
    global _pool_instance
    if _pool_instance:
        await _pool_instance.checkpoint_all()
        _pool_instance = None
