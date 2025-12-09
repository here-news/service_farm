"""
Event domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Set, TYPE_CHECKING
from enum import Enum

from utils.id_generator import (
    generate_event_id, validate_id, is_uuid, uuid_to_short_id
)

if TYPE_CHECKING:
    from .claim import Claim


class ClaimDecision(Enum):
    """Decision on how to handle a new claim"""
    MERGE = "merge"              # Duplicate - corroborate existing claim
    ADD = "add"                  # Novel but fits this event's topic
    DELEGATE = "delegate"        # Sub-event handles it better
    YIELD_SUBEVENT = "yield"     # Novel aspect - create sub-event
    REJECT = "reject"            # Doesn't belong here


@dataclass
class ExaminationResult:
    """Result of examining claims"""
    claims_added: List['Claim'] = field(default_factory=list)
    sub_events_created: List['Event'] = field(default_factory=list)
    claims_rejected: List['Claim'] = field(default_factory=list)


@dataclass
class Event:
    """
    Event domain model - storage-agnostic representation

    Represents a real-world event discovered from multiple pages.
    Events are RECURSIVE - can contain sub-events (phases/aspects).

    ID format: ev_xxxxxxxx (11 chars)
    """
    id: str  # Short ID: ev_xxxxxxxx
    canonical_name: str  # Canonical event name
    event_type: str  # FIRE, SHOOTING, PROTEST, etc.

    # Recursive structure
    parent_event_id: Optional[str] = None  # Short ID: ev_xxxxxxxx

    # NOTE: Claims are linked via Neo4j graph relationships (Event-[SUPPORTS]->Claim)
    # Use EventRepository.get_event_claims() to fetch claims for an event

    # Quality metrics
    confidence: float = 0.3  # Evidence strength (0-1)
    coherence: float = 0.5   # How well claims fit together (0-1)

    # Temporal bounds (matches database schema: event_start, event_end)
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None

    # Event properties
    status: str = 'provisional'  # provisional, stable, archived
    event_scale: str = 'micro'  # micro, meso, macro

    # Summary/description
    summary: Optional[str] = None
    location: Optional[str] = None

    # Counts (legacy - for compatibility)
    pages_count: int = 0
    claims_count: int = 0

    # Embedding (stored in PostgreSQL as vector)
    embedding: Optional[List[float]] = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Lazy-loaded relationships
    sub_events: Optional[List['Event']] = field(default=None, repr=False)
    entities: Optional[Set[str]] = field(default=None, repr=False)  # Set of en_xxxxxxxx

    def __post_init__(self):
        """Ensure IDs are in short format, convert UUIDs if needed"""
        if self.id:
            if is_uuid(self.id):
                self.id = uuid_to_short_id(self.id, 'event')
            elif not validate_id(self.id):
                self.id = generate_event_id()

        if self.parent_event_id and is_uuid(self.parent_event_id):
            self.parent_event_id = uuid_to_short_id(self.parent_event_id, 'event')

    @property
    def is_root(self) -> bool:
        """Check if this is a root event (no parent)"""
        return self.parent_event_id is None

    @property
    def is_sub_event(self) -> bool:
        """Check if this is a sub-event (has parent)"""
        return self.parent_event_id is not None

    @property
    def is_provisional(self) -> bool:
        """Check if event is provisional"""
        return self.status == 'provisional'

    @property
    def is_stable(self) -> bool:
        """Check if event is stable"""
        return self.status == 'stable'

    @property
    def has_temporal_bounds(self) -> bool:
        """Check if event has temporal bounds"""
        return self.event_start is not None and self.event_end is not None

    def update_status(self):
        """
        Update status based on confidence and claim count

        Note: claim_count now requires fetching from graph via repository
        """
        if self.confidence >= 0.7 and self.claims_count >= 3:
            self.status = 'stable'
        else:
            self.status = 'provisional'

    # Recursive examination methods (to be implemented in service layer)
    # These are placeholders - actual logic lives in EventService
    async def examine(self, claims: List['Claim']) -> ExaminationResult:
        """
        Recursively examine new claims and integrate into event structure

        This is a placeholder - actual implementation is in EventService
        to avoid circular dependencies and access to repositories
        """
        raise NotImplementedError("Use EventService.examine_claims() instead")
