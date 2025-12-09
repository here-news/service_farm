"""
Claim domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

from utils.id_generator import (
    generate_claim_id, validate_id, is_uuid, uuid_to_short_id
)

if TYPE_CHECKING:
    from .entity import Entity


@dataclass
class Claim:
    """
    Claim domain model - storage-agnostic representation

    Represents a factual assertion extracted from a page

    ID format: cl_xxxxxxxx (11 chars)
    """
    id: str  # Short ID: cl_xxxxxxxx
    page_id: str  # Short ID: pg_xxxxxxxx
    text: str

    # Claim properties
    event_time: Optional[datetime] = None  # When the fact occurred (ground truth time)
    reported_time: Optional[datetime] = None  # When we learned/published it
    confidence: float = 0.8
    modality: str = 'observation'  # observation, prediction, speculation, opinion

    # Update chains (for evolving facts)
    updates_claim_id: Optional[str] = None  # Points to claim this supersedes
    is_superseded: bool = False  # True if a newer claim has arrived
    topic_key: Optional[str] = None  # e.g. "casualty_count", "alarm_level"

    # Embedding (stored in PostgreSQL as vector)
    embedding: Optional[List[float]] = None

    # Metadata (includes entity_ids as JSON)
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None

    # Lazy-loaded entities (fetched from Neo4j via repository)
    entities: Optional[List['Entity']] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure IDs are in short format, convert UUIDs if needed"""
        if self.id:
            if is_uuid(self.id):
                self.id = uuid_to_short_id(self.id, 'claim')
            elif not validate_id(self.id):
                self.id = generate_claim_id()

        if self.page_id and is_uuid(self.page_id):
            self.page_id = uuid_to_short_id(self.page_id, 'page')

        if self.updates_claim_id and is_uuid(self.updates_claim_id):
            self.updates_claim_id = uuid_to_short_id(self.updates_claim_id, 'claim')

    @property
    def is_factual(self) -> bool:
        """Check if claim is factual (not opinion or speculation)"""
        return self.modality in ['observation', 'prediction']

    @property
    def is_timestamped(self) -> bool:
        """Check if claim has event time"""
        return self.event_time is not None

    @property
    def entity_ids(self) -> List[str]:
        """Extract entity IDs from metadata JSON (short format)"""
        entity_id_strings = self.metadata.get('entity_ids', [])
        # Convert any UUIDs to short format
        result = []
        for eid in entity_id_strings:
            if is_uuid(eid):
                result.append(uuid_to_short_id(eid, 'entity'))
            else:
                result.append(eid)
        return result

    @property
    def entity_names(self) -> List[str]:
        """Extract entity names from metadata JSON (for debugging)"""
        return self.metadata.get('entity_names', [])

    @property
    def is_update(self) -> bool:
        """Check if this claim updates another claim"""
        return self.updates_claim_id is not None

    @property
    def is_current(self) -> bool:
        """Check if this is the current (non-superseded) claim"""
        return not self.is_superseded
