"""
Claim domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .entity import Entity


@dataclass
class Claim:
    """
    Claim domain model - storage-agnostic representation

    Represents a factual assertion extracted from a page
    """
    id: uuid.UUID
    page_id: uuid.UUID
    text: str

    # Claim properties
    event_time: Optional[datetime] = None
    confidence: float = 0.8
    modality: str = 'observation'  # observation, prediction, speculation, opinion

    # Embedding (stored in PostgreSQL as vector)
    embedding: Optional[List[float]] = None

    # Metadata (includes entity_ids as JSON)
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None

    # Lazy-loaded entities (fetched from Neo4j via repository)
    entities: Optional[List['Entity']] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure IDs are UUIDs"""
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)
        if isinstance(self.page_id, str):
            self.page_id = uuid.UUID(self.page_id)

    @property
    def is_factual(self) -> bool:
        """Check if claim is factual (not opinion or speculation)"""
        return self.modality in ['observation', 'prediction']

    @property
    def is_timestamped(self) -> bool:
        """Check if claim has event time"""
        return self.event_time is not None

    @property
    def entity_ids(self) -> List[uuid.UUID]:
        """Extract entity IDs from metadata JSON"""
        entity_id_strings = self.metadata.get('entity_ids', [])
        return [uuid.UUID(eid) for eid in entity_id_strings]

    @property
    def entity_names(self) -> List[str]:
        """Extract entity names from metadata JSON (for debugging)"""
        return self.metadata.get('entity_names', [])
