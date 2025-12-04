"""
Claim domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import uuid


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

    # Metadata
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None

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
