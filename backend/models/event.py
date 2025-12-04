"""
Event domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import uuid


@dataclass
class Event:
    """
    Event domain model - storage-agnostic representation

    Represents a real-world event discovered from multiple pages
    """
    id: uuid.UUID
    title: str
    event_type: str  # FIRE, SHOOTING, PROTEST, etc.

    # Temporal bounds
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None

    # Event properties
    status: str = 'provisional'  # provisional, emerging, stable
    confidence: float = 0.5
    event_scale: str = 'micro'  # micro, meso, macro

    # Summary/description
    summary: Optional[str] = None
    location: Optional[str] = None

    # Counts
    pages_count: int = 0
    claims_count: int = 0

    # Embedding (stored in PostgreSQL as vector, computed from page embeddings)
    embedding: Optional[List[float]] = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure id is UUID"""
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)

    @property
    def is_provisional(self) -> bool:
        """Check if event is provisional (1 page)"""
        return self.status == 'provisional'

    @property
    def is_emerging(self) -> bool:
        """Check if event is emerging (2-4 pages)"""
        return self.status == 'emerging'

    @property
    def is_stable(self) -> bool:
        """Check if event is stable (5+ pages)"""
        return self.status == 'stable'

    @property
    def has_temporal_bounds(self) -> bool:
        """Check if event has temporal bounds"""
        return self.event_start is not None and self.event_end is not None

    def update_status_from_page_count(self):
        """Update status based on page count"""
        if self.pages_count >= 5:
            self.status = 'stable'
            self.confidence = 0.95
        elif self.pages_count >= 2:
            self.status = 'emerging'
            self.confidence = 0.75
        else:
            self.status = 'provisional'
            self.confidence = 0.5
