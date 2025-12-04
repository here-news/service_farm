"""
Phase domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import uuid


@dataclass
class Phase:
    """
    Phase domain model - storage-agnostic representation

    Represents a semantic phase within an event (e.g., "Fire Breakout", "Casualties")
    """
    id: uuid.UUID
    event_id: uuid.UUID
    name: str
    phase_type: str  # INCIDENT, CONSEQUENCE, RESPONSE, INVESTIGATION, etc.

    # Temporal bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Phase properties
    sequence: int = 1
    confidence: float = 0.9
    description: Optional[str] = None

    # Embedding (stored in PostgreSQL as vector, computed from claim embeddings)
    embedding: Optional[List[float]] = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure IDs are UUIDs"""
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)
        if isinstance(self.event_id, str):
            self.event_id = uuid.UUID(self.event_id)

    @property
    def is_incident_phase(self) -> bool:
        """Check if phase is an incident/initial event phase"""
        return self.phase_type == 'INCIDENT'

    @property
    def is_consequence_phase(self) -> bool:
        """Check if phase represents consequences"""
        return self.phase_type == 'CONSEQUENCE'

    @property
    def is_response_phase(self) -> bool:
        """Check if phase represents emergency response"""
        return self.phase_type == 'RESPONSE'

    @property
    def is_investigation_phase(self) -> bool:
        """Check if phase represents investigation"""
        return self.phase_type == 'INVESTIGATION'
