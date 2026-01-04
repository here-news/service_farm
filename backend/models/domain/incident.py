"""
Incident domain model - L3 membrane over L2 surfaces

An Incident groups surfaces that share anchor entities and compatible
companion contexts. It represents a coherent "happening" like:
- A fire breaking out
- A policy announcement
- An arrest

Key properties:
- anchor_entities: Core entities that define the incident identity
- companion_entities: Context entities (for bridge immunity check)
- surfaces: L2 surfaces contained in this incident
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Set, Dict

from utils.id_generator import generate_id, validate_id, is_uuid, uuid_to_short_id


@dataclass
class Incident:
    """
    L3 Incident - storage-agnostic representation.

    Represents a membrane over L2 surfaces that share anchor entities
    and compatible companion contexts.

    ID format: incident_xxxxxxxx
    """
    id: str

    # Entity signals
    anchor_entities: Set[str] = field(default_factory=set)  # Core identity
    companion_entities: Set[str] = field(default_factory=set)  # Context

    # Surface membership
    surface_ids: Set[str] = field(default_factory=set)

    # Temporal bounds (derived from surfaces)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Canonical title (LLM-generated or derived)
    canonical_title: str = ""

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure sets are proper sets."""
        if isinstance(self.anchor_entities, list):
            self.anchor_entities = set(self.anchor_entities)
        if isinstance(self.companion_entities, list):
            self.companion_entities = set(self.companion_entities)
        if isinstance(self.surface_ids, list):
            self.surface_ids = set(self.surface_ids)

    def __hash__(self):
        return hash(self.id)

    @property
    def surface_count(self) -> int:
        return len(self.surface_ids)

    @property
    def all_entities(self) -> Set[str]:
        """All entities (anchors + companions)."""
        return self.anchor_entities | self.companion_entities

    def add_surface(self, surface_id: str, anchors: Set[str], companions: Set[str]) -> None:
        """Add a surface to this incident."""
        self.surface_ids.add(surface_id)
        self.anchor_entities.update(anchors)
        self.companion_entities.update(companions)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "id": self.id,
            "anchor_entities": list(self.anchor_entities),
            "companion_entities": list(self.companion_entities),
            "surface_count": self.surface_count,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "canonical_title": self.canonical_title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
