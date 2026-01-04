"""
Case domain model - L4 membrane over L3 incidents

A Case groups related incidents that share binding evidence:
- At least 2 incidents
- At least 1 non-trivial binding constraint (shared anchors, time proximity)

Cases represent user-facing "stories" like:
- "Hong Kong Fire Response" (fire + rescue + political response incidents)
- "Epstein Investigation" (arrest + court proceedings + political fallout)

Key properties:
- incident_ids: L3 incidents contained in this case
- primary_entities: Key entities across all incidents
- binding_evidence: Why these incidents belong together
- title/description: LLM-generated narrative (never entity concatenation)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Set
import hashlib

from utils.id_generator import generate_id


@dataclass
class Case:
    """
    L4 Case - storage-agnostic representation.

    Represents a user-facing event that emerges from clustering
    L3 incidents with binding evidence.

    ID format: case_xxxxxxxxxxxx (stable hash of incident cluster)

    Invariants:
    - Must have >= 2 incidents
    - Must have binding evidence (shared entities, time proximity)
    - Title must describe the STORY, not just entity names
    """
    id: str

    # Incident membership
    incident_ids: Set[str] = field(default_factory=set)

    # Entity signals (aggregated from incidents)
    primary_entities: List[str] = field(default_factory=list)  # Most connected entities

    # Narrative (LLM-generated)
    title: str = ""
    description: str = ""

    # Classification
    case_type: str = "developing"  # breaking, developing, ongoing, resolved

    # Binding evidence (why these incidents belong together)
    binding_evidence: List[str] = field(default_factory=list)

    # Aggregate stats
    surface_count: int = 0
    source_count: int = 0
    claim_count: int = 0

    # Temporal bounds (derived from incidents)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure sets are proper sets."""
        if isinstance(self.incident_ids, list):
            self.incident_ids = set(self.incident_ids)
        if isinstance(self.primary_entities, set):
            self.primary_entities = list(self.primary_entities)

    def __hash__(self):
        return hash(self.id)

    @property
    def incident_count(self) -> int:
        return len(self.incident_ids)

    @property
    def is_valid(self) -> bool:
        """Check if case meets minimum requirements."""
        return len(self.incident_ids) >= 2

    @staticmethod
    def generate_stable_id(incident_ids: Set[str]) -> str:
        """Generate stable case ID from incident cluster."""
        sorted_incidents = sorted(incident_ids)
        signature = f"case:{','.join(sorted_incidents[:5])}"
        hash_hex = hashlib.sha256(signature.encode()).hexdigest()[:12]
        return f"case_{hash_hex}"

    def add_incident(self, incident_id: str) -> None:
        """Add an incident to this case."""
        self.incident_ids.add(incident_id)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "primary_entities": self.primary_entities,
            "case_type": self.case_type,
            "incident_count": self.incident_count,
            "surface_count": self.surface_count,
            "source_count": self.source_count,
            "claim_count": self.claim_count,
            "binding_evidence": self.binding_evidence,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_event_summary(self) -> dict:
        """Convert to event summary format for API compatibility."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "primary_entity": self.primary_entities[0] if self.primary_entities else "Unknown",
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "source_count": self.source_count,
            "surface_count": self.surface_count,
            "case_type": self.case_type,
        }
