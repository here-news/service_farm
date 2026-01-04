"""
Story domain model - Unified API-facing object for L3/L4

A Story is the user-facing representation of either:
- scale="incident": L3 tight membrane (single happening)
- scale="case": L4 loose membrane (grouped happenings)

This unifies what downstream consumes while keeping kernel precision.
The kernel uses Incident/Case internally; this is the API projection.

Key principle: One type, two scales. No "Event/Incident/View" confusion.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Set, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .incident import Incident
    from .case import Case


StoryScale = Literal["incident", "case"]


@dataclass
class Story:
    """
    Unified API-facing story object.

    Combines incident (L3) and case (L4) into single response type.
    Scale field distinguishes which layer this represents.

    ID format:
    - scale="incident": incident_xxxxxxxx
    - scale="case": case_xxxxxxxxxxxx
    """
    id: str
    scale: StoryScale

    # Stable identity (deterministic hash for reproducibility)
    scope_signature: str = ""

    # Content (LLM-generated or derived)
    title: str = ""
    description: str = ""

    # Entities
    primary_entities: List[str] = field(default_factory=list)

    # Stats
    surface_count: int = 0
    source_count: int = 0
    claim_count: int = 0
    incident_count: int = 0  # Only meaningful for scale="case"

    # Case type (only for scale="case")
    # - "developing": CaseCore formed by k>=2 motif recurrence
    # - "entity_storyline": EntityCase formed by focal entity with rotating companions
    case_type: Optional[str] = None

    # Temporal bounds
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Nested structure (optional, for detail views)
    # For scale="case": list of incident stories
    incidents: Optional[List['Story']] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __hash__(self):
        return hash(self.id)

    @property
    def primary_entity(self) -> str:
        """First primary entity for summary display."""
        return self.primary_entities[0] if self.primary_entities else "Unknown"

    @property
    def is_incident(self) -> bool:
        return self.scale == "incident"

    @property
    def is_case(self) -> bool:
        return self.scale == "case"

    def to_summary(self) -> dict:
        """Convert to summary dict for list views."""
        result = {
            "id": self.id,
            "scale": self.scale,
            "title": self.title,
            "description": self.description,
            "primary_entity": self.primary_entity,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "source_count": self.source_count,
            "surface_count": self.surface_count,
            "incident_count": self.incident_count if self.is_case else None,
        }
        if self.is_case and self.case_type:
            result["case_type"] = self.case_type
        return result

    def to_detail(self) -> dict:
        """Convert to detail dict for single-item views."""
        result = self.to_summary()
        result.update({
            "primary_entities": self.primary_entities,
            "claim_count": self.claim_count,
            "scope_signature": self.scope_signature,
            "incidents": [i.to_summary() for i in self.incidents] if self.incidents else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        })
        return result

    @classmethod
    def from_incident(
        cls,
        incident: 'Incident',
        title: str = "",
        description: str = "",
    ) -> 'Story':
        """
        Create Story from L3 Incident.

        Args:
            incident: Internal Incident domain object
            title: LLM-generated or derived title
            description: LLM-generated description

        Returns:
            Story with scale="incident"
        """
        return cls(
            id=incident.id,
            scale="incident",
            scope_signature=incident.id,  # Incident ID is its signature
            title=title or incident.canonical_title or f"{list(incident.anchor_entities)[0]} Incident" if incident.anchor_entities else "Incident",
            description=description,
            primary_entities=list(incident.anchor_entities)[:5],
            surface_count=incident.surface_count,
            source_count=0,  # Needs aggregation from surfaces
            claim_count=0,   # Needs aggregation from surfaces
            incident_count=0,
            time_start=incident.time_start,
            time_end=incident.time_end,
            created_at=incident.created_at,
            updated_at=incident.updated_at,
        )

    @classmethod
    def from_case(cls, case: 'Case') -> 'Story':
        """
        Create Story from L4 Case.

        Args:
            case: Internal Case domain object

        Returns:
            Story with scale="case"
        """
        return cls(
            id=case.id,
            scale="case",
            scope_signature=case.id,  # Case ID is hash of incidents
            title=case.title,
            description=case.description,
            primary_entities=case.primary_entities,
            surface_count=case.surface_count,
            source_count=case.source_count,
            claim_count=case.claim_count,
            incident_count=case.incident_count,
            time_start=case.time_start,
            time_end=case.time_end,
            created_at=case.created_at,
            updated_at=case.updated_at,
        )

    @classmethod
    def from_neo4j_row(cls, row: dict) -> 'Story':
        """
        Create Story from Neo4j query result.

        Expects row with: id, scale, title, description, primary_entities,
        surface_count, source_count, claim_count, incident_count,
        time_start, time_end, scope_signature, case_type (optional)
        """
        return cls(
            id=row['id'],
            scale=row.get('scale', 'incident'),
            scope_signature=row.get('scope_signature', row['id']),
            title=row.get('title') or "",
            description=row.get('description') or "",
            primary_entities=row.get('primary_entities') or [],
            surface_count=row.get('surface_count') or 0,
            source_count=row.get('source_count') or 0,
            claim_count=row.get('claim_count') or 0,
            incident_count=row.get('incident_count') or 0,
            case_type=row.get('case_type'),
            time_start=cls._parse_time(row.get('time_start')),
            time_end=cls._parse_time(row.get('time_end')),
            created_at=cls._parse_time(row.get('created_at')),
            updated_at=cls._parse_time(row.get('updated_at')),
        )

    @classmethod
    def from_kernel(cls, kernel_story) -> 'Story':
        """
        Convert kernel Story (reee.types.Story) to domain Story.

        This bridges the kernel layer to the API/domain layer.
        """
        return cls(
            id=kernel_story.id,
            scale=kernel_story.scale,
            scope_signature=kernel_story.scope_signature,
            title=kernel_story.title,
            description=kernel_story.description,
            primary_entities=kernel_story.primary_entities,
            surface_count=kernel_story.surface_count,
            source_count=kernel_story.source_count,
            claim_count=kernel_story.claim_count,
            incident_count=kernel_story.incident_count,
            time_start=kernel_story.time_start,
            time_end=kernel_story.time_end,
        )

    @staticmethod
    def _parse_time(val) -> Optional[datetime]:
        """Parse time from various formats."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace('Z', '+00:00'))
            except:
                return None
        return None
