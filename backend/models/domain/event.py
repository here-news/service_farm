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
class NarrativeSection:
    """
    A section of a structured narrative.

    Each section covers a specific topic (casualties, response, investigation, etc.)
    and contains prose with embedded claim/entity references.
    """
    topic: str              # Topic key: "casualties", "response", "investigation", etc.
    title: str              # Display title: "Casualties", "Emergency Response", etc.
    content: str            # Prose with [cl_xxx] and [en_xxx] markers
    claim_ids: List[str] = field(default_factory=list)  # Claims used in this section

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "title": self.title,
            "content": self.content,
            "claim_ids": self.claim_ids
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NarrativeSection':
        return cls(
            topic=data.get("topic", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            claim_ids=data.get("claim_ids", [])
        )


@dataclass
class KeyFigure:
    """A key numeric figure extracted from claims (death toll, injuries, etc.)"""
    label: str              # "death_toll", "injuries", "arrests", etc.
    value: str              # "160", "76", etc.
    claim_id: str           # Source claim for this figure
    supersedes: Optional[str] = None  # Earlier claim_id this figure supersedes

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "claim_id": self.claim_id,
            "supersedes": self.supersedes
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KeyFigure':
        return cls(
            label=data.get("label", ""),
            value=data.get("value", ""),
            claim_id=data.get("claim_id", ""),
            supersedes=data.get("supersedes")
        )


@dataclass
class StructuredNarrative:
    """
    Structured narrative for an event.

    Unlike free-form summary text, this structure allows:
    - Independent section updates when new claims arrive
    - Frontend rendering of sections separately
    - Key figures extracted explicitly (not buried in prose)
    - Pattern-aware presentation (progressive, contradictory, consensus)

    The canonical_name from Event serves as the headline - no need for
    LLM-generated "The X: A Tragic Event" style titles.
    """
    sections: List[NarrativeSection] = field(default_factory=list)
    key_figures: List[KeyFigure] = field(default_factory=list)
    pattern: str = "unknown"  # "consensus", "progressive", "contradictory"
    consensus_date: Optional[str] = None
    generated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "sections": [s.to_dict() for s in self.sections],
            "key_figures": [f.to_dict() for f in self.key_figures],
            "pattern": self.pattern,
            "consensus_date": self.consensus_date,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StructuredNarrative':
        if not data:
            return cls()
        return cls(
            sections=[NarrativeSection.from_dict(s) for s in data.get("sections", [])],
            key_figures=[KeyFigure.from_dict(f) for f in data.get("key_figures", [])],
            pattern=data.get("pattern", "unknown"),
            consensus_date=data.get("consensus_date"),
            generated_at=datetime.fromisoformat(data["generated_at"]) if data.get("generated_at") else None
        )

    def to_flat_text(self) -> str:
        """Convert to flat text for backwards compatibility with Event.summary"""
        parts = []
        for section in self.sections:
            if section.title:
                parts.append(f"**{section.title}**")
            parts.append(section.content)
            parts.append("")  # Blank line between sections
        return "\n".join(parts).strip()

    def get_section(self, topic: str) -> Optional[NarrativeSection]:
        """Get a section by topic key"""
        for section in self.sections:
            if section.topic == topic:
                return section
        return None


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

    # NOTE: Claims are linked via Neo4j graph relationships (Event-[INTAKES]->Claim)
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
    summary: Optional[str] = None  # Legacy flat text narrative
    narrative: Optional[StructuredNarrative] = None  # Structured narrative
    location: Optional[str] = None

    # Counts (legacy - for compatibility)
    pages_count: int = 0
    claims_count: int = 0

    # Embedding (stored in PostgreSQL as vector)
    embedding: Optional[List[float]] = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    # Semantic versioning: major.minor
    # - Minor bump: narrative regeneration, claim additions
    # - Major bump: coherence leap (≥0.1 increase), pattern change, significant restructure
    version_major: int = 0
    version_minor: int = 1

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

    @property
    def version(self) -> str:
        """Semantic version string (major.minor)"""
        return f"{self.version_major}.{self.version_minor}"

    def bump_minor(self) -> str:
        """Bump minor version (narrative update, claim addition). Returns new version."""
        self.version_minor += 1
        return self.version

    def bump_major(self, reset_minor: bool = True) -> str:
        """Bump major version (coherence leap, pattern change). Returns new version."""
        self.version_major += 1
        if reset_minor:
            self.version_minor = 0
        return self.version

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
