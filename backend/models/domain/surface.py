"""
Surface domain model - L2 identity cluster

A Surface is a bundle of claims connected by identity edges.
Claims in a surface are about the "same referent" (same entity, same proposition).

Key properties:
- support: Epistemic confidence in the identity (NOT for threshold modulation)
- entropy: Uncertainty in typed values within this surface
- sources: Independent attestations
- anchor_entities: Discriminative entities that identify this surface
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Set, TYPE_CHECKING
import math

from utils.id_generator import generate_id, validate_id, is_uuid, uuid_to_short_id

if TYPE_CHECKING:
    from .claim import Claim


@dataclass
class Surface:
    """
    L2 Surface - storage-agnostic representation.

    Represents an identity cluster of claims about the same referent.

    ID format: sf_xxxxxxxx (11 chars)
    """
    id: str  # Short ID: sf_xxxxxxxx

    # Claim membership (unique claim IDs)
    claim_ids: Set[str] = field(default_factory=set)

    # Entity signals
    entities: Set[str] = field(default_factory=set)  # All entities mentioned
    anchor_entities: Set[str] = field(default_factory=set)  # Discriminative anchors

    # Sources (for support computation)
    sources: Set[str] = field(default_factory=set)

    # Temporal bounds
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Support (diagnostic, not decision input)
    # "How well-attested is this identity cluster?"
    support: float = 0.0

    # Centroid embedding (stored in PostgreSQL)
    centroid: Optional[List[float]] = None

    # Params version (for reproducibility)
    params_version: int = 1

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Lazy-loaded claims (from Neo4j)
    claims: Optional[List['Claim']] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure ID is in short format."""
        if self.id:
            if is_uuid(self.id):
                self.id = uuid_to_short_id(self.id, 'surface')
            elif not validate_id(self.id):
                self.id = generate_id('surface')

        # Ensure sets
        if isinstance(self.claim_ids, list):
            self.claim_ids = set(self.claim_ids)
        if isinstance(self.entities, list):
            self.entities = set(self.entities)
        if isinstance(self.anchor_entities, list):
            self.anchor_entities = set(self.anchor_entities)
        if isinstance(self.sources, list):
            self.sources = set(self.sources)

    def __hash__(self):
        return hash(self.id)

    @property
    def claim_count(self) -> int:
        return len(self.claim_ids)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def compute_support(self) -> float:
        """
        Compute support score (epistemic confidence in identity).

        Support = f(claim_count, source_diversity)
        Used for: prioritization, compute allocation
        NOT for: lowering thresholds, weighting posteriors
        """
        # Claim count (diminishing returns)
        claim_mass = math.log(1 + len(self.claim_ids))

        # Source diversity (independent attestations)
        diversity_mass = math.log(1 + len(self.sources)) * 2

        self.support = claim_mass + diversity_mass
        return self.support

    def add_claim(self, claim: 'Claim', publisher_id: Optional[str] = None) -> None:
        """Add a claim to this surface."""
        self.claim_ids.add(claim.id)

        # Update entities
        if hasattr(claim, 'entity_ids'):
            self.entities.update(claim.entity_ids)

        # Update anchors (if claim has them)
        if hasattr(claim, 'anchor_entities') and claim.anchor_entities:
            self.anchor_entities.update(claim.anchor_entities)

        # Update sources (use publisher_id for true source diversity)
        # Falls back to page_id if publisher not available
        source = publisher_id or (claim.page_id if hasattr(claim, 'page_id') else None)
        if source:
            self.sources.add(source)

        # Update time bounds (handle string timestamps)
        from dateutil.parser import parse as parse_date

        event_time = claim.event_time
        if isinstance(event_time, str):
            event_time = parse_date(event_time)

        # Also convert surface times if they're strings
        if isinstance(self.time_start, str):
            self.time_start = parse_date(self.time_start)
        if isinstance(self.time_end, str):
            self.time_end = parse_date(self.time_end)

        if event_time:
            if self.time_start is None or event_time < self.time_start:
                self.time_start = event_time
            if self.time_end is None or event_time > self.time_end:
                self.time_end = event_time

        self.updated_at = datetime.utcnow()

    def merge_from(self, other: 'Surface') -> None:
        """Merge another surface into this one."""
        self.claim_ids.update(other.claim_ids)
        self.entities.update(other.entities)
        self.anchor_entities.update(other.anchor_entities)
        self.sources.update(other.sources)

        # Update time bounds
        if other.time_start:
            if self.time_start is None or other.time_start < self.time_start:
                self.time_start = other.time_start
        if other.time_end:
            if self.time_end is None or other.time_end > self.time_end:
                self.time_end = other.time_end

        self.compute_support()
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "id": self.id,
            "claim_count": self.claim_count,
            "source_count": self.source_count,
            "entities": list(self.entities),
            "anchor_entities": list(self.anchor_entities),
            "sources": list(self.sources),
            "support": round(self.support, 2),
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
