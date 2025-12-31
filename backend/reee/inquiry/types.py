"""
Inquiry Types
=============

Data structures for the Inquiry MVP layer.
These wrap REEE structures for user-facing workflow.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Set, Optional, Any, Literal


# =============================================================================
# ENUMS
# =============================================================================

class InquiryStatus(Enum):
    """Lifecycle status of an inquiry."""
    OPEN = "open"                 # Accepting evidence
    RESOLVED = "resolved"         # Posterior >= 95% stable
    STALE = "stale"               # No activity for 7+ days
    CLOSED = "closed"             # Manually closed by creator


class RigorLevel(Enum):
    """
    Rigor badge for inquiry.

    A: Typed + calibrated noise model
    B: Typed + generic priors
    C: Exploratory (no headline "resolved")
    """
    A = "A"  # High rigor - typed, calibrated
    B = "B"  # Medium rigor - typed, generic priors
    C = "C"  # Exploratory - no resolution badge


class ContributionType(Enum):
    """Type of user contribution."""
    EVIDENCE = "evidence"               # Supporting claim with source
    REFUTATION = "refutation"           # Counter-evidence
    ATTRIBUTION = "attribution"         # "A said B reported p"
    SCOPE_CORRECTION = "scope_correction"  # "This is a different incident"
    DISAMBIGUATION = "disambiguation"   # Entity clarification


class TaskType(Enum):
    """System-generated task types (from meta-claims)."""
    NEED_PRIMARY_SOURCE = "need_primary_source"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    SINGLE_SOURCE_ONLY = "single_source_only"
    HIGH_ENTROPY = "high_entropy"
    STALE = "stale"


# =============================================================================
# INQUIRY SCHEMA (typed target)
# =============================================================================

@dataclass
class InquirySchema:
    """
    Defines the typed target for an inquiry.

    Maps to REEE's typed_belief domains.
    """
    # Schema type
    schema_type: Literal[
        "monotone_count",    # death_count, missing_count
        "categorical",       # legal_status, verdict
        "report_truth",      # "did Reuters report p?"
        "quote_authenticity", # "did X say Y?"
        "boolean",           # yes/no
        "custom"             # user-defined
    ] = "boolean"

    # For categorical: allowed values
    categories: List[str] = field(default_factory=list)

    # For count: expected scale
    count_scale: Literal["small", "medium", "large"] = "medium"
    count_max: int = 500
    count_monotone: bool = True

    # For custom: hypothesis set
    hypotheses: List[str] = field(default_factory=list)

    # Rigor level (determines if "resolved" badge shown)
    rigor: RigorLevel = RigorLevel.B


# =============================================================================
# CONTRIBUTION (user submission)
# =============================================================================

@dataclass
class Contribution:
    """
    User submission to an inquiry.

    Becomes L0 claims after processing.
    """
    id: str = field(default_factory=lambda: f"contrib_{uuid.uuid4().hex[:8]}")
    inquiry_id: str = ""
    user_id: str = ""

    # Type and content
    type: ContributionType = ContributionType.EVIDENCE
    text: str = ""  # Quote or description
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    timestamp: Optional[datetime] = None  # Time of original claim

    # Extracted value (for typed inquiries)
    extracted_value: Optional[Any] = None
    observation_kind: Optional[str] = None  # point, lower_bound, interval

    # Processing state
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False
    claim_ids: List[str] = field(default_factory=list)  # L0 claims created

    # Impact (computed after processing)
    posterior_impact: float = 0.0  # Change in MAP probability


# =============================================================================
# TASK (from meta-claims)
# =============================================================================

@dataclass
class InquiryTask:
    """
    Actionable task for an inquiry.

    Generated from REEE meta-claims.
    """
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    inquiry_id: str = ""

    type: TaskType = TaskType.SINGLE_SOURCE_ONLY
    description: str = ""

    # Bounty (credits)
    bounty: float = 0.0

    # Status
    created_at: datetime = field(default_factory=datetime.utcnow)
    claimed_by: Optional[str] = None
    completed: bool = False
    completed_at: Optional[datetime] = None

    # Link to meta-claim
    meta_claim_id: Optional[str] = None


# =============================================================================
# INQUIRY (main entity)
# =============================================================================

@dataclass
class Inquiry:
    """
    User-scoped question with typed target and live evidence state.

    This is the main entity for the MVP.
    """
    id: str = field(default_factory=lambda: f"inq_{uuid.uuid4().hex[:8]}")

    # Question
    title: str = ""  # "How many people died in the Wang Fuk Court fire?"
    description: str = ""  # Additional context

    # Scope (prevents cross-incident contamination)
    scope_entities: Set[str] = field(default_factory=set)  # Required entities
    scope_time_start: Optional[datetime] = None
    scope_time_end: Optional[datetime] = None
    scope_keywords: List[str] = field(default_factory=list)

    # Schema (typed target)
    schema: InquirySchema = field(default_factory=InquirySchema)

    # Stakes
    total_stake: float = 0.0
    stakes: Dict[str, float] = field(default_factory=dict)  # user_id -> amount

    # Lifecycle
    status: InquiryStatus = InquiryStatus.OPEN
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    # REEE integration
    surface_ids: Set[str] = field(default_factory=set)  # L2 surfaces in scope
    event_ids: Set[str] = field(default_factory=set)    # L3 events in scope

    # Current belief state (summary, not full posterior)
    posterior_map: Optional[Any] = None        # MAP value
    posterior_probability: float = 0.0         # P(MAP)
    entropy_bits: float = 0.0
    normalized_entropy: float = 0.0
    credible_interval: tuple = (None, None)    # 95% CI

    # Contributions and tasks
    contribution_count: int = 0
    open_tasks: int = 0

    # Resolution criteria tracking
    stable_since: Optional[datetime] = None  # When P(MAP) >= 0.95 started
    blocking_tasks: List[str] = field(default_factory=list)

    def is_resolvable(self) -> bool:
        """Check if inquiry meets resolution criteria."""
        if self.schema.rigor == RigorLevel.C:
            return False  # Exploratory never shows "resolved"

        if self.posterior_probability < 0.95:
            return False

        if self.blocking_tasks:
            return False

        if self.stable_since is None:
            return False

        # Need 24 hours of stability
        hours_stable = (datetime.utcnow() - self.stable_since).total_seconds() / 3600
        return hours_stable >= 24

    def summary(self) -> Dict:
        """Summary for API response."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "rigor": self.schema.rigor.value,
            "schema_type": self.schema.schema_type,
            "posterior": {
                "map": self.posterior_map,
                "probability": round(self.posterior_probability, 3),
                "entropy_bits": round(self.entropy_bits, 2),
                "normalized_entropy": round(self.normalized_entropy, 3),
                "credible_interval": self.credible_interval,
            },
            "stake": self.total_stake,
            "contributions": self.contribution_count,
            "open_tasks": self.open_tasks,
            "created_at": self.created_at.isoformat(),
            "resolvable": self.is_resolvable(),
        }
