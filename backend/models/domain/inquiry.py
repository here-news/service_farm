"""
Inquiry domain models

Storage: PostgreSQL (inquiries, contributions, inquiry_stakes, inquiry_tasks tables)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Set, Dict, Any
from enum import Enum
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class InquiryStatus(Enum):
    """Lifecycle status of an inquiry."""
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"
    CLOSED = "closed"


class RigorLevel(Enum):
    """
    Rigor badge for inquiry.

    A: Typed + calibrated noise model (highest rigor)
    B: Typed + generic priors (medium rigor)
    C: Exploratory - no resolution badge shown
    """
    A = "A"
    B = "B"
    C = "C"


class SchemaType(Enum):
    """Type of question schema."""
    MONOTONE_COUNT = "monotone_count"  # death_count, missing_count
    CATEGORICAL = "categorical"         # legal_status, verdict
    BOOLEAN = "boolean"                  # yes/no
    REPORT_TRUTH = "report_truth"        # "did Reuters report p?"
    QUOTE_AUTHENTICITY = "quote_authenticity"  # "did X say Y?"
    FORECAST = "forecast"                # prediction
    CUSTOM = "custom"


class ContributionType(Enum):
    """Type of user contribution."""
    EVIDENCE = "evidence"
    REFUTATION = "refutation"
    ATTRIBUTION = "attribution"
    SCOPE_CORRECTION = "scope_correction"
    DISAMBIGUATION = "disambiguation"


class TaskType(Enum):
    """System-generated task types (from meta-claims)."""
    NEED_PRIMARY_SOURCE = "need_primary_source"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    SINGLE_SOURCE_ONLY = "single_source_only"
    HIGH_ENTROPY = "high_entropy"
    STALE = "stale"


class TransactionType(Enum):
    """Types of credit transactions."""
    STAKE = "stake"
    REWARD = "reward"
    PURCHASE = "purchase"
    REFUND = "refund"
    SIGNUP_BONUS = "signup_bonus"
    ADMIN_ADJUSTMENT = "admin_adjustment"


# =============================================================================
# INQUIRY SCHEMA
# =============================================================================

@dataclass
class InquirySchema:
    """
    Defines the typed target for an inquiry.
    Maps to REEE's typed_belief domains.
    """
    schema_type: str = "boolean"  # Use string for flexibility

    # For categorical: allowed values
    categories: List[str] = field(default_factory=list)

    # For count: expected scale
    count_scale: str = "medium"  # small, medium, large
    count_max: int = 500
    count_monotone: bool = True

    # For custom: hypothesis set
    hypotheses: List[str] = field(default_factory=list)

    # Rigor level (determines if "resolved" badge shown)
    rigor: RigorLevel = field(default=RigorLevel.B)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return {
            "schema_type": self.schema_type,
            "categories": self.categories,
            "count_scale": self.count_scale,
            "count_max": self.count_max,
            "count_monotone": self.count_monotone,
            "hypotheses": self.hypotheses,
        }

    @classmethod
    def from_dict(cls, data: Dict, rigor: str = "B") -> "InquirySchema":
        """Create from dictionary."""
        return cls(
            schema_type=data.get("schema_type", "boolean"),
            categories=data.get("categories", []),
            count_scale=data.get("count_scale", "medium"),
            count_max=data.get("count_max", 500),
            count_monotone=data.get("count_monotone", True),
            hypotheses=data.get("hypotheses", []),
            rigor=RigorLevel(rigor)
        )


# =============================================================================
# INQUIRY
# =============================================================================

@dataclass
class Inquiry:
    """
    User-scoped question with typed target and live evidence state.
    This is the main entity for the Inquiry MVP.

    Storage: PostgreSQL (inquiries table)
    """
    id: str = ""
    title: str = ""
    description: str = ""

    # Status and rigor
    status: InquiryStatus = InquiryStatus.OPEN
    rigor_level: str = "B"
    schema_type: str = "boolean"
    schema_config: Dict = field(default_factory=dict)

    # Scope constraints
    scope_entities: List[str] = field(default_factory=list)
    scope_keywords: List[str] = field(default_factory=list)
    scope_time_start: Optional[datetime] = None
    scope_time_end: Optional[datetime] = None

    # Current belief state
    posterior_map: Any = None
    posterior_prob: float = 0.0
    entropy_bits: float = 0.0
    normalized_entropy: float = 0.0
    credible_interval: Optional[tuple] = None

    # Stakes
    total_stake: float = 0.0
    distributed: float = 0.0

    # Counts
    contribution_count: int = 0
    open_tasks_count: int = 0

    # Metadata
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    stable_since: Optional[datetime] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"inq_{uuid.uuid4().hex[:8]}"
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

    @property
    def schema(self) -> InquirySchema:
        """Get the schema object."""
        return InquirySchema.from_dict(self.schema_config, self.rigor_level)

    def is_resolvable(self) -> bool:
        """Check if inquiry meets resolution criteria."""
        if RigorLevel(self.rigor_level) == RigorLevel.C:
            return False  # Exploratory never shows "resolved"

        if self.posterior_prob < 0.95:
            return False

        if self.open_tasks_count > 0:
            return False

        if self.stable_since is None:
            return False

        # Need 24 hours of stability
        hours_stable = (datetime.utcnow() - self.stable_since).total_seconds() / 3600
        return hours_stable >= 24

    def to_summary(self) -> Dict:
        """Summary for API response (list view)."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value if isinstance(self.status, InquiryStatus) else self.status,
            "rigor": self.rigor_level,
            "schema_type": self.schema_type,
            "posterior_map": self.posterior_map,
            "posterior_probability": round(self.posterior_prob, 3),
            "entropy_bits": round(self.entropy_bits, 2),
            "stake": self.total_stake,
            "contributions": self.contribution_count,
            "open_tasks": self.open_tasks_count,
            "resolvable": self.is_resolvable(),
            "scope_entities": self.scope_entities,
        }

    def to_detail(self) -> Dict:
        """Full detail for API response."""
        return {
            **self.to_summary(),
            "description": self.description,
            "normalized_entropy": round(self.normalized_entropy, 3),
            "credible_interval": self.credible_interval,
            "scope_keywords": self.scope_keywords,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


# =============================================================================
# CONTRIBUTION
# =============================================================================

@dataclass
class Contribution:
    """
    User submission to an inquiry.
    Becomes L0 claims after processing.

    Storage: PostgreSQL (contributions table)
    """
    id: str = ""
    inquiry_id: str = ""
    user_id: Optional[str] = None
    user_name: str = "Anonymous"

    # Type and content
    type: str = "evidence"  # ContributionType value
    text: str = ""
    source_url: Optional[str] = None
    source_name: Optional[str] = None

    # Extracted value (for typed inquiries)
    extracted_value: Any = None
    observation_kind: Optional[str] = None

    # Processing state
    processed: bool = False
    claim_ids: List[str] = field(default_factory=list)

    # Impact and rewards
    posterior_impact: float = 0.0
    reward_earned: float = 0.0

    # Metadata
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"contrib_{uuid.uuid4().hex[:8]}"
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to API response format."""
        return {
            "id": self.id,
            "inquiry_id": self.inquiry_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "type": self.type,
            "text": self.text,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "extracted_value": self.extracted_value,
            "observation_kind": self.observation_kind,
            "processed": self.processed,
            "impact": round(self.posterior_impact, 3),
            "posterior_impact": round(self.posterior_impact, 3),
            "reward_earned": self.reward_earned,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# INQUIRY TASK
# =============================================================================

@dataclass
class InquiryTask:
    """
    Actionable task for an inquiry.
    Generated from REEE meta-claims.

    Storage: PostgreSQL (inquiry_tasks table)
    """
    id: str = ""
    inquiry_id: str = ""

    type: str = "single_source_only"  # TaskType value
    description: str = ""
    bounty: float = 0.0

    # Status
    claimed_by: Optional[str] = None
    claimed_by_name: Optional[str] = None
    claimed_at: Optional[datetime] = None
    completed: bool = False
    completed_at: Optional[datetime] = None

    # Link to meta-claim
    meta_claim_id: Optional[str] = None

    # Metadata
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"task_{uuid.uuid4().hex[:8]}"
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to API response format."""
        return {
            "id": self.id,
            "inquiry_id": self.inquiry_id,
            "type": self.type,
            "description": self.description,
            "bounty": self.bounty,
            "claimed_by": self.claimed_by,
            "claimed_by_name": self.claimed_by_name,
            "completed": self.completed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# STAKE
# =============================================================================

@dataclass
class Stake:
    """
    User stake/bounty on an inquiry.

    Storage: PostgreSQL (inquiry_stakes table)
    """
    id: int = 0
    inquiry_id: str = ""
    user_id: str = ""
    user_name: str = "Anonymous"
    amount: float = 0.0
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "amount": self.amount,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# CREDIT TRANSACTION
# =============================================================================

@dataclass
class CreditTransaction:
    """
    Record of credit movement for a user.

    Storage: PostgreSQL (credit_transactions table)
    """
    id: int = 0
    user_id: str = ""
    amount: float = 0.0  # positive=credit, negative=debit
    balance_after: float = 0.0

    transaction_type: str = "stake"  # TransactionType value
    reference_type: Optional[str] = None  # inquiry, contribution, task
    reference_id: Optional[str] = None

    description: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "amount": self.amount,
            "balance_after": self.balance_after,
            "transaction_type": self.transaction_type,
            "reference_type": self.reference_type,
            "reference_id": self.reference_id,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
