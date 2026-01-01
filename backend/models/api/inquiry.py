"""
Inquiry API Models (Pydantic schemas)

Request/Response models for the Inquiry MVP API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS FOR VALIDATION
# =============================================================================

class InquiryStatusEnum(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"
    CLOSED = "closed"


class RigorLevelEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class SchemaTypeEnum(str, Enum):
    MONOTONE_COUNT = "monotone_count"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    REPORT_TRUTH = "report_truth"
    QUOTE_AUTHENTICITY = "quote_authenticity"
    FORECAST = "forecast"
    CUSTOM = "custom"


class ContributionTypeEnum(str, Enum):
    EVIDENCE = "evidence"
    REFUTATION = "refutation"
    ATTRIBUTION = "attribution"
    SCOPE_CORRECTION = "scope_correction"
    DISAMBIGUATION = "disambiguation"


class TaskTypeEnum(str, Enum):
    NEED_PRIMARY_SOURCE = "need_primary_source"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    SINGLE_SOURCE_ONLY = "single_source_only"
    HIGH_ENTROPY = "high_entropy"
    STALE = "stale"


class ObservationKindEnum(str, Enum):
    POINT = "point"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    INTERVAL = "interval"
    APPROXIMATE = "approximate"
    NONE = "none"


# =============================================================================
# REQUEST MODELS
# =============================================================================

class InquirySchemaInput(BaseModel):
    """Schema definition for new inquiry."""
    schema_type: SchemaTypeEnum = SchemaTypeEnum.BOOLEAN
    categories: List[str] = []
    count_scale: str = "medium"
    count_max: int = 500
    count_monotone: bool = True
    hypotheses: List[str] = []
    rigor: RigorLevelEnum = RigorLevelEnum.B


class CreateInquiryInput(BaseModel):
    """Input for creating an inquiry."""
    title: str = Field(..., min_length=10, max_length=500, description="The question to investigate")
    description: str = Field("", max_length=2000, description="Additional context")
    schema: InquirySchemaInput = InquirySchemaInput()
    scope_entities: List[str] = Field([], description="Required entities for scope")
    scope_keywords: List[str] = Field([], description="Keywords to match")
    scope_time_start: Optional[datetime] = None
    scope_time_end: Optional[datetime] = None
    initial_stake: float = Field(0.0, ge=0, description="Initial bounty stake")


class ContributionInput(BaseModel):
    """Input for adding a contribution."""
    type: ContributionTypeEnum = ContributionTypeEnum.EVIDENCE
    text: str = Field(..., min_length=10, max_length=2000, description="Quote or description")
    source_url: Optional[str] = Field(None, max_length=2000, description="Source URL")
    source_name: Optional[str] = Field(None, max_length=255, description="Source name")
    extracted_value: Optional[Any] = Field(None, description="Extracted value for typed inquiries")
    observation_kind: ObservationKindEnum = ObservationKindEnum.POINT


class StakeInput(BaseModel):
    """Input for adding stake."""
    amount: float = Field(..., gt=0, description="Stake amount in credits")


class ClaimTaskInput(BaseModel):
    """Input for claiming a task."""
    pass  # Just need the task_id from path


class CompleteTaskInput(BaseModel):
    """Input for completing a task."""
    contribution_id: Optional[str] = Field(None, description="ID of contribution that completes task")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class InquirySummary(BaseModel):
    """Summary response for inquiry (list view)."""
    id: str
    title: str
    status: str
    rigor: str
    schema_type: str
    posterior_map: Optional[Any] = None
    posterior_probability: float
    entropy_bits: float
    stake: float
    contributions: int
    open_tasks: int
    resolvable: bool
    scope_entities: List[str] = []
    cover_image: Optional[str] = None
    resolved_ago: Optional[str] = None

    class Config:
        from_attributes = True


class BeliefStateResponse(BaseModel):
    """Belief state summary."""
    map: Optional[Any] = None
    map_probability: float
    entropy_bits: float
    normalized_entropy: float
    observation_count: int
    total_log_score: float


class InquiryDetail(BaseModel):
    """Detailed response for inquiry."""
    id: str
    title: str
    description: str
    status: str
    rigor: str
    schema_type: str
    scope_entities: List[str]
    scope_keywords: List[str]
    posterior: BeliefStateResponse
    stake: float
    contributions: int
    open_tasks: int
    created_at: str
    updated_at: str
    resolvable: bool
    credible_interval: Optional[List[Any]] = None

    class Config:
        from_attributes = True


class ContributionResponse(BaseModel):
    """Response for a contribution."""
    id: str
    inquiry_id: str
    user_id: Optional[str] = None
    user_name: str = "Anonymous"
    type: str
    text: str
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    extracted_value: Optional[Any] = None
    observation_kind: Optional[str] = None
    processed: bool
    impact: float
    posterior_impact: float
    reward_earned: float
    created_at: str

    class Config:
        from_attributes = True


class TaskResponse(BaseModel):
    """Response for a task."""
    id: str
    inquiry_id: str
    type: str
    description: str
    bounty: float
    claimed_by: Optional[str] = None
    claimed_by_name: Optional[str] = None
    completed: bool
    created_at: str

    class Config:
        from_attributes = True


class StakeResponse(BaseModel):
    """Response for a stake."""
    user_id: str
    user_name: str
    amount: float
    created_at: str


class TransactionResponse(BaseModel):
    """Response for a credit transaction."""
    id: int
    amount: float
    balance_after: float
    transaction_type: str
    reference_type: Optional[str] = None
    reference_id: Optional[str] = None
    description: Optional[str] = None
    created_at: str


class ResolutionResponse(BaseModel):
    """Resolution status response."""
    status: str
    resolvable: bool
    stable_since: Optional[str] = None
    hours_stable: Optional[float] = None
    blocking_tasks: List[str] = []


class TraceResponse(BaseModel):
    """Full epistemic trace response."""
    inquiry: InquirySummary
    belief_state: BeliefStateResponse
    contributions: List[ContributionResponse]
    tasks: List[TaskResponse]
    stakes: List[StakeResponse]
    resolution: ResolutionResponse
    posterior_top_10: Optional[List[Dict[str, Any]]] = None
    surfaces: Optional[List[Dict[str, Any]]] = None


class ContributionResultResponse(BaseModel):
    """Response after adding a contribution."""
    contribution: ContributionResponse
    updated_posterior: BeliefStateResponse


class StakeResultResponse(BaseModel):
    """Response after adding a stake."""
    total_stake: float
    user_balance: float


class UserCreditsResponse(BaseModel):
    """Response for user credits."""
    user_id: str
    credits_balance: float
    recent_transactions: List[TransactionResponse] = []
