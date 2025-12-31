"""
Inquiry API Endpoints
=====================

REST API for the Inquiry MVP.

Endpoints:
- POST /api/inquiry - Create inquiry
- GET /api/inquiry - List inquiries
- GET /api/inquiry/{id} - Get inquiry detail
- POST /api/inquiry/{id}/contribute - Add contribution
- POST /api/inquiry/{id}/stake - Add stake
- GET /api/inquiry/{id}/trace - Get epistemic trace
- GET /api/inquiry/{id}/tasks - Get tasks
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

# Import inquiry engine
from reee.inquiry import (
    InquiryEngine, Inquiry, InquirySchema, InquiryStatus, RigorLevel,
    Contribution, ContributionType, InquiryTask
)

router = APIRouter()

# Global engine instance (in production, use dependency injection)
_engine: Optional[InquiryEngine] = None


def get_engine() -> InquiryEngine:
    """Get or create the inquiry engine."""
    global _engine
    if _engine is None:
        _engine = InquiryEngine()
    return _engine


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class InquirySchemaInput(BaseModel):
    """Schema definition for new inquiry."""
    schema_type: str = "boolean"  # monotone_count, categorical, boolean, etc.
    categories: List[str] = []
    count_scale: str = "medium"
    count_max: int = 500
    count_monotone: bool = True
    rigor: str = "B"


class CreateInquiryInput(BaseModel):
    """Input for creating an inquiry."""
    title: str = Field(..., min_length=10, max_length=500)
    description: str = ""
    schema: InquirySchemaInput = InquirySchemaInput()
    scope_entities: List[str] = []
    scope_keywords: List[str] = []
    initial_stake: float = 0.0


class ContributionInput(BaseModel):
    """Input for adding a contribution."""
    type: str = "evidence"  # evidence, refutation, attribution, etc.
    text: str = Field(..., min_length=10)
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    extracted_value: Optional[Any] = None
    observation_kind: str = "point"  # point, lower_bound, interval, approximate


class StakeInput(BaseModel):
    """Input for adding stake."""
    amount: float = Field(..., gt=0)


class InquirySummary(BaseModel):
    """Summary response for inquiry."""
    id: str
    title: str
    status: str
    rigor: str
    schema_type: str
    posterior_map: Optional[Any]
    posterior_probability: float
    entropy_bits: float
    stake: float
    contributions: int
    open_tasks: int
    resolvable: bool


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
    posterior_map: Optional[Any]
    posterior_probability: float
    entropy_bits: float
    normalized_entropy: float
    credible_interval: tuple
    stake: float
    contributions: int
    open_tasks: int
    created_at: str
    updated_at: str
    resolvable: bool


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/inquiry", response_model=InquirySummary)
async def create_inquiry(
    input: CreateInquiryInput,
    user_id: str = "anonymous",  # Would come from auth in production
    engine: InquiryEngine = Depends(get_engine)
):
    """Create a new inquiry."""
    # Convert schema
    schema = InquirySchema(
        schema_type=input.schema.schema_type,
        categories=input.schema.categories,
        count_scale=input.schema.count_scale,
        count_max=input.schema.count_max,
        count_monotone=input.schema.count_monotone,
        rigor=RigorLevel(input.schema.rigor),
    )

    inquiry = engine.create_inquiry(
        title=input.title,
        description=input.description,
        schema=schema,
        created_by=user_id,
        scope_entities=input.scope_entities,
        scope_keywords=input.scope_keywords,
        initial_stake=input.initial_stake,
    )

    return _to_summary(inquiry)


@router.get("/inquiry", response_model=List[InquirySummary])
async def list_inquiries(
    status: Optional[str] = None,
    order_by: str = "stake",
    limit: int = 50,
    engine: InquiryEngine = Depends(get_engine)
):
    """List inquiries with optional filtering."""
    status_enum = InquiryStatus(status) if status else None
    inquiries = engine.list_inquiries(status=status_enum, limit=limit, order_by=order_by)
    return [_to_summary(i) for i in inquiries]


@router.get("/inquiry/{inquiry_id}")
async def get_inquiry(
    inquiry_id: str,
    engine: InquiryEngine = Depends(get_engine)
):
    """Get inquiry detail."""
    inquiry = engine.get_inquiry(inquiry_id)
    if not inquiry:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    return {
        "id": inquiry.id,
        "title": inquiry.title,
        "description": inquiry.description,
        "status": inquiry.status.value,
        "rigor": inquiry.schema.rigor.value,
        "schema_type": inquiry.schema.schema_type,
        "scope_entities": list(inquiry.scope_entities),
        "scope_keywords": inquiry.scope_keywords,
        "posterior": {
            "map": inquiry.posterior_map,
            "probability": round(inquiry.posterior_probability, 3),
            "entropy_bits": round(inquiry.entropy_bits, 2),
            "normalized_entropy": round(inquiry.normalized_entropy, 3),
            "credible_interval": inquiry.credible_interval,
        },
        "stake": inquiry.total_stake,
        "contributions": inquiry.contribution_count,
        "open_tasks": inquiry.open_tasks,
        "created_at": inquiry.created_at.isoformat(),
        "updated_at": inquiry.updated_at.isoformat(),
        "resolvable": inquiry.is_resolvable(),
    }


@router.post("/inquiry/{inquiry_id}/contribute")
async def add_contribution(
    inquiry_id: str,
    input: ContributionInput,
    user_id: str = "anonymous",
    engine: InquiryEngine = Depends(get_engine)
):
    """Add a contribution to an inquiry."""
    try:
        contribution = await engine.add_contribution(
            inquiry_id=inquiry_id,
            user_id=user_id,
            contribution_type=ContributionType(input.type),
            text=input.text,
            source_url=input.source_url,
            source_name=input.source_name,
            extracted_value=input.extracted_value,
            observation_kind=input.observation_kind,
        )

        inquiry = engine.get_inquiry(inquiry_id)

        return {
            "contribution": {
                "id": contribution.id,
                "type": contribution.type.value,
                "processed": contribution.processed,
                "impact": round(contribution.posterior_impact, 3),
            },
            "updated_posterior": {
                "map": inquiry.posterior_map,
                "probability": round(inquiry.posterior_probability, 3),
                "entropy_bits": round(inquiry.entropy_bits, 2),
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/inquiry/{inquiry_id}/stake")
async def add_stake(
    inquiry_id: str,
    input: StakeInput,
    user_id: str = "anonymous",
    engine: InquiryEngine = Depends(get_engine)
):
    """Add stake to an inquiry."""
    try:
        new_total = engine.add_stake(inquiry_id, user_id, input.amount)
        return {"total_stake": new_total}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/inquiry/{inquiry_id}/trace")
async def get_trace(
    inquiry_id: str,
    engine: InquiryEngine = Depends(get_engine)
):
    """Get full epistemic trace for an inquiry."""
    trace = engine.get_trace(inquiry_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Inquiry not found")
    return trace


@router.get("/inquiry/{inquiry_id}/tasks")
async def get_tasks(
    inquiry_id: str,
    include_completed: bool = False,
    engine: InquiryEngine = Depends(get_engine)
):
    """Get tasks for an inquiry."""
    tasks = engine.get_tasks(inquiry_id, include_completed=include_completed)
    return [
        {
            "id": t.id,
            "type": t.type.value,
            "description": t.description,
            "bounty": t.bounty,
            "completed": t.completed,
            "claimed_by": t.claimed_by,
        }
        for t in tasks
    ]


# =============================================================================
# HELPERS
# =============================================================================

def _to_summary(inquiry: Inquiry) -> InquirySummary:
    """Convert inquiry to summary response."""
    return InquirySummary(
        id=inquiry.id,
        title=inquiry.title,
        status=inquiry.status.value,
        rigor=inquiry.schema.rigor.value,
        schema_type=inquiry.schema.schema_type,
        posterior_map=inquiry.posterior_map,
        posterior_probability=round(inquiry.posterior_probability, 3),
        entropy_bits=round(inquiry.entropy_bits, 2),
        stake=inquiry.total_stake,
        contributions=inquiry.contribution_count,
        open_tasks=inquiry.open_tasks,
        resolvable=inquiry.is_resolvable(),
    )
