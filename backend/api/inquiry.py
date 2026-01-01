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
- POST /api/inquiry/{id}/tasks/{task_id}/claim - Claim task
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import logging
import random

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC MODELS (inline to avoid import issues)
# =============================================================================

class InquirySchemaInput(BaseModel):
    schema_type: str = "boolean"
    categories: List[str] = []
    count_scale: str = "medium"
    count_max: int = 500
    rigor: str = "B"


class CreateInquiryInput(BaseModel):
    title: str = Field(..., min_length=10, max_length=500)
    description: str = ""
    inquiry_schema: InquirySchemaInput = InquirySchemaInput()
    scope_entities: List[str] = []
    scope_keywords: List[str] = []
    initial_stake: float = 0.0


class ContributionInput(BaseModel):
    type: str = "evidence"
    text: str = Field(..., min_length=10)
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    extracted_value: Optional[Any] = None
    observation_kind: str = "point"


class StakeInput(BaseModel):
    amount: float = Field(..., gt=0)


# =============================================================================
# SIMULATED DATA FOR DEMO
# =============================================================================

SIMULATED_INQUIRIES = [
    {
        "id": "sim_resolved_1",
        "title": "Did Elon Musk acquire Twitter in 2022?",
        "status": "resolved",
        "rigor": "A",
        "schema_type": "boolean",
        "posterior_map": "true",
        "posterior_probability": 0.99,
        "entropy_bits": 0.08,
        "stake": 250.00,
        "contributions": 12,
        "open_tasks": 0,
        "resolvable": True,
        "scope_entities": ["Elon Musk", "Twitter"],
    },
    {
        "id": "sim_bounty_1",
        "title": "How many Russian soldiers have died in Ukraine as of Dec 2024?",
        "status": "open",
        "rigor": "B",
        "schema_type": "monotone_count",
        "posterior_map": 315000,
        "posterior_probability": 0.32,
        "entropy_bits": 4.8,
        "stake": 5000.00,
        "contributions": 24,
        "open_tasks": 4,
        "resolvable": False,
        "scope_entities": ["Russian Armed Forces", "Ukraine"],
    },
    {
        "id": "sim_bounty_2",
        "title": "Will GPT-5 be released before July 2025?",
        "status": "open",
        "rigor": "B",
        "schema_type": "forecast",
        "posterior_map": "true",
        "posterior_probability": 0.62,
        "entropy_bits": 0.96,
        "stake": 2500.00,
        "contributions": 15,
        "open_tasks": 2,
        "resolvable": False,
        "scope_entities": ["OpenAI", "GPT-5"],
    },
    {
        "id": "sim_contested_1",
        "title": "How many people were killed in the Gaza hospital explosion?",
        "status": "open",
        "rigor": "B",
        "schema_type": "monotone_count",
        "posterior_map": 300,
        "posterior_probability": 0.18,
        "entropy_bits": 5.2,
        "stake": 800.00,
        "contributions": 31,
        "open_tasks": 5,
        "resolvable": False,
        "scope_entities": ["Al-Ahli Hospital", "Gaza"],
    },
]


def generate_simulated_trace(inquiry):
    """Generate simulated trace for demo."""
    return {
        "inquiry": {"id": inquiry["id"], "title": inquiry["title"]},
        "belief_state": {
            "map": inquiry.get("posterior_map"),
            "map_probability": inquiry.get("posterior_probability", 0),
            "entropy_bits": inquiry.get("entropy_bits", 0),
            "normalized_entropy": min(1, inquiry.get("entropy_bits", 0) / 5),
            "observation_count": inquiry.get("contributions", 0),
            "total_log_score": -inquiry.get("entropy_bits", 0) * 2
        },
        "contributions": [],
        "tasks": [],
        "stakes": [],
        "resolution": {
            "status": inquiry.get("status"),
            "resolvable": inquiry.get("resolvable", False),
            "blocking_tasks": []
        }
    }


# =============================================================================
# LAZY IMPORTS FOR OPTIONAL DEPENDENCIES
# =============================================================================

_imports_ready = False
_inquiry_repo_cls = None
_user_repo_cls = None
_get_db_pool = None
_get_current_user_optional = None


def _ensure_imports():
    global _imports_ready, _inquiry_repo_cls, _user_repo_cls, _get_db_pool
    global _get_current_user_optional
    if _imports_ready:
        return True
    try:
        from repositories.inquiry_repository import InquiryRepository
        from repositories.user_repository import UserRepository
        from repositories import get_db_pool
        from middleware.auth import get_current_user_optional
        _inquiry_repo_cls = InquiryRepository
        _user_repo_cls = UserRepository
        _get_db_pool = get_db_pool
        _get_current_user_optional = get_current_user_optional
        _imports_ready = True
        return True
    except ImportError as e:
        logger.warning(f"Could not import inquiry dependencies: {e}")
        return False


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async def get_inquiry_repo():
    """Get inquiry repository instance."""
    if not _ensure_imports():
        return None
    pool = await _get_db_pool()
    return _inquiry_repo_cls(pool)


async def get_user_repo():
    """Get user repository instance."""
    if not _ensure_imports():
        return None
    pool = await _get_db_pool()
    return _user_repo_cls(pool)


async def get_current_user(request: Request):
    """Get current user from auth middleware."""
    if not _ensure_imports():
        return None
    return await _get_current_user_optional(request)


# =============================================================================
# LIST INQUIRIES
# =============================================================================

@router.get("/inquiry")
async def list_inquiries(
    status: Optional[str] = None,
    order_by: str = "stake",
    limit: int = 50,
):
    """List inquiries with optional filtering."""
    repo = await get_inquiry_repo()

    if repo:
        try:
            inquiries = await repo.list_inquiries(
                status=status,
                order_by=order_by,
                limit=limit
            )
            if inquiries:
                return [
                    {
                        "id": inq["id"],
                        "title": inq["title"],
                        "status": inq["status"],
                        "rigor": inq["rigor_level"],
                        "schema_type": inq["schema_type"],
                        "posterior_map": inq.get("posterior_map"),
                        "posterior_probability": inq.get("posterior_prob", 0),
                        "entropy_bits": inq.get("entropy_bits", 0),
                        "stake": inq.get("total_stake", 0),
                        "contributions": inq.get("contribution_count", 0),
                        "open_tasks": inq.get("open_tasks_count", 0),
                        "resolvable": inq.get("posterior_prob", 0) >= 0.95,
                        "scope_entities": inq.get("scope_entities", [])
                    }
                    for inq in inquiries
                ]
        except Exception as e:
            logger.warning(f"Database error listing inquiries: {e}")

    # Fallback to simulated data
    return SIMULATED_INQUIRIES


# =============================================================================
# GET INQUIRY DETAIL
# =============================================================================

@router.get("/inquiry/{inquiry_id}")
async def get_inquiry(inquiry_id: str):
    """Get inquiry detail."""
    # Check if simulated
    if inquiry_id.startswith("sim_"):
        sim = next((i for i in SIMULATED_INQUIRIES if i["id"] == inquiry_id), None)
        if sim:
            return sim
        raise HTTPException(status_code=404, detail="Inquiry not found")

    repo = await get_inquiry_repo()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    inquiry = await repo.get_inquiry(inquiry_id)
    if not inquiry:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    return {
        "id": inquiry["id"],
        "title": inquiry["title"],
        "description": inquiry["description"],
        "status": inquiry["status"],
        "rigor": inquiry["rigor_level"],
        "schema_type": inquiry["schema_type"],
        "scope_entities": inquiry.get("scope_entities", []),
        "scope_keywords": inquiry.get("scope_keywords", []),
        "posterior": {
            "map": inquiry.get("posterior_map"),
            "probability": round(inquiry.get("posterior_prob", 0), 3),
            "entropy_bits": round(inquiry.get("entropy_bits", 0), 2),
            "normalized_entropy": round(inquiry.get("normalized_entropy", 0), 3),
            "credible_interval": inquiry.get("credible_interval"),
        },
        "stake": inquiry.get("total_stake", 0),
        "contributions": inquiry.get("contribution_count", 0),
        "open_tasks": inquiry.get("open_tasks_count", 0),
        "created_at": inquiry.get("created_at"),
        "updated_at": inquiry.get("updated_at"),
        "resolvable": inquiry.get("posterior_prob", 0) >= 0.95,
    }


# =============================================================================
# CREATE INQUIRY
# =============================================================================

@router.post("/inquiry")
async def create_inquiry(input: CreateInquiryInput, request: Request):
    """Create a new inquiry."""
    current_user = await get_current_user(request)
    user_id = str(current_user.user_id) if current_user else None

    repo = await get_inquiry_repo()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    inquiry = await repo.create_inquiry(
        title=input.title,
        description=input.description,
        schema_type=input.inquiry_schema.schema_type,
        schema_config=input.inquiry_schema.model_dump(),
        rigor_level=input.inquiry_schema.rigor,
        scope_entities=input.scope_entities,
        scope_keywords=input.scope_keywords,
        created_by=user_id
    )

    # Add initial stake if provided
    if input.initial_stake > 0 and user_id:
        user_repo = await get_user_repo()
        if user_repo:
            user = await user_repo.get_by_id(user_id)
            if user and user.credits_balance >= input.initial_stake:
                success = await user_repo.deduct_credits(user_id, int(input.initial_stake))
                if success:
                    await repo.add_stake(inquiry["id"], user_id, input.initial_stake)

    logger.info(f"Created inquiry {inquiry['id']}: {input.title[:50]}")

    return {
        "id": inquiry["id"],
        "title": inquiry["title"],
        "status": inquiry["status"],
        "rigor": inquiry["rigor_level"],
        "schema_type": inquiry["schema_type"],
        "posterior_probability": 0,
        "entropy_bits": 0,
        "stake": inquiry.get("total_stake", 0),
        "contributions": 0,
        "open_tasks": 0,
        "resolvable": False,
        "scope_entities": inquiry.get("scope_entities", [])
    }


# =============================================================================
# ADD CONTRIBUTION
# =============================================================================

@router.post("/inquiry/{inquiry_id}/contribute")
async def add_contribution(inquiry_id: str, input: ContributionInput, request: Request):
    """Add a contribution to an inquiry."""
    current_user = await get_current_user(request)
    user_id = str(current_user.user_id) if current_user else None

    # Handle simulated inquiries - demo mode
    if inquiry_id.startswith("sim_"):
        return {
            "contribution": {
                "id": f"contrib_demo_{inquiry_id[-4:]}",
                "type": input.type,
                "processed": True,
                "impact": 0.05,
                "posterior_impact": 0.05,
            },
            "updated_posterior": {
                "map": None,
                "probability": 0.35,
                "entropy_bits": 4.5,
            },
            "demo_mode": True,
            "message": "Demo mode - contribution not saved"
        }

    repo = await get_inquiry_repo()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    inquiry = await repo.get_inquiry(inquiry_id)
    if not inquiry:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    contribution = await repo.add_contribution(
        inquiry_id=inquiry_id,
        user_id=user_id,
        contribution_type=input.type,
        text=input.text,
        source_url=input.source_url,
        source_name=input.source_name,
        extracted_value=input.extracted_value,
        observation_kind=input.observation_kind
    )

    # Mock impact for now
    mock_impact = random.uniform(0.01, 0.1)
    await repo.update_contribution_impact(
        contribution_id=contribution["id"],
        posterior_impact=mock_impact,
        processed=True
    )

    inquiry = await repo.get_inquiry(inquiry_id)

    return {
        "contribution": {
            "id": contribution["id"],
            "type": contribution["type"],
            "processed": True,
            "impact": round(mock_impact, 3),
            "posterior_impact": round(mock_impact, 3),
        },
        "updated_posterior": {
            "map": inquiry.get("posterior_map"),
            "probability": round(inquiry.get("posterior_prob", 0), 3),
            "entropy_bits": round(inquiry.get("entropy_bits", 0), 2),
        }
    }


# =============================================================================
# ADD STAKE
# =============================================================================

@router.post("/inquiry/{inquiry_id}/stake")
async def add_stake(inquiry_id: str, input: StakeInput, request: Request):
    """Add stake to an inquiry."""
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required to stake")

    user_id = str(current_user.user_id)

    # Handle simulated inquiries
    if inquiry_id.startswith("sim_"):
        return {
            "total_stake": input.amount + 5000,
            "user_balance": 950,
            "demo_mode": True,
            "message": "Demo mode - stake not saved"
        }

    repo = await get_inquiry_repo()
    user_repo = await get_user_repo()
    if not repo or not user_repo:
        raise HTTPException(status_code=503, detail="Database not available")

    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.credits_balance < input.amount:
        raise HTTPException(status_code=400, detail="Insufficient credits")

    inquiry = await repo.get_inquiry(inquiry_id)
    if not inquiry:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    success = await user_repo.deduct_credits(user_id, int(input.amount))
    if not success:
        raise HTTPException(status_code=400, detail="Insufficient credits")

    await repo.add_stake(inquiry_id, user_id, input.amount)

    new_balance = user.credits_balance - input.amount
    await repo.record_transaction(
        user_id=user_id,
        amount=-input.amount,
        balance_after=new_balance,
        transaction_type="stake",
        reference_type="inquiry",
        reference_id=inquiry_id,
        description=f"Stake on: {inquiry['title'][:50]}"
    )

    total_stake = await repo.get_total_stake(inquiry_id)

    return {
        "total_stake": total_stake,
        "user_balance": new_balance
    }


# =============================================================================
# GET TRACE
# =============================================================================

@router.get("/inquiry/{inquiry_id}/trace")
async def get_trace(inquiry_id: str):
    """Get full epistemic trace for an inquiry."""
    if inquiry_id.startswith("sim_"):
        sim = next((i for i in SIMULATED_INQUIRIES if i["id"] == inquiry_id), None)
        if sim:
            return generate_simulated_trace(sim)
        raise HTTPException(status_code=404, detail="Inquiry not found")

    repo = await get_inquiry_repo()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    trace = await repo.get_trace(inquiry_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    return trace


# =============================================================================
# GET TASKS
# =============================================================================

@router.get("/inquiry/{inquiry_id}/tasks")
async def get_tasks(inquiry_id: str, include_completed: bool = False):
    """Get tasks for an inquiry."""
    if inquiry_id.startswith("sim_"):
        return []

    repo = await get_inquiry_repo()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    tasks = await repo.get_tasks(inquiry_id, include_completed=include_completed)
    return tasks


# =============================================================================
# CLAIM TASK
# =============================================================================

@router.post("/inquiry/{inquiry_id}/tasks/{task_id}/claim")
async def claim_task(inquiry_id: str, task_id: str, request: Request):
    """Claim a task."""
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    repo = await get_inquiry_repo()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = str(current_user.user_id)

    success = await repo.claim_task(task_id, user_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task already claimed or not found")

    return {"status": "claimed", "task_id": task_id}


# =============================================================================
# GET CONTRIBUTIONS
# =============================================================================

@router.get("/inquiry/{inquiry_id}/contributions")
async def get_contributions(inquiry_id: str, order_by: str = "recent", limit: int = 50):
    """Get contributions for an inquiry."""
    if inquiry_id.startswith("sim_"):
        return []

    repo = await get_inquiry_repo()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    contributions = await repo.get_contributions(inquiry_id, order_by=order_by, limit=limit)
    return contributions
