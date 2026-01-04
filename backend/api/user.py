"""
User API Endpoints
==================

User profile, credits, and transaction history.

Endpoints:
- GET /api/user/credits - Get credit balance
- GET /api/user/transactions - Get transaction history
- GET /api/user/profile - Get user profile with stats
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# Lazy imports
_imports_ready = False
_user_repo_cls = None
_inquiry_repo_cls = None
_get_db_pool = None
_get_current_user_optional = None


def _ensure_imports():
    global _imports_ready, _user_repo_cls, _inquiry_repo_cls, _get_db_pool
    global _get_current_user_optional
    if _imports_ready:
        return True
    try:
        from repositories.user_repository import UserRepository
        from repositories.inquiry_repository import InquiryRepository
        from repositories import get_db_pool
        from middleware.auth import get_current_user_optional
        _user_repo_cls = UserRepository
        _inquiry_repo_cls = InquiryRepository
        _get_db_pool = get_db_pool
        _get_current_user_optional = get_current_user_optional
        _imports_ready = True
        return True
    except ImportError as e:
        logger.warning(f"Could not import user dependencies: {e}")
        return False


async def get_current_user(request: Request):
    """Get current user from auth middleware."""
    if not _ensure_imports():
        return None
    return await _get_current_user_optional(request)


@router.get("/user/credits")
async def get_credits(request: Request):
    """
    Get current user's credit balance.

    Returns:
        credits_balance: Current balance
        can_stake: Whether user can stake (balance > 0)
    """
    if not _ensure_imports():
        raise HTTPException(status_code=503, detail="Service not available")

    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    pool = await _get_db_pool()
    user_repo = _user_repo_cls(pool)

    user = await user_repo.get_by_id(str(current_user.user_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "credits_balance": user.credits_balance,
        "can_stake": user.credits_balance > 0,
        "reputation": user.reputation
    }


@router.get("/user/transactions")
async def get_transactions(request: Request, limit: int = 50):
    """
    Get user's credit transaction history.

    Returns list of transactions with:
    - amount (positive=credit, negative=debit)
    - balance_after
    - transaction_type (stake, reward, purchase, etc.)
    - reference info
    - created_at
    """
    if not _ensure_imports():
        raise HTTPException(status_code=503, detail="Service not available")

    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    pool = await _get_db_pool()
    inquiry_repo = _inquiry_repo_cls(pool)

    transactions = await inquiry_repo.get_user_transactions(
        user_id=str(current_user.user_id),
        limit=limit
    )

    return {
        "transactions": transactions,
        "total": len(transactions)
    }


@router.get("/user/profile")
async def get_profile(request: Request):
    """
    Get user profile with contribution stats.

    Returns:
    - User info (name, email, picture)
    - Credits and reputation
    - Contribution statistics
    """
    if not _ensure_imports():
        raise HTTPException(status_code=503, detail="Service not available")

    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    pool = await _get_db_pool()
    user_repo = _user_repo_cls(pool)

    user = await user_repo.get_by_id(str(current_user.user_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get contribution stats (simplified - can expand later)
    # TODO: Add actual stats queries

    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "picture_url": user.picture_url,
        "credits_balance": user.credits_balance,
        "reputation": user.reputation,
        "subscription_tier": user.subscription_tier,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "stats": {
            "total_contributions": 0,  # TODO: Query from contributions table
            "total_stakes": 0,  # TODO: Query from stakes table
            "total_rewards": 0,  # TODO: Query from transactions
        }
    }


@router.get("/user/stakes")
async def get_user_stakes(request: Request):
    """
    Get all inquiries the user has staked on.
    """
    if not _ensure_imports():
        raise HTTPException(status_code=503, detail="Service not available")

    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    pool = await _get_db_pool()

    # Query user's stakes with inquiry info
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT s.inquiry_id, s.amount, s.created_at,
                   i.title, i.status, i.posterior_prob, i.total_stake
            FROM inquiry_stakes s
            JOIN inquiries i ON s.inquiry_id = i.id
            WHERE s.user_id = $1
            ORDER BY s.created_at DESC
        """, current_user.user_id)

    stakes = [
        {
            "inquiry_id": row["inquiry_id"],
            "inquiry_title": row["title"],
            "inquiry_status": row["status"],
            "stake_amount": float(row["amount"]),
            "total_stake": float(row["total_stake"]),
            "posterior_prob": float(row["posterior_prob"] or 0),
            "staked_at": row["created_at"].isoformat()
        }
        for row in rows
    ]

    return {
        "stakes": stakes,
        "total_staked": sum(s["stake_amount"] for s in stakes)
    }


@router.get("/user/contributions")
async def get_user_contributions(request: Request, limit: int = 50):
    """
    Get user's contributions across all inquiries.
    """
    if not _ensure_imports():
        raise HTTPException(status_code=503, detail="Service not available")

    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Login required")

    pool = await _get_db_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT c.id, c.inquiry_id, c.contribution_type, c.text,
                   c.posterior_impact, c.reward_earned, c.created_at,
                   i.title as inquiry_title
            FROM contributions c
            JOIN inquiries i ON c.inquiry_id = i.id
            WHERE c.user_id = $1
            ORDER BY c.created_at DESC
            LIMIT $2
        """, current_user.user_id, limit)

    contributions = [
        {
            "id": row["id"],
            "inquiry_id": row["inquiry_id"],
            "inquiry_title": row["inquiry_title"],
            "type": row["contribution_type"],
            "text": row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"],
            "impact": float(row["posterior_impact"] or 0),
            "reward_earned": float(row["reward_earned"] or 0),
            "created_at": row["created_at"].isoformat()
        }
        for row in rows
    ]

    return {
        "contributions": contributions,
        "total": len(contributions),
        "total_impact": sum(c["impact"] for c in contributions),
        "total_rewards": sum(c["reward_earned"] for c in contributions)
    }
