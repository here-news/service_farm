"""
User domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class User:
    """
    User domain model - storage-agnostic representation

    Storage: PostgreSQL (users table)

    Note: Users keep UUID format (not short IDs like events/pages)
    This maintains compatibility with existing auth systems and standards.
    """
    user_id: str  # UUID format (not short ID)
    email: str
    google_id: str

    # Profile
    name: Optional[str] = None
    picture_url: Optional[str] = None

    # Credits and reputation system
    credits_balance: int = 1000
    reputation: int = 0

    # Subscription/status
    subscription_tier: Optional[str] = None  # 'free', 'premium', etc.
    is_active: bool = True

    # Timestamps
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Generate UUID if not provided"""
        if not self.user_id:
            self.user_id = str(uuid.uuid4())

    @property
    def has_credits(self) -> bool:
        """Check if user has any credits"""
        return self.credits_balance > 0

    @property
    def can_unlock_chat(self) -> bool:
        """Check if user has enough credits to unlock a chat session"""
        CHAT_UNLOCK_COST = 10
        return self.credits_balance >= CHAT_UNLOCK_COST

    def deduct_credits(self, amount: int) -> bool:
        """
        Attempt to deduct credits from balance.

        Args:
            amount: Number of credits to deduct

        Returns:
            True if successful, False if insufficient credits
        """
        if self.credits_balance >= amount:
            self.credits_balance -= amount
            return True
        return False

    def add_reputation(self, points: int):
        """Add reputation points"""
        self.reputation = max(0, self.reputation + points)
