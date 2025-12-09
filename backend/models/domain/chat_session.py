"""
Chat session domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from utils.id_generator import generate_chat_session_id, validate_id


class ChatSessionStatus(str, Enum):
    """Chat session status"""
    ACTIVE = "active"
    EXHAUSTED = "exhausted"  # Message limit reached


@dataclass
class ChatSession:
    """
    Chat session domain model - storage-agnostic representation

    Represents a premium AI chat session for an event.
    Users pay credits to unlock, get limited messages.

    Storage: PostgreSQL (chat_sessions table)

    ID format: cs_xxxxxxxx (11 chars)
    """
    id: str  # Short ID: cs_xxxxxxxx
    event_id: str  # ev_xxxxxxxx
    user_id: str  # UUID format

    # Message tracking
    message_count: int = 0
    max_messages: int = 100

    # Credits
    cost: int = 10  # Credits spent to unlock

    # Status
    status: ChatSessionStatus = ChatSessionStatus.ACTIVE

    # Timestamps
    unlocked_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and generate ID if needed"""
        if not self.id or not validate_id(self.id):
            self.id = generate_chat_session_id()

        if not self.unlocked_at:
            self.unlocked_at = datetime.utcnow()

        if not self.created_at:
            self.created_at = datetime.utcnow()

    @property
    def remaining_messages(self) -> int:
        """Calculate remaining messages"""
        return max(0, self.max_messages - self.message_count)

    @property
    def is_exhausted(self) -> bool:
        """Check if message limit reached"""
        return self.message_count >= self.max_messages

    @property
    def can_send_message(self) -> bool:
        """Check if user can send another message"""
        return self.status == ChatSessionStatus.ACTIVE and not self.is_exhausted

    def increment_message_count(self) -> bool:
        """
        Increment message count and update status if needed.

        Returns:
            True if successful, False if exhausted
        """
        if self.is_exhausted:
            return False

        self.message_count += 1
        self.last_message_at = datetime.utcnow()

        # Update status if limit reached
        if self.is_exhausted:
            self.status = ChatSessionStatus.EXHAUSTED

        return True
