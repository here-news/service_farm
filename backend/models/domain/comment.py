"""
Comment domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from utils.id_generator import generate_comment_id, validate_id


class ReactionType(str, Enum):
    """Comment reaction types"""
    SUPPORT = "support"
    REFUTE = "refute"
    QUESTION = "question"
    COMMENT = "comment"


@dataclass
class Comment:
    """
    Comment domain model - storage-agnostic representation

    Storage: PostgreSQL (comments table)

    Comments can be attached to either:
    - Events (event_id) - for commenting on events
    - Pages (page_id) - for commenting on specific articles

    ID format: cm_xxxxxxxx (11 chars)
    """
    id: str  # Short ID: cm_xxxxxxxx
    user_id: str  # UUID format
    text: str

    # Parent entity (at least one must be set)
    event_id: Optional[str] = None  # ev_xxxxxxxx
    page_id: Optional[str] = None   # pg_xxxxxxxx

    # Threading support
    parent_comment_id: Optional[str] = None  # cm_xxxxxxxx

    # Reaction type (optional classification)
    reaction_type: Optional[ReactionType] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and generate ID if needed"""
        if not self.id or not validate_id(self.id):
            self.id = generate_comment_id()

        # Validate that at least one parent is set
        if not self.event_id and not self.page_id:
            raise ValueError("Comment must have either event_id or page_id")

    @property
    def is_reply(self) -> bool:
        """Check if this is a reply to another comment"""
        return self.parent_comment_id is not None

    @property
    def is_root_comment(self) -> bool:
        """Check if this is a top-level comment"""
        return self.parent_comment_id is None
