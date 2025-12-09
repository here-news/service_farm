"""
Pydantic models for Chat Sessions
"""

from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
from app.database.models import ChatSessionStatus


class ChatSessionResponse(BaseModel):
    """Chat session response model"""
    id: str
    story_id: str
    user_id: str
    message_count: int
    cost: int
    status: ChatSessionStatus
    unlocked_at: datetime
    last_message_at: Optional[datetime] = None
    created_at: datetime

    @field_validator('user_id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        """Convert UUID to string"""
        if isinstance(v, UUID):
            return str(v)
        return v

    model_config = {"from_attributes": True}


class UnlockChatRequest(BaseModel):
    """Request to unlock chat for a story"""
    story_id: str


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str  # 'user' or 'assistant'
    content: str


class SendMessageRequest(BaseModel):
    """Request to send a chat message"""
    story_id: str
    message: str
    conversation_history: List[ChatMessage] = []


class SendMessageResponse(BaseModel):
    """Response after sending a message"""
    message: str
    message_count: int
    remaining_messages: int
    session_status: ChatSessionStatus
