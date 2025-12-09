"""
Pydantic models for User
"""

from pydantic import BaseModel, EmailStr, field_serializer
from datetime import datetime
from typing import Optional
from uuid import UUID


class UserCreate(BaseModel):
    """Model for creating a user"""
    email: EmailStr
    name: str
    picture: Optional[str] = None
    google_id: str


class UserPublic(BaseModel):
    """Public user model (minimal info)"""
    user_id: UUID
    email: str
    name: str
    picture_url: Optional[str] = None

    model_config = {
        "from_attributes": True,
        "populate_by_name": True
    }

    @field_serializer('user_id')
    def serialize_user_id(self, value: UUID) -> str:
        return str(value)

    def model_dump(self, **kwargs):
        """Override to include aliases for backward compatibility"""
        data = super().model_dump(**kwargs)
        # Add aliases for frontend compatibility
        data['id'] = data.get('user_id')
        data['picture'] = data.get('picture_url')
        return data


class UserResponse(BaseModel):
    """Full user response model"""
    user_id: UUID
    email: str
    name: str
    picture_url: Optional[str] = None
    google_id: str
    credits_balance: int
    reputation: int
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = {
        "from_attributes": True,
        "populate_by_name": True
    }

    @field_serializer('user_id')
    def serialize_user_id(self, value: UUID) -> str:
        return str(value)

    def model_dump(self, **kwargs):
        """Override to include aliases for backward compatibility"""
        data = super().model_dump(**kwargs)
        # Add aliases for frontend compatibility
        data['id'] = data.get('user_id')
        data['picture'] = data.get('picture_url')
        data['credits'] = data.get('credits_balance')
        return data
