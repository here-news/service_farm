"""
Extraction Task Models
Handles URL extraction tasks and results
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Boolean, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.schema import Index
from ..database.connection import Base
import uuid


def generate_uuid():
    """Generate UUID for task_id"""
    return uuid.uuid4()


class ExtractionTask(Base):
    """Extraction task model - stored in tasks.extraction_tasks"""
    __tablename__ = "extraction_tasks"
    __table_args__ = {'schema': 'tasks'}

    # Primary key (UUID column in database)
    id = Column(UUID(as_uuid=False), primary_key=True)

    # Basic task info
    url = Column(Text, nullable=False, index=True)
    canonical_url = Column(Text, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True, index=True)

    # Status tracking
    status = Column(String, nullable=False, default="pending", index=True)
    # Status values: pending, processing, completed, failed, blocked

    current_stage = Column(String, nullable=True)
    # Stage values: extraction, cleaning, resolution, semantization

    # Task results (JSONB for flexibility)
    result = Column(JSONB, nullable=True)
    # Contains: title, content_text, word_count, screenshot_url, meta_description, etc.

    semantic_data = Column(JSONB, nullable=True)
    # Contains: claims, entities (people, orgs, locations)

    preview_meta = Column(JSONB, nullable=True)
    # Preview metadata for quick display

    story_match = Column(JSONB, nullable=True)
    # Story matching results

    # Error handling
    error_message = Column(Text, nullable=True)
    block_reason = Column(Text, nullable=True)

    # Story assignment
    target_story_id = Column(UUID(as_uuid=False), nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index('idx_extraction_tasks_status_created', 'status', 'created_at'),
        Index('idx_extraction_tasks_canonical_url', 'canonical_url'),
        Index('idx_extraction_tasks_user_created', 'user_id', 'created_at'),
        {'schema': 'tasks'}
    )


class UserURL(Base):
    """User-URL relationship tracking"""
    __tablename__ = "user_urls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    task_id = Column(String, nullable=False, index=True)  # Links to ExtractionTask.id
    url = Column(Text, nullable=False)

    # Submission metadata
    submitted_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    credits_spent = Column(Integer, default=10)

    # User notes and organization
    user_notes = Column(Text, nullable=True)
    tags = Column(JSONB, nullable=True)  # Array of tags

    # Bookmarking
    is_bookmarked = Column(Boolean, default=False)
    bookmarked_at = Column(DateTime(timezone=True), nullable=True)

    # Sharing
    is_public = Column(Boolean, default=False)
    share_token = Column(String(64), unique=True, nullable=True)

    # Preservation tracking
    preservation_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Indexes
    __table_args__ = (
        Index('idx_user_urls_user_submitted', 'user_id', 'submitted_at'),
        Index('idx_user_urls_bookmarked', 'user_id', 'is_bookmarked'),
    )
