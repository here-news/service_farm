"""
Page domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import uuid


@dataclass
class Page:
    """
    Page domain model - storage-agnostic representation

    Attributes correspond to core.pages table but abstracted from storage
    """
    id: uuid.UUID
    url: str
    title: Optional[str] = None
    content_text: Optional[str] = None

    # Metadata
    canonical_url: Optional[str] = None
    byline: Optional[str] = None
    site_name: Optional[str] = None
    domain: Optional[str] = None
    language: Optional[str] = None
    word_count: int = 0
    pub_time: Optional[datetime] = None

    # Page gist/summary
    gist: Optional[str] = None

    # Embedding (stored in PostgreSQL as vector)
    embedding: Optional[List[float]] = None

    # Status tracking
    extraction_status: str = 'pending'
    semantic_status: str = 'pending'

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Additional metadata as dict
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure id is UUID"""
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)

    @property
    def has_content(self) -> bool:
        """Check if page has sufficient content"""
        return bool(self.content_text) and self.word_count >= 100

    @property
    def is_extracted(self) -> bool:
        """Check if page has been extracted"""
        return self.extraction_status == 'completed'

    @property
    def is_semantically_analyzed(self) -> bool:
        """Check if page has been semantically analyzed"""
        return self.semantic_status == 'completed'
