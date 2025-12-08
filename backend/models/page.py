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

    Storage strategy (handled by PageRepository):
    - Content (content_text, embedding): PostgreSQL
    - Metadata (title, url, etc.): Neo4j Page node
    """
    id: uuid.UUID
    url: str
    title: Optional[str] = None
    content_text: Optional[str] = None

    # Metadata
    canonical_url: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    thumbnail_url: Optional[str] = None
    language: Optional[str] = None
    word_count: int = 0
    pub_time: Optional[datetime] = None
    metadata_confidence: float = 0.0

    # Page gist/summary (from semantic worker)
    gist: Optional[str] = None

    # Embedding (stored in PostgreSQL as vector)
    embedding: Optional[List[float]] = None

    # Status: 'stub', 'preview', 'extracted', 'semantic_complete', 'failed'
    status: str = 'stub'

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Additional metadata as dict (for backward compatibility)
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
        return self.status in ('extracted', 'semantic_complete')

    @property
    def needs_extraction(self) -> bool:
        """Check if page needs extraction"""
        return self.status in ('stub', 'preview')

    @property
    def is_semantically_analyzed(self) -> bool:
        """Check if page has been semantically analyzed"""
        return self.status == 'semantic_complete'
