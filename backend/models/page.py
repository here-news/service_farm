"""
Page domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from utils.id_generator import generate_page_id, validate_id, is_uuid, uuid_to_short_id


@dataclass
class Page:
    """
    Page domain model - storage-agnostic representation

    Storage strategy (handled by PageRepository):
    - Content (content_text, embedding): PostgreSQL
    - Metadata (title, url, etc.): Neo4j Page node

    ID format: pg_xxxxxxxx (11 chars)
    """
    id: str  # Short ID: pg_xxxxxxxx
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

    # Publisher info
    domain: Optional[str] = None       # e.g. "bbc.com"
    site_name: Optional[str] = None    # e.g. "BBC News" (from og:site_name)

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
        """Ensure id is in short format, convert UUIDs if needed"""
        if self.id:
            if is_uuid(self.id):
                self.id = uuid_to_short_id(self.id, 'page')
            elif not validate_id(self.id):
                # Generate new ID if invalid
                self.id = generate_page_id()

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
