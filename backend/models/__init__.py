"""
Domain Models - Storage-agnostic data structures

These models represent the core domain entities independent of storage layer.
Workers and services operate on these models, not raw database rows.

Architecture:
- Domain models are pure Python objects (dataclasses)
- Storage details (PostgreSQL, Neo4j) are abstracted via repositories
- Business logic operates on these models, not database rows

Knowledge Pipeline Models:
- Mention: Ephemeral entity reference from extraction (not persisted)
- Entity: Canonical knowledge base entry (persisted in Neo4j)
- Source: Publisher/author with credibility tracking
"""

from .page import Page
from .claim import Claim
from .entity import Entity
from .event import Event
from .mention import Mention, MentionRelationship, ExtractionResult
from .source import Source, CredibilityEvent
from .relationships import (
    ClaimEntityLink,
    PageEventLink,
    EventRelationship
)

__all__ = [
    # Core entities
    'Page',
    'Claim',
    'Entity',
    'Event',

    # Knowledge pipeline (extraction → identification)
    'Mention',
    'MentionRelationship',
    'ExtractionResult',

    # Source credibility
    'Source',
    'CredibilityEvent',

    # Relationships
    'ClaimEntityLink',
    'PageEventLink',
    'EventRelationship',
]
