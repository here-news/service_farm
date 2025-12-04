"""
Domain Models - Storage-agnostic data structures

These models represent the core domain entities independent of storage layer.
Workers and services operate on these models, not raw database rows.

Architecture:
- Domain models are pure Python objects (dataclasses)
- Storage details (PostgreSQL, Neo4j) are abstracted via repositories
- Business logic operates on these models, not database rows
"""

from .page import Page
from .claim import Claim
from .entity import Entity
from .event import Event
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

    # Relationships
    'ClaimEntityLink',
    'PageEventLink',
    'EventRelationship',
]
