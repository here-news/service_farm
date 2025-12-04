"""
Entity domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import uuid


@dataclass
class Entity:
    """
    Entity domain model - storage-agnostic representation

    Represents a named entity (person, organization, location, etc.)
    """
    id: uuid.UUID
    canonical_name: str
    entity_type: str  # PERSON, ORG, GPE, LOC, etc.

    # Alternative names/aliases
    aliases: List[str] = field(default_factory=list)

    # Mention count (how many times referenced across all claims)
    mention_count: int = 0

    # Metadata
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure id is UUID"""
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)

    @property
    def is_person(self) -> bool:
        """Check if entity is a person"""
        return self.entity_type == 'PERSON'

    @property
    def is_organization(self) -> bool:
        """Check if entity is an organization"""
        return self.entity_type in ['ORG', 'ORGANIZATION']

    @property
    def is_location(self) -> bool:
        """Check if entity is a location"""
        return self.entity_type in ['GPE', 'LOC', 'LOCATION']

    def add_alias(self, alias: str):
        """Add an alias to this entity"""
        if alias not in self.aliases and alias != self.canonical_name:
            self.aliases.append(alias)
