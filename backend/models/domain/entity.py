"""
Entity domain model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from utils.id_generator import (
    generate_entity_id, validate_id, is_uuid, uuid_to_short_id
)


@dataclass
class Entity:
    """
    Entity domain model - storage-agnostic representation

    Represents a named entity (person, organization, location, etc.)

    ID format: en_xxxxxxxx (11 chars)
    """
    id: str  # Short ID: en_xxxxxxxx
    canonical_name: str
    entity_type: str  # PERSON, ORGANIZATION, LOCATION

    # Alternative names/aliases
    aliases: List[str] = field(default_factory=list)

    # Mention count (how many times referenced across all claims)
    mention_count: int = 0

    # AI-generated profile summary (from semantic worker)
    profile_summary: Optional[str] = None

    # Wikidata enrichment (from Wikidata worker)
    wikidata_qid: Optional[str] = None
    wikidata_label: Optional[str] = None
    wikidata_description: Optional[str] = None
    wikidata_image: Optional[str] = None  # Wikimedia Commons thumbnail URL

    # Status: 'pending', 'checked', 'enriched'
    status: str = 'pending'

    # Confidence score from enrichment
    confidence: float = 0.0

    # Additional metadata (coordinates, thumbnail, etc.)
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure id is in short format, convert UUIDs if needed"""
        if self.id:
            if is_uuid(self.id):
                self.id = uuid_to_short_id(self.id, 'entity')
            elif not validate_id(self.id):
                self.id = generate_entity_id()

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
