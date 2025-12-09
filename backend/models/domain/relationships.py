"""
Relationship domain models

Represents connections between entities in the knowledge graph

ID format: {prefix}_xxxxxxxx (11 chars)
"""
from dataclasses import dataclass
from typing import Optional

from utils.id_generator import is_uuid, uuid_to_short_id


@dataclass
class ClaimEntityLink:
    """
    Link between a Claim and an Entity

    Represents that an entity is mentioned in a claim with a specific role
    """
    claim_id: str  # cl_xxxxxxxx
    entity_id: str  # en_xxxxxxxx
    relationship_type: str  # MENTIONS, ACTOR, SUBJECT, LOCATION

    # Confidence in this relationship
    confidence: float = 0.9

    def __post_init__(self):
        """Ensure IDs are in short format"""
        if is_uuid(self.claim_id):
            self.claim_id = uuid_to_short_id(self.claim_id, 'claim')
        if is_uuid(self.entity_id):
            self.entity_id = uuid_to_short_id(self.entity_id, 'entity')

    @property
    def is_actor(self) -> bool:
        """Entity is the actor/agent in the claim"""
        return self.relationship_type == 'ACTOR'

    @property
    def is_subject(self) -> bool:
        """Entity is the subject/patient in the claim"""
        return self.relationship_type == 'SUBJECT'

    @property
    def is_location(self) -> bool:
        """Entity is a location in the claim"""
        return self.relationship_type == 'LOCATION'

    @property
    def is_mention(self) -> bool:
        """Generic mention (default)"""
        return self.relationship_type == 'MENTIONS'


@dataclass
class PageEventLink:
    """
    Link between a Page and an Event

    Represents that a page contributes to an event
    """
    page_id: str  # pg_xxxxxxxx
    event_id: str  # ev_xxxxxxxx

    # Score/confidence in this attachment
    attachment_score: float = 0.5

    def __post_init__(self):
        """Ensure IDs are in short format"""
        if is_uuid(self.page_id):
            self.page_id = uuid_to_short_id(self.page_id, 'page')
        if is_uuid(self.event_id):
            self.event_id = uuid_to_short_id(self.event_id, 'event')


@dataclass
class EventRelationship:
    """
    Relationship between two Events

    Represents causal, temporal, or hierarchical relationships
    """
    from_event_id: str  # ev_xxxxxxxx
    to_event_id: str  # ev_xxxxxxxx
    relationship_type: str  # CAUSED, TRIGGERED, PART_OF, RELATED_TO

    # Confidence in this relationship
    confidence: float = 0.8

    # Optional metadata
    metadata: dict = None

    def __post_init__(self):
        """Ensure IDs are in short format"""
        if is_uuid(self.from_event_id):
            self.from_event_id = uuid_to_short_id(self.from_event_id, 'event')
        if is_uuid(self.to_event_id):
            self.to_event_id = uuid_to_short_id(self.to_event_id, 'event')
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_causal(self) -> bool:
        """This event caused another"""
        return self.relationship_type == 'CAUSED'

    @property
    def is_trigger(self) -> bool:
        """This event triggered another"""
        return self.relationship_type == 'TRIGGERED'

    @property
    def is_part_of(self) -> bool:
        """This event is part of another (hierarchical)"""
        return self.relationship_type == 'PART_OF'
