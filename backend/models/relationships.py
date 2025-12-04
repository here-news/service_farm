"""
Relationship domain models

Represents connections between entities in the knowledge graph
"""
from dataclasses import dataclass
from typing import Optional
import uuid


@dataclass
class ClaimEntityLink:
    """
    Link between a Claim and an Entity

    Represents that an entity is mentioned in a claim with a specific role
    """
    claim_id: uuid.UUID
    entity_id: uuid.UUID
    relationship_type: str  # MENTIONS, ACTOR, SUBJECT, LOCATION

    # Confidence in this relationship
    confidence: float = 0.9

    def __post_init__(self):
        """Ensure IDs are UUIDs"""
        if isinstance(self.claim_id, str):
            self.claim_id = uuid.UUID(self.claim_id)
        if isinstance(self.entity_id, str):
            self.entity_id = uuid.UUID(self.entity_id)

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
class PhaseClaimLink:
    """
    Link between a Phase and a Claim

    Represents that a claim supports/belongs to a phase
    """
    phase_id: uuid.UUID
    claim_id: uuid.UUID

    # Confidence that this claim belongs to this phase
    confidence: float = 0.9

    def __post_init__(self):
        """Ensure IDs are UUIDs"""
        if isinstance(self.phase_id, str):
            self.phase_id = uuid.UUID(self.phase_id)
        if isinstance(self.claim_id, str):
            self.claim_id = uuid.UUID(self.claim_id)


@dataclass
class PageEventLink:
    """
    Link between a Page and an Event

    Represents that a page contributes to an event
    """
    page_id: uuid.UUID
    event_id: uuid.UUID

    # Score/confidence in this attachment
    attachment_score: float = 0.5

    def __post_init__(self):
        """Ensure IDs are UUIDs"""
        if isinstance(self.page_id, str):
            self.page_id = uuid.UUID(self.page_id)
        if isinstance(self.event_id, str):
            self.event_id = uuid.UUID(self.event_id)


@dataclass
class EventRelationship:
    """
    Relationship between two Events

    Represents causal, temporal, or hierarchical relationships
    """
    from_event_id: uuid.UUID
    to_event_id: uuid.UUID
    relationship_type: str  # CAUSED, TRIGGERED, PART_OF, RELATED_TO

    # Confidence in this relationship
    confidence: float = 0.8

    # Optional metadata
    metadata: dict = None

    def __post_init__(self):
        """Ensure IDs are UUIDs"""
        if isinstance(self.from_event_id, str):
            self.from_event_id = uuid.UUID(self.from_event_id)
        if isinstance(self.to_event_id, str):
            self.to_event_id = uuid.UUID(self.to_event_id)
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
