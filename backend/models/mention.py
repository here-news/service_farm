"""
Mention domain model - ephemeral extraction output

Mentions are raw entity references extracted from text.
They are NOT persisted - they exist only during the knowledge pipeline
to be resolved into canonical Entity records.

Flow:
  LLM Extraction → Mentions → Identification → Entities
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Mention:
    """
    A mention is a raw entity reference in text (ephemeral, not persisted).

    Mentions capture what the LLM sees in the text, including context
    that helps with identification and disambiguation.

    Example:
        surface_form: "Building A"
        type_hint: "LOCATION"
        context: "fire spread to Building A, also known as the East Tower"
        description: "Residential tower in the housing complex"
        aliases: ["East Tower"]
    """
    id: str                        # Local ID within extraction (m1, m2, m3...)
    surface_form: str              # Raw text as it appears: "Block 6"
    type_hint: str                 # LLM's type guess: PERSON, ORGANIZATION, LOCATION
    context: str                   # Verbatim surrounding text from article

    # LLM-generated description of what/who this entity is
    description: str = ""

    # Other names for the same entity found in the text
    aliases: List[str] = field(default_factory=list)

    # Optional: character span in source text
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    # Which claims reference this mention
    claim_indices: List[int] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Mention):
            return self.id == other.id
        return False


@dataclass
class MentionRelationship:
    """
    A structural relationship between mentions (ephemeral).

    Captures relationships like PART_OF, LOCATED_IN that help
    constrain entity identification.

    Example:
        subject_id: "m1"  (Block 6)
        predicate: "PART_OF"
        object_id: "m2"   (Wang Fuk Court)
    """
    subject_id: str      # Mention ID (e.g., "m1")
    predicate: str       # PART_OF, LOCATED_IN, WORKS_FOR, MEMBER_OF, AFFILIATED_WITH
    object_id: str       # Mention ID (e.g., "m2")


@dataclass
class ExtractionResult:
    """
    Complete output from the extraction stage.

    Contains all mentions, claims (referencing mentions), and relationships.
    This is passed to the identification stage for entity resolution.
    """
    # All mentions found in the text
    mentions: List[Mention] = field(default_factory=list)

    # Claims referencing mentions by ID
    claims: List[dict] = field(default_factory=list)

    # Structural relationships between mentions
    mention_relationships: List[MentionRelationship] = field(default_factory=list)

    # Page-level summary
    gist: str = ""

    # Overall extraction confidence
    confidence: float = 0.5

    # Extraction quality (0.0-1.0): ratio of proper named entities vs garbage
    # < 0.5 means too many generic terms, page should not pass to knowledge_complete
    extraction_quality: float = 0.5

    # Token usage for cost tracking
    token_usage: dict = field(default_factory=dict)

    def get_mention(self, mention_id: str) -> Optional[Mention]:
        """Get mention by ID."""
        for m in self.mentions:
            if m.id == mention_id:
                return m
        return None

    def get_mentions_by_type(self, type_hint: str) -> List[Mention]:
        """Get all mentions of a given type."""
        return [m for m in self.mentions if m.type_hint == type_hint]

    def get_relationships_for(self, mention_id: str) -> List[MentionRelationship]:
        """Get all relationships involving a mention (as subject or object)."""
        return [
            r for r in self.mention_relationships
            if r.subject_id == mention_id or r.object_id == mention_id
        ]
