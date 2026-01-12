"""
Evidence Contract - Input to the kernel.

ClaimEvidence is the ONLY input the kernel receives.
All LLM/DB enrichment happens BEFORE this is constructed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, FrozenSet, Tuple
import hashlib


@dataclass(frozen=True)
class TypedObservation:
    """Typed value observation for Jaynes inference.

    Used when a claim makes a quantitative assertion (e.g., "17 people died").
    """

    value: Any  # The observed value (numeric or categorical)
    unit: Optional[str] = None  # e.g., "people", "dollars", "percent"
    confidence: float = 0.5  # How confident is the extraction? [0, 1]
    authority: float = 0.5  # How authoritative is the source? [0, 1]


@dataclass(frozen=True)
class ClaimEvidence:
    """Minimal contract for claim input to kernel.

    This is the ONLY input the kernel receives.
    All LLM/DB enrichment happens BEFORE this is constructed.

    Design principles:
    - Immutable (frozen=True)
    - No optional DB/LLM calls inside
    - All fields have sensible defaults for weak evidence
    - Confidence fields fail independently
    """

    # Required identifiers
    claim_id: str
    text: str
    source_id: str  # Publisher/domain (e.g., "bbc.com")
    page_id: Optional[str] = None  # Specific page URL/ID (for page_scope fallback)

    # Derived evidence (may be weak/empty)
    entities: FrozenSet[str] = frozenset()
    anchors: FrozenSet[str] = frozenset()  # Subset of entities that define scope
    question_key: Optional[str] = None  # From LLM or pattern extraction
    time: Optional[datetime] = None

    # Confidence per field (fail independently)
    entity_confidence: float = 0.5
    question_key_confidence: float = 0.5
    time_confidence: float = 0.5

    # Optional enrichments (already computed, not live LLM)
    embedding: Optional[Tuple[float, ...]] = None
    typed_observation: Optional[TypedObservation] = None

    # Provenance (for replay/debugging)
    provider_versions: Dict[str, str] = field(default_factory=dict)

    @property
    def evidence_hash(self) -> str:
        """Deterministic hash of all evidence for cache invalidation."""
        components = [
            self.claim_id,
            self.text,
            self.source_id,
            self.page_id or "",
            ",".join(sorted(self.entities)),
            ",".join(sorted(self.anchors)),
            self.question_key or "",
            self.time.isoformat() if self.time else "",
            str(self.entity_confidence),
            str(self.question_key_confidence),
            str(self.time_confidence),
        ]
        content = "|".join(components)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def companions(self) -> FrozenSet[str]:
        """Entities that are not anchors (context entities)."""
        return self.entities - self.anchors

    def with_enrichment(self, **kwargs) -> "ClaimEvidence":
        """Create a new ClaimEvidence with additional enrichments.

        Useful for evidence providers to add data without mutation.
        """
        current = {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_id": self.source_id,
            "page_id": self.page_id,
            "entities": self.entities,
            "anchors": self.anchors,
            "question_key": self.question_key,
            "time": self.time,
            "entity_confidence": self.entity_confidence,
            "question_key_confidence": self.question_key_confidence,
            "time_confidence": self.time_confidence,
            "embedding": self.embedding,
            "typed_observation": self.typed_observation,
            "provider_versions": dict(self.provider_versions),
        }
        current.update(kwargs)

        # Handle provider_versions merge
        if "provider_versions" in kwargs and isinstance(kwargs["provider_versions"], dict):
            merged_versions = dict(self.provider_versions)
            merged_versions.update(kwargs["provider_versions"])
            current["provider_versions"] = merged_versions

        return ClaimEvidence(**current)
