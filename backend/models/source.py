"""
Source entity model - tracks publisher/author credibility

Sources are entities (organizations or persons) that publish content.
They have credibility scores that emerge from track record, not prejudice.

Credibility Model (No Prejudice):
- Base: Organization 0.51, Individual 0.49
- +0.01 if Wikidata-linkable (identifiable)
- Track record adjusts score over time
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import uuid


@dataclass
class CredibilityEvent:
    """A single event that affects credibility score."""
    timestamp: datetime
    event_type: str        # corroborated, contradicted, scoop_verified, retraction
    delta: float           # Score change (+/-)
    details: Optional[str] = None


@dataclass
class Source:
    """
    Source entity with emergent credibility.

    Extends the concept of Entity for publishers/authors.
    Credibility is earned through track record, not assigned by prejudice.
    """
    id: uuid.UUID
    canonical_name: str                    # "South China Morning Post"
    entity_type: str                       # ORGANIZATION or PERSON

    # Domain(s) this source publishes on
    domains: List[str] = field(default_factory=list)

    # Wikidata linking (just for identity, not credibility boost beyond +0.01)
    wikidata_qid: Optional[str] = None
    wikidata_label: Optional[str] = None

    # Track record (inputs to credibility calculation)
    claims_published: int = 0              # Total claims from this source
    claims_corroborated: int = 0           # Claims later verified by others
    claims_contradicted: int = 0           # Claims proven wrong
    claims_exclusive_verified: int = 0     # Scoops that held up
    retractions: int = 0                   # Corrections/retractions issued

    # Credibility history for auditing
    credibility_history: List[CredibilityEvent] = field(default_factory=list)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)

    @property
    def credibility_score(self) -> float:
        """
        Calculate credibility from track record only.
        No hardcoded source preferences - credibility is earned.

        Base:
        - Organization: 0.51
        - Person: 0.49

        Boosts (verifiable facts):
        - +0.01 if has Wikidata QID (identifiable entity)

        Track record:
        - +0.001 per corroborated claim
        - -0.002 per contradicted claim
        - +0.005 per verified exclusive scoop
        - -0.001 per retraction

        Returns: 0.10 - 0.95 (clamped)
        """
        # Base credibility
        if self.entity_type == "ORGANIZATION":
            base = 0.51
        else:
            base = 0.49

        # Small boost for verifiable identity
        identity_boost = 0.01 if self.wikidata_qid else 0.0

        # Track record adjustments
        corroboration_boost = self.claims_corroborated * 0.001
        contradiction_penalty = self.claims_contradicted * 0.002
        scoop_boost = self.claims_exclusive_verified * 0.005
        retraction_penalty = self.retractions * 0.001

        # Calculate
        score = (
            base
            + identity_boost
            + corroboration_boost
            - contradiction_penalty
            + scoop_boost
            - retraction_penalty
        )

        # Clamp to valid range
        return max(0.10, min(0.95, score))

    def record_corroboration(self, details: str = None):
        """Record that a claim from this source was corroborated."""
        self.claims_corroborated += 1
        self.credibility_history.append(CredibilityEvent(
            timestamp=datetime.utcnow(),
            event_type="corroborated",
            delta=0.001,
            details=details
        ))

    def record_contradiction(self, details: str = None):
        """Record that a claim from this source was contradicted."""
        self.claims_contradicted += 1
        self.credibility_history.append(CredibilityEvent(
            timestamp=datetime.utcnow(),
            event_type="contradicted",
            delta=-0.002,
            details=details
        ))

    def record_scoop_verified(self, details: str = None):
        """Record that an exclusive scoop was later verified."""
        self.claims_exclusive_verified += 1
        self.credibility_history.append(CredibilityEvent(
            timestamp=datetime.utcnow(),
            event_type="scoop_verified",
            delta=0.005,
            details=details
        ))

    def record_retraction(self, details: str = None):
        """Record that source issued a retraction/correction."""
        self.retractions += 1
        self.credibility_history.append(CredibilityEvent(
            timestamp=datetime.utcnow(),
            event_type="retraction",
            delta=-0.001,
            details=details
        ))

    @property
    def is_established(self) -> bool:
        """Source has enough track record for meaningful credibility."""
        return self.claims_published >= 10

    @property
    def track_record_summary(self) -> dict:
        """Summary of source track record."""
        return {
            "claims_published": self.claims_published,
            "corroborated": self.claims_corroborated,
            "contradicted": self.claims_contradicted,
            "exclusive_verified": self.claims_exclusive_verified,
            "retractions": self.retractions,
            "credibility_score": self.credibility_score
        }
