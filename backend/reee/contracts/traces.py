"""
Trace Contracts - Decision and belief update traces.

Every kernel decision emits a trace for:
- Debugging
- Explainability
- Replay validation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, FrozenSet, List
import uuid


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return f"trace_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class FeatureVector:
    """Numeric features that contributed to a decision.

    All features are optional - only populated if relevant.
    """

    # Overlap metrics
    anchor_overlap: float = 0.0  # Jaccard of anchors
    companion_jaccard: float = 0.0  # Jaccard of companions

    # Time
    time_delta_hours: Optional[float] = None

    # Structural
    motif_support: int = 0  # How many shared motifs

    # Confidence from evidence
    extraction_confidence: float = 0.5
    question_key_confidence: float = 0.5
    time_confidence: float = 0.5


@dataclass(frozen=True)
class DecisionTrace:
    """Membership/routing decision trace.

    Emitted for every routing decision:
    - surface_key: which surface does this claim join?
    - incident_membership: which incident does this surface join?
    """

    id: str
    decision_type: str  # "surface_key", "incident_membership"
    subject_id: str  # Claim or surface being processed
    target_id: Optional[str]  # Surface or incident joined (None if rejected)
    candidate_ids: FrozenSet[str]  # Alternatives considered
    outcome: str  # "created_new", "joined", "rejected", "key_explicit", etc.
    features: FeatureVector
    rules_fired: FrozenSet[str]  # Which rules determined outcome
    params_hash: str
    kernel_version: str
    timestamp: datetime

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Serialize for LLM explainer (outside kernel).

        Returns a dict suitable for inclusion in an LLM prompt.
        """
        return {
            "decision_type": self.decision_type,
            "subject": self.subject_id,
            "target": self.target_id,
            "outcome": self.outcome,
            "features": {
                "anchor_overlap": f"{self.features.anchor_overlap:.1%}",
                "companion_similarity": f"{self.features.companion_jaccard:.1%}",
                "time_gap": (
                    f"{self.features.time_delta_hours:.1f}h"
                    if self.features.time_delta_hours is not None
                    else "unknown"
                ),
                "extraction_confidence": f"{self.features.extraction_confidence:.0%}",
                "question_key_confidence": f"{self.features.question_key_confidence:.0%}",
            },
            "rules": sorted(self.rules_fired),
            "candidates_considered": len(self.candidate_ids),
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        """Full serialization for debugging."""
        return {
            "id": self.id,
            "decision_type": self.decision_type,
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "candidate_ids": sorted(self.candidate_ids),
            "outcome": self.outcome,
            "features": {
                "anchor_overlap": self.features.anchor_overlap,
                "companion_jaccard": self.features.companion_jaccard,
                "time_delta_hours": self.features.time_delta_hours,
                "motif_support": self.features.motif_support,
                "extraction_confidence": self.features.extraction_confidence,
                "question_key_confidence": self.features.question_key_confidence,
                "time_confidence": self.features.time_confidence,
            },
            "rules_fired": sorted(self.rules_fired),
            "params_hash": self.params_hash,
            "kernel_version": self.kernel_version,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class BeliefUpdateTrace:
    """Jaynes posterior update trace (L2 proposition inference).

    Emitted when a typed observation updates a surface's belief state.
    """

    id: str
    surface_id: str
    question_key: str
    claim_id: str

    # Prior state
    prior_entropy: float
    prior_map: Optional[float]  # MAP estimate before update
    prior_support: int  # Number of observations before

    # Observation
    observation_value: Any
    observation_confidence: float
    observation_authority: float
    noise_model: str  # "uniform", "calibrated"

    # Posterior state
    posterior_entropy: float
    posterior_map: Optional[float]  # MAP estimate after update
    posterior_support: int  # Number of observations after

    # Derived
    surprisal: float  # How unexpected was this observation?
    conflict_detected: bool  # Did this contradict consensus?
    timestamp: datetime

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Serialize for LLM explainer."""
        return {
            "surface": self.surface_id,
            "question": self.question_key,
            "observation": {
                "value": self.observation_value,
                "confidence": f"{self.observation_confidence:.0%}",
                "authority": f"{self.observation_authority:.0%}",
            },
            "prior": {
                "map": self.prior_map,
                "entropy": f"{self.prior_entropy:.2f}",
                "observations": self.prior_support,
            },
            "posterior": {
                "map": self.posterior_map,
                "entropy": f"{self.posterior_entropy:.2f}",
                "observations": self.posterior_support,
            },
            "surprisal": f"{self.surprisal:.2f}",
            "conflict": self.conflict_detected,
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        """Full serialization for debugging."""
        return {
            "id": self.id,
            "surface_id": self.surface_id,
            "question_key": self.question_key,
            "claim_id": self.claim_id,
            "prior_entropy": self.prior_entropy,
            "prior_map": self.prior_map,
            "prior_support": self.prior_support,
            "observation_value": self.observation_value,
            "observation_confidence": self.observation_confidence,
            "observation_authority": self.observation_authority,
            "noise_model": self.noise_model,
            "posterior_entropy": self.posterior_entropy,
            "posterior_map": self.posterior_map,
            "posterior_support": self.posterior_support,
            "surprisal": self.surprisal,
            "conflict_detected": self.conflict_detected,
            "timestamp": self.timestamp.isoformat(),
        }
