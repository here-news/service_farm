"""
Signal Contracts - Quality indicators and inquiry seeds.

EpistemicSignal replaces MetaClaim with clearer semantics:
- Signals are quality/status indicators
- InquirySeeds are actionable requests for more evidence

Signals connect to the inquiry system via explicit mapping.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Dict
import uuid


def generate_signal_id() -> str:
    """Generate a unique signal ID."""
    return f"sig_{uuid.uuid4().hex[:12]}"


def generate_inquiry_id() -> str:
    """Generate a unique inquiry ID."""
    return f"inq_{uuid.uuid4().hex[:12]}"


class SignalType(Enum):
    """Quality/status signal types.

    These replace the old MetaClaim types with clearer semantics.
    """

    # Data quality signals
    MISSING_TIME = "missing_time"  # Claim has no timestamp
    EXTRACTION_SPARSE = "extraction_sparse"  # Few entities extracted
    SCOPE_UNDERPOWERED = "scope_underpowered"  # All anchors are hubs

    # Processing block signals
    BRIDGE_BLOCKED = "bridge_blocked"  # Shared anchor but incompatible companions

    # Epistemic state signals
    HIGH_ENTROPY = "high_entropy"  # Can't decide on value
    CONFLICT = "conflict"  # Contradictory observations
    CORROBORATION_LACKING = "corroboration_lacking"  # Single source


class Severity(Enum):
    """Signal severity levels."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # May affect quality
    ERROR = "error"  # Blocks normal processing


@dataclass(frozen=True)
class EpistemicSignal:
    """Quality/status signal emitted by kernel.

    Signals are not errors - they're structured observations about
    the epistemic state of the system.
    """

    id: str
    signal_type: SignalType
    subject_id: str  # What this is about (claim/surface/incident ID)
    subject_type: str  # "claim", "surface", "incident"
    severity: Severity
    evidence: Dict[str, Any]  # Supporting data
    resolution_hint: Optional[str]  # What would resolve this
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence/logging."""
        return {
            "id": self.id,
            "signal_type": self.signal_type.value,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "resolution_hint": self.resolution_hint,
            "timestamp": self.timestamp.isoformat(),
        }


class InquiryType(Enum):
    """What kind of evidence would help.

    Maps from SignalType to actionable inquiry.
    """

    RESOLVE_VALUE = "resolve_value"  # Need authoritative source
    DISAMBIGUATE_SCOPE = "disambiguate_scope"  # Which referent?
    REQUEST_TIMESTAMP = "request_timestamp"  # When did this happen?
    IMPROVE_EXTRACTION = "improve_extraction"  # Better entity linking
    SEEK_CORROBORATION = "seek_corroboration"  # Find more sources


# Explicit mapping from signals to inquiries
SIGNAL_TO_INQUIRY: Dict[SignalType, InquiryType] = {
    SignalType.CONFLICT: InquiryType.RESOLVE_VALUE,
    SignalType.BRIDGE_BLOCKED: InquiryType.DISAMBIGUATE_SCOPE,
    SignalType.MISSING_TIME: InquiryType.REQUEST_TIMESTAMP,
    SignalType.EXTRACTION_SPARSE: InquiryType.IMPROVE_EXTRACTION,
    SignalType.CORROBORATION_LACKING: InquiryType.SEEK_CORROBORATION,
    SignalType.HIGH_ENTROPY: InquiryType.SEEK_CORROBORATION,
    # SCOPE_UNDERPOWERED doesn't map to inquiry (informational only)
}


@dataclass(frozen=True)
class InquirySeed:
    """Actionable request for evidence.

    Generated from signals, consumed by the inquiry system.
    """

    id: str
    inquiry_type: InquiryType
    subject_id: str  # What this is about
    priority: float  # Based on entropy reduction potential [0, 1]
    question: str  # Human-readable question
    current_state: Dict[str, Any]  # What we know now
    evidence_needed: str  # Description of what would resolve this
    source_signal_id: Optional[str]  # Link to originating signal

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "id": self.id,
            "inquiry_type": self.inquiry_type.value,
            "subject_id": self.subject_id,
            "priority": self.priority,
            "question": self.question,
            "current_state": self.current_state,
            "evidence_needed": self.evidence_needed,
            "source_signal_id": self.source_signal_id,
        }


def signal_to_inquiry(signal: EpistemicSignal) -> Optional[InquirySeed]:
    """Convert signal to actionable inquiry.

    Returns None if signal doesn't warrant an inquiry.
    """
    inquiry_type = SIGNAL_TO_INQUIRY.get(signal.signal_type)
    if inquiry_type is None:
        return None

    # Generate question based on type
    question = _generate_question(signal, inquiry_type)
    evidence_needed = _describe_needed_evidence(inquiry_type)
    priority = _compute_priority(signal)

    return InquirySeed(
        id=generate_inquiry_id(),
        inquiry_type=inquiry_type,
        subject_id=signal.subject_id,
        priority=priority,
        question=question,
        current_state=signal.evidence,
        evidence_needed=evidence_needed,
        source_signal_id=signal.id,
    )


def _generate_question(signal: EpistemicSignal, inquiry_type: InquiryType) -> str:
    """Generate human-readable question from signal."""
    templates = {
        InquiryType.RESOLVE_VALUE: f"What is the correct value for {signal.subject_id}?",
        InquiryType.DISAMBIGUATE_SCOPE: f"Are these referring to the same event? ({signal.subject_id})",
        InquiryType.REQUEST_TIMESTAMP: f"When did this happen? ({signal.subject_id})",
        InquiryType.IMPROVE_EXTRACTION: f"What entities are involved in {signal.subject_id}?",
        InquiryType.SEEK_CORROBORATION: f"Are there other sources for {signal.subject_id}?",
    }
    return templates.get(inquiry_type, f"Need more information about {signal.subject_id}")


def _describe_needed_evidence(inquiry_type: InquiryType) -> str:
    """Describe what evidence would resolve the inquiry."""
    descriptions = {
        InquiryType.RESOLVE_VALUE: "Authoritative source with confirmed value",
        InquiryType.DISAMBIGUATE_SCOPE: "Explicit confirmation of referent identity",
        InquiryType.REQUEST_TIMESTAMP: "Date/time from primary source or context",
        InquiryType.IMPROVE_EXTRACTION: "Better entity extraction or manual annotation",
        InquiryType.SEEK_CORROBORATION: "Independent source confirming the claim",
    }
    return descriptions.get(inquiry_type, "Additional evidence")


def _compute_priority(signal: EpistemicSignal) -> float:
    """Compute inquiry priority from signal.

    Higher priority = more urgent to resolve.
    """
    base_priority = {
        SignalType.CONFLICT: 0.9,  # High - blocking progress
        SignalType.BRIDGE_BLOCKED: 0.7,  # Medium-high
        SignalType.HIGH_ENTROPY: 0.6,  # Medium
        SignalType.CORROBORATION_LACKING: 0.5,  # Medium
        SignalType.MISSING_TIME: 0.4,  # Lower
        SignalType.EXTRACTION_SPARSE: 0.3,  # Lower
    }.get(signal.signal_type, 0.5)

    # Adjust by severity
    severity_multiplier = {
        Severity.ERROR: 1.2,
        Severity.WARNING: 1.0,
        Severity.INFO: 0.8,
    }.get(signal.severity, 1.0)

    return min(1.0, base_priority * severity_multiplier)
