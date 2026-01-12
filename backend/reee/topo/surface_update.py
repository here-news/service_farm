"""
Surface Update - Pure functions for surface key computation and claim application.

This module handles:
- compute_surface_key: Derives SurfaceKey from ClaimEvidence
- apply_claim_to_surface: Updates SurfaceState with new claim

Pure function - no DB, no LLM.
Uses scope.py and question_key.py for key derivation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, FrozenSet, Tuple

from ..contracts.evidence import ClaimEvidence
from ..contracts.state import SurfaceKey, SurfaceState
from ..contracts.traces import DecisionTrace, FeatureVector, generate_trace_id

from .scope import compute_scope_id, extract_primary_anchors
from .question_key import (
    extract_question_key,
    FallbackLevel,
    QuestionKeyResult,
)


@dataclass(frozen=True)
class SurfaceKeyParams:
    """Parameters for surface key computation."""
    kernel_version: str = "1.0.0"

    @property
    def params_hash(self) -> str:
        """Hash of parameters for trace reproducibility."""
        import hashlib
        content = f"surface_key|{self.kernel_version}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class SurfaceKeyResult:
    """Result of surface key computation."""
    key: SurfaceKey
    scope_id: str
    question_key: str
    question_key_result: QuestionKeyResult
    trace: DecisionTrace


def compute_surface_key(
    evidence: ClaimEvidence,
    params: SurfaceKeyParams = SurfaceKeyParams(),
) -> SurfaceKeyResult:
    """Compute SurfaceKey from ClaimEvidence.

    Pure function - uses scope.py and question_key.py.

    Flow:
    1. Extract anchors from evidence (or use provided)
    2. Compute scope_id from anchors
    3. Extract question_key using fallback chain
    4. Return SurfaceKey with trace

    Args:
        evidence: The claim evidence to process
        params: Surface key computation parameters

    Returns:
        SurfaceKeyResult with key and trace
    """
    # Step 1: Get anchors (use evidence.anchors if provided, else derive)
    if evidence.anchors:
        anchors = evidence.anchors
        all_hubs = False
    else:
        anchors, all_hubs = extract_primary_anchors(evidence.entities)

    # Step 2: Compute scope_id
    scope_id = compute_scope_id(anchors)

    # Step 3: Extract question_key
    qk_result = extract_question_key(
        text=evidence.text,
        entities=evidence.entities,
        anchors=anchors,
        page_id=evidence.page_id,
        claim_id=evidence.claim_id,
        explicit_key=evidence.question_key,
        explicit_confidence=evidence.question_key_confidence,
    )

    # Step 4: Create key
    key = SurfaceKey(
        scope_id=scope_id,
        question_key=qk_result.question_key,
    )

    # Step 5: Create trace
    outcome = _fallback_to_outcome(qk_result.fallback_level)
    rules_fired = _compute_key_rules(qk_result, all_hubs)

    trace = DecisionTrace(
        id=generate_trace_id(),
        decision_type="surface_key",
        subject_id=evidence.claim_id,
        target_id=key.signature,
        candidate_ids=frozenset(),  # No candidates for key derivation
        outcome=outcome,
        features=FeatureVector(
            question_key_confidence=qk_result.confidence,
            extraction_confidence=evidence.entity_confidence,
        ),
        rules_fired=rules_fired,
        params_hash=params.params_hash,
        kernel_version=params.kernel_version,
        timestamp=datetime.utcnow(),
    )

    return SurfaceKeyResult(
        key=key,
        scope_id=scope_id,
        question_key=qk_result.question_key,
        question_key_result=qk_result,
        trace=trace,
    )


def apply_claim_to_surface(
    surface: Optional[SurfaceState],
    evidence: ClaimEvidence,
    key: SurfaceKey,
) -> Tuple[SurfaceState, bool]:
    """Apply claim evidence to a surface.

    Pure function - returns new immutable SurfaceState.

    If surface is None, creates a new one.
    Otherwise, updates with the claim.

    Args:
        surface: Existing surface state (or None for new)
        evidence: The claim evidence to apply
        key: The computed surface key

    Returns:
        Tuple of (updated_surface, is_new)
    """
    if surface is None:
        # Create new surface
        new_surface = SurfaceState(key=key).with_claim(
            claim_id=evidence.claim_id,
            entities=evidence.entities,
            anchors=evidence.anchors or frozenset(),
            source_id=evidence.source_id,
            claim_time=evidence.time,
        )
        return new_surface, True
    else:
        # Update existing surface
        updated = surface.with_claim(
            claim_id=evidence.claim_id,
            entities=evidence.entities,
            anchors=evidence.anchors or frozenset(),
            source_id=evidence.source_id,
            claim_time=evidence.time,
        )
        return updated, False


def _fallback_to_outcome(level: FallbackLevel) -> str:
    """Map fallback level to outcome string for trace."""
    mapping = {
        FallbackLevel.EXPLICIT: "key_explicit",
        FallbackLevel.PATTERN: "key_pattern",
        FallbackLevel.ENTITY: "key_entity",
        FallbackLevel.PAGE_SCOPE: "key_page_scope",
        FallbackLevel.SINGLETON: "key_singleton",
    }
    return mapping.get(level, "key_unknown")


def _compute_key_rules(
    qk_result: QuestionKeyResult,
    all_hubs: bool,
) -> FrozenSet[str]:
    """Compute rules fired for surface key decision."""
    rules = set()

    # Fallback level rules
    level_rules = {
        FallbackLevel.EXPLICIT: "FALLBACK_EXPLICIT",
        FallbackLevel.PATTERN: "FALLBACK_PATTERN",
        FallbackLevel.ENTITY: "FALLBACK_ENTITY",
        FallbackLevel.PAGE_SCOPE: "FALLBACK_PAGE_SCOPE",
        FallbackLevel.SINGLETON: "FALLBACK_SINGLETON",
    }
    rules.add(level_rules.get(qk_result.fallback_level, "FALLBACK_UNKNOWN"))

    # Hub rules
    if all_hubs:
        rules.add("ALL_HUBS_FALLBACK")

    return frozenset(rules)


def merge_surface_entities(
    existing: SurfaceState,
    new_entities: FrozenSet[str],
    new_anchors: FrozenSet[str],
) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Merge entities from new claim into existing surface.

    Returns:
        (merged_entities, merged_anchors)
    """
    merged_entities = existing.entities | new_entities
    merged_anchors = existing.anchor_entities | new_anchors
    return merged_entities, merged_anchors


def expand_time_bounds(
    existing_start: Optional[datetime],
    existing_end: Optional[datetime],
    new_time: Optional[datetime],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Expand time bounds to include new time.

    Returns:
        (time_start, time_end)
    """
    if new_time is None:
        return existing_start, existing_end

    new_start = existing_start
    new_end = existing_end

    if existing_start is None or new_time < existing_start:
        new_start = new_time

    if existing_end is None or new_time > existing_end:
        new_end = new_time

    return new_start, new_end
