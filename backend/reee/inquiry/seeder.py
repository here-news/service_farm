"""
Inquiry Seeder: Proto-Inquiry Emergence from Weaver
====================================================

This module implements the bridge between weaver outputs and inquiries.

Concept:
- Surface already implies a question (claims answering same question_key)
- MetaClaim signals when that question has unresolved tension
- ProtoInquiry is the system-generated epistemic object that emerges

ProtoInquiry vs UserInquiry:
- ProtoInquiry: System-generated from surfaces + meta-claims (no contract)
- UserInquiry: User-adopted with resolution rules, stakes, deadlines (contract)

Emergence Rules:
1. typed surface + high_entropy → "What is the value of X for event Y?"
2. typed surface + unresolved_conflict → "Which value is correct for X?"
3. typed surface + single_source_only → "Can we corroborate claim about X?"
4. event fragmentation → "Are these surfaces one incident or multiple?"
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Set, Optional, Any, Tuple
import uuid

from ..types import Surface, MetaClaim, Parameters
from ..typed_belief import TypedBeliefState, Observation, CountDomain, CategoricalDomain


class ProtoInquiryType(Enum):
    """Type of question the proto-inquiry represents."""
    VALUE_RESOLUTION = "value_resolution"     # What is the value of X?
    CONFLICT_RESOLUTION = "conflict_resolution"  # Which value is correct?
    CORROBORATION = "corroboration"           # Can we verify single source?
    SCOPE_CLARIFICATION = "scope_clarification"  # Same event or multiple?
    TRUTH_VERIFICATION = "truth_verification"    # Did X actually happen?


class SchemaType(Enum):
    """Schema type for the inquiry answer domain."""
    MONOTONE_COUNT = "monotone_count"      # death_count, injury_count
    CATEGORICAL = "categorical"            # legal_status, verdict
    BOOLEAN = "boolean"                    # did_X_happen
    REPORT_TRUTH = "report_truth"          # is_claim_true
    QUOTE_AUTHENTICITY = "quote_authenticity"  # did_person_say_X


@dataclass
class ViewId:
    """
    Identifier for which projection of the field this inquiry is about.

    Prevents accidental merging across different projections:
    - incident vs case scale may answer differently
    - Different params produce different surfaces
    - Each snapshot is immutable
    """
    operator: str = "incident"       # "incident" | "case" | "pattern"
    scale: str = "tight"             # "tight" | "loose"
    params_hash: str = ""            # Hash of view parameters
    snapshot_id: str = ""            # Immutable reference to computation

    def __hash__(self):
        return hash((self.operator, self.scale, self.params_hash, self.snapshot_id))

    def signature(self) -> str:
        """Stable signature for dedup."""
        return f"{self.operator}:{self.scale}:{self.params_hash[:8]}"


@dataclass
class MetaClaimRef:
    """
    Reference to a meta-claim that triggered this proto-inquiry.

    Keeps emission explainable - a proto-inquiry is justified by
    one or more tensions (typed conflict + single-source + missing timestamp).
    """
    type: str                        # "typed_value_conflict" | "single_source_only" | etc.
    target_id: str                   # claim_id or surface_id
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "target_id": self.target_id,
            "evidence": self.evidence,
        }


@dataclass
class ScopeSignature:
    """
    Inferred scope for a proto-inquiry.

    The scope defines what the inquiry is about without
    requiring user specification.

    scope_signature = hash(entity_senses[], time_window, view_id)
    """
    # Which projection this is about (prevents cross-view merging)
    view_id: ViewId = field(default_factory=ViewId)

    # Entities that define the referent (sense-resolved IDs)
    anchor_entities: Set[str] = field(default_factory=set)

    # All related entities
    context_entities: Set[str] = field(default_factory=set)

    # Time window
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Source diversity (informs confidence)
    sources: Set[str] = field(default_factory=set)

    # Keywords (for matching)
    keywords: List[str] = field(default_factory=list)

    def signature_key(self) -> str:
        """
        Generate a stable key for deduplication.

        Derived from sense-resolved entity IDs + time window + view ID.
        """
        anchors = sorted(self.anchor_entities)[:3]
        entity_part = "|".join(anchors)
        view_part = self.view_id.signature() if self.view_id else ""
        return f"{entity_part}@{view_part}"


@dataclass
class ProtoInquiry:
    """
    System-generated inquiry from surface + meta-claim.

    This is the epistemic object that emerges from weaving.
    It can be "adopted" by users to become a full Inquiry.

    No stakes, no deadlines, no resolution rules - those
    come when a user adopts it.
    """
    id: str = field(default_factory=lambda: f"proto_{uuid.uuid4().hex[:12]}")

    # Source (what generated this)
    surface_id: str = ""
    meta_claim_ids: List[str] = field(default_factory=list)  # Deprecated: use triggers

    # Triggers: which meta-claims justified this proto-inquiry
    # Multiple tensions can trigger one inquiry (typed conflict + single-source + missing timestamp)
    triggers: List[MetaClaimRef] = field(default_factory=list)

    # What question is being asked
    inquiry_type: ProtoInquiryType = ProtoInquiryType.VALUE_RESOLUTION
    schema_type: SchemaType = SchemaType.CATEGORICAL

    # Question formulation
    target_variable: str = ""  # e.g., "death_count", "legal_status"
    question_text: str = ""    # Human-readable question

    # Scope (includes view_id to prevent cross-projection merging)
    scope: ScopeSignature = field(default_factory=ScopeSignature)

    # Current belief state (from surface claims)
    belief_state: Optional[TypedBeliefState] = None
    observations: List[Observation] = field(default_factory=list)

    # Epistemic summary
    posterior_map: Any = None           # Current best answer
    posterior_probability: float = 0.0  # Confidence in MAP
    entropy_bits: float = 0.0           # Uncertainty
    normalized_entropy: float = 0.0     # Normalized to [0,1]

    # Competing hypotheses (for display)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    # Tensions that triggered this (human-readable summary)
    tensions: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Epistemic accounting (critical distinction)
    typed_observation_count: int = 0   # Claims with extracted_value (feed posterior)
    supporting_claim_count: int = 0    # Context claims (same topic, no value)
    source_count: int = 0

    # Reported values (for conflict detection)
    reported_values: List[Any] = field(default_factory=list)

    # Adoption status
    adopted: bool = False
    adopted_by: Optional[str] = None
    adopted_at: Optional[datetime] = None
    user_inquiry_id: Optional[str] = None

    def priority_score(self) -> float:
        """
        Score for prioritizing which proto-inquiries to surface.

        Higher = more worth investigating.
        """
        score = 0.0

        # High entropy = uncertainty = worth resolving
        score += self.normalized_entropy * 30

        # Multiple sources = high stakes
        score += min(self.source_count * 5, 25)

        # Conflicts are urgent (actual conflicting values, not noise spillover)
        if len(self.reported_values) > 1 and len(set(self.reported_values)) > 1:
            score += 25  # Real conflict

        # Single source = needs corroboration
        if self.inquiry_type == ProtoInquiryType.CORROBORATION:
            score += 15

        # More typed observations = more signal
        score += min(self.typed_observation_count * 5, 25)

        return score

    def to_dict(self) -> Dict:
        """Convert to dict for API response."""
        return {
            "id": self.id,
            "inquiry_type": self.inquiry_type.value,
            "schema_type": self.schema_type.value,
            "target_variable": self.target_variable,
            "question_text": self.question_text,
            "scope": {
                "signature": self.scope.signature_key(),
                "view_id": {
                    "operator": self.scope.view_id.operator,
                    "scale": self.scope.view_id.scale,
                    "params_hash": self.scope.view_id.params_hash,
                    "snapshot_id": self.scope.view_id.snapshot_id,
                } if self.scope.view_id else None,
                "anchor_entities": list(self.scope.anchor_entities),
                "sources": list(self.scope.sources),
                "time_start": self.scope.time_start.isoformat() if self.scope.time_start else None,
                "time_end": self.scope.time_end.isoformat() if self.scope.time_end else None,
            },
            # Triggers: why this proto-inquiry emerged
            "triggers": [t.to_dict() for t in self.triggers],
            # Epistemic state
            "posterior_map": self.posterior_map,
            "posterior_probability": round(self.posterior_probability, 4),
            "entropy_bits": round(self.entropy_bits, 2),
            "normalized_entropy": round(self.normalized_entropy, 3),
            "hypotheses": self.hypotheses[:5],  # Top 5
            # Reported values (actual conflicts, not noise spillover)
            "reported_values": self.reported_values,
            "has_conflict": len(set(self.reported_values)) > 1,
            # Epistemic accounting
            "typed_observation_count": self.typed_observation_count,
            "supporting_claim_count": self.supporting_claim_count,
            "source_count": self.source_count,
            # Meta
            "tensions": self.tensions,
            "priority_score": round(self.priority_score(), 1),
            "adopted": self.adopted,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# QUESTION KEY TEMPLATES
# =============================================================================

QUESTION_TEMPLATES = {
    # Count variables
    "death_count": {
        "schema": SchemaType.MONOTONE_COUNT,
        "template": "How many people died in {event}?",
        "variable": "death_count",
    },
    "injury_count": {
        "schema": SchemaType.MONOTONE_COUNT,
        "template": "How many people were injured in {event}?",
        "variable": "injury_count",
    },
    "casualty_count": {
        "schema": SchemaType.MONOTONE_COUNT,
        "template": "What is the total casualty count for {event}?",
        "variable": "casualty_count",
    },
    "missing_count": {
        "schema": SchemaType.MONOTONE_COUNT,
        "template": "How many people are missing from {event}?",
        "variable": "missing_count",
    },

    # Categorical variables
    "cause": {
        "schema": SchemaType.CATEGORICAL,
        "template": "What caused {event}?",
        "variable": "cause",
    },
    "responsibility": {
        "schema": SchemaType.CATEGORICAL,
        "template": "Who is responsible for {event}?",
        "variable": "responsibility",
    },
    "legal_status": {
        "schema": SchemaType.CATEGORICAL,
        "template": "What is the legal status of {subject}?",
        "variable": "legal_status",
    },

    # Boolean / verification
    "quote_said": {
        "schema": SchemaType.QUOTE_AUTHENTICITY,
        "template": "Did {person} actually say \"{quote}\"?",
        "variable": "quote_authenticity",
    },
}


# =============================================================================
# INQUIRY SEEDER
# =============================================================================

class InquirySeeder:
    """
    Seeds proto-inquiries from surfaces and meta-claims.

    This is the bridge between weaving (epistemic structure)
    and inquiry (user-facing investigation).

    The seeder does NOT require user input to operate.
    It automatically detects inquiry-worthy surfaces.
    """

    def __init__(
        self,
        params: Parameters = None,
        min_entropy_threshold: float = 0.3,
        min_conflict_confidence: float = 0.5,
    ):
        self.params = params or Parameters()
        self.min_entropy_threshold = min_entropy_threshold
        self.min_conflict_confidence = min_conflict_confidence

        # Track already-seeded to avoid duplicates
        self._seeded_surfaces: Set[str] = set()

    def seed_from_surface(
        self,
        surface: Surface,
        meta_claims: List[MetaClaim],
        claims: Dict[str, Any],  # claim_id -> Claim
        event_name: Optional[str] = None,
        view_id: Optional[ViewId] = None,
    ) -> Optional[ProtoInquiry]:
        """
        Attempt to seed a proto-inquiry from a surface.

        Returns None if the surface doesn't warrant an inquiry.

        Args:
            surface: The L2 surface to analyze
            meta_claims: Meta-claims that reference this surface
            claims: Dict of claims by ID
            event_name: Optional event name for question formulation
            view_id: Which projection this inquiry is about
        """
        if surface.id in self._seeded_surfaces:
            return None

        # Find relevant meta-claims for this surface
        relevant_mc = [
            mc for mc in meta_claims
            if mc.target_id == surface.id or mc.target_id in surface.claim_ids
        ]

        if not relevant_mc:
            # No tensions = no inquiry needed
            return None

        # Determine inquiry type from meta-claims
        inquiry_type = self._determine_inquiry_type(relevant_mc)

        # Get question key from surface claims
        question_key = self._extract_question_key(surface, claims)
        if not question_key:
            # Can't form a typed inquiry without question key
            return None

        # Build proto-inquiry
        proto = self._build_proto_inquiry(
            surface=surface,
            meta_claims=relevant_mc,
            claims=claims,
            question_key=question_key,
            inquiry_type=inquiry_type,
            event_name=event_name,
            view_id=view_id,
        )

        self._seeded_surfaces.add(surface.id)
        return proto

    def seed_from_meta_claims(
        self,
        surfaces: Dict[str, Surface],
        meta_claims: List[MetaClaim],
        claims: Dict[str, Any],
        event_names: Dict[str, str] = None,  # surface_id -> event_name
        view_id: Optional[ViewId] = None,
    ) -> List[ProtoInquiry]:
        """
        Seed proto-inquiries from a batch of meta-claims.

        This is the main entry point for batch processing.

        Key behavior: Multiple surfaces with the same question_key and overlapping
        scope are merged into a single ProtoInquiry. This is because the *question*
        is the same - different surfaces just have different observations.

        Args:
            surfaces: Dict of surfaces by ID
            meta_claims: List of meta-claims to process
            claims: Dict of claims by ID
            event_names: Optional surface_id -> event_name mapping
            view_id: Which projection these surfaces came from
        """
        event_names = event_names or {}
        protos = []

        # Group meta-claims by target surface
        mc_by_surface: Dict[str, List[MetaClaim]] = {}
        for mc in meta_claims:
            if mc.target_type == "surface":
                mc_by_surface.setdefault(mc.target_id, []).append(mc)

        # Attempt to seed from each surface with tensions
        for surface_id, surface_mcs in mc_by_surface.items():
            if surface_id not in surfaces:
                continue

            surface = surfaces[surface_id]
            proto = self.seed_from_surface(
                surface=surface,
                meta_claims=surface_mcs,
                claims=claims,
                event_name=event_names.get(surface_id),
                view_id=view_id,
            )

            if proto:
                protos.append(proto)

        # Merge protos with same question_key and overlapping scope
        protos = self._merge_related_protos(protos)

        # Sort by priority
        protos.sort(key=lambda p: p.priority_score(), reverse=True)
        return protos

    def _merge_related_protos(self, protos: List[ProtoInquiry]) -> List[ProtoInquiry]:
        """
        Merge proto-inquiries that ask the same question about the same event.

        Key insight: Multiple surfaces may contain claims about the same typed
        variable (e.g., death_count) for the same incident. These should be
        ONE inquiry with multiple observations, not multiple inquiries.

        Merge strategy:
        1. Same target_variable (e.g., death_count) = same question
        2. For now, merge all with same variable (assumes single-event context)
        3. Future: use entity overlap to distinguish different events
        """
        if len(protos) <= 1:
            return protos

        # Group by target_variable only (merge all death_count together, etc.)
        # This assumes we're processing claims from a single event context
        groups: Dict[str, List[ProtoInquiry]] = {}
        for proto in protos:
            # Simple grouping: same variable = same inquiry
            key = proto.target_variable
            groups.setdefault(key, []).append(proto)

        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge this group into one proto-inquiry
                merged.append(self._merge_proto_group(group))

        return merged

    def _merge_proto_group(self, group: List[ProtoInquiry]) -> ProtoInquiry:
        """Merge a group of related proto-inquiries into one."""
        # Use first as base
        base = group[0]

        # Collect all observations and reported values from all protos
        all_observations = []
        all_reported_values = []
        all_tensions = []
        all_triggers = []
        all_sources = set()
        total_typed = 0
        total_supporting = 0

        for proto in group:
            all_observations.extend(proto.observations)
            all_reported_values.extend(proto.reported_values)
            all_tensions.extend(proto.tensions)
            all_triggers.extend(proto.triggers)
            all_sources.update(proto.scope.sources)
            total_typed += proto.typed_observation_count
            total_supporting += proto.supporting_claim_count

        # Rebuild belief state with merged observations
        belief_state = None
        if all_observations:
            belief_state = self._build_belief_state(base.schema_type, all_observations)

        # Get hypotheses from merged state
        hypotheses = []
        if belief_state:
            posterior = belief_state.compute_posterior()
            sorted_hypos = sorted(posterior.items(), key=lambda x: -x[1])[:10]
            hypotheses = [
                {"value": h[0], "probability": round(h[1], 4)}
                for h in sorted_hypos
            ]

        # Update inquiry type based on actual reported value conflicts
        inquiry_type = base.inquiry_type
        unique_reported = set(all_reported_values)
        if len(unique_reported) > 1:
            inquiry_type = ProtoInquiryType.CONFLICT_RESOLUTION

        # Deduplicate triggers by (type, target_id)
        seen_triggers = set()
        unique_triggers = []
        for t in all_triggers:
            key = (t.type, t.target_id)
            if key not in seen_triggers:
                seen_triggers.add(key)
                unique_triggers.append(t)

        # Create merged proto
        return ProtoInquiry(
            surface_id=base.surface_id,  # Keep first surface ID
            meta_claim_ids=[mc for p in group for mc in p.meta_claim_ids],
            triggers=unique_triggers,
            inquiry_type=inquiry_type,
            schema_type=base.schema_type,
            target_variable=base.target_variable,
            question_text=base.question_text,
            scope=ScopeSignature(
                view_id=base.scope.view_id,  # Use first proto's view_id
                anchor_entities=set().union(*(p.scope.anchor_entities for p in group)),
                context_entities=set().union(*(p.scope.context_entities for p in group)),
                sources=all_sources,
                time_start=min((p.scope.time_start for p in group if p.scope.time_start), default=None),
                time_end=max((p.scope.time_end for p in group if p.scope.time_end), default=None),
            ),
            belief_state=belief_state,
            observations=all_observations,
            posterior_map=belief_state.map_value if belief_state else None,
            posterior_probability=belief_state.map_probability if belief_state else 0.0,
            entropy_bits=belief_state.entropy() if belief_state else 0.0,
            normalized_entropy=belief_state.normalized_entropy() if belief_state else 0.0,
            hypotheses=hypotheses,
            tensions=list(set(all_tensions)),  # Deduplicate
            typed_observation_count=total_typed,
            supporting_claim_count=total_supporting,
            source_count=len(all_sources),
            reported_values=all_reported_values,
        )

    def _determine_inquiry_type(self, meta_claims: List[MetaClaim]) -> ProtoInquiryType:
        """Determine inquiry type from meta-claim types."""
        mc_types = {mc.type for mc in meta_claims}

        # Typed value conflict is the most specific - prioritize it
        if "typed_value_conflict" in mc_types:
            return ProtoInquiryType.CONFLICT_RESOLUTION
        if "unresolved_conflict" in mc_types:
            return ProtoInquiryType.CONFLICT_RESOLUTION
        if "single_source_only" in mc_types:
            return ProtoInquiryType.CORROBORATION
        if "high_entropy_surface" in mc_types:
            return ProtoInquiryType.VALUE_RESOLUTION

        return ProtoInquiryType.VALUE_RESOLUTION

    def _extract_question_key(
        self,
        surface: Surface,
        claims: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extract question_key from surface claims.

        CRITICAL: Only returns a question_key if the surface contains
        at least one claim with BOTH question_key AND extracted_value.

        We do NOT infer question_key from keywords - that would create
        proto-inquiries from untyped claims, contaminating the posterior.
        """
        for claim_id in surface.claim_ids:
            claim = claims.get(claim_id)
            if claim and hasattr(claim, 'question_key') and claim.question_key:
                # Must also have extracted_value to be a typed observation
                if hasattr(claim, 'extracted_value') and claim.extracted_value is not None:
                    return claim.question_key

        # NO FALLBACK: Only typed claims generate inquiries
        # Untyped claims that mention "death" are context, not observations
        return None

    def _build_proto_inquiry(
        self,
        surface: Surface,
        meta_claims: List[MetaClaim],
        claims: Dict[str, Any],
        question_key: str,
        inquiry_type: ProtoInquiryType,
        event_name: Optional[str],
        view_id: Optional[ViewId] = None,
    ) -> ProtoInquiry:
        """Build a ProtoInquiry from components."""
        # Get template
        template = QUESTION_TEMPLATES.get(question_key, {})
        schema_type = template.get("schema", SchemaType.CATEGORICAL)
        question_template = template.get("template", f"What is the {question_key}?")

        # Build scope from surface (includes view_id for projection tracking)
        scope = ScopeSignature(
            view_id=view_id or ViewId(),
            anchor_entities=surface.anchor_entities.copy() if hasattr(surface, 'anchor_entities') else set(),
            context_entities=surface.entities.copy() if hasattr(surface, 'entities') else set(),
            sources=surface.sources.copy() if hasattr(surface, 'sources') else set(),
            time_start=surface.time_window[0] if hasattr(surface, 'time_window') else None,
            time_end=surface.time_window[1] if hasattr(surface, 'time_window') else None,
        )

        # Format question
        event_desc = event_name or self._format_event_description(scope)
        question_text = question_template.format(event=event_desc, subject=event_desc)

        # Extract observations from claims (only typed claims with extracted_value)
        observations, reported_values = self._extract_observations(surface, claims, question_key)

        # Count supporting claims (same surface, but no extracted value)
        typed_claim_ids = {obs.claim_id for obs in observations if obs.claim_id}
        supporting_claim_count = len(surface.claim_ids) - len(typed_claim_ids)

        # Build belief state
        belief_state = self._build_belief_state(schema_type, observations)

        # Get hypotheses (credible interval, not "claimed values")
        hypotheses = []
        if belief_state:
            posterior = belief_state.compute_posterior()
            sorted_hypos = sorted(posterior.items(), key=lambda x: -x[1])[:10]
            hypotheses = [
                {"value": h[0], "probability": round(h[1], 4)}
                for h in sorted_hypos
            ]

        # Build triggers from meta-claims (explainable provenance)
        triggers = [
            MetaClaimRef(
                type=mc.type,
                target_id=mc.target_id,
                evidence=mc.evidence.copy() if mc.evidence else {},
            )
            for mc in meta_claims
        ]

        # Build tensions list (human-readable summary)
        tensions = [
            f"{mc.type}: {mc.evidence.get('claim_1_text', '')[:50]}..."
            if mc.type == "unresolved_conflict"
            else f"{mc.type}"
            for mc in meta_claims
        ]

        proto = ProtoInquiry(
            surface_id=surface.id,
            meta_claim_ids=[mc.id for mc in meta_claims],
            triggers=triggers,
            inquiry_type=inquiry_type,
            schema_type=schema_type,
            target_variable=question_key,
            question_text=question_text,
            scope=scope,
            belief_state=belief_state,
            observations=observations,
            posterior_map=belief_state.map_value if belief_state else None,
            posterior_probability=belief_state.map_probability if belief_state else 0.0,
            entropy_bits=belief_state.entropy() if belief_state else 0.0,
            normalized_entropy=belief_state.normalized_entropy() if belief_state else 0.0,
            hypotheses=hypotheses,
            tensions=tensions,
            typed_observation_count=len(observations),
            supporting_claim_count=supporting_claim_count,
            source_count=len(scope.sources),
            reported_values=reported_values,
        )

        return proto

    def _format_event_description(self, scope: ScopeSignature) -> str:
        """Format event description from scope."""
        if scope.anchor_entities:
            return ", ".join(list(scope.anchor_entities)[:2])
        if scope.context_entities:
            return ", ".join(list(scope.context_entities)[:2])
        return "this incident"

    def _extract_observations(
        self,
        surface: Surface,
        claims: Dict[str, Any],
        question_key: str,
    ) -> Tuple[List[Observation], List[Any]]:
        """
        Extract observations from surface claims.

        Only claims with extracted_value are treated as observations.
        Returns (observations, reported_values) where reported_values
        are the actual values claimed by sources.

        Respects observation_kind from extraction:
        - 'point': X = value (default)
        - 'lower_bound': X >= value ("at least N")
        - 'update': X = value with update semantics
        """
        observations = []
        reported_values = []

        for claim_id in surface.claim_ids:
            claim = claims.get(claim_id)
            if not claim:
                continue

            # Only claims with extracted value count as observations
            extracted_value = getattr(claim, 'extracted_value', None)
            if extracted_value is None:
                continue

            source = getattr(claim, 'source', 'unknown')
            obs_kind = getattr(claim, 'observation_kind', 'point')
            has_update = getattr(claim, 'has_update_language', False)

            # Create appropriate observation type
            if obs_kind == 'lower_bound':
                obs = Observation.lower_bound(
                    value=extracted_value,
                    source=source,
                    claim_id=claim_id,
                )
            else:
                obs = Observation.point(
                    value=extracted_value,
                    confidence=0.8,
                    source=source,
                    claim_id=claim_id,
                )

            obs.signals['is_update'] = has_update or (obs_kind == 'update')
            observations.append(obs)
            reported_values.append(extracted_value)

        return observations, reported_values

    def _build_belief_state(
        self,
        schema_type: SchemaType,
        observations: List[Observation],
    ) -> Optional[TypedBeliefState]:
        """Build belief state from observations."""
        if not observations:
            return None

        if schema_type == SchemaType.MONOTONE_COUNT:
            from ..typed_belief import count_belief
            bs = count_belief(max_count=500, monotone=True)
        elif schema_type == SchemaType.CATEGORICAL:
            # Need categories - extract from observations
            categories = list(set(
                obs.map_value for obs in observations
                if obs.map_value is not None
            ))
            if not categories:
                return None
            from ..typed_belief import categorical_belief
            bs = categorical_belief(categories=categories)
        else:
            return None

        # Add observations
        for obs in observations:
            bs.add_observation(obs)

        return bs
