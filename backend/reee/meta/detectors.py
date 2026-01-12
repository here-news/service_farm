"""
Tension Detection for Meta-Claims (Constraint-State Facts)
==========================================================

Meta-claims are observations ABOUT the epistemic state, not truth claims.
They are classified by constraint-state (see REEE1.md Section 5):

  - Incompatible constraints: typed_value_conflict, unresolved_conflict
  - Weak constraints: high_entropy_value, single_source_only
  - Missing constraints: coverage_gap, extraction_gap (future)

Additionally, the system must be SELF-AWARE about constraint availability:

  - Constraint availability: typed_coverage_zero, missing_time_high_rate, missing_semantic_signal
  - Binding diagnostics: insufficient_binding_evidence, quarantine_dominates_event

Jaynes-style rigor: don't pretend to make richer inferences than constraints allow.
If coverage(S,X)=0 for all X, say so explicitly instead of emitting only single_source_only.

They may trigger:
- ParameterChange (adjust thresholds)
- New L0 claims (verification, corroboration)
- Task generation (bounties, investigations)

INVARIANT 6: Meta-claims are never injected as world-claims.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..types import (
    Claim, Surface, Event, Relation, Parameters, MetaClaim
)


class TensionDetector:
    """
    Detects tensions in the epistemic state and emits meta-claims.
    """

    def __init__(
        self,
        claims: Dict[str, Claim],
        surfaces: Dict[str, Surface],
        edges: List[Tuple[str, str, Relation, float]],
        params: Parameters = None
    ):
        self.claims = claims
        self.surfaces = surfaces
        self.edges = edges
        self.params = params or Parameters()

    def detect_all(self) -> List[MetaClaim]:
        """
        Detect all constraint-state tensions and return meta-claims.

        Constraint-state types (see REEE1.md Section 5):
        - Incompatible: typed_value_conflict, unresolved_conflict
        - Weak (typed): high_entropy_value (H(X|E,S) > threshold) - REQUIRES typed coverage
        - Weak (geometric): high_dispersion_surface (D(surface) > threshold)
        - Missing: coverage_gap (expectedness model) - REQUIRES typed coverage
        """
        meta_claims = []

        # Check typed coverage to gate typed-dependent meta-claims
        typed_coverage = self._compute_typed_coverage()

        # Detect high-dispersion surfaces (geometric, not Jaynes)
        meta_claims.extend(self._detect_high_dispersion())

        # Detect single-source claims (weak constraint)
        meta_claims.extend(self._detect_single_source())

        # Detect unresolved conflicts (incompatible constraint)
        meta_claims.extend(self._detect_conflicts())

        # TYPED-DEPENDENT: Only emit if typed coverage > 0
        if typed_coverage > 0:
            # Detect typed value conflicts (incompatible constraint)
            meta_claims.extend(self._detect_typed_value_conflicts())

            # Detect high entropy values (Jaynes H(X|E,S) - requires TypedBeliefState)
            meta_claims.extend(self._detect_high_entropy_value())

            # Detect coverage gaps (missing constraint)
            meta_claims.extend(self._detect_coverage_gaps())

        return meta_claims

    def _compute_typed_coverage(self) -> float:
        """Compute fraction of claims with typed constraints (question_key + extracted_value)."""
        if not self.claims:
            return 0.0
        typed_count = sum(
            1 for c in self.claims.values()
            if getattr(c, 'question_key', None) and getattr(c, 'extracted_value', None) is not None
        )
        return typed_count / len(self.claims)

    def _detect_high_dispersion(self) -> List[MetaClaim]:
        """
        Detect surfaces with high semantic dispersion D(surface).

        This is GEOMETRIC (embedding variance), not Jaynes entropy.
        Indicates scope/mixing risk - the surface may contain mixed propositions.
        """
        meta_claims = []
        threshold = self.params.high_entropy_threshold  # Reuse threshold for now

        for surface in self.surfaces.values():
            # surface.entropy is actually D(surface) - semantic dispersion
            if surface.entropy > threshold:
                mc = MetaClaim(
                    type="high_dispersion_surface",
                    target_id=surface.id,
                    target_type="surface",
                    evidence={
                        'dispersion': surface.entropy,  # Renamed for clarity
                        'threshold': threshold,
                        'claim_count': len(surface.claim_ids),
                        'sources': list(surface.sources),
                        'meaning': 'Semantic dispersion (geometric) - may indicate mixed scope'
                    },
                    params_version=self.params.version
                )
                meta_claims.append(mc)

        return meta_claims

    def _detect_high_entropy_value(self) -> List[MetaClaim]:
        """
        Detect surfaces with high Jaynes entropy H(X|E,S) over typed variable.

        INVARIANT: Never emit if typed coverage is zero.

        Requires TypedBeliefState to be wired in. Currently a stub that
        will be implemented when TypedBeliefState is available.
        """
        # TODO: Wire TypedBeliefState.entropy() here
        # For now, return empty - we don't have TypedBeliefState yet
        #
        # When implemented:
        # 1. Group claims by question_key within each surface
        # 2. Build TypedBeliefState from extracted_values
        # 3. Compute H(X|E,S) = belief_state.entropy()
        # 4. Emit high_entropy_value if H > threshold
        return []

    def _detect_single_source(self) -> List[MetaClaim]:
        """Detect claims with only one source (needs corroboration)."""
        meta_claims = []

        for surface in self.surfaces.values():
            if len(surface.sources) == 1:
                mc = MetaClaim(
                    type="single_source_only",
                    target_id=surface.id,
                    target_type="surface",
                    evidence={
                        'source': list(surface.sources)[0],
                        'claim_count': len(surface.claim_ids)
                    },
                    params_version=self.params.version
                )
                meta_claims.append(mc)

        return meta_claims

    def _detect_conflicts(self) -> List[MetaClaim]:
        """Detect unresolved conflicts between claims."""
        meta_claims = []

        for c1, c2, rel, conf in self.edges:
            if rel == Relation.CONFLICTS:
                mc = MetaClaim(
                    type="unresolved_conflict",
                    target_id=f"{c1}:{c2}",
                    target_type="claim_pair",
                    evidence={
                        'claim_1': c1,
                        'claim_2': c2,
                        'confidence': conf,
                        'claim_1_text': self.claims[c1].text if c1 in self.claims else None,
                        'claim_2_text': self.claims[c2].text if c2 in self.claims else None
                    },
                    params_version=self.params.version
                )
                meta_claims.append(mc)

        return meta_claims

    def _detect_typed_value_conflicts(self) -> List[MetaClaim]:
        """
        Detect surfaces where typed claims have conflicting extracted_value.

        This catches conflicts that identity linking missed (e.g., when running
        without LLM, or when values differ but texts are similar).
        """
        meta_claims = []

        for surface in self.surfaces.values():
            # Collect typed claims in this surface
            typed_values = {}  # question_key -> {value -> [claim_ids]}
            for claim_id in surface.claim_ids:
                claim = self.claims.get(claim_id)
                if not claim:
                    continue

                qkey = getattr(claim, 'question_key', None)
                value = getattr(claim, 'extracted_value', None)

                if qkey and value is not None:
                    if qkey not in typed_values:
                        typed_values[qkey] = {}
                    if value not in typed_values[qkey]:
                        typed_values[qkey][value] = []
                    typed_values[qkey][value].append(claim_id)

            # Check for conflicts (multiple distinct values for same question_key)
            for qkey, value_map in typed_values.items():
                if len(value_map) > 1:
                    # Conflict detected!
                    values = list(value_map.keys())
                    mc = MetaClaim(
                        type="typed_value_conflict",
                        target_id=surface.id,
                        target_type="surface",
                        evidence={
                            'question_key': qkey,
                            'conflicting_values': values,
                            'claims_by_value': {
                                str(v): cids for v, cids in value_map.items()
                            },
                            'surface_sources': list(surface.sources),
                        },
                        params_version=self.params.version
                    )
                    meta_claims.append(mc)

        return meta_claims

    def _detect_coverage_gaps(self) -> List[MetaClaim]:
        """
        Detect coverage gaps using expectedness model (REEE1 Section 5.6).

        A coverage gap occurs when:
        - coverage(S,X) = 0 (no typed observation for question_key X in surface S)
        - expectedness(X|C) >= threshold (X is expected given context C)

        Simple baseline: use co-occurrence of question_keys with anchor entity types.
        If surfaces with similar anchors usually have X, but this surface doesn't,
        emit a coverage_gap meta-claim.
        """
        meta_claims = []

        # Build co-occurrence matrix: anchor_type -> {question_keys that appear}
        anchor_to_qkeys: Dict[str, set] = {}
        all_contexts = 0

        for surface in self.surfaces.values():
            surface_qkeys = set()
            for claim_id in surface.claim_ids:
                claim = self.claims.get(claim_id)
                if claim and getattr(claim, 'question_key', None):
                    surface_qkeys.add(claim.question_key)

            # Build anchor context (simplified: use anchor entities as context key)
            for anchor in surface.anchor_entities:
                if anchor not in anchor_to_qkeys:
                    anchor_to_qkeys[anchor] = set()
                anchor_to_qkeys[anchor].update(surface_qkeys)

            if surface_qkeys:
                all_contexts += 1

        # Minimum contexts to build expectations
        if all_contexts < 3:
            return meta_claims

        # For each surface, check for expected but missing question_keys
        for surface in self.surfaces.values():
            # Get observed question_keys in this surface
            observed_qkeys = set()
            for claim_id in surface.claim_ids:
                claim = self.claims.get(claim_id)
                if claim and getattr(claim, 'question_key', None):
                    observed_qkeys.add(claim.question_key)

            # Get expected question_keys based on anchors
            expected_qkeys: Dict[str, float] = {}
            for anchor in surface.anchor_entities:
                if anchor in anchor_to_qkeys:
                    for qkey in anchor_to_qkeys[anchor]:
                        if qkey not in expected_qkeys:
                            expected_qkeys[qkey] = 0
                        expected_qkeys[qkey] += 1

            # Normalize expectedness and check for gaps
            gap_threshold = 0.5  # At least half of anchors expect this qkey
            for qkey, count in expected_qkeys.items():
                if qkey not in observed_qkeys:
                    expectedness = count / len(surface.anchor_entities) if surface.anchor_entities else 0
                    if expectedness >= gap_threshold:
                        mc = MetaClaim(
                            type="coverage_gap",
                            target_id=surface.id,
                            target_type="surface",
                            evidence={
                                'missing_question_key': qkey,
                                'expectedness': round(expectedness, 2),
                                'threshold': gap_threshold,
                                'anchor_count': len(surface.anchor_entities),
                                'similar_anchors_with_qkey': count,
                            },
                            params_version=self.params.version
                        )
                        meta_claims.append(mc)

        return meta_claims


def detect_tensions(
    claims: Dict[str, Claim],
    surfaces: Dict[str, Surface],
    edges: List[Tuple[str, str, Relation, float]],
    params: Parameters = None
) -> List[MetaClaim]:
    """
    Convenience function to detect tensions.

    Returns list of meta-claims.
    """
    detector = TensionDetector(claims, surfaces, edges, params)
    return detector.detect_all()


def resolve_meta_claim(
    meta_claims: List[MetaClaim],
    meta_claim_id: str,
    resolution: str,
    actor: str = "system"
) -> Optional[MetaClaim]:
    """
    Mark a meta-claim as resolved.

    Args:
        meta_claims: List of meta-claims to search
        meta_claim_id: ID of the meta-claim to resolve
        resolution: How it was resolved
        actor: Who/what resolved it

    Returns:
        The resolved meta-claim, or None if not found
    """
    for mc in meta_claims:
        if mc.id == meta_claim_id:
            mc.resolved = True
            mc.resolution = f"{resolution} by {actor}"
            return mc
    return None


def get_unresolved(meta_claims: List[MetaClaim]) -> List[MetaClaim]:
    """Return meta-claims that haven't been resolved."""
    return [mc for mc in meta_claims if not mc.resolved]


def count_by_type(meta_claims: List[MetaClaim]) -> Dict[str, int]:
    """Count meta-claims by type."""
    counts: Dict[str, int] = {}
    for mc in meta_claims:
        counts[mc.type] = counts.get(mc.type, 0) + 1
    return counts


# =============================================================================
# VIEW-LEVEL DIAGNOSTICS (Constraint Availability + Binding)
# =============================================================================

@dataclass
class ConstraintAvailability:
    """
    Measures what constraints are AVAILABLE for inference.

    Jaynes-style rigor: if coverage(S,X)=0, you cannot compute typed uncertainty.
    The system must be explicit about this rather than pretending to diagnose.
    """
    # Typed constraint availability
    claims_with_question_key: int = 0
    claims_with_extracted_value: int = 0
    total_claims: int = 0

    # Temporal constraint availability
    surfaces_with_time: int = 0
    surfaces_without_time: int = 0

    # Semantic signal availability
    surfaces_with_centroid: int = 0
    surfaces_without_centroid: int = 0

    @property
    def typed_coverage_rate(self) -> float:
        """% of claims with typed constraints."""
        if self.total_claims == 0:
            return 0.0
        return self.claims_with_extracted_value / self.total_claims

    @property
    def time_coverage_rate(self) -> float:
        """% of surfaces with time window."""
        total = self.surfaces_with_time + self.surfaces_without_time
        if total == 0:
            return 0.0
        return self.surfaces_with_time / total

    @property
    def semantic_coverage_rate(self) -> float:
        """% of surfaces with centroid/embedding."""
        total = self.surfaces_with_centroid + self.surfaces_without_centroid
        if total == 0:
            return 0.0
        return self.surfaces_with_centroid / total


@dataclass
class BindingDiagnostics:
    """
    Measures why edges did or didn't form (gate diagnostics).
    """
    # Edge formation stats
    total_candidate_pairs: int = 0
    edges_formed: int = 0

    # Gate blocking reasons
    blocked_by_signals_met: int = 0      # signals_met < min_signals
    blocked_by_no_discriminative: int = 0  # no discriminative anchor
    blocked_by_temporal_unknown: int = 0   # time missing on one/both surfaces
    blocked_by_hub_anchor: int = 0         # shared anchor is hub

    # Membership diagnostics (per event)
    total_core: int = 0
    total_periphery: int = 0
    total_quarantine: int = 0

    @property
    def edge_formation_rate(self) -> float:
        if self.total_candidate_pairs == 0:
            return 0.0
        return self.edges_formed / self.total_candidate_pairs

    @property
    def quarantine_rate(self) -> float:
        """% of non-singleton surfaces that are quarantine."""
        total = self.total_core + self.total_periphery + self.total_quarantine
        if total == 0:
            return 0.0
        return self.total_quarantine / total


class ViewDiagnostics:
    """
    Computes constraint availability and binding diagnostics for a view.

    This makes the system "self-aware" about what constraints are missing
    and why inference is limited.
    """

    def __init__(
        self,
        surfaces: Dict[str, Surface],
        claims: Dict[str, Claim] = None,
        events: Dict[str, Event] = None,
        view_trace: Any = None,  # ViewTrace from incident/case view
    ):
        self.surfaces = surfaces
        self.claims = claims or {}
        self.events = events or {}
        self.view_trace = view_trace

    def compute_constraint_availability(self) -> ConstraintAvailability:
        """Compute what constraints are available for inference."""
        avail = ConstraintAvailability()

        # Typed constraint availability
        avail.total_claims = len(self.claims)
        for claim in self.claims.values():
            if getattr(claim, 'question_key', None):
                avail.claims_with_question_key += 1
            if getattr(claim, 'extracted_value', None) is not None:
                avail.claims_with_extracted_value += 1

        # Temporal constraint availability
        for surface in self.surfaces.values():
            tw = surface.time_window
            if tw and (tw[0] is not None or tw[1] is not None):
                avail.surfaces_with_time += 1
            else:
                avail.surfaces_without_time += 1

        # Semantic signal availability
        for surface in self.surfaces.values():
            if surface.centroid is not None:
                avail.surfaces_with_centroid += 1
            else:
                avail.surfaces_without_centroid += 1

        return avail

    def compute_binding_diagnostics(self) -> BindingDiagnostics:
        """Compute binding diagnostics from view trace and events."""
        diag = BindingDiagnostics()

        # From view trace (if available)
        if self.view_trace:
            gates = getattr(self.view_trace, 'gates_hit', {})
            diag.blocked_by_signals_met = gates.get('signals_met < min', 0)
            diag.blocked_by_no_discriminative = gates.get('no_discriminative', 0)
            diag.blocked_by_temporal_unknown = gates.get('temporal_unknown', 0)
            diag.edges_formed = getattr(self.view_trace, 'edges_computed', 0)

            # Estimate candidate pairs (n choose 2)
            n = len(self.surfaces)
            diag.total_candidate_pairs = n * (n - 1) // 2

        # From events (membership counts)
        for event in self.events.values():
            for sid, membership in event.memberships.items():
                level = membership.level.value
                if level == 'core':
                    diag.total_core += 1
                elif level == 'periphery':
                    diag.total_periphery += 1
                elif level == 'quarantine':
                    diag.total_quarantine += 1

        return diag

    def emit_meta_claims(self) -> List[MetaClaim]:
        """
        Emit meta-claims about constraint availability and binding.

        These are view-level diagnostics, not surface-level tensions.
        """
        meta_claims = []
        avail = self.compute_constraint_availability()
        binding = self.compute_binding_diagnostics()

        # Constraint availability meta-claims
        if avail.typed_coverage_rate < 0.05:  # <5% typed
            mc = MetaClaim(
                type="typed_coverage_zero",
                target_id="view",
                target_type="view",
                evidence={
                    'claims_with_question_key': avail.claims_with_question_key,
                    'claims_with_extracted_value': avail.claims_with_extracted_value,
                    'total_claims': avail.total_claims,
                    'typed_coverage_rate': round(avail.typed_coverage_rate, 3),
                    'implication': 'Cannot compute typed_value_conflict or coverage_gap without typed claims',
                }
            )
            meta_claims.append(mc)

        if avail.time_coverage_rate < 0.5:  # <50% have time
            mc = MetaClaim(
                type="missing_time_high_rate",
                target_id="view",
                target_type="view",
                evidence={
                    'surfaces_with_time': avail.surfaces_with_time,
                    'surfaces_without_time': avail.surfaces_without_time,
                    'time_coverage_rate': round(avail.time_coverage_rate, 3),
                    'implication': 'Cannot do reliable temporal binding; many surfaces lack timestamps',
                }
            )
            meta_claims.append(mc)

        if avail.semantic_coverage_rate < 0.1:  # <10% have centroids
            mc = MetaClaim(
                type="missing_semantic_signal",
                target_id="view",
                target_type="view",
                evidence={
                    'surfaces_with_centroid': avail.surfaces_with_centroid,
                    'surfaces_without_centroid': avail.surfaces_without_centroid,
                    'semantic_coverage_rate': round(avail.semantic_coverage_rate, 3),
                    'implication': 'Cannot compute semantic similarity signal; relying on anchor overlap only',
                }
            )
            meta_claims.append(mc)

        # Binding diagnostics meta-claims
        if binding.edge_formation_rate < 0.01:  # <1% edges formed
            mc = MetaClaim(
                type="insufficient_binding_evidence",
                target_id="view",
                target_type="view",
                evidence={
                    'edges_formed': binding.edges_formed,
                    'total_candidate_pairs': binding.total_candidate_pairs,
                    'edge_formation_rate': round(binding.edge_formation_rate, 4),
                    'blocked_by_signals_met': binding.blocked_by_signals_met,
                    'blocked_by_no_discriminative': binding.blocked_by_no_discriminative,
                    'blocked_by_temporal_unknown': binding.blocked_by_temporal_unknown,
                    'implication': 'Most candidate edges blocked by gates; consider relaxing thresholds or adding constraints',
                }
            )
            meta_claims.append(mc)

        if binding.quarantine_rate > 0.3:  # >30% quarantine
            mc = MetaClaim(
                type="quarantine_dominates_event",
                target_id="view",
                target_type="view",
                evidence={
                    'total_core': binding.total_core,
                    'total_periphery': binding.total_periphery,
                    'total_quarantine': binding.total_quarantine,
                    'quarantine_rate': round(binding.quarantine_rate, 3),
                    'implication': 'Events held together by weak anchor-only attachment; need stronger binding evidence',
                }
            )
            meta_claims.append(mc)

        return meta_claims

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary dict for reporting."""
        avail = self.compute_constraint_availability()
        binding = self.compute_binding_diagnostics()

        return {
            'constraint_availability': {
                'typed_coverage_rate': round(avail.typed_coverage_rate, 3),
                'time_coverage_rate': round(avail.time_coverage_rate, 3),
                'semantic_coverage_rate': round(avail.semantic_coverage_rate, 3),
                'total_claims': avail.total_claims,
                'claims_with_question_key': avail.claims_with_question_key,
                'surfaces_without_time': avail.surfaces_without_time,
                'surfaces_without_centroid': avail.surfaces_without_centroid,
            },
            'binding_diagnostics': {
                'total_candidate_pairs': binding.total_candidate_pairs,
                'edge_formation_rate': round(binding.edge_formation_rate, 4),
                'edges_formed': binding.edges_formed,
                'blocked_by_signals_met': binding.blocked_by_signals_met,
                'blocked_by_no_discriminative': binding.blocked_by_no_discriminative,
                'blocked_by_temporal_unknown': binding.blocked_by_temporal_unknown,
                'core': binding.total_core,
                'periphery': binding.total_periphery,
                'quarantine': binding.total_quarantine,
                'quarantine_rate': round(binding.quarantine_rate, 3),
            }
        }
