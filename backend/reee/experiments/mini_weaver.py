"""
Mini-Weaver Experiment: Incremental Organism Growth

This experiment tests the conservative, explainable, repairable growth loop
for topology formation. Claims arrive one at a time, and we trace every decision.

The Growth Loop:
0. World state: Surfaces (L2), Incidents (L3), Cases (L4)
1. Perception: Extract artifact (referents, time, proposition, confidence)
2. Candidate generation: Cheap recall of potential attachment points
3. L2 attachment: Same variable? → update posterior or DEFER/Inquiry
4. L3 routing: MERGE (spine) / PERIPHERY (metabolic) / DEFER (inquiry)
5. L4 update: Spine-only union-find, metabolic edges separate
6. Inquiry loop: Turn uncertainty into work

Key principle: High precision online, recover recall via repair.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, FrozenSet, Any, Literal, Tuple
from enum import Enum, auto
from datetime import datetime
import hashlib
import json


# =============================================================================
# World State (L2/L3/L4 existing structures)
# =============================================================================

@dataclass(frozen=True)
class ClaimGenes:
    """The 'genes' extracted from a claim - cached and replayable."""
    claim_id: str
    text: str
    referents: FrozenSet[str]  # Specific entities (identity witnesses)
    contexts: FrozenSet[str]   # Broad/ambient entities (never merge alone)
    time_point: Optional[datetime]
    proposition_key: str       # What variable this claim is about
    extracted_value: Optional[Any]  # Typed value if applicable
    confidence: float
    provenance: str            # Model/version that produced this
    hash: str                  # For caching


@dataclass
class SurfaceState:
    """L2 variable: a proposition about a scoped topic."""
    surface_id: str
    scope_id: str              # Hash of anchor entities
    proposition_key: str
    claim_ids: Set[str]
    values: List[Any]          # All extracted values
    posterior: float           # Current belief (0.0 to 1.0)
    entropy: float             # Uncertainty measure
    sources: Set[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class IncidentState:
    """L3 happening: a specific event in the world."""
    incident_id: str
    referents: FrozenSet[str]  # Identity-defining entities
    contexts: FrozenSet[str]   # Ambient but not identity
    surface_ids: Set[str]
    time_start: Optional[datetime]
    time_end: Optional[datetime]
    created_at: datetime


@dataclass
class CaseState:
    """L4 organism: spine-connected incidents + metabolism."""
    case_id: str
    incident_ids: Set[str]
    spine_edges: Set[Tuple[str, str]]  # Pairs that define membership
    metabolic_edges: Set[Tuple[str, str, str]]  # (from, to, relation)


# =============================================================================
# Decision Outcomes
# =============================================================================

class L2Action(Enum):
    """What to do with a claim at L2."""
    ATTACH_EXISTING = auto()   # Same variable, update posterior
    CREATE_NEW = auto()        # New variable
    DEFER = auto()             # Can't decide, emit inquiry


class L3Action(Enum):
    """What to do with a surface at L3."""
    MERGE = auto()             # Same happening → spine edge
    PERIPHERY = auto()         # Related but not identity → metabolic
    DEFER = auto()             # Can't decide, emit inquiry
    REJECT = auto()            # Clearly unrelated


class SignalType(Enum):
    """Epistemic signals detected during processing."""
    CONFLICT = auto()          # Values disagree
    HIGH_ENTROPY = auto()      # Too much uncertainty
    MISSING_TIME = auto()      # Time artifact absent
    MISSING_REFERENT = auto()  # No specific entities
    LOW_CONFIDENCE = auto()    # Extraction uncertain


@dataclass(frozen=True)
class InquirySeed:
    """A request for human/external evidence."""
    inquiry_id: str
    signal_type: SignalType
    subject_id: str            # What triggered this
    question: str              # Human-readable
    expected_evidence: str     # What would resolve it
    priority: float            # How urgent


@dataclass
class DecisionTrace:
    """Full audit trail for a decision."""
    decision_id: str
    claim_id: str
    phase: Literal["perception", "L2", "L3", "L4"]
    action: str
    reason: str
    features: Dict[str, Any]   # Quantitative inputs
    signals: List[SignalType]
    inquiries: List[InquirySeed]
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Mini-Weaver: The Organism
# =============================================================================

class MiniWeaver:
    """
    Conservative, incremental topology builder.

    Guarantees:
    - Every decision has a trace
    - Spine edges only from MERGE with referent witness
    - Metabolic edges never affect case membership
    - Uncertainty → InquirySeed, not bad merge
    """

    # Thresholds (tunable)
    MIN_CONFIDENCE_FOR_ATTACH = 0.6
    MIN_REFERENT_OVERLAP_FOR_MERGE = 1
    HIGH_ENTROPY_THRESHOLD = 0.8
    TIME_WINDOW_HOURS = 72  # For same-happening

    def __init__(self):
        # World state
        self.surfaces: Dict[str, SurfaceState] = {}
        self.incidents: Dict[str, IncidentState] = {}
        self.cases: Dict[str, CaseState] = {}

        # Artifact cache (genes)
        self.gene_cache: Dict[str, ClaimGenes] = {}

        # Union-find for cases
        self._case_parent: Dict[str, str] = {}

        # Audit trail
        self.traces: List[DecisionTrace] = []
        self.inquiries: List[InquirySeed] = []

        # Indices for candidate generation
        self._surface_by_scope: Dict[str, Set[str]] = {}
        self._surface_by_proposition: Dict[str, Set[str]] = {}
        self._incident_by_referent: Dict[str, Set[str]] = {}

    # -------------------------------------------------------------------------
    # Phase 1: Perception (extract genes, cached)
    # -------------------------------------------------------------------------

    def perceive(
        self,
        claim_id: str,
        text: str,
        entities: Set[str],
        anchor_entities: Set[str],
        event_time: Optional[datetime],
        source: str,
        # Pre-extracted if available
        proposition_key: Optional[str] = None,
        extracted_value: Optional[Any] = None,
        confidence: float = 0.8,
    ) -> Tuple[ClaimGenes, List[SignalType]]:
        """
        Extract or retrieve cached genes for a claim.

        Returns: (genes, signals)
        """
        # Check cache
        cache_key = hashlib.sha256(f"{claim_id}:{text}".encode()).hexdigest()[:16]
        if cache_key in self.gene_cache:
            return self.gene_cache[cache_key], []

        signals: List[SignalType] = []

        # Classify entities into referents vs contexts
        # For now: anchor_entities are referents, others are contexts
        # (In production, this would use LLM classification)
        referents = frozenset(anchor_entities)
        contexts = frozenset(entities - anchor_entities)

        # Signal if no referents
        if not referents:
            signals.append(SignalType.MISSING_REFERENT)

        # Signal if no time
        if event_time is None:
            signals.append(SignalType.MISSING_TIME)

        # Derive proposition key if not provided
        if proposition_key is None:
            # Fallback: scope + hash of text start
            scope = "_".join(sorted(anchor_entities)[:2])
            proposition_key = f"{scope}:{hashlib.md5(text[:50].encode()).hexdigest()[:8]}"

        # Signal if low confidence
        if confidence < self.MIN_CONFIDENCE_FOR_ATTACH:
            signals.append(SignalType.LOW_CONFIDENCE)

        genes = ClaimGenes(
            claim_id=claim_id,
            text=text,
            referents=referents,
            contexts=contexts,
            time_point=event_time,
            proposition_key=proposition_key,
            extracted_value=extracted_value,
            confidence=confidence,
            provenance=f"mini_weaver_v1:{source}",
            hash=cache_key,
        )

        self.gene_cache[cache_key] = genes
        return genes, signals

    # -------------------------------------------------------------------------
    # Phase 2: Candidate Generation (cheap recall)
    # -------------------------------------------------------------------------

    def find_surface_candidates(
        self,
        genes: ClaimGenes,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find existing surfaces this claim might attach to.

        Returns: [(surface_id, score), ...] sorted by score
        """
        candidates = []

        # By proposition key (exact match - highest priority)
        prop_matches = self._surface_by_proposition.get(genes.proposition_key, set())
        for sid in prop_matches:
            candidates.append((sid, 1.0))

        # By scope (entity overlap)
        scope_id = self._compute_scope_id(genes.referents)
        scope_matches = self._surface_by_scope.get(scope_id, set())
        for sid in scope_matches:
            if sid not in [c[0] for c in candidates]:
                candidates.append((sid, 0.7))

        # Sort and truncate
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def find_incident_candidates(
        self,
        genes: ClaimGenes,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find existing incidents this claim might belong to.

        Returns: [(incident_id, score), ...] sorted by score
        """
        candidates = []

        # By referent overlap
        for ref in genes.referents:
            for iid in self._incident_by_referent.get(ref, set()):
                incident = self.incidents[iid]
                overlap = len(genes.referents & incident.referents)
                total = len(genes.referents | incident.referents)
                jaccard = overlap / total if total > 0 else 0

                # Time proximity bonus
                time_bonus = 0.0
                if genes.time_point and incident.time_start:
                    hours_diff = abs((genes.time_point - incident.time_start).total_seconds() / 3600)
                    if hours_diff < self.TIME_WINDOW_HOURS:
                        time_bonus = 0.3 * (1 - hours_diff / self.TIME_WINDOW_HOURS)

                score = jaccard + time_bonus
                candidates.append((iid, score))

        # Dedupe and sort
        seen = {}
        for iid, score in candidates:
            if iid not in seen or score > seen[iid]:
                seen[iid] = score

        result = sorted(seen.items(), key=lambda x: -x[1])
        return result[:top_k]

    # -------------------------------------------------------------------------
    # Phase 3: L2 Attachment (metabolize into facet)
    # -------------------------------------------------------------------------

    def attach_l2(
        self,
        genes: ClaimGenes,
        perception_signals: List[SignalType],
    ) -> Tuple[str, L2Action, DecisionTrace]:
        """
        Decide whether to attach to existing surface or create new.

        Returns: (surface_id, action, trace)
        """
        signals = list(perception_signals)
        inquiries = []
        features = {
            "referent_count": len(genes.referents),
            "confidence": genes.confidence,
            "has_time": genes.time_point is not None,
        }

        candidates = self.find_surface_candidates(genes)

        # No candidates → create new
        if not candidates:
            surface = self._create_surface(genes)
            trace = DecisionTrace(
                decision_id=f"l2_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L2",
                action="CREATE_NEW",
                reason="no existing surface matches",
                features=features,
                signals=signals,
                inquiries=inquiries,
            )
            self.traces.append(trace)
            return surface.surface_id, L2Action.CREATE_NEW, trace

        # Check best candidate
        best_sid, best_score = candidates[0]
        best_surface = self.surfaces[best_sid]
        features["best_candidate_score"] = best_score

        # Exact proposition match → attach
        if best_surface.proposition_key == genes.proposition_key:
            # Check for conflict
            if genes.extracted_value is not None and best_surface.values:
                if genes.extracted_value not in best_surface.values:
                    signals.append(SignalType.CONFLICT)
                    # Don't defer on conflict - update and emit signal
                    inquiries.append(InquirySeed(
                        inquiry_id=f"conflict_{genes.claim_id}",
                        signal_type=SignalType.CONFLICT,
                        subject_id=best_sid,
                        question=f"Conflicting values for {best_surface.proposition_key}",
                        expected_evidence="corroborating source or correction",
                        priority=0.8,
                    ))

            self._attach_to_surface(genes, best_surface)
            trace = DecisionTrace(
                decision_id=f"l2_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L2",
                action="ATTACH_EXISTING",
                reason=f"proposition match: {genes.proposition_key}",
                features=features,
                signals=signals,
                inquiries=inquiries,
            )
            self.traces.append(trace)
            self.inquiries.extend(inquiries)
            return best_sid, L2Action.ATTACH_EXISTING, trace

        # Low confidence → defer
        if genes.confidence < self.MIN_CONFIDENCE_FOR_ATTACH:
            inquiries.append(InquirySeed(
                inquiry_id=f"lowconf_{genes.claim_id}",
                signal_type=SignalType.LOW_CONFIDENCE,
                subject_id=genes.claim_id,
                question="Extraction confidence too low",
                expected_evidence="re-extract with better model or human review",
                priority=0.5,
            ))
            # Create isolated surface but mark as deferred
            surface = self._create_surface(genes)
            trace = DecisionTrace(
                decision_id=f"l2_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L2",
                action="DEFER",
                reason="low confidence extraction",
                features=features,
                signals=signals,
                inquiries=inquiries,
            )
            self.traces.append(trace)
            self.inquiries.extend(inquiries)
            return surface.surface_id, L2Action.DEFER, trace

        # Default: create new surface
        surface = self._create_surface(genes)
        trace = DecisionTrace(
            decision_id=f"l2_{genes.claim_id}",
            claim_id=genes.claim_id,
            phase="L2",
            action="CREATE_NEW",
            reason="no exact proposition match",
            features=features,
            signals=signals,
            inquiries=inquiries,
        )
        self.traces.append(trace)
        return surface.surface_id, L2Action.CREATE_NEW, trace

    # -------------------------------------------------------------------------
    # Phase 4: L3 Routing (membrane decision)
    # -------------------------------------------------------------------------

    def route_l3(
        self,
        genes: ClaimGenes,
        surface_id: str,
        l2_trace: DecisionTrace,
    ) -> Tuple[str, L3Action, DecisionTrace]:
        """
        Decide whether surface belongs to existing incident or seeds new.

        Returns: (incident_id, action, trace)
        """
        signals = []
        inquiries = []
        features = {
            "referent_count": len(genes.referents),
            "has_time": genes.time_point is not None,
        }

        # Must have referents to merge
        if not genes.referents:
            incident = self._create_incident(genes, surface_id)
            inquiries.append(InquirySeed(
                inquiry_id=f"noref_{genes.claim_id}",
                signal_type=SignalType.MISSING_REFERENT,
                subject_id=genes.claim_id,
                question="No specific referents identified",
                expected_evidence="entity extraction or manual annotation",
                priority=0.6,
            ))
            trace = DecisionTrace(
                decision_id=f"l3_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L3",
                action="DEFER",
                reason="no referents for identity",
                features=features,
                signals=signals,
                inquiries=inquiries,
            )
            self.traces.append(trace)
            self.inquiries.extend(inquiries)
            return incident.incident_id, L3Action.DEFER, trace

        candidates = self.find_incident_candidates(genes)

        if not candidates:
            incident = self._create_incident(genes, surface_id)
            trace = DecisionTrace(
                decision_id=f"l3_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L3",
                action="CREATE_NEW",
                reason="no incident candidates",
                features=features,
                signals=signals,
                inquiries=inquiries,
            )
            self.traces.append(trace)
            return incident.incident_id, L3Action.REJECT, trace  # REJECT = seed new

        # Evaluate best candidate
        best_iid, best_score = candidates[0]
        best_incident = self.incidents[best_iid]
        features["best_candidate_score"] = best_score

        # Check referent overlap (the identity witness)
        referent_overlap = genes.referents & best_incident.referents
        features["referent_overlap"] = len(referent_overlap)

        # MERGE: requires referent witness (not just context)
        if len(referent_overlap) >= self.MIN_REFERENT_OVERLAP_FOR_MERGE:
            # Check time compatibility if both have time
            if genes.time_point and best_incident.time_start:
                hours_diff = abs((genes.time_point - best_incident.time_start).total_seconds() / 3600)
                features["time_diff_hours"] = hours_diff

                if hours_diff > self.TIME_WINDOW_HOURS:
                    # Too far apart → UPDATE_TO (sequence) not SAME_HAPPENING
                    # For now, treat as PERIPHERY
                    self._attach_to_incident(genes, surface_id, best_incident)
                    trace = DecisionTrace(
                        decision_id=f"l3_{genes.claim_id}",
                        claim_id=genes.claim_id,
                        phase="L3",
                        action="PERIPHERY",
                        reason=f"time gap too large ({hours_diff:.1f}h)",
                        features=features,
                        signals=signals,
                        inquiries=inquiries,
                    )
                    self.traces.append(trace)
                    return best_iid, L3Action.PERIPHERY, trace

            # Same happening! This is a MERGE (spine edge)
            self._attach_to_incident(genes, surface_id, best_incident)
            trace = DecisionTrace(
                decision_id=f"l3_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L3",
                action="MERGE",
                reason=f"referent witness: {referent_overlap}",
                features=features,
                signals=signals,
                inquiries=inquiries,
            )
            self.traces.append(trace)
            return best_iid, L3Action.MERGE, trace

        # Only context overlap → PERIPHERY (metabolic, not spine)
        context_overlap = genes.contexts & best_incident.contexts
        if context_overlap or (genes.contexts & best_incident.referents):
            self._attach_to_incident(genes, surface_id, best_incident)
            trace = DecisionTrace(
                decision_id=f"l3_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L3",
                action="PERIPHERY",
                reason="context overlap only (metabolic)",
                features=features,
                signals=signals,
                inquiries=inquiries,
            )
            self.traces.append(trace)
            return best_iid, L3Action.PERIPHERY, trace

        # No good match → create new incident
        incident = self._create_incident(genes, surface_id)
        trace = DecisionTrace(
            decision_id=f"l3_{genes.claim_id}",
            claim_id=genes.claim_id,
            phase="L3",
            action="REJECT",
            reason="no identity witness with candidates",
            features=features,
            signals=signals,
            inquiries=inquiries,
        )
        self.traces.append(trace)
        return incident.incident_id, L3Action.REJECT, trace

    # -------------------------------------------------------------------------
    # Phase 5: L4 Case Update (spine-only union-find)
    # -------------------------------------------------------------------------

    def update_l4(
        self,
        genes: ClaimGenes,
        incident_id: str,
        l3_action: L3Action,
        l3_trace: DecisionTrace,
    ) -> Optional[DecisionTrace]:
        """
        Update case structure based on L3 decision.

        Only MERGE creates spine edges that affect case membership.
        PERIPHERY creates metabolic edges that don't.
        """
        if l3_action == L3Action.REJECT or l3_action == L3Action.DEFER:
            # New incident, initialize in union-find
            self._case_parent[incident_id] = incident_id
            return None

        if l3_action == L3Action.MERGE:
            # Find the incident we merged into
            target_iid = l3_trace.features.get("merged_into")
            if target_iid is None:
                # This is the first claim for this incident
                self._case_parent[incident_id] = incident_id
                return None

            # Union the cases
            self._union_cases(incident_id, target_iid)

            # Record spine edge
            case_id = self._find_case(incident_id)
            if case_id not in self.cases:
                self.cases[case_id] = CaseState(
                    case_id=case_id,
                    incident_ids=set(),
                    spine_edges=set(),
                    metabolic_edges=set(),
                )
            self.cases[case_id].spine_edges.add(
                (min(incident_id, target_iid), max(incident_id, target_iid))
            )

            trace = DecisionTrace(
                decision_id=f"l4_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L4",
                action="SPINE_EDGE",
                reason=f"merged {incident_id} with {target_iid}",
                features={"case_id": case_id},
                signals=[],
                inquiries=[],
            )
            self.traces.append(trace)
            return trace

        if l3_action == L3Action.PERIPHERY:
            # Metabolic edge - does NOT affect union-find
            target_iid = incident_id  # The incident we attached to

            # Ensure both are in union-find (but not united)
            if incident_id not in self._case_parent:
                self._case_parent[incident_id] = incident_id

            # Record metabolic edge (if we have cross-incident info)
            # For now, just trace it
            trace = DecisionTrace(
                decision_id=f"l4_{genes.claim_id}",
                claim_id=genes.claim_id,
                phase="L4",
                action="METABOLIC_EDGE",
                reason="periphery attachment",
                features={"incident_id": incident_id},
                signals=[],
                inquiries=[],
            )
            self.traces.append(trace)
            return trace

        return None

    # -------------------------------------------------------------------------
    # Main Entry Point: Process a Claim
    # -------------------------------------------------------------------------

    def process_claim(
        self,
        claim_id: str,
        text: str,
        entities: Set[str],
        anchor_entities: Set[str],
        event_time: Optional[datetime] = None,
        source: str = "unknown",
        proposition_key: Optional[str] = None,
        extracted_value: Optional[Any] = None,
        confidence: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Process a single claim through the full growth loop.

        Returns summary of what happened.
        """
        # Phase 1: Perception
        genes, perception_signals = self.perceive(
            claim_id=claim_id,
            text=text,
            entities=entities,
            anchor_entities=anchor_entities,
            event_time=event_time,
            source=source,
            proposition_key=proposition_key,
            extracted_value=extracted_value,
            confidence=confidence,
        )

        # Phase 3: L2 Attachment
        surface_id, l2_action, l2_trace = self.attach_l2(genes, perception_signals)

        # Phase 4: L3 Routing
        incident_id, l3_action, l3_trace = self.route_l3(genes, surface_id, l2_trace)

        # Phase 5: L4 Update
        l4_trace = self.update_l4(genes, incident_id, l3_action, l3_trace)

        return {
            "claim_id": claim_id,
            "genes": genes,
            "surface_id": surface_id,
            "l2_action": l2_action.name,
            "incident_id": incident_id,
            "l3_action": l3_action.name,
            "case_id": self._find_case(incident_id) if incident_id in self._case_parent else None,
            "signals": perception_signals + l2_trace.signals + l3_trace.signals,
            "inquiries_emitted": len(l2_trace.inquiries) + len(l3_trace.inquiries),
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _compute_scope_id(self, referents: FrozenSet[str]) -> str:
        """Hash of sorted referents."""
        key = "|".join(sorted(referents))
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _create_surface(self, genes: ClaimGenes) -> SurfaceState:
        """Create a new surface from genes."""
        now = datetime.utcnow()
        surface_id = f"surf_{genes.hash[:8]}_{now.strftime('%H%M%S')}"
        scope_id = self._compute_scope_id(genes.referents)

        surface = SurfaceState(
            surface_id=surface_id,
            scope_id=scope_id,
            proposition_key=genes.proposition_key,
            claim_ids={genes.claim_id},
            values=[genes.extracted_value] if genes.extracted_value else [],
            posterior=genes.confidence,
            entropy=0.0,
            sources={genes.provenance},
            created_at=now,
            updated_at=now,
        )

        self.surfaces[surface_id] = surface

        # Update indices
        self._surface_by_scope.setdefault(scope_id, set()).add(surface_id)
        self._surface_by_proposition.setdefault(genes.proposition_key, set()).add(surface_id)

        return surface

    def _attach_to_surface(self, genes: ClaimGenes, surface: SurfaceState) -> None:
        """Attach a claim to an existing surface."""
        surface.claim_ids.add(genes.claim_id)
        if genes.extracted_value:
            surface.values.append(genes.extracted_value)
        surface.sources.add(genes.provenance)
        surface.updated_at = datetime.utcnow()

        # Update entropy based on value diversity
        if len(surface.values) > 1:
            unique_values = len(set(str(v) for v in surface.values))
            surface.entropy = unique_values / len(surface.values)

    def _create_incident(self, genes: ClaimGenes, surface_id: str) -> IncidentState:
        """Create a new incident from genes."""
        now = datetime.utcnow()
        incident_id = f"inc_{genes.hash[:8]}_{now.strftime('%H%M%S')}"

        incident = IncidentState(
            incident_id=incident_id,
            referents=genes.referents,
            contexts=genes.contexts,
            surface_ids={surface_id},
            time_start=genes.time_point,
            time_end=genes.time_point,
            created_at=now,
        )

        self.incidents[incident_id] = incident

        # Update indices
        for ref in genes.referents:
            self._incident_by_referent.setdefault(ref, set()).add(incident_id)

        # Initialize in union-find
        self._case_parent[incident_id] = incident_id

        return incident

    def _attach_to_incident(
        self,
        genes: ClaimGenes,
        surface_id: str,
        incident: IncidentState,
    ) -> None:
        """Attach a surface to an existing incident."""
        incident.surface_ids.add(surface_id)

        # Expand referents (union)
        incident.referents = incident.referents | genes.referents
        incident.contexts = incident.contexts | genes.contexts

        # Expand time window
        if genes.time_point:
            if incident.time_start is None:
                incident.time_start = genes.time_point
                incident.time_end = genes.time_point
            else:
                if genes.time_point < incident.time_start:
                    incident.time_start = genes.time_point
                if genes.time_point > incident.time_end:
                    incident.time_end = genes.time_point

        # Update indices
        for ref in genes.referents:
            self._incident_by_referent.setdefault(ref, set()).add(incident.incident_id)

    def _find_case(self, incident_id: str) -> str:
        """Find root of case containing incident."""
        if incident_id not in self._case_parent:
            return incident_id
        if self._case_parent[incident_id] != incident_id:
            self._case_parent[incident_id] = self._find_case(self._case_parent[incident_id])
        return self._case_parent[incident_id]

    def _union_cases(self, iid1: str, iid2: str) -> None:
        """Union two incidents into same case."""
        root1 = self._find_case(iid1)
        root2 = self._find_case(iid2)
        if root1 != root2:
            self._case_parent[root1] = root2

    # -------------------------------------------------------------------------
    # Analysis / Reporting
    # -------------------------------------------------------------------------

    def get_case_membership(self) -> Dict[str, Set[str]]:
        """Get all cases with their incident members."""
        cases = {}
        for iid in self.incidents:
            root = self._find_case(iid)
            cases.setdefault(root, set()).add(iid)
        return {k: v for k, v in cases.items() if len(v) >= 1}

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        cases = self.get_case_membership()
        return {
            "surfaces": len(self.surfaces),
            "incidents": len(self.incidents),
            "cases": len([c for c in cases.values() if len(c) >= 2]),
            "singletons": len([c for c in cases.values() if len(c) == 1]),
            "largest_case": max((len(c) for c in cases.values()), default=0),
            "traces": len(self.traces),
            "inquiries": len(self.inquiries),
            "actions": {
                "l2_attach": sum(1 for t in self.traces if t.phase == "L2" and t.action == "ATTACH_EXISTING"),
                "l2_create": sum(1 for t in self.traces if t.phase == "L2" and t.action == "CREATE_NEW"),
                "l2_defer": sum(1 for t in self.traces if t.phase == "L2" and t.action == "DEFER"),
                "l3_merge": sum(1 for t in self.traces if t.phase == "L3" and t.action == "MERGE"),
                "l3_periphery": sum(1 for t in self.traces if t.phase == "L3" and t.action == "PERIPHERY"),
                "l3_reject": sum(1 for t in self.traces if t.phase == "L3" and t.action == "REJECT"),
            },
            "signals": {
                s.name: sum(1 for t in self.traces for sig in t.signals if sig == s)
                for s in SignalType
            },
        }


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(claims: List[Dict]) -> Dict[str, Any]:
    """
    Run the mini-weaver experiment on a list of claims.

    Each claim should have:
    - id: str
    - text: str
    - entities: Set[str]
    - anchor_entities: Set[str]
    - event_time: Optional[datetime]
    - source: str
    """
    weaver = MiniWeaver()
    results = []

    for claim in claims:
        result = weaver.process_claim(
            claim_id=claim["id"],
            text=claim["text"],
            entities=set(claim.get("entities", [])),
            anchor_entities=set(claim.get("anchor_entities", [])),
            event_time=claim.get("event_time"),
            source=claim.get("source", "unknown"),
            proposition_key=claim.get("proposition_key"),
            extracted_value=claim.get("extracted_value"),
            confidence=claim.get("confidence", 0.8),
        )
        results.append(result)

    return {
        "results": results,
        "summary": weaver.summary(),
        "cases": weaver.get_case_membership(),
        "inquiries": [
            {
                "id": inq.inquiry_id,
                "type": inq.signal_type.name,
                "question": inq.question,
                "priority": inq.priority,
            }
            for inq in weaver.inquiries
        ],
    }


if __name__ == "__main__":
    # Simple test with synthetic data
    from datetime import timedelta

    base_time = datetime(2024, 11, 26, 10, 0, 0)

    test_claims = [
        {
            "id": "c1",
            "text": "Fire breaks out at Wang Fuk Court",
            "entities": {"Wang Fuk Court", "Tai Po", "Hong Kong"},
            "anchor_entities": {"Wang Fuk Court"},
            "event_time": base_time,
            "source": "reuters",
            "proposition_key": "wfc:fire_reported",
        },
        {
            "id": "c2",
            "text": "Two people confirmed dead in Wang Fuk Court fire",
            "entities": {"Wang Fuk Court", "Tai Po"},
            "anchor_entities": {"Wang Fuk Court"},
            "event_time": base_time + timedelta(hours=2),
            "source": "bbc",
            "proposition_key": "wfc:death_count",
            "extracted_value": 2,
        },
        {
            "id": "c3",
            "text": "Death toll rises to 17 in Wang Fuk Court fire",
            "entities": {"Wang Fuk Court", "Tai Po"},
            "anchor_entities": {"Wang Fuk Court"},
            "event_time": base_time + timedelta(hours=8),
            "source": "scmp",
            "proposition_key": "wfc:death_count",
            "extracted_value": 17,
        },
        {
            "id": "c4",
            "text": "Trump announces new tariffs",
            "entities": {"Donald Trump", "United States"},
            "anchor_entities": {"Donald Trump"},
            "event_time": base_time + timedelta(hours=1),
            "source": "cnn",
            "proposition_key": "trump:tariff_announcement",
        },
    ]

    result = run_experiment(test_claims)

    print("=" * 60)
    print("MINI-WEAVER EXPERIMENT RESULTS")
    print("=" * 60)
    print()
    print("Summary:")
    for k, v in result["summary"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
    print()
    print("Cases:")
    for case_id, incidents in result["cases"].items():
        print(f"  {case_id}: {len(incidents)} incidents - {incidents}")
    print()
    print("Inquiries emitted:")
    for inq in result["inquiries"]:
        print(f"  [{inq['type']}] {inq['question']}")
