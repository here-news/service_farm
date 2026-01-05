"""
Golden Trace Replay Runner
==========================

Replays golden trace fixtures through the REEE kernel and validates:
1. Surface partition matches expected
2. Event partition matches expected
3. Invariants hold (no core merge on semantic-only, etc.)
4. Meta-claims are emitted as expected
5. Typed posteriors converge correctly

Also provides streaming visualization endpoint for debugging.

Usage:
    # Run tests
    pytest backend/reee/tests/test_golden_trace.py -v

    # Run with streaming viz
    python -m reee.tests.test_golden_trace --viz
"""

import os
import sys
import yaml
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    pytest = None

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import Claim, Surface, Event, Constraint, ConstraintType, ConstraintLedger
from reee.builders import PrincipledSurfaceBuilder, MotifConfig, context_compatible


# ============================================================================
# TRACE LOADER
# ============================================================================

@dataclass
class TraceEntity:
    id: str
    canonical: str
    type: str
    role: str = "unknown"
    note: Optional[str] = None
    authority: Optional[float] = None


@dataclass
class TraceClaim:
    id: str
    publisher: str
    reported_time: Optional[datetime]
    event_time: Optional[datetime]
    anchor_entities: List[str]
    entities: List[str]
    roles: Dict[str, List[str]]
    question_key: str  # L2 identity key
    typed_observation: Optional[Dict]
    gist: str
    noise_type: Optional[str] = None  # adversarial_numeric, copy_noise, time_noise, extraction_sparsity
    syndicated_from: Optional[str] = None  # Original publisher for syndicated content


@dataclass
class TraceExpectedDelta:
    after_claim: str
    surfaces: Dict[str, Dict]
    events: Dict[str, Dict]
    typed_state: Optional[Dict]
    explanation: Dict


@dataclass
class TraceAssertion:
    type: str
    params: Dict


@dataclass
class GoldenTrace:
    """Loaded golden trace fixture."""
    name: str
    description: str
    version: str
    invariants: List[str]
    entities: Dict[str, TraceEntity]
    gists: Dict[str, str]
    claims: List[TraceClaim]
    expected_deltas: List[TraceExpectedDelta]
    assertions: List[TraceAssertion]

    @classmethod
    def load(cls, path: Path) -> "GoldenTrace":
        """Load trace from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        meta = data.get("meta", {})
        entities = {
            k: TraceEntity(id=k, **v)
            for k, v in data.get("entities", {}).items()
        }
        gists = data.get("gists", {})

        claims = []
        for c in data.get("claims", []):
            # Handle timestamps - may be missing or null
            reported_time = None
            if c.get("reported_time"):
                reported_time = datetime.fromisoformat(c["reported_time"].replace("Z", "+00:00"))

            event_time = None
            if c.get("event_time"):
                event_time = datetime.fromisoformat(c["event_time"].replace("Z", "+00:00"))

            # Get gist from inline or gists dict
            gist = c.get("gist") or gists.get(c.get("gist_key", c["id"]), "")

            claims.append(TraceClaim(
                id=c["id"],
                publisher=c["publisher"],
                reported_time=reported_time,
                event_time=event_time,
                anchor_entities=c.get("anchor_entities", []),
                entities=c.get("entities", []),
                roles=c.get("roles", {}),
                question_key=c.get("question_key", "unknown"),
                typed_observation=c.get("typed_observation"),
                gist=gist,
                noise_type=c.get("noise_type"),
                syndicated_from=c.get("syndicated_from"),
            ))

        expected_deltas = []
        for d in data.get("expected_deltas", []):
            expected_deltas.append(TraceExpectedDelta(
                after_claim=d["after_claim"],
                surfaces=d.get("surfaces", {}),
                events=d.get("events", {}),
                typed_state=d.get("typed_state"),
                explanation=d.get("explanation", {}),
            ))

        assertions = []
        for a in data.get("assertions", []):
            assertions.append(TraceAssertion(
                type=a["type"],
                params={k: v for k, v in a.items() if k != "type"}
            ))

        return cls(
            name=meta.get("name", "unknown"),
            description=meta.get("description", ""),
            version=meta.get("version", "1.0"),
            invariants=meta.get("invariants", []),
            entities=entities,
            gists=gists,
            claims=claims,
            expected_deltas=expected_deltas,
            assertions=assertions,
        )


# ============================================================================
# EXPLANATION BUNDLE
# ============================================================================

@dataclass
class IdentityDecision:
    """Explains surface join/create decision."""
    action: str  # "new_surface" or "attach"
    surface_id: str
    constraints_used: List[Dict] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ContextCheckResult:
    """Result of context compatibility check."""
    entity: str
    s1_companions: Set[str]
    s2_companions: Set[str]
    overlap: float
    result: str  # "compatible", "incompatible", "underpowered"


@dataclass
class MetaClaim:
    """Meta-claim emitted during processing."""
    type: str
    entity: Optional[str] = None
    surface: Optional[str] = None
    between: Optional[Tuple[str, str]] = None
    reason: str = ""
    evidence: Dict = field(default_factory=dict)


@dataclass
class StepExplanation:
    """Full explanation bundle for one claim processing step."""
    claim_id: str
    identity_decision: IdentityDecision
    context_checks: List[ContextCheckResult] = field(default_factory=list)
    meta_claims: List[MetaClaim] = field(default_factory=list)
    typed_update: Optional[Dict] = None
    relation: Optional[str] = None  # CONFIRMS, REFINES, SUPERSEDES, CONFLICTS


# ============================================================================
# TRACE KERNEL (simplified REEE for testing)
# ============================================================================

@dataclass
class L3Incident:
    """L3 Incident: membrane over L2 surfaces."""
    id: str
    surface_ids: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)


class TraceKernel:
    """
    Minimal REEE kernel for trace replay.

    Implements principled L2 + L3 architecture:
    - L2 Surface = (scope_id, question_key) - SCOPED propositions
    - L3 Incident = membrane over surfaces (tight time/motif/context)

    CRITICAL INVARIANT (Transitional Option A'):
    - Surface key = (scope_id, question_key), NOT question_key alone
    - scope_id derived from claim's anchor entities
    - Same question_key with different referents = DIFFERENT surfaces
    - This prevents cross-event contamination via generic predicates

    No database dependencies.
    """

    def __init__(self, trace: GoldenTrace):
        self.trace = trace

        # L2 State: Surfaces by (scope_id, question_key)
        self.claims: Dict[str, Claim] = {}
        self.surfaces: Dict[str, Surface] = {}
        self.surface_question_key: Dict[str, str] = {}  # surface_id -> question_key
        self.surface_scope: Dict[str, str] = {}  # surface_id -> scope_id
        # REFACTORED: Key by (scope_id, question_key) tuple, not question_key alone
        self.scoped_key_to_surface: Dict[Tuple[str, str], str] = {}

        # L3 State: Incidents
        self.incidents: Dict[str, L3Incident] = {}
        self.surface_to_incident: Dict[str, str] = {}

        # Indices
        self.claim_to_surface: Dict[str, str] = {}
        self.entity_to_claims: Dict[str, Set[str]] = defaultdict(set)
        self.entity_to_surfaces: Dict[str, Set[str]] = defaultdict(set)

        # Meta-claims
        self.meta_claims: List[MetaClaim] = []

        # Typed posteriors per surface
        self.typed_posteriors: Dict[str, Dict[str, Dict]] = defaultdict(dict)

        # Config
        self.min_motif_k = 2
        self.companion_overlap_threshold = 0.15
        self.ledger = ConstraintLedger()

        # Counters
        self.surface_counter = 0
        self.incident_counter = 0

        # Hub entities to suppress from scope computation
        self.hub_entities = {"United States", "China", "European Union", "United Nations", "Asia", "Europe"}

    def _entity_id_to_name(self, entity_id: str) -> str:
        """Convert entity ID (E001) to canonical name."""
        if entity_id in self.trace.entities:
            return self.trace.entities[entity_id].canonical
        return entity_id

    def _resolve_entities(self, entity_ids: List[str]) -> Set[str]:
        """Convert entity IDs to canonical names."""
        return {self._entity_id_to_name(eid) for eid in entity_ids}

    def _compute_scope_id(self, anchor_entities: Set[str]) -> str:
        """
        Compute scope identifier from anchor entities.

        CRITICAL: This determines surface identity scope.
        - Filters out hub entities (too generic to scope by)
        - Hashes sorted remaining anchors
        - Returns deterministic scope_id

        Example:
        - {John Lee, Hong Kong} -> "scope_johnlee_hongkong"
        - {Gavin Newsom, California} -> "scope_gavinnewsom_california"
        """
        # Filter out hub entities that are too generic
        scoping_anchors = anchor_entities - self.hub_entities

        if not scoping_anchors:
            # Fallback: all anchors are hubs, use them anyway but emit warning
            scoping_anchors = anchor_entities
            if anchor_entities:
                self.meta_claims.append(MetaClaim(
                    type="scope_underpowered",
                    reason=f"All anchor entities are hubs: {sorted(anchor_entities)}",
                    evidence={"anchor_entities": sorted(anchor_entities)}
                ))

        # Create deterministic scope_id from sorted anchors
        # Normalize: lowercase, remove spaces, sort
        normalized = sorted(a.lower().replace(" ", "") for a in scoping_anchors)
        scope_id = "scope_" + "_".join(normalized[:3])  # Limit to top 3 for readability

        return scope_id

    def process_claim(self, trace_claim: TraceClaim) -> StepExplanation:
        """
        Process one claim and return explanation bundle.

        L2: Route by question_key (identity)
        L3: Group surfaces by motif/context (incident formation)
        """
        # Convert to REEE Claim
        entities = self._resolve_entities(trace_claim.entities)
        anchor_entities = self._resolve_entities(trace_claim.anchor_entities)
        question_key = trace_claim.question_key

        claim = Claim(
            id=trace_claim.id,
            text=trace_claim.gist,
            source=trace_claim.publisher,
            entities=entities,
            anchor_entities=anchor_entities,
            event_time=trace_claim.event_time,
            timestamp=trace_claim.reported_time,
        )

        self.claims[claim.id] = claim

        # =====================================================================
        # NOISE CHECKS: Emit blockers/warnings for problematic claims
        # =====================================================================

        # Check for missing timestamp
        if trace_claim.event_time is None and trace_claim.reported_time is None:
            self.meta_claims.append(MetaClaim(
                type="missing_time",
                entity=None,
                surface=None,
                reason="Timestamp unavailable - monotone constraints not applied",
                evidence={"claim_id": claim.id, "consequence": "temporal_ordering_blocked"}
            ))

        # Check for extraction sparsity (too few entities for context check)
        if len(entities) < 2:
            self.meta_claims.append(MetaClaim(
                type="context_underpowered",
                entity=None,
                surface=None,
                reason=f"Insufficient entities for context check ({len(entities)} < 2)",
                evidence={
                    "claim_id": claim.id,
                    "entity_count": len(entities),
                    "minimum_required": 2,
                }
            ))

        # Check for copy noise (syndicated content)
        if trace_claim.syndicated_from:
            self.meta_claims.append(MetaClaim(
                type="syndication_detected",
                entity=None,
                surface=None,
                reason=f"Syndicated from {trace_claim.syndicated_from} - effective weight reduced",
                evidence={
                    "claim_id": claim.id,
                    "original_publisher": trace_claim.syndicated_from,
                    "republisher": trace_claim.publisher,
                }
            ))

        # Update entity indices
        for entity in entities:
            self.entity_to_claims[entity].add(claim.id)

        # =====================================================================
        # L2: Surface by (scope_id, question_key) - SCOPED identity routing
        # =====================================================================
        # CRITICAL INVARIANT: Same question_key with different referents
        # must produce DIFFERENT surfaces. This prevents cross-event contamination.

        # Step 1: Compute scope_id from anchor entities
        scope_id = self._compute_scope_id(anchor_entities)

        # Step 2: Build scoped key = (scope_id, question_key)
        scoped_key = (scope_id, question_key)

        if scoped_key in self.scoped_key_to_surface:
            # Attach to existing surface for this (scope, question_key)
            surface_id = self.scoped_key_to_surface[scoped_key]
            surface = self.surfaces[surface_id]

            surface.claim_ids.add(claim.id)
            surface.entities.update(entities)
            surface.anchor_entities.update(anchor_entities)
            if claim.source:
                surface.sources.add(claim.source)

            self.claim_to_surface[claim.id] = surface_id

            # Determine relation type
            relation = self._determine_relation(claim, surface, trace_claim.typed_observation)

            # Process typed observation
            typed_update = None
            if trace_claim.typed_observation:
                typed_update = self._update_typed_posterior(
                    surface_id,
                    trace_claim.typed_observation,
                    claim.id
                )

            l2_decision = IdentityDecision(
                action="attach",
                surface_id=surface_id,
                constraints_used=[
                    {"type": "scoped_key", "scope": scope_id, "question_key": question_key}
                ],
                reason=f"Same scoped key: ({scope_id}, {question_key})"
            )

            # L3: Surface already in an incident, claim joins that incident
            incident_id = self.surface_to_incident.get(surface_id)

            return StepExplanation(
                claim_id=claim.id,
                identity_decision=l2_decision,
                relation=relation,
                typed_update=typed_update,
                meta_claims=self._collect_meta_claims_for_step(claim.id),
            )

        else:
            # Create new L2 surface for this (scope_id, question_key)
            self.surface_counter += 1
            # Surface ID includes scope for debuggability
            surface_id = f"S_{scope_id}_{question_key}"

            surface = Surface(
                id=surface_id,
                claim_ids={claim.id},
                entities=entities,
                anchor_entities=anchor_entities,
                sources={claim.source} if claim.source else set(),
                formation_method="scoped_question_key",
            )

            self.surfaces[surface_id] = surface
            self.claim_to_surface[claim.id] = surface_id
            self.surface_question_key[surface_id] = question_key
            self.surface_scope[surface_id] = scope_id
            # CRITICAL: Use scoped key, not global question_key
            self.scoped_key_to_surface[scoped_key] = surface_id

            # Update entity-surface index
            for entity in entities:
                self.entity_to_surfaces[entity].add(surface_id)

            # Process typed observation
            typed_update = None
            if trace_claim.typed_observation:
                typed_update = self._update_typed_posterior(
                    surface_id,
                    trace_claim.typed_observation,
                    claim.id
                )

            l2_decision = IdentityDecision(
                action="new_surface",
                surface_id=surface_id,
                constraints_used=[
                    {"type": "scoped_key", "scope": scope_id, "question_key": question_key}
                ],
                reason=f"New scoped key: ({scope_id}, {question_key})"
            )

            # =================================================================
            # L3: Incident membrane formation
            # =================================================================
            l3_decision = self._process_incident_membrane(surface_id, anchor_entities, entities)

            return StepExplanation(
                claim_id=claim.id,
                identity_decision=l2_decision,
                typed_update=typed_update,
                meta_claims=self._collect_meta_claims_for_step(claim.id),
            )

    def _process_incident_membrane(
        self,
        surface_id: str,
        anchor_entities: Set[str],
        entities: Set[str]
    ) -> Dict:
        """
        L3: Process incident membrane for a new surface.

        Decides whether to:
        1. Create new incident
        2. Attach to existing incident (if context compatible)
        3. Block bridge (if disjoint companions)
        """
        surface = self.surfaces[surface_id]

        # Find candidate incidents (share anchor entities)
        candidates = []
        for inc_id, incident in self.incidents.items():
            shared_anchors = anchor_entities & incident.anchor_entities

            if len(shared_anchors) >= self.min_motif_k:
                # Check context compatibility
                surface_companions = entities - shared_anchors
                incident_companions = incident.entities - shared_anchors

                if surface_companions and incident_companions:
                    intersection = surface_companions & incident_companions
                    union = surface_companions | incident_companions
                    overlap = len(intersection) / len(union) if union else 0.0

                    if overlap >= self.companion_overlap_threshold:
                        candidates.append((inc_id, overlap, shared_anchors))
                    else:
                        # Bridge blocked - emit detailed meta-claim
                        # Identify which shared entity failed context check
                        blocking_entity = list(shared_anchors)[0] if shared_anchors else None
                        self.meta_claims.append(MetaClaim(
                            type="bridge_blocked",
                            entity=blocking_entity,
                            between=(surface_id, inc_id),
                            reason=f"{blocking_entity} does not bind {surface_id}↔{inc_id} because ctx({blocking_entity}|{surface_id}) and ctx({blocking_entity}|{inc_id}) are incompatible",
                            evidence={
                                "blocking_entity": blocking_entity,
                                "surface_companions": sorted(list(surface_companions)[:5]),  # Top 5 for display
                                "incident_companions": sorted(list(incident_companions)[:5]),
                                "overlap": overlap,
                                "threshold": self.companion_overlap_threshold,
                                "shared_anchors": sorted(list(shared_anchors)),
                            }
                        ))

        if candidates:
            # Attach to best matching incident
            candidates.sort(key=lambda x: -x[1])
            best_inc_id, overlap, shared = candidates[0]
            incident = self.incidents[best_inc_id]

            incident.surface_ids.add(surface_id)
            incident.entities.update(entities)
            self.surface_to_incident[surface_id] = best_inc_id

            return {
                "action": "attach_to_incident",
                "incident_id": best_inc_id,
                "shared_anchors": list(shared),
                "overlap": overlap
            }

        else:
            # Create new incident
            self.incident_counter += 1
            incident_id = f"I_{self.incident_counter:03d}"

            incident = L3Incident(
                id=incident_id,
                surface_ids={surface_id},
                anchor_entities=anchor_entities.copy(),
                entities=entities.copy(),
            )

            self.incidents[incident_id] = incident
            self.surface_to_incident[surface_id] = incident_id

            return {
                "action": "new_incident",
                "incident_id": incident_id,
                "anchor_entities": list(anchor_entities)
            }

    def _find_candidate_surfaces(self, claim: Claim) -> List[Tuple[str, float, List[Dict]]]:
        """
        Find surfaces that share a motif (k >= 2 entities) with claim.

        Uses ANCHOR entity overlap as primary signal - this prevents
        hub entities from bridging unrelated incidents.

        Returns list of (surface_id, score, constraints_used).
        """
        candidates = []

        for sid, surface in self.surfaces.items():
            shared = claim.entities & surface.entities
            shared_anchors = claim.anchor_entities & surface.anchor_entities

            # PRIMARY CHECK: Anchor overlap (most restrictive)
            # Anchors represent the core identity of the claim (where + when)
            if len(shared_anchors) >= self.min_motif_k:
                # Strong match - anchors align
                score = len(shared_anchors) * 3 + len(shared)
                constraints = [{
                    "type": "anchor_motif",
                    "shared_anchors": list(shared_anchors),
                    "shared_entities": list(shared),
                    "k": len(shared_anchors),
                }]
                candidates.append((sid, score, constraints))

            elif len(shared) >= self.min_motif_k:
                # SECONDARY CHECK: Context compatibility
                # Shared entities must have overlapping companion sets
                # This prevents hub entities (Hong Kong, John Lee) from bridging

                # Build companion sets for shared entities
                claim_companions = claim.entities - shared
                surface_companions = surface.entities - shared

                # Jaccard overlap of companions
                if claim_companions and surface_companions:
                    intersection = claim_companions & surface_companions
                    union = claim_companions | surface_companions
                    overlap = len(intersection) / len(union) if union else 0.0

                    if overlap >= 0.15:
                        # Compatible contexts
                        score = len(shared) + overlap * 2
                        constraints = [{
                            "type": "motif_overlap",
                            "shared_entities": list(shared),
                            "companion_overlap": overlap,
                            "k": len(shared),
                        }]
                        candidates.append((sid, score, constraints))
                    else:
                        # Incompatible - emit blocked meta-claim
                        self.meta_claims.append(MetaClaim(
                            type="bridge_blocked",
                            surface=sid,
                            reason=f"Disjoint companions: claim has {claim_companions}, surface has {surface_companions}",
                            evidence={"overlap": overlap, "shared": list(shared)}
                        ))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[1])
        return candidates

    def _determine_relation(
        self,
        claim: Claim,
        surface: Surface,
        typed_obs: Optional[Dict]
    ) -> str:
        """Determine relation type (CONFIRMS, REFINES, SUPERSEDES, CONFLICTS)."""
        if not typed_obs:
            return "CONFIRMS"

        question = typed_obs.get("question")
        value = typed_obs.get("value")

        if question not in self.typed_posteriors.get(surface.id, {}):
            return "CONFIRMS"

        posterior = self.typed_posteriors[surface.id][question]
        current_mean = posterior.get("posterior_mean", 0)

        # Check for conflict
        if abs(value - current_mean) > current_mean * 0.5:
            confidence = typed_obs.get("confidence", 0.5)
            if confidence < 0.5:
                # Low-confidence conflict
                self.meta_claims.append(MetaClaim(
                    type="typed_value_conflict",
                    surface=surface.id,
                    reason=f"{question}: {value} vs consensus {current_mean:.1f}",
                    evidence={"conflicting_value": value, "consensus": current_mean}
                ))
                return "CONFLICTS"

        # Check for supersession (official source)
        if typed_obs.get("modifier") == "confirmed":
            return "SUPERSEDES"

        # Default to refinement
        return "REFINES"

    def _update_typed_posterior(
        self,
        surface_id: str,
        typed_obs: Dict,
        claim_id: str,
        trace_claim: Optional['TraceClaim'] = None
    ) -> Dict:
        """
        Update typed belief state for observation.

        Handles both numeric (Bayesian posterior) and categorical values.
        Detects outliers and emits appropriate meta-claims.
        Returns update summary.
        """
        value = typed_obs.get("value")
        confidence = typed_obs.get("confidence", 0.5)
        authority = typed_obs.get("source_authority", 0.5)

        if surface_id not in self.typed_posteriors:
            self.typed_posteriors[surface_id] = {}

        # Use question_key from surface (already set during L2 routing)
        question = self.surface_question_key.get(surface_id, "unknown")

        is_numeric = isinstance(value, (int, float))
        weight = confidence * authority

        if question not in self.typed_posteriors[surface_id]:
            # Initialize belief state
            if is_numeric:
                self.typed_posteriors[surface_id][question] = {
                    "type": "numeric",
                    "posterior_mean": float(value),
                    "posterior_std": 5.0,
                    "observations": 1,
                    "values": [(value, weight, claim_id)],
                    "outliers": [],
                }
            else:
                # Categorical: track value distribution
                self.typed_posteriors[surface_id][question] = {
                    "type": "categorical",
                    "current_value": value,
                    "observations": 1,
                    "values": [(value, weight, claim_id)],
                    "distribution": {value: weight},
                }
        else:
            posterior = self.typed_posteriors[surface_id][question]

            # Check for outlier BEFORE adding to values
            is_outlier = False
            if posterior.get("type") == "numeric" and is_numeric and len(posterior["values"]) >= 2:
                existing_values = [v for v, _, _ in posterior["values"]]
                existing_mean = sum(existing_values) / len(existing_values)
                existing_std = (sum((v - existing_mean) ** 2 for v in existing_values) / len(existing_values)) ** 0.5
                existing_std = max(existing_std, 1.0)  # Minimum std to avoid division issues

                # Outlier if > 3 std deviations from mean
                z_score = abs(value - existing_mean) / existing_std
                if z_score > 3:
                    is_outlier = True
                    posterior.setdefault("outliers", []).append({
                        "value": value,
                        "claim_id": claim_id,
                        "z_score": z_score,
                        "authority": authority,
                    })

                    # Emit conflict meta-claim for outlier
                    self.meta_claims.append(MetaClaim(
                        type="typed_conflict",
                        surface=surface_id,
                        reason=f"Outlier detected: {value} vs consensus {existing_mean:.1f} (z={z_score:.1f})",
                        evidence={
                            "outlier_value": value,
                            "outlier_claim": claim_id,
                            "consensus_mean": existing_mean,
                            "consensus_n": len(existing_values),
                            "z_score": z_score,
                            "authority": authority,
                        }
                    ))

            # Add observation
            posterior["values"].append((value, weight, claim_id))
            posterior["observations"] += 1

            if posterior.get("type") == "numeric" and is_numeric:
                # Bayesian update (simplified weighted average)
                total_weight = sum(w for _, w, _ in posterior["values"])
                if total_weight > 0:
                    new_mean = sum(v * w for v, w, _ in posterior["values"]) / total_weight
                    posterior["posterior_mean"] = new_mean

                # Check for general conflict (not just outliers)
                if len(posterior["values"]) > 1:
                    values = [v for v, _, _ in posterior["values"]]
                    spread = max(values) - min(values)
                    if spread > posterior["posterior_mean"] * 0.3:
                        posterior["conflict_detected"] = True
            else:
                # Categorical update
                if value not in posterior.get("distribution", {}):
                    posterior.setdefault("distribution", {})[value] = 0
                posterior["distribution"][value] += weight

                # Update current value to highest weight
                if posterior["distribution"]:
                    best_value = max(posterior["distribution"].items(), key=lambda x: x[1])[0]
                    posterior["current_value"] = best_value

        return self.typed_posteriors[surface_id][question].copy()

    def _collect_meta_claims_for_step(self, claim_id: str) -> List[MetaClaim]:
        """Collect meta-claims emitted during this step."""
        # Return and clear
        claims = [mc for mc in self.meta_claims]
        self.meta_claims.clear()
        return claims

    def check_context_compatibility(
        self,
        s1_id: str,
        s2_id: str
    ) -> ContextCheckResult:
        """
        Check context compatibility between two surfaces.

        Used for event formation decisions.
        """
        s1 = self.surfaces[s1_id]
        s2 = self.surfaces[s2_id]

        # Find shared entities
        shared = s1.entities & s2.entities

        results = []
        for entity in shared:
            # Companions = other entities that co-occur with this entity
            s1_companions = s1.entities - {entity}
            s2_companions = s2.entities - {entity}

            if len(s1_companions) < 1 or len(s2_companions) < 1:
                results.append(ContextCheckResult(
                    entity=entity,
                    s1_companions=s1_companions,
                    s2_companions=s2_companions,
                    overlap=0.0,
                    result="underpowered"
                ))
            else:
                overlap = len(s1_companions & s2_companions) / min(len(s1_companions), len(s2_companions))
                result = "compatible" if overlap > 0.15 else "incompatible"

                if result == "incompatible":
                    # Emit bridge blocked meta-claim
                    self.meta_claims.append(MetaClaim(
                        type="bridge_blocked",
                        entity=entity,
                        between=(s1_id, s2_id),
                        reason=f"Companions disjoint: {s1_companions} vs {s2_companions}"
                    ))

                results.append(ContextCheckResult(
                    entity=entity,
                    s1_companions=s1_companions,
                    s2_companions=s2_companions,
                    overlap=overlap,
                    result=result
                ))

        # Return first (or most relevant) result
        return results[0] if results else None

    def get_state_snapshot(self) -> Dict:
        """Get current state for assertions."""
        return {
            "claims": {cid: self.claim_to_surface.get(cid) for cid in self.claims},
            "L2_surfaces": {
                sid: {
                    "claims": sorted(s.claim_ids),
                    "question_key": self.surface_question_key.get(sid),
                    "entities": sorted(s.entities),
                }
                for sid, s in self.surfaces.items()
            },
            "L3_incidents": {
                iid: {
                    "surfaces": sorted(i.surface_ids),
                    "anchor_entities": sorted(i.anchor_entities),
                }
                for iid, i in self.incidents.items()
            },
            "surface_to_incident": dict(self.surface_to_incident),
            "typed_posteriors": dict(self.typed_posteriors),
        }


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"

if HAS_PYTEST:
    @pytest.fixture
    def bridge_immunity_trace():
        return GoldenTrace.load(FIXTURES_DIR / "golden_trace_bridge_immunity.yaml")

    @pytest.fixture
    def typed_conflict_trace():
        return GoldenTrace.load(FIXTURES_DIR / "golden_trace_typed_conflict.yaml")

    @pytest.fixture
    def companion_incompatibility_trace():
        return GoldenTrace.load(FIXTURES_DIR / "golden_trace_companion_incompatibility.yaml")

    @pytest.fixture
    def noise_outlier_trace():
        return GoldenTrace.load(FIXTURES_DIR / "golden_trace_noise_outlier.yaml")


def load_bridge_immunity_trace():
    """Non-pytest loader."""
    return GoldenTrace.load(FIXTURES_DIR / "golden_trace_bridge_immunity.yaml")


def load_typed_conflict_trace():
    """Non-pytest loader."""
    return GoldenTrace.load(FIXTURES_DIR / "golden_trace_typed_conflict.yaml")


def load_companion_incompatibility_trace():
    """Non-pytest loader."""
    return GoldenTrace.load(FIXTURES_DIR / "golden_trace_companion_incompatibility.yaml")


def load_noise_outlier_trace():
    """Non-pytest loader."""
    return GoldenTrace.load(FIXTURES_DIR / "golden_trace_noise_outlier.yaml")


# ============================================================================
# TESTS
# ============================================================================

class TestBridgeImmunity:
    """Test that hub entities don't cause false merges at L3."""

    def test_load_trace(self, bridge_immunity_trace):
        """Verify trace loads correctly."""
        trace = bridge_immunity_trace
        assert trace.name == "bridge_immunity"
        assert len(trace.claims) == 6
        assert len(trace.entities) == 7

    def test_l2_surfaces_by_scoped_question_key(self, bridge_immunity_trace):
        """Verify L2 surfaces are created by (scope_id, question_key)."""
        trace = bridge_immunity_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        state = kernel.get_state_snapshot()

        # Should have 6 L2 surfaces (scoped by anchor entities):
        # - fire_death_count (HK+TaiPo)
        # - fire_status (HK+TaiPo) - C002
        # - fire_status (JohnLee+HK+TaiPo) - C003 has different anchors!
        # - fire_cause (HK+TaiPo)
        # - policy_announcement (JohnLee+HK)
        # - policy_status (JohnLee+HK)
        #
        # C003 creates a separate fire_status surface because John Lee visiting
        # fire victims has different anchor entities than the core fire claims.
        # This is CORRECT - different referents = different surfaces.
        assert len(state["L2_surfaces"]) == 6, \
            f"Expected 6 L2 surfaces, got {len(state['L2_surfaces'])}: {list(state['L2_surfaces'].keys())}"

        # Check question_key appears in surface IDs (now with scope prefix)
        surface_ids = list(state["L2_surfaces"].keys())
        assert any("fire_death_count" in sid for sid in surface_ids), \
            f"No surface for fire_death_count in {surface_ids}"
        assert any("fire_status" in sid for sid in surface_ids), \
            f"No surface for fire_status in {surface_ids}"
        assert any("policy_announcement" in sid for sid in surface_ids), \
            f"No surface for policy_announcement in {surface_ids}"

    def test_l3_incidents_separate(self, bridge_immunity_trace):
        """Verify L3 incidents remain separate (fire vs policy)."""
        trace = bridge_immunity_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        state = kernel.get_state_snapshot()

        # Should have 3 L3 incidents:
        # 1. Core fire incident (HK+TaiPo): fire_death_count, fire_status, fire_cause
        # 2. John Lee visits fire (HK+JohnLee+TaiPo): separate fire_status surface
        # 3. Policy incident (HK+JohnLee): policy_announcement, policy_status
        #
        # C003 (John Lee visits fire victims) creates its own incident because
        # its anchor motif differs from both core fire and policy.
        # This is correct - bridging claims get their own scoped surface/incident.
        assert len(state["L3_incidents"]) == 3, \
            f"Expected 3 L3 incidents, got {len(state['L3_incidents'])}: {list(state['L3_incidents'].keys())}"

        # Classify surfaces by question_key substring (fire_* vs policy_*)
        for iid, inc in state["L3_incidents"].items():
            surfaces = inc["surfaces"]
            has_fire = any("fire_" in s for s in surfaces)
            has_policy = any("policy_" in s for s in surfaces)
            # Should not mix fire and policy in same incident
            assert not (has_fire and has_policy), \
                f"Incident {iid} mixes fire and policy surfaces: {surfaces}"


class TestTypedConflict:
    """Test typed value posterior updates and conflicts (weighted aggregation baseline)."""

    def test_load_trace(self, typed_conflict_trace):
        """Verify trace loads correctly."""
        trace = typed_conflict_trace
        # Renamed to be honest about what it tests
        assert trace.name == "weighted_aggregation_baseline"
        assert len(trace.claims) == 5

    def test_single_surface(self, typed_conflict_trace):
        """All claims should join same surface (same question_key)."""
        trace = typed_conflict_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        # Should have exactly 1 surface (all claims share same question_key)
        assert len(kernel.surfaces) == 1

        # All claims in same surface
        surface_id = list(kernel.surfaces.keys())[0]
        for cid in kernel.claims:
            assert kernel.claim_to_surface[cid] == surface_id

    def test_posterior_converges(self, typed_conflict_trace):
        """Posterior should converge via weighted mean (non-Jaynes baseline)."""
        trace = typed_conflict_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        # Get final posterior - question_key is "death_count" in this trace
        surface_id = list(kernel.surfaces.keys())[0]
        question_key = kernel.surface_question_key.get(surface_id)
        posterior = kernel.typed_posteriors[surface_id].get(question_key)

        assert posterior is not None, f"Should have posterior for {question_key}"

        # Check posterior is in reasonable range
        mean = posterior.get("posterior_mean", 0)
        assert 10 <= mean <= 20, f"Posterior mean {mean} should be in [10, 20]"
        assert posterior["observations"] == 5

    def test_conflict_flag_set(self, typed_conflict_trace):
        """Conflict should be detected when values diverge significantly."""
        trace = typed_conflict_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        # Get posterior
        surface_id = list(kernel.surfaces.keys())[0]
        question_key = kernel.surface_question_key.get(surface_id)
        posterior = kernel.typed_posteriors[surface_id].get(question_key, {})

        # Should have conflict_detected flag (values range from 4 to 17)
        assert posterior.get("conflict_detected", False), \
            "Conflict should be detected with divergent values"


class TestCompanionIncompatibility:
    """Test L3 membrane blocks merge when companions are disjoint."""

    def test_load_trace(self, companion_incompatibility_trace):
        """Verify trace loads correctly."""
        trace = companion_incompatibility_trace
        assert trace.name == "companion_incompatibility"
        assert len(trace.claims) == 6

    def test_separate_incidents_despite_shared_anchors(self, companion_incompatibility_trace):
        """Two incidents with same anchors but disjoint companions should NOT merge."""
        trace = companion_incompatibility_trace
        kernel = TraceKernel(trace)

        # Process all claims
        for tc in trace.claims:
            kernel.process_claim(tc)

        state = kernel.get_state_snapshot()

        # Should have 2 L3 incidents (policy + fire response)
        assert len(state["L3_incidents"]) == 2, \
            f"Expected 2 incidents, got {len(state['L3_incidents'])}: {list(state['L3_incidents'].keys())}"

        # Both incidents should share anchors (John Lee, Hong Kong)
        for iid, inc in state["L3_incidents"].items():
            anchors = set(inc["anchor_entities"])
            assert "John Lee" in anchors or "Hong Kong" in anchors, \
                f"Incident {iid} should have John Lee or Hong Kong as anchor"

    def test_bridge_blocked_meta_claim_emitted(self, companion_incompatibility_trace):
        """Meta-claim explaining bridge block should be emitted."""
        trace = companion_incompatibility_trace
        kernel = TraceKernel(trace)

        all_meta_claims = []
        for tc in trace.claims:
            exp = kernel.process_claim(tc)
            all_meta_claims.extend(exp.meta_claims)

        # Should have at least one bridge_blocked meta-claim
        bridge_blocked = [mc for mc in all_meta_claims if mc.type == "bridge_blocked"]
        assert len(bridge_blocked) >= 1, \
            f"Expected bridge_blocked meta-claim, got: {[mc.type for mc in all_meta_claims]}"

    def test_incidental_entity_not_anchor(self, companion_incompatibility_trace):
        """United Nations (mentioned in passing) should not become anchor."""
        trace = companion_incompatibility_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        state = kernel.get_state_snapshot()

        # Check all incidents - UN should not be in anchor_entities
        for iid, inc in state["L3_incidents"].items():
            assert "United Nations" not in inc["anchor_entities"], \
                f"United Nations should not be anchor in {iid}"


class TestNoiseOutlier:
    """Test robustness to adversarial numeric outliers and noise."""

    def test_load_trace(self, noise_outlier_trace):
        """Verify trace loads correctly."""
        trace = noise_outlier_trace
        assert trace.name == "noise_outlier"
        assert len(trace.claims) == 8

    def _find_surface_by_question_key(self, kernel, qk: str) -> str:
        """Find surface ID containing the given question_key."""
        for sid, qkey in kernel.surface_question_key.items():
            if qkey == qk:
                return sid
        return None

    def test_outlier_impact_bounded(self, noise_outlier_trace):
        """Outlier (47) should have minimal impact on posterior (close to 4)."""
        trace = noise_outlier_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        # Get posterior for fire_death_count (find surface by question_key)
        surface_id = self._find_surface_by_question_key(kernel, "fire_death_count")
        assert surface_id, "Should have surface for fire_death_count"
        posterior = kernel.typed_posteriors.get(surface_id, {}).get("fire_death_count", {})
        assert posterior, "Should have fire_death_count posterior"

        mean = posterior.get("posterior_mean", 0)
        # Posterior should be in range [4, 8] - outlier (47) has low weight
        assert 4 <= mean <= 8, \
            f"Posterior mean {mean:.2f} should be between 4 and 8 (outlier impact bounded)"

    def test_outlier_conflict_meta_claim(self, noise_outlier_trace):
        """typed_conflict meta-claim should be emitted for outlier."""
        trace = noise_outlier_trace
        kernel = TraceKernel(trace)

        all_meta_claims = []
        for tc in trace.claims:
            exp = kernel.process_claim(tc)
            all_meta_claims.extend(exp.meta_claims)

        # Should have typed_conflict for C004 (outlier)
        typed_conflicts = [mc for mc in all_meta_claims if mc.type == "typed_conflict"]
        assert len(typed_conflicts) >= 1, \
            f"Expected typed_conflict meta-claim, got: {[mc.type for mc in all_meta_claims]}"

        # Check outlier is in evidence
        outlier_conflict = next((mc for mc in typed_conflicts if mc.evidence.get("outlier_value") == 47), None)
        assert outlier_conflict, "Should have typed_conflict for outlier value 47"

    def test_outlier_preserved_in_values(self, noise_outlier_trace):
        """Outlier should remain visible in reported values."""
        trace = noise_outlier_trace
        kernel = TraceKernel(trace)

        for tc in trace.claims:
            kernel.process_claim(tc)

        surface_id = self._find_surface_by_question_key(kernel, "fire_death_count")
        posterior = kernel.typed_posteriors.get(surface_id, {}).get("fire_death_count", {})
        values = [v for v, _, _ in posterior.get("values", [])]

        assert 47 in values, f"Outlier value 47 should be preserved in values: {values}"

    def test_syndication_detected(self, noise_outlier_trace):
        """Syndicated content should emit syndication_detected meta-claim."""
        trace = noise_outlier_trace
        kernel = TraceKernel(trace)

        all_meta_claims = []
        for tc in trace.claims:
            exp = kernel.process_claim(tc)
            all_meta_claims.extend(exp.meta_claims)

        syndication = [mc for mc in all_meta_claims if mc.type == "syndication_detected"]
        assert len(syndication) >= 2, \
            f"Expected 2+ syndication_detected, got {len(syndication)}"

    def test_missing_time_emits_blocker(self, noise_outlier_trace):
        """Missing timestamp should emit missing_time meta-claim."""
        trace = noise_outlier_trace
        kernel = TraceKernel(trace)

        all_meta_claims = []
        for tc in trace.claims:
            exp = kernel.process_claim(tc)
            all_meta_claims.extend(exp.meta_claims)

        missing_time = [mc for mc in all_meta_claims if mc.type == "missing_time"]
        assert len(missing_time) >= 1, \
            f"Expected missing_time meta-claim for C007, got: {[mc.type for mc in all_meta_claims]}"

    def test_sparse_extraction_emits_underpowered(self, noise_outlier_trace):
        """Sparse extraction (1 entity) should emit context_underpowered."""
        trace = noise_outlier_trace
        kernel = TraceKernel(trace)

        all_meta_claims = []
        for tc in trace.claims:
            exp = kernel.process_claim(tc)
            all_meta_claims.extend(exp.meta_claims)

        underpowered = [mc for mc in all_meta_claims if mc.type == "context_underpowered"]
        assert len(underpowered) >= 1, \
            f"Expected context_underpowered for C008, got: {[mc.type for mc in all_meta_claims]}"


# ============================================================================
# STREAMING VIZ (optional)
# ============================================================================

async def run_trace_with_viz(trace_path: str, port: int = 8082):
    """Run trace with streaming visualization."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    trace = GoldenTrace.load(Path(trace_path))
    kernel = TraceKernel(trace)

    app = FastAPI(title="Golden Trace Replay")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    async def index():
        return HTMLResponse(content=VIZ_HTML)

    @app.get("/api/stream")
    async def stream():
        async def generate():
            for tc in trace.claims:
                exp = kernel.process_claim(tc)
                state = kernel.get_state_snapshot()

                yield f"data: {json.dumps({'explanation': asdict(exp), 'state': state})}\n\n"
                await asyncio.sleep(2)

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/trace")
    async def get_trace():
        return {"name": trace.name, "claims": len(trace.claims)}

    print(f"🔬 Golden Trace Viz: http://localhost:{port}")
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


VIZ_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Golden Trace Replay</title>
    <style>
        body { font-family: monospace; background: #0a0a12; color: #e0e0e0; padding: 20px; }
        .step { background: #151520; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 3px solid #4a9eff; }
        .step h3 { color: #4a9eff; margin: 0 0 10px 0; }
        .decision { color: #4aff9e; }
        .conflict { color: #ff4a4a; }
        pre { background: #0a0a0f; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>🔬 Golden Trace Replay</h1>
    <button onclick="start()">▶ Start</button>
    <div id="steps"></div>
    <script>
        function start() {
            const es = new EventSource('/api/stream');
            es.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'complete') { es.close(); return; }
                const exp = data.explanation;
                const div = document.createElement('div');
                div.className = 'step';
                div.innerHTML = `
                    <h3>Claim: ${exp.claim_id}</h3>
                    <div class="decision">${exp.identity_decision.action}: ${exp.identity_decision.surface_id}</div>
                    <div>Reason: ${exp.identity_decision.reason}</div>
                    ${exp.relation ? `<div>Relation: ${exp.relation}</div>` : ''}
                    ${exp.meta_claims.length ? `<div class="conflict">Meta-claims: ${exp.meta_claims.map(m => m.type).join(', ')}</div>` : ''}
                `;
                document.getElementById('steps').appendChild(div);
            };
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import sys
    if "--viz" in sys.argv:
        trace_file = sys.argv[sys.argv.index("--viz") + 1] if len(sys.argv) > sys.argv.index("--viz") + 1 else str(FIXTURES_DIR / "golden_trace_bridge_immunity.yaml")
        asyncio.run(run_trace_with_viz(trace_file))
    else:
        pytest.main([__file__, "-v"])
