"""
Principled Case Builder
=======================

Builds Cases (L4) from Incidents (L3) using kernel-native evidence.

This implements the principled emergence approach for L4 case formation:
1. Consume L3 justification.core_motifs as primary evidence
2. Compute CROSS-INCIDENT motif support (L4 recurrence, not L3 internal)
3. Apply hubness suppression at L4 scale
4. Use shared_supported_motif for CORE edges (exact motif recurrence)
5. Use motif chains for PERIPHERY attachment only (never core)
6. Apply anti-percolation rule: core requires shared_supported_motif

THEORY (from user guidance):
- Anchor overlap is NOT a valid sufficient statistic for L4
- Motif chains percolate via high-context-entropy bridge entities
  (e.g., "Bondi Beach", "Trump") even when not above frequency threshold
- The right criterion is SHARED SUPPORTED MOTIF (exact recurrence)
- Chain-only edges must be periphery (attachment), never core (merge)

CONSTRAINT TYPES FOR L4:
- shared_motif: Same motif in ≥2 incidents, cross-incident support ≥2 → CORE eligible
- motif_chain: 2-hop via shared entity → PERIPHERY only (never core)
- time_compatible_case: Temporal proximity → supports but doesn't create core
- anchor_not_hub: Non-hub shared anchor → supports but doesn't create core

CORE ELIGIBILITY (anti-percolation rule):
- ≥2 constraints total
- ≥1 shared_motif constraint (chains don't qualify)

Usage:
    builder = PrincipledCaseBuilder()
    result = builder.build_from_incidents(incidents)
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional, Any

from ..types import (
    Event, Surface, Story, StoryScale,
    Constraint, ConstraintType, ConstraintLedger,
    MembershipLevel, EventJustification,
)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class MotifProfile:
    """
    Incident's motif profile extracted from justification.core_motifs.

    This is the L3→L4 interface: incidents expose their motifs,
    L4 clusters by motif recurrence.
    """
    incident_id: str
    motifs: List[frozenset]  # List of entity sets
    motif_support: Dict[frozenset, int]  # motif → support count from L3
    anchor_entities: Set[str]
    time_window: Tuple[Optional[datetime], Optional[datetime]]


@dataclass
class L4Hubness:
    """
    L4-scale hubness for an entity.

    Different from L3 dispersion: L4 hubness is about participation
    across incidents, not co-anchor cohesion.
    """
    entity: str
    incident_count: int  # How many incidents contain this entity
    incident_fraction: float  # What fraction of all incidents
    is_hub: bool  # True if too ubiquitous to bind cases
    explanation: str


@dataclass
class CaseEdge:
    """An edge between two incidents in a case, with explainability."""
    incident1_id: str
    incident2_id: str
    is_core: bool  # Core edge (can merge) vs periphery (attaches only)
    constraints: List[Constraint]  # Evidence for this edge
    shared_motifs: List[frozenset]
    motif_chains: List[Tuple[frozenset, frozenset]]  # (A's motif, B's motif) chains
    shared_anchors: Set[str]
    hub_anchors: Set[str]  # Anchors suppressed as hubs
    explanation: str


@dataclass
class EntityCase:
    """
    Entity-centric case (lens-like view).

    Unlike CaseCore (partition-like), EntityCase allows overlap:
    - Focal entity defines the case
    - Incidents attach with membership weights
    - Same incident can appear in multiple EntityCases

    This handles star-shaped storylines (Jimmy Lai pattern):
    - One focal entity with rotating companions
    - No pair recurs (k=2 recurrence fails)
    - But the focal entity ties the narrative

    Anti-percolation: Hub entities (Hong Kong, USA, Time) may attach
    but never define the case. Only non-hub entities can be focal.
    """
    entity: str  # Focal entity
    entity_case_id: str  # Stable ID
    incident_ids: Set[str]  # All attached incidents
    core_incident_ids: Set[str]  # Where entity is primary subject
    periphery_incident_ids: Set[str]  # Where entity is supporting
    total_incidents: int
    time_start: Optional[datetime]
    time_end: Optional[datetime]
    companion_entities: Dict[str, int]  # Co-occurring entities → count
    sample_headlines: List[str]  # Representative surfaces
    membership_weights: Dict[str, MembershipLevel]  # incident_id → membership
    is_hub: bool  # True if entity is hub (affects case validity)
    explanation: str


@dataclass
class CaseBuilderResult:
    """Result of case building."""
    cases: Dict[str, Story]  # case_id → Story(scale="case") - CaseCore (partition-like)
    entity_cases: Dict[str, EntityCase]  # entity → EntityCase - lens-like views
    motif_profiles: Dict[str, MotifProfile]  # incident_id → profile
    hubness: Dict[str, L4Hubness]  # entity → hubness
    edges: List[CaseEdge]  # All considered edges
    ledger: ConstraintLedger
    stats: Dict[str, Any]


# =============================================================================
# PRINCIPLED CASE BUILDER
# =============================================================================

class PrincipledCaseBuilder:
    """
    Builds cases from incidents using kernel-native evidence.

    This replaces the ad-hoc anchor overlap criterion in canonical_worker
    with principled motif-based clustering.
    """

    def __init__(
        self,
        min_motif_size: int = 2,  # Minimum entities in a motif to count
        hub_fraction_threshold: float = 0.3,  # Entity in >30% of incidents = hub
        hub_min_incidents: int = 3,  # Need ≥3 incidents to compute hubness
        time_window_days: int = 90,  # Case-scale time window (3 months)
        min_incidents_for_case: int = 2,  # Minimum incidents to form a CaseCore
        min_incidents_for_entity_case: int = 5,  # Minimum for EntityCase (lens)
    ):
        self.min_motif_size = min_motif_size
        self.hub_fraction_threshold = hub_fraction_threshold
        self.hub_min_incidents = hub_min_incidents
        self.time_window_days = time_window_days
        self.min_incidents_for_case = min_incidents_for_case
        self.min_incidents_for_entity_case = min_incidents_for_entity_case
        self.ledger = ConstraintLedger()

    def build_from_incidents(
        self,
        incidents: Dict[str, Event],
        surfaces: Dict[str, Surface] = None,
    ) -> CaseBuilderResult:
        """
        Build cases from incidents using motif-based clustering.

        Strategy:
        1. Extract motif profiles from each incident's justification
        2. Compute L4-scale hubness for all entities
        3. Build inverted index: motif → incidents
        4. For each incident pair sharing motif(s), evaluate constraints
        5. Cluster into cases using core/periphery discipline

        Args:
            incidents: Dict of incident_id → Event (from L3)
            surfaces: Optional surface dict for additional metadata

        Returns:
            CaseBuilderResult with cases, profiles, hubness, and ledger
        """
        self.ledger = ConstraintLedger()

        # Step 1: Extract motif profiles from incidents
        profiles = self._extract_motif_profiles(incidents)

        # Step 2: Compute L4-scale hubness
        hubness = self._compute_l4_hubness(profiles)

        # Step 3: Build motif → incidents index
        motif_to_incidents = self._build_motif_index(profiles)

        # Step 4: Build candidate edges from shared motifs
        edges = self._build_candidate_edges(
            profiles, hubness, motif_to_incidents
        )

        # Step 5: Cluster into CaseCores (partition-like)
        cases = self._cluster_into_cases(edges, incidents, profiles)

        # Step 6: Build EntityCases (lens-like) for star-shaped storylines
        entity_cases = self._build_entity_cases(incidents, hubness, profiles)

        # Count incidents without motif profiles (blocked from L4 core)
        incidents_without_motifs = sum(1 for p in profiles.values() if not p.motifs)

        # Count incidents covered by CaseCores vs EntityCases only
        casecore_incidents = set()
        for case in cases.values():
            casecore_incidents.update(case.incident_ids)

        entitycase_only_incidents = set()
        for ec in entity_cases.values():
            entitycase_only_incidents.update(ec.incident_ids)
        entitycase_only_incidents -= casecore_incidents

        stats = {
            "incidents": len(incidents),
            "profiles_extracted": len(profiles),
            "incidents_with_motifs": len(profiles) - incidents_without_motifs,
            "incidents_without_motifs": incidents_without_motifs,
            "unique_motifs": len(motif_to_incidents),
            "hub_entities": sum(1 for h in hubness.values() if h.is_hub),
            "candidate_edges": len(edges),
            "core_edges": sum(1 for e in edges if e.is_core),
            "periphery_edges": sum(1 for e in edges if not e.is_core),
            "cases_formed": len(cases),
            "entity_cases_formed": len(entity_cases),
            "entitycase_only_incidents": len(entitycase_only_incidents),
            "casecore_incidents": len(casecore_incidents),
            "constraints": len(self.ledger.constraints),
        }

        return CaseBuilderResult(
            cases=cases,
            entity_cases=entity_cases,
            motif_profiles=profiles,
            hubness=hubness,
            edges=edges,
            ledger=self.ledger,
            stats=stats,
        )

    def _extract_motif_profiles(
        self,
        incidents: Dict[str, Event]
    ) -> Dict[str, MotifProfile]:
        """
        Extract motif profiles from incident anchor entities.

        L4 motif strategy:
        - Generate candidate motifs from anchor-entity pairs (k=2)
        - Motifs are anchor-pair based (not L3 core_motifs)
        - Cross-incident support computed in _build_motif_index
        - Only motifs with cross-incident support ≥2 become core-eligible

        This approach works because:
        1. L3 core_motifs are incident-internal (rarely recur)
        2. Anchor pairs ARE the discriminative signal
        3. Cross-incident recurrence of same pair = case binding
        """
        profiles = {}

        for incident_id, incident in incidents.items():
            motifs = []
            motif_support = {}

            # Strategy: Generate k=2 motifs from anchor entities
            # These are candidate motifs; cross-incident support computed later
            anchors = list(incident.anchor_entities)

            if len(anchors) >= 2:
                from itertools import combinations
                for pair in combinations(sorted(anchors), 2):
                    motif = frozenset(pair)
                    motifs.append(motif)
                    motif_support[motif] = 1  # L4 support computed in index

            # Also include L3 core_motifs if available (additive, not replacement)
            if incident.justification and incident.justification.core_motifs:
                for motif_entry in incident.justification.core_motifs:
                    entities = motif_entry.get("entities", [])
                    support = motif_entry.get("support", 1)

                    if len(entities) >= self.min_motif_size:
                        motif = frozenset(entities)
                        if motif not in motifs:
                            motifs.append(motif)
                        motif_support[motif] = max(
                            motif_support.get(motif, 0), support
                        )

            profiles[incident_id] = MotifProfile(
                incident_id=incident_id,
                motifs=motifs,
                motif_support=motif_support,
                anchor_entities=incident.anchor_entities.copy(),
                time_window=incident.time_window,
            )

            # Log profile
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"Incident {incident_id} has {len(motifs)} candidate motifs from {len(anchors)} anchors",
                evidence={
                    "incident_id": incident_id,
                    "anchor_count": len(anchors),
                    "motif_count": len(motifs),
                    "sample_motifs": [list(m) for m in motifs[:3]],
                    "source": "anchor_pairs",
                },
                provenance="motif_profile_extraction"
            ), scope=incident_id)

        return profiles

    def _compute_l4_hubness(
        self,
        profiles: Dict[str, MotifProfile]
    ) -> Dict[str, L4Hubness]:
        """
        Compute L4-scale hubness for all entities.

        L4 hubness is different from L3 dispersion:
        - L3 dispersion: Does entity's co-anchors form a cohesive cluster?
        - L4 hubness: Does entity appear in too many incidents to be discriminative?

        An entity like "Hong Kong" appearing in 40% of incidents should not
        bind cases - it's too ubiquitous.
        """
        # Count entity appearances across incidents
        entity_incidents: Dict[str, Set[str]] = defaultdict(set)

        for incident_id, profile in profiles.items():
            # Count from anchor_entities
            for entity in profile.anchor_entities:
                entity_incidents[entity].add(incident_id)

            # Also count from motifs
            for motif in profile.motifs:
                for entity in motif:
                    entity_incidents[entity].add(incident_id)

        total_incidents = len(profiles)
        hubness = {}

        for entity, inc_set in entity_incidents.items():
            count = len(inc_set)
            fraction = count / total_incidents if total_incidents > 0 else 0

            # Hubness determination
            if total_incidents < self.hub_min_incidents:
                # Not enough incidents to determine hubness
                is_hub = False
                explanation = f"Only {total_incidents} incidents, cannot determine hubness"
            elif fraction >= self.hub_fraction_threshold:
                is_hub = True
                explanation = (
                    f"HUB: appears in {count}/{total_incidents} incidents "
                    f"({fraction:.0%} >= {self.hub_fraction_threshold:.0%} threshold). "
                    f"Too ubiquitous to bind cases."
                )
            else:
                is_hub = False
                explanation = (
                    f"NOT hub: appears in {count}/{total_incidents} incidents "
                    f"({fraction:.0%} < {self.hub_fraction_threshold:.0%} threshold). "
                    f"Can bind cases."
                )

            hubness[entity] = L4Hubness(
                entity=entity,
                incident_count=count,
                incident_fraction=fraction,
                is_hub=is_hub,
                explanation=explanation,
            )

            # Log hub entities
            if is_hub:
                self.ledger.add(Constraint(
                    constraint_type=ConstraintType.STRUCTURAL,
                    assertion=f"Entity '{entity}' is L4 hub (suppressed)",
                    evidence={
                        "entity": entity,
                        "incident_count": count,
                        "incident_fraction": round(fraction, 3),
                        "threshold": self.hub_fraction_threshold,
                        "explanation": explanation,
                    },
                    provenance="l4_hub_detection"
                ), scope=f"hub:{entity}")

        return hubness

    def _build_motif_index(
        self,
        profiles: Dict[str, MotifProfile]
    ) -> Dict[frozenset, Set[str]]:
        """
        Build inverted index: motif → incident_ids.

        Also computes cross-incident support for motifs.
        A motif's L4 support = number of distinct incidents containing it.
        This is different from L3 support (surfaces within an incident).
        """
        motif_to_incidents: Dict[frozenset, Set[str]] = defaultdict(set)

        for incident_id, profile in profiles.items():
            for motif in profile.motifs:
                motif_to_incidents[motif].add(incident_id)

        # Log motifs with cross-incident support ≥ 2
        supported_motifs = {
            m: incs for m, incs in motif_to_incidents.items()
            if len(incs) >= 2
        }

        if supported_motifs:
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"Found {len(supported_motifs)} motifs with cross-incident support ≥ 2",
                evidence={
                    "count": len(supported_motifs),
                    "top_motifs": [
                        {"motif": list(m)[:4], "support": len(incs)}
                        for m, incs in sorted(
                            supported_motifs.items(),
                            key=lambda x: -len(x[1])
                        )[:10]
                    ],
                },
                provenance="cross_incident_motif_support"
            ), scope="l4_global")

        return motif_to_incidents

    def _build_candidate_edges(
        self,
        profiles: Dict[str, MotifProfile],
        hubness: Dict[str, L4Hubness],
        motif_to_incidents: Dict[frozenset, Set[str]],
    ) -> List[CaseEdge]:
        """
        Build candidate edges between incidents.

        Edge formation rules:
        1. shared_motif: Same motif in both incidents → strong structural evidence
        2. motif_chain: Overlapping motifs (2-hop) → weaker structural evidence
        3. time_compatible: Within case-scale time window → temporal evidence
        4. anchor_not_hub: Shared anchor that isn't a hub → structural evidence

        Core eligibility (anti-trap):
        - ≥2 constraints total
        - ≥1 non-semantic
        """
        edges = []
        seen_pairs = set()

        # Strategy 1: Direct motif sharing
        for motif, incident_ids in motif_to_incidents.items():
            if len(incident_ids) < 2:
                continue

            for inc1, inc2 in combinations(sorted(incident_ids), 2):
                pair_key = (inc1, inc2)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                edge = self._evaluate_edge(
                    inc1, inc2, profiles, hubness, motif_to_incidents
                )
                if edge:
                    edges.append(edge)

        # Strategy 2: Motif chains (2-hop via overlapping motifs)
        # Find pairs that don't share a motif but have overlapping motifs
        for inc1, inc2 in combinations(sorted(profiles.keys()), 2):
            pair_key = (inc1, inc2)
            if pair_key in seen_pairs:
                continue

            # Check for motif chain
            profile1 = profiles[inc1]
            profile2 = profiles[inc2]

            has_chain = False
            for m1 in profile1.motifs:
                for m2 in profile2.motifs:
                    # Chain: motifs overlap (share at least 1 non-hub entity)
                    overlap = m1 & m2
                    non_hub_overlap = {
                        e for e in overlap
                        if not hubness.get(e, L4Hubness(e, 0, 0, False, "")).is_hub
                    }
                    if non_hub_overlap:
                        has_chain = True
                        break
                if has_chain:
                    break

            if has_chain:
                seen_pairs.add(pair_key)
                edge = self._evaluate_edge(
                    inc1, inc2, profiles, hubness, motif_to_incidents
                )
                if edge:
                    edges.append(edge)

        return edges

    def _evaluate_edge(
        self,
        inc1: str,
        inc2: str,
        profiles: Dict[str, MotifProfile],
        hubness: Dict[str, L4Hubness],
        motif_to_incidents: Dict[frozenset, Set[str]],
    ) -> Optional[CaseEdge]:
        """
        Evaluate a candidate edge between two incidents.

        Returns CaseEdge if there's any evidence, None otherwise.
        Determines core vs periphery based on anti-trap rule.
        """
        profile1 = profiles[inc1]
        profile2 = profiles[inc2]
        pair_key = f"{inc1}:{inc2}"

        constraints = []
        shared_motifs = []
        motif_chains = []
        shared_anchors = set()
        hub_anchors = set()

        # === CONSTRAINT 1: Shared supported motifs ===
        # A motif counts as "supported" if it appears in ≥2 incidents (cross-incident)
        # This is the primary L4 merge criterion - exact motif recurrence
        for motif in profile1.motifs:
            if motif in profile2.motifs:
                shared_motifs.append(motif)

                # Check cross-incident support (L4 support, not L3)
                cross_incident_support = len(motif_to_incidents.get(motif, set()))

                # Check if motif contains non-hub entities
                non_hub_entities = [
                    e for e in motif
                    if not hubness.get(e, L4Hubness(e, 0, 0, False, "")).is_hub
                ]

                # Only emit shared_motif constraint if:
                # 1. Has non-hub entities
                # 2. Has cross-incident support ≥ 2 (by definition, since both incidents have it)
                if non_hub_entities and cross_incident_support >= 2:
                    constraints.append(Constraint(
                        constraint_type=ConstraintType.STRUCTURAL,
                        assertion=f"Shared supported motif {set(motif)} (L4 support={cross_incident_support})",
                        evidence={
                            "motif": list(motif),
                            "non_hub_entities": non_hub_entities,
                            "cross_incident_support": cross_incident_support,
                            "support1": profile1.motif_support.get(motif, 0),
                            "support2": profile2.motif_support.get(motif, 0),
                        },
                        provenance="shared_motif"
                    ))
                    self.ledger.add(constraints[-1], scope=pair_key)

        # === CONSTRAINT 2: Motif chains ===
        for m1 in profile1.motifs:
            for m2 in profile2.motifs:
                if m1 == m2:
                    continue  # Already counted as shared

                overlap = m1 & m2
                non_hub_overlap = {
                    e for e in overlap
                    if not hubness.get(e, L4Hubness(e, 0, 0, False, "")).is_hub
                }

                if non_hub_overlap and len(non_hub_overlap) >= 1:
                    motif_chains.append((m1, m2))

                    # Only count chain if not already have shared motif
                    if not any(m1 == sm or m2 == sm for sm in shared_motifs):
                        constraints.append(Constraint(
                            constraint_type=ConstraintType.STRUCTURAL,
                            assertion=f"Motif chain via {non_hub_overlap}",
                            evidence={
                                "motif1": list(m1),
                                "motif2": list(m2),
                                "overlap": list(non_hub_overlap),
                            },
                            provenance="motif_chain"
                        ))
                        self.ledger.add(constraints[-1], scope=pair_key)

        # === CONSTRAINT 3: Time compatibility ===
        time1_start = profile1.time_window[0]
        time2_start = profile2.time_window[0]

        if time1_start and time2_start:
            delta = abs((time1_start - time2_start).days)
            if delta <= self.time_window_days:
                constraints.append(Constraint(
                    constraint_type=ConstraintType.TEMPORAL,
                    assertion=f"Time compatible ({delta} days apart)",
                    evidence={
                        "time1": time1_start.isoformat() if time1_start else None,
                        "time2": time2_start.isoformat() if time2_start else None,
                        "delta_days": delta,
                        "threshold_days": self.time_window_days,
                    },
                    provenance="time_compatible_case"
                ))
                self.ledger.add(constraints[-1], scope=pair_key)

        # === CONSTRAINT 4: Shared non-hub anchors ===
        shared = profile1.anchor_entities & profile2.anchor_entities
        for anchor in shared:
            h = hubness.get(anchor, L4Hubness(anchor, 0, 0, False, ""))
            if h.is_hub:
                hub_anchors.add(anchor)
            else:
                shared_anchors.add(anchor)
                constraints.append(Constraint(
                    constraint_type=ConstraintType.STRUCTURAL,
                    assertion=f"Shared non-hub anchor '{anchor}'",
                    evidence={
                        "anchor": anchor,
                        "hubness_fraction": h.incident_fraction,
                        "is_hub": h.is_hub,
                    },
                    provenance="anchor_not_hub"
                ))
                self.ledger.add(constraints[-1], scope=pair_key)

        # === NO EVIDENCE ===
        if not constraints:
            return None

        # === L4 ANTI-PERCOLATION RULE: Core requires shared_supported_motif ===
        #
        # The key insight: motif_chain edges percolate via high-context-entropy
        # bridge entities (e.g., "Bondi Beach", "Trump") even when those entities
        # aren't above the simple frequency-based hub threshold.
        #
        # Solution: Chain-only edges are PERIPHERY, never core.
        # Core eligibility requires shared_supported_motif (exact motif recurrence).
        #
        # This stops the mega-case percolation while preserving legitimate merges.

        non_semantic = [c for c in constraints if c.constraint_type != ConstraintType.SEMANTIC]

        # Only shared_motif counts for core eligibility (not chain)
        shared_motif_constraints = [
            c for c in constraints
            if c.provenance == "shared_motif"
        ]
        chain_constraints = [
            c for c in constraints
            if c.provenance == "motif_chain"
        ]

        has_shared_motif = len(shared_motif_constraints) >= 1
        has_chain_only = len(chain_constraints) >= 1 and not has_shared_motif

        # Core eligibility: ≥2 constraints AND ≥1 shared_motif constraint
        # Chain-only edges are explicitly periphery (attachment only)
        is_core = (
            len(constraints) >= 2 and
            len(non_semantic) >= 1 and
            has_shared_motif  # NOT has_motif_evidence - chains don't qualify
        )

        if is_core:
            reason = f"CORE: {len(constraints)} constraints ({len(shared_motif_constraints)} shared_motif)"
        elif has_chain_only:
            reason = f"PERIPHERY: chain-only evidence (chains cannot form cores)"
        elif has_shared_motif:
            reason = f"PERIPHERY: has shared_motif but < 2 total constraints"
        else:
            reason = f"PERIPHERY: anchor-only (no motif evidence)"

        # Generate explanation
        parts = []
        if shared_motifs:
            parts.append(f"shared motifs: {[list(m)[:3] for m in shared_motifs[:3]]}")
        if motif_chains:
            parts.append(f"motif chains: {len(motif_chains)}")
        if shared_anchors:
            parts.append(f"shared anchors: {shared_anchors}")
        if hub_anchors:
            parts.append(f"hub anchors (suppressed): {hub_anchors}")

        explanation = f"{reason}. Evidence: {'; '.join(parts)}"

        return CaseEdge(
            incident1_id=inc1,
            incident2_id=inc2,
            is_core=is_core,
            constraints=constraints,
            shared_motifs=shared_motifs,
            motif_chains=motif_chains,
            shared_anchors=shared_anchors,
            hub_anchors=hub_anchors,
            explanation=explanation,
        )

    def _cluster_into_cases(
        self,
        edges: List[CaseEdge],
        incidents: Dict[str, Event],
        profiles: Dict[str, MotifProfile],
    ) -> Dict[str, Story]:
        """
        Cluster incidents into cases using core edges.

        Core edges define connected components.
        Periphery edges attach but don't merge.
        """
        # Build adjacency for core edges only
        core_adj: Dict[str, Set[str]] = defaultdict(set)
        edge_lookup: Dict[Tuple[str, str], CaseEdge] = {}

        for edge in edges:
            if edge.is_core:
                core_adj[edge.incident1_id].add(edge.incident2_id)
                core_adj[edge.incident2_id].add(edge.incident1_id)
            edge_lookup[(edge.incident1_id, edge.incident2_id)] = edge
            edge_lookup[(edge.incident2_id, edge.incident1_id)] = edge

        # Find connected components on core graph
        visited = set()
        cases = {}
        case_idx = 0

        for incident_id in incidents:
            if incident_id in visited:
                continue

            # BFS for component
            component = set()
            component_edges = []
            queue = [incident_id]

            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)

                for neighbor in core_adj.get(curr, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
                        edge = edge_lookup.get((curr, neighbor))
                        if edge and edge.is_core:
                            component_edges.append(edge)

            # Skip single-incident "cases"
            if len(component) < self.min_incidents_for_case:
                continue

            # Build case
            case = self._build_case(
                f"case_{case_idx:04d}",
                component,
                component_edges,
                incidents,
                profiles,
            )

            if case:
                cases[case.id] = case
                case_idx += 1

                # Log case formation
                self.ledger.add(Constraint(
                    constraint_type=ConstraintType.STRUCTURAL,
                    assertion=f"Case {case.id} formed from {len(component)} incidents",
                    evidence={
                        "case_id": case.id,
                        "incident_count": len(component),
                        "incidents": list(component),
                        "core_edges": len(component_edges),
                        "backbone_entities": list(case.anchor_entities)[:5],
                    },
                    provenance="case_formation"
                ), scope=case.id)

        return cases

    def _build_case(
        self,
        case_id: str,
        incident_ids: Set[str],
        edges: List[CaseEdge],
        incidents: Dict[str, Event],
        profiles: Dict[str, MotifProfile],
    ) -> Optional[Story]:
        """
        Build a Case (Story with scale="case") from incidents.
        """
        # Aggregate data from incidents
        all_surfaces = set()
        all_entities = set()
        anchor_entities = set()
        total_claims = 0
        total_sources = 0
        times = []

        # Collect shared motifs from edges as backbone evidence
        backbone_motifs: Dict[frozenset, int] = defaultdict(int)
        for edge in edges:
            for motif in edge.shared_motifs:
                backbone_motifs[motif] += 1

        for inc_id in incident_ids:
            incident = incidents.get(inc_id)
            if not incident:
                continue

            all_surfaces.update(incident.surface_ids)
            all_entities.update(incident.entities)
            anchor_entities.update(incident.anchor_entities)
            total_claims += incident.total_claims
            total_sources += incident.total_sources

            if incident.time_window[0]:
                times.append(incident.time_window[0])
            if incident.time_window[1]:
                times.append(incident.time_window[1])

        # Determine primary entities from most shared motifs
        entity_backbone_score: Dict[str, int] = defaultdict(int)
        for motif, count in backbone_motifs.items():
            for entity in motif:
                entity_backbone_score[entity] += count

        # Sort by backbone score
        primary_entities = sorted(
            entity_backbone_score.keys(),
            key=lambda e: -entity_backbone_score[e]
        )[:5]

        # Compute time window
        time_start = min(times) if times else None
        time_end = max(times) if times else None

        # Generate title from primary entities
        if primary_entities:
            title = f"{primary_entities[0]}"
            if len(primary_entities) > 1:
                title += f" & {primary_entities[1]}"
            title += ": Ongoing Story"
        else:
            title = "Unnamed Case"

        # Generate description
        description = (
            f"Story spanning {len(incident_ids)} incidents "
            f"with {total_claims} claims from {total_sources} sources. "
            f"Primary entities: {', '.join(primary_entities[:5])}"
        )

        # Build justification
        justification = EventJustification(
            core_motifs=[
                {"entities": list(m), "support": c}
                for m, c in sorted(backbone_motifs.items(), key=lambda x: -x[1])[:10]
            ],
            representative_surfaces=list(all_surfaces)[:3],
            canonical_handle=title,
        )

        # Create Story with scale="case"
        story = Story(
            id=case_id,
            scale="case",
            title=title,
            description=description,
            primary_entities=primary_entities,
            anchor_entities=anchor_entities,
            surface_ids=all_surfaces,
            incident_ids=incident_ids,
            time_start=time_start,
            time_end=time_end,
            surface_count=len(all_surfaces),
            source_count=total_sources,
            claim_count=total_claims,
            incident_count=len(incident_ids),
            justification=justification,
        )

        # Compute stable ID
        story.compute_scope_signature()

        return story

    def _build_entity_cases(
        self,
        incidents: Dict[str, Event],
        hubness: Dict[str, L4Hubness],
        profiles: Dict[str, MotifProfile],
    ) -> Dict[str, EntityCase]:
        """
        Build EntityCases (lens-like views) for star-shaped storylines.

        This handles cases like Jimmy Lai where:
        - One focal entity appears across many incidents
        - Companion entities rotate (no pair recurs)
        - k=2 motif recurrence fails but narrative coheres

        Strategy:
        1. Find entities appearing in ≥ min_incidents_for_entity_case incidents
        2. Filter out hub entities (they cannot DEFINE cases)
        3. Classify incidents as core (entity is primary) vs periphery (supporting)
        4. Build EntityCase with membership weights

        Anti-percolation rule:
        - Hub entities (>30% of incidents) can ATTACH to EntityCases
        - But hubs can NEVER be the focal entity of an EntityCase
        """
        entity_cases = {}

        # Step 1: Count entity appearances across incidents
        entity_to_incidents: Dict[str, Set[str]] = defaultdict(set)

        for incident_id, incident in incidents.items():
            for entity in incident.anchor_entities:
                entity_to_incidents[entity].add(incident_id)

        # Step 2: Find focal entity candidates (non-hub, sufficient incidents)
        focal_candidates = []
        for entity, inc_set in entity_to_incidents.items():
            if len(inc_set) < self.min_incidents_for_entity_case:
                continue

            h = hubness.get(entity, L4Hubness(entity, 0, 0, False, ""))
            if h.is_hub:
                # Hub entities cannot define EntityCases
                self.ledger.add(Constraint(
                    constraint_type=ConstraintType.STRUCTURAL,
                    assertion=f"Entity '{entity}' is hub - cannot define EntityCase",
                    evidence={
                        "entity": entity,
                        "incident_count": len(inc_set),
                        "hub_fraction": h.incident_fraction,
                        "reason": "Anti-percolation: hubs attach but never define",
                    },
                    provenance="entity_case_hub_blocked"
                ), scope=f"entity:{entity}")
                continue

            focal_candidates.append((entity, inc_set, h))

        # Step 3: Build EntityCase for each focal entity
        for entity, inc_set, h in focal_candidates:
            # Classify incidents as core vs periphery
            core_incidents = set()
            periphery_incidents = set()
            membership_weights = {}

            for inc_id in inc_set:
                incident = incidents.get(inc_id)
                if not incident:
                    continue

                # Core: entity is in anchor_entities (primary subject)
                # Periphery: entity appears but not anchored
                if entity in incident.anchor_entities:
                    # Further refinement: core if entity is top-2 anchor by recurrence
                    # For now, all anchor appearances are core
                    core_incidents.add(inc_id)
                    membership_weights[inc_id] = MembershipLevel.CORE
                else:
                    periphery_incidents.add(inc_id)
                    membership_weights[inc_id] = MembershipLevel.PERIPHERY

            # Collect companion entities (co-occurring with focal)
            companion_counts: Dict[str, int] = defaultdict(int)
            for inc_id in inc_set:
                incident = incidents.get(inc_id)
                if not incident:
                    continue
                for companion in incident.anchor_entities:
                    if companion != entity:
                        companion_counts[companion] += 1

            # Top companions
            top_companions = dict(
                sorted(companion_counts.items(), key=lambda x: -x[1])[:20]
            )

            # Time bounds
            times = []
            for inc_id in inc_set:
                incident = incidents.get(inc_id)
                if incident and incident.time_window[0]:
                    times.append(incident.time_window[0])
                if incident and incident.time_window[1]:
                    times.append(incident.time_window[1])

            time_start = min(times) if times else None
            time_end = max(times) if times else None

            # Sample headlines from justifications
            sample_headlines = []
            for inc_id in list(inc_set)[:5]:
                incident = incidents.get(inc_id)
                if incident and incident.justification:
                    handle = incident.justification.canonical_handle
                    if handle:
                        sample_headlines.append(handle)

            # Generate explanation
            explanation = (
                f"Star-shaped storyline: {entity} appears in {len(inc_set)} incidents "
                f"({len(core_incidents)} core, {len(periphery_incidents)} periphery). "
                f"Top companions: {list(top_companions.keys())[:5]}. "
                f"Hub status: {'YES (attached but not defining)' if h.is_hub else 'NO (can define case)'}."
            )

            # Create stable ID
            entity_case_id = f"ec_{hashlib.md5(entity.encode()).hexdigest()[:12]}"

            entity_case = EntityCase(
                entity=entity,
                entity_case_id=entity_case_id,
                incident_ids=inc_set,
                core_incident_ids=core_incidents,
                periphery_incident_ids=periphery_incidents,
                total_incidents=len(inc_set),
                time_start=time_start,
                time_end=time_end,
                companion_entities=top_companions,
                sample_headlines=sample_headlines,
                membership_weights=membership_weights,
                is_hub=h.is_hub,
                explanation=explanation,
            )

            entity_cases[entity] = entity_case

            # Log EntityCase formation
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"EntityCase '{entity}' formed with {len(inc_set)} incidents",
                evidence={
                    "entity": entity,
                    "entity_case_id": entity_case_id,
                    "total_incidents": len(inc_set),
                    "core_incidents": len(core_incidents),
                    "periphery_incidents": len(periphery_incidents),
                    "top_companions": list(top_companions.keys())[:5],
                    "is_hub": h.is_hub,
                },
                provenance="entity_case_formation"
            ), scope=entity_case_id)

        return entity_cases


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PrincipledCaseBuilder',
    'CaseBuilderResult',
    'MotifProfile',
    'L4Hubness',
    'CaseEdge',
    'EntityCase',
]
