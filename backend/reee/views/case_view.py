"""
Case View - Story-level aggregation of incidents
=================================================

While IncidentView (L3) produces tight membranes with anti-percolation,
CaseView (L4) aggregates incidents into larger "story" or "case" structures.

THEORY (from REEE1.md section 8.2-8.3):
- Cases use DIFFERENT signals than incidents (not same signals with looser thresholds)
- Primary signal: Entity relation BACKBONE (not just shared entities)
- Dispersion-based hubness: An entity's co-anchors determine hub vs backbone
  - If co-anchors co-occur with each other → BACKBONE (binds)
  - If co-anchors are disjoint → HUB (suppressed)

THREE SIGNALS FOR CASE BINDING (2-of-3 gate):
1. BACKBONE OVERLAP: Events share a backbone entity (not a hub)
2. RELATION BACKBONE: Events share a stable entity-pair (e.g., Jimmy Lai ↔ Apple Daily)
3. SEMANTIC SIMILARITY: Event centroids have high cosine similarity

DISPERSION FORMULA:
    co_anchors = {b : b co-occurs with a in some incident}
    cohesion = |{(b1,b2) : b1,b2 co-occur in some incident}| / |all pairs in co_anchors|
    dispersion = 1 - cohesion

CLASSIFICATION:
    freq < threshold → NEUTRAL (not enough signal)
    freq >= threshold AND dispersion < 0.7 → BACKBONE (binds)
    freq >= threshold AND dispersion >= 0.7 → HUB (suppressed)

EXPLAINABILITY:
Every case formation decision is recorded in the constraint ledger with:
- Why each entity is classified as backbone/hub
- Which signals contributed to case binding
- The evidence supporting each decision

Usage:
    case_builder = CaseViewBuilder()
    cases = await case_builder.build_from_events(events, surfaces)
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Set, Optional, Any, Tuple

from ..types import Event, Surface, Constraint, ConstraintType, ConstraintLedger


@dataclass
class EntityRole:
    """Classification of an entity's structural role with explainability."""
    entity: str
    frequency: int  # Number of incidents containing this entity
    dispersion: float  # 0 = cohesive backbone, 1 = pure hub
    role: str  # "backbone", "hub", or "neutral"
    co_anchors: Set[str] = field(default_factory=set)
    cohesion_pairs: int = 0  # How many co-anchor pairs co-occur
    total_pairs: int = 0  # Total possible pairs

    # Explainability
    explanation: str = ""  # Human-readable explanation of classification
    evidence_incidents: List[str] = field(default_factory=list)  # Which incidents this entity appears in


@dataclass
class CaseEdge:
    """An edge between two events in a case, with explainability."""
    event1_id: str
    event2_id: str
    score: float  # Combined signal score
    signals: Dict[str, float] = field(default_factory=dict)  # Which signals contributed
    shared_backbones: Set[str] = field(default_factory=set)
    shared_relation_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    semantic_similarity: float = 0.0
    explanation: str = ""  # Human-readable explanation


@dataclass
class Case:
    """A story-level aggregation of incidents (L4) with explainability."""
    id: str
    backbone_entities: Set[str] = field(default_factory=set)  # All backbone entities
    primary_backbone: str = ""  # The strongest backbone
    event_ids: Set[str] = field(default_factory=set)  # Incident IDs
    surface_ids: Set[str] = field(default_factory=set)

    # Aggregated properties
    total_claims: int = 0
    total_sources: int = 0
    all_entities: Set[str] = field(default_factory=set)
    time_window: tuple = (None, None)

    # Backbone evidence
    backbone_score: float = 0.0  # Average (1 - dispersion) of backbones

    # Edge evidence (explainability)
    edges: List[CaseEdge] = field(default_factory=list)

    # Formation explanation
    formation_reason: str = ""

    # Narrative facets (different aspects of the case)
    facets: List[str] = field(default_factory=list)

    # Canonical presentation
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass
class CaseViewResult:
    """Result of case view building with full explainability."""
    cases: Dict[str, Case]
    entity_roles: Dict[str, EntityRole]  # All entity classifications
    relation_pairs: Dict[Tuple[str, str], int]  # Relation backbones (entity pairs that recur)
    ledger: ConstraintLedger
    stats: Dict[str, Any]


class CaseViewBuilder:
    """
    Builds Case structures from Incidents using dispersion-based backbone detection.

    Strategy (from REEE1.md section 8.2):
    1. Compute dispersion for all entities across incidents
    2. Classify entities as backbone/hub/neutral
    3. Identify relation backbones (entity pairs that recur)
    4. Compute semantic similarity between event centroids
    5. Form cases using 2-of-3 signal gate with core/periphery clustering
    6. Record all decisions for explainability
    """

    def __init__(
        self,
        min_incidents: int = 2,  # Minimum incidents for an entity to be considered
        min_claims: int = 3,     # Minimum total claims for a case
        hub_threshold: float = 0.7,  # Dispersion >= this = hub
        semantic_threshold: float = 0.5,  # Cosine similarity >= this = semantic signal
        core_threshold: float = 0.4,  # Edge score >= this = core edge
        periphery_threshold: float = 0.2,  # Edge score >= this = periphery attachment
    ):
        self.min_incidents = min_incidents
        self.min_claims = min_claims
        self.hub_threshold = hub_threshold
        self.semantic_threshold = semantic_threshold
        self.core_threshold = core_threshold
        self.periphery_threshold = periphery_threshold
        self.ledger = ConstraintLedger()

    async def build_from_events(
        self,
        events: Dict[str, Event],
        surfaces: Dict[str, Surface],
    ) -> CaseViewResult:
        """
        Build cases from events using 3-signal approach with explainability.
        """
        self.ledger = ConstraintLedger()

        # Step 1: Map entities to events and compute entity sets per event
        entity_to_events, event_entity_sets = self._map_entities_to_events(events, surfaces)

        # Step 2: Compute dispersion for each entity
        entity_roles = self._compute_entity_dispersion(entity_to_events, event_entity_sets)

        # Step 3: Identify backbone entities
        backbones = {
            e: r for e, r in entity_roles.items()
            if r.role == "backbone"
        }

        # Step 4: Identify relation backbones (entity pairs that recur across events)
        relation_pairs = self._compute_relation_backbones(event_entity_sets, entity_roles)

        # Step 5: Compute event centroids for semantic similarity
        event_centroids = self._compute_event_centroids(events, surfaces)

        # Step 6: Build case edges using 3 signals
        case_edges = self._compute_case_edges(
            events, event_entity_sets, backbones, relation_pairs, event_centroids
        )

        # Step 7: Form cases using core/periphery clustering on edges
        cases = self._form_cases_from_edges(
            case_edges, events, surfaces, backbones
        )

        # Step 8: Compute case properties
        for case in cases.values():
            self._compute_case_properties(case, events, surfaces)

        stats = {
            "events": len(events),
            "entities_analyzed": len(entity_roles),
            "backbones": len(backbones),
            "hubs": len([r for r in entity_roles.values() if r.role == "hub"]),
            "relation_pairs": len(relation_pairs),
            "case_edges": len(case_edges),
            "cases": len(cases),
            "avg_events_per_case": sum(len(c.event_ids) for c in cases.values()) / len(cases) if cases else 0,
            "constraints": len(self.ledger.constraints),
        }

        return CaseViewResult(
            cases=cases,
            entity_roles=entity_roles,
            relation_pairs=relation_pairs,
            ledger=self.ledger,
            stats=stats
        )

    def _map_entities_to_events(
        self,
        events: Dict[str, Event],
        surfaces: Dict[str, Surface],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Map each entity to the events it appears in, and events to their entities."""
        entity_to_events = defaultdict(set)
        event_entity_sets = {}

        for event in events.values():
            event_entities = set()
            for sid in event.surface_ids:
                s = surfaces.get(sid)
                if s:
                    event_entities.update(s.entities)

            event_entity_sets[event.id] = event_entities

            for entity in event_entities:
                entity_to_events[entity].add(event.id)

        return entity_to_events, event_entity_sets

    def _compute_entity_dispersion(
        self,
        entity_to_events: Dict[str, Set[str]],
        event_entity_sets: Dict[str, Set[str]],
    ) -> Dict[str, EntityRole]:
        """
        Compute dispersion for each entity with explanations.

        DISPERSION FORMULA:
            co_anchors = {b : b co-occurs with a in some incident}
            cohesion = |{(b1,b2) : b1,b2 co-occur}| / |all pairs|
            dispersion = 1 - cohesion

        Low dispersion = backbone (co-anchors form cohesive cluster)
        High dispersion = hub (co-anchors are disjoint)
        """
        entity_roles = {}

        for entity, event_ids in entity_to_events.items():
            freq = len(event_ids)

            if freq < self.min_incidents:
                entity_roles[entity] = EntityRole(
                    entity=entity,
                    frequency=freq,
                    dispersion=0.0,
                    role="neutral",
                    explanation=f"Appears in only {freq} incident(s), below threshold of {self.min_incidents}"
                )
                continue

            # Find all co-anchors
            co_anchors = set()
            for eid in event_ids:
                entities = event_entity_sets.get(eid, set())
                co_anchors.update(entities - {entity})

            if len(co_anchors) < 2:
                entity_roles[entity] = EntityRole(
                    entity=entity,
                    frequency=freq,
                    dispersion=0.0,
                    role="backbone",
                    co_anchors=co_anchors,
                    evidence_incidents=list(event_ids),
                    explanation=f"Only {len(co_anchors)} co-anchor(s), defaulting to backbone"
                )
                continue

            # Count how many co-anchor pairs co-occur in some event
            co_anchor_list = list(co_anchors)
            total_pairs = len(co_anchor_list) * (len(co_anchor_list) - 1) // 2
            cohesion_pairs = 0

            for i, a1 in enumerate(co_anchor_list):
                for a2 in co_anchor_list[i+1:]:
                    a1_events = entity_to_events.get(a1, set())
                    a2_events = entity_to_events.get(a2, set())
                    if a1_events & a2_events:
                        cohesion_pairs += 1

            cohesion = cohesion_pairs / total_pairs if total_pairs > 0 else 0.0
            dispersion = 1.0 - cohesion

            # Classify and generate explanation
            if dispersion >= self.hub_threshold:
                role = "hub"
                explanation = (
                    f"HUB: {freq} incidents, dispersion={dispersion:.2f} >= {self.hub_threshold}. "
                    f"Co-anchors are disjoint ({cohesion_pairs}/{total_pairs} pairs co-occur). "
                    f"This entity bridges unrelated contexts and should NOT bind cases."
                )
            else:
                role = "backbone"
                explanation = (
                    f"BACKBONE: {freq} incidents, dispersion={dispersion:.2f} < {self.hub_threshold}. "
                    f"Co-anchors form cohesive cluster ({cohesion_pairs}/{total_pairs} pairs co-occur). "
                    f"This entity binds related incidents into a case."
                )

            entity_roles[entity] = EntityRole(
                entity=entity,
                frequency=freq,
                dispersion=dispersion,
                role=role,
                co_anchors=co_anchors,
                cohesion_pairs=cohesion_pairs,
                total_pairs=total_pairs,
                evidence_incidents=list(event_ids),
                explanation=explanation
            )

            # Log to constraint ledger
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"Entity '{entity}' classified as {role}",
                evidence={
                    "entity": entity,
                    "frequency": freq,
                    "dispersion": round(dispersion, 3),
                    "cohesion": round(cohesion, 3),
                    "co_anchors_count": len(co_anchors),
                    "cohesion_pairs": cohesion_pairs,
                    "total_pairs": total_pairs,
                    "threshold": self.hub_threshold,
                    "explanation": explanation,
                },
                provenance="dispersion_classification"
            ), scope=entity)

        return entity_roles

    def _compute_relation_backbones(
        self,
        event_entity_sets: Dict[str, Set[str]],
        entity_roles: Dict[str, EntityRole],
    ) -> Dict[Tuple[str, str], int]:
        """
        Identify relation backbones: entity pairs that recur across events.

        These represent stable relationships like:
        - Jimmy Lai ↔ Apple Daily
        - Do Kwon ↔ Terraform Labs
        """
        pair_counts = defaultdict(int)
        pair_events = defaultdict(set)

        for event_id, entities in event_entity_sets.items():
            # Only consider non-hub entities for relation pairs
            valid_entities = [
                e for e in entities
                if entity_roles.get(e, EntityRole(e, 0, 0.0, "neutral")).role != "hub"
            ]

            for e1, e2 in combinations(sorted(valid_entities), 2):
                pair_counts[(e1, e2)] += 1
                pair_events[(e1, e2)].add(event_id)

        # Filter to pairs appearing in 2+ events
        relation_pairs = {
            pair: count for pair, count in pair_counts.items()
            if count >= self.min_incidents
        }

        # Log significant relation pairs
        for pair, count in relation_pairs.items():
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"Relation backbone: {pair[0]} ↔ {pair[1]}",
                evidence={
                    "entity1": pair[0],
                    "entity2": pair[1],
                    "occurrence_count": count,
                    "events": list(pair_events[pair]),
                },
                provenance="relation_backbone"
            ), scope=f"rel:{pair[0]}:{pair[1]}")

        return relation_pairs

    def _compute_event_centroids(
        self,
        events: Dict[str, Event],
        surfaces: Dict[str, Surface],
    ) -> Dict[str, List[float]]:
        """Compute centroid embeddings for each event from its surfaces."""
        event_centroids = {}

        for event in events.values():
            embeddings = []
            for sid in event.surface_ids:
                s = surfaces.get(sid)
                if s and s.centroid:
                    embeddings.append(s.centroid)

            if embeddings:
                # Average the embeddings
                dim = len(embeddings[0])
                centroid = [0.0] * dim
                for emb in embeddings:
                    for i, v in enumerate(emb):
                        centroid[i] += v
                centroid = [v / len(embeddings) for v in centroid]
                event_centroids[event.id] = centroid

        return event_centroids

    def _compute_case_edges(
        self,
        events: Dict[str, Event],
        event_entity_sets: Dict[str, Set[str]],
        backbones: Dict[str, EntityRole],
        relation_pairs: Dict[Tuple[str, str], int],
        event_centroids: Dict[str, List[float]],
    ) -> List[CaseEdge]:
        """
        Compute case edges between event pairs using 3 signals.

        SIGNALS:
        1. Backbone overlap (shared backbone entity)
        2. Relation backbone (shared entity pair)
        3. Semantic similarity (centroid cosine similarity)
        """
        edges = []
        event_ids = list(events.keys())

        for i, e1_id in enumerate(event_ids):
            for e2_id in event_ids[i+1:]:
                e1_entities = event_entity_sets.get(e1_id, set())
                e2_entities = event_entity_sets.get(e2_id, set())

                signals = {}
                shared_backbones = set()
                shared_relations = set()
                semantic_sim = 0.0

                # Signal 1: Backbone overlap
                for entity in e1_entities & e2_entities:
                    if entity in backbones:
                        shared_backbones.add(entity)
                if shared_backbones:
                    # Score based on number of shared backbones and their strength
                    backbone_scores = [1 - backbones[b].dispersion for b in shared_backbones]
                    signals["backbone_overlap"] = max(backbone_scores)

                # Signal 2: Relation backbone
                for pair, count in relation_pairs.items():
                    e1_has_pair = pair[0] in e1_entities and pair[1] in e1_entities
                    e2_has_pair = pair[0] in e2_entities and pair[1] in e2_entities
                    if e1_has_pair and e2_has_pair:
                        shared_relations.add(pair)
                if shared_relations:
                    # Score based on relation frequency
                    signals["relation_backbone"] = min(1.0, len(shared_relations) * 0.3)

                # Signal 3: Semantic similarity
                c1 = event_centroids.get(e1_id)
                c2 = event_centroids.get(e2_id)
                if c1 and c2:
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(c1, c2))
                    mag1 = sum(a * a for a in c1) ** 0.5
                    mag2 = sum(b * b for b in c2) ** 0.5
                    if mag1 > 0 and mag2 > 0:
                        semantic_sim = dot / (mag1 * mag2)
                        if semantic_sim >= self.semantic_threshold:
                            signals["semantic_similarity"] = semantic_sim

                # Apply 2-of-3 gate
                signal_count = len(signals)
                if signal_count >= 2:
                    # Combined score (average of active signals)
                    score = sum(signals.values()) / len(signals)

                    # Generate explanation
                    signal_parts = []
                    if "backbone_overlap" in signals:
                        signal_parts.append(f"shared backbones: {shared_backbones}")
                    if "relation_backbone" in signals:
                        signal_parts.append(f"shared relations: {[f'{p[0]}↔{p[1]}' for p in shared_relations]}")
                    if "semantic_similarity" in signals:
                        signal_parts.append(f"semantic similarity: {semantic_sim:.2f}")

                    explanation = f"2-of-3 gate passed with {signal_count} signals: {'; '.join(signal_parts)}"

                    edge = CaseEdge(
                        event1_id=e1_id,
                        event2_id=e2_id,
                        score=score,
                        signals=signals,
                        shared_backbones=shared_backbones,
                        shared_relation_pairs=shared_relations,
                        semantic_similarity=semantic_sim,
                        explanation=explanation
                    )
                    edges.append(edge)

                    # Log edge
                    self.ledger.add(Constraint(
                        constraint_type=ConstraintType.STRUCTURAL,
                        assertion=f"Case edge: {e1_id} ↔ {e2_id}",
                        evidence={
                            "events": [e1_id, e2_id],
                            "score": round(score, 3),
                            "signals": {k: round(v, 3) for k, v in signals.items()},
                            "shared_backbones": list(shared_backbones),
                            "shared_relations": [list(p) for p in shared_relations],
                            "semantic_similarity": round(semantic_sim, 3),
                            "explanation": explanation,
                        },
                        provenance="case_edge"
                    ), scope=f"edge:{e1_id}:{e2_id}")

        return edges

    def _form_cases_from_edges(
        self,
        edges: List[CaseEdge],
        events: Dict[str, Event],
        surfaces: Dict[str, Surface],
        backbones: Dict[str, EntityRole],
    ) -> Dict[str, Case]:
        """
        Form cases using core/periphery clustering on case edges.

        Core formation: Strong edges (score >= core_threshold) define case cores
        Periphery: Weaker edges attach but don't merge cores
        """
        # Build adjacency for core edges only
        core_adj = defaultdict(set)
        edge_lookup = {}

        for edge in edges:
            if edge.score >= self.core_threshold:
                core_adj[edge.event1_id].add(edge.event2_id)
                core_adj[edge.event2_id].add(edge.event1_id)
            edge_lookup[(edge.event1_id, edge.event2_id)] = edge
            edge_lookup[(edge.event2_id, edge.event1_id)] = edge

        # Find connected components on core graph
        visited = set()
        cases = {}
        case_idx = 0

        for event_id in events:
            if event_id in visited:
                continue
            if event_id not in core_adj:
                continue

            # BFS for component
            component = set()
            component_edges = []
            queue = [event_id]

            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)

                for neighbor in core_adj[curr]:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        # Record edge
                        edge = edge_lookup.get((curr, neighbor))
                        if edge:
                            component_edges.append(edge)

            if len(component) < self.min_incidents:
                continue

            # Calculate total claims
            total_claims = sum(
                events[eid].total_claims for eid in component if eid in events
            )
            if total_claims < self.min_claims:
                continue

            # Find all backbones in this case
            case_backbones = set()
            for eid in component:
                for sid in events[eid].surface_ids:
                    s = surfaces.get(sid)
                    if s:
                        for entity in s.entities:
                            if entity in backbones:
                                case_backbones.add(entity)

            # Find primary backbone (most frequent in case)
            backbone_freq = defaultdict(int)
            for eid in component:
                for sid in events[eid].surface_ids:
                    s = surfaces.get(sid)
                    if s:
                        for entity in s.entities:
                            if entity in backbones:
                                backbone_freq[entity] += 1

            primary_backbone = max(backbone_freq.items(), key=lambda x: x[1])[0] if backbone_freq else ""

            # Collect surfaces
            surface_ids = set()
            for eid in component:
                event = events.get(eid)
                if event:
                    surface_ids.update(event.surface_ids)

            # Calculate backbone score
            if case_backbones:
                backbone_score = sum(1 - backbones[b].dispersion for b in case_backbones) / len(case_backbones)
            else:
                backbone_score = 0.0

            # Formation explanation
            signal_summary = defaultdict(int)
            for edge in component_edges:
                for signal in edge.signals:
                    signal_summary[signal] += 1

            formation_reason = (
                f"Case formed from {len(component)} incidents connected by {len(component_edges)} core edges. "
                f"Signals: {dict(signal_summary)}. "
                f"Primary backbone: {primary_backbone} (dispersion={backbones[primary_backbone].dispersion:.2f})." if primary_backbone else ""
            )

            case_id = f"C{case_idx:03d}"
            case = Case(
                id=case_id,
                backbone_entities=case_backbones,
                primary_backbone=primary_backbone,
                event_ids=component.copy(),
                surface_ids=surface_ids,
                backbone_score=backbone_score,
                edges=component_edges,
                formation_reason=formation_reason,
            )

            cases[case_id] = case
            case_idx += 1

            # Log case formation
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"Case '{case_id}' formed around '{primary_backbone}'",
                evidence={
                    "case_id": case_id,
                    "primary_backbone": primary_backbone,
                    "all_backbones": list(case_backbones),
                    "event_count": len(component),
                    "events": list(component),
                    "edge_count": len(component_edges),
                    "backbone_score": round(backbone_score, 3),
                    "formation_reason": formation_reason,
                },
                provenance="case_formation"
            ), scope=case_id)

        return cases

    def _compute_case_properties(
        self,
        case: Case,
        events: Dict[str, Event],
        surfaces: Dict[str, Surface],
    ):
        """Compute derived properties for a case."""
        all_entities = set()
        all_sources = set()
        total_claims = 0
        times = []

        for eid in case.event_ids:
            event = events.get(eid)
            if not event:
                continue

            total_claims += event.total_claims

            for sid in event.surface_ids:
                s = surfaces.get(sid)
                if s:
                    all_entities.update(s.entities)
                    all_sources.update(s.sources)
                    if s.time_window[0]:
                        times.append(s.time_window[0])
                    if s.time_window[1]:
                        times.append(s.time_window[1])

        case.total_claims = total_claims
        case.total_sources = len(all_sources)
        case.all_entities = all_entities

        if times:
            try:
                case.time_window = (min(times), max(times))
            except TypeError:
                pass

        # Generate title from backbones
        if case.backbone_entities:
            backbone_list = sorted(case.backbone_entities)[:3]
            if len(backbone_list) == 1:
                case.title = f"{backbone_list[0]}: Ongoing Story"
            else:
                case.title = f"{backbone_list[0]} & {backbone_list[1]}: Related Story"
        else:
            case.title = "Unnamed Case"

        case.description = (
            f"Story spanning {len(case.event_ids)} incidents "
            f"with {total_claims} claims from {len(all_sources)} sources. "
            f"Backbone entities: {', '.join(sorted(case.backbone_entities)[:5])}"
        )
