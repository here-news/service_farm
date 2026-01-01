"""
Relation Backbone for CaseView
==============================

Entity relations as a signal for case-level binding.

Relations are propositions about entity connections:
- "Do Kwon founded Terraform Labs"
- "Jimmy Lai owns Apple Daily"

These relations, when corroborated, provide a backbone for case formation:
- Related anchors count as one signal in CaseView scoring
- 1-hop only (direct relations, not transitive)
- Require corroboration threshold before contributing to cores

Key constraint: Relations must be contestable (surfaces with belief state)
before they contribute to core formation. For now, we use a simpler
corroboration count as a proxy.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict


@dataclass
class RelationEdge:
    """A single relation between two entities."""
    entity1: str
    entity2: str
    relation_type: str  # e.g., "founded", "owns", "member_of"
    corroboration_count: int = 1  # Number of claims supporting this
    claim_ids: Set[str] = field(default_factory=set)

    @property
    def is_corroborated(self) -> bool:
        """Whether this relation has sufficient corroboration."""
        return self.corroboration_count >= 2


@dataclass
class RelationBackbone:
    """
    A graph of corroborated entity relations.

    Used by CaseView to add "related anchors" as a binding signal.
    """
    edges: Dict[Tuple[str, str], RelationEdge] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Stats
    total_relations: int = 0
    corroborated_relations: int = 0

    def add_relation(
        self,
        entity1: str,
        entity2: str,
        relation_type: str = "related",
        claim_id: Optional[str] = None,
    ):
        """Add or update a relation between two entities."""
        # Normalize order for undirected lookup
        key = tuple(sorted([entity1, entity2]))

        if key in self.edges:
            edge = self.edges[key]
            edge.corroboration_count += 1
            if claim_id:
                edge.claim_ids.add(claim_id)
        else:
            self.edges[key] = RelationEdge(
                entity1=key[0],
                entity2=key[1],
                relation_type=relation_type,
                corroboration_count=1,
                claim_ids={claim_id} if claim_id else set(),
            )
            self.total_relations += 1

        # Update adjacency
        self.adjacency[entity1].add(entity2)
        self.adjacency[entity2].add(entity1)

    def get_related(self, entity: str, min_corroboration: int = 1) -> Set[str]:
        """Get entities related to this entity (1-hop only)."""
        related = set()
        for other in self.adjacency.get(entity, set()):
            key = tuple(sorted([entity, other]))
            edge = self.edges.get(key)
            if edge and edge.corroboration_count >= min_corroboration:
                related.add(other)
        return related

    def are_related(
        self,
        entity1: str,
        entity2: str,
        min_corroboration: int = 1
    ) -> bool:
        """Check if two entities are related (1-hop)."""
        key = tuple(sorted([entity1, entity2]))
        edge = self.edges.get(key)
        return edge is not None and edge.corroboration_count >= min_corroboration

    def find_related_pairs(
        self,
        anchors1: Set[str],
        anchors2: Set[str],
        min_corroboration: int = 1,
    ) -> Set[Tuple[str, str]]:
        """Find pairs of related anchors between two sets."""
        pairs = set()
        for a1 in anchors1:
            related = self.get_related(a1, min_corroboration)
            for a2 in anchors2:
                if a2 in related:
                    pairs.add((a1, a2))
        return pairs

    def finalize(self):
        """Compute final statistics."""
        self.corroborated_relations = sum(
            1 for e in self.edges.values() if e.is_corroborated
        )


def build_relation_backbone_from_incidents(
    incidents: Dict[str, Any],
    min_co_occurrence: int = 2,
) -> RelationBackbone:
    """
    Build a relation backbone from incident anchor co-occurrence.

    This is a simple heuristic: if two anchors appear together in
    multiple incidents, they're probably related.

    For a more principled approach, relations would come from:
    - Explicit relation extraction from text
    - Wikidata/knowledge base lookups
    - User annotations

    Args:
        incidents: Dict of incident_id -> Event
        min_co_occurrence: Min incidents where anchors must co-occur

    Returns:
        RelationBackbone with entity relations
    """
    backbone = RelationBackbone()

    # Count co-occurrences of anchor pairs
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_incidents: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for inc_id, inc in incidents.items():
        anchors = list(inc.anchor_entities)
        for i, a1 in enumerate(anchors):
            for a2 in anchors[i+1:]:
                key = tuple(sorted([a1, a2]))
                pair_counts[key] += 1
                pair_incidents[key].add(inc_id)

    # Add relations that meet threshold
    for (a1, a2), count in pair_counts.items():
        if count >= min_co_occurrence:
            backbone.add_relation(
                entity1=a1,
                entity2=a2,
                relation_type="co_occurs",
            )
            # Set corroboration to actual count
            key = tuple(sorted([a1, a2]))
            backbone.edges[key].corroboration_count = count

    backbone.finalize()
    return backbone


def print_backbone_report(backbone: RelationBackbone, top_k: int = 10):
    """Print a human-readable backbone report."""
    print("=" * 70)
    print("RELATION BACKBONE")
    print("=" * 70)
    print(f"Total relations: {backbone.total_relations}")
    print(f"Corroborated (count >= 2): {backbone.corroborated_relations}")
    print()

    # Sort by corroboration
    sorted_edges = sorted(
        backbone.edges.values(),
        key=lambda e: e.corroboration_count,
        reverse=True
    )

    print(f"Top {top_k} by corroboration:")
    print("-" * 70)
    for edge in sorted_edges[:top_k]:
        status = "✓" if edge.is_corroborated else " "
        print(
            f"  {status} {edge.entity1} <-> {edge.entity2}: "
            f"count={edge.corroboration_count}, type={edge.relation_type}"
        )
