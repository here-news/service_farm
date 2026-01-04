"""
Principled Surface Builder
==========================

Builds surfaces using motif-based clustering instead of embedding similarity.

This implements the principled emergence approach:
1. Each claim is a hyperedge over its entity set
2. Motifs are k-sets (k≥2) that appear in multiple claims
3. Surfaces are connected components of motif-sharing claims
4. All decisions are tracked in the constraint ledger

THEORY:
- Bianconi higher-order networks: claims are hyperedges, not pairwise
- Graded evidence: log(support+1) instead of hard thresholds
- Anti-trap rule: cores require ≥2 constraints, ≥1 non-semantic

Usage:
    builder = PrincipledSurfaceBuilder()
    surfaces, motifs, ledger = await builder.build_from_claims(claims)
"""

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional, Any

from ..types import (
    Claim, Surface, Constraint, ConstraintType, ConstraintLedger, Motif
)


@dataclass
class MotifConfig:
    """Configuration for motif detection."""
    min_k: int = 2  # Minimum motif size
    min_support: int = 2  # Minimum support threshold
    graded: bool = True  # Use graded evidence (log) instead of step function
    triangle_bonus: float = 1.5  # Extra weight for k>=3 motifs


@dataclass
class SurfaceBuilderResult:
    """Result of surface building."""
    surfaces: Dict[str, Surface]
    motifs: Dict[str, Motif]
    ledger: ConstraintLedger
    stats: Dict[str, Any]


class PrincipledSurfaceBuilder:
    """
    Builds surfaces using motif-based clustering.

    This replaces the embedding-similarity approach in weaver_worker.py
    with a principled higher-order network approach.
    """

    def __init__(self, config: MotifConfig = None):
        self.config = config or MotifConfig()
        self.ledger = ConstraintLedger()

    async def build_from_claims(
        self,
        claims: List[Claim],
        existing_surfaces: Dict[str, Surface] = None
    ) -> SurfaceBuilderResult:
        """
        Build surfaces from claims using motif clustering.

        Args:
            claims: List of claims to cluster
            existing_surfaces: Optional existing surfaces to extend

        Returns:
            SurfaceBuilderResult with surfaces, motifs, and ledger
        """
        self.ledger = ConstraintLedger()

        # Step 1: Form hyperedges (each claim is a hyperedge over its entities)
        hyperedges = self._form_hyperedges(claims)

        # Step 2: Detect motifs (k-sets with support ≥ min_support)
        motifs = self._detect_motifs(hyperedges)

        # Step 3: Form surfaces via connected components of motif-sharing
        surfaces = self._form_surfaces(claims, hyperedges, motifs)

        # Step 4: Compute surface properties
        for surface in surfaces.values():
            self._compute_surface_properties(surface, claims)

        stats = {
            "claims": len(claims),
            "hyperedges": len(hyperedges),
            "motifs": len(motifs),
            "surfaces": len(surfaces),
            "constraints": len(self.ledger.constraints),
            "config": {
                "min_k": self.config.min_k,
                "min_support": self.config.min_support,
                "graded": self.config.graded,
                "triangle_bonus": self.config.triangle_bonus,
            }
        }

        return SurfaceBuilderResult(
            surfaces=surfaces,
            motifs=motifs,
            ledger=self.ledger,
            stats=stats
        )

    def _form_hyperedges(
        self,
        claims: List[Claim]
    ) -> Dict[str, Set[str]]:
        """
        Form hyperedges from claims.

        Each claim becomes a hyperedge over its entity set.
        This is Bianconi's higher-order representation.
        """
        hyperedges = {}

        for claim in claims:
            if len(claim.entities) < 1:
                continue

            hyperedges[claim.id] = claim.entities.copy()

            # Add structural constraint
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"Entities {claim.entities} co-occur in claim",
                evidence={
                    "claim_id": claim.id,
                    "entity_count": len(claim.entities),
                    "entities": list(claim.entities)
                },
                provenance="hyperedge_formation"
            ), scope=claim.id)

        return hyperedges

    def _detect_motifs(
        self,
        hyperedges: Dict[str, Set[str]]
    ) -> Dict[str, Motif]:
        """
        Detect motifs (repeated k-sets) across hyperedges.

        A motif is a k-set (k ≥ min_k) that appears in multiple claims.
        Uses graded evidence (log) for robustness.
        """
        # Count all k-sets
        kset_counts = defaultdict(int)
        kset_claims = defaultdict(set)

        for claim_id, entities in hyperedges.items():
            entity_list = list(entities)
            n = len(entity_list)

            # Generate all k-subsets where k >= min_k
            for k in range(self.config.min_k, n + 1):
                for subset in combinations(entity_list, k):
                    kset = frozenset(subset)
                    kset_counts[kset] += 1
                    kset_claims[kset].add(claim_id)

        # Compute motif weights
        motifs = {}

        if self.config.graded:
            # Graded: log-evidence with k-bonus
            threshold = math.log(self.config.min_support + 1) * 0.5
            for kset, count in kset_counts.items():
                k = len(kset)
                k_bonus = self.config.triangle_bonus if k >= 3 else 1.0
                weight = math.log(count + 1) * k_bonus

                if weight >= threshold:
                    motif_id = f"mtf_{hash(kset) % 100000:05d}"
                    motifs[motif_id] = Motif(
                        id=motif_id,
                        entities=set(kset),
                        support=count,
                        weight=weight,
                        claim_ids=kset_claims[kset].copy()
                    )
        else:
            # Step function (original)
            for kset, count in kset_counts.items():
                if count >= self.config.min_support:
                    k = len(kset)
                    k_bonus = self.config.triangle_bonus if k >= 3 else 1.0
                    weight = count * k_bonus

                    motif_id = f"mtf_{hash(kset) % 100000:05d}"
                    motifs[motif_id] = Motif(
                        id=motif_id,
                        entities=set(kset),
                        support=count,
                        weight=weight,
                        claim_ids=kset_claims[kset].copy()
                    )

        # Add constraints for motifs
        for motif in motifs.values():
            self.ledger.add(Constraint(
                constraint_type=ConstraintType.STRUCTURAL,
                assertion=f"Motif {motif.entities} appears {motif.support} times",
                evidence={
                    "motif_id": motif.id,
                    "entities": list(motif.entities),
                    "support": motif.support,
                    "weight": motif.weight,
                    "k": motif.k,
                    "claims": list(motif.claim_ids)
                },
                provenance="motif_detection"
            ), scope=motif.id)

        return motifs

    def _form_surfaces(
        self,
        claims: List[Claim],
        hyperedges: Dict[str, Set[str]],
        motifs: Dict[str, Motif]
    ) -> Dict[str, Surface]:
        """
        Form surfaces via connected components of motif-sharing claims.

        Two claims are in the same surface if they share a supported motif.
        This replaces embedding similarity with structural evidence.

        NOTE: Only motifs with support >= 2 can form edges (they need 2+ claims).
        """
        # Build claim-claim edges based on shared motifs
        claim_edges = defaultdict(set)
        claim_motifs = defaultdict(set)  # Track which motifs each claim belongs to

        for motif in motifs.values():
            # Only use motifs with 2+ claims for surface formation
            if motif.support < 2:
                continue

            claims_with_motif = list(motif.claim_ids)

            for claim_id in claims_with_motif:
                claim_motifs[claim_id].add(motif.id)

            # Connect all pairs of claims that share this motif
            for i, c1 in enumerate(claims_with_motif):
                for c2 in claims_with_motif[i+1:]:
                    claim_edges[c1].add(c2)
                    claim_edges[c2].add(c1)

                    # Add constraint for this edge
                    pair_key = f"{min(c1, c2)}:{max(c1, c2)}"
                    self.ledger.add(Constraint(
                        constraint_type=ConstraintType.STRUCTURAL,
                        assertion=f"Claims share motif {motif.entities}",
                        evidence={
                            "motif_id": motif.id,
                            "motif_entities": list(motif.entities),
                            "support": motif.support
                        },
                        provenance="surface_formation"
                    ), scope=pair_key)

        # Find connected components via BFS
        visited = set()
        surfaces = {}
        surface_idx = 0

        claim_lookup = {c.id: c for c in claims}

        for claim_id in hyperedges:
            if claim_id in visited:
                continue

            # BFS to find component
            component = set()
            queue = [claim_id]

            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                queue.extend(claim_edges[curr] - visited)

            # Create surface from component
            surface_id = f"S{surface_idx:03d}"
            surface = Surface(
                id=surface_id,
                claim_ids=component,
                formation_method="motif",
                constraint_ledger=ConstraintLedger()
            )

            # Collect motif IDs for this surface
            for cid in component:
                surface.motif_ids.update(claim_motifs.get(cid, set()))

            # Collect entities
            for cid in component:
                if cid in hyperedges:
                    surface.entities.update(hyperedges[cid])

            surfaces[surface_id] = surface
            surface_idx += 1

        # Add surface-level motif constraints for surfaces that share motifs
        surface_ids = list(surfaces.keys())
        for i, s1_id in enumerate(surface_ids):
            for s2_id in surface_ids[i+1:]:
                s1 = surfaces[s1_id]
                s2 = surfaces[s2_id]

                # Check if surfaces share any motifs
                shared_motif_ids = s1.motif_ids & s2.motif_ids
                if shared_motif_ids:
                    pair_key = f"{s1_id}:{s2_id}"
                    for mid in shared_motif_ids:
                        motif = motifs.get(mid)
                        if motif:
                            self.ledger.add(Constraint(
                                constraint_type=ConstraintType.STRUCTURAL,
                                assertion=f"Surfaces share motif {motif.entities}",
                                evidence={
                                    "motif_id": mid,
                                    "motif_entities": list(motif.entities),
                                    "support": motif.support
                                },
                                provenance="surface_motif_sharing"
                            ), scope=pair_key)

        return surfaces

    def _compute_surface_properties(
        self,
        surface: Surface,
        claims: List[Claim]
    ):
        """
        Compute derived properties for a surface.
        """
        claim_lookup = {c.id: c for c in claims}

        # Collect sources and timestamps
        sources = set()
        times = []
        embeddings = []

        for cid in surface.claim_ids:
            claim = claim_lookup.get(cid)
            if claim:
                if claim.source:
                    sources.add(claim.source)

                # Time from event_time or timestamp
                t = claim.event_time or claim.timestamp
                if t:
                    # Normalize to datetime
                    if isinstance(t, str):
                        from datetime import datetime
                        try:
                            t = datetime.fromisoformat(t.replace('Z', '+00:00'))
                        except:
                            t = None
                    if t:
                        times.append(t)

                if claim.embedding:
                    embeddings.append(claim.embedding)

                # Anchor entities
                if claim.anchor_entities:
                    surface.anchor_entities.update(claim.anchor_entities)

        surface.sources = sources
        surface.mass = len(surface.claim_ids)

        # Generate canonical_title and key_facts from claim texts
        claim_texts = []
        for cid in surface.claim_ids:
            claim = claim_lookup.get(cid)
            if claim and claim.text:
                claim_texts.append(claim.text.strip())

        if claim_texts:
            # Pick the longest claim as canonical title (most descriptive)
            claim_texts.sort(key=len, reverse=True)
            # Truncate to reasonable length for title
            title = claim_texts[0]
            if len(title) > 150:
                title = title[:147] + "..."
            surface.canonical_title = title

            # Store top 3 distinct claims as key facts
            seen = set()
            for text in claim_texts[:10]:
                # Dedupe similar texts
                key = text[:50].lower()
                if key not in seen:
                    seen.add(key)
                    surface.key_facts.append(text[:200])
                    if len(surface.key_facts) >= 3:
                        break

        # Time window
        if times:
            # Ensure all times are comparable
            try:
                surface.time_window = (min(times), max(times))
            except TypeError:
                # Mixed types, skip time window
                pass

        # Centroid (average of embeddings)
        if embeddings:
            dim = len(embeddings[0])
            centroid = [0.0] * dim
            for emb in embeddings:
                for i, v in enumerate(emb):
                    centroid[i] += v
            centroid = [v / len(embeddings) for v in centroid]
            surface.centroid = centroid


# =============================================================================
# CONTEXT COMPATIBILITY (for event formation)
# =============================================================================

@dataclass
class ContextResult:
    """Result of context compatibility check."""
    entity: str
    compatible: bool
    underpowered: bool
    overlap: float
    companions1_size: int
    companions2_size: int
    reason: str


def context_compatible(
    entity: str,
    surface1: Surface,
    surface2: Surface,
    min_companions: int = 2,
    overlap_threshold: float = 0.15
) -> ContextResult:
    """
    Check if an entity's context is compatible between two surfaces.

    This prevents percolation by rejecting entities that bridge
    unrelated topics (disjoint companion sets).

    Returns:
        ContextResult with compatibility decision and evidence
    """
    companions1 = surface1.entities - {entity}
    companions2 = surface2.entities - {entity}

    c1_size = len(companions1)
    c2_size = len(companions2)

    # Check if underpowered
    if c1_size < min_companions or c2_size < min_companions:
        return ContextResult(
            entity=entity,
            compatible=False,
            underpowered=True,
            overlap=0.0,
            companions1_size=c1_size,
            companions2_size=c2_size,
            reason=f"underpowered: companions=({c1_size},{c2_size}), need >= {min_companions}"
        )

    # Jaccard overlap
    intersection = len(companions1 & companions2)
    union = len(companions1 | companions2)
    overlap = intersection / union if union > 0 else 0.0

    if overlap >= overlap_threshold:
        return ContextResult(
            entity=entity,
            compatible=True,
            underpowered=False,
            overlap=overlap,
            companions1_size=c1_size,
            companions2_size=c2_size,
            reason=f"compatible: overlap={overlap:.2f} >= {overlap_threshold}"
        )
    else:
        return ContextResult(
            entity=entity,
            compatible=False,
            underpowered=False,
            overlap=overlap,
            companions1_size=c1_size,
            companions2_size=c2_size,
            reason=f"blocked: overlap={overlap:.2f} < {overlap_threshold}"
        )
