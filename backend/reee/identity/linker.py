"""
Identity Linker: L0 -> L2 Claim Relationship Detection
======================================================

Two-path design:
1. Claims WITH question_key -> bucket lookup -> rule-based classification
2. Claims WITHOUT question_key -> embedding gate -> LLM classification

Embedding is ONLY for GATING (reducing comparisons), never for classification.
question_key is the L1 indexing primitive.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from ..types import Claim, Relation, Parameters, Surface
from .question_key import extract_question_key, classify_within_bucket


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between embeddings."""
    if not a or not b:
        return 0.0
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class IdentityLinker:
    """
    Links claims via identity relationships for L2 surface formation.

    Clean two-path design:
    - Path A: Claims WITH question_key -> bucket -> anchor check -> rule-based
    - Path B: Claims WITHOUT question_key -> embedding gate -> LLM
    """

    def __init__(self, llm: 'AsyncOpenAI' = None, params: Parameters = None):
        self.llm = llm
        self.params = params or Parameters()

        # Claim storage
        self.claims: Dict[str, Claim] = {}

        # Question Key Index (q1/q2 pattern)
        self.question_index: Dict[str, List[str]] = {}

        # Identity edges (claim-to-claim)
        self.edges: List[Tuple[str, str, Relation, float]] = []

    async def add_claim(
        self,
        claim: Claim,
        extract_qkey: bool = True
    ) -> Dict:
        """
        Add claim and compute relationships to existing claims.

        Returns: {claim_id, question_key, relations: [...]}
        """
        # Store claim
        self.claims[claim.id] = claim

        # Step 1: Extract question_key if not already set
        if extract_qkey and not claim.question_key:
            await extract_question_key(claim, self.llm)

        # Step 2: Index by question_key
        if claim.question_key:
            if claim.question_key not in self.question_index:
                self.question_index[claim.question_key] = []
            self.question_index[claim.question_key].append(claim.id)

        relations_found = []

        # =====================================================================
        # PATH A: Claims WITH question_key -> bucket -> anchor check -> rule-based
        # =====================================================================
        if claim.question_key:
            bucket_claim_ids = self.question_index.get(claim.question_key, [])
            candidates = [
                self.claims[cid] for cid in bucket_claim_ids
                if cid != claim.id and cid in self.claims
            ]

            for other in candidates:
                # ANCHOR CHECK: Claims must share anchors or have high entity overlap
                shared_anchors = claim.anchor_entities & other.anchor_entities
                shared_entities = claim.entities & other.entities

                if not shared_anchors and len(shared_entities) < 2:
                    continue  # Different events, skip

                # Within-bucket classification (rule-based, no LLM needed)
                relation, confidence, reasoning = classify_within_bucket(claim, other)

                if confidence >= self.params.identity_confidence_threshold:
                    self.edges.append((claim.id, other.id, relation, confidence))
                    relations_found.append({
                        'other_id': other.id,
                        'relation': relation.value,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'method': 'question_key'
                    })

        # =====================================================================
        # PATH B: Claims WITHOUT question_key -> embedding gate -> LLM
        # =====================================================================
        else:
            for other_id, other in self.claims.items():
                if other_id == claim.id:
                    continue

                # Embedding gate (reduces comparisons, not classification)
                if not self._should_compare(claim, other):
                    continue

                # LLM classification
                relation, confidence, reasoning = await self._classify_llm(claim, other)

                if relation != Relation.UNRELATED and confidence >= self.params.identity_confidence_threshold:
                    self.edges.append((claim.id, other_id, relation, confidence))
                    relations_found.append({
                        'other_id': other_id,
                        'relation': relation.value,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'method': 'embedding_llm'
                    })

        return {
            'claim_id': claim.id,
            'question_key': claim.question_key,
            'extracted_value': claim.extracted_value,
            'relations': relations_found
        }

    def _affinity(self, claim: Claim, other: Claim) -> float:
        """Compute affinity score (no LLM)."""
        semantic = cosine_similarity(claim.embedding, other.embedding)
        entity = jaccard(claim.entities, other.entities)
        anchor = 1.0 if (claim.anchor_entities & other.anchor_entities) else 0.0
        return 0.5 * semantic + 0.3 * entity + 0.2 * anchor

    def _should_compare(self, claim: Claim, other: Claim, threshold: float = 0.15) -> bool:
        """Gate: should we call the expensive intrinsic check?"""
        if claim.id == other.id:
            return False

        # Low affinity threshold
        if self._affinity(claim, other) < threshold:
            return False

        # Anchor gate only at very high semantic (prevent obvious bleeding)
        semantic = cosine_similarity(claim.embedding, other.embedding)
        if semantic > 0.85:
            has_anchors = bool(claim.anchor_entities or other.anchor_entities)
            if has_anchors and not (claim.anchor_entities & other.anchor_entities):
                return False

        return True

    async def _classify_llm(
        self,
        claim: Claim,
        other: Claim
    ) -> Tuple[Relation, float, str]:
        """LLM classification of claim relationship."""
        if not self.llm:
            # Fallback: use affinity as heuristic
            aff = self._affinity(claim, other)
            if aff > 0.8:
                return Relation.CONFIRMS, aff, "High affinity (no LLM)"
            return Relation.UNRELATED, 1.0 - aff, "Low affinity (no LLM)"

        prompt = f"""Compare these two claims and determine their epistemic relationship.

CLAIM A: "{claim.text}"
SOURCE A: {claim.source}
ENTITIES A: {list(claim.entities)[:5]}

CLAIM B: "{other.text}"
SOURCE B: {other.source}
ENTITIES B: {list(other.entities)[:5]}

STEP 1: Do both claims refer to the SAME real-world fact/event?
- Look for shared specific entities (people, places, organizations)
- Generic terms like "officials" or "Hong Kong" are not sufficient
- Must be the SAME incident, not just same topic

STEP 2: If same fact, what is the relationship?
- CONFIRMS: Same claim stated by different source
- REFINES: One adds detail to the other
- SUPERSEDES: One updates/corrects the other (with temporal marker)
- CONFLICTS: They contradict each other

If NOT same fact:
- UNRELATED: Different events or unclear connection

Return JSON:
{{
  "same_fact": true/false,
  "shared_entities": ["list of shared specific entities"],
  "relation": "CONFIRMS|REFINES|SUPERSEDES|CONFLICTS|UNRELATED",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence explanation"
}}"""

        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)

            rel_str = result.get('relation', 'UNRELATED').upper()
            relation = Relation[rel_str] if rel_str in Relation.__members__ else Relation.UNRELATED
            confidence = float(result.get('confidence', 0.5))
            reasoning = result.get('reasoning', '')

            return relation, confidence, reasoning

        except Exception as e:
            return Relation.UNRELATED, 0.0, f"Error: {e}"

    def compute_surfaces(self) -> Dict[str, Surface]:
        """
        Compute surfaces from IDENTITY edges only (connected components).

        Surfaces represent same-fact sub-topologies. Only Tier-1 kernel
        relations are used: CONFIRMS, REFINES, SUPERSEDES, CONFLICTS.
        """
        from collections import defaultdict

        # Build adjacency from identity relations ONLY
        adj = defaultdict(set)
        for c1, c2, rel, conf in self.edges:
            if rel in (Relation.CONFIRMS, Relation.REFINES, Relation.SUPERSEDES, Relation.CONFLICTS):
                adj[c1].add(c2)
                adj[c2].add(c1)

        # Find connected components
        visited = set()
        components = []

        for claim_id in self.claims:
            if claim_id in visited:
                continue

            component = set()
            stack = [claim_id]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                stack.extend(adj[curr] - visited)

            components.append(component)

        # Create surfaces
        surfaces = {}
        for i, component in enumerate(components):
            surface_id = f"S{i+1:03d}"

            claims = [self.claims[cid] for cid in component]
            edges = [(c1, c2, rel) for c1, c2, rel, _ in self.edges
                     if c1 in component and c2 in component]

            surface = self._surface_from_claims(surface_id, claims, edges)
            surfaces[surface_id] = surface

        return surfaces

    def _surface_from_claims(
        self,
        surface_id: str,
        claims: List[Claim],
        edges: List[Tuple[str, str, Relation]]
    ) -> Surface:
        """Compute surface properties from claims."""
        if not claims:
            return Surface(id=surface_id)

        # Centroid (average embedding)
        embeddings = [c.embedding for c in claims if c.embedding]
        centroid = np.mean(embeddings, axis=0).tolist() if embeddings else None

        # Entropy (semantic variance)
        if centroid and len(embeddings) > 1:
            distances = [1 - cosine_similarity(e, centroid) for e in embeddings]
            entropy = float(np.mean(distances))
        else:
            entropy = 0.0

        # Mass (source count)
        sources = set(c.source for c in claims if c.source)
        mass = len(sources)

        # Entities
        entities = set()
        anchor_entities = set()
        for c in claims:
            entities.update(c.entities)
            anchor_entities.update(c.anchor_entities)

        # Time window
        timestamps = [c.timestamp for c in claims if c.timestamp]
        time_window = (min(timestamps), max(timestamps)) if timestamps else (None, None)

        return Surface(
            id=surface_id,
            claim_ids={c.id for c in claims},
            centroid=centroid,
            entropy=entropy,
            mass=mass,
            sources=sources,
            entities=entities,
            anchor_entities=anchor_entities,
            time_window=time_window,
            internal_edges=edges
        )
