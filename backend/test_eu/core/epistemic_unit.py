"""
Epistemic Unit: Fractal Architecture for Knowledge Emergence
=============================================================

Architecture:
  L0 ClaimObservation: raw, append-only, provenance (immutable)
  L1 Proposition: deduplicated facts, version chains, conflicts
  L2 Surface: bundles of claims connected by IDENTITY edges only
  L3 Event: groups surfaces by ABOUTNESS edges (soft, graded)
  L4 Narrative: temporal/causal/discourse edges between events
  L5 Meaning: frames, stakes, questions (interpretive layer)

INVARIANTS:

1. L0 immutability
   - Claims append-only, never deleted, never modified
   - Cold storage allowed, but lineage preserved

2. Parameter versioning
   - Parameters append-only (changes logged, not overwritten)
   - Each change attributed: actor, trigger, rationale
   - (L0, params@v) → deterministic L1-L5

3. Identity/Aboutness separation
   - L2 surfaces: IDENTITY edges only (CONFIRMS/REFINES/SUPERSEDES/CONFLICTS)
   - L3 events: ABOUTNESS edges only (between surfaces, not claims)
   - NEVER mix these graphs

4. Derived state purity
   - L1-L5 = f(L0, parameters)
   - No external mutation of derived layers
   - Recompute is the only update path

5. Stable core relations
   - {CONFIRMS, REFINES, SUPERSEDES, CONFLICTS, UNRELATED}
   - Domain differences via extraction, not new relations

6. Meta-claims are observations
   - Emitted about epistemic state (tension, gaps, conflicts)
   - May trigger ParameterChange or new L0 claims
   - Never directly injected as world-claims
"""

import json
import numpy as np
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Set, Optional, Tuple, Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI


# =============================================================================
# PARAMETER VERSIONING (Invariant 2)
# =============================================================================

@dataclass
class ParameterChange:
    """
    System action that affects L1-L5 computation.

    Parameters are versioned and attributed because they change
    derived layer outcomes without new evidence. Treat like
    "system actions" with provenance for reproducibility.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # What changed
    parameter: str = ""             # e.g., "identity_threshold", "hub_max_df"
    old_value: Any = None
    new_value: Any = None

    # Provenance (who/what/why)
    actor: str = "system"           # "system:tension_detector", "human:operator@xyz"
    trigger: Optional[str] = None   # meta-claim ID that prompted this, if any
    rationale: str = ""             # human-readable explanation

    # Reproducibility
    topology_version: Optional[str] = None  # snapshot ID before change
    affects_layers: List[str] = field(default_factory=list)  # ["L2", "L3"]


@dataclass
class Parameters:
    """
    Versioned parameter set for epistemic computation.

    All derived state (L1-L5) is deterministic given (L0, params@version).
    """
    version: int = 1

    # L2 Surface formation (identity edges)
    identity_confidence_threshold: float = 0.5  # min confidence for identity relation

    # L3 Event formation (aboutness edges)
    hub_max_df: int = 3             # anchors in more surfaces than this get zero weight
    aboutness_min_signals: int = 2   # require N of 3 signals for aboutness edge
    aboutness_threshold: float = 0.35  # min score to link surfaces into events

    # Tension detection
    high_entropy_threshold: float = 0.6  # surface entropy above this triggers meta-claim

    # History
    changes: List[ParameterChange] = field(default_factory=list)

    def update(
        self,
        parameter: str,
        new_value: Any,
        actor: str = "system",
        trigger: Optional[str] = None,
        rationale: str = ""
    ) -> ParameterChange:
        """Update a parameter with full provenance tracking."""
        old_value = getattr(self, parameter, None)

        change = ParameterChange(
            parameter=parameter,
            old_value=old_value,
            new_value=new_value,
            actor=actor,
            trigger=trigger,
            rationale=rationale,
            topology_version=f"v{self.version}",
            affects_layers=self._affected_layers(parameter)
        )

        setattr(self, parameter, new_value)
        self.version += 1
        self.changes.append(change)

        return change

    def _affected_layers(self, parameter: str) -> List[str]:
        """Determine which layers are affected by a parameter change."""
        if parameter.startswith("identity"):
            return ["L2", "L3"]  # surfaces affect events
        elif parameter.startswith("aboutness") or parameter == "hub_max_df":
            return ["L3"]
        elif parameter.startswith("high_entropy"):
            return []  # only affects meta-claims
        return ["L2", "L3"]


# =============================================================================
# META-CLAIMS (Invariant 6)
# =============================================================================

MetaClaimType = Literal[
    "high_stakes_low_evidence",  # → verification bounty
    "unresolved_conflict",        # → adjudication task
    "single_source_only",         # → corroboration request
    "high_entropy_surface",       # → investigation prompt
    "bridge_node_detected",       # → potential split candidate
    "stale_event",                # → decay candidate
]

@dataclass
class MetaClaim:
    """
    Observation about the epistemic state itself.

    These are NOT truth claims about the world. They are observations
    about the topology that may trigger operational actions:
    - ParameterChange (adjust thresholds)
    - New L0 claims (verification, corroboration)
    - Task generation (bounties, investigations)

    Meta-claims flow OUT of the core as observations.
    They are consumed by operational layers.
    They are NEVER injected back as world-claims.
    """
    id: str = field(default_factory=lambda: f"mc_{uuid.uuid4().hex[:8]}")
    type: MetaClaimType = "high_entropy_surface"
    target_id: str = ""           # node/surface/event ID
    target_type: str = "surface"  # "claim", "surface", "event"

    # Evidence for this meta-claim
    evidence: Dict = field(default_factory=dict)

    # When generated
    generated_at: datetime = field(default_factory=datetime.utcnow)
    params_version: int = 1       # parameter version when generated

    # Resolution tracking
    resolved: bool = False
    resolution: Optional[str] = None  # "parameter_updated", "new_claim_added", "dismissed"


# =============================================================================
# RELATIONS
# =============================================================================

class Relation(Enum):
    """Level 0 relations (claim-to-claim only)."""
    CONFIRMS = "confirms"       # Same fact, different source
    REFINES = "refines"         # Adds detail to same fact
    SUPERSEDES = "supersedes"   # Updates/corrects prior claim
    CONFLICTS = "conflicts"     # Contradicts existing claim
    UNRELATED = "unrelated"     # Different facts


class Association(Enum):
    """Higher-level associations (surface-to-surface, event-to-event)."""
    SAME = "same"               # Identity: should merge
    RELATED = "related"         # Association: edge only
    DISTINCT = "distinct"       # No connection


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def temporal_overlap(t1: Tuple[datetime, datetime], t2: Tuple[datetime, datetime]) -> float:
    """Overlap coefficient for time windows."""
    if not t1 or not t2 or not t1[0] or not t2[0]:
        return 0.0

    start = max(t1[0], t2[0])
    end = min(t1[1], t2[1])

    if start > end:
        return 0.0  # No overlap

    overlap = (end - start).total_seconds()
    span1 = (t1[1] - t1[0]).total_seconds() or 1
    span2 = (t2[1] - t2[0]).total_seconds() or 1

    return overlap / min(span1, span2)


# =============================================================================
# LEVEL 0: CLAIM (Atomic, Intrinsic Relationships)
# =============================================================================

@dataclass
class Claim:
    """
    L0: Atomic epistemic unit. Append-only, immutable.

    The claim itself knows how to relate to other claims.
    LLM provides the classification rigor.

    Question Key (q1/q2 pattern):
        question_key identifies WHICH QUESTION this claim answers.
        Claims only relate (CONFIRMS/SUPERSEDES/CONFLICTS) if they
        answer the SAME question. This is the indexing primitive
        for L1 proposition formation and version chains.

        Examples:
          "13 dead" → question_key="death_count", value=13
          "Fire on Floor 8" → question_key="origin_location", value="Floor 8"
          "Started at 3am" → question_key="start_time", value="3am"
    """
    id: str
    text: str
    source: str
    embedding: Optional[List[float]] = None
    entities: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)  # PERSON, ORG
    timestamp: Optional[datetime] = None

    # Metadata
    page_id: Optional[str] = None
    event_time: Optional[datetime] = None

    # Question Key (q1/q2 pattern) — L1 proposition indexing
    question_key: Optional[str] = None      # "death_count", "origin_location", etc.
    extracted_value: Optional[Any] = None   # 13, "Floor 8", etc.
    value_unit: Optional[str] = None        # "people", "floors", "hours", etc.
    has_update_language: bool = False       # "rises to", "now at", "updated to"
    is_monotonic: Optional[bool] = None     # True = value only increases (death toll)

    def __hash__(self):
        return hash(self.id)

    # =========================================================================
    # EXTENSION: Fast approximation for gating
    # =========================================================================

    def affinity(self, other: 'Claim') -> float:
        """
        Extension: Compute affinity score (no LLM).

        Used to gate which pairs get the expensive intrinsic check.
        """
        semantic = cosine_similarity(self.embedding, other.embedding)
        entity = jaccard(self.entities, other.entities)
        anchor = 1.0 if (self.anchor_entities & other.anchor_entities) else 0.0

        # Weighted combination
        return 0.5 * semantic + 0.3 * entity + 0.2 * anchor

    def event_affinity(
        self,
        other: 'Claim',
        entity_idf: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Soft event association score with weighted evidence signals.

        Events are latent causes; this score estimates P(same event).

        Signals:
        - semantic: embedding similarity
        - entity_strength: shared entities weighted by IDF (specificity)
        - source_diversity: different sources = more evidence
        - anchor_match: shared person/org anchors (high value)
        """
        w = weights or {
            'semantic': 0.35,
            'entity_strength': 0.25,
            'anchor_match': 0.30,
            'source_diversity': 0.10
        }

        # 1. Semantic similarity
        semantic = 0.0
        if self.embedding and other.embedding:
            semantic = cosine_similarity(self.embedding, other.embedding)

        # 2. Entity strength with IDF weighting
        shared_entities = self.entities & other.entities
        entity_strength = 0.0
        if shared_entities and entity_idf:
            # Sum of IDF weights for shared entities (specificity)
            total_weight = sum(entity_idf.get(e, 1.0) for e in shared_entities)
            # Normalize by max possible
            max_weight = max(
                sum(entity_idf.get(e, 1.0) for e in self.entities),
                sum(entity_idf.get(e, 1.0) for e in other.entities),
                1.0
            )
            entity_strength = min(total_weight / max_weight, 1.0)
        elif shared_entities:
            # Fallback: simple jaccard
            entity_strength = jaccard(self.entities, other.entities)

        # 3. Anchor match (high value - person/org are specific)
        shared_anchors = self.anchor_entities & other.anchor_entities
        anchor_match = 0.0
        if shared_anchors:
            # More shared anchors = higher score
            anchor_match = min(len(shared_anchors) / 2.0, 1.0)

        # 4. Source diversity (different sources = more evidence)
        source_diversity = 1.0 if self.source != other.source else 0.5

        # Combine with weights
        score = (
            w['semantic'] * semantic +
            w['entity_strength'] * entity_strength +
            w['anchor_match'] * anchor_match +
            w['source_diversity'] * source_diversity
        )

        return score

    def should_compare(self, other: 'Claim', threshold: float = 0.15) -> bool:
        """
        Gate: should we call the expensive intrinsic check?

        Strategy: Low threshold + anchor gate at extreme similarity.
        For better recall, use get_top_k_candidates() instead.
        """
        if self.id == other.id:
            return False

        # Low affinity threshold
        if self.affinity(other) < threshold:
            return False

        # Anchor gate only at very high semantic (prevent obvious bleeding)
        semantic = cosine_similarity(self.embedding, other.embedding)
        if semantic > 0.85:
            has_anchors = bool(self.anchor_entities or other.anchor_entities)
            if has_anchors and not (self.anchor_entities & other.anchor_entities):
                return False

        return True

    def get_top_k_candidates(
        self,
        others: List['Claim'],
        k: int = 5
    ) -> List[Tuple['Claim', float]]:
        """
        k-NN approach: return top-k most similar claims.

        Better recall than threshold-based gating.
        """
        scored = []
        for other in others:
            if other.id == self.id:
                continue
            aff = self.affinity(other)
            scored.append((other, aff))

        # Sort by affinity descending
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    # =========================================================================
    # QUESTION KEY: q1/q2 extraction and candidate generation
    # =========================================================================

    async def extract_question_key(self, llm: 'AsyncOpenAI' = None) -> Dict:
        """
        Extract question_key metadata from claim text.

        The question_key identifies WHICH QUESTION this claim answers.
        Claims with the same question_key are candidates for relation
        classification (CONFIRMS/SUPERSEDES/CONFLICTS).

        Returns dict with:
            question_key: str (e.g., "death_count", "origin_location")
            extracted_value: Any (e.g., 13, "Floor 8")
            value_unit: str (e.g., "people", "floors")
            has_update_language: bool
            is_monotonic: bool
        """
        if not llm:
            # Rule-based fallback
            return self._extract_question_key_rules()

        prompt = f"""Extract the question this claim answers and its value.

CLAIM: "{self.text}"

What specific question does this claim answer? Focus on the PRIMARY assertion.

Examples:
- "13 dead in fire" → question: "death_count", value: 13, unit: "people"
- "Fire started on Floor 8" → question: "origin_floor", value: 8, unit: "floor"
- "Death toll rises to 17" → question: "death_count", value: 17, has_update: true
- "Jimmy Lai faces trial" → question: "legal_status", value: "facing_trial"
- "Trump said X" → question: "trump_statement", value: "X"

Return JSON:
{{
  "question_key": "short_snake_case_key",
  "extracted_value": <number or string>,
  "value_unit": "unit if applicable or null",
  "has_update_language": true/false,
  "is_monotonic": true/false (true for counts that only increase like death tolls),
  "reasoning": "one sentence"
}}"""

        try:
            response = await llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)

            # Update self with extracted values
            self.question_key = result.get('question_key')
            self.extracted_value = result.get('extracted_value')
            self.value_unit = result.get('value_unit')
            self.has_update_language = result.get('has_update_language', False)
            self.is_monotonic = result.get('is_monotonic')

            return result

        except Exception as e:
            return {'error': str(e)}

    def _extract_question_key_rules(self) -> Dict:
        """
        Rule-based question_key extraction (no LLM).

        Handles common patterns:
        - Death/casualty counts
        - Injury counts
        - Location mentions
        - Time mentions
        """
        import re
        text_lower = self.text.lower()

        # Death/casualty patterns
        death_patterns = [
            r'(\d+)\s*(?:people\s+)?(?:dead|killed|died|deaths?|fatalities)',
            r'death\s+toll\s*(?:of|:)?\s*(\d+)',
            r'(\d+)\s+(?:people\s+)?(?:were\s+)?killed',
        ]
        for pattern in death_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = int(match.group(1))
                has_update = any(w in text_lower for w in ['rises', 'risen', 'climbs', 'reaches', 'now', 'updated'])
                self.question_key = "death_count"
                self.extracted_value = value
                self.value_unit = "people"
                self.has_update_language = has_update
                self.is_monotonic = True
                return {
                    'question_key': 'death_count',
                    'extracted_value': value,
                    'value_unit': 'people',
                    'has_update_language': has_update,
                    'is_monotonic': True
                }

        # Injury patterns
        injury_patterns = [
            r'(\d+)\s*(?:people\s+)?(?:injured|wounded|hurt)',
            r'(\d+)\s+(?:others?\s+)?(?:were\s+)?injured',
        ]
        for pattern in injury_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = int(match.group(1))
                has_update = any(w in text_lower for w in ['rises', 'risen', 'climbs', 'reaches', 'now'])
                self.question_key = "injury_count"
                self.extracted_value = value
                self.value_unit = "people"
                self.has_update_language = has_update
                self.is_monotonic = True
                return {
                    'question_key': 'injury_count',
                    'extracted_value': value,
                    'value_unit': 'people',
                    'has_update_language': has_update,
                    'is_monotonic': True
                }

        # No pattern matched - return None (will use embedding fallback)
        return {
            'question_key': None,
            'extracted_value': None,
            'reasoning': 'No rule-based pattern matched'
        }

    def same_question(self, other: 'Claim') -> bool:
        """
        Do these claims answer the same question?

        This is the q1/q2 gate: claims only relate if same question_key.
        """
        if not self.question_key or not other.question_key:
            return False  # Can't determine without question_key
        return self.question_key == other.question_key

    def get_question_bucket_candidates(
        self,
        others: List['Claim']
    ) -> List['Claim']:
        """
        Get candidates with the SAME question_key.

        This is the primary indexing primitive for L1 proposition formation.
        Claims in the same bucket are candidates for:
        - CONFIRMS (same value, different source)
        - SUPERSEDES (different value + update language)
        - CONFLICTS (different value, no update)
        """
        if not self.question_key:
            return []  # No bucket without question_key

        return [
            c for c in others
            if c.id != self.id and c.question_key == self.question_key
        ]

    def classify_within_bucket(self, other: 'Claim') -> Tuple[Relation, float, str]:
        """
        Classify relationship to another claim with SAME question_key.

        Since they answer the same question, classification is mostly rule-based:
        - Same value → CONFIRMS
        - Different value + update language → SUPERSEDES
        - Different value, no update → CONFLICTS

        Returns: (relation, confidence, reasoning)
        """
        assert self.question_key == other.question_key, "Must have same question_key"

        # Same value = CONFIRMS
        if self.extracted_value == other.extracted_value:
            return (
                Relation.CONFIRMS,
                0.9,
                f"Same {self.question_key}: {self.extracted_value}"
            )

        # Different values - check for update language
        if self.has_update_language or other.has_update_language:
            # Determine which supersedes which based on:
            # 1. Update language
            # 2. Monotonicity (for counts)
            # 3. Timestamp

            newer_claim = self
            older_claim = other

            # If monotonic (death toll), higher value supersedes
            if self.is_monotonic and isinstance(self.extracted_value, (int, float)):
                if self.extracted_value > other.extracted_value:
                    newer_claim, older_claim = self, other
                else:
                    newer_claim, older_claim = other, self

            # Otherwise use timestamp if available
            elif self.timestamp and other.timestamp:
                if self.timestamp > other.timestamp:
                    newer_claim, older_claim = self, other
                else:
                    newer_claim, older_claim = other, self

            return (
                Relation.SUPERSEDES,
                0.85,
                f"{self.question_key}: {older_claim.extracted_value} → {newer_claim.extracted_value}"
            )

        # Different values, no update language = CONFLICTS
        return (
            Relation.CONFLICTS,
            0.8,
            f"Conflicting {self.question_key}: {self.extracted_value} vs {other.extracted_value}"
        )

    # =========================================================================
    # INTRINSIC: The atomic relationship classification
    # =========================================================================

    async def relates_to(
        self,
        other: 'Claim',
        llm: 'AsyncOpenAI' = None
    ) -> Tuple[Relation, float, str]:
        """
        INTRINSIC: Classify relationship to another claim.

        This is the atomic operation that cannot be approximated.
        Returns: (relation, confidence, reasoning)
        """
        if not llm:
            # Fallback: use affinity as heuristic
            aff = self.affinity(other)
            if aff > 0.8:
                return Relation.CONFIRMS, aff, "High affinity (no LLM)"
            return Relation.UNRELATED, 1.0 - aff, "Low affinity (no LLM)"

        prompt = f"""Compare these two claims and determine their epistemic relationship.

CLAIM A: "{self.text}"
SOURCE A: {self.source}
ENTITIES A: {list(self.entities)[:5]}

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
            response = await llm.chat.completions.create(
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


# =============================================================================
# LEVEL 1+: SURFACE (Virtual, Emergent Properties)
# =============================================================================

@dataclass
class AboutnessLink:
    """
    Soft aboutness edge between surfaces (Tier-2).

    These edges represent "same event, different aspect" associations.
    They are NOT identity edges and should NOT be used to merge surfaces.
    They are used to cluster surfaces into events.
    """
    target_id: str              # other surface ID
    score: float                # aboutness score (0-1)
    evidence: Dict = field(default_factory=dict)  # breakdown of signals


@dataclass
class Surface:
    """
    L2: Bundle of claims connected by IDENTITY edges.

    A surface represents a single proposition/fact that may have multiple
    claims from different sources. Internal edges are IDENTITY relations:
    - CONFIRMS: same claim, different source
    - REFINES: adds detail to same fact
    - SUPERSEDES: updates value with temporal marker
    - CONFLICTS: contradicts but still same fact (disagreement recorded)

    Aboutness (L3 event-level) is stored separately in about_links.
    This separation is INVARIANT 3.
    """
    id: str
    claim_ids: Set[str] = field(default_factory=set)

    # Computed properties (from claims)
    centroid: Optional[List[float]] = None
    entropy: float = 0.0
    mass: float = 0.0  # Total source count
    sources: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    time_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None)

    # Semantic properties (LLM interpretation)
    canonical_title: Optional[str] = None
    description: Optional[str] = None
    key_facts: List[str] = field(default_factory=list)

    # Internal structure: IDENTITY edges that formed this surface
    internal_edges: List[Tuple[str, str, Relation]] = field(default_factory=list)

    # External structure: ABOUTNESS edges to other surfaces (soft, graded)
    # These are NOT used for surface formation, only for event clustering
    about_links: List[AboutnessLink] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    # =========================================================================
    # COMPUTED: Build from claims
    # =========================================================================

    @classmethod
    def from_claims(cls, surface_id: str, claims: List[Claim], edges: List[Tuple[str, str, Relation]] = None) -> 'Surface':
        """Compute surface properties from claims."""
        if not claims:
            return cls(id=surface_id)

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

        return cls(
            id=surface_id,
            claim_ids={c.id for c in claims},
            centroid=centroid,
            entropy=entropy,
            mass=mass,
            sources=sources,
            entities=entities,
            anchor_entities=anchor_entities,
            time_window=time_window,
            internal_edges=edges or []
        )

    # =========================================================================
    # EXTENSION: Fast approximation for gating
    # =========================================================================

    def affinity(self, other: 'Surface') -> float:
        """Extension: Compute affinity to another surface."""
        semantic = cosine_similarity(self.centroid, other.centroid) if self.centroid and other.centroid else 0.0
        entity = jaccard(self.entities, other.entities)
        anchor = jaccard(self.anchor_entities, other.anchor_entities)
        temporal = temporal_overlap(self.time_window, other.time_window)

        return 0.4 * semantic + 0.3 * anchor + 0.2 * entity + 0.1 * temporal

    # =========================================================================
    # IDENTITY vs ASSOCIATION (level-specific semantics)
    # =========================================================================

    def identity_check(self, other: 'Surface', threshold: float = 0.7) -> bool:
        """
        Are we the SAME surface? (Should merge)

        Stricter than association - requires anchor overlap.
        """
        # Must share anchor entities
        if not (self.anchor_entities & other.anchor_entities):
            return False

        # High affinity
        if self.affinity(other) < threshold:
            return False

        # Temporal overlap
        if self.time_window[0] and other.time_window[0]:
            days_apart = abs((self.time_window[0] - other.time_window[0]).days)
            if days_apart > 7:
                return False

        return True

    def association_check(self, other: 'Surface') -> float:
        """
        How RELATED are we? (Edge strength, no merge)

        Softer than identity - allows topic-level connections.
        """
        return self.affinity(other)

    # =========================================================================
    # SEMANTIC INTERPRETATION (LLM)
    # =========================================================================

    async def interpret(self, claims: List[Claim], llm: 'AsyncOpenAI') -> None:
        """
        LLM: Lift computed structure to semantic meaning.

        Generates canonical_title, description, key_facts.
        """
        # Get claim texts
        claim_texts = [c.text for c in claims if c.id in self.claim_ids][:10]

        if not claim_texts:
            self.canonical_title = f"Surface {self.id}"
            self.description = "No claims"
            return

        prompt = f"""Based on these related claims, generate semantic interpretation:

CLAIMS:
{chr(10).join(f'- {t}' for t in claim_texts)}

ENTITIES: {list(self.entities)[:10]}
SOURCES: {len(self.sources)} independent sources
TIME: {self.time_window[0]} to {self.time_window[1]}

Generate:
1. canonical_title: Short (3-6 word) reusable title for this event/topic
2. description: One paragraph summary (50-100 words)
3. key_facts: List of 3-5 main facts (confirmed by multiple sources first)

Return JSON:
{{
  "canonical_title": "...",
  "description": "...",
  "key_facts": ["...", "..."]
}}"""

        try:
            response = await llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)

            self.canonical_title = result.get('canonical_title', f'Surface {self.id}')
            self.description = result.get('description', '')
            self.key_facts = result.get('key_facts', [])

        except Exception as e:
            self.canonical_title = f"Surface {self.id}"
            self.description = f"Error: {e}"


# =============================================================================
# LEVEL 2: EVENT (Virtual, Higher-order emergence)
# =============================================================================

@dataclass
class Event:
    """
    Higher-level virtual unit. Emergent from surface relationships.

    May group multiple related surfaces into a coherent narrative.
    """
    id: str
    surface_ids: Set[str] = field(default_factory=set)

    # Computed from surfaces
    centroid: Optional[List[float]] = None
    total_claims: int = 0
    total_sources: int = 0
    entities: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    time_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None)

    # Semantic interpretation
    canonical_title: Optional[str] = None
    narrative: Optional[str] = None
    timeline: List[Dict] = field(default_factory=list)

    @classmethod
    def from_surfaces(cls, event_id: str, surfaces: List[Surface]) -> 'Event':
        """Compute event properties from surfaces."""
        if not surfaces:
            return cls(id=event_id)

        # Centroid (average of surface centroids)
        centroids = [s.centroid for s in surfaces if s.centroid]
        centroid = np.mean(centroids, axis=0).tolist() if centroids else None

        # Aggregate
        total_claims = sum(len(s.claim_ids) for s in surfaces)
        all_sources = set()
        all_entities = set()
        all_anchors = set()

        for s in surfaces:
            all_sources.update(s.sources)
            all_entities.update(s.entities)
            all_anchors.update(s.anchor_entities)

        # Time window (span across all surfaces)
        starts = [s.time_window[0] for s in surfaces if s.time_window[0]]
        ends = [s.time_window[1] for s in surfaces if s.time_window[1]]
        time_window = (
            min(starts) if starts else None,
            max(ends) if ends else None
        )

        return cls(
            id=event_id,
            surface_ids={s.id for s in surfaces},
            centroid=centroid,
            total_claims=total_claims,
            total_sources=len(all_sources),
            entities=all_entities,
            anchor_entities=all_anchors,
            time_window=time_window
        )

    def affinity(self, other: 'Event') -> float:
        """Affinity to another event."""
        semantic = cosine_similarity(self.centroid, other.centroid) if self.centroid and other.centroid else 0.0
        anchor = jaccard(self.anchor_entities, other.anchor_entities)
        temporal = temporal_overlap(self.time_window, other.time_window)

        return 0.4 * semantic + 0.4 * anchor + 0.2 * temporal

    async def interpret(self, surfaces: List[Surface], llm: 'AsyncOpenAI') -> None:
        """Generate narrative interpretation."""
        surface_summaries = []
        for s in surfaces:
            if s.id in self.surface_ids and s.canonical_title:
                surface_summaries.append(f"- {s.canonical_title}: {s.description[:100]}...")

        if not surface_summaries:
            self.canonical_title = f"Event {self.id}"
            return

        prompt = f"""Based on these related topics/surfaces, generate event narrative:

SURFACES:
{chr(10).join(surface_summaries)}

KEY ENTITIES: {list(self.anchor_entities)[:5]}
TIME SPAN: {self.time_window[0]} to {self.time_window[1]}
SOURCES: {self.total_sources} independent sources

Generate:
1. canonical_title: Event title (3-8 words)
2. narrative: Coherent narrative connecting these surfaces (100-150 words)
3. timeline: Key moments in chronological order

Return JSON:
{{
  "canonical_title": "...",
  "narrative": "...",
  "timeline": [{{"time": "...", "event": "..."}}, ...]
}}"""

        try:
            response = await llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)

            self.canonical_title = result.get('canonical_title', f'Event {self.id}')
            self.narrative = result.get('narrative', '')
            self.timeline = result.get('timeline', [])

        except Exception as e:
            self.canonical_title = f"Event {self.id}"
            self.narrative = f"Error: {e}"


# =============================================================================
# EMERGENCE ENGINE: Level-agnostic clustering
# =============================================================================

class EmergenceEngine:
    """
    Orchestrates emergence across levels.

    Architecture:
      L0 → L2: Claims → Surfaces (via IDENTITY edges from Claim.relates_to)
      L2 → L3: Surfaces → Events (via ABOUTNESS edges from compute_surface_aboutness)

    Invariants enforced:
      - L0 is append-only (claims never deleted)
      - Parameters are versioned (all changes tracked)
      - Identity/Aboutness separation (L2 uses identity, L3 uses aboutness)
      - Meta-claims emitted for tension detection
    """

    def __init__(self, llm: 'AsyncOpenAI' = None, params: Parameters = None):
        self.llm = llm
        self.params = params or Parameters()

        # L0: Claims (append-only — INVARIANT 1)
        self.claims: Dict[str, Claim] = {}

        # Question Key Index (q1/q2 pattern for L1 proposition formation)
        # Maps question_key → list of claim_ids
        self.question_index: Dict[str, List[str]] = {}

        # L2: Surfaces (computed from L0 + params)
        self.surfaces: Dict[str, Surface] = {}

        # L3: Events (computed from L2 + params)
        self.events: Dict[str, Event] = {}

        # Identity edges (claim-to-claim, Tier-1)
        self.claim_edges: List[Tuple[str, str, Relation, float]] = []

        # Aboutness edges (surface-to-surface, Tier-2)
        self.surface_aboutness: List[Tuple[str, str, float, Dict]] = []

        # Meta-claims (observations about epistemic state)
        self.meta_claims: List[MetaClaim] = []

        # Counters
        self._surface_counter = 0
        self._event_counter = 0

    # =========================================================================
    # LEVEL 0 → 1: Claims to Surfaces
    # =========================================================================

    async def add_claim(
        self,
        claim: Claim,
        extract_question_key: bool = True
    ) -> Dict:
        """
        Add claim and compute relationships to existing claims.

        Clean two-path design:
        1. Claims WITH question_key → bucket lookup → rule-based classification
        2. Claims WITHOUT question_key → embedding gate → LLM classification

        Embedding is only used for GATING (reducing comparisons), never for
        classification. question_key is the L1 indexing primitive.

        Returns: {claim_id, question_key, relations: [...]}
        """
        # Store claim (INVARIANT 1: append-only)
        self.claims[claim.id] = claim

        # Step 1: Extract question_key if not already set
        if extract_question_key and not claim.question_key:
            await claim.extract_question_key(self.llm)

        # Step 2: Index by question_key
        if claim.question_key:
            if claim.question_key not in self.question_index:
                self.question_index[claim.question_key] = []
            self.question_index[claim.question_key].append(claim.id)

        relations_found = []

        # =====================================================================
        # PATH A: Claims WITH question_key → bucket → rule-based
        # =====================================================================
        if claim.question_key:
            bucket_claim_ids = self.question_index.get(claim.question_key, [])
            candidates = [
                self.claims[cid] for cid in bucket_claim_ids
                if cid != claim.id and cid in self.claims
            ]

            for other in candidates:
                # Within-bucket classification (rule-based, no LLM needed)
                relation, confidence, reasoning = claim.classify_within_bucket(other)

                if confidence >= self.params.identity_confidence_threshold:
                    self.claim_edges.append((claim.id, other.id, relation, confidence))
                    relations_found.append({
                        'other_id': other.id,
                        'relation': relation.value,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'method': 'question_key'
                    })

        # =====================================================================
        # PATH B: Claims WITHOUT question_key → embedding gate → LLM
        # =====================================================================
        else:
            for other_id, other in self.claims.items():
                if other_id == claim.id:
                    continue

                # Embedding gate (reduces comparisons, not classification)
                if not claim.should_compare(other):
                    continue

                # LLM classification
                relation, confidence, reasoning = await claim.relates_to(other, self.llm)

                if relation != Relation.UNRELATED and confidence >= self.params.identity_confidence_threshold:
                    self.claim_edges.append((claim.id, other_id, relation, confidence))
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

    def compute_surfaces(self) -> List[Surface]:
        """
        Compute surfaces from IDENTITY edges only (connected components).

        Surfaces represent same-fact sub-topologies. Only Tier-1 kernel
        relations are used: CONFIRMS, REFINES, SUPERSEDES, CONFLICTS.

        Aboutness (event-level association) is handled separately by
        compute_surface_aboutness() and compute_events().
        """
        from collections import defaultdict

        # Build adjacency from identity relations ONLY
        # These are all "same fact" relations - even CONFLICTS
        adj = defaultdict(set)
        for c1, c2, rel, conf in self.claim_edges:
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
        self.surfaces = {}
        for component in components:
            self._surface_counter += 1
            surface_id = f"S{self._surface_counter:03d}"

            claims = [self.claims[cid] for cid in component]
            edges = [(c1, c2, rel) for c1, c2, rel, _ in self.claim_edges
                     if c1 in component and c2 in component]

            surface = Surface.from_claims(surface_id, claims, edges)
            self.surfaces[surface_id] = surface

        return list(self.surfaces.values())

    # =========================================================================
    # TIER-2: Surface Aboutness (soft event-level associations)
    # =========================================================================

    def compute_surface_aboutness(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Compute soft aboutness edges between surfaces (TIER-2).

        These edges represent "same event, different aspect" associations.
        They are NOT identity edges and MUST NOT be used to merge surfaces.
        They are used by compute_events() to cluster surfaces into events.

        This method also populates Surface.about_links for each surface.

        Uses parameters (INVARIANT 2):
            - hub_max_df: Anchors in more surfaces than this get zero weight
            - aboutness_min_signals: Require N of 3 signals for edge

        Returns:
            List of (surface_id_1, surface_id_2, score, evidence_dict)
        """
        # Use versioned parameters
        hub_max_df = self.params.hub_max_df
        min_signals = self.params.aboutness_min_signals
        import math
        from collections import defaultdict

        if not self.surfaces:
            return []

        surfaces = list(self.surfaces.values())
        n_surfaces = len(surfaces)

        # Compute anchor document frequency across surfaces
        anchor_df = defaultdict(int)
        for s in surfaces:
            for anchor in s.anchor_entities:
                anchor_df[anchor] += 1

        # Compute anchor IDF with hub penalty
        anchor_idf = {}
        for anchor, df in anchor_df.items():
            if df > hub_max_df:
                anchor_idf[anchor] = 0.0  # Hub penalty: zero weight
            else:
                anchor_idf[anchor] = math.log(1 + n_surfaces / df)

        # Compute entity IDF across surfaces
        entity_df = defaultdict(int)
        for s in surfaces:
            for e in s.entities:
                entity_df[e] += 1
        entity_idf = {
            e: math.log(1 + n_surfaces / df)
            for e, df in entity_df.items()
        }

        aboutness_edges = []

        for i, s1 in enumerate(surfaces):
            for s2 in surfaces[i+1:]:
                evidence = {}
                signals_met = 0

                # Signal 1: Strong anchor overlap (IDF-weighted)
                shared_anchors = s1.anchor_entities & s2.anchor_entities
                anchor_score = 0.0
                if shared_anchors:
                    anchor_score = sum(anchor_idf.get(a, 0) for a in shared_anchors)
                    # Normalize by max possible
                    max_anchor = max(
                        sum(anchor_idf.get(a, 0) for a in s1.anchor_entities),
                        sum(anchor_idf.get(a, 0) for a in s2.anchor_entities),
                        1.0
                    )
                    anchor_score = min(anchor_score / max_anchor, 1.0)
                    if anchor_score > 0.3:
                        signals_met += 1
                evidence['anchor_score'] = anchor_score
                evidence['shared_anchors'] = list(shared_anchors)

                # Signal 2: Semantic similarity (centroid)
                semantic_score = 0.0
                if s1.centroid and s2.centroid:
                    semantic_score = cosine_similarity(s1.centroid, s2.centroid)
                    if semantic_score > 0.5:
                        signals_met += 1
                evidence['semantic_score'] = semantic_score

                # Signal 3: Entity overlap (IDF-weighted, excluding anchors)
                shared_entities = s1.entities & s2.entities
                entity_score = 0.0
                if shared_entities:
                    entity_weight = sum(entity_idf.get(e, 0) for e in shared_entities)
                    max_entity = max(
                        sum(entity_idf.get(e, 0) for e in s1.entities),
                        sum(entity_idf.get(e, 0) for e in s2.entities),
                        1.0
                    )
                    entity_score = min(entity_weight / max_entity, 1.0)
                    if entity_score > 0.3:
                        signals_met += 1
                evidence['entity_score'] = entity_score

                # Signal 4: Source diversity (different sources = more evidence)
                source_overlap = len(s1.sources & s2.sources)
                source_diversity = 1.0 if source_overlap == 0 else 0.5
                evidence['source_diversity'] = source_diversity

                # 2-of-3 constraint: require min_signals to create edge
                if signals_met < min_signals:
                    continue

                # Compute overall aboutness score
                score = (
                    0.35 * anchor_score +
                    0.30 * semantic_score +
                    0.25 * entity_score +
                    0.10 * source_diversity
                )

                evidence['signals_met'] = signals_met

                aboutness_edges.append((s1.id, s2.id, score, evidence))

                # Populate Surface.about_links (bidirectional)
                s1.about_links.append(AboutnessLink(
                    target_id=s2.id,
                    score=score,
                    evidence=evidence.copy()
                ))
                s2.about_links.append(AboutnessLink(
                    target_id=s1.id,
                    score=score,
                    evidence=evidence.copy()
                ))

        # Store for later use
        self.surface_aboutness = aboutness_edges

        return aboutness_edges

    def compute_events(self) -> List['Event']:
        """
        Cluster surfaces into events based on ABOUTNESS edges (L3).

        This uses the surface↔surface aboutness graph (NOT identity).
        Surfaces with strong aboutness links are grouped into the same event.

        INVARIANT 3: Only aboutness edges used here, never identity.

        Uses parameters (INVARIANT 2):
            - aboutness_threshold: Min score to link surfaces into events
        """
        from collections import defaultdict

        # Use versioned parameters
        aboutness_threshold = self.params.aboutness_threshold

        if not self.surface_aboutness:
            self.compute_surface_aboutness()

        # Build aboutness adjacency (above threshold only)
        adj = defaultdict(set)
        for s1_id, s2_id, score, _ in self.surface_aboutness:
            if score >= aboutness_threshold:
                adj[s1_id].add(s2_id)
                adj[s2_id].add(s1_id)

        # Find connected components of strong aboutness
        visited = set()
        event_groups = []

        for surface_id in self.surfaces:
            if surface_id in visited:
                continue

            group = set()
            stack = [surface_id]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                group.add(curr)
                stack.extend(adj[curr] - visited)

            event_groups.append(group)

        # Create events from surface groups
        self.events = {}
        for group in event_groups:
            self._event_counter += 1
            event_id = f"E{self._event_counter:03d}"

            surfaces_in_event = [self.surfaces[sid] for sid in group]
            event = Event.from_surfaces(event_id, surfaces_in_event)
            self.events[event_id] = event

        return list(self.events.values())

    # =========================================================================
    # META-CLAIMS: Tension Detection (INVARIANT 6)
    # =========================================================================

    def detect_tensions(self) -> List[MetaClaim]:
        """
        Detect tensions in the epistemic state and emit meta-claims.

        Meta-claims are observations ABOUT the topology, not truth claims.
        They may trigger:
        - ParameterChange (adjust thresholds)
        - New L0 claims (verification, corroboration)
        - Task generation (bounties, investigations)

        Tension types detected:
        - high_entropy_surface: Surface has high semantic dispersion
        - single_source_only: Claim has only one source
        - unresolved_conflict: CONFLICTS edge without resolution
        - bridge_node_detected: Single anchor connecting dense parts
        """
        new_meta_claims = []

        # Detect high-entropy surfaces
        for surface in self.surfaces.values():
            if surface.entropy > self.params.high_entropy_threshold:
                mc = MetaClaim(
                    type="high_entropy_surface",
                    target_id=surface.id,
                    target_type="surface",
                    evidence={
                        'entropy': surface.entropy,
                        'threshold': self.params.high_entropy_threshold,
                        'claim_count': len(surface.claim_ids),
                        'sources': list(surface.sources)
                    },
                    params_version=self.params.version
                )
                new_meta_claims.append(mc)

        # Detect single-source claims (corroboration needed)
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
                new_meta_claims.append(mc)

        # Detect unresolved conflicts
        for c1, c2, rel, conf in self.claim_edges:
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
                new_meta_claims.append(mc)

        # Store and return
        self.meta_claims.extend(new_meta_claims)
        return new_meta_claims

    def get_unresolved_meta_claims(self) -> List[MetaClaim]:
        """Return meta-claims that haven't been resolved."""
        return [mc for mc in self.meta_claims if not mc.resolved]

    def resolve_meta_claim(
        self,
        meta_claim_id: str,
        resolution: str,
        actor: str = "system"
    ) -> Optional[MetaClaim]:
        """
        Mark a meta-claim as resolved.

        Args:
            meta_claim_id: ID of the meta-claim to resolve
            resolution: How it was resolved (e.g., "parameter_updated", "new_claim_added")
            actor: Who/what resolved it
        """
        for mc in self.meta_claims:
            if mc.id == meta_claim_id:
                mc.resolved = True
                mc.resolution = f"{resolution} by {actor}"
                return mc
        return None

    def attach_singletons(self, min_anchor_overlap: int = 1) -> int:
        """
        Attach singleton surfaces to larger surfaces via weak links.

        Uses anchor entity overlap (not LLM) to connect orphaned claims.
        Returns number of attachments made.

        Weak links are flagged - can be dropped if tension rises.
        """
        singletons = [s for s in self.surfaces.values() if len(s.claim_ids) == 1]
        larger = [s for s in self.surfaces.values() if len(s.claim_ids) > 1]

        if not singletons or not larger:
            return 0

        attachments = 0
        to_remove = []

        for singleton in singletons:
            # Find best larger surface by anchor overlap
            best_match = None
            best_overlap = 0

            for candidate in larger:
                overlap = len(singleton.anchor_entities & candidate.anchor_entities)
                if overlap >= min_anchor_overlap and overlap > best_overlap:
                    best_overlap = overlap
                    best_match = candidate

            if best_match:
                # Merge singleton into larger surface (weak link)
                singleton_claim_id = list(singleton.claim_ids)[0]
                best_match.claim_ids.add(singleton_claim_id)

                # Merge entities
                best_match.entities.update(singleton.entities)
                best_match.anchor_entities.update(singleton.anchor_entities)

                # Flag as weak link
                if not hasattr(best_match, 'weak_links'):
                    best_match.weak_links = set()
                best_match.weak_links.add(singleton_claim_id)

                to_remove.append(singleton.id)
                attachments += 1

        # Remove merged singletons
        for sid in to_remove:
            del self.surfaces[sid]

        return attachments

    def merge_surfaces_by_anchor_prototype(self, max_anchor_frequency: int = 2) -> int:
        """
        Merge surfaces that share a strong/specific anchor.

        Strong anchor = appears in few surfaces (specific to event)
        vs global figure (appears in many surfaces, e.g. "Donald Trump")

        Args:
            max_anchor_frequency: anchors appearing in more surfaces than this
                                  are considered "global" and ignored

        Returns: number of merges performed
        """
        from collections import defaultdict

        if not self.surfaces:
            return 0

        # Count anchor frequency across surfaces
        anchor_counts = defaultdict(int)
        for s in self.surfaces.values():
            for anchor in s.anchor_entities:
                anchor_counts[anchor] += 1

        # Identify strong anchors (not global figures)
        strong_anchors = {
            anchor for anchor, count in anchor_counts.items()
            if count <= max_anchor_frequency
        }

        # Build anchor -> surfaces mapping (only strong anchors)
        anchor_to_surfaces = defaultdict(set)
        for sid, s in self.surfaces.items():
            for anchor in s.anchor_entities:
                if anchor in strong_anchors:
                    anchor_to_surfaces[anchor].add(sid)

        # Union-find for surface merging
        parent = {sid: sid for sid in self.surfaces}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        # Merge surfaces sharing strong anchors
        merges = 0
        for anchor, sids in anchor_to_surfaces.items():
            sids = list(sids)
            for i in range(len(sids) - 1):
                if union(sids[i], sids[i + 1]):
                    merges += 1

        # Rebuild surfaces from merged components
        components = defaultdict(set)
        for sid in self.surfaces:
            components[find(sid)].add(sid)

        # Create new merged surfaces
        new_surfaces = {}
        for component_sids in components.values():
            if len(component_sids) == 1:
                # No merge, keep original
                sid = list(component_sids)[0]
                new_surfaces[sid] = self.surfaces[sid]
            else:
                # Merge multiple surfaces
                self._surface_counter += 1
                new_id = f"S{self._surface_counter:03d}"

                all_claim_ids = set()
                all_entities = set()
                all_anchors = set()

                for sid in component_sids:
                    s = self.surfaces[sid]
                    all_claim_ids.update(s.claim_ids)
                    all_entities.update(s.entities)
                    all_anchors.update(s.anchor_entities)

                claims = [self.claims[cid] for cid in all_claim_ids]
                merged = Surface.from_claims(new_id, claims, [])
                new_surfaces[new_id] = merged

        self.surfaces = new_surfaces
        return merges

    # NOTE: compute_events() is now defined in TIER-2 section above
    # It uses aboutness edges between surfaces, not identity checks

    # =========================================================================
    # SEMANTIC INTERPRETATION
    # =========================================================================

    async def interpret_all(self) -> None:
        """Generate semantic interpretation for all surfaces and events."""
        if not self.llm:
            return

        claims_list = list(self.claims.values())
        surfaces_list = list(self.surfaces.values())

        # Interpret surfaces
        for surface in surfaces_list:
            await surface.interpret(claims_list, self.llm)

        # Interpret events
        for event in self.events.values():
            await event.interpret(surfaces_list, self.llm)

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def summary(self) -> Dict:
        """Summary of current state including parameters and meta-claims."""
        return {
            # Core counts
            'claims': len(self.claims),
            'claim_edges': len(self.claim_edges),
            'surfaces': len(self.surfaces),
            'surface_aboutness_edges': len(self.surface_aboutness),
            'events': len(self.events),

            # Question Key Index (q1/q2 pattern)
            'question_index': {
                'buckets': len(self.question_index),
                'claims_with_key': sum(len(v) for v in self.question_index.values()),
                'largest_bucket': max((len(v) for v in self.question_index.values()), default=0),
                'keys': list(self.question_index.keys())[:10]  # Show first 10 keys
            },

            # Parameters (INVARIANT 2: versioned)
            'params': {
                'version': self.params.version,
                'identity_confidence_threshold': self.params.identity_confidence_threshold,
                'hub_max_df': self.params.hub_max_df,
                'aboutness_min_signals': self.params.aboutness_min_signals,
                'aboutness_threshold': self.params.aboutness_threshold,
                'high_entropy_threshold': self.params.high_entropy_threshold,
                'changes': len(self.params.changes)
            },

            # Meta-claims (INVARIANT 6: observations about state)
            'meta_claims': {
                'total': len(self.meta_claims),
                'unresolved': len(self.get_unresolved_meta_claims()),
                'by_type': self._count_meta_claims_by_type()
            },

            # Details
            'surfaces_detail': [
                {
                    'id': s.id,
                    'title': s.canonical_title,
                    'claims': len(s.claim_ids),
                    'sources': len(s.sources),
                    'entropy': round(s.entropy, 3),
                    'about_links': len(s.about_links)
                }
                for s in self.surfaces.values()
            ],
            'events_detail': [
                {
                    'id': e.id,
                    'title': e.canonical_title,
                    'surfaces': len(e.surface_ids),
                    'total_claims': e.total_claims,
                    'sources': e.total_sources
                }
                for e in self.events.values()
            ]
        }

    def _count_meta_claims_by_type(self) -> Dict[str, int]:
        """Count meta-claims by type."""
        counts: Dict[str, int] = {}
        for mc in self.meta_claims:
            counts[mc.type] = counts.get(mc.type, 0) + 1
        return counts


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 60)
        print("EPISTEMIC UNIT TEST")
        print("=" * 60)
        print("\nINVARIANTS:")
        print("  1. L0 immutability (claims append-only)")
        print("  2. Parameter versioning")
        print("  3. Identity/Aboutness separation")
        print("  4. Derived state purity")
        print("  5. Stable core relations")
        print("  6. Meta-claims are observations")

        # Create test claims
        claims = [
            Claim(id="c1", text="Fire kills 13 in Hong Kong high-rise", source="BBC",
                  entities={"Hong Kong", "fire", "high-rise"}, anchor_entities=set()),
            Claim(id="c2", text="Hong Kong fire death toll reaches 13", source="Reuters",
                  entities={"Hong Kong", "fire"}, anchor_entities=set()),
            Claim(id="c3", text="13 dead in HK apartment blaze", source="SCMP",
                  entities={"Hong Kong", "fire", "apartment"}, anchor_entities=set()),
            Claim(id="c4", text="Jimmy Lai trial continues in Hong Kong", source="BBC",
                  entities={"Jimmy Lai", "Hong Kong", "trial"}, anchor_entities={"Jimmy Lai"}),
            Claim(id="c5", text="Lai faces national security charges", source="Guardian",
                  entities={"Jimmy Lai", "national security"}, anchor_entities={"Jimmy Lai"}),
        ]

        # Test with custom parameters
        print("\n--- Parameters (INVARIANT 2) ---")
        params = Parameters()
        print(f"  Version: {params.version}")
        print(f"  identity_confidence_threshold: {params.identity_confidence_threshold}")
        print(f"  hub_max_df: {params.hub_max_df}")

        # Test parameter update with provenance
        change = params.update(
            parameter="identity_confidence_threshold",
            new_value=0.6,
            actor="test:manual",
            rationale="Testing parameter versioning"
        )
        print(f"\n  Parameter updated: {change.parameter} {change.old_value} → {change.new_value}")
        print(f"  Version now: {params.version}")
        print(f"  Actor: {change.actor}")

        # Test emergence (no LLM)
        print("\n--- Emergence Engine ---")
        engine = EmergenceEngine(llm=None, params=params)

        for claim in claims:
            result = await engine.add_claim(claim)
            if result['relations']:
                print(f"  {claim.id}: {len(result['relations'])} relations")

        # Compute surfaces (IDENTITY edges only - INVARIANT 3)
        print("\n--- Surfaces (L2: IDENTITY edges only) ---")
        surfaces = engine.compute_surfaces()
        print(f"  Total: {len(surfaces)}")
        for s in surfaces:
            print(f"  {s.id}: {len(s.claim_ids)} claims, entropy={s.entropy:.3f}")

        # Compute aboutness (separate from identity - INVARIANT 3)
        print("\n--- Surface Aboutness (L3: soft edges) ---")
        aboutness = engine.compute_surface_aboutness()
        print(f"  Aboutness edges: {len(aboutness)}")
        for s in surfaces:
            print(f"  {s.id}: {len(s.about_links)} about_links")

        # Compute events from aboutness
        print("\n--- Events (L3: from ABOUTNESS, not identity) ---")
        events = engine.compute_events()
        print(f"  Total: {len(events)}")
        for e in events:
            print(f"  {e.id}: {len(e.surface_ids)} surfaces")

        # Test meta-claims (INVARIANT 6)
        print("\n--- Meta-claims (INVARIANT 6: observations) ---")
        meta_claims = engine.detect_tensions()
        print(f"  Detected: {len(meta_claims)}")
        for mc in meta_claims[:3]:
            print(f"  - {mc.type}: {mc.target_id}")

        # Summary
        print("\n--- Summary ---")
        summary = engine.summary()
        print(f"  Claims: {summary['claims']}")
        print(f"  Surfaces: {summary['surfaces']}")
        print(f"  Events: {summary['events']}")
        print(f"  Params version: {summary['params']['version']}")
        print(f"  Meta-claims: {summary['meta_claims']}")

    asyncio.run(test())
