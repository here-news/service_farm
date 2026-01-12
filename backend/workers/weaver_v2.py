"""
Weaver v2 - LLM-Assisted Epistemic Topology
============================================

Architecture: propose → verify → commit with backtracking/reconcile capability.

Key differences from principled_weaver.py:
1. LLM adjudication for semantic grounding (when needed)
2. Embedding-based candidate recall (maximize recall)
3. Self-consistency checks (2-3 samples)
4. Commit with uncertainty levels (auto/periphery/defer)

Preserves from principled_weaver:
- Data model: Surface, Incident, Case, MetaClaim, TypedPosterior
- Scoped surface identity: (scope_id, question_key)
- Bridge immunity: anchor overlap + companion compatibility
- Typed posteriors with outlier detection
- Motif computation for L3→L4
- Redis viz emission

LLM is called conditionally (not every claim):
- When pattern matching fails or low confidence
- When top-2 candidates are close (ambiguous)
- When bridge risk detected
- When typed conflict needs disambiguation

Two core prompts:
- Prompt A (Proposition Schema): "Same random variable?" → Surface identity
- Prompt B (Happening Schema): "Same event instance?" → Incident membership
"""

import asyncio
import json
import logging
import hashlib
import os
import sys
from typing import Optional, List, Set, Dict, Tuple, Any, FrozenSet
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import numpy as np

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
from repositories.claim_repository import ClaimRepository
from repositories.surface_repository import SurfaceRepository
from models.domain.claim import Claim
from models.domain.surface import Surface
from utils.id_generator import generate_id

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES (preserved from principled_weaver)
# =============================================================================

@dataclass
class MetaClaim:
    """Meta-claim emitted during processing."""
    type: str  # bridge_blocked, typed_conflict, missing_time, llm_adjudicated, etc.
    claim_id: str
    surface_id: Optional[str] = None
    incident_id: Optional[str] = None
    reason: str = ""
    evidence: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TypedPosterior:
    """Typed belief state for a surface."""
    question_key: str
    value_type: str  # "numeric" or "categorical"
    posterior_mean: Optional[float] = None
    current_value: Optional[Any] = None
    observations: int = 0
    values: List[Tuple[Any, float, str]] = field(default_factory=list)
    outliers: List[Dict] = field(default_factory=list)
    conflict_detected: bool = False


@dataclass
class L3Incident:
    """L3 Incident: membrane over L2 surfaces."""
    id: str
    surface_ids: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    companion_entities: Set[str] = field(default_factory=set)
    core_motifs: List[Dict] = field(default_factory=list)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    canonical_title: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class L4Case:
    """L4 Case: membrane over L3 incidents."""
    id: str
    incident_ids: Set[str] = field(default_factory=set)
    relation_backbone: List[Tuple[str, str, str]] = field(default_factory=list)
    core_entities: Set[str] = field(default_factory=set)
    case_type: str = "unknown"
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    canonical_title: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LinkResult:
    """Result of linking a claim to L2/L3/L4."""
    claim_id: str
    surface_id: str
    incident_id: str
    case_id: Optional[str] = None
    is_new_surface: bool = False
    is_new_incident: bool = False
    is_new_case: bool = False
    question_key: str = ""
    meta_claims: List[MetaClaim] = field(default_factory=list)


# =============================================================================
# NEW: Commitment levels and LLM artifacts
# =============================================================================

class CommitLevel(Enum):
    """Confidence level for topology decisions."""
    AUTO = "auto"           # High confidence → merge automatically
    PERIPHERY = "periphery" # Medium → soft link, no merge
    DEFER = "defer"         # Low → inquiry seed, needs more evidence


@dataclass
class PropositionSchema:
    """LLM-extracted proposition schema (Prompt A output)."""
    proposition_key: str          # Semantic key for the proposition
    value: Optional[Any] = None   # Extracted value (number, status, etc.)
    unit: Optional[str] = None    # Unit if numeric
    modality: str = "asserted"    # asserted, possible, denied, questioned
    roles: Dict[str, str] = field(default_factory=dict)  # entity → role mapping
    confidence: float = 0.5
    raw_response: str = ""
    model_hash: str = ""


@dataclass
class HappeningSchema:
    """LLM-extracted happening schema (Prompt B output)."""
    verdict: str = "unknown"      # same, different, related
    happening_key: Optional[str] = None  # If same/related, the shared key
    anchor_roles: Dict[str, str] = field(default_factory=dict)
    time_grounding: Optional[str] = None
    location_grounding: Optional[str] = None
    confidence: float = 0.5
    raw_response: str = ""
    model_hash: str = ""


@dataclass
class LLMArtifact:
    """Persisted LLM output for replay/reconcile."""
    id: str
    prompt_type: str  # "proposition" or "happening"
    claim_id: str
    candidate_id: Optional[str]  # Surface or incident being compared
    prompt_hash: str
    model_id: str
    response: Dict
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# CANDIDATE GENERATION (maximize recall)
# =============================================================================

@dataclass
class SurfaceCandidate:
    """Candidate surface for proposition matching."""
    surface_id: str
    question_key: str
    scope_id: str
    score: float  # Combined score from all signals
    signals: Dict[str, float] = field(default_factory=dict)  # signal_name → score


@dataclass
class IncidentCandidate:
    """Candidate incident for happening matching."""
    incident_id: str
    shared_anchors: Set[str] = field(default_factory=set)
    anchor_overlap: float = 0.0
    companion_overlap: float = 0.0
    is_underpowered: bool = False
    score: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)


class CandidateGenerator:
    """
    Generate candidates using multiple signals.

    Uses SurfaceRepository for data model abstraction (handles Neo4j + PostgreSQL).

    Signals:
    - Embeddings: nearest neighbors of claim text (via pgvector)
    - Graph: shared entities, anchors, time proximity (via Neo4j)
    - Lexical: normalized phrases, numbers
    """

    def __init__(self, neo4j: Neo4jService, db_pool: asyncpg.Pool, surface_repo: SurfaceRepository):
        self.neo4j = neo4j
        self.db_pool = db_pool
        self.surface_repo = surface_repo  # Use repository for data access

        # Weights for combining signals
        self.embedding_weight = 0.4
        self.entity_weight = 0.3
        self.time_weight = 0.2
        self.lexical_weight = 0.1

    async def find_surface_candidates(
        self,
        claim_embedding: Optional[List[float]],
        entities: Set[str],
        anchors: Set[str],
        scope_id: str,
        claim_time: Optional[datetime],
        top_k: int = 10,
    ) -> List[SurfaceCandidate]:
        """Find candidate surfaces for a claim using repository layer."""
        candidates: Dict[str, SurfaceCandidate] = {}

        # Compute time window
        time_window_days = 14
        if claim_time:
            time_window = (
                claim_time - timedelta(days=time_window_days),
                claim_time + timedelta(days=time_window_days),
            )
        else:
            now = datetime.utcnow()
            time_window = (now - timedelta(days=time_window_days), now + timedelta(days=1))

        # Signal 1: Embedding similarity via SurfaceRepository
        if claim_embedding:
            embedding_surface_ids = await self.surface_repo.find_candidates_by_embedding(
                claim_embedding, time_window, top_k * 2
            )
            for surface_id in embedding_surface_ids:
                if surface_id not in candidates:
                    candidates[surface_id] = await self._load_surface_candidate(surface_id)
                # Rough similarity score (could be refined by loading centroid and computing)
                candidates[surface_id].signals['embedding'] = 0.8  # Placeholder

        # Signal 2: Anchor overlap via SurfaceRepository
        if anchors:
            anchor_surface_ids = await self.surface_repo.find_candidates_by_anchor(
                anchors, time_window, top_k * 2
            )
            for surface_id in anchor_surface_ids:
                if surface_id not in candidates:
                    candidates[surface_id] = await self._load_surface_candidate(surface_id)
                candidates[surface_id].signals['anchor'] = 0.7  # Placeholder

        # Signal 3: Entity overlap (graph query)
        entity_matches = await self._find_by_entities(entities, scope_id, top_k * 2)
        for surface_id, overlap in entity_matches:
            if surface_id not in candidates:
                candidates[surface_id] = await self._load_surface_candidate(surface_id)
            candidates[surface_id].signals['entity'] = overlap

        # Signal 4: Time proximity (if available)
        if claim_time:
            time_matches = await self._find_by_time(claim_time, scope_id, top_k * 2)
            for surface_id, proximity in time_matches:
                if surface_id not in candidates:
                    candidates[surface_id] = await self._load_surface_candidate(surface_id)
                candidates[surface_id].signals['time'] = proximity

        # Compute combined scores
        for candidate in candidates.values():
            candidate.score = (
                candidate.signals.get('embedding', 0) * self.embedding_weight +
                candidate.signals.get('entity', 0) * self.entity_weight +
                candidate.signals.get('anchor', 0) * 0.3 +  # Anchor weight
                candidate.signals.get('time', 0) * self.time_weight +
                candidate.signals.get('lexical', 0) * self.lexical_weight
            )

        # Sort by score, return top_k
        sorted_candidates = sorted(candidates.values(), key=lambda c: -c.score)
        return sorted_candidates[:top_k]

    async def find_incident_candidates(
        self,
        anchor_entities: Set[str],
        companion_entities: Set[str],
        scope_id: str,
        claim_time: Optional[datetime],
        min_shared_anchors: int = 2,
        time_window_days: int = 14,
    ) -> List[IncidentCandidate]:
        """Find candidate incidents for routing."""
        candidates = []

        # Query incidents with anchor overlap
        results = await self.neo4j._execute_read("""
            MATCH (i:Incident)
            WHERE any(a IN $anchors WHERE a IN i.anchor_entities)
            RETURN i.id as id,
                   i.anchor_entities as anchors,
                   i.companion_entities as companions,
                   i.time_start as time_start
        """, {'anchors': list(anchor_entities)})

        for row in results:
            inc_anchors = set(row.get('anchors') or [])
            inc_companions = set(row.get('companions') or [])

            shared_anchors = anchor_entities & inc_anchors
            if len(shared_anchors) < min_shared_anchors:
                continue

            # Compute overlaps
            anchor_overlap = len(shared_anchors) / max(len(anchor_entities), 1)

            if companion_entities and inc_companions:
                intersection = companion_entities & inc_companions
                union = companion_entities | inc_companions
                companion_overlap = len(intersection) / len(union) if union else 0.0
                is_underpowered = False
            else:
                companion_overlap = 0.5  # Benefit of doubt
                is_underpowered = True

            # Combined score
            score = anchor_overlap * 0.6 + companion_overlap * 0.4

            candidates.append(IncidentCandidate(
                incident_id=row['id'],
                shared_anchors=shared_anchors,
                anchor_overlap=anchor_overlap,
                companion_overlap=companion_overlap,
                is_underpowered=is_underpowered,
                score=score,
                signals={
                    'anchor': anchor_overlap,
                    'companion': companion_overlap,
                }
            ))

        # Sort by score
        candidates.sort(key=lambda c: -c.score)
        return candidates

    async def _find_by_entities(
        self,
        entities: Set[str],
        scope_id: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """Find surfaces by entity overlap."""
        if not entities:
            return []

        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            WHERE s.scope_id = $scope_id
              AND any(e IN $entities WHERE e IN s.entities)
            WITH s,
                 size([e IN $entities WHERE e IN s.entities]) as shared,
                 size(s.entities) as total
            RETURN s.id as id,
                   toFloat(shared) / toFloat(size($entities) + total - shared) as jaccard
            ORDER BY jaccard DESC
            LIMIT $limit
        """, {
            'scope_id': scope_id,
            'entities': list(entities),
            'limit': limit,
        })

        return [(r['id'], r['jaccard']) for r in results]

    async def _find_by_time(
        self,
        claim_time: datetime,
        scope_id: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """Find surfaces by time proximity."""
        window_hours = 72  # 3 days

        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            WHERE s.scope_id = $scope_id
              AND s.time_start IS NOT NULL
            WITH s,
                 abs(duration.between(datetime($claim_time), datetime(s.time_start)).hours) as hours_diff
            WHERE hours_diff <= $window
            RETURN s.id as id,
                   1.0 - (toFloat(hours_diff) / $window) as proximity
            ORDER BY proximity DESC
            LIMIT $limit
        """, {
            'scope_id': scope_id,
            'claim_time': claim_time.isoformat(),
            'window': window_hours,
            'limit': limit,
        })

        return [(r['id'], r['proximity']) for r in results]

    async def _load_surface_candidate(self, surface_id: str) -> SurfaceCandidate:
        """Load surface metadata for candidate."""
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface {id: $id})
            RETURN s.question_key as qk, s.scope_id as scope
        """, {'id': surface_id})

        if results:
            return SurfaceCandidate(
                surface_id=surface_id,
                question_key=results[0].get('qk') or '',
                scope_id=results[0].get('scope') or '',
                score=0.0,
            )
        return SurfaceCandidate(surface_id=surface_id, question_key='', scope_id='', score=0.0)


# =============================================================================
# LLM ADJUDICATION
# =============================================================================

class LLMAdjudicator:
    """
    LLM-based semantic adjudication.

    Two prompts:
    - Prompt A: Proposition Schema (Surface identity)
    - Prompt B: Happening Schema (Incident membership)

    Features:
    - Self-consistency (2-3 samples)
    - Caching by (claim_id, candidate_id, prompt_hash)
    - Structured JSON output
    """

    # Prompt templates - v2 with predicate/object structure
    PROMPT_VERSION = "v2.1"  # For cache invalidation

    PROPOSITION_PROMPT = """Analyze this claim and extract its proposition schema.

CLAIM: {claim_text}

ENTITIES MENTIONED: {entities}

Return JSON with:
{{
  "predicate": "the type of assertion - be SPECIFIC not generic (e.g., 'death_count', 'passage_status', 'arrest_status', 'relationship_denial')",
  "object": "specific referent if any (e.g., 'epstein_files_act', 'tai_po_fire', 'trump_epstein_ties'), or null if truly generic",
  "value": "extracted value if numeric or categorical, null otherwise",
  "unit": "unit if value is numeric (e.g., 'people', 'dollars'), null otherwise",
  "negated": true if the claim denies/negates something, false otherwise,
  "modality": "one of: asserted, possible, denied, questioned",
  "time_scope": "point (specific moment), range (period), or unknown",
  "time_anchor": "ISO date or description if extractable, null otherwise",
  "roles": {{"entity_name": "role in proposition (e.g., 'subject', 'location', 'cause')"}},
  "confidence": 0.0-1.0
}}

IMPORTANT: Avoid overly generic predicates like "policy_status", "event_status", "person_status".
Instead use specific predicates like "legislation_passage", "fire_death_count", "arrest_announcement".

The question_key will be: "{{predicate}}:{{object or 'generic'}}"

Examples:
- "10 people died in Tai Po fire" → predicate="death_count", object="tai_po_fire"
- "Epstein Files Act passed unanimously" → predicate="legislation_passage", object="epstein_files_act"
- "Trump denied ties to Epstein" → predicate="relationship_denial", object="trump_epstein", negated=false (he's asserting the denial)
- "Fire spread to 3 buildings" → predicate="fire_spread_extent", object=null (or building name if specific)

Focus on WHAT random variable this claim is about, not the specific value."""

    PROPOSITION_COMPARE_PROMPT = """Do these two claims assert the same proposition (same random variable)?

CLAIM A: {claim_a_text}
CLAIM B: {claim_b_text}

A proposition is the same if both claims are making assertions about the SAME underlying variable,
even if the values differ. For example:
- "10 people died" and "12 people died" → SAME proposition (death count)
- "10 people died" and "He was arrested" → DIFFERENT propositions

Return JSON:
{{
  "same_proposition": true/false,
  "shared_key": "proposition key if same, null otherwise",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0
}}"""

    HAPPENING_PROMPT = """Are these two propositions about the same real-world happening (event instance)?

PROPOSITION A (from {source_a}):
  Text: {claim_a_text}
  Entities: {entities_a}
  Time: {time_a}

PROPOSITION B (from {source_b}):
  Text: {claim_b_text}
  Entities: {entities_b}
  Time: {time_b}

Two propositions are about the SAME happening if they refer to the same event instance,
not just share entities. For example:
- "Fire in Tai Po kills 10" and "Tai Po fire death toll rises to 12" → SAME happening
- "Fire in Tai Po" and "Fire in Central" → DIFFERENT happenings (same entity Hong Kong, different events)

Return JSON:
{{
  "verdict": "same" | "different" | "related",
  "happening_key": "shared happening identifier if same/related",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0
}}"""

    # LLM Veto Policy thresholds
    VETO_CONFIDENCE_THRESHOLD = 0.4  # Below this → singleton fallback
    LOW_CONFIDENCE_THRESHOLD = 0.6   # Below this → periphery commit
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # Above this → auto commit

    def __init__(self, openai_api_key: Optional[str] = None, neo4j: Optional[Neo4jService] = None):
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model = "gpt-4o-mini"  # Cost-effective for adjudication
        self.neo4j = neo4j  # For artifact persistence
        self.cache: Dict[str, Any] = {}  # In-memory cache keyed by text hash
        self.artifacts: List[LLMArtifact] = []  # Pending artifacts to persist
        self.cache_hits = 0
        self.cache_misses = 0

    async def extract_proposition(
        self,
        claim_text: str,
        entities: Set[str],
        claim_id: str,
        num_samples: int = 2,
    ) -> PropositionSchema:
        """Extract proposition schema from claim (Prompt A).

        LLM Veto Policy:
        - confidence < VETO_THRESHOLD → singleton fallback (reject LLM output)
        - confidence < LOW_THRESHOLD → periphery commit (soft link)
        - confidence >= HIGH_THRESHOLD → auto commit (full merge)
        """
        prompt = self.PROPOSITION_PROMPT.format(
            claim_text=claim_text,
            entities=', '.join(sorted(entities)) if entities else 'none',
        )
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]

        responses = await self._call_llm_with_consistency(
            prompt, claim_id, None, "proposition", num_samples
        )

        if not responses:
            # LLM call failed - singleton fallback
            return PropositionSchema(proposition_key=f"singleton_{claim_id}", confidence=0.1)

        # Check consistency on predicate field (new v2.1 schema)
        consistent, merged = self._check_consistency(responses, key_field='predicate')

        base_confidence = merged.get('confidence', 0.5)
        if not consistent:
            logger.warning(f"Inconsistent proposition extraction for {claim_id}")
            # Inconsistent responses → apply veto penalty
            merged = responses[0]
            base_confidence = base_confidence * 0.5

        # Create artifact for persistence (use 'meta' type for LLM artifacts)
        artifact = LLMArtifact(
            id=generate_id('meta'),  # LLM artifacts use 'meta' type ID
            prompt_type="proposition",
            claim_id=claim_id,
            candidate_id=None,
            prompt_hash=prompt_hash,
            model_id=self.model,
            response=merged,
        )
        self.artifacts.append(artifact)

        # Extract predicate and object from v2.1 schema
        predicate = merged.get('predicate', '')
        obj = merged.get('object')

        # Validate predicate - veto generic predicates
        GENERIC_PREDICATES = {
            'policy_status', 'event_status', 'person_status', 'status',
            'action', 'statement', 'claim', 'fact', 'news', 'report',
            'update', 'information', 'announcement', 'event'
        }
        predicate_is_generic = predicate.lower() in GENERIC_PREDICATES

        # Compose question_key from predicate:object
        if predicate and not predicate_is_generic:
            # Normalize object: lowercase, underscores, alphanumeric only
            if obj:
                obj_normalized = obj.lower().replace(' ', '_').replace('-', '_')
                obj_normalized = ''.join(c for c in obj_normalized if c.isalnum() or c == '_')
                proposition_key = f"{predicate}:{obj_normalized}"
            else:
                proposition_key = f"{predicate}:generic"
        else:
            # Fallback for legacy format or missing predicate
            proposition_key = merged.get('proposition_key', f"singleton_{claim_id}")

        # Apply LLM Veto Policy
        final_confidence = base_confidence

        # Veto if predicate is too generic
        if predicate_is_generic:
            logger.info(f"LLM veto for {claim_id}: generic predicate '{predicate}'")
            proposition_key = f"singleton_{claim_id}"
            final_confidence = 0.1

        # Veto if confidence too low
        if final_confidence < self.VETO_CONFIDENCE_THRESHOLD:
            logger.info(f"LLM veto for {claim_id}: confidence {final_confidence:.2f} < {self.VETO_CONFIDENCE_THRESHOLD}")
            proposition_key = f"singleton_{claim_id}"
            final_confidence = 0.1  # Mark as singleton

        return PropositionSchema(
            proposition_key=proposition_key,
            value=merged.get('value'),
            unit=merged.get('unit'),
            modality=merged.get('modality', 'asserted'),
            roles=merged.get('roles', {}),
            confidence=final_confidence,
            raw_response=json.dumps(merged),
            model_hash=self._model_hash(),
        )

    async def compare_propositions(
        self,
        claim_a_text: str,
        claim_b_text: str,
        claim_a_id: str,
        claim_b_id: str,
        num_samples: int = 2,
    ) -> Tuple[bool, float, str]:
        """Compare two claims for proposition sameness."""
        prompt = self.PROPOSITION_COMPARE_PROMPT.format(
            claim_a_text=claim_a_text,
            claim_b_text=claim_b_text,
        )

        responses = await self._call_llm_with_consistency(
            prompt, claim_a_id, claim_b_id, "prop_compare", num_samples
        )

        if not responses:
            return False, 0.1, "LLM call failed"

        consistent, merged = self._check_consistency(responses, key_field='same_proposition')

        same = merged.get('same_proposition', False)
        confidence = merged.get('confidence', 0.5)
        if not consistent:
            confidence *= 0.5

        return same, confidence, merged.get('reasoning', '')

    async def compare_happenings(
        self,
        claim_a_text: str,
        claim_b_text: str,
        entities_a: Set[str],
        entities_b: Set[str],
        time_a: Optional[datetime],
        time_b: Optional[datetime],
        source_a: str,
        source_b: str,
        claim_a_id: str,
        incident_id: str,
        num_samples: int = 2,
    ) -> HappeningSchema:
        """Compare claim and incident for happening sameness (Prompt B)."""
        prompt = self.HAPPENING_PROMPT.format(
            claim_a_text=claim_a_text,
            claim_b_text=claim_b_text,
            entities_a=', '.join(sorted(entities_a)) if entities_a else 'none',
            entities_b=', '.join(sorted(entities_b)) if entities_b else 'none',
            time_a=time_a.isoformat() if time_a else 'unknown',
            time_b=time_b.isoformat() if time_b else 'unknown',
            source_a=source_a,
            source_b=source_b,
        )

        responses = await self._call_llm_with_consistency(
            prompt, claim_a_id, incident_id, "happening", num_samples
        )

        if not responses:
            return HappeningSchema(verdict="unknown", confidence=0.1)

        consistent, merged = self._check_consistency(responses, key_field='verdict')

        confidence = merged.get('confidence', 0.5)
        if not consistent:
            confidence *= 0.5

        return HappeningSchema(
            verdict=merged.get('verdict', 'unknown'),
            happening_key=merged.get('happening_key'),
            confidence=confidence,
            raw_response=json.dumps(merged),
            model_hash=self._model_hash(),
        )

    def _normalize_text_for_cache(self, text: str) -> str:
        """Normalize text for cache key - catches paraphrases."""
        import re
        # Lowercase, collapse whitespace, remove punctuation
        normalized = text.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized.strip()

    async def _call_llm_with_consistency(
        self,
        prompt: str,
        claim_id: str,
        candidate_id: Optional[str],
        prompt_type: str,
        num_samples: int,
    ) -> List[Dict]:
        """Call LLM multiple times for self-consistency.

        Cache key uses prompt hash (not claim_id) so identical texts share cache.
        """
        # Use prompt hash as cache key - identical prompts get same result
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        cache_key = f"{self.PROMPT_VERSION}:{prompt_type}:{prompt_hash}"

        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for {claim_id} (key={cache_key[:20]})")
            return self.cache[cache_key]

        self.cache_misses += 1
        responses = []
        for _ in range(num_samples):
            response = await self._call_llm(prompt)
            if response:
                responses.append(response)

        if responses:
            self.cache[cache_key] = responses

        return responses

    async def _call_llm(self, prompt: str) -> Optional[Dict]:
        """Make single LLM call."""
        if not self.api_key:
            logger.warning("No OpenAI API key configured")
            return None

        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise semantic analyzer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low for consistency
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _check_consistency(
        self,
        responses: List[Dict],
        key_field: str,
    ) -> Tuple[bool, Dict]:
        """Check if responses are consistent on key field."""
        if len(responses) < 2:
            return True, responses[0] if responses else {}

        # Check if key field matches across responses
        key_values = [r.get(key_field) for r in responses]
        consistent = len(set(str(v) for v in key_values)) == 1

        # Merge: use first response, average confidence
        merged = responses[0].copy()
        if 'confidence' in merged:
            confidences = [r.get('confidence', 0.5) for r in responses]
            merged['confidence'] = sum(confidences) / len(confidences)

        return consistent, merged

    def _model_hash(self) -> str:
        """Hash of model + prompt versions for artifact tracking."""
        return hashlib.md5(f"{self.model}:v1".encode()).hexdigest()[:8]

    async def persist_artifacts(self) -> int:
        """Persist all pending LLM artifacts to Neo4j.

        Returns number of artifacts persisted.
        """
        if not self.neo4j or not self.artifacts:
            return 0

        count = 0
        for artifact in self.artifacts:
            try:
                await self.neo4j._execute_write("""
                    CREATE (a:LLMArtifact {
                        id: $id,
                        prompt_type: $prompt_type,
                        claim_id: $claim_id,
                        candidate_id: $candidate_id,
                        prompt_hash: $prompt_hash,
                        model_id: $model_id,
                        response: $response,
                        created_at: datetime()
                    })
                """, {
                    'id': artifact.id,
                    'prompt_type': artifact.prompt_type,
                    'claim_id': artifact.claim_id,
                    'candidate_id': artifact.candidate_id or '',
                    'prompt_hash': artifact.prompt_hash,
                    'model_id': artifact.model_id,
                    'response': json.dumps(artifact.response),
                })
                count += 1
            except Exception as e:
                logger.warning(f"Failed to persist LLM artifact {artifact.id}: {e}")

        self.artifacts.clear()
        return count

    async def load_cached_response(
        self,
        claim_id: str,
        candidate_id: Optional[str],
        prompt_hash: str,
    ) -> Optional[Dict]:
        """Load cached LLM response by hash (for replay/reconcile)."""
        if not self.neo4j:
            return None

        results = await self.neo4j._execute_read("""
            MATCH (a:LLMArtifact {
                claim_id: $claim_id,
                prompt_hash: $prompt_hash
            })
            WHERE a.candidate_id = $candidate_id OR ($candidate_id = '' AND a.candidate_id = '')
            RETURN a.response as response
            ORDER BY a.created_at DESC
            LIMIT 1
        """, {
            'claim_id': claim_id,
            'candidate_id': candidate_id or '',
            'prompt_hash': prompt_hash,
        })

        if results:
            try:
                return json.loads(results[0]['response'])
            except:
                pass
        return None


# =============================================================================
# WEAVER V2 WORKER
# =============================================================================

class WeaverV2Worker:
    """
    Weaver v2: LLM-assisted topology with propose → verify → commit.

    Improvements over principled_weaver:
    1. Embedding-based candidate recall
    2. LLM adjudication when uncertain
    3. Self-consistency checks
    4. Commit with uncertainty levels
    """

    QUEUE_NAME = "claims:pending"

    # Thresholds
    MIN_SHARED_ANCHORS = 2
    COMPANION_OVERLAP_THRESHOLD = 0.15
    TIME_WINDOW_DAYS = 14
    OUTLIER_Z_THRESHOLD = 3.0

    # LLM trigger thresholds
    LLM_CONFIDENCE_THRESHOLD = 0.6  # Below this, call LLM
    CANDIDATE_AMBIGUITY_THRESHOLD = 0.1  # Score gap below this → ambiguous

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        neo4j_service: Neo4jService,
        job_queue: JobQueue,
        worker_id: int = 1,
        enable_llm: bool = True,
    ):
        self.db_pool = db_pool
        self.neo4j = neo4j_service
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.enable_llm = enable_llm

        # Repositories
        self.claim_repo = ClaimRepository(db_pool, neo4j_service)
        self.surface_repo = SurfaceRepository(db_pool, neo4j_service)

        # Components - pass surface_repo to CandidateGenerator for data model access
        self.candidate_gen = CandidateGenerator(neo4j_service, db_pool, self.surface_repo)
        # Pass neo4j to adjudicator for artifact persistence
        self.adjudicator = LLMAdjudicator(neo4j=neo4j_service) if enable_llm else None

        # State (preserved from principled_weaver)
        self.scoped_key_to_surface: Dict[Tuple[str, str], str] = {}
        self.surfaces: Dict[str, Surface] = {}
        self.incidents: Dict[str, L3Incident] = {}
        self.surface_to_incident: Dict[str, str] = {}
        self.cases: Dict[str, L4Case] = {}
        self.incident_to_case: Dict[str, str] = {}
        self.typed_posteriors: Dict[str, TypedPosterior] = {}

        # Hub entities
        self.hub_entities = {
            "United States", "China", "European Union", "United Nations",
            "Asia", "Europe", "North America", "South America", "Africa"
        }

        # Meta-claims buffer
        self.pending_meta_claims: List[MetaClaim] = []

        # Stats
        self.claims_processed = 0
        self.surfaces_created = 0
        self.incidents_created = 0
        self.llm_calls = 0
        self.meta_claims_emitted = 0

        # Redis for viz
        self.redis_client = None
        self.viz_channel = "weaver:events"

        self.running = False

    async def initialize(self):
        """Load existing state from database."""
        logger.info("📥 Loading existing L2/L3 state...")

        # Load surfaces
        surfaces = await self._load_existing_surfaces()
        for surface in surfaces:
            self.surfaces[surface.id] = surface
            if surface.question_key and surface.scope_id:
                self.scoped_key_to_surface[(surface.scope_id, surface.question_key)] = surface.id

        # Load incidents
        incidents = await self._load_existing_incidents()
        for inc in incidents:
            self.incidents[inc.id] = inc
            for sid in inc.surface_ids:
                self.surface_to_incident[sid] = inc.id

        logger.info(f"   Loaded {len(self.scoped_key_to_surface)} surfaces, {len(self.incidents)} incidents")

    async def _load_existing_surfaces(self) -> List[Surface]:
        """Load surfaces from Neo4j."""
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface)
            RETURN s.id as id, s.question_key as question_key, s.scope_id as scope_id,
                   s.anchor_entities as anchors, s.entities as entities
            LIMIT 10000
        """)

        surfaces = []
        for row in results:
            surface = Surface(
                id=row['id'],
                anchor_entities=set(row.get('anchors') or []),
                entities=set(row.get('entities') or []),
                question_key=row.get('question_key'),
                scope_id=row.get('scope_id'),
            )
            surfaces.append(surface)
        return surfaces

    async def _load_existing_incidents(self) -> List[L3Incident]:
        """Load incidents from Neo4j."""
        results = await self.neo4j._execute_read("""
            MATCH (i:Incident)
            OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
            WITH i, collect(s.id) as surface_ids
            RETURN i.id as id, i.anchor_entities as anchors,
                   i.companion_entities as companions, surface_ids
            LIMIT 5000
        """)

        incidents = []
        for row in results:
            inc = L3Incident(
                id=row['id'],
                surface_ids=set(row.get('surface_ids') or []),
                anchor_entities=set(row.get('anchors') or []),
                companion_entities=set(row.get('companions') or []),
            )
            incidents.append(inc)
        return incidents

    async def process_claim(self, claim_id: str) -> Optional[LinkResult]:
        """
        Process a single claim through v2 pipeline.

        Flow:
        1. Load claim with entities
        2. Extract proposition (pattern → LLM if needed)
        3. Find surface candidates (embedding + graph)
        4. Adjudicate surface match (rules → LLM if ambiguous)
        5. Find incident candidates
        6. Adjudicate incident match (rules → LLM if ambiguous)
        7. Commit with appropriate level
        """
        self.pending_meta_claims = []

        # 1. Load claim
        claim = await self.claim_repo.get_by_id(claim_id)
        if not claim:
            logger.warning(f"⚠️ Claim {claim_id} not found")
            return None

        claim = await self.claim_repo.hydrate_entities(claim)

        entities = self._get_entity_names(claim)
        anchor_entities = self._extract_anchors(claim)
        companion_entities = entities - anchor_entities
        claim_time = self._get_claim_time(claim)

        # 2. Compute scope
        scope_id = self._compute_scope_id(anchor_entities)

        # 3. Extract proposition (question_key)
        question_key, qk_confidence = await self._extract_question_key(
            claim, entities, anchor_entities
        )

        # 4. Find/create surface
        surface_id, is_new_surface = await self._route_to_surface(
            claim, question_key, qk_confidence, scope_id, entities, anchor_entities
        )

        # 5. Route to incident
        incident_id, is_new_incident = await self._route_to_incident(
            surface_id, anchor_entities, companion_entities, claim_time
        )

        if is_new_incident:
            self.incidents_created += 1

        # 6. Persist meta-claims
        for mc in self.pending_meta_claims:
            await self._persist_meta_claim(mc)
        self.meta_claims_emitted += len(self.pending_meta_claims)

        # 7. Emit viz event
        result = LinkResult(
            claim_id=claim_id,
            surface_id=surface_id,
            incident_id=incident_id,
            is_new_surface=is_new_surface,
            is_new_incident=is_new_incident,
            question_key=question_key,
            meta_claims=self.pending_meta_claims.copy()
        )

        await self._emit_viz_event(result)

        self.claims_processed += 1
        return result

    async def _extract_question_key(
        self,
        claim: Claim,
        entities: Set[str],
        anchors: Set[str],
    ) -> Tuple[str, float]:
        """
        Extract question_key with pattern matching → LLM fallback.

        Returns (question_key, confidence).
        """
        text = (claim.text or "").lower()

        # Try pattern matching first (cheap)
        pattern_key = self._match_patterns(text)
        if pattern_key:
            return pattern_key, 0.8

        # No pattern match - need LLM or singleton
        if self.enable_llm and self.adjudicator:
            # Call LLM for proposition extraction
            schema = await self.adjudicator.extract_proposition(
                claim.text or "",
                entities,
                claim.id,
            )
            self.llm_calls += 1

            if schema.confidence >= self.LLM_CONFIDENCE_THRESHOLD:
                self.pending_meta_claims.append(MetaClaim(
                    type="llm_adjudicated",
                    claim_id=claim.id,
                    reason=f"LLM extracted proposition_key: {schema.proposition_key}",
                    evidence={
                        "proposition_key": schema.proposition_key,
                        "confidence": schema.confidence,
                        "modality": schema.modality,
                    }
                ))
                return schema.proposition_key, schema.confidence

        # Fallback to singleton
        self.pending_meta_claims.append(MetaClaim(
            type="extraction_sparse",
            claim_id=claim.id,
            reason="No pattern match, using singleton",
            evidence={"fallback": "singleton"}
        ))
        return f"singleton_{claim.id}", 0.1

    def _match_patterns(self, text: str) -> Optional[str]:
        """Pattern matching for common proposition types."""
        # Death patterns
        death_patterns = ['kill', 'dead', 'death', 'fatality', 'died', 'perish']
        if any(p in text for p in death_patterns):
            event_type = self._infer_event_type(text)
            return f"{event_type}_death_count"

        # Injury patterns
        injury_patterns = ['injur', 'wound', 'hurt', 'hospitali']
        if any(p in text for p in injury_patterns):
            event_type = self._infer_event_type(text)
            return f"{event_type}_injury_count"

        # Status patterns
        status_patterns = ['status', 'condition', 'ongoing', 'active', 'resolved']
        if any(p in text for p in status_patterns):
            event_type = self._infer_event_type(text)
            return f"{event_type}_status"

        # Policy patterns
        policy_patterns = ['announc', 'policy', 'legislation', 'bill', 'reform']
        if any(p in text for p in policy_patterns):
            return "policy_announcement"

        return None

    def _infer_event_type(self, text: str) -> str:
        """Infer event type from text."""
        if 'fire' in text or 'blaze' in text:
            return "fire"
        if 'flood' in text:
            return "flood"
        if 'earthquake' in text or 'quake' in text:
            return "earthquake"
        if 'storm' in text or 'typhoon' in text:
            return "storm"
        if 'crash' in text or 'accident' in text:
            return "accident"
        return "incident"

    async def _route_to_surface(
        self,
        claim: Claim,
        question_key: str,
        qk_confidence: float,
        scope_id: str,
        entities: Set[str],
        anchor_entities: Set[str],
    ) -> Tuple[str, bool]:
        """Route claim to surface with candidate comparison."""
        scoped_key = (scope_id, question_key)

        # Check if surface already exists for this scoped key (cache first)
        if scoped_key in self.scoped_key_to_surface:
            surface_id = self.scoped_key_to_surface[scoped_key]
            await self._update_surface(surface_id, claim, entities, anchor_entities)
            return surface_id, False

        # Check database if not in cache (handle incomplete cache)
        existing = await self.neo4j._execute_read("""
            MATCH (s:Surface {scope_id: $scope, question_key: $qk})
            RETURN s.id as id
            LIMIT 1
        """, {'scope': scope_id, 'qk': question_key})

        if existing:
            surface_id = existing[0]['id']
            self.scoped_key_to_surface[scoped_key] = surface_id  # Update cache
            await self._update_surface(surface_id, claim, entities, anchor_entities)
            return surface_id, False

        # Get embedding for candidate search
        embedding = await self.claim_repo.get_embedding(claim.id)
        claim_time = self._get_claim_time(claim)

        # Find candidates (if not singleton)
        if not question_key.startswith("singleton_") and qk_confidence >= 0.5:
            candidates = await self.candidate_gen.find_surface_candidates(
                embedding, entities, anchor_entities, scope_id, claim_time, top_k=5
            )

            # Check for ambiguous top candidates
            if len(candidates) >= 2:
                score_gap = candidates[0].score - candidates[1].score
                if score_gap < self.CANDIDATE_AMBIGUITY_THRESHOLD and self.enable_llm:
                    # Ambiguous - use LLM to adjudicate
                    same, confidence, reason = await self.adjudicator.compare_propositions(
                        claim.text or "",
                        await self._get_surface_text(candidates[0].surface_id),
                        claim.id,
                        candidates[0].surface_id,
                    )
                    self.llm_calls += 1

                    if same and confidence >= self.LLM_CONFIDENCE_THRESHOLD:
                        # Merge with existing surface
                        surface_id = candidates[0].surface_id
                        await self._update_surface(surface_id, claim, entities, anchor_entities)
                        self.pending_meta_claims.append(MetaClaim(
                            type="llm_adjudicated",
                            claim_id=claim.id,
                            surface_id=surface_id,
                            reason=f"LLM confirmed same proposition: {reason}",
                            evidence={"confidence": confidence}
                        ))
                        return surface_id, False

        # Create new surface
        surface_id = await self._create_surface(
            claim, question_key, entities, anchor_entities, scope_id
        )
        self.scoped_key_to_surface[scoped_key] = surface_id
        self.surfaces_created += 1

        return surface_id, True

    async def _route_to_incident(
        self,
        surface_id: str,
        anchor_entities: Set[str],
        companion_entities: Set[str],
        claim_time: Optional[datetime],
    ) -> Tuple[str, bool]:
        """
        Route surface to incident with bridge immunity.

        Preserved from principled_weaver with optional LLM adjudication.
        """
        # Check if already routed
        if surface_id in self.surface_to_incident:
            return self.surface_to_incident[surface_id], False

        # Find candidates
        candidates = await self.candidate_gen.find_incident_candidates(
            anchor_entities, companion_entities, "", claim_time,
            self.MIN_SHARED_ANCHORS, self.TIME_WINDOW_DAYS
        )

        # Filter by companion compatibility (bridge immunity)
        compatible = []
        for candidate in candidates:
            if candidate.is_underpowered:
                compatible.append(candidate)
            elif candidate.companion_overlap >= self.COMPANION_OVERLAP_THRESHOLD:
                compatible.append(candidate)
            else:
                # Bridge blocked
                blocking = next(iter(candidate.shared_anchors)) if candidate.shared_anchors else None
                self.pending_meta_claims.append(MetaClaim(
                    type="bridge_blocked",
                    surface_id=surface_id,
                    incident_id=candidate.incident_id,
                    reason=f"{blocking} does not bind due to incompatible companions",
                    evidence={
                        "blocking_entity": blocking,
                        "companion_overlap": candidate.companion_overlap,
                        "threshold": self.COMPANION_OVERLAP_THRESHOLD,
                    }
                ))

        if compatible:
            # Join best compatible incident
            best = compatible[0]

            # Load incident if not in cache
            if best.incident_id not in self.incidents:
                inc_data = await self.neo4j._execute_read("""
                    MATCH (i:Incident {id: $id})
                    OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
                    WITH i, collect(s.id) as sids
                    RETURN i.anchor_entities as anchors, i.companion_entities as companions, sids
                """, {'id': best.incident_id})
                if inc_data:
                    self.incidents[best.incident_id] = L3Incident(
                        id=best.incident_id,
                        surface_ids=set(inc_data[0].get('sids') or []),
                        anchor_entities=set(inc_data[0].get('anchors') or []),
                        companion_entities=set(inc_data[0].get('companions') or []),
                    )

            incident = self.incidents.get(best.incident_id)
            if not incident:
                # Still not found, create new
                incident = L3Incident(
                    id=best.incident_id,
                    surface_ids=set(),
                    anchor_entities=best.shared_anchors,
                    companion_entities=set(),
                )
                self.incidents[best.incident_id] = incident

            incident.surface_ids.add(surface_id)
            incident.anchor_entities.update(anchor_entities)
            incident.companion_entities.update(companion_entities)
            self.surface_to_incident[surface_id] = best.incident_id

            await self._persist_incident(incident)
            return best.incident_id, False
        else:
            # Create new incident
            incident_id = generate_id('incident')
            incident = L3Incident(
                id=incident_id,
                surface_ids={surface_id},
                anchor_entities=anchor_entities.copy(),
                companion_entities=companion_entities.copy(),
            )

            self.incidents[incident_id] = incident
            self.surface_to_incident[surface_id] = incident_id

            await self._persist_incident(incident)
            return incident_id, True

    # ==========================================================================
    # Helper methods (preserved from principled_weaver)
    # ==========================================================================

    def _get_entity_names(self, claim: Claim) -> Set[str]:
        """Get entity names as strings."""
        if not claim.entities:
            return set()
        names = set()
        for entity in claim.entities:
            if hasattr(entity, 'canonical_name'):
                names.add(entity.canonical_name)
            elif hasattr(entity, 'name'):
                names.add(entity.name)
            else:
                names.add(str(entity))
        return names

    def _extract_anchors(self, claim: Claim) -> Set[str]:
        """Extract anchor entities (high-IDF, non-hub)."""
        if not claim.entities:
            return set()

        anchors = set()
        for entity in claim.entities:
            name = entity.canonical_name if hasattr(entity, 'canonical_name') else str(entity)

            if hasattr(entity, 'entity_type') and entity.entity_type in ('PERSON', 'ORG'):
                anchors.add(name)
                continue

            if hasattr(entity, 'mention_count') and entity.mention_count is not None:
                if entity.mention_count < 50:
                    anchors.add(name)
                    continue

            if hasattr(entity, 'entity_type') and entity.entity_type in ('GPE', 'LOC'):
                if len(claim.entities) > 1:
                    continue

            anchors.add(name)

        return anchors

    def _compute_scope_id(self, anchor_entities: Set[str]) -> str:
        """Compute scope_id from anchors (hub filtering)."""
        scoping_anchors = anchor_entities - self.hub_entities

        if not scoping_anchors:
            scoping_anchors = anchor_entities

        normalized = sorted(a.lower().replace(" ", "").replace("'", "") for a in scoping_anchors)
        primary = normalized[:2]

        if not primary:
            return "scope_unscoped"

        return "scope_" + "_".join(primary)

    def _get_claim_time(self, claim: Claim) -> Optional[datetime]:
        """Get best available time for claim."""
        if claim.event_time:
            if isinstance(claim.event_time, datetime):
                return claim.event_time
            try:
                from dateutil.parser import parse
                return parse(str(claim.event_time))
            except:
                pass

        if claim.reported_time:
            if isinstance(claim.reported_time, datetime):
                return claim.reported_time
            try:
                from dateutil.parser import parse
                return parse(str(claim.reported_time))
            except:
                pass

        return None

    async def _get_surface_text(self, surface_id: str) -> str:
        """Get representative text for a surface."""
        results = await self.neo4j._execute_read("""
            MATCH (s:Surface {id: $id})-[:CONTAINS]->(c:Claim)
            RETURN c.text as text
            LIMIT 1
        """, {'id': surface_id})

        return results[0]['text'] if results else ""

    async def _create_surface(
        self,
        claim: Claim,
        question_key: str,
        entities: Set[str],
        anchor_entities: Set[str],
        scope_id: str,
    ) -> str:
        """Create new L2 surface."""
        surface_id = generate_id('surface')

        surface = Surface(
            id=surface_id,
            claim_ids={claim.id},
            entities=entities,
            anchor_entities=anchor_entities,
            sources={claim.page_id} if claim.page_id else set(),
            question_key=question_key,
            scope_id=scope_id,
        )

        embedding = await self.claim_repo.get_embedding(claim.id)
        if embedding:
            surface.centroid = embedding

        claim_time = self._get_claim_time(claim)
        if claim_time:
            surface.time_start = claim_time
            surface.time_end = claim_time

        await self.surface_repo.save(surface)

        await self.neo4j._execute_write("""
            MATCH (s:Surface {id: $id})
            SET s.question_key = $qk, s.scope_id = $scope, s.anchor_entities = $anchors
        """, {
            'id': surface_id,
            'qk': question_key,
            'scope': scope_id,
            'anchors': list(anchor_entities)
        })

        self.surfaces[surface_id] = surface
        return surface_id

    async def _update_surface(
        self,
        surface_id: str,
        claim: Claim,
        entities: Set[str],
        anchor_entities: Set[str],
    ):
        """Update existing surface with new claim."""
        from dateutil.parser import parse as parse_date

        surface = await self.surface_repo.get_by_id(surface_id)
        if not surface:
            return

        surface.claim_ids.add(claim.id)
        surface.entities.update(entities)
        surface.anchor_entities.update(anchor_entities)
        if claim.page_id:
            surface.sources.add(claim.page_id)

        # Always ensure surface times are datetime (not strings from DB)
        if surface.time_start and isinstance(surface.time_start, str):
            try:
                surface.time_start = parse_date(surface.time_start)
            except:
                surface.time_start = None
        if surface.time_end and isinstance(surface.time_end, str):
            try:
                surface.time_end = parse_date(surface.time_end)
            except:
                surface.time_end = None

        claim_time = self._get_claim_time(claim)
        if claim_time:
            if surface.time_start is None or claim_time < surface.time_start:
                surface.time_start = claim_time
            if surface.time_end is None or claim_time > surface.time_end:
                surface.time_end = claim_time

        await self.surface_repo.save(surface)

    async def _persist_incident(self, incident: L3Incident):
        """Save incident to Neo4j."""
        await self.neo4j._execute_write("""
            MERGE (i:Incident {id: $id})
            SET i:Story,
                i.anchor_entities = $anchors,
                i.companion_entities = $companions,
                i.updated_at = datetime()
            WITH i
            UNWIND $surface_ids as sid
            MATCH (s:Surface {id: sid})
            MERGE (i)-[:CONTAINS]->(s)
        """, {
            'id': incident.id,
            'anchors': list(incident.anchor_entities),
            'companions': list(incident.companion_entities),
            'surface_ids': list(incident.surface_ids),
        })

    async def _persist_meta_claim(self, mc: MetaClaim):
        """Save meta-claim to database."""
        await self.neo4j._execute_write("""
            CREATE (m:MetaClaim {
                id: $id,
                type: $type,
                claim_id: $claim_id,
                surface_id: $surface_id,
                incident_id: $incident_id,
                reason: $reason,
                evidence: $evidence,
                created_at: datetime()
            })
        """, {
            'id': generate_id('meta'),
            'type': mc.type,
            'claim_id': mc.claim_id or '',
            'surface_id': mc.surface_id or '',
            'incident_id': mc.incident_id or '',
            'reason': mc.reason,
            'evidence': json.dumps(mc.evidence),
        })

    async def _emit_viz_event(self, result: LinkResult):
        """Emit event for real-time visualization."""
        if not self.redis_client:
            return

        try:
            event = {
                "type": "claim_processed",
                "timestamp": datetime.utcnow().isoformat(),
                "claim_id": result.claim_id,
                "L2": {
                    "surface_id": result.surface_id,
                    "question_key": result.question_key,
                    "is_new": result.is_new_surface,
                },
                "L3": {
                    "incident_id": result.incident_id,
                    "is_new": result.is_new_incident,
                },
                "meta_claims": [
                    {"type": mc.type, "reason": mc.reason[:100]}
                    for mc in result.meta_claims
                ],
                "stats": {
                    "claims_processed": self.claims_processed,
                    "surfaces": self.surfaces_created,
                    "incidents": self.incidents_created,
                    "llm_calls": self.llm_calls,
                }
            }

            await self.redis_client.publish(self.viz_channel, json.dumps(event))
        except Exception:
            pass

    async def run(self, mode: str = 'poll'):
        """Main processing loop."""
        self.running = True
        await self.initialize()
        logger.info(f"🧬 Weaver v2 Worker {self.worker_id} starting in {mode} mode...")

        if mode == 'poll':
            await self._run_poll_mode()
        else:
            await self._run_queue_mode()

    async def _run_poll_mode(self):
        """Poll for unprocessed claims."""
        batch_size = 50
        poll_interval = 5.0

        while self.running:
            try:
                unprocessed = await self._find_unprocessed_claims(batch_size)

                if not unprocessed:
                    await asyncio.sleep(poll_interval)
                    continue

                logger.info(f"📥 Found {len(unprocessed)} unprocessed claims")

                for claim_id in unprocessed:
                    if not self.running:
                        break

                    try:
                        result = await self.process_claim(claim_id)
                        if result and self.claims_processed % 100 == 0:
                            logger.info(
                                f"📊 Progress: {self.claims_processed} claims → "
                                f"{self.surfaces_created} surfaces, "
                                f"{self.incidents_created} incidents, "
                                f"{self.llm_calls} LLM calls"
                            )
                    except Exception as e:
                        logger.error(f"❌ Error processing {claim_id}: {e}")

            except Exception as e:
                logger.error(f"❌ Poll error: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)

    async def _run_queue_mode(self):
        """Process from Redis queue."""
        while self.running:
            try:
                item = await self.job_queue.dequeue(self.QUEUE_NAME, timeout=5.0)
                if item is None:
                    continue

                claim_id = self._parse_queue_item(item)
                if claim_id:
                    await self.process_claim(claim_id)
            except Exception as e:
                logger.error(f"❌ Error: {e}", exc_info=True)

    async def _find_unprocessed_claims(self, limit: int) -> List[str]:
        """Find claims not yet processed."""
        results = await self.neo4j._execute_read("""
            MATCH (c:Claim)
            WHERE NOT (c)<-[:CONTAINS]-(:Surface)
              AND coalesce(c.weaver_failed, false) = false
            RETURN c.id as id
            ORDER BY c.created_at ASC
            LIMIT $limit
        """, {'limit': limit})
        return [r['id'] for r in results]

    def _parse_queue_item(self, item: bytes) -> Optional[str]:
        """Parse queue item."""
        try:
            data = json.loads(item)
            if isinstance(data, dict):
                return data.get('claim_id') or data.get('id')
            return str(data)
        except:
            return item.decode() if isinstance(item, bytes) else str(item)

    async def stop(self):
        """Stop worker."""
        self.running = False
        logger.info(
            f"🛑 Weaver v2 stopped. "
            f"{self.claims_processed} claims, {self.surfaces_created} surfaces, "
            f"{self.incidents_created} incidents, {self.llm_calls} LLM calls"
        )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run weaver v2."""
    import redis.asyncio as redis

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=2,
        max_size=10
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379'))
    job_queue = JobQueue(redis_client)

    mode = 'poll' if '--poll' in sys.argv else 'queue'
    enable_llm = '--no-llm' not in sys.argv

    worker = WeaverV2Worker(db_pool, neo4j, job_queue, enable_llm=enable_llm)
    worker.redis_client = redis_client

    try:
        await worker.run(mode=mode)
    except KeyboardInterrupt:
        await worker.stop()
    finally:
        await db_pool.close()
        await neo4j.close()
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
