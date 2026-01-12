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

# Import kernel trace contracts for explainability
from reee.contracts.traces import (
    DecisionTrace, FeatureVector, BeliefUpdateTrace, generate_trace_id
)
from reee.contracts.signals import (
    EpistemicSignal, SignalType, Severity, InquirySeed, signal_to_inquiry
)
from reee.explain.formatter import format_trace, TraceStyle

# Import case formation contracts for spine-based L4
from reee.contracts.case_formation import (
    EdgeType, EntityRole,
    IncidentRoleArtifact as ContractRoleArtifact,  # Keep for type reference
    SpineGateResult, EntityDFEstimate,
    evaluate_spine_gate, should_suppress_entity,
    invariant_referent_never_suppressed,
)

# Import the proven LLM-based role labeling from relational_experiment
from workers.relational_experiment import (
    label_incidents_roles_batch,
    IncidentContext,
    IncidentRoleArtifact as ExperimentRoleArtifact,
)

# Import the compiler package for membrane-based topology decisions
# The compiler is the SOLE AUTHORITY for topology mutations (spine edges, case merges)
from reee.compiler import (
    # Structural language
    Action,
    EdgeType as CompilerEdgeType,  # Avoid conflict with contracts.EdgeType
    ReferentRole,
    # Artifacts
    Referent,
    IncidentArtifact,
    # Decisions
    MembraneDecision,
    # Parameters
    CompilerParams,
    DEFAULT_PARAMS,
    # Core functions
    compile_pair,
    assert_invariants,
    # Weaver compiler
    compile_incidents,
    compile_incremental,
    CompiledEdge,
    Case as CompilerCase,
    CompilationResult,
    UnionFind,
    # Artifact extraction
    extract_artifact,
    ExtractionResult,
    InquirySeed as CompilerInquirySeed,
)


class CaseFormationMode(Enum):
    """Mode for L4 case formation."""
    SIMILARITY = "similarity"    # Current: connected-components on affinity graph
    RELATIONAL = "relational"    # New: spine + metabolic edges
    COMPILER = "compiler"        # Compiler-based: artifacts → membrane → union-find
    SHADOW = "shadow"            # Run both, compare results (production validation)


class CaseRoleMode(Enum):
    """Mode for incident role labeling.

    ARTIFACT_ONLY (default): Roles must come from LLM artifacts or persisted artifacts.
                             No heuristics. If unavailable, incident cannot contribute
                             spine edges (degrades to DEFER).

    LEGACY_HEURISTIC:        Emergency debugging only. Uses keyword-based heuristics.
                             NEVER use in production.
    """
    ARTIFACT_ONLY = "artifact_only"      # Default: LLM or persisted artifacts only
    LEGACY_HEURISTIC = "legacy_heuristic"  # Emergency debugging ONLY


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
class ReferentKey:
    """Grounded identity key for incident/case membrane.

    This is the "species marker" that defines what an incident is ABOUT.
    Computed from event_type + role-filtered entity IDs.

    Examples:
        fire: ReferentKey(event_type="fire", facility="Wang Fuk Court", location="Tai Po District")
        trial: ReferentKey(event_type="trial", defendant="Jimmy Lai", court="West Kowloon")
        legislation: ReferentKey(event_type="legislation", bill="Article 23")

    Two incidents with the same ReferentKey are about the SAME happening.
    Different ReferentKeys → different happenings (may have relations, but don't merge).
    """
    event_type: str  # fire, trial, legislation, election, etc.
    # Role-filtered entity names (will become IDs when we have entity canonicalization)
    facility: Optional[str] = None  # building, location of physical event
    location: Optional[str] = None  # district, city, country
    defendant: Optional[str] = None  # for trials
    court: Optional[str] = None  # for trials
    bill: Optional[str] = None  # for legislation
    actor: Optional[str] = None  # main actor (person/org) - weak signal
    time_bucket: Optional[str] = None  # YYYY-MM or YYYY-WW for temporal grounding

    def signature(self) -> str:
        """Stable signature for membrane matching."""
        parts = [self.event_type]
        # Order matters: most specific first
        if self.facility:
            parts.append(f"fac:{self.facility}")
        if self.location:
            parts.append(f"loc:{self.location}")
        if self.defendant:
            parts.append(f"def:{self.defendant}")
        if self.court:
            parts.append(f"crt:{self.court}")
        if self.bill:
            parts.append(f"bill:{self.bill}")
        if self.time_bucket:
            parts.append(f"t:{self.time_bucket}")
        return "|".join(parts)

    def matches(self, other: 'ReferentKey') -> Tuple[bool, float]:
        """Check if two ReferentKeys match (same happening).

        Returns (is_match, confidence).
        - is_match: True if same event_type AND same grounded referents
        - confidence: 0-1, higher if more fields match
        """
        if self.event_type != other.event_type:
            return False, 0.0

        # Count matching grounded fields
        matches = 0
        total = 0
        for field in ['facility', 'location', 'defendant', 'court', 'bill']:
            v1 = getattr(self, field)
            v2 = getattr(other, field)
            if v1 and v2:
                total += 1
                if v1 == v2:
                    matches += 1
                else:
                    # Different grounded values = different happenings
                    return False, 0.0

        if total == 0:
            # No grounded fields to compare
            return False, 0.0

        return True, matches / total


# Event type patterns for detection from predicates
EVENT_TYPE_PATTERNS = {
    'fire': ['fire', 'blaze', 'burn', 'arson', 'inferno'],
    'trial': ['trial', 'verdict', 'sentencing', 'prosecution', 'acquittal', 'conviction'],
    'legislation': ['bill', 'legislation', 'amendment', 'ordinance', 'law_passed'],
    'election': ['election', 'vote', 'ballot', 'campaign', 'candidate'],
    'death': ['death', 'died', 'killed', 'fatality', 'murder', 'assassination'],
    'appointment': ['appointed', 'nomination', 'resign', 'leadership'],
    'protest': ['protest', 'demonstration', 'rally', 'march'],
    'accident': ['accident', 'crash', 'collision', 'incident'],
}


def detect_event_type(predicates: Set[str], question_keys: Set[str]) -> Optional[str]:
    """Detect event type from predicates and question_keys."""
    text = ' '.join(predicates) + ' ' + ' '.join(question_keys)
    text = text.lower()

    for event_type, patterns in EVENT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                return event_type
    return None


@dataclass
class L3Incident:
    """L3 Incident: membrane over L2 surfaces."""
    id: str
    surface_ids: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    companion_entities: Set[str] = field(default_factory=set)
    core_motifs: List[Dict] = field(default_factory=list)
    question_keys: Set[str] = field(default_factory=set)  # Semantic keys from surfaces
    embedding: Optional[List[float]] = None  # Average of surface centroids (for L4 affinity)
    referent_key: Optional[ReferentKey] = None  # Grounded identity (membrane key)
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
    kernel_signature: str = ""  # Stable ID for deterministic case formation
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IncidentAffinity:
    """Affinity between two incidents for case formation."""
    incident1_id: str
    incident2_id: str
    object_overlap: float  # Jaccard of referent objects (e.g., wang_fuk_court_fire)
    predicate_overlap: float  # Jaccard of predicates (e.g., death_count, injury_count)
    embedding_similarity: float  # Cosine similarity of incident embeddings
    motif_overlap: float  # Jaccard of motif entity sets
    anchor_overlap: float  # Jaccard of anchor entities
    time_proximity: float  # 1.0 if same day, decays with distance
    total_affinity: float  # Weighted sum
    evidence_mass: float = 0.0  # IDF-weighted shared anchor mass (debug/audit)
    shared_mass_ratio: float = 0.0  # evidence_mass / min(total_mass) (debug/audit)

    @staticmethod
    def _parse_question_keys(keys: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Parse question_keys into predicates and objects.

        question_key format: predicate:object or predicate
        Examples:
            death_count:wang_fuk_court_fire → (death_count, wang_fuk_court_fire)
            policy_announcement → (policy_announcement, None)
        """
        predicates = set()
        objects = set()
        for key in keys:
            if ':' in key:
                parts = key.split(':', 1)
                predicates.add(parts[0])
                objects.add(parts[1])
            else:
                predicates.add(key)
        return predicates, objects

    @classmethod
    def compute(
        cls,
        inc1: 'L3Incident',
        inc2: 'L3Incident',
        embedding_weight: float = 0.40,  # Semantic similarity (tie-breaker)
        motif_weight: float = 0.25,  # Structural evidence
        anchor_weight: float = 0.25,  # IDF-weighted entity overlap
        time_weight: float = 0.10,
        max_time_days: float = 30.0,
        entity_idf: Dict[str, float] = None,  # IDF weights per entity
        hub_entities: Set[str] = None,  # DEPRECATED
        embedding_threshold: float = 0.50,  # Embedding confirmation for weak entity evidence
        evidence_mass_threshold: float = 2.5,  # Min IDF-weighted evidence (~1 specific entity)
        min_shared_mass_ratio: float = 0.55,  # Shared mass / min(total mass) must exceed this
    ) -> 'IncidentAffinity':
        """Compute affinity between two incidents.

        STABILIZATION MODE: object_overlap is FROZEN (set to 0) until we have
        grounded ReferentKey. LLM "object" strings are not identity primitives.

        Uses IDF-weighted anchor overlap instead of hub suppression:
        - w(e) = log((N + 1) / (df(e) + 1))
        - overlap = sum(w for shared) / sum(w for union)
        - evidence_mass = sum(w for shared) - must exceed threshold

        This makes "Xi Jinping" (high df) contribute near-zero,
        while "Wang Fuk Court" (low df) contributes strongly.

        Args:
            entity_idf: IDF weights per entity (computed by CaseBuilder)
            hub_entities: DEPRECATED - kept for compatibility
            embedding_threshold: Min embedding similarity to consider as evidence
            evidence_mass_threshold: Min weighted overlap mass to consider as evidence
        """
        entity_idf = entity_idf or {}

        # =====================
        # PARSE QUESTION_KEYS (for predicate_overlap only)
        # =====================
        preds1, _ = cls._parse_question_keys(inc1.question_keys or set())
        preds2, _ = cls._parse_question_keys(inc2.question_keys or set())

        # =====================
        # OBJECT OVERLAP - FROZEN
        # =====================
        object_overlap = 0.0  # FROZEN until ReferentKey is implemented

        # =====================
        # PREDICATE OVERLAP (SUPPORTING SIGNAL ONLY)
        # =====================
        predicate_overlap = 0.0
        if preds1 and preds2:
            predicate_overlap = len(preds1 & preds2) / len(preds1 | preds2)

        # =====================
        # EMBEDDING SIMILARITY (SEMANTIC)
        # =====================
        embedding_similarity = 0.0
        if inc1.embedding and inc2.embedding:
            emb1 = np.array(inc1.embedding)
            emb2 = np.array(inc2.embedding)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 > 0 and norm2 > 0:
                embedding_similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))

        # =====================
        # MOTIF OVERLAP (STRUCTURAL)
        # =====================
        # Use IDF to weight motif entities too
        motifs1 = {frozenset(m['entities']) for m in inc1.core_motifs}
        motifs2 = {frozenset(m['entities']) for m in inc2.core_motifs}
        motif_overlap = 0.0
        if motifs1 and motifs2:
            # Simple Jaccard for now (could IDF-weight motifs later)
            motif_overlap = len(motifs1 & motifs2) / len(motifs1 | motifs2)

        # =====================
        # ANCHOR OVERLAP (IDF-WEIGHTED)
        # =====================
        # Use IDF weights instead of hub suppression
        # w(e) = log((N+1)/(df(e)+1)), pre-computed by CaseBuilder
        anchors1 = inc1.anchor_entities
        anchors2 = inc2.anchor_entities

        anchor_overlap = 0.0
        evidence_mass = 0.0
        shared_mass_ratio = 0.0

        if anchors1 and anchors2:
            shared = anchors1 & anchors2
            union = anchors1 | anchors2

            # Weighted sums
            # Missing IDF defaults to 0.0 (conservative) to avoid accidental linking.
            shared_weight = sum(entity_idf.get(e, 0.0) for e in shared)
            union_weight = sum(entity_idf.get(e, 0.0) for e in union)
            total1 = sum(entity_idf.get(e, 0.0) for e in anchors1)
            total2 = sum(entity_idf.get(e, 0.0) for e in anchors2)

            if union_weight > 0:
                anchor_overlap = shared_weight / union_weight

            evidence_mass = shared_weight
            min_total = min(total1, total2)
            if min_total > 0:
                shared_mass_ratio = shared_weight / min_total

        # =====================
        # TIME PROXIMITY
        # =====================
        time_proximity = 0.0
        if inc1.time_start and inc2.time_start:
            delta_days = abs((inc1.time_start - inc2.time_start).total_seconds()) / 86400
            if delta_days <= max_time_days:
                time_proximity = 1.0 - (delta_days / max_time_days)

        # =====================
        # STRUCTURAL EVIDENCE GATE
        # =====================
        # Embeddings are for RECALL + TIE-BREAK, not identity.
        # Must have STRUCTURAL evidence to link:
        # 1. Motif overlap > 0 (shared entity pairs), OR
        # 2. Sufficient evidence mass (IDF-weighted anchor overlap)
        #
        # Embeddings boost affinity but cannot open the gate alone.
        has_structural_evidence = motif_overlap > 0
        # Entity evidence:
        # 1. Strong: high IDF-weighted mass (multiple discriminative entities)
        # 2. Confirmed: high ratio + embedding confirmation (same topic)
        has_strong_entity_evidence = evidence_mass >= evidence_mass_threshold
        has_confirmed_entity_evidence = (
            shared_mass_ratio >= min_shared_mass_ratio and
            embedding_similarity >= embedding_threshold
        )
        has_entity_evidence = has_strong_entity_evidence or has_confirmed_entity_evidence

        if not (has_structural_evidence or has_entity_evidence):
            # No sufficient evidence - cannot link
            total = 0.0
        else:
            # Structural evidence present - embeddings contribute to affinity
            total = (embedding_weight * embedding_similarity +
                     motif_weight * motif_overlap +
                     anchor_weight * anchor_overlap +
                     time_weight * time_proximity +
                     0.05 * predicate_overlap)

        return cls(
            incident1_id=inc1.id,
            incident2_id=inc2.id,
            object_overlap=object_overlap,
            predicate_overlap=predicate_overlap,
            embedding_similarity=embedding_similarity,
            motif_overlap=motif_overlap,
            anchor_overlap=anchor_overlap,
            time_proximity=time_proximity,
            evidence_mass=evidence_mass,
            shared_mass_ratio=shared_mass_ratio,
            total_affinity=total,
        )


# =============================================================================
# Commitment levels and Fallback Chain (must be before LinkResult)
# =============================================================================

class CommitLevel(Enum):
    """Confidence level for topology decisions."""
    AUTO = "auto"           # High confidence → merge automatically
    PERIPHERY = "periphery" # Medium → soft link, no merge
    DEFER = "defer"         # Low → inquiry seed, needs more evidence


class FallbackChain(Enum):
    """
    Question key extraction fallback chain.

    Each level represents decreasing semantic precision.
    The kernel's explicit hierarchy - now with LLM as EXPLICIT.
    """
    EXPLICIT = "explicit"     # LLM-extracted proposition_key (highest)
    PATTERN = "pattern"       # Matched typed question pattern
    ENTITY = "entity"         # Derived from anchor entities
    PAGE_SCOPE = "page_scope" # Fell back to page/source scope
    SINGLETON = "singleton"   # No collapse possible (lowest)


# Confidence thresholds for each fallback level
FALLBACK_CONFIDENCE = {
    FallbackChain.EXPLICIT: 0.85,
    FallbackChain.PATTERN: 0.75,
    FallbackChain.ENTITY: 0.50,
    FallbackChain.PAGE_SCOPE: 0.30,
    FallbackChain.SINGLETON: 0.10,
}


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
    # Kernel-style traces for explainability
    decision_traces: List[DecisionTrace] = field(default_factory=list)
    signals: List[EpistemicSignal] = field(default_factory=list)
    fallback_chain_level: Optional[FallbackChain] = None

    def format_traces(self, style: TraceStyle = TraceStyle.SHORT) -> List[str]:
        """Format all decision traces for display."""
        return [format_trace(t, style) for t in self.decision_traces]


# =============================================================================
# LLM artifacts
# =============================================================================


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
    5. Kernel-style traces for explainability (DecisionTrace, EpistemicSignal)
    """

    QUEUE_NAME = "claims:pending"
    KERNEL_VERSION = "v2.1"  # For trace attribution

    # Thresholds
    MIN_SHARED_ANCHORS = 2
    COMPANION_OVERLAP_THRESHOLD = 0.15
    TIME_WINDOW_DAYS = 14
    OUTLIER_Z_THRESHOLD = 3.0

    # LLM trigger thresholds
    LLM_CONFIDENCE_THRESHOLD = 0.6  # Below this, call LLM
    CANDIDATE_AMBIGUITY_THRESHOLD = 0.1  # Score gap below this → ambiguous

    # Compute params hash for trace attribution
    @classmethod
    def _compute_params_hash(cls) -> str:
        """Hash of key parameters for trace attribution."""
        params = f"{cls.MIN_SHARED_ANCHORS}:{cls.COMPANION_OVERLAP_THRESHOLD}:{cls.TIME_WINDOW_DAYS}"
        return hashlib.md5(params.encode()).hexdigest()[:8]

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

        # Kernel-style trace/signal buffers (cleared per claim)
        self.pending_traces: List[DecisionTrace] = []
        self.pending_signals: List[EpistemicSignal] = []
        self.current_fallback_level: Optional[FallbackChain] = None
        self.params_hash = self._compute_params_hash()

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
                   i.companion_entities as companions,
                   i.core_motifs as motifs,
                   surface_ids
            LIMIT 5000
        """)

        incidents = []
        for row in results:
            # Parse motifs from JSON string
            motifs_raw = row.get('motifs')
            core_motifs = []
            if motifs_raw:
                try:
                    core_motifs = json.loads(motifs_raw) if isinstance(motifs_raw, str) else motifs_raw
                except (json.JSONDecodeError, TypeError):
                    pass

            inc = L3Incident(
                id=row['id'],
                surface_ids=set(row.get('surface_ids') or []),
                anchor_entities=set(row.get('anchors') or []),
                companion_entities=set(row.get('companions') or []),
                core_motifs=core_motifs,
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
        # Clear per-claim buffers
        self.pending_meta_claims = []
        self.pending_traces = []
        self.pending_signals = []
        self.current_fallback_level = None

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

        # 7. Build result with traces for explainability
        result = LinkResult(
            claim_id=claim_id,
            surface_id=surface_id,
            incident_id=incident_id,
            is_new_surface=is_new_surface,
            is_new_incident=is_new_incident,
            question_key=question_key,
            meta_claims=self.pending_meta_claims.copy(),
            decision_traces=self.pending_traces.copy(),
            signals=self.pending_signals.copy(),
            fallback_chain_level=self.current_fallback_level,
        )

        # Log traces in DEBUG mode
        if logger.isEnabledFor(logging.DEBUG):
            for trace in result.decision_traces:
                logger.debug(f"TRACE: {format_trace(trace, TraceStyle.LOG)}")

        # Process signals → inquiries (ensures signals don't become dead-ends)
        await self._process_signals_to_inquiries()

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
        Extract question_key using explicit fallback chain:
        EXPLICIT(LLM) → PATTERN → ENTITY → PAGE_SCOPE → SINGLETON

        Returns (question_key, confidence).
        Emits DecisionTrace for transparency.
        """
        text = (claim.text or "").lower()
        rules_fired: Set[str] = set()

        # ==================
        # Level 1: PATTERN (cheap, deterministic)
        # ==================
        pattern_key = self._match_patterns(text)
        if pattern_key:
            self.current_fallback_level = FallbackChain.PATTERN
            rules_fired.add("FALLBACK_PATTERN")
            self._emit_surface_key_trace(
                claim.id, None, "key_pattern", pattern_key, rules_fired,
                FeatureVector(question_key_confidence=0.75)
            )
            return pattern_key, FALLBACK_CONFIDENCE[FallbackChain.PATTERN]

        # ==================
        # Level 2: EXPLICIT (LLM-extracted)
        # ==================
        if self.enable_llm and self.adjudicator:
            schema = await self.adjudicator.extract_proposition(
                claim.text or "",
                entities,
                claim.id,
            )
            self.llm_calls += 1

            if schema.confidence >= self.LLM_CONFIDENCE_THRESHOLD:
                self.current_fallback_level = FallbackChain.EXPLICIT
                rules_fired.add("FALLBACK_EXPLICIT")
                self._emit_surface_key_trace(
                    claim.id, None, "key_explicit", schema.proposition_key, rules_fired,
                    FeatureVector(question_key_confidence=schema.confidence)
                )
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

        # ==================
        # Level 3: ENTITY (derive from anchors)
        # ==================
        # Guards:
        # 1. Require ≥2 anchors for ENTITY fallback (single anchor is underpowered)
        # 2. Filter out attribution/publication entities (they're not referents)
        if len(anchors) >= 2:
            # Filter out attribution entities (publications, journals, news orgs)
            referent_anchors = self._filter_attribution_entities(anchors)

            if len(referent_anchors) >= 2:
                entity_key = "_".join(sorted(list(referent_anchors)[:2]))
                self.current_fallback_level = FallbackChain.ENTITY
                rules_fired.add("FALLBACK_ENTITY")
                rules_fired.add("ENTITY_MULTI_ANCHOR")
                self._emit_surface_key_trace(
                    claim.id, None, "key_entity", entity_key, rules_fired,
                    FeatureVector(question_key_confidence=0.5)
                )
                return entity_key, FALLBACK_CONFIDENCE[FallbackChain.ENTITY]
            else:
                # Had anchors but all were attribution entities
                rules_fired.add("ENTITY_ATTRIBUTION_VETO")
        elif len(anchors) == 1:
            # Single anchor - check if it's a referent (not attribution)
            single_anchor = list(anchors)[0]
            if not self._is_attribution_entity(single_anchor):
                # Single referent anchor - use PAGE_SCOPE with anchor hint
                self.current_fallback_level = FallbackChain.PAGE_SCOPE
                rules_fired.add("FALLBACK_PAGE_SCOPE")
                rules_fired.add("SINGLE_REFERENT_ANCHOR")
                page_key = f"page_{claim.page_id or 'unknown'}_{single_anchor[:20]}"
                self._emit_surface_key_trace(
                    claim.id, None, "key_page_scope", page_key, rules_fired,
                    FeatureVector(question_key_confidence=0.3)
                )
                # Emit signal for underpowered scope
                self._emit_signal(
                    SignalType.SCOPE_UNDERPOWERED,
                    claim.id, "claim",
                    {"reason": "Single referent anchor - using page scope", "anchor": single_anchor},
                    "Add more entity context for better scoping"
                )
                return page_key, FALLBACK_CONFIDENCE[FallbackChain.PAGE_SCOPE]
            else:
                # Single attribution entity - force singleton
                rules_fired.add("SINGLE_ATTRIBUTION_VETO")

        # ==================
        # Level 4: SINGLETON (no collapse possible)
        # ==================
        self.current_fallback_level = FallbackChain.SINGLETON
        rules_fired.add("FALLBACK_SINGLETON")
        singleton_key = f"singleton_{claim.id}"
        self._emit_surface_key_trace(
            claim.id, None, "key_singleton", singleton_key, rules_fired,
            FeatureVector(question_key_confidence=0.1)
        )
        # Emit signal for sparse extraction
        self._emit_signal(
            SignalType.EXTRACTION_SPARSE,
            claim.id, "claim",
            {"reason": "No pattern/LLM/entity fallback available"},
            "Re-extract with better entity recognition"
        )
        return singleton_key, FALLBACK_CONFIDENCE[FallbackChain.SINGLETON]

    def _emit_surface_key_trace(
        self,
        subject_id: str,
        target_id: Optional[str],
        outcome: str,
        question_key: str,
        rules: Set[str],
        features: FeatureVector,
    ):
        """Emit a DecisionTrace for surface key extraction."""
        trace = DecisionTrace(
            id=generate_trace_id(),
            decision_type="surface_key",
            subject_id=subject_id,
            target_id=target_id,
            candidate_ids=frozenset(),
            outcome=outcome,
            features=features,
            rules_fired=frozenset(rules),
            params_hash=self.params_hash,
            kernel_version=self.KERNEL_VERSION,
            timestamp=datetime.utcnow(),
        )
        self.pending_traces.append(trace)

    def _emit_signal(
        self,
        signal_type: SignalType,
        subject_id: str,
        subject_type: str,
        evidence: Dict[str, Any],
        resolution_hint: Optional[str] = None,
    ):
        """Emit an EpistemicSignal."""
        from reee.contracts.signals import generate_signal_id
        signal = EpistemicSignal(
            id=generate_signal_id(),
            signal_type=signal_type,
            subject_id=subject_id,
            subject_type=subject_type,
            severity=Severity.WARNING,
            evidence=evidence,
            resolution_hint=resolution_hint,
            timestamp=datetime.utcnow(),
        )
        self.pending_signals.append(signal)

    def _emit_incident_routing_trace(
        self,
        surface_id: str,
        target_id: str,
        outcome: str,
        candidates: List[str],
        rules: Set[str],
        features: FeatureVector,
    ):
        """Emit a DecisionTrace for incident routing."""
        trace = DecisionTrace(
            id=generate_trace_id(),
            decision_type="incident_membership",
            subject_id=surface_id,
            target_id=target_id,
            candidate_ids=frozenset(candidates),
            outcome=outcome,
            features=features,
            rules_fired=frozenset(rules),
            params_hash=self.params_hash,
            kernel_version=self.KERNEL_VERSION,
            timestamp=datetime.utcnow(),
        )
        self.pending_traces.append(trace)

    def _emit_surface_routing_trace(
        self,
        claim_id: str,
        target_id: str,
        outcome: str,
        candidates: FrozenSet[str],
        rules: Set[str],
        features: FeatureVector,
    ):
        """Emit a DecisionTrace for surface routing (L2 membership)."""
        trace = DecisionTrace(
            id=generate_trace_id(),
            decision_type="surface_membership",
            subject_id=claim_id,
            target_id=target_id,
            candidate_ids=candidates,
            outcome=outcome,
            features=features,
            rules_fired=frozenset(rules),
            params_hash=self.params_hash,
            kernel_version=self.KERNEL_VERSION,
            timestamp=datetime.utcnow(),
        )
        self.pending_traces.append(trace)

    async def _process_signals_to_inquiries(self):
        """
        Convert pending signals to InquirySeeds and persist/log them.

        This ensures signals don't become dead-ends like the old MetaClaim system.
        Each signal that warrants action gets converted to an actionable inquiry.
        """
        if not self.pending_signals:
            return

        inquiries_generated = 0
        for signal in self.pending_signals:
            # Convert signal to inquiry using the contracts module
            inquiry = signal_to_inquiry(signal)
            if inquiry is None:
                # Signal doesn't warrant inquiry (informational only)
                logger.debug(f"Signal {signal.signal_type.value} (informational, no inquiry)")
                continue

            inquiries_generated += 1

            # Log the inquiry at INFO level for visibility
            logger.info(
                f"INQUIRY: type={inquiry.inquiry_type.value} "
                f"priority={inquiry.priority:.2f} "
                f"subject={inquiry.subject_id} "
                f"question=\"{inquiry.question[:60]}...\""
            )

            # Persist inquiry to Neo4j for later processing
            try:
                await self.neo4j._execute_write("""
                    CREATE (inq:InquirySeed {
                        id: $id,
                        inquiry_type: $inquiry_type,
                        subject_id: $subject_id,
                        priority: $priority,
                        question: $question,
                        current_state: $current_state,
                        evidence_needed: $evidence_needed,
                        source_signal_id: $source_signal_id,
                        status: 'pending',
                        created_at: datetime()
                    })
                """, {
                    'id': inquiry.id,
                    'inquiry_type': inquiry.inquiry_type.value,
                    'subject_id': inquiry.subject_id,
                    'priority': inquiry.priority,
                    'question': inquiry.question,
                    'current_state': json.dumps(inquiry.current_state),
                    'evidence_needed': inquiry.evidence_needed,
                    'source_signal_id': inquiry.source_signal_id or '',
                })
            except Exception as e:
                logger.warning(f"Failed to persist InquirySeed {inquiry.id}: {e}")

            # Also persist the source signal for audit trail
            try:
                await self.neo4j._execute_write("""
                    MERGE (sig:EpistemicSignal {id: $id})
                    SET sig.signal_type = $signal_type,
                        sig.subject_id = $subject_id,
                        sig.subject_type = $subject_type,
                        sig.severity = $severity,
                        sig.evidence = $evidence,
                        sig.resolution_hint = $resolution_hint,
                        sig.created_at = datetime($timestamp)
                """, {
                    'id': signal.id,
                    'signal_type': signal.signal_type.value,
                    'subject_id': signal.subject_id,
                    'subject_type': signal.subject_type,
                    'severity': signal.severity.value,
                    'evidence': json.dumps(signal.evidence),
                    'resolution_hint': signal.resolution_hint or '',
                    'timestamp': signal.timestamp.isoformat(),
                })
            except Exception as e:
                logger.warning(f"Failed to persist EpistemicSignal {signal.id}: {e}")

        if inquiries_generated > 0:
            logger.debug(f"Generated {inquiries_generated} inquiries from {len(self.pending_signals)} signals")

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
            # Emit trace for joining by scoped key (cache hit)
            self._emit_surface_routing_trace(
                claim.id, surface_id, "joined_by_scoped_key",
                frozenset(), {"SCOPED_KEY_CACHE_HIT"},
                FeatureVector(question_key_confidence=qk_confidence)
            )
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
            # Emit trace for joining by scoped key (DB hit)
            self._emit_surface_routing_trace(
                claim.id, surface_id, "joined_by_scoped_key",
                frozenset(), {"SCOPED_KEY_DB_HIT"},
                FeatureVector(question_key_confidence=qk_confidence)
            )
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
            # Emit trace for joining existing incident
            self._emit_incident_routing_trace(
                surface_id=surface_id,
                target_id=best.incident_id,
                outcome="joined",
                candidates=[c.incident_id for c in candidates],
                rules={"ANCHOR_OVERLAP_PASS", "COMPANION_COMPATIBLE"},
                features=FeatureVector(
                    anchor_overlap=best.anchor_overlap,
                    companion_jaccard=best.companion_overlap,
                )
            )

            # Load incident if not in cache
            if best.incident_id not in self.incidents:
                inc_data = await self.neo4j._execute_read("""
                    MATCH (i:Incident {id: $id})
                    OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
                    WITH i, collect(s.id) as sids
                    RETURN i.anchor_entities as anchors,
                           i.companion_entities as companions,
                           i.core_motifs as motifs,
                           sids
                """, {'id': best.incident_id})
                if inc_data:
                    # Parse motifs from JSON string
                    motifs_raw = inc_data[0].get('motifs')
                    core_motifs = []
                    if motifs_raw:
                        try:
                            core_motifs = json.loads(motifs_raw) if isinstance(motifs_raw, str) else motifs_raw
                        except (json.JSONDecodeError, TypeError):
                            pass

                    self.incidents[best.incident_id] = L3Incident(
                        id=best.incident_id,
                        surface_ids=set(inc_data[0].get('sids') or []),
                        anchor_entities=set(inc_data[0].get('anchors') or []),
                        companion_entities=set(inc_data[0].get('companions') or []),
                        core_motifs=core_motifs,
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

            # Emit trace for creating new incident
            self._emit_incident_routing_trace(
                surface_id=surface_id,
                target_id=incident_id,
                outcome="created_new",
                candidates=[c.incident_id for c in candidates],  # All rejected candidates
                rules={"NO_COMPATIBLE_CANDIDATES"},
                features=FeatureVector()
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

    # Attribution entity patterns - these are sources, not referents
    ATTRIBUTION_PATTERNS = {
        # Publications/Journals
        'journal', 'magazine', 'newspaper', 'gazette', 'times', 'post',
        'herald', 'tribune', 'chronicle', 'review', 'quarterly',
        # News organizations
        'news', 'media', 'press', 'broadcasting', 'network', 'wire',
        'reuters', 'associated press', 'afp', 'bbc', 'cnn', 'fox',
        # Publishers
        'publishing', 'publications', 'press', 'books',
        # Generic attribution markers
        'report', 'study', 'analysis', 'institute', 'foundation',
        'according to', 'says', 'stated', 'announced',
    }

    def _is_attribution_entity(self, entity_name: str) -> bool:
        """Check if entity is an attribution source rather than a referent.

        Attribution entities are publications, journals, news orgs - they're
        WHO said something, not WHAT/WHO the claim is about.
        """
        name_lower = entity_name.lower()

        # Check for attribution patterns
        for pattern in self.ATTRIBUTION_PATTERNS:
            if pattern in name_lower:
                return True

        # Check for "Journal of X" or "X Journal" patterns
        if 'journal of' in name_lower or name_lower.endswith(' journal'):
            return True

        # Check for news domain patterns (e.g., "BBC News", "CNN")
        if any(org in name_lower for org in ['bbc', 'cnn', 'reuters', 'ap ', 'afp']):
            return True

        return False

    def _filter_attribution_entities(self, entities: Set[str]) -> Set[str]:
        """Filter out attribution entities, keeping only referents.

        Referents are the subjects of claims (people, places, events).
        Attribution entities are sources (journals, news orgs, publishers).
        """
        return {e for e in entities if not self._is_attribution_entity(e)}

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
        # Keep in-memory cache in sync (used for incident motif computation).
        self.surfaces[surface_id] = surface

    def _compute_incident_motifs(self, incident: L3Incident, min_support: int = 2) -> List[Dict]:
        """Extract motifs (entity k-sets) from surfaces in an incident.

        A motif is a set of 2+ entities that co-occur across multiple surfaces.
        This provides structural evidence for L4 case formation.

        Args:
            incident: The incident to compute motifs for
            min_support: Minimum number of surfaces an entity pair must appear in

        Returns:
            List of motif dicts with entities, support count, and weight
        """
        from itertools import combinations
        from collections import Counter

        # Collect entity sets from each surface
        surface_entity_sets = []
        for sid in incident.surface_ids:
            surface = self.surfaces.get(sid)
            if surface and surface.entities:
                surface_entity_sets.append(frozenset(surface.entities))

        if len(surface_entity_sets) < 2:
            return []

        # Count 2-set co-occurrences across surfaces
        pair_counts: Counter = Counter()
        for entity_set in surface_entity_sets:
            # Generate all pairs from this surface
            entities = sorted(entity_set)[:10]  # Limit to avoid combinatorial explosion
            for pair in combinations(entities, 2):
                pair_counts[frozenset(pair)] += 1

        # Filter to supported motifs
        motifs = []
        for pair, count in pair_counts.most_common(20):  # Top 20 motifs
            if count >= min_support:
                motifs.append({
                    "entities": sorted(pair),
                    "support": count,
                    "weight": count / len(surface_entity_sets),  # Normalized support
                })

        return motifs

    async def _persist_incident(self, incident: L3Incident):
        """Save incident to Neo4j with motifs."""
        # Compute motifs from surfaces
        incident.core_motifs = self._compute_incident_motifs(incident)

        # Serialize motifs for Neo4j
        motifs_json = json.dumps(incident.core_motifs) if incident.core_motifs else "[]"

        await self.neo4j._execute_write("""
            MERGE (i:Incident {id: $id})
            SET i:Story,
                i.anchor_entities = $anchors,
                i.companion_entities = $companions,
                i.core_motifs = $motifs,
                i.updated_at = datetime()
            WITH i
            UNWIND $surface_ids as sid
            MATCH (s:Surface {id: sid})
            MERGE (i)-[:CONTAINS]->(s)
        """, {
            'id': incident.id,
            'anchors': list(incident.anchor_entities),
            'companions': list(incident.companion_entities),
            'motifs': motifs_json,
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
# L4 CASE BUILDER - Deterministic emergence over incident affinity
# =============================================================================

class CaseBuilder:
    """
    Forms L4 Cases as connected components over incident affinity graph.

    This is a pure view computation - deterministic emergence from L3.
    Cases = connected components where incidents have affinity above threshold.

    Affinity uses multiple signals, but it enforces a hard gate to avoid
    "giant components" caused by transitive chaining on weak overlaps.

    Signals (weighted):
    - Embedding similarity (tie-breaker): semantic similarity of incident embeddings
    - Motif overlap: structural recurrence across surfaces
    - Anchor overlap: IDF-weighted entity overlap (discriminativeness-aware)
    - Time proximity: weak regularizer
    - Predicate overlap: tiny supporting signal only

    Gate (must pass at least one):
    - Motif overlap > 0, OR
    - Entity evidence is both strong and concentrated:
      evidence_mass >= threshold AND shared_mass_ratio >= min_shared_mass_ratio

    Embeddings do NOT open the gate by themselves (recall/tie-break only).
    """

    # Use same channel as weaver viz events - canonical worker subscribes
    VIZ_CHANNEL = "weaver:events"

    def __init__(
        self,
        neo4j: 'Neo4jService',
        db_pool: 'asyncpg.Pool' = None,  # PostgreSQL for surface centroids
        redis_client=None,  # Optional Redis for queue emission
        affinity_threshold: float = 0.20,  # Minimum affinity to link incidents
        # STABILIZATION MODE: object_weight frozen, weights redistributed
        embedding_weight: float = 0.40,  # Semantic similarity (primary when available)
        motif_weight: float = 0.25,  # Structural evidence
        anchor_weight: float = 0.25,  # Entity overlap (non-hub)
        time_weight: float = 0.10,
        max_time_days: float = 30.0,
        embedding_threshold: float = 0.75,  # Min embedding similarity as evidence
        # Entity-evidence gate parameters (to prevent "giant components")
        entity_df_cap: Optional[int] = None,  # Max df considered "discriminative"; default sqrt(N)
        min_shared_mass_ratio: float = 0.55,  # Shared mass / min(total mass) must exceed this
    ):
        self.neo4j = neo4j
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.affinity_threshold = affinity_threshold
        self.embedding_weight = embedding_weight
        self.motif_weight = motif_weight
        self.anchor_weight = anchor_weight
        self.time_weight = time_weight
        self.max_time_days = max_time_days
        self.embedding_threshold = embedding_threshold
        self.entity_df_cap = entity_df_cap
        self.min_shared_mass_ratio = min_shared_mass_ratio

        self.incidents: Dict[str, L3Incident] = {}
        self.cases: Dict[str, L4Case] = {}
        self.hub_entities: Set[str] = set()

    async def load_incidents(self) -> List[L3Incident]:
        """Load all incidents with their motifs AND embeddings from surface centroids.

        Incident embedding = average of surface centroids (from PostgreSQL).
        This is the PRIMARY semantic signal for L4 case formation.
        """
        results = await self.neo4j._execute_read("""
            MATCH (i:Incident)
            OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
            WITH i, collect(s.id) as surface_ids, collect(s.question_key) as question_keys
            RETURN i.id as id,
                   i.anchor_entities as anchors,
                   i.companion_entities as companions,
                   i.core_motifs as motifs,
                   i.time_start as time_start,
                   surface_ids,
                   question_keys
            LIMIT 10000
        """)

        # Collect all surface IDs to batch-load centroids
        all_surface_ids = set()
        incident_data = []
        for row in results:
            surface_ids = set(row.get('surface_ids') or [])
            all_surface_ids.update(surface_ids)
            incident_data.append(row)

        # Batch load surface centroids from PostgreSQL
        surface_centroids = {}
        if self.db_pool and all_surface_ids:
            surface_centroids = await self._load_surface_centroids(list(all_surface_ids))
            logger.info(f"Loaded {len(surface_centroids)} surface centroids from PostgreSQL")

        incidents = []
        for row in incident_data:
            # Parse motifs
            motifs_raw = row.get('motifs')
            core_motifs = []
            if motifs_raw:
                try:
                    core_motifs = json.loads(motifs_raw) if isinstance(motifs_raw, str) else motifs_raw
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse time
            time_start = None
            if row.get('time_start'):
                ts = row['time_start']
                if hasattr(ts, 'to_native'):
                    time_start = ts.to_native()
                elif isinstance(ts, datetime):
                    time_start = ts

            # Parse question_keys from surfaces
            raw_keys = row.get('question_keys') or []
            question_keys = {k for k in raw_keys if k}

            # Compute incident embedding = average of surface centroids
            surface_ids = set(row.get('surface_ids') or [])
            incident_embedding = self._compute_incident_embedding(surface_ids, surface_centroids)

            inc = L3Incident(
                id=row['id'],
                surface_ids=surface_ids,
                anchor_entities=set(row.get('anchors') or []),
                companion_entities=set(row.get('companions') or []),
                core_motifs=core_motifs,
                question_keys=question_keys,
                embedding=incident_embedding,
                time_start=time_start,
            )
            incidents.append(inc)
            self.incidents[inc.id] = inc

        # Log embedding coverage
        emb_count = sum(1 for i in incidents if i.embedding)
        logger.info(f"Loaded {len(incidents)} incidents:")
        logger.info(f"  {emb_count}/{len(incidents)} have embeddings (semantic signal)")

        return incidents

    async def _load_surface_centroids(self, surface_ids: List[str]) -> Dict[str, List[float]]:
        """Batch load surface centroids from PostgreSQL (pgvector format)."""
        if not surface_ids:
            return {}

        async with self.db_pool.acquire() as conn:
            # Register pgvector codec for proper decoding
            await conn.set_type_codec(
                'vector',
                encoder=lambda v: v,
                decoder=lambda v: np.array([float(x) for x in v[1:-1].split(',')]),
                schema='public',
                format='text'
            )

            rows = await conn.fetch("""
                SELECT surface_id, centroid
                FROM content.surface_centroids
                WHERE surface_id = ANY($1)
            """, surface_ids)

            centroids = {}
            for r in rows:
                if r['centroid'] is not None:
                    # pgvector returns numpy array via codec
                    centroids[r['surface_id']] = r['centroid'].tolist()
            return centroids

    def _compute_incident_embedding(
        self,
        surface_ids: Set[str],
        surface_centroids: Dict[str, List[float]]
    ) -> Optional[List[float]]:
        """Compute incident embedding as average of surface centroids."""
        embeddings = []
        expected_dim = 1536  # OpenAI text-embedding-3-small

        for sid in surface_ids:
            if sid in surface_centroids:
                centroid = surface_centroids[sid]
                # Validate dimension
                if len(centroid) == expected_dim:
                    embeddings.append(centroid)

        if not embeddings:
            return None

        # Average the embeddings
        avg = np.mean(np.array(embeddings), axis=0)
        return avg.tolist()

    def _compute_entity_idf(
        self,
        incidents: List[L3Incident],
    ) -> Dict[str, float]:
        """Compute IDF weights for entities.

        w(e) = log((N + 1) / (df(e) + 1))

        This replaces hub detection with data-driven weighting:
        - "Xi Jinping" (high df) → low weight → contributes little to overlap
        - "Wang Fuk Court" (low df) → high weight → contributes strongly

        Returns dict mapping entity name → IDF weight.
        """
        import math
        from collections import Counter

        N = len(incidents)
        entity_df = Counter()

        for inc in incidents:
            for ent in inc.anchor_entities:
                entity_df[ent] += 1

        entity_idf = {}
        for ent, df in entity_df.items():
            entity_idf[ent] = math.log((N + 1) / (df + 1))

        # Log top and bottom entities
        sorted_by_idf = sorted(entity_idf.items(), key=lambda x: x[1])
        if sorted_by_idf:
            logger.debug(f"Lowest IDF (hub-like): {sorted_by_idf[:3]}")
            logger.debug(f"Highest IDF (discriminative): {sorted_by_idf[-3:]}")

        return entity_idf

    def _compute_evidence_mass_threshold(
        self,
        incidents: List[L3Incident],
        entity_df_cap: Optional[int],
    ) -> float:
        """Compute an evidence-mass threshold from corpus size.

        With IDF defined as w(e)=log((N+1)/(df(e)+1)), a single shared entity opens the
        entity gate iff its IDF >= threshold.

        We set threshold as the IDF corresponding to df_cap (defaults to sqrt(N)):
            threshold = log((N+1)/(df_cap+1))
        This avoids hard-coding a "hub list" while preventing single-entity chaining.
        """
        import math

        N = max(1, len(incidents))
        df_cap = entity_df_cap if entity_df_cap is not None else max(3, int(math.sqrt(N)))
        df_cap = max(1, min(df_cap, N))
        return math.log((N + 1) / (df_cap + 1))

    def build_affinity_graph(
        self,
        incidents: List[L3Incident]
    ) -> Dict[str, List[IncidentAffinity]]:
        """Build affinity graph between incidents.

        Uses IDF-weighted anchor overlap instead of hub suppression.
        Hard gate requires:
        - Motif overlap > 0, OR
        - Entity evidence is strong + concentrated (mass + ratio)
        """
        from collections import defaultdict

        graph: Dict[str, List[IncidentAffinity]] = defaultdict(list)

        # Compute entity IDF weights (replaces hub detection)
        self.entity_idf = self._compute_entity_idf(incidents)
        evidence_mass_threshold = self._compute_evidence_mass_threshold(
            incidents, entity_df_cap=self.entity_df_cap
        )

        # All incidents with any signal are candidates
        candidates = []
        for inc in incidents:
            has_semantic = bool(inc.question_keys)
            has_embedding = bool(inc.embedding)
            has_structural = bool(inc.core_motifs)
            has_anchors = bool(inc.anchor_entities)
            if has_semantic or has_embedding or has_structural or has_anchors:
                candidates.append(inc)

        logger.info(f"Building affinity graph for {len(candidates)} candidate incidents")
        semantic_count = sum(1 for c in candidates if c.question_keys)
        embedding_count = sum(1 for c in candidates if c.embedding)
        logger.info(f"  {semantic_count} have question_keys, {embedding_count} have embeddings")
        logger.info(
            f"  Entity gate: evidence_mass>={evidence_mass_threshold:.2f}, "
            f"shared_mass_ratio>={self.min_shared_mass_ratio:.2f}"
        )

        for i, inc1 in enumerate(candidates):
            for inc2 in candidates[i+1:]:
                affinity = IncidentAffinity.compute(
                    inc1, inc2,
                    embedding_weight=self.embedding_weight,
                    motif_weight=self.motif_weight,
                    anchor_weight=self.anchor_weight,
                    time_weight=self.time_weight,
                    max_time_days=self.max_time_days,
                    entity_idf=self.entity_idf,  # IDF weights (replaces hub_entities)
                    embedding_threshold=self.embedding_threshold,
                    evidence_mass_threshold=evidence_mass_threshold,
                    min_shared_mass_ratio=self.min_shared_mass_ratio,
                )

                if affinity.total_affinity >= self.affinity_threshold:
                    graph[inc1.id].append(affinity)
                    # Create reverse edge
                    reverse = IncidentAffinity(
                        incident1_id=inc2.id,
                        incident2_id=inc1.id,
                        object_overlap=affinity.object_overlap,
                        predicate_overlap=affinity.predicate_overlap,
                        embedding_similarity=affinity.embedding_similarity,
                        motif_overlap=affinity.motif_overlap,
                        anchor_overlap=affinity.anchor_overlap,
                        time_proximity=affinity.time_proximity,
                        evidence_mass=affinity.evidence_mass,
                        shared_mass_ratio=affinity.shared_mass_ratio,
                        total_affinity=affinity.total_affinity,
                    )
                    graph[inc2.id].append(reverse)

        return graph

    def prune_to_mutual_knn(
        self,
        graph: Dict[str, List[IncidentAffinity]],
        k: int = 5,
    ) -> Dict[str, List[IncidentAffinity]]:
        """Prune affinity graph to mutual k-nearest neighbors.

        For each node, keep only top-k neighbors by affinity.
        Keep edge only if BOTH endpoints have each other in their top-k.

        This prevents chaining through "bridge" nodes that are weakly
        connected to many unrelated incidents.
        """
        from collections import defaultdict

        # Step 1: For each node, find its top-k neighbors
        top_k: Dict[str, Set[str]] = {}
        for node_id, edges in graph.items():
            # Sort by affinity descending, take top k
            sorted_edges = sorted(edges, key=lambda e: e.total_affinity, reverse=True)
            top_k[node_id] = {e.incident2_id for e in sorted_edges[:k]}

        # Step 2: Build mutual kNN graph - keep edge only if reciprocal
        mutual_graph: Dict[str, List[IncidentAffinity]] = defaultdict(list)
        for node_id, edges in graph.items():
            for edge in edges:
                neighbor_id = edge.incident2_id
                # Check mutual: node in neighbor's top-k AND neighbor in node's top-k
                if (neighbor_id in top_k.get(node_id, set()) and
                    node_id in top_k.get(neighbor_id, set())):
                    mutual_graph[node_id].append(edge)

        original_edges = sum(len(e) for e in graph.values()) // 2
        mutual_edges = sum(len(e) for e in mutual_graph.values()) // 2
        logger.info(f"Mutual kNN pruning (k={k}): {original_edges} → {mutual_edges} edges")

        return mutual_graph

    def find_connected_components(
        self,
        graph: Dict[str, List[IncidentAffinity]],
        use_mutual_knn: bool = True,
        knn_k: int = 5,
    ) -> List[Set[str]]:
        """Find connected components in affinity graph.

        Args:
            graph: Affinity graph from build_affinity_graph
            use_mutual_knn: If True, prune to mutual kNN first to prevent chaining
            knn_k: Number of neighbors for mutual kNN
        """
        # Optionally prune to prevent giant components from chaining
        if use_mutual_knn:
            graph = self.prune_to_mutual_knn(graph, k=knn_k)

        visited = set()
        components = []

        for start_id in graph:
            if start_id in visited:
                continue

            # BFS from this node
            component = set()
            queue = [start_id]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                component.add(node)

                # Add neighbors
                for affinity in graph.get(node, []):
                    if affinity.incident2_id not in visited:
                        queue.append(affinity.incident2_id)

            if len(component) >= 2:  # Only multi-incident cases
                components.append(component)

        return components

    def compute_case_signature(self, incident_ids: Set[str]) -> str:
        """
        Compute stable kernel signature for a case.

        Signature is deterministic hash of sorted incident IDs.
        This allows idempotent MERGE on cases.
        """
        import hashlib
        sorted_ids = sorted(incident_ids)
        content = "|".join(sorted_ids)
        return f"case_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    def build_case(self, incident_ids: Set[str]) -> L4Case:
        """Build a case from a set of incidents."""
        kernel_sig = self.compute_case_signature(incident_ids)

        # Aggregate properties from incidents
        all_anchors = set()
        all_companions = set()
        times = []

        for inc_id in incident_ids:
            inc = self.incidents.get(inc_id)
            if inc:
                all_anchors.update(inc.anchor_entities)
                all_companions.update(inc.companion_entities)
                if inc.time_start:
                    times.append(inc.time_start)

        # Core entities = anchors that appear in 2+ incidents
        anchor_counts = {}
        for inc_id in incident_ids:
            inc = self.incidents.get(inc_id)
            if inc:
                for a in inc.anchor_entities:
                    anchor_counts[a] = anchor_counts.get(a, 0) + 1

        core_entities = {a for a, c in anchor_counts.items() if c >= 2}

        return L4Case(
            id=generate_id('case'),
            incident_ids=incident_ids,
            core_entities=core_entities,
            kernel_signature=kernel_sig,
            time_start=min(times) if times else None,
            time_end=max(times) if times else None,
        )

    async def build_cases(self) -> List[L4Case]:
        """
        Main entry point: build all cases from current incidents.

        Returns list of L4Case objects (not yet persisted).
        """
        incidents = await self.load_incidents()
        logger.info(f"Loaded {len(incidents)} incidents for L4 case formation")

        graph = self.build_affinity_graph(incidents)
        logger.info(f"Built affinity graph with {len(graph)} nodes with edges")

        components = self.find_connected_components(graph)
        logger.info(f"Found {len(components)} connected components (multi-incident cases)")

        cases = []
        for component in components:
            case = self.build_case(component)
            cases.append(case)
            self.cases[case.id] = case
            logger.debug(f"  Case {case.kernel_signature}: {len(component)} incidents, {len(case.core_entities)} core entities")

        return cases

    async def persist_case(self, case: L4Case):
        """Persist a case to Neo4j idempotently using kernel_signature."""
        await self.neo4j._execute_write("""
            MERGE (c:Case {kernel_signature: $kernel_sig})
            ON CREATE SET
                c.id = $id,
                c.created_at = datetime()
            SET c.incident_ids = $incident_ids,
                c.core_entities = $core_entities,
                c.time_start = $time_start,
                c.time_end = $time_end,
                c.updated_at = datetime()
            WITH c
            UNWIND $incident_ids as inc_id
            MATCH (i:Incident {id: inc_id})
            MERGE (c)-[:CONTAINS]->(i)
        """, {
            'id': case.id,
            'kernel_sig': case.kernel_signature,
            'incident_ids': list(case.incident_ids),
            'core_entities': list(case.core_entities),
            'time_start': case.time_start.isoformat() if case.time_start else None,
            'time_end': case.time_end.isoformat() if case.time_end else None,
        })

    async def emit_case_event(self, case: L4Case, is_new: bool = True):
        """Emit case event to viz channel (shared with canonical worker).

        Literal data format - just IDs and facts, no instructions.
        """
        if not self.redis_client:
            return

        try:
            event = {
                "type": "case_formed" if is_new else "case_updated",
                "timestamp": datetime.utcnow().isoformat(),
                "case_id": case.id,
                "kernel_signature": case.kernel_signature,
                "incident_ids": list(case.incident_ids),
                "core_entities": list(case.core_entities),
            }

            # Publish to viz channel - canonical worker subscribes to same channel
            await self.redis_client.publish(self.VIZ_CHANNEL, json.dumps(event))
            logger.debug(f"Emitted case event: {case.kernel_signature}")
        except Exception as e:
            logger.warning(f"Failed to emit case event: {e}")

    async def run(self) -> List[L4Case]:
        """Build, persist, and emit all cases."""
        cases = await self.build_cases()

        new_cases = 0
        for case in cases:
            # Check if case already exists
            existing = await self.neo4j._execute_read("""
                MATCH (c:Case {kernel_signature: $sig})
                RETURN c.id as id
            """, {'sig': case.kernel_signature})

            is_new = not existing
            await self.persist_case(case)
            await self.emit_case_event(case, is_new=is_new)

            if is_new:
                new_cases += 1

        logger.info(f"✅ Persisted {len(cases)} L4 cases ({new_cases} new)")
        return cases

    # =========================================================================
    # RELATIONAL CASE FORMATION (Spine + Metabolic)
    # =========================================================================

    async def build_role_artifacts(
        self,
        incidents: List[L3Incident],
        llm_client=None,
    ) -> Dict[str, ContractRoleArtifact]:
        """
        Build per-incident role artifacts via LLM labeling.

        Uses label_incidents_roles_batch from relational_experiment for proper
        LLM-based role detection, then adapts to contracts format.

        Returns dict mapping incident_id -> ContractRoleArtifact.
        """
        if not llm_client:
            # No LLM = conservative fallback (no referents)
            logger.warning("[RELATIONAL] No LLM client - returning empty artifacts")
            return {}

        # Resolve entity metadata (stable IDs + coarse types) for better role labeling.
        all_anchor_names: Set[str] = set()
        for inc in incidents:
            all_anchor_names.update(inc.anchor_entities or set())

        entity_rows = await self.neo4j._execute_read("""
            MATCH (e:Entity)
            WHERE e.canonical_name IN $names
            RETURN e.canonical_name as name,
                   e.id as id,
                   coalesce(e.entity_type, e.type, 'UNKNOWN') as type
        """, {"names": list(all_anchor_names)})
        entity_by_name = {r["name"]: {"id": r["id"], "name": r["name"], "type": r.get("type") or "UNKNOWN"} for r in entity_rows}

        # Pull a couple of claim snippets per incident to reduce "anchor-only" ambiguity.
        incident_ids = [inc.id for inc in incidents]
        snippet_rows = await self.neo4j._execute_read("""
            UNWIND $ids as iid
            MATCH (i:Incident {id: iid})
            OPTIONAL MATCH (i)-[:CONTAINS]->(s:Surface)
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
            WITH iid, collect(DISTINCT c.text) as texts
            RETURN iid as id, texts[0..3] as snippets
        """, {"ids": incident_ids})
        snippets_by_incident = {r["id"]: [t for t in (r.get("snippets") or []) if t] for r in snippet_rows}

        # Convert L3Incident -> IncidentContext for relational_experiment
        contexts: List[IncidentContext] = []
        for inc in incidents:
            if not inc.anchor_entities:
                continue

            # Build anchors list with id/name for LLM prompt
            anchors = []
            for ent in inc.anchor_entities:
                anchors.append(entity_by_name.get(ent, {"id": ent, "name": ent, "type": "UNKNOWN"}))

            ctx = IncidentContext(
                id=inc.id,
                anchors=anchors,
                question_keys=list(inc.question_keys)[:6],
                time_start=inc.time_start,
                claim_snippets=snippets_by_incident.get(inc.id, []),
            )
            contexts.append(ctx)

        # Call LLM-based labeling
        exp_artifacts = await label_incidents_roles_batch(
            contexts, llm_client, model="gpt-4o-mini", batch_size=10
        )

        # Adapt ExperimentRoleArtifact -> ContractRoleArtifact
        contract_artifacts: Dict[str, ContractRoleArtifact] = {}
        for inc_id, exp_art in exp_artifacts.items():
            # Map role strings to EntityRole enum
            role_map: Dict[str, EntityRole] = {}
            for ent_id, role_str in exp_art.role_map.items():
                if role_str == "referent_facility":
                    role_map[ent_id] = EntityRole.REFERENT_FACILITY
                elif role_str == "referent_location":
                    role_map[ent_id] = EntityRole.REFERENT_LOCATION
                elif role_str == "referent_person":
                    role_map[ent_id] = EntityRole.REFERENT_PERSON
                elif role_str == "referent_object":
                    role_map[ent_id] = EntityRole.REFERENT_OBJECT
                elif role_str == "broad_location":
                    role_map[ent_id] = EntityRole.BROAD_LOCATION
                elif role_str == "authority":
                    role_map[ent_id] = EntityRole.AUTHORITY
                elif role_str == "responder":
                    role_map[ent_id] = EntityRole.RESPONDER
                elif role_str == "publisher":
                    role_map[ent_id] = EntityRole.PUBLISHER
                elif role_str == "commentator":
                    role_map[ent_id] = EntityRole.COMMENTARY
                else:
                    role_map[ent_id] = EntityRole.CONTEXT

            # Ensure referent ids are role-backed; if LLM returns referents without roles,
            # backfill a coarse role using the entity type as a prior (still deterministic).
            for ent_id in exp_art.referent_entity_ids:
                if ent_id in role_map:
                    continue
                # Find type for this entity_id from contexts (ids are stable)
                ent_type = None
                ctx = next((c for c in contexts if c.id == inc_id), None)
                if ctx:
                    for a in ctx.anchors:
                        if a.get("id") == ent_id:
                            ent_type = (a.get("type") or "UNKNOWN").upper()
                            break
                if ent_type == "PERSON":
                    role_map[ent_id] = EntityRole.REFERENT_PERSON
                elif ent_type == "LOCATION":
                    role_map[ent_id] = EntityRole.REFERENT_LOCATION
                else:
                    # Conservative: don't assume ORG/UNKNOWN is a referent.
                    role_map[ent_id] = EntityRole.CONTEXT

            # Compute referent set from roles (do not trust raw referent list blindly).
            referent_ids = {eid for eid, role in role_map.items() if role.is_referent}

            # Get incident time
            inc = self.incidents.get(inc_id)
            time_start = inc.time_start if inc else None

            contract_artifacts[inc_id] = ContractRoleArtifact(
                incident_id=inc_id,
                referent_entity_ids=frozenset(referent_ids),
                role_map=role_map,
                time_start=time_start,
            )

        logger.info(f"Built {len(contract_artifacts)} role artifacts from {len(incidents)} incidents via LLM")
        return contract_artifacts

    def compute_global_df(
        self,
        incidents: List[L3Incident],
        name_to_id: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, int], int]:
        """
        Compute global document frequency for all entities.

        Returns (df_map, total_incidents).
        """
        from collections import Counter

        df_map = Counter()
        for inc in incidents:
            for ent in inc.anchor_entities:
                if name_to_id and ent in name_to_id:
                    df_map[name_to_id[ent]] += 1
                else:
                    df_map[ent] += 1

        return dict(df_map), len(incidents)

    def filter_artifacts_by_df(
        self,
        artifacts: Dict[str, ContractRoleArtifact],
        global_df: Dict[str, int],
        global_total_incidents: int,
        threshold_fraction: float = 0.05,
    ) -> Dict[str, ContractRoleArtifact]:
        """
        Filter artifacts to remove broad context-like entities from referent sets.

        This implements the contract semantics:
        - Context roles may be suppressed if globally broad (DF-based).
        - Broad locations must not be treated as identity referents.
        - True referents (facility/person/object) are never suppressed.

        It also applies a deterministic correction:
        - If the LLM labels an entity as REFERENT_LOCATION but it is globally broad,
          downgrade it to BROAD_LOCATION (context) and remove it from referents.

        Args:
            artifacts: Original artifacts from LLM
            global_df: Entity document frequency map
            global_total_incidents: Total incidents in corpus
            threshold_fraction: Entities appearing in > this fraction are suppressed

        Returns:
            New artifacts with filtered referent_entity_ids
        """
        filtered: Dict[str, ContractRoleArtifact] = {}
        suppressed_entities: Dict[str, int] = {}

        # Local DF (within artifact population) for shrinkage.
        local_df: Dict[str, int] = {}
        for art in artifacts.values():
            for eid in (art.role_map or {}).keys():
                local_df[eid] = local_df.get(eid, 0) + 1

        for inc_id, art in artifacts.items():
            # Copy role_map so we can apply deterministic corrections safely.
            role_map = dict(art.role_map or {})

            filtered_referents: Set[str] = set()
            for eid in art.referent_entity_ids:
                role = role_map.get(eid, EntityRole.CONTEXT)

                df_g = int(global_df.get(eid, 0))
                df_l = int(local_df.get(eid, 0))
                df_shrunk = EntityDFEstimate.compute(
                    entity_id=eid,
                    df_global=df_g,
                    df_local=df_l,
                    alpha=0.9,
                ).df_shrunk

                # Deterministic correction: overly-broad locations cannot be referents.
                if role == EntityRole.REFERENT_LOCATION:
                    if df_shrunk > threshold_fraction * global_total_incidents:
                        role = EntityRole.BROAD_LOCATION
                        role_map[eid] = role

                if should_suppress_entity(
                    entity_id=eid,
                    role=role,
                    df_shrunk=float(df_shrunk),
                    global_total_incidents=global_total_incidents,
                    threshold_fraction=threshold_fraction,
                ):
                    suppressed_entities[eid] = suppressed_entities.get(eid, 0) + 1
                    continue

                # Keep as referent only if still a referent role.
                if role.is_referent:
                    filtered_referents.add(eid)

            # Create new artifact with filtered referents
            filtered[inc_id] = ContractRoleArtifact(
                incident_id=inc_id,
                referent_entity_ids=frozenset(filtered_referents),
                role_map=role_map,
                time_start=art.time_start,
            )

        if suppressed_entities:
            total_suppressed = sum(suppressed_entities.values())
            top_suppressed = sorted(suppressed_entities.items(), key=lambda x: -x[1])[:5]
            logger.info(
                f"DF filtering suppressed {total_suppressed} high-df referents: "
                f"{[(e, c, global_df.get(e, 0)) for e, c in top_suppressed]}"
            )

        return filtered

    def build_spine_edges(
        self,
        artifacts: Dict[str, ContractRoleArtifact],
        global_df: Dict[str, int],
        global_total_incidents: int,
        time_closeness_days: float = 30.0,
        df_threshold_fraction: float = 0.05,
    ) -> List[Tuple[str, str, SpineGateResult]]:
        """
        Build spine edges between incidents using evaluate_spine_gate.

        Applies DF-based filtering first to suppress high-frequency context
        entities that the LLM may have mislabeled as referents.

        Only creates edges for incident pairs that pass the spine gate.
        Returns list of (incident_a_id, incident_b_id, SpineGateResult).
        """
        # Step 1: Filter artifacts to remove high-DF context entities
        filtered_artifacts = self.filter_artifacts_by_df(
            artifacts, global_df, global_total_incidents,
            threshold_fraction=df_threshold_fraction,
        )

        spine_edges = []
        incident_ids = list(filtered_artifacts.keys())

        for i, id_a in enumerate(incident_ids):
            for id_b in incident_ids[i+1:]:
                art_a = filtered_artifacts[id_a]
                art_b = filtered_artifacts[id_b]

                # Skip if no referents after filtering
                if not art_a.referent_entity_ids or not art_b.referent_entity_ids:
                    continue

                # Evaluate spine gate
                result = evaluate_spine_gate(
                    art_a, art_b,
                    time_closeness_days=time_closeness_days,
                )

                if result.is_spine:
                    spine_edges.append((id_a, id_b, result))

        logger.info(f"Built {len(spine_edges)} spine edges from {len(artifacts)} incidents")
        return spine_edges

    def union_find_cases(
        self,
        incident_ids: List[str],
        spine_edges: List[Tuple[str, str, SpineGateResult]],
    ) -> List[Set[str]]:
        """
        Build cases from spine edges using union-find.

        Returns list of incident ID sets (each set = one case).
        """
        # Initialize union-find
        parent = {iid: iid for iid in incident_ids}
        rank = {iid: 0 for iid in incident_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Union all spine edges
        for id_a, id_b, _ in spine_edges:
            if id_a in parent and id_b in parent:
                union(id_a, id_b)

        # Collect components
        from collections import defaultdict
        components: Dict[str, Set[str]] = defaultdict(set)
        for iid in incident_ids:
            root = find(iid)
            components[root].add(iid)

        # Return only multi-incident cases
        return [comp for comp in components.values() if len(comp) >= 2]

    async def build_cases_relational(self, llm_client=None) -> List[L4Case]:
        """
        Build cases using spine-based relational formation.

        This is the new approach using contracts:
        1. Per-incident role labeling (LLM-based)
        2. Global DF computation
        3. Spine edge formation via evaluate_spine_gate
        4. Union-find over spine edges

        Args:
            llm_client: OpenAI-compatible client for role labeling
        """
        incidents = await self.load_incidents()
        logger.info(f"[RELATIONAL] Loaded {len(incidents)} incidents")

        # Step 1: Build role artifacts (requires LLM)
        artifacts = await self.build_role_artifacts(incidents, llm_client=llm_client)
        if not artifacts:
            logger.warning("[RELATIONAL] No artifacts - LLM may be unavailable")
            return []

        # Step 2: Compute global DF (use stable entity IDs when possible)
        all_anchor_names: Set[str] = set()
        for inc in incidents:
            all_anchor_names.update(inc.anchor_entities or set())
        entity_rows = await self.neo4j._execute_read("""
            MATCH (e:Entity)
            WHERE e.canonical_name IN $names
            RETURN e.canonical_name as name, e.id as id
        """, {"names": list(all_anchor_names)})
        name_to_id = {r["name"]: r["id"] for r in entity_rows}

        global_df, global_total = self.compute_global_df(incidents, name_to_id=name_to_id)
        logger.info(f"[RELATIONAL] Global corpus: {global_total} incidents, {len(global_df)} unique entities")

        # Step 3: Build spine edges
        spine_edges = self.build_spine_edges(
            artifacts, global_df, global_total,
            time_closeness_days=30.0,
        )

        # Step 4: Union-find to form cases
        incident_ids = list(artifacts.keys())
        components = self.union_find_cases(incident_ids, spine_edges)
        logger.info(f"[RELATIONAL] Found {len(components)} cases from spine edges")

        # Build case objects
        cases = []
        for component in components:
            case = self.build_case(component)
            cases.append(case)
            self.cases[case.id] = case

        return cases

    async def run_shadow(self, llm_client=None) -> Tuple[List[L4Case], List[L4Case]]:
        """
        Run both similarity and relational case formation for comparison.

        Args:
            llm_client: OpenAI-compatible client for role labeling (required for relational)

        Returns (similarity_cases, relational_cases).
        Use this to validate relational approach before switching.
        """
        logger.info("=== SHADOW MODE: Running both case formation strategies ===")

        # Similarity-based
        incidents = await self.load_incidents()
        graph = self.build_affinity_graph(incidents)
        sim_components = self.find_connected_components(graph)
        sim_cases = [self.build_case(comp) for comp in sim_components]
        logger.info(f"[SIMILARITY] {len(sim_cases)} cases, largest={max(len(c.incident_ids) for c in sim_cases) if sim_cases else 0}")

        # Clear state for relational
        self.incidents.clear()
        self.cases.clear()

        # Relational
        rel_cases = await self.build_cases_relational(llm_client=llm_client)
        logger.info(f"[RELATIONAL] {len(rel_cases)} cases, largest={max(len(c.incident_ids) for c in rel_cases) if rel_cases else 0}")

        # Compare
        sim_incident_sets = {frozenset(c.incident_ids) for c in sim_cases}
        rel_incident_sets = {frozenset(c.incident_ids) for c in rel_cases}

        only_sim = len(sim_incident_sets - rel_incident_sets)
        only_rel = len(rel_incident_sets - sim_incident_sets)
        both = len(sim_incident_sets & rel_incident_sets)

        logger.info(f"[SHADOW] Comparison: {both} identical, {only_sim} only-similarity, {only_rel} only-relational")

        return sim_cases, rel_cases

    async def build_cases_compiler(
        self,
        llm_client=None,
        params: CompilerParams = DEFAULT_PARAMS,
    ) -> Tuple[List[L4Case], CompilationResult]:
        """
        Build cases using the membrane compiler architecture.

        This is the strict 4-stage pipeline:
        1. load - Load incidents from Neo4j
        2. artifact_get_or_extract - Extract artifacts via LLM (referents + contexts)
        3. compile - Run all pairs through membrane (MERGE/PERIPHERY/DEFER)
        4. persist+emit - Form cases via union-find, persist to Neo4j

        The compiler is the SOLE AUTHORITY for topology decisions.
        No heuristics, no DF filtering - pure structural decisions.

        Args:
            llm_client: OpenAI-compatible client for artifact extraction
            params: CompilerParams (thresholds for membrane decisions)

        Returns:
            Tuple of (L4Case list for weaver compatibility, CompilationResult for audit)
        """
        logger.info("[COMPILER] Starting 4-stage compilation pipeline")

        # Stage 1: LOAD
        incidents = await self.load_incidents()
        logger.info(f"[COMPILER] Stage 1 (load): {len(incidents)} incidents")

        if not incidents:
            return [], CompilationResult(
                cases=[],
                all_edges=[],
                deferred=[],
                inquiries=[],
                stats={"error": "no_incidents"},
            )

        # Build incident dicts for compiler
        incident_dicts = []
        for inc in incidents:
            incident_dicts.append({
                "id": inc.id,
                "title": inc.canonical_title or "",
                "anchor_entities": list(inc.anchor_entities or []),
                "time_start": inc.time_start,
            })

        # Build entity lookup from Neo4j
        all_entity_names: Set[str] = set()
        for inc in incidents:
            all_entity_names.update(inc.anchor_entities or set())

        entity_rows = await self.neo4j._execute_read("""
            MATCH (e:Entity)
            WHERE e.canonical_name IN $names
            RETURN e.id as id, e.canonical_name as canonical_name, e.name as name
        """, {"names": list(all_entity_names)})
        entity_lookup = {r["id"]: r for r in entity_rows}

        # Stage 2: ARTIFACT EXTRACTION (via compiler)
        if llm_client is None:
            logger.warning("[COMPILER] No LLM client - cannot extract artifacts")
            return [], CompilationResult(
                cases=[],
                all_edges=[],
                deferred=[],
                inquiries=[],
                stats={"error": "no_llm_client"},
            )

        logger.info(f"[COMPILER] Stage 2 (artifact_extract): processing {len(incident_dicts)} incidents")

        # Stage 3: COMPILE (via membrane)
        # This calls compile_incidents which does:
        # - extract_artifacts_batch (Stage 2)
        # - generate_candidates_by_referent_overlap
        # - compile_pair for each candidate (membrane decisions)
        # - union-find on MERGE edges
        result = await compile_incidents(
            incidents=incident_dicts,
            entity_lookup=entity_lookup,
            llm_client=llm_client,
            params=params,
        )

        logger.info(f"[COMPILER] Stage 3 (compile): {result.stats}")

        # Stage 4: PERSIST + EMIT
        # Convert CompilerCase to L4Case for compatibility
        l4_cases = []
        for compiler_case in result.cases:
            # Find incidents in this case
            case_incidents = {iid for iid in compiler_case.incident_ids}

            # Build L4Case
            l4_case = L4Case(
                id=compiler_case.case_id,
                incident_ids=case_incidents,
                relation_backbone=[
                    (e.incident_a, e.incident_b, e.decision.edge_type.value if e.decision.edge_type else "spine")
                    for e in compiler_case.spine_edges
                ],
                core_entities=self._extract_case_entities(case_incidents),
                case_type="compiler",
                canonical_title=self._generate_case_title(case_incidents),
                kernel_signature=compiler_case.case_id,
            )
            l4_cases.append(l4_case)
            self.cases[l4_case.id] = l4_case

        logger.info(f"[COMPILER] Stage 4 (persist): {len(l4_cases)} cases formed")

        # Log DEFER edges for human review
        if result.deferred:
            logger.warning(f"[COMPILER] {len(result.deferred)} DEFER edges need attention")
            for edge in result.deferred[:5]:
                logger.warning(f"  DEFER: {edge.incident_a[:8]} <-> {edge.incident_b[:8]}: {edge.decision.reason}")

        # Log InquirySeed for human disambiguation
        if result.inquiries:
            logger.info(f"[COMPILER] {len(result.inquiries)} InquirySeed requests for human review")

        return l4_cases, result

    def _extract_case_entities(self, incident_ids: Set[str]) -> Set[str]:
        """Extract core entities from incidents in a case."""
        entities = set()
        for iid in incident_ids:
            inc = self.incidents.get(iid)
            if inc:
                entities.update(inc.anchor_entities or set())
        return entities

    def _generate_case_title(self, incident_ids: Set[str]) -> str:
        """Generate a title for a case from its incidents."""
        # Use first incident's title as basis
        for iid in sorted(incident_ids):
            inc = self.incidents.get(iid)
            if inc and inc.canonical_title:
                return inc.canonical_title
        return f"Case with {len(incident_ids)} incidents"


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
