"""
Unified Weaver Experiment v2

Implements the Universal Logic Flow:
  Perceive → Route → Evaluate Compatibility → Compile → Execute → Emit → Repair

KEY INVARIANTS (v2):
1. No silent uncertainty: if candidates exist and you don't assimilate, emit LINK or DEFER + InquirySeed
2. No unaudited membership: every spine membership change has an auditable DecisionRecord
3. No heuristic spine edges: LLM-off mode forces DEFER, never ASSIMILATE
4. Deterministic fingerprints: all sets sorted before use

This is a standalone experiment - does NOT modify REEE.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import uuid
import hashlib
import json
import numpy as np


# =============================================================================
# EDGE TYPE VOCABULARY
# =============================================================================

class EdgeType(Enum):
    """Edge types with semantic meaning."""
    # Spine edges (affect membership via union-find)
    SAME_HAPPENING = auto()  # Same event, different reports
    UPDATE_TO = auto()       # Later report supersedes/updates earlier

    # Metabolic edges (relation-only, no membership change)
    PHASE_OF = auto()        # Sequential phase of same story
    RESPONSE_TO = auto()     # Reaction/consequence
    CONTEXT_FOR = auto()     # Background/context
    CAUSES = auto()          # Causal link

SPINE_EDGES = {EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO}
METABOLIC_EDGES = {EdgeType.PHASE_OF, EdgeType.RESPONSE_TO, EdgeType.CONTEXT_FOR, EdgeType.CAUSES}


# =============================================================================
# COMPATIBILITY RESULT (Output of Evaluate step)
# =============================================================================

@dataclass
class Evidence:
    """Single piece of evidence supporting a compatibility judgment."""
    type: str                    # "entity_overlap", "semantic_sim", "time_proximity", "llm_judgment"
    value: float                 # Strength (0-1)
    detail: str                  # Human-readable explanation

@dataclass
class CompatibilityResult:
    """Typed output from compatibility evaluation."""
    target_id: str               # ID of candidate target

    # Edge type probability distribution (sums to 1 over all types + NONE)
    edge_probs: Dict[EdgeType, float] = field(default_factory=dict)
    p_none: float = 0.0          # Probability of no relation

    # Assimilation probability (should we merge?)
    p_assimilate: float = 0.0    # P(membership change)

    # Confidence in this judgment
    confidence: float = 0.0      # 0-1, based on evidence quality

    # Evidence trail
    evidence: List[Evidence] = field(default_factory=list)

    # LLM judgment gate - REQUIRED for spine actions
    has_llm_judgment: bool = False  # True only if LLM actually returned a judgment

    # LLM audit trail - hashes for reproducibility
    llm_prompt_hash: Optional[str] = None    # SHA256 of prompt sent to LLM
    llm_response_hash: Optional[str] = None  # SHA256 of response received from LLM

    @property
    def p_spine(self) -> float:
        """Total probability of spine relationship."""
        return sum(self.edge_probs.get(e, 0.0) for e in SPINE_EDGES)

    @property
    def p_metabolic(self) -> float:
        """Total probability of metabolic relationship."""
        return sum(self.edge_probs.get(e, 0.0) for e in METABOLIC_EDGES)

    @property
    def best_edge(self) -> Optional[EdgeType]:
        """Most likely edge type (None if p_none is highest)."""
        if not self.edge_probs:
            return None
        best = max(self.edge_probs.items(), key=lambda x: x[1])
        if best[1] > self.p_none:
            return best[0]
        return None


# =============================================================================
# COMPILED ACTIONS (Output of Compile step)
# =============================================================================

class ActionType(Enum):
    ASSIMILATE = auto()   # Membership change (spine edge)
    LINK = auto()         # Relation only (metabolic edge)
    DEFER = auto()        # Uncertain, emit inquiry
    REJECT = auto()       # No relation, create new structure

@dataclass
class InquirySeed:
    """Question to resolve uncertainty."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    context: Dict = field(default_factory=dict)
    priority: float = 0.5  # Higher = more urgent


# =============================================================================
# DECISION RECORD (Auditable provenance for every membership change)
# =============================================================================

def _stable_hash(obj: Any) -> str:
    """Compute stable hash for any JSON-serializable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]

@dataclass
class DecisionRecord:
    """
    Immutable audit record for every compiler decision.

    Every spine membership change MUST have one of these.
    Can be persisted as JSONL or Neo4j nodes.
    """
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Stage in the pipeline
    stage: str = ""  # "route", "evaluate", "compile", "execute", "repair_promotion"

    # Source object
    source_type: str = ""  # "incident" or "case"
    source_id: str = ""

    # Routing info
    candidates: List[Dict] = field(default_factory=list)  # [{target_id, route_score}]

    # Evaluation info
    compat_results: List[Dict] = field(default_factory=list)  # [{target_id, p_spine, p_metabolic, confidence}]

    # Compiler params
    compiler_version: str = "v2.0"
    thresholds: Dict[str, float] = field(default_factory=dict)

    # Chosen action
    action: str = ""  # ASSIMILATE, LINK, DEFER, REJECT
    target_id: Optional[str] = None
    edge_type: Optional[str] = None

    # LLM provenance (if used)
    llm_model: Optional[str] = None
    llm_prompt_hash: Optional[str] = None
    llm_response_hash: Optional[str] = None

    # Fingerprint hashes for reproducibility
    source_fingerprint_hash: Optional[str] = None
    target_fingerprint_hash: Optional[str] = None

    # Inquiry if emitted
    inquiry_id: Optional[str] = None

    # For repair promotions
    promotion_reasoning: Optional[str] = None
    shared_identifiers: List[str] = field(default_factory=list)
    anti_witnesses: List[str] = field(default_factory=list)

    def to_jsonl(self) -> str:
        """Serialize to JSONL format for persistence."""
        return json.dumps({
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "candidates": self.candidates,
            "compat_results": self.compat_results,
            "compiler_version": self.compiler_version,
            "thresholds": self.thresholds,
            "action": self.action,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "llm_model": self.llm_model,
            "llm_prompt_hash": self.llm_prompt_hash,
            "llm_response_hash": self.llm_response_hash,
            "source_fingerprint_hash": self.source_fingerprint_hash,
            "target_fingerprint_hash": self.target_fingerprint_hash,
            "inquiry_id": self.inquiry_id,
            "promotion_reasoning": self.promotion_reasoning,
            "shared_identifiers": self.shared_identifiers,
            "anti_witnesses": self.anti_witnesses,
        }, sort_keys=True)


@dataclass
class CompiledAction:
    """Deterministic output from compiler."""
    action: ActionType
    target_id: Optional[str] = None      # Target to merge/link with
    edge_type: Optional[EdgeType] = None # Edge to create
    inquiry: Optional[InquirySeed] = None  # For DEFER

    # Provenance
    source_compat: Optional[CompatibilityResult] = None
    compiler_version: str = "v1.0"


# =============================================================================
# COMPILER: CompatibilityResult -> CompiledAction
# =============================================================================

# Thresholds (tunable)
SPINE_THRESHOLD = 0.7       # P(spine) >= this → ASSIMILATE
CONFIDENCE_THRESHOLD = 0.7  # Confidence >= this to act decisively
DEFER_LOWER = 0.4           # Below this → REJECT (new structure)
METABOLIC_THRESHOLD = 0.5   # P(metabolic) >= this → LINK

# Current thresholds as dict for DecisionRecord
CURRENT_THRESHOLDS = {
    "spine_threshold": SPINE_THRESHOLD,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "defer_lower": DEFER_LOWER,
    "metabolic_threshold": METABOLIC_THRESHOLD,
}

def compile_action(
    compat: CompatibilityResult,
    source_summary: str = "",
) -> CompiledAction:
    """
    Deterministic compiler: CompatibilityResult -> CompiledAction

    Decision tree:
    1. If P(spine) >= 0.7 AND confidence >= 0.7 AND has_llm_judgment → ASSIMILATE
    2. If 0.4 <= P(spine) < 0.7 OR confidence in [0.4, 0.7) OR NOT has_llm_judgment:
       - DEFER with inquiry
       - BUT still LINK with best metabolic edge (don't isolate)
    3. If P(spine) < 0.4 AND P(metabolic) >= 0.5 → LINK only
    4. Otherwise → REJECT (create new structure)

    KEY INVARIANT (v3): Spine actions REQUIRE has_llm_judgment=True.
    This ensures spine edges are NEVER created from heuristics alone, even if
    the LLM client is configured but the call failed/returned empty.
    """
    p_spine = compat.p_spine
    p_metabolic = compat.p_metabolic
    conf = compat.confidence

    # ARCHITECTURAL INVARIANT: Spine actions require actual LLM judgment evidence
    # has_llm_judgment is True ONLY if LLM actually returned a valid judgment
    has_llm_evidence = compat.has_llm_judgment

    # INVARIANT: If no LLM judgment evidence, force all spine-eligible to DEFER
    # This prevents heuristic-only spine edges (even if LLM client is configured)
    if not has_llm_evidence and p_spine >= DEFER_LOWER:
        metabolic_edge = _best_metabolic_edge(compat) or EdgeType.CONTEXT_FOR
        inquiry = InquirySeed(
            question=f"No LLM judgment available - manual review needed for potential spine relationship",
            context={
                "source": source_summary[:100],
                "target_id": compat.target_id,
                "p_spine": p_spine,
                "p_metabolic": p_metabolic,
                "confidence": conf,
                "llm_unavailable": True,
            },
            priority=0.9,  # High priority - needs resolution
        )
        return CompiledAction(
            action=ActionType.DEFER,
            target_id=compat.target_id,
            edge_type=metabolic_edge,
            inquiry=inquiry,
            source_compat=compat,
        )

    # Zone 1: High confidence spine → ASSIMILATE (only if LLM judgment present)
    # KEY INVARIANT: Spine actions REQUIRE has_llm_judgment=True
    if p_spine >= SPINE_THRESHOLD and conf >= CONFIDENCE_THRESHOLD and has_llm_evidence:
        spine_edge = _best_spine_edge(compat) or EdgeType.SAME_HAPPENING
        return CompiledAction(
            action=ActionType.ASSIMILATE,
            target_id=compat.target_id,
            edge_type=spine_edge,
            source_compat=compat,
        )

    # Zone 2: Uncertain → DEFER + provisional LINK
    if (DEFER_LOWER <= p_spine < SPINE_THRESHOLD) or \
       (p_spine >= DEFER_LOWER and conf < CONFIDENCE_THRESHOLD):
        metabolic_edge = _best_metabolic_edge(compat) or EdgeType.CONTEXT_FOR
        inquiry = InquirySeed(
            question=f"Is this the same happening or a related but distinct event?",
            context={
                "source": source_summary,
                "target_id": compat.target_id,
                "p_spine": p_spine,
                "p_metabolic": p_metabolic,
                "confidence": conf,
            },
            priority=1.0 - conf,
        )
        return CompiledAction(
            action=ActionType.DEFER,
            target_id=compat.target_id,
            edge_type=metabolic_edge,
            inquiry=inquiry,
            source_compat=compat,
        )

    # Zone 3: Low spine but high metabolic → LINK only
    if p_spine < DEFER_LOWER and p_metabolic >= METABOLIC_THRESHOLD:
        metabolic_edge = _best_metabolic_edge(compat)
        return CompiledAction(
            action=ActionType.LINK,
            target_id=compat.target_id,
            edge_type=metabolic_edge,
            source_compat=compat,
        )

    # Zone 4: No relation → REJECT
    return CompiledAction(
        action=ActionType.REJECT,
        source_compat=compat,
    )

def _best_spine_edge(compat: CompatibilityResult) -> Optional[EdgeType]:
    """Get highest-probability spine edge."""
    spine_probs = [(e, compat.edge_probs.get(e, 0)) for e in SPINE_EDGES]
    if not spine_probs:
        return None
    best = max(spine_probs, key=lambda x: x[1])
    return best[0] if best[1] > 0 else None

def _best_metabolic_edge(compat: CompatibilityResult) -> Optional[EdgeType]:
    """Get highest-probability metabolic edge."""
    metabolic_probs = [(e, compat.edge_probs.get(e, 0)) for e in METABOLIC_EDGES]
    if not metabolic_probs:
        return None
    best = max(metabolic_probs, key=lambda x: x[1])
    return best[0] if best[1] > 0 else None

def compile_best(
    compats: List[CompatibilityResult],
    source_summary: str = "",
) -> CompiledAction:
    """
    Select best action from multiple candidates.

    KEY POLICY: If candidates exist, never REJECT outright.
    Instead, DEFER+LINK to keep the organism coherent.
    """
    if not compats:
        return CompiledAction(action=ActionType.REJECT)

    # Sort by combined score: P(spine) * confidence + P(metabolic) * 0.5
    # This gives metabolic relationships a fighting chance
    scored = [(c, c.p_spine * c.confidence + c.p_metabolic * 0.5) for c in compats]
    scored.sort(key=lambda x: -x[1])

    best_compat = scored[0][0]
    action = compile_action(best_compat, source_summary)

    # KEY POLICY: Ban REJECT when candidates exist
    # If compiler says REJECT but we have candidates, downgrade to DEFER+LINK
    if action.action == ActionType.REJECT and len(compats) > 0:
        # Find best metabolic edge type
        metabolic_edge = _best_metabolic_edge(best_compat) or EdgeType.CONTEXT_FOR
        inquiry = InquirySeed(
            question="Is this a distinct event or a phase of the existing case?",
            context={
                "source": source_summary[:100],
                "target_id": best_compat.target_id,
                "p_spine": best_compat.p_spine,
                "p_metabolic": best_compat.p_metabolic,
                "rejected_but_linked": True,
            },
            priority=0.7,  # Medium priority - we're uncertain
        )
        return CompiledAction(
            action=ActionType.DEFER,  # Defer instead of reject
            target_id=best_compat.target_id,
            edge_type=metabolic_edge,
            inquiry=inquiry,
            source_compat=best_compat,
        )

    return action


# =============================================================================
# ROUTER: Generate candidates via embeddings + cheap constraints
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class Router:
    """
    Route step: Generate candidate targets using embeddings + cheap constraints.
    NO DECISIONS here - just candidate generation.
    """

    def __init__(
        self,
        embedding_k: int = 20,  # Top-k by embedding similarity
        time_window_hours: int = 72,  # Time proximity window
        entity_boost: float = 0.3,  # Boost for entity overlap
    ):
        self.embedding_k = embedding_k
        self.time_window_hours = time_window_hours
        self.entity_boost = entity_boost

    def route_incident_to_cases(
        self,
        incident_embedding: Optional[np.ndarray],
        incident_entities: Set[str],
        incident_time: Optional[datetime],
        cases: Dict[str, "Case"],
    ) -> List[str]:
        """
        Generate candidate case IDs for an incident.

        Returns case IDs sorted by routing score (embedding sim + entity overlap).
        """
        if not cases:
            return []

        candidates = []

        for case_id, case in cases.items():
            score = 0.0

            # Embedding similarity
            if incident_embedding is not None and case.embedding is not None:
                sim = cosine_similarity(incident_embedding, case.embedding)
                score += sim

            # Entity overlap boost
            overlap = incident_entities & case.entities
            if overlap:
                score += self.entity_boost * min(len(overlap), 3)

            # Time proximity (within window)
            if incident_time and case.latest_time:
                hours_diff = abs((incident_time - case.latest_time).total_seconds() / 3600)
                if hours_diff <= self.time_window_hours:
                    score += 0.1 * (1 - hours_diff / self.time_window_hours)

            if score > 0:
                candidates.append((case_id, score))

        # Sort by score descending, take top k
        candidates.sort(key=lambda x: -x[1])
        return candidates[:self.embedding_k]  # Return (case_id, score) tuples for audit trail


# =============================================================================
# EVALUATOR: LLM-based compatibility assessment
# =============================================================================

class Evaluator:
    """
    Evaluate step: Compute typed compatibility for each candidate.
    Uses LLM for semantic judgment + heuristics for priors.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def evaluate_incident_case(
        self,
        incident_summary: str,
        incident_entities: Set[str],
        incident_embedding: Optional[np.ndarray],
        case_summary: str,
        case_entities: Set[str],
        case_embedding: Optional[np.ndarray],
        case_id: str,
    ) -> CompatibilityResult:
        """
        Evaluate compatibility between an incident and a case.

        Returns CompatibilityResult with edge probabilities.
        """
        evidence = []

        # Entity overlap evidence
        overlap = incident_entities & case_entities
        entity_score = min(len(overlap) / max(len(incident_entities), 1), 1.0)
        if overlap:
            evidence.append(Evidence(
                type="entity_overlap",
                value=entity_score,
                detail=f"Shared entities: {', '.join(list(overlap)[:3])}"
            ))

        # Semantic similarity evidence
        sem_sim = 0.0
        if incident_embedding is not None and case_embedding is not None:
            sem_sim = cosine_similarity(incident_embedding, case_embedding)
            evidence.append(Evidence(
                type="semantic_sim",
                value=sem_sim,
                detail=f"Embedding cosine similarity: {sem_sim:.3f}"
            ))

        # LLM judgment (if available)
        llm_probs = None
        if self.llm_client and incident_summary and case_summary:
            llm_probs = await self._llm_evaluate(incident_summary, case_summary)
            if llm_probs:
                evidence.append(Evidence(
                    type="llm_judgment",
                    value=llm_probs.get("confidence", 0.5),
                    detail=f"LLM: {llm_probs.get('relation', 'unknown')}"
                ))

        # Compute edge probabilities
        edge_probs = self._compute_edge_probs(entity_score, sem_sim, llm_probs)

        # Confidence based on evidence quality
        confidence = self._compute_confidence(evidence, llm_probs)

        return CompatibilityResult(
            target_id=case_id,
            edge_probs=edge_probs,
            p_none=1.0 - sum(edge_probs.values()),
            p_assimilate=edge_probs.get(EdgeType.SAME_HAPPENING, 0) + edge_probs.get(EdgeType.UPDATE_TO, 0),
            confidence=confidence,
            evidence=evidence,
            # KEY INVARIANT: has_llm_judgment is True ONLY if LLM actually returned a valid judgment
            # This gate prevents spine edges from heuristics alone
            has_llm_judgment=llm_probs is not None,
            # Audit trail: LLM call hashes for reproducibility
            llm_prompt_hash=llm_probs.get("prompt_hash") if llm_probs else None,
            llm_response_hash=llm_probs.get("response_hash") if llm_probs else None,
        )

    def _compute_edge_probs(
        self,
        entity_score: float,
        sem_sim: float,
        llm_probs: Optional[Dict],
    ) -> Dict[EdgeType, float]:
        """Compute edge type probabilities from evidence."""

        # If LLM provided explicit probabilities, use them
        if llm_probs and "edge_probs" in llm_probs:
            return llm_probs["edge_probs"]

        # Otherwise, heuristic combination
        # High entity overlap + high semantic sim → likely SAME_HAPPENING
        combined = (entity_score * 0.4 + sem_sim * 0.6)

        probs = {}
        if combined >= 0.7:
            probs[EdgeType.SAME_HAPPENING] = combined * 0.8
            probs[EdgeType.UPDATE_TO] = combined * 0.2
        elif combined >= 0.5:
            probs[EdgeType.SAME_HAPPENING] = combined * 0.5
            probs[EdgeType.PHASE_OF] = combined * 0.3
            probs[EdgeType.CONTEXT_FOR] = combined * 0.2
        elif combined >= 0.3:
            probs[EdgeType.PHASE_OF] = combined * 0.4
            probs[EdgeType.CONTEXT_FOR] = combined * 0.4
            probs[EdgeType.RESPONSE_TO] = combined * 0.2
        else:
            probs[EdgeType.CONTEXT_FOR] = combined * 0.5

        return probs

    def _compute_confidence(
        self,
        evidence: List[Evidence],
        llm_probs: Optional[Dict],
    ) -> float:
        """Compute confidence based on evidence quality."""
        if not evidence:
            return 0.3

        # LLM confidence is authoritative if available
        if llm_probs and "confidence" in llm_probs:
            return llm_probs["confidence"]

        # Otherwise, average evidence values with bonus for multiple sources
        avg = sum(e.value for e in evidence) / len(evidence)
        bonus = min(len(evidence) * 0.1, 0.2)  # Up to 0.2 bonus
        return min(avg + bonus, 1.0)

    async def _llm_evaluate(
        self,
        incident_summary: str,
        case_summary: str,
    ) -> Optional[Dict]:
        """Call LLM to evaluate relationship."""
        if not self.llm_client:
            return None

        prompt = f"""Analyze the relationship between these two news items:

ITEM A (new incident):
{incident_summary[:500]}

ITEM B (existing case):
{case_summary[:500]}

What is the relationship?
- SAME_HAPPENING: Same event reported differently
- UPDATE_TO: A is an update/follow-up to B
- PHASE_OF: A is a subsequent phase of the same story
- CONTEXT_FOR: A provides background for B
- RESPONSE_TO: A is a reaction to B
- UNRELATED: No meaningful connection

Respond in JSON format:
{{"relation": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            import json
            response_content = response.choices[0].message.content
            result = json.loads(response_content)

            # Convert relation to edge probs
            relation = result.get("relation", "UNRELATED")
            confidence = float(result.get("confidence", 0.5))

            edge_probs = {}
            if relation == "SAME_HAPPENING":
                edge_probs[EdgeType.SAME_HAPPENING] = 0.9
            elif relation == "UPDATE_TO":
                edge_probs[EdgeType.UPDATE_TO] = 0.9
            elif relation == "PHASE_OF":
                edge_probs[EdgeType.PHASE_OF] = 0.9
            elif relation == "CONTEXT_FOR":
                edge_probs[EdgeType.CONTEXT_FOR] = 0.9
            elif relation == "RESPONSE_TO":
                edge_probs[EdgeType.RESPONSE_TO] = 0.9

            return {
                "relation": relation,
                "confidence": confidence,
                "edge_probs": edge_probs,
                # Audit trail: hashes for reproducibility
                "prompt_hash": _stable_hash(prompt),
                "response_hash": _stable_hash(response_content),
            }
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CaseFingerprint:
    """
    Rich identity evidence for a case, used in LLM compatibility checks.

    This is NOT an entity list - it's structured semantic evidence:
    - what_happened: Free-text short description of what occurred (1 clause)
    - distinguishers: Key facts that identify THIS happening vs others
    - Top snippets (actual text spans that identify the happening)
    - Time window (when did this happen)
    - Loci phrases (specific location mentions, not just entity names)

    NO keyword-based type classification - let LLM compare free-text cards.
    """
    what_happened: str = ""  # Free-text: "High-rise fire killed 17 in Wang Fuk Court"
    distinguishers: List[str] = field(default_factory=list)  # ["Wang Fuk Court Block 7", "17 dead", "Nov 2024"]
    snippets: List[str] = field(default_factory=list)  # Top 3 identifying snippets
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    loci: List[str] = field(default_factory=list)  # Specific place phrases from text
    incident_count: int = 0

    def to_prompt_block(self) -> str:
        """Format fingerprint for LLM prompt - no taxonomies, just evidence."""
        lines = []

        # What happened (MOST IMPORTANT - free text description)
        if self.what_happened:
            lines.append(f"WHAT HAPPENED: {self.what_happened}")

        # Distinguishing facts (key identity markers)
        if self.distinguishers:
            lines.append("DISTINGUISHING FACTS:")
            for d in self.distinguishers[:5]:
                lines.append(f"  - {d}")

        # Time window
        if self.time_start and self.time_end:
            if self.time_start.date() == self.time_end.date():
                lines.append(f"TIME: {self.time_start.strftime('%Y-%m-%d')}")
            else:
                lines.append(f"TIME: {self.time_start.strftime('%Y-%m-%d')} to {self.time_end.strftime('%Y-%m-%d')}")
        elif self.time_start:
            lines.append(f"TIME: {self.time_start.strftime('%Y-%m-%d')}")

        # Location phrases (specific, not broad)
        if self.loci:
            lines.append(f"LOCATIONS: {', '.join(self.loci[:3])}")

        # Snippets (supporting evidence)
        if self.snippets:
            lines.append("EVIDENCE SNIPPETS:")
            for i, s in enumerate(self.snippets[:2], 1):
                lines.append(f"  {i}. \"{s[:150]}\"")

        lines.append(f"SOURCES: {self.incident_count} reports")

        return "\n".join(lines)


@dataclass
class Incident:
    """An incident (happening) in the system."""
    id: str
    summary: str = ""
    entities: Set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    time: Optional[datetime] = None
    surface_ids: Set[str] = field(default_factory=set)
    # Rich content for fingerprinting
    snippets: List[str] = field(default_factory=list)  # Key text spans
    facets: Set[str] = field(default_factory=set)  # Mentioned aspects

@dataclass
class Case:
    """A case (story/organism) grouping related incidents."""
    id: str
    incident_ids: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    latest_time: Optional[datetime] = None
    earliest_time: Optional[datetime] = None
    title: str = ""
    summaries: List[str] = field(default_factory=list)  # Track incident summaries
    # Rich content for fingerprinting
    all_snippets: List[str] = field(default_factory=list)
    all_facets: Set[str] = field(default_factory=set)

    def update_from_incidents(self, incidents: Dict[str, Incident]):
        """Update case embedding, entities, and summary from member incidents.

        DETERMINISM INVARIANT: Always iterate incident_ids in sorted order
        so that summaries, snippets, embeddings are collected consistently.
        """
        self.entities = set()
        embeddings = []
        times = []
        self.summaries = []
        self.all_snippets = []
        self.all_facets = set()

        # CRITICAL: Sort incident_ids for deterministic iteration
        for iid in sorted(self.incident_ids):
            if iid in incidents:
                inc = incidents[iid]
                self.entities.update(inc.entities)
                if inc.embedding is not None:
                    embeddings.append(inc.embedding)
                if inc.time:
                    times.append(inc.time)
                if inc.summary:
                    self.summaries.append(inc.summary)
                # Collect snippets and facets
                self.all_snippets.extend(inc.snippets)
                self.all_facets.update(inc.facets)

        # Centroid embedding
        if embeddings:
            self.embedding = np.mean(embeddings, axis=0)

        # Time range
        if times:
            self.latest_time = max(times)
            self.earliest_time = min(times)

        # Generate title from first summary or entities
        if self.summaries:
            self.title = self.summaries[0][:200]  # Use first incident summary
        elif self.entities:
            self.title = f"Case about {', '.join(sorted(list(self.entities))[:3])}"

    def get_summary_for_comparison(self) -> str:
        """Get a summary suitable for LLM comparison (legacy method)."""
        if self.summaries:
            # Use first 2 summaries to represent the case
            return " | ".join(self.summaries[:2])[:500]
        elif self.entities:
            return f"Case about: {', '.join(sorted(list(self.entities))[:5])}"
        return ""

    def build_fingerprint(self) -> CaseFingerprint:
        """
        Build a rich fingerprint for LLM compatibility checks.

        DETERMINISM INVARIANT: All lists derived from sets must be sorted.

        LOCI FIX (D2): Don't use raw entity list for loci.
        Instead, extract specific location phrases from snippets,
        or leave empty and rely on snippets + time for identity.
        """
        # Select top snippets - prefer diverse ones
        # Snippets are already in deterministic order from sorted incident iteration
        snippets = []
        if self.all_snippets:
            snippets = self.all_snippets[:3]
        elif self.summaries:
            snippets = self.summaries[:3]

        # Extract "what happened" from first summary (free-text)
        what_happened = ""
        if self.summaries:
            what_happened = self.summaries[0][:200]
        elif self.title:
            what_happened = self.title

        # DETERMINISM: Sort entities before taking top N
        sorted_entities = sorted(self.entities)

        # D2 FIX: Don't use entity list as loci - it smuggles broad places
        # Instead, extract locus phrases from snippets or leave empty
        loci = self._extract_loci_from_snippets(snippets)

        # Distinguishers: sorted entities + time + count
        distinguishers = sorted_entities[:3]

        # Add time window
        if self.earliest_time:
            time_str = self.earliest_time.strftime('%Y-%m-%d')
            if self.latest_time and self.latest_time.date() != self.earliest_time.date():
                time_str = f"{self.earliest_time.strftime('%Y-%m-%d')} to {self.latest_time.strftime('%Y-%m-%d')}"
            distinguishers.append(f"Time: {time_str}")

        # Add incident count as evidence mass
        distinguishers.append(f"{len(self.incident_ids)} sources")

        return CaseFingerprint(
            what_happened=what_happened,
            distinguishers=distinguishers[:5],
            snippets=snippets,
            time_start=self.earliest_time,
            time_end=self.latest_time,
            loci=loci,
            incident_count=len(self.incident_ids),
        )

    def _extract_loci_from_snippets(self, snippets: List[str]) -> List[str]:
        """
        NO-OP: Loci extraction is deferred to the LLM.

        ARCHITECTURAL INVARIANT: No keyword-based heuristics for semantic extraction.
        The LLM typed compatibility check receives the full snippets and can derive
        location context semantically. Using keyword heuristics here would bias
        the witness list and reintroduce the Tai Po/HK bridging problem.

        Returns empty list - the fingerprint includes full snippets for LLM analysis.
        """
        # REMOVED: Keyword heuristics that could introduce bridging bias
        # The LLM sees snippets in the fingerprint and derives location semantically
        return []


# =============================================================================
# UNIFIED WEAVER
# =============================================================================

class UnifiedWeaver:
    """
    Main weaver implementing Universal Logic Flow v2.

    Flow: Perceive → Route → Evaluate → Compile → Execute → Emit

    KEY INVARIANTS (v2):
    1. Every time we start a new case while candidates existed,
       we MUST emit at least one metabolic link to the best candidate.
    2. Every spine membership change MUST emit a DecisionRecord.
    3. LLM unavailable → no ASSIMILATE, only DEFER with high-priority inquiry.
    """

    def __init__(self, llm_client=None, decision_log_path: Optional[str] = None, verbose: bool = True):
        self.incidents: Dict[str, Incident] = {}
        self.cases: Dict[str, Case] = {}
        self.metabolic_edges: List[Tuple[str, str, EdgeType]] = []
        self.deferred: List[Tuple[str, CompiledAction]] = []
        self.inquiries: List[InquirySeed] = []

        self.router = Router()
        self.evaluator = Evaluator(llm_client)

        # Union-find for case membership
        self._parent: Dict[str, str] = {}

        # Track case growth for re-evaluation
        self._case_growth: Dict[str, int] = {}  # case_id -> last known size

        # Decision audit log
        self.decision_records: List[DecisionRecord] = []
        self.decision_log_path = decision_log_path  # Optional JSONL file
        self.verbose = verbose  # Real-time audit logging

        # Stats (single source of truth - updated ONLY in _update_stats)
        self.stats = {
            "assimilate": 0,
            "link": 0,
            "defer": 0,
            "reject": 0,
            "metabolic_edges": 0,
            "spine_edges": 0,
        }

    def _find(self, x: str) -> str:
        """Union-find: find root."""
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self._find(self._parent[x])
        return self._parent[x]

    def _union(self, x: str, y: str):
        """Union-find: merge sets."""
        px, py = self._find(x), self._find(y)
        if px != py:
            self._parent[px] = py

    async def process_incident(
        self,
        incident_id: str,
        summary: str,
        entities: Set[str],
        embedding: Optional[np.ndarray] = None,
        time: Optional[datetime] = None,
        snippets: Optional[List[str]] = None,
        facets: Optional[Set[str]] = None,
    ) -> CompiledAction:
        """
        Process a new incident through the full pipeline.

        Returns the compiled action taken.

        KEY INVARIANTS (v2):
        - If we create a new case while candidates existed, emit metabolic link
        - Emit DecisionRecord for every spine membership change
        - LLM unavailable → force DEFER, never ASSIMILATE
        """
        # 1. PERCEIVE: Create incident artifact
        incident = Incident(
            id=incident_id,
            summary=summary,
            entities=entities,
            embedding=embedding,
            time=time,
            snippets=snippets or [],
            facets=facets or set(),
        )
        self.incidents[incident_id] = incident

        # 2. ROUTE: Generate candidate cases (returns (case_id, route_score) tuples)
        candidates_with_scores = self.router.route_incident_to_cases(
            incident_embedding=embedding,
            incident_entities=entities,
            incident_time=time,
            cases=self.cases,
        )
        # Build route score lookup for DecisionRecord (Fix 4)
        route_scores = {cid: score for cid, score in candidates_with_scores}
        candidates = [cid for cid, _ in candidates_with_scores]

        # 3. EVALUATE: Compute compatibility for each candidate
        compats = []
        for case_id in candidates:
            case = self.cases[case_id]
            # Use case's summary method for better LLM comparison
            case_summary = case.get_summary_for_comparison() if hasattr(case, 'get_summary_for_comparison') else case.title
            compat = await self.evaluator.evaluate_incident_case(
                incident_summary=summary,
                incident_entities=entities,
                incident_embedding=embedding,
                case_summary=case_summary,
                case_entities=case.entities,
                case_embedding=case.embedding,
                case_id=case_id,
            )
            compats.append(compat)

        # 4. COMPILE: Deterministic action selection
        # Spine decisions are gated by compat.has_llm_judgment inside compile_action
        action = compile_best(compats, summary)

        # 5. EXECUTE: Apply action
        self._execute(incident_id, action, compats)

        # 6. EMIT: Create DecisionRecord for audit trail
        record = self._create_decision_record(
            incident_id=incident_id,
            candidates=candidates,
            compats=compats,
            action=action,
            summary=summary,
            route_scores=route_scores,  # Fix 4: Include route scores
        )
        self._emit_decision_record(record, verbose=self.verbose)

        # 7. Update stats (single source of truth)
        self._update_stats(action)

        if action.inquiry:
            self.inquiries.append(action.inquiry)

        return action

    def _create_decision_record(
        self,
        incident_id: str,
        candidates: List[str],
        compats: List[CompatibilityResult],
        action: CompiledAction,
        summary: str,
        route_scores: Optional[Dict[str, float]] = None,  # Fix 4: Accept route scores
    ) -> DecisionRecord:
        """Create an auditable decision record."""
        route_scores = route_scores or {}

        # Build candidate info with route scores (Fix 4)
        candidate_info = [
            {"target_id": cid, "route_score": route_scores.get(cid, 0.0)}
            for cid in candidates
        ]

        # Build compat results summary with has_llm_judgment flag and LLM hashes (Fix 3)
        compat_info = [
            {
                "target_id": c.target_id,
                "p_spine": c.p_spine,
                "p_metabolic": c.p_metabolic,
                "confidence": c.confidence,
                "has_llm_judgment": c.has_llm_judgment,  # Fix 3: Include LLM judgment flag
                "llm_prompt_hash": c.llm_prompt_hash,    # Audit: LLM prompt hash
                "llm_response_hash": c.llm_response_hash,  # Audit: LLM response hash
            }
            for c in compats
        ]

        return DecisionRecord(
            stage="compile",
            source_type="incident",
            source_id=incident_id,
            candidates=candidate_info,
            compat_results=compat_info,
            thresholds=CURRENT_THRESHOLDS.copy(),
            action=action.action.name,
            target_id=action.target_id,
            edge_type=action.edge_type.name if action.edge_type else None,
            llm_model="gpt-4o-mini" if self.evaluator.llm_client else None,
            inquiry_id=action.inquiry.id if action.inquiry else None,
            # Fix 3: Add source fingerprint hash for reproducibility
            source_fingerprint_hash=_stable_hash(summary) if summary else None,
        )

    def _emit_decision_record(self, record: DecisionRecord, verbose: bool = True):
        """Emit decision record to log and optional file.

        Args:
            record: The DecisionRecord to emit
            verbose: If True, print human-readable audit line to stdout
        """
        self.decision_records.append(record)

        # Print real-time audit log for debugging
        if verbose:
            n = len(self.decision_records)
            target_str = f" -> {record.target_id[:8]}" if record.target_id else ""
            edge_str = f" ({record.edge_type})" if record.edge_type else ""
            candidates_str = f" [{len(record.candidates)} candidates]" if record.candidates else " [no candidates]"
            print(f"[{n:04d}] {record.action:10}{target_str}{edge_str}{candidates_str} | {record.source_id[:8]}")

        # Write to JSONL file if configured
        if self.decision_log_path:
            try:
                with open(self.decision_log_path, 'a') as f:
                    f.write(record.to_jsonl() + '\n')
            except Exception as e:
                print(f"Warning: Failed to write decision record: {e}")

    def _update_stats(self, action: CompiledAction):
        """Update stats from action (single source of truth)."""
        action_name = action.action.name.lower()
        self.stats[action_name] = self.stats.get(action_name, 0) + 1

        # Track edge types
        if action.action == ActionType.ASSIMILATE:
            self.stats["spine_edges"] += 1
        elif action.action in (ActionType.LINK, ActionType.DEFER):
            self.stats["metabolic_edges"] += 1

    def _execute(
        self,
        incident_id: str,
        action: CompiledAction,
        compats: List[CompatibilityResult] = None,
    ):
        """
        Execute compiled action.

        KEY INVARIANT: When creating a new case while candidates existed,
        we MUST emit at least one metabolic link to the best candidate.
        This creates cross-component tissue for self-healing.
        """
        compats = compats or []

        if action.action == ActionType.ASSIMILATE:
            # Add incident to existing case (spine edge)
            case = self.cases[action.target_id]
            case.incident_ids.add(incident_id)
            self._union(incident_id, action.target_id)
            case.update_from_incidents(self.incidents)
            # Track growth for re-evaluation triggers
            self._case_growth[action.target_id] = len(case.incident_ids)

        elif action.action == ActionType.LINK:
            # Create metabolic edge (relation only, no membership change)
            self.metabolic_edges.append((incident_id, action.target_id, action.edge_type))
            # Still need to create a case for this incident
            new_case_id = self._create_case_for_incident(incident_id)
            # NOTE: Stats updated by _update_stats(), not here

        elif action.action == ActionType.DEFER:
            # Create metabolic edge + track for repair
            self.metabolic_edges.append((incident_id, action.target_id, action.edge_type))
            self.deferred.append((incident_id, action))
            # Create case but mark as provisional
            self._create_case_for_incident(incident_id)
            # NOTE: Stats updated by _update_stats(), not here

        elif action.action == ActionType.REJECT:
            # Create new case
            new_case_id = self._create_case_for_incident(incident_id)

            # KEY INVARIANT: If candidates existed, we MUST emit metabolic links
            # This ensures cross-component tissue for repair
            if compats:
                # Link to best candidate (by combined score)
                best_compat = max(compats, key=lambda c: c.p_spine * c.confidence + c.p_metabolic * 0.5)
                edge_type = _best_metabolic_edge(best_compat) or EdgeType.CONTEXT_FOR
                self.metabolic_edges.append((incident_id, best_compat.target_id, edge_type))

                # For high-scoring candidates (top 3 by embedding), also emit LINK
                # This creates richer tissue for repair
                for compat in compats[:3]:
                    if compat.target_id != best_compat.target_id:
                        edge = _best_metabolic_edge(compat) or EdgeType.CONTEXT_FOR
                        self.metabolic_edges.append((incident_id, compat.target_id, edge))
                # NOTE: Stats updated by _update_stats(), not here
                # The metabolic_edges list is used for soft-reject tissue creation

    def _create_case_for_incident(self, incident_id: str) -> str:
        """Create a new case containing just this incident."""
        case_id = f"case_{incident_id[:8]}"
        case = Case(
            id=case_id,
            incident_ids={incident_id},
        )
        case.update_from_incidents(self.incidents)
        self.cases[case_id] = case
        self._parent[incident_id] = case_id
        return case_id

    def get_case_membership(self) -> Dict[str, Set[str]]:
        """Get current case membership."""
        membership = {}
        for case_id, case in self.cases.items():
            membership[case_id] = case.incident_ids.copy()
        return membership

    async def stitch_via_metabolic_edges(
        self,
        max_rounds: int = 5,
        min_confidence: float = 0.7,
    ) -> Dict:
        """
        REPAIR PHASE: Multi-round promotion of metabolic edges to spine edges.

        KEY INVARIANT: Promotions are decided by LLM typed compatibility, NOT entity overlap.
        Entity overlap only routes candidates; the actual spine decision comes from
        semantic fingerprint comparison.

        Flow:
        1. Metabolic edges propose candidate case pairs
        2. LLM evaluates typed compatibility on case fingerprints
        3. Only SAME_HAPPENING/UPDATE_TO at high confidence → spine promotion
        4. Iterate until convergence or max_rounds

        Returns dict with merge stats and promotions.
        """
        total_result = {
            "merges": 0,
            "promotions": [],
            "skipped_llm_reject": 0,
            "skipped_low_confidence": 0,
            "rounds": 0,
        }

        for round_num in range(max_rounds):
            round_result = await self._stitch_one_round_llm(min_confidence=min_confidence)

            total_result["merges"] += round_result["merges"]
            total_result["promotions"].extend(round_result["promotions"])
            total_result["skipped_llm_reject"] += round_result["skipped_llm_reject"]
            total_result["skipped_low_confidence"] += round_result["skipped_low_confidence"]
            total_result["rounds"] = round_num + 1

            # Stop if no new merges (convergence)
            if round_result["merges"] == 0:
                break

        return total_result

    async def _stitch_one_round_llm(self, min_confidence: float = 0.7) -> Dict:
        """
        Single round of LLM-gated metabolic edge stitching.

        NO hard-coded keyword gates. LLM compares free-text evidence cards
        (what_happened, distinguishers) and decides spine promotion directly.
        """
        result = {
            "merges": 0,
            "promotions": [],
            "skipped_llm_reject": 0,
            "skipped_low_confidence": 0,
        }

        if not self.evaluator.llm_client:
            # Without LLM, no promotions (fail-safe)
            return result

        # Build incident -> case mapping
        incident_to_case = {}
        for case_id, case in self.cases.items():
            for iid in case.incident_ids:
                incident_to_case[iid] = case_id

        # Collect unique case pairs from metabolic edges
        processed_pairs = set()
        candidate_pairs = []

        for incident_id, target_id, edge_type in self.metabolic_edges:
            source_case_id = incident_to_case.get(incident_id)
            target_case_id = incident_to_case.get(target_id, target_id)

            if not source_case_id or not target_case_id:
                continue
            if source_case_id == target_case_id:
                continue

            pair = tuple(sorted([source_case_id, target_case_id]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            source_case = self.cases.get(source_case_id)
            target_case = self.cases.get(target_case_id)

            if source_case and target_case:
                candidate_pairs.append((source_case_id, target_case_id, source_case, target_case))

        # Evaluate each candidate pair via LLM typed compatibility
        for source_case_id, target_case_id, source_case, target_case in candidate_pairs:
            # Build fingerprints for rich identity evidence
            source_fingerprint = source_case.build_fingerprint()
            target_fingerprint = target_case.build_fingerprint()

            # Also get summaries as fallback
            source_summary = source_case.get_summary_for_comparison()
            target_summary = target_case.get_summary_for_comparison()

            if not source_summary and not source_fingerprint.snippets:
                continue
            if not target_summary and not target_fingerprint.snippets:
                continue

            # LLM typed compatibility check with fingerprints
            llm_result = await self._llm_typed_compatibility(
                source_summary,
                target_summary,
                fingerprint_a=source_fingerprint,
                fingerprint_b=target_fingerprint,
            )

            if not llm_result:
                result["skipped_llm_reject"] += 1
                continue

            relation = llm_result.get("relation", "UNRELATED")
            confidence = llm_result.get("confidence", 0.0)

            # Only spine-eligible relations at high confidence trigger promotion
            if relation not in ("SAME_HAPPENING", "UPDATE_TO"):
                result["skipped_llm_reject"] += 1
                continue

            if confidence < min_confidence:
                result["skipped_low_confidence"] += 1
                continue

            # PROMOTION: LLM confirms spine relationship
            promotion = {
                "source_case": source_case_id,
                "target_case": target_case_id,
                "relation": relation,
                "confidence": confidence,
                "reasoning": llm_result.get("reasoning", ""),
            }
            result["promotions"].append(promotion)

            # Merge smaller into larger
            if len(source_case.incident_ids) > len(target_case.incident_ids):
                source_case, target_case = target_case, source_case
                source_case_id, target_case_id = target_case_id, source_case_id

            # AUDIT: Emit DecisionRecord for repair promotion (Fix 2)
            # Every spine membership change MUST have an auditable record
            # Use _emit_decision_record to ensure consistent output (JSONL, verbose logging)
            decision_record = DecisionRecord(
                stage="repair_promotion",
                source_type="case",
                source_id=source_case_id,
                target_id=target_case_id,
                action="ASSIMILATE",
                edge_type=relation,
                thresholds=CURRENT_THRESHOLDS,
                compat_results=[{
                    "target_id": target_case_id,
                    "p_spine": 1.0 if relation in ("SAME_HAPPENING", "UPDATE_TO") else 0.0,
                    "p_metabolic": 0.0 if relation in ("SAME_HAPPENING", "UPDATE_TO") else 1.0,
                    "confidence": confidence,
                    "has_llm_judgment": True,  # Repair promotions always have LLM judgment
                    "llm_prompt_hash": llm_result.get("prompt_hash"),    # LLM audit trail
                    "llm_response_hash": llm_result.get("response_hash"),  # LLM audit trail
                }],
                llm_model="gpt-4o-mini",
                promotion_reasoning=llm_result.get("reasoning", ""),
                shared_identifiers=llm_result.get("shared_identifiers", []),
                anti_witnesses=llm_result.get("anti_witnesses", []),
                # Hash fields for reproducibility (Fix 3)
                source_fingerprint_hash=_stable_hash(source_fingerprint.to_dict()) if source_fingerprint else None,
                target_fingerprint_hash=_stable_hash(target_fingerprint.to_dict()) if target_fingerprint else None,
            )
            # Route through _emit_decision_record for consistent audit output (verbose + JSONL)
            self._emit_decision_record(decision_record)

            # Merge source into target
            target_case.incident_ids.update(source_case.incident_ids)
            target_case.update_from_incidents(self.incidents)

            # Update mappings
            for iid in source_case.incident_ids:
                incident_to_case[iid] = target_case_id
                self._parent[iid] = target_case_id

            # Remove source case
            if source_case_id in self.cases:
                del self.cases[source_case_id]
            result["merges"] += 1

        return result

    async def _llm_typed_compatibility(
        self,
        case_a_summary: str,
        case_b_summary: str,
        fingerprint_a: Optional[CaseFingerprint] = None,
        fingerprint_b: Optional[CaseFingerprint] = None,
    ) -> Optional[Dict]:
        """
        LLM typed compatibility check for spine promotion with STRUCTURED IDENTITY WITNESSES.

        The LLM must return:
        - event_signature_a/b: {what_happened, locus_phrase, time_hint}
        - shared_identifiers: specific matching facts (NOT broad locations)
        - anti_witnesses: reasons they might be different

        COMPILER RULE: Spine promotion requires 2+ matching identifiers among:
        {specific locus phrase, time overlap, distinctive numeric fact}

        Returns relation type + confidence + witnesses, or None on failure.
        """
        if not self.evaluator.llm_client:
            return None

        # Build evidence blocks - prefer fingerprints over thin summaries
        if fingerprint_a and (fingerprint_a.what_happened or fingerprint_a.snippets):
            evidence_a = fingerprint_a.to_prompt_block()
        else:
            evidence_a = f"WHAT HAPPENED: {case_a_summary[:400]}"

        if fingerprint_b and (fingerprint_b.what_happened or fingerprint_b.snippets):
            evidence_b = fingerprint_b.to_prompt_block()
        else:
            evidence_b = f"WHAT HAPPENED: {case_b_summary[:400]}"

        prompt = f"""Compare these two news case cards. Are they about the SAME physical happening?

=== CASE A ===
{evidence_a}

=== CASE B ===
{evidence_b}

DECISION RULES (in priority order):

1. WHAT HAPPENED IS PRIMARY: Focus on what physically occurred, not where.
   - "Fire at Building X" vs "Fire at Building Y" = DIFFERENT happenings (UNRELATED)
   - "Fire at Building X" vs "Death toll update for Building X fire" = SAME happening (UPDATE_TO)

2. SHARED LOCATION IS NOT ENOUGH:
   - Two events in the same city/district/area are NOT the same happening
   - Broad locations like "Hong Kong", "Tai Po District", "China" do NOT count as identity witnesses
   - Only SPECIFIC places (building names, exact addresses) count

3. DISTINGUISHING FACTS MUST ALIGN:
   - Compare specific identifiers: building names, death tolls, dates, key actors
   - If distinguishing facts conflict, they are UNRELATED

Possible relations:
- SAME_HAPPENING: Same physical event reported by different sources
- UPDATE_TO: Direct follow-up with new facts about the same event
- PHASE_OF: Sequential phases of evolving story (causally connected but distinct events)
- CONTEXT_FOR: One provides background for the other
- UNRELATED: Different happenings, even if they share location or topic

You MUST provide structured identity witnesses. Respond in JSON:
{{
  "relation": "SAME_HAPPENING|UPDATE_TO|PHASE_OF|CONTEXT_FOR|UNRELATED",
  "confidence": 0.0-1.0,
  "event_signature_a": {{
    "what_happened": "short clause describing the physical event",
    "locus_phrase": "specific place name (NOT broad like 'Hong Kong')",
    "time_hint": "when it happened if known"
  }},
  "event_signature_b": {{
    "what_happened": "short clause describing the physical event",
    "locus_phrase": "specific place name (NOT broad like 'Hong Kong')",
    "time_hint": "when it happened if known"
  }},
  "shared_identifiers": ["list of SPECIFIC matching facts - NOT broad locations like Hong Kong/Tai Po"],
  "anti_witnesses": ["reasons they might be different events"],
  "reasoning": "brief explanation"
}}"""

        try:
            response = await self.evaluator.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            import json
            result = json.loads(response.choices[0].message.content)

            # COMPILER RULE: Enforce 2+ shared identifiers for spine promotion
            # The LLM is responsible for listing ONLY specific identifiers (not broad locations)
            # We just enforce the count requirement here - no Python filtering of what "counts"
            relation = result.get("relation", "UNRELATED")
            if relation in ("SAME_HAPPENING", "UPDATE_TO"):
                shared_ids = result.get("shared_identifiers", [])
                anti_witnesses = result.get("anti_witnesses", [])

                # Require at least 2 shared identifiers for spine promotion
                if len(shared_ids) < 2:
                    # Downgrade based on anti-witnesses
                    if anti_witnesses:
                        result["relation"] = "UNRELATED"
                        result["downgrade_reason"] = f"Insufficient identifiers ({len(shared_ids)}<2), has anti-witnesses"
                    else:
                        result["relation"] = "CONTEXT_FOR"
                        result["downgrade_reason"] = f"Insufficient identifiers ({len(shared_ids)}<2), no anti-witnesses"

            return result

        except Exception as e:
            print(f"LLM typed compatibility error: {e}")
            return None

    def re_evaluate_singletons_against_grown_cases(self, growth_threshold: int = 3) -> Dict:
        """
        Growth-triggered candidate surfacing: When cases grow significantly,
        identify singleton cases that might benefit from re-evaluation.

        ARCHITECTURAL INVARIANT: This method is now a NO-OP for membership changes.

        Previous behavior (REMOVED):
        - Used keyword heuristics to identify "location anchors"
        - Directly merged singletons into grown cases
        - Modified union-find without LLM verification

        This was a "backdoor compiler" that violated the typed-witness architecture.
        All membership changes MUST go through the LLM-gated pipeline:
        _route() -> _evaluate() -> _compile() -> _execute()

        Current behavior:
        - Returns candidate pairs for potential re-evaluation
        - Actual merging requires re-processing through process_incident() with LLM
        - No heuristic-based membership changes

        Returns dict with candidate statistics (no membership changes made).
        """
        result = {
            "re_evaluated": 0,  # Always 0 - no direct re-evaluation
            "promoted": 0,      # Always 0 - no direct promotion
            "still_separate": 0,
            "candidates_surfaced": 0,  # New: pairs identified for potential LLM review
            "grown_cases": [],
            "singleton_candidates": [],
        }

        # Find cases that have grown significantly
        grown_cases = []
        for case_id, case in self.cases.items():
            old_size = self._case_growth.get(case_id, 1)
            new_size = len(case.incident_ids)
            if new_size - old_size >= growth_threshold:
                grown_cases.append(case_id)
                self._case_growth[case_id] = new_size

        if not grown_cases:
            return result

        result["grown_cases"] = grown_cases

        # Find singleton cases (size 1-2) that share ANY entities with grown cases
        # NOTE: We do NOT use keyword heuristics - just identify shared entities
        singleton_cases = [
            case_id for case_id, case in self.cases.items()
            if len(case.incident_ids) <= 2 and case_id not in grown_cases
        ]

        for singleton_id in singleton_cases:
            singleton = self.cases.get(singleton_id)
            if not singleton:
                continue

            for grown_id in grown_cases:
                grown = self.cases.get(grown_id)
                if not grown:
                    continue

                # Check for ANY shared entities (no keyword filtering)
                shared = singleton.entities & grown.entities

                if shared:
                    # Surface as candidate for potential LLM review
                    # NOTE: We do NOT merge here - that requires LLM verification
                    result["candidates_surfaced"] += 1
                    result["singleton_candidates"].append({
                        "singleton_id": singleton_id,
                        "grown_id": grown_id,
                        "shared_entities": list(shared),
                    })
                    break  # Only surface once per singleton
            else:
                result["still_separate"] += 1

        # IMPORTANT: No membership changes were made
        # Callers should use the candidates_surfaced data to queue
        # incidents for re-processing through the LLM-gated pipeline
        return result

    def summary(self) -> Dict:
        """Get summary stats."""
        return {
            "incidents": len(self.incidents),
            "cases": len(self.cases),
            "metabolic_edges": len(self.metabolic_edges),
            "deferred": len(self.deferred),
            "inquiries": len(self.inquiries),
            "actions": self.stats.copy(),
        }
