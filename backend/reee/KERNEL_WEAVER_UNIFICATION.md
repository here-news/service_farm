# Kernel-Weaver Unification Plan

## Design Decisions Locked

These decisions are final and should not be revisited during implementation:

1. **Kernel is pure** - No DB/Redis/LLM imports. LLM is an EvidenceProvider (outside kernel) + optional Explainer (outside kernel). Kernel receives `ClaimEvidence`, returns `TopologyDelta`.

2. **Partition = scope_id** - Time is a constraint + candidate-retrieval filter, not a partition boundary. Sliding window for incident candidates, no week-bin artifacts.

3. **Surface identity = (scope_id, question_key)** - `page_id` included in evidence for fallback chain. `source_id` is publisher/domain, `page_id` is specific URL.

4. **Global reconcile defaults to MERGE** - User annotations stored separately from kernel-owned structure (separate `:Annotation` nodes or `user_*` properties). REPLACE is available but requires explicit opt-in.

5. **Deterministic merge keys required** - Surfaces use `signature = hash(scope_id, question_key)`. Incidents use `signature = hash(sorted_anchors, time_model)`. No random IDs for identity - only for internal references.

6. **Traces are mandatory** - Every decision emits a `DecisionTrace`. No silent mutations. Signals connect to inquiry system via explicit mapping.

7. **TopologyKernel naming** - Avoids collision with deprecated `reee.kernel.EpistemicKernel`. New kernel lives in `reee/kernel/` (note: directory, not module).

---

## Problem Statement

Two parallel L2/L3 kernels exist:
- `backend/reee/` - Pure epistemic kernel (types, membrane, builders)
- `backend/workers/principled_weaver.py` - Production L2/L3 with duplicated logic

Goal: Unify into a single kernel that supports both realtime and global reconciliation.

---

## Core Architectural Principles

### 1. Kernel Purity (No LLM Inside)

The kernel must be **pure, deterministic, and replayable**:

- **No LLM calls** - LLM is an evidence provider, not part of the kernel
- **No DB imports** - persistence is the worker's job
- **No network calls** - all evidence arrives via contracts

```
┌─────────────────────────────────────────────────────────────────┐
│                      OUTSIDE KERNEL                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ LLM Provider │  │ DB Provider  │  │ Embedding    │          │
│  │ (optional)   │  │ (entities)   │  │ Provider     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────────┬┴─────────────────┘                   │
│                          ▼                                      │
│                  ┌───────────────┐                              │
│                  │ ClaimEvidence │  (persisted artifacts)       │
│                  └───────┬───────┘                              │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      KERNEL (Pure)                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ TopologyKernel.step(snapshot, evidence) → TopologyDelta     │ │
│  │   - compute_scope_id()                                      │ │
│  │   - extract_question_key()                                  │ │
│  │   - route_to_surface()                                      │ │
│  │   - route_to_incident()                                     │ │
│  │   - update_belief() (Jaynes)                                │ │
│  │   - emit traces, signals, inquiries                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      OUTSIDE KERNEL                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Worker: apply_delta(delta) → Neo4j + Postgres            │   │
│  │ Explainer: format_trace(trace) or llm_explain(trace)     │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 2. Partitioning Strategy

**Partition = `scope_id` only** (not time bins):

- Time is a **constraint within** the kernel, not a partition boundary
- Avoids week-boundary artifacts
- Sliding time window for candidate retrieval

```python
# Realtime: load scope snapshot + temporally-nearby incidents
async def load_partition(self, scope_id: str, claim_time: Optional[datetime]):
    """Load surfaces in scope + incidents within sliding window."""
    surfaces = await self.load_surfaces_by_scope(scope_id)

    # Load incidents that overlap [claim_time - 14d, claim_time + 14d]
    # Plus unknown-time incidents in this scope
    incidents = await self.load_incidents_in_window(
        scope_id,
        time_center=claim_time,
        window_days=14,
        include_unknown_time=True
    )
    return PartitionSnapshot(surfaces=surfaces, incidents=incidents)
```

### 3. Deterministic Identity (Merge Keys)

For global reconcile to converge, use **stable signatures** as merge keys:

```python
# Surface identity: deterministic from (scope_id, question_key)
def compute_surface_signature(scope_id: str, question_key: str) -> str:
    content = f"surface|{scope_id}|{question_key}"
    return f"sf_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

# Incident identity: deterministic from sorted anchors + time_start
# NOTE: time_start (not bin) used for signature - sliding window for membership
def compute_incident_signature(anchors: FrozenSet[str], time_start: Optional[datetime]) -> str:
    sorted_anchors = ",".join(sorted(anchors)[:10])
    # Use ISO week for time component (coarse enough to be stable)
    time_component = "unknown"
    if time_start:
        time_component = f"{time_start.year}-W{time_start.isocalendar()[1]:02d}"
    content = f"incident|{sorted_anchors}|{time_component}"
    return f"inc_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
```

Persistence uses `MERGE` on signature, not random IDs.

---

## Phase 1: Define Kernel Contracts

**Location**: `backend/reee/contracts/`

### 1.1 Evidence Contract (`evidence.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, FrozenSet, Tuple

@dataclass(frozen=True)
class TypedObservation:
    """Typed value observation for Jaynes inference."""
    value: Any
    unit: Optional[str] = None
    confidence: float = 0.5
    authority: float = 0.5

@dataclass(frozen=True)
class ClaimEvidence:
    """Minimal contract for claim input to kernel.

    This is the ONLY input the kernel receives.
    All LLM/DB enrichment happens BEFORE this is constructed.
    """
    # Required identifiers
    claim_id: str
    text: str
    source_id: str              # Publisher/domain
    page_id: Optional[str]      # Specific page (for page_scope fallback)

    # Derived evidence (may be weak/empty)
    entities: FrozenSet[str] = frozenset()
    anchors: FrozenSet[str] = frozenset()
    question_key: Optional[str] = None   # From LLM or pattern extraction
    time: Optional[datetime] = None

    # Confidence per field (fail independently)
    entity_confidence: float = 0.5
    question_key_confidence: float = 0.5
    time_confidence: float = 0.5

    # Optional enrichments (already computed, not live LLM)
    embedding: Optional[Tuple[float, ...]] = None
    typed_observation: Optional[TypedObservation] = None

    # Provenance (for replay/debugging)
    provider_versions: Dict[str, str] = field(default_factory=dict)
    evidence_hash: str = ""     # Hash of all evidence for cache invalidation
```

### 1.2 State Contracts (`state.py`)

```python
@dataclass(frozen=True)
class SurfaceKey:
    """Stable identity for a surface."""
    scope_id: str
    question_key: str

    @property
    def signature(self) -> str:
        """Deterministic merge key."""
        content = f"surface|{self.scope_id}|{self.question_key}"
        return f"sf_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

@dataclass
class SurfaceState:
    """Surface state for kernel computation."""
    # Identity (stable)
    key: SurfaceKey
    signature: str              # = key.signature

    # Mutable state
    claim_ids: FrozenSet[str]
    entities: FrozenSet[str]
    anchor_entities: FrozenSet[str]
    sources: FrozenSet[str]

    # Time bounds
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Belief state (Jaynes)
    posterior_entropy: float = 0.0
    posterior_map: Optional[Any] = None
    observation_count: int = 0

    # Versioning
    kernel_version: str = ""
    params_hash: str = ""
    evidence_hash: str = ""

@dataclass
class IncidentState:
    """Incident state for kernel computation."""
    # Identity (stable)
    id: str                     # Generated, but MERGE uses signature
    signature: str              # Deterministic from anchors + time model

    # Membership
    surface_ids: FrozenSet[str]
    anchor_entities: FrozenSet[str]
    companion_entities: FrozenSet[str]

    # Motifs (L3→L4 contract)
    core_motifs: Tuple[Dict, ...] = ()

    # Time bounds
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Versioning
    kernel_version: str = ""
    params_hash: str = ""

@dataclass(frozen=True)
class PartitionSnapshot:
    """Snapshot of a scope partition for kernel computation."""
    scope_id: str
    surfaces: Tuple[SurfaceState, ...]
    incidents: Tuple[IncidentState, ...]

    # Index for fast lookup
    surface_by_key: Dict[SurfaceKey, SurfaceState] = field(default_factory=dict)
    incident_by_id: Dict[str, IncidentState] = field(default_factory=dict)
```

### 1.3 Delta Contract (`delta.py`)

```python
@dataclass(frozen=True)
class Link:
    """Edge to create/update."""
    from_id: str
    relation: str               # "CONTAINS", "MEMBER_OF", etc.
    to_id: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TopologyDelta:
    """Kernel output: what changed and why."""

    # Structural changes (kernel-owned)
    surface_upserts: List[SurfaceState] = field(default_factory=list)
    incident_upserts: List[IncidentState] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    stale_ids: List[str] = field(default_factory=list)

    # Traces (immutable history)
    decision_traces: List[DecisionTrace] = field(default_factory=list)
    belief_traces: List[BeliefUpdateTrace] = field(default_factory=list)

    # Signals (quality indicators, replaces MetaClaim)
    signals: List[EpistemicSignal] = field(default_factory=list)

    # Inquiry seeds (actionable)
    inquiries: List[InquirySeed] = field(default_factory=list)
```

### 1.4 Trace Contracts (`traces.py`)

```python
@dataclass(frozen=True)
class FeatureVector:
    """Numeric features for decision."""
    anchor_overlap: float = 0.0
    companion_jaccard: float = 0.0
    time_delta_hours: Optional[float] = None
    motif_support: int = 0
    extraction_confidence: float = 0.5
    question_key_confidence: float = 0.5
    time_confidence: float = 0.5

@dataclass(frozen=True)
class DecisionTrace:
    """Membership/routing decision trace."""
    id: str
    decision_type: str          # "surface_key", "incident_membership"
    subject_id: str             # Claim or surface being processed
    target_id: Optional[str]    # Surface or incident joined
    candidate_ids: FrozenSet[str]
    outcome: str                # "created_new", "joined", "rejected"
    features: FeatureVector
    rules_fired: FrozenSet[str] # Which rules determined outcome
    params_hash: str
    kernel_version: str
    timestamp: datetime

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Serialize for LLM explainer (outside kernel)."""
        return {
            "decision_type": self.decision_type,
            "subject": self.subject_id,
            "target": self.target_id,
            "outcome": self.outcome,
            "features": {
                "anchor_overlap": f"{self.features.anchor_overlap:.1%}",
                "companion_similarity": f"{self.features.companion_jaccard:.1%}",
                "time_gap": f"{self.features.time_delta_hours:.1f}h"
                    if self.features.time_delta_hours else "unknown",
            },
            "rules": list(self.rules_fired),
        }

@dataclass(frozen=True)
class BeliefUpdateTrace:
    """Jaynes posterior update trace (L2 proposition inference)."""
    id: str
    surface_id: str
    question_key: str
    claim_id: str

    # Prior state
    prior_entropy: float
    prior_map: Optional[float]
    prior_support: int

    # Observation
    observation_value: Any
    observation_confidence: float
    observation_authority: float
    noise_model: str            # "uniform", "calibrated"

    # Posterior state
    posterior_entropy: float
    posterior_map: Optional[float]
    posterior_support: int

    # Derived
    surprisal: float            # How unexpected
    conflict_detected: bool
    timestamp: datetime
```

### 1.5 Signal Contracts (`signals.py`)

```python
class SignalType(Enum):
    """Quality/status signal types (replaces MetaClaim)."""
    # Data quality
    MISSING_TIME = "missing_time"
    EXTRACTION_SPARSE = "extraction_sparse"
    SCOPE_UNDERPOWERED = "scope_underpowered"

    # Processing blocks
    BRIDGE_BLOCKED = "bridge_blocked"

    # Epistemic state
    HIGH_ENTROPY = "high_entropy"
    CONFLICT = "conflict"
    CORROBORATION_LACKING = "corroboration_lacking"

@dataclass(frozen=True)
class EpistemicSignal:
    """Quality/status signal (replaces MetaClaim)."""
    id: str
    signal_type: SignalType
    subject_id: str
    subject_type: str           # "claim", "surface", "incident"
    severity: str               # "info", "warning", "error"
    evidence: Dict[str, Any]
    resolution_hint: Optional[str]
    timestamp: datetime

class InquiryType(Enum):
    """What kind of evidence would help."""
    RESOLVE_VALUE = "resolve_value"
    DISAMBIGUATE_SCOPE = "disambiguate_scope"
    REQUEST_TIMESTAMP = "request_timestamp"
    IMPROVE_EXTRACTION = "improve_extraction"
    SEEK_CORROBORATION = "seek_corroboration"

@dataclass(frozen=True)
class InquirySeed:
    """Actionable request for evidence."""
    id: str
    inquiry_type: InquiryType
    subject_id: str
    priority: float
    question: str               # Human-readable
    current_state: Dict[str, Any]
    evidence_needed: str
    source_signal_id: Optional[str]  # Link to originating signal
```

### 1.6 Inquiry Integration

Signals connect to `backend/reee/inquiry/` via explicit mapping:

```python
# In backend/reee/inquiry/signal_router.py

SIGNAL_TO_INQUIRY: Dict[SignalType, InquiryType] = {
    SignalType.CONFLICT: InquiryType.RESOLVE_VALUE,
    SignalType.BRIDGE_BLOCKED: InquiryType.DISAMBIGUATE_SCOPE,
    SignalType.MISSING_TIME: InquiryType.REQUEST_TIMESTAMP,
    SignalType.EXTRACTION_SPARSE: InquiryType.IMPROVE_EXTRACTION,
    SignalType.CORROBORATION_LACKING: InquiryType.SEEK_CORROBORATION,
    SignalType.HIGH_ENTROPY: InquiryType.SEEK_CORROBORATION,
}

def signal_to_inquiry(signal: EpistemicSignal) -> Optional[InquirySeed]:
    """Convert signal to actionable inquiry for webapp_seeder."""
    inquiry_type = SIGNAL_TO_INQUIRY.get(signal.signal_type)
    if not inquiry_type:
        return None

    return InquirySeed(
        id=generate_id("inquiry"),
        inquiry_type=inquiry_type,
        subject_id=signal.subject_id,
        priority=_compute_priority(signal),
        question=_generate_question(signal, inquiry_type),
        current_state=signal.evidence,
        evidence_needed=_describe_needed(inquiry_type),
        source_signal_id=signal.id,
    )

# webapp_seeder.py consumes InquirySeed objects
```

### 1.7 Deliverables

- [ ] `backend/reee/contracts/__init__.py`
- [ ] `backend/reee/contracts/evidence.py`
- [ ] `backend/reee/contracts/state.py`
- [ ] `backend/reee/contracts/delta.py`
- [ ] `backend/reee/contracts/traces.py`
- [ ] `backend/reee/contracts/signals.py`
- [ ] `backend/reee/inquiry/signal_router.py`
- [ ] Unit tests for serialization/immutability

---

## Phase 2: Extract Weaver Logic to Pure Kernel

**Location**: `backend/reee/kernel/`

**Note**: Use `TopologyKernel` as the name (not `EpistemicKernel` which conflicts with deprecated `reee.kernel`).

### 2.1 Scope Computation (`scope.py`)

```python
DEFAULT_HUB_ENTITIES = frozenset({
    "United States", "China", "European Union", "United Nations",
    "Asia", "Europe", "North America", "South America", "Africa"
})

def compute_scope_id(
    anchor_entities: FrozenSet[str],
    hub_entities: FrozenSet[str] = DEFAULT_HUB_ENTITIES
) -> str:
    """Compute deterministic scope_id from anchors.

    Pure function - no DB, no LLM.
    """
    scoping = anchor_entities - hub_entities
    if not scoping:
        scoping = anchor_entities  # Fallback: all anchors are hubs

    normalized = sorted(a.lower().replace(" ", "").replace("'", "") for a in scoping)
    primary = normalized[:2]  # Top 2 for scope drift tolerance

    return "scope_" + "_".join(primary) if primary else "scope_unscoped"
```

### 2.2 Question Key Extraction (`question_key.py`)

```python
class FallbackLevel(Enum):
    EXPLICIT = 1      # Trusted LLM-extracted or external key
    PATTERN = 2       # Pattern-matched (death_count, policy_status)
    ENTITY = 3        # Entity-derived (about_X_Y)
    PAGE_SCOPE = 4    # Page/source fallback
    SINGLETON = 5     # Never collapse

@dataclass(frozen=True)
class QuestionKeyResult:
    question_key: str
    fallback_level: FallbackLevel
    confidence: float

def extract_question_key(
    text: str,
    entities: FrozenSet[str],
    anchors: FrozenSet[str],
    page_id: Optional[str],
    claim_id: str,
    explicit_key: Optional[str] = None,
    explicit_confidence: float = 0.9
) -> QuestionKeyResult:
    """Extract question_key with explicit fallback chain.

    Pure function - no DB, no LLM (LLM result arrives via explicit_key).

    Fallback hierarchy:
    1. explicit_key if provided and confident
    2. Pattern-extracted from text (death_count, etc.)
    3. Entity-derived (about_X_Y)
    4. page_scope_{page_id}
    5. singleton_{claim_id}
    """
    # Level 1: Explicit (from LLM or upstream)
    if explicit_key and explicit_confidence >= 0.7:
        return QuestionKeyResult(explicit_key, FallbackLevel.EXPLICIT, explicit_confidence)

    # Level 2: Pattern matching
    pattern_key = _match_pattern(text)
    if pattern_key:
        return QuestionKeyResult(pattern_key, FallbackLevel.PATTERN, 0.8)

    # Level 3: Entity-derived
    if anchors:
        sorted_anchors = sorted(anchors)[:2]
        key = "about_" + "_".join(_normalize(a) for a in sorted_anchors)
        return QuestionKeyResult(key, FallbackLevel.ENTITY, 0.6)

    # Level 4: Page scope
    if page_id:
        return QuestionKeyResult(f"page_scope_{page_id}", FallbackLevel.PAGE_SCOPE, 0.4)

    # Level 5: Singleton (never collapse)
    return QuestionKeyResult(f"singleton_{claim_id}", FallbackLevel.SINGLETON, 0.1)
```

### 2.3 Surface Update (`surface_update.py`)

```python
def compute_surface_key(
    evidence: ClaimEvidence,
    hub_entities: FrozenSet[str] = DEFAULT_HUB_ENTITIES
) -> Tuple[SurfaceKey, DecisionTrace]:
    """Compute surface identity with trace."""
    scope_id = compute_scope_id(evidence.anchors, hub_entities)

    qk_result = extract_question_key(
        text=evidence.text,
        entities=evidence.entities,
        anchors=evidence.anchors,
        page_id=evidence.page_id,
        claim_id=evidence.claim_id,
        explicit_key=evidence.question_key,
        explicit_confidence=evidence.question_key_confidence,
    )

    key = SurfaceKey(scope_id=scope_id, question_key=qk_result.question_key)

    trace = DecisionTrace(
        id=generate_trace_id(),
        decision_type="surface_key",
        subject_id=evidence.claim_id,
        target_id=key.signature,
        candidate_ids=frozenset(),
        outcome=f"key_{qk_result.fallback_level.name.lower()}",
        features=FeatureVector(
            question_key_confidence=qk_result.confidence,
            extraction_confidence=evidence.entity_confidence,
        ),
        rules_fired=frozenset({f"FALLBACK_{qk_result.fallback_level.name}"}),
        params_hash=...,
        kernel_version=...,
        timestamp=datetime.utcnow(),
    )

    return key, trace

def update_surface_belief(
    surface: SurfaceState,
    observation: TypedObservation,
    claim_id: str,
    noise_model: NoiseModel = UniformNoise()
) -> Tuple[SurfaceState, BeliefUpdateTrace]:
    """Update Jaynes posterior with trace.

    Uses TypedBeliefState from reee.typed_belief.
    """
    # ... implementation using TypedBeliefState
```

### 2.4 Incident Routing (`incident_routing.py`)

```python
@dataclass
class RoutingCandidate:
    incident_id: str
    anchor_overlap: float       # Jaccard
    companion_jaccard: float
    time_delta_hours: Optional[float]
    compatible: bool
    block_reason: Optional[str]

def find_incident_candidates(
    surface_anchors: FrozenSet[str],
    surface_companions: FrozenSet[str],
    surface_time: Optional[datetime],
    incidents: Tuple[IncidentState, ...],
    params: IncidentRoutingParams
) -> List[RoutingCandidate]:
    """Find candidate incidents.

    Pure function - no DB.
    Time is a constraint, not a partition boundary.
    """
    candidates = []
    for inc in incidents:
        # Anchor overlap
        shared = surface_anchors & inc.anchor_entities
        if len(shared) < params.min_shared_anchors:
            continue

        anchor_overlap = len(shared) / len(surface_anchors | inc.anchor_entities)

        # Companion compatibility
        companion_jaccard = 0.0
        block_reason = None
        compatible = True

        if surface_companions and inc.companion_entities:
            intersection = surface_companions & inc.companion_entities
            union = surface_companions | inc.companion_entities
            companion_jaccard = len(intersection) / len(union) if union else 0.0

            if companion_jaccard < params.companion_threshold:
                compatible = False
                block_reason = f"companion_disjoint({companion_jaccard:.2f})"

        # Time compatibility (sliding window, not bin)
        time_delta = None
        if surface_time and inc.time_start:
            time_delta = abs((surface_time - inc.time_start).total_seconds() / 3600)
            if time_delta > params.time_window_hours:
                compatible = False
                block_reason = f"time_gap({time_delta:.0f}h)"

        candidates.append(RoutingCandidate(
            incident_id=inc.id,
            anchor_overlap=anchor_overlap,
            companion_jaccard=companion_jaccard,
            time_delta_hours=time_delta,
            compatible=compatible,
            block_reason=block_reason,
        ))

    return candidates

def route_to_incident(
    surface_id: str,
    surface_anchors: FrozenSet[str],
    surface_time: Optional[datetime],
    candidates: List[RoutingCandidate],
    params: IncidentRoutingParams
) -> Tuple[str, bool, DecisionTrace, Optional[EpistemicSignal]]:
    """Decide which incident to join (or create new).

    Returns (incident_id, is_new, trace, optional_signal).
    """
    compatible = [c for c in candidates if c.compatible]
    blocked = [c for c in candidates if not c.compatible]

    signals = []

    # Emit bridge_blocked signals
    for b in blocked:
        signals.append(EpistemicSignal(
            id=generate_id("signal"),
            signal_type=SignalType.BRIDGE_BLOCKED,
            subject_id=surface_id,
            subject_type="surface",
            severity="info",
            evidence={
                "incident_id": b.incident_id,
                "anchor_overlap": b.anchor_overlap,
                "block_reason": b.block_reason,
            },
            resolution_hint="Check if these are actually the same incident",
            timestamp=datetime.utcnow(),
        ))

    if compatible:
        # Join best match
        best = max(compatible, key=lambda c: c.anchor_overlap)
        outcome = "joined"
        target_id = best.incident_id
        is_new = False
    else:
        # Create new incident with deterministic signature
        target_id = compute_incident_signature(surface_anchors, surface_time)
        outcome = "created_new"
        is_new = True

    trace = DecisionTrace(...)
    return target_id, is_new, trace, signals[0] if signals else None
```

### 2.5 Main Kernel (`__init__.py`)

```python
class TopologyKernel:
    """Pure epistemic kernel for L2/L3 emergence.

    NO DB imports. NO LLM calls. Pure computation.
    """

    def __init__(self, params: KernelParams, version: str = "1.0.0"):
        self.params = params
        self.version = version
        self.params_hash = compute_params_hash(params)

    def step(
        self,
        snapshot: PartitionSnapshot,
        evidence: ClaimEvidence
    ) -> TopologyDelta:
        """Process single claim, return delta.

        Pure function - all state passed in, delta passed out.
        """
        traces = []
        signals = []
        inquiries = []

        # 1. Compute surface key
        surface_key, key_trace = compute_surface_key(evidence)
        traces.append(key_trace)

        # 2. Find or create surface
        surface, surface_updated = self._upsert_surface(
            snapshot, surface_key, evidence
        )

        # 3. Route to incident
        candidates = find_incident_candidates(
            surface.anchor_entities,
            surface.entities - surface.anchor_entities,
            evidence.time,
            snapshot.incidents,
            self.params.incident_routing,
        )

        incident_id, is_new, routing_trace, signal = route_to_incident(
            surface.signature,
            surface.anchor_entities,
            candidates,
            self.params.incident_routing,
        )
        traces.append(routing_trace)
        if signal:
            signals.append(signal)

        # 4. Update belief if typed observation
        belief_trace = None
        if evidence.typed_observation:
            surface, belief_trace = update_surface_belief(
                surface, evidence.typed_observation, evidence.claim_id
            )
            traces.append(belief_trace)

        # 5. Emit signals for data quality issues
        signals.extend(self._check_data_quality(evidence))

        # 6. Convert signals to inquiry seeds
        for sig in signals:
            inquiry = signal_to_inquiry(sig)
            if inquiry:
                inquiries.append(inquiry)

        return TopologyDelta(
            surface_upserts=[surface],
            incident_upserts=[...],
            links=[Link(surface.signature, "CONTAINS", evidence.claim_id)],
            decision_traces=traces,
            belief_traces=[belief_trace] if belief_trace else [],
            signals=signals,
            inquiries=inquiries,
        )
```

### 2.6 Deliverables

- [ ] `backend/reee/kernel/__init__.py` (TopologyKernel)
- [ ] `backend/reee/kernel/scope.py`
- [ ] `backend/reee/kernel/question_key.py`
- [ ] `backend/reee/kernel/surface_update.py`
- [ ] `backend/reee/kernel/incident_routing.py`
- [ ] `backend/reee/kernel/motifs.py`
- [ ] `backend/reee/kernel/params.py`
- [ ] Unit tests for each module (no DB required)

---

## Phase 3: Build Prototype Weaver

**Location**: `backend/workers/kernel_weaver.py`

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      KernelWeaver                               │
├─────────────────────────────────────────────────────────────────┤
│  1. Load claim from DB                                          │
│  2. Build ClaimEvidence (via providers - LLM cached here)       │
│  3. Load scope snapshot + incidents in sliding window           │
│  4. Call kernel.step(snapshot, evidence) → delta                │
│  5. Apply delta to DB (MERGE by signature)                      │
│  6. Route inquiries to inquiry system                           │
│  7. Optionally generate explanations (outside kernel)           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Evidence Providers (Outside Kernel)

```python
# backend/workers/evidence_providers.py

class EvidenceProvider(Protocol):
    """Interface for evidence enrichment (OUTSIDE kernel)."""
    async def enrich(self, claim: Claim) -> Dict[str, Any]:
        """Return enrichment data to merge into ClaimEvidence."""

class EntityProvider(EvidenceProvider):
    """Hydrate entities from DB."""
    async def enrich(self, claim: Claim) -> Dict[str, Any]:
        entities = await self.repo.get_entities(claim.id)
        return {
            "entities": frozenset(e.canonical_name for e in entities),
            "anchors": frozenset(e.canonical_name for e in entities if e.is_anchor),
            "entity_confidence": 0.8,  # DB entities are trusted
        }

class LLMExtractionProvider(EvidenceProvider):
    """Optional: LLM-based extraction (results are CACHED).

    LLM is called ONCE per claim, result persisted.
    Kernel sees only the persisted artifact.
    """
    async def enrich(self, claim: Claim) -> Dict[str, Any]:
        # Check cache first
        cached = await self.cache.get(f"llm_extraction:{claim.id}")
        if cached:
            return cached

        # Call LLM (only if not cached)
        result = await self.llm.extract(claim.text)

        # Persist for replay
        enrichment = {
            "question_key": result.question_key,
            "question_key_confidence": result.confidence,
            "typed_observation": result.typed_observation,
            "provider_versions": {"llm": self.model_version},
        }
        await self.cache.set(f"llm_extraction:{claim.id}", enrichment)

        return enrichment
```

### 3.3 Persistence with Merge Keys

```python
async def apply_delta(self, delta: TopologyDelta):
    """Apply delta using MERGE on signatures (not random IDs)."""

    for surface in delta.surface_upserts:
        # MERGE by signature ensures convergence
        await self.neo4j._execute_write("""
            MERGE (s:Surface {signature: $signature})
            SET s.scope_id = $scope_id,
                s.question_key = $question_key,
                s.entities = $entities,
                s.anchor_entities = $anchors,
                s.kernel_version = $kernel_version,
                s.params_hash = $params_hash,
                s.updated_at = datetime()
        """, {
            'signature': surface.signature,
            'scope_id': surface.key.scope_id,
            'question_key': surface.key.question_key,
            'entities': list(surface.entities),
            'anchors': list(surface.anchor_entities),
            'kernel_version': surface.kernel_version,
            'params_hash': surface.params_hash,
        })
```

### 3.4 Shadow Mode Safety

```python
if WEAVER_MODE == "shadow":
    # Shadow mode: compute only, DO NOT persist
    legacy_result = await legacy_weaver.process(claim_id)
    kernel_delta = kernel_weaver.compute_only(claim_id)  # No apply_delta!

    await log_comparison(legacy_result, kernel_delta)

    # Only legacy persists
    return legacy_result
```

### 3.5 Deliverables

- [ ] `backend/workers/kernel_weaver.py`
- [ ] `backend/workers/evidence_providers.py`
- [ ] `backend/reee/tests/integration/test_kernel_weaver.py`
- [ ] Docker compose for isolated test environment

---

## Phase 4: Validate Against Real Data

### 4.1 Corpus Selection

Select ~100 claims covering key scenarios:

```python
SCENARIOS = [
    "multi_source_fire_incident",
    "bridge_immunity_test",
    "typed_conflict",
    "sparse_extraction",
    "singleton_fallback",
    "sliding_time_window",      # Not "time_boundary" - no bins
]
```

### 4.2 Success Criteria

| Metric | Requirement |
|--------|-------------|
| Invariant preservation | 100% (no cross-scope, hub suppression) |
| Surface identity match | ≥95% |
| Incident routing match | ≥90% |
| Explainable diffs | 100% (every diff has trace) |
| Convergence | Global reconcile idempotent |

### 4.3 Deliverables

- [ ] `backend/reee/tests/scripts/select_validation_corpus.py`
- [ ] `backend/reee/tests/scripts/run_comparison.py`
- [ ] Comparison report
- [ ] Golden fixture updates

---

## Phase 5: Integration + Migration

### 5.1 Reconcile Modes

```python
class ReconcileMode(Enum):
    VALIDATE = "validate"   # Diff only, no persistence
    MERGE = "merge"         # Update kernel-owned fields, preserve annotations
    REPLACE = "replace"     # Full replace (use with caution)

# Kernel-owned vs user-owned separation
KERNEL_OWNED_FIELDS = {
    "scope_id", "question_key", "signature",
    "entities", "anchor_entities", "sources",
    "kernel_version", "params_hash", "evidence_hash",
}

USER_OWNED_FIELDS = {
    "canonical_title",      # Editorial override
    "user_annotations",     # Manual notes
    "manual_membership",    # Human correction
}
```

### 5.2 Migration Steps

1. **Shadow mode** (1 week): Compute + diff, legacy persists
2. **Kernel mode** (1 week): New claims use kernel
3. **Reconcile** (1 run): Global rebuild with VALIDATE, then MERGE
4. **Retire legacy**: Remove duplication

### 5.3 Deliverables

- [ ] Feature flag implementation
- [ ] Shadow mode (compute-only, no persist)
- [ ] Reconcile job with VALIDATE/MERGE modes
- [ ] Rollback runbook

---

## Open Questions (Deferred)

These are explicitly deferred to future phases:

1. **Surface signature stability on extraction improvement**: If question_key extraction improves (better LLM), surfaces may get new signatures. Deferred: handle via `superseded_by` annotation or migration tooling in Phase N.

2. **Bayesian membrane**: Phase N could add `MembershipPosteriorTrace` for probabilistic routing (posterior over {join A, join B, create new}). Not in scope for initial unification.

3. **Annotation storage format**: Current decision: separate `:Annotation` nodes linked to kernel-owned nodes. Alternative (`user_*` properties) rejected for cleaner REPLACE semantics.

---

## Success Metrics

1. **Kernel isolation**: No DB/LLM imports in `reee/kernel/`
2. **Trace coverage**: Every decision has a DecisionTrace
3. **Replay safety**: `kernel.step()` is deterministic given same evidence
4. **Convergence**: Two reconcile runs produce identical topology
5. **Explainability**: `format_trace()` produces useful debugging output
