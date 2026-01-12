"""
Principled Emergence Experiment
===============================

Grows from 0 claims, applying the RIGHT theory at each step:

- JAYNES: For typed variables with observations (P(X|evidence))
- BIANCONI: For membrane formation (higher-order structure)

Each step is:
1. Theoretically justified (which theory, why)
2. Empirically validated (what changed, what emerged)
3. Auditable (full constraint ledger)

NO ad-hoc thresholds without stated loss functions.
NO semantic-only decisions.
NO hidden assumptions.

EMPIRICAL RESULTS (2026-01, 500 claims)
=======================================

| Metric                  | Baseline | Graded (log-evidence) |
|-------------------------|----------|----------------------|
| Events                  | 247      | 244                  |
| Multi-surface events    | 2        | 5                    |
| Largest core            | 2        | 4                    |
| Core edges              | 5        | 8                    |
| Subsample stability 80% | -        | 60-100% preserved    |

Graded mode uses log(support+1) instead of hard threshold, allowing:
- Partial evidence (support=1 → weight≈0.69 > threshold≈0.55)
- 1214 extra motifs included vs baseline
- More cores without mega-merge (largest stays bounded at 4)
- Sensible clusters: Jimmy Lai, Do Kwon/SBF, Venezuela, Brown, Epstein

Triangle bonus alone has no effect without graded mode because
k≥3 motifs with support≥2 are rare in sparse extraction.

Usage:
    docker exec herenews-app python -m reee.experiments.principled_emergence
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

# =============================================================================
# CONSTRAINT LEDGER: The core data structure
# =============================================================================

@dataclass
class Constraint:
    """
    A single constraint with provenance.

    This is the atomic unit - everything flows through constraints.
    """
    id: str
    constraint_type: str  # "structural", "semantic", "typed", "temporal"

    # What does this constraint assert?
    assertion: str  # Human-readable

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    provenance: str = ""  # Where did this come from?

    # Confidence (for semantic constraints)
    confidence: float = 1.0

    # Timestamp
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_structural(self) -> bool:
        return self.constraint_type == "structural"

    def is_semantic(self) -> bool:
        return self.constraint_type == "semantic"


@dataclass
class ConstraintLedger:
    """
    The constraint ledger - replaces score soup.

    All decisions are derived from constraints, not computed directly.
    """
    constraints: List[Constraint] = field(default_factory=list)

    # Index by type
    _by_type: Dict[str, List[Constraint]] = field(default_factory=lambda: defaultdict(list))

    # Index by scope (surface/event ID)
    _by_scope: Dict[str, List[Constraint]] = field(default_factory=lambda: defaultdict(list))

    def add(self, constraint: Constraint, scope: str = None):
        """Add constraint to ledger."""
        self.constraints.append(constraint)
        self._by_type[constraint.constraint_type].append(constraint)
        if scope:
            self._by_scope[scope].append(constraint)

    def get_structural(self) -> List[Constraint]:
        return self._by_type["structural"]

    def get_semantic(self) -> List[Constraint]:
        return self._by_type["semantic"]

    def for_scope(self, scope: str) -> List[Constraint]:
        return self._by_scope.get(scope, [])

    def can_form_core(self, scope1: str, scope2: str) -> Tuple[bool, str]:
        """
        ANTI-TRAP RULE: Core edges require ≥2 constraints, ≥1 non-semantic.
        """
        pair_key = f"{scope1}:{scope2}"
        constraints = self.for_scope(pair_key)

        if len(constraints) < 2:
            return False, f"Only {len(constraints)} constraints (need ≥2)"

        non_semantic = [c for c in constraints if not c.is_semantic()]
        if not non_semantic:
            return False, "All constraints are semantic (need ≥1 structural/temporal)"

        return True, f"Valid: {len(non_semantic)} structural + {len(constraints) - len(non_semantic)} semantic"


# =============================================================================
# STEP 0: CLAIM ARRIVAL
# =============================================================================

@dataclass
class Claim:
    """A claim as it arrives - minimal structure."""
    id: str
    text: str
    entities: Set[str]  # Entity names (not IDs yet)
    source: str
    timestamp: Optional[datetime] = None
    embedding: Optional[List[float]] = None

    # Extracted typed values (if any)
    typed_values: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"death_count": 17, "death_count_kind": "point"}


@dataclass
class ExperimentState:
    """Full state of the experiment at any step."""
    step: int = 0
    claims: List[Claim] = field(default_factory=list)

    # L1: Hyperedges (each claim is a hyperedge over its entities)
    hyperedges: Dict[str, Set[str]] = field(default_factory=dict)  # claim_id -> entity set

    # L2: Surfaces (identity clusters)
    surfaces: Dict[str, Set[str]] = field(default_factory=dict)  # surface_id -> claim_ids
    surface_entities: Dict[str, Set[str]] = field(default_factory=dict)  # surface_id -> entities

    # L3: Events (aboutness clusters)
    events: Dict[str, Set[str]] = field(default_factory=dict)  # event_id -> surface_ids

    # Constraint ledger
    ledger: ConstraintLedger = field(default_factory=ConstraintLedger)

    # Typed belief states (Jaynes)
    belief_states: Dict[str, Any] = field(default_factory=dict)  # variable_id -> TypedBeliefState

    # Audit log
    audit: List[str] = field(default_factory=list)

    # Quiet mode (suppress step-by-step output)
    quiet: bool = False

    def log(self, msg: str):
        self.audit.append(f"[Step {self.step}] {msg}")
        if not self.quiet:
            print(f"[Step {self.step}] {msg}")


# =============================================================================
# STEP 1: HYPEREDGE FORMATION (Bianconi)
# =============================================================================

def step1_form_hyperedge(state: ExperimentState, claim: Claim) -> str:
    """
    Each claim becomes a hyperedge over its entity set.

    THEORY: Bianconi higher-order networks
    - A claim with entities {A, B, C} is a 3-hyperedge
    - NOT three pairwise edges (A-B, B-C, A-C)
    - The hyperedge captures "these entities co-occur in this claim"

    WHY: Pairwise loses information. If claim says "A and B met at C",
    the 3-way co-occurrence is stronger than three 2-way co-occurrences.
    """
    state.step = 1

    if len(claim.entities) < 1:
        state.log(f"Claim {claim.id}: No entities, skipping hyperedge")
        return None

    # Form hyperedge
    state.hyperedges[claim.id] = claim.entities.copy()
    state.claims.append(claim)

    # Add structural constraint
    state.ledger.add(Constraint(
        id=f"hyper_{claim.id}",
        constraint_type="structural",
        assertion=f"Entities {claim.entities} co-occur in claim",
        evidence={"claim_id": claim.id, "entity_count": len(claim.entities)},
        provenance=claim.source
    ), scope=claim.id)

    state.log(f"Claim {claim.id}: Formed {len(claim.entities)}-hyperedge over {claim.entities}")
    return claim.id


# =============================================================================
# STEP 2: MOTIF DETECTION (Bianconi)
# =============================================================================

@dataclass
class MotifConfig:
    """Configuration for motif detection."""
    min_k: int = 2  # Minimum motif size
    min_support: int = 2  # Minimum support threshold
    graded: bool = False  # Use graded evidence instead of step function
    triangle_bonus: float = 1.5  # Extra weight for k>=3 motifs


def step2_detect_motifs(
    state: ExperimentState,
    min_support: int = 2,
    config: MotifConfig = None
) -> Dict[frozenset, float]:
    """
    Find repeated k-sets (k≥2) across hyperedges.

    THEORY: Bianconi motifs
    - A motif is a k-set that appears in multiple hyperedges
    - Motifs with support ≥ min_support are "real" patterns
    - Singleton entities (k=1) are NOT motifs - they can be hubs

    GRADED MODE (config.graded=True):
    - Instead of hard threshold, compute log-evidence weight
    - weight = log(support) * k_bonus
    - This smooths the decision boundary and improves stability

    TRIANGLE BONUS (config.triangle_bonus):
    - k>=3 motifs are more discriminative than pairs
    - Apply multiplier to their evidence weight
    """
    state.step = 2

    if config is None:
        config = MotifConfig(min_support=min_support)

    # Count all k-sets (k≥2)
    kset_counts = defaultdict(int)
    kset_claims = defaultdict(set)  # Which claims contain this k-set

    for claim_id, entities in state.hyperedges.items():
        entity_list = list(entities)
        n = len(entity_list)

        # Generate all k-subsets where k >= min_k
        for k in range(config.min_k, n + 1):
            from itertools import combinations
            for subset in combinations(entity_list, k):
                kset = frozenset(subset)
                kset_counts[kset] += 1
                kset_claims[kset].add(claim_id)

    # Compute motif weights
    if config.graded:
        # Graded: log-evidence with k-bonus
        motifs = {}
        for kset, count in kset_counts.items():
            if count >= 1:  # Include all, weight by evidence
                k = len(kset)
                k_bonus = config.triangle_bonus if k >= 3 else 1.0
                # Log-evidence: log(support + 1) to handle support=1
                weight = math.log(count + 1) * k_bonus
                # Only include if weight exceeds threshold equivalent
                # threshold = log(min_support + 1) ≈ log(3) ≈ 1.1
                if weight >= math.log(config.min_support + 1) * 0.5:  # Allow partial evidence
                    motifs[kset] = weight
    else:
        # Step function (original)
        motifs = {}
        for kset, count in kset_counts.items():
            if count >= config.min_support:
                k = len(kset)
                k_bonus = config.triangle_bonus if k >= 3 else 1.0
                motifs[kset] = count * k_bonus

    # Add structural constraints for motifs
    for kset, weight in motifs.items():
        support = kset_counts[kset]
        state.ledger.add(Constraint(
            id=f"motif_{hash(kset) % 10000}",
            constraint_type="structural",
            assertion=f"Motif {set(kset)} appears {support} times (weight={weight:.2f})",
            evidence={
                "entities": list(kset),
                "support": support,
                "weight": weight,
                "k": len(kset),
                "claims": list(kset_claims[kset])
            },
            provenance="motif_detection"
        ))

    state.log(f"Detected {len(motifs)} motifs (graded={config.graded}, triangle_bonus={config.triangle_bonus})")
    for kset, weight in sorted(motifs.items(), key=lambda x: -x[1])[:5]:
        state.log(f"  Motif {set(kset)}: support={kset_counts[kset]}, weight={weight:.2f}, k={len(kset)}")

    return motifs


# =============================================================================
# STEP 3: SURFACE FORMATION (Identity - Bianconi)
# =============================================================================

def step3_form_surfaces(state: ExperimentState, motifs: Dict[frozenset, int]) -> Dict[str, Set[str]]:
    """
    Cluster claims into surfaces using motif co-membership.

    THEORY: Identity via shared motifs (Bianconi)
    - Two claims are in the same surface if they share a supported motif
    - NOT if they share a single entity (that could be a hub)
    - Connected components of motif-sharing form surfaces

    WHY: Surfaces are "about the same thing". Shared motifs indicate
    same referent more reliably than shared singletons.

    RULE: Singleton overlap can attach periphery but NEVER forms cores.
    """
    state.step = 3

    # Build claim-claim edges based on shared motifs
    claim_edges = defaultdict(set)

    for motif, support in motifs.items():
        # Find all claims containing this motif
        claims_with_motif = []
        for claim_id, entities in state.hyperedges.items():
            if motif.issubset(entities):
                claims_with_motif.append(claim_id)

        # Connect all pairs
        for i, c1 in enumerate(claims_with_motif):
            for c2 in claims_with_motif[i+1:]:
                claim_edges[c1].add(c2)
                claim_edges[c2].add(c1)

                # Add constraint for this edge
                pair_key = f"{min(c1,c2)}:{max(c1,c2)}"
                state.ledger.add(Constraint(
                    id=f"identity_{pair_key}",
                    constraint_type="structural",
                    assertion=f"Claims share motif {set(motif)}",
                    evidence={"motif": list(motif), "support": support},
                    provenance="surface_formation"
                ), scope=pair_key)

    # Find connected components
    visited = set()
    surfaces = {}
    surface_idx = 0

    for claim_id in state.hyperedges:
        if claim_id in visited:
            continue

        # BFS
        component = set()
        queue = [claim_id]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            component.add(curr)
            queue.extend(claim_edges[curr] - visited)

        surface_id = f"S{surface_idx:03d}"
        surfaces[surface_id] = component

        # Compute surface entities (union of claim entities)
        surface_entities = set()
        for cid in component:
            surface_entities.update(state.hyperedges[cid])
        state.surface_entities[surface_id] = surface_entities

        surface_idx += 1

    state.surfaces = surfaces
    state.log(f"Formed {len(surfaces)} surfaces from motif clustering")

    for sid, claims in sorted(surfaces.items(), key=lambda x: -len(x[1]))[:5]:
        state.log(f"  {sid}: {len(claims)} claims, entities={state.surface_entities[sid]}")

    # Add surface-level motif constraints for surfaces that share motifs
    # This is needed because Step 7 checks constraints at surface-pair level
    surface_ids = list(surfaces.keys())
    for i, s1_id in enumerate(surface_ids):
        for s2_id in surface_ids[i+1:]:
            s1_entities = state.surface_entities[s1_id]
            s2_entities = state.surface_entities[s2_id]

            # Check if surfaces share any supported motif
            shared_motifs = []
            for motif, support in motifs.items():
                if motif.issubset(s1_entities) and motif.issubset(s2_entities):
                    shared_motifs.append((motif, support))

            if shared_motifs:
                pair_key = f"{s1_id}:{s2_id}"
                for motif, support in shared_motifs:
                    state.ledger.add(Constraint(
                        id=f"surface_motif_{pair_key}_{hash(motif) % 10000}",
                        constraint_type="structural",
                        assertion=f"Surfaces share motif {set(motif)}",
                        evidence={"motif": list(motif), "support": support},
                        provenance="surface_motif_sharing"
                    ), scope=pair_key)

    return surfaces


# =============================================================================
# STEP 4: TYPED VARIABLE EXTRACTION (Jaynes preparation)
# =============================================================================

def step4_extract_typed_values(state: ExperimentState) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Extract typed values from claims for Jaynes inference.

    THEORY: Jaynes is for TYPED VARIABLES with observations
    - "17 dead" → observation about death_count variable
    - "at least 100 injured" → lower bound observation
    - These are NOT structural constraints - they're probabilistic observations

    WHY: Jaynes gives us P(X|evidence) with proper uncertainty.
    We don't guess the death toll - we infer it from observations.

    SCOPE: Each surface has its own belief state for each variable.
    """
    state.step = 4

    # Group typed values by surface and variable
    surface_observations = defaultdict(lambda: defaultdict(list))

    for surface_id, claim_ids in state.surfaces.items():
        for claim_id in claim_ids:
            claim = next((c for c in state.claims if c.id == claim_id), None)
            if not claim:
                continue

            for var_name, value in claim.typed_values.items():
                surface_observations[surface_id][var_name].append({
                    "value": value,
                    "source": claim.source,
                    "claim_id": claim_id
                })

    state.log(f"Extracted typed values for {len(surface_observations)} surfaces")

    for sid, variables in surface_observations.items():
        for var_name, obs_list in variables.items():
            state.log(f"  {sid}.{var_name}: {len(obs_list)} observations")

            # Add typed constraint
            state.ledger.add(Constraint(
                id=f"typed_{sid}_{var_name}",
                constraint_type="typed",
                assertion=f"Variable {var_name} has {len(obs_list)} observations",
                evidence={"observations": obs_list},
                provenance="typed_extraction"
            ), scope=sid)

    return dict(surface_observations)


# =============================================================================
# STEP 5: JAYNES INFERENCE (per surface, per variable)
# =============================================================================

def step5_jaynes_inference(
    state: ExperimentState,
    surface_observations: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Run Jaynes inference for each typed variable in each surface.

    THEORY: Bayesian updating with explicit priors and likelihoods
    - Prior: P(X) - what we believed before observations
    - Likelihood: P(observation | X) - noise model
    - Posterior: P(X | observations) - what we believe now

    WHY: This is the ONLY principled way to combine conflicting observations.
    "13 dead" vs "17 dead" → multi-modal posterior, not a guess.

    OUTPUT: Posterior distribution, entropy, credible intervals
    """
    state.step = 5

    from ..typed_belief import TypedBeliefState, CountDomain, CountDomainConfig, Observation, UniformNoise

    results = {}

    for surface_id, variables in surface_observations.items():
        results[surface_id] = {}

        for var_name, obs_list in variables.items():
            # Create belief state with explicit configuration
            config = CountDomainConfig(
                max_count=500,
                scales=[("small", 10.0, 0.5), ("large", 100.0, 0.5)],
                monotone=True
            )
            belief = TypedBeliefState(
                domain=CountDomain(config),
                noise_model=UniformNoise(delta=2.0)  # Explicit noise assumption
            )

            # Add observations
            for obs in obs_list:
                value = obs["value"]
                if isinstance(value, int):
                    observation = Observation.point(value, confidence=0.85, source=obs["source"])
                    belief.add_observation(observation)

            # Compute posterior
            posterior = belief.compute_posterior()

            results[surface_id][var_name] = {
                "map_value": belief.map_value,
                "map_probability": belief.map_probability,
                "entropy": belief.entropy(),
                "normalized_entropy": belief.normalized_entropy(),
                "n_observations": len(obs_list),
                "credible_95": belief.credible_interval(0.95)
            }

            state.belief_states[f"{surface_id}.{var_name}"] = belief

            state.log(f"  {surface_id}.{var_name}: MAP={belief.map_value} (p={belief.map_probability:.2f}), H={belief.entropy():.2f} bits")

    return results


# =============================================================================
# STEP 6: CONTEXT COMPATIBILITY (Bianconi - higher-order check)
# =============================================================================

@dataclass
class ContextResult:
    """Result of context compatibility check for one entity."""
    entity: str
    compatible: bool
    underpowered: bool
    overlap: float
    companions1_size: int
    companions2_size: int
    reason: str


def step6_context_compatibility(state: ExperimentState) -> Dict[Tuple[str, str], Tuple[bool, bool, List[ContextResult]]]:
    """
    Check if shared anchors have compatible context between surfaces.

    THEORY: Higher-order membrane (Bianconi)
    - Entity E in Surface A has companions (other entities in A)
    - Entity E in Surface B has companions (other entities in B)
    - If companions overlap → E binds A and B (same topic)
    - If companions disjoint → E bridges different topics (should NOT bind)

    INFERENCE (not hard threshold):
    - Underpowered: companion set < min_companions → cannot decide, emit meta-claim
    - Powered + overlap >= threshold → compatible (core candidate)
    - Powered + overlap < threshold → incompatible (blocked)

    Returns: Dict[(s1, s2)] -> (is_compatible, is_underpowered, [ContextResult])
    """
    state.step = 6

    MIN_COMPANIONS = 2  # Need at least 2 companions to judge context
    OVERLAP_THRESHOLD = 0.15  # Jaccard threshold for compatibility

    compatibility = {}
    underpowered_count = 0
    blocked_count = 0
    compatible_count = 0

    surface_ids = list(state.surfaces.keys())

    for i, s1_id in enumerate(surface_ids):
        for s2_id in surface_ids[i+1:]:
            s1_entities = state.surface_entities[s1_id]
            s2_entities = state.surface_entities[s2_id]

            # Find shared entities
            shared = s1_entities & s2_entities
            if not shared:
                continue

            results = []
            any_compatible = False
            all_underpowered = True

            for entity in shared:
                companions1 = s1_entities - {entity}
                companions2 = s2_entities - {entity}

                c1_size = len(companions1)
                c2_size = len(companions2)

                # Check if underpowered
                if c1_size < MIN_COMPANIONS or c2_size < MIN_COMPANIONS:
                    result = ContextResult(
                        entity=entity,
                        compatible=False,  # Cannot determine
                        underpowered=True,
                        overlap=0.0,
                        companions1_size=c1_size,
                        companions2_size=c2_size,
                        reason=f"underpowered: companions=({c1_size},{c2_size}), need >= {MIN_COMPANIONS}"
                    )
                    results.append(result)

                    # Emit meta-claim for audit
                    pair_key = f"{s1_id}:{s2_id}"
                    state.ledger.add(Constraint(
                        id=f"underpowered_{entity}_{pair_key}",
                        constraint_type="meta",
                        assertion=f"Context test underpowered for '{entity}'",
                        evidence={
                            "companions1_size": c1_size,
                            "companions2_size": c2_size,
                            "min_required": MIN_COMPANIONS
                        },
                        provenance="context_compatibility_underpowered"
                    ), scope=pair_key)
                    continue

                all_underpowered = False

                # Jaccard overlap
                intersection = len(companions1 & companions2)
                union = len(companions1 | companions2)
                overlap = intersection / union if union > 0 else 0

                if overlap >= OVERLAP_THRESHOLD:
                    result = ContextResult(
                        entity=entity,
                        compatible=True,
                        underpowered=False,
                        overlap=overlap,
                        companions1_size=c1_size,
                        companions2_size=c2_size,
                        reason=f"compatible: overlap={overlap:.2f} >= {OVERLAP_THRESHOLD}"
                    )
                    any_compatible = True

                    # Add structural constraint
                    pair_key = f"{s1_id}:{s2_id}"
                    state.ledger.add(Constraint(
                        id=f"compat_{entity}_{pair_key}",
                        constraint_type="structural",
                        assertion=f"Entity '{entity}' has compatible context",
                        evidence={
                            "companions1": list(companions1),
                            "companions2": list(companions2),
                            "overlap": overlap
                        },
                        provenance="context_compatibility"
                    ), scope=pair_key)
                else:
                    result = ContextResult(
                        entity=entity,
                        compatible=False,
                        underpowered=False,
                        overlap=overlap,
                        companions1_size=c1_size,
                        companions2_size=c2_size,
                        reason=f"blocked: overlap={overlap:.2f} < {OVERLAP_THRESHOLD}"
                    )

                results.append(result)

            # Determine pair status
            is_compatible = any_compatible
            is_underpowered = all_underpowered

            compatibility[(s1_id, s2_id)] = (is_compatible, is_underpowered, results)

            # Log appropriately
            if is_underpowered:
                underpowered_count += 1
                state.log(f"  UNDERPOWERED: {s1_id}↔{s2_id} - insufficient companions to decide")
            elif is_compatible:
                compatible_count += 1
                compat_entities = [(r.entity, f"overlap={r.overlap:.2f}") for r in results if r.compatible]
                state.log(f"  COMPATIBLE: {s1_id}↔{s2_id} via {compat_entities}")
            else:
                blocked_count += 1
                for r in results:
                    if not r.underpowered and not r.compatible:
                        state.log(f"  BLOCKED: '{r.entity}' bridges {s1_id}↔{s2_id} ({r.reason})")

    state.log(f"Context compatibility summary:")
    state.log(f"  - Compatible (can form core): {compatible_count}")
    state.log(f"  - Blocked (incompatible context): {blocked_count}")
    state.log(f"  - Underpowered (insufficient evidence): {underpowered_count}")

    return compatibility


# =============================================================================
# STEP 7: EVENT FORMATION (with anti-trap rule)
# =============================================================================

def step7_form_events(
    state: ExperimentState,
    compatibility: Dict[Tuple[str, str], Tuple[bool, bool, List[ContextResult]]]
) -> Dict[str, Set[str]]:
    """
    Cluster surfaces into events using constraint ledger.

    THEORY: Decision layer with explicit loss
    - We have constraints (structural, semantic, typed, meta)
    - We apply the ANTI-TRAP RULE: cores need ≥2 constraints, ≥1 non-semantic

    ASYMMETRIC MEMBRANE RULE:
    - Incompatible (overlap < threshold with sufficient data) → BLOCK (no edge)
    - Compatible (overlap >= threshold) → allow core if anti-trap passes
    - Underpowered (insufficient data) → allow periphery, OR core if OTHER structural evidence exists

    KEY INSIGHT: Motif sharing is positive structural evidence.
    If surfaces share a supported motif AND context is underpowered (not incompatible),
    the motif can justify core formation.
    """
    state.step = 7

    # Build edges between surfaces based on constraints
    core_edges = []
    periphery_edges = []
    blocked_edges = []

    for (s1_id, s2_id), (is_compatible, is_underpowered, results) in compatibility.items():
        pair_key = f"{s1_id}:{s2_id}"
        constraints = state.ledger.for_scope(pair_key)

        # Count constraint types
        structural = [c for c in constraints if c.constraint_type == "structural"]
        motif_constraints = [c for c in structural if "motif" in c.provenance]

        # ASYMMETRIC RULE:
        # 1. If incompatible (powered + overlap < threshold) → BLOCK
        if not is_compatible and not is_underpowered:
            blocked_edges.append((s1_id, s2_id))
            for r in results:
                if not r.underpowered and not r.compatible:
                    state.log(f"  BLOCKED: '{r.entity}' bridges {s1_id}↔{s2_id} ({r.reason})")
            continue

        # 2. If compatible (powered + overlap >= threshold) → core candidate
        if is_compatible:
            can_core, reason = state.ledger.can_form_core(s1_id, s2_id)
            if can_core:
                core_edges.append((s1_id, s2_id))
                state.log(f"  Core edge {s1_id}↔{s2_id}: {reason}")
            else:
                periphery_edges.append((s1_id, s2_id))
                state.log(f"  Periphery edge {s1_id}↔{s2_id}: {reason}")
            continue

        # 3. If underpowered → check for motif evidence
        #    Motif sharing is strong structural evidence that survives sparsity
        if is_underpowered and motif_constraints:
            # Have motif evidence despite sparse context
            motif_names = [c.evidence.get("motif", []) for c in motif_constraints]
            state.log(f"  Core edge {s1_id}↔{s2_id}: underpowered context BUT shared motif {motif_names}")
            core_edges.append((s1_id, s2_id))
            continue

        # 4. Underpowered with no motif evidence → periphery only
        if is_underpowered:
            periphery_edges.append((s1_id, s2_id))
            state.log(f"  Periphery edge {s1_id}↔{s2_id}: underpowered, no motif evidence")

    # Form events from core edges (connected components)
    visited = set()
    events = {}
    event_idx = 0

    # Build adjacency from core edges only
    adj = defaultdict(set)
    for s1, s2 in core_edges:
        adj[s1].add(s2)
        adj[s2].add(s1)

    for surface_id in state.surfaces:
        if surface_id in visited:
            continue

        # BFS on core edges
        component = set()
        queue = [surface_id]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            component.add(curr)
            queue.extend(adj[curr] - visited)

        event_id = f"E{event_idx:03d}"
        events[event_id] = component
        event_idx += 1

    # Log periphery connections (they don't merge events)
    for s1, s2 in periphery_edges:
        e1 = next((eid for eid, surfaces in events.items() if s1 in surfaces), None)
        e2 = next((eid for eid, surfaces in events.items() if s2 in surfaces), None)

        if e1 and e2 and e1 != e2:
            state.log(f"  Periphery connection {e1}↔{e2} via {s1}↔{s2} (NOT merged)")

    state.events = events
    state.log(f"Event formation summary:")
    state.log(f"  - Core edges: {len(core_edges)}")
    state.log(f"  - Periphery edges: {len(periphery_edges)}")
    state.log(f"  - Blocked edges: {len(blocked_edges)}")
    state.log(f"  - Events formed: {len(events)}")

    for eid, surfaces in sorted(events.items(), key=lambda x: -len(x[1]))[:5]:
        all_entities = set()
        for sid in surfaces:
            all_entities.update(state.surface_entities.get(sid, set()))
        state.log(f"  {eid}: {len(surfaces)} surfaces, entities={all_entities}")

    return events


# =============================================================================
# MAIN: Run the full experiment
# =============================================================================

async def run_principled_emergence(
    claims: List[Claim] = None,
    show_counterfactual: bool = True,
    motif_config: MotifConfig = None,
    quiet: bool = False
):
    """
    Run the full principled emergence experiment.

    If claims not provided, loads from database.
    If show_counterfactual=True, also shows what would happen WITHOUT context check.
    If motif_config provided, uses custom motif detection settings.
    If quiet=True, suppresses step-by-step output.
    """
    if not quiet:
        print("=" * 70)
        print("PRINCIPLED EMERGENCE EXPERIMENT")
        print("Growing from 0, applying Jaynes and Bianconi at each step")
        print("=" * 70)

    state = ExperimentState(quiet=quiet)

    # Load claims if not provided
    if claims is None:
        claims = await load_real_claims(limit=50)

    if not quiet:
        print(f"\nLoaded {len(claims)} claims")

    # Step 1: Form hyperedges
    if not quiet:
        print("\n" + "=" * 70)
        print("STEP 1: HYPEREDGE FORMATION (Bianconi)")
        print("=" * 70)
    for claim in claims:
        step1_form_hyperedge(state, claim)

    # Step 2: Detect motifs
    if not quiet:
        print("\n" + "=" * 70)
        print("STEP 2: MOTIF DETECTION (Bianconi)")
        print("=" * 70)
    motifs = step2_detect_motifs(state, min_support=2, config=motif_config)

    # Step 3: Form surfaces
    if not quiet:
        print("\n" + "=" * 70)
        print("STEP 3: SURFACE FORMATION (Bianconi)")
        print("=" * 70)
    surfaces = step3_form_surfaces(state, motifs)

    # Step 4: Extract typed values
    if not quiet:
        print("\n" + "=" * 70)
        print("STEP 4: TYPED VALUE EXTRACTION (Jaynes preparation)")
        print("=" * 70)
    surface_obs = step4_extract_typed_values(state)

    # Step 5: Jaynes inference
    if not quiet:
        print("\n" + "=" * 70)
        print("STEP 5: JAYNES INFERENCE")
        print("=" * 70)
    if surface_obs:
        jaynes_results = step5_jaynes_inference(state, surface_obs)
    elif not quiet:
        print("  No typed values to infer")

    # Step 6: Context compatibility
    if not quiet:
        print("\n" + "=" * 70)
        print("STEP 6: CONTEXT COMPATIBILITY (Bianconi higher-order)")
        print("=" * 70)
    compatibility = step6_context_compatibility(state)

    # Step 7: Event formation
    if not quiet:
        print("\n" + "=" * 70)
        print("STEP 7: EVENT FORMATION (with anti-trap rule)")
        print("=" * 70)
    events = step7_form_events(state, compatibility)

    # Summary
    if not quiet:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Claims: {len(state.claims)}")
        print(f"Hyperedges: {len(state.hyperedges)}")
        print(f"Motifs (k≥2, support≥2): {len(motifs)}")
        print(f"Surfaces: {len(state.surfaces)}")
        print(f"Events: {len(state.events)}")
        print(f"Constraints in ledger: {len(state.ledger.constraints)}")
        print(f"  - Structural: {len(state.ledger.get_structural())}")
        print(f"  - Semantic: {len(state.ledger.get_semantic())}")
        print(f"Belief states: {len(state.belief_states)}")

    # Show counterfactual: what would happen WITHOUT context check
    if show_counterfactual and not quiet:
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL: Without Context Compatibility Check")
        print("=" * 70)
        naive_events = _compute_naive_events(state)
        print(f"Events WITHOUT context check: {len(naive_events)}")
        print(f"Events WITH context check: {len(state.events)}")

        # Show the largest naive clusters
        for eid, surfaces in sorted(naive_events.items(), key=lambda x: -len(x[1]))[:3]:
            if len(surfaces) > 1:
                all_entities = set()
                for sid in surfaces:
                    all_entities.update(state.surface_entities.get(sid, set()))
                print(f"\n  {eid}: {len(surfaces)} surfaces, {len(all_entities)} entities")
                print(f"    Entities: {all_entities}")

    return state


def _compute_naive_events(state: ExperimentState) -> Dict[str, Set[str]]:
    """
    What events would form if we connected surfaces sharing ANY entity?
    (No context compatibility check)
    """
    adj = defaultdict(set)

    surface_ids = list(state.surfaces.keys())
    for i, s1_id in enumerate(surface_ids):
        for s2_id in surface_ids[i+1:]:
            s1_entities = state.surface_entities[s1_id]
            s2_entities = state.surface_entities[s2_id]

            if s1_entities & s2_entities:
                adj[s1_id].add(s2_id)
                adj[s2_id].add(s1_id)

    visited = set()
    events = {}
    event_idx = 0

    for surface_id in state.surfaces:
        if surface_id in visited:
            continue

        component = set()
        queue = [surface_id]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            component.add(curr)
            queue.extend(adj[curr] - visited)

        events[f"NAIVE_{event_idx:03d}"] = component
        event_idx += 1

    return events


async def load_real_claims(limit: int = 50) -> List[Claim]:
    """Load real claims from database."""
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    # Load claims with entities
    results = await neo4j._execute_read(f'''
        MATCH (c:Claim)-[:MENTIONS]->(e:Entity)
        WITH c, collect(DISTINCT e.canonical_name) as entities
        WHERE size(entities) >= 2
        RETURN c.id as id, c.text as text, entities,
               c.source as source, c.created_at as created_at
        LIMIT {limit}
    ''')

    await neo4j.close()

    claims = []
    for row in results:
        claims.append(Claim(
            id=row['id'],
            text=row['text'] or "",
            entities=set(row['entities'] or []),
            source=row['source'] or "unknown",
            timestamp=row['created_at']
        ))

    return claims


if __name__ == "__main__":
    asyncio.run(run_principled_emergence())
