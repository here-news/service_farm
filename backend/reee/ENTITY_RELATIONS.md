# Entity Relations as Aboutness Signal

## Problem

Surfaces with different anchor framings don't connect:
- Surface A: `{Do Kwon, Southern District of NY}`
- Surface B: `{Terraform Labs}`

No anchor overlap → no aboutness edge → fragmentation.

But claims **state** the relation: "Terraform Labs was co-founded by Do Kwon"

## Solution

Extract entity-to-entity relations from claims as **typed propositions**. Use the resulting entity graph as a **bounded additional signal** for aboutness.

## Core Principle: Relations Are Propositions

Each extracted relation is a typed proposition:

```
relation(subject_entity, relation_type, object_entity)
```

Example:
```
relation(Do Kwon, FOUNDED, Terraform Labs)
```

This proposition:
- Has provenance (source claim, extractor version, timestamp)
- Can be corroborated (multiple claims assert the same relation)
- Can conflict (claim A says FOUNDED, claim B says NOT_FOUNDED)
- Forms an **EntityRelationSurface** via identity linking (CONFIRMS/CONFLICTS)

Relations are NOT bare edges in a graph. They are epistemic objects with the same status as other propositions.

### Jaynes-Style Belief Over Relations

A relation proposition has a **belief state**, not a binary truth value:

```python
@dataclass
class RelationBeliefState:
    """
    Bayesian belief over whether relation(A, R, B) holds.

    Updated via Jaynes-style evidence accumulation:
    - Multiple claims asserting same relation → higher confidence
    - Conflicting claims → posterior reflects uncertainty
    - No evidence → prior (usually skeptical)
    """
    relation: str  # e.g., "FOUNDED"
    subject_id: str
    object_id: str

    # Posterior distribution over {asserted, negated, uncertain}
    # NOT a single float - we need to distinguish:
    # - "probably true" vs "probably false" vs "no idea"
    p_asserted: float   # P(relation holds | evidence)
    p_negated: float    # P(relation does not hold | evidence)
    p_uncertain: float  # P(unknown | evidence) - hedged/ambiguous

    # Evidence summary
    asserting_claims: List[str]  # Claims that assert this relation
    negating_claims: List[str]   # Claims that deny this relation
    uncertain_claims: List[str]  # Claims with hedged language

    # Source quality weighting (not all claims equal)
    asserting_weight: float  # Sum of source credibility for assertions
    negating_weight: float   # Sum of source credibility for negations

    @property
    def entropy(self) -> float:
        """Shannon entropy over the posterior distribution."""
        from math import log2
        h = 0.0
        for p in [self.p_asserted, self.p_negated, self.p_uncertain]:
            if p > 0:
                h -= p * log2(p)
        return h

    @property
    def is_safe_to_use(self) -> bool:
        """Should this relation be used as aboutness binder?"""
        # Require:
        # 1. High probability of assertion
        # 2. Low probability of negation (not just count, but weighted)
        # 3. Low entropy (confident posterior)
        # 4. Multiple independent sources (not single extraction)
        return (
            self.p_asserted > 0.8 and
            self.p_negated < 0.1 and  # Even one strong negation matters
            self.entropy < 0.5 and
            len(self.asserting_claims) >= 2  # Corroboration required
        )
```

**Key principle**: A relation only becomes a safe aboutness binder when:
1. High P(asserted) from multiple independent claims (corroboration)
2. Low P(negated) - even one strong negation blocks usage
3. Low entropy - the posterior is confident, not spread across states
4. Not just "many uncertain claims" - a pile of "allegedly" doesn't equal "asserted"

This prevents extraction errors from becoming sticky bridges - a single wrong extraction can be overridden by conflicting evidence.

### Unified Weaving: Relations Use Core Logic

Entity relations flow through the **same weaving pipeline** as world propositions:

```
Claim "Do Kwon founded Terraform Labs"
    ↓ extraction
RelationAssertion(Do Kwon, FOUNDED, Terraform Labs, polarity=asserted)
    ↓ identity linking (same as any proposition)
RelationSurface (groups claims asserting/denying this relation)
    ↓ belief state computation (Jaynes posterior)
RelationBeliefState (posterior, entropy, evidence)
    ↓ if is_safe_to_use
Entity graph edge (for aboutness binding)
```

This is NOT a separate system. It's the universal weaver applied to relation propositions:

- **L0**: Relation assertions extracted from claims (with provenance)
- **L2**: RelationSurfaces via identity linking (CONFIRMS, CONFLICTS, REFINES)
- **Belief**: Posterior over relation truth via typed belief state
- **Output**: Entity graph as a derived view with epistemic grounding

The same invariants apply:
- Append-only assertions
- Versioned parameters
- Replayable from (assertions, params@version)
- Conflicts surface as tensions, not silent overwrites

## Guardrails (Critical)

Entity relations are a sharp tool. Without constraints, they reintroduce percolation.

### 1. Only 1-hop relations

Use only direct relations (A → B), not transitive chains (A → B → C).

```python
# YES: Do Kwon --FOUNDED--> Terraform Labs
# NO:  Do Kwon --FOUNDED--> Terraform Labs --CREATED--> Luna --CRASHED--> ...
```

### 2. Only strong relation types

Start with high-precision relations that truly imply shared frame:

| Relation | Implies |
|----------|---------|
| FOUNDED / CO_FOUNDED | Person created organization |
| CEO_OF / LEADS | Person runs organization |
| SUBSIDIARY_OF / PARENT_OF | Org hierarchy |
| CREATED / TOKEN_OF | Org created asset |
| OWNED_BY / OWNS | Ownership |
| PART_OF | Structural membership |

**Avoid** as binders:
- MET_WITH, SPOKE_TO (discourse, not structural)
- MENTIONED, CRITICIZED (too weak)
- RELATED_TO (too vague)

### 3. Multi-signal requirement

"Related anchors" counts as **one signal**, not sufficient alone.

Still require at least one additional compatibility:
- **Incident view**: temporal overlap (within Δ days)
- **Case view**: strong semantic similarity OR shared non-hub entities

```python
# Aboutness edge requires:
#   (shared_anchor OR related_anchor)
#   AND (temporal_compatible OR semantic_score > 0.6)
#   AND signals_met >= 2
```

### 4. Relations are versioned and contestable

Entity relations are **derived claims** with:
- Provenance (source claim ID, extraction timestamp)
- Extractor version (model/prompt version)
- Confidence score
- Ability to conflict (multiple claims may assert different relations)

They should form **EntityRelationSurfaces** (identity over relation propositions).

### 5. Bridge-resistant event formation

Even with entity relations, do NOT use plain connected components.

The relation graph adds edges, but event formation must remain:
- Core/periphery based
- Require discriminative signal (not just any relation)
- Time-bounded for incident views

## Scale Policy (Explicit)

Entity relations are **primarily for case/saga views**, not incident views.

| Scale | Time Window | Entity Relations as Binder | Product Use |
|-------|-------------|---------------------------|-------------|
| **Incident** | Δ ≤ 3 days | Only with temporal overlap | Breaking news, daily digest |
| **Case** | Δ ≤ 30 days | Primary signal (1-hop) | Story tracker, inquiry scope |
| **Saga** | Δ ≤ 1 year | Multi-hop chains allowed | Long-form narrative |

### Policy Rules

1. **Incident view**: Relation-graph binding is DISABLED unless surfaces are time-compatible.
   - "Do Kwon sentencing" (Dec 11) and "Terraform collapse" (May 2022) do NOT merge at incident level.
   - Same-day surfaces about Do Kwon and Terraform Labs CAN merge if relation exists.

2. **Case view**: Relation-graph binding is the PRIMARY mechanism for aggregation.
   - Surfaces with related anchors form the same case regardless of exact timing.
   - Still require bridge-resistant clustering (not plain connected components).

3. **Saga view**: Reserved for editorial/narrative products, not automated weaving.

### Default Behavior

```python
def should_use_relation_binding(view_scale: str, time_delta_days: int) -> bool:
    if view_scale == 'incident':
        return time_delta_days <= 3  # Only if time-compatible
    elif view_scale == 'case':
        return True  # Primary signal
    elif view_scale == 'saga':
        return True  # Allow freely
    return False
```

## Relation Vocabulary (Minimal)

Start with 6 relation types, tiered by percolation risk:

### Tier 1: High-Precision (use freely as aboutness signal)

```python
TIER1_RELATIONS = {
    'FOUNDED',      # Person founded/co-founded Org
    'CEO_OF',       # Person leads/runs Org (current)
    'SUBSIDIARY_OF', # Org is subsidiary of parent Org
    'CREATED',      # Org/Person created product/token/asset
}
```

These are structural, stable, and rarely ambiguous.

### Tier 2: High-Risk Percolators (require extra gating)

```python
TIER2_RELATIONS = {
    'OWNS',         # Person/Org owns asset - can be broad
    'PART_OF',      # X is part of Y - too general without domain constraint
}
```

Tier 2 relations require:
- Higher confidence threshold (P(asserted) ≥ 0.9 vs 0.8 for Tier 1)
- Corroboration from ≥ 3 claims (vs 2 for Tier 1)
- **Domain scoping** (critical):

```python
TIER2_DOMAIN_CONSTRAINTS = {
    'OWNS': {
        'allowed_domains': ['corporate', 'financial'],
        'subject_types': ['PERSON', 'ORGANIZATION'],
        'object_types': ['ORGANIZATION', 'ASSET', 'PROPERTY'],
        # Reject: "owns the narrative", "owns the debate"
    },
    'PART_OF': {
        'allowed_domains': ['corporate', 'organizational'],
        'subject_types': ['ORGANIZATION', 'DIVISION'],
        'object_types': ['ORGANIZATION'],
        # Reject: "part of the problem", "part of history"
        # Accept: "X is a subsidiary of Y", "X division of Y"
    },
}
```

Without domain scoping, PART_OF and OWNS will percolate across politics/economy/culture.

### Polarity and Hedging

Each relation extraction must include:

```python
@dataclass
class RelationAssertion:
    subject_id: str           # Entity ID (not raw text)
    relation: str             # FOUNDED, CEO_OF, etc.
    object_id: str            # Entity ID (not raw text)

    polarity: Literal['asserted', 'negated', 'uncertain']
    confidence: float         # 0.0 - 1.0

    # Provenance
    source_claim_id: str
    extractor_version: str
    extracted_at: datetime

    # Hedge indicators detected
    hedge_markers: List[str]  # ["allegedly", "reportedly", etc.]
```

Hedge/negation markers that affect polarity:
- **Negated**: "not", "never", "denied", "no longer"
- **Uncertain**: "allegedly", "reportedly", "claimed to", "accused of", "may have"

Relations with `polarity != 'asserted'` should NOT be used as aboutness binders until corroborated.

## Extraction Strategy

### Fast Path (inline, high-precision)

Pattern matching over **recognized entity spans**, not raw tokens.

```python
def extract_relations(claim_text: str, entity_spans: List[EntitySpan]) -> List[RelationAssertion]:
    """
    Extract relations by finding patterns BETWEEN recognized entity spans.

    entity_spans contains positions and IDs of already-extracted entities.
    We look for relation-indicating text between pairs of entities.
    """
    relations = []

    for i, ent_a in enumerate(entity_spans):
        for ent_b in entity_spans[i+1:]:
            # Get text between the two entity spans
            between_text = claim_text[ent_a.end:ent_b.start].lower().strip()

            # Check for relation patterns
            if matches_pattern(between_text, FOUNDED_PATTERNS):
                relations.append(RelationAssertion(
                    subject_id=ent_a.entity_id,
                    relation='FOUNDED',
                    object_id=ent_b.entity_id,
                    polarity=detect_polarity(between_text),
                    confidence=0.8,
                    source_claim_id=claim.id,
                    ...
                ))

    return relations

# Patterns operate on text between entities, not raw tokens
FOUNDED_PATTERNS = [
    r',?\s*(co-?)?founded\s*',
    r',?\s*(co-?)?founder of\s*',
    r',?\s*who founded\s*',
]

CEO_PATTERNS = [
    r',?\s*CEO of\s*',
    r',?\s*chief executive of\s*',
    r',?\s*who leads\s*',
    r',?\s*head of\s*',
]
```

This approach:
- Works with multi-word entities, aliases, non-Latin names
- Relies on entity extraction as prerequisite (entities already identified)
- Avoids false matches on unrecognized text

**Directionality Warning**: Entity order in text does NOT reliably indicate subject/object roles.

```python
# "Terraform Labs, founded by Do Kwon" → subject=Do Kwon, object=Terraform Labs
# "Do Kwon founded Terraform Labs" → subject=Do Kwon, object=Terraform Labs
# Text order differs, but semantic roles are the same.

def extract_with_directionality(
    between_text: str,
    ent_a: EntitySpan,
    ent_b: EntitySpan,
    pattern: str,
) -> RelationAssertion:
    """
    Role assignment requires syntactic cues, not just position.

    If pattern contains passive markers ("founded by", "created by", "led by"),
    the SECOND entity is the subject (agent).

    If pattern is active ("founded", "created", "leads"),
    the FIRST entity is the subject.

    If ambiguous, emit UNDIRECTED candidate with lower confidence.
    """
    if is_passive_pattern(pattern):
        subject, obj = ent_b, ent_a
        confidence = 0.8
    elif is_active_pattern(pattern):
        subject, obj = ent_a, ent_b
        confidence = 0.8
    else:
        # Ambiguous - emit undirected with low confidence
        subject, obj = ent_a, ent_b  # arbitrary order
        confidence = 0.4  # Requires corroboration to become usable

    return RelationAssertion(
        subject_id=subject.entity_id,
        object_id=obj.entity_id,
        confidence=confidence,
        ...
    )
```

Low-confidence undirected extractions should NOT be used as aboutness binders until corroborated by a high-confidence directed extraction.

### Slow Path (async, LLM-based)

Background RelationWeaver for:
- Ambiguous patterns
- Implicit relations
- Relation confidence scoring

Output is stored as derived assertions, not raw L0 claims.

## Implementation Notes

1. Entity relations don't replace anchor overlap—they extend it
2. Relations must be bidirectional in the graph (FOUNDED implies FOUNDED_BY)
3. IDF penalty still applies: if an entity is in many relation edges, downweight
4. The relation graph is a **view**, not ground truth—it can be recomputed

## Example: Do Kwon Case

Claims containing relation statements:
- "Terraform Labs was co-founded by Do Kwon in 2018"
- "Do Kwon created two virtual currencies, TerraUSD and Luna"

Extracted relations:
```
Do Kwon --FOUNDED--> Terraform Labs
Do Kwon --CREATED--> TerraUSD
Do Kwon --CREATED--> Luna
Terraform Labs --CREATED--> Luna (if stated)
```

Effect on aboutness:
- Surface `{Do Kwon}` and Surface `{Terraform Labs}` are now 1-hop related
- If they also pass temporal gate and have semantic > 0.5, they can connect
- This enables case-level aggregation without polluting incident-level clustering

## Evaluation Criteria

Before using entity relations as aboutness binders, measure these metrics:

### 1. False Merge Rate

Compare clustering with and without relation signal:

```python
def false_merge_rate(
    events_with_relations: Dict[str, Event],
    events_without_relations: Dict[str, Event],
    ground_truth: Dict[str, str],  # claim_id -> legacy_event
) -> float:
    """
    What fraction of merges introduced by relations are incorrect?

    A merge is "false" if it combines claims from different ground truth events.
    """
    # Find merges that only exist with relations
    new_merges = find_new_merges(events_with_relations, events_without_relations)

    false_count = 0
    for merged_event in new_merges:
        claim_ids = get_claims(merged_event)
        gt_labels = {ground_truth[cid] for cid in claim_ids if cid in ground_truth}
        if len(gt_labels) > 1:
            false_count += 1

    return false_count / len(new_merges) if new_merges else 0.0
```

**Threshold**: False merge rate < 10%

### 2. Bridge Ratio

How many cross-topic connections does each relation create?

```python
def bridge_ratio(relation: RelationAssertion, surfaces: Dict[str, Surface]) -> float:
    """
    For a given relation (A -> B), how many surface pairs does it connect
    that have NO other binding signal (semantic, temporal, entity)?

    High bridge ratio = potential percolator.
    """
    a_surfaces = surfaces_with_anchor(relation.subject_id)
    b_surfaces = surfaces_with_anchor(relation.object_id)

    pure_bridges = 0
    for s1 in a_surfaces:
        for s2 in b_surfaces:
            if not has_other_binding_signal(s1, s2):
                pure_bridges += 1

    return pure_bridges / (len(a_surfaces) * len(b_surfaces))
```

**Threshold**: Bridge ratio < 0.3 per relation

### 3. Stability Under Shuffle

Does the relation-based clustering remain stable under claim order perturbation?

```python
def shuffle_stability(claims: List[Claim], n_runs: int = 10) -> float:
    """
    Run clustering N times with shuffled claim order.
    Measure Jaccard similarity of resulting event assignments.
    """
    assignments = []
    for _ in range(n_runs):
        shuffled = shuffle(claims)
        events = run_emergence_pipeline(shuffled)
        assignments.append(claim_to_event_map(events))

    return mean_pairwise_jaccard(assignments)
```

**Threshold**: Stability > 0.85

### 4. Cross-Topic Mixing Rate

What fraction of emerged events contain claims from multiple ground-truth topics?

```python
def cross_topic_mixing(
    events: Dict[str, Event],
    claim_to_topic: Dict[str, str],  # claim_id -> topic label
) -> float:
    """
    For each emerged event, what fraction of its claims come from
    a topic different from the dominant one?
    """
    total_claims = 0
    mixed_claims = 0

    for event in events.values():
        claims = get_claims(event)
        topics = [claim_to_topic[c] for c in claims if c in claim_to_topic]
        if not topics:
            continue

        dominant = Counter(topics).most_common(1)[0][0]
        for topic in topics:
            total_claims += 1
            if topic != dominant:
                mixed_claims += 1

    return mixed_claims / total_claims if total_claims else 0.0
```

**Threshold**: Cross-topic mixing < 5%

### 5. Relation Extraction Precision

For extracted relations, what fraction are correct when spot-checked?

```python
def relation_precision(
    extracted: List[RelationAssertion],
    sample_size: int = 50,
) -> float:
    """
    Sample extracted relations and verify correctness.
    Returns fraction that are true relations.
    """
    sample = random.sample(extracted, min(sample_size, len(extracted)))
    correct = sum(1 for r in sample if human_verify(r))
    return correct / len(sample)
```

**Threshold**: Precision > 90% for Tier 1 relations

### Evaluation Protocol

Before enabling relation-graph binding in production:

1. Run extraction on corpus, measure relation precision
2. Run emergence with relations OFF, measure baseline B³/purity
3. Run emergence with relations ON, measure:
   - B³ improvement (should increase recall)
   - False merge rate (should be < 10%)
   - Cross-topic mixing (should be < 5%)
   - Stability (should be > 0.85)
4. If any metric fails threshold, tighten relation constraints before enabling
