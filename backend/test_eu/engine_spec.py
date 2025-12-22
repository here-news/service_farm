"""
UNIVERSAL EPISTEMIC ENGINE - COMPLETE SPECIFICATION

This document presents the engine in full detail.
"""

print("""
╔══════════════════════════════════════════════════════════════════════╗
║           UNIVERSAL EPISTEMIC ENGINE - SPECIFICATION                  ║
╚══════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════
1. CORE CONCEPT
═══════════════════════════════════════════════════════════════════════

The engine transforms raw claims into structured knowledge:

    CLAIMS → EDGES → EVENTS → DISTILLED OUTPUT

Each stage uses the SAME underlying operation: measure epistemic affinity.


═══════════════════════════════════════════════════════════════════════
2. DATA STRUCTURES
═══════════════════════════════════════════════════════════════════════

CLAIM:
    id:           unique identifier
    text:         the assertion
    page_id:      source page
    entity_ids:   [list of entities mentioned]
    timestamp:    when published (if available)

EVENT:
    id:           unique identifier
    claim_ids:    [claims in this event]
    entity_surface: {all entities from all claims}
    embedding_centroid: average of claim embeddings
    mass:         sum of source credibilities
    sources:      {distinct source domains}

SOURCE CREDIBILITY (priors):
    bbc.com:        0.90
    reuters.com:    0.88
    theguardian.com: 0.85
    ...
    unknown:        0.50


═══════════════════════════════════════════════════════════════════════
3. THE CORE OPERATION: compute_affinity()
═══════════════════════════════════════════════════════════════════════

This is the SINGLE operation that powers everything.

    compute_affinity(A, B) → score ∈ [0, 1]

Where A and B can be:
    - claim, event    (should claim join event?)
    - event, event    (should events merge?)

SIGNALS COMBINED:

    ┌─────────────────────────────────────────────────────────┐
    │  SEMANTIC SIMILARITY (60% weight)                       │
    │                                                         │
    │  sim = cosine(embedding(A), embedding(B))               │
    │                                                         │
    │  Uses: text-embedding-3-small or similar                │
    │  Captures: topical relatedness                          │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │  ENTITY OVERLAP (25% weight)                            │
    │                                                         │
    │  For claim→event:                                       │
    │    overlap = |claim.entities ∩ event.entity_surface|    │
    │              / |claim.entities|                         │
    │                                                         │
    │  For event→event:                                       │
    │    jaccard = |A.entities ∩ B.entities|                  │
    │              / |A.entities ∪ B.entities|                │
    │                                                         │
    │  Captures: shared actors, locations, objects            │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │  ENTITY SPECIFICITY (15% weight)                        │
    │                                                         │
    │  specificity(entity) = 1 - freq(entity) / max_freq      │
    │                                                         │
    │  "Wang Fuk Court" (rare) → high specificity             │
    │  "Hong Kong" (common) → low specificity                 │
    │                                                         │
    │  Captures: precise vs vague overlap                     │
    └─────────────────────────────────────────────────────────┘

COMBINED SCORE:

    affinity = 0.60 × semantic
             + 0.25 × entity_overlap
             + 0.15 × entity_specificity


═══════════════════════════════════════════════════════════════════════
4. PHASE 1: CLAIM PROCESSING
═══════════════════════════════════════════════════════════════════════

For each incoming claim:

    ┌─────────────────────────────────────────────────────────┐
    │  for event in existing_events:                          │
    │      score = compute_affinity(claim, event)             │
    │                                                         │
    │  best_event = argmax(score)                             │
    │                                                         │
    │  if best_score >= THRESHOLD (0.45):                     │
    │      JOIN: add claim to best_event                      │
    │      - event.claim_ids.append(claim.id)                 │
    │      - event.entity_surface.update(claim.entities)      │
    │      - event.mass += source_credibility                 │
    │      - update event.embedding_centroid                  │
    │  else:                                                  │
    │      SEED: create new event from claim                  │
    └─────────────────────────────────────────────────────────┘

GROWTH DYNAMICS:

    As event grows:
    - entity_surface expands → more entities to match
    - mass increases → (optional) mass bonus in scoring
    - centroid stabilizes → better semantic anchor

    Result: larger events become "stickier"


═══════════════════════════════════════════════════════════════════════
5. PHASE 2: EVENT MERGE
═══════════════════════════════════════════════════════════════════════

Same operation, applied to events:

    ┌─────────────────────────────────────────────────────────┐
    │  for each pair (event_A, event_B):                      │
    │      score = compute_affinity(event_A, event_B)         │
    │                                                         │
    │  if score >= MERGE_THRESHOLD (0.50):                    │
    │      MERGE: combine B into A                            │
    │      - A.claim_ids.extend(B.claim_ids)                  │
    │      - A.entity_surface.update(B.entity_surface)        │
    │      - A.mass += B.mass                                 │
    │      - update A.embedding_centroid                      │
    │      - remove B from events                             │
    └─────────────────────────────────────────────────────────┘

WHY MERGE WORKS:

    Initial fragmentation happens because:
    - Early claims seed separate events
    - Claims with different entity subsets don't connect

    Merge fixes this because:
    - Event centroids are more stable than single claims
    - Entity surfaces are larger, more overlap likely
    - Same affinity logic catches what claim-level missed


═══════════════════════════════════════════════════════════════════════
6. PHASE 3: DISTILLATION
═══════════════════════════════════════════════════════════════════════

Each event produces a distilled output:

    ┌─────────────────────────────────────────────────────────┐
    │  DISTILLED EVENT                                        │
    │  ═══════════════                                        │
    │                                                         │
    │  QUANTITATIVE:                                          │
    │    mass:      sum of source credibilities               │
    │    claims:    count                                     │
    │    sources:   count of distinct domains                 │
    │    entities:  key entities by specificity               │
    │                                                         │
    │  QUALITATIVE:                                           │
    │    what_we_know:   top claims by credibility            │
    │    what_differs:   claims with differing values         │
    │    evolution:      if timestamps, show progression      │
    └─────────────────────────────────────────────────────────┘

HANDLING DIFFERS (same topic, different values):

    ┌─────────────────────────────────────────────────────────┐
    │  IF timestamps available:                               │
    │      current_state = latest claim                       │
    │      history = earlier claims                           │
    │                                                         │
    │  IF no timestamps:                                      │
    │      show range: "Death toll: 36-160 (evolving)"        │
    │      OR use claim order as proxy                        │
    │      OR mark as "uncertain/contested"                   │
    └─────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
7. EDGE TYPES
═══════════════════════════════════════════════════════════════════════

The engine recognizes three edge types:

    CORROBORATES:
        - High semantic similarity (>0.85)
        - Claims assert the same thing
        - Strengthens confidence

    DIFFERS:
        - Moderate similarity (0.5-0.85)
        - Shared entities
        - Same topic, different details
        - Could be UPDATE or CONTRADICTION
        - Resolved at distillation time

    INDEPENDENT:
        - Low similarity (<0.5)
        - No shared entities
        - Different topics
        - Claims don't connect


═══════════════════════════════════════════════════════════════════════
8. JAYNESIAN QUANTITIES
═══════════════════════════════════════════════════════════════════════

Each event has Bayesian properties:

    MASS (evidence weight):
        mass = Σ credibility_i × (1 + corroboration_bonus)

        - More credible sources → higher mass
        - Corroboration multiplies confidence
        - Represents total evidence strength

    COHERENCE (internal agreement):
        coherence = corroborates / (corroborates + differs)

        - 1.0 = all claims agree
        - 0.5 = half agree, half differ
        - 0.0 = total disagreement

    ENTROPY (uncertainty):
        H = -Σ p_i × log(p_i) over contested values

        - 0 = certainty (one value)
        - High = uncertainty (multiple values)
        - Weighted by source credibility


═══════════════════════════════════════════════════════════════════════
9. THE COMPLETE PIPELINE
═══════════════════════════════════════════════════════════════════════

    ┌──────────────┐
    │   CLAIMS     │  1215 raw claims from various sources
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   PROCESS    │  compute_affinity(claim, events)
    │              │  → JOIN or SEED
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   EVENTS     │  231 initial events (some fragmented)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   MERGE      │  compute_affinity(event, event)
    │              │  → same rules, higher level
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   EVENTS     │  221 merged events
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   DISTILL    │  Extract: mass, coherence, what_we_know
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   OUTPUT     │  Structured events with Jaynesian quantities
    └──────────────┘


═══════════════════════════════════════════════════════════════════════
10. KEY PROPERTIES
═══════════════════════════════════════════════════════════════════════

    FRACTAL:     Same compute_affinity() at all levels

    SCALABLE:    Entity routing reduces O(N²) to O(N × avg_entities)

    INCREMENTAL: Claims can be processed as they arrive

    UNIVERSAL:   No domain-specific patterns
                 Works for any event type

    JAYNESIAN:   Proper uncertainty quantification
                 Credibility weighting
                 Evidence accumulation


═══════════════════════════════════════════════════════════════════════
11. EXAMPLE: HONG KONG FIRE
═══════════════════════════════════════════════════════════════════════

INPUT:  77 fire-related claims from 24 sources
        Death toll mentions: 36, 44, 100, 128, 160

PROCESS:
    Claim 1: "36 killed..." → SEED E1
    Claim 2: "Firefighters..." → SEED E2 (different entities)
    Claim 3: "Fire in Tai Po..." → JOIN E2 (semantic + entity match)
    ...

MERGE:
    E2 → E1 (score=0.52)
    E3 → E1 (score=0.63)

OUTPUT:
    ┌─────────────────────────────────────────────────────────┐
    │  HONG KONG FIRE EVENT                                   │
    │  ════════════════════                                   │
    │                                                         │
    │  Claims:    140                                         │
    │  Mass:      81.6                                        │
    │  Sources:   24                                          │
    │  Coherence: 86%                                         │
    │                                                         │
    │  WHAT WE KNOW:                                          │
    │    - Fire at Wang Fuk Court, Tai Po, Hong Kong          │
    │    - Multiple high-rise buildings affected              │
    │    - 279 reported missing at peak                       │
    │                                                         │
    │  EVOLVING:                                              │
    │    Death toll: 36 → 44 → 100 → 128 → 160               │
    │    Current: 160 (latest report)                         │
    │                                                         │
    │  KEY ENTITIES:                                          │
    │    Wang Fuk Court, Tai Po, John Lee, Xi Jinping,        │
    │    Fire Services Department, Transport Department       │
    └─────────────────────────────────────────────────────────┘

""")
