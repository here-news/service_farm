"""
BACKEND MIGRATION PLAN: Current System → UEE Architecture

This document outlines how to evolve the current backend to enable
the Universal Epistemic Engine vision while preserving working components.
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BACKEND MIGRATION PLAN                                    ║
║                    Current System → UEE Architecture                         ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
1. CURRENT ARCHITECTURE (What We Have)
═══════════════════════════════════════════════════════════════════════════════

    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    │  URL submitted                                                        │
    │       │                                                               │
    │       ▼                                                               │
    │  ┌──────────────────┐     ┌──────────────────┐                       │
    │  │ ExtractionWorker │ ──→ │ queue:semantic   │                       │
    │  │                  │     │ :high            │                       │
    │  │ URL → Page       │     │ (page_id)        │                       │
    │  └──────────────────┘     └────────┬─────────┘                       │
    │                                    │                                  │
    │                                    ▼                                  │
    │  ┌──────────────────────────────────────────────────────────────┐    │
    │  │                    KnowledgeWorker                            │    │
    │  │                                                               │    │
    │  │  STAGE 0: Publisher identification                            │    │
    │  │  STAGE 1: LLM extraction (claims, entities, relationships)    │    │
    │  │  STAGE 2: Entity identification (local + Wikidata)            │    │
    │  │  STAGE 3: Entity creation + QID update                        │    │
    │  │  STAGE 4: Linking (Page→Claim, Claim→Entity)                  │    │
    │  │  STAGE 4e: Page embedding generation                          │    │
    │  │  STAGE 5: Integrity check                                     │    │
    │  │                                                               │    │
    │  │  Output: page_id → queue:event:high                           │    │
    │  └───────────────────────────────────┬──────────────────────────┘    │
    │                                      │                                │
    │                                      ▼                                │
    │  ┌──────────────────────────────────────────────────────────────┐    │
    │  │                      EventWorker                              │    │
    │  │                                                               │    │
    │  │  Receives: page_id                                            │    │
    │  │  Fetches: ALL claims for page                                 │    │
    │  │  Routes: ALL claims to ONE event (page-level routing)         │    │
    │  │                                                               │    │
    │  │  Scoring: 0.40 × entity_jaccard + 0.60 × semantic_cosine      │    │
    │  │  Threshold: 0.20 (join existing) or create new                │    │
    │  └───────────────────────────────────┬──────────────────────────┘    │
    │                                      │                                │
    │                                      ▼                                │
    │  ┌──────────────────────────────────────────────────────────────┐    │
    │  │                     LiveEventPool                             │    │
    │  │                                                               │    │
    │  │  - Maintains active events in memory                          │    │
    │  │  - Routes pages to events                                     │    │
    │  │  - Bootstraps new events                                      │    │
    │  │  - Runs metabolism cycles (hourly)                            │    │
    │  │  - Hibernates dormant events                                  │    │
    │  │                                                               │    │
    │  │  Uses: ClaimTopologyService for Bayesian analysis             │    │
    │  └──────────────────────────────────────────────────────────────┘    │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘

    CURRENT ISSUES:
    ───────────────
    • Page-level routing: ALL claims go to ONE event
    • 50% orphan claims (claims that don't fit get lost)
    • Duplicate events (Trump vs BBC appears twice)
    • No event merging (similar events stay separate)
    • No claim-level affinity scoring
    • No Jaynesian state on events (mass, heat, entropy)


═══════════════════════════════════════════════════════════════════════════════
2. TARGET ARCHITECTURE (UEE)
═══════════════════════════════════════════════════════════════════════════════

    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    │  URL submitted          Community UI                                  │
    │       │                     │                                         │
    │       ▼                     ▼                                         │
    │  ┌──────────────────┐  ┌──────────────────┐                          │
    │  │ ExtractionWorker │  │ ClaimSubmission  │ ← NEW: Direct claim      │
    │  │ (unchanged)      │  │ API              │    input from users      │
    │  └────────┬─────────┘  └────────┬─────────┘                          │
    │           │                     │                                     │
    │           ▼                     │                                     │
    │  ┌──────────────────┐           │                                     │
    │  │ KnowledgeWorker  │           │                                     │
    │  │ (minor change)   │───────────┤                                     │
    │  │                  │           │                                     │
    │  │ NEW: Emits to    │           │                                     │
    │  │ claim pool, not  │           │                                     │
    │  │ event queue      │           │                                     │
    │  └──────────────────┘           │                                     │
    │                                 │                                     │
    │                                 ▼                                     │
    │  ┌──────────────────────────────────────────────────────────────┐    │
    │  │                      CLAIM POOL                               │    │
    │  │                                                               │    │
    │  │  queue:claims:pending                                         │    │
    │  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐                       │    │
    │  │  │ c1  │ c2  │ c3  │ c4  │ c5  │ ... │                       │    │
    │  │  └─────┴─────┴─────┴─────┴─────┴─────┘                       │    │
    │  │                                                               │    │
    │  │  Each claim has:                                              │    │
    │  │  - id, text, embedding, entity_ids                            │    │
    │  │  - page_id (source), submitter_id (if user)                   │    │
    │  │  - energy_stake (if from community)                           │    │
    │  └───────────────────────────────┬──────────────────────────────┘    │
    │                                  │                                    │
    │                                  ▼                                    │
    │  ┌──────────────────────────────────────────────────────────────┐    │
    │  │                       UEE WORKER                              │    │
    │  │                                                               │    │
    │  │  For each claim:                                              │    │
    │  │  1. compute_affinity(claim, all_active_events)                │    │
    │  │     - 0.60 × semantic + 0.25 × entity_overlap                 │    │
    │  │     + 0.15 × entity_specificity                               │    │
    │  │  2. If max_affinity > 0.45: JOIN event                        │    │
    │  │     Else: SEED new event                                      │    │
    │  │  3. Update event state: mass, heat, entity_surface            │    │
    │  │  4. Compute edge type vs existing claims (topology)           │    │
    │  │  5. Compute REWARD for submitter (if applicable)              │    │
    │  │                                                               │    │
    │  │  Periodic tasks:                                              │    │
    │  │  - merge_events() using same affinity rules                   │    │
    │  │  - decay_heat() based on time                                 │    │
    │  │  - compute_entropy() from contradictions                      │    │
    │  │  - hibernate dormant events                                   │    │
    │  └───────────────────────────────┬──────────────────────────────┘    │
    │                                  │                                    │
    │                                  ▼                                    │
    │  ┌──────────────────────────────────────────────────────────────┐    │
    │  │                    LIVE EVENT POOL                            │    │
    │  │                    (Enhanced)                                 │    │
    │  │                                                               │    │
    │  │  Each event now has Jaynesian state:                          │    │
    │  │  - mass: Σ source_credibility                                 │    │
    │  │  - heat: Σ exp(-λt) for recent claims                         │    │
    │  │  - entropy: H(contested values)                               │    │
    │  │  - coherence: corroborates / total_edges                      │    │
    │  │  - entity_surface: set of all entity_ids                      │    │
    │  │  - embedding_centroid: mean of claim embeddings               │    │
    │  │                                                               │    │
    │  │  display_score = mass × heat × log(sources) × (1-entropy)     │    │
    │  └───────────────────────────────┬──────────────────────────────┘    │
    │                                  │                                    │
    │                                  ▼                                    │
    │                          DISPLAY LAYER                                │
    │                          (ranked by display_score)                    │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
3. WHAT STAYS, WHAT CHANGES
═══════════════════════════════════════════════════════════════════════════════

    KEEP AS-IS:
    ───────────
    ✓ ExtractionWorker - URL → Page extraction works well
    ✓ KnowledgeWorker STAGES 0-4 - Entity/claim extraction works well
    ✓ ClaimTopologyService - Bayesian analysis (reuse in UEE)
    ✓ Neo4j graph structure (Page, Claim, Entity nodes)
    ✓ PostgreSQL storage (page content, embeddings)
    ✓ LiveEvent.examine() logic - claim examination
    ✓ Redis queues infrastructure

    MODIFY:
    ───────
    ○ KnowledgeWorker output:
      Current: enqueue page_id to queue:event:high
      New: enqueue EACH claim_id to queue:claims:pending

    ○ LiveEventPool.route_page_claims():
      Current: routes all page claims together
      New: route_claim() - routes ONE claim at a time

    ○ LiveEvent state:
      Current: claims list, last_update, narrative
      New: + mass, heat, entropy, coherence, entity_surface, embedding_centroid

    REPLACE:
    ────────
    ✗ EventWorker → UEEWorker
      - Claim-level processing instead of page-level
      - compute_affinity() as core operation
      - Reward computation for community claims

    ADD NEW:
    ────────
    + ClaimPool service (Redis-backed claim queue)
    + compute_affinity() function (the fractal core)
    + compute_event_affinity() for event merging
    + merge_events() periodic task
    + Jaynesian state updates on events
    + display_score computation


═══════════════════════════════════════════════════════════════════════════════
4. TOPOLOGY INTEGRATION
═══════════════════════════════════════════════════════════════════════════════

    CURRENT ClaimTopologyService:
    ─────────────────────────────
    - Generates claim embeddings
    - Computes similarity network
    - Classifies relations: corroborates, contradicts, updates, complements
    - Computes Bayesian posteriors
    - Detects contradictions
    - Tracks superseded_by chains

    HOW IT INTEGRATES WITH UEE:
    ───────────────────────────

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  UEE Worker receives claim                                          │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │ STEP 1: Route claim using compute_affinity()                 │   │
    │  │                                                              │   │
    │  │ affinity = 0.60 × semantic + 0.25 × entity + 0.15 × spec     │   │
    │  │                                                              │   │
    │  │ This uses embeddings already (semantic component)            │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │ STEP 2: After joining event, compute edges to existing claims│   │
    │  │                                                              │   │
    │  │ Use ClaimTopologyService.classify_relations() to determine:  │   │
    │  │ - CORROBORATES (same fact, different source)                 │   │
    │  │ - DIFFERS (different value - resolve later with timestamps)  │   │
    │  │ - COMPLEMENTS (related but different aspect)                 │   │
    │  │                                                              │   │
    │  │ Store edges in Neo4j:                                        │   │
    │  │ (Claim)-[:CORROBORATES]->(Claim)                             │   │
    │  │ (Claim)-[:DIFFERS]->(Claim)                                  │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │ STEP 3: Update event Jaynesian state                         │   │
    │  │                                                              │   │
    │  │ mass += source_credibility(claim)                            │   │
    │  │ heat += 1.0 (recency)                                        │   │
    │  │ coherence = corroborates / (corroborates + differs)          │   │
    │  │ entropy = compute from contested values                       │   │
    │  │                                                              │   │
    │  │ Reuse ClaimTopologyService.extract_numbers() for death tolls │   │
    │  │ Reuse ClaimTopologyService.compute_posterior() for priors    │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    KEY INSIGHT:
    ─────────────
    ClaimTopologyService already does most of what UEE needs:
    - Embeddings → semantic affinity
    - Relation classification → edge types
    - Bayesian posteriors → claim plausibility
    - Contradiction detection → entropy computation

    We're not replacing it - we're LIFTING it to event level:
    - Current: runs WITHIN event (after claims assigned)
    - UEE: runs DURING routing (to compute affinity) + AFTER (to update state)


═══════════════════════════════════════════════════════════════════════════════
5. IMPLEMENTATION PHASES
═══════════════════════════════════════════════════════════════════════════════

    ╔═══════════════════════════════════════════════════════════════════════╗
    ║ PHASE 1: Claim Pool + Claim-Level Routing (Week 1-2)                  ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║ Goal: Route claims individually, not by page                          ║
    ║                                                                       ║
    ║ Tasks:                                                                ║
    ║ 1.1 Create ClaimPool service (Redis queue wrapper)                    ║
    ║     - queue:claims:pending                                            ║
    ║     - enqueue(claim_id, metadata)                                     ║
    ║     - dequeue() → claim_id                                            ║
    ║                                                                       ║
    ║ 1.2 Modify KnowledgeWorker output                                     ║
    ║     - After creating claims, enqueue EACH claim_id                    ║
    ║     - Keep page_id in claim metadata for source tracking              ║
    ║                                                                       ║
    ║ 1.3 Create UEEWorker (basic version)                                  ║
    ║     - Dequeue from claim pool                                         ║
    ║     - Fetch claim with embedding + entities                           ║
    ║     - compute_affinity() against active events                        ║
    ║     - Route: JOIN or SEED                                             ║
    ║                                                                       ║
    ║ 1.4 Update LiveEventPool for claim-level routing                      ║
    ║     - route_claim(claim) instead of route_page_claims(claims)         ║
    ║     - Accumulate entity_surface per event                             ║
    ║     - Maintain embedding_centroid (running mean)                      ║
    ║                                                                       ║
    ║ Validation: Run on existing data, compare event assignments           ║
    ╚═══════════════════════════════════════════════════════════════════════╝

    ╔═══════════════════════════════════════════════════════════════════════╗
    ║ PHASE 2: Jaynesian State + Event Merging (Week 3-4)                   ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║ Goal: Add mass/heat/entropy and automatic event merging               ║
    ║                                                                       ║
    ║ Tasks:                                                                ║
    ║ 2.1 Add Jaynesian state to Event model                                ║
    ║     - mass: float (evidence weight)                                   ║
    ║     - heat: float (recency score)                                     ║
    ║     - entropy: float (uncertainty)                                    ║
    ║     - coherence: float (agreement ratio)                              ║
    ║     - sources: Set[str] (distinct source domains)                     ║
    ║                                                                       ║
    ║ 2.2 Update state on claim addition                                    ║
    ║     - mass += source_credibility (from publisher prior)               ║
    ║     - heat = recency-weighted sum                                     ║
    ║     - Recompute coherence from edge types                             ║
    ║                                                                       ║
    ║ 2.3 Add entropy computation                                           ║
    ║     - Extract numeric claims (death toll, etc.)                       ║
    ║     - Compute distribution over values                                ║
    ║     - H = -Σ p_i log p_i                                              ║
    ║                                                                       ║
    ║ 2.4 Implement merge_events()                                          ║
    ║     - compute_event_affinity(event_a, event_b)                        ║
    ║     - Same formula as claim→event                                     ║
    ║     - If affinity > 0.50: merge smaller into larger                   ║
    ║     - Run periodically (e.g., every 15 minutes)                       ║
    ║                                                                       ║
    ║ 2.5 Add display_score computation                                     ║
    ║     - display = mass × heat × log(1+sources) × (1-0.5×entropy)        ║
    ║                                                                       ║
    ║ Validation: Check event merging on HK Fire data (should consolidate)  ║
    ╚═══════════════════════════════════════════════════════════════════════╝

    ╔═══════════════════════════════════════════════════════════════════════╗
    ║ PHASE 3: Edge Computation + Topology Integration (Week 5-6)           ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║ Goal: Compute claim edges during routing, persist in Neo4j            ║
    ║                                                                       ║
    ║ Tasks:                                                                ║
    ║ 3.1 Integrate ClaimTopologyService into UEEWorker                     ║
    ║     - After claim joins event, classify relations                     ║
    ║     - Use incremental_update() for efficiency                         ║
    ║                                                                       ║
    ║ 3.2 Persist edges in Neo4j                                            ║
    ║     - (Claim)-[:CORROBORATES {similarity: 0.87}]->(Claim)             ║
    ║     - (Claim)-[:DIFFERS {value_a: 36, value_b: 160}]->(Claim)         ║
    ║                                                                       ║
    ║ 3.3 Resolve DIFFERS at distillation                                   ║
    ║     - When generating event summary                                   ║
    ║     - Use timestamps: later claim wins                                ║
    ║     - Track evolution chain                                           ║
    ║                                                                       ║
    ║ 3.4 Update coherence/entropy from edges                               ║
    ║     - coherence = corroborates / (corroborates + differs)             ║
    ║     - entropy from DIFFERS edges with numeric values                  ║
    ║                                                                       ║
    ║ Validation: Check HK Fire death toll evolution (36→160)               ║
    ╚═══════════════════════════════════════════════════════════════════════╝

    ╔═══════════════════════════════════════════════════════════════════════╗
    ║ PHASE 4: Display Ranking + API (Week 7-8)                             ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║ Goal: Surface events by display_score, expose via API                 ║
    ║                                                                       ║
    ║ Tasks:                                                                ║
    ║ 4.1 Add event tier classification                                     ║
    ║     - HEADLINE: display > 100, sources >= 5                           ║
    ║     - SIGNIFICANT: display > 50, sources >= 3                         ║
    ║     - EMERGING: display > 20, high heat                               ║
    ║     - INTERNAL: display < 20 (not shown)                              ║
    ║                                                                       ║
    ║ 4.2 Create ranked events API                                          ║
    ║     - GET /api/events/live → top events by display_score              ║
    ║     - Include mass, heat, entropy, sources count                      ║
    ║     - Include claim count, entity list                                ║
    ║                                                                       ║
    ║ 4.3 Add heat decay job                                                ║
    ║     - Periodic task (every hour)                                      ║
    ║     - heat = heat × exp(-λ × hours_elapsed)                           ║
    ║     - Events naturally drop from display as they cool                 ║
    ║                                                                       ║
    ║ 4.4 Update frontend to use new API                                    ║
    ║     - Show events by tier                                             ║
    ║     - Display uncertainty indicators                                  ║
    ║     - Show source count                                               ║
    ║                                                                       ║
    ║ Validation: Check that major stories surface correctly                ║
    ╚═══════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
6. DATA MODEL CHANGES
═══════════════════════════════════════════════════════════════════════════════

    CURRENT Event model:
    ────────────────────
    Event
      id: str
      canonical_name: str
      status: str
      created_at: datetime
      updated_at: datetime
      embedding: List[float]
      narrative: str

    ENHANCED Event model:
    ─────────────────────
    Event
      id: str
      canonical_name: str
      status: str
      created_at: datetime
      updated_at: datetime
      embedding: List[float]        # embedding_centroid
      narrative: str

      # NEW: Jaynesian state
      mass: float                   # Σ source_credibility
      heat: float                   # recency-weighted activity
      entropy: float                # uncertainty level
      coherence: float              # corroborates / total_edges
      sources: Set[str]             # distinct source domains
      entity_surface: Set[str]      # all entity IDs in event

      # NEW: Display
      display_score: float          # computed from above
      tier: str                     # HEADLINE, SIGNIFICANT, EMERGING, INTERNAL

    Neo4j schema additions:
    ───────────────────────
    (Event) node:
      + mass: float
      + heat: float
      + entropy: float
      + coherence: float
      + display_score: float
      + tier: string

    (Claim)-[:CORROBORATES]->(Claim)
      similarity: float
      computed_at: datetime

    (Claim)-[:DIFFERS]->(Claim)
      value_a: string
      value_b: string
      resolved_by: string (claim_id that supersedes)
      computed_at: datetime


═══════════════════════════════════════════════════════════════════════════════
7. KEY CODE CHANGES
═══════════════════════════════════════════════════════════════════════════════

    FILE: backend/workers/knowledge_worker.py
    ──────────────────────────────────────────
    CHANGE: Output to claim pool instead of event queue

    # CURRENT (line ~285-289)
    await self.job_queue.enqueue('queue:event:high', {
        'page_id': str(page_id),
        'url': url,
        'claims_count': len(claim_ids)
    })

    # NEW
    for claim_id in claim_ids:
        await self.job_queue.enqueue('queue:claims:pending', {
            'claim_id': str(claim_id),
            'page_id': str(page_id),
            'source': 'knowledge_worker'
        })


    FILE: backend/workers/uee_worker.py (NEW)
    ─────────────────────────────────────────
    class UEEWorker:
        async def process_claim(self, claim_id: str):
            # 1. Fetch claim with embedding + entities
            claim = await self.claim_repo.get_by_id(claim_id)
            await self.claim_repo.hydrate_entities(claim)

            # 2. Compute affinity against active events
            best_event, best_score = await self.pool.compute_best_affinity(claim)

            # 3. Route
            if best_score > 0.45:
                await self.pool.add_claim_to_event(claim, best_event)
            else:
                await self.pool.seed_new_event(claim)

            # 4. Compute edges (optional, can be batched)
            await self.compute_edges_for_claim(claim)

            # 5. Update event state
            await self.update_event_state(claim.event_id)


    FILE: backend/services/live_event_pool.py
    ─────────────────────────────────────────
    CHANGE: Add compute_affinity() and route_claim()

    def compute_affinity(self, claim: Claim, event: Event) -> float:
        # Semantic similarity (60%)
        semantic = cosine_similarity(claim.embedding, event.embedding_centroid)

        # Entity overlap (25%)
        shared = set(claim.entity_ids) & event.entity_surface
        entity_score = len(shared) / max(len(claim.entity_ids), 1)

        # Entity specificity (15%)
        if shared:
            specificity = sum(
                1.0 - (self.entity_freq.get(e, 1) / self.max_freq)
                for e in shared
            ) / len(shared)
        else:
            specificity = 0

        return 0.60 * semantic + 0.25 * entity_score + 0.15 * specificity


    FILE: backend/models/domain/event.py
    ────────────────────────────────────
    CHANGE: Add Jaynesian state fields

    @dataclass
    class Event:
        # ... existing fields ...

        # Jaynesian state
        mass: float = 0.0
        heat: float = 0.0
        entropy: float = 0.0
        coherence: float = 1.0
        sources: Set[str] = field(default_factory=set)
        entity_surface: Set[str] = field(default_factory=set)
        embedding_centroid: List[float] = None

        def compute_display_score(self) -> float:
            diversity = math.log(1 + len(self.sources))
            entropy_penalty = 1 - 0.5 * self.entropy
            return self.mass * self.heat * diversity * entropy_penalty


═══════════════════════════════════════════════════════════════════════════════
8. MIGRATION SAFETY
═══════════════════════════════════════════════════════════════════════════════

    BACKWARDS COMPATIBILITY:
    ────────────────────────
    • Keep old EventWorker running during Phase 1
    • Write to both old queue AND new claim pool
    • Compare results between old and new routing
    • Validate event assignments match (minus improvements)

    ROLLBACK PLAN:
    ──────────────
    • Each phase is independent
    • Can revert to previous phase if issues
    • Neo4j schema is additive (new fields, not removed)
    • PostgreSQL changes are additive

    TESTING STRATEGY:
    ─────────────────
    • Unit tests for compute_affinity()
    • Integration tests with snapshot data
    • Compare UEE events vs current events vs newsroom expectations
    • Validate HK Fire consolidation (should have 140+ claims)
    • Validate death toll evolution (36 → 160)

""")
