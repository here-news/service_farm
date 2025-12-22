"""
CLAIM POOL + UEE WORKER ARCHITECTURE

A metabolic system where:
1. Claims flow in from community
2. UEE processes and routes to events
3. High-quality events surface to users
4. Users contribute more → feedback loop
5. Economic layer prevents gaming, rewards alignment

This is an INFORMATION ORGANISM with metabolism.
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CLAIM POOL + UEE ARCHITECTURE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
1. THE FLOW
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   UPSTREAM (Community)                                                  │
    │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
    │   │ User A   │  │ User B   │  │ Crawler  │  │ API Feed │               │
    │   │ submits  │  │ shares   │  │ finds    │  │ pushes   │               │
    │   │ URL      │  │ claim    │  │ articles │  │ updates  │               │
    │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
    │        │             │             │             │                      │
    │        └─────────────┴─────────────┴─────────────┘                      │
    │                          │                                              │
    │                          ▼                                              │
    │   ┌──────────────────────────────────────────────────────────────┐      │
    │   │                    CLAIM POOL                                 │      │
    │   │                                                               │      │
    │   │  queue:claims:pending                                         │      │
    │   │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                 │      │
    │   │  │ c1  │ c2  │ c3  │ c4  │ c5  │ c6  │ ... │                 │      │
    │   │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                 │      │
    │   │                                                               │      │
    │   │  Each claim has:                                              │      │
    │   │  - text, entities, embedding                                  │      │
    │   │  - source_url, submitter_id                                   │      │
    │   │  - energy_stake (anti-spam)                                   │      │
    │   └──────────────────────────────────────────────────────────────┘      │
    │                          │                                              │
    │                          ▼                                              │
    │   ┌──────────────────────────────────────────────────────────────┐      │
    │   │                    UEE WORKER                                 │      │
    │   │                                                               │      │
    │   │  For each claim:                                              │      │
    │   │  1. compute_affinity(claim, all_events)                       │      │
    │   │  2. Route: JOIN event or SEED new event                       │      │
    │   │  3. Update event: mass, heat, entropy                         │      │
    │   │  4. Compute REWARD for submitter                              │      │
    │   │                                                               │      │
    │   │  Periodically:                                                │      │
    │   │  - merge_events() (same rules)                                │      │
    │   │  - decay_heat() (time-based)                                  │      │
    │   │  - hibernate dormant events                                   │      │
    │   └──────────────────────────────────────────────────────────────┘      │
    │                          │                                              │
    │                          ▼                                              │
    │   ┌──────────────────────────────────────────────────────────────┐      │
    │   │                 LIVE EVENT POOL                               │      │
    │   │                                                               │      │
    │   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │      │
    │   │  │ HK Fire │  │ J. Lai  │  │ Bondi   │  │ ...     │          │      │
    │   │  │ m=81.6  │  │ m=77.1  │  │ m=48.2  │  │         │          │      │
    │   │  │ h=0.92  │  │ h=0.85  │  │ h=0.45  │  │         │          │      │
    │   │  │ H=0.12  │  │ H=0.08  │  │ H=0.05  │  │         │          │      │
    │   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │      │
    │   │                                                               │      │
    │   │  Events ranked by: display = mass × heat × log(sources)       │      │
    │   └──────────────────────────────────────────────────────────────┘      │
    │                          │                                              │
    │                          ▼                                              │
    │   ┌──────────────────────────────────────────────────────────────┐      │
    │   │                 DISPLAY LAYER                                 │      │
    │   │                                                               │      │
    │   │  HEADLINE tier: HK Fire, Jimmy Lai                            │      │
    │   │  SIGNIFICANT tier: Bondi, Brown                               │      │
    │   │  EMERGING tier: Venezuela, Do Kwon                            │      │
    │   │                                                               │      │
    │   │  Users see high-scoring events → contribute more              │      │
    │   └──────────────────────────────────────────────────────────────┘      │
    │                          │                                              │
    │                          ▼                                              │
    │                   FEEDBACK LOOP                                         │
    │                  (more claims in)                                       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
2. ECONOMIC LAYER: ENERGY & REWARDS
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ENERGY STAKE (Anti-Gaming)                                             │
    │                                                                         │
    │  To submit a claim, user must stake energy:                             │
    │                                                                         │
    │     stake = BASE_COST × reputation_factor                               │
    │                                                                         │
    │  Where reputation_factor:                                               │
    │     - New user: 1.0 (normal cost)                                       │
    │     - Trusted contributor: 0.5 (discount)                               │
    │     - Suspicious pattern: 2.0+ (penalty)                                │
    │                                                                         │
    │  If claim is LOW QUALITY (spam, duplicate, rejected):                   │
    │     → stake is BURNED (lost)                                            │
    │                                                                         │
    │  If claim is HIGH QUALITY:                                              │
    │     → stake returned + REWARD                                           │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  REWARD SIGNALS (What We Incentivize)                                   │
    │                                                                         │
    │  1. CORROBORATION REWARD                                                │
    │     - Claim CORROBORATES existing claims in event                       │
    │     - Increases event coherence                                         │
    │     - reward = stake × corroboration_factor                             │
    │                                                                         │
    │  2. ENTROPY REDUCTION REWARD                                            │
    │     - Claim provides DEFINITIVE answer to uncertainty                   │
    │     - Resolves "death toll 36 or 160?" → "confirmed 160"                │
    │     - reward = stake × entropy_delta × source_credibility               │
    │                                                                         │
    │  3. SOURCE DIVERSITY BONUS                                              │
    │     - Claim from NEW source not yet in event                            │
    │     - Adds independent verification                                     │
    │     - reward = stake × novelty_factor                                   │
    │                                                                         │
    │  4. BRIDGE BONUS                                                        │
    │     - Claim connects two events (shared entities)                       │
    │     - Enables meta-narrative discovery                                  │
    │     - reward = stake × bridge_value                                     │
    │                                                                         │
    │  5. EARLY SIGNAL BONUS                                                  │
    │     - First to contribute to emerging event                             │
    │     - Decays as event grows (early bird advantage)                      │
    │     - reward = stake × (1 / log(event_mass + 1))                        │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ATTACK VECTORS & DEFENSES                                              │
    │                                                                         │
    │  ATTACK 1: Spam flood                                                   │
    │  Defense: Energy stake + rate limiting                                  │
    │           Burned stakes make spam expensive                             │
    │                                                                         │
    │  ATTACK 2: Sybil (fake accounts)                                        │
    │  Defense: New accounts have low reputation = high stake cost            │
    │           Reputation builds slowly over time                            │
    │                                                                         │
    │  ATTACK 3: Coordinated disinformation                                   │
    │  Defense: Source credibility weighting                                  │
    │           Low-credibility sources add less mass                         │
    │           Cross-source contradiction detection                          │
    │                                                                         │
    │  ATTACK 4: Gaming rewards (fake corroboration)                          │
    │  Defense: Only credible sources increase coherence                      │
    │           Same-source claims don't corroborate                          │
    │           Semantic similarity threshold (not just entity match)         │
    │                                                                         │
    │  ATTACK 5: Entropy manipulation                                         │
    │  Defense: Only claims from NEW credible sources reduce entropy          │
    │           Bayesian update requires evidence, not assertion              │
    └─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
3. THE METABOLISM EQUATIONS
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EVENT STATE                                                            │
    │                                                                         │
    │  Each event e has:                                                      │
    │     mass(t)    = Σ credibility_i × recency_weight_i                     │
    │     heat(t)    = Σ exp(-λ × (now - claim_time_i))                       │
    │     entropy(t) = -Σ p_i × log(p_i)  over contested values               │
    │     coherence  = corroborates / (corroborates + contradicts)            │
    │     sources    = |distinct domains|                                     │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CLAIM CONTRIBUTION                                                     │
    │                                                                         │
    │  When claim c joins event e:                                            │
    │                                                                         │
    │     Δmass = source_credibility(c)                                       │
    │     Δheat = 1.0 (new claim is "hot")                                    │
    │     Δentropy = depends on whether c resolves or adds uncertainty        │
    │     Δcoherence = depends on c's relation to existing claims             │
    │                                                                         │
    │  REWARD(c) = f(Δmass, Δheat, -Δentropy, Δcoherence, novelty)            │
    │                                                                         │
    │  Negative entropy change = entropy REDUCTION = GOOD                     │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DISPLAY RANKING                                                        │
    │                                                                         │
    │  display_score(e) = mass × heat_factor × diversity_factor               │
    │                     × (1 - entropy_penalty)                             │
    │                                                                         │
    │  Where:                                                                 │
    │     heat_factor = min(1.5, 0.5 + heat/max_heat)                         │
    │     diversity_factor = log(1 + sources)                                 │
    │     entropy_penalty = 0.5 × entropy  (high uncertainty = lower rank)    │
    │                                                                         │
    │  Events with high mass + recent activity + multiple sources             │
    │  + resolved uncertainty → TOP OF DISPLAY                                │
    └─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
4. ARCHITECTURAL FIT
═══════════════════════════════════════════════════════════════════════════════

    Current System                    Proposed Extension
    ══════════════════════════════════════════════════════════════════════════

    ┌─────────────────┐              ┌─────────────────┐
    │ KnowledgeWorker │              │ Community UI    │
    │ (crawler)       │              │ (user submit)   │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ queue:event:high│     →        │ CLAIM POOL      │
    │ (page-level)    │              │ (claim-level)   │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ EventWorker     │     →        │ UEE WORKER      │
    │ (page routing)  │              │ (claim routing) │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ LiveEventPool   │     →        │ LiveEventPool   │
    │ (no economics)  │              │ + ECONOMICS     │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ Display         │     →        │ RANKED DISPLAY  │
    │ (all events)    │              │ + REWARDS       │
    └─────────────────┘              └─────────────────┘


    KEY CHANGES:

    1. Claim Pool replaces page-level queue
       - Claims processed individually
       - Community can contribute directly
       - Energy stake required

    2. UEE Worker replaces EventWorker
       - Same affinity logic, claim-level
       - Computes rewards per claim
       - Updates event Jaynesian quantities

    3. Economic layer added
       - Stake/reward mechanism
       - Reputation system
       - Anti-gaming defenses

    4. Display becomes merit-based
       - Ranked by display_score
       - High-quality events surface
       - Feedback loop to community


═══════════════════════════════════════════════════════════════════════════════
5. THE FEEDBACK LOOP (Information Metabolism)
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  EVENT LIFECYCLE                                                        │
    │                                                                         │
    │  1. EMERGENCE                                                           │
    │     - First claim seeds new event                                       │
    │     - Low mass, high entropy, uncertain                                 │
    │     - Internal only (not displayed)                                     │
    │                                                                         │
    │  2. GROWTH                                                              │
    │     - More claims join                                                  │
    │     - Mass increases, entropy may increase or decrease                  │
    │     - If quality → reaches EMERGING tier → displayed                    │
    │                                                                         │
    │  3. ATTRACTION                                                          │
    │     - Displayed event attracts attention                                │
    │     - Users see it, contribute more claims                              │
    │     - Contributors rewarded → more contribution                         │
    │                                                                         │
    │  4. MATURATION                                                          │
    │     - Event reaches HEADLINE tier                                       │
    │     - High mass, low entropy (converged)                                │
    │     - Multiple credible sources confirm                                 │
    │                                                                         │
    │  5. COOLING                                                             │
    │     - Heat decays (no new claims)                                       │
    │     - Drops from HEADLINE → SIGNIFICANT → archive                       │
    │     - Mass preserved (historical record)                                │
    │                                                                         │
    │  6. HIBERNATION                                                         │
    │     - Event archived, searchable                                        │
    │     - Can be REACTIVATED if new claims arrive                           │
    │     - (e.g., new development in cold case)                              │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    This is a LIVING SYSTEM:

    - Events are organisms that consume claims (energy)
    - They grow, compete for attention, mature, and hibernate
    - The economic layer is the metabolism
    - Users are symbiotic contributors
    - Quality emerges from the dynamics, not from curation


═══════════════════════════════════════════════════════════════════════════════
6. IMPLEMENTATION CONSIDERATIONS
═══════════════════════════════════════════════════════════════════════════════

    PHASE 1: Claim Pool + UEE Worker
    ─────────────────────────────────
    - Replace page-level routing with claim-level
    - Add claim queue with basic validation
    - Implement compute_affinity() as core operation
    - No economics yet (just proper routing)

    PHASE 2: Jaynesian Quantities
    ─────────────────────────────────
    - Add mass, heat, entropy to events
    - Implement display_score ranking
    - Add DIFFERS edge handling at distillation
    - Surface events by tier

    PHASE 3: Economic Layer
    ─────────────────────────────────
    - Add energy stake mechanism
    - Implement reward computation
    - Build reputation system
    - Add anti-gaming checks

    PHASE 4: Community Integration
    ─────────────────────────────────
    - UI for claim submission
    - Show rewards and reputation
    - Leaderboards / gamification
    - Feedback loop complete


═══════════════════════════════════════════════════════════════════════════════
7. WHY THIS WORKS (Jaynesian Foundation)
═══════════════════════════════════════════════════════════════════════════════

    The system works because it aligns INCENTIVES with TRUTH:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRUTH-SEEKING INCENTIVES                                               │
    │                                                                         │
    │  Reward CORROBORATION:                                                  │
    │     → Users incentivized to find CONFIRMING sources                     │
    │     → More independent confirmation = higher confidence                 │
    │     → This is Bayesian: P(true|n sources) >> P(true|1 source)           │
    │                                                                         │
    │  Reward ENTROPY REDUCTION:                                              │
    │     → Users incentivized to RESOLVE uncertainty                         │
    │     → Find the definitive answer, not just opinions                     │
    │     → This is information theory: knowledge = entropy reduction         │
    │                                                                         │
    │  Penalize CONTRADICTION (without resolution):                           │
    │     → Adding conflicting claims without evidence is costly              │
    │     → Incentivizes RESOLUTION, not chaos                                │
    │                                                                         │
    │  Penalize SPAM:                                                         │
    │     → Low-quality claims burn stake                                     │
    │     → Gaming is expensive                                               │
    └─────────────────────────────────────────────────────────────────────────┘

    The economic layer makes TRUTH the rational strategy.

    Lying is expensive (burned stakes, reputation damage).
    Truth is profitable (rewards, reputation building).

    This is NOT moderation. This is MARKET DESIGN.
    The system self-organizes toward truth.

""")
