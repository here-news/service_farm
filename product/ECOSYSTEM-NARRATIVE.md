# The Two Loops: An Ecosystem Narrative

> How the Epistemic and Community loops create a self-reinforcing truth-finding engine.

This document illustrates the HereNews ecosystem through concrete scenarios, showing how machine-driven evidence processing and human participation accelerate each other through epochs of growth.

---

## Economic Model

**Exchange Rate: 1 credit = $0.01 (1 cent)**

| Credits | USD | Example |
|---------|-----|---------|
| 100 | $1.00 | Small task bounty |
| 1,000 | $10.00 | New user signup bonus |
| 5,000 | $50.00 | Significant contribution reward |
| 50,000 | $500.00 | Major inquiry total stake |
| 500,000 | $5,000.00 | High-profile contested inquiry |

This rate makes micro-contributions viable while allowing serious bounties for important work.

---

## The Core Flywheel

```
                        ┌─────────────────────────────────┐
                        │      COMMUNITY LOOP             │
                        │                                 │
                        │  Stakes → Tasks → Contributions │
                        │      ↑              │           │
                        │      │              ↓           │
                        │  Rewards ←── Impact ←── Work    │
                        └─────────┬───────────────────────┘
                                  │
                    ◄─────────────┼─────────────►
                    Contributions │ Tasks emerge
                    become Claims │ from Gaps
                                  │
                        ┌─────────┴───────────────────────┐
                        │      EPISTEMIC LOOP             │
                        │                                 │
                        │  Claims → Surfaces → Events     │
                        │      ↓              │           │
                        │   Identity      Aboutness       │
                        │   Edges ───────── Edges         │
                        │      ↓              ↓           │
                        │     Meta-claims (Gaps)          │
                        └─────────────────────────────────┘
```

**The Key Insight**: Each loop feeds the other:
- The **Epistemic Loop** detects what we don't know (gaps, conflicts, single-source claims)
- The **Community Loop** mobilizes humans to fill those gaps with evidence
- Human contributions become claims, improving epistemic quality
- Improved epistemic quality reveals new, deeper gaps
- Bounties incentivize work on the highest-value gaps

---

## Scenario 1: The Wang Fuk Court Fire

*An incident-level event showing how the loops bootstrap from breaking news.*

### Epoch 0: Breaking News (Hour 0-2)

**Epistemic Loop activates:**

At 11:47 AM on November 26, 2025, news breaks of a fire at Wang Fuk Court in Tai Po, Hong Kong.

```
EXTRACTION WORKER processes initial URLs:
├── bbc.com/news/world-asia-...  → 4 claims extracted
├── reuters.com/world/asia/...   → 3 claims extracted
└── scmp.com/news/hong-kong/...  → 5 claims extracted

WEAVER (REEE) forms initial surfaces:
├── Surface S001: "Fire at residential building in Tai Po"
│   └── Claims: 12 claims, 3 sources
│   └── Entities: {Wang Fuk Court, Tai Po, Fire Services}
│   └── Entropy: 0.3 (low - consistent reports)
│
└── Surface S002: "Casualties reported"
    └── Claims: 5 claims, 3 sources
    └── Entropy: 0.7 (high - conflicting numbers)
```

**Meta-claims emitted:**
```
MC_001: single_source_only on S002
        Evidence: {source: 'scmp.com', claim_count: 2}

MC_002: high_dispersion_surface on S002
        Evidence: {dispersion: 0.7, meaning: 'casualty counts vary widely'}

MC_003: coverage_gap
        Evidence: {missing_type: 'official', expectedness: 0.9}
```

### Epoch 1: Community Responds (Hour 2-6)

**Alice**, a freelance journalist in Hong Kong, sees the inquiry on the homepage:

```
┌──────────────────────────────────────────────────────────────┐
│ 💰 TOP BOUNTIES                                              │
├──────────────────────────────────────────────────────────────┤
│ How many died in the Wang Fuk Court fire?                    │
│ $0.00 staked  •  3 contributions  •  5.2 bits entropy        │
│ [📊 4-12 deaths] 18% confidence                              │
│                                                              │
│ ⚠️ NEEDS: Official source, Corroboration                     │
└──────────────────────────────────────────────────────────────┘
```

Alice thinks: "I have contacts at the Fire Services Department. This could be valuable."

She stakes 5,000 credits ($50) to signal importance:

```
Community Action:
├── Alice stakes 5,000 credits ($50) on inquiry INQ_001
├── credit_transactions logged: {type: 'stake', amount: -5000}
├── inquiries.total_stake updated: 0 → 5,000 ($50)
└── Task bounties auto-funded from stake pool
```

**Bob**, an OSINT researcher, claims Task_001:

```
Task_001: "Find official government source"
├── Type: need_primary_source
├── Bounty: 750 credits / $7.50 (15% of stake pool)
├── Claimed by: Bob
└── Status: in_progress
```

Bob finds the Fire Services Department press release:

```
Bob's Contribution:
├── text: "Fire Services confirmed 11 fatalities as of 15:30"
├── source_url: "https://www.hkfsd.gov.hk/press/..."
├── extracted_value: 11
├── observation_kind: "point"
```

**Epistemic Loop processes Bob's contribution:**

```
CONTRIBUTION → CLAIM PIPELINE:
├── Contribution processed → Claim CL_089 created
├── CL_089 linked to Surface S002
├── S002 entropy recalculated: 0.7 → 0.4
├── MC_003 (coverage_gap) marked resolved
└── Posterior updated: P(deaths=11) increases

REWARD CALCULATION:
├── Entropy reduction: 0.3 bits
├── Impact score: 0.12
├── Bob's impact reward: 5,000 * 0.12 * 0.7 = 420 credits ($4.20)
└── Task_001 marked complete, Bob receives 750 credit bounty ($7.50)
```

Bob's balance: +1,170 credits ($11.70) for 20 minutes of work.

### Epoch 2: Conflicting Evidence (Hour 6-24)

**Carol**, a medical researcher, contributes from a hospital source:

```
Carol's Contribution:
├── text: "Queen Elizabeth Hospital received 7 deceased,
│         4 more at Prince of Wales Hospital"
├── source_url: "https://www.ha.org.hk/..."
├── extracted_value: 11
├── observation_kind: "aggregate"
```

But then **local news** reports differently:

```
NEW CLAIMS FROM EXTRACTION:
├── Source: Oriental Daily
├── Claim: "At least 13 confirmed dead including 2 children"
├── extracted_value: 13
└── Conflicts with: CL_089 (value: 11)
```

**REEE detects tension:**

```
Meta-claims:
├── MC_007: unresolved_conflict
│   ├── claim_1: CL_089 (Fire Services: 11)
│   ├── claim_2: CL_112 (Oriental Daily: 13)
│   └── confidence: 0.82
│
└── MC_008: high_dispersion_surface on S002
    └── dispersion: 0.65 (increased from resolved sources)
```

**New tasks emerge** (stake pool has grown to 15,000 credits / $150 from additional stakers):

```
Task_003: "Resolve conflict between official and media count"
├── Type: unresolved_conflict
├── Bounty: 2,000 credits / $20 (auto-funded from growing stake)
├── Description: "Fire Services says 11, Oriental Daily says 13"
└── Status: open
```

**David**, a Hong Kong resident, claims the task:

```
David's Investigation:
├── Visits Ha.gov.hk mortality records (2 hours later)
├── Finds update: "2 additional deaths from injuries"
├── Source: Hospital Authority bulletin, timestamped
│
David's Contribution:
├── text: "HA updated: 2 additional deaths at 21:00,
│         bringing total to 13. Initial count was ED arrivals only."
├── extracted_value: 13
├── observation_kind: "supersedes"
├── links_to: CL_089 (supersedes relationship)
```

**Epistemic Loop resolves:**

```
CONFLICT RESOLUTION:
├── CL_089 marked as superseded by CL_145
├── MC_007 (unresolved_conflict) resolved
├── Posterior: P(deaths=13) = 0.89
├── S002 entropy: 0.65 → 0.22
│
DAVID'S REWARD:
├── Conflict resolution bonus: 2,000 credits / $20 (task bounty)
├── Entropy reduction impact: 0.43 bits → 840 credits / $8.40
└── Total: 2,840 credits ($28.40)
```

### Epoch 3: Case-Level Pattern Emerges (Week 1-4)

As more incidents get processed, **REEE detects a pattern:**

```
CASE VIEW FORMATION:
├── Event: Wang Fuk Court Fire (Nov 26)
├── Event: Kwun Tong Industrial Fire (Dec 3)
├── Event: Sham Shui Po Tenement Fire (Dec 12)
│
SHARED ANCHORS (high-IDF):
├── "Hong Kong"
├── "Fire Services Department"
├── "building safety"
├── "elderly residents"
│
CASE CREATED:
└── "Hong Kong Building Safety Crisis 2025"
    ├── 3 events, 340 claims, 28 sources
    └── Parent inquiry auto-generated
```

**Proto-inquiry emerges:**

```
REEE SEEDER generates:
├── ProtoInquiry: "Is Hong Kong building code enforcement adequate?"
├── Type: categorical
├── Categories: [adequate, inadequate, reform_needed]
├── Evidence:
│   ├── Pattern of similar incidents
│   ├── Fire safety concerns mentioned in 47 claims
│   └── Official inspection gaps cited by 3 sources
```

**Emma**, a policy analyst, promotes this to a full Inquiry and stakes 20,000 credits ($200):

```
Community Action:
├── Emma stakes 20,000 credits ($200)
├── ProtoInquiry promoted to Inquiry INQ_045
├── Tasks auto-generated from case-level gaps
│
Tasks Created:
├── Task_089: "Find Building Department inspection records" - 3,000 credits ($30)
├── Task_090: "Compare fire safety regulations HK vs Singapore" - 2,500 credits ($25)
├── Task_091: "Locate government audit reports on enforcement" - 4,000 credits ($40)
```

### Epoch Summary: Value Created

After 4 weeks on the Wang Fuk Court fire and related cases:

```
EPISTEMIC VALUE:
├── 3 events with 89-95% posterior confidence
├── 1 case-level pattern identified
├── 340 claims from 28 sources integrated
├── 12 conflicts resolved with provenance
├── Entropy reduced: 34.7 bits total
│
COMMUNITY VALUE:                                    ECONOMICS
├── 47 contributions from 23 users
├── 85,000 credits staked across inquiries         = $850 total staked
├── 58,000 credits distributed as rewards          = $580 paid to contributors
├── 4 users earned >5,000 credits each             = >$50 each (power contributors)
│
TOP CONTRIBUTOR EARNINGS (4 weeks):
├── Bob (OSINT): 8,200 credits                     = $82
├── David (local): 6,400 credits                   = $64
├── Grace (HR): 5,100 credits                      = $51
└── Carol (medical): 4,300 credits                 = $43
│
FLYWHEEL METRICS:
├── Avg time to official source: 2.3 hours
├── Conflict resolution time: 4.1 hours
├── User retention (contributed again): 78%
├── Effective hourly rate for top contributors: ~$15-25/hr
└── Stake/contribution ratio increasing each epoch
```

---

## Scenario 2: The Contested Question

*A high-entropy inquiry showing how the system handles genuine uncertainty.*

### The Question

```
┌──────────────────────────────────────────────────────────────┐
│ ⚔️ HIGHLY CONTESTED                                          │
├──────────────────────────────────────────────────────────────┤
│ How many Russian soldiers have died in Ukraine (Dec 2024)?   │
│ 500,000 credits ($5,000) staked  •  24 contributions         │
│ 4.8 bits entropy  •  [📊 315,000] 32% confidence             │
│                                                              │
│ Sources conflict: 🇺🇦 UA claims 350k  •  🇷🇺 RU claims 45k   │
│ 4 tasks open  •  High stakes, high uncertainty               │
└──────────────────────────────────────────────────────────────┘
```

This is a **Rigor Level C** inquiry - genuine world-truth uncertainty.

**Economic context**: At $5,000 staked, this is a high-value inquiry attracting serious researchers. Task bounties range from $50-200 each.

### The Evidence Landscape

```
SURFACES:
├── S_UA_OFFICIAL: Ukrainian General Staff claims
│   ├── Claims: 23
│   ├── Source type: official (one side)
│   ├── Value range: 340,000 - 380,000
│   └── Entropy: 0.8 (internally consistent, but single perspective)
│
├── S_RU_OFFICIAL: Russian MoD statements
│   ├── Claims: 8
│   ├── Source type: official (other side)
│   ├── Value range: 40,000 - 50,000
│   └── Entropy: 0.4 (consistent but implausible given other signals)
│
├── S_OSINT: Open-source intelligence
│   ├── Claims: 31
│   ├── Sources: Mediazona, BBC Russian, iStories
│   ├── Method: Obituaries, social media, cemetery surveys
│   ├── Value range: 70,000 - 120,000
│   └── Entropy: 1.2 (methodological variation)
│
└── S_WESTERN_INTEL: US/UK intelligence estimates
    ├── Claims: 12
    ├── Sources: Leaked assessments, official briefings
    ├── Value range: 280,000 - 350,000
    └── Entropy: 0.9 (depends on definitions)
```

**Meta-claims reveal the real issue:**

```
MC_201: typed_coverage_zero for "verified_individual_deaths"
        → We have estimates, not verified unit-level data

MC_202: high_dispersion_surface on combined evidence
        → 4.8 bits entropy = genuine uncertainty

MC_203: unresolved_conflict between methodologies
        → OSINT counts verified deaths (lower bound)
        → Intel estimates include wounded/missing (higher)
```

### Community Approaches the Problem

**Frank**, a conflict researcher, proposes a scope split:

```
Frank's Contribution:
├── type: scope_correction
├── text: "This inquiry conflates three different metrics:
│         1. Verified deaths (OSINT method)
│         2. Combat losses (killed+wounded+missing)
│         3. Total military losses (all causes)"
└── proposal: Split into 3 sub-inquiries with clear definitions
```

**System response:**

```
REEE SEEDER creates sub-inquiries:
├── INQ_201a: "Verified Russian military deaths (OSINT methodology)"
│   ├── Scope: Deaths confirmed through individual documentation
│   ├── Current estimate: 75,000 (Mediazona floor)
│   ├── Entropy: 1.1 bits (narrower uncertainty)
│   └── Rigor: A (artifact-based)
│
├── INQ_201b: "Total Russian casualties (killed+wounded+missing)"
│   ├── Scope: All combat losses using standard 3:1 ratio
│   ├── Current estimate: 280,000-350,000
│   ├── Entropy: 1.4 bits
│   └── Rigor: B (methodology-based)
│
└── INQ_201c: "Russian military deaths from all causes"
    ├── Scope: Including accidents, disease, suicide
    ├── Current estimate: 90,000-400,000 (wide range)
    ├── Entropy: 2.8 bits
    └── Rigor: C (contested)
```

**Frank's reward for scope correction:**

```
IMPACT CALCULATION:
├── Created 3 tractable sub-inquiries from 1 contested mega-inquiry
├── Reduced effective entropy: 4.8 → avg(1.1, 1.4, 2.8) = 1.77 bits
├── Entropy reduction: 3.03 bits
├── Stake pool participating: 500,000 credits ($5,000)
│
REWARD:
├── Scope correction bonus: 15,000 credits ($150) - 3% of stake
├── Entropy impact reward: 4,500 credits ($45)
└── Total: 19,500 credits ($195) for intellectual contribution
│
ECONOMICS NOTE:
├── Frank spent ~2 hours on analysis and proposal
├── Effective rate: ~$97/hour
└── High-value intellectual work properly compensated
```

### The Flywheel on Contested Questions

```
BEFORE SCOPE SPLIT:
├── 1 inquiry, 4.8 bits entropy, 32% confidence
├── Contributions hit ceiling (conflicting methodologies)
├── Stake accumulating but no resolution path
│
AFTER SCOPE SPLIT:
├── 3 inquiries with clear resolution paths
├── INQ_201a: Resolvable by counting verified deaths
├── INQ_201b: Resolvable by methodology agreement
├── INQ_201c: Explicitly flagged as high-uncertainty index
│
COMMUNITY RESPONSE:
├── New contributions target specific sub-inquiries
├── OSINT contributors focus on INQ_201a (tractable)
├── Analysts focus on INQ_201b (methodology)
├── INQ_201c accepted as "index" not "answer"
```

---

## Scenario 3: Entity-Centric Growth

*How the system builds persistent knowledge about entities.*

### The Entity: Jimmy Lai

From our data, Jimmy Lai appears in 117 claims across 13 sources with clear tension.

**Initial State:**

```
ENTITY: Jimmy Lai (en_jimmylai)
├── Wikidata: Q708255
├── Type: Person (media executive)
├── Claims mentioning: 117
├── Events involving: 3
│   ├── ev_jimmylai_trial: "Jimmy Lai National Security Trial"
│   ├── ev_appledaily_closure: "Apple Daily Shutdown"
│   └── ev_hk_pressfreedom: "Hong Kong Press Freedom Decline"
│
TENSION DETECTED:
├── Pole A: "Political prisoner under oppressive law"
│   └── Sources: CNBC, DW, Independent, THEFP, RSF
├── Pole B: "Defendant in legitimate legal proceeding"
│   └── Sources: The Standard, SCMP
│
Auto-generated inquiries:
├── INQ_301: "Is Jimmy Lai's trial fair by international standards?"
│   ├── Type: categorical (fair/unfair/disputed)
│   ├── Rigor: C (contested world-truth)
│   └── Entropy: 1.5 bits
│
├── INQ_302: "How long has Jimmy Lai been imprisoned?"
│   ├── Type: monotone_count (days)
│   ├── Rigor: A (record-truth)
│   └── Entropy: 0.1 bits (easily resolved)
│
└── INQ_303: "What is Jimmy Lai's current health status?"
    ├── Type: categorical
    ├── Rigor: B (attestation-based)
    └── Entropy: 1.8 bits (family claims vs official silence)
```

### Grace Builds an Entity Profile

**Grace**, a human rights researcher, systematically works on Jimmy Lai inquiries:

```
Grace's Contributions (over 2 weeks):
│
├── INQ_302 (imprisonment duration):
│   ├── Found: Court records showing detention start Dec 2020
│   ├── Calculated: 1,827 days as of Dec 2024
│   └── Resolved with 98% confidence
│
├── INQ_301 (trial fairness):
│   ├── Scope correction: "Fairness" needs definition
│   ├── Created sub-inquiry: "Right to jury trial"
│   │   └── Evidence: NSL cases have no jury (record-truth)
│   ├── Created sub-inquiry: "Access to counsel of choice"
│   │   └── Evidence: UK barrister denied entry (record-truth)
│   └── Parent inquiry becomes index of sub-findings
│
└── INQ_303 (health status):
    ├── Found: Son's testimony to UK Parliament
    ├── Found: Hong Kong government denial
    └── Marked as "contested attestation" (Rigor B ceiling)
```

**Grace's cumulative impact:**

```
ENTITY PROFILE IMPROVEMENT:
├── 5 inquiries touched
├── 2 resolved to high confidence
├── 3 properly scoped with sub-inquiries
├── Jimmy Lai entity page now shows structured findings
│
GRACE'S REWARDS:                                    ECONOMICS
├── Task completions: 18,000 credits               = $180
├── Entropy reduction: 9,500 credits               = $95
├── Scope corrections: 6,000 credits               = $60
├── Total: 33,500 credits over 2 weeks             = $335
│
ECONOMICS ANALYSIS:
├── Hours invested: ~25 hours over 2 weeks
├── Effective rate: $13.40/hour
├── Sustainable part-time income for domain expert
│
GRACE'S REPUTATION:
├── Contribution count: 23
├── Accuracy rate: 94%
├── Specialization: Legal/Human Rights
└── Visible on entity contributor leaderboard
```

---

## The Acceleration Effect

### How Epochs Compound

```
EPOCH 1 (Launch):                                   ECONOMICS
├── 10 seed inquiries
├── 50 initial users
├── 1,000 credits in circulation                    = $10 total
├── Avg resolution time: 7 days
├── Bounties too small to attract professionals
│
EPOCH 2 (Month 1):                                  ECONOMICS
├── 10 → 45 inquiries (auto-generated from events)
├── 50 → 180 users (attracted by bounties)
├── 1,000 → 8,000 credits staked                    = $80 total
├── Avg resolution time: 4 days (more contributors)
├── Early adopters earning $5-10/week
│
EPOCH 3 (Month 3):                                  ECONOMICS
├── 45 → 200 inquiries
├── 180 → 600 users
├── 8,000 → 35,000 credits staked                   = $350 total
├── Avg resolution time: 2 days
├── Case-level patterns emerging
├── Top contributors earning $30-50/week
│
EPOCH 4 (Month 6):                                  ECONOMICS
├── 200 → 800 inquiries
├── 35,000 → 150,000 credits staked                 = $1,500 total
├── Entity profiles: 2,000+ with structured findings
├── Power contributors: $100-200/week (side income)
├── Institutional users staking for research value
├── Resolution time: <24 hours for record-truth inquiries
│
EPOCH 5 (Year 1):                                   ECONOMICS
├── 800 → 5,000+ inquiries
├── 150,000 → 1,000,000 credits staked              = $10,000 total
├── Monthly rewards distributed: ~$7,000
├── Power contributors: $300-500/week (significant income)
├── Expert contributors: $50-100/hr effective rate
└── System becomes self-sustaining
```

### The Flywheel Mechanics

**Why it accelerates:**

1. **More stakes → More bounties → More contributors**
   - Early contributors earn well
   - Word spreads, new contributors join
   - Competition improves quality

2. **More contributions → Better epistemic quality → More trust**
   - High-confidence resolutions become reference points
   - Institutional users see value
   - Stakes increase for important questions

3. **Better quality → More events processed → More gaps detected**
   - REEE gets smarter with more data
   - Meta-claims become more precise
   - Tasks become more specific and tractable

4. **More tasks → More specialization → Faster resolution**
   - Contributors specialize (OSINT, legal, medical)
   - Matching improves (right task to right person)
   - Resolution time drops

---

## Characters Summary

| Name | Role | Specialty | Key Actions |
|------|------|-----------|-------------|
| **Alice** | Journalist | Local sources | Stakes early, signals importance |
| **Bob** | OSINT researcher | Primary docs | Claims tasks, finds official sources |
| **Carol** | Medical researcher | Health data | Contributes hospital records |
| **David** | HK resident | Local knowledge | Resolves conflicts with ground truth |
| **Emma** | Policy analyst | Patterns | Promotes proto-inquiries, stakes on cases |
| **Frank** | Conflict researcher | Methodology | Scope corrections, sub-inquiry splits |
| **Grace** | HR researcher | Legal/rights | Systematic entity profile building |

---

## Key Invariants Demonstrated

1. **Claims are immutable** - Bob's Fire Services claim wasn't edited when superseded; new claim added with SUPERSEDES relation

2. **Contributions become claims** - All human contributions entered the epistemic loop as L0 claims

3. **Tasks emerge from gaps** - Every task was auto-generated from a meta-claim (missing source, conflict, high entropy)

4. **Rewards proportional to impact** - David's 28.4 credits came from measurable entropy reduction + task bounty

5. **Stakes fund work** - Alice's 50 credits created the bounty pool that paid Bob and David

6. **Scope corrections are valuable** - Frank earned $195 for intellectual contribution, not just document-finding

7. **Entity profiles accumulate** - Grace's 23 contributions built permanent, structured knowledge about Jimmy Lai

---

## The End State Vision

After 12 months of operation:

```
THE ECOSYSTEM:                                      ECONOMICS
├── 5,000+ active inquiries
├── 50,000+ resolved inquiries (reference library)
├── 10,000+ entity profiles with structured findings
├── 500+ power contributors                         = $250k/month to contributors
├── Total credits in circulation: 50M               = $500k total economy
│
MONTHLY FLOWS (Year 1 End):
├── Stakes added: ~2M credits/month                 = $20k/month staked
├── Rewards distributed: ~1.5M credits/month        = $15k/month to contributors
├── Top 10 contributors: avg 50k credits/month      = $500/month each
├── Top 100 contributors: avg 15k credits/month     = $150/month each
│
THE PRODUCT:
├── "What's the answer?" → Check HereNews first
├── Journalists cite resolution traces
├── Researchers use API for structured data
├── Institutions stake on questions they need answered
│   └── Single institutional stake: 500k-5M credits = $5k-$50k
│
THE MOAT:
├── Network effects: More contributors = faster resolution
├── Data effects: Historical traces create unique dataset
├── Trust effects: Track record of accuracy builds reputation
└── Economic effects: Credit system creates real value exchange
    └── Total value transacted Year 1: ~$200k
```

---

## Appendix: Credit Economics

### Flow of Value

```
NEW USER:
├── Signup bonus: 1,000 credits                     = $10 starting balance
├── Can stake or contribute immediately
├── Typical first contribution reward: 200-500 cr  = $2-5
│
STAKING (Example: 5,000 credits / $50):
├── User stakes 5,000 credits on inquiry
├── 5,000 credits deducted from balance
├── 5,000 credits added to inquiry bounty pool
├── Tasks auto-funded from pool (10-20% each):
│   └── 3-5 tasks at 500-1,000 credits ($5-10) each
│
CONTRIBUTING (Example: Resolving a task):
├── User submits contribution
├── Contribution processed → claim created
├── Impact calculated (entropy reduction)
├── Reward formula: stake_pool × impact × 0.7
├── Example: 5,000 × 0.15 × 0.7 = 525 credits      = $5.25
│
TASK COMPLETION (Example: "Find official source"):
├── User claims task (bounty: 750 credits / $7.50)
├── User submits contribution solving task
├── Task marked complete
├── User receives:
│   └── Task bounty: 750 credits                   = $7.50
│   └── Impact reward: 420 credits                 = $4.20
│   └── Total: 1,170 credits                       = $11.70
│
RESOLUTION:
├── Inquiry reaches 95% confidence
├── Remaining stake pool distributed:
│   ├── 70% to contributors (proportional to impact)
│   ├── 30% returned to stakers
│   └── Example: 5,000 cr pool → 3,500 to contributors, 1,500 to staker
```

### Sustainable Economics

For the system to be sustainable (1 credit = $0.01):

```
VALUE IN (Monthly, Year 1 End):
├── Credit purchases: ~$15k/month
│   └── avg $30/user × 500 active buyers
├── Institutional subscriptions: ~$10k/month
│   └── 20 orgs × $500/month avg
├── API access fees: ~$5k/month
│   └── Research/journalism licenses
├── Total revenue: ~$30k/month = $360k/year
│
VALUE OUT (Monthly):
├── Contributor rewards: ~$15k/month            (50% of revenue)
├── Infrastructure (servers, APIs): ~$5k/month  (17% of revenue)
├── Quality control/moderation: ~$3k/month      (10% of revenue)
├── Operating margin: ~$7k/month                (23% of revenue)
│
BREAK-EVEN ANALYSIS:
├── Fixed costs: ~$8k/month (infra + team)
├── Break-even: 800k credits staked/month       = $8k/month
├── Current trajectory: Profitable by Month 9
│
SCALING (Year 2+):
├── Early: Subsidize with seed funding (~$200k)
├── Growth: Credits have real exchange value
├── Mature: Self-sustaining marketplace
└── Target: $1M+ annual value transacted
```

---

## Conclusive Summary: The Three Value Dimensions

The scenarios in this document—Wang Fuk Court Fire, Russian Casualties, and Jimmy Lai—demonstrate how the two-loop architecture creates value across three dimensions that compound over time.

### Economic Value: A New Market for Truth

| Metric | Demonstrated Value | Year 1 Projection |
|--------|-------------------|-------------------|
| Total stakes | 85,000 credits ($850) in fire scenario | $500k total economy |
| Contributor earnings | $82 top earner (4 weeks) | $15k/month distributed |
| Cost per resolution | ~$25 avg per inquiry | 10-100x cheaper than traditional |
| Effective hourly rate | $15-25/hr for skilled work | $20-100/hr for experts |

**Key Insight**: The credit economy creates a functioning market where:
- **Stakers** pay for answers they value (information as commodity)
- **Contributors** earn for work proportional to impact (meritocratic rewards)
- **The system** captures value from entropy reduction (truth has measurable price)

At 1 credit = $0.01, the Wang Fuk Court fire resolution cost $850 total—a fraction of what a single journalist's week costs. The market prices truth efficiently.

### Epistemic Value: Structured Uncertainty and Provenance

| Metric | Before HereNews | After HereNews |
|--------|-----------------|----------------|
| Casualty certainty | "500 dead" (wrong) | "4-12 range, 18% confidence" → "13 confirmed, 89%" |
| Conflict resolution | Days/weeks, buried corrections | 4.1 hours avg, visible trace |
| Source attribution | "Sources say" | Full provenance to specific document |
| Methodology transparency | Hidden editorial judgment | Rigor levels (A/B/C), entropy displayed |

**The Three Scenarios Demonstrate:**

1. **Wang Fuk Fire** (Incident-level): How meta-claims detect gaps (`single_source_only`, `coverage_gap`) and tasks auto-generate to fill them. Entropy reduced from 5.2 bits to <1 bit in 24 hours.

2. **Russian Casualties** (Contested question): How scope correction (Frank's contribution) transforms an intractable 4.8-bit question into three tractable sub-inquiries. The system admits what it doesn't know instead of false certainty.

3. **Jimmy Lai** (Entity-centric): How structured findings accumulate into entity profiles. Grace's 23 contributions created permanent, citable knowledge—not ephemeral social media posts.

**Key Insight**: The epistemic loop makes uncertainty visible and corrections valuable. When the system doesn't know something, it says so with numbers. When evidence conflicts, both sides show with confidence levels. This is epistemology encoded into incentives.

### Social Value: Trust, Coordination, and Public Good

| Dimension | Traditional Model | HereNews Model |
|-----------|------------------|----------------|
| Who decides truth | Editors, platforms, officials | Distributed verification + posteriors |
| Accountability | None (quiet corrections) | Full trace, reputation effects |
| Expertise recognition | Credentials-based | Performance-based (accuracy rate) |
| Coordination | Ad-hoc, duplicated work | Structured tasks, parallel contribution |
| Polarization | Amplified by engagement optimization | Reduced by uncertainty visibility |

**Social Value Created:**

1. **Trust Through Transparency**
   - Every claim traced to source
   - Every update visible in posterior evolution
   - Contributors build reputation through accuracy, not credentials
   - Alice (journalist), Bob (OSINT), David (local resident) all contribute on equal footing

2. **Coordination Without Central Authority**
   - Tasks auto-generated from epistemic gaps
   - Bounties direct attention to highest-value work
   - Cross-verification prevents single points of failure
   - 47 contributions from 23 users on fire scenario—coordinated by incentives, not management

3. **Truth as Public Good**
   - Resolved inquiries become reference library
   - Entity profiles accumulate structured findings
   - Provenance traces citable by journalists and researchers
   - Knowledge compounds rather than disappearing into news cycle

**Key Insight**: The community loop transforms truth-finding from a zero-sum competition (who gets the scoop) into a positive-sum collaboration (who contributes to resolution). Contributors compete on quality, not speed. The result is a public good—verifiable knowledge—funded by those who value it.

### The Compounding Effect

```
YEAR 1 VALUE CREATION:

ECONOMIC:
├── Total value transacted: ~$200k
├── Contributors paid: ~$150k
├── Cost per bit of entropy reduced: ~$10-50
└── Sustainable by Month 9

EPISTEMIC:
├── 50,000+ resolved inquiries (reference library)
├── 10,000+ entity profiles with structured findings
├── Avg entropy reduction: 2.3 bits per inquiry
└── 12 conflicts resolved with full provenance (fire scenario alone)

SOCIAL:
├── 500+ contributors building reputation
├── Trust established through track record
├── Knowledge compounds (each resolution makes next easier)
└── Alternative to both broken journalism and chaotic social media
```

### Why This Matters

The three scenarios show that HereNews isn't just a better news platform—it's a new institution for collective sense-making:

- **Not journalism**: No editorial gatekeeping, no narrative framing, distributed verification
- **Not social media**: No engagement optimization, explicit uncertainty, permanent accountability
- **Not prediction markets**: Works on non-binary questions, rewards evidence not speculation
- **Not Wikipedia**: Real-time, incentivized contribution, machine-assisted gap detection

The two-loop architecture creates something new: a **market for epistemic labor** where truth-finding is economically sustainable, epistemically rigorous, and socially beneficial.

When Alice stakes $50 on the fire inquiry, she's not betting—she's funding investigation. When Bob finds the official source and earns $11.70, he's not gaming a system—he's being paid for genuine epistemic work. When the system shows "13 deaths, 89% confidence" instead of "officials say 13," it's not hedging—it's being honest about what we know.

This is the value proposition: **truth that pays for itself, uncertainty that's visible, and knowledge that compounds.**

---

*This narrative demonstrates that the two-loop architecture creates a coherent, self-reinforcing system where machine intelligence and human contribution multiply each other's value—economically, epistemically, and socially.*
