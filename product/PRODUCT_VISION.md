# HERE.news Product Vision: The Epistemic Live Organism

## Core Concept

HERE.news is a **live organism** for truth-finding, not a social feed or one-shot synthesis model. The system operates as a continuous loop where:

1. **People care** (homepage) → stake/bounty attention on inquiries
2. **REEE weaves evidence** (claims → surfaces/views) and emits proto-inquiries + tasks where uncertainty/conflict exists
3. **Humans do work** (OSINT, primary docs, disambiguation, scope splits) by completing tasks or adding new inquiries-in-context
4. **REEE updates posteriors + trace**; "resolved" and "needs evidence" become measurable states, not vibes

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE EPISTEMIC LOOP                          │
│                                                                 │
│   ┌──────────┐    stake     ┌──────────┐    emit tasks         │
│   │ Homepage │ ──────────► │   REEE   │ ──────────────┐       │
│   │(Attention│   bounties   │ (Weaver) │   proto-inq   │       │
│   │ Market)  │              └────┬─────┘               │       │
│   └────▲─────┘                   │                     ▼       │
│        │                    updates                ┌───────┐   │
│        │                  posteriors              │ Human │   │
│        │                         │                │  Work │   │
│        │                         ▼                │(Tasks)│   │
│        │              ┌──────────────────┐        └───┬───┘   │
│        └──────────────│  Inquiry Pages   │◄───────────┘       │
│          "resolved"   │ (REEE2 Contract) │  contributions     │
│                       └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Objects

| Object | Definition | Invariant |
|--------|------------|-----------|
| **ProtoInquiry** | REEE-derived question from detected uncertainty/conflict in evidence | Auto-generated; becomes Inquiry when staked or manually promoted |
| **Inquiry** | Epistemic contract with target, resolver policy, bounty, and trace | Contracts can only append L0 evidence or propose versioned params (no silent edits) |
| **Event/Case** | REEE1 view grouping claims into narrative timeline at a zoom level | View is a lens, not a container; same claims can appear in multiple views |
| **Task** | Meta-claim action request (verify source, disambiguate entity, scope-split) | Tasks produce evidence about evidence; completion rewards provenance work |
| **Surface** | Claim cluster at a specific confidence/zoom level | Surfaces are versioned snapshots; inquiry chips pin to specific surface versions |

**Key invariant**: Contracts append-only at L0. Parameter changes (scope, deadline, resolver) create new versions with migration path, never silent overwrites.

---

## Key Pages (Unified Mental Model)

### 1. Homepage: The Attention Market

**Purpose**: Surface what matters, where uncertainty is high, and where resolution creates value.

**Sections**:
- **Top-Staked Inquiries**: Where the most bounty capital flows
- **Newly Resolved (Policy)**: Fresh certainty that affects decisions
- **High-Entropy/High-Stakes**: "Needs evidence" — uncertainty that matters
- **Parent Event/Case Chips**: Entry points to narrative contexts

**Business alignment**: Bounties create liquidity; resolution creates trust/retention.
**Epistemic alignment**: Attention flows to genuine uncertainty, not engagement-bait.

**Ranking factors** (operational definition of "high stakes / relevance"):
1. `stake` — total bounty committed (demand signal)
2. `entropy` — posterior uncertainty in bits (epistemic need)
3. `recency` — time since last evidence update (staleness penalty)
4. `conflict_count` — unresolved CONFLICTS relations in trace (controversy signal)
5. `task_backlog` — open tasks awaiting completion (work availability)
6. `impact_tags` — manual or derived tags (policy, safety, financial) that boost visibility

Default sort: `0.4*stake + 0.3*entropy + 0.2*conflict_count + 0.1*task_backlog`, with recency decay.

### 2. Event/Case Page: REEE1 View

**Purpose**: Narrative timeline with inquiry chips embedded at the right zoom level.

**Zoom semantics**:
- **Incident view**: Single discrete event (e.g., "Wang Fuk Court Fire on Nov 26")
- **Case view**: Aggregated narrative spanning related incidents (e.g., "Hong Kong Building Safety Crisis 2025")
- Toggle between views; same claims can appear at both levels with different groupings

**Structure**:
- **Timeline**: Chronological events with confidence states
- **Inquiry Chips**: Embedded at relevant points showing:
  - Current belief (posterior)
  - Active tasks count
  - Bounty available
  - **Pinned to specific surface version** — chips don't silently regroup when REEE re-clusters
- **Context Creation**: Users can create new inquiries "inside" this narrative context

**Snapshot pinning**: When an inquiry chip is placed in a view, it pins to the current surface version. If REEE later re-clusters claims, the chip shows a "view updated" indicator rather than silently changing position. User can accept new placement or keep pinned.

**Example**: Wang Fuk Court Fire
- Event timeline shows incident progression
- Chips for "How many died?" (contested), "What caused it?" (under investigation)
- Users can add scoped inquiries: "Were fire doors operational?"

### 3. Inquiry Page: REEE2 Contract

**Purpose**: A formal epistemic contract with explicit terms.

**Components**:
- **Target**: What exactly are we trying to know? (typed: count, boolean, date, etc.)
- **Resolver Policy**: How will this be decided? (artifact list, deadline rules, evidence standards)
- **Bounty/Task Board**: What work is needed? What's the reward?
- **Epistemic Trace**: Full provenance pinned to snapshots
  - Claims → Sources → Surfaces → Views
  - Relationship symbols (first-class relations matching REEE engine):
    - `=` CONFIRMS — new evidence supports existing claim
    - `↑` REFINES — narrows scope or adds precision to existing claim
    - `→` SUPERSEDES — replaces prior claim (e.g., correction, update)
    - `!` CONFLICTS — contradicts existing claim (drives entropy up)
    - `+` NOVEL — new claim with no prior relation in this surface

### 4. Entity Page: Entity-Centric Entry Point

**Purpose**: Access events and inquiries through entities (people, organizations, locations).

**Structure**:
- **Entity Profile**: Verified facts, Wikidata link, related entities
- **Inquiry List**: Questions about this entity
  - "How many biological children does Elon Musk have?" (typed: count)
  - "What is Elon Musk's current net worth?" (typed: monetary)
- **Event Timeline**: Events involving this entity
- **Inquiry Wizard**: Create new entity-scoped inquiries with:
  - Time reference (when does this apply?)
  - Related entities (context)
  - Similar inquiry detection (prevent duplicates)

---

## Rigor Framework: Record-Truth vs World-Truth

### The Classic Failure Mode

Conflating "someone said/reported" with "it is true."

### The Solution: Proposition Scaffolding

**Example: "Is Trump mentioned in the Epstein files?"**

Default scaffold into separate propositions:

1. **Record-Truth (Rigor A)**:
   - "Does official 2025 disclosure document contain reference to 'Donald Trump' (with disambiguation)?"
   - Resolver = specific artifact list + parsing rules
   - Decidable: YES/NO based on document content

2. **World-Truth (Rigor C)**:
   - "Did Trump associate with Epstein?"
   - Becomes a bundle of A/B sub-inquiries:
     - Named documents (flight logs, visitor records)
     - Sworn testimony (depositions, court records)
     - Physical evidence (photos with metadata)
   - Each sub-inquiry has its own resolver

### Rigor Levels

| Level | Name | Evidence Standard | Resolver | Output |
|-------|------|-------------------|----------|--------|
| A | Record-Truth | Specific artifact exists/contains | Parsing rules on defined docs | Decidable YES/NO |
| B | Attestation-Truth | Someone claims X under penalty | Court records, sworn statements | Decidable with provenance |
| C | World-Truth | Bundle of A/B evidence | Weighted posterior from sub-inquiries | **Index/scorecard** unless upgraded |
| D | System/Meta | Evidence about the inquiry system itself | Admin/governance rules | Affects contract params |

**Critical note on C**: World-truth inquiries output an **index** (list of supporting/conflicting evidence with weights) rather than a binary answer, unless the contract explicitly upgrades to a decision threshold. This prevents false certainty on inherently contested questions.

---

## Polls: Evidence About Beliefs, Not Truth

### What Polls Are

A poll is a claim like:
> "N participants believe hypothesis H with distribution D at time T"

### How to Use Polls

**Useful for**:
- Prioritization (what do people want to know?)
- Forecasting (prediction markets)
- Attention routing (what's controversial?)

**NOT useful for**:
- Directly updating world-truth posteriors
- Resolving factual inquiries by majority vote

**Exception**: Forecast-market-style resolvers where the inquiry contract explicitly defines poll/market as the resolution mechanism (e.g., "What will the prediction market price be for X on date Y?")

**Surface type**: Polls create **belief-about-beliefs surfaces** — distinct from world-truth surfaces. These track "what do people think?" separately from "what is true?", unless a contract explicitly collapses them (and then it's a forecast, not an epistemic inquiry).

---

## Quality/Safety Constraints

### 1. Reward Provenance, Not Probability Movement

**Reward**:
- Task completion (primary source located, timestamp verified)
- Provenance quality (disambiguation, scope corrections)
- Novel evidence surfacing

**Don't reward**:
- "Moving the number" without evidence
- Engagement-farming (hot takes, controversy)
- Circular sourcing (news citing news)

### 2. Enforce Resolver Rules for Deadline Questions

**Absence-of-event can be decidable** if the contract defines:
- The artifact to check (official record, authoritative source)
- The deadline (after which non-occurrence is confirmed)
- The checking procedure (who/how verifies)

Example: "Will X resign by Dec 31, 2025?"
- Artifact: Official announcement from org
- Deadline: Dec 31, 2025 23:59 UTC
- Procedure: Check official channels, absence = NO

### 3. Privacy/Ethics Boundaries

**Entity pages must forbid**:
- Doxxing (non-public personal addresses, family members not in public life)
- Non-public personal data (medical records, private communications)
- Speculation presented as fact

**OSINT must be bounded by**:
- Admissible evidence policy
- Public-source requirement
- Consent for private individuals

---

## Page Alignment Checklist

### Homepage ✓
- [ ] Top-staked inquiries section
- [ ] Newly resolved (with policy impact indicator)
- [ ] High-entropy/high-stakes "needs evidence"
- [ ] Event/case chips as entry points
- [ ] Trending entities with hot inquiries

### Event/Case Page ✓
- [ ] Narrative timeline with dates
- [ ] Embedded inquiry chips showing belief state
- [ ] Task count per inquiry
- [ ] "Create inquiry in context" flow
- [ ] Wikipedia-style structure for credibility

### Inquiry Page ✓
- [ ] Explicit target type (count, boolean, date, text)
- [ ] Resolver policy displayed
- [ ] Bounty breakdown
- [ ] Task board
- [ ] Epistemic trace with relationship symbols
- [ ] Snapshot pinning

### Entity Page ✓
- [ ] Entity profile with verified facts
- [ ] Wikidata/external links
- [ ] Inquiry list (filterable: open/resolved)
- [ ] Inquiry wizard with:
  - [ ] Time reference
  - [ ] Context entities
  - [ ] Similar inquiry detection
- [ ] Related entities

---

## Implementation Priority

### Phase 1: Core Loop (MVP)
1. Homepage attention market (top bounties, needs evidence)
2. Inquiry page with bounty/task/trace
3. Basic entity pages

### Phase 2: Context Integration
1. Event pages with embedded inquiry chips
2. Entity-scoped inquiry creation
3. Rigor level indicators

### Phase 3: REEE Integration
1. Automated proto-inquiry generation
2. Posterior updates from evidence
3. Conflict/uncertainty detection
4. Task queue optimization

---

## Success Metrics

**Epistemic Health**:
- Resolution rate (inquiries resolved per week)
- Evidence quality (% with primary sources)
- Disagreement resolution time

**Business Health**:
- Bounty volume ($ staked weekly)
- Contributor growth (active task completers)
- Retention (return visitors after resolution)

**Trust Indicators**:
- Accuracy on verifiable inquiries
- Source diversity per inquiry
- Reversal rate (resolved inquiries re-opened)
