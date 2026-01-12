# User Story: Topology + Inquiry Working Together (Concrete)

This is a concrete, end-to-end user story that maps product UX to the deep “micro → macro” system:

- **Topology** (Surfaces → Incidents → Cases) provides structure.
- **Inquiry** turns uncertainty and contradictions into actionable work.
- **Spine vs metabolic edges** ensure “relatedness ≠ identity” (no giant components).

The goal is a user experience where people can:
1) see what happened, 2) see what is uncertain, 3) contribute evidence, 4) watch the system update, and
5) understand why.

---

## Cast (Personas)

- **Mina (Reader)**: wants a clean story view and explicit uncertainty.
- **Bo (OSINT researcher)**: likes tasks with bounties; can quickly find primary sources.
- **Asha (Local journalist)**: has access to official updates; cares about attribution and correctness.
- **Rui (Moderator / Power user)**: resolves scope disputes; approves high-impact merges/splits.

---

## Scenario: “Wang Fuk Court Fire” becomes an organism (and stays pure)

### T0 — Breaking news appears (minutes)

**Mina** opens HERE.news and sees a trending card:

> **Wang Fuk Court Fire (Hong Kong)**  
> Status: **Updating**  
> Confidence: **Low/Medium** (multiple conflicting counts)  
> Open work: **3 inquiries** · **5 tasks** · **$120 bounty**

When Mina clicks, she lands on a **Case page** (L4 organism):

### What the Case page looks like (concrete UI)

The page has four panels that stay stable even as the story evolves:

1) **Spine timeline (membership)**
- A list of incident “beats” (updates) with timestamps.
- Each beat expands into its surfaces (facets) and citations.

2) **Facets (L2 surfaces)**
- A “what variables exist?” grid: deaths, injuries, evacuations, cause, arrests, etc.
- Each facet shows a distribution + confidence + citation count + last-updated.

3) **Open inquiries & tasks**
- “What we still need” items with bounties and task buttons.
- Clicking a task shows the exact required evidence (source type + timestamp rules).

4) **Metabolism map (relations)**
- A directed graph of incident-to-incident relations (response, investigation, charges, dispute).
- Each edge is explicitly labeled **non-membership** unless it is a spine edge.

**A) Spine timeline (membership-defining)**
- A spine is a minimal set of identity-strong links: `same_happening` / `update_to`.
- This is what prevents unrelated incidents from joining by “Hong Kong” alone.

**B) Metabolism map (non-membership links)**
- Directed edges like `response_to`, `investigates`, `charges`, `disputes`, `context_for`.
- These edges explain *how* things relate without forcing membership merges.

**C) Inquiry panel**
- Explicit “What we don’t know yet” items, each backed by signals from topology.

---

## The micro loop: Claim → Surface (random variable) → belief update

### T1 — The system ingests claims and creates L2 surfaces

The extraction pipeline turns URLs into **Claims** (atomic statements) and emits evidence artifacts
(entities, roles, predicate family, values).

The system creates/updates **Surfaces** (L2 = random variables), such as:

- `death_count:<referent>` (typed numeric)
- `evacuation_count:<referent>`
- `cause_hypothesis:<referent>` (categorical)
- `arrest_status:<referent>` (categorical)

Each Surface has:
- a **belief state** (Jaynes-style posterior + entropy)
- citations (claims)
- traces (“why this claim belongs here”)

**Mina sees this as “Facets” on the case page:**

- **Deaths**: [8–13] (entropy high; conflicting sources)
- **Evacuated**: ~700 (entropy low; consistent)
- **Cause**: unknown (no corroborated evidence)

---

## Inquiry emergence: signals → inquiry seeds → tasks

### T2 — Uncertainty becomes explicit work

From topology, the system emits **EpistemicSignals** (machine-readable quality facts), e.g.:
- `CONFLICT` on a surface (values disagree)
- `CORROBORATION_LACKING` (single-source only)
- `MISSING_TIME` (no timestamp)
- `BRIDGE_BLOCKED` (scope ambiguity prevented a merge)

Signals are converted into **InquirySeeds** (actionable prompts) and shown to users as:

**Open Inquiries**
1) “How many people died in the Wang Fuk Court fire?”  
   - reason: conflict + high entropy on `death_count`
   - needs: official update / hospital bulletin / timestamped corroboration
2) “Were renovations ongoing at time of fire?”  
   - reason: corroboration lacking on `renovation_status`
3) “What caused the fire?”  
   - reason: missing evidence (only speculative sources so far)

Each InquirySeed spawns concrete **Tasks**:
- “Find Fire Services Department press release”
- “Find Hospital Authority casualty bulletin”
- “Extract timeline timestamps from official updates”

---

## User interaction: stake → task → contribution → update

### T3 — Mina funds what matters

Mina clicks **Add bounty** on the “Deaths” inquiry.

- Mina’s stake increases the inquiry’s bounty pool.
- The system automatically increases task bounties for that inquiry.

From Mina’s perspective:
- “I’m not betting. I’m funding investigation.”

What the UI records immediately:
- “Bounty pool increased”
- “New tasks posted”
- “Subscription prompt” (“Notify me when deaths confidence changes”)

### T4 — Bo claims a task and submits evidence

Bo claims: “Find official Fire Services Department update”.

Bo submits a URL + quote + screenshot:
- The system creates a **Contribution** (product object)
- The contribution is ingested into the epistemic loop as a **Claim**
- The claim attaches to the correct Surface via semantic extraction
- The Surface posterior updates (Jaynes) and entropy drops

**Mina sees the case page change in real time (or on refresh):**
- Deaths posterior tightens: `[8–13] → [11–12]`
- Confidence rises (entropy drops)
- The inquiry shows “1 task completed” and credits paid out

Bo sees:
- task bounty paid
- impact reward based on information gain (entropy reduction)

What happens behind the scenes (so UI can explain it later):
- Bo’s submission is converted into a Claim (with source URL and timestamp).
- The claim is attached to a Surface only if it matches the surface’s variable key.
- The Surface posterior updates (Jaynes) and emits a belief trace (prior → posterior).

---

## Preventing contamination: scope disputes become metabolic, not merges

### T5 — A contaminant claim appears

A new incident elsewhere in Hong Kong mentions “Hong Kong” and “Tai Po” (broad context) but is not
about Wang Fuk Court.

The system prevents a false merge by applying the **membrane witness rule**:
- Only **referent roles** can create spine edges (membership).
- Broad locations and authorities create at most metabolic edges.

What Mina sees:
- The case remains pure (membership unchanged).
- A new **metabolic** link may appear: `context_for` (“same district context”), clearly marked as non-membership.
- If ambiguity is high, the system emits an inquiry: “Is this the same incident or a different fire?”

What Rui (moderator) can do:
- Review a “scope correction” card (“Split/merge suggestion”)
- Approve a split/merge if the boundary evidence is strong
- See a trace of which referent(s) caused the proposed merge

What a “scope correction” interaction looks like:
1) Rui clicks “This is a different incident”
2) Rui selects which incident beat it belongs to (or “new incident”)
3) Rui selects the witness (facility/time) that justifies separation
4) The system replays the relevant traces and shows “expected effect”:
   - membership change (spine) vs relation change (metabolic)
5) Rui confirms; the system logs a reversible decision

---

## Macro emergence: phases, causal structure, and narrative without re-merging

### T6 — The story evolves into phases

Over days, new incidents appear that are not new events, but new phases:
- investigation updates
- arrests/charges related to renovation compliance
- government policy responses (safety inspections)

These are *not* merged into one Surface (random variable) and are *not* separate unrelated stories.
They become **metabolic edges** inside the organism:

- `phase_of`: outbreak → response → investigation → trial
- `charges_or_prosecutes`: legal action emerges
- `response_to`: policy response / inspections

**Mina sees a coherent organism:**
- A stable spine (“this is the Wang Fuk Court fire case”)
- A readable sequence of phases (metabolism)
- Each phase has its own surfaces and inquiries (“what caused it?”, “who is responsible?”)

### Where causal-effect fits (interaction without identity collapse)

Asha adds a contribution:
> “Fire started during renovation; flame-retardant compliance was questioned by inspectors.”

The system creates:
- new facets (surfaces): `compliance_status:<referent>`, `cause_hypothesis:<referent>`
- new inquiries: “Which code requirement was violated?” (if evidence is weak/contradictory)
- metabolic edges:
  - `causes_or_triggers` (renovation conditions → fire outbreak) **if** supported
  - `investigates` (inspections → compliance facet)

Mina sees a “Causal chain” view:
- Fire outbreak
  - Investigated by: Fire Services / Urban Renewal Authority
  - Possible trigger: renovation conditions (confidence tagged)
  - Dispute edge if claims conflict (“denies_or_disputes”)

---

## Expensive reasoning only when it matters

### T7 — Boundary refinement (selective)

When a candidate edge would:
- connect large components, or
- flip case membership, or
- resolve a major conflict inquiry,

the system escalates from cheap semantics to expensive adjudication.

From a product standpoint:
- “This merge requires higher rigor” (a visible badge)
- “Awaiting review” (with explainable reasons)

Example of when the system escalates:
- Two incidents share a person referent (“Jimmy Lai”) but not a facility/time witness.
- The deterministic gate downgrades to `context_for` by default.
- If a user tries to force membership, the UI requires higher rigor and triggers expensive adjudication.

---

## What the user walks away with

Mina’s experience after 48 hours:
- A stable story object (Case) that doesn’t bloat into a giant component.
- Clear uncertainty and progress (“here’s what we know / don’t know”).
- Actions to take (tasks) with aligned incentives (bounties).
- A change log of why the system believes what it believes (traces + citations).

Bo’s experience:
- High-leverage tasks surfaced by real epistemic gaps, not vague “fact checking”.
- Rewards tied to measurable information gain.

Rui’s experience:
- Moderation is rare and focused on boundary edges (where membership is contested).
- Decisions are explainable: which referents, which roles, which witnesses, which df/broadness demotions.
