# REEE Inquiry MVP

## Goal
Ship a first prototype product that lets users:
1) ask “is this true?” questions (inquiries),
2) add evidence/refutations,
3) stake small credits to prioritize verification,
4) see a live plausibility distribution with an auditable epistemic trace,
5) receive actionable “next steps” (meta-claim tasks) that recruit more evidence.

The MVP must preserve REEE’s constitution:
- L0 claims append-only
- identity vs aboutness separation
- belief updates are typed and explicit
- meta-claims drive participation, not hidden editorial judgment

---

## Why reuse REEE (and not build a new engine)
REEE already provides the epistemic substrate required for an inquiry product:

- **L0 append-only provenance log**: user submissions become immutable evidence.
- **L2 Surfaces (identity components)**: isolates “same fact” disputes and updates.
- **Typed belief (Jaynes)**: produces a posterior + entropy + credible sets for typed variables.
- **L3 Events (aboutness)**: groups different aspects without corrupting identity.
- **Meta-claims**: turns epistemic gaps into explicit tasks (“need primary source”, “conflict unresolved”, “single-source only”).

Building a new engine would reintroduce common failure modes (topic blobs, silent overwrites, untraceable “truth”). The product needs a UI/workflow layer, not a different epistemic kernel.

---

## Product Concept: “Inquiry Market with Epistemic Trace”
An Inquiry is a user-scoped question with a typed target and a live evidence state.

Users can:
- open a new inquiry,
- stake credits to signal importance and fund verification tasks,
- contribute evidence in structured form,
- see the current plausibility distribution and exactly why it moved.

REEE provides the evidence topology; the Inquiry layer provides scoping, workflow, and incentives.

---

## MVP Scope (Prototype 1)

### Must-have user stories
1) **Browse**
- See “Top staked” inquiries and “Newly resolved (≥95%)”.

2) **Search**
- Search inquiries by keywords and entities.

3) **Create inquiry**
- Create a new inquiry with:
  - title/question
  - scope (entities + optional time window)
  - target schema (typed variable or boolean)
  - initial evidence (optional links/quotes)
  - stake (starting from $0.01 credit)

4) **Participate**
- Add contributions to an inquiry:
  - evidence (quote + URL + timestamp)
  - refutation
  - attribution (“A said B reported p”)
  - scope correction (“this is a different incident”)
  - entity disambiguation (“Charlie Kirk (TPUSA) vs victim”)

5) **See plausibility + trace**
- Inquiry page shows:
  - plausibility distribution (posterior)
  - uncertainty (entropy / credible set)
  - timeline of contributions with “impact”
  - epistemic trace pane:
    - L0 claims (provenance)
    - L2 surfaces (same-fact clusters)
    - belief state per surface/target
    - L3 events (context)
    - meta-claims/tasks

6) **Task loop**
- System emits tasks (meta-claims) users can “claim”:
  - “need primary source”
  - “unresolved conflict”
  - “single-source only”
- Users submit evidence to complete tasks and receive credits.

### MVP non-goals (explicit)
- No full social network features (follows, feeds beyond inquiry lists).
- No general L4 narrative generation.
- No “final truth oracle” claims; only posterior + provenance.
- No heavy automated crawling; evidence arrives via user submissions in MVP.

---

## Inquiry “Types” in MVP (Unlimited, but graded)
We do not hard-limit types. Instead we support:

- **Template schemas (high rigor)**
  - Monotone count (death toll, missing count)
  - Categorical (legal status, verdict)
  - Report-truth (“did Reuters report p?”)
  - Quote-authenticity (“did X say Y?”)

- **Custom schema (lower rigor)**
  - User defines hypothesis set and what evidence would confirm/refute.
  - The UI displays a “Rigor Level” badge:
    - A: typed + calibrated
    - B: typed + generic priors
    - C: exploratory (no headline “≥95% resolved”)

MVP should compute the headline “resolved ≥95%” only for A/B.

---

## Key Product Screens

### 1) Home
Sections:
- Top staked inquiries
- Newly resolved inquiries (≥95% for ≥24h)
- “Needs evidence” (open tasks / high entropy)

Each inquiry card shows:
- question
- current posterior (e.g., 97/3)
- top meta-claim badge (“needs primary source”)
- total stake, contributors, last update

### 2) Search
Results grouped:
- inquiries
- entities/events (optional)
Filters:
- open/resolved
- rigor level
- time window

### 3) Inquiry detail
Three panes:
1) **Answer pane**
   - posterior
   - credible set / entropy
   - resolution status and criteria

2) **Community pane**
   - contributions timeline
   - stake controls
   - “add evidence/refute/attribution” actions

3) **Epistemic trace pane**
   - claims (L0)
   - surfaces (L2) with conflicts/supersedes
   - belief state summary per target
   - events (L3) context grouping
   - meta-claim tasks with bounties

### 4) New inquiry wizard
Steps:
- question
- scope
- schema/template
- initial evidence (optional)
- stake

---

## How REEE Integrates (Technical)
The MVP adds a wrapper layer, not a new kernel:

### New components (application layer)
- `InquiryEngine`
  - stores Inquiry objects, binds them to REEE state
  - computes inquiry-level posterior from relevant surfaces
  - turns meta-claims into tasks
- `ScopeResolver`
  - selects candidate surfaces/events for an inquiry
  - prevents cross-incident contamination
- `ContributionAdapter`
  - turns user submissions into L0 claims (with provenance)
  - performs attribution factoring where possible

REEE remains unchanged as the evidence processor.

### Dataflow
1) User submission → ContributionAdapter → new L0 claims appended
2) REEE updates:
   - identity edges → surfaces
   - belief update inside surfaces (typed belief)
   - aboutness edges → events
   - meta-claims emitted
3) InquiryEngine reads REEE outputs and updates:
   - inquiry posterior
   - task list
   - trace snapshot pointer

---

## Resolution Criteria (MVP policy)
An inquiry can be labeled “Resolved” if:
- posterior ≥ 0.95 for the leading hypothesis for ≥ 24 hours, and
- no “blocking” tasks remain (e.g., “needs primary source” unresolved), and
- posterior is stable under the last N contributions (no large oscillations).

---

## Metrics of MVP success
Epistemic:
- reduction in entropy over time per inquiry
- fraction of inquiries that reach resolution with multi-source support
- false resolution rate (later reversed) — must be low

Participation:
- task completion rate
- time-to-primary-source for high-stakes inquiries
- contribution quality (measured by posterior impact + later stability)

---

## MVP Deliverables Checklist
- Home page with top staked + newly resolved lists
- Search inquiries
- Inquiry detail page with:
  - posterior + uncertainty
  - contributions timeline
  - epistemic trace pane
  - open tasks list
- New inquiry creation wizard
- Credits/staking ledger (minimal)
- Evidence submission forms and provenance capture

---

## Risks & mitigations
- **False certainty from ill-typed inquiries**
  - Mitigation: rigor badge; only “resolved” for typed A/B
- **Percolation / event contamination**
  - Mitigation: strict scope resolver; multi-signal aboutness; bridge warnings
- **Gaming with money**
  - Mitigation: stakes affect visibility + task bounty only, never belief
- **Low-quality evidence spam**
  - Mitigation: provenance requirements + duplicate detection + low marginal rewards

