# Relational Weaver Experiment Guide (Essential Directions)

This guide proposes a minimal, high-signal experiment to validate a stronger topology:
**identity membranes + relational metabolism**.

The motivation: “same referent / same variable” is necessary for safety, but insufficient for real
world narratives where objects interact (cause/effect, phases, responses, disputes, frames).
We should stop using *relatedness* as *identity* and instead represent interactions as typed edges.

---

## 0) What You’re Testing (Hypotheses)

### H1 — Identity alone oscillates
Tuning L4 by similarity thresholds produces either:
- giant components (transitive chaining), or
- fragmentation (no edges).

### H2 — Relations stabilize organisms
If we separate:
- **spine edges** (identity-strong, membership-defining)
- **metabolic edges** (non-membership, explanatory interactions)

…then we can build coherent cases without giant-component pathology, while still “seeing the whole”.

### H3 — Cheap semantics is enough for the rough organism
A cheap LLM + embeddings can create reliable *typed edges* for a “rough organism”; expensive LLM is
reserved for finalization on borderline edges.

---

## 1) Definitions (Minimal, Universal)

### Objects
- **L2 Surface**: “one random variable” (Jaynes/TBP updates live here).
- **L3 Incident**: “one happening instance” (membrane over surfaces).
- **L4 Case**: “organism” over incidents **with a spine** (not connected-components over similarity).

### Edge Types
We distinguish edges by whether they define membership.

**Spine edges** (membership-defining for L4):
- `same_happening` (same event instance)
- `update_to_same_happening` (death toll rises, investigation updates, timeline continuation)

**Metabolic edges** (non-membership; link without fusing):
- `causes_or_triggers`
- `response_to` (institutional response, deployments, evacuations, arrests as response)
- `phase_of` (phase transition: outbreak → response → investigation → legal)
- `disputes` / `denies`
- `context_for` (background context; should never imply same-case membership)
- `analogy_or_frame` (narrative framing; explicitly non-identity)

Rule of thumb: if an edge is “interesting but not identity”, it must be metabolic, not spine.

---

## 2) Minimal Experiment Scope

Pick one topic cluster where you already observe fragmentation and/or giant components.

Recommended: **Wang Fuk Court fire (WFC)** + a distractor mix (unrelated incidents that used to chain).

Dataset size:
- Start with incidents that mention a WFC anchor + 200 random non-WFC incidents in same time window.
- This is large enough to test false merges and chaining, small enough to iterate fast.

---

## 3) Inputs Available Today (No New Ingestion Needed)

From the current graph + worker state:
- incident → surfaces (`(i:Incident)-[:CONTAINS]->(s:Surface)`)
- surface `question_key`, `anchor_entities`, claim ids
- surface centroids in PostgreSQL (`content.surface_centroids`) if present
- existing LLM artifacts/traces if enabled in `backend/workers/principled_weaver.py`

---

## 4) The Experimental Pipeline (Propose → Verify → Commit)

### Stage A — Candidate generation (maximize recall; cheap)
For each incident `I`, build candidate incident set `C(I)` from:
- embedding nearest neighbors (if embeddings exist)
- shared anchors (graph)
- time window overlap
- optionally: shared predicates (from question_key prefixes)

Important: candidate generation may be permissive; no merges happen here.

### Stage B — Typed relation proposal (cheap LLM)
For each `(I, J)` in candidates, produce a structured relation:

Output schema (store as artifact; replayable):
- `relation_type`: from the edge types above
- `confidence`: 0–1
- `reason`: short string
- `evidence`: small structured fields (shared entities, time delta, key phrases)

**Only** propose one of:
- spine edge types, or
- metabolic edge types, or
- `unrelated`

### Stage C — Deterministic gates (membranes; no LLM)
Convert proposed relations into graph edges with hard rules:

1) `same_happening` / `update_to_same_happening` can create spine edges **only if**:
   - bridge immunity checks pass (no contradiction vetoes), and
   - time compatibility is acceptable (unless relation explicitly permits long-range).

2) `context_for`, `analogy_or_frame` **never** create spine edges.

3) If relation confidence is medium, create a metabolic edge, not spine.

### Stage D — Case formation from spine only
Build cases using **only spine edges**, using one of:
- mutual-kNN pruned spine graph + connected components, or
- union-find over spine edges only (no similarity edges).

Then attach metabolic edges inside/around cases without affecting membership.

### Stage E — Optional expensive LLM finalization (small budget)
Run expensive adjudication only on:
- spine edges with low cohesion (edges that connect large components),
- edges whose removal splits/merges cases,
- edges with conflict signals (typed_conflict/bridge_blocked-like).

---

## 5) What to Measure (Success Criteria)

### Topology health (structure)
- Largest case size (should drop vs giant component baseline)
- Number of WFC incidents inside the main WFC case (should increase vs fragmentation baseline)
- Purity: % of non-WFC incidents inside the WFC case (should be near zero)

### Edge quality
- Distribution of relation types
- Spine-to-metabolic ratio (expect few spine edges, more metabolic)
- “Bridge edges” count: edges that connect previously unrelated clusters

### Explainability
- Every spine edge has: relation type + confidence + evidence + trace id
- Borderline edges are flagged for refinement (queue)

### Cost/latency
- cheap LLM calls per incident (target: O(k), where k is candidate limit)
- expensive LLM calls per run (target: small, only on boundary edges)

---

## 6) Experimental Variants (Compare Paths)

Run these variants on the same incident subset:

1) **Baseline (current L4)**: similarity/affinity components (what you have now).
2) **Spine-only (no LLM)**: motifs + strict gates only.
3) **Cheap-LMM relations**: add typed relations, cases from spine only.
4) **Cheap + expensive boundary**: add expensive only for boundary edges.

The “win” is variant 3 yielding a coherent WFC case without absorbing unrelated incidents.

---

## 7) Implementation Notes (Minimal Touch Points)

To keep changes localized and reversible:
- Add a “relational experiment mode” that does **not** overwrite production case membership unless enabled.
- Persist relation artifacts and typed edges with an `experiment_id` label/property for cleanup.
- Emit events for canonical worker:
  - `case_formed_provisional`
  - `case_edge_proposed`
  - `case_needs_refinement`

Canonical worker should enrich (titles/summaries) but not decide membership.

---

## 8) Why This Should Work (Intuition)

Connected-components over similarity turns “related” into “identical”. That is the root of both:
- giant components (transitive chaining), and
- threshold oscillation (too loose vs too strict).

By moving “relatedness” into explicit metabolic edges, you keep global coherence without fusing
membranes. Cases become organisms with a spine and metabolism, not blobs.

---

## 9) Minimal Next Action

Implement just enough to run Variant 3 on the WFC subset:
- cheap relation proposer (LLM) + artifact persistence
- deterministic gate to map relations → spine vs metabolic edges
- case formation from spine edges only
- report metrics (largest component, WFC purity/coverage)

Once this validates, generalize to more clusters and introduce expensive refinement only on boundary
edges.

