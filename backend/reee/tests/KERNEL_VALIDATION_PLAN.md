# Kernel Validation Environment - Build Plan

## What "Perfect" Means

**Perfect ≠ one huge golden graph**

Perfect = stable invariants + deterministic replay + explainable diffs

Concretely:
- Deterministic IDs (hash-based)
- Deterministic ordering (explicit ORDER BY everywhere)
- Tests assert invariants + scenario-local memberships, not full topology snapshots
- Any failure prints: (a) which invariant broke, (b) minimal counterexample, (c) which constraints caused it

---

## Milestone 1: Test Neo4j Substrate
**Goal:** Single pytest that boots Neo4j, loads fixture, runs kernel, tears down.

### Acceptance Criteria
- [ ] `docker-compose.test.yml` starts Neo4j on port 7688
- [ ] `TestNeo4jManager.connect()` succeeds
- [ ] `TestNeo4jManager.load_fixture()` loads 10 claims in deterministic order
- [ ] `TestNeo4jManager.snapshot()` returns ordered dict
- [ ] `TestNeo4jManager.clear_all()` leaves zero nodes
- [ ] Full cycle completes in < 30 seconds

### Stop/Go Gate
```bash
pytest backend/reee/tests/integration/test_neo4j_lifecycle.py -v
# Must pass: test_connect, test_load_fixture, test_snapshot, test_clear
```

### Deliverables
- `docker-compose.test.yml` (working)
- `reee/tests/db/test_neo4j.py` (tested)
- `reee/tests/integration/test_neo4j_lifecycle.py`
- Tiny fixture: `fixtures/micro_10_claims.json`

---

## Milestone 2: Star Story Archetype (WFC Pattern)
**Goal:** Prove pipeline with the case that broke motif recurrence.

### Why This Archetype
- Spine + rotating companions
- No recurring pairs (k=2 fails)
- Tests membrane Core-A/Core-B logic
- Has leaked candidates to test periphery/blocked

### Acceptance Criteria
- [ ] `golden_micro_star_wfc.yaml` with 20-30 claims
- [ ] StoryBuilder produces exactly 1 story for WFC spine
- [ ] Core-A = incidents where WFC is anchor
- [ ] Core-B = 0 (no structural witnesses without ledger)
- [ ] Periphery candidates tracked with blocked reasons
- [ ] `core_leak_rate == 0.0`
- [ ] All invariants pass:
  - `no_semantic_only_core`
  - `no_hub_story_definition`
  - `scoped_surface_isolation`

### Stop/Go Gate
```bash
pytest backend/reee/tests/integration/test_star_story_archetype.py -v
# Must pass: test_wfc_forms_one_story, test_core_leak_zero, test_invariants_hold
```

### Deliverables
- `fixtures/golden_micro_star_wfc.yaml`
- `reee/tests/integration/test_star_story_archetype.py`
- Invariant assertions wired to real kernel output

---

## Milestone 3: Retrieval Completeness Test
**Goal:** Prove on-demand context loading doesn't miss true members.

### Why This Matters
- Neo4j stores full graph
- Kernel loads bounded context (time window + top-k)
- Must not miss true core members
- Must emit `insufficient_context` if budget too small

### Acceptance Criteria
- [ ] Test with retrieval budget: `top_k=20, time_window=7d`
- [ ] Candidate pool includes all true WFC core members
- [ ] If budget too small, kernel emits meta-claim not wrong merge
- [ ] Recall metric computed and asserted

### Stop/Go Gate
```bash
pytest backend/reee/tests/integration/test_retrieval_completeness.py -v
# Must pass: test_full_recall_within_budget, test_insufficient_context_emits_metaclaim
```

### Deliverables
- `reee/tests/integration/test_retrieval_completeness.py`
- Context budget simulation in test harness

---

## Milestone 4: Adversary Archetypes (Hub + Scope Pollution)
**Goal:** Lock anti-percolation under pressure.

### Adversary 1: Hub Entity
- "Hong Kong" / "United States" appears in 30%+ of incidents
- Must NOT define stories
- Must NOT create core merges between unrelated incidents

### Adversary 2: Scope Pollution
- Same `question_key=policy_announcement` in two unrelated scopes
- Must produce SEPARATE surfaces
- `(scope_id, question_key)` invariant holds

### Acceptance Criteria
- [ ] Hub entity in 30%+ incidents → 0 stories defined by hub
- [ ] Scope pollution → surfaces isolated by scope
- [ ] No mega-case formation (max case size < 50)
- [ ] Invariants:
  - `no_hub_story_definition`
  - `scoped_surface_isolation`
  - `max_case_size_below(50)`

### Stop/Go Gate
```bash
pytest backend/reee/tests/integration/test_adversary_archetypes.py -v
# Must pass: test_hub_cannot_define_story, test_scope_isolation, test_no_megacase
```

### Deliverables
- `fixtures/golden_adversary_hub.yaml`
- `fixtures/golden_adversary_scope.yaml`
- `reee/tests/integration/test_adversary_archetypes.py`

---

## Milestone 5: Macro Corpus Generator (~1000 claims)
**Goal:** Systematic coverage of all 8 archetypes.

### Prerequisites
- Milestones 1-4 complete and green
- Test harness proven stable

### 8 Archetypes
| # | Archetype | Claims | Key Invariants |
|---|-----------|--------|----------------|
| 1 | Star Story (WFC) | 150 | core_leak=0, 1 story per spine |
| 2 | Dyad Story (Do Kwon + Terraform) | 80 | multi-spine promotion, PMI>2 |
| 3 | Hub Adversary | 200 | hub never defines story |
| 4 | Homonym Adversary | 60 | disambiguation, no merge |
| 5 | Scope Pollution | 100 | surface isolation |
| 6 | Time Missingness (50%) | 120 | conservative blocking |
| 7 | Typed Conflicts | 90 | Jaynes posterior, conflicts flagged |
| 8 | Related Storyline | 100 | RELATED_STORY link, not member |

### Acceptance Criteria
- [ ] `golden_corpus_generator.py` with `seed=42`
- [ ] `corpus_manifest.yaml` with:
  - Scenarios + expected invariants
  - Named anchors/stories with expected memberships
  - Quantitative envelopes (ranges, not exact values)
- [ ] All 8 archetypes pass their invariants
- [ ] Aggregate quantitative report:
  - `stories_range: [40, 80]`
  - `periphery_rate: [0.05, 0.25]`
  - `witness_scarcity: < 0.40`
  - `max_case_size: < 50`

### Stop/Go Gate
```bash
pytest backend/reee/tests/integration/test_macro_corpus.py -v
# Must pass: test_all_archetypes_invariants, test_quantitative_bounds
```

### Deliverables
- `golden_macro/corpus_generator.py`
- `golden_macro/corpus_manifest.yaml`
- `golden_macro/corpus.json` (generated, gitignored)
- `reee/tests/integration/test_macro_corpus.py`

---

## Milestone 6: Replay Snapshots
**Goal:** Freeze real-world slices for regression testing.

### What Gets Recorded
- Kernel INPUTS: incidents, anchors, time windows, constraints
- NOT raw pages or HTML

### Use Cases
- "WFC leak" snapshot - ensure fix doesn't regress
- "Jimmy Lai trial" snapshot - multi-mode story
- "Hong Kong policy" snapshot - hub pressure

### Acceptance Criteria
- [ ] `record_snapshot(kernel_inputs)` → JSON file
- [ ] `replay_snapshot(path)` → same topology
- [ ] Diff tool shows: which decisions changed, why
- [ ] CI runs replay tests on every PR

### Stop/Go Gate
```bash
pytest backend/reee/tests/integration/test_replay_snapshots.py -v
# Must pass: test_wfc_replay_deterministic, test_diff_explains_changes
```

### Deliverables
- `reee/tests/replay/recorder.py`
- `reee/tests/replay/replayer.py`
- `fixtures/replay_wfc_snapshot.json` (already exists, wire it up)
- `reee/tests/integration/test_replay_snapshots.py`

---

## Deterministic ID Scheme

Use in test generator first, align production later.

```python
# Stable, content-based IDs
claim_id    = sha1(corpus_id + idx + text)[:12]
entity_id   = sha1(canonical_name)[:12]
scope_id    = sha1(sorted(nonhub_anchors))[:12]
surface_id  = sha1(scope_id + question_key)[:12]
incident_id = sha1(sorted(surface_ids) + mode_bin)[:12]
story_id    = sha1(focal_set + mode_id)[:12]
```

---

## Timeline Estimate

| Milestone | Days | Cumulative |
|-----------|------|------------|
| 1. Neo4j Substrate | 1 | 1 |
| 2. Star Story | 1 | 2 |
| 3. Retrieval Completeness | 0.5-1 | 3 |
| 4. Adversary Archetypes | 1 | 4 |
| 5. Macro Generator | 2-3 | 6-7 |
| 6. Replay Snapshots | 1-2 | 7-9 |

**Total: ~7-9 days for complete kernel validation environment**

---

## Current Status

- [x] Plan document created
- [x] **Milestone 1: Test Neo4j Substrate** ✓ (5 tests pass, 12.17s)
- [x] **Milestone 2: Star Story Archetype** ✓ (10 tests pass, 0.14s)
- [x] **Milestone 3: Retrieval Completeness** ✓ (7 tests pass, 0.11s)
- [x] **Milestone 4: Adversary Archetypes** ✓ (13 tests pass, 0.08s)
- [x] **Milestone 5: Macro Corpus Generator** ✓ (17 tests pass, 1.94s)
- [x] **Milestone 6: Replay Snapshots** ✓ (16 tests pass, 0.20s)

**ALL MILESTONES COMPLETE** - Total: 68 tests across 6 test modules

---

## Post-Milestone: Persistent Topology Validation

### Neo4j Persistence + Topology Semantics Test
- `scripts/load_corpus_to_neo4j.py` - Load macro corpus for persistent inspection
- `test_topology_semantics.py` - 19 tests validating graph structure

### How to Use

```bash
# Load corpus into test Neo4j (persists data)
docker exec herenews-app python /app/reee/tests/scripts/load_corpus_to_neo4j.py

# Run topology semantics tests (requires loaded data)
docker exec herenews-app python -m pytest /app/reee/tests/integration/test_topology_semantics.py -v

# Query test Neo4j directly
docker exec herenews-neo4j-test cypher-shell -u neo4j -p test_password "MATCH (st:Story) RETURN st.spine, st.core_a_count ORDER BY st.core_a_count DESC"
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Structural Invariants | 4 | Every story has spine, incidents have anchors, etc. |
| Archetype Behavior | 5 | Star story, dyad, hub detection, scope isolation |
| Story Quality | 4 | No mega-cases, leak rates, coverage |
| Graph Connectivity | 3 | Entity-incident, claim-surface paths |
| Archetype Statistics | 2 | All 8 archetypes present with counts |
| Summary | 1 | Print topology overview |

**Total: 87 tests** (68 base + 19 topology semantics)

---

## Phase 2: Real Kernel Validation with Decision Traces

### Acceptance Checkpoint
After running kernel_validator.py, you can answer "why is incident X in story Y?" by following stored MEMBERSHIP_DECISION edges in Neo4j - not by re-running code.

### Isolated Test Infrastructure
Uses `docker-compose.test.yml` with isolated test-runner container that ONLY has test Neo4j credentials. Production databases cannot be contaminated.

### How to Use (Isolated)

```bash
# Start isolated test environment
docker-compose -f docker-compose.test.yml up -d

# Load corpus and run kernel with decision traces
docker exec herenews-test-runner python -m reee.tests.scripts.load_corpus_to_neo4j
docker exec herenews-test-runner python -m reee.tests.scripts.kernel_validator

# Run hard invariant tests (MUST ALWAYS PASS)
docker exec herenews-test-runner python -m pytest reee/tests/integration/test_hard_invariants.py -v

# Run soft envelope tests (warnings only)
docker exec herenews-test-runner python -m pytest reee/tests/integration/test_soft_envelopes.py -v

# Run retrieval completeness tests
docker exec herenews-test-runner python -m pytest reee/tests/integration/test_retrieval_completeness.py::TestNeo4jRetrievalCompleteness -v

# Query decision traces directly
docker exec herenews-neo4j-test cypher-shell -u neo4j -p test_password "
MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st:Story)
RETURN i.id, st.spine, d.membership, d.core_reason, d.blocked_reason
LIMIT 5
"
```

### Decision Trace Schema
Per candidate:
- `candidate_id`, `target_id` (story)
- `membership`: CORE_A, CORE_B, PERIPHERY, REJECT
- `core_reason`: ANCHOR, WARRANT, null
- `witnesses`: [constraint_ids]
- `blocked_reason`: string or null
- `timestamp`, `kernel_version`, `params_hash`

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| **Hard Invariants** | 15 | MUST ALWAYS PASS - kernel safety guarantees |
| Scoped Surface Isolation | 2 | (scope_id, question_key) uniquely identifies surface |
| Semantic-Only Cannot Core | 2 | CORE requires structural link, not just semantic |
| Chain-Only Cannot Merge | 1 | Single witness chain insufficient for CORE_B |
| Core Leak Rate | 2 | Leak rate correctly computed and bounded |
| Blocked Reasons Visible | 3 | All non-core candidates have blocked_reason |
| Hub Cannot Define Story | 1 | Hub entities don't become story spines |
| Decision Trace Complete | 3 | All traces have version, timestamp, params_hash |
| Summary | 1 | Print invariant summary |
| **Soft Envelopes** | 6 | Warnings only - parameter tuning indicators |
| Story Count Range | 1 | 20-80 stories expected |
| Max Core Size | 1 | No core larger than 30 |
| Hub Count | 1 | 1-10 hubs expected |
| Periphery Rate | 1 | 5-40% periphery expected |
| Witness Scarcity | 1 | <50% CORE_B without witnesses |
| Summary | 1 | Print envelope summary |
| **Retrieval Completeness** | 5 | Bounded retrieval doesn't miss core members |
| Entity Retrieval | 2 | CORE_A retrievable via spine entity |
| Budget Sufficiency | 1 | Top-k covers all cores |
| Graph Connectivity | 1 | All incidents reachable from entities |
| Context Gap Detection | 1 | PERIPHERY has blocked_reason |

**Total: 113 tests** (68 base + 19 topology + 26 decision traces)
