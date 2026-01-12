# Kernel Validation Environment Plan

## Goals

1. **Deterministic testing** - Same inputs → same topology + same decisions
2. **Invariant-based assertions** - Not brittle exact snapshots
3. **Local Neo4j** - Full graph testing without production dependency
4. **Replay capability** - Record once, replay forever
5. **Explanation audit** - Every decision has traceable reasoning

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Validation Environment                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Golden Micro │    │ Golden Macro │    │   Replay     │       │
│  │   Traces     │    │   Corpus     │    │  Snapshots   │       │
│  │  (YAML/JSON) │    │ (~1000 claims)│    │ (real slices)│       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             ▼                                    │
│                   ┌──────────────────┐                          │
│                   │   Test Neo4j     │                          │
│                   │   (Containerized)│                          │
│                   └────────┬─────────┘                          │
│                            │                                     │
│         ┌──────────────────┼──────────────────┐                 │
│         ▼                  ▼                  ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   L2/L3/L4  │   │  Membrane   │   │ Retrieval   │           │
│  │   Builders  │   │  Decisions  │   │ Completeness│           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                     │
│                   ┌──────────────────┐                          │
│                   │   Invariant      │                          │
│                   │   Assertions     │                          │
│                   └──────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Local Neo4j Test Infrastructure

### 1.1 Docker Compose for Test Neo4j

```yaml
# docker-compose.test.yml
services:
  neo4j-test:
    image: neo4j:5.15-community
    ports:
      - "7688:7687"  # Different port from production
      - "7475:7474"
    environment:
      NEO4J_AUTH: neo4j/test_password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j-test-data:/data
    tmpfs:
      - /logs  # Ephemeral logs for speed

volumes:
  neo4j-test-data:
```

### 1.2 Test Database Manager

```python
# reee/tests/db/test_neo4j.py
class TestNeo4jManager:
    """Manages test Neo4j lifecycle."""

    async def setup_fresh(self):
        """Clear and initialize test database."""
        await self.clear_all()
        await self.create_indexes()

    async def load_fixture(self, fixture_path: str):
        """Load deterministic fixture into Neo4j."""
        # Load claims, surfaces, incidents in deterministic order
        # Use hash-based IDs for stability

    async def snapshot(self) -> dict:
        """Export current state for comparison."""
        # Returns ordered dict of all nodes/relationships
```

## Phase 2: Golden Corpus Design

### 2.1 Story Archetypes (8 patterns)

| Archetype | Description | Claims | Expected Invariants |
|-----------|-------------|--------|---------------------|
| **Star Story** | Spine + rotating companions (WFC pattern) | 150 | 1 story, multiple modes, facet gaps emitted |
| **Dyad Story** | Two-spine interaction (Do Kwon + Terraform) | 80 | Multi-spine promotion succeeds, PMI > 2.0 |
| **Hub Adversary** | Ubiquitous entities (Hong Kong, USA) | 200 | Never define stories, only lenses/periphery |
| **Homonym Adversary** | Same name, different entities | 60 | Disambiguation, no accidental merge |
| **Scope Pollution** | Same question_key across incidents | 100 | Scoped surfaces never shared |
| **Time Missingness** | 50% claims missing timestamps | 120 | Conservative blocking, blockers emitted |
| **Typed Conflicts** | Numeric conflicts with supersession | 90 | Jaynes posterior evolves, conflicts flagged |
| **Related Storyline** | Adjacent but not member | 100 | RELATED_STORY link, not membership |

**Total: ~900 claims + 100 edge cases = ~1000 claims**

### 2.2 Fixture Schema

```yaml
# golden_macro/corpus_manifest.yaml
corpus:
  version: "1.0"
  seed: 42  # For reproducible generation

archetypes:
  - name: star_story_wfc
    template: star_story
    params:
      spine: "Wang Fuk Court"
      companion_pool: ["Fire Services", "John Lee", "Chris Tang", ...]
      facets: [fire_death_count, fire_injury_count, fire_cause, ...]
      time_modes: 2
      claims_per_mode: 75

    expected:
      stories: 1
      modes: 2
      core_incidents_range: [20, 40]
      periphery_incidents_range: [0, 10]
      facet_gaps: [fire_status]  # Expected missing

  - name: hub_adversary_hk
    template: hub_adversary
    params:
      hub_entity: "Hong Kong"
      appearance_fraction: 0.35  # 35% of all incidents

    expected:
      stories_defined_by_hub: 0  # Must be zero!
      hub_blocked_count_min: 50

invariants:
  safety:
    - no_semantic_only_core
    - no_chain_percolation
    - core_leak_rate_zero
    - scoped_surface_isolation

  quantitative:
    - stories_range: [15, 30]
    - periphery_rate_range: [0.05, 0.20]
    - witness_scarcity_max: 0.40
```

### 2.3 Invariant Assertion Framework

```python
# reee/tests/invariants/kernel_invariants.py

class KernelInvariants:
    """Invariants that must always hold."""

    @staticmethod
    def no_semantic_only_core(stories: List[CompleteStory]) -> bool:
        """Core membership requires structural witness."""
        for story in stories:
            for inc_id in story.core_b_ids:
                decision = story.membrane_decisions.get(inc_id)
                if decision and not decision.witnesses:
                    return False
        return True

    @staticmethod
    def no_hub_story_definition(stories: List[CompleteStory],
                                 hub_entities: Set[str]) -> bool:
        """Hub entities cannot define stories."""
        for story in stories:
            if story.spine in hub_entities:
                return False
        return True

    @staticmethod
    def scoped_surface_isolation(surfaces: Dict[str, Surface]) -> bool:
        """Same (scope_id, question_key) never shared across scopes."""
        seen = {}
        for surf in surfaces.values():
            key = (surf.scope_id, surf.question_key)
            if key in seen and seen[key] != surf.id:
                return False
            seen[key] = surf.id
        return True
```

## Phase 3: Test Structure

### 3.1 Directory Layout

```
reee/tests/
├── golden_micro/           # Hand-authored traces (existing)
│   ├── bridge_immunity.yaml
│   ├── typed_conflict.yaml
│   └── companion_incompatibility.yaml
│
├── golden_macro/           # Generated corpus
│   ├── corpus_manifest.yaml
│   ├── generator.py        # Deterministic generator
│   ├── archetypes/
│   │   ├── star_story.py
│   │   ├── dyad_story.py
│   │   ├── hub_adversary.py
│   │   └── ...
│   └── expected/
│       ├── wfc_story.yaml  # Named scenario expectations
│       └── ...
│
├── replay_snapshots/       # Frozen real-world slices
│   ├── wfc_fire_2025.json
│   └── jimmy_lai_trial.json
│
├── db/                     # Neo4j test infrastructure
│   ├── test_neo4j.py
│   ├── fixtures/
│   │   └── full_corpus.cypher
│   └── conftest.py         # pytest fixtures
│
├── invariants/             # Invariant assertion library
│   ├── kernel_invariants.py
│   ├── scenario_invariants.py
│   └── quantitative_bounds.py
│
└── integration/            # Full-stack tests with Neo4j
    ├── test_full_weave.py
    ├── test_retrieval_completeness.py
    └── test_explanation_audit.py
```

### 3.2 Pytest Integration

```python
# reee/tests/db/conftest.py

@pytest.fixture(scope="session")
async def test_neo4j():
    """Session-scoped test Neo4j connection."""
    manager = TestNeo4jManager(
        uri="bolt://localhost:7688",
        user="neo4j",
        password="test_password"
    )
    await manager.connect()
    yield manager
    await manager.close()

@pytest.fixture
async def fresh_db(test_neo4j):
    """Per-test fresh database."""
    await test_neo4j.setup_fresh()
    yield test_neo4j

@pytest.fixture
async def corpus_db(test_neo4j):
    """Database loaded with golden corpus."""
    await test_neo4j.setup_fresh()
    await test_neo4j.load_fixture("golden_macro/corpus.json")
    yield test_neo4j
```

## Phase 4: Explanation Audit

### 4.1 Deterministic Explanations

```python
# Explanation tokens (no LLM, deterministic)
EXPLANATION_TOKENS = {
    "CORE_A": "spine '{spine}' is anchor in incident",
    "CORE_B": "2+ structural witnesses: {witnesses}",
    "CORE_B_BLOCKED": "insufficient witnesses: {reason}",
    "PERIPHERY": "semantic-only, no structural binding",
    "REJECT_HUB": "only hub anchors shared: {hubs}",
    "REJECT_UNRELATED": "no focal overlap",
}

def explain_decision(decision: MembershipDecision) -> str:
    """Generate deterministic explanation."""
    token = ...  # Select based on decision
    return EXPLANATION_TOKENS[token].format(**decision.__dict__)
```

### 4.2 Audit Trail Assertions

```python
def test_every_decision_has_explanation(corpus_db, stories):
    """Every membership decision must have traceable reasoning."""
    for story in stories:
        for inc_id, decision in story.membrane_decisions.items():
            assert decision.membership is not None
            if decision.membership == Membership.CORE:
                assert decision.core_reason is not None
            if decision.membership == Membership.PERIPHERY:
                assert decision.blocked_reason is not None
```

## Phase 5: Implementation Steps

### Step 1: Neo4j Test Infrastructure (Day 1)
- [ ] Create `docker-compose.test.yml`
- [ ] Implement `TestNeo4jManager`
- [ ] Add pytest fixtures for Neo4j lifecycle
- [ ] Verify container startup/teardown

### Step 2: Corpus Generator (Day 2-3)
- [ ] Define archetype templates
- [ ] Implement deterministic claim generator
- [ ] Generate ~1000 claims with seed=42
- [ ] Create corpus manifest with expected invariants

### Step 3: Invariant Framework (Day 3-4)
- [ ] Implement `KernelInvariants` class
- [ ] Add scenario-specific invariant assertions
- [ ] Add quantitative bounds checking
- [ ] Wire into pytest

### Step 4: Full Integration Tests (Day 4-5)
- [ ] `test_full_weave.py` - End-to-end kernel execution
- [ ] `test_retrieval_completeness.py` - No missed candidates
- [ ] `test_explanation_audit.py` - All decisions explained

### Step 5: CI Integration (Day 5)
- [ ] Add Neo4j service to CI workflow
- [ ] Run kernel validation on PR
- [ ] Cache corpus for speed

## Success Criteria

1. **Determinism**: Running tests 10x produces identical results
2. **Invariant coverage**: All 8 archetypes have passing invariant tests
3. **No brittleness**: Kernel changes don't require golden output rewrites (unless intentional)
4. **Explanability**: 100% of decisions have audit trail
5. **Speed**: Full corpus test completes in < 60 seconds

## Notes

- **No LLM in tests**: All explanations are template-based
- **Stable IDs**: All IDs derived from content hash, not insertion order
- **Ordered queries**: All Neo4j queries use explicit ORDER BY
- **Seed everything**: Random choices use fixed seed for reproducibility
