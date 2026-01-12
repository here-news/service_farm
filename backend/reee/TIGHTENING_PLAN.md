# REEE Module Tightening Plan

## Current State Analysis

Based on file audit, the package has ~80 Python files across:
- Core types/builders
- Multiple deprecated paths
- Experiments
- Tests
- Views

## Import Consumer Map

### Workers (Production)
```
canonical_worker.py:
  - from reee.types import Surface, Event, EntityLens
  - from reee.builders import PrincipledSurfaceBuilder, PrincipledEventBuilder, MotifConfig
  - from reee.builders.story_builder import StoryBuilder, StoryBuilderResult
  - from reee.types import EventJustification (conditional)

claim_loader.py:
  - from reee.types import Claim
```

### Debug Scripts (Non-production)
```
test_case_dispersion.py:
  - from reee.views.case_view import CaseViewBuilder, EntityRole
  - from reee.types import Event, Surface

test_event_justification.py:
  - from reee.builders import PrincipledEventBuilder, PrincipledSurfaceBuilder
  - from reee.types import Claim

weave_viz.py:
  - from reee.tests.test_golden_trace import GoldenTrace, TraceKernel (conditional)
```

### Safe to Fence (Not imported by production)
These are NOT imported by workers/endpoints:
- engine.py
- kernel.py
- comparator.py
- extractor.py
- event_interpreter.py
- event_weaver.py
- topology.py
- force_field.py
- evaluation.py
- experiment.py
- interpretation.py
- belief_state.py
- visualize.py
- aboutness/ (entire directory)

## Target Structure

```
backend/reee/
├── __init__.py          # Minimal public API (see below)
├── types.py             # Stable: Surface, Incident, Story, Lens, Constraint
├── membrane.py          # Pure decision tables (Membership, CoreReason, classify_*)
├── typed_belief.py      # Jaynes inference (pure)
│
├── builders/
│   ├── __init__.py
│   ├── surface_builder.py    # KEEP: PrincipledSurfaceBuilder
│   ├── incident_builder.py   # RENAME from event_builder.py
│   ├── story_builder.py      # KEEP: StoryBuilder (authoritative L4)
│   └── lens_builder.py       # NEW: Entity profile/lens (no membrane)
│
├── inference/
│   └── typed_belief.py       # MOVE from root (or keep at root)
│
├── validation/
│   ├── __init__.py
│   ├── invariants.py         # Hard invariants (from tests)
│   └── bounds.py             # Soft envelopes (from tests)
│
├── tests/                    # Keep as-is, well-organized
│
├── deprecated/                   # NEW: Deprecated code graveyard
│   ├── aboutness/            # MOVE from root
│   ├── engine.py             # MOVE
│   ├── kernel.py             # MOVE
│   ├── comparator.py         # MOVE
│   ├── extractor.py          # MOVE
│   ├── event_interpreter.py  # MOVE
│   ├── event_weaver.py       # MOVE
│   ├── topology.py           # MOVE
│   ├── force_field.py        # MOVE
│   └── case_builder.py       # MOVE from builders/
│
└── experiments/              # Keep, already fenced
```

## Module Classification

### KERNEL (Pure, No DB/LLM imports)

| Module | Status | Action |
|--------|--------|--------|
| `types.py` | KEEP | Stable types |
| `membrane.py` | KEEP | Decision tables (already pure) |
| `typed_belief.py` | KEEP | Jaynes (pure) |
| `builders/story_builder.py` | KEEP | Authoritative L4 |
| `builders/surface_builder.py` | KEEP | L2 builder |
| `builders/event_builder.py` | RENAME | → `incident_builder.py` |
| `identity/question_key.py` | KEEP | Pure keying |

### ATTIC (Move to `deprecated/`)

| Module | Reason |
|--------|--------|
| `engine.py` | Legacy structure builder |
| `kernel.py` | Old belief kernel |
| `comparator.py` | Unused |
| `extractor.py` | LLM-coupled |
| `event_interpreter.py` | Legacy |
| `event_weaver.py` | Legacy |
| `topology.py` | Legacy visualization |
| `force_field.py` | Unused |
| `epistemic_unit.py` | Superseded by types.py |
| `evaluation.py` | Legacy |
| `experiment.py` | One-off |
| `interpretation.py` | Superseded by typed_belief |
| `belief_state.py` | Superseded by typed_belief |
| `visualize.py` | Dev-only |
| `aboutness/` | Superseded by membrane |
| `builders/case_builder.py` | Superseded by story_builder |
| `adapters/` | Worker glue |
| `repositories/` | DB layer |

### KEEP AS INTERNAL

| Module | Reason |
|--------|--------|
| `views/` | Explicit projections (may migrate to lens_builder) |
| `meta/detectors.py` | Quality signals |
| `inquiry/` | Seeder system (active) |
| `identity/linker.py` | Entity resolution |

### EXPERIMENTS (Already fenced)

All files under `experiments/` - no action needed.

## Vocabulary Collapse

### Current Confusion

| Old Term | New Term | Notes |
|----------|----------|-------|
| Event | Incident | L3 |
| Case/EntityCase | Story | L4 |
| Anchor/Companion | (keep) | Within Incident |
| Claim → Surface | (keep) | L1 → L2 |
| Story.scale="incident" | DELETE | Use type Incident |
| Story.scale="case" | DELETE | Use type Story |

### Implementation

1. Keep `Event` as internal computation type
2. Rename `PrincipledEventBuilder` → `IncidentBuilder`
3. Remove `StoryScale` enum
4. Add `Incident` as distinct type (not just Story with scale)

## Public API (`__init__.py`)

### EXPORT (Stable)

```python
# Types
from .types import Claim, Surface, Incident, Story, Lens
from .types import Constraint, ConstraintType

# Membrane
from .membrane import (
    Membership, CoreReason, LinkType,
    MembershipDecision,
    classify_incident_membership,
)

# Builders (pure)
from .builders import (
    PrincipledSurfaceBuilder,
    IncidentBuilder,  # renamed from PrincipledEventBuilder
    StoryBuilder,
)

# Inference
from .typed_belief import TypedBeliefState
```

### DO NOT EXPORT

- Engine, EmergenceEngine
- AboutnessScorer
- ClaimExtractor, ClaimComparator
- Views (internal)
- Workers, repositories, experiments

## Centralize Membership Decisions

### Rule

Only `membrane.classify_*` decides CORE/PERIPHERY/REJECT.

### Implementation

1. `StoryBuilder.build_from_incidents()` calls `membrane.classify_incident_membership()`
2. Remove any fallback thresholds in builders
3. Add: `classify_surface_membership()` for future Surface → Incident

### Constraint Source Enforcement

```python
# In membrane.py
def is_valid_structural_constraint(c: Constraint) -> bool:
    """
    A constraint counts as structural witness ONLY if:
    1. kind in WITNESS_KINDS (time, geo, event_type, motif, context)
    2. source != "semantic_proposal"
    """
    if c.source == "semantic_proposal":
        return False  # Never structural, even if kind="geo"
    return c.kind in WITNESS_KINDS
```

## Decision Traces as First-Class

### Current

Decision traces only exist in Neo4j (via kernel_validator).

### Target

```python
@dataclass
class DecisionTrace:
    candidate_id: str
    target_id: str
    decision: MembershipDecision
    constraints_used: List[str]
    params_hash: str
    timestamp: str

@dataclass
class StoryBuilderResult:
    stories: Dict[str, CompleteStory]
    spines: Dict[str, SpineData]
    traces: List[DecisionTrace]  # NEW: Always populated
```

This makes non-DB runs explainable.

## Implementation Order (Safe Sequence)

### Phase 1: Fence + Deprecate (Zero breakage risk)

**Do NOT move files yet. Add deprecation banners + stop exporting.**

1. Add deprecation banners to legacy modules (no imports change):
   ```python
   # engine.py
   """
   DEPRECATED: Use reee.builders.story_builder instead.
   This module is scheduled for removal in v2.0.
   """
   import warnings
   warnings.warn("reee.engine is deprecated, use reee.builders", DeprecationWarning)
   ```

2. Stop exporting legacy from `reee/__init__.py`:
   - Remove: Engine, EmergenceEngine, AboutnessScorer, ClaimExtractor, etc.
   - Keep: types, builders, membrane (already clean)

3. Add membrane to public exports:
   ```python
   from .membrane import (
       Membership, CoreReason, LinkType,
       MembershipDecision, classify_incident_membership,
   )
   ```

### Phase 2: Clean Worker Imports (One PR)

1. canonical_worker.py already uses clean imports:
   - types: Surface, Event, Claim ✓
   - builders: story_builder ✓

2. Remove conditional EntityCase import:
   - Replace with CompleteStory or remove dead code

3. Fix broken debug scripts:
   - test_event_justification.py uses non-existent `reee.canonical` path

### Phase 3: Physical Move to deprecated (After imports clean)

Only after nothing imports them:
```bash
mkdir backend/reee/deprecated
mv engine.py kernel.py comparator.py extractor.py \
   event_interpreter.py event_weaver.py topology.py \
   force_field.py evaluation.py experiment.py \
   interpretation.py belief_state.py visualize.py \
   aboutness/ \
   backend/reee/deprecated/
```

### Phase 4: Vocabulary Rename (Cosmetic, last)

1. Rename `PrincipledEventBuilder` → `IncidentBuilder`
2. Keep `Event` as internal computation type
3. Update Neo4j labels if needed (separate migration)

### Phase 5: Decision Traces in Memory

1. Define `DecisionTrace` dataclass once in membrane.py:
   ```python
   @dataclass
   class DecisionTrace:
       candidate_id: str
       target_id: str
       decision: MembershipDecision
       constraints_used: List[str]
       params_hash: str
   ```

2. Add to `StoryBuilderResult`:
   ```python
   @dataclass
   class StoryBuilderResult:
       stories: Dict[str, CompleteStory]
       spines: Dict[str, SpineData]
       traces: List[DecisionTrace] = field(default_factory=list)  # NEW
   ```

3. Reuse for Neo4j persistence (kernel_validator) and tests

## Quick Wins (Do First)

1. **Lint test**: Add `test_kernel_isolation.py` that fails if kernel modules import neo4j/psycopg/openai ✅ DONE
2. **Quarantine legacy implementations**: move real code into `deprecated/_*.py`, keep import stubs in place ✅ DONE
3. **Export membrane**: Add membrane types to public API ✅ DONE
4. **Rename test**: Update terminology in test names (Event → Incident)

## Completed (2026-01-05)

### Phase 1-2: Deprecation + Clean Imports ✅

1. **EntityLens type created** (`types.py`)
   - Immutable (`frozen=True`) replacement for `EntityCase`
   - Uses `frozenset` and `tuple` for immutability
   - Factory method `EntityLens.create()` for construction

2. **`to_lens()` method added** to `CompleteStory` (`story_builder.py`)
   - Returns immutable `EntityLens`
   - `to_entity_case()` deprecated with warning

3. **`canonical_worker.py` migrated**
   - Now uses `to_lens()` + `EntityLens.create()`
   - No more mutability leak (companions built externally)
   - No deprecated imports

4. **ALL deprecated modules quarantined** with stubs:
   - `builders/case_builder.py` now stubs → `deprecated/_case_builder.py`
   - `engine.py` now stubs → `deprecated/_engine.py`
   - `kernel.py` now stubs → `deprecated/_kernel.py`
   - `interpretation.py` now stubs → `deprecated/_interpretation.py`
   - `extractor.py` now stubs → `deprecated/_extractor.py`
   - `comparator.py` now stubs → `deprecated/_comparator.py`
   - `aboutness/*.py` now stubs → `deprecated/_aboutness/*.py`

   Each stub:
   - Emits `DeprecationWarning` on import
   - Logs warning for production visibility
   - `REEE_STRICT_DEPRECATIONS=1` makes import fail after removal date
   - Re-exports from deprecated/ for backward compatibility

5. **Enforcement tests added**
   - `test_no_deprecated_imports.py` (4 tests)
   - `test_kernel_isolation.py` (11 tests)
   - `test_deprecation_warnings.py` (6 tests)

### Removal Schedule

| Module | Deprecated | Removal |
|--------|------------|---------|
| `builders/case_builder.py` | 2026-01-05 | 2026-02-01 |
| `engine.py` | 2026-01-05 | 2026-02-01 |
| `kernel.py` | 2026-01-05 | 2026-02-01 |
| `interpretation.py` | 2026-01-05 | 2026-02-01 |
| `extractor.py` | 2026-01-05 | 2026-02-01 |
| `comparator.py` | 2026-01-05 | 2026-02-01 |
| `aboutness/` | 2026-01-05 | 2026-02-01 |
| `story_builder.to_entity_case()` | 2026-01-05 | 2026-02-01 |

See `deprecated/RELIC.md` for full migration guide.

## Remaining Work

### Phase 3: Migrate Internal Consumers

1. **`views/incident.py`** still *uses* deprecated aboutness scoring for edge computation:
   - Import is now lazy (warning only occurs if `compute_edges()` is called) ✅
   - Full migration requires refactoring to membrane-based scoring (deferred)

2. **`builders/__init__.py`** still exports `PrincipledCaseBuilder`:
   - Keep for backward compatibility until removal date
   - Now lazy-loaded to avoid warnings on `import reee.builders` ✅

3. **`reee/__init__.py`** still exports deprecated symbols:
   - `Engine`, `EmergenceEngine`
   - `AboutnessScorer`, `compute_aboutness_edges`, `compute_events_from_aboutness`
   - `interpret_all`, `interpret_surface`, `interpret_event`
   - `ClaimExtractor`, `ExtractedClaim`
   - `ClaimComparator`, `ComparisonRelation`
   - `EpistemicKernel`, `Belief`
   - Keep until removal date, then remove
   - Now lazy-loaded to avoid warnings on `import reee` ✅

### Enforcement Tests Added ✅

- `test_no_structure_from_deprecated.py` (8 tests)
  - Ensures kernel modules (types, membrane, typed_belief, builders/*) don't import deprecated code
  - Only checks top-level imports (allows transitional imports inside deprecated methods)
  - Verifies RELIC.md documents all deprecated modules

### Phase 4: Vocabulary Rename (Cosmetic, last)

1. Rename `PrincipledEventBuilder` → `IncidentBuilder`
2. Keep `Event` as internal computation type
3. Update Neo4j labels if needed (separate migration)

### Phase 5: Decision Traces in Memory

Already defined in plan - implement when needed.

## Metrics After Tightening

| Metric | Before | Target |
|--------|--------|--------|
| Root-level .py files | ~20 | 5 (types, membrane, typed_belief, __init__, constraints) |
| builders/ files | 4 | 4 (surface, incident, story, lens) |
| deprecated/ files | 0 | ~15 |
| Public exports | ~80 | ~20 |
| DB-importing kernel modules | unknown | 0 |

## Soft Envelopes Stay Soft

Do NOT tune these down as goals:
- periphery_rate (43.8%) - diagnostic for witness scarcity
- hub_count (1) - may need hub detection tuning

Track as time series across builds, not targets.
