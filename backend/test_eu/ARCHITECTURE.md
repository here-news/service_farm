# Epistemic Kernel Architecture

## Current State (Messy)

40+ Python files with overlapping implementations. Need consolidation.

## Target Architecture

```
test_eu/
├── core/                      # Core kernel components
│   ├── __init__.py
│   ├── kernel.py              # Main EpistemicKernel class (from kernel_complete.py)
│   ├── extractor.py           # Structured value extraction (from relate_updates.py)
│   ├── comparator.py          # Claim comparison (q1/q2 from universal_kernel.py)
│   └── types.py               # Shared types: Claim, Belief, Aspect, Rule
│
├── quality/                   # Quality evaluation
│   ├── __init__.py
│   ├── evaluator.py           # Quality scoring (from EPISTEMIC_QUALITY.md)
│   ├── jaynes.py              # Jaynes alignment checks
│   └── topology.py            # Belief topology visualization
│
├── reporters/                 # Output generation
│   ├── __init__.py
│   ├── narrator.py            # Epistemic narrator (what we know/don't know)
│   └── prose.py               # LLM prose generation
│
├── server/                    # API server
│   ├── __init__.py
│   └── api.py                 # FastAPI endpoints
│
├── tests/                     # Test cases
│   └── test_kernel.py
│
├── _archive/                  # Deprecated experiments (move old files here)
│
└── docs/                      # Documentation
    ├── JAYNES_ALIGNMENT.md
    ├── EPISTEMIC_QUALITY.md
    └── SEMANTIC_PATTERNS.md
```

## Key Components

### 1. Extractor (Priority 1)
Consolidates extraction from:
- `relate_updates.py`: `extract_structure()` → attrs, temporal_markers
- `universal_kernel.py`: q1/q2 question extraction
- `uee_server.py`: `numeric_value`, `is_monotonic`

```python
# core/extractor.py
class ClaimExtractor:
    def extract(self, text: str) -> ExtractedClaim:
        return ExtractedClaim(
            attrs={'death_toll': 160, 'location': 'Wang Fuk Court'},
            numeric_value=160.0,
            temporal_markers=['rises to'],
            question='count of deaths',
            is_update=True
        )
```

### 2. Comparator (Priority 2)
Uses q1/q2 pattern from `universal_kernel.py`:

```python
# core/comparator.py
class ClaimComparator:
    def compare(self, c1: ExtractedClaim, c2: ExtractedClaim) -> Relation:
        if c1.question != c2.question:
            return Relation.NOVEL  # Different questions = no relation
        if c1.numeric_value == c2.numeric_value:
            return Relation.CONFIRMS
        if c2.is_update and c2.numeric_value > c1.numeric_value:
            return Relation.SUPERSEDES
        return Relation.DIVERGENT
```

### 3. Kernel (Priority 3)
Clean interface from `kernel_complete.py`:

```python
# core/kernel.py
class EpistemicKernel:
    def __init__(self):
        self.extractor = ClaimExtractor()
        self.comparator = ClaimComparator()
        self.state = EpistemicState()

    async def process_claim(self, text: str, source: str) -> ProcessResult:
        extracted = self.extractor.extract(text)
        # Find matching beliefs, compare, update state
        ...
```

## Files to Keep (Active)

| File | Purpose | Keep |
|------|---------|------|
| `kernel_complete.py` | Main kernel | → `core/kernel.py` |
| `relate_updates.py` | Extraction | → `core/extractor.py` |
| `universal_kernel.py` | q1/q2 pattern | → `core/comparator.py` |
| `uee_server.py` | API server | → `server/api.py` |
| `epistemic_narrator.py` | Reports | → `reporters/narrator.py` |
| `kernel_topology.py` | Visualization | → `quality/topology.py` |

## Files to Archive (Experimental)

Move to `_archive/`:
- `basic_form*.py` - early experiments
- `breathing_event.py` - not used
- `causal_emergence.py` - not used
- `cluster_mass.py` - not used
- `contribution_simulation.py` - simulation only
- `*_simulation.py` - simulations
- `jimmy_lai_*.py` - case-specific tests
- Files prefixed with `_trash_`

## Migration Plan

### Phase 1: Create core/extractor.py
1. Extract `extract_structure()` from `relate_updates.py`
2. Add q1/q2 question extraction from `universal_kernel.py`
3. Add `numeric_value` handling from `uee_server.py`
4. Update `kernel_complete.py` to import from extractor

### Phase 2: Create core/comparator.py
1. Extract comparison logic from `universal_kernel.py`
2. Integrate with kernel's `_classify_relationship()`

### Phase 3: Reorganize files
1. Create directory structure
2. Move active files to new locations
3. Move deprecated files to `_archive/`
4. Update imports

### Phase 4: Clean kernel_complete.py
1. Remove inline extraction (use extractor)
2. Remove inline comparison (use comparator)
3. Focus on state management only

## Quality Metrics Target

| Metric | Current | Target |
|--------|---------|--------|
| Extraction | 17% | 80% |
| Consolidation | 61% | 90% |
| Completeness | 100% | 100% |
| Coherence | 100% | 100% |
| Calibration | 100% | 100% |
| Traceability | 100% | 100% |
