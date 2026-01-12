# Relic: Pre-Declutter Snapshot

**Snapshot tag:** `relic/pre-declutter-20251231`

This document records the declutter performed on 2024-12-31, consolidating
the codebase after the REEE views implementation.

## What Was Removed/Ignored

### Ignored (local state, not committed)

| Path | Reason |
|------|--------|
| `.claude/` | Local Claude Code settings |
| `product/node_modules/` | npm dependencies |
| `product/test-results/` | Playwright test outputs |
| `backend/data/*.json`, `backend/data/*.csv` | Local data exports |

### Consolidated (duplicates resolved)

| Removed/Superseded | Canonical Location | Reason |
|--------------------|-------------------|--------|
| `backend/test_eu/core/aboutness/` | `backend/reee/aboutness/` | Duplicate module |
| `backend/test_eu/core/identity/` | `backend/reee/identity/` | Duplicate module |
| `backend/test_eu/core/meta/` | `backend/reee/meta/` | Duplicate module |
| `backend/test_eu/core/engine.py` | `backend/reee/engine.py` | Duplicate module |
| `backend/test_eu/core/types.py` | `backend/reee/types.py` | Duplicate module |
| `backend/test_eu/core/interpretation.py` | `backend/reee/interpretation.py` | Duplicate module |
| `backend/test_eu/REEE.md` | `backend/reee/REEE1.md` | Superseded docs |

**Note:** `backend/test_eu/core/__init__.py` is kept as a backward-compat shim
that re-exports from `reee`.

### Repository Structure After Declutter

```
backend/
├── api/                    # FastAPI HTTP layer
├── services/               # App-layer business logic (neo4j, event, entity, etc.)
├── repositories/           # App-layer DB adapters (claim, event, page, user, inquiry)
├── reee/                   # REEE epistemic library
│   ├── views/              # Multi-scale views (incident, case)
│   ├── identity/           # L0→L2 identity
│   ├── aboutness/          # L2→L3 aboutness
│   ├── meta/               # Meta-claims and tensions
│   ├── experiments/        # Runnable experiment scripts
│   ├── inquiry/            # Inquiry seeding/resolution
│   ├── repositories/       # REEE-specific repos (surface)
│   ├── REEE.md             # Original architecture doc
│   └── REEE1.md            # Views architecture (current)
├── test_eu/
│   └── core/__init__.py    # Backward-compat shim only
├── data/                   # Simulated inquiry data
├── models/                 # Pydantic models (api/domain)
└── scripts/                # Utility and experiment scripts
```

## How to Access Old State

```bash
# View the pre-declutter state
git show relic/pre-declutter-20251231

# Checkout to explore (detached HEAD)
git checkout relic/pre-declutter-20251231

# Return to current branch
git checkout feature/epistemic-webapp
```

## Key Architectural Decisions

1. **Views over Layers**: `backend/reee/views/` is the canonical path for
   event formation (IncidentEventView, CaseView), superseding hardcoded
   hub logic in `aboutness/scorer.py`.

2. **Dispersion-based Hubness**: Hub detection uses co-anchor dispersion,
   not global IDF. Backbones (low dispersion) bind; hubs (high dispersion)
   are suppressed.

3. **Single DB Boundary**: `backend/repositories/` for app-layer,
   `backend/reee/repositories/` for REEE-specific persistence.

4. **test_eu is Legacy**: New code goes in `backend/reee/`. The `test_eu`
   shim exists only for backward compatibility.

## Pending Removals

These directories/files are marked for removal once all imports are migrated:

| Path | Reason | Blocker |
|------|--------|---------|
| `backend/test_eu/` | Full duplicate of `backend/reee/` | Verify no external imports remain |

To check blockers:
```bash
# Find imports of test_eu outside the test_eu directory
grep -r "from test_eu" backend/ --include="*.py" | grep -v "test_eu/"
grep -r "import test_eu" backend/ --include="*.py" | grep -v "test_eu/"
```
