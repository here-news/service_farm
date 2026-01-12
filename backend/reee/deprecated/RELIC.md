# REEE Deprecated Modules (RELIC)

This document maps deprecated modules to their replacements.

## Removal Schedule

| Module | Deprecated | Removal Target | Replacement | Status |
|--------|------------|----------------|-------------|--------|
| `builders/case_builder.py` | 2026-01-05 | 2026-02-01 | `StoryBuilder` + `EntityLens` | STUB |
| `engine.py` | 2026-01-05 | 2026-02-01 | `PrincipledSurfaceBuilder` + `PrincipledEventBuilder` + `StoryBuilder` | STUB |
| `kernel.py` | 2026-01-05 | 2026-02-01 | `TypedBeliefState` | STUB |
| `comparator.py` | 2026-01-05 | 2026-02-01 | `TypedBeliefState` | STUB |
| `extractor.py` | 2026-01-05 | 2026-02-01 | `claim_loader.py` in workers | STUB |
| `interpretation.py` | 2026-01-05 | 2026-02-01 | worker layer + `TypedBeliefState` | STUB |
| `aboutness/` | 2026-01-05 | 2026-02-01 | `membrane` + `StoryBuilder` | STUB |

## Strict Mode

Set `REEE_STRICT_DEPRECATIONS=1` environment variable to make deprecated imports
fail with RuntimeError after the removal date. This is useful for CI/testing to
ensure no deprecated code paths remain.

## Migration Paths

### `case_builder.py` → `StoryBuilder` + `EntityLens`

**Old (deprecated):**
```python
from reee.builders.case_builder import PrincipledCaseBuilder, EntityCase

builder = PrincipledCaseBuilder()
result = builder.build_from_incidents(incidents)
for entity_case in result.entity_cases.values():
    process(entity_case)
```

**New:**
```python
from reee.builders.story_builder import StoryBuilder
from reee.types import EntityLens

builder = StoryBuilder()
result = builder.build_from_incidents(incidents)
for story in result.stories.values():
    lens = story.to_lens()
    # Build companions externally (lens is immutable)
    enriched_lens = EntityLens.create(
        entity=lens.entity,
        incident_ids=lens.incident_ids,
        companion_counts=compute_companions(lens.incident_ids),
    )
    process(enriched_lens)
```

### `engine.py` → `StoryBuilder`

**Old (deprecated):**
```python
from reee import Engine
engine = Engine()
result = engine.process(claims)
```

**New:**
```python
from reee.builders import PrincipledSurfaceBuilder, PrincipledEventBuilder, StoryBuilder

# L2: Claims → Surfaces
surface_builder = PrincipledSurfaceBuilder()
surface_result = await surface_builder.build_from_claims(claims)

# L3: Surfaces → Incidents
event_builder = PrincipledEventBuilder()
event_result = await event_builder.build_from_surfaces(surface_result.surfaces)

# L4: Incidents → Stories
story_builder = StoryBuilder()
story_result = story_builder.build_from_incidents(event_result.events)
```

### `kernel.py` → `TypedBeliefState`

**Old (deprecated):**
```python
from reee import EpistemicKernel, Belief
kernel = EpistemicKernel()
belief = kernel.update(observation)
```

**New:**
```python
from reee import TypedBeliefState, CountDomain, Observation

belief = TypedBeliefState(domain=CountDomain(max_value=100))
belief.update(Observation(value=10, source="source1", confidence=0.9))
posterior = belief.posterior()
```

### `aboutness/` → `PrincipledSurfaceBuilder` + `membrane`

**Old (deprecated):**
```python
from reee.aboutness import AboutnessScorer, compute_aboutness_edges
scorer = AboutnessScorer()
edges = compute_aboutness_edges(surfaces)
```

**New:**
```python
from reee.builders import PrincipledSurfaceBuilder, PrincipledEventBuilder
from reee.membrane import classify_incident_membership

# Aboutness is now internal to builders
surface_builder = PrincipledSurfaceBuilder()
event_builder = PrincipledEventBuilder()

# Membership decisions go through membrane
decision = classify_incident_membership(incident, focal_set, constraints)
```

## Stub Behavior

When importing a deprecated module:

1. **During deprecation period**: `DeprecationWarning` is raised, module works
2. **After removal date**: `RuntimeError` is raised with migration pointer

Example stub:
```python
# reee/builders/case_builder.py (stub)
import warnings
from datetime import date

REMOVAL_DATE = date(2026, 2, 1)

if date.today() >= REMOVAL_DATE:
    raise RuntimeError(
        "reee.builders.case_builder has been removed. "
        "Use StoryBuilder + EntityLens instead. "
        "See reee/deprecated/RELIC.md for migration guide."
    )

warnings.warn(
    "reee.builders.case_builder is deprecated and will be removed on 2026-02-01. "
    "Use StoryBuilder + EntityLens instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from deprecated location for compatibility
from ..deprecated._case_builder import *
```
