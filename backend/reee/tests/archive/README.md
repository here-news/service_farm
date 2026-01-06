# Archived Tests

These tests were for `case_builder.py` (PrincipledCaseBuilder) which has been replaced by `story_builder.py` (StoryBuilder).

## Archived Files (42 tests total)

| File | Tests | Purpose |
|------|-------|---------|
| test_case_builder_hub_suppression.py | 11 | Hub entity detection and suppression |
| test_case_builder_chain.py | 10 | Motif chain continuity |
| test_case_builder_motif_recurrence.py | 10 | k=2 motif recurrence for core edges |
| test_entity_case.py | 11 | Star-shaped EntityCase formation |

## Key Assertions Preserved

The following invariants were ported to story_builder tests:

1. **Hub suppression**: Entities >20% cannot create core edges
2. **Anti-percolation**: No mega-case formation from hub linkage
3. **Spine membership**: Core-A is automatic, Core-B needs 2 structural witnesses
4. **Membrane decision audit**: All decisions recorded with provenance

## Migration Date

2026-01-05 - Migrated from case_builder → story_builder
