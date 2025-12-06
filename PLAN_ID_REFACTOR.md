# ID Format Refactor Plan

## New Format
```
pg_xxxxxxxx  - page (8 chars base36)
cl_xxxxxxxx  - claim
en_xxxxxxxx  - entity (all types)
ev_xxxxxxxx  - event
```
- Lowercase base36: a-z, 0-9
- 8 chars = 2.8 trillion IDs per type
- Total length: 11 chars (vs UUID's 36)

## Changes Required

### 1. Create ID Generator Utility
**File:** `backend/utils/id_generator.py`
- `generate_id(prefix: str) -> str`
- Base36 encoding using `secrets.token_bytes()`
- Validation: `is_valid_id(id: str, prefix: str) -> bool`

### 2. Update PostgreSQL Schema
**File:** `db/schema_gen2.sql`
- Remove unused tables: `core.entities`, `core.edges`, `core.events`, junction tables
- Keep: `core.pages`, `core.claims`
- Change ID columns: `UUID` → `VARCHAR(11)`
- Add entity_id_mappings table for merge tracking:
  ```sql
  CREATE TABLE core.entity_id_mappings (
      old_id VARCHAR(11) PRIMARY KEY,
      new_id VARCHAR(11) NOT NULL,
      merged_at TIMESTAMPTZ DEFAULT NOW()
  );
  ```

### 3. Update Neo4j Schema
- Entity nodes: `id` property uses `en_xxxxxxxx`
- Event nodes: `id` property uses `ev_xxxxxxxx`
- Claim nodes: `id` property uses `cl_xxxxxxxx`

### 4. Update Domain Models
**Files:** `backend/models/*.py`
- Change `id: uuid.UUID` → `id: str`
- Remove UUID parsing in `__post_init__`
- Update type hints

### 5. Update Repositories
**Files:** `backend/repositories/*.py`
- Use `id_generator.generate_id()` instead of `uuid.uuid4()`
- Update SQL queries (no `::uuid` casts)
- Update Neo4j queries

### 6. Update Workers
**Files:** `backend/workers/*.py`
- Update ID parsing (no `uuid.UUID()` calls)
- Update claim metadata handling

### 7. Migration Script
**File:** `db/migrate_to_short_ids.py`
- Map existing UUIDs to new short IDs
- Update PostgreSQL tables
- Update Neo4j nodes
- Update claim metadata entity_ids arrays

## Execution Order

1. Create `id_generator.py` utility
2. Update domain models (backward compatible - accept both formats temporarily)
3. Write migration script
4. Run migration on dev data
5. Update schema file for fresh installs
6. Update repositories to generate new format
7. Update workers
8. Remove UUID compatibility code

## Rollback Plan
- Keep UUID→short_id mapping table permanently
- Can reconstruct UUIDs if needed
