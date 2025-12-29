# MVP: Collaborative Epistemic Investigation System

## Vision

A system where **humans and engine collaborate** to converge toward truth:
- Engine processes claims using RELATE operation
- Humans add evidence, flag corrections, resolve conflicts
- All state changes recorded for audit trail
- Narratives released with version control

## Core Data Model

```
EVENT
├── id, title, created_at
├── phase: emerging | converging | stable | contested
├── metrics: { entropy, coherence, mass, claim_count }
│
├── BELIEF_STATES[] (engine-discovered)
│   ├── metric_name (discovered, not hardcoded)
│   ├── current_belief, entropy
│   ├── prior: monotonic_increasing | monotonic_decreasing | none
│   ├── history: [{time, value, relation, claim_id}]
│   └── support: {corroborations, contradictions}
│
├── CLAIMS[]
│   ├── id, text, source, pub_time
│   ├── relation_to_event: {type, confidence, related_to}
│   └── user_flags: [{type: CR, user_id, reason, status}]
│
├── CONTRADICTIONS[]
│   ├── topic, values[], support[], entropy
│   ├── status: active | resolved | dismissed
│   └── resolution: {method, by, at, explanation}
│
├── NARRATIVE_VERSIONS[]
│   ├── version, released_at, released_by
│   ├── prose (generated from belief states)
│   ├── facts[] with entropy markers
│   └── diff_from_previous
│
└── AUDIT_LOG[]
    ├── action: claim_added | cr_filed | conflict_resolved | version_released
    ├── actor: engine | user_id
    ├── timestamp
    └── state_snapshot
```

## Operations

### 1. ADD_CLAIM (Engine or User)
```
Input: claim_text, source, pub_time
Process:
  1. Extract metrics via LLM
  2. For each metric found:
     - RELATE to existing belief state
     - Classify: CORROBORATES | UPDATES | REFINES | CONTRADICTS
  3. Update belief states
  4. Recompute entropy
  5. Check for new contradictions
  6. Record to audit log
Output: Updated event state, relation results
```

### 2. FILE_CR (User)
```
Input: target (claim_id | belief_state | narrative_section), reason, suggested_fix
Process:
  1. Create CR record
  2. Flag target as "under review"
  3. Notify reviewers
  4. Record to audit log
Output: CR_id
```

### 3. RESOLVE_CONFLICT (User or Engine)
```
Input: contradiction_id, resolution_method, explanation
Methods:
  - TEMPORAL_UPDATE: Later value supersedes (engine can auto-resolve)
  - EVIDENCE_WEIGHT: One side has 3x+ support
  - USER_JUDGMENT: Human decides with explanation
  - DISMISS: Not a real conflict
Process:
  1. Mark contradiction as resolved
  2. Update affected belief states
  3. Recompute entropy
  4. Record to audit log
Output: Updated event state
```

### 4. RELEASE_VERSION (User)
```
Input: release_notes
Process:
  1. Generate prose from current belief states (LLM)
  2. Compute diff from previous version
  3. Create version record
  4. Record to audit log
Output: version_id, prose, diff
```

### 5. GET_STATE (Read)
```
Input: event_id, [version]
Output: Full event state at current or specified version
```

## UI Screens

### Screen 1: Progressive Emergence
- Live feed of claims as they arrive
- Each claim shows RELATE result (corroborates/updates/contradicts)
- Belief states update in real-time
- Entropy trajectory chart updates live
- Phase indicator changes: emerging → converging → stable

### Screen 2: Conflict Resolution
- List of active contradictions
- For each: show competing values, support counts, entropy
- Actions: Resolve (with method), Dismiss, Request more evidence
- History of resolved conflicts

### Screen 3: Correction Requests
- List of open CRs
- For each: target, reason, suggested fix, status
- Actions: Accept, Reject, Request clarification
- Link to audit log

### Screen 4: Version Control
- Timeline of released versions
- For each: diff view, prose, release notes
- Compare any two versions
- Rollback capability

### Screen 5: Collaborative Dashboard
- Who added what claims
- Who resolved what conflicts
- Who filed/addressed CRs
- Contribution stats

## API Endpoints

```
POST /events                     - Create event
GET  /events/:id                 - Get event state
POST /events/:id/claims          - Add claim
POST /events/:id/claims/:cid/cr  - File CR on claim
GET  /events/:id/contradictions  - List contradictions
POST /events/:id/contradictions/:cid/resolve - Resolve
POST /events/:id/versions        - Release version
GET  /events/:id/versions        - List versions
GET  /events/:id/versions/:vid   - Get specific version
GET  /events/:id/audit           - Audit log
```

## State Persistence

Using existing infrastructure:
- **Neo4j**: Event graph (events, claims, entities, relationships)
- **PostgreSQL**: Audit log, versions, CRs (structured data)
- **Redis**: Real-time state for live UI updates

## Implementation Plan

### Phase 1: Core Engine (test_eu/uee_server.py)
- Wrap existing UEE code in Flask/FastAPI
- State persistence to file/SQLite for MVP
- Endpoints: create_event, add_claim, get_state

### Phase 2: Conflict & CR (test_eu/uee_server.py)
- Add contradiction tracking
- Add CR filing and resolution
- Endpoints: contradictions, resolve, file_cr

### Phase 3: Version Control (test_eu/uee_server.py)
- Narrative generation from belief states
- Version snapshots and diffs
- Endpoints: versions, release

### Phase 4: Collaborative UI (frontend/prototypes/collaborative-epistemic.html)
- Progressive emergence view
- Conflict resolution interface
- CR management
- Version timeline

## Success Criteria

1. **Progressive Emergence**: Can watch event emerge from claims
2. **Conflict Resolution**: Can see and resolve contradictions
3. **CR Support**: Can flag issues and track resolution
4. **Version Control**: Can release, diff, and audit narratives
5. **Collaboration**: Multiple users can participate, all recorded

## Files to Create

1. `test_eu/uee_server.py` - Backend API with UEE
2. `test_eu/uee_state.py` - State persistence layer
3. `frontend/prototypes/collaborative-epistemic.html` - Full UI
4. `test_eu/uee_models.py` - Data models (Event, Claim, BeliefState, etc.)
