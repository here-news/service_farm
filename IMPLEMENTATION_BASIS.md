# Service Farm Gen2 - Implementation Basis

**Date:** 2025-11-29
**Status:** Architecture validated, ready for implementation

---

## Executive Summary

This document outlines the foundation for Service Farm Gen2, based on validated architectural patterns from the demo implementation. Gen2 represents a complete rewrite with:
- **PostgreSQL-first architecture** (replacing Neo4j-centric design)
- **Recursive multi-pass event clustering** (replacing single-pass story clustering)
- **Worker-based modular design** (clean separation of concerns)
- **Proven extraction quality from Gen1** (reusing battle-tested components)

---

## What We Validated in Demo

### ✅ Core Architecture Patterns

**1. Recursive Multi-Pass Clustering**
- **Pass 1:** Tight temporal clusters (2-day window + 2+ entity overlap)
- **Pass 2:** Bridge temporal gaps (14-day relaxed window + 4+ entity overlap)
- **Pass 3:** Transitive merging (graph DFS for connected components)
- **Result:** Successfully merged 82 claims across 26 days into 1 unified event

**2. Temporal Phase Detection**
- Automatically detects phases within large merged clusters
- Re-clusters with strict temporal window to find natural breakpoints
- Creates hierarchical micro/macro event structure
- **Result:** 4 temporal phases emerged from Comey case data

**3. Error-Tolerant Design**
- Handles wrong timestamps (filtered >6-month background references)
- Tolerates metadata gaps
- Entity-driven bridging compensates for temporal errors
- **Result:** Robust against real-world data quality issues

**4. Incremental Updates**
- New articles correctly merge into existing events
- No duplicate event creation
- Maintains hierarchy integrity
- **Result:** CBS and CNN dismissal articles properly integrated

---

## What to Borrow from Gen1

### ✅ Keep & Reuse

**1. Complete Extraction Pipeline**
Location: `service_farm_gen1/workers/extraction_worker.py` (and related modules)
- Content cleaning and normalization
- Entity extraction (NER)
- Claim extraction with confidence scoring
- Temporal extraction (with enhancements)
- **Why:** Battle-tested, high quality, proven success rate

**2. Semantic Analyzer**
Location: `service_farm_gen1/services/semantic_analyzer.py`
- Primary event time extraction
- Claim type classification
- Entity disambiguation
- **Enhancements needed:** 6-month background filter for timestamps

**3. GCS Persistence**
Location: `service_farm_gen1/services/gcs_persistence.py`
- File storage patterns
- Content archiving
- **Why:** Production-proven reliability

**4. Access Control (if needed)**
Location: `service_farm_gen1/services/access_control.py`
- Authentication/authorization patterns
- **Why:** Security is already implemented

### ❌ Replace / Don't Migrate

**1. Neo4j-Centric Patterns**
- Gen1 uses Neo4j as primary data store
- Gen2 uses PostgreSQL with proper relational schemas
- **Rationale:** Simpler ops, better query performance, standard SQL

**2. Story-Based Clustering**
- Gen1 clusters into "stories" with simple heuristics
- Gen2 uses recursive event formation with multi-signal clustering
- **Rationale:** More sophisticated, validated algorithm

**3. Monolithic Worker**
- Gen1 has combined extraction + processing worker
- Gen2 separates: extraction → semantic → event workers
- **Rationale:** Cleaner separation, better scalability

---

## Gen2 Architecture Overview

### Database Schema (PostgreSQL)

```sql
-- Core entities
pages (id, url, title, content_text, status, created_at)
claims (id, page_id, text, confidence, event_time, created_at)
entities (id, canonical_name, entity_type, created_at)
claim_entities (claim_id, entity_id)

-- Event hierarchy
events (
  id, title, summary, event_type,
  event_scale,  -- micro | meso | macro | story
  parent_event_id,  -- For hierarchy
  relationship_type,  -- PHASE_OF | PART_OF
  start_time, end_time,
  created_at, updated_at
)
page_events (page_id, event_id)
event_entities (event_id, entity_id)
```

### Worker Pipeline

```
URL → Extraction Worker → Semantic Worker → Event Worker
       ↓                   ↓                  ↓
     Redis Queue         Redis Queue        PostgreSQL
     (extract)           (semantic)          (events)
```

**Extraction Worker** (from Gen1)
- Download & clean content
- Extract entities (NER)
- Extract claims with timestamps
- Store in PostgreSQL (`pages`, `claims`, `entities`)
- Queue for semantic processing

**Semantic Worker** (from Gen1 + enhancements)
- Classify claim types
- Disambiguate entities
- Filter background timestamps
- Mark status = 'entities_extracted'
- Queue for event processing

**Event Worker** (new in Gen2)
- Fetch pending pages with claims
- Run recursive multi-pass clustering
- Detect temporal phases
- Create/update event hierarchy
- Link pages to events

### Key Configuration

```python
# Temporal clustering
TEMPORAL_WINDOW_DAYS = 2  # Pass 1 strict window
RELAXED_WINDOW_DAYS = 14   # Pass 2 bridging
MIN_ENTITY_OVERLAP = 2     # Pass 1
BRIDGE_ENTITY_OVERLAP = 4  # Pass 2
TRANSITIVE_OVERLAP = 3     # Pass 3

# Phase detection
MIN_CLUSTER_SIZE_FOR_PHASES = 20  # claims
MIN_TIME_SPAN_FOR_PHASES = 7      # days
MIN_PHASE_SIZE = 3                # claims
```

---

## Migration Strategy

### Phase 1: Setup Gen2 Repository
1. Rename `service_farm/` → `service_farm_gen1/`
2. Move `service_farm/demo/` → `service_farm/`
3. Clean up demo artifacts (keep validated code)
4. Set up proper git repo structure

### Phase 2: Port Gen1 Extraction
1. Copy extraction worker from Gen1
2. Adapt to Gen2 PostgreSQL schemas
3. Remove Neo4j dependencies
4. Keep Redis queue patterns
5. Test with sample URLs

### Phase 3: Enhance Semantic Analysis
1. Port semantic analyzer from Gen1
2. Add 6-month timestamp filter
3. Integrate with Gen2 schemas
4. Test claim classification quality

### Phase 4: Finalize Event Worker
1. Clean up demo event worker code
2. Add production configs
3. Implement monitoring/logging
4. Performance tuning for large datasets

### Phase 5: Data Migration
1. Run both Gen1 and Gen2 in parallel
2. Reprocess Gen1 pages through Gen2 pipeline
3. Validate event formation quality
4. Compare Gen1 stories vs Gen2 events

### Phase 6: Cutover
1. Point browser extension to Gen2 API
2. Migrate webapp to Gen2 database
3. Archive Gen1 (keep as reference)
4. Monitor Gen2 in production

---

## Directory Structure

```
service_farm/  (Gen2)
├── backend/
│   ├── api.py              # FastAPI endpoints
│   ├── workers.py          # Event worker (validated)
│   ├── semantic_analyzer.py # (from Gen1 + enhancements)
│   └── extraction_worker.py # (from Gen1)
├── services/
│   ├── gcs_persistence.py  # (from Gen1)
│   ├── entity_resolver.py  # (from Gen1/NER)
│   └── access_control.py   # (from Gen1 if needed)
├── db/
│   └── schema.sql          # PostgreSQL schemas
├── docker-compose.yml      # Backend + Workers + Postgres + Redis
├── requirements.txt
└── README.md

service_farm_gen1/  (Archive)
├── ... (original Gen1 code)
└── ARCHIVED.md  # Why archived, what was migrated
```

---

## Open Research Questions

### 1. Semantic Phase Detection
**Problem:** Current phase detection is purely temporal (2-day clustering). Cannot distinguish:
- Indictment vs dismissal (same entities, same day)
- Grand jury issues vs court hearings

**Potential Solutions:**
- Add claim type clustering (group by action/announcement/allegation/observation)
- Embedding-based similarity (see `experiment_embedding_matching.py`)
- LLM-based phase naming (generate descriptive titles)

**Timeline:** Post-migration research

### 2. Entity Evolution Tracking
**Problem:** Entities change roles over time (defendant → witness, prosecutor → judge)

**Approach:**
- Track entity relationships per event
- Temporal entity graphs
- Role-based entity clustering

**Timeline:** Future enhancement

### 3. Cross-Event Causality
**Problem:** Events can cause other events (indictment → motion to dismiss → dismissal)

**Approach:**
- Detect causal language in claims
- Build event dependency graphs
- Reasoning over event sequences

**Timeline:** Advanced feature

---

## Success Metrics

### Immediate (Phase 1-4)
- ✅ Gen2 can process URLs end-to-end
- ✅ Event formation produces hierarchies
- ✅ Quality matches or exceeds Gen1 extraction

### Short-term (Phase 5-6)
- ✅ Gen1 data successfully migrated
- ✅ Gen2 running in production
- ✅ Browser extension working with Gen2

### Long-term (Research)
- Semantic phase detection implemented
- Event causality detection working
- Multi-scale event hierarchy validated at scale

---

## Next Steps

1. **Rename & Move:**
   ```bash
   cd /media/im3/plus/lab4/re_news
   mv service_farm service_farm_gen1
   mv service_farm_gen1/demo service_farm
   ```

2. **Clean Up Demo:**
   - Remove test scripts
   - Keep validated `workers.py`, `api.py`
   - Add production configs

3. **Port Extraction Worker:**
   - Copy from Gen1
   - Adapt to PostgreSQL
   - Test with Comey URLs

4. **Document as You Go:**
   - Update this file with decisions
   - Track migration progress
   - Note any architecture changes

---

## References

- `docs/EVENT_FORMATION_MECHANISM.md` - Event emergence theory
- `docs/EVENT_DATA_STRUCTURE_SPEC.md` - Event schema spec
- `docs/EVENT_EMERGENCE_COMPLETE_SUMMARY.md` - Clustering validation
- `experiment_embedding_matching.py` - Semantic clustering research
- Gen1 codebase: `../service_farm_gen1/`

---

**Last Updated:** 2025-11-29
**Next Review:** After Phase 2 completion
