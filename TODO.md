# Service Farm Gen2 - TODO

**Status:** Gen2 extraction pipeline complete, moving to semantic & event processing

---

## Completed ✅

- [x] Gen2 PostgreSQL schema with migrations
- [x] Instant best-shot API with iframely integration (`POST /api/v2/artifacts`)
- [x] Multi-method extraction worker (Newspaper3k → Trafilatura → Readability)
- [x] Rogue URL extraction system with browser extension v2.2.0
- [x] Autonomous extraction worker with decision-making
- [x] Image extraction support
- [x] RESTful endpoint naming (`/artifacts` instead of `/url`)

---

## TODO

### Phase 2.5: Enhance Extraction Worker (metadata verification)
**Goal:** Use extraction worker as fallback and verification for iframely

- [ ] Update extraction worker to verify/correct iframely metadata
- [ ] Extract metadata even when iframely succeeds (for comparison/correction)
- [ ] Update page metadata if worker finds better values (title, author, date)
- [ ] Ensure metadata extraction works as fallback when iframely fails
- [ ] Future-proof for potential iframely retirement

### Phase 3: Semantic Worker - Port from Gen1 with enhancements
**Goal:** Extract entities and claims from content

- [ ] Port semantic analyzer from Gen1 (`service_farm_gen1/services/semantic_analyzer.py`)
- [ ] Adapt to Gen2 PostgreSQL schemas (claims, entities, claim_entities tables)
- [ ] Add 6-month timestamp filter for background references
- [ ] Integrate with Gen2 worker queue system
- [ ] Test claim classification quality
- [ ] Commission event worker after semantic extraction

### Phase 4: Event Worker - Implement recursive multi-pass clustering
**Goal:** Form events from claims using multi-pass clustering

- [ ] Implement Pass 1: Tight temporal clusters (2-day window + 2+ entity overlap)
- [ ] Implement Pass 2: Bridge temporal gaps (14-day relaxed + 4+ entity overlap)
- [ ] Implement Pass 3: Transitive merging (graph DFS for connected components)
- [ ] Add temporal phase detection for large clusters
- [ ] Create event hierarchy (micro/meso/macro/story scales)
- [ ] Link pages to events
- [ ] Add incremental update support (new articles merge into existing events)

### Phase 5: Data Migration - Migrate Gen1 data through Gen2 pipeline
**Goal:** Reprocess existing data through Gen2 for validation

- [ ] Set up parallel Gen1/Gen2 operation
- [ ] Export Gen1 page URLs
- [ ] Reprocess through Gen2 extraction pipeline
- [ ] Compare event formation quality (Gen1 stories vs Gen2 events)
- [ ] Validate semantic extraction improvements
- [ ] Performance benchmarking

---

## Future Enhancements (Post-Migration)

- [ ] Semantic phase detection (claim type clustering)
- [ ] Embedding-based similarity clustering
- [ ] Entity evolution tracking (role changes over time)
- [ ] Cross-event causality detection
- [ ] LLM-based phase naming

---

**Last Updated:** 2025-11-30
