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
- [x] **Phase 3: Semantic Worker implementation** (claims-first approach)
  - [x] SemanticWorker class adapted to Gen2 schema
  - [x] Integration with Demo's EnhancedSemanticAnalyzer
  - [x] Entity extraction FROM claims (no orphan entities)
  - [x] Entity deduplication via DB uniqueness constraint
  - [x] Page embedding generation (pgvector)
  - [x] Docker compose configuration
  - [x] **Entity type fixes** - Extract from WHO/WHERE prefixes instead of entities_dict
  - [x] **Entity centrality tracking** - Count claims per entity (mention_count)
  - [x] **LLM-based entity descriptions** - Generate concise profiles (max 20 words)
- [x] **Phase 3.1: Test & Validate Semantic Worker**
  - [x] Created test UI (`frontend/index.html`) with Gen2 API endpoints
  - [x] 4-stage pipeline visualization (Preview → Extraction → Semantic → Complete)
  - [x] Claims visualization with WHO/WHERE/WHEN metadata
  - [x] Entity display with tooltips showing descriptions
  - [x] Central entity highlighting based on mention counts
  - [x] Embedded claims/entities in `/url/{page_id}` to avoid CORS
- [x] **Phase 2.5: Metadata extraction enhancements**
  - [x] Newspaper3k metadata capture (title, author, description, thumbnail)
  - [x] Preserve iframely metadata when available
  - [x] metadata_confidence score (0.0-1.0) based on completeness
  - [x] Smart metadata merging (prefer extraction, fallback to iframely)

---

## TODO

### Phase 3.2: Additional Testing & Refinement
**Goal:** More comprehensive validation before event worker

- [ ] Test with 5-10 diverse articles (opinion, press release, historical)
- [ ] Validate entity descriptions quality across different domains
- [ ] Test metadata_confidence edge cases (paywall sites, dynamic content)
- [ ] Performance benchmarking (throughput, latency)

### Phase 3.2: Entity Enhancements (post-testing)
- [ ] Wikidata linking (QIDs, descriptions, thumbnails)
- [ ] Coreference resolution ("Trump" vs "Donald Trump")
- [ ] Fuzzy deduplication (85% similarity threshold)
- [ ] Metadata entity resolution (author, source as entities)

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
