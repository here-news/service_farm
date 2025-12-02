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
- [ ] Deep extraction: add Playwright iframe-chasing mode and semantic heuristic; write /tmp/deep_playwright_tests.md scenarios and rerun parent+embed tests

### Phase 3.2: Entity Enhancements (post-testing)
- [ ] Wikidata linking (QIDs, descriptions, thumbnails)
- [ ] Coreference resolution ("Trump" vs "Donald Trump")
- [ ] Fuzzy deduplication (85% similarity threshold)
- [ ] Metadata entity resolution (author, source as entities)
- [ ] Authoritative ID normalization (Wikidata/GeoNames/ISNI/ORCID/LEI) with per-ID confidence
- [ ] Backfill existing entities via lookup worker; persist `authoritative_ids` map and provenance
- [ ] Alias graph (surface forms, transliterations) to strengthen canonical matching
- [ ] Claim-level role tagging (subject/agent/location/patient/source) aggregated to participant centrality
- [ ] Entity resolution quality gates (minimum confidence, coverage alerts) and dashboards

### Phase 4: Event Worker - Holistic Enrichment Architecture ✨ **NEW APPROACH**
**Goal:** Form events through incremental holistic enrichment (NOT claim clustering)

#### Completed ✅
- [x] **Architectural pivot:** Direct page → event enrichment (no claims needed)
- [x] **Holistic enrichment framework** (`workers/holistic_enrichment.py`)
  - [x] FactBelief class with timeline tracking
  - [x] Incremental contradiction resolution (consensus, temporal evolution, Bayesian override)
  - [x] Dual representation: structured ontology + synthesized story (5W+H)
  - [x] Milestone resolution with fuzzy matching
- [x] **Event network builder** (`test_event_network.py`)
  - [x] Multi-signal similarity scoring (embeddings)
  - [x] Decision logic: Attach (>0.65), Relate (0.45-0.65), Spawn (<0.45)
  - [x] Relationship types: SIMILAR_TO, CAUSED_BY, EXTENDS, PHASE_OF
- [x] **Event ontology structure:**
  - [x] casualties: deaths/missing/injured with FactBeliefs
  - [x] timeline: start/end/milestones with temporal resolution
  - [x] locations: primary/district
  - [x] response: evacuations/arrests/casualties
  - [x] story: WHO/WHEN/WHERE/WHAT/WHY/HOW synthesis
- [x] **Clean slate test:** Hong Kong fire event (5 pages → 98.6% coherence)
- [x] **Deduplication:** Milestone fuzzy matching (80% similarity threshold)

#### TODO - Production Integration
- [ ] **Integrate into event_worker.py:**
  - [ ] Replace claim-based logic with HolisticEventEnricher
  - [ ] Add EventNetworkBuilder for similarity-based decisions
  - [ ] Preserve entity extraction from semantic_worker (event surface)
  - [ ] Make claims optional (for provenance/debug only)
- [ ] **semantic_worker.py updates:**
  - [ ] Keep entity extraction (needed for event surface)
  - [ ] Make claim extraction optional flag
  - [ ] Add page embedding if not already generated
- [ ] **Sub-events cleanup:**
  - [ ] Decide: Eliminate micro-event records OR keep for relationships?
  - [ ] If keep: Define how they relate to ontology sections
  - [ ] Remove old micro-event generation code

#### TODO - Evolution & Refinement
- [ ] **Event propagation:**
  - [ ] Detect coherence upgrades (0.65→0.7) as trigger
  - [ ] Extract facts from event(A) story → enrich related event(B)
  - [ ] Update relationship weights based on propagation success
- [ ] **Timeline & phasing:**
  - [ ] Auto-detect event phases (outbreak → response → investigation)
  - [ ] Better milestone extraction beyond "upgraded to level X"
  - [ ] Event lifecycle tracking (active → concluded → historical)
- [ ] **Opinion/review detection:**
  - [ ] Classify opinion vs factual reporting
  - [ ] Add to event context without spawning new events
  - [ ] Track as "supporting_content" to boost plausibility
- [ ] **Scalability:**
  - [ ] Batch processing for multiple pages
  - [ ] Cache repeated enrichments
  - [ ] Optimize LLM calls (1 per page currently)
- [ ] **Quality & monitoring:**
  - [ ] Coherence thresholds for event quality gates
  - [ ] Detect low-quality enrichments
  - [ ] Human-in-the-loop for ambiguous decisions (similarity 0.60-0.70)

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

**Last Updated:** 2025-12-02 (Session: Holistic enrichment architecture complete)
