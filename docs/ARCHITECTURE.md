# Architecture: Breathing Knowledge Base

> **Mission**: Build a living, evidence-driven knowledge base that evolves with reality, combining machine curation with community consensus and economic incentives.

**Version**: 2.2 (Consolidated)
**Date**: 2025-11-28
**Status**: Design Complete, Implementation Phase 1 Ready

---

## Design Philosophy

### The "Wikipedia Problem"

**Wikipedia** is humanity's greatest collaborative knowledge project, but it has fundamental limitations:

1. **Static by nature** - Articles freeze until someone manually updates them
2. **Source opacity** - Citations exist, but relationship to claims is loose
3. **Edit wars** - Conflicts resolved by moderator judgment, not evidence weight
4. **Single narrative** - One article per topic, erasing valid alternative perspectives
5. **Language silos** - Cross-language linking is manual and incomplete

### Our Approach: Beyond Wikipedia

We're building a knowledge base that **breathes**:

1. **Evidence-first** - Every fact anchored to source artifacts (URLs, images, videos, documents)
2. **Claim-atomic** - Facts are atomic, not paragraphs - can be recombined and verified independently
3. **Graph-native** - Relationships are first-class citizens with their own confidence
4. **Multi-lingual by default** - Entities link across languages via canonical IDs
5. **Confidence evolves** - Trust rises with evidence, falls with age or contradictions
6. **Community + economics** - Disputes resolved by weighted voting (reputation + stake)
7. **Facts, not narratives** - System produces events; humans curate stories

**Result**: A knowledge base that adapts to reality in real-time, not article edit-time.

---

## Core Principles

### 1. Resource-First, Not Pipeline

**Anti-pattern (Current systems)**:
```
URL submitted → Pipeline processes everything → Return result or error
```
- User waits for entire pipeline (2-10 seconds)
- Any step fails → entire request fails
- Can't return partial results

**Our pattern**:
```
URL submitted → Create resource stub → Return immediately → Workers enrich asynchronously
```
- User gets response in < 100ms
- Workers run independently (one fails ≠ all fail)
- Progressive enhancement: stub → partial → enriched → complete

**Why**: Users need speed; systems need resilience. Separate concerns.

**Industry reference**: [REST Maturity Model Level 3 (HATEOAS)](https://martinfowler.com/articles/richardsonMaturityModel.html), [Event-Driven Architecture](https://martinfowler.com/articles/201701-event-driven.html)

---

### 2. Confidence Lives in the Graph, Not Just Nodes

**Anti-pattern**:
```
Entity {confidence: 0.85}  ← Single magic number
```

**Our pattern**:
```
Node confidence = f(
    semantic signals (model scores, external authority),
    structural signals (graph topology, edge patterns),
    temporal factors (freshness, decay),
    contradiction penalties (disputes, conflicts)
)
```

**Why**: Confidence in "Louvre Museum" shouldn't just come from NER extraction. It should reflect:
- How many sources mention it (evidence diversity)
- Across how many languages (cross-validation)
- How consistent the mentions are (low contradictions)
- How fresh the evidence is (temporal decay)
- Whether it's disputed (community signals)

**Industry reference**: [PageRank](https://en.wikipedia.org/wiki/PageRank) (link-based authority), [Knowledge Graph Embedding Confidence](https://arxiv.org/abs/1503.00759)

---

### 3. Instant vs Lazy: Strict Latency Discipline

**Anti-pattern**:
```
API calls external service (Wikidata) → 1-3 seconds → User waits
```

**Our pattern**:
```
Instant Tier (< 500ms):  DB/cache only, return stub
Lazy Tier (async):       External APIs, LLMs, heavy computation
```

**Why**:
- 500ms is [Jakob Nielsen's threshold](https://www.nngroup.com/articles/response-times-3-important-limits/) for "feels instant"
- External services are unreliable (timeouts, rate limits, outages)
- User experience > completeness

**Enforcement**: Hard timeout on instant tier. Any code path violating 500ms = architectural violation.

**Industry reference**: [Microservices Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html), [CQRS](https://martinfowler.com/bliki/CQRS.html) (read/write separation)

---

### 4. Facts Over Narratives

**Anti-pattern (Most news systems)**:
```
Ingest article → Summarize → Store summary
```
- Summary loses nuance
- Can't track what changed over time
- Sources disconnected from claims

**Our pattern**:
```
Article → Extract atomic claims → Cluster into events → (Stop)
Narratives/stories = community layer (not system-generated)
```

**Why**:
- Claims are verifiable ("44 people died")
- Narratives are subjective ("deadliest fire in decades")
- System should build facts; humans interpret meaning

**Event ≠ Story**:
- **Event**: Factual happening backed by evidence ("Hong Kong apartment fire, Nov 26, 2025")
- **Story**: Human narrative ("Safety failures led to tragedy") ← Community creates this

**Industry reference**: [Semantic Web/RDF Triple Stores](https://www.w3.org/TR/rdf11-primer/) (subject-predicate-object facts)

---

### 5. Gated Entity Creation

**Anti-pattern**:
```
Extract "police", "the incident", "someone" → Create 3 entities
```
→ Entity explosion with noise

**Our pattern**: Only create entity if it passes **value criteria**:

1. **External authority** (Wikidata, Wikipedia match)
2. **Repeated mention** (3+ times across corpus)
3. **User intent** (manually created)
4. **Specificity** (proper noun, multi-word, not generic)
5. **Event significance** (key actor in high-confidence event)
6. **Graph coherence** (local neighborhood has high structural confidence)

**Why**: Unbounded entity growth = search pollution. Quality > quantity.

**Industry reference**: [Entity Linking](https://en.wikipedia.org/wiki/Entity_linking), [Named Entity Recognition Filtering](https://nlp.stanford.edu/projects/kbp/)

---

### 6. Multi-Language as First-Class, Not Afterthought

**Anti-pattern**:
```
Entity stored in English → Translation added later as separate field
"Hong Kong" ≠ "香港" (different entities)
```

**Our pattern**:
```typescript
Entity {
  canonical_id: "e1",
  canonical_name: "Hong Kong",  // English default
  names_by_language: {
    "en": ["Hong Kong", "HK", "HKSAR"],
    "zh": ["香港", "香港特别行政区"],
    "fr": ["Hong Kong"],
    ...
  },
  wikidata_qid: "Q8646"  // Bridge for cross-language linking
}
```

**Why**:
- Global system needs global support
- Chinese "香港" + English "Hong Kong" + French "Hong Kong" = **same entity**
- Wikidata provides canonical cross-language mapping

**Industry reference**: [Internationalization (i18n) Best Practices](https://www.w3.org/International/questions/qa-i18n), [Wikidata Language-Independent IDs](https://www.wikidata.org/)

---

### 7. Temporal Decay: Forgetting as Feature

**Anti-pattern**:
```
Evidence from 2020 has same weight as evidence from yesterday
```

**Our pattern**:
```
edge_weight(t) = base_confidence × exp(-decay_rate × age)
```

**Why**:
- Old information becomes stale
- Prevents graph from being dominated by historical patterns
- Forces reconfirmation of facts over time

**Decay rates by edge type**:
- Slow: Wikidata links (half-life ~700 days)
- Medium: Entity mentions (half-life ~70 days)
- Fast: Co-occurrence patterns (half-life ~14 days)
- Never: Contradictions, user corrections

**Result**: Graph "breathes" - forgets stale info, reinforces reconfirmed facts.

**Industry reference**: [Temporal Knowledge Graphs](https://arxiv.org/abs/2004.04926), [Time-Aware Recommender Systems](https://dl.acm.org/doi/10.1145/2043932.2043988)

---

## Resource Model

### Hierarchy

```
Artifact (evidence container)
    ↓ extraction
Content (normalized text/data)
    ↓ entity extraction
Entity (canonical reference)
    ↓ claim extraction + clustering
Event (factual happening)
    ↓ [STOP - system boundary]

Community Layer (outside system):
    - Stories (human narratives)
    - Collections (saved events)
    - Threads (discussions)
```

### Why This Hierarchy?

**Artifact abstraction**:
- URL, image, video, document all become "artifact"
- Different extractors plug in, but downstream is uniform
- Future-proof: new evidence types = new extractor, rest unchanged

**Content normalization**:
- Extract text from anything (Trafilatura for URLs, OCR for images, Whisper for video)
- Downstream doesn't care about source format

**Entity canonicalization**:
- "Hong Kong" + "HK" + "香港" → single canonical entity
- Prevents duplication, enables cross-reference

**Event as factual endpoint**:
- System stops at verifiable facts
- Interpretation left to humans

---

## Confidence: From Nodes to Graph Topology

### Traditional Approach (Naive)

```
entity.confidence = ner_model_score  // 0.0-1.0
```

**Problems**:
- Ignores how many sources mention entity
- Ignores cross-language validation
- Ignores graph neighborhood coherence
- Static (doesn't evolve with time)

### Our Approach (Graph-Aware)

**Confidence has three components**:

#### 1. Semantic Confidence
"Is the entity semantically correct?"
- NER model scores
- Wikidata match (external authority)
- Human verification

#### 2. Structural Confidence
"Does the entity fit coherently in the graph?"
- **Evidence diversity**: Unique sources, languages, temporal spread
- **Edge consistency**: Low contradiction rate
- **Local density**: Well-connected neighborhood
- **Centrality**: Important in local subgraph

#### 3. Temporal Freshness
"Is the evidence recent?"
- Edge weights decay over time
- Reconfirmed edges maintain weight
- Stale entities drop in confidence → trigger re-enrichment

**Combined**:
```
confidence = weighted_average(semantic, structural, temporal) - contradiction_penalty
```

### Why This Matters

**Example: Louvre Museum**
- 50 sources in 4 languages over 30 days → high diversity
- Wikidata Q19675 match → high semantic
- Few contradictions → high consistency
- Recently mentioned → high temporal freshness
- **Result**: confidence = 0.95

**Example: "John's Pizza" (local shop)**
- 1 source, 1 mention → low diversity
- No Wikidata match → low semantic
- No contradictions (but few edges) → medium consistency
- Recent → high temporal freshness
- **Result**: confidence = 0.55 (correctly low for obscure entity)

**Industry reference**: [Graph Neural Networks for KG Completion](https://arxiv.org/abs/1902.08564), [Trust Rank (Spam Detection)](https://en.wikipedia.org/wiki/TrustRank)

---

## Two-Tier Execution Model

### Tier 1: Instant (< 500ms, Synchronous)

**Allowed**:
- PostgreSQL queries (indexed)
- Redis cache lookups
- In-memory computation
- Stub creation

**Forbidden**:
- External API calls (Wikidata, Wikipedia, etc.)
- LLM calls (GPT, Claude, etc.)
- Heavy graph traversals
- File I/O

**Enforcement**: Hard timeout decorator on all instant tier endpoints.

**Endpoints**:
- `POST /artifacts` → Create stub, enqueue extraction
- `GET /entities/search` → Fuzzy lookup in DB/cache
- `GET /entities/{id}` → Return current state (might be partial)
- `GET /events/{id}` → Return current state

### Tier 2: Lazy (Async, Queued)

**Priority levels**:

| Priority | Latency Target | Use Case |
|----------|---------------|----------|
| `critical` | < 5s | User-submitted URL just now |
| `high` | < 30s | User viewing entity page |
| `normal` | < 5min | Background entity/event processing |
| `low` | < 1hr | Wikidata enrichment, alias discovery |
| `batch` | No limit | Nightly duplicate detection, archival |

**Workers**:
1. Content extraction (URL, image, video, document)
2. Entity extraction (NER + fuzzy matching)
3. Claim extraction (LLM structured output)
4. Event clustering (temporal + entity + semantic)
5. Entity enrichment (Wikidata, Wikipedia, aliases)
6. Confidence recomputation (graph topology analysis)
7. Conflict resolution (weighted voting)

**Why separate tiers?**
- Instant tier = user-facing, must be fast and reliable
- Lazy tier = system-facing, can be slow and can fail/retry
- Users never blocked by slow external services

**Industry reference**: [Async Request-Reply Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/async-request-reply), [Priority Queue Pattern](https://www.enterpriseintegrationpatterns.com/patterns/messaging/MessagePriority.html)

---

## Conflict Resolution: Weighted Voting

### Scenario: "2000 People Say Louvre Is in Berlin"

**Naive approach**: Count votes → 2000 > 50 → Berlin wins → Wrong!

**Our approach**: Weight by multiple factors

#### Vote Weight Formula
```
weight = edge_confidence × source_credibility × stake × temporal_decay
```

**Example calculation**:

**Paris claim** (50 sources, tier1):
- Avg edge confidence: 0.95
- Source credibility: tier1 = 0.95, multiplier = 3.0
- Stake: implicit reputation = 1.0
- Temporal: fresh (0.98)
- **Weight per source**: 0.95 × 0.95 × 3.0 × 0.98 = 2.64
- **Total**: 2.64 × 50 = 132.0

**Berlin claim** (2000 sources, tier3):
- Avg edge confidence: 0.60
- Source credibility: tier3 = 0.65, multiplier = 0.3
- Stake: low reputation = 0.5
- Temporal: fresh (1.0)
- **Weight per source**: 0.60 × 0.65 × 0.3 × 0.5 × 1.0 = 0.059
- **Total**: 0.059 × 2000 = 118.0

**Result**: Paris wins (132 > 118) - quality beats quantity.

### Why This Works

1. **Source credibility**: Tier1 sources (BBC, Reuters) weighted higher than tier3 (blogs)
2. **Stake**: Users with reputation have more weight (Sybil attack resistance)
3. **Confidence**: Low-confidence extractions penalized
4. **Temporal**: Old evidence decays (prevents historical bias)

**Industry reference**: [Reputation Systems](https://en.wikipedia.org/wiki/Reputation_system), [Stake-Weighted Voting (Blockchain)](https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/), [Wisdom of Crowds](https://en.wikipedia.org/wiki/The_Wisdom_of_Crowds)

---

## Extensibility: Adding New Capabilities

### New Artifact Type (Example: Video)

**What to add**:
1. Video extractor worker (Whisper for audio, frame OCR for visuals)
2. Artifact metadata schema for video (duration, format, has_audio, has_subtitles)

**What stays the same**:
- API endpoints (`POST /artifacts`)
- Downstream pipeline (Content → Entity → Event)
- All other workers

**Why**: Artifact abstraction decouples input format from processing logic.

### New Language (Example: Arabic)

**What to add**:
1. spaCy model for Arabic (`ar_core_news_sm`)
2. Language detection route to Arabic NER pipeline

**What stays the same**:
- Entity schema (already has `names_by_language`)
- Cross-language linking (via Wikidata)
- All other languages

**Why**: Multi-language is first-class, not bolted on.

### New Worker (Example: Image Classification)

**What to add**:
1. Worker that classifies images (GPT-4V, CLIP)
2. Queue listener for `classify_image` jobs
3. Schema for image labels/tags

**What stays the same**:
- Instant tier (still < 500ms)
- Other workers (run independently)
- API (new endpoint optional)

**Why**: Workers are independent, communicate via queue only.

---

## Best Practices We Follow

### 1. **Idempotency**
- Same URL submitted 10 times → same resource, updated metadata
- No duplicate entities, no duplicate events
- `ON CONFLICT` clauses in SQL, unique constraints

### 2. **Graceful Degradation**
- Wikidata down → skip enrichment, return partial entity
- One worker fails → others continue
- No cascading failures

### 3. **Observable System**
- Every worker logs: job_id, duration, success/failure
- Confidence scores are transparent (semantic + structural + temporal breakdown)
- Graph health metrics (duplicate rate, orphan entities, contradiction rate)

### 4. **Event-Driven Architecture**
- Workers don't call each other directly
- All communication via message queue (Redis)
- Easy to add new workers, remove old ones

### 5. **Separation of Concerns**
- Instant tier = user experience
- Lazy tier = correctness/completeness
- Never mix the two

### 6. **Data Provenance**
- Every entity tracks: extraction_method, sources, confidence_method
- Every edge tracks: evidence_sources, first_seen, last_seen
- Audit trail for disputes/corrections

### 7. **Version Everything**
- Entities have `updated_at` timestamp
- Events have version history
- Graph snapshots for rollback

---

## What We Explicitly Don't Do

### 1. **No Automatic Story Generation**
**Why**: Stories are interpretive. System produces facts (events); humans curate narratives.

### 2. **No Synchronous External API Calls in Instant Tier**
**Why**: Violates latency budget. Use lazy tier.

### 3. **No Unbounded Entity Creation**
**Why**: Gating prevents noise. "Police" mentioned once ≠ create entity.

### 4. **No Static Confidence**
**Why**: Confidence must evolve with evidence. Use temporal decay + recomputation.

### 5. **No Single-Source-of-Truth for Location**
**Why**: Some entities have multiple valid locations (traveling person, distributed org). Store all with confidence.

### 6. **No Ignoring Contradictions**
**Why**: Contradictions are data. Store as edges, resolve via voting, surface to users.

---

## Trade-Offs

### Fast vs Complete
**Choice**: Fast (instant tier < 500ms)
**Trade-off**: Initial response may be partial (status: "stub" or "partial")
**Mitigation**: Progressive enhancement, users can poll for updates

### Quality vs Quantity
**Choice**: Quality (gated entity creation)
**Trade-off**: Some valid entities might be missed initially
**Mitigation**: Repeated mentions or user creation fills gaps over time

### Automation vs Accuracy
**Choice**: Automation with confidence scores
**Trade-off**: Some errors will occur (NER mistakes, clustering errors)
**Mitigation**: Community corrections, weighted voting, confidence transparency

### Global vs Tenant-Specific
**Choice**: Global graph with per-tenant overlays (SaaS-ready)
**Trade-off**: More complex data model
**Mitigation**: Clear separation, tenant_id on all resources

---

## Success Metrics

### Performance
- Instant tier: < 500ms p95
- URL extraction: < 5s p95
- Entity extraction: < 10s p95
- Event formation: < 30s p95

### Quality
- Entity duplication: < 10% (down from 56% in old system)
- Cross-language linking: > 80% accuracy
- Event coherence: > 0.85 average
- Wikidata linkage: > 85% for prominent entities

### Scale
- 1000+ pages/minute ingestion
- 100k+ entities in graph
- 10k+ events formed
- Sub-second entity search (autocomplete)

### User Experience
- 100% of URL submissions return stub in < 200ms
- 90% of pages reach "complete" in < 30s
- Dispute resolution triggered within 1 minute of threshold
- Confidence evolution visible in real-time

---

## Implementation Roadmap

### Phase 1: Core Resource Flow (2 weeks)
- Artifact + Content resources (PostgreSQL)
- Instant tier endpoints (`POST /artifacts`, `GET /entities/search`)
- URL extraction worker (lazy tier)
- Entity extraction worker with fuzzy matching
- Test with Hong Kong Fire case (EN + ZH sources)

### Phase 2: Graph Confidence (2 weeks)
- Edge schema (PostgreSQL + Neo4j)
- Structural confidence computation
- Temporal decay mechanism
- Confidence recomputation worker (hourly for hot entities)
- Test confidence evolution over 30 days

### Phase 3: Multi-Language (1 week)
- Language detection in instant tier
- Cross-language entity linking (Wikidata bridge)
- Multi-language event merging
- Test with 3+ languages (EN, ZH, FR)

### Phase 4: Conflict Resolution (1 week)
- Dispute API endpoint
- Weighted voting worker
- Stake integration (future: economic layer)
- Test with "2000 wrong" scenario

### Phase 5: Extensibility (2 weeks)
- Image artifact extractor (OCR)
- Video artifact extractor (Whisper + frame OCR)
- Document artifact extractor (PyPDF2)
- Test mixed inputs (URL + image + video about same event)

---

## Conclusion

This architecture is designed to build a **breathing knowledge base** that:

1. **Scales** from webapp to mobile to SaaS
2. **Evolves** with evidence (not locked to edit history)
3. **Respects** reality (multi-source, multi-language, temporal decay)
4. **Empowers** community (stake-weighted dispute resolution)
5. **Focuses** on facts (events, not stories)
6. **Responds** instantly (< 500ms for user-facing APIs)
7. **Degrades** gracefully (workers fail independently)

**Beyond Wikipedia**: Where Wikipedia freezes knowledge in articles, we let evidence flow through a living graph. Where Wikipedia resolves conflicts through moderator judgment, we use weighted consensus. Where Wikipedia separates languages, we link them via canonical IDs.

**The result**: A knowledge base that breathes - rising and falling with the tide of evidence, adapting to reality faster than any human-edited encyclopedia could.

---

## References

- [REST Maturity Model](https://martinfowler.com/articles/richardsonMaturityModel.html)
- [Event-Driven Architecture](https://martinfowler.com/articles/201701-event-driven.html)
- [Response Time Limits](https://www.nngroup.com/articles/response-times-3-important-limits/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Knowledge Graph Embeddings](https://arxiv.org/abs/1503.00759)
- [Temporal Knowledge Graphs](https://arxiv.org/abs/2004.04926)
- [Wikidata](https://www.wikidata.org/)
- [PageRank](https://en.wikipedia.org/wiki/PageRank)
- [Reputation Systems](https://en.wikipedia.org/wiki/Reputation_system)
