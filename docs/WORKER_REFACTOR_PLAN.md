# Worker Refactor Plan: Demo → Production Gen2

**Goal:** Refactor demo workers to production-grade using Gen1 battle-tested components + Gen2 architecture

**Date:** 2025-11-29
**Status:** Planning

---

## Current State Analysis

### Demo Workers (What We Have)

**Location:** `backend/worker_*.py`, `backend/workers.py`

**Architecture:**
```
Redis Queue → Worker → PostgreSQL (demo schema)
```

**Good patterns (Keep!):**
- ✅ Redis-based async queuing (simple, fast, proven)
- ✅ Independent worker processes (clean separation)
- ✅ Uses production `semantic_analyzer.py`
- ✅ Modular design (workers are independent)

**Wrong assumptions (Fix in Gen2):**
- ❌ **Pipeline thinking** - Demo assumes URL → content → entities → claims → events
- ❌ **Sequential processing** - Each step waits for previous step
- ❌ **Deterministic flow** - Assumes every URL follows same path

**Reality for Gen2 (Event-Driven):**
- ✅ **Events trigger events** - Page extracted → MAY trigger entity extraction
- ✅ **Probability-driven** - Confidence thresholds determine what happens next
- ✅ **Non-linear** - Multiple paths based on content type, language, quality
- ✅ **Autonomous workers** - Each worker decides what to do based on state

**Technical issues (Also fix):**
- ❌ Uses demo schema (flat tables, no `core.*` namespacing)
- ❌ Simple extraction (urllib + trafilatura only, ~40% success rate)
- ❌ No browser pool for JS-heavy sites

---

### Gen1 Workers (Critical Analysis)

**Location:** `../service_farm_gen1/worker.py`, `../service_farm_gen1/services/`

**What Gen1 Does Well (Borrow):**

#### 1. **Extraction Service** (`services/extraction_service/`)
```python
# Gen1 has sophisticated extraction with fallbacks
services/extraction_service/
├── browser_pool.py                    # ✅ Playwright pool (3 concurrent browsers)
├── extractors/
│   ├── lightweight_mobile_extractor.py   # ✅ Fast mobile UA
│   ├── pooled_playwright_extractor.py    # ✅ JS rendering
│   └── social_bot_extractor.py           # ✅ Social media
├── intelligence/
│   ├── method_selector.py                # ✅ Choose best extractor per domain
│   ├── failure_classifier.py             # ✅ Why extraction failed
│   └── domain_intelligence_store.py      # ✅ Learn from past attempts
└── smart_extraction_orchestrator.py      # ✅ Orchestrates all above
```

**Why this matters:**
- Gen1 extraction success rate: ~85% (vs demo: ~40%)
- Handles paywalls, JS-heavy sites, social media
- Intelligent fallback strategies

#### 2. **Task Store Patterns** (`services/shared/task_store_postgres_async.py`)
```python
# Gen1 has atomic task claiming
async def poll_task(self, timeout_seconds=5):
    """
    Atomically claim next pending task using PostgreSQL row-level locks

    SELECT ... FOR UPDATE SKIP LOCKED
    - Only one worker gets the task
    - No race conditions
    - Graceful degradation if no tasks
    """
```

**Why this matters:**
- Demo uses Redis queue (simple but loses tasks on crash)
- Gen1 uses PostgreSQL as queue (durable, atomic)
- Can scale workers without duplicate processing

#### 3. **Browser Pool** (`services/extraction_service/browser_pool.py`)
```python
class BrowserPool:
    """
    Manages 3 concurrent Playwright browsers

    - Semaphore limits concurrency
    - Browsers are reused (not recreated per request)
    - Automatic cleanup on errors
    - Proxy rotation support
    """
```

**Why this matters:**
- Demo has no browser support (can't handle JS)
- Gen1 browser pool handles CNN, NYT, etc.

#### 4. **Error Handling** (`handlers.py`)
```python
# Gen1 has sophisticated error classification
try:
    result = await extract()
except Exception as e:
    error_type = classify_error(e)  # paywall | bot_detected | timeout | etc.

    if error_type == "bot_detected":
        # Try different UA / proxy
        await retry_with_different_method()
    elif error_type == "paywall":
        # Mark as paywall, don't retry
        await mark_paywall(task_id)
    elif error_type == "timeout":
        # Retry with longer timeout
        await retry_with_extended_timeout()
```

**Why this matters:**
- Demo just fails (loses data)
- Gen1 retries intelligently (saves ~30% of "failed" extractions)

#### 5. **Monitoring** (Gen1 patterns)
```python
# Gen1 logs detailed metrics
{
    "worker_name": "worker-3",
    "task_id": "abc123",
    "url": "...",
    "extraction_method": "playwright_mobile",
    "duration_ms": 2450,
    "success": true,
    "word_count": 450,
    "retry_count": 1
}
```

---

## Gen2 Worker Architecture (Event-Driven)

### Core Principles

1. **Event-driven, not pipeline** - Workers react to state changes, not sequential flow
2. **Probability-driven** - Confidence thresholds determine next actions
3. **Autonomous workers** - Each worker decides independently what to do
4. **Schema-aware** - Use `core.*` tables
5. **Observable** - Detailed logging, metrics

### Queue-Driven Flow (Efficient, No Duplication)

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  POST /artifacts → Create stub (status='stub')               │
│  Enqueue extraction job → Returns immediately (< 100ms)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ LPUSH queue:extraction:high
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Redis Job Queues                        │
│  queue:extraction:high  (FIFO)                               │
│  queue:semantic:high    (FIFO)                               │
│  queue:event:high       (FIFO)                               │
│  queue:enrichment:high  (FIFO)                               │
└─────────────────────────────────────────────────────────────┘
       │               │               │               │
       │ BRPOP         │ BRPOP         │ BRPOP         │ BRPOP
       ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ Extraction   │ │ Semantic     │ │ Event    │ │ Enrichment   │
│ Worker 1,2,3 │ │ Worker 1,2   │ │ Worker 1 │ │ Worker 1     │
│              │ │              │ │          │ │              │
│ Decides:     │ │ Decides:     │ │ Decides: │ │ Decides:     │
│ - Worth?     │ │ - Quality?   │ │ - Enough │ │ - Has QID?   │
│ - Method?    │ │ - Language?  │ │   claims │ │ - Worth it?  │
│ - Retry?     │ │              │ │   for    │ │              │
│              │ │              │ │   event? │ │              │
│ Enqueues:    │ │ Enqueues:    │ │          │ │              │
│ → semantic   │ │ → event      │ │ Enqueues:│ │              │
│ OR retry     │ │ → enrichment │ │ → enrich │ │              │
└──────────────┘ └──────────────┘ └──────────┘ └──────────────┘
       │                     │               │              │
       │   All update PostgreSQL state (status, confidence)  │
       ▼                     ▼               ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                   PostgreSQL Gen2 Schema                      │
│  core.pages (status: stub → extracting → extracted)         │
│  core.entities (confidence evolves)                          │
│  core.claims (linked as discovered)                          │
│  core.events (formed when threshold met)                     │
└──────────────────────────────────────────────────────────────┘
```

**Key: Each job consumed by exactly ONE worker (BRPOP = atomic pop)**

### Key Differences from Pipeline

**Pipeline (Old thinking):**
```
URL → [wait] → Content → [wait] → Entities → [wait] → Claims → [wait] → Event
```
- Sequential, blocking
- Every URL follows same path
- Rigid

**Event-Driven (Gen2):**
```
URL created
  ↓
  IF confidence(extractable) > 0.7 → Try extraction
    ↓
    IF extracted AND word_count > 100 → Extract entities
      ↓
      IF entities.count > 3 → Extract claims
        ↓
        IF claims.count > 5 AND temporal_overlap > 0.8 → Form event
          ↓
          IF event.confidence > 0.9 → Enrich with Wikidata
```
- Non-blocking, probabilistic
- Different paths for different content
- Adaptive

### Example: Low-Quality Page

**Pipeline approach:** Waste resources processing garbage
```
URL → Extract (fails) → Retry → Extract (succeeds, 20 words) →
Extract entities (none) → Extract claims (none) → No event → WASTED TIME
```

**Event-driven approach:** Stop early
```
URL created → Extract → word_count=20 → confidence(useful)=0.2 → STOP
(No entity extraction triggered, saved resources)
```

### Example: High-Quality Page

**Pipeline approach:** Same as above, just happens to succeed
```
URL → Extract → Entities → Claims → Event (maybe)
```

**Event-driven approach:** Adaptive processing
```
URL created → Extract → word_count=800 → confidence(useful)=0.95 →
Extract entities (15 found) → confidence(has_claims)=0.9 → Extract claims (12 found) →
Claims overlap temporally → confidence(event)=0.85 → Form event →
Event has 4+ entities → confidence(enrichable)=0.9 → Enrich from Wikidata
```

---

## Refactor Tasks (Priority Order)

### Phase 1: Event Bus + Worker Foundation (Week 1)

**Goal:** Event-driven infrastructure, not job queue pipeline

#### Task 1.1: Redis Queue System (Proven from Demo)
**File:** `backend/services/job_queue.py` (adapt from demo)

```python
class JobQueue:
    """
    Redis-based job queue system

    Queues (one per worker type):
    - 'queue:extraction:high'  → Extraction workers consume
    - 'queue:semantic:high'    → Semantic workers consume
    - 'queue:event:high'       → Event workers consume
    - 'queue:enrichment:high'  → Enrichment workers consume

    Workers use BRPOP (blocking pop) for efficient consumption
    Each job is consumed by exactly ONE worker (round-robin)
    """

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def enqueue(self, queue_name: str, job: dict):
        """
        Add job to queue

        Example:
            await queue.enqueue('queue:extraction:high', {
                'page_id': '...',
                'url': 'https://...'
            })
        """
        await self.redis.lpush(queue_name, json.dumps(job))

    async def dequeue(self, queue_name: str, timeout: int = 5) -> dict | None:
        """
        Blocking pop from queue (BRPOP)

        Blocks until job available or timeout
        Returns None on timeout
        """
        result = await self.redis.brpop(queue_name, timeout=timeout)
        if result:
            return json.loads(result[1])
        return None
```

**Why queues instead of pub/sub events:**
- ✅ **Efficient**: Each job consumed by exactly ONE worker (no duplicate work)
- ✅ **Simple**: No optimistic locking needed
- ✅ **Proven**: Demo pattern worked well
- ✅ **Scalable**: Add more workers → automatic load balancing

---

#### Task 1.2: Worker Base Class
**File:** `backend/services/worker_base.py` (new)

```python
class BaseWorker:
    """
    Base class for all Gen2 workers

    Borrowed from Gen1:
    - Signal handling (graceful shutdown)
    - Health checks
    - Metrics logging

    Borrowed from Demo:
    - Redis queue consumption (BRPOP)
    - Simple, efficient worker loop

    New for Gen2:
    - Autonomous decision-making (should_process)
    - Confidence-threshold driven
    - Schema-aware (core.* tables)
    """

    def __init__(self, pool: asyncpg.Pool, job_queue: JobQueue, worker_name: str, queue_name: str):
        self.pool = pool
        self.job_queue = job_queue
        self.worker_name = worker_name
        self.queue_name = queue_name
        self.running = False

    async def start(self):
        """
        Main worker loop (from demo pattern)

        Continuously:
        1. BRPOP from queue (blocks until job available)
        2. Decide if should process (autonomous)
        3. Process if threshold met
        4. Enqueue next jobs
        """
        print(f"[{self.worker_name}] Started, listening on {self.queue_name}")

        while self.running:
            try:
                # Blocking pop from queue
                job = await self.job_queue.dequeue(self.queue_name, timeout=5)

                if job:
                    # Fetch current state from PostgreSQL
                    state = await self.get_state(job)

                    # Autonomous decision
                    should_process, confidence = await self.should_process(state)

                    if should_process:
                        await self.process(job, state)
                    else:
                        logger.info(f"Skipping job {job}: confidence={confidence:.2f}")

            except Exception as e:
                logger.error(f"[{self.worker_name}] Error: {e}")
                await asyncio.sleep(1)

    async def process(self, job: dict, state: dict):
        """Override in subclass - do the actual work"""
        raise NotImplementedError

    async def should_process(self, state: dict) -> tuple[bool, float]:
        """
        Autonomous decision based on state

        Returns:
            (should_process, confidence)

        Example:
            if state['word_count'] > 100 and state['language'] == 'en':
                return (True, 0.9)
            return (False, 0.2)
        """
        raise NotImplementedError

    async def get_state(self, job: dict) -> dict:
        """Fetch current state from PostgreSQL"""
        raise NotImplementedError
```

**Borrow patterns from:**
- Demo: BRPOP queue consumption
- Gen1: Signal handling, metrics
**Critical: Uses queues (efficient), NOT pub/sub events**

---

### Phase 2: Extraction Worker (Week 1-2)

**Goal:** Production-quality URL extraction

#### Task 2.1: Port Browser Pool
**File:** `backend/services/browser_pool.py` (copy from Gen1)

```bash
# Copy Gen1 browser pool with minimal changes
cp ../service_farm_gen1/services/extraction_service/browser_pool.py backend/services/
```

**Changes needed:**
- Update imports (Gen1 → Gen2 paths)
- Keep all Gen1 logic (semaphore, browser lifecycle, etc.)

---

#### Task 2.2: Port Extraction Orchestrator
**File:** `backend/services/extraction_orchestrator.py` (adapted from Gen1)

**Borrow from Gen1:**
- `smart_extraction_orchestrator.py`
- All extractors (`extractors/`)
- Intelligence modules (`intelligence/`)

**Adapt for Gen2:**
```python
class ExtractionOrchestrator:
    """
    Orchestrates URL extraction with intelligent fallbacks

    Gen1 patterns:
    - Method selection per domain
    - Browser pool for JS sites
    - Failure classification

    Gen2 changes:
    - Updates core.pages (not extraction_tasks)
    - Uses system.worker_jobs
    """

    async def extract_url(self, page_id: uuid, url: str) -> dict:
        """
        1. Choose best extraction method (Gen1 intelligence)
        2. Try extraction
        3. If fail, classify error and retry
        4. Update core.pages
        5. Queue semantic worker
        """
```

**Copy from Gen1:**
```bash
cp -r ../service_farm_gen1/services/extraction_service backend/services/
# Then refactor to use Gen2 schema
```

---

#### Task 2.3: Extraction Worker Implementation (Event-Driven)
**File:** `backend/workers/extraction_worker.py` (refactor from demo)

```python
from services.worker_base import BaseWorker
from services.extraction_orchestrator import ExtractionOrchestrator

class ExtractionWorker(BaseWorker):
    """
    Event-Driven Extraction Worker

    Subscribes to: ['page.created', 'page.retry_extraction']

    Decides autonomously:
    - Is this URL worth extracting? (confidence threshold)
    - Which extraction method to use?
    - Should I retry on failure?

    Emits:
    - 'page.extracted'     (on success, word_count > threshold)
    - 'page.low_quality'   (extracted but not useful)
    - 'page.failed'        (permanent failure)
    """

    def __init__(self, pool, event_bus, worker_name):
        super().__init__(pool, event_bus, worker_name)
        self.subscribed_events = ['page.created', 'page.retry_extraction']
        self.orchestrator = ExtractionOrchestrator(pool)

        # Confidence thresholds (configurable)
        self.MIN_WORD_COUNT = 100
        self.MIN_EXTRACTION_CONFIDENCE = 0.7

    async def process(self, job: dict, state: dict):
        """
        Process extraction job (consumed from queue)

        Job structure:
        {
            'page_id': '...',
            'url': 'https://...',
            'retry_count': 0
        }
        """
        page_id = job['page_id']
        url = job['url']
        retry_count = job.get('retry_count', 0)

        # Update status to extracting
        await self.update_page_status(page_id, 'extracting')

        try:
            # Extract (Gen1 intelligence)
            result = await self.orchestrator.extract_url(page_id, url)

            # Save result
            await self.save_extraction_result(page_id, result)

            # Enqueue next job based on quality
            if result['word_count'] >= self.MIN_WORD_COUNT:
                # High quality → enqueue semantic analysis
                await self.job_queue.enqueue('queue:semantic:high', {
                    'page_id': page_id
                })
                await self.update_page_status(page_id, 'extracted')
            else:
                # Low quality → don't waste resources
                await self.update_page_status(page_id, 'low_quality')
                logger.info(f"Page {page_id} is low quality (word_count={result['word_count']})")

        except Exception as e:
            # Classify error (Gen1 pattern)
            error_type = await self.classify_error(e)

            # Retry logic
            if error_type in ['bot_detected', 'timeout'] and retry_count < 3:
                # Re-enqueue with incremented retry count
                await self.job_queue.enqueue('queue:extraction:high', {
                    'page_id': page_id,
                    'url': url,
                    'retry_count': retry_count + 1,
                    'previous_error': error_type
                })
                logger.warning(f"Retrying {page_id} (attempt {retry_count + 1}/3, reason: {error_type})")
            else:
                # Permanent failure
                await self.update_page_status(page_id, 'failed')
                await self.save_error(page_id, str(e), error_type)
                logger.error(f"Page {page_id} permanently failed: {error_type}")

    async def should_process(self, page: dict) -> tuple[bool, float]:
        """
        Autonomous decision: Is this URL worth extracting?

        Factors:
        - Domain reputation (from intelligence store)
        - Content type hints (from URL)
        - Previous extraction attempts
        """
        # Check if already extracted
        if page['status'] in ['extracted', 'failed']:
            return (False, 0.0)

        # Check domain intelligence (Gen1 pattern)
        domain_score = await self.orchestrator.get_domain_confidence(page['url'])

        # Check content type hints
        content_type_score = self.estimate_content_type(page['url'])

        # Combined confidence
        confidence = (domain_score * 0.6 + content_type_score * 0.4)

        return (confidence >= self.MIN_EXTRACTION_CONFIDENCE, confidence)

    def estimate_content_type(self, url: str) -> float:
        """
        Quick heuristics for URL quality

        Examples:
        - .pdf → 0.3 (might not extract well)
        - news site → 0.9
        - social media → 0.6
        """
        if url.endswith('.pdf'):
            return 0.3
        if any(domain in url for domain in ['nytimes.com', 'bbc.com', 'reuters.com']):
            return 0.9
        if any(domain in url for domain in ['twitter.com', 'facebook.com']):
            return 0.6
        return 0.5  # Default

    async def classify_error(self, error: Exception) -> str:
        """
        Classify extraction error (from Gen1)

        Returns: 'bot_detected' | 'timeout' | 'paywall' | 'network' | 'unknown'
        """
        error_str = str(error).lower()

        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout'
        elif 'captcha' in error_str or '403' in error_str:
            return 'bot_detected'
        elif 'paywall' in error_str or 'subscription' in error_str:
            return 'paywall'
        elif 'network' in error_str or 'connection' in error_str:
            return 'network'
        else:
            return 'unknown'
```

**Key differences from demo:**
- ✅ Uses Gen2 schema (core.pages, not demo tables)
- ✅ **Retry logic with error classification** (from Gen1)
- ✅ Autonomous `should_process()` decision (probability-driven)
- ✅ Quality-based routing (high quality → semantic, low quality → stop)
- ✅ No duplicate work (Redis queue ensures one worker per job)

---

### Phase 3: Semantic Worker (Week 2)

**Goal:** Entity + claim extraction

#### Task 3.1: Port Semantic Analyzer
**Status:** ✅ Already have `backend/semantic_analyzer.py` (production quality)

**Changes needed:**
- Use `core.entities`, `core.claims`, `core.edges`
- Add Wikidata QID lookup
- Store in Gen2 schema

---

#### Task 3.2: Semantic Worker Implementation (Event-Driven)
**File:** `backend/workers/semantic_worker.py` (refactor from demo)

```python
from services.worker_base import BaseWorker
from semantic_analyzer import SemanticAnalyzer

class SemanticWorker(BaseWorker):
    """
    Event-Driven Semantic Worker

    Subscribes to: ['page.extracted', 'page.updated']

    Decides autonomously:
    - Is content quality high enough? (word_count, language)
    - Are there likely entities/claims to extract?
    - Should I run expensive LLM analysis?

    Emits:
    - 'entity.discovered'  (for each new entity)
    - 'claim.extracted'    (for each new claim)
    - 'page.semantic_done' (analysis complete)
    """

    def __init__(self, pool, event_bus, worker_name):
        super().__init__(pool, event_bus, worker_name)
        self.subscribed_events = ['page.extracted', 'page.updated']
        self.semantic_analyzer = SemanticAnalyzer()

        # Confidence thresholds
        self.MIN_WORD_COUNT = 100
        self.MIN_ENTITY_CONFIDENCE = 0.6
        self.MIN_CLAIM_CONFIDENCE = 0.7

    async def on_event(self, event_type: str, payload: dict):
        """
        React to page.extracted event

        1. Fetch page content
        2. Decide if semantic analysis worth it
        3. Run semantic_analyzer (entities + claims)
        4. Fuzzy match entities (deduplication)
        5. Store results
        6. Emit events for each discovered entity/claim
        """
        page_id = payload['page_id']

        # 1. Get current state
        page = await self.get_page_with_content(page_id)

        # 2. Autonomous decision
        should_analyze, confidence = await self.should_process(page)

        if not should_analyze:
            logger.info(f"Skipping semantic analysis for {page_id}: confidence={confidence:.2f}")
            return

        try:
            # 3. Run semantic analysis (production semantic_analyzer.py)
            result = await self.semantic_analyzer.analyze(
                text=page['content_text'],
                language=page['language']
            )

            # 4. Process entities (fuzzy matching to avoid duplicates)
            for entity_data in result['entities']:
                # Check if entity already exists (fuzzy match)
                existing = await self.fuzzy_match_entity(
                    entity_data['name'],
                    entity_data['type']
                )

                if existing:
                    entity_id = existing['id']
                    # Update confidence if higher
                    await self.update_entity_confidence(entity_id, entity_data)
                else:
                    # Create new entity
                    entity_id = await self.create_entity(entity_data)

                    # Emit event for new entity
                    await self.event_bus.publish('entity.discovered', {
                        'entity_id': entity_id,
                        'entity_type': entity_data['type'],
                        'canonical_name': entity_data['name'],
                        'confidence': entity_data['confidence']
                    })

                # Link page to entity
                await self.link_page_entity(page_id, entity_id, entity_data['mention_count'])

            # 5. Process claims
            for claim_data in result['claims']:
                if claim_data['confidence'] >= self.MIN_CLAIM_CONFIDENCE:
                    claim_id = await self.create_claim(page_id, claim_data)

                    # Emit event for new claim
                    await self.event_bus.publish('claim.extracted', {
                        'claim_id': claim_id,
                        'page_id': page_id,
                        'event_time': claim_data.get('event_time'),
                        'confidence': claim_data['confidence']
                    })

            # 6. Emit completion event
            await self.event_bus.publish('page.semantic_done', {
                'page_id': page_id,
                'entity_count': len(result['entities']),
                'claim_count': len([c for c in result['claims'] if c['confidence'] >= self.MIN_CLAIM_CONFIDENCE])
            })

        except Exception as e:
            logger.error(f"Semantic analysis failed for {page_id}: {e}")
            await self.event_bus.publish('page.semantic_failed', {
                'page_id': page_id,
                'error': str(e)
            })

    async def should_process(self, page: dict) -> tuple[bool, float]:
        """
        Autonomous decision: Is semantic analysis worth it?

        Factors:
        - Word count (too short → low value)
        - Language (supported languages only)
        - Content type (article vs list vs navigation)
        - Already analyzed?
        """
        # Already analyzed
        if page.get('semantic_analyzed_at'):
            return (False, 0.0)

        # Too short
        if page['word_count'] < self.MIN_WORD_COUNT:
            return (False, 0.3)

        # Language check
        if page['language'] not in ['en', 'zh', 'es', 'fr']:
            return (False, 0.4)

        # Content type estimation (heuristic)
        content_quality = self.estimate_content_quality(page['content_text'])

        return (content_quality >= 0.7, content_quality)

    def estimate_content_quality(self, text: str) -> float:
        """
        Quick heuristics for content quality

        - Sentence count
        - Paragraph structure
        - Named entity density
        """
        sentences = text.split('.')
        if len(sentences) < 5:
            return 0.4

        # Check for paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) < 2:
            return 0.6

        # Good quality article-like content
        return 0.9

    async def fuzzy_match_entity(self, name: str, entity_type: str) -> dict:
        """
        Fuzzy match to avoid duplicate entities

        Uses PostgreSQL similarity (pg_trgm extension)
        """
        query = """
        SELECT id, canonical_name, similarity(canonical_name, $1) AS score
        FROM core.entities
        WHERE entity_type = $2
          AND similarity(canonical_name, $1) > 0.7
        ORDER BY score DESC
        LIMIT 1
        """
        return await self.pool.fetchrow(query, name, entity_type)
```

**Key differences from pipeline version:**
- Event-driven: Reacts to `page.extracted`, not polled jobs
- Probability-driven: `should_process()` checks content quality
- Emits granular events: `entity.discovered`, `claim.extracted` (not just "done")
- Autonomous decisions: Skips low-quality content early

---

### Phase 4: Event Worker (Week 2-3)

**Goal:** Claim clustering into events

#### Task 4.1: Event Worker (Event-Driven Clustering)
**File:** `backend/workers/event_worker.py` (from demo + IMPLEMENTATION_BASIS.md)

```python
from services.worker_base import BaseWorker
from services.event_clustering import EventClusteringService

class EventWorker(BaseWorker):
    """
    Event-Driven Event Formation Worker

    Subscribes to: ['claim.extracted', 'page.semantic_done']

    Decides autonomously:
    - Are there enough claims to form/update an event?
    - Do claims have temporal overlap?
    - Is cross-language clustering possible?

    Emits:
    - 'event.formed'   (new event created)
    - 'event.updated'  (existing event updated with new claims)
    - 'event.merged'   (two events merged)

    Based on IMPLEMENTATION_BASIS.md:
    - Recursive multi-pass clustering
    - Temporal phase detection
    """

    def __init__(self, pool, event_bus, worker_name):
        super().__init__(pool, event_bus, worker_name)
        self.subscribed_events = ['claim.extracted', 'page.semantic_done']
        self.clustering_service = EventClusteringService(pool)

        # Confidence thresholds
        self.MIN_CLAIMS_FOR_EVENT = 3
        self.MIN_TEMPORAL_OVERLAP = 0.7
        self.MIN_EVENT_COHERENCE = 0.6

    async def on_event(self, event_type: str, payload: dict):
        """
        React to claim.extracted or page.semantic_done

        Strategy:
        - Don't process every single claim individually
        - Wait for page.semantic_done (batch processing)
        - Check if page's claims can form/join events
        """
        if event_type == 'page.semantic_done':
            page_id = payload['page_id']

            # 1. Get all claims from this page
            claims = await self.get_page_claims(page_id)

            if len(claims) < self.MIN_CLAIMS_FOR_EVENT:
                logger.info(f"Page {page_id} has only {len(claims)} claims, skipping event formation")
                return

            # 2. Autonomous decision: Worth attempting clustering?
            should_cluster, confidence = await self.should_process(claims)

            if not should_cluster:
                logger.info(f"Skipping event clustering for {page_id}: confidence={confidence:.2f}")
                return

            # 3. Run recursive clustering (IMPLEMENTATION_BASIS.md)
            try:
                result = await self.clustering_service.cluster_claims(
                    claims=claims,
                    time_window_short=timedelta(days=2),
                    time_window_long=timedelta(days=14)
                )

                # 4. Process each discovered event cluster
                for cluster in result['clusters']:
                    # Check if this is a new event or update to existing
                    existing_event = await self.find_matching_event(cluster)

                    if existing_event:
                        # Update existing event
                        await self.update_event(existing_event['id'], cluster)

                        await self.event_bus.publish('event.updated', {
                            'event_id': existing_event['id'],
                            'new_claim_count': len(cluster['claim_ids']),
                            'confidence': cluster['coherence_score']
                        })
                    else:
                        # Create new event
                        event_id = await self.create_event(cluster)

                        await self.event_bus.publish('event.formed', {
                            'event_id': event_id,
                            'claim_count': len(cluster['claim_ids']),
                            'entity_count': len(cluster['entity_ids']),
                            'languages': cluster['languages'],
                            'confidence': cluster['coherence_score'],
                            'event_start': cluster['event_start'],
                            'event_end': cluster['event_end']
                        })

            except Exception as e:
                logger.error(f"Event clustering failed for page {page_id}: {e}")

    async def should_process(self, claims: list) -> tuple[bool, float]:
        """
        Autonomous decision: Worth attempting event clustering?

        Factors:
        - Number of claims (more → higher confidence)
        - Temporal distribution (concentrated → higher confidence)
        - Entity overlap (shared entities → higher confidence)
        """
        if len(claims) < self.MIN_CLAIMS_FOR_EVENT:
            return (False, 0.2)

        # Check temporal distribution
        temporal_concentration = self.calculate_temporal_concentration(claims)

        # Check entity overlap
        entity_overlap = self.calculate_entity_overlap(claims)

        # Combined confidence
        confidence = (temporal_concentration * 0.6 + entity_overlap * 0.4)

        return (confidence >= self.MIN_EVENT_COHERENCE, confidence)

    def calculate_temporal_concentration(self, claims: list) -> float:
        """
        How concentrated are claims temporally?

        Example:
        - All claims within 2 days → 0.95
        - Claims spread over 30 days → 0.3
        """
        if not claims:
            return 0.0

        event_times = [c['event_time'] for c in claims if c.get('event_time')]
        if len(event_times) < 2:
            return 0.5

        time_span = max(event_times) - min(event_times)
        days_span = time_span.total_seconds() / 86400

        if days_span < 2:
            return 0.95
        elif days_span < 7:
            return 0.8
        elif days_span < 14:
            return 0.6
        else:
            return 0.3

    def calculate_entity_overlap(self, claims: list) -> float:
        """
        How many entities are shared across claims?

        High overlap → likely same event
        """
        # This would query core.claim_entities
        # Simplified here
        return 0.7  # Placeholder

    async def find_matching_event(self, cluster: dict) -> dict:
        """
        Check if cluster matches existing event

        Uses:
        - Temporal overlap
        - Entity overlap
        - Location similarity
        """
        query = """
        SELECT e.*
        FROM core.events e
        WHERE e.event_start <= $1 + interval '2 days'
          AND e.event_end >= $2 - interval '2 days'
          AND e.confidence >= $3
        ORDER BY
            tsrange(e.event_start, e.event_end) <-> tsrange($1, $2)
        LIMIT 1
        """
        return await self.pool.fetchrow(
            query,
            cluster['event_start'],
            cluster['event_end'],
            0.6
        )
```

**Key differences from pipeline version:**
- Event-driven: Reacts to `page.semantic_done` (batch), not continuous polling
- Probability-driven: `should_process()` checks temporal concentration
- Non-linear: Can update existing events OR create new ones
- Autonomous: Decides event boundaries based on confidence

---

### Phase 5: Enrichment Worker (Week 3)

**Goal:** Wikidata enrichment

#### Task 5.1: Wikidata Service
**File:** `backend/services/wikidata_service.py` (new)

```python
class WikidataService:
    """
    Enrich entities with Wikidata

    1. Search Wikidata by name
    2. Get QID
    3. Fetch properties (P31, P17, etc.)
    4. Store in core.entities (wikidata_qid, wikidata_properties)
    5. Create core.edges from Wikidata relationships
    """
```

---

## Migration Checklist

### From Gen1 (Copy)
- [x] Browser pool (`browser_pool.py`)
- [x] Extraction orchestrator (`smart_extraction_orchestrator.py`)
- [x] All extractors (`extractors/`)
- [x] Intelligence modules (`intelligence/`)
- [x] Task store patterns (atomic claiming)
- [x] Error classification (`failure_classifier.py`)

### From Demo (Refactor)
- [x] Worker structure (`worker_*.py`)
- [x] Semantic analyzer (already production quality)
- [x] Event clustering logic

### New for Gen2
- [x] Job queue (`system.worker_jobs`)
- [x] Schema-aware workers (`core.*` tables)
- [x] Wikidata enrichment
- [x] Multi-language support

---

## Success Criteria

### Extraction Worker
- ✅ Handles JS-heavy sites (uses browser pool)
- ✅ Success rate > 80% (Gen1 level)
- ✅ Intelligent retries (classify errors)
- ✅ Updates `core.pages` correctly

### Semantic Worker
- ✅ Extracts entities with fuzzy matching (no duplicates)
- ✅ Links Wikidata QIDs
- ✅ Creates claims with temporal info
- ✅ Stores in `core.entities`, `core.claims`

### Event Worker
- ✅ Forms events from claim clusters
- ✅ Detects temporal phases
- ✅ Creates hierarchy (micro/meso/macro)
- ✅ Links cross-language pages

### Overall
- ✅ End-to-end: URL → events in < 30s (p95)
- ✅ No duplicate entities
- ✅ Graceful error handling
- ✅ Observable (logs, metrics)

---

## File Structure (After Refactor)

```
backend/
├── workers/
│   ├── extraction_worker.py       # URL → content (Gen1 patterns)
│   ├── semantic_worker.py         # Content → entities/claims
│   ├── event_worker.py            # Claims → events
│   └── enrichment_worker.py       # Wikidata enrichment
│
├── services/
│   ├── worker_base.py             # Base class (Gen1 patterns)
│   ├── job_queue.py               # PostgreSQL queue (Gen1 patterns)
│   ├── browser_pool.py            # From Gen1 (copied)
│   ├── extraction_orchestrator.py # From Gen1 (adapted)
│   ├── extractors/                # From Gen1 (copied)
│   ├── intelligence/              # From Gen1 (copied)
│   ├── semantic_analyzer.py       # Existing (production quality)
│   └── wikidata_service.py        # New for Gen2
│
├── main.py                        # FastAPI API
└── requirements.txt
```

---

## Next Steps

1. **Start with Phase 1** - Job queue + base worker (foundation)
2. **Copy Gen1 extraction service** - Browser pool + orchestrator
3. **Refactor extraction worker** - Use Gen2 schema
4. **Test with real URLs** - Validate extraction quality
5. **Then semantic + event workers** - Complete pipeline

---

## Questions Resolved

### ✅ Architecture Paradigm
**Question:** Pipeline or event-driven?
**Answer:** Event-driven and probability-driven
- Workers react to events (Redis pub/sub)
- Autonomous decision-making based on confidence thresholds
- Non-linear processing (not A→B→C)

### ✅ Job Queue System
**Question:** PostgreSQL queue or Redis?
**Answer:** Redis event bus for coordination, PostgreSQL for state
- Redis: Event notifications (`page.created`, `entity.discovered`, etc.)
- PostgreSQL: Source of truth (current state, confidence scores)
- No job claiming/locking (workers decide independently)

## Questions Still to Resolve

1. **Database connection:** Gen1 uses remote PostgreSQL (IPv6). Should Gen2 workers connect to same DB or use local test DB first?
2. **Browser pool resources:** Do we have GPU for headless Chrome in prod? Or use lightweight mobile extractors?
3. **Wikidata rate limits:** Need API key? Or use public endpoint with rate limiting?
4. **Monitoring:** Use Gen1's logging patterns or set up new observability (Prometheus, etc.)?
5. **Event deduplication:** Multiple workers might react to same event - how to prevent duplicate work?
6. **Confidence threshold configuration:** Should thresholds be per-worker env vars or in database?

---

## Event Flow Examples

### Example 1: High-Quality News Article (Happy Path)

```
1. User submits URL via API
   ↓
   FastAPI creates core.pages (status='stub')
   ↓
   Emit: page.created {page_id, url}

2. ExtractionWorker hears page.created
   ↓
   Fetch page state → should_process() → confidence=0.9 ✅
   ↓
   Extract using browser pool → word_count=850
   ↓
   Update core.pages (status='extracted')
   ↓
   Emit: page.extracted {page_id, word_count=850, language='en'}

3. SemanticWorker hears page.extracted
   ↓
   Fetch page content → should_process() → confidence=0.95 ✅
   ↓
   Run semantic_analyzer → 12 entities, 8 claims
   ↓
   Store in core.entities, core.claims
   ↓
   Emit: entity.discovered (×12)
   Emit: claim.extracted (×8)
   Emit: page.semantic_done {page_id, entity_count=12, claim_count=8}

4. EventWorker hears page.semantic_done
   ↓
   Fetch claims → should_process() → confidence=0.85 ✅
   ↓
   Run clustering → finds matching event
   ↓
   Update core.events (add new claims)
   ↓
   Emit: event.updated {event_id, new_claim_count=8}

5. EnrichmentWorker hears entity.discovered (×12)
   ↓
   For each entity → should_process() → check if has QID
   ↓
   Fetch Wikidata → 9 entities enriched, 3 not found
   ↓
   Update core.entities (wikidata_qid, wikidata_properties)
   ↓
   Emit: entity.enriched (×9)

Total time: ~15 seconds
Total events: 24 emitted
Workers involved: 4 (Extraction, Semantic, Event, Enrichment)
```

### Example 2: Low-Quality URL (Early Stop)

```
1. User submits URL via API
   ↓
   FastAPI creates core.pages (status='stub')
   ↓
   Emit: page.created {page_id, url}

2. ExtractionWorker hears page.created
   ↓
   Fetch page state → should_process() → confidence=0.9 ✅
   ↓
   Extract → word_count=25 (short article)
   ↓
   Update core.pages (status='extracted')
   ↓
   Emit: page.low_quality {page_id, word_count=25}

3. SemanticWorker hears page.low_quality
   ↓
   (Worker ignores low_quality events)

STOP - No further processing, saved resources!

Total time: ~3 seconds
Total events: 2 emitted
Workers involved: 1 (Extraction only)
```

### Example 3: Extraction Failure with Retry

```
1. User submits URL via API
   ↓
   FastAPI creates core.pages (status='stub')
   ↓
   Enqueue: queue:extraction:high {page_id, url, retry_count=0}

2. ExtractionWorker (Worker-1) pops job from queue
   ↓
   Fetch page state → should_process() → confidence=0.85 ✅
   ↓
   Extract using lightweight method → FAILS (bot detected)
   ↓
   Classify error → error_type='bot_detected', retry_count=0
   ↓
   Re-enqueue: queue:extraction:high {page_id, url, retry_count=1, previous_error='bot_detected'}

3. ExtractionWorker (Worker-2 or Worker-1) pops retry job
   ↓
   Fetch page state → should_process() → confidence=0.85 ✅
   ↓
   Orchestrator sees previous_error='bot_detected' → uses browser pool (different method)
   ↓
   Extract using browser pool → word_count=650 ✅
   ↓
   Update core.pages (status='extracted')
   ↓
   Enqueue: queue:semantic:high {page_id}

(Continues as Example 1...)

Total retries: 1
Success rate: Improved from 40% → 85% (Gen1 level)
```

---

## Queue and Job Types Reference

### Queues

```
queue:extraction:high     → Extraction workers (URL → content)
queue:semantic:high       → Semantic workers (content → entities/claims)
queue:event:high          → Event workers (claims → events)
queue:enrichment:high     → Enrichment workers (entities → Wikidata)
```

### Job Structures

**Extraction Job:**
```json
{
  "page_id": "uuid",
  "url": "https://...",
  "retry_count": 0,
  "previous_error": null  // or "bot_detected", "timeout", etc.
}
```

**Semantic Job:**
```json
{
  "page_id": "uuid"
}
```

**Event Job:**
```json
{
  "page_id": "uuid",
  "claim_count": 8
}
```

**Enrichment Job:**
```json
{
  "entity_id": "uuid",
  "canonical_name": "Hong Kong",
  "entity_type": "LOCATION"
}
```

### Failure Handling

**Retriable errors** (max 3 attempts):
- `bot_detected` - Retry with browser pool
- `timeout` - Retry with longer timeout
- `network` - Retry after backoff

**Non-retriable errors** (permanent failure):
- `paywall` - Cannot access content
- `unknown` - Unexpected error (after 3 attempts)
- `404` - Page not found

**Retry behavior:**
```python
if retry_count < 3 and error_type in ['bot_detected', 'timeout', 'network']:
    # Re-enqueue with incremented retry_count
    await job_queue.enqueue(queue_name, {
        ...job,
        'retry_count': retry_count + 1,
        'previous_error': error_type
    })
else:
    # Mark as failed, don't retry
    await update_status(page_id, 'failed')
```

---

**Ready to start implementation?** Let me know which phase to begin with!
