# HereNews Architecture

> Single source of truth for system architecture.
> Last updated: 2026-01-05

## Deployment Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REMOTE SERVER (../infra)                            │
│                     Yggdrasil IPv6 + IPv4 accessible                        │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │   PostgreSQL    │  │     Neo4j       │  │     Redis       │            │
│   │   (pgvector)    │  │   (Knowledge    │  │    (Queues)     │            │
│   │                 │  │     Graph)      │  │                 │            │
│   │   Port: 5432    │  │   Port: 7687    │  │   Port: 6380    │            │
│   │   phi_here_db   │  │   uwps-neo4j    │  │   uwps-redis    │            │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    Network: Yggdrasil IPv6 / localhost
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LOCAL / APP SERVER (service_farm)                      │
│                                                                             │
│   ┌─────────────────────────────┐    ┌─────────────────────────────────┐   │
│   │      APP-API Container      │    │       WORKERS Container         │   │
│   │        herenews-api         │    │        herenews-workers         │   │
│   │                             │    │                                 │   │
│   │  ┌───────────────────────┐  │    │  ┌─────────────────────────┐   │   │
│   │  │   FastAPI (uvicorn)   │  │    │  │   Extraction Worker     │   │   │
│   │  │      Port: 8000       │  │    │  │   URL → Page (content)  │   │   │
│   │  └───────────────────────┘  │    │  └─────────────────────────┘   │   │
│   │                             │    │                                 │   │
│   │  ┌───────────────────────┐  │    │  ┌─────────────────────────┐   │   │
│   │  │   Frontend Static     │  │    │  │   Knowledge Worker      │   │   │
│   │  │   (Next.js build)     │  │    │  │   Page → Claims/Entities│   │   │
│   │  └───────────────────────┘  │    │  └─────────────────────────┘   │   │
│   │                             │    │                                 │   │
│   │  Endpoints:                 │    │  ┌─────────────────────────┐   │   │
│   │  /api/* → REST APIs         │    │  │   Weaver Worker (REEE)  │   │   │
│   │  /* → SPA routes            │    │  │   Claims → Surfaces →   │   │   │
│   │                             │    │  │   Incidents (L3)        │   │   │
│   └─────────────────────────────┘    │  └─────────────────────────┘   │   │
│                                      │                                 │   │
│                                      │  ┌─────────────────────────┐   │   │
│                                      │  │   Canonical Worker      │   │   │
│                                      │  │   Incidents → Cases (L4)│   │   │
│                                      │  └─────────────────────────┘   │   │
│                                      │                                 │   │
│                                      │  Queues:                       │   │
│                                      │  queue:extraction:high         │   │
│                                      │  queue:semantic:high           │   │
│                                      │  claims:pending                │   │
│                                      └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Separate Containers?

| Concern | APP-API | WORKERS |
|---------|---------|---------|
| **Scaling** | Horizontal for traffic | Based on queue depth |
| **Resources** | Lightweight, fast response | CPU/memory intensive (LLM) |
| **Deployment** | Update API without workers | Update workers without API |
| **Failure** | API crash doesn't affect workers | Worker crash doesn't affect API |
| **Restart** | Fast restart, no queue drain | Can restart one worker type |

### Container Configuration

```yaml
# docker-compose.yml (service_farm)
services:
  api:
    image: herenews-api
    ports: ["8000:8000"]
    volumes: [./frontend/dist:/static]
    command: uvicorn main:app --host 0.0.0.0 --port 8000

  workers:
    image: herenews-workers
    command: python run_workers.py
    # Or run specific workers:
    # command: python run_workers.py --only extraction
    # command: python run_workers.py --only knowledge
    # command: python run_workers.py --only weaver
    # command: python run_workers.py --only canonical
```

### Network Topology

```
Browser → [8000] API Container → [5432/7687/6380] Remote Databases
                     ↓
              Redis Queues
                     ↓
          Workers Container → [5432/7687] Remote Databases
```

---

## The Two Loops

HereNews runs on two interconnected feedback loops:

```
                    ┌──────────────────────────────────────────────────────────┐
                    │              COMMUNITY LOOP (Human-Driven)               │
                    │                                                          │
                    │   ┌─────────┐    ┌─────────┐    ┌─────────┐             │
                    │   │  Users  │───▶│ Contrib │───▶│ Stakes  │             │
                    │   │         │    │ utions  │    │ (Bounty)│             │
                    │   └────┬────┘    └────┬────┘    └────┬────┘             │
                    │        │              │              │                   │
                    │        │    ┌─────────▼─────────┐    │                   │
                    │        │    │     INQUIRIES     │◀───┘                   │
                    │        │    │  (Questions with  │                        │
                    │        │    │   Belief States)  │                        │
                    │        │    └─────────┬─────────┘                        │
                    │        │              │                                  │
                    │        │    ┌─────────▼─────────┐                        │
                    │        └───▶│      TASKS        │──────▶ Rewards         │
                    │             │  (From MetaClaims)│                        │
                    │             └───────────────────┘                        │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               │ Contributions become Claims
                                               │ Tasks emerge from Gaps
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EPISTEMIC LOOP (Machine-Driven)                       │
│                                                                              │
│   URLs ──▶ Pages ──▶ Claims ──▶ Surfaces ──▶ Incidents ──▶ Cases           │
│                         │           │           │                            │
│                         ▼           ▼           ▼                            │
│                     Entities    Identity    Aboutness                        │
│                                  Edges       Edges                           │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         META-CLAIMS                                   │  │
│   │   Gaps, Conflicts, Single-Source, High-Entropy → Generate TASKS      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Loop Interaction

1. **Epistemic → Community**: Meta-claims (gaps, conflicts) become Tasks with bounties
2. **Community → Epistemic**: User contributions become Claims, feeding back into surfaces
3. **Stakes accelerate**: Higher bounty = more contributor attention = faster resolution
4. **Rewards align incentives**: Information gain (entropy reduction) = credit reward

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                        │
│                    Inquiry → Story → Entity Oriented                         │
│                         (Next.js + React)                                    │
│                       Served as static from API                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪════════════════════════════════════════════
                        APP-API CONTAINER (herenews-api)
════════════════════════════════════╪════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 API                                          │
│                    FastAPI (endpoints_*.py, api/*.py)                        │
│                              Port 8000                                       │
│                                                                              │
│   /api/stories     - Stories (L3 incidents + L4 cases)                      │
│   /api/events      - Canonical cases (legacy endpoint name)                 │
│   /api/surfaces    - L2 identity clusters                                   │
│   /api/inquiry     - Epistemic inquiries with belief states                 │
│   /api/entity      - Entities with Wikidata enrichment                      │
│   /api/claim       - Atomic claims with provenance                          │
│   /api/page        - Source pages                                           │
│   /api/user        - Credits, transactions, profile                         │
│   /api/auth        - Google OAuth, sessions                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA MODELS                                       │
│                      models/domain/*.py                                      │
│                                                                              │
│   Story     - Unified API view of L3/L4 (scale=incident|case)               │
│   Incident  - L3 membrane cluster of surfaces (same happening)              │
│   Case      - L4 membrane cluster of incidents (same storyline)             │
│   Surface   - L2 scoped proposition ((scope_id, question_key))              │
│   Claim     - L0 atomic observation with provenance                         │
│   Entity    - Named thing (person, place, org) with Wikidata link           │
│   Inquiry   - Question with typed belief state and resolution status        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            REPOSITORIES                                      │
│                      repositories/*.py                                       │
│                                                                              │
│   StoryRepository    - Unified reads (incidents + cases)                    │
│   CaseRepository     - Neo4j cases (L4)                                     │
│   EventRepository    - Legacy recursive event worker (deprecated)           │
│   SurfaceRepository  - Neo4j for graph, PostgreSQL for centroids            │
│   ClaimRepository    - Neo4j nodes, PostgreSQL for embeddings               │
│   EntityRepository   - Neo4j nodes with Wikidata properties                 │
│   InquiryRepository  - PostgreSQL for inquiry state                         │
│   UserRepository     - PostgreSQL for users, credits                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪════════════════════════════════════════════
                     REMOTE DATABASE SERVER (../infra)
════════════════════════════════════╪════════════════════════════════════════════
                                    ▼
┌──────────────────────────────┬──────────────────────────────────────────────┐
│         Neo4j                │              PostgreSQL                       │
│    (Knowledge Graph)         │           (Relational + Vector)               │
│       Port 7687              │              Port 5432                        │
│                              │                                               │
│  (:Page)-[:EMITS]->(:Claim)  │  core.pages          - page metadata          │
│  (:Claim)-[:MENTIONS]->      │  core.claims         - claim text + embedding │
│           (:Entity)          │  core.entities       - entity metadata        │
│  (:Surface)-[:CONTAINS]->    │  content.surface_centroids                    │
│             (:Claim)         │  core.claim_embeddings                        │
│  (:Incident)-[:CONTAINS]->   │  public.users        - user accounts          │
│             (:Surface)       │  public.credit_transactions                   │
│  (:Case)-[:CONTAINS]->       │  inquiry.inquiries   - inquiry state          │
│            (:Incident)       │  inquiry.contributions                        │
└──────────────────────────────┴──────────────────────────────────────────────┘
                                    ▲
                                    │
                              Redis (6380)
                                    │
════════════════════════════════════╪════════════════════════════════════════════
                     WORKERS CONTAINER (herenews-workers)
════════════════════════════════════╪════════════════════════════════════════════
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORKERS                                         │
│                         (Background 24x7)                                    │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Extraction  │  │ Knowledge   │  │   Weaver    │  │ Canonical   │         │
│  │   Worker    │  │   Worker    │  │   Worker    │  │   Worker    │         │
│  │             │  │             │  │   (REEE)    │  │             │         │
│  │ URL→Page    │  │ Page→       │  │ Claims→     │  │ Incidents→  │         │
│  │ (content)   │  │ Claims/Ents │  │ Surfaces→   │  │ Cases (L4)  │         │
│  │             │  │ + Wikidata  │  │ Incidents   │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                                              │
│  Redis Queues: queue:extraction:high, queue:semantic:high, claims:pending   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Epistemic Layers (REEE)

The system implements a layered epistemic architecture:

```
L0  Claim      Atomic observation from a source (immutable)
    ↓
L2  Surface    Scoped proposition: (scope_id, question_key)
    ↓
L3  Incident   Membrane over surfaces: "same happening"
    ↓
L4  Case       Membrane over incidents: "same storyline" (plus entity lens)
```

### L0: Claim
- Extracted from source pages (LLM-assisted) by KnowledgeWorker
- Contains: text, provenance (page/publisher), reported_time, entities, optional typed values, optional embedding
- Immutable once created (append-only)

### L2: Surface
- **Scoped proposition container** keyed by `(scope_id, question_key)`, not `question_key` alone
- Purpose: aggregate competing reports about the same *typed proposition within a referent scope*
- Supports typed belief states (Jaynes) for variables when typed constraints exist
- Prevents global predicate buckets (e.g., `policy_announcement` across unrelated places)

### L3: Incident
- Groups L2 surfaces into a single happening via a **membrane** (anti-percolation)
- Inputs: anchor entities, companion context, reported_time windows
- Outputs: membership levels (CORE/PERIPHERY/QUARANTINE) + audit reasons/meta-claims
- Important: semantic similarity/LLM output can propose candidates, but **semantic-only evidence must not create core merges**

### Views (reee/views/)
- **IncidentEventView**: Tight temporal window (days), discriminative anchors
- **CaseView**: Loose temporal binding, storyline clustering (still bridge-resistant)
- **EntityCase**: Lens-like entity storyline (overlapping membership; handles star-shaped stories where k=2 recurrence fails)

## Frontend Architecture

```
/                  - Homepage (redirects to /inquiry)
/inquiry           - Active inquiries ranked by stake/entropy
/inquiry/{id}      - Inquiry detail: belief state, contributions, tasks
/event/{slug}      - Story detail (incident/case): surfaces, claims, tensions
/entity/{id}       - Entity profile: related stories, claims, Wikidata
/profile           - User profile: credits, stakes, contributions
/page/{id}         - Source page: extracted claims, entities
/archive           - Page archive/explorer
/graph             - Knowledge graph visualization
/map               - Geographic event map
```

### Key Components
- **layout/**: Layout, Header, SearchBar, UserProfile
- **inquiry/**: InquiryCard, InquiryCarousel
- **event/**: TimelineView, TimelineCard, TopologyView, EpicenterMapCard, EventSidebar, EventNarrativeContent, DebateThread
- **epistemic/**: EpistemicStateCard, QuestList, ContributionModal, DivergentValuesCard, AccountabilityChainCard

## API Endpoints

### Stories (preferred)
```
GET  /api/stories?scale=incident    List incidents (L3)
GET  /api/stories?scale=case        List cases (L4)
GET  /api/stories/{id}              Story detail (incident/case)
GET  /api/stories/{id}/surfaces     Surfaces for a story
GET  /api/stories/{id}/incidents    Incidents inside a case
```

### Cases (legacy endpoint name)
```
GET  /api/events                    List cases (L4 only)
GET  /api/events/{id}               Case detail
```

### Surfaces
```
GET  /api/surfaces                  List all surfaces
GET  /api/surfaces/{id}             Surface with claims
GET  /api/surfaces/by-event/{id}    Surfaces for event
GET  /api/surfaces/stats            Aggregate statistics
```

### Inquiry
```
GET  /api/inquiry                   List inquiries
GET  /api/inquiry/{id}              Inquiry detail
GET  /api/inquiry/{id}/trace        Full epistemic trace
POST /api/inquiry/{id}/contribute   Add contribution
POST /api/inquiry/{id}/stake        Add stake (bounty)
```

### Entities & Claims
```
GET  /api/entity/{id}               Entity with stories, claims
GET  /api/claim/{id}                Claim with source, entities
GET  /api/page/{id}                 Page with claims, entities
```

## Workers

### Boundary Rules (No Crosslines)

This repo intentionally separates **product state** from **epistemic state**. The fastest way to lose coherence is letting multiple layers write the same concept in different places.

| Component | Writes | Must NOT Write |
|----------|--------|----------------|
| APP-API (`backend/main.py`, `backend/api/*.py`) | PostgreSQL `public.*` (users/inquiries/stakes/tasks), Redis enqueue | Neo4j core topology (`:Surface/:Incident/:Case`), claim extraction outputs |
| ExtractionWorker (`backend/workers/extraction_worker.py`) | PostgreSQL `core.pages` (content + metadata), Redis enqueue `queue:semantic:high` | Neo4j claims/entities/topology |
| KnowledgeWorker (`backend/workers/knowledge_worker.py`) | Neo4j `:Claim/:Entity` + mention links, Wikidata resolution; enqueue `claims:pending` | L2/L3/L4 topology nodes |
| Weaver (`backend/workers/principled_weaver.py`) | Neo4j `:Surface/:Incident/:MetaClaim` + membership edges; PostgreSQL indices/centroids | User/community tables (`public.*`) |
| Canonical (`backend/workers/canonical_worker.py`) | Neo4j `:Case` (and story labels/fields), titles/descriptions | L0 deletion; competing L3 formation |
| Inquiry resolver (`backend/run_inquiry_resolver.py`) | PostgreSQL `public.*` posterior/resolution state | Rewriting core topology directly |

**Healing mode**: some workers provide a controlled “rebuild derived layers” operation (delete/recompute L2+). This must never delete L0 claims/pages.

### Extraction Worker
**Queue**: `queue:extraction:high`
**Input**: URL
**Output**: Page content in PostgreSQL + enqueue semantic job

1. Fetch URL content (Iframely/direct)
2. Store content + metadata in PostgreSQL (`core.pages`)
3. Enqueue `queue:semantic:high` for KnowledgeWorker

### Knowledge Worker
**Queue**: `queue:semantic:high`
**Input**: Page ID
**Output**: Claims + entities in Neo4j (with Wikidata resolution), then queue claims for weaving

1. LLM extracts claim candidates, mentions, and (optional) typed slots from page content
2. Resolve mentions to entity IDs (local + Wikidata), dedupe by QID
3. Persist `:Claim` and `:Entity` nodes + links in Neo4j
4. Enqueue claims to `claims:pending` for the Weaver

### Weaver Worker (REEE Core)
**Queue**: `claims:pending` (or continuous loop)
**Input**: New claims
**Output**: Surfaces (L2), Incidents (L3), Meta-claims

1. Load recent claims from Neo4j
2. Compute `(scope_id, question_key)` and route claim to an L2 surface
3. Update surface-level typed belief states (when typed constraints exist)
4. Form L3 incidents via membrane rules (bridge immunity, time windows, hub suppression)
5. Persist derived topology to Neo4j + PostgreSQL indices/centroids

### Canonical Worker
**Input**: L3 incidents
**Output**: L4 cases (CaseCore + EntityCase), plus titles/descriptions (optionally LLM-assisted)

- Case formation should be constraint-ledger native (motifs/constraints), not raw anchor intersection
- Canonical worker should be presentation/persistence oriented, not a competing structure pipeline

### Inquiry Resolver
**Input**: Inquiries + linked surfaces/incidents
**Output**: Posterior updates, resolution checks, bounties/tasks (PostgreSQL `public` schema)

## Data Flow Example

```
User submits URL via Browser
       │
═══════╪═══════════════════════════════════════════
       │  APP-API CONTAINER
═══════╪═══════════════════════════════════════════
       ▼
┌──────────────────┐
│   API (8000)     │  → Validates URL, queues to Redis
└──────────────────┘
       │
       ▼  Redis: queue:extraction:high
═══════╪═══════════════════════════════════════════
       │  WORKERS CONTAINER
═══════╪═══════════════════════════════════════════
       ▼
┌──────────────────┐
│ Extraction Worker│  → Page content in PostgreSQL
└──────────────────┘
       │  Redis: queue:semantic:high
       ▼
┌──────────────────┐
│ Knowledge Worker │  → Claims + Entities in Neo4j
└──────────────────┘
       │  Redis: claims:pending
       ▼
┌──────────────────┐
│  Weaver Worker   │  → Claims grouped into scoped Surfaces (L2)
│     (REEE)       │  → Surfaces clustered into Incidents (L3)
└──────────────────┘
       │
       ▼
       ▼  Neo4j/PostgreSQL updated
═══════╪═══════════════════════════════════════════
       │  APP-API CONTAINER
═══════╪═══════════════════════════════════════════
       ▼
┌──────────────────┐
│   API (8000)     │  → Frontend fetches event, surfaces, tensions
└──────────────────┘
       │
       ▼
┌──────────────────┐
│    Frontend      │  → User sees inquiry, can contribute
└──────────────────┘
```

## Community Layer (Public Schema)

The community loop uses PostgreSQL `public` schema for user-facing data:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PUBLIC SCHEMA                                      │
│                    (User/Community/Finance Layer)                            │
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │    users    │────▶│   stakes    │────▶│  inquiries  │                    │
│  │             │     │             │     │             │                    │
│  │ user_id     │     │ inquiry_id  │     │ id          │                    │
│  │ email       │     │ user_id     │     │ title       │                    │
│  │ google_id   │     │ amount      │     │ schema_type │                    │
│  │ credits     │     └─────────────┘     │ posterior   │                    │
│  │ reputation  │                         │ entropy     │                    │
│  └──────┬──────┘     ┌─────────────┐     │ total_stake │                    │
│         │            │contributions│     └──────┬──────┘                    │
│         │            │             │            │                           │
│         └───────────▶│ inquiry_id  │◀───────────┘                           │
│                      │ user_id     │                                        │
│                      │ type        │     ┌─────────────┐                    │
│                      │ text        │     │   tasks     │                    │
│                      │ source_url  │     │             │                    │
│                      │ impact      │     │ inquiry_id  │                    │
│                      │ reward      │     │ type        │                    │
│                      └─────────────┘     │ bounty      │                    │
│                                          │ claimed_by  │                    │
│  ┌─────────────────────────────────┐     │ completed   │                    │
│  │     credit_transactions         │     │ meta_claim  │                    │
│  │                                 │     └─────────────┘                    │
│  │ user_id, amount, balance_after  │                                        │
│  │ type: stake|reward|purchase     │                                        │
│  │ reference_type, reference_id    │                                        │
│  └─────────────────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### User Journey

```
1. User signs in (Google OAuth)
   └─▶ users.credits = 1000 (new user bonus)

2. User views inquiry with high bounty
   └─▶ inquiries.total_stake shown

3. User contributes evidence
   └─▶ contributions created
   └─▶ Contribution processed → becomes Claim in core schema
   └─▶ Posterior updated → contributions.impact computed
   └─▶ credits earned proportional to impact

4. User stakes on inquiry
   └─▶ stakes created
   └─▶ users.credits decreased
   └─▶ inquiries.total_stake increased
   └─▶ credit_transactions logged

5. User claims task (from meta-claim)
   └─▶ tasks.claimed_by = user_id
   └─▶ User submits contribution
   └─▶ tasks.completed = true
   └─▶ User receives task bounty
```

### Reward Economics

| Action | Credit Flow | Calculation |
|--------|-------------|-------------|
| New user | +1000 | Welcome bonus |
| Add stake | -N | Deducted from balance |
| Contribution | +reward | `stake_pool * impact * 0.7` |
| Complete task | +bounty | Task-specific bounty |
| Inquiry resolved | refund? | TBD: stake return policy |

### Inquiry Schema Types

Inquiries have typed schemas that determine how contributions update the posterior:

| Type | Example | Posterior Shape |
|------|---------|-----------------|
| `boolean` | "Did X happen?" | P(true), P(false) |
| `monotone_count` | "How many dead?" | Lower/upper bounds |
| `categorical` | "Who is responsible?" | P(A), P(B), P(C)... |
| `report_truth` | "Is this report accurate?" | P(true), P(false) |
| `forecast` | "Will X happen by date?" | Time-decaying probability |

### Task Generation

Tasks are auto-generated from meta-claims (epistemic gaps):

| Meta-Claim Type | Task Generated | Default Bounty |
|-----------------|----------------|----------------|
| `single_source_only` | "Find corroborating source" | 10 |
| `unresolved_conflict` | "Resolve contradiction" | 20 |
| `need_primary_source` | "Find official source" | 15 |
| `coverage_gap` | "Fill information gap" | 10 |
| `high_dispersion` | "Clarify scope" | 10 |

## Key Files

### APP-API Container

```
backend/
├── main.py                    # FastAPI app, router registration
├── Dockerfile.api             # API-only container
├── endpoints_events.py        # Event, entity, claim, page endpoints
├── api/
│   ├── auth.py               # Google OAuth, sessions
│   ├── inquiry.py            # Inquiry CRUD, contribute, stake
│   ├── user.py               # Credits, transactions, profile
│   ├── contributions.py      # User contributions
│   ├── surfaces.py           # Surface endpoints
│   ├── coherence.py          # Coherence feed
│   ├── map.py                # Geographic endpoints
│   └── preview.py            # URL preview
├── models/domain/             # Core domain models
│   ├── event.py              # Event (L3)
│   ├── surface.py            # Surface (L2)
│   ├── claim.py              # Claim (L0)
│   ├── entity.py             # Entity
│   └── inquiry.py            # Inquiry, Contribution, Task
├── repositories/              # Data access (Neo4j + PostgreSQL)
│   ├── event_repository.py
│   ├── surface_repository.py
│   ├── claim_repository.py
│   ├── entity_repository.py
│   ├── inquiry_repository.py
│   └── user_repository.py
└── config.py                  # Database connection settings

frontend/
├── app/
│   ├── App.tsx               # Router: routes → pages
│   ├── main.tsx              # Entry point
│   ├── InquiryPage.tsx       # /inquiry - inquiry list
│   ├── InquiryDetailPage.tsx # /inquiry/:id - inquiry detail
│   ├── EventPage.tsx         # /event/:slug - event detail
│   ├── EntityPage.tsx        # /entity/:id - entity profile
│   ├── ProfilePage.tsx       # /profile - user profile
│   ├── PagePage.tsx          # /page/:id - source page
│   ├── ArchivePage.tsx       # /archive - page explorer
│   ├── GraphPage.tsx         # /graph - knowledge graph
│   └── MapPage.tsx           # /map - geographic map
├── components/
│   ├── layout/               # Layout, Header, SearchBar, UserProfile
│   ├── inquiry/              # InquiryCard, InquiryCarousel
│   ├── event/                # TimelineView, TopologyView, etc.
│   └── epistemic/            # EpistemicStateCard, QuestList, etc.
├── types/
│   ├── inquiry.ts            # Inquiry, Contribution, Task types
│   └── user.ts               # User type
├── hooks/                    # useEpistemicState, useInView
├── data/                     # simulatedInquiries (demo data)
└── utils/                    # timeFormat helpers
```

### WORKERS Container

```
backend/
├── Dockerfile.workers         # Workers-only container
├── run_workers.py             # Unified worker manager (API + workers in dev)
├── run_extraction_worker.py
├── run_knowledge_worker.py
├── run_inquiry_resolver.py
├── workers/
│   ├── extraction_worker.py
│   ├── knowledge_worker.py
│   ├── principled_weaver.py
│   └── canonical_worker.py
└── reee/                      # REEE epistemic core (used by workers)
    ├── types.py              # Core types (Claim, Surface, Event, MetaClaim)
    ├── views/
    │   ├── incident.py       # IncidentEventView (L3)
    │   └── case.py           # CaseView (L4)
    ├── meta/
    │   └── detectors.py      # TensionDetector (meta-claims)
    └── inquiry/
        ├── seeder.py         # ProtoInquiry emergence
        └── webapp_seeder.py  # Bridge to webapp
```

### Infrastructure (../infra)

```
infra/
├── docker-compose.yml        # Database services only
├── .env                      # Credentials (not in git)
├── postgres/
│   └── init/                 # Init scripts for PostgreSQL
├── start.sh                  # Start databases
├── stop.sh                   # Stop databases
├── status.sh                 # Check database health
└── backup.sh                 # Backup databases
```

### Database Migrations

```
product/migrations/
├── 001_create_inquiry_tables.sql  # Inquiry schema
└── 002_create_user_tables.sql     # Public schema (users, credits)
```

## Configuration

### Environment Variables (.env)

```bash
# ═══════════════════════════════════════════════════════════════
# REMOTE DATABASE CONNECTIONS (../infra server)
# ═══════════════════════════════════════════════════════════════

# PostgreSQL (phi_here_db)
POSTGRES_HOST=200:xxxx:xxxx:xxxx:xxxx  # Yggdrasil IPv6 or localhost
POSTGRES_PORT=5432
POSTGRES_DB=phi_here
POSTGRES_USER=phi_user
POSTGRES_PASSWORD=phi_password_dev

# Neo4j (uwps-neo4j)
NEO4J_URI=bolt://200:xxxx:xxxx:xxxx:xxxx:7687  # Yggdrasil IPv6 or localhost
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# Redis (uwps-redis)
REDIS_URL=redis://200:xxxx:xxxx:xxxx:xxxx:6380  # Note: port 6380, not 6379

# ═══════════════════════════════════════════════════════════════
# EXTERNAL APIs
# ═══════════════════════════════════════════════════════════════

OPENAI_API_KEY=sk-...
IFRAMELY_API_KEY=...

# ═══════════════════════════════════════════════════════════════
# AUTH (Optional - for Google OAuth)
# ═══════════════════════════════════════════════════════════════

GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
JWT_SECRET_KEY=...
```

### Docker Compose Services

**Remote (../infra/docker-compose.yml)** - Database infrastructure:
```yaml
services:
  postgres:   # PostgreSQL + pgvector (phi_here_db, port 5432)
  neo4j:      # Neo4j graph database (uwps-neo4j, port 7687)
  redis:      # Job queues (uwps-redis, port 6380)
```

**Local (./docker-compose.yml)** - Application:
```yaml
services:
  api:        # FastAPI + static frontend (herenews-api, port 8000)
  workers:    # Background workers (herenews-workers, no port)
```

### Remote Database Connection

From app/workers to remote databases:
```bash
# PostgreSQL
POSTGRES_HOST=<yggdrasil-ipv6-or-localhost>
POSTGRES_PORT=5432
POSTGRES_DB=phi_here
POSTGRES_USER=phi_user

# Neo4j
NEO4J_URI=bolt://<yggdrasil-ipv6-or-localhost>:7687
NEO4J_USER=neo4j

# Redis
REDIS_URL=redis://<yggdrasil-ipv6-or-localhost>:6380
```

## Invariants

### Epistemic Loop
1. **Claims are immutable** - Once created, never modified
2. **Surfaces are emergent** - Computed from claim relations, not manually created
3. **Incidents/Cases are emergent** - Derived from surfaces/incidents via membrane rules, not manually created
4. **Meta-claims are diagnostic** - Observations about topology, not truth claims
5. **All computation is reproducible** - Given (L0, params), derive same L2-L4

### Community Loop
6. **Inquiries have typed schemas** - Enable proper Bayesian updates
7. **Credits are audited** - Every transaction logged in credit_transactions
8. **Contributions become claims** - User evidence enters epistemic loop
9. **Tasks emerge from gaps** - Meta-claims auto-generate bounty tasks
10. **Rewards proportional to impact** - Information gain (entropy reduction) = credit reward

### Loop Coupling
11. **Contributions → Claims** - User contributions processed into L0 claims
12. **Meta-claims → Tasks** - Epistemic gaps become claimable tasks
13. **Stakes → Bounties** - User stakes fund task rewards
14. **Resolution → Distribution** - Inquiry resolution triggers reward payout

## See Also

### Epistemic Theory
- `backend/reee/REEE1.md` - Epistemic theory and invariants
- `backend/reee/REEE2.md` - Implementation details

### Product & Community
- `product/ECOSYSTEM-NARRATIVE.md` - **How the two loops work together** (scenarios with Alice, Bob, etc.)
- `product/ECOSYSTEM-NARRATIVE-2.md` - **Why this system beats journalism/social media** (Gaza, Lab Leak, MH17, Biden laptop)
- `product/PRODUCT_VISION.md` - Core concept and page alignment
- `product/PRODUCT-DEFINITION.md` - MVP1 product definition
- `product/MVP1-SPECIFICATION.md` - Feature specifications
- `product/migrations/` - Database schema for community layer

### API Reference
- `API.md` - Quick reference for frontend developers
