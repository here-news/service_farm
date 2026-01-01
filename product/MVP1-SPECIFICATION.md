# HERE.news Inquiry MVP1 - Product Specification

**Version:** 1.0
**Last Updated:** 2025-12-31
**Status:** Active Development

## 1. Overview

### 1.1 Product Vision
HERE.news Inquiry is a collaborative fact-finding platform that applies Bayesian epistemics to answer contested questions. Users contribute evidence, stake credits on outcomes, and earn rewards for valuable contributions that reduce uncertainty.

### 1.2 Core Concept
Unlike prediction markets (which bet on outcomes) or social media fact-checking (which relies on authority), Inquiry:
- **Tracks belief states** using proper Bayesian inference
- **Rewards evidence quality** not just popularity
- **Makes uncertainty explicit** through entropy and confidence metrics
- **Provides epistemic transparency** showing how conclusions were reached

### 1.3 Target Users (MVP1)
1. **Evidence Hunters** - News junkies who want to earn rewards for research
2. **Truth Seekers** - Users frustrated by conflicting information
3. **Domain Experts** - People with specialized knowledge to contribute
4. **Stakers** - Users who want to back important questions with bounties

## 2. Feature Specification

### 2.1 Inquiry Listing (`/inquiry`)

#### Functional Requirements
| ID | Feature | Priority | Status |
|----|---------|----------|--------|
| F1.1 | Display carousels by category (Resolved, Top Bounties, Contested) | P0 | Done |
| F1.2 | Sort all open questions by bounty/entropy/activity | P0 | Done |
| F1.3 | Search by title and entity keywords | P1 | Done |
| F1.4 | FAB button to create new inquiry | P0 | Done |
| F1.5 | Show key metrics per card (bounty, entropy, contributions) | P0 | Done |
| F1.6 | Pagination/infinite scroll for large lists | P2 | Pending |

#### UI Components
- **Carousel**: Horizontal scrolling cards with peek arrows
- **InquiryCard**: Shows title, status badge, metrics, entity tags
- **Sorting dropdown**: By Bounty / By Uncertainty / By Activity
- **Search bar**: Filters current view or navigates with `?q=`

### 2.2 Inquiry Detail (`/inquiry/:id`)

#### Functional Requirements
| ID | Feature | Priority | Status |
|----|---------|----------|--------|
| F2.1 | Display current best answer with confidence | P0 | Done |
| F2.2 | Show probability distribution chart | P0 | Done |
| F2.3 | Display bounty pool with add-bounty input | P0 | Done |
| F2.4 | Community contributions feed (social style) | P0 | Done |
| F2.5 | Simple share box with text + file attachment | P0 | Done |
| F2.6 | Evidence gaps panel (system-generated tasks) | P0 | Done |
| F2.7 | Claim clusters (surfaces) visualization | P1 | Done |
| F2.8 | Resolution status indicator | P0 | Done |
| F2.9 | Recent rewards list (who earned what) | P1 | Done |
| F2.10 | URL preview inline for source links | P1 | Partial |

#### Layout
```
+------------------------------------------+
| < Back to questions                       |
| [OPEN] [Rigor B]     How many people...  |
| [Entity Tags]                             |
+------------------------------------------+
|                        |                 |
| Context/Description    | Bounty Pool     |
|                        | $5000.00        |
| Current Best: 315,000  | [Add Bounty]    |
| Confidence: 32%        |                 |
| [=====----] bar        | [Recent Rewards]|
|                        |                 |
| Stats: Sources|Entropy | Community (3)   |
|                        | [Share box]     |
| Resolution Status      |                 |
| [=] Gathering Evidence | [Contributions] |
|                        | - Sarah: "..."  |
| Distribution Chart     | - Mike: "..."   |
| [Bars for top values]  |                 |
|                        |                 |
| Evidence Gaps          |                 |
| - Need primary source  |                 |
|                        |                 |
| Claim Clusters         |                 |
| - Primary Sources (4)  |                 |
| - Wire Services (6)    |                 |
+------------------------------------------+
```

### 2.3 Bounty System

#### Functional Requirements
| ID | Feature | Priority | Status |
|----|---------|----------|--------|
| F3.1 | Display total bounty pool | P0 | Done (UI) |
| F3.2 | Add bounty from user credits | P0 | Mock only |
| F3.3 | Show distributed bounty amount | P1 | Mock only |
| F3.4 | Recent rewards list | P1 | Mock data |
| F3.5 | Task claiming with bounty reward | P1 | UI only |
| F3.6 | Automatic reward distribution on contribution | P2 | Not started |

#### Business Rules
1. **Minimum stake**: $0.01
2. **Maximum single stake**: No limit (MVP1)
3. **Bounty distribution**:
   - 70% to contributors proportional to impact
   - 20% to task completers
   - 10% platform fee

### 2.4 Contribution System

#### Functional Requirements
| ID | Feature | Priority | Status |
|----|---------|----------|--------|
| F4.1 | Natural text input (AI classifies type) | P0 | Done (UI) |
| F4.2 | Character limit with counter (500) | P0 | Done |
| F4.3 | File attachment (images, PDFs) | P2 | UI only |
| F4.4 | URL detection and preview | P1 | Partial |
| F4.5 | Show contribution type badge | P0 | Done |
| F4.6 | Show impact percentage | P0 | Done (mock) |
| F4.7 | Sort by recent/impact | P0 | Done |
| F4.8 | Verify/Flag actions | P2 | UI only |

#### Contribution Types
1. **Evidence**: Supporting claim with source
2. **Refutation**: Counter-evidence
3. **Attribution**: Source chain ("A said B reported...")
4. **Scope Correction**: "This is a different incident"
5. **Disambiguation**: Entity clarification

### 2.5 Credit System

#### Functional Requirements
| ID | Feature | Priority | Status |
|----|---------|----------|--------|
| F5.1 | Display user credit balance | P0 | DB ready, UI partial |
| F5.2 | New users get 1000 credits | P0 | Done |
| F5.3 | Deduct credits for bounty stakes | P1 | API ready |
| F5.4 | Add credits for contributions | P1 | Not started |
| F5.5 | Credit purchase flow | P3 | Not in MVP1 |
| F5.6 | Transaction history | P3 | Not in MVP1 |

### 2.6 User Authentication

#### Functional Requirements
| ID | Feature | Priority | Status |
|----|---------|----------|--------|
| F6.1 | Google OAuth login | P0 | Done |
| F6.2 | Session persistence (JWT cookie) | P0 | Done |
| F6.3 | Display user info in header | P1 | Partial |
| F6.4 | Logout | P0 | Done |
| F6.5 | Guest mode (view only) | P0 | Done |

## 3. Data Models

### 3.1 Existing Tables (PostgreSQL)

#### users
```sql
user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid()
email           VARCHAR NOT NULL UNIQUE
google_id       VARCHAR NOT NULL UNIQUE
name            VARCHAR
picture_url     TEXT
credits_balance INTEGER NOT NULL DEFAULT 1000
reputation      INTEGER NOT NULL DEFAULT 0
subscription_tier VARCHAR
is_active       BOOLEAN NOT NULL DEFAULT true
created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
last_login      TIMESTAMPTZ
```

### 3.2 New Tables Required (MVP1)

#### inquiries
```sql
CREATE TABLE inquiries (
  id              VARCHAR(20) PRIMARY KEY,  -- inq_abc12345
  title           TEXT NOT NULL,
  description     TEXT DEFAULT '',
  status          VARCHAR(20) NOT NULL DEFAULT 'open',
  rigor_level     VARCHAR(1) NOT NULL DEFAULT 'B',
  schema_type     VARCHAR(30) NOT NULL DEFAULT 'boolean',
  schema_config   JSONB DEFAULT '{}',

  -- Scope
  scope_entities  TEXT[] DEFAULT '{}',
  scope_keywords  TEXT[] DEFAULT '{}',
  scope_time_start TIMESTAMPTZ,
  scope_time_end  TIMESTAMPTZ,

  -- Current state
  posterior_map   JSONB,
  posterior_prob  DECIMAL(5,4) DEFAULT 0.0,
  entropy_bits    DECIMAL(6,3) DEFAULT 0.0,
  credible_interval JSONB,

  -- Stakes
  total_stake     DECIMAL(12,2) DEFAULT 0.0,
  distributed     DECIMAL(12,2) DEFAULT 0.0,

  -- Metadata
  created_by      UUID REFERENCES users(user_id),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at     TIMESTAMPTZ,

  -- Resolution tracking
  stable_since    TIMESTAMPTZ,

  CONSTRAINT valid_status CHECK (status IN ('open', 'resolved', 'stale', 'closed'))
);
```

#### inquiry_stakes
```sql
CREATE TABLE inquiry_stakes (
  id              SERIAL PRIMARY KEY,
  inquiry_id      VARCHAR(20) REFERENCES inquiries(id),
  user_id         UUID REFERENCES users(user_id),
  amount          DECIMAL(12,2) NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(inquiry_id, user_id)
);
```

#### contributions
```sql
CREATE TABLE contributions (
  id              VARCHAR(20) PRIMARY KEY,  -- contrib_abc123
  inquiry_id      VARCHAR(20) REFERENCES inquiries(id),
  user_id         UUID REFERENCES users(user_id),

  -- Content
  type            VARCHAR(30) NOT NULL DEFAULT 'evidence',
  text            TEXT NOT NULL,
  source_url      TEXT,
  source_name     VARCHAR(255),

  -- Extracted value
  extracted_value JSONB,
  observation_kind VARCHAR(20),

  -- Processing
  processed       BOOLEAN DEFAULT false,
  claim_ids       TEXT[] DEFAULT '{}',

  -- Impact
  posterior_impact DECIMAL(5,4) DEFAULT 0.0,
  reward_earned   DECIMAL(12,2) DEFAULT 0.0,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT valid_type CHECK (type IN ('evidence', 'refutation', 'attribution', 'scope_correction', 'disambiguation'))
);
```

#### inquiry_tasks
```sql
CREATE TABLE inquiry_tasks (
  id              VARCHAR(20) PRIMARY KEY,
  inquiry_id      VARCHAR(20) REFERENCES inquiries(id),

  type            VARCHAR(30) NOT NULL,
  description     TEXT NOT NULL,
  bounty          DECIMAL(12,2) DEFAULT 0.0,

  claimed_by      UUID REFERENCES users(user_id),
  completed       BOOLEAN DEFAULT false,
  completed_at    TIMESTAMPTZ,

  meta_claim_id   VARCHAR(50),

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

#### credit_transactions
```sql
CREATE TABLE credit_transactions (
  id              SERIAL PRIMARY KEY,
  user_id         UUID REFERENCES users(user_id),
  amount          DECIMAL(12,2) NOT NULL,  -- positive=credit, negative=debit
  balance_after   DECIMAL(12,2) NOT NULL,

  transaction_type VARCHAR(30) NOT NULL,  -- stake, reward, purchase, refund
  reference_type  VARCHAR(30),            -- inquiry, contribution, task
  reference_id    VARCHAR(50),

  description     TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

## 4. API Endpoints

### 4.1 Existing (Backend Ready)
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | /api/auth/status | Check auth status | No |
| GET | /api/auth/login | Initiate Google OAuth | No |
| GET | /api/auth/callback | OAuth callback | No |
| GET | /api/auth/logout | Logout | Yes |
| GET | /api/auth/me | Get current user | Yes |

### 4.2 Inquiry API (Implemented but In-Memory)
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | /api/inquiry | Create inquiry | Yes |
| GET | /api/inquiry | List inquiries | No |
| GET | /api/inquiry/:id | Get inquiry detail | No |
| POST | /api/inquiry/:id/contribute | Add contribution | Yes |
| POST | /api/inquiry/:id/stake | Add stake | Yes |
| GET | /api/inquiry/:id/trace | Get epistemic trace | No |
| GET | /api/inquiry/:id/tasks | Get tasks | No |

### 4.3 Needed Endpoints
| Method | Endpoint | Description | Priority |
|--------|----------|-------------|----------|
| GET | /api/user/credits | Get credit balance | P0 |
| GET | /api/user/transactions | Credit history | P2 |
| POST | /api/inquiry/:id/tasks/:taskId/claim | Claim task | P1 |
| POST | /api/inquiry/:id/tasks/:taskId/complete | Submit task | P1 |
| GET | /api/artifacts?url=... | URL preview metadata | P1 |

## 5. Testing Plan

### 5.1 Playwright E2E Tests (MVP1)
```
tests/
  explore-inquiry.spec.ts     # Current exploration tests
  inquiry-flow.spec.ts        # Full user journey
  bounty-system.spec.ts       # Credit/bounty tests
  auth-flow.spec.ts           # Login/logout tests
  contribution-flow.spec.ts   # Add contribution tests
```

### 5.2 Key User Journeys
1. **Guest browsing**: View inquiries, read details, see distributions
2. **Login flow**: Google OAuth -> redirect to /inquiry
3. **Create inquiry**: FAB -> form -> submit -> view detail
4. **Add contribution**: Login -> detail page -> share box -> post
5. **Add bounty**: Login -> detail page -> enter amount -> add
6. **Claim task**: Login -> detail page -> click claim -> submit

## 6. Implementation Phases

### Phase 1: Foundation (Current)
- [x] Frontend prototype with simulated data
- [x] Auth system (Google OAuth, JWT)
- [x] User table with credits
- [x] In-memory inquiry engine

### Phase 2: Persistence (Next)
- [ ] Create PostgreSQL tables for inquiries
- [ ] Implement inquiry repository
- [ ] Connect API to real database
- [ ] Credit deduction for stakes

### Phase 3: Integration
- [ ] Connect to REEE typed_belief for real posteriors
- [ ] URL preview integration
- [ ] Real contribution processing
- [ ] Task generation from meta-claims

### Phase 4: Economics
- [ ] Automatic reward distribution
- [ ] Task completion rewards
- [ ] Credit purchase flow

## 7. Open Questions

1. **Task bounty source**: From inquiry stake pool or system-generated?
2. **Reputation calculation**: How does contribution impact affect reputation?
3. **Rigor level control**: Can users change rigor after creation?
4. **Stake refunds**: When/if stakes get returned on resolution?
5. **Multi-answer inquiries**: Support for categorical with >2 options?

## 8. Success Metrics (MVP1)

| Metric | Target |
|--------|--------|
| Inquiries created | 10+ per week |
| Contributions per inquiry | 5+ average |
| User retention (7-day) | 30% |
| Resolution rate | 20% within 30 days |
| Average stake per inquiry | $50+ |

---

*Document maintained by the product team. Update as features are implemented.*
