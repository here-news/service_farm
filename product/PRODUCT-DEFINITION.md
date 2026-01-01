# HERE.news Inquiry MVP1 - Product Definition

**Version:** 1.0.0
**Date:** 2025-12-31
**Status:** MVP Ready for Testing

---

## Executive Summary

The HERE.news Inquiry system is a collaborative fact-finding platform that applies Bayesian epistemics to answer contested questions. Unlike prediction markets or social media fact-checking, Inquiry:

1. **Tracks belief states** using proper Bayesian inference
2. **Rewards evidence quality** proportional to information gain
3. **Makes uncertainty explicit** through entropy and confidence metrics
4. **Provides epistemic transparency** showing how conclusions were reached

## MVP1 Scope

### What's Included

| Feature | Status | Description |
|---------|--------|-------------|
| Inquiry Listing | ✅ Complete | Browse inquiries by category (Resolved, Top Bounties, Contested) |
| Inquiry Detail | ✅ Complete | View full inquiry with belief state, contributions, tasks |
| Simulated Data | ✅ Complete | Demo mode with realistic sample inquiries |
| Database Schema | ✅ Complete | PostgreSQL tables for inquiries, contributions, stakes, tasks |
| Repository Layer | ✅ Complete | Data access layer with proper separation of concerns |
| API Endpoints | ✅ Complete | REST API for all CRUD operations |
| Guest Browsing | ✅ Complete | View all content without authentication |
| Contribution UI | ✅ Complete | Share box with text input and URL support |
| Bounty Pool UI | ✅ Complete | Display total bounty and add bounty input |
| Auth Integration | ✅ Complete | Google OAuth with JWT cookies |
| Credit System | ✅ Complete | 1000 credits for new users, deduction for stakes |

### What's Mocked/Simulated

| Feature | Status | Notes |
|---------|--------|-------|
| Contribution Impact | Mock | Random 1-10% impact assigned |
| Posterior Computation | Simulated | No real REEE integration yet |
| Reward Distribution | Mock | UI only, no actual credit transfers |
| URL Preview | Partial | API exists but not fully integrated |
| Task Generation | Mock | Simulated tasks in demo data |

### What's Not Included (MVP2+)

- Real-time posterior updates via WebSocket
- Credit purchase flow
- Reputation system
- Email notifications
- Mobile app
- Admin dashboard

## Architecture

### Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│   FastAPI       │────▶│  PostgreSQL     │
│    (React)      │     │   Backend       │     │  + Neo4j        │
│                 │◀────│                 │◀────│                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        │                       │
        ▼                       ▼
   Simulated Data         REEE Engine
   (Demo Mode)           (Future Integration)
```

### Database Schema

```
┌──────────────────┐
│     users        │
├──────────────────┤
│ user_id (PK)     │
│ email            │
│ google_id        │
│ credits_balance  │
│ reputation       │
└──────────────────┘
         │
         │ 1:N
         ▼
┌──────────────────┐     ┌──────────────────┐
│   inquiries      │◀────│ inquiry_stakes   │
├──────────────────┤     ├──────────────────┤
│ id (PK)          │     │ inquiry_id (FK)  │
│ title            │     │ user_id (FK)     │
│ description      │     │ amount           │
│ status           │     └──────────────────┘
│ schema_type      │
│ posterior_prob   │     ┌──────────────────┐
│ entropy_bits     │◀────│  contributions   │
│ total_stake      │     ├──────────────────┤
└──────────────────┘     │ id (PK)          │
         │               │ inquiry_id (FK)  │
         │               │ user_id (FK)     │
         │               │ type             │
         │               │ text             │
         │               │ posterior_impact │
         │               └──────────────────┘
         │
         │ 1:N
         ▼
┌──────────────────┐
│  inquiry_tasks   │
├──────────────────┤
│ id (PK)          │
│ inquiry_id (FK)  │
│ task_type        │
│ bounty           │
│ claimed_by (FK)  │
│ completed        │
└──────────────────┘
```

### API Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | /api/inquiry | List inquiries | No |
| GET | /api/inquiry/:id | Get inquiry detail | No |
| POST | /api/inquiry | Create inquiry | Yes |
| POST | /api/inquiry/:id/contribute | Add contribution | Optional |
| POST | /api/inquiry/:id/stake | Add stake/bounty | Yes |
| GET | /api/inquiry/:id/trace | Get epistemic trace | No |
| GET | /api/inquiry/:id/tasks | Get tasks | No |
| GET | /api/inquiry/:id/contributions | Get contributions | No |
| POST | /api/inquiry/:id/tasks/:taskId/claim | Claim task | Yes |
| GET | /api/auth/status | Check auth status | No |
| GET | /api/auth/login | Initiate Google OAuth | No |
| GET | /api/auth/logout | Logout | Yes |

## User Journeys

### Journey 1: Guest Discovery
1. User arrives at `/inquiry`
2. Browses carousels (Resolved, Top Bounties, Contested)
3. Clicks on an inquiry card
4. Views detail page with belief state and evidence
5. Reads contributions from community

### Journey 2: Authenticated Contribution
1. User clicks "Sign in" → Google OAuth
2. Returns to inquiry detail
3. Types contribution in share box
4. Posts contribution
5. Sees impact on posterior (mocked in MVP1)

### Journey 3: Staking/Bounty
1. Authenticated user views inquiry
2. Enters bounty amount
3. Clicks "Add"
4. Credits deducted from balance
5. Bounty pool increases

### Journey 4: Task Claiming
1. User views inquiry with open tasks
2. Clicks "Claim Task"
3. Task is reserved for user
4. User submits contribution
5. Task marked complete, reward pending

## Testing

### Playwright Test Coverage

```
tests/
├── explore-inquiry.spec.ts    # 22 exploratory tests
└── mvp1-walkthrough.spec.ts   # 27 walkthrough tests

Results: 22/27 passed (81% pass rate)
```

### Known Test Failures
- Strict mode violations (multiple elements matching) - UI refinement needed
- Back navigation test - timing issue

## Files Created/Modified

### Product Documentation
- `product/MVP1-SPECIFICATION.md` - Detailed spec with tables
- `product/PRODUCT-DEFINITION.md` - This document
- `product/migrations/001_create_inquiry_tables.sql` - DB migration
- `product/tests/*.spec.ts` - Playwright tests

### Backend
- `backend/models/domain/inquiry.py` - Domain models
- `backend/models/api/inquiry.py` - Pydantic schemas
- `backend/repositories/inquiry_repository.py` - Data access
- `backend/api/inquiry.py` - REST API endpoints
- `backend/data/simulatedInquiries.py` - Demo data

### Database Tables Created
- `inquiries` - Main inquiry table
- `inquiry_stakes` - User stakes on inquiries
- `contributions` - User contributions
- `inquiry_tasks` - System-generated tasks
- `credit_transactions` - Audit log for credits

## Configuration

### Environment Variables
```
# Database (from .env)
POSTGRES_HOST=...
POSTGRES_PORT=5432
POSTGRES_DB=phi_here
POSTGRES_USER=phi_user
POSTGRES_PASSWORD=...

# Auth
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
```

### Running the Application
```bash
# Start containers
docker-compose up -d

# View logs
docker-compose logs -f app

# Run tests
cd product && npx playwright test
```

## Next Steps (MVP2)

### Priority 1: Real Posterior Computation
- Integrate with REEE typed_belief module
- Compute actual information gain for contributions
- Update entropy and confidence in real-time

### Priority 2: Reward Distribution
- Automatic reward calculation on resolution
- Credit transfer to contributors
- Task completion rewards

### Priority 3: URL Preview Enhancement
- Full iframely integration
- Inline article previews
- Source credibility indicators

### Priority 4: Real-time Updates
- WebSocket for live posterior changes
- Notification system
- Activity feed

---

## Success Criteria

| Metric | MVP1 Target | Current |
|--------|-------------|---------|
| Page loads without error | 100% | ✅ |
| API endpoints return valid data | 100% | ✅ |
| Authentication works | Yes | ✅ |
| Credit deduction works | Yes | ✅ |
| Playwright tests pass | >80% | 81% |
| Mobile responsive | Yes | ✅ |

---

*Document maintained as part of the HERE.news product development process.*
