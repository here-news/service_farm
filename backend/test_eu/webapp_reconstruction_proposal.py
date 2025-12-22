"""
WEBAPP RECONSTRUCTION PROPOSAL
==============================

Building the Epistemic Narrative System on Existing Components

CURRENT WORKING COMPONENTS
--------------------------
✅ ExtractionWorker: URL → page metadata + raw claims
✅ KnowledgeWorker: claims → entities + dedup + Neo4j storage
✅ EventWorker: page-level event routing (to enhance, not replace yet)
✅ Frontend: React SPA with event/entity pages
✅ Infrastructure: PostgreSQL, Neo4j, Redis queues

THE VISION
----------
Transform from "news aggregator" to "epistemic workspace" where:
1. Users SEE the narrative evolving (not just final state)
2. Users SEE what's uncertain/missing (gaps become invitations)
3. Users CAN contribute (claims, sources, perspectives)
4. System REWARDS coherence and entropy reduction

RECONSTRUCTION STRATEGY
-----------------------
Start from the WEBAPP (user-facing) and work backwards:
- Design the ideal UX first
- Identify API requirements
- Enhance backend to serve new UX
- Keep existing pipeline running in parallel

This way: users see value immediately, backend evolves incrementally.
"""

# =============================================================================
# PHASE 1: WEBAPP FOUNDATION (Week 1-2)
# =============================================================================

PHASE_1 = """
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: EPISTEMIC EVENT PAGE                                              │
│  "Show what we know AND what we don't know"                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Current Event Page:
  - Title, summary, claims list, entity tags
  - Static, no temporal context

New Event Page Structure:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  EVENT: Jimmy Lai Trial                                                 │
  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
  │                                                                         │
  │  📊 EPISTEMIC STATE           │  🔥 HEAT: 78/100  │  📈 156 claims     │
  │  ┌─────────────────────────┐  │                   │                    │
  │  │ ████████░░ 80% covered  │  │  Sources: 26     │  12 entities       │
  │  │ 3 open questions        │  │  Last: 2hr ago   │  4 perspectives    │
  │  └─────────────────────────┘  │                   │                    │
  │                                                                         │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  📰 NARRATIVE (Current Understanding)                                   │
  │  ┌─────────────────────────────────────────────────────────────────────┐│
  │  │ Hong Kong media mogul Jimmy Lai's national security trial began... ││
  │  │ [BBC, Reuters, SCMP +23 sources]                                   ││
  │  │                                                                     ││
  │  │ Prosecutors allege Lai colluded with foreign forces...             ││
  │  │ [HK Gov, SCMP] ⚖️ DISPUTED by [Amnesty, RSF]                       ││
  │  │                                                                     ││
  │  │ Lai faces potential life imprisonment...                           ││
  │  │ [Confirmed by 18 sources]                                          ││
  │  └─────────────────────────────────────────────────────────────────────┘│
  │                                                                         │
  │  ❓ OPEN QUESTIONS (Contribute!)                                        │
  │  ┌─────────────────────────────────────────────────────────────────────┐│
  │  │ 🎯 "Direct HK Government official statement"    +15 pts  [Add]     ││
  │  │ 🎯 "UK Government formal position"              +12 pts  [Add]     ││
  │  │ 🎯 "Human rights org assessment"                +20 pts  [Add]     ││
  │  └─────────────────────────────────────────────────────────────────────┘│
  │                                                                         │
  │  ⚖️ PERSPECTIVES                                                        │
  │  ┌─────────────────────────────────────────────────────────────────────┐│
  │  │ Prosecution view    ████████████░░░░░░  65%  (9 claims)            ││
  │  │ Defense/Rights view ████░░░░░░░░░░░░░░  22%  (3 claims)            ││
  │  │ Neutral reporting   ██░░░░░░░░░░░░░░░░  13%  (2 claims)            ││
  │  │                                                                     ││
  │  │ ⚠️ Note: Narrative currently leans prosecution. More defense       ││
  │  │    perspective sources would improve balance.                       ││
  │  └─────────────────────────────────────────────────────────────────────┘│
  │                                                                         │
  │  📜 CLAIM STREAM (Chronological)                                        │
  │  ┌─────────────────────────────────────────────────────────────────────┐│
  │  │ ┌────┐                                                              ││
  │  │ │ ▼  │ [Dec 19, 14:32] BBC                                         ││
  │  │ └────┘ "Trial of Jimmy Lai begins in Hong Kong..."                 ││
  │  │        ✓ Corroborated by 5 sources                                 ││
  │  │                                                                     ││
  │  │ ┌────┐                                                              ││
  │  │ │ ▼  │ [Dec 19, 15:01] Reuters                                     ││
  │  │ └────┘ "Lai pleaded not guilty to sedition charges..."             ││
  │  │        ✓ Corroborated by 3 sources                                 ││
  │  │                                                                     ││
  │  │ ┌────┐                                                              ││
  │  │ │ ▼  │ [Dec 19, 16:45] Amnesty International                       ││
  │  │ └────┘ "Trial is politically motivated persecution..."             ││
  │  │        ⚖️ Contradicts prosecution narrative                        ││
  │  │        ? Seeking: Official rebuttal or supporting evidence         ││
  │  │                                                                     ││
  │  └─────────────────────────────────────────────────────────────────────┘│
  │                                                                         │
  │  [View Full Timeline] [View All Sources] [Contribute Evidence]         │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

Key UI Components to Build:
1. EpistemicStateCard - shows coverage %, open questions count
2. NarrativeSummary - generated summary with source attribution
3. QuestList - open questions with bounties
4. PerspectiveBalance - visualization of viewpoint distribution
5. ClaimStream - chronological claim timeline with corroboration status
"""

# =============================================================================
# PHASE 2: COMMUNITY CONTRIBUTION SYSTEM (Week 2-3)
# =============================================================================

PHASE_2 = """
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: CONTRIBUTION SYSTEM                                               │
│  "Turn gaps into invitations, contributions into recognition"               │
└─────────────────────────────────────────────────────────────────────────────┘

CONTRIBUTION TYPES:

1. SOURCE SUBMISSION
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  📎 Submit Source                                                       │
   │  ─────────────────────────────────────────────────────────────────────  │
   │  URL: [https://example.com/article                              ]       │
   │                                                                         │
   │  Relevant to Quest: [Dropdown: "HK Gov statement" +15pts]              │
   │                     [Dropdown: "Human rights assessment" +20pts]        │
   │                     [Dropdown: "General contribution" +5pts]            │
   │                                                                         │
   │  Your note (optional): [                                        ]       │
   │                                                                         │
   │  [ Submit for Review ]                                                  │
   └─────────────────────────────────────────────────────────────────────────┘

2. CLAIM ANNOTATION
   - User can highlight existing claim and add:
     - Supporting evidence (link to source)
     - Counter-evidence (contradicting source)
     - Context (additional background)
     - Question (request for clarification)

3. PERSPECTIVE TAGGING
   - User can tag claims with perspective labels
   - Helps system identify epistemic poles
   - Crowdsourced categorization

CONTRIBUTION FLOW:

  User sees Quest → Submits URL → ExtractionWorker processes →
  Claims linked to event → User gets credit if:
    - Source is new (not duplicate)
    - Claims match quest criteria
    - Community validates (optional)

RECOGNITION SYSTEM:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  👤 Your Profile                                                        │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  🏆 Contribution Score: 245 pts                                         │
  │                                                                         │
  │  Recent Contributions:                                                  │
  │  ✓ Added UK Gov source to Jimmy Lai case         +12 pts   2 days ago  │
  │  ✓ First to report HK Fire death toll update     +25 pts   5 days ago  │
  │  ✓ Corroborated claim in Taiwan election story   +8 pts    1 week ago  │
  │                                                                         │
  │  Badges:                                                                │
  │  🔍 First Source (5x)  |  ⚖️ Balance Seeker (3x)  |  📰 News Scout     │
  │                                                                         │
  │  [View All Contributions] [Leaderboard]                                 │
  └─────────────────────────────────────────────────────────────────────────┘

API ENDPOINTS NEEDED:

POST /api/contributions/submit
  - url: string
  - event_id: string (optional)
  - quest_id: string (optional)
  - note: string (optional)
  - Returns: contribution_id, status

GET /api/contributions/mine
  - Returns: list of user's contributions with status

GET /api/events/{id}/quests
  - Returns: open quests for this event

POST /api/claims/{id}/annotate
  - type: "support" | "counter" | "context" | "question"
  - content: string
  - source_url: string (optional)
"""

# =============================================================================
# PHASE 3: REAL-TIME NARRATIVE UPDATES (Week 3-4)
# =============================================================================

PHASE_3 = """
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: LIVE NARRATIVE EVOLUTION                                          │
│  "Watch the story unfold in real-time"                                      │
└─────────────────────────────────────────────────────────────────────────────┘

WEBSOCKET INTEGRATION:

  Client connects to: ws://api/events/{event_id}/stream

  Server pushes:
  {
    "type": "new_claim",
    "claim": {...},
    "source": "BBC",
    "affects": ["narrative_summary", "perspective_balance"],
    "timestamp": "2024-12-22T10:30:00Z"
  }

  {
    "type": "corroboration",
    "claim_id": "...",
    "corroborated_by": "Reuters",
    "new_confidence": 0.85,
    "timestamp": "..."
  }

  {
    "type": "quest_fulfilled",
    "quest_id": "...",
    "fulfilled_by": "user_123",
    "points_awarded": 15,
    "timestamp": "..."
  }

UI UPDATES:

  - New claims slide into timeline with animation
  - Corroboration badges update in real-time
  - Quest completion shows celebration animation
  - Perspective balance bar animates changes
  - "X users watching this story" indicator

NOTIFICATION SYSTEM:

  Users can subscribe to:
  - Specific events ("Notify me of updates to Jimmy Lai trial")
  - Quest fulfillment ("Alert me when someone answers my question")
  - Major narrative shifts ("Breaking: death toll updated")

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  🔔 Notifications                                                       │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  ● Jimmy Lai Trial: New claim from UK Foreign Office          2m ago   │
  │  ● HK Fire: Death toll updated to 160 (was 156)              15m ago   │
  │  ○ Your quest "Gov statement" fulfilled by @NewsHunter       1hr ago   │
  └─────────────────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# PHASE 4: BACKEND ENHANCEMENTS (Parallel Track)
# =============================================================================

PHASE_4 = """
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: BACKEND EVOLUTION (Runs in parallel)                              │
│  "Enhance without breaking"                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

STRATEGY: Keep existing pipeline, add new capabilities alongside

CURRENT PIPELINE (Keep running):
  URL → ExtractionWorker → KnowledgeWorker → EventWorker → Display

NEW CAPABILITIES TO ADD:

1. EPISTEMIC STATE COMPUTATION (New service)

   EpistemicStateWorker:
   - Listens to: queue:event:updated
   - Computes: mass, heat, entropy, coverage for event
   - Stores: event_epistemic_state table
   - Publishes: WebSocket updates

   class EpistemicState:
       event_id: str
       mass: float          # Total evidence weight
       heat: float          # Recency factor (0-1)
       entropy: float       # Uncertainty measure
       coverage: float      # % of expected aspects covered
       perspective_balance: Dict[str, float]
       open_quests: List[Quest]
       last_computed: datetime

2. QUEST GENERATOR (New service)

   QuestWorker:
   - Listens to: queue:event:updated
   - Analyzes: gaps in coverage, perspective imbalance
   - Generates: Quest objects with bounties
   - Stores: quests table

   class Quest:
       id: str
       event_id: str
       quest_type: str      # "missing_source", "perspective_gap", "verification"
       description: str
       bounty: int
       criteria: Dict       # What fulfills this quest
       status: str          # "open", "pending_review", "fulfilled"
       fulfilled_by: str    # user_id
       created_at: datetime

3. NARRATIVE GENERATOR (Enhanced existing)

   Current: Basic summary from claims
   New: Attributed narrative with uncertainty markers

   def generate_narrative(event_id):
       claims = get_claims(event_id)

       # Group by topic/aspect
       aspects = cluster_claims_by_topic(claims)

       narrative_parts = []
       for aspect, aspect_claims in aspects.items():
           # Generate sentence with attribution
           sentence = synthesize_sentence(aspect_claims)
           sources = [c.source for c in aspect_claims]
           confidence = compute_confidence(aspect_claims)

           narrative_parts.append({
               'text': sentence,
               'sources': sources,
               'confidence': confidence,
               'disputed_by': find_contradictions(aspect_claims),
               'needs': identify_gaps(aspect_claims)
           })

       return narrative_parts

4. CONTRIBUTION PROCESSOR (New service)

   ContributionWorker:
   - Receives: user URL submissions
   - Validates: not duplicate, relevant to event/quest
   - Queues: to ExtractionWorker with contribution metadata
   - Credits: user when claims extracted

DATABASE ADDITIONS:

  -- Epistemic state per event
  CREATE TABLE event_epistemic_state (
      event_id UUID PRIMARY KEY REFERENCES events(id),
      mass FLOAT,
      heat FLOAT,
      entropy FLOAT,
      coverage FLOAT,
      perspective_balance JSONB,
      computed_at TIMESTAMP
  );

  -- Quests
  CREATE TABLE quests (
      id UUID PRIMARY KEY,
      event_id UUID REFERENCES events(id),
      quest_type VARCHAR(50),
      description TEXT,
      bounty INT,
      criteria JSONB,
      status VARCHAR(20),
      fulfilled_by UUID REFERENCES users(id),
      created_at TIMESTAMP,
      fulfilled_at TIMESTAMP
  );

  -- User contributions
  CREATE TABLE contributions (
      id UUID PRIMARY KEY,
      user_id UUID REFERENCES users(id),
      contribution_type VARCHAR(50),
      event_id UUID REFERENCES events(id),
      quest_id UUID REFERENCES quests(id),
      source_url TEXT,
      status VARCHAR(20),
      points_awarded INT,
      created_at TIMESTAMP,
      processed_at TIMESTAMP
  );

  -- Claim annotations
  CREATE TABLE claim_annotations (
      id UUID PRIMARY KEY,
      claim_id UUID REFERENCES claims(id),
      user_id UUID REFERENCES users(id),
      annotation_type VARCHAR(50),
      content TEXT,
      source_url TEXT,
      created_at TIMESTAMP
  );
"""

# =============================================================================
# IMPLEMENTATION ROADMAP
# =============================================================================

ROADMAP = """
┌─────────────────────────────────────────────────────────────────────────────┐
│  IMPLEMENTATION ROADMAP                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

WEEK 1: Foundation
━━━━━━━━━━━━━━━━━━
  Frontend:
  □ Create EpistemicEventPage component shell
  □ Build ClaimStream component (timeline view)
  □ Build PerspectiveBalance visualization
  □ Design QuestCard component

  Backend:
  □ Add event_epistemic_state table
  □ Add quests table
  □ Create GET /api/events/{id}/epistemic-state endpoint
  □ Create GET /api/events/{id}/quests endpoint

WEEK 2: Core Features
━━━━━━━━━━━━━━━━━━━━━
  Frontend:
  □ Build ContributionModal (URL submission)
  □ Build UserProfile page with contribution history
  □ Integrate epistemic state display on event pages
  □ Add quest list to event pages

  Backend:
  □ Create EpistemicStateWorker (compute mass/heat/entropy)
  □ Create QuestWorker (generate quests from gaps)
  □ Create POST /api/contributions/submit endpoint
  □ Add contributions table

WEEK 3: Community Features
━━━━━━━━━━━━━━━━━━━━━━━━━━
  Frontend:
  □ Build notification system UI
  □ Add claim annotation interface
  □ Create leaderboard page
  □ Add "watching this story" indicator

  Backend:
  □ Create ContributionWorker (process submissions)
  □ Add WebSocket support for real-time updates
  □ Implement notification service
  □ Add claim_annotations table

WEEK 4: Polish & Launch
━━━━━━━━━━━━━━━━━━━━━━━
  Frontend:
  □ Animation and transitions for real-time updates
  □ Mobile responsiveness
  □ Onboarding flow for new contributors
  □ Feedback collection widget

  Backend:
  □ Performance optimization
  □ Rate limiting for contributions
  □ Spam detection for submissions
  □ Analytics/metrics collection

PARALLEL TRACK: Narrative Enhancement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  □ Improve claim clustering for narrative generation
  □ Add contradiction detection to NarrativeGenerator
  □ Implement source credibility scoring
  □ Build perspective detection model
"""

# =============================================================================
# STARTING POINT: MINIMAL VIABLE EPISTEMIC PAGE
# =============================================================================

STARTING_POINT = """
┌─────────────────────────────────────────────────────────────────────────────┐
│  STARTING POINT: Day 1 Implementation                                       │
│  "Get something visible immediately"                                        │
└─────────────────────────────────────────────────────────────────────────────┘

TODAY'S GOAL: Transform current event page to show epistemic awareness

STEP 1: Add epistemic indicators to existing event page

  Current EventPage.tsx shows:
  - event.title
  - event.summary
  - event.claims (list)
  - event.entities (tags)

  ADD to EventPage.tsx:
  - Source count and diversity indicator
  - Claim timeline (sort claims by time)
  - Simple "gaps" section (hardcoded queries for now)

STEP 2: Create new API endpoint

  GET /api/events/{id}/epistemic
  Returns:
  {
    "source_count": 26,
    "source_diversity": {
      "wire": 3,
      "international": 8,
      "local": 5,
      "official": 1,
      "ngo": 2
    },
    "claim_count": 156,
    "temporal_span": {
      "first": "2024-12-19T10:00:00Z",
      "last": "2024-12-22T08:30:00Z"
    },
    "has_contradiction": true,
    "gaps": [
      {"type": "missing_source", "description": "No direct government statement"},
      {"type": "perspective", "description": "Limited defense perspective"}
    ]
  }

STEP 3: Quick Quest UI

  Even before full quest system, show static "gaps" as invitation:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  ❓ Help improve this story                                             │
  │                                                                         │
  │  This narrative could be stronger with:                                 │
  │  • Direct statement from HK Government                                  │
  │  • More human rights organization perspectives                          │
  │  • UK/US government official positions                                  │
  │                                                                         │
  │  [Suggest a Source]                                                     │
  └─────────────────────────────────────────────────────────────────────────┘

This gives users IMMEDIATE visibility into:
1. How well-sourced the story is
2. What perspectives are missing
3. That they can contribute

And it requires MINIMAL backend changes to start.
"""

# =============================================================================
# FILE STRUCTURE FOR NEW COMPONENTS
# =============================================================================

FILE_STRUCTURE = """
frontend/src/
├── components/
│   ├── epistemic/
│   │   ├── EpistemicStateCard.tsx      # Shows mass/heat/coverage
│   │   ├── PerspectiveBalance.tsx      # Perspective distribution viz
│   │   ├── ClaimTimeline.tsx           # Chronological claim stream
│   │   ├── QuestCard.tsx               # Single quest display
│   │   ├── QuestList.tsx               # List of open quests
│   │   ├── ContributionModal.tsx       # URL submission dialog
│   │   └── SourceDiversity.tsx         # Source type breakdown
│   ├── narrative/
│   │   ├── AttributedSentence.tsx      # Sentence with source badges
│   │   ├── NarrativeSummary.tsx        # Full narrative with markers
│   │   └── UncertaintyMarker.tsx       # ? indicator component
│   └── community/
│       ├── ContributionHistory.tsx     # User's past contributions
│       ├── Leaderboard.tsx             # Top contributors
│       └── NotificationBell.tsx        # Notification dropdown
├── pages/
│   └── EpistemicEventPage.tsx          # New event page layout
└── hooks/
    ├── useEpistemicState.ts            # Fetch epistemic state
    ├── useEventStream.ts               # WebSocket subscription
    └── useContributions.ts             # User contribution actions

backend/
├── api/
│   ├── epistemic.py                    # Epistemic state endpoints
│   ├── quests.py                       # Quest endpoints
│   └── contributions.py                # Contribution endpoints
├── workers/
│   ├── epistemic_state_worker.py       # Compute epistemic quantities
│   ├── quest_worker.py                 # Generate quests from gaps
│   └── contribution_worker.py          # Process user submissions
└── services/
    ├── narrative_generator.py          # Generate attributed narrative
    ├── gap_detector.py                 # Detect missing aspects
    └── perspective_analyzer.py         # Detect epistemic poles
"""

if __name__ == "__main__":
    print("=" * 80)
    print("WEBAPP RECONSTRUCTION PROPOSAL")
    print("=" * 80)
    print(PHASE_1)
    print(PHASE_2)
    print(PHASE_3)
    print(PHASE_4)
    print(ROADMAP)
    print(STARTING_POINT)
    print(FILE_STRUCTURE)
