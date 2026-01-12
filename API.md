# HereNews API Reference

> Quick reference for frontend developers.
> Base URL: `http://localhost:8000/api`

## Events

### List Events
```
GET /events
GET /events?status=active&limit=50&min_coherence=0.3
```
Returns: `{ events: [...], total: number }`

### Get Event
```
GET /event/{event_id}
```
Returns: `{ event, children, parent, entities, claims, page_thumbnails, thought }`

### Event Surfaces
```
GET /surfaces/by-event/{event_id}
```
Returns: `{ event_id, surfaces: [...], total }`

### Event Tensions (Meta-claims)
```
GET /event/{event_id}/tensions
```
Returns: `{ event_id, tensions: [...], total, source_count, claim_count }`

Tension types:
- `single_source_only` - Needs corroboration
- `unresolved_conflict` - Conflicting claims
- `need_primary_source` - Missing official/wire sources

### Event Epistemic State
```
GET /event/{event_id}/epistemic
```
Returns: `{ source_diversity, coverage, heat, gaps, has_contradiction }`

### Event Topology (for visualization)
```
GET /event/{event_id}/topology
```
Returns: `{ claims, relationships, update_chains, contradictions, organism_state }`

## Surfaces

### List Surfaces
```
GET /surfaces
GET /surfaces?limit=50&offset=0&min_claims=2
```
Returns: `{ surfaces: [...], total, limit, offset }`

### Get Surface
```
GET /surfaces/{surface_id}
```
Returns: `{ surface, claims, internal_relations }`

### Surface Stats
```
GET /surfaces/stats
```
Returns: `{ total_surfaces, avg_claims_per_surface, top_anchors }`

## Inquiry

### List Inquiries
```
GET /inquiry
GET /inquiry?status=open&limit=20
```
Returns: `{ inquiries: [...], total }`

### Get Inquiry
```
GET /inquiry/{inquiry_id}
```
Returns: `InquiryDetail` with belief state, schema

### Inquiry Trace (full epistemic state)
```
GET /inquiry/{inquiry_id}/trace
```
Returns: `{ inquiry, belief_state, observations, contributions, surfaces, tasks, resolution }`

### Add Contribution
```
POST /inquiry/{inquiry_id}/contribute
Body: { type, text, source_url?, extracted_value?, observation_kind? }
```

### Add Stake
```
POST /inquiry/{inquiry_id}/stake
Body: { amount: number }
```

## User (Community Loop)

### Get Credits
```
GET /user/credits
Auth: Required
```
Returns: `{ credits_balance, can_stake, reputation }`

### Get Transactions
```
GET /user/transactions?limit=50
Auth: Required
```
Returns: `{ transactions: [...], total }`

### Get Profile
```
GET /user/profile
Auth: Required
```
Returns: `{ user_id, email, name, credits_balance, reputation, stats }`

### Get User Stakes
```
GET /user/stakes
Auth: Required
```
Returns: `{ stakes: [...], total_staked }`

### Get User Contributions
```
GET /user/contributions?limit=50
Auth: Required
```
Returns: `{ contributions: [...], total, total_impact, total_rewards }`

## Auth

### Check Status
```
GET /auth/status
```
Returns: `{ authenticated: bool, user: UserResponse | null }`

### Login
```
GET /auth/login
```
Redirects to Google OAuth

### Logout
```
GET /auth/logout
```
Clears session cookie

### Get Current User
```
GET /auth/me
Auth: Required
```
Returns: `UserResponse`

## Entities

### Get Entity
```
GET /entity/{entity_id}
```
Returns: `{ entity, claims, claims_count, related_events }`

Entity includes Wikidata enrichment:
- `wikidata_qid`, `wikidata_label`, `wikidata_description`
- `image_url`, `latitude`, `longitude`

## Claims

### Get Claim
```
GET /claim/{claim_id}
```
Returns: `{ claim, source, entities }`

## Pages

### List Pages
```
GET /pages
GET /pages?limit=50&offset=0&status=semantic_complete
```
Returns: `{ pages: [...], total }`

### Get Page
```
GET /page/{page_id}
```
Returns: `{ page, claims, claims_count, entities, entities_count }`

## ID Formats

All IDs use short format:
- Event: `ev_xxxxxxxx`
- Surface: `sf_xxxxxxxx`
- Claim: `cl_xxxxxxxx`
- Entity: `en_xxxxxxxx`
- Page: `pg_xxxxxxxx`
- Inquiry: `iq_xxxxxxxx`

## TypeScript Types

See `frontend/app/types/inquiry.ts` for full type definitions.

Key types:
```typescript
interface InquirySummary {
  id: string
  title: string
  status: 'open' | 'resolved' | 'stale' | 'closed'
  rigor: 'A' | 'B' | 'C'
  schema_type: SchemaType
  posterior_probability: number
  entropy_bits: number
  stake: number
  contributions: number
  open_tasks: number
}

interface Surface {
  id: string
  name: string
  claim_count: number
  sources: string[]
  entities?: string[]
  in_scope: boolean
  entropy?: number
}

interface InquiryTask {
  id: string
  type: 'need_primary_source' | 'unresolved_conflict' | 'single_source_only' | 'high_entropy'
  description: string
  bounty: number
  completed: boolean
}
```

## Error Responses

```json
{
  "detail": "Error message"
}
```

Common status codes:
- `400` - Invalid ID format
- `404` - Resource not found
- `500` - Server error
