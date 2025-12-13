# PagePage Feature Plan

> Quick reference for implementing `/page/{pg_id}` UI

---

## Current State

### Existing Endpoints
| Endpoint | Purpose | File |
|----------|---------|------|
| `GET /api/url/{page_id}` | Page status + basic metadata | `backend/endpoints.py:430` |
| `GET /api/claim/{claim_id}` | Single claim + source page + entities | `backend/endpoints_events.py:306` |
| `GET /api/claims?page_id=xxx` | List claims for a page | `backend/endpoints.py:533` |
| `GET /api/entities?page_id=xxx` | List entities mentioned in page | `backend/endpoints.py:543` |

### Frontend Routes (No page route yet)
```
/app/                    → HomePage
/app/event/:eventSlug    → EventPage
/app/entity/:entityId    → EntityPage  ← Reference pattern
/app/graph               → GraphPage
/app/map                 → MapPage
```

---

## Decision: `/page/{pg_id}` with Anchor Links

### URL Pattern
```
/app/page/pg_abc123           → Full page view
/app/page/pg_abc123#cl_xyz789 → Jump to specific claim
```

### Why This Approach
- **Claims are contextual** - they come from a page, make more sense viewed there
- **Anchor links** (`#cl_xxx`) allow deep-linking to specific claims
- **Follows architecture** - "atomic facts" philosophy; claims are atoms, pages are evidence containers
- **Matches existing patterns** - EntityPage established the template
- **No separate `/claim/{cl_id}` page needed** (for now) - claims too atomic to display alone

---

## Implementation Plan

### 1. Backend: `GET /api/page/{page_id}`

**File:** `backend/endpoints.py` (add new endpoint)

**Response Structure:**
```json
{
  "page": {
    "id": "pg_xxxxxxxx",
    "url": "https://...",
    "canonical_url": "https://...",
    "title": "Article Title",
    "description": "Meta description...",
    "author": "John Doe",
    "thumbnail_url": "https://...",
    "site_name": "BBC News",
    "domain": "bbc.com",
    "word_count": 2500,
    "language": "en",
    "status": "event_complete",
    "pub_time": "2025-12-10T...",
    "created_at": "2025-12-11T...",
    "metadata_confidence": 0.85
  },
  "claims": [
    {
      "id": "cl_xxxxxxxx",
      "text": "44 people died in the fire",
      "event_time": "2025-11-26T...",
      "confidence": 0.92,
      "modality": "observation",
      "entities": [
        {"id": "en_xxx", "canonical_name": "Hong Kong", "entity_type": "LOCATION"}
      ]
    }
  ],
  "claims_count": 12,
  "entities": [
    {
      "id": "en_xxxxxxxx",
      "canonical_name": "Hong Kong",
      "entity_type": "LOCATION",
      "mention_count": 5,
      "wikidata_qid": "Q8646"
    }
  ],
  "entities_count": 5
}
```

**Repository Methods to Use:**
- `PageRepository.get_by_id(page_id)`
- `PageRepository.get_claims(page_id)`
- `PageRepository.get_entities(page_id)`

### 2. Frontend: `PagePage.tsx`

**File:** `frontend/app/PagePage.tsx`

**UI Sections:**

| Section | Content |
|---------|---------|
| **Header** | Title, thumbnail, source domain badge, pub date, status badge |
| **Stats Bar** | Word count, claim count, entity count, confidence % |
| **Claims List** | Each claim card with `id="cl_xxx"` anchor, confidence bar, timestamp, linked entity chips |
| **Entities Mentioned** | Grid of entity cards (clickable → EntityPage) |
| **Source Footer** | External link to original URL, processing timestamps |

**Anchor Scroll Behavior:**
```typescript
useEffect(() => {
  if (location.hash) {
    const element = document.getElementById(location.hash.slice(1));
    element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    element?.classList.add('highlight-pulse'); // CSS animation
  }
}, [location.hash]);
```

### 3. Route Registration

**File:** `frontend/app/App.tsx`

```typescript
<Route path="/app/page/:pageId" element={<PagePage />} />
```

---

## Status Badge Colors

| Status | Color | Meaning |
|--------|-------|---------|
| `stub` | Gray | Just submitted |
| `preview` | Yellow | Metadata fetched |
| `extracted` | Blue | Content extracted |
| `knowledge_complete` | Purple | Claims/entities extracted |
| `event_complete` | Green | Fully processed |
| `failed` | Red | Processing failed |

---

## Linking Patterns

### From EventPage → PagePage
```tsx
<Link to={`/app/page/${claim.page_id}#${claim.id}`}>
  View in source
</Link>
```

### From EntityPage → PagePage (claim context)
```tsx
<Link to={`/app/page/${claim.page_id}#${claim.id}`}>
  {claim.text}
</Link>
```

### From PagePage → EntityPage
```tsx
<Link to={`/app/entity/${entity.id}`}>
  {entity.canonical_name}
</Link>
```

---

## Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| **Add** | `backend/endpoints.py` | Add `GET /api/page/{page_id}` endpoint |
| **Create** | `frontend/app/PagePage.tsx` | New page component |
| **Modify** | `frontend/app/App.tsx` | Add route |
| **Modify** | `frontend/app/types/story.ts` | Add PageResponse type |
| **Optional** | Other pages | Add "View source" links to claims |

---

## Future Considerations

### Maybe Later: `/claim/{cl_id}` Page
- If claims become more complex (with replies, disputes, votes)
- When community features need claim-level discussion
- For now: `/api/claim/{cl_id}` endpoint exists for programmatic access

### Content Preview
- Could show snippet of `content_text` around each claim
- Highlight where in the article each claim was extracted from

### Claim Comparison
- Side-by-side view of same claim from different pages
- Useful for contradiction detection UI

---

*Reference: See `docs/architecture.principles.md` for full system design*
