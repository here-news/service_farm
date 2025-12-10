# Event Visualization Components

## Summary

Successfully created 3 new React components from legacy HTML visualizations and integrated them into the EventPage.

## New Components

### 1. TimelineView (`app/components/event/TimelineView.tsx`)
- **Features:**
  - Interactive horizontal timeline with pan & zoom
  - Claims grouped by time proximity (within 1 hour)
  - Alternating card heights for visual interest
  - Mouse drag to pan, scroll to zoom
  - Time markers with formatted timestamps
  - Confidence indicators on each claim

- **Props:**
  - `claims: Claim[]` - Array of claims with event_time

- **Implementation:**
  - Pure React with CSS transforms
  - No external dependencies
  - Responsive canvas with dynamic width

### 2. GraphView (`app/components/event/GraphView.tsx`)
- **Features:**
  - Network graph visualization using HTML Canvas
  - Node types: Event (center), Entities (circle), Claims (outer)
  - Color-coded by entity type
  - Connections showing relationships
  - Interactive legend

- **Props:**
  - `entities: Entity[]`
  - `claims: Claim[]`
  - `eventName: string`

- **Implementation:**
  - Canvas 2D rendering
  - Static layout (no force simulation for simplicity)
  - Entity type colors: Person (orange), Organization (purple), Location (green)

### 3. MapView (`app/components/event/MapView.tsx`)
- **Features:**
  - Location-focused view
  - Filters and displays location entities
  - Lists all identified locations
  - Placeholder for future Leaflet integration

- **Props:**
  - `entities: Entity[]`
  - `eventName: string`

- **Current State:**
  - Displays location list
  - Ready for Leaflet.js integration
  - Shows count of identified locations

## Integration

All components integrated into `EventPage.tsx`:
- **Narrative tab**: ReactMarkdown rendering of event summary
- **Timeline tab**: TimelineView component ✅
- **Graph tab**: GraphView component ✅
- **Map tab**: MapView component ✅

## Build Output

- **New bundle**: `index-Ce4pu1qP.js` (731 KB)
- **CSS**: `index-OpMGW_vh.css` (48 KB)
- **Total modules**: 1,299 transformed

## Legacy Files Preserved

Original HTML files saved in `frontend/page_samples/`:
- `timeline.html` - Full D3-based cascading timeline
- `event_graph.html` - 3D force-directed graph
- `map.html` - Leaflet map with markers
- `event.html` - Main event page

## Future Enhancements

### Timeline
- [ ] Add D3 for smoother animations
- [ ] Implement force-directed vertical spacing
- [ ] Add claim filtering by confidence
- [ ] Export timeline as image

### Graph
- [ ] Integrate react-force-graph-3d for interactive 3D
- [ ] Add node dragging
- [ ] Implement zoom and pan
- [ ] Show claim text on hover

### Map
- [ ] Integrate Leaflet.js
- [ ] Geocode location names to coordinates
- [ ] Add clustering for multiple nearby locations
- [ ] Show claims associated with each location

## Testing

To test the new components:

1. Navigate to any event: `http://localhost:7272/app/event/ev_xxxxxxxx`
2. Click through the tabs:
   - **Timeline**: Drag and zoom the timeline
   - **Graph**: View the network visualization
   - **Map**: See identified locations

## Dependencies

Currently using only React built-ins. Future additions:
- `leaflet` + `react-leaflet` for maps
- `d3` for advanced timeline animations
- `react-force-graph` for 3D graph (optional)
