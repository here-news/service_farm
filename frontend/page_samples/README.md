# Legacy Event Visualization Pages

These are the original standalone HTML pages for event visualizations, recovered from git commit `6d13f91`.

## Files

1. **timeline.html** - Interactive timeline with cascading layout
2. **map.html** - Geographic map view using Leaflet
3. **event_graph.html** - 3D force-directed graph visualization
4. **event.html** - Main event detail page

## Libraries Used

- **Leaflet** (1.9.4) - Map visualization
- **3D Force Graph** - Network graph visualization (d3-based)
- **D3.js** - Timeline and data visualization

## Current Implementation

The current React app (`frontend/app/EventPage.tsx`) has simplified versions of these views:
- **Narrative tab**: Displays event summary (working)
- **Timeline tab**: Simple list grouped by time (needs enhancement)
- **Graph tab**: Placeholder only
- **Map tab**: Placeholder only

## Integration Plan

To integrate these rich visualizations into the React app:

### Option 1: Extract Components (Recommended)
1. Convert the inline JavaScript to React components
2. Use same libraries: Leaflet for maps, D3 for timeline
3. Keep the same visual design and interactions

### Option 2: iframe Integration (Quick)
1. Serve these HTML files as-is
2. Load them in iframes within the React tab views
3. Pass event data via postMessage

### Option 3: Hybrid
1. Use Leaflet React wrapper for map
2. Port timeline D3 code to a React component
3. Use react-force-graph-3d for the graph

## Key Features to Port

### Timeline
- Horizontal cascading layout
- Zoom controls
- Claim grouping by time
- Smooth animations

### Map
- Location markers for entities
- Claim clustering
- Info popups
- Custom styling

### Graph
- 3D force-directed layout
- Node types: Claims, Entities, Events
- Interactive camera controls
- Color coding by entity type

## Next Steps

1. Review current EventPage.tsx tabs
2. Choose integration approach
3. Install needed npm packages (leaflet, d3, react-force-graph)
4. Port one visualization at a time (start with timeline)
