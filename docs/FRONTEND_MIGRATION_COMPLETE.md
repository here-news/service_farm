# ✅ Frontend Migration Complete

## Summary

Successfully migrated the React frontend from `../webapp/frontend/` to unified structure.

---

## Directory Structure

```
service_farm/
├── frontend/                    # ✅ Main React/TypeScript app
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── utils/               # Utility functions
│   │   └── main.ts              # Entry point
│   ├── app/                     # App-specific code
│   ├── node_modules/            # Dependencies (already installed)
│   ├── package.json
│   ├── vite.config.ts           # ✅ Already configured!
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── index.html
│
├── frontend-legacy/             # ✅ Preserved HTML tools
│   ├── event.html               # Event viewer
│   ├── timeline.html            # Timeline visualization
│   ├── map.html                 # Map interface
│   └── index.html               # Dashboard
│
├── static/                      # Build output (npm run build)
│   └── (generated assets)
│
└── backend/                     # Python backend (already restructured)
```

---

## Frontend Tech Stack

**Framework:** React 18 + TypeScript
**Build Tool:** Vite 5
**Styling:** Tailwind CSS
**Routing:** React Router 6
**Visualizations:**
- react-force-graph-2d (knowledge graph)
- Leaflet + react-leaflet (maps)
- Lucide React (icons)

---

## Configuration (Already Set!)

### vite.config.ts
```typescript
export default defineConfig({
  plugins: [react()],
  base: '/app/',                    // Matches main.py route
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // Proxy to FastAPI
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: '../static',           // ✅ Output to service_farm/static/
    emptyOutDir: true
  }
})
```

**Perfect!** Build output already configured for unified deployment.

---

## Development Workflow

### Option 1: Development Mode (Hot Reload)

**Terminal 1 - Backend:**
```bash
cd /media/im3/plus/lab4/re_news/service_farm
python main.py
# Runs on http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd /media/im3/plus/lab4/re_news/service_farm/frontend
npm run dev
# Runs on http://localhost:5173
# API calls proxied to :8000
```

### Option 2: Production Build

```bash
cd frontend
npm run build
# Outputs to ../static/

# Backend serves static/ at http://localhost:8000/app
cd ..
python main.py
```

---

## What Was Moved

### ✅ Preserved (frontend-legacy/)
- **event.html** - Event detail viewer with claims
- **timeline.html** - Timeline visualization
- **map.html** - Geographic map interface
- **index.html** - Dashboard/navigation
- **event_graph.html** - Graph visualization
- **sections.html** - Section-based views

**Use Case:** Quick debugging, standalone tools, legacy references

### ✅ Main App (frontend/)
- **React SPA** - Full featured web application
- **Components** - Reusable UI components
- **Routing** - Client-side routing with React Router
- **API Integration** - Configured to call `/api/*` endpoints
- **Build Pipeline** - TypeScript + Vite + Tailwind

---

## API Integration

Frontend is configured to call backend API at `/api/*`:

```typescript
// Example API calls (in frontend/src/)
const response = await fetch('/api/events');          // GET events
const auth = await fetch('/api/auth/status');         // Check auth
const comments = await fetch('/api/comments/story/ev_xxx');
```

**Development:** Vite proxy forwards `/api` → `http://localhost:8000`
**Production:** Backend serves frontend from `/app`, API from `/api`

---

## Main.py Routes (Already Configured)

```python
# Static assets
app.mount("/app/assets", StaticFiles(directory="static/assets"))

# SPA route
@app.get("/app", response_class=HTMLResponse)
async def app_route():
    """Serves frontend SPA"""
    return HTMLResponse(content=open("static/index.html").read())

# Catch-all for client-side routing
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serves index.html for all non-API routes"""
    return HTMLResponse(content=open("static/index.html").read())
```

---

## Next Steps

### 1. Install Dependencies (if needed)
```bash
cd frontend
npm install
```

### 2. Development
```bash
# Terminal 1: Backend
python main.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

Visit: `http://localhost:5173`

### 3. Production Build
```bash
cd frontend
npm run build
# ✅ Outputs to ../static/

# Backend serves from /app
cd .. && python main.py
```

Visit: `http://localhost:8000/app`

### 4. Docker Integration (future)
```dockerfile
# Multi-stage build in Dockerfile
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build
# Output: /app/static/

FROM python:3.11
COPY --from=frontend-build /app/static /app/static
# Backend serves static/
```

---

## Benefits

✅ **Unified deployment** - Single server serves both frontend and API
✅ **Clean separation** - Frontend code in frontend/, legacy tools preserved
✅ **Already configured** - Vite config outputs to correct directory
✅ **Development ready** - Hot reload + API proxy works out of box
✅ **Production ready** - Build outputs to static/, main.py serves it
✅ **Type-safe** - TypeScript throughout frontend
✅ **Modern stack** - React 18, Vite 5, Tailwind CSS

---

## File Counts

- **frontend/**: 21 node_modules dirs, React components, TypeScript source
- **frontend-legacy/**: 18 HTML files (preserved for reference)
- **static/**: Will contain built assets after `npm run build`

---

**✅ Frontend migration complete! Ready for development.**
