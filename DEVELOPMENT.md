# Development Guide

## Quick Start

```bash
# Start all services
docker-compose up -d

# Access app
open http://localhost:7272/app/
```

## Frontend Development

After making changes to `frontend/app/`:

```bash
# Rebuild and deploy frontend (one command)
docker-compose build frontend && docker-compose run --rm frontend
```

The app container will automatically serve the new files (they're volume-mounted).

**No need to restart the app container** - just hard refresh your browser (Ctrl+Shift+R / Cmd+Shift+R).

## Backend Development

The backend has hot-reload enabled. Just edit files in `backend/` and uvicorn will automatically restart.

## Troubleshooting

### Frontend not updating
1. Check browser is loading new JS file: DevTools → Network → look for `index-*.js`
2. Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
3. Clear browser cache completely

### API errors
```bash
# Check app logs
docker-compose logs -f app

# Check all services
docker-compose ps
```

## Architecture

- **Frontend**: React + TypeScript + Vite → builds to `/static/`
- **Backend**: FastAPI (Python) → serves `/api/*` and `/app/*` (SPA)
- **Static files**: Volume-mounted at `./static` → `/app/static`
