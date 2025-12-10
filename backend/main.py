"""
HereNews Service Farm - FastAPI Backend
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router as artifacts_router
from endpoints_rogue import router as rogue_router
from endpoints_legacy import router as legacy_router
from endpoints_events import router as events_router

# Add backend to path for API imports
sys.path.insert(0, str(Path(__file__).parent))

# Import coherence feed (standalone, no auth dependencies)
coherence_router = None
try:
    from api import coherence
    coherence_router = coherence.router
    print("✅ Coherence feed loaded")
except ImportError as e:
    print(f"⚠️  Coherence feed not available: {e}")

# Try to import community feature routers (may fail if dependencies missing)
community_routers = []
try:
    from api import auth, comments, chat, preview, submissions, extraction, event_page, map
    community_routers = [
        (auth.router, "/api/auth", ["Authentication"]),
        (comments.router, "/api/comments", ["Comments"]),
        (chat.router, "/api/chat", ["Chat"]),
        (preview.router, "/api/preview", ["Preview"]),
        (submissions.router, "/api/submissions", ["Submissions"]),
        (extraction.router, "/api/extraction", ["Extraction"]),
        (event_page.router, "/api/event", ["Event Pages"]),
        (map.router, "/api/map", ["Map"]),
    ]
    print("✅ Community features loaded successfully")
except ImportError as e:
    print(f"⚠️  Community features not available (missing dependencies): {e}")
    print("   Install: authlib, python-jose, itsdangerous, email-validator")

app = FastAPI(
    title="HereNews Service Farm",
    description="Knowledge graph extraction and event detection service",
    version="2.0.0"
)

# Session middleware for OAuth (if available)
try:
    from starlette.middleware.sessions import SessionMiddleware
    from config import get_settings
    settings = get_settings()
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.jwt_secret_key,
        session_cookie="session",
        max_age=86400,  # 24 hours
        same_site="lax",
        https_only=False
    )
    print("✅ Session middleware enabled")
except ImportError:
    print("⚠️  Session middleware not available (missing itsdangerous)")

# Enable CORS for webapp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints - all under /api/*
app.include_router(artifacts_router, prefix="/api", tags=["Artifacts"])
app.include_router(rogue_router, prefix="/api", tags=["Rogue Extraction"])
app.include_router(events_router, prefix="/api", tags=["Events"])

# Coherence feed (standalone)
if coherence_router:
    app.include_router(coherence_router, prefix="/api/coherence", tags=["Coherence"])

# Legacy demo endpoints (archived)
app.include_router(legacy_router, prefix="/api/demo", tags=["Legacy Demo"])

# Community feature routers (if available)
for router, prefix, tags in community_routers:
    app.include_router(router, prefix=prefix, tags=tags)

# Static files for frontend assets
static_path = Path("/app/static")
if static_path.exists():
    app.mount("/app/assets", StaticFiles(directory=str(static_path / "assets")), name="assets")

@app.get("/")
async def root():
    return {"message": "HereNews Platform", "app_url": "/app"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/health")
async def api_health():
    return {"status": "ok", "service": "service_farm"}

# Fallback auth endpoint (if community features not loaded)
if not community_routers:
    @app.get("/api/auth/status")
    async def auth_status_fallback():
        """Fallback when auth system is not available"""
        return {
            "authenticated": False,
            "user": None,
            "message": "Auth system not enabled"
        }

# Frontend SPA routes (must be last) - /app/* serves the React app
@app.get("/app", response_class=HTMLResponse)
@app.get("/app/", response_class=HTMLResponse)
@app.get("/app/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str = ""):
    """Serve React frontend for all /app/* routes (SPA routing)"""
    index_path = Path("/app/static/index.html")
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>Frontend Not Built</h1><p>Run: docker run --rm -v $(pwd)/frontend:/app -v $(pwd)/static:/static -w /app node:18-alpine npm run build</p>",
            status_code=503
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
