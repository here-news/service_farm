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
from endpoints_events import router as events_router
import httpx

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

# Import auth router (core feature, load separately)
auth_router = None
try:
    from api import auth
    auth_router = auth.router
    print("✅ Auth loaded")
except ImportError as e:
    print(f"⚠️  Auth not available: {e}")

# Import contributions router (epistemic layer)
contributions_router = None
try:
    from api import contributions
    contributions_router = contributions.router
    print("✅ Contributions loaded")
except ImportError as e:
    print(f"⚠️  Contributions not available: {e}")

# Import inquiry router (MVP)
inquiry_router = None
try:
    from api import inquiry
    inquiry_router = inquiry.router
    print("✅ Inquiry MVP loaded")
except ImportError as e:
    print(f"⚠️  Inquiry not available: {e}")

# Try to import other feature routers
feature_routers = []
for module_name, prefix, tags in [
    ("map", "/api/map", ["Map"]),
    ("preview", "/api/preview", ["Preview"]),
]:
    try:
        module = __import__(f"api.{module_name}", fromlist=[module_name])
        feature_routers.append((module.router, prefix, tags))
    except ImportError as e:
        print(f"⚠️  {module_name} not available: {e}")

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

# Auth router (core) - already has /api/auth prefix in router
if auth_router:
    app.include_router(auth_router)

# Contributions router (epistemic layer)
if contributions_router:
    app.include_router(contributions_router, prefix="/api", tags=["Contributions"])

# Inquiry router (MVP)
if inquiry_router:
    app.include_router(inquiry_router, prefix="/api", tags=["Inquiry"])

# Feature routers (if available)
for router, prefix, tags in feature_routers:
    app.include_router(router, prefix=prefix, tags=tags)

# Static files for frontend assets
# Check /static first (Docker image), then /app/static (local dev)
static_path = Path("/static") if Path("/static").exists() else Path("/app/static")
if static_path.exists() and (static_path / "assets").exists():
    # Mount at both /assets (for root landing) and /app/assets (for app routes)
    app.mount("/assets", StaticFiles(directory=str(static_path / "assets")), name="root-assets")
    app.mount("/app/assets", StaticFiles(directory=str(static_path / "assets")), name="app-assets")

# Serve prototypes (design mockups)
prototypes_path = Path("/app/prototypes") if Path("/app/prototypes").exists() else Path("/media/im3/plus/lab4/re_news/service_farm/frontend/prototypes")
if prototypes_path.exists():
    app.mount("/prototypes", StaticFiles(directory=str(prototypes_path), html=True), name="prototypes")
    print(f"✅ Prototypes served at /prototypes")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve React frontend at root (landing page)"""
    index_path = Path("/static/index.html") if Path("/static/index.html").exists() else Path("/app/static/index.html")
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>Frontend Not Built</h1><p>Rebuild the app container: docker-compose build app</p>",
            status_code=503
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/health")
async def api_health():
    return {"status": "ok", "service": "service_farm"}

# Fallback auth endpoint (if auth not loaded)
if not auth_router:
    @app.get("/api/auth/status")
    async def auth_status_fallback():
        """Fallback when auth system is not available"""
        return {
            "authenticated": False,
            "user": None,
            "message": "Auth system not enabled"
        }

# UEE Proxy - Forward /api/uee/* to the internal UEE server
UEE_INTERNAL_URL = "http://localhost:5050/api"

@app.get("/api/uee/events")
async def uee_list_events():
    """Proxy to UEE: list investigations"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{UEE_INTERNAL_URL}/events")
        return response.json()

@app.post("/api/uee/events")
async def uee_create_event(data: dict):
    """Proxy to UEE: create investigation"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{UEE_INTERNAL_URL}/events", json=data)
        return response.json()

@app.get("/api/uee/events/{event_id}")
async def uee_get_event(event_id: str):
    """Proxy to UEE: get investigation state"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{UEE_INTERNAL_URL}/events/{event_id}")
        return response.json()

@app.post("/api/uee/events/{event_id}/claims")
async def uee_add_claim(event_id: str, claim: dict):
    """Proxy to UEE: add claim"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{UEE_INTERNAL_URL}/events/{event_id}/claims",
            json=claim
        )
        return response.json()

@app.get("/api/uee/events/{event_id}/narrative")
async def uee_get_narrative(event_id: str):
    """Proxy to UEE: generate semantic narrative"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{UEE_INTERNAL_URL}/events/{event_id}/narrative")
        return response.json()

# Frontend SPA routes (must be last)
# Helper to serve the SPA
async def _serve_spa():
    """Serve React frontend index.html"""
    index_path = Path("/static/index.html") if Path("/static/index.html").exists() else Path("/app/static/index.html")
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>Frontend Not Built</h1><p>Rebuild the app container: docker-compose build app</p>",
            status_code=503
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# SPA routes - /app/* (legacy)
@app.get("/app", response_class=HTMLResponse)
@app.get("/app/", response_class=HTMLResponse)
@app.get("/app/{full_path:path}", response_class=HTMLResponse)
async def serve_spa_app(full_path: str = ""):
    return await _serve_spa()

# SPA routes - direct paths (new)
@app.get("/graph", response_class=HTMLResponse)
@app.get("/map", response_class=HTMLResponse)
@app.get("/archive", response_class=HTMLResponse)
@app.get("/inquiry", response_class=HTMLResponse)
async def serve_spa_direct():
    return await _serve_spa()

@app.get("/inquiry/{inquiry_path:path}", response_class=HTMLResponse)
async def serve_spa_inquiry(inquiry_path: str):
    return await _serve_spa()

@app.get("/event/{event_slug:path}", response_class=HTMLResponse)
async def serve_spa_event(event_slug: str):
    return await _serve_spa()

@app.get("/event-inquiry/{event_slug:path}", response_class=HTMLResponse)
async def serve_spa_event_inquiry(event_slug: str):
    return await _serve_spa()

@app.get("/entity/{entity_id:path}", response_class=HTMLResponse)
async def serve_spa_entity(entity_id: str):
    return await _serve_spa()

@app.get("/entity-inquiry/{entity_slug:path}", response_class=HTMLResponse)
async def serve_spa_entity_inquiry(entity_slug: str):
    return await _serve_spa()

@app.get("/story/{story_path:path}", response_class=HTMLResponse)
async def serve_spa_story(story_path: str):
    return await _serve_spa()

@app.get("/page/{page_id:path}", response_class=HTMLResponse)
async def serve_spa_page(page_id: str):
    return await _serve_spa()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
