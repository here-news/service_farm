"""
HereNews Service Farm - FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router as artifacts_router
from endpoints_rogue import router as rogue_router
from endpoints_legacy import router as legacy_router

app = FastAPI(
    title="HereNews Service Farm",
    description="Knowledge graph extraction and event detection service",
    version="2.0.0"
)

# Enable CORS for webapp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main API v2 endpoints
app.include_router(artifacts_router, prefix="/api/v2", tags=["Artifacts"])
app.include_router(rogue_router, prefix="/api/v2", tags=["Rogue Extraction"])

# Legacy demo endpoints (archived)
app.include_router(legacy_router, prefix="/api/demo", tags=["Legacy Demo"])

@app.get("/")
async def root():
    return {
        "service": "HereNews Service Farm",
        "version": "2.0.0",
        "architecture": "Queue-driven workers with autonomous decision-making",
        "endpoints": {
            "submit_artifact": "POST /api/v2/artifacts?url=...",
            "get_artifact": "GET /api/v2/artifacts/{artifact_id}",
            "queue_stats": "GET /api/v2/queue/stats",
            "rogue_tasks": "GET /api/v2/rogue/tasks",
            "demo_legacy": "POST /api/demo/artifacts/draft?url=..."
        },
        "workers": {
            "extraction": 3,
            "semantic": 2,
            "event": 1,
            "enrichment": 1
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
