"""
HereNews Service Farm - FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router as demo_router
from endpoints_gen2 import router as gen2_router

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

# Gen2 endpoints (production)
app.include_router(gen2_router, prefix="/api/v2", tags=["Gen2"])

# Demo endpoints (legacy, for comparison)
app.include_router(demo_router, prefix="/api/demo", tags=["Demo"])

@app.get("/")
async def root():
    return {
        "service": "HereNews Service Farm",
        "version": "2.0.0",
        "architecture": "Queue-driven workers with autonomous decision-making",
        "endpoints": {
            "submit_url": "POST /api/v2/url?url=...",
            "get_status": "GET /api/v2/url/{page_id}",
            "queue_stats": "GET /api/v2/queue/stats",
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
