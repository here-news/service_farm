"""
Demo Backend - FastAPI app
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router

app = FastAPI(title="Breathing KB Demo")

# Enable CORS for webapp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {
        "service": "Breathing Knowledge Base Demo",
        "version": "0.1.0",
        "endpoints": {
            "submit_url": "POST /api/artifacts/draft?url=...",
            "poll_status": "GET /api/artifacts/draft/{artifact_id}"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
