"""
Kernel Server - FastAPI wrapper for Belief Kernel
==================================================

Minimal server that exposes the Belief Kernel via REST API.
Compatible with journalist-workbench.html frontend.

NO adhoc rules. NO complex routing. Just beliefs and LLM reasoning.
"""

import os
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

from belief_kernel import BeliefKernel, Belief

# =============================================================================
# PYDANTIC MODELS (API contracts)
# =============================================================================

class EntityInput(BaseModel):
    id: Optional[str] = None
    canonical_name: str
    entity_type: str

class ClaimInput(BaseModel):
    id: Optional[str] = None
    text: str
    source: str = "Unknown"
    pub_time: Optional[str] = None
    entities: List[EntityInput] = []

class CreateEventInput(BaseModel):
    title: str = "New Investigation"


# =============================================================================
# INVESTIGATION STATE (wraps kernel)
# =============================================================================

@dataclass
class Investigation:
    """An investigation backed by a Belief Kernel."""
    id: str
    title: str
    kernel: BeliefKernel = field(default_factory=BeliefKernel)
    prose_cache: Optional[str] = None
    prose_coherence: float = 0.0  # Coherence when prose was generated

    def to_dict(self) -> Dict:
        """Convert to API response format."""
        kernel_data = self.kernel.to_api_response()
        return {
            "id": self.id,
            "title": self.title or kernel_data.get("title", "Developing Story"),
            "phase": kernel_data.get("phase", "emerging"),
            "entropy": kernel_data.get("entropy", 1.0),
            "coherence": kernel_data.get("coherence", 0.0),
            "claim_count": kernel_data.get("claim_count", 0),
            "belief_count": kernel_data.get("belief_count", 0),
            "beliefs": kernel_data.get("beliefs", []),
            "conflicts": kernel_data.get("conflicts", []),
            "relations": kernel_data.get("relations", {}),
            "llm_calls": kernel_data.get("llm_calls", 0),
            # Additional fields for compatibility - claims as objects
            "claims": [
                {
                    "id": f"claim_{i}",
                    "text": h["claim"],
                    "source": h["source"],
                    "relation": h["result"].get("relation", "COMPATIBLE"),
                    "reasoning": h["result"].get("reasoning", "")
                }
                for i, h in enumerate(self.kernel.history)
            ],
            "excluded_claims": [],  # Kernel doesn't exclude claims
            "topics": list(set(b.text.split()[0] for b in self.kernel.beliefs)) if self.kernel.beliefs else []
        }


# =============================================================================
# SERVER
# =============================================================================

app = FastAPI(title="Belief Kernel Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Store investigations in memory
investigations: Dict[str, Investigation] = {}

# LLM client
llm = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))


@app.get("/api/events")
async def list_investigations():
    """List all investigations."""
    return [
        {
            'id': inv.id,
            'title': inv.title,
            'claim_count': len(inv.kernel.history),
            'belief_count': len(inv.kernel.beliefs),
            'phase': inv.kernel._get_phase(),
            'entropy': inv.kernel.compute_entropy(),
            'coherence': inv.kernel.compute_coherence()
        }
        for inv in investigations.values()
    ]


@app.post("/api/events")
async def create_investigation(data: CreateEventInput):
    """Create a new investigation."""
    inv_id = str(uuid.uuid4())[:8]
    inv = Investigation(id=inv_id, title=data.title)
    investigations[inv_id] = inv
    return inv.to_dict()


@app.get("/api/events/{event_id}")
async def get_investigation(event_id: str):
    """Get investigation state."""
    inv = investigations.get(event_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return inv.to_dict()


@app.post("/api/events/{event_id}/claims")
async def add_claim(event_id: str, claim: ClaimInput):
    """Process a claim through the belief kernel."""
    inv = investigations.get(event_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")

    # Process through kernel
    result = await inv.kernel.process(
        claim=claim.text,
        source=claim.source,
        llm=llm
    )

    return {
        "status": "processed",
        "relation": result.get("relation"),
        "reasoning": result.get("reasoning"),
        "belief_count": len(inv.kernel.beliefs),
        "coherence": inv.kernel.compute_coherence(),
        "entropy": inv.kernel.compute_entropy()
    }


@app.get("/api/events/{event_id}/narrative")
async def get_narrative(event_id: str):
    """
    Get prose narrative from current beliefs.

    Coherence Leap: Only regenerate prose when coherence has increased
    significantly since last generation. This prevents unnecessary LLM calls
    and ensures prose stability.
    """
    inv = investigations.get(event_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")

    current_coherence = inv.kernel.compute_coherence()
    current_entropy = inv.kernel.compute_entropy()

    # Coherence leap threshold: regenerate if coherence increased by 0.1+
    # or if we don't have prose yet
    COHERENCE_LEAP_THRESHOLD = 0.1

    should_regenerate = (
        inv.prose_cache is None or
        current_coherence - inv.prose_coherence >= COHERENCE_LEAP_THRESHOLD
    )

    if should_regenerate:
        inv.prose_cache = await inv.kernel.generate_prose(llm)
        inv.prose_coherence = current_coherence

    return {
        "narrative": inv.prose_cache,
        "coherence": current_coherence,
        "entropy": current_entropy,
        "phase": inv.kernel._get_phase(),
        "belief_count": len(inv.kernel.beliefs),
        "regenerated": should_regenerate,
        "last_coherence": inv.prose_coherence
    }


@app.get("/api/events/{event_id}/beliefs")
async def get_beliefs(event_id: str):
    """Get just the beliefs (for debugging)."""
    inv = investigations.get(event_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")

    return {
        "beliefs": [
            {
                "text": b.text,
                "sources": b.sources,
                "source_count": len(b.sources),
                "supersedes": b.supersedes
            }
            for b in inv.kernel.beliefs
        ],
        "conflicts": inv.kernel.conflicts,
        "coherence": inv.kernel.compute_coherence(),
        "entropy": inv.kernel.compute_entropy()
    }


@app.get("/api/events/{event_id}/history")
async def get_history(event_id: str):
    """Get processing history (for debugging)."""
    inv = investigations.get(event_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")

    return {
        "history": inv.kernel.history,
        "summary": inv.kernel.summary()
    }


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == '__main__':
    import uvicorn
    print("=" * 60)
    print("BELIEF KERNEL SERVER")
    print("=" * 60)
    print("\nCore approach:")
    print("  - Domain knowledge in prompt, not code")
    print("  - One LLM call per claim")
    print("  - NO topics, NO clustering, NO thresholds")
    print("  - Coherence leap for prose regeneration")
    print("\nEndpoints:")
    print("  GET  /api/events           - List investigations")
    print("  POST /api/events           - Create investigation")
    print("  GET  /api/events/{id}      - Get state")
    print("  POST /api/events/{id}/claims - Add claim")
    print("  GET  /api/events/{id}/narrative - Get prose")
    print("\nStarting on port 5050...")
    uvicorn.run(app, host='0.0.0.0', port=5050)
