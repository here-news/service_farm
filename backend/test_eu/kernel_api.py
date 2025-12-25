"""
Kernel API - Layer 5
====================

FastAPI endpoints for the enriched belief kernel.
Provides UI-ready data for the kernel-ui.html prototype.

Endpoints:
- POST /api/kernel/process - Process claims through kernel
- GET /api/kernel/topology/{event_id} - Get enriched topology
- GET /api/kernel/prose/{event_id} - Generate/regenerate prose
- GET /api/kernel/trajectory/{event_id} - Get entropy trajectory
"""

import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

from kernel_enriched import EnrichedKernel
from kernel_prose import generate_prose, generate_prose_from_dict


# ============================================================================
# Request/Response Models
# ============================================================================

class ClaimInput(BaseModel):
    """Single claim for processing."""
    id: Optional[str] = None
    text: str
    source: str
    event_time: Optional[str] = None
    modality: Optional[str] = "observation"
    entities: Optional[List[Dict]] = None


class ProcessRequest(BaseModel):
    """Request to process claims through kernel."""
    event_id: str
    claims: List[ClaimInput]


class BeliefResponse(BaseModel):
    """Enriched belief for UI."""
    id: str
    text: str
    sources: List[str]
    claim_ids: List[str]
    entity_ids: List[str]
    certainty: float
    category: Optional[str]
    supersedes_id: Optional[str]
    supersedes_text: Optional[str]


class ConflictResponse(BaseModel):
    """Conflict for UI."""
    id: str
    new_claim: str
    existing_belief_id: Optional[str]
    existing_belief_text: Optional[str]
    topic: Optional[str]
    reasoning: str


class EntropyPointResponse(BaseModel):
    """Entropy trajectory point."""
    claim_index: int
    entropy: float
    coherence: float
    belief_count: int
    conflict_count: int


class TopologyResponse(BaseModel):
    """Full enriched topology for UI."""
    event_id: str
    beliefs: List[BeliefResponse]
    conflicts: List[ConflictResponse]
    entropy_trajectory: List[EntropyPointResponse]
    metrics: Dict[str, Any]
    relations: Dict[str, int]
    entity_lookup: Dict[str, str]


class ProseResponse(BaseModel):
    """Generated prose narrative."""
    event_id: str
    prose: str
    model: str
    generated_at: str


# ============================================================================
# In-Memory State (for prototype; production would use Redis/DB)
# ============================================================================

# Store kernels by event_id
_kernels: Dict[str, EnrichedKernel] = {}
_topologies: Dict[str, Dict] = {}


def get_or_create_kernel(event_id: str) -> EnrichedKernel:
    """Get existing kernel or create new one."""
    if event_id not in _kernels:
        _kernels[event_id] = EnrichedKernel()
    return _kernels[event_id]


def get_llm() -> AsyncOpenAI:
    """Get OpenAI client."""
    return AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/api/kernel", tags=["kernel"])


@router.post("/process", response_model=Dict[str, Any])
async def process_claims(request: ProcessRequest):
    """
    Process claims through the enriched kernel.
    Returns processing summary and updated topology.
    """
    kernel = get_or_create_kernel(request.event_id)
    llm = get_llm()

    results = []
    for claim in request.claims:
        # Convert to object-like dict for kernel
        claim_obj = type('Claim', (), {
            'id': claim.id,
            'text': claim.text,
            'event_time': datetime.fromisoformat(claim.event_time) if claim.event_time else None,
            'modality': claim.modality,
            'entities': [
                type('Entity', (), {'id': e.get('id'), 'canonical_name': e.get('name')})()
                for e in (claim.entities or [])
            ]
        })()

        result = await kernel.process(claim_obj, claim.source, llm)
        results.append({
            'claim_id': claim.id,
            'relation': result.get('relation'),
            'reasoning': result.get('reasoning')
        })

    # Save topology
    topology_dict = kernel.to_dict()
    _topologies[request.event_id] = topology_dict

    return {
        'event_id': request.event_id,
        'claims_processed': len(request.claims),
        'results': results,
        'metrics': topology_dict['metrics']
    }


@router.get("/topology/{event_id}", response_model=TopologyResponse)
async def get_topology(event_id: str):
    """
    Get enriched topology for an event.
    Returns UI-ready data with belief IDs, entity links, certainty scores.
    """
    # Check in-memory first
    if event_id in _topologies:
        t = _topologies[event_id]
    elif event_id in _kernels:
        t = _kernels[event_id].to_dict()
        _topologies[event_id] = t
    else:
        # Try to load from file (for testing)
        try:
            path = f'/app/test_eu/results/topology_{event_id}.json'
            with open(path) as f:
                t = json.load(f)
        except FileNotFoundError:
            raise HTTPException(404, f"Topology not found for event: {event_id}")

    return TopologyResponse(
        event_id=event_id,
        beliefs=[BeliefResponse(**b) for b in t.get('beliefs', [])],
        conflicts=[ConflictResponse(**c) for c in t.get('conflicts', [])],
        entropy_trajectory=[
            EntropyPointResponse(**e) for e in t.get('entropy_trajectory', [])
        ],
        metrics=t.get('metrics', {}),
        relations=t.get('relations', {}),
        entity_lookup=t.get('entity_lookup', {})
    )


@router.get("/prose/{event_id}", response_model=ProseResponse)
async def get_prose(event_id: str, model: str = "gpt-5.2"):
    """
    Generate or regenerate prose from topology.
    Can specify different models to compare output.
    """
    # Get topology
    if event_id in _topologies:
        topology_dict = _topologies[event_id]
    elif event_id in _kernels:
        topology_dict = _kernels[event_id].to_dict()
    else:
        try:
            path = f'/app/test_eu/results/topology_{event_id}.json'
            with open(path) as f:
                topology_dict = json.load(f)
        except FileNotFoundError:
            raise HTTPException(404, f"Topology not found for event: {event_id}")

    # Generate prose
    llm = get_llm()
    prose = await generate_prose_from_dict(topology_dict, llm, model)

    return ProseResponse(
        event_id=event_id,
        prose=prose,
        model=model,
        generated_at=datetime.now().isoformat()
    )


@router.get("/trajectory/{event_id}")
async def get_trajectory(event_id: str):
    """
    Get entropy trajectory for convergence visualization.
    Shows how entropy/coherence evolved as claims were processed.
    """
    # Get topology
    if event_id in _topologies:
        t = _topologies[event_id]
    elif event_id in _kernels:
        t = _kernels[event_id].to_dict()
    else:
        try:
            path = f'/app/test_eu/results/topology_{event_id}.json'
            with open(path) as f:
                t = json.load(f)
        except FileNotFoundError:
            raise HTTPException(404, f"Topology not found for event: {event_id}")

    trajectory = t.get('entropy_trajectory', [])

    return {
        'event_id': event_id,
        'trajectory': trajectory,
        'final_metrics': t.get('metrics', {})
    }


@router.delete("/reset/{event_id}")
async def reset_kernel(event_id: str):
    """Reset kernel state for an event (for testing)."""
    if event_id in _kernels:
        del _kernels[event_id]
    if event_id in _topologies:
        del _topologies[event_id]
    return {'status': 'reset', 'event_id': event_id}


# ============================================================================
# Integration helper (for main.py)
# ============================================================================

def include_kernel_router(app):
    """Add kernel routes to FastAPI app."""
    app.include_router(router)


# ============================================================================
# Standalone test server
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="Belief Kernel API")
    app.include_router(router)

    @app.get("/")
    async def root():
        return {
            "service": "Belief Kernel API",
            "endpoints": [
                "POST /api/kernel/process",
                "GET /api/kernel/topology/{event_id}",
                "GET /api/kernel/prose/{event_id}",
                "GET /api/kernel/trajectory/{event_id}",
                "DELETE /api/kernel/reset/{event_id}"
            ]
        }

    uvicorn.run(app, host="0.0.0.0", port=8001)
