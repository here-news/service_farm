"""
REEE Simulator - Deterministic test harness with streaming visualization.

Creates a synthetic claim dataset that demonstrates REEE mechanics:
1. Claims arrive one-by-one
2. Each claim is semantically explained (what it's about)
3. Surface formation is explained (why claims cluster)
4. Event emergence is explained (why surfaces merge)

Uses small LLM calls for semantic gisting.

Usage:
    python -m debug.reee_simulator
    # Then open http://localhost:8081/simulate in browser
"""

import os
import sys
import json
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI

# ============================================================================
# SIMULATED CLAIM DATASET
# ============================================================================

# A deterministic scenario: Hong Kong press freedom case
SIMULATED_CLAIMS = [
    # Wave 1: Initial arrest news
    {
        "id": "sim_c001",
        "text": "Jimmy Lai, founder of Apple Daily, was arrested by Hong Kong police under the national security law.",
        "entities": ["Jimmy Lai", "Apple Daily", "Hong Kong"],
        "source": "reuters.com",
        "event_time": "2024-12-10T08:00:00Z",
    },
    {
        "id": "sim_c002",
        "text": "Hong Kong authorities detained media tycoon Jimmy Lai on charges of collusion with foreign forces.",
        "entities": ["Jimmy Lai", "Hong Kong"],
        "source": "bbc.com",
        "event_time": "2024-12-10T08:30:00Z",
    },
    {
        "id": "sim_c003",
        "text": "The arrest of Jimmy Lai marks another blow to press freedom in Hong Kong.",
        "entities": ["Jimmy Lai", "Hong Kong"],
        "source": "nytimes.com",
        "event_time": "2024-12-10T09:00:00Z",
    },

    # Wave 2: International response
    {
        "id": "sim_c004",
        "text": "US Secretary of State condemned the arrest of Jimmy Lai as an attack on freedom of the press.",
        "entities": ["Jimmy Lai", "United States"],
        "source": "state.gov",
        "event_time": "2024-12-10T14:00:00Z",
    },
    {
        "id": "sim_c005",
        "text": "The UK government expressed deep concern over Jimmy Lai's detention in Hong Kong.",
        "entities": ["Jimmy Lai", "United Kingdom", "Hong Kong"],
        "source": "gov.uk",
        "event_time": "2024-12-10T15:00:00Z",
    },

    # Wave 3: Related but different - Apple Daily closure
    {
        "id": "sim_c006",
        "text": "Apple Daily newspaper announced it will cease publication following asset freeze.",
        "entities": ["Apple Daily", "Hong Kong"],
        "source": "scmp.com",
        "event_time": "2024-12-11T06:00:00Z",
    },
    {
        "id": "sim_c007",
        "text": "Staff at Apple Daily printed one million copies for final edition as paper shuts down.",
        "entities": ["Apple Daily", "Hong Kong"],
        "source": "guardian.com",
        "event_time": "2024-12-11T10:00:00Z",
    },

    # Wave 4: Separate event - China response
    {
        "id": "sim_c008",
        "text": "China's Foreign Ministry spokesperson said Hong Kong affairs are internal matters.",
        "entities": ["China", "Hong Kong"],
        "source": "xinhua.net",
        "event_time": "2024-12-10T16:00:00Z",
    },
    {
        "id": "sim_c009",
        "text": "Beijing criticized Western interference in Hong Kong's judicial process.",
        "entities": ["Beijing", "Hong Kong"],
        "source": "globaltimes.cn",
        "event_time": "2024-12-10T17:00:00Z",
    },

    # Wave 5: Trial begins (later event)
    {
        "id": "sim_c010",
        "text": "Jimmy Lai's national security trial began in Hong Kong's High Court.",
        "entities": ["Jimmy Lai", "Hong Kong"],
        "source": "reuters.com",
        "event_time": "2024-12-18T09:00:00Z",
    },
    {
        "id": "sim_c011",
        "text": "Prosecutors allege Jimmy Lai conspired with foreign powers to impose sanctions on Hong Kong.",
        "entities": ["Jimmy Lai", "Hong Kong"],
        "source": "bbc.com",
        "event_time": "2024-12-18T10:00:00Z",
    },
    {
        "id": "sim_c012",
        "text": "Jimmy Lai pleaded not guilty to all charges at the start of his trial.",
        "entities": ["Jimmy Lai", "Hong Kong"],
        "source": "ap.com",
        "event_time": "2024-12-18T11:00:00Z",
    },
]


# ============================================================================
# REEE TYPES (simplified for simulation)
# ============================================================================

@dataclass
class SimClaim:
    id: str
    text: str
    entities: Set[str]
    source: str
    event_time: datetime
    embedding: Optional[List[float]] = None
    gist: Optional[str] = None  # LLM-generated summary


@dataclass
class SimSurface:
    id: str
    claim_ids: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    gist: Optional[str] = None  # What this surface is about
    formation_reason: Optional[str] = None  # Why claims clustered


@dataclass
class SimEvent:
    id: str
    surface_ids: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    gist: Optional[str] = None  # What this event is about
    formation_reason: Optional[str] = None  # Why surfaces merged


# ============================================================================
# STREAMING EVENTS
# ============================================================================

@dataclass
class StreamEvent:
    """Event sent to the visualization."""
    type: str  # claim_arrived, surface_formed, surface_attached, event_formed, etc.
    timestamp: str
    data: Dict[str, Any]
    explanation: str  # Human-readable explanation


# ============================================================================
# REEE SIMULATOR
# ============================================================================

class REEESimulator:
    """
    Simulates REEE processing with step-by-step streaming.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.openai = AsyncOpenAI() if use_llm else None

        # State
        self.claims: Dict[str, SimClaim] = {}
        self.surfaces: Dict[str, SimSurface] = {}
        self.events: Dict[str, SimEvent] = {}

        # Indices
        self.entity_to_claims: Dict[str, Set[str]] = defaultdict(set)
        self.claim_to_surface: Dict[str, str] = {}
        self.surface_to_event: Dict[str, str] = {}

        # Counters
        self.surface_counter = 0
        self.event_counter = 0

    async def gist_claim(self, claim: SimClaim) -> str:
        """Generate a one-line gist of what the claim is about."""
        if not self.use_llm:
            return f"Claim about {', '.join(list(claim.entities)[:2])}"

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"In 10 words or less, what is this news claim about?\n\n{claim.text}"
                }],
                max_tokens=30,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Claim about {', '.join(list(claim.entities)[:2])}"

    async def gist_surface(self, surface: SimSurface, claims: List[SimClaim]) -> str:
        """Generate a gist of what the surface represents."""
        if not self.use_llm:
            return f"Surface about {', '.join(list(surface.entities)[:3])}"

        claim_texts = [c.text for c in claims[:3]]
        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"These claims form a coherent surface (cluster). In 15 words, what proposition do they jointly assert?\n\n" +
                              "\n".join(f"- {t}" for t in claim_texts)
                }],
                max_tokens=40,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Surface about {', '.join(list(surface.entities)[:3])}"

    async def explain_attachment(self, claim: SimClaim, surface: SimSurface) -> str:
        """Explain why a claim attaches to a surface."""
        shared = claim.entities & surface.entities
        return f"Shares entities [{', '.join(shared)}] with existing surface"

    async def explain_surface_formation(self, claims: List[SimClaim]) -> str:
        """Explain why claims form a new surface."""
        if len(claims) == 1:
            return "First claim with this entity combination - new surface"

        shared = set.intersection(*[c.entities for c in claims])
        return f"Claims share motif [{', '.join(shared)}] - structural clustering"

    async def explain_event_formation(self, surfaces: List[SimSurface]) -> str:
        """Explain why surfaces form an event."""
        shared = set.intersection(*[s.entities for s in surfaces])
        return f"Surfaces share entities [{', '.join(shared)}] with compatible context"

    async def process_claim(self, claim_data: dict) -> List[StreamEvent]:
        """
        Process a single claim and yield stream events.

        Returns list of events to stream to visualization.
        """
        events = []

        # Parse claim
        claim = SimClaim(
            id=claim_data["id"],
            text=claim_data["text"],
            entities=set(claim_data["entities"]),
            source=claim_data["source"],
            event_time=datetime.fromisoformat(claim_data["event_time"].replace("Z", "+00:00"))
        )

        # Generate gist
        claim.gist = await self.gist_claim(claim)
        self.claims[claim.id] = claim

        # Update entity index
        for entity in claim.entities:
            self.entity_to_claims[entity].add(claim.id)

        # Stream: Claim arrived
        events.append(StreamEvent(
            type="claim_arrived",
            timestamp=datetime.utcnow().isoformat(),
            data={
                "claim_id": claim.id,
                "text": claim.text,
                "entities": list(claim.entities),
                "source": claim.source,
                "gist": claim.gist,
            },
            explanation=f"📥 New claim: {claim.gist}"
        ))

        # Find potential surfaces to attach to
        # Look for surfaces with overlapping entities
        candidate_surfaces = []
        for entity in claim.entities:
            for other_claim_id in self.entity_to_claims[entity]:
                if other_claim_id != claim.id and other_claim_id in self.claim_to_surface:
                    surface_id = self.claim_to_surface[other_claim_id]
                    if surface_id not in [s[0] for s in candidate_surfaces]:
                        surface = self.surfaces[surface_id]
                        shared = claim.entities & surface.entities
                        if len(shared) >= 2:  # Motif requirement: k >= 2
                            candidate_surfaces.append((surface_id, len(shared)))

        if candidate_surfaces:
            # Attach to best matching surface
            candidate_surfaces.sort(key=lambda x: -x[1])
            best_surface_id = candidate_surfaces[0][0]
            surface = self.surfaces[best_surface_id]

            # Attach claim
            surface.claim_ids.add(claim.id)
            surface.entities.update(claim.entities)
            surface.sources.add(claim.source)
            self.claim_to_surface[claim.id] = best_surface_id

            # Update surface gist
            surface_claims = [self.claims[cid] for cid in surface.claim_ids]
            surface.gist = await self.gist_surface(surface, surface_claims)

            # Stream: Claim attached to surface
            attachment_reason = await self.explain_attachment(claim, surface)
            events.append(StreamEvent(
                type="claim_attached",
                timestamp=datetime.utcnow().isoformat(),
                data={
                    "claim_id": claim.id,
                    "surface_id": best_surface_id,
                    "shared_entities": list(claim.entities & surface.entities),
                    "surface_claim_count": len(surface.claim_ids),
                    "surface_gist": surface.gist,
                },
                explanation=f"🔗 Attached to {best_surface_id}: {attachment_reason}"
            ))

        else:
            # Form new surface
            self.surface_counter += 1
            surface_id = f"S{self.surface_counter:03d}"

            surface = SimSurface(
                id=surface_id,
                claim_ids={claim.id},
                entities=claim.entities.copy(),
                sources={claim.source},
            )
            surface.gist = await self.gist_surface(surface, [claim])
            surface.formation_reason = await self.explain_surface_formation([claim])

            self.surfaces[surface_id] = surface
            self.claim_to_surface[claim.id] = surface_id

            # Stream: New surface formed
            events.append(StreamEvent(
                type="surface_formed",
                timestamp=datetime.utcnow().isoformat(),
                data={
                    "surface_id": surface_id,
                    "claim_ids": list(surface.claim_ids),
                    "entities": list(surface.entities),
                    "gist": surface.gist,
                },
                explanation=f"✨ New surface {surface_id}: {surface.formation_reason}"
            ))

        # Check for event formation/updates
        await self._check_event_formation(events)

        return events

    async def _check_event_formation(self, events: List[StreamEvent]):
        """Check if any surfaces should form or merge into events."""

        # For each surface not yet in an event, check if it can form one
        unassigned_surfaces = [
            sid for sid in self.surfaces
            if sid not in self.surface_to_event
        ]

        if len(unassigned_surfaces) < 2:
            return

        # Check pairs of unassigned surfaces for compatibility
        for i, s1_id in enumerate(unassigned_surfaces):
            for s2_id in unassigned_surfaces[i+1:]:
                s1 = self.surfaces[s1_id]
                s2 = self.surfaces[s2_id]

                # Check entity overlap (k >= 2)
                shared = s1.entities & s2.entities
                if len(shared) >= 2:
                    # Form event
                    self.event_counter += 1
                    event_id = f"E{self.event_counter:03d}"

                    event = SimEvent(
                        id=event_id,
                        surface_ids={s1_id, s2_id},
                        entities=s1.entities | s2.entities,
                    )

                    surface_list = [s1, s2]
                    event.formation_reason = await self.explain_event_formation(surface_list)

                    # Generate event gist
                    all_claims = []
                    for sid in event.surface_ids:
                        for cid in self.surfaces[sid].claim_ids:
                            all_claims.append(self.claims[cid])

                    if self.use_llm:
                        try:
                            texts = [c.text for c in all_claims[:5]]
                            response = await self.openai.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{
                                    "role": "user",
                                    "content": f"These claims describe an event. In 20 words, what happened?\n\n" +
                                              "\n".join(f"- {t}" for t in texts)
                                }],
                                max_tokens=50,
                                temperature=0
                            )
                            event.gist = response.choices[0].message.content.strip()
                        except:
                            event.gist = f"Event involving {', '.join(list(shared)[:3])}"
                    else:
                        event.gist = f"Event involving {', '.join(list(shared)[:3])}"

                    self.events[event_id] = event
                    self.surface_to_event[s1_id] = event_id
                    self.surface_to_event[s2_id] = event_id

                    # Stream: Event formed
                    events.append(StreamEvent(
                        type="event_formed",
                        timestamp=datetime.utcnow().isoformat(),
                        data={
                            "event_id": event_id,
                            "surface_ids": list(event.surface_ids),
                            "entities": list(event.entities),
                            "gist": event.gist,
                            "claim_count": sum(len(self.surfaces[sid].claim_ids) for sid in event.surface_ids),
                        },
                        explanation=f"🌟 Event {event_id} emerged: {event.formation_reason}"
                    ))

                    return  # One event at a time for clarity

    def get_state(self) -> Dict:
        """Get current state for visualization."""
        return {
            "claims": {cid: {
                "id": c.id,
                "text": c.text[:100],
                "entities": list(c.entities),
                "gist": c.gist,
                "surface_id": self.claim_to_surface.get(cid),
            } for cid, c in self.claims.items()},
            "surfaces": {sid: {
                "id": s.id,
                "claim_ids": list(s.claim_ids),
                "entities": list(s.entities),
                "gist": s.gist,
                "event_id": self.surface_to_event.get(sid),
            } for sid, s in self.surfaces.items()},
            "events": {eid: {
                "id": e.id,
                "surface_ids": list(e.surface_ids),
                "entities": list(e.entities),
                "gist": e.gist,
            } for eid, e in self.events.items()},
            "stats": {
                "claims": len(self.claims),
                "surfaces": len(self.surfaces),
                "events": len(self.events),
            }
        }

    def reset(self):
        """Reset simulator state."""
        self.claims.clear()
        self.surfaces.clear()
        self.events.clear()
        self.entity_to_claims.clear()
        self.claim_to_surface.clear()
        self.surface_to_event.clear()
        self.surface_counter = 0
        self.event_counter = 0


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="REEE Simulator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

simulator = REEESimulator(use_llm=True)


@app.get("/simulate")
async def simulate_page():
    """Serve the simulation visualization page."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>REEE Simulator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Monaco', monospace;
            background: #0a0a12;
            color: #e0e0e0;
            display: flex;
            height: 100vh;
        }

        /* Left panel - Event log */
        .log-panel {
            width: 400px;
            background: #0f0f18;
            border-right: 1px solid #2a2a4e;
            display: flex;
            flex-direction: column;
        }
        .log-header {
            padding: 15px;
            border-bottom: 1px solid #2a2a4e;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .log-header h2 { color: #7b8cff; font-size: 1em; }
        .log-content {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        .log-entry {
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 6px;
            border-left: 3px solid #333;
            background: #151520;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .log-entry.claim_arrived { border-left-color: #4a9eff; }
        .log-entry.claim_attached { border-left-color: #4aff9e; }
        .log-entry.surface_formed { border-left-color: #ff9e4a; }
        .log-entry.event_formed { border-left-color: #ff4a9e; }
        .log-type {
            font-size: 0.7em;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .log-explanation {
            font-size: 0.9em;
            line-height: 1.4;
        }
        .log-data {
            font-size: 0.75em;
            color: #666;
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid #222;
        }

        /* Right panel - Visualization */
        .viz-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .viz-header {
            padding: 15px;
            border-bottom: 1px solid #2a2a4e;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4a9eff;
        }
        .stat-label {
            font-size: 0.7em;
            color: #666;
            text-transform: uppercase;
        }
        .controls {
            margin-left: auto;
            display: flex;
            gap: 10px;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.85em;
        }
        .btn-start {
            background: #4a9eff;
            color: white;
        }
        .btn-reset {
            background: #333;
            color: #ccc;
        }
        .btn-start:hover { background: #3a8eef; }
        .btn-reset:hover { background: #444; }

        /* Canvas area */
        .viz-canvas {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        #topology {
            width: 100%;
            height: 100%;
        }

        /* Floating gist panel */
        .gist-panel {
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(15, 15, 25, 0.95);
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
        }
        .gist-title {
            color: #ff9e4a;
            font-size: 0.8em;
            margin-bottom: 8px;
        }
        .gist-text {
            font-size: 0.9em;
            line-height: 1.5;
        }

        /* Speed control */
        .speed-control {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
            font-size: 0.8em;
        }
        .speed-control input {
            width: 100px;
        }
    </style>
</head>
<body>
    <div class="log-panel">
        <div class="log-header">
            <h2>📜 Event Log</h2>
            <span id="log-count">0 events</span>
        </div>
        <div class="log-content" id="log"></div>
    </div>

    <div class="viz-panel">
        <div class="viz-header">
            <div class="stat">
                <div class="stat-value" id="claim-count">0</div>
                <div class="stat-label">Claims</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="surface-count">0</div>
                <div class="stat-label">Surfaces</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="event-count">0</div>
                <div class="stat-label">Events</div>
            </div>

            <div class="speed-control">
                <label>Speed:</label>
                <input type="range" id="speed" min="500" max="5000" value="2000">
                <span id="speed-label">2.0s</span>
            </div>

            <div class="controls">
                <button class="btn-start" id="btn-start">▶ Start Simulation</button>
                <button class="btn-reset" id="btn-reset">↺ Reset</button>
            </div>
        </div>

        <div class="viz-canvas">
            <canvas id="topology"></canvas>
            <div class="gist-panel" id="gist-panel" style="display:none;">
                <div class="gist-title" id="gist-title">Selected Item</div>
                <div class="gist-text" id="gist-text">Click on a node to see details</div>
            </div>
        </div>
    </div>

    <script>
        const log = document.getElementById('log');
        const canvas = document.getElementById('topology');
        const ctx = canvas.getContext('2d');
        const gistPanel = document.getElementById('gist-panel');
        const speedSlider = document.getElementById('speed');
        const speedLabel = document.getElementById('speed-label');

        let eventSource = null;
        let nodes = [];  // {id, type, x, y, data}
        let edges = [];  // {source, target, type}

        // Layout parameters
        const CLAIM_RADIUS = 8;
        const SURFACE_RADIUS = 20;
        const EVENT_RADIUS = 35;

        speedSlider.addEventListener('input', () => {
            speedLabel.textContent = (speedSlider.value / 1000).toFixed(1) + 's';
        });

        function resizeCanvas() {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            draw();
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function addNode(id, type, data) {
            // Position based on type
            const w = canvas.width;
            const h = canvas.height;

            let x, y;
            if (type === 'claim') {
                // Claims on the left
                x = 50 + Math.random() * 100;
                y = 50 + nodes.filter(n => n.type === 'claim').length * 40;
            } else if (type === 'surface') {
                // Surfaces in the middle
                x = w * 0.4 + Math.random() * 100;
                y = 100 + nodes.filter(n => n.type === 'surface').length * 80;
            } else if (type === 'event') {
                // Events on the right
                x = w * 0.75;
                y = 150 + nodes.filter(n => n.type === 'event').length * 120;
            }

            nodes.push({ id, type, x, y, data });
        }

        function addEdge(sourceId, targetId, type) {
            edges.push({ source: sourceId, target: targetId, type });
        }

        function findNode(id) {
            return nodes.find(n => n.id === id);
        }

        function draw() {
            ctx.fillStyle = '#0a0a12';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw column labels
            ctx.fillStyle = '#333';
            ctx.font = '12px monospace';
            ctx.fillText('CLAIMS (L1)', 50, 30);
            ctx.fillText('SURFACES (L2)', canvas.width * 0.4, 30);
            ctx.fillText('EVENTS (L3)', canvas.width * 0.75, 30);

            // Draw edges
            edges.forEach(edge => {
                const source = findNode(edge.source);
                const target = findNode(edge.target);
                if (source && target) {
                    ctx.beginPath();
                    ctx.moveTo(source.x, source.y);
                    ctx.lineTo(target.x, target.y);
                    ctx.strokeStyle = edge.type === 'contains' ? 'rgba(74, 158, 255, 0.3)' : 'rgba(255, 158, 74, 0.3)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            });

            // Draw nodes
            nodes.forEach(node => {
                let radius, color;
                if (node.type === 'claim') {
                    radius = CLAIM_RADIUS;
                    color = '#4a9eff';
                } else if (node.type === 'surface') {
                    radius = SURFACE_RADIUS;
                    color = '#ff9e4a';
                } else {
                    radius = EVENT_RADIUS;
                    color = '#ff4a9e';
                }

                ctx.beginPath();
                ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#fff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(node.id, node.x, node.y + radius + 15);
            });
        }

        function addLogEntry(event) {
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + event.type;
            entry.innerHTML = `
                <div class="log-type">${event.type.replace('_', ' ')}</div>
                <div class="log-explanation">${event.explanation}</div>
                <div class="log-data">${JSON.stringify(event.data).slice(0, 100)}...</div>
            `;
            log.insertBefore(entry, log.firstChild);
            document.getElementById('log-count').textContent = log.children.length + ' events';
        }

        function updateStats(state) {
            document.getElementById('claim-count').textContent = state.stats.claims;
            document.getElementById('surface-count').textContent = state.stats.surfaces;
            document.getElementById('event-count').textContent = state.stats.events;
        }

        function handleEvent(event) {
            addLogEntry(event);

            if (event.type === 'claim_arrived') {
                addNode(event.data.claim_id, 'claim', event.data);
            } else if (event.type === 'surface_formed') {
                addNode(event.data.surface_id, 'surface', event.data);
                event.data.claim_ids.forEach(cid => {
                    addEdge(cid, event.data.surface_id, 'contains');
                });
            } else if (event.type === 'claim_attached') {
                addEdge(event.data.claim_id, event.data.surface_id, 'contains');
            } else if (event.type === 'event_formed') {
                addNode(event.data.event_id, 'event', event.data);
                event.data.surface_ids.forEach(sid => {
                    addEdge(sid, event.data.event_id, 'includes');
                });
            }

            draw();
        }

        document.getElementById('btn-start').addEventListener('click', async () => {
            const speed = parseInt(speedSlider.value);
            const btn = document.getElementById('btn-start');
            btn.disabled = true;
            btn.textContent = '⏳ Running...';

            // Start streaming simulation
            eventSource = new EventSource(`/api/simulate/stream?delay=${speed}`);

            eventSource.onmessage = (e) => {
                const data = JSON.parse(e.data);

                if (data.type === 'state') {
                    updateStats(data.state);
                } else if (data.type === 'complete') {
                    eventSource.close();
                    btn.disabled = false;
                    btn.textContent = '✓ Complete';
                } else {
                    handleEvent(data);
                }
            };

            eventSource.onerror = () => {
                eventSource.close();
                btn.disabled = false;
                btn.textContent = '▶ Start Simulation';
            };
        });

        document.getElementById('btn-reset').addEventListener('click', async () => {
            if (eventSource) eventSource.close();

            await fetch('/api/simulate/reset', { method: 'POST' });

            nodes = [];
            edges = [];
            log.innerHTML = '';
            document.getElementById('log-count').textContent = '0 events';
            document.getElementById('claim-count').textContent = '0';
            document.getElementById('surface-count').textContent = '0';
            document.getElementById('event-count').textContent = '0';
            document.getElementById('btn-start').textContent = '▶ Start Simulation';
            document.getElementById('btn-start').disabled = false;

            draw();
        });

        // Click handling for details
        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            for (const node of nodes) {
                const r = node.type === 'claim' ? CLAIM_RADIUS :
                          node.type === 'surface' ? SURFACE_RADIUS : EVENT_RADIUS;
                const dist = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);

                if (dist < r) {
                    gistPanel.style.display = 'block';
                    document.getElementById('gist-title').textContent =
                        `${node.type.toUpperCase()}: ${node.id}`;
                    document.getElementById('gist-text').textContent =
                        node.data.gist || node.data.text || JSON.stringify(node.data);
                    return;
                }
            }

            gistPanel.style.display = 'none';
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/api/simulate/stream")
async def stream_simulation(delay: int = Query(2000, ge=500, le=10000)):
    """Stream the simulation step by step."""

    async def event_generator():
        simulator.reset()

        for claim_data in SIMULATED_CLAIMS:
            # Process claim
            events = await simulator.process_claim(claim_data)

            # Stream each event
            for event in events:
                yield f"data: {json.dumps(asdict(event))}\n\n"
                await asyncio.sleep(0.3)  # Small delay between events

            # Stream current state
            state = simulator.get_state()
            yield f"data: {json.dumps({'type': 'state', 'state': state})}\n\n"

            # Delay before next claim
            await asyncio.sleep(delay / 1000)

        # Signal completion
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/api/simulate/reset")
async def reset_simulation():
    """Reset the simulator state."""
    simulator.reset()
    return {"status": "reset"}


@app.get("/api/simulate/state")
async def get_state():
    """Get current simulation state."""
    return simulator.get_state()


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "reee-simulator"}


if __name__ == "__main__":
    import uvicorn
    print("🧪 REEE Simulator running on http://localhost:8081")
    print("   Open http://localhost:8081/simulate to start")
    uvicorn.run(app, host="0.0.0.0", port=8081)
