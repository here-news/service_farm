"""
Weave Topology Visualization Server

A standalone FastAPI server for visualizing surface → event weaving.
Runs inside the workers container on port 8080.

Supports TWO modes:
1. LIVE MODE: Real-time visualization of production topology (PostgreSQL + Neo4j)
2. TRACE MODE: Replay golden traces for testing/debugging REEE mechanics

Design goals (per epistemic requirements):
1. Show space filling - where new claims land, animate joins
2. Separate entropies - geometric dispersion D vs value uncertainty H
3. Show inter-cluster distance and binding - edge evidence
4. Make constraint scarcity visible - HUD with coverage stats
5. Debug interactions - click for details
6. Golden trace replay with step-by-step explanation

Usage:
    python -m debug.weave_viz
    # or via run_workers.py (auto-started)

Endpoints:
    GET /                    - D3 visualization page (live mode)
    GET /trace               - Golden trace replay page
    GET /api/snapshot        - JSON snapshot of current topology
    GET /api/surface/{id}    - Surface details with claims
    GET /api/traces          - List available golden traces
    GET /api/trace/{name}/stream - SSE stream for trace replay
    GET /api/health          - Health check
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import numpy as np

from fastapi import FastAPI, Query, Path as PathParam
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
from services.neo4j_service import Neo4jService
from repositories.surface_repository import SurfaceRepository
from repositories.event_repository import EventRepository
from models.domain.surface import Surface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weave Topology Visualization",
    description="D3 visualization of surface → event weaving with epistemic observables",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database config from environment
PG_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'db'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'phi_here'),
    'user': os.getenv('POSTGRES_USER', 'phi_user'),
    'password': os.getenv('POSTGRES_PASSWORD', ''),
}

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

# Static files directory
STATIC_DIR = Path(__file__).parent / 'static'

# Golden trace fixtures directory
TRACES_DIR = Path(__file__).parent.parent / 'reee' / 'tests' / 'fixtures'

# Connection pool (lazy init)
_db_pool: Optional[asyncpg.Pool] = None
_neo4j: Optional[Neo4jService] = None

# Import golden trace components (lazy to avoid import errors if not available)
def get_trace_kernel():
    """Lazy import of trace kernel to avoid circular imports."""
    try:
        from reee.tests.test_golden_trace import GoldenTrace, TraceKernel
        return GoldenTrace, TraceKernel
    except ImportError as e:
        logger.warning(f"Could not import trace kernel: {e}")
        return None, None


async def get_db_pool() -> asyncpg.Pool:
    """Get or create database pool."""
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(**PG_CONFIG, min_size=1, max_size=5)
    return _db_pool


async def get_neo4j() -> Neo4jService:
    """Get or create Neo4j service."""
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        await _neo4j.connect()
    return _neo4j


def compute_2d_projection(centroids: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
    """
    Compute 2D PCA projection of surface centroids for visualization.

    Returns dict of surface_id -> (x, y) coordinates.
    """
    if not centroids:
        return {}

    ids = list(centroids.keys())
    vectors = np.array([centroids[sid] for sid in ids])

    if len(vectors) < 2:
        return {ids[0]: (0.5, 0.5)} if ids else {}

    # Simple PCA (2 components)
    vectors_centered = vectors - vectors.mean(axis=0)
    try:
        # SVD for PCA
        U, S, Vt = np.linalg.svd(vectors_centered, full_matrices=False)
        coords_2d = U[:, :2] * S[:2]

        # Normalize to [0, 1] range
        coords_2d -= coords_2d.min(axis=0)
        max_range = coords_2d.max(axis=0)
        max_range[max_range == 0] = 1  # Avoid division by zero
        coords_2d /= max_range

        return {ids[i]: (float(coords_2d[i, 0]), float(coords_2d[i, 1]))
                for i in range(len(ids))}
    except Exception as e:
        logger.warning(f"PCA projection failed: {e}")
        # Fallback: random layout
        return {sid: (float(hash(sid) % 1000) / 1000, float(hash(sid[::-1]) % 1000) / 1000)
                for sid in ids}


async def compute_dispersion(surface_id: str, db_pool: asyncpg.Pool) -> Optional[float]:
    """
    Compute geometric dispersion D(surface) - embedding spread within surface.

    D = mean pairwise cosine distance between claim embeddings in this surface.
    High D = semantically mixed / maybe mis-scoped.
    """
    from pgvector.asyncpg import register_vector

    async with db_pool.acquire() as conn:
        await register_vector(conn)

        # Get claim embeddings for this surface
        rows = await conn.fetch("""
            SELECT ce.embedding
            FROM content.claim_surfaces cs
            JOIN core.claim_embeddings ce ON cs.claim_id = ce.claim_id
            WHERE cs.surface_id = $1
            LIMIT 50
        """, surface_id)

        if len(rows) < 2:
            return 0.0

        embeddings = []
        for r in rows:
            if r['embedding'] is not None:
                if hasattr(r['embedding'], 'tolist'):
                    embeddings.append(r['embedding'].tolist())
                else:
                    embeddings.append(list(r['embedding']))

        if len(embeddings) < 2:
            return 0.0

        # Compute mean pairwise cosine distance
        vectors = np.array(embeddings)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = vectors / norms

        # Pairwise cosine similarities
        similarities = normalized @ normalized.T

        # Mean distance (1 - similarity) excluding diagonal
        n = len(embeddings)
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += 1 - similarities[i, j]
                count += 1

        return round(total_dist / count, 3) if count > 0 else 0.0


async def get_topology_snapshot(
    limit: int = 100,
    min_support: float = 0.0,
    include_centroids: bool = True,
    include_dispersion: bool = True
) -> Dict[str, Any]:
    """
    Build a snapshot of the current surface → event topology.
    Uses proper repositories to load data.

    Returns:
        {
            meta: { snapshot_time, counts, constraint_availability, gate_histogram },
            nodes: [ surfaces with projection + dispersion, events ],
            links: [ aboutness edges with evidence, membership edges ]
        }
    """
    nodes = []
    links = []
    meta = {
        'snapshot_time': datetime.utcnow().isoformat(),
        'params': {'limit': limit, 'min_support': min_support},
        'counts': {},
        'constraint_availability': {},
        'gate_histogram': {}
    }

    db_pool = await get_db_pool()
    neo4j = await get_neo4j()

    surface_repo = SurfaceRepository(db_pool, neo4j)

    # ============================================================
    # 1. Load surfaces using repository pattern
    # ============================================================
    surface_results = await neo4j._execute_read("""
        MATCH (s:Surface)
        WHERE s.support >= $min_support
        RETURN s.id as id
        ORDER BY s.support DESC
        LIMIT $limit
    """, {'min_support': min_support, 'limit': limit})

    surface_ids = [r['id'] for r in surface_results if r['id']]
    surfaces = await surface_repo.get_by_ids(surface_ids)

    # Build centroid map for projection
    centroids = {s.id: s.centroid for s in surfaces if s.centroid}
    projections = compute_2d_projection(centroids)

    # Compute dispersion for surfaces (sample for performance)
    dispersions = {}
    if include_dispersion:
        sample_ids = surface_ids[:30]  # Only compute for top surfaces
        for sid in sample_ids:
            dispersions[sid] = await compute_dispersion(sid, db_pool)

    def safe_isoformat(dt):
        """Convert datetime to isoformat string, handling strings and None."""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        if hasattr(dt, 'isoformat'):
            return dt.isoformat()
        return str(dt)

    for surface in surfaces:
        proj = projections.get(surface.id, (0.5, 0.5))
        node = {
            'id': surface.id,
            'kind': 'surface',
            'support': round(surface.support, 2),
            'claim_count': len(surface.claim_ids),
            'source_count': len(surface.sources),
            'time_start': safe_isoformat(surface.time_start),
            'time_end': safe_isoformat(surface.time_end),
            'anchors': list(surface.anchor_entities)[:5],
            'has_centroid': surface.centroid is not None,
            # Projection coordinates for 2D layout
            'x': proj[0],
            'y': proj[1],
            # Geometric dispersion D(surface)
            'dispersion': dispersions.get(surface.id, None),
            # Value entropy H (placeholder - needs typed coverage)
            'value_entropy': None,
        }
        nodes.append(node)

    meta['counts']['surfaces'] = len(surfaces)

    # ============================================================
    # 2. Load events from Neo4j
    # ============================================================
    event_results = await neo4j._execute_read("""
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[:CONTAINS]->(s:Surface)
        WITH e, count(s) as surface_count, collect(s.id) as member_ids
        RETURN e.id as id,
               e.canonical_name as title,
               e.event_start as time_start,
               e.event_end as time_end,
               e.coherence as coherence,
               surface_count,
               member_ids
        ORDER BY surface_count DESC
        LIMIT $limit
    """, {'limit': limit})

    event_ids = set()
    for r in event_results:
        if not r['id']:
            continue
        event_ids.add(r['id'])

        # Compute event centroid from member surfaces for projection
        member_centroids = [centroids[mid] for mid in (r['member_ids'] or []) if mid in centroids]
        if member_centroids:
            event_centroid = np.mean(member_centroids, axis=0).tolist()
            if r['id'] not in centroids:
                centroids[r['id']] = event_centroid

        event_proj = projections.get(r['id'], (0.5, 0.5))
        # Use mean of member projections if available
        member_projs = [projections[mid] for mid in (r['member_ids'] or []) if mid in projections]
        if member_projs:
            event_proj = (
                sum(p[0] for p in member_projs) / len(member_projs),
                sum(p[1] for p in member_projs) / len(member_projs)
            )

        node = {
            'id': r['id'],
            'kind': 'event',
            'title': r['title'] or 'Untitled',
            'surface_count': r['surface_count'] or 0,
            'coherence': round(r['coherence'] or 0, 2),
            'time_start': safe_isoformat(r['time_start']),
            'time_end': safe_isoformat(r['time_end']),
            'x': event_proj[0],
            'y': event_proj[1],
            'member_ids': r['member_ids'] or [],
        }
        nodes.append(node)

    meta['counts']['events'] = len(event_ids)

    # ============================================================
    # 3. Get membership edges with evidence
    # ============================================================
    if event_ids:
        membership_results = await neo4j._execute_read("""
            MATCH (e:Event)-[r:CONTAINS]->(s:Surface)
            WHERE e.id IN $event_ids
            RETURN e.id as event_id, s.id as surface_id,
                   r.level as level, r.weight as weight,
                   r.signals_met as signals_met,
                   r.time_delta_days as time_delta
        """, {'event_ids': list(event_ids)})

        for r in membership_results:
            links.append({
                'source': r['event_id'],
                'target': r['surface_id'],
                'kind': 'membership',
                'level': r['level'] or 'core',
                'weight': float(r['weight'] or 1.0),
                # Edge evidence
                'evidence': {
                    'signals_met': r['signals_met'] or 0,
                    'time_delta_days': r['time_delta'],
                }
            })

    meta['counts']['membership_edges'] = len([l for l in links if l['kind'] == 'membership'])

    # ============================================================
    # 4. Get aboutness edges with evidence
    # ============================================================
    if surface_ids and include_centroids:
        from pgvector.asyncpg import register_vector

        async with db_pool.acquire() as conn:
            await register_vector(conn)

            # Get pairwise similarities with more detail
            aboutness_rows = await conn.fetch("""
                SELECT
                    s1.surface_id as source,
                    s2.surface_id as target,
                    1 - (s1.centroid <=> s2.centroid) as similarity,
                    s1.time_start as t1_start, s1.time_end as t1_end,
                    s2.time_start as t2_start, s2.time_end as t2_end
                FROM content.surface_centroids s1
                JOIN content.surface_centroids s2 ON s1.surface_id < s2.surface_id
                WHERE s1.surface_id = ANY($1) AND s2.surface_id = ANY($1)
                  AND 1 - (s1.centroid <=> s2.centroid) > 0.65
                ORDER BY similarity DESC
                LIMIT 300
            """, surface_ids)

            for row in aboutness_rows:
                # Compute signals
                signals = []
                semantic_sim = float(row['similarity'])
                if semantic_sim > 0.8:
                    signals.append('semantic')

                # Check temporal overlap
                t1_start, t1_end = row['t1_start'], row['t1_end']
                t2_start, t2_end = row['t2_start'], row['t2_end']
                time_overlap = False
                if t1_start and t2_start and t1_end and t2_end:
                    if t1_start <= t2_end and t2_start <= t1_end:
                        time_overlap = True
                        signals.append('temporal')

                links.append({
                    'source': row['source'],
                    'target': row['target'],
                    'kind': 'aboutness',
                    'weight': round(semantic_sim, 3),
                    # Edge evidence
                    'evidence': {
                        'semantic_sim': round(semantic_sim, 3),
                        'time_overlap': time_overlap,
                        'signals': signals,
                    }
                })

    meta['counts']['aboutness_edges'] = len([l for l in links if l['kind'] == 'aboutness'])

    # ============================================================
    # 5. Compute constraint availability metrics
    # ============================================================
    total_surfaces = meta['counts']['surfaces']
    if total_surfaces > 0:
        surfaces_with_time = len([n for n in nodes if n['kind'] == 'surface' and n.get('time_start')])
        surfaces_with_centroid = len([n for n in nodes if n['kind'] == 'surface' and n.get('has_centroid')])
        multi_source = len([n for n in nodes if n['kind'] == 'surface' and n.get('source_count', 0) > 1])

        meta['constraint_availability'] = {
            'time_coverage': round(surfaces_with_time / total_surfaces * 100, 1),
            'centroid_coverage': round(surfaces_with_centroid / total_surfaces * 100, 1),
            'multi_source': round(multi_source / total_surfaces * 100, 1),
            'typed_coverage': 0.0,  # TODO: implement when typed extraction ready
        }

        # Gate histogram - count edge signal patterns
        signal_counts = {'semantic_only': 0, 'temporal_only': 0, 'both': 0, 'none': 0}
        for link in links:
            if link['kind'] == 'aboutness':
                signals = link.get('evidence', {}).get('signals', [])
                if 'semantic' in signals and 'temporal' in signals:
                    signal_counts['both'] += 1
                elif 'semantic' in signals:
                    signal_counts['semantic_only'] += 1
                elif 'temporal' in signals:
                    signal_counts['temporal_only'] += 1
                else:
                    signal_counts['none'] += 1
        meta['gate_histogram'] = signal_counts
    else:
        meta['constraint_availability'] = {
            'time_coverage': 0, 'centroid_coverage': 0,
            'multi_source': 0, 'typed_coverage': 0
        }
        meta['gate_histogram'] = {}

    return {
        'meta': meta,
        'nodes': nodes,
        'links': links
    }


@app.get("/")
async def index():
    """Serve the D3 visualization page."""
    return FileResponse(STATIC_DIR / 'weave.html')


@app.get("/api/snapshot")
async def snapshot(
    limit: int = Query(100, ge=10, le=500),
    min_support: float = Query(0.0, ge=0.0),
    centroids: bool = Query(True),
    dispersion: bool = Query(True)
):
    """Get topology snapshot as JSON."""
    try:
        data = await get_topology_snapshot(
            limit=limit,
            min_support=min_support,
            include_centroids=centroids,
            include_dispersion=dispersion
        )
        return JSONResponse(content=data)
    except Exception as e:
        logger.exception("Snapshot error")
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.get("/api/surface/{surface_id}")
async def surface_detail(surface_id: str):
    """Get detailed surface info with claims for inspection."""
    try:
        db_pool = await get_db_pool()
        neo4j = await get_neo4j()

        surface_repo = SurfaceRepository(db_pool, neo4j)
        surface = await surface_repo.get_with_claims(surface_id)

        if not surface:
            return JSONResponse(content={'error': 'Surface not found'}, status_code=404)

        dispersion = await compute_dispersion(surface_id, db_pool)

        def safe_iso(dt):
            if dt is None:
                return None
            if isinstance(dt, str):
                return dt
            if hasattr(dt, 'isoformat'):
                return dt.isoformat()
            return str(dt)

        return {
            'id': surface.id,
            'support': round(surface.support, 2),
            'claim_count': len(surface.claim_ids),
            'source_count': len(surface.sources),
            'sources': list(surface.sources),
            'anchors': list(surface.anchor_entities),
            'time_start': safe_iso(surface.time_start),
            'time_end': safe_iso(surface.time_end),
            'dispersion': dispersion,
            'has_centroid': surface.centroid is not None,
            'claims': [
                {
                    'id': c.id,
                    'text': c.text[:200] if c.text else '',
                    'event_time': safe_iso(c.event_time),
                    'reported_time': safe_iso(c.reported_time),
                    'time_source': 'event_time' if c.event_time else ('reported_time' if c.reported_time else 'none'),
                    'page_id': c.page_id,
                }
                for c in (surface.claims or [])[:20]
            ]
        }
    except Exception as e:
        logger.exception("Surface detail error")
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "weave-viz"}


# ============================================================================
# GOLDEN TRACE ENDPOINTS
# ============================================================================

# =============================================================================
# REAL-TIME WEAVER STREAM
# =============================================================================

@app.get("/api/live/stream")
async def stream_live_weaver():
    """
    Stream real-time weaver events via SSE.

    Consumes from Redis pub/sub channel 'weaver:events'.
    Events are emitted by principled_weaver as claims are processed.

    Structure:
    - type: "claim_processed"
    - L2: surface routing decision
    - L3: incident routing decision
    - L4: case routing decision (if applicable)
    - meta_claims: any noise/conflict detections
    - stats: running totals
    """
    import redis.asyncio as redis

    redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')

    async def generate():
        try:
            redis_client = redis.from_url(redis_url)
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("weaver:events")

            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'channel': 'weaver:events'})}\n\n"

            async for message in pubsub.listen():
                if message['type'] == 'message':
                    yield f"data: {message['data'].decode()}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            try:
                await pubsub.unsubscribe("weaver:events")
                await redis_client.close()
            except:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/live")
async def live_page():
    """Real-time weaver visualization page."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>REEE Live Weaver</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Monaco', 'Menlo', monospace;
            background: #0a0a12;
            color: #e0e0e0;
            padding: 20px;
        }
        h1 { color: #4aff9e; margin-bottom: 10px; }
        .status { color: #888; margin-bottom: 20px; }
        .status.connected { color: #4aff9e; }
        .status.error { color: #ff4a4a; }

        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat {
            background: #1a1a2e;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4a9eff;
        }
        .stat-label {
            font-size: 0.8em;
            color: #888;
            text-transform: uppercase;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .panel {
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 15px;
            max-height: 500px;
            overflow-y: auto;
        }
        .panel h3 {
            color: #4a9eff;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2a2a4e;
        }

        .event-card {
            background: #0f0f1a;
            border: 1px solid #2a2a4e;
            border-left: 3px solid #4a9eff;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
            animation: slideIn 0.3s ease;
        }
        .event-card.new-surface { border-left-color: #4aff9e; }
        .event-card.new-incident { border-left-color: #ff9e4a; }
        .event-card.meta-claim { border-left-color: #ff4a9e; }

        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .event-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .claim-id { font-weight: bold; color: #4a9eff; }
        .timestamp { font-size: 0.75em; color: #666; }

        .event-layers {
            font-size: 0.85em;
            color: #aaa;
        }
        .layer-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75em;
            margin-right: 5px;
        }
        .layer-badge.L2 { background: #2a4a6e; color: #8ab4f8; }
        .layer-badge.L3 { background: #4a3a2e; color: #f8b48a; }
        .layer-badge.L4 { background: #3a2a4e; color: #b48af8; }
        .layer-badge.meta { background: #4a2a3e; color: #f88ab4; }

        .meta-claims {
            margin-top: 5px;
            font-size: 0.8em;
            color: #ff9e4a;
        }
    </style>
</head>
<body>
    <h1>🔴 REEE Live Weaver</h1>
    <div class="status" id="status">Connecting...</div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value" id="claims-count">0</div>
            <div class="stat-label">Claims</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="surfaces-count">0</div>
            <div class="stat-label">L2 Surfaces</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="incidents-count">0</div>
            <div class="stat-label">L3 Incidents</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="cases-count">0</div>
            <div class="stat-label">L4 Cases</div>
        </div>
    </div>

    <!-- D3 Force Graph -->
    <div id="graph-container" style="background:#0f0f1a;border:1px solid #2a2a4e;border-radius:8px;margin-bottom:20px;">
        <svg id="topology-graph" width="100%" height="400"></svg>
    </div>

    <div class="main-grid">
        <div class="panel">
            <h3>Recent Events</h3>
            <div id="events-stream"></div>
        </div>
        <div class="panel">
            <h3>Meta-Claims</h3>
            <div id="meta-stream"></div>
        </div>
    </div>

    <script>
        const status = document.getElementById('status');
        const eventsStream = document.getElementById('events-stream');
        const metaStream = document.getElementById('meta-stream');
        const maxEvents = 50;

        // D3 Force Graph Setup
        const svg = d3.select('#topology-graph');
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = 400;
        svg.attr('width', width).attr('height', height);

        const g = svg.append('g');

        // Zoom behavior
        svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', (e) => g.attr('transform', e.transform)));

        // Graph data
        const nodes = [];
        const links = [];
        const nodeMap = new Map();  // id -> node
        const incidentToSurfaces = new Map();  // incident_id -> Set of surface_ids

        // Force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30))
            .on('tick', ticked);

        // Link and node groups
        let linkGroup = g.append('g').attr('class', 'links');
        let nodeGroup = g.append('g').attr('class', 'nodes');
        let labelGroup = g.append('g').attr('class', 'labels');

        function ticked() {
            linkGroup.selectAll('line')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeGroup.selectAll('circle')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labelGroup.selectAll('text')
                .attr('x', d => d.x)
                .attr('y', d => d.y + 25);
        }

        function updateGraph() {
            // Update links
            const link = linkGroup.selectAll('line').data(links, d => d.source.id + '-' + d.target.id);
            link.enter().append('line')
                .attr('stroke', '#4a4a6e')
                .attr('stroke-width', 2)
                .attr('stroke-opacity', 0.6);
            link.exit().remove();

            // Update nodes
            const node = nodeGroup.selectAll('circle').data(nodes, d => d.id);
            node.enter().append('circle')
                .attr('r', d => d.type === 'incident' ? 18 : 12)
                .attr('fill', d => d.type === 'incident' ? '#ff9e4a' : '#4a9eff')
                .attr('stroke', d => d.isNew ? '#4aff9e' : '#fff')
                .attr('stroke-width', d => d.isNew ? 3 : 1.5)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .append('title').text(d => d.label || d.id);
            node.exit().remove();

            // Update labels
            const label = labelGroup.selectAll('text').data(nodes, d => d.id);
            label.enter().append('text')
                .attr('text-anchor', 'middle')
                .attr('fill', '#888')
                .attr('font-size', '10px')
                .text(d => d.label || d.id.slice(-6));
            label.exit().remove();

            // Restart simulation
            simulation.nodes(nodes);
            simulation.force('link').links(links);
            simulation.alpha(0.3).restart();
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
        }
        function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
        }

        function addToGraph(data) {
            if (!data.L2 || !data.L3) return;

            const surfaceId = data.L2.surface_id;
            const incidentId = data.L3.incident_id;
            const questionKey = data.L2.question_key || '';

            // Add surface node if new
            if (!nodeMap.has(surfaceId)) {
                const surfaceNode = {
                    id: surfaceId,
                    type: 'surface',
                    label: questionKey.slice(0, 15) || surfaceId.slice(-6),
                    isNew: data.L2.is_new
                };
                nodes.push(surfaceNode);
                nodeMap.set(surfaceId, surfaceNode);
            }

            // Add incident node if new
            if (!nodeMap.has(incidentId)) {
                const incidentNode = {
                    id: incidentId,
                    type: 'incident',
                    label: 'L3:' + incidentId.slice(-4),
                    isNew: data.L3.is_new
                };
                nodes.push(incidentNode);
                nodeMap.set(incidentId, incidentNode);
                incidentToSurfaces.set(incidentId, new Set());
            }

            // Link surface to incident
            const surfaceSet = incidentToSurfaces.get(incidentId);
            if (!surfaceSet.has(surfaceId)) {
                surfaceSet.add(surfaceId);
                links.push({ source: incidentId, target: surfaceId });
            }

            // Clear isNew after a moment
            setTimeout(() => {
                const sn = nodeMap.get(surfaceId);
                const in_ = nodeMap.get(incidentId);
                if (sn) sn.isNew = false;
                if (in_) in_.isNew = false;
            }, 3000);

            updateGraph();
        }

        function connect() {
            const es = new EventSource('/api/live/stream');

            es.onopen = () => {
                status.textContent = '● Connected to weaver stream';
                status.className = 'status connected';
            };

            es.onerror = () => {
                status.textContent = '● Connection error - retrying...';
                status.className = 'status error';
            };

            es.onmessage = (e) => {
                const data = JSON.parse(e.data);

                if (data.type === 'connected') {
                    return;
                }

                if (data.type === 'claim_processed') {
                    // Update D3 graph
                    addToGraph(data);

                    // Update stats
                    if (data.stats) {
                        document.getElementById('claims-count').textContent = data.stats.claims_processed || 0;
                        document.getElementById('surfaces-count').textContent = data.stats.surfaces || 0;
                        document.getElementById('incidents-count').textContent = data.stats.incidents || 0;
                        document.getElementById('cases-count').textContent = data.stats.cases || 0;
                    }

                    // Add event card
                    const card = document.createElement('div');
                    let cardClass = 'event-card';
                    if (data.L2 && data.L2.is_new) cardClass += ' new-surface';
                    else if (data.L3 && data.L3.is_new) cardClass += ' new-incident';

                    card.className = cardClass;
                    card.innerHTML = `
                        <div class="event-header">
                            <span class="claim-id">${data.claim_id}</span>
                            <span class="timestamp">${new Date(data.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <div class="event-layers">
                            <span class="layer-badge L2">${data.L2.is_new ? 'NEW' : ''} ${data.L2.surface_id}</span>
                            <span class="layer-badge L3">${data.L3.is_new ? 'NEW' : ''} ${data.L3.incident_id}</span>
                            ${data.L4 ? `<span class="layer-badge L4">${data.L4.is_new ? 'NEW' : ''} ${data.L4.case_id}</span>` : ''}
                        </div>
                        ${data.L2.question_key ? `<div style="color:#666;font-size:0.8em;margin-top:3px;">question_key: ${data.L2.question_key}</div>` : ''}
                    `;

                    eventsStream.insertBefore(card, eventsStream.firstChild);

                    // Trim old events
                    while (eventsStream.children.length > maxEvents) {
                        eventsStream.removeChild(eventsStream.lastChild);
                    }

                    // Add meta-claims
                    if (data.meta_claims && data.meta_claims.length > 0) {
                        for (const mc of data.meta_claims) {
                            const metaCard = document.createElement('div');
                            metaCard.className = 'event-card meta-claim';
                            metaCard.innerHTML = `
                                <div class="event-header">
                                    <span class="layer-badge meta">${mc.type}</span>
                                    <span class="claim-id">${data.claim_id}</span>
                                </div>
                                <div style="font-size:0.85em;color:#aaa;">${mc.reason}</div>
                            `;
                            metaStream.insertBefore(metaCard, metaStream.firstChild);
                        }

                        while (metaStream.children.length > maxEvents) {
                            metaStream.removeChild(metaStream.lastChild);
                        }
                    }
                }
            };
        }

        connect();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/api/traces")
async def list_traces():
    """List available golden trace fixtures."""
    traces = []
    if TRACES_DIR.exists():
        for f in TRACES_DIR.glob("golden_trace_*.yaml"):
            try:
                import yaml
                with open(f) as fp:
                    data = yaml.safe_load(fp)
                meta = data.get("meta", {})
                traces.append({
                    "name": f.stem.replace("golden_trace_", ""),
                    "file": f.name,
                    "description": meta.get("description", ""),
                    "version": meta.get("version", "1.0"),
                    "claims": len(data.get("claims", [])),
                    "invariants": meta.get("invariants", []),
                })
            except Exception as e:
                logger.warning(f"Could not load trace {f}: {e}")
    return {"traces": traces}


@app.get("/api/trace/{name}/stream")
async def stream_trace(
    name: str,
    delay: float = Query(1.5, ge=0.5, le=5.0, description="Delay between claims in seconds")
):
    """
    Stream a golden trace replay via SSE.

    Each event contains:
    - claim: the incoming claim data
    - explanation: L2/L3 decision details
    - state: current L2 surfaces and L3 incidents
    """
    GoldenTrace, TraceKernel = get_trace_kernel()
    if not GoldenTrace:
        return JSONResponse(
            content={"error": "Trace kernel not available"},
            status_code=500
        )

    trace_file = TRACES_DIR / f"golden_trace_{name}.yaml"
    if not trace_file.exists():
        return JSONResponse(
            content={"error": f"Trace '{name}' not found"},
            status_code=404
        )

    async def generate():
        try:
            trace = GoldenTrace.load(trace_file)
            kernel = TraceKernel(trace)

            # Send trace metadata first
            yield f"data: {json.dumps({'type': 'meta', 'name': trace.name, 'description': trace.description, 'total_claims': len(trace.claims), 'entities': {k: {'canonical': v.canonical, 'type': v.type} for k, v in trace.entities.items()}})}\n\n"

            await asyncio.sleep(0.5)

            # Process each claim
            for i, tc in enumerate(trace.claims):
                exp = kernel.process_claim(tc)
                state = kernel.get_state_snapshot()

                # Build claim data for visualization
                claim_data = {
                    "id": tc.id,
                    "publisher": tc.publisher,
                    "gist": tc.gist,
                    "question_key": tc.question_key,
                    "anchor_entities": tc.anchor_entities,
                    "entities": tc.entities,
                    "typed_observation": tc.typed_observation,
                }

                # Build explanation
                explanation = {
                    "L2_decision": {
                        "action": exp.identity_decision.action,
                        "surface_id": exp.identity_decision.surface_id,
                        "reason": exp.identity_decision.reason,
                    },
                    "relation": exp.relation,
                    "meta_claims": [
                        {
                            "type": mc.type,
                            "entity": mc.entity,
                            "reason": mc.reason,
                            "evidence": mc.evidence,
                        }
                        for mc in exp.meta_claims
                    ],
                }

                if exp.typed_update:
                    explanation["typed_update"] = {
                        "type": exp.typed_update.get("type"),
                        "value": exp.typed_update.get("posterior_mean") or exp.typed_update.get("current_value"),
                        "observations": exp.typed_update.get("observations"),
                        "conflict": exp.typed_update.get("conflict_detected", False),
                    }

                event = {
                    "type": "claim",
                    "step": i + 1,
                    "total": len(trace.claims),
                    "claim": claim_data,
                    "explanation": explanation,
                    "state": state,
                }

                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(delay)

            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'final_state': kernel.get_state_snapshot()})}\n\n"

        except Exception as e:
            logger.exception(f"Trace stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/trace")
async def trace_page():
    """Golden trace replay visualization page with D3 force graph."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>REEE Golden Trace Replay</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Monaco', 'Menlo', monospace;
            background: #0a0a12;
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #4a9eff; margin-bottom: 10px; }
        h2 { color: #8888aa; font-size: 0.9em; margin-bottom: 20px; }

        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 8px;
        }
        select, button {
            padding: 8px 16px;
            border: 1px solid #4a4a6e;
            background: #2a2a4e;
            color: #e0e0e0;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover { background: #3a3a5e; }
        button.playing { background: #4a6a4e; }

        .main-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 20px;
        }

        .panel {
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 15px;
        }
        .panel h3 {
            color: #4a9eff;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2a2a4e;
        }

        /* Force graph */
        #graph-container {
            background: #0a0a12;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            height: 500px;
            position: relative;
        }
        #graph-container svg {
            width: 100%;
            height: 100%;
        }
        .legend {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(26, 26, 46, 0.9);
            padding: 10px;
            border-radius: 4px;
            font-size: 0.75em;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 5px;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .legend-rect {
            width: 20px;
            height: 12px;
            border-radius: 2px;
            border: 1px dashed;
        }

        /* Claim stream */
        .claim-stream {
            max-height: 300px;
            overflow-y: auto;
        }
        .claim-card {
            background: #0f0f1a;
            border: 1px solid #2a2a4e;
            border-left: 3px solid #4a9eff;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }
        .claim-card.new {
            border-left-color: #4aff9e;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .claim-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .claim-id { font-weight: bold; color: #4a9eff; }
        .claim-qkey {
            font-size: 0.8em;
            padding: 2px 8px;
            background: #2a2a4e;
            border-radius: 10px;
            color: #aaa;
        }
        .claim-gist { color: #ccc; font-size: 0.9em; }
        .claim-decision {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #2a2a4e;
            font-size: 0.85em;
        }
        .decision-action {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            margin-right: 10px;
        }
        .decision-action.new_surface { background: #2a4a3e; color: #4aff9e; }
        .decision-action.attach { background: #2a3a5e; color: #4a9eff; }
        .meta-claim {
            margin-top: 8px;
            padding: 8px;
            background: #1a1a2e;
            border-left: 2px solid #ff9e4a;
            font-size: 0.8em;
            color: #ffaa66;
        }

        /* L2/L3 state */
        .layer-section { margin-bottom: 20px; }
        .layer-section h4 {
            color: #8888aa;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .surface-item, .incident-item {
            background: #0f0f1a;
            border: 1px solid #2a2a4e;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 8px;
            font-size: 0.85em;
        }
        .surface-item { border-left: 3px solid #4a9eff; }
        .incident-item { border-left: 3px solid #ff9e4a; }
        .item-id { font-weight: bold; margin-bottom: 5px; }
        .item-claims { color: #888; font-size: 0.9em; }
        .item-anchors { color: #666; font-size: 0.85em; margin-top: 4px; }

        /* Typed state */
        .typed-state {
            background: #0f0f1a;
            border: 1px solid #2a2a4e;
            border-radius: 4px;
            padding: 12px;
        }
        .posterior-value {
            font-size: 2em;
            color: #4aff9e;
            font-weight: bold;
        }
        .posterior-meta { color: #888; font-size: 0.85em; margin-top: 5px; }
        .conflict-badge {
            display: inline-block;
            padding: 2px 8px;
            background: #4a2a2a;
            color: #ff6666;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
        }

        /* Progress */
        .progress-bar {
            height: 4px;
            background: #2a2a4e;
            border-radius: 2px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4a9eff, #7b5cff);
            transition: width 0.3s ease;
        }

        .status-line {
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            color: #666;
            margin-bottom: 10px;
        }
        .status-item { display: flex; gap: 6px; align-items: center; }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4a9eff;
        }
        .status-dot.complete { background: #4aff9e; }
        .status-dot.error { background: #ff4a4a; }
    </style>
</head>
<body>
    <h1>🧪 Golden Trace Replay</h1>
    <h2>Step-by-step visualization of REEE L2/L3 mechanics</h2>

    <div class="controls">
        <select id="trace-select">
            <option value="">Select a trace...</option>
        </select>
        <button id="play-btn" disabled>▶ Play</button>
        <button id="reset-btn" disabled>↺ Reset</button>
        <label>
            Delay: <input type="range" id="delay" min="0.5" max="3" step="0.5" value="1.5">
            <span id="delay-val">1.5s</span>
        </label>
    </div>

    <div class="progress-bar">
        <div class="progress-fill" id="progress" style="width: 0%"></div>
    </div>

    <div class="status-line">
        <div class="status-item">
            <div class="status-dot" id="status-dot"></div>
            <span id="status-text">Select a trace to begin</span>
        </div>
        <div id="step-counter"></div>
    </div>

    <!-- Force Graph -->
    <div class="panel" style="margin-bottom: 20px;">
        <h3>L2/L3 Topology Graph</h3>
        <div id="graph-container">
            <svg id="graph-svg"></svg>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#4a9eff"></div> L2 Surface</div>
                <div class="legend-item"><div class="legend-dot" style="background:#ff9e4a"></div> Claim</div>
                <div class="legend-item"><div class="legend-rect" style="border-color:#4aff9e"></div> L3 Incident</div>
            </div>
        </div>
    </div>

    <div class="main-grid">
        <div class="panel">
            <h3>Claim Stream</h3>
            <div class="claim-stream" id="claim-stream"></div>
        </div>

        <div class="panel">
            <h3>Current State</h3>

            <div class="layer-section">
                <h4>L2 Surfaces (by question_key)</h4>
                <div id="l2-surfaces"></div>
            </div>

            <div class="layer-section">
                <h4>L3 Incidents (by anchor motif)</h4>
                <div id="l3-incidents"></div>
            </div>

            <div class="layer-section">
                <h4>Typed Belief States</h4>
                <div id="typed-states"></div>
            </div>
        </div>
    </div>

    <script>
        const traceSelect = document.getElementById('trace-select');
        const playBtn = document.getElementById('play-btn');
        const resetBtn = document.getElementById('reset-btn');
        const delayInput = document.getElementById('delay');
        const delayVal = document.getElementById('delay-val');
        const progress = document.getElementById('progress');
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const stepCounter = document.getElementById('step-counter');
        const claimStream = document.getElementById('claim-stream');
        const l2Surfaces = document.getElementById('l2-surfaces');
        const l3Incidents = document.getElementById('l3-incidents');
        const typedStates = document.getElementById('typed-states');

        let eventSource = null;
        let isPlaying = false;
        let entities = {};

        // ============================================================
        // D3 Force Graph Setup
        // ============================================================
        const container = document.getElementById('graph-container');
        const svg = d3.select('#graph-svg');
        const width = container.clientWidth;
        const height = container.clientHeight;

        svg.attr('viewBox', [0, 0, width, height]);

        // Graph data
        let graphNodes = [];
        let graphLinks = [];
        let incidentHulls = [];

        // Force simulation
        const simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(60))
            .force('charge', d3.forceManyBody().strength(-150))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(25));

        // SVG groups for layering
        const hullGroup = svg.append('g').attr('class', 'hulls');
        const linkGroup = svg.append('g').attr('class', 'links');
        const nodeGroup = svg.append('g').attr('class', 'nodes');
        const labelGroup = svg.append('g').attr('class', 'labels');

        // Color scales
        const incidentColors = d3.scaleOrdinal()
            .range(['rgba(74, 255, 158, 0.15)', 'rgba(255, 158, 74, 0.15)', 'rgba(158, 74, 255, 0.15)']);
        const incidentStrokes = d3.scaleOrdinal()
            .range(['#4aff9e', '#ff9e4a', '#9e4aff']);

        function updateGraph(state) {
            const surfaces = state.L2_surfaces || {};
            const incidents = state.L3_incidents || {};
            const claims = state.claims || {};

            // Build nodes: claims + surfaces
            const newNodes = [];
            const nodeIds = new Set();

            // Add surface nodes
            Object.entries(surfaces).forEach(([sid, s]) => {
                newNodes.push({
                    id: sid,
                    type: 'surface',
                    label: s.question_key || sid,
                    claims: s.claims || []
                });
                nodeIds.add(sid);
            });

            // Add claim nodes
            Object.entries(claims).forEach(([cid, sid]) => {
                if (!nodeIds.has(cid)) {
                    newNodes.push({
                        id: cid,
                        type: 'claim',
                        label: cid,
                        surface: sid
                    });
                    nodeIds.add(cid);
                }
            });

            // Build links: claim -> surface
            const newLinks = [];
            Object.entries(claims).forEach(([cid, sid]) => {
                if (nodeIds.has(cid) && nodeIds.has(sid)) {
                    newLinks.push({ source: cid, target: sid });
                }
            });

            // Preserve positions for existing nodes
            const oldPositions = {};
            graphNodes.forEach(n => {
                oldPositions[n.id] = { x: n.x, y: n.y, vx: n.vx, vy: n.vy };
            });

            newNodes.forEach(n => {
                if (oldPositions[n.id]) {
                    Object.assign(n, oldPositions[n.id]);
                }
            });

            graphNodes = newNodes;
            graphLinks = newLinks;

            // Build incident hulls
            incidentHulls = Object.entries(incidents).map(([iid, inc], i) => ({
                id: iid,
                surfaces: inc.surfaces || [],
                anchors: inc.anchor_entities || [],
                colorIndex: i
            }));

            // Update simulation
            simulation.nodes(graphNodes);
            simulation.force('link').links(graphLinks);
            simulation.alpha(0.5).restart();

            renderGraph();
        }

        function renderGraph() {
            // Links
            const link = linkGroup.selectAll('line')
                .data(graphLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

            link.exit().remove();

            link.enter().append('line')
                .attr('stroke', '#4a4a6e')
                .attr('stroke-width', 1.5)
                .attr('stroke-opacity', 0.6)
              .merge(link);

            // Nodes
            const node = nodeGroup.selectAll('circle')
                .data(graphNodes, d => d.id);

            node.exit().remove();

            const nodeEnter = node.enter().append('circle')
                .attr('r', d => d.type === 'surface' ? 18 : 10)
                .attr('fill', d => d.type === 'surface' ? '#4a9eff' : '#ff9e4a')
                .attr('stroke', '#fff')
                .attr('stroke-width', 2)
                .attr('cursor', 'pointer')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            nodeEnter.append('title')
                .text(d => d.label);

            nodeEnter.merge(node)
                .attr('r', d => d.type === 'surface' ? 18 : 10)
                .attr('fill', d => d.type === 'surface' ? '#4a9eff' : '#ff9e4a');

            // Labels
            const label = labelGroup.selectAll('text')
                .data(graphNodes.filter(d => d.type === 'surface'), d => d.id);

            label.exit().remove();

            label.enter().append('text')
                .attr('font-size', '10px')
                .attr('fill', '#ccc')
                .attr('text-anchor', 'middle')
                .attr('dy', 30)
              .merge(label)
                .text(d => d.label.replace('S_', ''));
        }

        function renderHulls() {
            const hull = hullGroup.selectAll('path')
                .data(incidentHulls, d => d.id);

            hull.exit().remove();

            hull.enter().append('path')
                .attr('fill', (d, i) => incidentColors(i))
                .attr('stroke', (d, i) => incidentStrokes(i))
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '5,3')
              .merge(hull)
                .attr('d', d => {
                    // Get positions of surfaces in this incident
                    const points = d.surfaces
                        .map(sid => graphNodes.find(n => n.id === sid))
                        .filter(n => n && n.x !== undefined)
                        .map(n => [n.x, n.y]);

                    if (points.length < 2) {
                        // Single surface: draw circle
                        if (points.length === 1) {
                            const [x, y] = points[0];
                            return `M ${x-30},${y} a 30,30 0 1,0 60,0 a 30,30 0 1,0 -60,0`;
                        }
                        return '';
                    }

                    // Expand points for padding
                    const padded = [];
                    points.forEach(([x, y]) => {
                        padded.push([x - 25, y - 25]);
                        padded.push([x + 25, y - 25]);
                        padded.push([x - 25, y + 25]);
                        padded.push([x + 25, y + 25]);
                    });

                    const hullPoints = d3.polygonHull(padded);
                    if (!hullPoints) return '';

                    return 'M' + hullPoints.map(p => p.join(',')).join('L') + 'Z';
                });
        }

        // Simulation tick
        simulation.on('tick', () => {
            linkGroup.selectAll('line')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeGroup.selectAll('circle')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labelGroup.selectAll('text')
                .attr('x', d => d.x)
                .attr('y', d => d.y);

            renderHulls();
        });

        // Drag handlers
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        function resetGraph() {
            graphNodes = [];
            graphLinks = [];
            incidentHulls = [];
            hullGroup.selectAll('*').remove();
            linkGroup.selectAll('*').remove();
            nodeGroup.selectAll('*').remove();
            labelGroup.selectAll('*').remove();
            simulation.nodes([]);
            simulation.force('link').links([]);
        }

        // ============================================================
        // Load available traces
        fetch('/api/traces')
            .then(r => r.json())
            .then(data => {
                data.traces.forEach(t => {
                    const opt = document.createElement('option');
                    opt.value = t.name;
                    opt.textContent = `${t.name} (${t.claims} claims) - ${t.description}`;
                    traceSelect.appendChild(opt);
                });
            });

        traceSelect.onchange = () => {
            playBtn.disabled = !traceSelect.value;
            resetBtn.disabled = true;
            reset();
        };

        delayInput.oninput = () => {
            delayVal.textContent = delayInput.value + 's';
        };

        function reset() {
            claimStream.innerHTML = '';
            l2Surfaces.innerHTML = '<div style="color:#666">No surfaces yet</div>';
            l3Incidents.innerHTML = '<div style="color:#666">No incidents yet</div>';
            typedStates.innerHTML = '<div style="color:#666">No typed values yet</div>';
            progress.style.width = '0%';
            stepCounter.textContent = '';
            statusText.textContent = 'Ready';
            statusDot.className = 'status-dot';
            entities = {};
            resetGraph();
        }

        resetBtn.onclick = () => {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            isPlaying = false;
            playBtn.textContent = '▶ Play';
            playBtn.className = '';
            reset();
        };

        playBtn.onclick = () => {
            if (isPlaying) {
                // Pause
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
                isPlaying = false;
                playBtn.textContent = '▶ Play';
                playBtn.className = '';
                statusText.textContent = 'Paused';
                return;
            }

            // Start
            reset();
            isPlaying = true;
            playBtn.textContent = '⏸ Pause';
            playBtn.className = 'playing';
            resetBtn.disabled = false;
            statusText.textContent = 'Streaming...';
            statusDot.className = 'status-dot';

            const delay = delayInput.value;
            eventSource = new EventSource(`/api/trace/${traceSelect.value}/stream?delay=${delay}`);

            eventSource.onmessage = (e) => {
                const data = JSON.parse(e.data);

                if (data.type === 'meta') {
                    statusText.textContent = `Playing: ${data.name}`;
                    entities = data.entities || {};
                    return;
                }

                if (data.type === 'claim') {
                    const pct = (data.step / data.total) * 100;
                    progress.style.width = pct + '%';
                    stepCounter.textContent = `Step ${data.step} of ${data.total}`;

                    // Add claim card
                    addClaimCard(data.claim, data.explanation);

                    // Update state panels
                    updateState(data.state);

                    // Update force graph
                    updateGraph(data.state);
                }

                if (data.type === 'complete') {
                    statusText.textContent = 'Complete';
                    statusDot.className = 'status-dot complete';
                    isPlaying = false;
                    playBtn.textContent = '▶ Play';
                    playBtn.className = '';
                    eventSource.close();
                }

                if (data.type === 'error') {
                    statusText.textContent = 'Error: ' + data.message;
                    statusDot.className = 'status-dot error';
                    isPlaying = false;
                    playBtn.textContent = '▶ Play';
                    playBtn.className = '';
                }
            };

            eventSource.onerror = () => {
                statusText.textContent = 'Connection error';
                statusDot.className = 'status-dot error';
            };
        };

        function resolveEntity(eid) {
            if (entities[eid]) return entities[eid].canonical;
            return eid;
        }

        function addClaimCard(claim, explanation) {
            const card = document.createElement('div');
            card.className = 'claim-card new';

            const anchors = (claim.anchor_entities || []).map(resolveEntity).join(', ');
            const metaClaims = (explanation.meta_claims || []).map(mc =>
                `<div class="meta-claim"><strong>${mc.type}</strong>: ${mc.reason}</div>`
            ).join('');

            let typedInfo = '';
            if (explanation.typed_update) {
                const tu = explanation.typed_update;
                typedInfo = `<div style="margin-top:8px;color:#888">
                    Typed: ${tu.value?.toFixed?.(2) || tu.value} (${tu.observations} obs)
                    ${tu.conflict ? '<span class="conflict-badge">CONFLICT</span>' : ''}
                </div>`;
            }

            card.innerHTML = `
                <div class="claim-header">
                    <span class="claim-id">${claim.id}</span>
                    <span class="claim-qkey">${claim.question_key}</span>
                </div>
                <div class="claim-gist">${claim.gist}</div>
                <div style="color:#666;font-size:0.8em;margin-top:5px">
                    ${claim.publisher} · anchors: ${anchors}
                </div>
                <div class="claim-decision">
                    <span class="decision-action ${explanation.L2_decision.action}">${explanation.L2_decision.action}</span>
                    → ${explanation.L2_decision.surface_id}
                    ${explanation.relation ? `<span style="color:#888;margin-left:10px">(${explanation.relation})</span>` : ''}
                </div>
                ${metaClaims}
                ${typedInfo}
            `;

            claimStream.insertBefore(card, claimStream.firstChild);

            // Remove 'new' class after animation
            setTimeout(() => card.classList.remove('new'), 300);
        }

        function updateState(state) {
            // L2 Surfaces
            const surfaces = state.L2_surfaces || {};
            if (Object.keys(surfaces).length === 0) {
                l2Surfaces.innerHTML = '<div style="color:#666">No surfaces yet</div>';
            } else {
                l2Surfaces.innerHTML = Object.entries(surfaces).map(([sid, s]) => `
                    <div class="surface-item">
                        <div class="item-id">${sid}</div>
                        <div class="item-claims">claims: ${(s.claims || []).join(', ')}</div>
                        <div class="item-anchors">qkey: ${s.question_key || 'unknown'}</div>
                    </div>
                `).join('');
            }

            // L3 Incidents
            const incidents = state.L3_incidents || {};
            if (Object.keys(incidents).length === 0) {
                l3Incidents.innerHTML = '<div style="color:#666">No incidents yet</div>';
            } else {
                l3Incidents.innerHTML = Object.entries(incidents).map(([iid, inc]) => `
                    <div class="incident-item">
                        <div class="item-id">${iid}</div>
                        <div class="item-claims">surfaces: ${(inc.surfaces || []).join(', ')}</div>
                        <div class="item-anchors">anchors: ${(inc.anchor_entities || []).join(', ')}</div>
                    </div>
                `).join('');
            }

            // Typed states
            const posteriors = state.typed_posteriors || {};
            if (Object.keys(posteriors).length === 0) {
                typedStates.innerHTML = '<div style="color:#666">No typed values yet</div>';
            } else {
                let html = '';
                for (const [sid, questions] of Object.entries(posteriors)) {
                    for (const [qkey, p] of Object.entries(questions)) {
                        const value = p.posterior_mean ?? p.current_value ?? '?';
                        const displayValue = typeof value === 'number' ? value.toFixed(2) : value;
                        html += `
                            <div class="typed-state">
                                <div style="color:#888;font-size:0.8em;margin-bottom:5px">${sid} · ${qkey}</div>
                                <span class="posterior-value">${displayValue}</span>
                                ${p.conflict_detected ? '<span class="conflict-badge">CONFLICT</span>' : ''}
                                <div class="posterior-meta">
                                    ${p.observations} observations · type: ${p.type || 'unknown'}
                                </div>
                            </div>
                        `;
                    }
                }
                typedStates.innerHTML = html || '<div style="color:#666">No typed values</div>';
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/api/stream")
async def stream_topology(interval: float = Query(2.0, ge=0.5, le=10.0)):
    """
    Server-Sent Events stream of topology changes.

    Polls the database and emits snapshots at the specified interval.
    Use this to visualize the rebuild process in real-time.
    """
    async def event_generator():
        last_counts = {}

        while True:
            try:
                # Get lightweight snapshot
                data = await get_topology_snapshot(
                    limit=100,
                    min_support=0.0,
                    include_centroids=True,
                    include_dispersion=False  # Skip dispersion for speed
                )

                counts = data['meta']['counts']

                # Detect changes
                changed = counts != last_counts
                last_counts = counts.copy()

                # Build SSE message
                event_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'counts': counts,
                    'constraint_availability': data['meta']['constraint_availability'],
                    'changed': changed,
                    'nodes': len(data['nodes']),
                    'links': len(data['links']),
                }

                yield f"data: {json.dumps(event_data)}\n\n"

                await asyncio.sleep(interval)

            except Exception as e:
                logger.exception("Stream error")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/live-old")
async def live_page_old():
    """
    Old live streaming visualization page (deprecated).

    Shows real-time topology changes during rebuild.
    Use /live instead for L2/L3 weaver events.
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>REEE Live Topology (Old)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Monaco', 'Menlo', monospace;
            background: #0a0a0f;
            color: #e0e0e0;
            padding: 20px;
        }
        h1 {
            color: #4a9eff;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 15px;
        }
        .card h2 {
            color: #8888aa;
            font-size: 0.8em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .big-number {
            font-size: 2.5em;
            color: #4a9eff;
            font-weight: bold;
        }
        .changed { animation: pulse 0.5s ease-out; }
        @keyframes pulse {
            0% { background: #2a4a6e; }
            100% { background: #1a1a2e; }
        }
        .progress-bar {
            height: 8px;
            background: #2a2a4e;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4a9eff, #7b5cff);
            transition: width 0.3s ease;
        }
        .log {
            background: #0f0f15;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-size: 0.85em;
        }
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid #1a1a2e;
        }
        .log-entry.change { color: #4aff9e; }
        .log-entry.error { color: #ff4a4a; }
        .timestamp { color: #666; margin-right: 10px; }
        .status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        .status.connected { background: #1a4a2e; color: #4aff9e; }
        .status.disconnected { background: #4a1a1a; color: #ff4a4a; }

        #canvas-container {
            background: #0f0f15;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
            padding: 10px;
            margin-top: 20px;
        }
        #topology-canvas {
            width: 100%;
            height: 400px;
        }
        .legend {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.8em;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot-surface { background: #4a9eff; }
        .dot-event { background: #ff9e4a; }
        .dot-new { background: #4aff9e; animation: blink 1s infinite; }
        @keyframes blink {
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <h1>🔮 REEE Live Topology Stream</h1>

    <div class="grid">
        <div class="card" id="surface-card">
            <h2>Surfaces</h2>
            <div class="big-number" id="surface-count">-</div>
            <div class="progress-bar">
                <div class="progress-fill" id="centroid-bar" style="width: 0%"></div>
            </div>
            <small style="color:#666">Centroid coverage</small>
        </div>

        <div class="card" id="event-card">
            <h2>Events</h2>
            <div class="big-number" id="event-count">-</div>
            <div class="progress-bar">
                <div class="progress-fill" id="time-bar" style="width: 0%"></div>
            </div>
            <small style="color:#666">Time coverage</small>
        </div>

        <div class="card" id="edge-card">
            <h2>Edges</h2>
            <div class="big-number" id="edge-count">-</div>
            <div class="progress-bar">
                <div class="progress-fill" id="multi-bar" style="width: 0%"></div>
            </div>
            <small style="color:#666">Multi-source</small>
        </div>
    </div>

    <div style="display:flex; align-items:center; gap:20px; margin-bottom:10px;">
        <span class="status" id="connection-status">Connecting...</span>
        <small id="last-update" style="color:#666">-</small>
    </div>

    <div class="log" id="log">
        <div class="log-entry">Waiting for connection...</div>
    </div>

    <div id="canvas-container">
        <canvas id="topology-canvas"></canvas>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot dot-surface"></div> Surface</div>
            <div class="legend-item"><div class="legend-dot dot-event"></div> Event</div>
            <div class="legend-item"><div class="legend-dot dot-new"></div> New (this session)</div>
        </div>
    </div>

    <script>
        const log = document.getElementById('log');
        const surfaceCount = document.getElementById('surface-count');
        const eventCount = document.getElementById('event-count');
        const edgeCount = document.getElementById('edge-count');
        const centroidBar = document.getElementById('centroid-bar');
        const timeBar = document.getElementById('time-bar');
        const multiBar = document.getElementById('multi-bar');
        const status = document.getElementById('connection-status');
        const lastUpdate = document.getElementById('last-update');

        // Track initial counts to detect new items
        let initialCounts = null;
        let nodePositions = {};

        function addLog(msg, type = '') {
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + type;
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="timestamp">${time}</span>${msg}`;
            log.insertBefore(entry, log.firstChild);

            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }

        function drawTopology(nodes, links) {
            const canvas = document.getElementById('topology-canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas size
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;

            const w = canvas.width;
            const h = canvas.height;
            const padding = 50;

            // Clear
            ctx.fillStyle = '#0f0f15';
            ctx.fillRect(0, 0, w, h);

            // Draw edges first
            ctx.strokeStyle = 'rgba(100, 100, 150, 0.3)';
            ctx.lineWidth = 1;
            links.forEach(link => {
                const source = nodePositions[link.source];
                const target = nodePositions[link.target];
                if (source && target) {
                    ctx.beginPath();
                    ctx.moveTo(source.x, source.y);
                    ctx.lineTo(target.x, target.y);
                    ctx.stroke();
                }
            });

            // Draw nodes
            nodes.forEach(node => {
                const x = padding + (node.x || 0.5) * (w - 2 * padding);
                const y = padding + (node.y || 0.5) * (h - 2 * padding);

                nodePositions[node.id] = { x, y };

                const isSurface = node.kind === 'surface';
                const size = isSurface ? 5 + Math.min(node.claim_count || 1, 20) : 8;
                const color = isSurface ? '#4a9eff' : '#ff9e4a';

                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();

                // Border
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }

        // Connect to SSE stream
        const eventSource = new EventSource('/api/stream?interval=2');

        eventSource.onopen = () => {
            status.textContent = 'Connected';
            status.className = 'status connected';
            addLog('Connected to stream');
        };

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                addLog('Error: ' + data.error, 'error');
                return;
            }

            // Update counts
            const counts = data.counts || {};
            surfaceCount.textContent = counts.surfaces || 0;
            eventCount.textContent = counts.events || 0;
            edgeCount.textContent = (counts.membership_edges || 0) + (counts.aboutness_edges || 0);

            // Update progress bars
            const ca = data.constraint_availability || {};
            centroidBar.style.width = (ca.centroid_coverage || 0) + '%';
            timeBar.style.width = (ca.time_coverage || 0) + '%';
            multiBar.style.width = (ca.multi_source || 0) + '%';

            // Log changes
            if (data.changed) {
                addLog(`Surfaces: ${counts.surfaces}, Events: ${counts.events}, Edges: ${edgeCount.textContent}`, 'change');

                // Animate cards
                document.getElementById('surface-card').classList.add('changed');
                document.getElementById('event-card').classList.add('changed');
                document.getElementById('edge-card').classList.add('changed');

                setTimeout(() => {
                    document.getElementById('surface-card').classList.remove('changed');
                    document.getElementById('event-card').classList.remove('changed');
                    document.getElementById('edge-card').classList.remove('changed');
                }, 500);
            }

            // Update timestamp
            lastUpdate.textContent = 'Last update: ' + new Date(data.timestamp).toLocaleTimeString();

            // Fetch full snapshot periodically for canvas (every 10 updates)
            if (Math.random() < 0.1 || data.changed) {
                fetch('/api/snapshot?limit=100&dispersion=false')
                    .then(r => r.json())
                    .then(snapshot => {
                        drawTopology(snapshot.nodes || [], snapshot.links || []);
                    })
                    .catch(e => console.error('Snapshot fetch error:', e));
            }
        };

        eventSource.onerror = () => {
            status.textContent = 'Disconnected';
            status.className = 'status disconnected';
            addLog('Connection lost, reconnecting...', 'error');
        };

        // Initial snapshot
        fetch('/api/snapshot?limit=100&dispersion=false')
            .then(r => r.json())
            .then(snapshot => {
                drawTopology(snapshot.nodes || [], snapshot.links || []);
                addLog('Initial snapshot loaded: ' + (snapshot.nodes?.length || 0) + ' nodes');
            });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup connections on shutdown."""
    global _db_pool, _neo4j
    if _db_pool:
        await _db_pool.close()
    if _neo4j:
        await _neo4j.close()


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('WEAVE_VIZ_PORT', 8080))
    print(f"🔮 Weave Topology Visualization running on port {port}")
    print(f"   Open http://localhost:{port} to view")
    uvicorn.run(app, host='0.0.0.0', port=port)
