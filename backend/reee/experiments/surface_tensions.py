"""
Surface Tension Detection Experiment
=====================================

Shows TOP EVENTS with semantic descriptions and their GAPS (meta-claims).

Per REEE1 Section 5 (Constraint-State Theory), gaps emerge from:
- Incompatible constraints: typed_value_conflict, unresolved_conflict
- Weak constraints: high_entropy_value, single_source_only
- Missing constraints: coverage_gap, extraction_gap

Run:
    docker exec herenews-app python -m reee.experiments.surface_tensions
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from reee.types import Surface, Parameters, MetaClaim
from reee.views.incident import build_incident_events, IncidentViewParams
from reee.meta.detectors import TensionDetector, ViewDiagnostics, count_by_type


def log(msg: str):
    """Print with immediate flush for progressive output."""
    print(msg, flush=True)


@dataclass
class EventGaps:
    """Gaps detected for a single event."""
    event_id: str
    topic: str  # Semantic description from anchors/claims
    surface_count: int = 0
    source_count: int = 0

    # Constraint-state gaps (per REEE1 Section 5)
    single_source_surfaces: List[str] = field(default_factory=list)
    typed_value_conflicts: List[Dict] = field(default_factory=list)
    high_entropy_surfaces: List[str] = field(default_factory=list)
    coverage_gaps: List[Dict] = field(default_factory=list)

    # Structural weaknesses
    quarantine_surfaces: List[Tuple[str, List[str]]] = field(default_factory=list)
    corroboration_rate: float = 0.0  # % of surfaces with >1 source

    @property
    def total_gaps(self) -> int:
        return (
            len(self.single_source_surfaces) +
            len(self.typed_value_conflicts) +
            len(self.high_entropy_surfaces) +
            len(self.coverage_gaps)
        )

    @property
    def inquiry_priority(self) -> float:
        """Higher = more urgent inquiry potential."""
        # Conflicts are highest priority (incompatible constraints)
        score = len(self.typed_value_conflicts) * 10
        # Low corroboration is next (weak constraints)
        score += (1 - self.corroboration_rate) * self.surface_count * 2
        # Quarantine surfaces indicate weak evidence
        score += len(self.quarantine_surfaces) * 1
        return score


def parse_datetime(val) -> Optional[datetime]:
    """Parse datetime from string or return as-is if already datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            # Try ISO format
            return datetime.fromisoformat(val.replace('Z', '+00:00'))
        except:
            try:
                # Try common format
                return datetime.strptime(val[:19], '%Y-%m-%dT%H:%M:%S')
            except:
                return None
    return None


async def load_surfaces_and_claims() -> Tuple[Dict[str, Surface], Dict[str, str], Dict[str, List[str]]]:
    """
    Load surfaces from Neo4j + centroids from PostgreSQL.

    Returns:
        (surfaces, claim_texts, page_titles)
    """
    import asyncpg
    import os
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to PostgreSQL for centroids
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=1,
        max_size=5
    )

    surfaces = {}
    claim_texts = {}  # claim_id -> text
    page_titles = {}  # page_id -> title

    try:
        # Load surfaces from Neo4j
        rows = await neo4j._execute_read('''
            MATCH (s:Surface)
            RETURN s.id as id, s.sources as sources, s.entities as entities,
                   s.anchor_entities as anchor_entities,
                   s.time_start as time_start, s.time_end as time_end
        ''')

        for row in rows:
            sid = row['id']
            time_start = parse_datetime(row.get('time_start'))
            time_end = parse_datetime(row.get('time_end'))

            surfaces[sid] = Surface(
                id=sid,
                claim_ids=set(),
                sources=set(row['sources'] or []),
                entities=set(row['entities'] or []),
                anchor_entities=set(row['anchor_entities'] or []),
                time_window=(time_start, time_end),
            )

        # Load centroids from PostgreSQL
        from pgvector.asyncpg import register_vector
        async with db_pool.acquire() as conn:
            await register_vector(conn)
            centroid_rows = await conn.fetch('''
                SELECT surface_id, centroid
                FROM content.surface_centroids
                WHERE centroid IS NOT NULL
            ''')

            for row in centroid_rows:
                sid = row['surface_id']
                if sid in surfaces:
                    centroid = row['centroid']
                    if hasattr(centroid, 'tolist'):
                        surfaces[sid].centroid = centroid.tolist()
                    elif isinstance(centroid, (list, tuple)):
                        surfaces[sid].centroid = list(centroid)

        # Load claim texts for semantic context
        rows = await neo4j._execute_read('''
            MATCH (c:Claim)
            WHERE c.text IS NOT NULL
            RETURN c.id as id, c.text as text
            LIMIT 500
        ''')
        for row in rows:
            claim_texts[row['id']] = row['text']

        # Load page titles for context
        rows = await neo4j._execute_read('''
            MATCH (p:Page)
            WHERE p.title IS NOT NULL
            RETURN p.id as id, p.title as title
        ''')
        for row in rows:
            page_titles[row['id']] = row['title']

    finally:
        await neo4j.close()
        await db_pool.close()

    return surfaces, claim_texts, page_titles


def infer_event_topic(event, surfaces: Dict[str, Surface], page_titles: Dict[str, str]) -> str:
    """
    Infer semantic topic from anchor entities and page titles.

    Returns a human-readable topic description.
    """
    # Get top anchor entities (excluding generic hubs)
    hub_entities = {'Hong Kong', 'China', 'United States', 'US', 'UK'}
    anchors = [a for a in event.anchor_entities if a not in hub_entities]

    # Get page titles from sources
    titles = []
    for sid in list(event.surface_ids)[:5]:
        s = surfaces.get(sid)
        if s:
            for src in list(s.sources)[:2]:
                if src in page_titles:
                    titles.append(page_titles[src])

    # Build topic string
    if anchors:
        topic = " + ".join(anchors[:3])
    elif titles:
        # Extract key phrases from titles
        title = titles[0][:60]
        topic = title
    else:
        topic = f"Event with {len(event.surface_ids)} surfaces"

    return topic


def analyze_event_gaps(
    event,
    surfaces: Dict[str, Surface],
    meta_claims: List[MetaClaim],
    page_titles: Dict[str, str]
) -> EventGaps:
    """
    Analyze gaps for a single event based on constraint-state theory.
    """
    topic = infer_event_topic(event, surfaces, page_titles)

    gaps = EventGaps(
        event_id=event.id,
        topic=topic,
        surface_count=len(event.surface_ids),
        source_count=event.total_sources,
    )

    # Build surface -> meta-claim index
    surface_meta = defaultdict(list)
    for mc in meta_claims:
        if mc.target_type == 'surface':
            surface_meta[mc.target_id].append(mc)

    # Analyze each surface in event
    multi_source_count = 0
    for sid in event.surface_ids:
        s = surfaces.get(sid)
        if not s:
            continue

        # Check source diversity
        if len(s.sources) > 1:
            multi_source_count += 1
        else:
            gaps.single_source_surfaces.append(sid)

        # Check meta-claims for this surface
        for mc in surface_meta.get(sid, []):
            if mc.type == 'typed_value_conflict':
                gaps.typed_value_conflicts.append({
                    'surface': sid,
                    'variable': mc.evidence.get('question_key'),
                    'values': mc.evidence.get('conflicting_values'),
                })
            elif mc.type == 'high_entropy_value':
                gaps.high_entropy_surfaces.append(sid)
            elif mc.type == 'coverage_gap':
                gaps.coverage_gaps.append({
                    'surface': sid,
                    'missing': mc.evidence.get('missing_question_key'),
                    'expectedness': mc.evidence.get('expectedness'),
                })

    # Check quarantine surfaces (structural weakness)
    for sid, membership in event.memberships.items():
        if membership.level.value == 'quarantine':
            backbones = membership.evidence.get('shared_backbones', [])
            gaps.quarantine_surfaces.append((sid, backbones))

    # Compute corroboration rate
    if gaps.surface_count > 0:
        gaps.corroboration_rate = multi_source_count / gaps.surface_count

    return gaps


async def run_semantic_event_gaps():
    """
    Main: Show top events with semantic descriptions and their gaps.

    Per Jaynes-style rigor: the system must be explicit about what constraints
    are MISSING before diagnosing gaps. Don't pretend to make richer inferences
    than the constraints allow.
    """
    log("=" * 70)
    log("EPISTEMIC DIAGNOSTIC REPORT")
    log("=" * 70)
    log("")
    log("Per REEE1 Section 5 (Constraint-State Theory):")
    log("  The system must be self-aware about constraint availability.")
    log("  If coverage(S,X)=0 for all X, say so explicitly.")
    log("")

    # Load data
    log("Loading surfaces and context from Neo4j...")
    surfaces, claim_texts, page_titles = await load_surfaces_and_claims()
    log(f"   Surfaces: {len(surfaces)}")
    log(f"   Claim texts: {len(claim_texts)}")
    log(f"   Page titles: {len(page_titles)}")
    log("")

    if not surfaces:
        log("No surfaces found!")
        return

    # Build incident events
    log("Building L3 incident events...")
    params = IncidentViewParams(
        hub_detection='time_mode',
        min_signals=2,
        require_discriminative_anchor=True,
    )
    result = build_incident_events(surfaces, params)
    log(f"   Multi-surface events: {result.trace.events_formed}")
    log(f"   Singletons: {result.trace.singletons}")
    log("")

    # =========================================================================
    # PART 1: CONSTRAINT AVAILABILITY (What CAN'T we infer?)
    # =========================================================================
    log("=" * 70)
    log("PART 1: CONSTRAINT AVAILABILITY")
    log("=" * 70)
    log("")
    log("Jaynes-style rigor: What constraints are MISSING that block inference?")
    log("")

    # Use ViewDiagnostics for self-aware reporting
    view_diag = ViewDiagnostics(
        surfaces=surfaces,
        claims={},  # No claims loaded
        events=result.events,
        view_trace=result.trace,
    )

    summary = view_diag.get_summary()
    avail = summary['constraint_availability']
    binding = summary['binding_diagnostics']

    # Typed constraints
    log("📊 TYPED CONSTRAINTS:")
    log(f"   Claims loaded: {avail['total_claims']}")
    log(f"   Claims with question_key: {avail['claims_with_question_key']}")
    log(f"   Typed coverage rate: {avail['typed_coverage_rate']:.1%}")
    if avail['typed_coverage_rate'] < 0.05:
        log("   ⚠️  BLOCKED: Cannot compute typed_value_conflict or coverage_gap")
        log("      → Need claims with question_key + extracted_value")
    log("")

    # Temporal constraints
    log("⏱️  TEMPORAL CONSTRAINTS:")
    total_surfaces = avail['surfaces_without_time'] + len(surfaces) - avail['surfaces_without_time']
    log(f"   Surfaces with time: {len(surfaces) - avail['surfaces_without_time']}/{len(surfaces)}")
    log(f"   Time coverage rate: {avail['time_coverage_rate']:.1%}")
    if avail['time_coverage_rate'] < 0.5:
        log("   ⚠️  BLOCKED: Cannot do reliable temporal binding")
        log("      → Need timestamps on surfaces")
    log("")

    # Semantic constraints
    log("🔗 SEMANTIC CONSTRAINTS:")
    log(f"   Surfaces with centroid: {len(surfaces) - avail['surfaces_without_centroid']}/{len(surfaces)}")
    log(f"   Semantic coverage rate: {avail['semantic_coverage_rate']:.1%}")
    if avail['semantic_coverage_rate'] < 0.1:
        log("   ⚠️  BLOCKED: Cannot compute semantic similarity signal")
        log("      → Relying on anchor overlap only")
    log("")

    # =========================================================================
    # PART 2: BINDING DIAGNOSTICS (Why are events weak?)
    # =========================================================================
    log("=" * 70)
    log("PART 2: BINDING DIAGNOSTICS")
    log("=" * 70)
    log("")
    log("Why did edges form (or not form)?")
    log("")

    log(f"📈 EDGE FORMATION:")
    log(f"   Total candidate pairs: {binding['total_candidate_pairs']:,}")
    log(f"   Edges formed: {binding['edges_formed']}")
    log(f"   Edge formation rate: {binding['edge_formation_rate']:.2%}")
    log("")

    log(f"🚧 GATES BLOCKING EDGES:")
    log(f"   Blocked by signals_met < min: {binding['blocked_by_signals_met']}")
    log(f"   Blocked by no discriminative anchor: {binding['blocked_by_no_discriminative']}")
    log(f"   Blocked by temporal unknown: {binding['blocked_by_temporal_unknown']}")
    log("")

    log(f"👥 MEMBERSHIP DISTRIBUTION:")
    log(f"   Core: {binding['core']}")
    log(f"   Periphery: {binding['periphery']}")
    log(f"   Quarantine: {binding['quarantine']}")
    log(f"   Quarantine rate: {binding['quarantine_rate']:.1%}")
    if binding['quarantine_rate'] > 0.3:
        log("   ⚠️  Events held together by weak anchor-only attachment")
    log("")

    # =========================================================================
    # PART 3: VIEW-LEVEL META-CLAIMS (Self-awareness)
    # =========================================================================
    log("=" * 70)
    log("PART 3: INFERENCE BLOCKERS (View-Level Meta-Claims)")
    log("=" * 70)
    log("")

    view_meta_claims = view_diag.emit_meta_claims()
    if view_meta_claims:
        for mc in view_meta_claims:
            log(f"🔴 {mc.type}")
            log(f"   {mc.evidence.get('implication', '')}")
            log("")
    else:
        log("✅ No major inference blockers detected")
        log("")

    # =========================================================================
    # PART 4: ACTIONABLE PROTO-INQUIRIES
    # =========================================================================
    log("=" * 70)
    log("PART 4: ACTIONABLE PROTO-INQUIRIES")
    log("=" * 70)
    log("")
    log("Given the constraint availability above, the RIGHT questions to ask are:")
    log("")

    # Priority 1: Fill constraint gaps
    if avail['typed_coverage_rate'] < 0.05:
        log("🔴 PRIORITY 1: Extract typed variables")
        log("   Task: Run typed extraction on claims to populate question_key + extracted_value")
        log("   Unlocks: typed_value_conflict, coverage_gap detection")
        log("")

    if avail['time_coverage_rate'] < 0.5:
        log("🔴 PRIORITY 2: Resolve time")
        log("   Task: Add timestamps to surfaces (from page pub_time or claim event_time)")
        log("   Unlocks: Reliable temporal binding, incident scoping")
        log("")

    if avail['semantic_coverage_rate'] < 0.1:
        log("🟡 PRIORITY 3: Compute embeddings")
        log("   Task: Generate surface centroids from claim embeddings")
        log("   Unlocks: Semantic similarity signal for edge scoring")
        log("")

    # Priority 2: Strengthen weak events
    if binding['quarantine_rate'] > 0.3:
        log("🟡 PRIORITY 4: Strengthen weak events")
        log(f"   {binding['quarantine']} quarantine surfaces need more evidence")
        log("   Task: Find additional sources or discriminative anchors")
        log("")

    # =========================================================================
    # PART 5: TOP EVENTS (only after showing what we CAN'T infer)
    # =========================================================================
    log("=" * 70)
    log("PART 5: TOP EVENTS WITH CURRENT GAPS")
    log("=" * 70)
    log("")
    log("Note: These gaps are LIMITED by constraint availability above.")
    log("      Currently can only detect: single_source_only, quarantine attachment")
    log("")

    # Detect surface-level meta-claims (limited by available constraints)
    detector = TensionDetector(
        claims={},
        surfaces=surfaces,
        edges=[],
        params=Parameters()
    )
    meta_claims = detector.detect_all()

    # Get multi-surface events sorted by size
    events_by_size = sorted(
        [(eid, e) for eid, e in result.events.items() if len(e.surface_ids) > 1],
        key=lambda x: -len(x[1].surface_ids)
    )

    event_gaps = []
    for eid, event in events_by_size:
        gaps = analyze_event_gaps(event, surfaces, meta_claims, page_titles)
        event_gaps.append(gaps)

    # Sort by inquiry priority
    event_gaps.sort(key=lambda g: -g.inquiry_priority)

    # Display top events
    for i, gaps in enumerate(event_gaps[:5], 1):
        log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log(f"#{i} {gaps.topic}")
        log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log(f"   Surfaces: {gaps.surface_count} | Sources: {gaps.source_count}")
        log(f"   Corroboration: {gaps.corroboration_rate:.0%} multi-source")

        # Membership breakdown
        event = result.events.get(gaps.event_id)
        if event:
            core = len(event.core_surfaces())
            periphery = len(event.periphery_surfaces())
            quarantine = len(event.quarantine_surfaces())
            log(f"   Membership: {core} core, {periphery} periphery, {quarantine} quarantine")

        log("")
        log("   DETECTABLE GAPS (limited by constraint availability):")

        if gaps.single_source_surfaces:
            n = len(gaps.single_source_surfaces)
            log(f"   • {n}/{gaps.surface_count} single-source (need corroboration)")

        if gaps.quarantine_surfaces:
            log(f"   • {len(gaps.quarantine_surfaces)} quarantine (weak anchor-only attachment)")

        if gaps.typed_value_conflicts:
            log(f"   • {len(gaps.typed_value_conflicts)} typed conflicts")

        if not gaps.single_source_surfaces and not gaps.quarantine_surfaces and not gaps.typed_value_conflicts:
            log("   ✅ No detectable gaps (but may have undetectable ones)")

        log("")

    # Final summary
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("")
    log("CONSTRAINT SCARCITY is the primary blocker, not 'no gaps'.")
    log("")
    log("Current state:")
    log(f"  • Typed coverage: {avail['typed_coverage_rate']:.0%} (need >5% for conflict detection)")
    log(f"  • Time coverage: {avail['time_coverage_rate']:.0%} (need >50% for temporal binding)")
    log(f"  • Semantic coverage: {avail['semantic_coverage_rate']:.0%} (need >10% for similarity)")
    log(f"  • Quarantine rate: {binding['quarantine_rate']:.0%} (want <30%)")
    log("")
    log("The RIGHT proto-inquiries are:")
    log("  1. Extract typed variables (unlocks typed_value_conflict)")
    log("  2. Resolve timestamps (unlocks temporal binding)")
    log("  3. Compute embeddings (unlocks semantic similarity)")
    log("  4. THEN ask for corroboration (once scope is stable)")


if __name__ == "__main__":
    asyncio.run(run_semantic_event_gaps())
