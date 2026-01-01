"""
Event Emergence Experiment
==========================

Tests whether REEE can emerge events from raw claims to replace legacy event worker.

Pipeline:
1. Load N claims from Neo4j (ignoring legacy event attachments)
2. Run IdentityLinker to form surfaces (L0 → L2)
3. Run AboutnessScorer to compute aboutness edges (L2 → L2)
4. Cluster surfaces into emerged events (L2 → L3)
5. Compare emerged events to legacy events (ground truth)

Goal: Validate that weaver-emerged events match or improve upon legacy events.

Run:
    docker exec herenews-app python -m reee.experiments.event_emergence
    docker exec herenews-app python -m reee.experiments.event_emergence --claims 500
"""

import asyncio
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

from reee import Claim, Parameters
from reee.types import Surface, Event
from reee.identity import IdentityLinker
from reee.aboutness.scorer import AboutnessScorer, compute_events_from_aboutness
from reee.aboutness.metrics import evaluate_clustering, print_evaluation_report
from reee.experiments.loader import create_context, log


@dataclass
class LegacyEvent:
    """Legacy event from database."""
    id: str
    name: str
    claim_ids: Set[str]


@dataclass
class EmergenceResult:
    """Result of event emergence experiment."""
    # Input
    total_claims: int
    claims_with_embeddings: int
    claims_with_entities: int

    # L2: Surfaces
    surfaces_formed: int
    identity_edges: int
    avg_claims_per_surface: float

    # L3: Events
    events_emerged: int
    aboutness_edges: int
    avg_surfaces_per_event: float

    # Comparison
    legacy_events: int
    precision: float  # emerged events that match legacy
    recall: float     # legacy events that were recovered

    # Largest emerged events
    top_events: List[Tuple[str, int, Set[str]]]  # (event_id, claim_count, anchor_entities)


async def load_all_claims(ctx, limit: int = 200) -> Tuple[List[Claim], Dict[str, str]]:
    """
    Load claims from Neo4j ignoring legacy event attachments.

    Returns:
        (claims, claim_to_legacy_event) - claims and their legacy event mapping for comparison
    """
    from pgvector.asyncpg import register_vector
    from datetime import datetime

    # Hub locations that should not count as anchor entities
    HUB_LOCATIONS = {'Hong Kong', 'China', 'United States', 'UK', 'United Kingdom', 'US'}

    # Get claims with their legacy event (if any) - include pub_time for temporal gate
    claims_data = await ctx.neo4j._execute_read('''
        MATCH (c:Claim)
        WHERE c.text IS NOT NULL
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        OPTIONAL MATCH (e:Event)-[:INTAKES]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        WITH c, p, e, collect({name: ent.canonical_name, type: ent.entity_type}) as entities
        RETURN c.id as id, c.text as text, p.domain as source,
               entities, e.id as legacy_event_id, e.canonical_name as legacy_event_name,
               p.pub_time as pub_time, c.event_time as event_time
        LIMIT $limit
    ''', {'limit': limit})

    claims = []
    claim_to_legacy = {}

    async with ctx.db_pool.acquire() as conn:
        await register_vector(conn)

        for row in claims_data:
            if not row['text']:
                continue

            # Get embedding
            embedding = await conn.fetchval(
                'SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1',
                row['id']
            )

            # Process entities
            all_entities = set()
            anchor_entities = set()

            for ent in row['entities']:
                ent_name = ent.get('name')
                if ent_name:
                    all_entities.add(ent_name)
                    ent_type = ent.get('type')

                    # Anchor: PERSON, ORG, or non-hub LOCATION
                    if ent_type in ('PERSON', 'ORGANIZATION', 'ORG'):
                        anchor_entities.add(ent_name)
                    elif ent_type == 'LOCATION' and ent_name not in HUB_LOCATIONS:
                        anchor_entities.add(ent_name)

            # Convert embedding
            emb = None
            if embedding is not None:
                try:
                    if len(embedding) > 0:
                        emb = [float(x) for x in embedding]
                except:
                    pass

            # Parse timestamp (prefer pub_time, fall back to event_time)
            timestamp = None
            pub_time = row.get('pub_time')
            event_time = row.get('event_time')

            if pub_time:
                try:
                    if isinstance(pub_time, str):
                        timestamp = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                    else:
                        timestamp = pub_time
                except:
                    pass

            if not timestamp and event_time:
                try:
                    if isinstance(event_time, str):
                        timestamp = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                    else:
                        timestamp = event_time
                except:
                    pass

            claim = Claim(
                id=row['id'],
                text=row['text'],
                source=row['source'] or 'unknown',
                embedding=emb,
                entities=all_entities,
                anchor_entities=anchor_entities,
                timestamp=timestamp
            )
            claims.append(claim)

            if row['legacy_event_id']:
                claim_to_legacy[claim.id] = row['legacy_event_name'] or row['legacy_event_id']

    return claims, claim_to_legacy


async def load_legacy_events(ctx) -> Dict[str, LegacyEvent]:
    """Load legacy events with their claim IDs."""
    events_data = await ctx.neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WITH e, collect(c.id) as claim_ids
        RETURN e.id as id, e.canonical_name as name, claim_ids
    ''')

    return {
        e['id']: LegacyEvent(
            id=e['id'],
            name=e['name'],
            claim_ids=set(e['claim_ids'])
        )
        for e in events_data
    }


def compute_cluster_overlap(
    emerged_events: Dict[str, Event],
    legacy_events: Dict[str, LegacyEvent],
    surfaces: Dict[str, Surface],
) -> Tuple[float, float, Dict]:
    """
    Compute precision/recall between emerged and legacy events.

    Uses claim overlap to measure alignment.
    """
    # Map emerged events to their claim IDs
    emerged_claim_sets = {}
    for event_id, event in emerged_events.items():
        claims = set()
        for surface_id in event.surface_ids:
            if surface_id in surfaces:
                claims.update(surfaces[surface_id].claim_ids)
        emerged_claim_sets[event_id] = claims

    # For each legacy event, find best matching emerged event
    legacy_matches = {}
    for legacy_id, legacy in legacy_events.items():
        best_match = None
        best_jaccard = 0.0

        for emerged_id, emerged_claims in emerged_claim_sets.items():
            overlap = len(legacy.claim_ids & emerged_claims)
            union = len(legacy.claim_ids | emerged_claims)
            if union > 0:
                jaccard = overlap / union
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_match = emerged_id

        if best_match and best_jaccard > 0.3:  # Threshold for "match"
            legacy_matches[legacy_id] = (best_match, best_jaccard)

    # Precision: what fraction of emerged events match some legacy event
    emerged_matched = set(m[0] for m in legacy_matches.values())
    precision = len(emerged_matched) / len(emerged_events) if emerged_events else 0.0

    # Recall: what fraction of legacy events were recovered
    recall = len(legacy_matches) / len(legacy_events) if legacy_events else 0.0

    return precision, recall, legacy_matches


async def run_experiment(num_claims: int = 200):
    """Run the event emergence experiment."""
    log("=" * 70)
    log("Event Emergence Experiment")
    log("=" * 70)
    log(f"Testing with {num_claims} claims")
    log("")

    ctx = await create_context()

    try:
        # 1. Load claims
        log("📚 Loading claims from database...")
        claims, claim_to_legacy = await load_all_claims(ctx, limit=num_claims)

        claims_with_emb = sum(1 for c in claims if c.embedding)
        claims_with_ent = sum(1 for c in claims if c.entities)

        log(f"   Loaded: {len(claims)} claims")
        log(f"   With embeddings: {claims_with_emb}")
        log(f"   With entities: {claims_with_ent}")
        log(f"   Attached to legacy events: {len(claim_to_legacy)}")

        # 2. Load legacy events for comparison
        log("")
        log("📦 Loading legacy events for comparison...")
        legacy_events = await load_legacy_events(ctx)
        log(f"   Legacy events: {len(legacy_events)}")
        for eid, ev in list(legacy_events.items())[:5]:
            log(f"   - {ev.name}: {len(ev.claim_ids)} claims")

        # 3. Run IdentityLinker to form surfaces
        log("")
        log("🧬 Running IdentityLinker (L0 → L2)...")
        params = Parameters(
            identity_confidence_threshold=0.45,
            hub_max_df=5,
            aboutness_min_signals=2,
            aboutness_threshold=0.15,
        )

        linker = IdentityLinker(llm=None, params=params)
        claims_dict = {c.id: c for c in claims}

        for claim in claims:
            await linker.add_claim(claim, extract_qkey=False)

        surfaces = linker.compute_surfaces()

        log(f"   Surfaces formed: {len(surfaces)}")
        log(f"   Identity edges: {len(linker.edges)}")

        if surfaces:
            avg_claims = sum(len(s.claim_ids) for s in surfaces.values()) / len(surfaces)
            log(f"   Avg claims/surface: {avg_claims:.1f}")

        # Show top surfaces
        sorted_surfaces = sorted(surfaces.values(), key=lambda s: len(s.claim_ids), reverse=True)
        log("")
        log("   Top surfaces by claim count:")
        for s in sorted_surfaces[:5]:
            anchors = list(s.anchor_entities)[:3]
            log(f"   - {s.id}: {len(s.claim_ids)} claims, anchors: {anchors}")

        # 4. Run AboutnessScorer (L2 → L2)
        log("")
        log("🔗 Computing aboutness edges (L2 → L2)...")
        scorer = AboutnessScorer(surfaces, params)
        aboutness_edges = scorer.compute_all_edges()

        log(f"   Aboutness edges: {len(aboutness_edges)}")

        # Show sample edges
        if aboutness_edges:
            log("   Sample edges:")
            for s1, s2, score, evidence in aboutness_edges[:3]:
                signals = evidence.get('signals_met', 0)
                log(f"   - {s1[:8]} ↔ {s2[:8]}: score={score:.2f}, signals={signals}")

        # 5. Cluster into events (L2 → L3)
        log("")
        log("🌐 Clustering surfaces into events (L2 → L3)...")
        emerged_events = compute_events_from_aboutness(surfaces, aboutness_edges, params)

        log(f"   Events emerged: {len(emerged_events)}")

        if emerged_events:
            avg_surfaces = sum(len(e.surface_ids) for e in emerged_events.values()) / len(emerged_events)
            log(f"   Avg surfaces/event: {avg_surfaces:.1f}")

        # Show top emerged events
        sorted_events = sorted(
            emerged_events.values(),
            key=lambda e: e.total_claims,
            reverse=True
        )

        log("")
        log("   Top emerged events:")
        for e in sorted_events[:8]:
            anchors = list(e.anchor_entities)[:4]
            log(f"   - {e.id}: {e.total_claims} claims, {len(e.surface_ids)} surfaces")
            log(f"     Anchors: {anchors}")

        # 6. B³ + Purity/Completeness Evaluation
        log("")
        log("=" * 70)
        log("B³ + PURITY/COMPLETENESS EVALUATION")
        log("=" * 70)

        eval_result = evaluate_clustering(emerged_events, surfaces, claim_to_legacy)

        log("")
        log("B³ METRICS (claim-level, proper clustering evaluation):")
        log(f"   Precision: {eval_result.b3.precision:.1%}")
        log(f"   Recall:    {eval_result.b3.recall:.1%}")
        log(f"   F1:        {eval_result.b3.f1:.1%}")
        log("")
        log("PURITY / COMPLETENESS (event-level):")
        log(f"   Purity:      {eval_result.purity.purity:.1%} (emerged events are clean)")
        log(f"   Completeness:{eval_result.purity.completeness:.1%} (legacy events are captured)")
        log("")

        # Per-legacy-event breakdown
        log("Per-legacy B³ breakdown:")
        for name, prec in sorted(eval_result.b3.event_precision.items(), key=lambda x: -x[1])[:8]:
            rec = eval_result.b3.event_recall.get(name, 0.0)
            log(f"   {name[:35]:35s}: P={prec:.0%} R={rec:.0%}")

        # Also show old exact-match metrics for comparison
        log("")
        log("(Exact-match metrics for reference - expected to be low):")
        precision, recall, matches = compute_cluster_overlap(
            emerged_events, legacy_events, surfaces
        )
        log(f"   Exact Precision: {precision:.1%}")
        log(f"   Exact Recall: {recall:.1%}")

        # 7. Detailed analysis of large emerged events
        log("")
        log("=" * 70)
        log("DETAILED: Large Emerged Events")
        log("=" * 70)

        for e in sorted_events[:5]:
            log("")
            log(f"[{e.id}] {e.total_claims} claims, {len(e.surface_ids)} surfaces")
            log(f"    Anchors: {list(e.anchor_entities)[:5]}")
            log(f"    Sources: {e.total_sources}")

            # Check which legacy events these claims belong to
            event_claims = set()
            for sid in e.surface_ids:
                if sid in surfaces:
                    event_claims.update(surfaces[sid].claim_ids)

            legacy_dist = defaultdict(int)
            for cid in event_claims:
                if cid in claim_to_legacy:
                    legacy_dist[claim_to_legacy[cid]] += 1

            if legacy_dist:
                log(f"    Legacy event distribution:")
                for leg_name, count in sorted(legacy_dist.items(), key=lambda x: -x[1])[:3]:
                    log(f"      - {leg_name}: {count} claims")

        # 8. Summary
        log("")
        log("=" * 70)
        log("SUMMARY")
        log("=" * 70)
        log(f"   Input claims: {len(claims)}")
        log(f"   Surfaces formed: {len(surfaces)}")
        log(f"   Events emerged: {len(emerged_events)}")
        log(f"   Legacy events: {len(legacy_events)}")
        log("")
        log("   B³ METRICS:")
        log(f"      Precision: {eval_result.b3.precision:.1%}")
        log(f"      Recall:    {eval_result.b3.recall:.1%}")
        log(f"      F1:        {eval_result.b3.f1:.1%}")
        log("")
        log("   PURITY/COMPLETENESS:")
        log(f"      Purity:      {eval_result.purity.purity:.1%}")
        log(f"      Completeness:{eval_result.purity.completeness:.1%}")
        log("")

        # Interpret results using B³ F1 and purity
        b3_f1 = eval_result.b3.f1
        purity = eval_result.purity.purity

        if b3_f1 > 0.7 and purity > 0.85:
            log("✅ Event emergence is viable!")
            log("   High B³ F1 and purity - clusters match legacy events well.")
        elif b3_f1 > 0.5 or purity > 0.8:
            log("⚠️  Partial success - reasonable clustering quality.")
            log("   May need parameter tuning for better recall/completeness.")
        else:
            log("❌ Low B³ F1 - investigate pipeline.")
            log("   Check: over-fragmentation (low recall) or over-merging (low precision).")

    finally:
        await ctx.close()


def main():
    parser = argparse.ArgumentParser(description='Event Emergence Experiment')
    parser.add_argument('--claims', type=int, default=200, help='Number of claims to process')
    args = parser.parse_args()

    asyncio.run(run_experiment(num_claims=args.claims))


if __name__ == "__main__":
    main()
