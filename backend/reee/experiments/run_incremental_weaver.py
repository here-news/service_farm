#!/usr/bin/env python3
"""
Run Incremental Weaver Experiment on Real Claims

This script loads claims from Neo4j/PostgreSQL and processes them through
the incremental weaver to test top-down routing with bottom-up growth.

Usage:
    # Inside docker:
    python -m reee.experiments.run_incremental_weaver [--wfc-only] [--limit N]

    # From host:
    docker exec herenews-app python -m reee.experiments.run_incremental_weaver
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Set, Optional

sys.path.insert(0, '/app/backend')

from reee.experiments.incremental_weaver import (
    IncrementalWeaver,
    ClaimArtifact,
    WeaverConfig,
    RouteLevel,
)


# Hub locations that should not count as anchor entities (from loader.py)
HUB_LOCATIONS = frozenset({
    'Hong Kong', 'China', 'United States', 'UK', 'United Kingdom', 'US',
    'New York', 'Washington', 'Beijing', 'London'
})


async def load_wfc_claims_from_neo4j(neo4j, limit: int = 50) -> List[Dict]:
    """
    Load Wang Fuk Court claims from Neo4j.

    The data model:
    - Incident has anchor_entities property (list of entity names)
    - Incident -[:CONTAINS]-> Surface
    - Surface -[:CONTAINS]-> Claim (claims are nodes, not property!)
    - Claim has text, event_time
    - Page -[:EMITS]-> Claim for source info

    Returns claims with:
    - id, text, entities, anchor_entities, event_time, source
    """
    # Get claims directly via relationship chain
    query = """
    MATCH (i:Incident)-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
    RETURN c.id as id,
           c.text as text,
           c.event_time as event_time,
           p.domain as source,
           s.anchor_entities as surface_anchors,
           i.anchor_entities as incident_anchors,
           s.time_start as time_start,
           s.sources as sources,
           i.id as incident_id,
           s.id as surface_id,
           s.question_key as question_key
    ORDER BY s.time_start, c.event_time
    LIMIT $limit
    """
    results = await neo4j._execute_read(query, {"limit": limit})

    claims = []
    seen_ids = set()

    for r in results:
        claim_id = r.get("id")
        if not claim_id or claim_id in seen_ids:
            continue
        seen_ids.add(claim_id)

        if not r.get("text"):
            continue

        # Combine anchors from incident and surface
        incident_anchors = set(r.get("incident_anchors") or [])
        surface_anchors = set(r.get("surface_anchors") or [])
        anchor_entities = incident_anchors | surface_anchors

        # Filter hub locations from anchors
        anchor_entities = {e for e in anchor_entities if e not in HUB_LOCATIONS}

        # Parse event_time
        event_time = None
        raw_time = r.get("event_time") or r.get("time_start")
        if raw_time:
            try:
                if isinstance(raw_time, str):
                    event_time = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                elif hasattr(raw_time, "to_native"):
                    event_time = raw_time.to_native()
            except Exception:
                pass

        claims.append({
            "id": claim_id,
            "text": r.get("text", "")[:500],
            "entities": incident_anchors | surface_anchors,
            "anchor_entities": anchor_entities,
            "event_time": event_time,
            "source": r.get("source", "unknown"),
            "incident_id": r.get("incident_id"),
            "surface_id": r.get("surface_id"),
            "question_key": r.get("question_key"),
        })

    return claims


async def load_mixed_claims(neo4j, wfc_limit: int = 30, distractor_limit: int = 20) -> tuple:
    """
    Load WFC claims plus distractors for testing membrane isolation.

    Returns: (all_claims, wfc_claim_ids)
    """
    # Load WFC claims
    wfc_claims = await load_wfc_claims_from_neo4j(neo4j, limit=wfc_limit)
    wfc_ids = {c["id"] for c in wfc_claims}

    # Load distractors (claims from non-WFC incidents) via correct relationship chain
    query = """
    MATCH (i:Incident)-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
    WHERE NOT 'Wang Fuk Court' IN i.anchor_entities
    OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
    RETURN c.id as id,
           c.text as text,
           i.anchor_entities as anchor_entities,
           s.time_start as event_time,
           p.domain as source,
           i.id as incident_id
    ORDER BY s.time_start
    LIMIT $limit
    """
    results = await neo4j._execute_read(query, {"limit": distractor_limit})

    distractor_claims = []
    seen_ids = set()

    for r in results:
        claim_id = r.get("id")
        if not claim_id or claim_id in seen_ids or claim_id in wfc_ids:
            continue
        seen_ids.add(claim_id)

        if not r.get("text"):
            continue

        anchor_entities = set(r.get("anchor_entities", []) or [])
        anchor_entities = {e for e in anchor_entities if e not in HUB_LOCATIONS}

        event_time = None
        if r.get("event_time"):
            try:
                if isinstance(r["event_time"], str):
                    event_time = datetime.fromisoformat(r["event_time"].replace("Z", "+00:00"))
                elif hasattr(r["event_time"], "to_native"):
                    event_time = r["event_time"].to_native()
            except Exception:
                pass

        distractor_claims.append({
            "id": claim_id,
            "text": r.get("text", "")[:500],
            "entities": set(r.get("anchor_entities", []) or []),
            "anchor_entities": anchor_entities,
            "event_time": event_time,
            "source": r.get("source", "unknown"),
            "incident_id": r.get("incident_id"),
        })

    # Interleave by time
    all_claims = wfc_claims + distractor_claims
    all_claims.sort(key=lambda c: c.get("event_time") or datetime.min)

    return all_claims, wfc_ids


def claim_to_artifact(claim: Dict) -> ClaimArtifact:
    """Convert a claim dict to ClaimArtifact."""
    anchor = frozenset(claim.get("anchor_entities", []))
    all_ents = frozenset(claim.get("entities", set()))

    # Referents = anchor entities (identity witnesses)
    # Contexts = everything else (ambient)
    referents = anchor
    contexts = all_ents - anchor

    # Derive proposition key from claim (simplified - in production use extractor)
    # For now, use a hash of the claim text's first phrase
    text = claim.get("text", "")
    prop_key = f"prop_{hash(text[:50]) % 10000:04d}"

    return ClaimArtifact(
        claim_id=claim["id"],
        text=text,
        referents=referents,
        contexts=contexts,
        anchor_entities=anchor,
        event_time=claim.get("event_time"),
        proposition_key=prop_key,
        source=claim.get("source", "unknown"),
        confidence=0.8,
        embedding=claim.get("embedding"),
    )


async def run_wfc_experiment(neo4j, limit: int = 30) -> Dict:
    """
    Run experiment specifically on Wang Fuk Court claims.

    Tests that claims with shared WFC anchor form a coherent case.
    """
    print("=" * 70)
    print("INCREMENTAL WEAVER EXPERIMENT: Wang Fuk Court Claims")
    print("=" * 70)
    print()

    # Load WFC claims
    claims = await load_wfc_claims_from_neo4j(neo4j, limit=limit)
    print(f"Loaded {len(claims)} claims containing 'Wang Fuk Court'")
    print()

    # Show claim arrival order
    print("Claim arrival order (by event_time):")
    print("-" * 60)
    for i, c in enumerate(claims[:10], 1):
        time_str = c["event_time"].strftime("%Y-%m-%d %H:%M") if c["event_time"] else "no-time"
        anchors = list(c["anchor_entities"])[:3]
        source = (c.get("source") or "unknown")[:12]
        print(f"  {i:2d}. [{time_str}] {source:12s} {anchors}")
    if len(claims) > 10:
        print(f"  ... and {len(claims) - 10} more")
    print()

    # Create weaver with hub filtering
    config = WeaverConfig(hub_entities=HUB_LOCATIONS)
    weaver = IncrementalWeaver(config=config)

    print("Processing claims through incremental weaver...")
    print("-" * 60)

    for i, claim in enumerate(claims, 1):
        artifact = claim_to_artifact(claim)
        result = weaver.process_claim(artifact)

        # Show progress every 5 claims
        if i % 5 == 0 or i == len(claims):
            summary = weaver.summary()
            print(f"  [{i:3d}/{len(claims)}] surfaces={summary['surfaces']} "
                  f"incidents={summary['incidents']} cases={summary['cases']} "
                  f"[{result.level.name}] {result.action[:30]}")

    print()

    # Final summary
    summary = weaver.summary()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"Claims processed: {len(claims)}")
    print(f"Surfaces created: {summary['surfaces']}")
    print(f"Incidents created: {summary['incidents']}")
    print(f"Cases formed: {summary['cases']}")
    print(f"Multi-incident cases: {summary['multi_incident_cases']}")
    print(f"Largest case: {summary['largest_case']} incidents")
    print()

    print("Routing decisions by level:")
    for level, count in summary["decisions_by_level"].items():
        print(f"  {level}: {count}")
    print()

    # Find WFC case
    wfc_case = weaver.get_case_for_entity("Wang Fuk Court")
    if wfc_case:
        print(f"WFC Case Analysis ({wfc_case.case_id[:12]}):")
        print(f"  Incidents: {len(wfc_case.incident_ids)}")
        print(f"  Membrane entities: {sorted(list(wfc_case.membrane))[:8]}")
        print(f"  Spine edges: {len(wfc_case.spine_edges)}")

        # Show incident details
        print("  Sample incidents:")
        for iid in list(wfc_case.incident_ids)[:5]:
            inc = weaver.incidents.get(iid)
            if inc:
                refs = sorted(list(inc.referents))[:3]
                print(f"    {iid[:8]}: {len(inc.surface_ids)} surfaces, refs={refs}")

    return {
        "claims": len(claims),
        "summary": summary,
        "wfc_case": wfc_case.case_id if wfc_case else None,
    }


async def run_mixed_experiment(neo4j, wfc_limit: int = 30, distractor_limit: int = 20) -> Dict:
    """
    Run experiment on mixed claims (WFC + distractors).

    Tests the weaver's ability to:
    1. Form a coherent WFC case
    2. Reject distractors (form separate cases)
    3. Not contaminate WFC case with unrelated claims
    """
    print("=" * 70)
    print("INCREMENTAL WEAVER EXPERIMENT: Mixed Claims (WFC + Distractors)")
    print("=" * 70)
    print()

    # Load mixed claims
    all_claims, wfc_ids = await load_mixed_claims(neo4j, wfc_limit, distractor_limit)

    print(f"Loaded {len(wfc_ids)} WFC claims + {len(all_claims) - len(wfc_ids)} distractors")
    print(f"Total: {len(all_claims)} claims")
    print()

    # Create weaver
    config = WeaverConfig(hub_entities=HUB_LOCATIONS)
    weaver = IncrementalWeaver(config=config)

    # Process all claims
    for claim in all_claims:
        artifact = claim_to_artifact(claim)
        weaver.process_claim(artifact)

    # Analyze results
    summary = weaver.summary()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"Claims processed: {len(all_claims)} ({len(wfc_ids)} WFC, {len(all_claims) - len(wfc_ids)} distractors)")
    print(f"Cases formed: {summary['cases']}")
    print(f"Largest case: {summary['largest_case']} incidents")
    print()

    # Find WFC case
    wfc_case = weaver.get_case_for_entity("Wang Fuk Court")

    if wfc_case:
        # Calculate purity/coverage
        case_claim_ids = set()
        for iid in wfc_case.incident_ids:
            inc = weaver.incidents.get(iid)
            if inc:
                for sid in inc.surface_ids:
                    surf = weaver.surfaces.get(sid)
                    if surf:
                        case_claim_ids.update(surf.claim_ids)

        wfc_in_case = case_claim_ids & wfc_ids
        purity = len(wfc_in_case) / len(case_claim_ids) if case_claim_ids else 0
        coverage = len(wfc_in_case) / len(wfc_ids) if wfc_ids else 0
        contaminants = case_claim_ids - wfc_ids

        print(f"WFC Case Analysis ({wfc_case.case_id[:12]}):")
        print(f"  Claims in case: {len(case_claim_ids)}")
        print(f"  WFC claims: {len(wfc_in_case)}")
        print(f"  Purity: {purity:.1%}")
        print(f"  Coverage: {coverage:.1%}")
        print(f"  Contaminants: {len(contaminants)}")
        print()

        if contaminants:
            print("  Contaminant claims (should be investigated):")
            for cid in list(contaminants)[:5]:
                # Find the claim in our data
                for c in all_claims:
                    if c["id"] == cid:
                        print(f"    {cid[:12]}: {list(c['anchor_entities'])[:3]}")
                        break
    else:
        print("No WFC case found!")
        purity = 0
        coverage = 0

    print()
    print("Routing decisions by level:")
    for level, count in summary["decisions_by_level"].items():
        print(f"  {level}: {count}")

    return {
        "claims": len(all_claims),
        "wfc_claims": len(wfc_ids),
        "summary": summary,
        "purity": purity if wfc_case else 0,
        "coverage": coverage if wfc_case else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run Incremental Weaver Experiment")
    parser.add_argument("--limit", type=int, default=30, help="Max WFC claims to process")
    parser.add_argument("--mixed", action="store_true", help="Run mixed (WFC + distractors) experiment")
    parser.add_argument("--distractor-limit", type=int, default=20, help="Max distractor claims")
    args = parser.parse_args()

    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        if args.mixed:
            result = await run_mixed_experiment(
                neo4j,
                wfc_limit=args.limit,
                distractor_limit=args.distractor_limit,
            )
        else:
            result = await run_wfc_experiment(neo4j, args.limit)

        print()
        print("Experiment complete!")
        print()

        # Summary metrics for CI
        print("CI Metrics:")
        print(f"  claims_processed: {result.get('claims', 0)}")
        print(f"  cases_formed: {result.get('summary', {}).get('cases', 0)}")
        print(f"  largest_case: {result.get('summary', {}).get('largest_case', 0)}")
        if "purity" in result:
            print(f"  wfc_purity: {result['purity']:.3f}")
            print(f"  wfc_coverage: {result['coverage']:.3f}")

    finally:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
