#!/usr/bin/env python3
"""
Run Mini-Weaver Experiment on Real Claims

This script loads claims from Neo4j/PostgreSQL and processes them through
the mini-weaver to test incremental organism growth.

Usage:
    python -m backend.reee.experiments.run_mini_weaver [--limit N] [--filter ENTITY]
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, '/app/backend')

from services.neo4j_service import Neo4jService
from reee.experiments.mini_weaver import MiniWeaver, run_experiment


async def load_claims_from_neo4j(
    neo4j: Neo4jService,
    limit: int = 50,
    filter_entity: Optional[str] = None,
    order_by_time: bool = True,
) -> List[Dict]:
    """
    Load claims from Neo4j for experiment.

    Args:
        limit: Max claims to load
        filter_entity: Only claims containing this entity
        order_by_time: If True, order by event_time (simulates arrival order)

    Returns:
        List of claim dicts ready for mini-weaver
    """
    # Build query
    if filter_entity:
        query = f"""
        MATCH (c:Claim)-[:MENTIONS]->(e:Entity)
        WHERE e.name = $filter_entity
        WITH c
        MATCH (c)-[:MENTIONS]->(e2:Entity)
        WITH c, collect(DISTINCT e2.name) as entities
        RETURN c.id as id,
               c.text as text,
               entities,
               c.event_time as event_time,
               c.source as source,
               c.page_id as page_id
        ORDER BY c.event_time
        LIMIT $limit
        """
        params = {"filter_entity": filter_entity, "limit": limit}
    else:
        query = """
        MATCH (c:Claim)-[:MENTIONS]->(e:Entity)
        WITH c, collect(DISTINCT e.name) as entities
        RETURN c.id as id,
               c.text as text,
               entities,
               c.event_time as event_time,
               c.source as source,
               c.page_id as page_id
        ORDER BY c.event_time
        LIMIT $limit
        """
        params = {"limit": limit}

    results = await neo4j._execute_read(query, params)

    claims = []
    for r in results:
        entities = set(r.get("entities", []))

        # Derive anchor entities (heuristic: first 2 most specific)
        # In production, this would use the extractor
        anchors = set(list(entities)[:2]) if len(entities) > 2 else entities

        # Parse event_time
        event_time = None
        if r.get("event_time"):
            try:
                if isinstance(r["event_time"], str):
                    event_time = datetime.fromisoformat(r["event_time"].replace("Z", "+00:00"))
                elif hasattr(r["event_time"], "to_native"):
                    event_time = r["event_time"].to_native()
            except Exception:
                pass

        claims.append({
            "id": r["id"],
            "text": r.get("text", "")[:500],  # Truncate long text
            "entities": entities,
            "anchor_entities": anchors,
            "event_time": event_time,
            "source": r.get("source", "unknown"),
        })

    return claims


async def run_wfc_experiment(neo4j: Neo4jService, limit: int = 30) -> Dict:
    """
    Run experiment specifically on Wang Fuk Court claims.

    This is our controlled Tier-B dataset for testing.
    """
    print("=" * 70)
    print("MINI-WEAVER EXPERIMENT: Wang Fuk Court Claims")
    print("=" * 70)
    print()

    # Load WFC claims
    claims = await load_claims_from_neo4j(
        neo4j,
        limit=limit,
        filter_entity="Wang Fuk Court",
        order_by_time=True,
    )

    print(f"Loaded {len(claims)} claims containing 'Wang Fuk Court'")
    print()

    # Show claim arrival order
    print("Claim arrival order (simulated):")
    print("-" * 60)
    for i, c in enumerate(claims[:10], 1):
        time_str = c["event_time"].strftime("%Y-%m-%d %H:%M") if c["event_time"] else "no-time"
        anchors = list(c["anchor_entities"])[:2]
        print(f"  {i:2d}. [{time_str}] {c['source'][:10]:10s} {anchors}")
    if len(claims) > 10:
        print(f"  ... and {len(claims) - 10} more")
    print()

    # Run the mini-weaver
    weaver = MiniWeaver()

    print("Processing claims through mini-weaver...")
    print("-" * 60)

    for i, claim in enumerate(claims, 1):
        result = weaver.process_claim(
            claim_id=claim["id"],
            text=claim["text"],
            entities=claim["entities"],
            anchor_entities=claim["anchor_entities"],
            event_time=claim["event_time"],
            source=claim["source"],
        )

        # Show progress every 5 claims
        if i % 5 == 0 or i == len(claims):
            summary = weaver.summary()
            print(f"  [{i:3d}/{len(claims)}] surfaces={summary['surfaces']} "
                  f"incidents={summary['incidents']} cases={summary['cases']} "
                  f"inquiries={summary['inquiries']}")

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
    print(f"Singletons: {summary['singletons']}")
    print(f"Largest case: {summary['largest_case']} incidents")
    print()

    print("L2 Actions:")
    for action, count in summary["actions"].items():
        if action.startswith("l2_"):
            print(f"  {action}: {count}")
    print()

    print("L3 Actions:")
    for action, count in summary["actions"].items():
        if action.startswith("l3_"):
            print(f"  {action}: {count}")
    print()

    print("Signals detected:")
    for signal, count in summary["signals"].items():
        if count > 0:
            print(f"  {signal}: {count}")
    print()

    print(f"Inquiries emitted: {summary['inquiries']}")
    if weaver.inquiries:
        print("Sample inquiries:")
        for inq in weaver.inquiries[:5]:
            print(f"  [{inq.signal_type.name}] {inq.question[:50]}...")
    print()

    # Show case structure
    cases = weaver.get_case_membership()
    print("Case structure:")
    for case_id, incidents in sorted(cases.items(), key=lambda x: -len(x[1])):
        if len(incidents) >= 2:
            # Get unique anchors in this case
            case_anchors = set()
            for iid in incidents:
                if iid in weaver.incidents:
                    case_anchors.update(weaver.incidents[iid].referents)
            print(f"  Case {case_id[:8]}: {len(incidents)} incidents")
            print(f"    Anchors: {sorted(case_anchors)[:5]}")

    return {
        "claims": len(claims),
        "summary": summary,
        "cases": {k: list(v) for k, v in cases.items()},
    }


async def run_mixed_experiment(neo4j: Neo4jService, limit: int = 50) -> Dict:
    """
    Run experiment on mixed claims (WFC + distractors).

    This tests the membrane's ability to reject unrelated claims.
    """
    print("=" * 70)
    print("MINI-WEAVER EXPERIMENT: Mixed Claims (WFC + Distractors)")
    print("=" * 70)
    print()

    # Load WFC claims
    wfc_claims = await load_claims_from_neo4j(
        neo4j,
        limit=limit // 2,
        filter_entity="Wang Fuk Court",
        order_by_time=True,
    )

    # Load distractor claims (not WFC)
    query = """
    MATCH (c:Claim)-[:MENTIONS]->(e:Entity)
    WHERE NOT (c)-[:MENTIONS]->(:Entity {name: 'Wang Fuk Court'})
    WITH c, collect(DISTINCT e.name) as entities
    WHERE size(entities) >= 2
    RETURN c.id as id,
           c.text as text,
           entities,
           c.event_time as event_time,
           c.source as source
    ORDER BY c.event_time
    LIMIT $limit
    """
    distractor_results = await neo4j._execute_read(query, {"limit": limit - len(wfc_claims)})

    distractor_claims = []
    for r in distractor_results:
        entities = set(r.get("entities", []))
        anchors = set(list(entities)[:2])

        event_time = None
        if r.get("event_time"):
            try:
                if isinstance(r["event_time"], str):
                    event_time = datetime.fromisoformat(r["event_time"].replace("Z", "+00:00"))
            except Exception:
                pass

        distractor_claims.append({
            "id": r["id"],
            "text": r.get("text", "")[:500],
            "entities": entities,
            "anchor_entities": anchors,
            "event_time": event_time,
            "source": r.get("source", "unknown"),
        })

    print(f"Loaded {len(wfc_claims)} WFC claims + {len(distractor_claims)} distractors")

    # Interleave claims by time (simulate real arrival)
    all_claims = wfc_claims + distractor_claims
    all_claims.sort(key=lambda c: c["event_time"] or datetime.min)

    print(f"Total: {len(all_claims)} claims")
    print()

    # Tag claims for ground truth
    wfc_ids = {c["id"] for c in wfc_claims}

    # Run mini-weaver
    weaver = MiniWeaver()

    for claim in all_claims:
        weaver.process_claim(
            claim_id=claim["id"],
            text=claim["text"],
            entities=claim["entities"],
            anchor_entities=claim["anchor_entities"],
            event_time=claim["event_time"],
            source=claim["source"],
        )

    # Analyze WFC purity
    cases = weaver.get_case_membership()

    # Find case containing most WFC claims
    wfc_case = None
    wfc_case_id = None
    for case_id, incidents in cases.items():
        case_claims = set()
        for iid in incidents:
            if iid in weaver.incidents:
                for sid in weaver.incidents[iid].surface_ids:
                    if sid in weaver.surfaces:
                        case_claims.update(weaver.surfaces[sid].claim_ids)

        wfc_in_case = case_claims & wfc_ids
        if len(wfc_in_case) > 0:
            if wfc_case is None or len(wfc_in_case) > len(wfc_case & wfc_ids):
                wfc_case = case_claims
                wfc_case_id = case_id

    summary = weaver.summary()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"Total claims: {len(all_claims)} ({len(wfc_claims)} WFC, {len(distractor_claims)} distractors)")
    print(f"Cases formed: {summary['cases']}")
    print(f"Largest case: {summary['largest_case']} incidents")
    print()

    if wfc_case:
        wfc_in_case = wfc_case & wfc_ids
        purity = len(wfc_in_case) / len(wfc_case) if wfc_case else 0
        coverage = len(wfc_in_case) / len(wfc_ids) if wfc_ids else 0
        contamination = wfc_case - wfc_ids

        print(f"WFC Case Analysis ({wfc_case_id[:8] if wfc_case_id else 'N/A'}):")
        print(f"  Claims in case: {len(wfc_case)}")
        print(f"  WFC claims: {len(wfc_in_case)}")
        print(f"  Purity: {purity:.1%}")
        print(f"  Coverage: {coverage:.1%}")
        print(f"  Contaminants: {len(contamination)}")

    print()
    print("L3 Decisions (membrane effectiveness):")
    print(f"  MERGE (spine): {summary['actions']['l3_merge']}")
    print(f"  PERIPHERY (metabolic): {summary['actions']['l3_periphery']}")
    print(f"  REJECT (new incident): {summary['actions']['l3_reject']}")

    return {
        "claims": len(all_claims),
        "wfc_claims": len(wfc_claims),
        "distractor_claims": len(distractor_claims),
        "summary": summary,
        "purity": purity if wfc_case else 0,
        "coverage": coverage if wfc_case else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run Mini-Weaver Experiment")
    parser.add_argument("--limit", type=int, default=30, help="Max claims to process")
    parser.add_argument("--mixed", action="store_true", help="Run mixed (WFC + distractors) experiment")
    args = parser.parse_args()

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        if args.mixed:
            result = await run_mixed_experiment(neo4j, args.limit)
        else:
            result = await run_wfc_experiment(neo4j, args.limit)

        print()
        print("Experiment complete!")

    finally:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
