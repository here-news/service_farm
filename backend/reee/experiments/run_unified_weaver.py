#!/usr/bin/env python3
"""
Run Unified Weaver Experiment

Tests the new Universal Logic Flow framework on WFC incidents.
Computes incident embeddings by aggregating claim embeddings via surfaces.
"""

import os
import sys
import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Set

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from openai import AsyncOpenAI
from services.neo4j_service import Neo4jService
from reee.experiments.unified_weaver import UnifiedWeaver, ActionType


async def load_incident_embeddings(
    neo4j: Neo4jService,
    db_pool,
    incident_ids: List[str]
) -> Dict[str, np.ndarray]:
    """
    Compute incident embeddings by aggregating claim embeddings.

    Path: Incident → Surfaces → Claims → Embeddings
    Aggregation: Mean of claim embeddings per incident
    """
    embeddings = {}

    # Get incident -> surface -> claim mapping from Neo4j
    query = """
    UNWIND $incident_ids as iid
    MATCH (i:Incident {id: iid})-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
    RETURN iid as incident_id, collect(DISTINCT c.id) as claim_ids
    """
    results = await neo4j._execute_read(query, {"incident_ids": incident_ids})

    # Build incident -> claims map
    incident_claims = {}
    all_claim_ids = set()
    for r in results:
        iid = r["incident_id"]
        claims = r.get("claim_ids", []) or []
        if claims:
            incident_claims[iid] = claims
            all_claim_ids.update(claims)

    if not all_claim_ids:
        print("  [warn] No claims found via surface relationships")
        return embeddings

    # Load claim embeddings from PostgreSQL
    claim_embeddings = {}
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT claim_id, embedding
            FROM core.claim_embeddings
            WHERE claim_id = ANY($1)
        """, list(all_claim_ids))

        import json as _json
        for row in rows:
            if row["embedding"]:
                # pgvector returns as string - parse it
                emb_data = row["embedding"]
                if isinstance(emb_data, str):
                    # Parse pgvector string format "[0.1,0.2,...]"
                    emb_data = _json.loads(emb_data)
                claim_embeddings[row["claim_id"]] = np.array(emb_data, dtype=np.float32)

    print(f"  Loaded {len(claim_embeddings)}/{len(all_claim_ids)} claim embeddings")

    # Aggregate claim embeddings into incident embeddings (mean)
    for iid, claim_ids in incident_claims.items():
        embs = [claim_embeddings[cid] for cid in claim_ids if cid in claim_embeddings]
        if embs:
            embeddings[iid] = np.mean(embs, axis=0)

    print(f"  Computed {len(embeddings)}/{len(incident_ids)} incident embeddings")

    return embeddings


async def load_wfc_incidents(
    neo4j: Neo4jService,
    db_pool,
    limit: int = 50
) -> List[Dict]:
    """Load WFC incidents with aggregated embeddings."""

    # Get incidents from Neo4j
    query = """
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN i.id as id, i.summary as summary, i.anchor_entities as entities
    ORDER BY i.id
    LIMIT $limit
    """
    results = await neo4j._execute_read(query, {"limit": limit})

    incidents = []
    for r in results:
        incidents.append({
            "id": r["id"],
            "summary": r.get("summary", "") or "",
            "entities": set(r.get("entities", []) or []),
            "embedding": None,
        })

    # Load embeddings via claim aggregation
    if incidents and db_pool:
        print(f"Loading embeddings for {len(incidents)} incidents...")
        incident_ids = [i["id"] for i in incidents]
        emb_map = await load_incident_embeddings(neo4j, db_pool, incident_ids)

        for inc in incidents:
            inc["embedding"] = emb_map.get(inc["id"])

    return incidents


async def run_experiment():
    print("=" * 70)
    print("UNIFIED WEAVER EXPERIMENT")
    print("=" * 70)
    print("Framework: Perceive → Route → Evaluate → Compile → Execute → Emit")
    print()

    # Connect to services
    neo4j = Neo4jService()
    await neo4j.connect()

    import asyncpg
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD'),
    )

    llm_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Load WFC incidents with embeddings
    print("Loading WFC incidents...")
    incidents = await load_wfc_incidents(neo4j, db_pool, limit=50)
    print(f"Loaded {len(incidents)} incidents")

    emb_count = sum(1 for i in incidents if i.get("embedding") is not None)
    print(f"Incidents with embeddings: {emb_count}/{len(incidents)}")
    print()

    # Initialize weaver with LLM
    weaver = UnifiedWeaver(llm_client=llm_client)

    # Process incidents
    print("Processing incidents through unified weaver...")
    print("-" * 70)

    for i, inc in enumerate(incidents, 1):
        action = await weaver.process_incident(
            incident_id=inc["id"],
            summary=inc["summary"],
            entities=inc["entities"],
            embedding=inc.get("embedding"),
            time=datetime.now(),  # Would use real time in production
        )

        if i % 5 == 0 or i == len(incidents):
            summary = weaver.summary()
            print(f"  [{i:3d}/{len(incidents)}] "
                  f"cases={summary['cases']} "
                  f"A={summary['actions']['assimilate']} "
                  f"L={summary['actions']['link']} "
                  f"D={summary['actions']['defer']} "
                  f"R={summary['actions']['reject']}")

    print()

    # REPAIR PHASE: LLM-gated promotion (no entity-overlap heuristics)
    print("Running LLM-gated metabolic edge stitching...")
    print("(Promotions decided by typed compatibility, not entity overlap)")
    stitch_result = await weaver.stitch_via_metabolic_edges(max_rounds=5, min_confidence=0.7)
    print(f"Repair rounds: {stitch_result['rounds']}")
    print(f"Promotions: {stitch_result['merges']} spine edges")
    print(f"LLM rejected: {stitch_result['skipped_llm_reject']}")
    print(f"Low confidence: {stitch_result['skipped_low_confidence']}")
    if stitch_result['promotions']:
        print("Sample promotions:")
        for p in stitch_result['promotions'][:3]:
            print(f"  {p['relation']} (conf={p['confidence']:.2f}): {p['reasoning'][:60]}...")
    print()

    # Skip growth re-evaluation for now (needs LLM gate too)
    print("(Growth re-evaluation skipped - needs LLM gate)")
    print()

    # Final summary
    summary = weaver.summary()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"Incidents processed: {summary['incidents']}")
    print(f"Cases formed: {summary['cases']}")
    print(f"Metabolic edges: {summary['metabolic_edges']}")
    print(f"Deferred decisions: {summary['deferred']}")
    print(f"Inquiries emitted: {summary['inquiries']}")
    print()
    print("Action breakdown:")
    print(f"  ASSIMILATE (spine): {summary['actions']['assimilate']}")
    print(f"  LINK (metabolic):   {summary['actions']['link']}")
    print(f"  DEFER (uncertain):  {summary['actions']['defer']}")
    print(f"  REJECT (new case):  {summary['actions']['reject']}")
    print()

    # WFC analysis
    wfc_incident_ids = {inc["id"] for inc in incidents if "Wang Fuk Court" in inc["entities"]}

    # Find case containing most WFC incidents
    membership = weaver.get_case_membership()
    best_case = None
    best_overlap = 0

    for case_id, incident_ids in membership.items():
        overlap = len(incident_ids & wfc_incident_ids)
        if overlap > best_overlap:
            best_overlap = overlap
            best_case = case_id

    if best_case:
        case_incidents = membership[best_case]
        wfc_in_case = len(case_incidents & wfc_incident_ids)
        purity = wfc_in_case / len(case_incidents) if case_incidents else 0
        coverage = wfc_in_case / len(wfc_incident_ids) if wfc_incident_ids else 0

        print("WFC Case Analysis:")
        print(f"  Best case: {best_case}")
        print(f"  Case size: {len(case_incidents)}")
        print(f"  WFC incidents: {wfc_in_case}")
        print(f"  Purity: {purity:.1%}")
        print(f"  Coverage: {coverage:.1%}")

        contaminants = case_incidents - wfc_incident_ids
        if contaminants:
            print(f"  Contaminants: {len(contaminants)}")
        else:
            print("  No contaminants!")

    print()

    # Show all cases
    print("All cases:")
    for case_id, incident_ids in sorted(membership.items(), key=lambda x: -len(x[1])):
        case = weaver.cases[case_id]
        entities = sorted(list(case.entities))[:3]
        print(f"  {case_id}: {len(incident_ids)} incidents - {entities}")

    print()
    print("=" * 70)
    print("CI METRICS")
    print("=" * 70)
    print(f"  incidents_processed: {len(incidents)}")
    print(f"  cases_formed: {summary['cases']}")
    print(f"  assimilations: {summary['actions']['assimilate']}")
    print(f"  deferrals: {summary['deferred']}")
    if best_case:
        print(f"  wfc_purity: {purity:.3f}")
        print(f"  wfc_coverage: {coverage:.3f}")

    await db_pool.close()
    await neo4j.close()

    print()
    print("Experiment complete!")


if __name__ == "__main__":
    asyncio.run(run_experiment())
