#!/usr/bin/env python3
"""
Tier-B Validation: Unified Weaver with Distractors

Tests the unified weaver on WFC + 100 distractors to validate
that repair dynamics don't accidentally stitch contaminants.
"""

import os
import sys
import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Set

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from openai import AsyncOpenAI
from services.neo4j_service import Neo4jService
from reee.experiments.unified_weaver import UnifiedWeaver


async def load_incident_snippets_and_facets(
    neo4j: Neo4jService,
    incident_ids: List[str]
) -> Dict[str, Dict]:
    """
    Load claim snippets and detect facets for fingerprinting.

    Returns dict: incident_id -> {"snippets": [...], "facets": set(...)}
    """
    result = {iid: {"snippets": [], "facets": set()} for iid in incident_ids}

    # Get claim text via surfaces
    query = """
    UNWIND $incident_ids as iid
    MATCH (i:Incident {id: iid})-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
    RETURN iid as incident_id, collect(DISTINCT c.text)[..5] as claim_texts
    """
    results = await neo4j._execute_read(query, {"incident_ids": incident_ids})

    # Facet detection keywords
    FACET_KEYWORDS = {
        "death_toll": ["killed", "dead", "died", "fatalities", "death toll", "deaths"],
        "injuries": ["injured", "hospitalized", "wounded"],
        "evacuation": ["evacuated", "evacuation", "fled", "escape"],
        "arrests": ["arrested", "arrest", "detained", "custody"],
        "investigation": ["investigation", "investigating", "probe", "inquiry"],
        "fire": ["fire", "blaze", "flames", "burning"],
        "rescue": ["rescue", "rescued", "firefighters", "responders"],
        "cause": ["caused by", "cause of", "started by", "sparked by"],
    }

    for r in results:
        iid = r["incident_id"]
        claim_texts = r.get("claim_texts", []) or []

        # Store snippets (top claim texts)
        result[iid]["snippets"] = claim_texts[:3]

        # Detect facets from claim text
        combined_text = " ".join(claim_texts).lower()
        for facet, keywords in FACET_KEYWORDS.items():
            if any(kw in combined_text for kw in keywords):
                result[iid]["facets"].add(facet)

    return result


async def load_incident_embeddings(
    neo4j: Neo4jService,
    db_pool,
    incident_ids: List[str]
) -> Dict[str, np.ndarray]:
    """Load incident embeddings via claim aggregation."""
    embeddings = {}

    query = """
    UNWIND $incident_ids as iid
    MATCH (i:Incident {id: iid})-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
    RETURN iid as incident_id, collect(DISTINCT c.id) as claim_ids
    """
    results = await neo4j._execute_read(query, {"incident_ids": incident_ids})

    incident_claims = {}
    all_claim_ids = set()
    for r in results:
        iid = r["incident_id"]
        claims = r.get("claim_ids", []) or []
        if claims:
            incident_claims[iid] = claims
            all_claim_ids.update(claims)

    if not all_claim_ids:
        return embeddings

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
                emb_data = row["embedding"]
                if isinstance(emb_data, str):
                    emb_data = _json.loads(emb_data)
                claim_embeddings[row["claim_id"]] = np.array(emb_data, dtype=np.float32)

    for iid, claim_ids in incident_claims.items():
        embs = [claim_embeddings[cid] for cid in claim_ids if cid in claim_embeddings]
        if embs:
            embeddings[iid] = np.mean(embs, axis=0)

    return embeddings


async def load_tier_b_incidents(
    neo4j: Neo4jService,
    db_pool,
) -> tuple[List[Dict], Set[str]]:
    """Load Tier-B dataset: All WFC + 100 distractors."""

    # Get WFC incidents
    wfc_query = """
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN i.id as id, i.summary as summary, i.anchor_entities as entities
    ORDER BY i.id
    """
    wfc_results = await neo4j._execute_read(wfc_query)

    # Get 100 distractor incidents
    distractor_query = """
    MATCH (i:Incident)
    WHERE NOT 'Wang Fuk Court' IN i.anchor_entities
    WITH i ORDER BY i.id LIMIT 100
    RETURN i.id as id, i.summary as summary, i.anchor_entities as entities
    """
    distractor_results = await neo4j._execute_read(distractor_query)

    all_results = wfc_results + distractor_results
    wfc_ids = {r["id"] for r in wfc_results}

    incidents = []
    for r in all_results:
        incidents.append({
            "id": r["id"],
            "summary": r.get("summary", "") or "",
            "entities": set(r.get("entities", []) or []),
            "embedding": None,
            "snippets": [],
            "facets": set(),
        })

    incident_ids = [i["id"] for i in incidents]

    # Load snippets and facets for fingerprinting
    print(f"Loading snippets and facets for {len(incidents)} incidents...")
    snippets_map = await load_incident_snippets_and_facets(neo4j, incident_ids)
    for inc in incidents:
        data = snippets_map.get(inc["id"], {})
        inc["snippets"] = data.get("snippets", [])
        inc["facets"] = data.get("facets", set())

    snippet_count = sum(1 for i in incidents if i.get("snippets"))
    print(f"  Incidents with snippets: {snippet_count}/{len(incidents)}")

    # Load embeddings
    if incidents and db_pool:
        print(f"Loading embeddings for {len(incidents)} incidents...")
        emb_map = await load_incident_embeddings(neo4j, db_pool, incident_ids)

        for inc in incidents:
            inc["embedding"] = emb_map.get(inc["id"])

        emb_count = sum(1 for i in incidents if i.get("embedding") is not None)
        print(f"  Incidents with embeddings: {emb_count}/{len(incidents)}")

    return incidents, wfc_ids


async def run_tier_b():
    print("=" * 70)
    print("TIER-B VALIDATION: Unified Weaver with Distractors")
    print("=" * 70)
    print("Goal: Validate that repair dynamics don't stitch contaminants")
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

    # Load Tier-B dataset
    print("Loading Tier-B dataset (WFC + 100 distractors)...")
    incidents, wfc_ground_truth = await load_tier_b_incidents(neo4j, db_pool)

    wfc_count = len(wfc_ground_truth)
    distractor_count = len(incidents) - wfc_count
    print(f"Dataset: {wfc_count} WFC + {distractor_count} distractors = {len(incidents)} total")
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
            time=datetime.now(),
            snippets=inc.get("snippets", []),
            facets=inc.get("facets", set()),
        )

        if i % 10 == 0 or i == len(incidents):
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
    # TODO: Convert re_evaluate_singletons_against_grown_cases to LLM-gated
    print("(Growth re-evaluation skipped - needs LLM gate)")
    print()

    # Final summary
    summary = weaver.summary()

    print("=" * 70)
    print("TIER-B RESULTS")
    print("=" * 70)
    print()
    print(f"Incidents processed: {summary['incidents']}")
    print(f"Cases formed: {summary['cases']}")
    print(f"Metabolic edges: {summary['metabolic_edges']}")
    print(f"Deferred decisions: {summary['deferred']}")
    print()

    # WFC analysis
    membership = weaver.get_case_membership()

    # Find case containing most WFC incidents
    best_case = None
    best_overlap = 0

    for case_id, incident_ids in membership.items():
        overlap = len(incident_ids & wfc_ground_truth)
        if overlap > best_overlap:
            best_overlap = overlap
            best_case = case_id

    if best_case:
        case_incidents = membership[best_case]
        wfc_in_case = len(case_incidents & wfc_ground_truth)
        non_wfc_in_case = len(case_incidents - wfc_ground_truth)
        purity = wfc_in_case / len(case_incidents) if case_incidents else 0
        coverage = wfc_in_case / len(wfc_ground_truth) if wfc_ground_truth else 0

        print("WFC Case Analysis:")
        print(f"  Best case: {best_case}")
        print(f"  Case size: {len(case_incidents)}")
        print(f"  WFC incidents: {wfc_in_case}")
        print(f"  Non-WFC (contaminants): {non_wfc_in_case}")
        print(f"  Purity: {purity:.1%}")
        print(f"  Coverage: {coverage:.1%}")

        contaminants = case_incidents - wfc_ground_truth
        if contaminants:
            print(f"\n  CONTAMINANTS FOUND ({len(contaminants)}):")
            for iid in list(contaminants)[:5]:
                inc_data = next((i for i in incidents if i["id"] == iid), None)
                if inc_data:
                    print(f"    {iid[:12]}: {list(inc_data['entities'])[:3]}")
        else:
            print("\n  No contaminants - clean membrane!")

    print()

    # Show distractor cases
    print("Distractor Analysis:")
    distractor_ids = {i["id"] for i in incidents} - wfc_ground_truth
    distractor_cases = []
    for case_id, incident_ids in membership.items():
        distractor_overlap = len(incident_ids & distractor_ids)
        wfc_overlap = len(incident_ids & wfc_ground_truth)
        if distractor_overlap > 0 and wfc_overlap == 0:
            distractor_cases.append((case_id, len(incident_ids)))

    print(f"  Pure distractor cases: {len(distractor_cases)}")
    if distractor_cases:
        distractor_cases.sort(key=lambda x: -x[1])
        print("  Largest distractor cases:")
        for case_id, size in distractor_cases[:3]:
            case = weaver.cases.get(case_id)
            if case:
                entities = sorted(list(case.entities))[:3]
                print(f"    {case_id}: {size} incidents - {entities}")

    print()
    print("=" * 70)
    print("CI METRICS (TIER-B)")
    print("=" * 70)
    print(f"  tier: B")
    print(f"  incidents_processed: {len(incidents)}")
    print(f"  wfc_count: {wfc_count}")
    print(f"  distractor_count: {distractor_count}")
    print(f"  cases_formed: {summary['cases']}")
    if best_case:
        print(f"  wfc_purity: {purity:.3f}")
        print(f"  wfc_coverage: {coverage:.3f}")
        print(f"  contaminants: {non_wfc_in_case}")

    await db_pool.close()
    await neo4j.close()

    print()
    print("Tier-B validation complete!")


if __name__ == "__main__":
    asyncio.run(run_tier_b())
