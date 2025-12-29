"""
Emergence Curve Experiment
==========================

Tests B³ F1 and Completeness at different claims-per-event levels.

Usage:
    docker exec herenews-app python -m test_eu.run_emergence_curve
"""

import asyncio
import sys
import os
sys.path.insert(0, '/app/backend')

import asyncpg
from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector
from services.neo4j_service import Neo4jService
from test_eu.core.epistemic_unit import Claim, EmergenceEngine
from test_eu.core.evaluation import GroundTruth, LabeledClaim, evaluate_clustering
from test_eu.core.visualize import plot_emergence_curve


async def main():
    print("=" * 70)
    print("EMERGENCE CURVE EXPERIMENT")
    print("=" * 70)

    db_pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "phi_here"),
        user=os.getenv("POSTGRES_USER", "phi_user"),
        password=os.getenv("POSTGRES_PASSWORD", "phi_password_dev"),
        min_size=1, max_size=5
    )

    neo4j = Neo4jService(
        uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    await neo4j.connect()

    llm = AsyncOpenAI()

    # Get 3 events with lots of claims
    events = await neo4j._execute_read("""
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WITH e, count(c) as cnt
        WHERE cnt >= 20
        RETURN e.id as id, e.canonical_name as name, cnt
        ORDER BY cnt DESC
        LIMIT 3
    """, {})

    print("Events:")
    for e in events:
        name = e["name"][:40]
        cnt = e["cnt"]
        print(f"  - {name} ({cnt} claims)")

    # Test at different claims-per-event levels
    levels = [2, 4, 6, 8, 10]
    results = []

    for n_claims in levels:
        print(f"\n--- Testing {n_claims} claims per event ---")

        labeled_claims = []
        claims = []

        for ev in events:
            claims_data = await neo4j._execute_read("""
                MATCH (e:Event {id: $eid})-[:INTAKES]->(c:Claim)
                WHERE c.text IS NOT NULL
                OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
                OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
                WITH c, p, collect({name: ent.canonical_name, type: ent.entity_type}) as entities
                RETURN c.id as id, c.text as text, p.domain as source, entities
                ORDER BY rand()
                LIMIT $limit
            """, {"eid": ev["id"], "limit": n_claims})

            async with db_pool.acquire() as conn:
                await register_vector(conn)
                for row in claims_data:
                    if not row["text"]:
                        continue
                    embedding = await conn.fetchval(
                        "SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1",
                        row["id"]
                    )

                    all_ent = set()
                    anchor_ent = set()
                    for ent in row["entities"]:
                        ent_name = ent.get("name")
                        if ent_name:
                            all_ent.add(ent_name)
                            ent_type = ent.get("type")
                            if ent_type in ("PERSON", "ORGANIZATION", "ORG"):
                                anchor_ent.add(ent_name)

                    emb = None
                    if embedding is not None and len(embedding) > 0:
                        emb = [float(x) for x in embedding]

                    source = row["source"] or "unknown"
                    event_label = ev["name"][:25]

                    claim = Claim(
                        id=row["id"],
                        text=row["text"],
                        source=source,
                        embedding=emb,
                        entities=all_ent,
                        anchor_entities=anchor_ent
                    )
                    labeled_claims.append(LabeledClaim(claim=claim, event_label=event_label))
                    claims.append(claim)

        gt = GroundTruth(
            claims=labeled_claims,
            events={ev["name"][:25]: ev["name"] for ev in events},
            confounders=[]
        )

        # Run engine
        engine = EmergenceEngine(llm=llm)
        for claim in claims:
            await engine.add_claim(claim)
        surfaces = engine.compute_surfaces()

        # Evaluate
        metrics = evaluate_clustering(surfaces, gt)

        print(f"  Claims: {len(claims)}, Surfaces: {len(surfaces)}")
        print(f"  B3 F1: {metrics.b3_f1:.1%}, Completeness: {metrics.completeness:.1%}")

        results.append({
            "claims_per_event": n_claims,
            "b3_f1": metrics.b3_f1,
            "b3_precision": metrics.b3_precision,
            "b3_recall": metrics.b3_recall,
            "completeness": metrics.completeness,
            "num_clusters": metrics.num_clusters,
            "num_gt_events": len(events)
        })

    # Generate visualization
    print("\n--- Generating Emergence Curve ---")
    os.makedirs("/app/backend/test_eu/results", exist_ok=True)
    path = plot_emergence_curve(
        results,
        output_path="/app/backend/test_eu/results/emergence_curve.png"
    )
    print(f"Saved: {path}")

    # Print summary table
    print("\n--- Summary Table ---")
    print("Claims/Event |   B3 P |   B3 R |  B3 F1 |  Compl | Clusters")
    print("-" * 60)
    for r in results:
        cpe = r["claims_per_event"]
        p = r["b3_precision"]
        rec = r["b3_recall"]
        f1 = r["b3_f1"]
        comp = r["completeness"]
        nc = r["num_clusters"]
        print(f"{cpe:>12} | {p:>6.1%} | {rec:>6.1%} | {f1:>6.1%} | {comp:>6.1%} | {nc:>8}")

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
