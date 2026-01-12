#!/usr/bin/env python3
"""
Run Model Selection Weaver Experiment on Real Claims

This script tests the universal grouping approach using:
- Embeddings loaded from PostgreSQL via ClaimRepository (domain model pattern)
- Cheap LLM (gpt-4o-mini) for proposition key extraction
- Model selection objective: ΔScore = gain - cost

Usage:
    # Inside docker:
    python -m reee.experiments.run_model_selection_weaver [--limit N]

    # From host:
    docker exec herenews-app python -m reee.experiments.run_model_selection_weaver
"""

import os
import sys
import asyncio
import argparse
import hashlib
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple

sys.path.insert(0, '/app/backend')

import asyncpg
from openai import AsyncOpenAI

# Use domain models - embeddings are loaded from database
from models.domain.claim import Claim
from models.domain.surface import Surface
from models.domain.incident import Incident
from repositories.claim_repository import ClaimRepository
from workers.claim_loader import load_claims_with_embeddings

from reee.experiments.model_selection_weaver import (
    ModelSelectionWeaver,
    CoherenceMetrics,
    GroupingDecision,
    WeaverEmission,
)

# Hub locations that should not count as anchor entities
HUB_LOCATIONS = frozenset({
    'Hong Kong', 'China', 'United States', 'UK', 'United Kingdom', 'US',
    'New York', 'Washington', 'Beijing', 'London', 'Tai Po', 'Hong Kong Fire Services'
})


async def load_wfc_claims_with_embeddings(
    neo4j,
    claim_repo: ClaimRepository,
    limit: int = 50
) -> List[Claim]:
    """
    Load Wang Fuk Court claims from Neo4j with embeddings from PostgreSQL.
    Uses domain model pattern: Neo4j for graph data, PostgreSQL for embeddings.
    """
    query = """
    MATCH (i:Incident)-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    WITH c, p, i, s, collect(DISTINCT e.canonical_name) as entities
    RETURN c.id as id,
           c.text as text,
           c.event_time as event_time,
           p.id as page_id,
           p.domain as source,
           entities,
           s.anchor_entities as surface_anchors,
           i.anchor_entities as incident_anchors,
           s.time_start as time_start,
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

        entities = set(e for e in (r.get("entities") or []) if e)
        incident_anchors = set(r.get("incident_anchors") or [])
        surface_anchors = set(r.get("surface_anchors") or [])
        all_entities = incident_anchors | surface_anchors | entities
        anchor_entities = {e for e in all_entities if e not in HUB_LOCATIONS}

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

        # Create Claim domain object using reee.types.Claim
        from reee.types import Claim as ReeClaim
        claim = ReeClaim(
            id=claim_id,
            text=r.get("text", "")[:500],
            source=r.get("source", "unknown"),
            page_id=r.get("page_id"),
            entities=all_entities,
            anchor_entities=anchor_entities,
            event_time=event_time,
        )
        # Store extra metadata
        claim.incident_id = r.get("incident_id")
        claim.surface_id = r.get("surface_id")
        claim.question_key = r.get("question_key")
        claims.append(claim)

    # Batch fetch embeddings from PostgreSQL
    if claims and claim_repo:
        from pgvector.asyncpg import register_vector
        claim_ids = [c.id for c in claims]
        claim_lookup = {c.id: c for c in claims}
        embeddings_found = 0

        async with claim_repo.db_pool.acquire() as conn:
            await register_vector(conn)
            results = await conn.fetch("""
                SELECT claim_id, embedding
                FROM core.claim_embeddings
                WHERE claim_id = ANY($1)
            """, claim_ids)

            for row in results:
                claim = claim_lookup.get(row['claim_id'])
                if claim and row['embedding'] is not None:
                    emb = row['embedding']
                    if hasattr(emb, 'tolist'):
                        claim.embedding = emb.tolist()
                    elif isinstance(emb, (list, tuple)):
                        claim.embedding = list(emb)
                    embeddings_found += 1

        print(f"  Loaded {embeddings_found}/{len(claims)} embeddings from database")

    return claims


async def load_mixed_claims_with_embeddings(
    neo4j,
    claim_repo: ClaimRepository,
    wfc_limit: int = 30,
    distractor_limit: int = 20
) -> Tuple[List, Set[str]]:
    """
    Load WFC claims plus distractors with embeddings from database.
    Returns: (all_claims, wfc_claim_ids)
    """
    from reee.types import Claim as ReeClaim

    wfc_claims = await load_wfc_claims_with_embeddings(neo4j, claim_repo, limit=wfc_limit)
    wfc_ids = {c.id for c in wfc_claims}

    # Load distractors
    query = """
    MATCH (i:Incident)-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(c:Claim)
    WHERE NOT 'Wang Fuk Court' IN i.anchor_entities
    OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    WITH c, p, i, s, collect(DISTINCT e.canonical_name) as entities
    RETURN c.id as id,
           c.text as text,
           entities,
           i.anchor_entities as anchor_entities,
           s.time_start as event_time,
           p.id as page_id,
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

        entities = set(e for e in (r.get("entities") or []) if e)
        all_entities = set(r.get("anchor_entities", []) or []) | entities
        anchor_entities = {e for e in all_entities if e not in HUB_LOCATIONS}

        event_time = None
        if r.get("event_time"):
            try:
                if isinstance(r["event_time"], str):
                    event_time = datetime.fromisoformat(r["event_time"].replace("Z", "+00:00"))
                elif hasattr(r["event_time"], "to_native"):
                    event_time = r["event_time"].to_native()
            except Exception:
                pass

        # Create Claim domain object
        claim = ReeClaim(
            id=claim_id,
            text=r.get("text", "")[:500],
            source=r.get("source", "unknown"),
            page_id=r.get("page_id"),
            entities=all_entities,
            anchor_entities=anchor_entities,
            event_time=event_time,
        )
        claim.incident_id = r.get("incident_id")
        distractor_claims.append(claim)

    # Batch fetch embeddings for distractors
    if distractor_claims and claim_repo:
        from pgvector.asyncpg import register_vector
        claim_ids = [c.id for c in distractor_claims]
        claim_lookup = {c.id: c for c in distractor_claims}
        embeddings_found = 0

        async with claim_repo.db_pool.acquire() as conn:
            await register_vector(conn)
            results = await conn.fetch("""
                SELECT claim_id, embedding
                FROM core.claim_embeddings
                WHERE claim_id = ANY($1)
            """, claim_ids)

            for row in results:
                claim = claim_lookup.get(row['claim_id'])
                if claim and row['embedding'] is not None:
                    emb = row['embedding']
                    if hasattr(emb, 'tolist'):
                        claim.embedding = emb.tolist()
                    elif isinstance(emb, (list, tuple)):
                        claim.embedding = list(emb)
                    embeddings_found += 1

        print(f"  Loaded {embeddings_found}/{len(distractor_claims)} distractor embeddings from database")

    # Interleave by time
    all_claims = wfc_claims + distractor_claims
    all_claims.sort(key=lambda c: c.event_time or datetime.min)

    return all_claims, wfc_ids


async def extract_proposition_keys_batch(
    llm_client: AsyncOpenAI,
    claims: List[Dict],
    batch_size: int = 20,
) -> List[str]:
    """
    Extract proposition keys using cheap LLM (gpt-4o-mini).

    Proposition key = the question this claim answers.
    E.g., "How many people died?" → "death_toll"
    """
    proposition_keys = []

    system_prompt = """You are extracting proposition keys from news claims.
A proposition key identifies WHAT QUESTION the claim answers.

Output format: one word or short phrase, snake_case.

Examples:
- "13 people were killed" → death_toll
- "The fire started on the 23rd floor" → fire_origin
- "Wang Fuk Court is a residential building" → building_type
- "Firefighters arrived at 6:45 AM" → response_time
- "An elderly woman was rescued from her balcony" → rescue_event
- "The cause is under investigation" → cause_status

Only output the key, nothing else."""

    for i in range(0, len(claims), batch_size):
        batch = claims[i:i + batch_size]
        batch_texts = [c.text[:200] for c in batch]

        # Format as numbered list for efficient batching
        prompt = "Extract proposition key for each claim:\n\n"
        for j, text in enumerate(batch_texts, 1):
            prompt += f"{j}. {text}\n"
        prompt += "\nOutput one key per line (numbered):"

        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            lines = response.choices[0].message.content.strip().split("\n")

            # Parse numbered responses
            for j, line in enumerate(lines):
                if j >= len(batch):
                    break
                # Extract key from "1. key" or just "key"
                key = line.strip()
                if ". " in key:
                    key = key.split(". ", 1)[1]
                key = key.strip().lower().replace(" ", "_")[:30]
                if not key:
                    key = f"unknown_{i+j}"
                proposition_keys.append(key)

            # Pad if we got fewer responses
            while len(proposition_keys) < i + len(batch):
                proposition_keys.append(f"unknown_{len(proposition_keys)}")

        except Exception as e:
            print(f"  Proposition extraction batch {i//batch_size + 1} failed: {e}")
            # Fallback: use hash of text
            for c in batch:
                h = hashlib.md5(c.text[:50].encode()).hexdigest()[:8]
                proposition_keys.append(f"prop_{h}")

    return proposition_keys


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


async def generate_incident_descriptions(
    llm_client: AsyncOpenAI,
    weaver: ModelSelectionWeaver,
    batch_size: int = 10,
) -> Dict[str, str]:
    """
    Generate descriptions for incidents using LLM.
    Returns: Dict mapping incident_id -> description
    """
    system_prompt = """You are generating concise descriptions for news incidents.
Each incident contains claims (evidence snippets) about what happened.

Generate a 1-2 sentence description that captures:
1. What happened (the core event)
2. Key entities involved (who/what/where)
3. The narrative role (is this a cause, effect, response, etc.)

Be factual and concise. Output format: one description per line."""

    descriptions = {}
    incident_ids = list(weaver.incidents.keys())

    for i in range(0, len(incident_ids), batch_size):
        batch_ids = incident_ids[i:i + batch_size]

        # Build context for each incident
        incident_contexts = []
        for iid in batch_ids:
            incident = weaver.incidents[iid]

            # Gather claim texts from surfaces
            claim_texts = []
            for sid in incident.surface_ids:
                surface = weaver.surfaces.get(sid)
                if surface:
                    # Get claim text (from first claim as representative)
                    for cid in list(surface.claim_ids)[:2]:
                        claim_texts.append(f"- {surface.proposition_key}")

            referents = sorted(list(incident.referents))[:5]
            incident_contexts.append({
                "id": iid,
                "referents": referents,
                "claims": claim_texts[:5],
            })

        # Format prompt
        prompt = "Generate a description for each incident:\n\n"
        for j, ctx in enumerate(incident_contexts, 1):
            prompt += f"{j}. Referents: {ctx['referents']}\n"
            prompt += f"   Claims: {'; '.join(ctx['claims'][:3])}\n\n"
        prompt += "Output one description per line (numbered):"

        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            lines = response.choices[0].message.content.strip().split("\n")

            for j, line in enumerate(lines):
                if j >= len(batch_ids):
                    break
                desc = line.strip()
                if ". " in desc:
                    desc = desc.split(". ", 1)[1]
                descriptions[batch_ids[j]] = desc

        except Exception as e:
            print(f"  Description batch {i//batch_size + 1} failed: {e}")
            for iid in batch_ids:
                descriptions[iid] = f"Incident involving {list(weaver.incidents[iid].referents)[:2]}"

    return descriptions


async def generate_case_descriptions(
    llm_client: AsyncOpenAI,
    weaver: ModelSelectionWeaver,
    incident_descriptions: Dict[str, str],
) -> Dict[str, Tuple[str, str]]:
    """
    Generate titles and summaries for cases using LLM.
    Returns: Dict mapping case_id -> (title, summary)
    """
    system_prompt = """You are generating titles and summaries for news story cases.
A case is a coherent story formed from multiple related incidents.

For each case, generate:
1. A short headline (5-10 words)
2. A 2-3 sentence summary explaining the story arc

Format:
TITLE: [headline]
SUMMARY: [summary paragraph]"""

    case_descriptions = {}

    for case_id, case in weaver.cases.items():
        # Gather incident descriptions
        incident_descs = []
        for iid in list(case.incident_ids)[:5]:  # Top 5 incidents
            desc = incident_descriptions.get(iid, "")
            if desc:
                incident_descs.append(desc)

        membrane_entities = sorted(list(case.membrane))[:5]

        prompt = f"""Generate a title and summary for this news story case:

Membrane entities (key subjects): {membrane_entities}
Number of incidents: {len(case.incident_ids)}

Related incidents:
{chr(10).join(f'- {d}' for d in incident_descs)}

Generate a compelling but factual title and summary:"""

        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            text = response.choices[0].message.content.strip()

            # Parse title and summary
            title = ""
            summary = ""

            for line in text.split("\n"):
                if line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()
                elif line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                elif summary:  # Continue summary across lines
                    summary += " " + line.strip()

            if not title:
                title = f"Story: {', '.join(membrane_entities[:2])}"
            if not summary:
                summary = f"A developing story involving {', '.join(membrane_entities[:3])}."

            case_descriptions[case_id] = (title, summary)

        except Exception as e:
            print(f"  Case description failed for {case_id[:8]}: {e}")
            case_descriptions[case_id] = (
                f"Story: {', '.join(membrane_entities[:2])}",
                f"A story involving {', '.join(membrane_entities[:3])}."
            )

    return case_descriptions


async def run_experiment(
    neo4j,
    claim_repo: ClaimRepository,
    llm_client: AsyncOpenAI,
    limit: int = 30,
    mixed: bool = False,
    distractor_limit: int = 20,
) -> Dict:
    """
    Run the model selection weaver experiment.
    Uses Claim domain objects with embeddings loaded from database.
    """
    print("=" * 70)
    print("MODEL SELECTION WEAVER EXPERIMENT")
    print("=" * 70)
    print(f"Approach: Universal grouping via compression vs contamination")
    print(f"LLM: gpt-4o-mini (proposition keys)")
    print(f"Embeddings: loaded from PostgreSQL (core.claim_embeddings)")
    print()

    # Load claims with embeddings from database
    if mixed:
        all_claims, wfc_ids = await load_mixed_claims_with_embeddings(
            neo4j, claim_repo, limit, distractor_limit
        )
        print(f"Loaded {len(wfc_ids)} WFC claims + {len(all_claims) - len(wfc_ids)} distractors")
    else:
        all_claims = await load_wfc_claims_with_embeddings(neo4j, claim_repo, limit=limit)
        wfc_ids = {c.id for c in all_claims}
        print(f"Loaded {len(all_claims)} WFC claims")

    # Count claims with embeddings
    claims_with_emb = sum(1 for c in all_claims if c.embedding)
    print(f"Claims with embeddings: {claims_with_emb}/{len(all_claims)}")
    print()

    # Extract proposition keys (still uses LLM)
    print("Extracting proposition keys (LLM batch)...")
    proposition_keys = await extract_proposition_keys_batch(llm_client, all_claims)
    print(f"  Extracted {len(proposition_keys)} keys")

    # Show sample proposition keys
    print("\nSample proposition keys:")
    for i in range(min(5, len(all_claims))):
        print(f"  {proposition_keys[i]}: {all_claims[i].text[:60]}...")
    print()

    # Store proposition keys in claim
    for i, claim in enumerate(all_claims):
        claim.proposition_key = proposition_keys[i]

    # Create weaver
    weaver = ModelSelectionWeaver(
        merge_threshold=0.5,
        closure_risk_max=0.3,
        hub_entities=HUB_LOCATIONS,
    )

    print("Processing claims through model selection weaver...")
    print("-" * 60)

    for i, claim in enumerate(all_claims, 1):
        # Use domain model attributes directly
        anchor_entities = claim.anchor_entities or set()
        all_entities = claim.entities or set()

        surface_id, incident_id, case_id, emissions = weaver.process_claim(
            claim_id=claim.id,
            text=claim.text,
            proposition_key=getattr(claim, 'proposition_key', '') or getattr(claim, 'question_key', ''),
            source=claim.source or "unknown",
            entities=all_entities,
            anchor_entities=anchor_entities,
            event_time=claim.event_time,
            embedding=claim.embedding,  # From domain model, loaded from DB
        )

        # Show progress every 10 claims
        if i % 10 == 0 or i == len(all_claims):
            summary = weaver.summary()
            emission_types = [e.emission_type.name for e in emissions]
            print(f"  [{i:3d}/{len(all_claims)}] surfaces={summary['surfaces']} "
                  f"incidents={summary['incidents']} cases={summary['cases']} "
                  f"emissions={emission_types[:2]}")

    print()

    # Final summary
    summary = weaver.summary()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"Claims processed: {len(all_claims)}")
    print(f"Surfaces created: {summary['surfaces']}")
    print(f"Incidents created: {summary['incidents']}")
    print(f"Cases formed: {summary['cases']}")
    print(f"Multi-incident cases: {summary['multi_incident_cases']}")
    print(f"Largest case: {summary['largest_case']} incidents")
    print(f"Average case health: {summary['avg_case_health']:.3f}")
    print(f"Deferred decisions: {summary['deferred_count']}")
    print()

    print("Emission breakdown:")
    for etype, count in summary["emission_types"].items():
        if count > 0:
            print(f"  {etype}: {count}")
    print()

    # Generate descriptions for incidents and cases
    print("Generating incident descriptions...")
    incident_descriptions = await generate_incident_descriptions(llm_client, weaver)
    print(f"  Generated {len(incident_descriptions)} incident descriptions")

    # Store descriptions in incidents
    for iid, desc in incident_descriptions.items():
        if iid in weaver.incidents:
            weaver.incidents[iid].description = desc

    print("Generating case titles and summaries...")
    case_descriptions = await generate_case_descriptions(llm_client, weaver, incident_descriptions)
    print(f"  Generated {len(case_descriptions)} case descriptions")

    # Store descriptions in cases
    for cid, (title, case_summary) in case_descriptions.items():
        if cid in weaver.cases:
            weaver.cases[cid].title = title
            weaver.cases[cid].summary = case_summary

    print()

    # Find WFC case
    wfc_case = None
    wfc_case_id = None
    for case_id, case in weaver.cases.items():
        if "Wang Fuk Court" in case.membrane:
            wfc_case = case
            wfc_case_id = case_id
            break

    if wfc_case:
        # Compute purity/coverage
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

        print(f"WFC Case Analysis ({wfc_case_id[:12]}...):")
        print(f"  Incidents: {len(wfc_case.incident_ids)}")
        print(f"  Claims in case: {len(case_claim_ids)}")
        print(f"  WFC claims: {len(wfc_in_case)}")
        print(f"  Purity: {purity:.1%}")
        print(f"  Coverage: {coverage:.1%}")
        print(f"  Contaminants: {len(contaminants)}")
        print()

        # Show case metrics
        case_metrics = wfc_case.metrics(weaver.incidents, weaver.surfaces)
        print(f"  Case coherence metrics:")
        print(f"    Entropy: {case_metrics.entropy:.3f}")
        print(f"    Conflict density: {case_metrics.conflict_density:.3f}")
        print(f"    Boundary pressure: {case_metrics.boundary_pressure:.3f}")
        print(f"    Closure risk: {case_metrics.closure_risk:.3f}")
        print(f"    Health score: {case_metrics.health_score():.3f}")
        print()

        # Show membrane entities
        print(f"  Membrane entities: {sorted(list(wfc_case.membrane))[:10]}")

        # Show generated title and summary
        if wfc_case.title:
            print()
            print(f"  Generated title: {wfc_case.title}")
        if wfc_case.summary:
            print(f"  Generated summary: {wfc_case.summary[:200]}...")
    else:
        print("No WFC case found!")
        purity = 0
        coverage = 0

    # Show all cases summary
    print()
    print("All cases:")
    for cid, case in sorted(weaver.cases.items(), key=lambda x: -len(x[1].incident_ids)):
        metrics = case.metrics(weaver.incidents, weaver.surfaces)
        title = case.title[:40] + "..." if len(case.title) > 40 else case.title
        print(f"  {cid[:8]}: {len(case.incident_ids)} incidents, "
              f"health={metrics.health_score():.2f}")
        if title:
            print(f"           \"{title}\"")

    return {
        "claims": len(all_claims),
        "wfc_claims": len(wfc_ids),
        "summary": summary,
        "purity": purity if wfc_case else 0,
        "coverage": coverage if wfc_case else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run Model Selection Weaver Experiment")
    parser.add_argument("--limit", type=int, default=30, help="Max WFC claims to process")
    parser.add_argument("--mixed", action="store_true", help="Run mixed (WFC + distractors) experiment")
    parser.add_argument("--distractor-limit", type=int, default=20, help="Max distractor claims")
    args = parser.parse_args()

    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService()
    await neo4j.connect()

    # Create PostgreSQL pool for ClaimRepository
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD'),
    )

    # Create ClaimRepository to access embeddings via domain model pattern
    claim_repo = ClaimRepository(db_pool=db_pool, neo4j_service=neo4j)

    llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        result = await run_experiment(
            neo4j,
            claim_repo,  # Pass ClaimRepository instead of db_pool
            llm_client,
            limit=args.limit,
            mixed=args.mixed,
            distractor_limit=args.distractor_limit,
        )

        print()
        print("Experiment complete!")
        print()

        # CI Metrics
        print("CI Metrics:")
        print(f"  claims_processed: {result.get('claims', 0)}")
        print(f"  cases_formed: {result.get('summary', {}).get('cases', 0)}")
        print(f"  largest_case: {result.get('summary', {}).get('largest_case', 0)}")
        if "purity" in result:
            print(f"  wfc_purity: {result['purity']:.3f}")
            print(f"  wfc_coverage: {result['coverage']:.3f}")
            print(f"  avg_case_health: {result['summary'].get('avg_case_health', 0):.3f}")

    finally:
        await db_pool.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
