"""
Evaluate Epistemic Stability and Accuracy

Tests:
1. Stability - Run synthesis N times, measure variance
2. Accuracy - Extract facts from narrative, verify against claims
3. Gap Analysis - Which gaps would most improve confidence
4. Convergence - Does re-synthesis with feedback improve quality
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, '/app')

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from rapidfuzz import fuzz


@dataclass
class EvalResult:
    stability_score: float  # 0-1, how consistent across runs
    fact_accuracy: float    # 0-1, facts match source claims
    source_coverage: float  # 0-1, claims represented
    gap_priorities: List[Dict]  # ranked gaps by impact
    contradictions: List[Dict]  # detected conflicts


async def fetch_claims(driver, event_id: str) -> List[Dict]:
    """Fetch all claims for evaluation"""
    query = """
    MATCH (e:Event {id: $event_id})-[:INTAKES]->(c:Claim)
    OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
    RETURN c.id as id, c.text as text, c.event_time as time, p.url as url
    """
    async with driver.session() as session:
        result = await session.run(query, event_id=event_id)
        return await result.data()


async def synthesize_narrative(client: AsyncOpenAI, claims: List[Dict]) -> Dict:
    """Single synthesis pass"""
    claims_text = "\n".join([f"- {c['text']}" for c in claims[:50]])

    prompt = f"""Synthesize these claims into a structured narrative.

CLAIMS:
{claims_text}

Return JSON with:
{{
    "facts": [
        {{"statement": "...", "confidence": 0.0-1.0, "claim_ids": [...]}},
        ...
    ],
    "uncertainties": ["what we don't know..."],
    "contradictions": [{{"topic": "...", "positions": ["A says...", "B says..."]}}]
}}

Extract 5-10 key facts. Be precise about numbers and dates."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"facts": [], "uncertainties": [], "contradictions": []}


async def test_stability(client: AsyncOpenAI, claims: List[Dict], runs: int = 3) -> Tuple[float, List[Dict]]:
    """Run synthesis multiple times, measure consistency"""
    print(f"  Running {runs} synthesis passes...")

    all_facts = []
    for i in range(runs):
        result = await synthesize_narrative(client, claims)
        all_facts.append(result.get('facts', []))
        print(f"    Pass {i+1}: {len(result.get('facts', []))} facts extracted")

    # Compare facts across runs using fuzzy matching
    if len(all_facts) < 2:
        return 1.0, all_facts[0] if all_facts else []

    # Find common facts (appear in all runs with similar wording)
    base_facts = all_facts[0]
    common_count = 0

    for base_fact in base_facts:
        appears_in_all = True
        for other_facts in all_facts[1:]:
            found = False
            for other_fact in other_facts:
                similarity = fuzz.token_sort_ratio(
                    base_fact.get('statement', ''),
                    other_fact.get('statement', '')
                )
                if similarity > 70:
                    found = True
                    break
            if not found:
                appears_in_all = False
                break
        if appears_in_all:
            common_count += 1

    stability = common_count / len(base_facts) if base_facts else 0
    return stability, base_facts


async def verify_facts_against_claims(facts: List[Dict], claims: List[Dict]) -> Tuple[float, List[Dict]]:
    """Check if extracted facts are supported by claims"""
    print("  Verifying facts against source claims...")

    verified = []
    for fact in facts:
        statement = fact.get('statement', '')

        # Find supporting claims
        supporting = []
        for claim in claims:
            similarity = fuzz.partial_ratio(statement.lower(), claim['text'].lower())
            if similarity > 60:
                supporting.append({
                    'claim_id': claim['id'],
                    'similarity': similarity,
                    'text': claim['text'][:100]
                })

        verified.append({
            'statement': statement,
            'confidence': fact.get('confidence', 0),
            'supporting_claims': len(supporting),
            'verified': len(supporting) > 0,
            'top_match': supporting[0] if supporting else None
        })

    accuracy = sum(1 for v in verified if v['verified']) / len(verified) if verified else 0
    return accuracy, verified


async def analyze_gaps(client: AsyncOpenAI, claims: List[Dict], facts: List[Dict]) -> List[Dict]:
    """Identify gaps and estimate their impact"""
    print("  Analyzing gaps...")

    claims_summary = "\n".join([c['text'][:100] for c in claims[:30]])
    facts_summary = "\n".join([f.get('statement', '')[:100] for f in facts])

    prompt = f"""Analyze what's MISSING from this coverage.

EXISTING FACTS:
{facts_summary}

SOURCE CLAIMS AVAILABLE:
{claims_summary}

Identify 3-5 gaps. For each, estimate:
- impact: how much would filling this improve the narrative (0.0-1.0)
- type: "missing_source", "unanswered_question", "conflicting_info", "temporal_gap"

Return JSON:
{{
    "gaps": [
        {{"question": "...", "impact": 0.8, "type": "...", "suggested_source": "..."}}
    ]
}}"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        gaps = result.get('gaps', [])
        # Sort by impact
        gaps.sort(key=lambda x: -x.get('impact', 0))
        return gaps
    except:
        return []


async def test_convergence(client: AsyncOpenAI, claims: List[Dict], iterations: int = 2) -> List[Dict]:
    """Test if iterative refinement improves quality"""
    print(f"  Testing convergence over {iterations} iterations...")

    history = []
    previous_narrative = None

    for i in range(iterations):
        # Synthesize
        if previous_narrative:
            # Include previous narrative as context for refinement
            result = await synthesize_with_feedback(client, claims, previous_narrative)
        else:
            result = await synthesize_narrative(client, claims)

        facts = result.get('facts', [])
        accuracy, verified = await verify_facts_against_claims(facts, claims)

        history.append({
            'iteration': i + 1,
            'fact_count': len(facts),
            'accuracy': accuracy,
            'avg_confidence': sum(f.get('confidence', 0) for f in facts) / len(facts) if facts else 0
        })

        previous_narrative = result
        print(f"    Iteration {i+1}: {len(facts)} facts, {accuracy:.0%} accuracy")

    return history


async def synthesize_with_feedback(client: AsyncOpenAI, claims: List[Dict], previous: Dict) -> Dict:
    """Refine synthesis based on previous attempt"""
    claims_text = "\n".join([f"- {c['text']}" for c in claims[:50]])
    prev_facts = "\n".join([f"- {f.get('statement', '')}" for f in previous.get('facts', [])])
    prev_gaps = ", ".join(previous.get('uncertainties', [])[:3])

    prompt = f"""Refine this narrative synthesis.

CLAIMS:
{claims_text}

PREVIOUS SYNTHESIS:
{prev_facts}

IDENTIFIED GAPS: {prev_gaps}

Improve the synthesis:
1. Fill gaps if claims support it
2. Increase precision (exact numbers, dates)
3. Resolve contradictions if possible

Return JSON with facts, uncertainties, contradictions."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return previous


async def main():
    # Setup
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', '')

    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Target event
    event_id = "ev_pth3a8dc"  # Wang Fuk Court Fire

    print(f"\n{'='*60}")
    print("EPISTEMIC EVALUATION: Wang Fuk Court Fire")
    print(f"{'='*60}\n")

    try:
        # Fetch claims
        claims = await fetch_claims(driver, event_id)
        print(f"Loaded {len(claims)} claims\n")

        # Test 1: Stability
        print("TEST 1: Stability (consistency across runs)")
        print("-" * 40)
        stability, base_facts = await test_stability(client, claims, runs=3)
        print(f"Result: {stability:.0%} of facts consistent across runs\n")

        # Test 2: Accuracy
        print("TEST 2: Accuracy (facts supported by claims)")
        print("-" * 40)
        accuracy, verified = await verify_facts_against_claims(base_facts, claims)
        print(f"Result: {accuracy:.0%} of facts verified against claims\n")

        # Show verified facts
        print("Verified facts:")
        for v in verified[:5]:
            status = "✓" if v['verified'] else "✗"
            print(f"  {status} {v['statement'][:60]}...")
            if v['top_match']:
                print(f"      Matched: {v['top_match']['text'][:50]}... ({v['top_match']['similarity']}%)")
        print()

        # Test 3: Gap Analysis
        print("TEST 3: Gap Analysis (what's missing)")
        print("-" * 40)
        gaps = await analyze_gaps(client, claims, base_facts)
        print(f"Identified {len(gaps)} gaps:\n")
        for i, gap in enumerate(gaps[:5], 1):
            print(f"  {i}. [{gap.get('impact', 0):.0%} impact] {gap.get('question', '')[:60]}")
            print(f"     Type: {gap.get('type', 'unknown')} | Source: {gap.get('suggested_source', 'N/A')}")
        print()

        # Test 4: Convergence
        print("TEST 4: Convergence (does refinement help)")
        print("-" * 40)
        convergence = await test_convergence(client, claims, iterations=3)
        print("\nConvergence history:")
        for h in convergence:
            print(f"  Iteration {h['iteration']}: {h['fact_count']} facts, "
                  f"{h['accuracy']:.0%} accuracy, {h['avg_confidence']:.0%} avg confidence")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Stability:  {stability:.0%}")
        print(f"  Accuracy:   {accuracy:.0%}")
        print(f"  Top Gap:    {gaps[0].get('question', 'N/A')[:50]}..." if gaps else "  Top Gap:    None identified")
        print(f"  Converges:  {'Yes' if convergence[-1]['accuracy'] >= convergence[0]['accuracy'] else 'Needs more iteration'}")

        # Write results
        output = {
            'event_id': event_id,
            'evaluated_at': datetime.utcnow().isoformat(),
            'claim_count': len(claims),
            'stability': stability,
            'accuracy': accuracy,
            'verified_facts': verified,
            'gaps': gaps,
            'convergence': convergence
        }

        with open('/app/prototypes/data/evaluation.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to /app/prototypes/data/evaluation.json")

    finally:
        await driver.close()


if __name__ == '__main__':
    asyncio.run(main())
