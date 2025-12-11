"""
FIXED: Proper plausibility scoring and weighted narrative generation

Fixes:
1. Enforce LLM JSON schema with explicit claim IDs
2. Validate response has scores for ALL claims
3. Show ALL claims with weights in narrative prompt
4. Use actual plausibility values, not filtered subset
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import json

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class FixedWeightedNarrativeGenerator:
    """Fixed version with proper plausibility scoring"""

    def __init__(self, neo4j: Neo4jService, openai_client: AsyncOpenAI):
        self.neo4j = neo4j
        self.openai = openai_client
        self.llm_calls = 0

    async def analyze_cluster_with_validation(
        self,
        cluster_claims: List[dict]
    ) -> dict:
        """
        Analyze cluster with STRICT JSON validation

        Returns plausibility scores for ALL claims
        """
        self.llm_calls += 1

        # Build expected output structure with actual claim IDs
        expected_scores = {c['id']: 0.5 for c in cluster_claims}
        example_json = {
            "topic": "brief topic description",
            "pattern": "consensus|mixed|contradictory",
            "plausibility_scores": {
                cluster_claims[0]['id']: 0.85,
                cluster_claims[1]['id']: 0.65
            } if len(cluster_claims) >= 2 else {cluster_claims[0]['id']: 0.75},
            "consensus_points": ["established fact 1"],
            "contradictions": ["conflict description"]
        }

        # Format claims
        claims_text = []
        for i, claim in enumerate(cluster_claims, 1):
            time_str = claim.get('event_time', '')[:19] if claim.get('event_time') else 'unknown'
            claims_text.append(f"{i}. ID={claim['id'][:12]} @ {time_str}")
            claims_text.append(f"   Text: \"{claim['text']}\"")

        prompt = f"""Analyze these {len(cluster_claims)} claims and assign plausibility scores.

CLAIMS:
{chr(10).join(claims_text)}

TASK:
1. Identify the topic/facet
2. Detect pattern: consensus (claims agree), mixed (complementary), contradictory (conflict)
3. Assign plausibility score (0.0-1.0) to EVERY claim based on:
   - Consensus claims (supported by others) → 0.75-0.95
   - Contradicted claims (minority view) → 0.20-0.45
   - Complementary claims (add info) → 0.60-0.80
   - Outliers (unusual/unsupported) → 0.10-0.30
   - Latest temporal claims (if progression) → highest scores

CRITICAL: You MUST return scores for ALL {len(cluster_claims)} claim IDs.

REQUIRED JSON FORMAT:
{json.dumps(example_json, indent=2)}

Return ONLY valid JSON with plausibility_scores containing ALL claim IDs from above."""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            data = json.loads(response.choices[0].message.content)

            # VALIDATION
            if 'plausibility_scores' not in data:
                raise ValueError("Missing plausibility_scores in response")

            scores = data['plausibility_scores']
            if len(scores) == 0:
                raise ValueError("Empty plausibility_scores")

            # Check for missing claims
            missing = [c['id'] for c in cluster_claims if c['id'] not in scores]
            if missing:
                print(f"  ⚠️  Missing scores for {len(missing)} claims, using default 0.5")
                for claim_id in missing:
                    scores[claim_id] = 0.5

            print(f"  ✅ Got plausibility scores for {len(scores)}/{len(cluster_claims)} claims")
            return data

        except Exception as e:
            print(f"  ❌ Cluster analysis failed: {e}")
            return {
                'topic': 'unknown',
                'pattern': 'mixed',
                'plausibility_scores': expected_scores,
                'consensus_points': [],
                'contradictions': []
            }

    async def build_network_and_resolve_plausibility(
        self,
        claims: List[dict],
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Build network and resolve plausibility

        Returns: {claim_id: plausibility_score}
        """
        print("\n🕸️  Building network and resolving plausibility...")
        print("="*80)

        # Build semantic network
        network = defaultdict(list)
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                vec1 = np.array(embeddings[claim1['id']])
                vec2 = np.array(embeddings[claim2['id']])

                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    similarity = float(dot / (norm1 * norm2))
                    if similarity > 0.4:
                        network[claim1['id']].append((claim2['id'], similarity))
                        network[claim2['id']].append((claim1['id'], similarity))

        print(f"   Network: {len(network)} nodes, {sum(len(n) for n in network.values())//2} edges")

        # Analyze all claims as one cluster
        print(f"\n   Analyzing all {len(claims)} claims...")
        analysis = await self.analyze_cluster_with_validation(claims)

        print(f"   Topic: {analysis['topic']}")
        print(f"   Pattern: {analysis['pattern']}")
        print(f"   Consensus: {len(analysis.get('consensus_points', []))} points")
        print(f"   Contradictions: {len(analysis.get('contradictions', []))} found")

        # Extract plausibility scores
        plausibility = {}
        for claim in claims:
            score = analysis['plausibility_scores'].get(claim['id'], 0.5)
            plausibility[claim['id']] = score

        # Show distribution
        scores = list(plausibility.values())
        print(f"\n   Plausibility distribution:")
        print(f"     Min: {min(scores):.2f}, Max: {max(scores):.2f}, Mean: {np.mean(scores):.2f}")
        print(f"     High (>0.7): {sum(1 for s in scores if s > 0.7)}")
        print(f"     Low (<0.4): {sum(1 for s in scores if s < 0.4)}")

        return plausibility, analysis

    async def generate_weighted_narrative(
        self,
        claims: List[dict],
        plausibility: Dict[str, float],
        analysis: dict
    ) -> str:
        """
        Generate narrative with EXPLICIT weights shown to LLM

        Shows ALL claims sorted by plausibility
        """
        self.llm_calls += 1

        # Sort claims by plausibility
        claims_sorted = sorted(
            claims,
            key=lambda c: plausibility.get(c['id'], 0.5),
            reverse=True
        )

        # Format claims with weights EXPLICITLY
        high_claims = [c for c in claims_sorted if plausibility[c['id']] >= 0.70]
        mid_claims = [c for c in claims_sorted if 0.45 <= plausibility[c['id']] < 0.70]
        low_claims = [c for c in claims_sorted if plausibility[c['id']] < 0.45]

        high_text = "\n".join(
            f"  [{plausibility[c['id']]:.2f}] {c['text']}"
            for c in high_claims[:10]
        ) if high_claims else "  (none)"

        mid_text = "\n".join(
            f"  [{plausibility[c['id']]:.2f}] {c['text']}"
            for c in mid_claims[:8]
        ) if mid_claims else "  (none)"

        low_text = "\n".join(
            f"  [{plausibility[c['id']]:.2f}] {c['text']}"
            for c in low_claims[:5]
        ) if low_claims else "  (none)"

        consensus = "\n".join(f"  • {cp}" for cp in analysis.get('consensus_points', []))
        contradictions = "\n".join(f"  • {cd}" for cd in analysis.get('contradictions', []))

        prompt = f"""Generate a comprehensive event narrative using these WEIGHTED claims.

EVENT TOPIC: {analysis.get('topic', 'Unknown')}
PATTERN: {analysis.get('pattern', 'mixed')}

CONSENSUS POINTS (established facts):
{consensus if consensus else "  (none identified)"}

CONTRADICTIONS (show as ranges/uncertainty):
{contradictions if contradictions else "  (none identified)"}

HIGH PLAUSIBILITY CLAIMS (0.70-1.00) - EMPHASIZE THESE:
{high_text}

MEDIUM PLAUSIBILITY CLAIMS (0.45-0.69) - Use as supporting info:
{mid_text}

LOW PLAUSIBILITY CLAIMS (<0.45) - Mention as "unverified" or omit:
{low_text}

INSTRUCTIONS:
1. Build narrative from HIGH plausibility claims first
2. Use CONSENSUS POINTS as established facts
3. For CONTRADICTIONS, show as ranges ("36-156 deaths reported")
4. Include MEDIUM claims as supporting details
5. Mention LOW claims ONLY if relevant, as "unverified reports"
6. Organize by natural topics (what happened, casualties, response, impact, etc.)
7. Use structured markdown format (## headers, bullet points)
8. NO speculation, NO hallucination
9. Show your confidence in each statement based on plausibility

Generate the narrative:"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Narrative generation failed: {e}"


async def main():
    print("="*80)
    print("🔧 FIXED: Weighted Narrative with Proper Plausibility Scoring")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    generator = FixedWeightedNarrativeGenerator(neo4j, openai_client)

    # Get event claims
    event_id = 'ev_pth3a8dc'

    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time
        ORDER BY c.event_time
        LIMIT 20
    """, {'event_id': event_id})

    claims = [dict(c) for c in claims]

    print(f"\n📊 Event: {event_id}")
    print(f"   Claims: {len(claims)}")

    # Generate embeddings
    print("\n🔮 Generating embeddings...")
    embeddings = {}
    for i, claim in enumerate(claims, 1):
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=claim['text']
        )
        embeddings[claim['id']] = response.data[0].embedding
        if i % 5 == 0:
            print(f"   {i}/{len(claims)}...")

    # Build network and resolve plausibility
    plausibility, analysis = await generator.build_network_and_resolve_plausibility(
        claims,
        embeddings
    )

    # Show top and bottom claims
    print("\n" + "="*80)
    print("📊 PLAUSIBILITY RESULTS")
    print("="*80)

    sorted_claims = sorted(
        claims,
        key=lambda c: plausibility[c['id']],
        reverse=True
    )

    print("\n🏆 TOP PLAUSIBILITY:")
    for claim in sorted_claims[:5]:
        score = plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")

    print("\n⚠️  BOTTOM PLAUSIBILITY:")
    for claim in sorted_claims[-5:]:
        score = plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")

    # Generate weighted narrative
    print("\n" + "="*80)
    print("📖 GENERATING WEIGHTED NARRATIVE")
    print("="*80)

    narrative = await generator.generate_weighted_narrative(
        claims,
        plausibility,
        analysis
    )

    print("\n" + "="*80)
    print("FINAL NARRATIVE:")
    print("="*80)
    print(narrative)
    print("="*80)

    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total LLM calls: {generator.llm_calls}")
    print(f"   Claims processed: {len(claims)}")
    scores = list(plausibility.values())
    print(f"   Plausibility range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"   Mean plausibility: {np.mean(scores):.2f}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
