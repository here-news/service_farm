"""
Test Bayesian claim plausibility metabolism with real Hong Kong fire data

This test:
1. Loads existing claims from pages 2, 3, 4 (already in graph)
2. Builds claim similarity network (AGREES/CONTRADICTS relationships)
3. Applies Bayesian updates to calculate plausibility scores
4. Shows which claims should be prioritized for narrative generation
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class BayesianMetabolism:
    """Bayesian plausibility updates for claim networks"""

    def __init__(self, neo4j: Neo4jService, openai_client: AsyncOpenAI):
        self.neo4j = neo4j
        self.openai = openai_client
        self.contradiction_cache = {}

    async def calculate_semantic_similarity(self, claim1: dict, claim2: dict) -> float:
        """Calculate cosine similarity between claim embeddings"""
        if not claim1.get('embedding') or not claim2.get('embedding'):
            return 0.0

        vec1 = np.array(claim1['embedding'])
        vec2 = np.array(claim2['embedding'])

        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            return float(dot / (norm1 * norm2))
        return 0.0

    async def extract_metrics(self, claim_text: str) -> List[Tuple[str, float]]:
        """
        Extract all quantitative metrics from claim using LLM

        Returns: [(metric_type, value), ...]
        Example: [('deaths', 156), ('missing', 279)]
        """
        # Check cache first
        cache_key = f"metrics_{hash(claim_text)}"
        if cache_key in self.contradiction_cache:
            return self.contradiction_cache[cache_key]

        prompt = f"""Extract all quantitative metrics from this claim. Return ONLY valid JSON.

Claim: "{claim_text}"

Return JSON object with metrics array:
{{
  "metrics": [
    {{"type": "deaths", "value": 156}},
    {{"type": "injured", "value": 12}}
  ]
}}

Valid metric types: deaths, injured, missing, evacuated, hospitalized, buildings, firefighters, displaced

If no metrics found, return: {{"metrics": []}}"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )

            import json
            data = json.loads(response.choices[0].message.content)
            metrics = [(m['type'], float(m['value'])) for m in data.get('metrics', [])]

            # Cache result
            self.contradiction_cache[cache_key] = metrics
            return metrics

        except Exception as e:
            print(f"  ⚠️  LLM metric extraction failed: {e}")
            return []

    async def are_contradictory(self, claim1: dict, claim2: dict, similarity: float) -> Tuple[bool, float]:
        """
        Detect if claims contradict using LLM-extracted metrics

        Returns: (is_contradictory, metric_difference)
        """
        # Extract metrics from both claims
        metrics1 = await self.extract_metrics(claim1['text'])
        metrics2 = await self.extract_metrics(claim2['text'])

        # Check for same metric type with different values
        for type1, val1 in metrics1:
            for type2, val2 in metrics2:
                if type1 == type2:
                    diff_pct = abs(val1 - val2) / max(val1, val2)
                    if diff_pct > 0.15:  # >15% difference = contradiction
                        print(f"  ⚠️  Numeric contradiction: {type1} {val1} vs {val2} (diff: {diff_pct:.1%}, sim: {similarity:.2f})")
                        return (True, diff_pct)

        return (False, 0.0)

    async def build_claim_network(self, claims: List[dict]) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Build claim relationship network

        Returns: {claim_id: [(related_claim_id, similarity, relationship_type)]}
        """
        network = defaultdict(list)

        print("\n🔗 Building claim relationship network...")
        print("="*80)

        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                sim = await self.calculate_semantic_similarity(claim1, claim2)

                if sim > 0.7:
                    # AGREES
                    network[claim1['id']].append((claim2['id'], sim, 'AGREES'))
                    network[claim2['id']].append((claim1['id'], sim, 'AGREES'))
                    print(f"✅ AGREES (sim={sim:.2f}):")
                    print(f"   {claim1['text'][:80]}...")
                    print(f"   {claim2['text'][:80]}...")

                else:
                    # Check for contradiction
                    is_contra, diff = await self.are_contradictory(claim1, claim2, sim)
                    if is_contra:
                        network[claim1['id']].append((claim2['id'], sim, 'CONTRADICTS'))
                        network[claim2['id']].append((claim1['id'], sim, 'CONTRADICTS'))
                        print(f"❌ CONTRADICTS (sim={sim:.2f}, diff={diff:.1%}):")
                        print(f"   {claim1['text'][:80]}...")
                        print(f"   {claim2['text'][:80]}...")

        return network

    async def update_plausibility_bayesian(
        self,
        claims: List[dict],
        network: Dict[str, List[Tuple[str, float, str]]]
    ) -> Dict[str, float]:
        """
        Apply Bayesian updates to claim plausibility scores

        Returns: {claim_id: plausibility_score}
        """
        print("\n🧮 Applying Bayesian plausibility updates...")
        print("="*80)

        # Initialize with priors (extraction confidence)
        plausibility = {c['id']: c['confidence'] for c in claims}

        # Iteration 1: Direct agreement/contradiction effects
        print("\n📊 Round 1: Direct relationship effects")
        for claim_id, relationships in network.items():
            agreements = [r for r in relationships if r[2] == 'AGREES']
            contradictions = [r for r in relationships if r[2] == 'CONTRADICTS']

            # Boost from agreements
            if agreements:
                avg_sim = np.mean([r[1] for r in agreements])
                boost = 0.3 * avg_sim * len(agreements)
                plausibility[claim_id] = min(0.99, plausibility[claim_id] * (1 + boost))
                print(f"  ↑ Claim {claim_id[:8]}: +{boost:.2f} from {len(agreements)} agreements")

            # Penalty from contradictions (depends on support)
            if contradictions:
                # Count support for this claim vs contradicting claims
                my_support = len(agreements)
                contra_support = []
                for contra_id, _, _ in contradictions:
                    contra_agreements = [r for r in network.get(contra_id, []) if r[2] == 'AGREES']
                    contra_support.append(len(contra_agreements))

                avg_contra_support = np.mean(contra_support) if contra_support else 0

                if my_support > avg_contra_support:
                    # Majority view - slight boost
                    plausibility[claim_id] = min(0.99, plausibility[claim_id] * 1.1)
                    print(f"  ↑ Claim {claim_id[:8]}: +10% (majority view, {my_support} vs {avg_contra_support:.1f} support)")
                elif my_support < avg_contra_support:
                    # Minority view - penalty
                    plausibility[claim_id] = max(0.01, plausibility[claim_id] * 0.3)
                    print(f"  ↓ Claim {claim_id[:8]}: -70% (minority view, {my_support} vs {avg_contra_support:.1f} support)")
                else:
                    # Contested - mark as uncertain
                    plausibility[claim_id] = plausibility[claim_id] * 0.6
                    print(f"  ~ Claim {claim_id[:8]}: -40% (contested, equal support)")

        # Iteration 2: Network propagation (neighbors influence)
        print("\n🌐 Round 2: Network propagation")
        for claim_id in plausibility:
            neighbors = network.get(claim_id, [])
            if neighbors:
                neighbor_scores = [plausibility[n[0]] for n in neighbors if n[2] == 'AGREES']
                if neighbor_scores:
                    neighbor_boost = np.mean(neighbor_scores) * 0.15
                    old_score = plausibility[claim_id]
                    plausibility[claim_id] = min(0.99, plausibility[claim_id] + neighbor_boost)
                    print(f"  → Claim {claim_id[:8]}: {old_score:.3f} → {plausibility[claim_id]:.3f} (neighbor boost)")

        return plausibility


async def main():
    print("="*80)
    print("🧪 BAYESIAN METABOLISM TEST")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    metabolism = BayesianMetabolism(neo4j, openai_client)

    # Load claims from the 3 test pages
    page_ids = [
        'pg_006iquvd',  # Page 2: Christianity Today
        'pg_00prszmp',  # Page 3: NY Post
        'pg_013v2wny',  # Page 4: DW
    ]

    print("\n📥 Loading claims from test pages...")
    all_claims = []

    for page_id in page_ids:
        claims = await neo4j._execute_read('''
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN c.id as id, c.text as text, c.confidence as confidence,
                   c.event_time as event_time
            ORDER BY c.event_time
        ''', {'page_id': page_id})

        page_title = await neo4j._execute_read('''
            MATCH (p:Page {id: $page_id})
            RETURN p.title as title
        ''', {'page_id': page_id})

        print(f"\n📄 {page_title[0]['title'][:70]}...")
        print(f"   Claims: {len(claims)}")

        for claim in claims:
            all_claims.append(claim)

    print(f"\n📊 Total claims: {len(all_claims)}")

    # Generate embeddings on-the-fly
    print("\n🔮 Generating claim embeddings...")
    for i, claim in enumerate(all_claims):
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=claim['text']
        )
        claim['embedding'] = response.data[0].embedding
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/{len(all_claims)} embeddings...")

    print(f"   ✅ All {len(all_claims)} embeddings generated")

    # Build relationship network
    network = await metabolism.build_claim_network(all_claims)

    # Apply Bayesian updates
    plausibility = await metabolism.update_plausibility_bayesian(all_claims, network)

    # Show final results
    print("\n" + "="*80)
    print("📈 FINAL PLAUSIBILITY SCORES")
    print("="*80)

    # Sort by plausibility
    sorted_claims = sorted(
        all_claims,
        key=lambda c: plausibility[c['id']],
        reverse=True
    )

    print("\n🏆 HIGH PLAUSIBILITY (narrative priority):")
    for claim in sorted_claims[:10]:
        score = plausibility[claim['id']]
        orig = claim['confidence']
        change = score - orig
        arrow = "↑" if change > 0.1 else "↓" if change < -0.1 else "→"
        print(f"\n{arrow} {score:.3f} (was {orig:.2f}, Δ={change:+.2f})")
        print(f"   {claim['text']}")

    print("\n" + "-"*80)
    print("\n⚠️  LOW PLAUSIBILITY (likely unreliable):")
    for claim in sorted_claims[-5:]:
        score = plausibility[claim['id']]
        orig = claim['confidence']
        change = score - orig
        print(f"\n↓ {score:.3f} (was {orig:.2f}, Δ={change:+.2f})")
        print(f"   {claim['text']}")

    # Summary statistics
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)

    plaus_values = list(plausibility.values())
    print(f"Mean plausibility: {np.mean(plaus_values):.3f}")
    print(f"Median plausibility: {np.median(plaus_values):.3f}")
    print(f"Std dev: {np.std(plaus_values):.3f}")
    print(f"Min: {np.min(plaus_values):.3f}")
    print(f"Max: {np.max(plaus_values):.3f}")

    # Count relationships
    total_agrees = sum(1 for rels in network.values() for r in rels if r[2] == 'AGREES') // 2
    total_contradicts = sum(1 for rels in network.values() for r in rels if r[2] == 'CONTRADICTS') // 2

    print(f"\nRelationships:")
    print(f"  AGREES: {total_agrees}")
    print(f"  CONTRADICTS: {total_contradicts}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
