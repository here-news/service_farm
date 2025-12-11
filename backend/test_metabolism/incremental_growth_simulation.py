"""
Simulate incremental event growth with plausibility updates

Read-only from DB, all computations local:
1. Start with 5 claims (initial event)
2. Add 5 more claims (page 2 arrives)
3. Add 5 more claims (page 3 arrives)
4. Add 5 more claims (page 4 arrives)

Show how plausibility scores change as network grows
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


class IncrementalGrowthSimulator:
    """Simulate event growth with incremental plausibility updates"""

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client
        self.llm_calls = 0

        # Local state (not saved to DB)
        self.embeddings = {}
        self.plausibility = {}
        self.network = defaultdict(list)
        self.all_claims = []

    async def analyze_cluster_with_validation(
        self,
        cluster_claims: List[dict]
    ) -> dict:
        """
        Analyze cluster with JSON validation
        """
        self.llm_calls += 1

        # Build expected output structure
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

            return data

        except Exception as e:
            print(f"  ❌ Cluster analysis failed: {e}")
            expected_scores = {c['id']: 0.5 for c in cluster_claims}
            return {
                'topic': 'unknown',
                'pattern': 'mixed',
                'plausibility_scores': expected_scores,
                'consensus_points': [],
                'contradictions': []
            }

    async def add_claims_and_update(
        self,
        new_claims: List[dict],
        batch_name: str
    ):
        """
        Add new claims and update plausibility scores incrementally

        Returns: (old_scores, new_scores) for comparison
        """
        print(f"\n{'='*80}")
        print(f"📥 Adding {batch_name}: {len(new_claims)} new claims")
        print(f"{'='*80}")

        # Store old scores for comparison
        old_scores = dict(self.plausibility)

        # Add new claims
        self.all_claims.extend(new_claims)

        # Generate embeddings for new claims only
        print(f"   🔮 Generating embeddings for new claims...")
        for claim in new_claims:
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=claim['text']
            )
            self.embeddings[claim['id']] = response.data[0].embedding

        # Rebuild network (add new edges)
        print(f"   🕸️  Building network with {len(self.all_claims)} claims...")
        edges_added = 0
        for new_claim in new_claims:
            for existing_claim in self.all_claims:
                if new_claim['id'] == existing_claim['id']:
                    continue

                vec1 = np.array(self.embeddings[new_claim['id']])
                vec2 = np.array(self.embeddings[existing_claim['id']])

                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    similarity = float(dot / (norm1 * norm2))
                    if similarity > 0.4:
                        self.network[new_claim['id']].append((existing_claim['id'], similarity))
                        self.network[existing_claim['id']].append((new_claim['id'], similarity))
                        edges_added += 1

        print(f"   Added {edges_added} new edges")

        # Re-analyze entire cluster
        print(f"   🧮 Re-analyzing all {len(self.all_claims)} claims...")
        analysis = await self.analyze_cluster_with_validation(self.all_claims)

        # Update plausibility scores
        for claim in self.all_claims:
            score = analysis['plausibility_scores'].get(claim['id'], 0.5)
            self.plausibility[claim['id']] = score

        # Show what changed
        print(f"\n   📊 Plausibility changes:")

        # New claims
        print(f"\n   🆕 New claims:")
        for claim in new_claims:
            score = self.plausibility[claim['id']]
            print(f"      {score:.2f}: {claim['text'][:70]}...")

        # Changed scores for existing claims
        print(f"\n   📈 Updated scores for existing claims:")
        changes = []
        for claim_id, new_score in self.plausibility.items():
            if claim_id in old_scores:
                old_score = old_scores[claim_id]
                if abs(new_score - old_score) > 0.05:
                    changes.append((claim_id, old_score, new_score))

        if changes:
            changes.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
            for claim_id, old_score, new_score in changes[:5]:
                change = "↑" if new_score > old_score else "↓"
                claim = next(c for c in self.all_claims if c['id'] == claim_id)
                print(f"      {old_score:.2f} → {new_score:.2f} {change}: {claim['text'][:60]}...")
        else:
            print(f"      (no significant changes)")

        # Summary stats
        scores = list(self.plausibility.values())
        print(f"\n   Distribution: min={min(scores):.2f}, max={max(scores):.2f}, mean={np.mean(scores):.2f}")
        print(f"   Topic: {analysis['topic']}")
        print(f"   Pattern: {analysis['pattern']}")
        print(f"   Consensus: {len(analysis.get('consensus_points', []))} points")
        print(f"   Contradictions: {len(analysis.get('contradictions', []))} found")

        return old_scores, dict(self.plausibility), analysis


async def main():
    print("="*80)
    print("🌱 INCREMENTAL EVENT GROWTH SIMULATION (LOCAL ONLY)")
    print("="*80)

    # Connect to DB (read-only)
    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    simulator = IncrementalGrowthSimulator(openai_client)

    # Get all claims from event
    event_id = 'ev_pth3a8dc'

    all_claims_from_db = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time
        ORDER BY c.event_time
    """, {'event_id': event_id})

    all_claims_from_db = [dict(c) for c in all_claims_from_db]

    print(f"\n📊 Event: {event_id}")
    print(f"   Total claims in DB: {len(all_claims_from_db)}")

    await neo4j.close()

    # Simulate incremental growth
    print(f"\n{'='*80}")
    print("SIMULATION: Event grows as pages arrive")
    print(f"{'='*80}")

    # Batch 1: Initial 5 claims
    batch1 = all_claims_from_db[:5]
    await simulator.add_claims_and_update(batch1, "Batch 1 (initial 5 claims)")

    # Batch 2: Next 5 claims
    batch2 = all_claims_from_db[5:10]
    await simulator.add_claims_and_update(batch2, "Batch 2 (claims 6-10)")

    # Batch 3: Next 5 claims
    batch3 = all_claims_from_db[10:15]
    await simulator.add_claims_and_update(batch3, "Batch 3 (claims 11-15)")

    # Batch 4: Last 5 claims
    batch4 = all_claims_from_db[15:20]
    await simulator.add_claims_and_update(batch4, "Batch 4 (claims 16-20)")

    # Final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal LLM calls: {simulator.llm_calls}")
    print(f"Final claim count: {len(simulator.all_claims)}")
    print(f"Network edges: {sum(len(neighbors) for neighbors in simulator.network.values()) // 2}")

    scores = list(simulator.plausibility.values())
    print(f"\nFinal plausibility distribution:")
    print(f"  Min: {min(scores):.2f}, Max: {max(scores):.2f}, Mean: {np.mean(scores):.2f}")
    print(f"  High (>0.7): {sum(1 for s in scores if s > 0.7)}")
    print(f"  Low (<0.4): {sum(1 for s in scores if s < 0.4)}")

    # Show top and bottom
    sorted_claims = sorted(
        simulator.all_claims,
        key=lambda c: simulator.plausibility[c['id']],
        reverse=True
    )

    print(f"\n🏆 TOP PLAUSIBILITY:")
    for claim in sorted_claims[:5]:
        score = simulator.plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")

    print(f"\n⚠️  BOTTOM PLAUSIBILITY:")
    for claim in sorted_claims[-5:]:
        score = simulator.plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")


if __name__ == "__main__":
    asyncio.run(main())
