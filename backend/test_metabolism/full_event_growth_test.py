"""
FULL Event Growth Test - ALL claims from fire-related pages

Proper approach:
1. Identify pages about Hong Kong fire (by URL or initial claim)
2. Get ALL claims from those pages (not just keyword-filtered)
3. Run incremental plausibility analysis
4. Generate weighted narrative

This simulates real production: event worker links pages to event,
then we analyze ALL claims from linked pages.
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict
from collections import defaultdict
import json

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class FullEventGrowthTest:
    """Test with ALL claims from fire-related pages"""

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client
        self.llm_calls = 0

        # Local state
        self.embeddings = {}
        self.plausibility = {}
        self.network = defaultdict(list)
        self.all_claims = []

    async def analyze_cluster_with_validation(
        self,
        cluster_claims: List[dict]
    ) -> dict:
        """Analyze cluster with JSON validation and temporal awareness"""
        self.llm_calls += 1

        example_json = {
            "topic": "brief topic description",
            "pattern": "consensus|mixed|contradictory|progressive",
            "plausibility_scores": {
                cluster_claims[0]['id']: 0.85,
                cluster_claims[1]['id']: 0.65
            } if len(cluster_claims) >= 2 else {cluster_claims[0]['id']: 0.75},
            "consensus_points": ["established fact 1"],
            "contradictions": ["conflict description"]
        }

        claims_text = []
        for i, claim in enumerate(cluster_claims, 1):
            time_str = claim.get('event_time', '')[:19] if claim.get('event_time') else 'unknown'
            claims_text.append(f"{i}. ID={claim['id'][:12]} @ {time_str}")
            claims_text.append(f"   Text: \"{claim['text']}\"")

        prompt = f"""Analyze these {len(cluster_claims)} claims and assign plausibility scores.

CLAIMS (sorted chronologically):
{chr(10).join(claims_text)}

TASK:
1. Identify the topic/facet
2. Detect pattern:
   - consensus: claims agree
   - mixed: complementary info
   - contradictory: genuine conflict (equal sources)
   - progressive: numbers UPDATE over time (e.g., death toll rises)

3. Assign plausibility score (0.0-1.0) to EVERY claim:

   For PROGRESSIVE patterns (casualty counts that increase over time):
   - LATEST temporal claim → HIGHEST score (0.80-0.95)
   - Earlier claims → moderate scores (0.60-0.75) - they were accurate at the time
   - Example: "4 deaths" @ Nov 26 → 0.70, "36 deaths" @ Nov 27 → 0.75, "156 deaths" @ Nov 28 → 0.85

   For CONTRADICTORY patterns (conflicting at same time):
   - If sources are EQUAL (e.g., 3 vs 3) → BOTH get ~0.55 (genuine uncertainty)
   - If sources are UNEQUAL (e.g., 2 vs 4) → majority wins (0.75-0.85), minority lower (0.30-0.45)

   For CONSENSUS:
   - Corroborated claims → 0.75-0.95
   - Complementary info → 0.60-0.80

   For OUTLIERS:
   - Single claim contradicting many → 0.10-0.30

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

            if 'plausibility_scores' not in data:
                raise ValueError("Missing plausibility_scores in response")

            scores = data['plausibility_scores']
            if len(scores) == 0:
                raise ValueError("Empty plausibility_scores")

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
        """Add new claims and update plausibility"""
        print(f"\n{'='*80}")
        print(f"📥 {batch_name}: Adding {len(new_claims)} claims")
        print(f"{'='*80}")

        old_scores = dict(self.plausibility)
        self.all_claims.extend(new_claims)

        # Generate embeddings
        print(f"   🔮 Generating embeddings...")
        for claim in new_claims:
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=claim['text']
            )
            self.embeddings[claim['id']] = response.data[0].embedding

        # Build network edges
        print(f"   🕸️  Building network edges...")
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

        total_edges = sum(len(neighbors) for neighbors in self.network.values()) // 2
        print(f"   Added {edges_added} edges (total: {total_edges})")

        # Re-analyze all claims
        print(f"   🧮 Re-analyzing all {len(self.all_claims)} claims...")
        analysis = await self.analyze_cluster_with_validation(self.all_claims)

        # Update scores
        for claim in self.all_claims:
            score = analysis['plausibility_scores'].get(claim['id'], 0.5)
            self.plausibility[claim['id']] = score

        # Show results
        print(f"\n   📊 Results:")
        print(f"   Topic: {analysis['topic']}")
        print(f"   Pattern: {analysis['pattern']}")
        print(f"   Consensus: {len(analysis.get('consensus_points', []))}")
        print(f"   Contradictions: {len(analysis.get('contradictions', []))}")

        # Show sample new claims
        print(f"\n   🆕 Sample new claims:")
        for claim in new_claims[:3]:
            score = self.plausibility[claim['id']]
            print(f"      {score:.2f}: {claim['text'][:70]}...")

        # Show significant changes
        changes = []
        for claim_id, new_score in self.plausibility.items():
            if claim_id in old_scores:
                old_score = old_scores[claim_id]
                if abs(new_score - old_score) > 0.05:
                    changes.append((claim_id, old_score, new_score))

        if changes:
            changes.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
            print(f"\n   📈 Top score changes:")
            for claim_id, old_score, new_score in changes[:3]:
                change = "↑" if new_score > old_score else "↓"
                claim = next(c for c in self.all_claims if c['id'] == claim_id)
                print(f"      {old_score:.2f} → {new_score:.2f} {change}: {claim['text'][:60]}...")

        scores = list(self.plausibility.values())
        print(f"\n   Range: {min(scores):.2f}-{max(scores):.2f}, Mean: {np.mean(scores):.2f}")

        return dict(self.plausibility), analysis


async def main():
    print("="*80)
    print("🌱 FULL EVENT GROWTH TEST - ALL CLAIMS FROM FIRE PAGES")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    tester = FullEventGrowthTest(openai_client)

    # Get all pages about Hong Kong fire (by keyword in ANY claim)
    fire_pages = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WHERE c.text CONTAINS 'Wang Fuk Court' OR c.text CONTAINS 'Tai Po'
           OR c.text CONTAINS 'Hong Kong fire'
        RETURN DISTINCT p.id as page_id, p.url as url
        ORDER BY p.id
    """)

    fire_page_ids = [p['page_id'] for p in fire_pages]
    print(f"\nIdentified {len(fire_page_ids)} pages about Hong Kong fire")

    # Get ALL claims from these pages (not filtered by keywords!)
    all_claims = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WHERE p.id IN $page_ids
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time, p.id as page_id
        ORDER BY c.event_time
    """, {'page_ids': fire_page_ids})

    all_claims = [dict(c) for c in all_claims]

    await neo4j.close()

    print(f"Total claims from fire pages: {len(all_claims)}")

    # Group claims by page
    claims_by_page = defaultdict(list)
    for claim in all_claims:
        claims_by_page[claim['page_id']].append(claim)

    print(f"\nClaims per page:")
    for page_id in sorted(claims_by_page.keys()):
        print(f"  {page_id}: {len(claims_by_page[page_id])} claims")

    # Simulate incremental growth: add pages one by one
    print(f"\n{'='*80}")
    print("INCREMENTAL GROWTH SIMULATION")
    print(f"{'='*80}")

    sorted_pages = sorted(fire_page_ids)

    # Batch 1: First page (all claims)
    batch1 = claims_by_page[sorted_pages[0]]
    await tester.add_claims_and_update(batch1, f"Page 1 ({sorted_pages[0]})")

    # Batch 2: Add second page
    if len(sorted_pages) > 1:
        batch2 = claims_by_page[sorted_pages[1]]
        await tester.add_claims_and_update(batch2, f"Page 2 ({sorted_pages[1]})")

    # Batch 3: Add third page
    if len(sorted_pages) > 2:
        batch3 = claims_by_page[sorted_pages[2]]
        await tester.add_claims_and_update(batch3, f"Page 3 ({sorted_pages[2]})")

    # Batch 4: Add remaining pages in one batch
    if len(sorted_pages) > 3:
        remaining = []
        for page_id in sorted_pages[3:]:
            remaining.extend(claims_by_page[page_id])
        await tester.add_claims_and_update(remaining, f"Remaining {len(sorted_pages)-3} pages")

    # Final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal LLM calls: {tester.llm_calls}")
    print(f"Total claims: {len(tester.all_claims)}")
    print(f"Network edges: {sum(len(neighbors) for neighbors in tester.network.values()) // 2}")

    scores = list(tester.plausibility.values())
    print(f"\nPlausibility distribution:")
    print(f"  Min: {min(scores):.2f}, Max: {max(scores):.2f}, Mean: {np.mean(scores):.2f}")
    print(f"  High (>0.7): {sum(1 for s in scores if s > 0.7)}")
    print(f"  Medium (0.5-0.7): {sum(1 for s in scores if 0.5 <= s < 0.7)}")
    print(f"  Low (<0.5): {sum(1 for s in scores if s < 0.5)}")

    # Show top and bottom
    sorted_claims = sorted(
        tester.all_claims,
        key=lambda c: tester.plausibility[c['id']],
        reverse=True
    )

    print(f"\n🏆 TOP 10 PLAUSIBILITY:")
    for claim in sorted_claims[:10]:
        score = tester.plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")

    print(f"\n⚠️  BOTTOM 10 PLAUSIBILITY:")
    for claim in sorted_claims[-10:]:
        score = tester.plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")


if __name__ == "__main__":
    asyncio.run(main())
