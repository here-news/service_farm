"""
Extended incremental growth simulation with MORE pages

Simulates realistic event growth:
1. Start with existing 20 claims (2 pages already linked)
2. Add 20 more claims from 7 additional related pages
3. Show how plausibility evolves with larger network
4. Generate narrative at each stage

Read-only from DB, all computations local
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


class ExtendedGrowthSimulator:
    """Simulate extended event growth with multiple pages"""

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
        """Analyze cluster with JSON validation"""
        self.llm_calls += 1

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
        """Add new claims and update plausibility scores incrementally"""
        print(f"\n{'='*80}")
        print(f"📥 Adding {batch_name}: {len(new_claims)} new claims")
        print(f"{'='*80}")

        old_scores = dict(self.plausibility)
        self.all_claims.extend(new_claims)

        # Generate embeddings for new claims only
        print(f"   🔮 Generating embeddings for {len(new_claims)} new claims...")
        for claim in new_claims:
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=claim['text']
            )
            self.embeddings[claim['id']] = response.data[0].embedding

        # Build network edges for new claims
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

        # Re-analyze entire cluster
        print(f"   🧮 Re-analyzing all {len(self.all_claims)} claims...")
        analysis = await self.analyze_cluster_with_validation(self.all_claims)

        # Update plausibility scores
        for claim in self.all_claims:
            score = analysis['plausibility_scores'].get(claim['id'], 0.5)
            self.plausibility[claim['id']] = score

        # Show changes
        print(f"\n   📊 Changes:")
        print(f"   Topic: {analysis['topic']}")
        print(f"   Pattern: {analysis['pattern']}")

        # New claims
        print(f"\n   🆕 New claims added:")
        for claim in new_claims[:3]:  # Top 3
            score = self.plausibility[claim['id']]
            print(f"      {score:.2f}: {claim['text'][:70]}...")

        # Significant score changes
        changes = []
        for claim_id, new_score in self.plausibility.items():
            if claim_id in old_scores:
                old_score = old_scores[claim_id]
                if abs(new_score - old_score) > 0.05:
                    changes.append((claim_id, old_score, new_score))

        if changes:
            changes.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
            print(f"\n   📈 Score changes (top 3):")
            for claim_id, old_score, new_score in changes[:3]:
                change = "↑" if new_score > old_score else "↓"
                claim = next(c for c in self.all_claims if c['id'] == claim_id)
                print(f"      {old_score:.2f} → {new_score:.2f} {change}: {claim['text'][:60]}...")

        # Stats
        scores = list(self.plausibility.values())
        print(f"\n   Range: {min(scores):.2f}-{max(scores):.2f}, Mean: {np.mean(scores):.2f}")
        print(f"   Consensus: {len(analysis.get('consensus_points', []))} | Contradictions: {len(analysis.get('contradictions', []))}")

        return dict(self.plausibility), analysis

    async def generate_narrative(self, analysis: dict) -> str:
        """Generate weighted narrative"""
        self.llm_calls += 1

        claims_sorted = sorted(
            self.all_claims,
            key=lambda c: self.plausibility[c['id']],
            reverse=True
        )

        high_claims = [c for c in claims_sorted if self.plausibility[c['id']] >= 0.70]
        mid_claims = [c for c in claims_sorted if 0.45 <= self.plausibility[c['id']] < 0.70]
        low_claims = [c for c in claims_sorted if self.plausibility[c['id']] < 0.45]

        high_text = "\n".join(
            f"  [{self.plausibility[c['id']]:.2f}] {c['text']}"
            for c in high_claims[:10]
        ) if high_claims else "  (none)"

        mid_text = "\n".join(
            f"  [{self.plausibility[c['id']]:.2f}] {c['text']}"
            for c in mid_claims[:8]
        ) if mid_claims else "  (none)"

        low_text = "\n".join(
            f"  [{self.plausibility[c['id']]:.2f}] {c['text']}"
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
3. For CONTRADICTIONS, show as ranges
4. Include MEDIUM claims as supporting details
5. Mention LOW claims ONLY if relevant, as "unverified reports"
6. Organize by natural topics
7. Use structured markdown format
8. NO speculation, NO hallucination

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
    print("🌱 EXTENDED EVENT GROWTH SIMULATION")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    simulator = ExtendedGrowthSimulator(openai_client)

    event_id = 'ev_pth3a8dc'

    # Get claims from pages already linked
    existing_claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time
        ORDER BY c.event_time
    """, {'event_id': event_id})
    existing_claims = [dict(c) for c in existing_claims]

    # Get page IDs already linked
    linked_pages = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page)
        RETURN DISTINCT p.id as page_id
    """, {'event_id': event_id})
    linked_ids = [p['page_id'] for p in linked_pages]

    # Get claims from related pages NOT yet linked
    new_claims = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WHERE (c.text CONTAINS 'Wang Fuk Court' OR c.text CONTAINS 'Tai Po' OR c.text CONTAINS 'Hong Kong fire')
        AND NOT p.id IN $linked_ids
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time, p.id as page_id
        ORDER BY c.event_time
    """, {'linked_ids': linked_ids})
    new_claims = [dict(c) for c in new_claims]

    await neo4j.close()

    print(f"\n📊 Event: {event_id}")
    print(f"   Existing claims (already linked): {len(existing_claims)}")
    print(f"   New claims (from related pages): {len(new_claims)}")
    print(f"   Total: {len(existing_claims) + len(new_claims)}")

    # Simulate growth
    print(f"\n{'='*80}")
    print("PHASE 1: Process existing claims")
    print(f"{'='*80}")

    await simulator.add_claims_and_update(existing_claims, "Existing claims")
    narrative1 = await simulator.generate_narrative(
        {'topic': 'Hong Kong fire', 'pattern': 'mixed', 'consensus_points': [], 'contradictions': []}
    )

    print(f"\n{'='*80}")
    print("PHASE 2: Add new pages")
    print(f"{'='*80}")

    await simulator.add_claims_and_update(new_claims, "New related pages")
    narrative2 = await simulator.generate_narrative(
        {'topic': 'Hong Kong fire', 'pattern': 'mixed', 'consensus_points': [], 'contradictions': []}
    )

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
    print(f"  Medium (0.5-0.7): {sum(1 for s in scores if 0.5 <= s < 0.7)}")
    print(f"  Low (<0.5): {sum(1 for s in scores if s < 0.5)}")

    sorted_claims = sorted(
        simulator.all_claims,
        key=lambda c: simulator.plausibility[c['id']],
        reverse=True
    )

    print(f"\n🏆 TOP 5 PLAUSIBILITY:")
    for claim in sorted_claims[:5]:
        score = simulator.plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")

    print(f"\n⚠️  BOTTOM 5 PLAUSIBILITY:")
    for claim in sorted_claims[-5:]:
        score = simulator.plausibility[claim['id']]
        print(f"   {score:.2f}: {claim['text'][:70]}...")

    print(f"\n{'='*80}")
    print("📖 FINAL NARRATIVE (with all pages)")
    print(f"{'='*80}")
    print(narrative2)


if __name__ == "__main__":
    asyncio.run(main())
