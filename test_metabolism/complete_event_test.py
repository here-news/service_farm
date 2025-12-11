"""
COMPLETE Event Topology Test

1. Find ALL pages about Hong Kong fire
2. Get ALL claims from those pages (no keyword filtering)
3. Process incrementally with progress updates
4. Generate weighted narrative
5. Output structured results to JSON file
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict
from collections import defaultdict
import json
from datetime import datetime

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class CompleteEventTest:
    """Complete event topology analysis"""

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client
        self.llm_calls = 0
        self.embeddings = {}
        self.plausibility = {}
        self.network = defaultdict(list)
        self.all_claims = []
        self.analyses = []

    async def analyze_with_validation(self, claims: List[dict]) -> dict:
        """Analyze claims with temporal awareness"""
        self.llm_calls += 1

        example_json = {
            "topic": "Fire incident at Wang Fuk Court",
            "pattern": "progressive",
            "plausibility_scores": {
                claims[0]['id']: 0.85,
                claims[1]['id']: 0.65
            } if len(claims) >= 2 else {claims[0]['id']: 0.75},
            "consensus_points": ["Fire occurred at Wang Fuk Court"],
            "contradictions": ["Death toll varies: 36 vs 156"]
        }

        claims_text = []
        for i, claim in enumerate(claims, 1):
            time_str = claim.get('event_time', '')[:19] if claim.get('event_time') else 'unknown'
            claims_text.append(f"{i}. ID={claim['id'][:12]} @ {time_str}")
            claims_text.append(f"   {claim['text']}")

        prompt = f"""Analyze {len(claims)} claims about a news event. Assign plausibility scores.

CLAIMS (chronologically ordered):
{chr(10).join(claims_text)}

SCORING RULES:

1. PROGRESSIVE UPDATES (numbers increase over time):
   - LATEST claim → HIGHEST score (0.80-0.95)
   - Earlier claims → moderate (0.65-0.80) - accurate at the time
   - Example: "4 deaths" @ Day 1 (0.70), "36 deaths" @ Day 2 (0.80), "156 deaths" @ Day 3 (0.90)

2. CONTRADICTORY (conflicting at same time):
   - Equal sources → both ~0.55 (genuine uncertainty)
   - Unequal sources → majority 0.75-0.85, minority 0.30-0.45

3. CONSENSUS (corroborated):
   - Multiple sources agree → 0.75-0.95

4. COMPLEMENTARY (different aspects):
   - Add context/details → 0.65-0.80

5. OUTLIERS:
   - Single unsupported claim → 0.10-0.30

CRITICAL: Return scores for ALL {len(claims)} claim IDs.

OUTPUT JSON:
{json.dumps(example_json, indent=2)}

Return ONLY valid JSON."""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            data = json.loads(response.choices[0].message.content)

            if 'plausibility_scores' not in data or len(data['plausibility_scores']) == 0:
                raise ValueError("Missing or empty plausibility_scores")

            scores = data['plausibility_scores']
            missing = [c['id'] for c in claims if c['id'] not in scores]
            if missing:
                for claim_id in missing:
                    scores[claim_id] = 0.5

            return data

        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            return {
                'topic': 'unknown',
                'pattern': 'mixed',
                'plausibility_scores': {c['id']: 0.5 for c in claims},
                'consensus_points': [],
                'contradictions': []
            }

    async def add_batch(self, new_claims: List[dict], batch_name: str):
        """Add batch of claims and update"""
        print(f"\n{'='*80}")
        print(f"📥 {batch_name}")
        print(f"{'='*80}")
        print(f"Adding {len(new_claims)} claims...")

        old_scores = dict(self.plausibility)
        self.all_claims.extend(new_claims)

        # Embeddings
        print(f"  🔮 Generating embeddings...")
        for i, claim in enumerate(new_claims):
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=claim['text']
            )
            self.embeddings[claim['id']] = response.data[0].embedding
            if (i + 1) % 10 == 0:
                print(f"     {i+1}/{len(new_claims)} embeddings...")

        # Network
        print(f"  🕸️  Building network...")
        edges_before = sum(len(neighbors) for neighbors in self.network.values()) // 2

        for new_claim in new_claims:
            for existing_claim in self.all_claims:
                if new_claim['id'] == existing_claim['id']:
                    continue

                vec1 = np.array(self.embeddings[new_claim['id']])
                vec2 = np.array(self.embeddings[existing_claim['id']])

                similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

                if similarity > 0.4:
                    self.network[new_claim['id']].append((existing_claim['id'], similarity))
                    self.network[existing_claim['id']].append((new_claim['id'], similarity))

        edges_after = sum(len(neighbors) for neighbors in self.network.values()) // 2
        print(f"     Network: {len(self.all_claims)} nodes, {edges_after} edges (+{edges_after - edges_before})")

        # Analyze
        print(f"  🧮 Analyzing all {len(self.all_claims)} claims...")
        analysis = await self.analyze_with_validation(self.all_claims)
        self.analyses.append(analysis)

        # Update scores
        for claim in self.all_claims:
            self.plausibility[claim['id']] = analysis['plausibility_scores'].get(claim['id'], 0.5)

        # Show results
        scores = list(self.plausibility.values())
        print(f"\n  📊 Analysis Results:")
        print(f"     Topic: {analysis['topic']}")
        print(f"     Pattern: {analysis['pattern']}")
        print(f"     Plausibility: min={min(scores):.2f}, max={max(scores):.2f}, mean={np.mean(scores):.2f}")
        print(f"     Consensus: {len(analysis.get('consensus_points', []))} points")
        print(f"     Contradictions: {len(analysis.get('contradictions', []))}")

        # Show changes
        if old_scores:
            changes = [(cid, old_scores[cid], self.plausibility[cid])
                      for cid in old_scores
                      if abs(self.plausibility[cid] - old_scores[cid]) > 0.05]
            if changes:
                changes.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
                print(f"\n  📈 Top score changes:")
                for cid, old, new in changes[:3]:
                    claim = next(c for c in self.all_claims if c['id'] == cid)
                    arrow = "↑" if new > old else "↓"
                    print(f"     {old:.2f}→{new:.2f} {arrow} {claim['text'][:60]}...")

    async def generate_narrative(self) -> str:
        """Generate weighted narrative"""
        self.llm_calls += 1
        print(f"\n{'='*80}")
        print(f"📖 GENERATING NARRATIVE")
        print(f"{'='*80}")

        analysis = self.analyses[-1]

        sorted_claims = sorted(self.all_claims, key=lambda c: self.plausibility[c['id']], reverse=True)

        high = [c for c in sorted_claims if self.plausibility[c['id']] >= 0.75]
        mid = [c for c in sorted_claims if 0.55 <= self.plausibility[c['id']] < 0.75]
        low = [c for c in sorted_claims if self.plausibility[c['id']] < 0.55]

        high_text = "\n".join(f"  [{self.plausibility[c['id']]:.2f}] {c['text']}" for c in high[:15])
        mid_text = "\n".join(f"  [{self.plausibility[c['id']]:.2f}] {c['text']}" for c in mid[:10])
        low_text = "\n".join(f"  [{self.plausibility[c['id']]:.2f}] {c['text']}" for c in low[:5])

        consensus = "\n".join(f"  • {p}" for p in analysis.get('consensus_points', []))
        contradictions = "\n".join(f"  • {c}" for c in analysis.get('contradictions', []))

        prompt = f"""Generate comprehensive event narrative using weighted claims.

TOPIC: {analysis['topic']}
PATTERN: {analysis['pattern']}

CONSENSUS POINTS:
{consensus or "  (none)"}

CONTRADICTIONS:
{contradictions or "  (none)"}

HIGH PLAUSIBILITY (≥0.75) - EMPHASIZE:
{high_text or "  (none)"}

MEDIUM PLAUSIBILITY (0.55-0.74) - Supporting info:
{mid_text or "  (none)"}

LOW PLAUSIBILITY (<0.55) - Mention as uncertain:
{low_text or "  (none)"}

INSTRUCTIONS:
1. Build from HIGH plausibility claims
2. Use consensus points as facts
3. Show contradictions as ranges/uncertainty
4. Include medium claims as context
5. Mark low claims as "unverified"
6. Organize by topics (what/when/casualties/response/impact)
7. Use markdown headers
8. NO speculation

Generate narrative:"""

        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content


async def main():
    print("="*80)
    print("🔥 COMPLETE EVENT TOPOLOGY TEST")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    # Step 1: Find fire-related pages
    print("\nStep 1: Finding fire-related pages...")
    fire_pages = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WHERE c.text CONTAINS 'Wang Fuk Court' OR c.text CONTAINS 'Tai Po'
        RETURN DISTINCT p.id as page_id, p.url as url
    """)

    page_ids = [p['page_id'] for p in fire_pages]
    print(f"Found {len(page_ids)} pages")
    for p in fire_pages:
        print(f"  - {p['page_id']}: {p['url'][:70]}...")

    # Step 2: Get ALL claims from these pages
    print(f"\nStep 2: Getting ALL claims from {len(page_ids)} pages...")
    all_claims = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WHERE p.id IN $page_ids
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time, p.id as page_id
        ORDER BY c.event_time, c.id
    """, {'page_ids': page_ids})

    all_claims = [dict(c) for c in all_claims]

    await neo4j.close()

    print(f"Total claims: {len(all_claims)}")

    # Group by page
    by_page = defaultdict(list)
    for c in all_claims:
        by_page[c['page_id']].append(c)

    print(f"\nClaims per page:")
    for pid in sorted(by_page.keys()):
        print(f"  {pid}: {len(by_page[pid])} claims")

    # Step 3: Process incrementally
    print(f"\n{'='*80}")
    print("INCREMENTAL PROCESSING")
    print(f"{'='*80}")

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    tester = CompleteEventTest(openai_client)

    sorted_pages = sorted(page_ids)

    # Add first 3 pages individually
    for i in range(min(3, len(sorted_pages))):
        await tester.add_batch(by_page[sorted_pages[i]], f"Page {i+1}/{len(sorted_pages)} ({sorted_pages[i]})")

    # Add remaining in batch
    if len(sorted_pages) > 3:
        remaining = []
        for pid in sorted_pages[3:]:
            remaining.extend(by_page[pid])
        await tester.add_batch(remaining, f"Remaining {len(sorted_pages)-3} pages ({len(remaining)} claims)")

    # Step 4: Generate narrative
    narrative = await tester.generate_narrative()

    print(f"\n{'='*80}")
    print("📖 FINAL NARRATIVE")
    print(f"{'='*80}")
    print(narrative)

    # Step 5: Output results
    print(f"\n{'='*80}")
    print("💾 SAVING RESULTS")
    print(f"{'='*80}")

    sorted_claims = sorted(tester.all_claims, key=lambda c: tester.plausibility[c['id']], reverse=True)

    high_plausibility = [
        {
            'id': c['id'],
            'text': c['text'],
            'plausibility': tester.plausibility[c['id']],
            'event_time': c.get('event_time')
        }
        for c in sorted_claims if tester.plausibility[c['id']] >= 0.75
    ]

    low_plausibility = [
        {'id': c['id'], 'plausibility': tester.plausibility[c['id']]}
        for c in sorted_claims if tester.plausibility[c['id']] < 0.55
    ]

    output = {
        'event_id': 'ev_pth3a8dc',
        'timestamp': datetime.utcnow().isoformat(),
        'summary': {
            'total_claims': len(tester.all_claims),
            'total_pages': len(page_ids),
            'network_edges': sum(len(n) for n in tester.network.values()) // 2,
            'llm_calls': tester.llm_calls,
            'plausibility_stats': {
                'min': float(min(tester.plausibility.values())),
                'max': float(max(tester.plausibility.values())),
                'mean': float(np.mean(list(tester.plausibility.values()))),
                'high_count': len(high_plausibility),
                'low_count': len(low_plausibility)
            }
        },
        'analysis': tester.analyses[-1],
        'narrative': narrative,
        'high_plausibility_claims': high_plausibility,
        'low_plausibility_claims': low_plausibility
    }

    output_file = '/tmp/event_topology_result.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"✅ Saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total claims: {len(tester.all_claims)}")
    print(f"  High plausibility (≥0.75): {len(high_plausibility)}")
    print(f"  Low plausibility (<0.55): {len(low_plausibility)}")
    print(f"  Network edges: {sum(len(n) for n in tester.network.values()) // 2}")
    print(f"  LLM calls: {tester.llm_calls}")

    print(f"\n🏆 TOP 10:")
    for c in sorted_claims[:10]:
        print(f"  {tester.plausibility[c['id']]:.2f}: {c['text'][:70]}...")

    print(f"\n⚠️  BOTTOM 5:")
    for c in sorted_claims[-5:]:
        print(f"  {tester.plausibility[c['id']]:.2f}: {c['text'][:70]}...")


if __name__ == "__main__":
    asyncio.run(main())
