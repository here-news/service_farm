"""
General claim network topology - not just metrics

This test demonstrates:
1. All claims connect via semantic similarity (not just metric-based)
2. Claims cluster by topic/facet (casualties, timeline, response, impact, etc.)
3. Within each cluster, claims agree or contradict
4. Plausibility resolution happens per-cluster
5. Event narrative uses high-plausibility claims from all clusters

Goal: See complete claim topology across ALL statement types,
      not just quantitative metrics
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import json

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class GeneralClaimNetworkAnalyzer:
    """Analyze complete claim network - all statement types"""

    def __init__(self, neo4j: Neo4jService, openai_client: AsyncOpenAI):
        self.neo4j = neo4j
        self.openai = openai_client
        self.llm_calls = 0

    # =========================================================================
    # STEP 1: Build semantic similarity network (ALL claims)
    # =========================================================================

    async def build_semantic_network(
        self,
        claims: List[dict],
        embeddings: Dict[str, List[float]],
        threshold: float = 0.5  # Lower threshold to capture more relationships
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Build claim network based on semantic similarity

        Returns: {claim_id: [(other_claim_id, similarity)]}
        """
        network = defaultdict(list)

        print(f"\n🕸️  Building semantic network (threshold={threshold})...")
        print("="*80)

        edges = 0
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                # Calculate cosine similarity
                vec1 = np.array(embeddings[claim1['id']])
                vec2 = np.array(embeddings[claim2['id']])

                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    similarity = float(dot / (norm1 * norm2))
                else:
                    similarity = 0.0

                # Connect if above threshold
                if similarity > threshold:
                    network[claim1['id']].append((claim2['id'], similarity))
                    network[claim2['id']].append((claim1['id'], similarity))
                    edges += 1

                    if similarity > 0.7:  # Print high-similarity edges
                        print(f"\n🔗 {similarity:.2f}:")
                        print(f"   {claim1['text'][:70]}...")
                        print(f"   {claim2['text'][:70]}...")

        print(f"\n📊 Network stats: {len(network)} nodes, {edges} edges")
        return dict(network)

    # =========================================================================
    # STEP 2: Detect claim clusters (topics/facets)
    # =========================================================================

    async def detect_clusters(
        self,
        claims: List[dict],
        network: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, Set[str]]:
        """
        Detect clusters of related claims using community detection

        Returns: {cluster_id: {claim_ids}}
        """
        print("\n📊 Detecting claim clusters...")
        print("="*80)

        # Simple community detection: connected components
        visited = set()
        clusters = {}
        cluster_id = 0

        for claim_id in claims:
            if claim_id['id'] in visited:
                continue

            # BFS to find connected component
            cluster = set()
            queue = [claim_id['id']]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.add(current)

                # Add neighbors
                for neighbor_id, sim in network.get(current, []):
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)

            if cluster:
                clusters[f"cluster_{cluster_id}"] = cluster
                cluster_id += 1

        # Print clusters
        for cid, claim_ids in clusters.items():
            if len(claim_ids) > 1:
                print(f"\n{cid}: {len(claim_ids)} claims")
                for claim_id in list(claim_ids)[:3]:
                    claim = next(c for c in claims if c['id'] == claim_id)
                    print(f"   - {claim['text'][:60]}...")

        return clusters

    # =========================================================================
    # STEP 3: Analyze each cluster for agreement/contradiction
    # =========================================================================

    async def analyze_cluster(
        self,
        cluster_claims: List[dict],
        network: Dict[str, List[Tuple[str, float]]]
    ) -> dict:
        """
        Analyze cluster to determine if claims agree or contradict

        Uses LLM to:
        1. Identify cluster topic/facet
        2. Detect agreements and contradictions
        3. Assign plausibility scores

        Returns: {
          'topic': str,
          'pattern': 'consensus|mixed|contradictory',
          'plausibility_scores': {claim_id: float},
          'reasoning': str
        }
        """
        if len(cluster_claims) == 1:
            return {
                'topic': 'isolated',
                'pattern': 'consensus',
                'plausibility_scores': {cluster_claims[0]['id']: 0.7},
                'reasoning': 'Single claim, no comparison'
            }

        self.llm_calls += 1

        # Format claims for LLM
        claims_text = []
        for i, claim in enumerate(cluster_claims, 1):
            time_str = claim.get('event_time', 'unknown')[:19] if claim.get('event_time') else 'unknown'
            claims_text.append(f"{i}. [{claim['id'][:8]}] @ {time_str}")
            claims_text.append(f"   \"{claim['text']}\"")

        prompt = f"""Analyze these {len(cluster_claims)} related claims from a news event:

{chr(10).join(claims_text)}

Task:
1. What topic/facet do these claims discuss? (casualties, timeline, cause, impact, response, etc.)
2. Do they agree, contradict, or provide complementary information?
3. Assign plausibility scores (0.0-1.0) based on:
   - **Consensus**: Claims that agree with others → higher scores
   - **Contradictions**: Claims that conflict → lower for outliers, higher for majority
   - **Complementary**: Different aspects of same topic → moderate-high scores
   - **Temporal progression**: Updates over time → latest gets highest if credible

Return JSON:
{{
  "topic": "topic name",
  "pattern": "consensus|mixed|contradictory",
  "plausibility_scores": {{
    "claim_id": 0.85,
    "claim_id": 0.45
  }},
  "reasoning": "Explain the pattern and why each score was assigned"
}}"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"  ⚠️  Cluster analysis failed: {e}")
            return {
                'topic': 'unknown',
                'pattern': 'mixed',
                'plausibility_scores': {c['id']: 0.6 for c in cluster_claims},
                'reasoning': f'Analysis failed: {e}'
            }

    # =========================================================================
    # STEP 4: Resolve plausibility across all clusters
    # =========================================================================

    async def resolve_plausibility(
        self,
        claims: List[dict],
        clusters: Dict[str, Set[str]],
        network: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, float]:
        """
        Resolve plausibility for all claims using cluster analysis

        Returns: {claim_id: plausibility_score}
        """
        print("\n🧮 Resolving plausibility from cluster topology...")
        print("="*80)

        # Initialize with priors
        plausibility = {c['id']: c.get('confidence', 0.6) for c in claims}

        # Analyze each cluster
        for cluster_id, claim_ids in clusters.items():
            if len(claim_ids) <= 1:
                continue

            cluster_claims = [c for c in claims if c['id'] in claim_ids]

            print(f"\n📊 Analyzing {cluster_id} ({len(cluster_claims)} claims)...")

            analysis = await self.analyze_cluster(cluster_claims, network)

            print(f"   Topic: {analysis['topic']}")
            print(f"   Pattern: {analysis['pattern']}")
            print(f"   Reasoning: {analysis['reasoning'][:100]}...")

            # Update plausibility scores
            for claim_id, score in analysis['plausibility_scores'].items():
                old = plausibility.get(claim_id, 0.6)
                # Weighted combination: 30% prior, 70% cluster analysis
                plausibility[claim_id] = 0.3 * old + 0.7 * score
                print(f"      {claim_id[:12]}: {old:.2f} → {plausibility[claim_id]:.2f}")

        return plausibility

    # =========================================================================
    # STEP 5: Generate narrative from topology
    # =========================================================================

    async def generate_narrative(
        self,
        claims: List[dict],
        plausibility: Dict[str, float],
        clusters: Dict[str, Set[str]],
        cluster_analyses: Dict[str, dict]
    ) -> str:
        """
        Generate event narrative using high-plausibility claims

        Organizes by topic clusters, prioritizes consensus claims
        """
        print("\n📖 Generating narrative from topology...")
        print("="*80)

        # Get high-plausibility claims
        winners = [
            (c, plausibility[c['id']])
            for c in claims
            if plausibility[c['id']] >= 0.65  # Lower threshold to include more
        ]
        winners.sort(key=lambda x: x[1], reverse=True)

        print(f"   {len(winners)} high-plausibility claims (≥0.65)")

        # Group winners by cluster topic
        topic_claims = defaultdict(list)
        for claim, score in winners:
            # Find cluster
            cluster_id = None
            for cid, claim_ids in clusters.items():
                if claim['id'] in claim_ids:
                    cluster_id = cid
                    break

            if cluster_id and cluster_id in cluster_analyses:
                topic = cluster_analyses[cluster_id]['topic']
                topic_claims[topic].append((claim, score))

        # Build narrative
        narrative_parts = [
            "# Event Narrative (from claim network topology)",
            "",
            f"**Total claims analyzed:** {len(claims)}",
            f"**High-plausibility claims:** {len(winners)}",
            f"**Topics identified:** {len(topic_claims)}",
            ""
        ]

        # Add claims by topic
        for topic, claims_list in sorted(topic_claims.items()):
            narrative_parts.append(f"## {topic.upper()}")
            narrative_parts.append("")

            for claim, score in claims_list[:5]:  # Top 5 per topic
                time_str = claim.get('event_time', '')[:19] if claim.get('event_time') else 'unknown time'
                narrative_parts.append(f"- **[{score:.2f}]** {claim['text']}")
                narrative_parts.append(f"  _{time_str}_")
                narrative_parts.append("")

        return "\n".join(narrative_parts)


async def main():
    print("="*80)
    print("🕸️  GENERAL CLAIM NETWORK TOPOLOGY")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    analyzer = GeneralClaimNetworkAnalyzer(neo4j, openai_client)

    # Get event and claims
    event_id = 'ev_pth3a8dc'  # Wang Fuk Court Fire

    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time
        ORDER BY c.event_time
    """, {'event_id': event_id})

    claims = [dict(c) for c in claims]

    print(f"\n📊 Event: {event_id}")
    print(f"   Claims: {len(claims)}")

    # STEP 1: Generate embeddings
    print("\n" + "="*80)
    print("STEP 1: Generate embeddings for ALL claims")
    print("="*80)

    embeddings = {}
    for i, claim in enumerate(claims[:20], 1):  # First 20 claims
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=claim['text']
        )
        embeddings[claim['id']] = response.data[0].embedding
        if i % 5 == 0:
            print(f"   Generated {i}/20 embeddings...")

    # STEP 2: Build semantic network
    print("\n" + "="*80)
    print("STEP 2: Build semantic network (ALL claim types)")
    print("="*80)

    network = await analyzer.build_semantic_network(
        claims[:20],
        embeddings,
        threshold=0.4  # Lower threshold to capture more relationships
    )

    # STEP 3: Detect clusters
    print("\n" + "="*80)
    print("STEP 3: Detect claim clusters (topics)")
    print("="*80)

    clusters = await analyzer.detect_clusters(claims[:20], network)

    # STEP 4: Analyze each cluster
    print("\n" + "="*80)
    print("STEP 4: Analyze clusters for agreement/contradiction")
    print("="*80)

    cluster_analyses = {}
    for cluster_id, claim_ids in clusters.items():
        if len(claim_ids) > 1:
            cluster_claims = [c for c in claims[:20] if c['id'] in claim_ids]
            analysis = await analyzer.analyze_cluster(cluster_claims, network)
            cluster_analyses[cluster_id] = analysis

    # STEP 5: Resolve plausibility
    print("\n" + "="*80)
    print("STEP 5: Resolve plausibility from topology")
    print("="*80)

    plausibility = await analyzer.resolve_plausibility(
        claims[:20],
        clusters,
        network
    )

    # Store in Neo4j
    for claim_id, score in plausibility.items():
        await neo4j._execute_write("""
            MATCH (c:Claim {id: $claim_id})
            SET c.plausibility = $score
        """, {'claim_id': claim_id, 'score': score})

    # STEP 6: Generate narrative
    print("\n" + "="*80)
    print("STEP 6: Generate narrative from topology")
    print("="*80)

    narrative = await analyzer.generate_narrative(
        claims[:20],
        plausibility,
        clusters,
        cluster_analyses
    )

    print(f"\n{narrative}")

    # Final summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    print(f"\nTotal LLM calls: {analyzer.llm_calls}")
    print(f"Claims processed: {len(claims[:20])}")
    print(f"Network edges: {sum(len(neighbors) for neighbors in network.values()) // 2}")
    print(f"Clusters detected: {len(clusters)}")

    scores = list(plausibility.values())
    print(f"\nPlausibility distribution:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  High (≥0.7): {sum(1 for s in scores if s >= 0.7)} claims")
    print(f"  Medium (0.5-0.7): {sum(1 for s in scores if 0.5 <= s < 0.7)} claims")
    print(f"  Low (<0.5): {sum(1 for s in scores if s < 0.5)} claims")

    # Show top and bottom claims
    print("\n🏆 TOP PLAUSIBILITY:")
    sorted_claims = sorted(
        [(c, plausibility[c['id']]) for c in claims[:20]],
        key=lambda x: x[1],
        reverse=True
    )
    for claim, score in sorted_claims[:5]:
        print(f"   {score:.3f}: {claim['text'][:70]}...")

    print("\n⚠️  BOTTOM PLAUSIBILITY:")
    for claim, score in sorted_claims[-3:]:
        print(f"   {score:.3f}: {claim['text'][:70]}...")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
