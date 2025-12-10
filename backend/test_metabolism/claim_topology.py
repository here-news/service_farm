"""
Test claim network topology and plausibility resolution

This test demonstrates:
1. Claims form a network based on semantic similarity (talking about same thing)
2. Some agree (AGREES), some contradict (CONTRADICTS)
3. After resolution, claims get plausibility scores
4. Event narrative uses high-plausibility claims (winners)
5. Visualization shows topology: clusters, consensus, outliers

Goal: See how claims cluster around facets, how contradictions resolve,
      and which claims "win" to form event narrative
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import json

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class ClaimTopologyAnalyzer:
    """Analyze claim network topology and plausibility resolution"""

    def __init__(self, neo4j: Neo4jService, openai_client: AsyncOpenAI):
        self.neo4j = neo4j
        self.openai = openai_client
        self.llm_calls = 0

    # =========================================================================
    # STEP 1: Extract and store metrics as ClaimMetric nodes
    # =========================================================================

    async def extract_and_store_metrics(
        self,
        claim_id: str,
        claim_text: str
    ) -> Dict[str, float]:
        """
        Extract metrics using LLM and store as ClaimMetric nodes

        Returns: {metric_type: value}
        """
        self.llm_calls += 1

        prompt = f"""Extract event-related quantitative metrics. Return ONLY valid JSON.

Claim: "{claim_text}"

Return JSON with metrics:
{{
  "deaths": 156,
  "injured": 12
}}

Valid types: deaths, injured, missing, evacuated, hospitalized, displaced

IMPORTANT:
- Only main event metrics (fire casualties)
- Ignore unrelated numbers (arrests, family losses, org counts)
- Return {{}} if no metrics

JSON:"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )

            metrics = json.loads(response.choices[0].message.content)

            # Store each metric as ClaimMetric node
            for metric_type, value in metrics.items():
                await self.neo4j._execute_write("""
                    MATCH (c:Claim {id: $claim_id})
                    MERGE (m:ClaimMetric {claim_id: $claim_id, metric_type: $metric_type})
                    SET m.value = $value,
                        m.extracted_at = datetime()
                    MERGE (c)-[:HAS_METRIC]->(m)
                """, {
                    'claim_id': claim_id,
                    'metric_type': metric_type,
                    'value': float(value)
                })

            return metrics

        except Exception as e:
            print(f"  ⚠️  Metric extraction failed: {e}")
            return {}

    # =========================================================================
    # STEP 2: Build claim similarity network (who talks about same thing)
    # =========================================================================

    async def build_similarity_network(
        self,
        claims: List[dict],
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Build claim network based on semantic similarity

        Returns: {claim_id: [(other_claim_id, similarity, relationship_type)]}

        Relationship types:
        - SIMILAR: High similarity (>0.7) - talking about same facet
        - AGREES: Similar AND same metric values
        - CONTRADICTS: Similar BUT different metric values
        """
        network = defaultdict(list)

        print("\n🕸️  Building claim similarity network...")
        print("="*80)

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

                # Only connect if talking about same thing (>0.6 similarity)
                if similarity > 0.6:
                    # Check if they agree or contradict on metrics
                    relationship = await self._determine_relationship(
                        claim1['id'],
                        claim2['id'],
                        similarity
                    )

                    if relationship:
                        network[claim1['id']].append((claim2['id'], similarity, relationship))
                        network[claim2['id']].append((claim1['id'], similarity, relationship))

                        # Print relationship
                        icon = "✅" if relationship == "AGREES" else "❌" if relationship == "CONTRADICTS" else "↔️"
                        print(f"\n{icon} {relationship} (sim={similarity:.2f}):")
                        print(f"   {claim1['text'][:70]}...")
                        print(f"   {claim2['text'][:70]}...")

        return dict(network)

    async def _determine_relationship(
        self,
        claim1_id: str,
        claim2_id: str,
        similarity: float
    ) -> str:
        """
        Determine if claims agree or contradict based on metrics

        Returns: 'AGREES' | 'CONTRADICTS' | 'SIMILAR'
        """
        # Get metrics for both claims
        metrics1 = await self._get_claim_metrics(claim1_id)
        metrics2 = await self._get_claim_metrics(claim2_id)

        if not metrics1 or not metrics2:
            return "SIMILAR"  # No metrics to compare

        # Check for overlapping metric types
        for m_type in metrics1:
            if m_type in metrics2:
                val1 = metrics1[m_type]
                val2 = metrics2[m_type]

                # Calculate percentage difference
                diff_pct = abs(val1 - val2) / max(val1, val2) if max(val1, val2) > 0 else 0

                if diff_pct > 0.15:  # >15% difference = contradiction
                    return "CONTRADICTS"

        # Similar metrics or no contradictions
        if similarity > 0.8:
            return "AGREES"

        return "SIMILAR"

    async def _get_claim_metrics(self, claim_id: str) -> Dict[str, float]:
        """Get metrics for a claim"""
        results = await self.neo4j._execute_read("""
            MATCH (c:Claim {id: $claim_id})-[:HAS_METRIC]->(m:ClaimMetric)
            RETURN m.metric_type as metric_type, m.value as value
        """, {'claim_id': claim_id})

        return {r['metric_type']: r['value'] for r in results}

    # =========================================================================
    # STEP 3: Resolve plausibility using network topology
    # =========================================================================

    async def resolve_plausibility(
        self,
        claims: List[dict],
        network: Dict[str, List[Tuple[str, float, str]]]
    ) -> Dict[str, float]:
        """
        Resolve claim plausibility using network topology

        Algorithm:
        1. Count agreements and contradictions for each claim
        2. Apply Bayesian updates:
           - Boost claims with many agreements (consensus)
           - Penalize claims with many contradictions (outliers)
           - Consider network effects (neighbors' plausibility)
        3. Iterate until convergence

        Returns: {claim_id: plausibility_score}
        """
        print("\n🧮 Resolving plausibility from network topology...")
        print("="*80)

        # Initialize with priors (extraction confidence)
        plausibility = {c['id']: c.get('confidence', 0.7) for c in claims}

        # Count relationships
        agrees_count = defaultdict(int)
        contradicts_count = defaultdict(int)

        for claim_id, neighbors in network.items():
            for neighbor_id, sim, rel_type in neighbors:
                if rel_type == "AGREES":
                    agrees_count[claim_id] += 1
                elif rel_type == "CONTRADICTS":
                    contradicts_count[claim_id] += 1

        print("\n📊 Network topology statistics:")
        for claim_id in plausibility:
            agrees = agrees_count.get(claim_id, 0)
            contradicts = contradicts_count.get(claim_id, 0)
            if agrees > 0 or contradicts > 0:
                print(f"   {claim_id[:12]}: {agrees} agrees, {contradicts} contradicts")

        # Round 1: Direct relationship effects
        print("\n🔄 Round 1: Direct relationship effects")
        for claim_id in plausibility:
            agrees = agrees_count.get(claim_id, 0)
            contradicts = contradicts_count.get(claim_id, 0)

            if agrees > 0 and contradicts == 0:
                # Pure consensus - boost
                boost = 0.3 * min(agrees / 5.0, 1.0)  # Max boost at 5 agreements
                old = plausibility[claim_id]
                plausibility[claim_id] = min(0.99, old + boost)
                print(f"   ↑ {claim_id[:12]}: {old:.2f} → {plausibility[claim_id]:.2f} (consensus, {agrees} agrees)")

            elif contradicts > agrees:
                # More contradictions than agreements - penalize
                penalty = 0.5  # Heavy penalty for being contradicted
                old = plausibility[claim_id]
                plausibility[claim_id] = max(0.10, old * penalty)
                print(f"   ↓ {claim_id[:12]}: {old:.2f} → {plausibility[claim_id]:.2f} (contradicted, {contradicts} vs {agrees})")

            elif contradicts > 0 and agrees > 0:
                # Mixed signals - moderate penalty
                old = plausibility[claim_id]
                plausibility[claim_id] = old * 0.8
                print(f"   ~ {claim_id[:12]}: {old:.2f} → {plausibility[claim_id]:.2f} (contested, {agrees} vs {contradicts})")

        # Round 2: Network propagation (neighbors influence)
        print("\n🌐 Round 2: Network propagation")
        for iteration in range(3):  # 3 iterations
            updates = {}

            for claim_id, neighbors in network.items():
                if not neighbors:
                    continue

                # Average plausibility of agreeing neighbors
                agreeing_neighbors = [
                    (neighbor_id, sim) for neighbor_id, sim, rel_type in neighbors
                    if rel_type == "AGREES"
                ]

                if agreeing_neighbors:
                    neighbor_scores = [plausibility[n_id] for n_id, sim in agreeing_neighbors]
                    avg_neighbor = np.mean(neighbor_scores)

                    # Influence proportional to number of agreeing neighbors
                    influence = 0.15 * min(len(agreeing_neighbors) / 3.0, 1.0)

                    old = plausibility[claim_id]
                    new = old * (1 - influence) + avg_neighbor * influence
                    updates[claim_id] = new

                    if iteration == 0:  # Only print first iteration
                        print(f"   → {claim_id[:12]}: {old:.2f} → {new:.2f} (influenced by {len(agreeing_neighbors)} neighbors)")

            # Apply updates
            plausibility.update(updates)

        return plausibility

    # =========================================================================
    # STEP 4: Cluster by metric type and analyze patterns
    # =========================================================================

    async def analyze_metric_clusters(
        self,
        event_id: str
    ) -> Dict[str, dict]:
        """
        Analyze clusters of claims by metric type

        Returns: {metric_type: {pattern, claims, reasoning}}
        """
        print("\n📊 Analyzing metric clusters...")
        print("="*80)

        # Get all metric types in event
        metric_types = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)-[:HAS_METRIC]->(m:ClaimMetric)
            RETURN DISTINCT m.metric_type as metric_type
        """, {'event_id': event_id})

        clusters = {}

        for row in metric_types:
            metric_type = row['metric_type']

            # Get cluster
            cluster = await self.neo4j._execute_read("""
                MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)-[:HAS_METRIC]->(m:ClaimMetric {metric_type: $metric_type})
                RETURN c.id as claim_id,
                       c.text as text,
                       c.event_time as time,
                       c.plausibility as plausibility,
                       m.value as value
                ORDER BY c.event_time
            """, {'event_id': event_id, 'metric_type': metric_type})

            cluster_list = [dict(r) for r in cluster]

            if len(cluster_list) >= 2:
                # Analyze with LLM
                analysis = await self._analyze_cluster_pattern(metric_type, cluster_list)
                clusters[metric_type] = {
                    'pattern': analysis['pattern'],
                    'claims': cluster_list,
                    'reasoning': analysis['reasoning']
                }

                print(f"\n📈 {metric_type.upper()}: {len(cluster_list)} claims")
                print(f"   Pattern: {analysis['pattern']}")
                print(f"   Reasoning: {analysis['reasoning'][:100]}...")
                for claim in cluster_list:
                    plaus = claim.get('plausibility', 0.5)
                    time_str = claim['time'][:19] if claim['time'] else 'unknown'
                    print(f"   {'✓' if plaus > 0.7 else '✗'} {claim['value']:>6.0f} @ {time_str} | plausibility={plaus:.2f}")

        return clusters

    async def _analyze_cluster_pattern(
        self,
        metric_type: str,
        cluster: List[dict]
    ) -> dict:
        """Analyze cluster pattern using LLM"""
        self.llm_calls += 1

        claims_text = []
        for i, item in enumerate(cluster, 1):
            time_str = item['time'][:19] if item['time'] else 'unknown'
            plaus = item.get('plausibility', 0.5)
            claims_text.append(
                f"{i}. {metric_type}={item['value']:.0f} @ {time_str} (plausibility={plaus:.2f})"
            )
            claims_text.append(f"   \"{item['text'][:80]}...\"")

        prompt = f"""Analyze these {metric_type} claims:

{chr(10).join(claims_text)}

Determine the pattern:

1. **PROGRESSION**: Numbers update over time (death toll rising)
2. **CONTRADICTION**: Conflicting reports
3. **CONSENSUS**: Similar numbers = agreement

Return JSON:
{{
  "pattern": "progression|contradiction|consensus",
  "reasoning": "Explain why"
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
            print(f"  ⚠️  Pattern analysis failed: {e}")
            return {'pattern': 'unknown', 'reasoning': str(e)}

    # =========================================================================
    # STEP 5: Generate narrative from high-plausibility claims
    # =========================================================================

    async def generate_narrative_from_topology(
        self,
        event_id: str,
        clusters: Dict[str, dict]
    ) -> str:
        """
        Generate event narrative using high-plausibility claims (winners)

        Uses topology resolution:
        - Consensus claims (high agreements) get priority
        - Progression patterns use latest values
        - Contradicted claims excluded
        """
        # Get high-plausibility claims
        winners = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            WHERE c.plausibility >= 0.7
            RETURN c.id as id, c.text as text, c.plausibility as plausibility
            ORDER BY c.plausibility DESC
            LIMIT 20
        """, {'event_id': event_id})

        print(f"\n📖 Narrative generation using {len(winners)} high-plausibility claims (winners)")

        # Build metrics summary from clusters
        metrics_summary = []
        for metric_type, cluster_data in clusters.items():
            pattern = cluster_data['pattern']
            claims = cluster_data['claims']

            if pattern == 'progression':
                # Use latest value
                latest = max(claims, key=lambda c: c['time'] or '2000')
                metrics_summary.append(f"{metric_type}: {latest['value']:.0f} (latest in progression)")
            elif pattern == 'consensus':
                # Use consensus value
                from collections import Counter
                values = [c['value'] for c in claims]
                consensus = Counter(values).most_common(1)[0][0]
                metrics_summary.append(f"{metric_type}: {consensus:.0f} (consensus)")
            else:
                # Show range for contradictions
                values = [c['value'] for c in claims]
                metrics_summary.append(f"{metric_type}: {min(values):.0f}-{max(values):.0f} (conflicting reports)")

        narrative_parts = [
            "## Verified Metrics (from topology resolution)",
            "\n".join(f"- {m}" for m in metrics_summary),
            "",
            "## High-Confidence Claims (plausibility ≥ 0.70)",
        ]

        for w in winners[:10]:
            narrative_parts.append(f"- [{w['plausibility']:.2f}] {w['text']}")

        return "\n".join(narrative_parts)


async def main():
    print("="*80)
    print("🕸️  CLAIM NETWORK TOPOLOGY & PLAUSIBILITY RESOLUTION")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    analyzer = ClaimTopologyAnalyzer(neo4j, openai_client)

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

    # STEP 1: Extract and store metrics
    print("\n" + "="*80)
    print("STEP 1: Extract and store metrics as ClaimMetric nodes")
    print("="*80)

    for i, claim in enumerate(claims[:15], 1):  # First 15 claims
        print(f"\n[{i}/15] Extracting metrics: {claim['text'][:60]}...")
        metrics = await analyzer.extract_and_store_metrics(claim['id'], claim['text'])
        if metrics:
            print(f"   Metrics: {metrics}")

    # STEP 2: Generate embeddings
    print("\n" + "="*80)
    print("STEP 2: Generate embeddings for similarity")
    print("="*80)

    embeddings = {}
    for i, claim in enumerate(claims[:15], 1):
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=claim['text']
        )
        embeddings[claim['id']] = response.data[0].embedding
        if i % 5 == 0:
            print(f"   Generated {i}/15 embeddings...")

    # STEP 3: Build similarity network
    print("\n" + "="*80)
    print("STEP 3: Build claim similarity network")
    print("="*80)

    network = await analyzer.build_similarity_network(claims[:15], embeddings)

    # STEP 4: Resolve plausibility
    print("\n" + "="*80)
    print("STEP 4: Resolve plausibility from topology")
    print("="*80)

    plausibility = await analyzer.resolve_plausibility(claims[:15], network)

    # Store plausibility scores
    for claim_id, score in plausibility.items():
        await neo4j._execute_write("""
            MATCH (c:Claim {id: $claim_id})
            SET c.plausibility = $score
        """, {'claim_id': claim_id, 'score': score})

    # STEP 5: Analyze metric clusters
    print("\n" + "="*80)
    print("STEP 5: Analyze metric clusters")
    print("="*80)

    clusters = await analyzer.analyze_metric_clusters(event_id)

    # STEP 6: Generate narrative
    print("\n" + "="*80)
    print("STEP 6: Generate narrative from topology winners")
    print("="*80)

    narrative = await analyzer.generate_narrative_from_topology(event_id, clusters)
    print(f"\n{narrative}")

    # Final summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    print(f"\nTotal LLM calls: {analyzer.llm_calls}")
    print(f"Claims processed: {len(claims[:15])}")
    print(f"Network edges: {sum(len(neighbors) for neighbors in network.values()) // 2}")
    print(f"Metric clusters: {len(clusters)}")

    scores = list(plausibility.values())
    print(f"\nPlausibility distribution:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  High (>0.7): {sum(1 for s in scores if s > 0.7)} claims")
    print(f"  Low (<0.3): {sum(1 for s in scores if s < 0.3)} claims")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
