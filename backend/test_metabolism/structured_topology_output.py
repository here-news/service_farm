"""
Output structured claim topology data for narrative generation

Pipeline:
1. Build claim network (semantic similarity)
2. Detect clusters (topic/facet groups)
3. Resolve plausibility (Bayesian + LLM analysis)
4. Output structured JSON with:
   - Claim nodes (text, plausibility, cluster, relationships)
   - Cluster summaries (topic, pattern, consensus)
   - Network statistics (density, agreement/contradiction counts)
5. Use structured data to generate weighted narrative with LLM

Goal: Separate topology analysis from narrative generation
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import json
from datetime import datetime

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class StructuredTopologyGenerator:
    """Generate structured topology data for narrative generation"""

    def __init__(self, neo4j: Neo4jService, openai_client: AsyncOpenAI):
        self.neo4j = neo4j
        self.openai = openai_client
        self.llm_calls = 0

    async def build_topology_data(
        self,
        event_id: str,
        claims: List[dict],
        embeddings: Dict[str, List[float]]
    ) -> dict:
        """
        Build complete topology data structure

        Returns: {
          'event_id': str,
          'total_claims': int,
          'claims': [
            {
              'id': str,
              'text': str,
              'plausibility': float,
              'cluster_id': str,
              'cluster_topic': str,
              'agreements': [claim_ids],
              'contradictions': [claim_ids],
              'similar_claims': [(claim_id, similarity)],
              'timestamp': str,
              'metadata': {...}
            }
          ],
          'clusters': {
            cluster_id: {
              'topic': str,
              'pattern': 'consensus|mixed|contradictory',
              'claim_ids': [str],
              'consensus_points': [str],
              'contradictions': [str],
              'plausibility_range': (float, float),
              'size': int
            }
          },
          'network_stats': {
            'edge_count': int,
            'avg_degree': float,
            'density': float,
            'agreement_ratio': float
          }
        }
        """
        print("\n🏗️  Building structured topology data...")
        print("="*80)

        # Step 1: Build network
        network, agreements, contradictions = await self._build_detailed_network(claims, embeddings)

        # Step 2: Detect clusters
        clusters_map = await self._detect_clusters(claims, network)

        # Step 3: Analyze each cluster
        cluster_analyses = {}
        for cluster_id, claim_ids in clusters_map.items():
            if len(claim_ids) > 1:
                cluster_claims = [c for c in claims if c['id'] in claim_ids]
                analysis = await self._analyze_cluster_detailed(cluster_claims, network)
                cluster_analyses[cluster_id] = analysis

        # Step 4: Resolve plausibility
        plausibility = await self._resolve_plausibility(claims, clusters_map, cluster_analyses)

        # Step 5: Build structured output
        structured_data = {
            'event_id': event_id,
            'total_claims': len(claims),
            'timestamp': datetime.utcnow().isoformat(),
            'claims': [],
            'clusters': {},
            'network_stats': {}
        }

        # Add claim data
        for claim in claims:
            claim_id = claim['id']

            # Find cluster
            cluster_id = None
            cluster_topic = 'unclustered'
            for cid, cids in clusters_map.items():
                if claim_id in cids:
                    cluster_id = cid
                    if cid in cluster_analyses:
                        cluster_topic = cluster_analyses[cid].get('topic', 'unknown')
                    break

            # Get relationships
            similar = [(n_id, sim) for n_id, sim in network.get(claim_id, [])[:5]]
            agrees = agreements.get(claim_id, [])
            contradicts = contradictions.get(claim_id, [])

            structured_data['claims'].append({
                'id': claim_id,
                'text': claim['text'],
                'plausibility': plausibility.get(claim_id, 0.5),
                'cluster_id': cluster_id,
                'cluster_topic': cluster_topic,
                'agreements': agrees,
                'contradictions': contradicts,
                'similar_claims': similar,
                'timestamp': claim.get('event_time'),
                'original_confidence': claim.get('confidence', 0.5),
                'metadata': {
                    'network_degree': len(network.get(claim_id, [])),
                    'agreement_count': len(agrees),
                    'contradiction_count': len(contradicts)
                }
            })

        # Add cluster data
        for cluster_id, analysis in cluster_analyses.items():
            claim_ids = list(clusters_map[cluster_id])
            plaus_scores = [plausibility.get(cid, 0.5) for cid in claim_ids]

            structured_data['clusters'][cluster_id] = {
                'topic': analysis.get('topic', 'unknown'),
                'pattern': analysis.get('pattern', 'mixed'),
                'claim_ids': claim_ids,
                'consensus_points': analysis.get('consensus_points', []),
                'contradictions': analysis.get('contradictions', []),
                'reasoning': analysis.get('reasoning', ''),
                'plausibility_range': (min(plaus_scores), max(plaus_scores)),
                'avg_plausibility': np.mean(plaus_scores),
                'size': len(claim_ids)
            }

        # Network stats
        total_edges = sum(len(neighbors) for neighbors in network.values()) // 2
        total_agreements = sum(len(agrees) for agrees in agreements.values()) // 2
        total_contradictions = sum(len(contradicts) for contradicts in contradictions.values()) // 2

        structured_data['network_stats'] = {
            'edge_count': total_edges,
            'avg_degree': total_edges * 2 / len(claims) if claims else 0,
            'density': total_edges / (len(claims) * (len(claims) - 1) / 2) if len(claims) > 1 else 0,
            'agreement_count': total_agreements,
            'contradiction_count': total_contradictions,
            'agreement_ratio': total_agreements / max(total_agreements + total_contradictions, 1)
        }

        return structured_data

    async def _build_detailed_network(
        self,
        claims: List[dict],
        embeddings: Dict[str, List[float]],
        threshold: float = 0.4
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Build network with agreement/contradiction detection

        Returns: (network, agreements, contradictions)
        """
        network = defaultdict(list)
        agreements = defaultdict(list)
        contradictions = defaultdict(list)

        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                vec1 = np.array(embeddings[claim1['id']])
                vec2 = np.array(embeddings[claim2['id']])

                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    similarity = float(dot / (norm1 * norm2))
                else:
                    continue

                if similarity > threshold:
                    network[claim1['id']].append((claim2['id'], similarity))
                    network[claim2['id']].append((claim1['id'], similarity))

                    # Determine agreement/contradiction
                    if similarity > 0.7:
                        # High similarity = likely agreement
                        agreements[claim1['id']].append(claim2['id'])
                        agreements[claim2['id']].append(claim1['id'])
                    elif 0.4 < similarity < 0.6:
                        # Medium similarity = potential contradiction
                        # (talking about same thing but different details)
                        contradictions[claim1['id']].append(claim2['id'])
                        contradictions[claim2['id']].append(claim1['id'])

        return dict(network), dict(agreements), dict(contradictions)

    async def _detect_clusters(
        self,
        claims: List[dict],
        network: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, Set[str]]:
        """Simple connected components clustering"""
        visited = set()
        clusters = {}
        cluster_id = 0

        for claim in claims:
            if claim['id'] in visited:
                continue

            cluster = set()
            queue = [claim['id']]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.add(current)

                for neighbor_id, sim in network.get(current, []):
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)

            if cluster:
                clusters[f"cluster_{cluster_id}"] = cluster
                cluster_id += 1

        return clusters

    async def _analyze_cluster_detailed(
        self,
        cluster_claims: List[dict],
        network: Dict[str, List[Tuple[str, float]]]
    ) -> dict:
        """
        Detailed cluster analysis with consensus/contradiction detection

        Returns: {
          'topic': str,
          'pattern': str,
          'plausibility_scores': {claim_id: float},
          'consensus_points': [str],  # Points of agreement
          'contradictions': [str],    # Points of disagreement
          'reasoning': str
        }
        """
        self.llm_calls += 1

        claims_text = []
        for i, claim in enumerate(cluster_claims, 1):
            time_str = claim.get('event_time', '')[:19] if claim.get('event_time') else 'unknown'
            claims_text.append(f"{i}. [{claim['id'][:8]}] @ {time_str}")
            claims_text.append(f"   \"{claim['text']}\"")

        prompt = f"""Analyze these {len(cluster_claims)} related claims:

{chr(10).join(claims_text)}

Task:
1. What topic/facet do they discuss?
2. Identify consensus points (facts most claims agree on)
3. Identify contradictions (conflicting statements)
4. Assign plausibility scores (0.0-1.0) based on:
   - Consensus claims: higher scores
   - Contradicted claims: lower scores
   - Complementary claims: moderate-high scores
   - Latest temporal claims: highest if credible

Return JSON:
{{
  "topic": "brief topic description",
  "pattern": "consensus|mixed|contradictory",
  "plausibility_scores": {{
    "claim_id": 0.85
  }},
  "consensus_points": [
    "Most claims agree that...",
    "Confirmed by multiple sources..."
  ],
  "contradictions": [
    "Claim X says A, but Claim Y says B"
  ],
  "reasoning": "Detailed explanation of analysis"
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
            return {
                'topic': 'unknown',
                'pattern': 'mixed',
                'plausibility_scores': {c['id']: 0.6 for c in cluster_claims},
                'consensus_points': [],
                'contradictions': [],
                'reasoning': f'Analysis failed: {e}'
            }

    async def _resolve_plausibility(
        self,
        claims: List[dict],
        clusters: Dict[str, Set[str]],
        cluster_analyses: Dict[str, dict]
    ) -> Dict[str, float]:
        """Resolve plausibility from cluster analyses"""
        plausibility = {c['id']: c.get('confidence', 0.6) for c in claims}

        for cluster_id, analysis in cluster_analyses.items():
            for claim_id, score in analysis.get('plausibility_scores', {}).items():
                old = plausibility.get(claim_id, 0.6)
                plausibility[claim_id] = 0.3 * old + 0.7 * score

        return plausibility

    async def generate_weighted_narrative(self, topology_data: dict) -> str:
        """
        Generate narrative from structured topology data using LLM

        Uses topology structure to create weighted, evidence-based narrative
        """
        self.llm_calls += 1

        # Prepare high-plausibility claims by cluster
        clusters_text = []

        for cluster_id, cluster_info in topology_data['clusters'].items():
            topic = cluster_info['topic']
            pattern = cluster_info['pattern']
            avg_plaus = cluster_info['avg_plausibility']

            # Get claims for this cluster
            cluster_claims = [
                c for c in topology_data['claims']
                if c['cluster_id'] == cluster_id and c['plausibility'] >= 0.6
            ]
            cluster_claims.sort(key=lambda x: x['plausibility'], reverse=True)

            if not cluster_claims:
                continue

            claims_list = []
            for claim in cluster_claims[:5]:  # Top 5 per cluster
                claims_list.append(
                    f"  - [{claim['plausibility']:.2f}] {claim['text']}"
                )

            consensus = "\n".join(f"  • {cp}" for cp in cluster_info.get('consensus_points', [])[:3])
            contradictions = "\n".join(f"  • {cd}" for cd in cluster_info.get('contradictions', [])[:2])

            cluster_block = f"""
## {topic.upper()}
Pattern: {pattern} | Avg Plausibility: {avg_plaus:.2f} | {cluster_info['size']} claims

Consensus Points:
{consensus if consensus else "  (none identified)"}

Contradictions:
{contradictions if contradictions else "  (none identified)"}

High-Plausibility Claims:
{chr(10).join(claims_list)}
"""
            clusters_text.append(cluster_block)

        # Network stats summary
        stats = topology_data['network_stats']
        stats_text = f"""
Event has {topology_data['total_claims']} claims forming a network with:
- {stats['edge_count']} semantic connections (density: {stats['density']:.2f})
- {stats['agreement_count']} agreements vs {stats['contradiction_count']} contradictions
- Agreement ratio: {stats['agreement_ratio']:.2f}
"""

        prompt = f"""Generate a comprehensive, factual event narrative using this structured claim topology data:

{stats_text}

CLAIM CLUSTERS BY TOPIC:
{chr(10).join(clusters_text)}

Instructions:
1. Synthesize information from HIGH-PLAUSIBILITY claims (≥0.60)
2. Weight claims by plausibility score - higher scores = more prominence
3. Use consensus points to establish facts
4. Acknowledge contradictions where they exist (show range: "36-156 deaths")
5. Organize by topic clusters (casualties, timeline, response, etc.)
6. Use structured format (markdown headers, bullet points)
7. NO speculation, NO editorial language
8. Show uncertainty when claims contradict

Generate a structured narrative that reflects the topology:"""

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
    print("📊 STRUCTURED TOPOLOGY → WEIGHTED NARRATIVE")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    generator = StructuredTopologyGenerator(neo4j, openai_client)

    # Get event and claims
    event_id = 'ev_pth3a8dc'

    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time
        ORDER BY c.event_time
        LIMIT 20
    """, {'event_id': event_id})

    claims = [dict(c) for c in claims]

    # Generate embeddings
    print("\n📊 Generating embeddings...")
    embeddings = {}
    for claim in claims:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=claim['text']
        )
        embeddings[claim['id']] = response.data[0].embedding

    # Build structured topology
    print("\n" + "="*80)
    print("STEP 1: Build structured topology data")
    print("="*80)

    topology_data = await generator.build_topology_data(event_id, claims, embeddings)

    # Print structured data
    print("\n📄 STRUCTURED TOPOLOGY DATA:")
    print(json.dumps(topology_data, indent=2, default=str))

    # Save to file
    output_file = f"/tmp/topology_{event_id}.json"
    with open(output_file, 'w') as f:
        json.dump(topology_data, f, indent=2, default=str)
    print(f"\n💾 Saved to: {output_file}")

    # Generate weighted narrative
    print("\n" + "="*80)
    print("STEP 2: Generate weighted narrative from topology")
    print("="*80)

    narrative = await generator.generate_weighted_narrative(topology_data)

    print("\n📖 WEIGHTED NARRATIVE:")
    print("="*80)
    print(narrative)
    print("="*80)

    # Summary
    print("\n📊 SUMMARY:")
    print(f"Total LLM calls: {generator.llm_calls}")
    print(f"Claims processed: {len(claims)}")
    print(f"Clusters detected: {len(topology_data['clusters'])}")
    print(f"Network density: {topology_data['network_stats']['density']:.2f}")
    print(f"Agreement ratio: {topology_data['network_stats']['agreement_ratio']:.2f}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
