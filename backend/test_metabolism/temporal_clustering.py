"""
Test temporal-aware metric clustering for Bayesian plausibility

This test demonstrates:
1. Clustering claims by metric type (deaths, missing, evacuated)
2. LLM analysis of each cluster to detect: progression, contradiction, or consensus
3. Temporal-aware plausibility scoring
4. Handling of developing news situations
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


class TemporalMetricAnalyzer:
    """Temporal-aware metric clustering and analysis"""

    def __init__(self, neo4j: Neo4jService, openai_client: AsyncOpenAI):
        self.neo4j = neo4j
        self.openai = openai_client
        self.cache = {}

    async def extract_metrics(self, claim: dict) -> List[Tuple[str, float]]:
        """Extract all quantitative metrics from claim using LLM"""
        cache_key = f"metrics_{hash(claim['text'])}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""Extract event-related quantitative metrics from this claim. Return ONLY valid JSON.

Claim: "{claim['text']}"

Return JSON with metrics array:
{{
  "metrics": [
    {{"type": "deaths", "value": 156}},
    {{"type": "injured", "value": 12}}
  ]
}}

Valid types: deaths, injured, missing, evacuated, hospitalized, buildings, firefighters, displaced

IMPORTANT:
- Only extract metrics about the MAIN EVENT (fire casualties)
- Ignore unrelated numbers (arrests, individual family losses, organizational counts)
- "3 arrested" is NOT deaths
- "60 church members" is NOT main event casualties
- "1 firefighter died" is deaths IF it's about the fire

If no metrics found, return: {{"metrics": []}}"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )

            data = json.loads(response.choices[0].message.content)
            metrics = [(m['type'], float(m['value'])) for m in data.get('metrics', [])]
            self.cache[cache_key] = metrics
            return metrics

        except Exception as e:
            print(f"  ⚠️  Metric extraction failed: {e}")
            return []

    async def cluster_by_metric(self, claims: List[dict]) -> Dict[str, List[dict]]:
        """
        Cluster claims by metric type

        Returns: {
          'deaths': [claim_with_death_count, ...],
          'missing': [claim_with_missing_count, ...],
          ...
        }
        """
        print("\n📊 Clustering claims by metric type...")
        print("="*80)

        clusters = defaultdict(list)

        for claim in claims:
            metrics = await self.extract_metrics(claim)
            for metric_type, value in metrics:
                clusters[metric_type].append({
                    'claim_id': claim['id'],
                    'text': claim['text'],
                    'value': value,
                    'time': claim.get('event_time'),
                    'confidence': claim.get('confidence', 0.5)
                })

        # Show clusters
        for metric_type, cluster in clusters.items():
            print(f"\n📈 {metric_type.upper()}: {len(cluster)} claims")
            for item in sorted(cluster, key=lambda x: x['time'] or '2000'):
                time_str = item['time'][:19] if item['time'] else 'unknown'
                print(f"   {item['value']:>6.0f} @ {time_str} | {item['text'][:60]}...")

        return dict(clusters)

    async def analyze_metric_cluster(self, metric_type: str, cluster: List[dict]) -> dict:
        """
        Analyze a cluster of claims about the same metric using LLM

        Returns: {
          'pattern': 'progression' | 'contradiction' | 'consensus',
          'plausibility_scores': {claim_id: float},
          'reasoning': str
        }
        """
        if len(cluster) <= 1:
            return {
                'pattern': 'consensus',
                'plausibility_scores': {cluster[0]['claim_id']: 1.0} if cluster else {},
                'reasoning': 'Single claim, no comparison needed'
            }

        # Sort by time
        sorted_cluster = sorted(cluster, key=lambda c: c['time'] or '2000-01-01')

        # Format for LLM
        claims_text = []
        for i, item in enumerate(sorted_cluster, 1):
            time_str = item['time'][:19] if item['time'] else 'unknown time'
            claims_text.append(f"{i}. [{item['claim_id'][:8]}] {metric_type}={item['value']:.0f} @ {time_str}")
            claims_text.append(f"   \"{item['text'][:100]}...\"")

        prompt = f"""Analyze these {metric_type} claims from a developing news event:

{chr(10).join(claims_text)}

Determine the pattern and plausibility of each claim:

1. **PROGRESSION**: Numbers update over time (death toll rising as more info arrives)
   - Most recent/final number is most plausible
   - Earlier numbers are updates, not wrong

2. **CONTRADICTION**: Conflicting reports at same time or numbers decrease
   - Penalize outliers
   - Trust consensus/majority

3. **CONSENSUS**: Similar/same numbers across claims
   - All claims corroborate each other
   - Boost all plausibility

Return JSON:
{{
  "pattern": "progression|contradiction|consensus",
  "plausibility_scores": {{
    "claim_id": 0.9,
    "claim_id": 0.7
  }},
  "reasoning": "Explain the pattern and scoring"
}}

Guidelines:
- Progression: latest gets 0.9-1.0, earlier gets 0.6-0.8
- Contradiction: consensus gets 0.8-0.9, outliers get 0.1-0.3
- Consensus: all get 0.85-0.95
- If timestamps unknown, assume simultaneous → check for contradiction"""

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
            # Fallback: treat as consensus
            return {
                'pattern': 'consensus',
                'plausibility_scores': {c['claim_id']: 0.7 for c in cluster},
                'reasoning': 'Analysis failed, defaulting to moderate confidence'
            }

    async def compute_temporal_plausibility(
        self,
        claims: List[dict],
        clusters: Dict[str, List[dict]]
    ) -> Dict[str, float]:
        """
        Compute plausibility scores using temporal cluster analysis

        Returns: {claim_id: plausibility_score}
        """
        print("\n🧮 Analyzing metric clusters with temporal awareness...")
        print("="*80)

        # Initialize with priors
        plausibility = {c['id']: c.get('confidence', 0.5) for c in claims}

        # Analyze each metric cluster
        for metric_type, cluster in clusters.items():
            if not cluster:
                continue

            print(f"\n📊 Analyzing {metric_type.upper()} cluster ({len(cluster)} claims)...")

            analysis = await self.analyze_metric_cluster(metric_type, cluster)

            print(f"   Pattern: {analysis['pattern']}")
            print(f"   Reasoning: {analysis['reasoning']}")
            print(f"   Scores:")

            for claim_id, score in analysis['plausibility_scores'].items():
                if claim_id in plausibility:
                    old = plausibility[claim_id]
                    # Combine prior with cluster-based score
                    plausibility[claim_id] = 0.3 * old + 0.7 * score
                    print(f"      {claim_id[:8]}: {old:.2f} → {plausibility[claim_id]:.2f}")

        return plausibility


async def main():
    print("="*80)
    print("🧪 TEMPORAL METRIC CLUSTERING TEST")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    analyzer = TemporalMetricAnalyzer(neo4j, openai_client)

    # Load claims from test pages
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

        for claim in claims:
            all_claims.append(claim)

    print(f"   Total claims: {len(all_claims)}")

    # Cluster by metric type
    clusters = await analyzer.cluster_by_metric(all_claims)

    # Compute temporal-aware plausibility
    plausibility = await analyzer.compute_temporal_plausibility(all_claims, clusters)

    # Show results
    print("\n" + "="*80)
    print("📈 FINAL TEMPORAL-AWARE PLAUSIBILITY SCORES")
    print("="*80)

    # Sort by plausibility
    sorted_claims = sorted(
        all_claims,
        key=lambda c: plausibility.get(c['id'], 0.5),
        reverse=True
    )

    print("\n🏆 HIGH PLAUSIBILITY (reliable metrics):")
    for claim in sorted_claims[:8]:
        score = plausibility.get(claim['id'], 0.5)
        print(f"\n{'↑' if score > 0.7 else '→'} {score:.3f}")
        print(f"   {claim['text']}")

    print("\n" + "-"*80)
    print("\n⚠️  LOW PLAUSIBILITY (unreliable/outdated):")
    for claim in sorted_claims[-5:]:
        score = plausibility.get(claim['id'], 0.5)
        print(f"\n↓ {score:.3f}")
        print(f"   {claim['text']}")

    # Statistics
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    scores = list(plausibility.values())
    print(f"Mean plausibility: {np.mean(scores):.3f}")
    print(f"Median plausibility: {np.median(scores):.3f}")
    print(f"Std dev: {np.std(scores):.3f}")

    print(f"\nMetric clusters:")
    for metric_type, cluster in clusters.items():
        values = [c['value'] for c in cluster]
        print(f"  {metric_type}: {len(cluster)} claims, range: {min(values):.0f}-{max(values):.0f}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
