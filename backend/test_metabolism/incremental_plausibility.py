"""
Test incremental plausibility updates for streaming claims

This test simulates claims arriving one-by-one and demonstrates:
1. Storing metrics on claim nodes (materialized)
2. Incremental cluster analysis (only affected clusters)
3. Heuristics for when to trigger full re-analysis
4. Performance comparison: batch vs incremental

Goal: Handle 100+ claims efficiently without O(N²) re-analysis
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


class IncrementalPlausibilityService:
    """Incremental plausibility scoring for streaming claims"""

    def __init__(self, neo4j: Neo4jService, openai_client: AsyncOpenAI):
        self.neo4j = neo4j
        self.openai = openai_client
        self.llm_calls = 0  # Track API usage

    async def extract_and_store_metrics(self, claim_id: str, claim_text: str) -> dict:
        """
        Extract metrics using LLM and store on claim node (materialized)

        Returns: {metric_type: value}
        Example: {'deaths': 156, 'injured': 12}
        """
        self.llm_calls += 1

        prompt = f"""Extract event-related quantitative metrics from this claim. Return ONLY valid JSON.

Claim: "{claim_text}"

Return JSON with metrics object:
{{
  "deaths": 156,
  "injured": 12
}}

Valid metric types: deaths, injured, missing, evacuated, hospitalized, buildings, firefighters, displaced

IMPORTANT:
- Only extract metrics about the MAIN EVENT (fire casualties)
- Ignore unrelated numbers (arrests, individual losses, organizational counts)
- Return empty object if no metrics: {{}}
"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )

            metrics = json.loads(response.choices[0].message.content)

            # Store metrics on claim node
            await self.neo4j._execute_write("""
                MATCH (c:Claim {id: $claim_id})
                SET c.metrics = $metrics,
                    c.metrics_extracted_at = datetime()
            """, {'claim_id': claim_id, 'metrics': metrics})

            return metrics

        except Exception as e:
            print(f"  ⚠️  Metric extraction failed: {e}")
            return {}

    async def get_metric_cluster(
        self,
        event_id: str,
        metric_type: str
    ) -> List[dict]:
        """
        Get all claims in event with given metric type

        Returns: [{claim_id, value, time, plausibility}, ...]
        """
        results = await self.neo4j._execute_read(f"""
            MATCH (e:Event {{id: $event_id}})-[:SUPPORTS]->(c:Claim)
            WHERE c.metrics.{metric_type} IS NOT NULL
            RETURN c.id as claim_id,
                   c.metrics.{metric_type} as value,
                   c.event_time as time,
                   c.plausibility as plausibility,
                   c.text as text
            ORDER BY c.event_time
        """, {'event_id': event_id, 'metric_type': metric_type})

        return [dict(r) for r in results]

    def needs_reanalysis(
        self,
        cluster: List[dict],
        new_value: float
    ) -> bool:
        """
        Heuristic: Does adding new_value require full cluster re-analysis?

        Returns True if:
        - First or second claim (establish baseline)
        - New value is outlier (>50% different from cluster mean)
        - Cluster small (<5 claims) so LLM analysis is cheap
        - Pattern change suspected (value decreases when all increasing)

        Returns False if:
        - Value within existing range (extends progression)
        - Corroborates consensus (matches majority)
        - Cluster large (>10 claims, expensive to re-analyze)
        """
        if not cluster:
            return True  # First claim, no cluster yet

        if len(cluster) <= 2:
            return True  # Need LLM to establish pattern

        values = [c['value'] for c in cluster]
        mean_val = np.mean(values)
        min_val = min(values)
        max_val = max(values)

        # Check if outlier
        if new_value < min_val * 0.5 or new_value > max_val * 2.0:
            print(f"    🔴 Outlier detected: {new_value} vs range [{min_val:.0f}-{max_val:.0f}]")
            return True

        # Check if value within range (no re-analysis needed)
        if min_val <= new_value <= max_val:
            print(f"    🟢 Within range: {new_value} in [{min_val:.0f}-{max_val:.0f}]")
            return False

        # Check if extends progression
        if new_value > max_val:
            # Check if values are increasing (progression pattern)
            is_increasing = all(
                cluster[i]['value'] <= cluster[i+1]['value']
                for i in range(len(cluster)-1)
                if cluster[i]['time'] and cluster[i+1]['time']
            )
            if is_increasing:
                print(f"    🟢 Extends progression: {new_value} > {max_val:.0f}")
                return False

        # Default: re-analyze for safety
        print(f"    🟡 Uncertain pattern, re-analyzing")
        return True

    async def analyze_cluster_llm(
        self,
        event_id: str,
        metric_type: str,
        cluster: List[dict]
    ) -> dict:
        """
        Analyze cluster using LLM (full re-analysis)

        Returns: {
          'pattern': 'progression|contradiction|consensus',
          'plausibility_scores': {claim_id: float},
          'reasoning': str
        }
        """
        self.llm_calls += 1

        if len(cluster) <= 1:
            return {
                'pattern': 'consensus',
                'plausibility_scores': {cluster[0]['claim_id']: 0.8} if cluster else {},
                'reasoning': 'Single claim, no comparison'
            }

        # Format for LLM
        claims_text = []
        for i, item in enumerate(cluster, 1):
            time_str = item['time'][:19] if item['time'] else 'unknown time'
            claims_text.append(
                f"{i}. [{item['claim_id'][:8]}] {metric_type}={item['value']:.0f} @ {time_str}"
            )
            claims_text.append(f"   \"{item['text'][:80]}...\"")

        prompt = f"""Analyze these {metric_type} claims from a developing news event:

{chr(10).join(claims_text)}

Determine the pattern and plausibility of each claim:

1. **PROGRESSION**: Numbers update over time (death toll rising as info arrives)
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
"""

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
                'pattern': 'consensus',
                'plausibility_scores': {c['claim_id']: 0.7 for c in cluster},
                'reasoning': 'Analysis failed, defaulting'
            }

    async def apply_incremental_update(
        self,
        claim_id: str,
        cluster: List[dict],
        new_value: float
    ) -> float:
        """
        Apply incremental Bayesian update WITHOUT full LLM re-analysis

        Heuristics:
        - If value in middle of range → moderate plausibility (0.7)
        - If value matches consensus → high plausibility (0.85)
        - If value is latest in progression → high plausibility (0.9)

        Returns: plausibility score
        """
        values = [c['value'] for c in cluster]
        min_val = min(values)
        max_val = max(values)

        # Check if matches consensus (most common value)
        from collections import Counter
        value_counts = Counter(values)
        consensus_value = value_counts.most_common(1)[0][0]
        consensus_count = value_counts.most_common(1)[0][1]

        if new_value == consensus_value and consensus_count >= 2:
            # Corroborates consensus
            plausibility = 0.85
            print(f"    ✅ Matches consensus: {new_value} ({consensus_count} claims)")
        elif new_value > max_val:
            # Latest in progression
            plausibility = 0.90
            print(f"    ✅ Latest in progression: {new_value} > {max_val:.0f}")
        elif min_val <= new_value <= max_val:
            # Within range
            plausibility = 0.70
            print(f"    ➡️  Within range: {new_value}")
        else:
            # Outlier
            plausibility = 0.30
            print(f"    ⚠️  Outlier: {new_value}")

        return plausibility

    async def update_plausibility_scores(
        self,
        scores: Dict[str, float]
    ):
        """Batch update plausibility scores in Neo4j"""
        for claim_id, score in scores.items():
            await self.neo4j._execute_write("""
                MATCH (c:Claim {id: $claim_id})
                SET c.plausibility = $score,
                    c.plausibility_updated_at = datetime()
            """, {'claim_id': claim_id, 'plausibility': score})

    async def process_claim_incremental(
        self,
        event_id: str,
        claim_id: str,
        claim_text: str
    ) -> Dict[str, float]:
        """
        Process new claim with incremental plausibility update

        Returns: {claim_id: plausibility_score} for all updated claims
        """
        print(f"\n{'='*80}")
        print(f"📥 Processing claim: {claim_id[:12]}")
        print(f"   {claim_text[:80]}...")

        # Step 1: Extract and store metrics
        metrics = await self.extract_and_store_metrics(claim_id, claim_text)
        print(f"   Metrics: {metrics}")

        if not metrics:
            print(f"   ⚠️  No metrics found, skipping plausibility update")
            return {}

        updated_scores = {}

        # Step 2: Process each metric type
        for metric_type, value in metrics.items():
            print(f"\n   📊 Processing {metric_type}={value}")

            # Get existing cluster
            cluster = await self.get_metric_cluster(event_id, metric_type)
            print(f"      Cluster size: {len(cluster)} claims")

            if self.needs_reanalysis(cluster, value):
                # Full LLM re-analysis
                print(f"      🔄 Full cluster re-analysis (LLM call)")
                # Add new claim to cluster for analysis
                cluster.append({
                    'claim_id': claim_id,
                    'value': value,
                    'time': None,  # Will be set by graph
                    'text': claim_text
                })
                analysis = await self.analyze_cluster_llm(event_id, metric_type, cluster)
                print(f"      Pattern: {analysis['pattern']}")
                print(f"      Reasoning: {analysis['reasoning'][:100]}...")

                # Update all claims in cluster
                for cid, score in analysis['plausibility_scores'].items():
                    updated_scores[cid] = score
            else:
                # Incremental update (no LLM call)
                print(f"      ⚡ Incremental update (no LLM)")
                score = await self.apply_incremental_update(claim_id, cluster, value)
                updated_scores[claim_id] = score

        # Step 3: Write scores to graph
        if updated_scores:
            await self.update_plausibility_scores(updated_scores)
            print(f"\n   ✅ Updated {len(updated_scores)} plausibility scores")

        return updated_scores


async def main():
    print("="*80)
    print("🧪 INCREMENTAL PLAUSIBILITY TEST")
    print("="*80)

    neo4j = Neo4jService()
    await neo4j.connect()

    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    service = IncrementalPlausibilityService(neo4j, openai_client)

    # Get test event and claims
    event_id = 'ev_pth3a8dc'  # Wang Fuk Court Fire event

    # Get all claims from event
    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as id, c.text as text, c.event_time as time
        ORDER BY c.event_time
    """, {'event_id': event_id})

    print(f"\n📊 Event: {event_id}")
    print(f"   Total claims: {len(claims)}")

    # Simulate claims arriving one by one
    print("\n" + "="*80)
    print("🔄 SIMULATING INCREMENTAL PROCESSING")
    print("="*80)

    total_updated = 0
    for i, claim in enumerate(claims[:20], 1):  # First 20 claims
        print(f"\n\n{'#'*80}")
        print(f"# CLAIM {i}/{min(20, len(claims))}")
        print(f"{'#'*80}")

        updated_scores = await service.process_claim_incremental(
            event_id=event_id,
            claim_id=claim['id'],
            claim_text=claim['text']
        )

        total_updated += len(updated_scores)

        # Show current state
        print(f"\n   📈 Total claims processed: {i}")
        print(f"   📈 Total scores updated: {total_updated}")
        print(f"   📈 Total LLM calls: {service.llm_calls}")
        print(f"   📈 Avg LLM calls per claim: {service.llm_calls/i:.2f}")

    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)

    n_claims = min(20, len(claims))
    print(f"\nClaims processed: {n_claims}")
    print(f"Total LLM calls: {service.llm_calls}")

    if n_claims > 0:
        print(f"Avg LLM calls per claim: {service.llm_calls/n_claims:.2f}")

        # Compare to batch approach
        batch_llm_calls = n_claims * 7  # Assume 7 metric types
        print(f"\nBatch approach would use: ~{batch_llm_calls} LLM calls")
        print(f"Incremental savings: {(1 - service.llm_calls/batch_llm_calls)*100:.1f}%")
    else:
        print("\n⚠️  No claims found in event")

    # Show final plausibility scores
    final_scores = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        WHERE c.plausibility IS NOT NULL
        RETURN c.id as id, c.text as text, c.plausibility as plausibility,
               c.metrics as metrics
        ORDER BY c.plausibility DESC
        LIMIT 10
    """, {'event_id': event_id})

    print("\n" + "="*80)
    print("🏆 TOP PLAUSIBILITY SCORES")
    print("="*80)

    for claim in final_scores:
        print(f"\n↑ {claim['plausibility']:.3f}")
        print(f"   Metrics: {claim['metrics']}")
        print(f"   {claim['text'][:100]}...")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
