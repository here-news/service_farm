"""
Fact Extraction and Resolution Engine

General approach (not domain-specific):
1. Extract atomic facts from claims via LLM
2. Group facts by type
3. Resolve contradictions using Bayesian logic + temporal analysis
4. Score claim coherence against resolved facts
5. Build ever-increasing precision as more claims arrive

Philosophy:
- Claims stay immutable (belong to pages)
- Facts are extracted/resolved computations
- Events build on coherent fact subsets
- Lower layers stabilize, upper layers inherit confidence
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI
import asyncpg
import math
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactResolutionEngine:
    """Extract facts from claims and resolve contradictions"""

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def extract_facts_from_claim(self, claim: Dict) -> List[Dict]:
        """
        Extract structured atomic facts from a claim using LLM

        Returns list of facts:
        {
            "type": "casualty_count" | "event_time" | "location" | "cause" | ...,
            "subtype": "deaths" | "injured" | "start" | "end" | ...,
            "value": actual value (number, string, timestamp),
            "confidence": 0.0-1.0,
            "precision": "exact" | "approximate" | "unknown",
            "source_claim_id": claim_id,
            "source_claim_text": claim text (for debugging)
        }
        """
        prompt = f"""Extract atomic facts from this claim. Return ONLY valid JSON array.

CLAIM: "{claim['text']}"
TIME: {claim.get('event_time', 'unknown')}
ENTITIES: {', '.join(claim.get('entities', [])) if claim.get('entities') else 'none'}

Extract facts in these categories:
- casualty_count (deaths, injured, hospitalized, missing, evacuated)
- event_time (start, end, milestone)
- location (specific places, buildings, districts)
- cause (what caused it, contributing factors)
- response (firefighting, rescue operations, arrests)
- impact (road closures, evacuations, damage)

Return JSON array:
[
  {{
    "type": "casualty_count",
    "subtype": "deaths",
    "value": 44,
    "confidence": 0.9,
    "precision": "exact"
  }},
  {{
    "type": "event_time",
    "subtype": "start",
    "value": "2025-11-26T14:51:00+00:00",
    "confidence": 0.95,
    "precision": "exact"
  }}
]

Only extract facts explicitly stated. If unsure, skip. Return [] if no facts."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract structured facts from news claims. Return only valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            facts = json.loads(content)

            # Enrich with metadata
            for fact in facts:
                fact['source_claim_id'] = str(claim['id'])
                fact['source_claim_text'] = claim['text'][:100]
                fact['extracted_at'] = datetime.utcnow().isoformat()

            return facts

        except Exception as e:
            logger.error(f"Fact extraction error: {e}")
            return []

    async def extract_facts_from_all_claims(self, claims: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Extract facts from all claims and group by type

        Returns: {
            "casualty_count": [{fact1}, {fact2}, ...],
            "event_time": [...],
            ...
        }
        """
        all_facts = defaultdict(list)

        logger.info(f"Extracting facts from {len(claims)} claims...")

        for i, claim in enumerate(claims, 1):
            facts = await self.extract_facts_from_claim(claim)
            logger.info(f"  Claim {i}/{len(claims)}: {len(facts)} facts extracted")

            for fact in facts:
                all_facts[fact['type']].append(fact)

        logger.info(f"Total fact types: {len(all_facts)}")
        for fact_type, facts in all_facts.items():
            logger.info(f"  {fact_type}: {len(facts)} facts")

        return dict(all_facts)

    def resolve_facts(self, facts_by_type: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Resolve contradictions for each fact type

        Returns: {
            "casualty_count": {
                "deaths": {
                    "resolved_value": 44,
                    "confidence": 0.92,
                    "resolution_type": "temporal_evolution" | "unanimous" | "bayesian_highest",
                    "evolution_path": "4→36→44" (if applicable),
                    "supporting_facts": [fact1, fact2, ...],
                    "contradictions": [{...}] (if any unresolved)
                },
                "injured": {...}
            },
            "event_time": {
                "start": {...}
            },
            ...
        }
        """
        resolved = {}

        for fact_type, facts in facts_by_type.items():
            # Group by subtype
            by_subtype = defaultdict(list)
            for fact in facts:
                by_subtype[fact.get('subtype', 'default')].append(fact)

            resolved[fact_type] = {}

            for subtype, subtype_facts in by_subtype.items():
                resolved[fact_type][subtype] = self._resolve_fact_group(
                    fact_type, subtype, subtype_facts
                )

        return resolved

    def _resolve_fact_group(self, fact_type: str, subtype: str, facts: List[Dict]) -> Dict:
        """Resolve a group of facts of same type/subtype"""

        if not facts:
            return None

        # Single fact - no resolution needed
        if len(facts) == 1:
            fact = facts[0]
            return {
                "resolved_value": fact['value'],
                "confidence": fact.get('confidence', 0.8),
                "resolution_type": "single_source",
                "supporting_facts": [fact],
                "contradictions": []
            }

        # Multiple facts - need resolution
        if fact_type == "casualty_count":
            return self._resolve_casualty_count(subtype, facts)
        elif fact_type == "event_time":
            return self._resolve_event_time(subtype, facts)
        elif fact_type == "location":
            return self._resolve_location(subtype, facts)
        else:
            # Generic resolution: pick most confident
            return self._resolve_generic(facts)

    def _resolve_casualty_count(self, subtype: str, facts: List[Dict]) -> Dict:
        """
        Resolve casualty counts - handle temporal evolution

        E.g., deaths: 4→36→44 over time
        """
        # Sort by claim timestamp if available
        sorted_facts = sorted(facts, key=lambda f: self._extract_timestamp(f), reverse=False)

        values = [f['value'] for f in sorted_facts]

        # Check if monotonically increasing (evolution pattern)
        is_evolution = all(values[i] <= values[i+1] for i in range(len(values)-1))

        if is_evolution and len(set(values)) > 1:
            # Temporal evolution: pick latest
            latest = sorted_facts[-1]
            evolution_path = '→'.join(map(str, sorted(set(values))))

            return {
                "resolved_value": latest['value'],
                "confidence": min(latest.get('confidence', 0.8) * 1.1, 0.98),  # Boost for evolution
                "resolution_type": "temporal_evolution",
                "evolution_path": evolution_path,
                "supporting_facts": sorted_facts,
                "contradictions": []
            }

        # Check if unanimous
        if len(set(values)) == 1:
            avg_confidence = sum(f.get('confidence', 0.8) for f in facts) / len(facts)
            return {
                "resolved_value": values[0],
                "confidence": min(avg_confidence * 1.2, 0.98),  # Boost for consensus
                "resolution_type": "unanimous",
                "supporting_facts": facts,
                "contradictions": []
            }

        # Contradictory - use Bayesian: highest confidence
        best_fact = max(facts, key=lambda f: f.get('confidence', 0.5))
        contradictions = [f for f in facts if f['value'] != best_fact['value']]

        return {
            "resolved_value": best_fact['value'],
            "confidence": best_fact.get('confidence', 0.7) * 0.8,  # Penalty for contradiction
            "resolution_type": "bayesian_highest_confidence",
            "supporting_facts": [best_fact],
            "contradictions": contradictions
        }

    def _resolve_event_time(self, subtype: str, facts: List[Dict]) -> Dict:
        """Resolve event timestamps - pick most precise"""
        # Sort by precision: exact > approximate > unknown
        precision_order = {"exact": 3, "approximate": 2, "unknown": 1}
        sorted_facts = sorted(
            facts,
            key=lambda f: (precision_order.get(f.get('precision', 'unknown'), 0), f.get('confidence', 0.5)),
            reverse=True
        )

        best = sorted_facts[0]

        # Check consensus
        values = [f['value'] for f in facts]
        if len(set(values)) == 1:
            resolution_type = "unanimous"
            confidence = min(best.get('confidence', 0.8) * 1.2, 0.98)
        else:
            resolution_type = "highest_precision"
            confidence = best.get('confidence', 0.8)

        return {
            "resolved_value": best['value'],
            "confidence": confidence,
            "resolution_type": resolution_type,
            "precision": best.get('precision', 'unknown'),
            "supporting_facts": sorted_facts if resolution_type == "unanimous" else [best],
            "contradictions": [] if resolution_type == "unanimous" else [f for f in facts if f['value'] != best['value']]
        }

    def _resolve_location(self, subtype: str, facts: List[Dict]) -> Dict:
        """Resolve locations - build hierarchical consensus"""
        # Locations can be hierarchical: "Wang Fuk Court" ⊂ "Tai Po" ⊂ "Hong Kong"
        # Just pick most specific for now

        sorted_facts = sorted(facts, key=lambda f: (len(f['value']), f.get('confidence', 0.5)), reverse=True)
        best = sorted_facts[0]

        return {
            "resolved_value": best['value'],
            "confidence": best.get('confidence', 0.8),
            "resolution_type": "most_specific",
            "supporting_facts": facts,
            "contradictions": []
        }

    def _resolve_generic(self, facts: List[Dict]) -> Dict:
        """Generic resolution: pick highest confidence"""
        best = max(facts, key=lambda f: f.get('confidence', 0.5))

        values = [f['value'] for f in facts]
        if len(set(values)) == 1:
            return {
                "resolved_value": best['value'],
                "confidence": min(best.get('confidence', 0.8) * 1.2, 0.98),
                "resolution_type": "unanimous",
                "supporting_facts": facts,
                "contradictions": []
            }

        return {
            "resolved_value": best['value'],
            "confidence": best.get('confidence', 0.7),
            "resolution_type": "highest_confidence",
            "supporting_facts": [best],
            "contradictions": [f for f in facts if f['value'] != best['value']]
        }

    def _extract_timestamp(self, fact: Dict) -> datetime:
        """Extract timestamp from fact for sorting"""
        # Try to get claim time from source
        # For now, just return a default
        return datetime.min

    def compute_claim_coherence(
        self,
        claim: Dict,
        claim_facts: List[Dict],
        resolved_facts: Dict[str, Dict]
    ) -> float:
        """
        Compute how well this claim supports the resolved facts

        Returns coherence score 0.0-1.0
        """
        if not claim_facts:
            return 0.5  # Neutral if no facts extracted

        coherence_scores = []

        for fact in claim_facts:
            fact_type = fact['type']
            subtype = fact.get('subtype', 'default')

            # Get resolved fact for this type/subtype
            resolved = resolved_facts.get(fact_type, {}).get(subtype)

            if not resolved:
                # Fact type not resolved (rare/novel info)
                coherence_scores.append(0.6)
                continue

            # Check if fact supports resolution
            if fact['value'] == resolved['resolved_value']:
                # Direct support
                coherence_scores.append(0.95)
            elif resolved['resolution_type'] == 'temporal_evolution':
                # Check if part of evolution path
                if str(fact['value']) in resolved.get('evolution_path', ''):
                    coherence_scores.append(0.85)  # Part of evolution
                else:
                    coherence_scores.append(0.3)  # Contradicts evolution
            else:
                # Contradiction
                coherence_scores.append(0.2)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5


async def test_fact_resolution(event_id: str):
    """Test fact resolution on an event"""
    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    engine = FactResolutionEngine(client)

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1,
        max_size=2
    )

    async with db_pool.acquire() as conn:
        # Get event
        event = await conn.fetchrow("""
            SELECT id, title FROM core.events WHERE id = $1
        """, event_id)

        print(f"\n{'='*80}")
        print(f"FACT RESOLUTION TEST: {event['title']}")
        print(f"{'='*80}\n")

        # Get claims
        claims = await conn.fetch("""
            SELECT
                c.id, c.text, c.event_time, c.confidence,
                ARRAY_AGG(DISTINCT e.canonical_name) FILTER (WHERE e.canonical_name IS NOT NULL) as entities
            FROM core.claims c
            JOIN core.pages p ON c.page_id = p.id
            JOIN core.page_events pe ON p.id = pe.page_id
            LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
            LEFT JOIN core.entities e ON ce.entity_id = e.id
            WHERE pe.event_id = $1
            GROUP BY c.id, c.text, c.event_time, c.confidence
        """, event_id)

        claims_list = [dict(c) for c in claims]
        print(f"Processing {len(claims_list)} claims...\n")

        # Extract facts
        facts_by_type = await engine.extract_facts_from_all_claims(claims_list)

        print(f"\n{'='*80}")
        print("RESOLUTION RESULTS")
        print(f"{'='*80}\n")

        # Resolve facts
        resolved_facts = engine.resolve_facts(facts_by_type)

        # Display resolutions
        for fact_type, subtypes in resolved_facts.items():
            print(f"📊 {fact_type.upper().replace('_', ' ')}")
            for subtype, resolution in subtypes.items():
                if not resolution:
                    continue

                print(f"  └─ {subtype}:")
                print(f"     Value: {resolution['resolved_value']}")
                print(f"     Confidence: {resolution['confidence']:.2f}")
                print(f"     Resolution: {resolution['resolution_type']}")

                if resolution.get('evolution_path'):
                    print(f"     Evolution: {resolution['evolution_path']}")

                if resolution.get('precision'):
                    print(f"     Precision: {resolution['precision']}")

                if resolution.get('contradictions'):
                    print(f"     ⚠️  Contradictions: {len(resolution['contradictions'])}")

                print()

        # Compute claim coherence scores
        print(f"{'='*80}")
        print("CLAIM COHERENCE SCORES")
        print(f"{'='*80}\n")

        # Map claim_id to extracted facts
        claim_facts_map = defaultdict(list)
        for fact_type, facts in facts_by_type.items():
            for fact in facts:
                claim_facts_map[fact['source_claim_id']].append(fact)

        coherence_scores = {}
        for claim in claims_list:
            claim_facts = claim_facts_map.get(str(claim['id']), [])
            coherence = engine.compute_claim_coherence(claim, claim_facts, resolved_facts)
            coherence_scores[str(claim['id'])] = coherence

            print(f"Claim: {claim['text'][:80]}...")
            print(f"  Coherence: {coherence:.2f}")
            print(f"  Facts extracted: {len(claim_facts)}")
            print()

        # Overall event coherence
        avg_coherence = sum(coherence_scores.values()) / len(coherence_scores) if coherence_scores else 0.5
        print(f"{'='*80}")
        print(f"OVERALL EVENT COHERENCE: {avg_coherence:.2f}")
        print(f"{'='*80}\n")

        # Store in enriched_json
        enriched_json = {
            "resolved_facts": resolved_facts,
            "claim_coherence_scores": coherence_scores,
            "overall_coherence": avg_coherence,
            "resolution_timestamp": datetime.utcnow().isoformat()
        }

        await conn.execute("""
            UPDATE core.events
            SET enriched_json = $2,
                coherence = $3
            WHERE id = $1
        """, event_id, json.dumps(enriched_json), avg_coherence)

        print("✅ Stored resolved facts in enriched_json")

    await db_pool.close()


if __name__ == '__main__':
    import sys
    event_id = sys.argv[1] if len(sys.argv) > 1 else '0c6bc931-94ea-4e14-9fc9-5c5ed3ebeb2a'
    asyncio.run(test_fact_resolution(event_id))
