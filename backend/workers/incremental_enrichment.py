"""
Incremental Event Enrichment Framework

Philosophy:
- Events are anchored entities that grow organically
- Each artifact enriches the event incrementally (not full re-computation)
- Facts accumulate with Bayesian priors
- Contradictions resolved as they arrive
- Transparent timeline showing belief evolution

Vision:
- See the "big picture" - how our understanding evolved
- Better truth through incremental refinement
- Sustainable - scales to 1000s of artifacts
"""
import asyncio
import asyncpg
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactBelief:
    """
    Represents our belief about a fact that evolves over time

    NOT a single resolved value, but a timeline with confidence evolution
    """
    def __init__(self, fact_type: str, subtype: str, initial_fact: Dict):
        self.fact_type = fact_type
        self.subtype = subtype

        # Timeline of belief updates
        self.timeline = [{
            "value": initial_fact['value'],
            "confidence": initial_fact.get('confidence', 0.7),
            "source": initial_fact['source_claim_id'],
            "timestamp": initial_fact.get('extracted_at'),
            "update_type": "initial"
        }]

        # Current belief (most recent/confident)
        self.current_value = initial_fact['value']
        self.current_confidence = initial_fact.get('confidence', 0.7)

        # Track contradictions
        self.contradictions = []
        self.resolution_history = []

    def update_with_new_fact(self, new_fact: Dict) -> Dict:
        """
        Update belief with new fact - incremental contradiction resolution

        Returns resolution decision for transparency
        """
        new_value = new_fact['value']
        new_confidence = new_fact.get('confidence', 0.7)

        # CASE 1: Unanimous - same value
        if new_value == self.current_value:
            # Consensus strengthens confidence
            old_confidence = self.current_confidence
            self.current_confidence = min(self.current_confidence * 1.15, 0.98)

            self.timeline.append({
                "value": new_value,
                "confidence": self.current_confidence,
                "source": new_fact['source_claim_id'],
                "timestamp": new_fact.get('extracted_at'),
                "update_type": "consensus"
            })

            resolution = {
                "type": "consensus",
                "action": "confidence_boost",
                "old_confidence": old_confidence,
                "new_confidence": self.current_confidence,
                "rationale": f"Multiple sources agree on {new_value}"
            }

            self.resolution_history.append(resolution)
            return resolution

        # CASE 2: Temporal evolution - increasing pattern
        if isinstance(new_value, (int, float)) and isinstance(self.current_value, (int, float)):
            if new_value > self.current_value:
                # Likely evolution (death toll rising)
                old_value = self.current_value
                self.current_value = new_value
                self.current_confidence = new_confidence

                self.timeline.append({
                    "value": new_value,
                    "confidence": new_confidence,
                    "source": new_fact['source_claim_id'],
                    "timestamp": new_fact.get('extracted_at'),
                    "update_type": "evolution"
                })

                resolution = {
                    "type": "temporal_evolution",
                    "action": "update_value",
                    "old_value": old_value,
                    "new_value": new_value,
                    "rationale": f"Death toll evolved from {old_value} to {new_value}"
                }

                self.resolution_history.append(resolution)
                return resolution

        # CASE 3: Bayesian override - higher confidence
        if new_confidence > self.current_confidence * 1.1:
            old_value = self.current_value
            old_confidence = self.current_confidence

            # Store old belief as potential contradiction
            self.contradictions.append({
                "value": self.current_value,
                "confidence": self.current_confidence,
                "superseded_by": new_fact['source_claim_id'],
                "reason": "higher_confidence_source"
            })

            self.current_value = new_value
            self.current_confidence = new_confidence * 0.9  # Slight penalty for contradiction

            self.timeline.append({
                "value": new_value,
                "confidence": self.current_confidence,
                "source": new_fact['source_claim_id'],
                "timestamp": new_fact.get('extracted_at'),
                "update_type": "override"
            })

            resolution = {
                "type": "bayesian_override",
                "action": "override_value",
                "old_value": old_value,
                "new_value": new_value,
                "old_confidence": old_confidence,
                "new_confidence": self.current_confidence,
                "rationale": f"Higher confidence source ({new_confidence:.2f} > {old_confidence:.2f})"
            }

            self.resolution_history.append(resolution)
            return resolution

        # CASE 4: Unresolved contradiction - flag but keep current belief
        self.contradictions.append({
            "value": new_value,
            "confidence": new_confidence,
            "source": new_fact['source_claim_id'],
            "flagged": True,
            "reason": "contradicts_current_belief"
        })

        self.timeline.append({
            "value": new_value,
            "confidence": new_confidence,
            "source": new_fact['source_claim_id'],
            "timestamp": new_fact.get('extracted_at'),
            "update_type": "contradiction_flagged"
        })

        resolution = {
            "type": "unresolved_contradiction",
            "action": "keep_current",
            "current_value": self.current_value,
            "conflicting_value": new_value,
            "rationale": f"Contradiction: {self.current_value} vs {new_value} - keeping current (higher confidence)"
        }

        self.resolution_history.append(resolution)
        return resolution

    def to_dict(self) -> Dict:
        """Export belief state"""
        return {
            "fact_type": self.fact_type,
            "subtype": self.subtype,
            "current_value": self.current_value,
            "current_confidence": self.current_confidence,
            "timeline": self.timeline,
            "contradictions": self.contradictions,
            "resolution_history": self.resolution_history,
            "update_count": len(self.timeline)
        }


class IncrementalEventEnricher:
    """
    Incrementally enrich events as artifacts arrive

    NOT batch processing - each artifact updates the event state
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def extract_facts_from_artifact(self, artifact: Dict) -> List[Dict]:
        """
        Extract facts from an artifact (page with claims)

        Returns list of facts with metadata
        """
        # Simple fact extraction from claims
        # In production, this would call LLM

        facts = []
        claims = artifact.get('claims', [])

        for claim in claims:
            # Parse if JSON string
            if isinstance(claim, str):
                claim = json.loads(claim)

            # Use our existing LLM extraction
            claim_facts = await self._extract_facts_llm(claim, artifact)
            facts.extend(claim_facts)

        return facts

    async def _extract_facts_llm(self, claim: Dict, artifact: Dict) -> List[Dict]:
        """Extract facts from single claim using LLM"""
        prompt = f"""Extract atomic facts from this claim. Return ONLY valid JSON array.

CLAIM: "{claim['text']}"
SOURCE: {artifact.get('url', 'unknown')}

Extract facts in these categories:
- casualty_count (deaths, injured, hospitalized, missing, evacuated)
- event_time (start, end, milestone)
- location (specific places, buildings, districts)
- cause (what caused it, contributing factors)
- response (firefighting, rescue operations, arrests)

Return JSON array:
[{{"type": "casualty_count", "subtype": "deaths", "value": 44, "confidence": 0.9, "precision": "exact"}}]

Only extract facts explicitly stated. Return [] if no facts."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract structured facts. Return only valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            facts = json.loads(content)

            # Enrich with metadata
            for fact in facts:
                fact['source_claim_id'] = str(claim['id'])
                fact['source_url'] = artifact.get('url')
                fact['artifact_timestamp'] = artifact.get('created_at')
                fact['extracted_at'] = datetime.utcnow().isoformat()

            return facts
        except Exception as e:
            logger.error(f"Fact extraction error: {e}")
            return []

    async def enrich_event_incrementally(
        self,
        event_id: str,
        artifacts: List[Dict]
    ) -> Dict:
        """
        Process artifacts incrementally, building up event state

        Args:
            event_id: Event UUID
            artifacts: List of artifacts (pages with claims) in chronological order

        Returns:
            Event state with fact beliefs and timeline
        """
        # Initialize event state
        event_state = {
            "event_id": event_id,
            "facts": {},  # {fact_type: {subtype: FactBelief}}
            "artifact_count": 0,
            "enrichment_timeline": []
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"INCREMENTAL ENRICHMENT: Processing {len(artifacts)} artifacts")
        logger.info(f"{'='*80}\n")

        for i, artifact in enumerate(artifacts, 1):
            logger.info(f"📄 Artifact {i}/{len(artifacts)}: {artifact.get('url', 'unknown')}")
            logger.info(f"   Created: {artifact.get('created_at')}")
            logger.info(f"   Claims: {len(artifact.get('claims', []))}")

            # Extract facts from this artifact
            new_facts = await self.extract_facts_from_artifact(artifact)
            logger.info(f"   Facts extracted: {len(new_facts)}\n")

            # Update event state with each fact
            updates = []

            for fact in new_facts:
                fact_type = fact['type']
                subtype = fact.get('subtype', 'default')

                # Initialize fact type if new
                if fact_type not in event_state['facts']:
                    event_state['facts'][fact_type] = {}

                # Initialize subtype if new
                if subtype not in event_state['facts'][fact_type]:
                    belief = FactBelief(fact_type, subtype, fact)
                    event_state['facts'][fact_type][subtype] = belief

                    update = {
                        "artifact": i,
                        "fact_type": fact_type,
                        "subtype": subtype,
                        "action": "initialized",
                        "value": fact['value']
                    }
                    logger.info(f"   🆕 NEW: {fact_type}.{subtype} = {fact['value']}")
                else:
                    # Update existing belief
                    belief = event_state['facts'][fact_type][subtype]
                    resolution = belief.update_with_new_fact(fact)

                    update = {
                        "artifact": i,
                        "fact_type": fact_type,
                        "subtype": subtype,
                        **resolution
                    }

                    emoji = {
                        "consensus": "✓",
                        "temporal_evolution": "📈",
                        "bayesian_override": "🔄",
                        "unresolved_contradiction": "⚠️"
                    }.get(resolution['type'], "•")

                    logger.info(f"   {emoji} {resolution['type'].upper()}: {fact_type}.{subtype}")
                    logger.info(f"      {resolution['rationale']}")

                updates.append(update)

            event_state['enrichment_timeline'].append({
                "artifact_index": i,
                "artifact_url": artifact.get('url'),
                "artifact_timestamp": artifact.get('created_at'),
                "facts_extracted": len(new_facts),
                "updates": updates
            })

            event_state['artifact_count'] = i
            logger.info("")

        return event_state

    def generate_big_picture(self, event_state: Dict) -> str:
        """
        Generate human-readable "big picture" summary

        Shows:
        - Final belief state for each fact
        - How beliefs evolved
        - What contradictions were resolved
        - Confidence progression
        """
        output = []
        output.append("\n" + "="*80)
        output.append("BIG PICTURE: Event Understanding")
        output.append("="*80 + "\n")

        output.append(f"Artifacts processed: {event_state['artifact_count']}\n")

        # For each fact type
        for fact_type, subtypes in event_state['facts'].items():
            output.append(f"\n📊 {fact_type.upper().replace('_', ' ')}")
            output.append("-" * 80)

            for subtype, belief in subtypes.items():
                belief_dict = belief.to_dict()

                output.append(f"\n  {subtype}:")
                output.append(f"    Current Value: {belief_dict['current_value']}")
                output.append(f"    Confidence: {belief_dict['current_confidence']:.2%}")
                output.append(f"    Updates: {belief_dict['update_count']}")

                # Evolution timeline
                if len(belief_dict['timeline']) > 1:
                    output.append(f"    Evolution:")
                    for t in belief_dict['timeline']:
                        output.append(f"      • {t['update_type']}: {t['value']} (confidence: {t['confidence']:.2f})")

                # Contradictions resolved
                if belief_dict['contradictions']:
                    output.append(f"    Contradictions: {len(belief_dict['contradictions'])}")
                    for c in belief_dict['contradictions'][:3]:  # Show first 3
                        output.append(f"      ⚠️  {c['value']} ({c['reason']})")

        output.append("\n" + "="*80 + "\n")

        return "\n".join(output)


async def test_incremental_enrichment(event_id: str):
    """Test incremental enrichment on Hong Kong fire event"""
    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    enricher = IncrementalEventEnricher(client)

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

        print(f"\nTesting incremental enrichment: {event['title']}\n")

        # Get artifacts (pages with claims) in chronological order
        artifacts_raw = await conn.fetch("""
            SELECT
                p.id, p.url, p.title, p.created_at,
                ARRAY_AGG(
                    jsonb_build_object(
                        'id', c.id,
                        'text', c.text,
                        'event_time', c.event_time,
                        'confidence', c.confidence
                    )
                ) as claims
            FROM core.pages p
            JOIN core.page_events pe ON p.id = pe.page_id
            LEFT JOIN core.claims c ON c.page_id = p.id
            WHERE pe.event_id = $1
            GROUP BY p.id, p.url, p.title, p.created_at
            ORDER BY p.created_at
        """, event_id)

        artifacts = [dict(a) for a in artifacts_raw]

        # Process incrementally
        event_state = await enricher.enrich_event_incrementally(event_id, artifacts)

        # Generate big picture
        big_picture = enricher.generate_big_picture(event_state)
        print(big_picture)

        # Store in enriched_json (convert datetimes to strings)
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        enriched_json_str = json.dumps({
            "incremental_enrichment": {
                "facts": {
                    fact_type: {
                        subtype: belief.to_dict()
                        for subtype, belief in subtypes.items()
                    }
                    for fact_type, subtypes in event_state['facts'].items()
                },
                "artifact_count": event_state['artifact_count'],
                "enrichment_timeline": event_state['enrichment_timeline'],
                "enriched_at": datetime.utcnow().isoformat()
            }
        }, default=serialize_datetime)

        await conn.execute("""
            UPDATE core.events
            SET enriched_json = $2
            WHERE id = $1
        """, event_id, enriched_json_str)

        print("✅ Stored incremental enrichment in enriched_json\n")

    await db_pool.close()


if __name__ == '__main__':
    import sys
    event_id = sys.argv[1] if len(sys.argv) > 1 else '0c6bc931-94ea-4e14-9fc9-5c5ed3ebeb2a'
    asyncio.run(test_incremental_enrichment(event_id))
