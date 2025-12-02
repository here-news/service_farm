"""
Generate dynamic micro-narratives for an event

Instead of creating separate micro event records:
1. Group claims by topic (simple keyword approach for now)
2. Synthesize corroborated description for each group via LLM
3. Store in parent event's enriched_json.micro_narratives[]

Micro-way test: See if this is more useful than raw claims
"""
import asyncio
import asyncpg
import os
import json
from openai import AsyncOpenAI
from typing import List, Dict
from collections import defaultdict


async def generate_micro_narratives(event_id: str):
    """Generate and store micro-narratives for an event"""

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            SELECT id, title, event_scale
            FROM core.events
            WHERE id = $1
        """, event_id)

        if not event:
            print(f"❌ Event {event_id} not found")
            await db_pool.close()
            return

        print(f"🎨 Generating micro-narratives for: {event['title']}")

        # Get all claims
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

        print(f"   Total claims: {len(claims)}")

        if len(claims) < 5:
            print("   ⚠️  Too few claims for meaningful narratives")
            await db_pool.close()
            return

        # Group claims by topic (simple keyword approach)
        groups = group_claims_by_topic(claims)

        print(f"   Identified {len(groups)} topic groups\n")

        # Synthesize each group
        micro_narratives = []

        for topic, group_claims in groups.items():
            if len(group_claims) < 2:
                continue

            print(f"   📖 Synthesizing: {topic} ({len(group_claims)} claims)")

            synthesis = await synthesize_group(client, topic, group_claims)

            if synthesis:
                micro_narratives.append({
                    'topic': topic,
                    'claim_count': len(group_claims),
                    'claim_ids': [str(c['id']) for c in group_claims],
                    **synthesis
                })

        # Store in enriched_json
        enriched = await conn.fetchval("""
            SELECT enriched_json FROM core.events WHERE id = $1
        """, event_id)

        enriched_dict = json.loads(enriched) if enriched else {}
        enriched_dict['micro_narratives'] = micro_narratives
        enriched_dict['micro_narratives_generated_at'] = asyncpg.pgproto.pgproto.encode_timestamp(
            asyncio.get_event_loop().time()
        ) if hasattr(asyncpg.pgproto.pgproto, 'encode_timestamp') else 'now'

        await conn.execute("""
            UPDATE core.events
            SET enriched_json = $2
            WHERE id = $1
        """, event_id, json.dumps(enriched_dict))

        print(f"\n✅ Stored {len(micro_narratives)} micro-narratives in enriched_json")

        # Display results
        print(f"\n{'='*80}")
        print(f"GENERATED MICRO-NARRATIVES")
        print(f"{'='*80}\n")

        for narrative in micro_narratives:
            print(f"📖 {narrative['topic'].upper()}")
            print(f"   Claims: {narrative['claim_count']}")
            print(f"   Confidence: {narrative.get('confidence', 'N/A')}")
            print(f"\n   {narrative.get('description', 'N/A')}\n")

            if narrative.get('contradictions'):
                print(f"   ⚠️  Contradictions:")
                for c in narrative['contradictions']:
                    print(f"      - {c}")
                print()

            print(f"{'-'*80}\n")

    await db_pool.close()


def group_claims_by_topic(claims: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group claims by topic using keyword matching

    Topics: casualties, fire_outbreak, rescue_operations, evacuations, investigation
    """
    groups = defaultdict(list)

    keywords_map = {
        'casualties': ['died', 'death', 'killed', 'firefighter died', 'missing', 'injured', 'hospitalized', 'victims'],
        'fire_outbreak': ['fire broke out', 'started', 'p.m.', 'alarm', 'upgraded', 'floors', 'scaffolding', 'renovation', 'blaze'],
        'rescue_operations': ['firefighters', 'temperature', 'difficult', 'grappling', 'intense heat', 'fire authorities', 'battling', 'rescue', 'operations'],
        'evacuations': ['evacuated', 'temporary housing', 'road closed', 'buses diverted', 'temporary shelters'],
        'investigation': ['arrested', 'manslaughter', 'police', 'investigation', 'determine the cause']
    }

    for claim in claims:
        text_lower = claim['text'].lower()
        matched = False

        for topic, keywords in keywords_map.items():
            if any(kw.lower() in text_lower for kw in keywords):
                groups[topic].append(dict(claim))
                matched = True
                break

        # If no match, add to general group
        if not matched:
            groups['general'].append(dict(claim))

    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


async def synthesize_group(client: AsyncOpenAI, topic: str, claims: List[Dict]) -> Dict:
    """Synthesize corroborated description for a claim group"""

    # Prepare claims text
    claims_text = ""
    for i, claim in enumerate(claims, 1):
        entities_str = ", ".join(claim['entities']) if claim['entities'] else "none"
        time_str = claim['event_time'].isoformat() if claim['event_time'] else "unknown"
        claims_text += f"{i}. [{time_str}] {claim['text']}\n   Entities: {entities_str}\n\n"

    prompt = f"""You are analyzing {len(claims)} claims about the topic: "{topic}".

CLAIMS:
{claims_text}

Your task:
1. Synthesize a corroborated 2-3 sentence description focusing ONLY on this specific topic
2. Resolve contradictions (e.g., death toll evolution: 4→36→44)
3. Extract relevant 5W+H for this specific aspect
4. Confidence score based on claim coherence

Return ONLY valid JSON:
{{
  "description": "2-3 sentence focused description of this specific aspect",
  "who": ["Key participants for this aspect"],
  "when": {{"start": "ISO or null", "end": "ISO or null", "precision": "exact|approximate|unknown"}},
  "where": ["Locations"],
  "what": "What happened regarding {topic}",
  "why": "Causal factors if mentioned, else null",
  "contradictions": ["List contradictions with resolution"],
  "confidence": 0.8
}}

Focus on this specific aspect, not the overall event.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a factual news synthesis assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        return json.loads(content)

    except Exception as e:
        print(f"   ❌ Synthesis error: {e}")
        return None


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        event_id = sys.argv[1]
    else:
        # Default to Hong Kong fire event
        event_id = '0c6bc931-94ea-4e14-9fc9-5c5ed3ebeb2a'

    asyncio.run(generate_micro_narratives(event_id))
