"""
Simple test: Can LLM synthesize useful descriptions from claim clusters?

Test on existing keyword-grouped micro events (casualties, fire outbreak, etc.)
"""
import asyncio
import asyncpg
import os
import json
from openai import AsyncOpenAI


async def synthesize_micro_event(event_id: str, event_title: str):
    """Synthesize description for a micro event from its claims"""

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
        # Get claims for this micro event (using keyword filtering from API)
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
            ORDER BY c.event_time NULLS LAST
        """, event_id)

        print(f"\n{'='*80}")
        print(f"🔬 Micro Event: {event_title}")
        print(f"   Claims: {len(claims)}")
        print(f"{'='*80}\n")

        if len(claims) < 2:
            print("⚠️  Too few claims to synthesize")
            await db_pool.close()
            return

        # Prepare claims for LLM
        claims_text = ""
        for i, claim in enumerate(claims, 1):
            entities_str = ", ".join(claim['entities']) if claim['entities'] else "none"
            time_str = claim['event_time'].isoformat() if claim['event_time'] else "unknown"
            claims_text += f"{i}. [{time_str}] {claim['text']}\n   Entities: {entities_str}\n   Confidence: {claim['confidence']}\n\n"

        prompt = f"""You are analyzing {len(claims)} claims about "{event_title}".

CLAIMS:
{claims_text}

Your task:
1. Synthesize a corroborated description of what these claims collectively tell us
2. Resolve any contradictions (e.g., if death tolls differ, track the evolution: 4→36→44)
3. Extract the 5W+H: WHO, WHEN, WHERE, WHAT (HOW), WHY
4. Give a confidence score (0.0-1.0) for how coherent these claims are

Return ONLY a JSON object:
{{
  "description": "2-3 sentence corroborated narrative synthesizing all claims",
  "who": ["Key participants/entities"],
  "when": {{"start": "ISO timestamp or null", "end": "ISO timestamp or null", "precision": "exact|approximate|unknown"}},
  "where": ["Locations"],
  "what": "What happened (1-2 sentences)",
  "why": "Causal factors or reasons (if mentioned, otherwise null)",
  "contradictions": ["List any contradictions with resolution"],
  "confidence": 0.8
}}

Be factual. If information is missing, use null.
"""

        print("📡 Calling LLM for synthesis...\n")

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a factual news synthesis assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            synthesis = json.loads(content)

            print(f"✨ SYNTHESIS:")
            print(f"   Confidence: {synthesis.get('confidence', 'N/A')}\n")
            print(f"   Description:")
            print(f"   {synthesis.get('description', 'N/A')}\n")

            if synthesis.get('what'):
                print(f"   What: {synthesis['what']}")
            if synthesis.get('who'):
                print(f"   Who: {', '.join(synthesis['who'][:5])}")
            if synthesis.get('where'):
                print(f"   Where: {', '.join(synthesis['where'])}")
            if synthesis.get('when'):
                when = synthesis['when']
                print(f"   When: {when.get('start', 'unknown')} (precision: {when.get('precision', 'unknown')})")
            if synthesis.get('why'):
                print(f"   Why: {synthesis['why']}")
            if synthesis.get('contradictions'):
                print(f"\n   ⚠️  Contradictions: {len(synthesis['contradictions'])}")
                for c in synthesis['contradictions']:
                    print(f"      - {c}")

        except Exception as e:
            print(f"❌ Synthesis error: {e}")

    await db_pool.close()


async def test_all_micro_events():
    """Test synthesis on all micro events from Hong Kong fire"""

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
        # Get micro events
        micro_events = await conn.fetch("""
            SELECT e.id, e.title
            FROM core.events e
            JOIN core.event_relationships r ON e.id = r.event_id
            WHERE r.related_event_id = '0c6bc931-94ea-4e14-9fc9-5c5ed3ebeb2a'
              AND r.relationship_type = 'PART_OF'
              AND e.event_scale = 'micro'
            ORDER BY e.title
        """)

        print(f"Found {len(micro_events)} micro events to test\n")

    await db_pool.close()

    for event in micro_events:
        await synthesize_micro_event(str(event['id']), event['title'])


if __name__ == '__main__':
    asyncio.run(test_all_micro_events())
