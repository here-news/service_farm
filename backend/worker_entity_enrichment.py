#!/usr/bin/env python3
"""
Entity Enrichment Worker

Builds progressive entity profiles from mentions across articles.
Triggered after semantic extraction completes.
"""
import asyncio
import json
from datetime import datetime
from openai import AsyncOpenAI
import asyncpg
import redis.asyncio as redis

openai_client = AsyncOpenAI()

class EntityEnrichmentWorker:
    def __init__(self, pool, redis_client):
        self.pool = pool
        self.redis = redis_client

    async def process(self, job: dict):
        """
        Enrich entities mentioned in a page

        Job format: {"page_id": "uuid"}
        """
        page_id = job['page_id']

        print(f"[EntityEnrichment] Processing page {page_id}")

        async with self.pool.acquire() as conn:
            # Get entities extracted for this page
            entities = await conn.fetch("""
                SELECT DISTINCT e.id, e.canonical_name, e.entity_type, e.language
                FROM entities e
                JOIN page_entities pe ON e.id = pe.entity_id
                WHERE pe.page_id = $1
            """, page_id)

            print(f"[EntityEnrichment] Found {len(entities)} entities to enrich")

            # Enrich each entity
            for entity in entities:
                await self.enrich_entity(
                    conn,
                    str(entity['id']),
                    entity['canonical_name'],
                    entity['entity_type'],
                    entity['language']
                )

            print(f"[EntityEnrichment] ✓ Completed page {page_id}")

    async def enrich_entity(self, conn, entity_id: str, name: str, entity_type: str, language: str):
        """
        Build/update profile for an entity based on all mentions
        """
        print(f"[EntityEnrichment] Enriching {name} ({entity_type})")

        # Get all pages mentioning this entity
        mentions = await conn.fetch("""
            SELECT p.title, c.text as claim, c.event_time
            FROM pages p
            JOIN page_entities pe ON p.id = pe.page_id
            JOIN entities e ON pe.entity_id = e.id
            LEFT JOIN claims c ON c.page_id = p.id
            WHERE e.id = $1 AND p.status IN ('entities_extracted', 'complete')
            ORDER BY p.created_at DESC
            LIMIT 20
        """, entity_id)

        if not mentions:
            print(f"[EntityEnrichment] No mentions found for {name}")
            return

        mention_count = len(set(m['title'] for m in mentions if m['title']))

        # Get current profile
        current = await conn.fetchrow("""
            SELECT profile_summary, profile_roles, profile_affiliations,
                   profile_key_facts, profile_locations
            FROM entities WHERE id = $1
        """, entity_id)

        # Build mention context
        mention_texts = []
        for m in mentions:
            if m['claim']:
                mention_texts.append({
                    "article": m['title'] or "Untitled",
                    "claim": m['claim']
                })

        # Only enrich if we have actual claims
        if not mention_texts:
            print(f"[EntityEnrichment] No claims found for {name}, skipping LLM enrichment")
            await conn.execute("""
                UPDATE entities SET mention_count = $1, updated_at = NOW()
                WHERE id = $2
            """, mention_count, entity_id)
            return

        # Use LLM to build/update profile
        profile = await self.synthesize_profile(
            name, entity_type, mention_texts[:10],  # Limit to 10 mentions for LLM
            current_profile={
                "summary": current['profile_summary'],
                "roles": current['profile_roles'],
                "affiliations": current['profile_affiliations'],
                "key_facts": current['profile_key_facts'],
                "locations": current['profile_locations']
            } if current else None
        )

        # Update entity with enriched profile
        await conn.execute("""
            UPDATE entities SET
                profile_summary = $1,
                profile_roles = $2,
                profile_affiliations = $3,
                profile_key_facts = $4,
                profile_locations = $5,
                mention_count = $6,
                last_enriched_at = NOW(),
                updated_at = NOW()
            WHERE id = $7
        """,
            profile.get('summary'),
            json.dumps(profile.get('roles', [])),
            json.dumps(profile.get('affiliations', [])),
            json.dumps(profile.get('key_facts', [])),
            json.dumps(profile.get('locations', [])),
            mention_count,
            entity_id
        )

        print(f"[EntityEnrichment] ✓ Enriched {name} - {mention_count} mentions")

    async def synthesize_profile(self, name: str, entity_type: str, mentions: list, current_profile: dict = None) -> dict:
        """
        Use LLM to synthesize entity profile from mentions
        """
        mention_list = "\n".join([
            f"- Article: {m['article']}\n  Claim: {m['claim']}"
            for m in mentions
        ])

        current_summary = "No existing profile." if not current_profile or not current_profile.get('summary') else f"Current profile: {current_profile['summary']}"

        prompt = f"""Build an entity profile for {name} ({entity_type}) based on mentions across articles.

{current_summary}

New mentions from {len(mentions)} articles:
{mention_list}

Create/update a structured profile. Return JSON:
{{
  "summary": "Brief description of who/what this entity is (2-3 sentences)",
  "roles": ["Role 1", "Role 2"],  // For PERSON: job titles, positions
  "affiliations": ["Org 1", "Org 2"],  // Organizations, groups associated with
  "key_facts": ["Fact 1", "Fact 2"],  // Important facts from the mentions
  "locations": ["Location 1"]  // For PERSON/ORG: where they operate. For GPE/LOC: related places
}}

Guidelines:
- For PERSON: roles, affiliations, key actions/events they're in
- For ORGANIZATION: type, key people, actions/statements
- For GPE/LOC: type (city/country), events there, related entities
- Accumulate new facts, don't repeat existing info
- Be factual, extract from mentions only
- Keep arrays concise (max 5 items each)
"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an entity profiling system. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"[EntityEnrichment] LLM error for {name}: {e}")
            return {
                "summary": f"{name} ({entity_type})",
                "roles": [],
                "affiliations": [],
                "key_facts": [],
                "locations": []
            }


async def main():
    """Main worker loop"""
    pool = await asyncpg.create_pool(
        host='demo-postgres',
        port=5432,
        user='demo_user',
        password='demo_pass',
        database='demo_phi_here',
        min_size=1,
        max_size=3
    )

    redis_client = await redis.from_url('redis://demo-redis:6379')

    worker = EntityEnrichmentWorker(pool, redis_client)

    print("[EntityEnrichment] Worker started, waiting for jobs...")

    while True:
        try:
            # Wait for jobs from queue
            job_data = await redis_client.brpop('queue:entity_enrichment:normal', timeout=5)

            if job_data:
                job = json.loads(job_data[1])
                await worker.process(job)

        except Exception as e:
            print(f"[EntityEnrichment] Error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
