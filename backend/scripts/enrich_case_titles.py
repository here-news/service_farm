#!/usr/bin/env python3
"""
Enrich Case Titles with Semantic Descriptions
==============================================

Uses LLM to generate meaningful titles for Cases based on their claims.

Usage:
    docker exec herenews-app python scripts/enrich_case_titles.py
    docker exec herenews-app python scripts/enrich_case_titles.py --limit 10
"""

import asyncio
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from services.neo4j_service import Neo4jService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_case_title(openai: AsyncOpenAI, claims: list[str], entities: list[str]) -> tuple[str, str]:
    """Generate semantic title and description for a case."""
    if not claims:
        return "Untitled Case", ""

    # Take first 5 claims, truncated
    sample_claims = [c[:300] for c in claims[:5]]
    claims_text = "\n- ".join(sample_claims)
    entities_text = ", ".join(entities[:10]) if entities else "Unknown"

    prompt = f"""Generate a concise news headline title and brief description for a news story based on these claims:

Key entities: {entities_text}

Sample claims:
- {claims_text}

Respond in JSON format:
{{"title": "A concise headline (5-10 words)", "description": "1-2 sentence summary of what this story is about"}}

Important:
- The title should describe the EVENT or STORY, not just list entity names
- Use active voice, present tense for headlines
- Be specific about what happened, not generic
- If claims are about a fire, incident, announcement, etc. - say that
"""

    try:
        response = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a news editor creating headlines. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200,
        )

        import json
        content = response.choices[0].message.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        data = json.loads(content)
        return data.get("title", "Untitled"), data.get("description", "")
    except Exception as e:
        logger.warning(f"LLM failed: {e}")
        return "Untitled Case", ""


async def enrich_cases(limit: int = None):
    """Enrich all Cases with semantic titles."""
    neo4j = Neo4jService()
    await neo4j.connect()

    openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        # Find Cases that need enrichment (no description or title is generic)
        query = '''
            MATCH (c:Case)
            WHERE c.description IS NULL OR c.description = ""
            OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)-[:CONTAINS]->(s:Surface)-[:CONTAINS]->(cl:Claim)
            WITH c, collect(DISTINCT cl.text) as claims
            RETURN c.id as id, c.canonical_title as current_title, c.core_entities as entities, claims
        '''
        if limit:
            query += f' LIMIT {limit}'

        results = await neo4j._execute_read(query)
        logger.info(f"Found {len(results)} cases to enrich")

        enriched = 0
        for row in results:
            case_id = row['id']
            claims = [c for c in (row['claims'] or []) if c]
            entities = row['entities'] or []

            if not claims:
                logger.info(f"Skipping {case_id}: no claims")
                continue

            logger.info(f"Enriching {case_id} with {len(claims)} claims...")
            title, description = await generate_case_title(openai, claims, entities)

            if title and title != "Untitled":
                await neo4j._execute_write('''
                    MATCH (c:Case {id: $id})
                    SET c.title = $title,
                        c.canonical_name = $title,
                        c.description = $description,
                        c.updated_at = datetime()
                ''', {
                    'id': case_id,
                    'title': title,
                    'description': description,
                })
                logger.info(f"  → {title}")
                enriched += 1
            else:
                logger.info(f"  → Skipped (no good title generated)")

        logger.info(f"✅ Enriched {enriched} cases")

    finally:
        await neo4j.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich Case titles with LLM")
    parser.add_argument('--limit', type=int, help='Limit number of cases to process')
    args = parser.parse_args()

    asyncio.run(enrich_cases(limit=args.limit))
