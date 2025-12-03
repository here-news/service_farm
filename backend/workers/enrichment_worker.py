"""
Enrichment Worker - Semantic clustering and LLM synthesis

Micro-way approach:
1. Take an event's claims
2. Cluster them semantically (embedding-based)
3. Use LLM to synthesize corroborated descriptions for each cluster
4. Store synthesis in enriched_json

Start small, evaluate usefulness before scaling.
"""
import os
import asyncio
import asyncpg
import numpy as np
from typing import List, Dict, Set
from openai import AsyncOpenAI
import json
from datetime import datetime
from sklearn.cluster import DBSCAN
import logging

from services.job_queue import JobQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnrichmentWorker:
    """Enrich events with synthesized micro-event descriptions"""

    def __init__(self, db_pool: asyncpg.Pool, job_queue: JobQueue, worker_id: int = 1):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def start(self):
        """Start processing enrichment jobs"""
        logger.info(f"🎨 enrichment-worker-{self.worker_id} started")

        while True:
            try:
                # Poll for enrichment jobs
                job = await self.job_queue.dequeue('enrichment_queue')

                if job:
                    event_id = job['event_id']
                    await self.enrich_event(event_id)
                else:
                    await asyncio.sleep(5)  # No jobs, wait

            except Exception as e:
                logger.error(f"❌ Enrichment worker error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def enrich_event(self, event_id: str):
        """
        Enrich event with synthesized micro-narratives

        Steps:
        1. Fetch event claims with embeddings
        2. Cluster claims semantically
        3. For each cluster, synthesize description via LLM
        4. Store synthesis in event's enriched_json
        """
        async with self.db_pool.acquire() as conn:
            logger.info(f"🎨 Enriching event {event_id}")

            # Get event
            event = await conn.fetchrow("""
                SELECT id, title, event_scale, enriched_json
                FROM core.events
                WHERE id = $1
            """, event_id)

            if not event:
                logger.warning(f"Event {event_id} not found")
                return

            # Get claims with embeddings
            claims = await conn.fetch("""
                SELECT
                    c.id, c.text, c.event_time, c.confidence, c.modality,
                    c.embedding,
                    ARRAY_AGG(DISTINCT e.canonical_name) FILTER (WHERE e.canonical_name IS NOT NULL) as entities
                FROM core.claims c
                JOIN core.pages p ON c.page_id = p.id
                JOIN core.page_events pe ON p.id = pe.page_id
                LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
                LEFT JOIN core.entities e ON ce.entity_id = e.id
                WHERE pe.event_id = $1
                  AND c.embedding IS NOT NULL
                GROUP BY c.id, c.text, c.event_time, c.confidence, c.modality, c.embedding
            """, event_id)

            if len(claims) < 3:
                logger.info(f"Event has only {len(claims)} claims, skipping enrichment")
                return

            logger.info(f"📊 Event '{event['title'][:50]}...' has {len(claims)} claims")

            # Cluster claims semantically
            clusters = await self._cluster_claims(claims)

            logger.info(f"📂 Found {len(clusters)} semantic clusters")

            # Synthesize description for each cluster
            syntheses = []
            for i, cluster_claims in enumerate(clusters):
                if len(cluster_claims) < 2:
                    continue  # Skip single-claim clusters

                logger.info(f"  🔬 Cluster {i+1}: {len(cluster_claims)} claims")

                synthesis = await self._synthesize_cluster(cluster_claims)
                if synthesis:
                    syntheses.append({
                        'cluster_id': i,
                        'claim_count': len(cluster_claims),
                        'claim_ids': [str(c['id']) for c in cluster_claims],
                        **synthesis
                    })

            # Synthesize canonical event title
            canonical_title = await self._synthesize_event_title(conn, event_id, claims)

            # Store syntheses in enriched_json
            enriched = json.loads(event['enriched_json']) if event['enriched_json'] else {}
            enriched['micro_narratives'] = syntheses
            enriched['enrichment_timestamp'] = datetime.utcnow().isoformat()
            if canonical_title:
                enriched['canonical_title'] = canonical_title

            # Update event with canonical title and enriched_json
            if canonical_title:
                await conn.execute("""
                    UPDATE core.events
                    SET title = $2, enriched_json = $3
                    WHERE id = $1
                """, event_id, canonical_title, json.dumps(enriched))
                logger.info(f"✅ Updated title to: {canonical_title}")
            else:
                await conn.execute("""
                    UPDATE core.events
                    SET enriched_json = $2
                    WHERE id = $1
                """, event_id, json.dumps(enriched))

            logger.info(f"✅ Enriched event with {len(syntheses)} micro-narratives")

    async def _synthesize_event_title(self, conn: asyncpg.Connection, event_id: str, claims: List[Dict]) -> str:
        """
        Synthesize a canonical event title from all pages and claims

        Format: "YYYY Location Event Type" (e.g., "2025 Hong Kong Tai Po Residential Fire")
        Not: article headlines like "Death toll rises as blaze engulfs high-rise"
        """
        # Get all page titles for this event
        pages = await conn.fetch("""
            SELECT p.title, p.url
            FROM core.pages p
            JOIN core.page_events pe ON p.id = pe.page_id
            WHERE pe.event_id = $1
            LIMIT 15
        """, event_id)

        if not pages:
            return None

        # Get key entities (locations, orgs, people)
        entities = await conn.fetch("""
            SELECT e.canonical_name, e.entity_type, COUNT(*) as mentions
            FROM core.entities e
            JOIN core.event_entities ee ON e.id = ee.entity_id
            WHERE ee.event_id = $1
            GROUP BY e.canonical_name, e.entity_type
            ORDER BY mentions DESC
            LIMIT 10
        """, event_id)

        # Prepare context for LLM
        page_titles = "\n".join([f"- {p['title']}" for p in pages])

        entity_list = []
        for e in entities:
            entity_list.append(f"- {e['canonical_name']} ({e['entity_type']}, {e['mentions']} mentions)")
        entities_text = "\n".join(entity_list)

        # Get sample claims for context
        sample_claims = "\n".join([f"- {c['text'][:100]}" for c in claims[:10]])

        # Get event date
        event_date = claims[0]['event_time'] if claims and claims[0].get('event_time') else None
        date_str = event_date.strftime("%Y") if event_date else "Recent"

        prompt = f"""Synthesize a canonical event title from multiple news articles about the same incident.

ARTICLE HEADLINES ({len(pages)} sources):
{page_titles}

KEY ENTITIES:
{entities_text}

SAMPLE CLAIMS:
{sample_claims}

Create a canonical event name following this format:
"{date_str} [Location] [Event Type]"

Examples of GOOD canonical titles:
- "2025 Hong Kong Tai Po Residential Fire"
- "2024 Baltimore Bridge Collapse"
- "2023 Turkey-Syria Earthquake"
- "2025 Los Angeles Wildfire"

Examples of BAD titles (these are article headlines, NOT event names):
- "Death toll rises as blaze engulfs high-rise"
- "Survivors asking how it was allowed to happen"
- "Fire kills dozens in apartment complex"

Requirements:
1. Start with year if known
2. Include primary location (city/district level, not just country)
3. End with event type (Fire, Earthquake, Shooting, Explosion, etc.)
4. Keep it under 8 words
5. Use proper nouns, not generic terms
6. Make it timeless (not "breaking" or "toll rises")

Return ONLY the canonical title, nothing else."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at naming historical events. Return only the canonical event title."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=50
            )

            title = response.choices[0].message.content.strip()

            # Remove quotes if present
            title = title.strip('"').strip("'")

            # Validate length
            if len(title) > 100:
                title = title[:100]

            logger.info(f"📝 Synthesized title: '{title}'")
            return title

        except Exception as e:
            logger.error(f"❌ Title synthesis error: {e}")
            return None

    async def _cluster_claims(self, claims: List[Dict]) -> List[List[Dict]]:
        """
        Cluster claims semantically using DBSCAN on embeddings

        Returns list of clusters, where each cluster is a list of claim dicts
        """
        # Extract embeddings
        embeddings = []
        claim_list = []

        for claim in claims:
            embedding = claim['embedding']

            # Parse embedding (could be pgvector string or list)
            if isinstance(embedding, str):
                if embedding.startswith('[') and embedding.endswith(']'):
                    embedding = [float(x.strip()) for x in embedding[1:-1].split(',')]
                else:
                    continue

            if embedding and len(embedding) > 0:
                embeddings.append(embedding)
                claim_list.append(dict(claim))

        if len(embeddings) < 3:
            return [claim_list]  # Return all claims as one cluster

        # Convert to numpy array
        X = np.array(embeddings)

        # DBSCAN clustering
        # eps: maximum distance between two samples to be in same cluster
        # min_samples: minimum cluster size
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(X)

        # Group claims by cluster label
        clusters_dict = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue  # Skip noise points

            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(claim_list[idx])

        clusters = list(clusters_dict.values())

        # Add noise points as individual clusters if needed
        # (or we could skip them)

        return clusters

    async def _synthesize_cluster(self, claims: List[Dict]) -> Dict:
        """
        Use LLM to synthesize a corroborated description from claim cluster

        Returns:
        {
            'title': 'Short title for this aspect',
            'description': 'Corroborated narrative',
            'who': [...],
            'when': {...},
            'where': [...],
            'what': 'What happened',
            'why': 'Causal factors (if mentioned)',
            'contradictions': [...],
            'confidence': 0.0-1.0
        }
        """
        # Prepare claims for LLM
        claims_text = ""
        for i, claim in enumerate(claims, 1):
            entities_str = ", ".join(claim['entities']) if claim['entities'] else "none"
            time_str = claim['event_time'].isoformat() if claim['event_time'] else "unknown"
            claims_text += f"{i}. [{time_str}] {claim['text']}\n   Entities: {entities_str}\n   Confidence: {claim['confidence']}\n\n"

        prompt = f"""You are analyzing a cluster of {len(claims)} related claims about the same aspect of an event.

CLAIMS:
{claims_text}

Your task:
1. Synthesize a corroborated description of what these claims collectively tell us
2. Resolve any contradictions (e.g., if death tolls differ, track the evolution: 4→36→44)
3. Extract the 5W+H: WHO, WHEN, WHERE, WHAT (HOW), WHY
4. Identify what topic/aspect these claims represent

Return ONLY a JSON object with this structure:
{{
  "title": "Short title (5-10 words) for this aspect/topic",
  "description": "2-3 sentence corroborated narrative synthesizing all claims",
  "who": ["List of key participants/entities mentioned"],
  "when": {{"start": "ISO timestamp or null", "end": "ISO timestamp or null", "precision": "exact|approximate|unknown"}},
  "where": ["List of locations"],
  "what": "What happened (1-2 sentences)",
  "why": "Causal factors or reasons (if mentioned, otherwise null)",
  "contradictions": ["List any contradictions found, with resolution if possible"],
  "confidence": 0.8
}}

Be factual. If information is missing, use null. Focus on corroboration across multiple claims.
"""

        try:
            response = await self.openai_client.chat.completions.create(
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
            return synthesis

        except Exception as e:
            logger.error(f"❌ LLM synthesis error: {e}")
            return None


async def main():
    """Main worker entry point"""
    worker_id = int(os.getenv('WORKER_ID', '1'))

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    worker = EnrichmentWorker(db_pool, job_queue, worker_id=worker_id)
    logger.info(f"🎨 Starting Enrichment worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info(f"🛑 Enrichment worker {worker_id} stopped")
    finally:
        await db_pool.close()
        await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
