"""
Test Event Network Architecture with Clean Slate

Process Hong Kong fire pages chronologically to demonstrate:
- event(n) + page → event(n+1) incremental enrichment
- Event network (not hierarchy) via relationships
- Holistic enrichment (no claims, direct page processing)
"""
import asyncio
import asyncpg
import os
import json
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from workers.holistic_enrichment import HolisticEventEnricher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventNetworkBuilder:
    """Build event network from pages using holistic enrichment"""

    def __init__(self, db_pool, openai_client):
        self.db_pool = db_pool
        self.enricher = HolisticEventEnricher(openai_client)

        # Thresholds
        self.attach_threshold = 0.65
        self.relate_threshold = 0.45

    async def find_similar_events(self, page_embedding, limit: int = 5) -> List[Dict]:
        """Find similar events by embedding similarity"""
        async with self.db_pool.acquire() as conn:
            # Convert embedding to list if it's a memoryview/bytes
            if isinstance(page_embedding, (memoryview, bytes)):
                import struct
                # pgvector stores as float32 array
                embedding_list = list(struct.unpack(f'{len(page_embedding)//4}f', bytes(page_embedding)))
            elif isinstance(page_embedding, str):
                # Parse string representation
                embedding_list = [float(x) for x in page_embedding.strip('[]').split(',')]
            else:
                embedding_list = list(page_embedding)

            # Convert to pgvector format
            embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

            candidates = await conn.fetch("""
                SELECT
                    e.id, e.title, e.coherence,
                    e.embedding <=> $1::vector as similarity
                FROM core.events e
                WHERE e.embedding IS NOT NULL
                ORDER BY e.embedding <=> $1::vector
                LIMIT $2
            """, embedding_str, limit)

            return [dict(c) for c in candidates]

    async def create_event_from_page(self, page: Dict) -> str:
        """Create new event from page using holistic enrichment"""
        async with self.db_pool.acquire() as conn:
            # Create event record
            event_id = await conn.fetchval("""
                INSERT INTO core.events (
                    title, summary, event_scale,
                    embedding, coherence
                )
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """,
                page['title'],
                page.get('description', '')[:500],
                'meso',  # Default scale
                page.get('embedding'),
                0.5  # Initial coherence
            )

            # Link page to event
            await conn.execute("""
                INSERT INTO core.page_events (page_id, event_id)
                VALUES ($1, $2)
            """, page['id'], event_id)

            logger.info(f"   🆕 Created event: {event_id}")
            logger.info(f"      Title: {page['title']}")

            # Initial enrichment with this page
            event_state = {
                "event_id": str(event_id),
                "ontology": {},
                "artifact_count": 0,
                "enrichment_timeline": []
            }

            result = await self.enricher.enrich_event_with_page(event_state, page)

            if 'error' not in result:
                # Store enriched ontology
                await self._store_event_state(conn, event_id, event_state)

            return str(event_id)

    async def enrich_existing_event(self, event_id: str, page: Dict):
        """Enrich existing event with new page"""
        async with self.db_pool.acquire() as conn:
            # Get current event state
            event = await conn.fetchrow("""
                SELECT id, title, enriched_json
                FROM core.events
                WHERE id = $1
            """, event_id)

            # Load existing ontology
            if event['enriched_json']:
                enriched = json.loads(event['enriched_json']) if isinstance(event['enriched_json'], str) else event['enriched_json']
                event_state = enriched.get('holistic_enrichment', {})

                # Ensure all required keys exist
                event_state.setdefault('event_id', str(event_id))
                event_state.setdefault('ontology', {})
                event_state.setdefault('artifact_count', 0)
                event_state.setdefault('enrichment_timeline', [])

                # Deserialize FactBeliefs from dictionaries
                event_state['ontology'] = self._deserialize_ontology(event_state['ontology'])
            else:
                event_state = {
                    "event_id": str(event_id),
                    "ontology": {},
                    "artifact_count": 0,
                    "enrichment_timeline": []
                }

            # Enrich with new page
            result = await self.enricher.enrich_event_with_page(event_state, page)

            if 'error' not in result:
                # Link page to event
                await conn.execute("""
                    INSERT INTO core.page_events (page_id, event_id)
                    VALUES ($1, $2)
                    ON CONFLICT (page_id, event_id) DO NOTHING
                """, page['id'], event_id)

                # Store updated state
                await self._store_event_state(conn, event_id, event_state)

    def _deserialize_ontology(self, ontology: Dict) -> Dict:
        """Deserialize FactBelief dictionaries back to FactBelief objects"""
        from workers.holistic_enrichment import FactBelief

        deserialized = {}

        for aspect_name, aspect_data in ontology.items():
            if aspect_name in ('story', 'narrative'):
                # These are plain dicts, not FactBeliefs
                deserialized[aspect_name] = aspect_data
            elif isinstance(aspect_data, dict):
                deserialized[aspect_name] = {}
                for key, val in aspect_data.items():
                    if isinstance(val, dict) and 'fact_type' in val and 'timeline' in val:
                        # This is a serialized FactBelief - reconstruct it
                        belief = FactBelief(
                            val['fact_type'],
                            val['subtype'],
                            val['current_value'],
                            val['current_confidence'],
                            'deserialized'
                        )
                        belief.timeline = val['timeline']
                        belief.contradictions = val.get('contradictions', [])
                        belief.resolution_history = val.get('resolution_history', [])
                        deserialized[aspect_name][key] = belief
                    else:
                        # Regular value (like milestones list)
                        deserialized[aspect_name][key] = val
            else:
                deserialized[aspect_name] = aspect_data

        return deserialized

    async def create_relationship(self, event_a_id: str, event_b_id: str,
                                  rel_type: str, confidence: float):
        """Create relationship between two events"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.event_relationships
                (event_id, related_event_id, relationship_type, confidence)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (event_id, related_event_id, relationship_type) DO NOTHING
            """, event_a_id, event_b_id, rel_type, confidence)

    async def _store_event_state(self, conn, event_id: str, event_state: Dict):
        """Store event ontology and update coherence"""
        from workers.holistic_enrichment import FactBelief
        from datetime import datetime

        # Serialize FactBeliefs
        def serialize_for_json(obj):
            if isinstance(obj, FactBelief):
                return obj.to_dict()
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        # Compute coherence
        coherence = self._compute_coherence(event_state)

        enriched_json = json.dumps({
            "holistic_enrichment": {
                "ontology": {
                    aspect_name: {
                        key: serialize_for_json(val)
                        for key, val in aspect_data.items()
                    } if isinstance(aspect_data, dict) else aspect_data
                    for aspect_name, aspect_data in event_state['ontology'].items()
                },
                "coherence": coherence,
                "artifact_count": event_state['artifact_count'],
                "enrichment_timeline": event_state['enrichment_timeline'],
                "enriched_at": datetime.utcnow().isoformat()
            }
        }, default=str)

        # Extract timeline bounds
        event_start = None
        event_end = None
        if 'timeline' in event_state['ontology']:
            timeline = event_state['ontology']['timeline']
            if 'start' in timeline:
                from workers.holistic_enrichment import FactBelief
                if isinstance(timeline['start'], FactBelief):
                    start_value = timeline['start'].current_value
                    # Parse if string
                    if isinstance(start_value, str):
                        from dateutil.parser import parse
                        event_start = parse(start_value)
                    else:
                        event_start = start_value
            if 'milestones' in timeline and timeline['milestones']:
                # Get latest milestone time
                milestones = timeline['milestones']
                if milestones:
                    end_value = milestones[-1].get('time')
                    if end_value and isinstance(end_value, str):
                        from dateutil.parser import parse
                        event_end = parse(end_value)
                    else:
                        event_end = end_value

        await conn.execute("""
            UPDATE core.events
            SET enriched_json = $2,
                coherence = $3,
                event_start = $4,
                event_end = $5,
                updated_at = NOW()
            WHERE id = $1
        """, event_id, enriched_json, coherence, event_start, event_end)

    def _compute_coherence(self, event_state: Dict) -> float:
        """Compute coherence from event state"""
        from workers.holistic_enrichment import FactBelief

        ontology = event_state.get('ontology', {})
        if not ontology:
            return 0.0

        confidences = []
        multi_source_count = 0
        total_facts = 0

        for aspect_name, aspect_data in ontology.items():
            if aspect_name in ('story', 'narrative'):
                continue

            if isinstance(aspect_data, dict):
                for key, belief in aspect_data.items():
                    if isinstance(belief, FactBelief):
                        confidences.append(belief.current_confidence)
                        total_facts += 1
                        if len(belief.timeline) > 1:
                            multi_source_count += 1

        if not confidences:
            return 0.5

        avg_confidence = sum(confidences) / len(confidences)
        corroboration_rate = multi_source_count / total_facts if total_facts > 0 else 0.0

        coherence = (avg_confidence * 0.7) + (corroboration_rate * 0.3)
        return round(coherence, 3)

    async def process_page(self, page: Dict):
        """
        Process a single page: find similar events, decide attach/create/relate
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📄 Processing: {page['title']}")
        logger.info(f"   URL: {page.get('url')}")
        logger.info(f"   Created: {page['created_at']}")
        logger.info(f"{'='*80}\n")

        # Find similar events
        if page.get('embedding'):
            similar_events = await self.find_similar_events(page['embedding'], limit=5)
        else:
            logger.warning("   ⚠️  No embedding for page, creating new event")
            event_id = await self.create_event_from_page(page)
            return event_id

        if not similar_events:
            logger.info("   📊 No similar events found → Creating new event")
            event_id = await self.create_event_from_page(page)
            return event_id

        # Check best match
        best_match = similar_events[0]
        similarity = 1.0 - best_match['similarity']  # Convert distance to similarity

        logger.info(f"   📊 Best match: {best_match['title']}")
        logger.info(f"      Similarity: {similarity:.3f}")
        logger.info(f"      Coherence: {best_match['coherence']:.3f}\n")

        # Decision logic
        if similarity >= self.attach_threshold:
            logger.info(f"   ✅ ATTACH (similarity {similarity:.3f} >= {self.attach_threshold})")
            await self.enrich_existing_event(str(best_match['id']), page)
            return str(best_match['id'])

        elif similarity >= self.relate_threshold:
            logger.info(f"   🔗 CREATE + RELATE (similarity {similarity:.3f} in [{self.relate_threshold}, {self.attach_threshold}))")

            # Create new event
            new_event_id = await self.create_event_from_page(page)

            # Determine relationship type (simple heuristic for now)
            rel_type = self._determine_relationship_type(page, best_match)
            logger.info(f"      Relationship: {rel_type}")

            # Create bidirectional relationship
            await self.create_relationship(new_event_id, str(best_match['id']), rel_type, similarity)

            return new_event_id

        else:
            logger.info(f"   🆕 CREATE STANDALONE (similarity {similarity:.3f} < {self.relate_threshold})")
            event_id = await self.create_event_from_page(page)
            return event_id

    def _determine_relationship_type(self, page: Dict, similar_event: Dict) -> str:
        """Determine relationship type based on content similarity"""
        # Simple heuristic based on title keywords
        title_lower = page['title'].lower()

        if 'similar' in title_lower or 'like' in title_lower or 'comparison' in title_lower:
            return 'SIMILAR_TO'
        elif 'caused' in title_lower or 'due to' in title_lower:
            return 'CAUSED_BY'
        elif 'response' in title_lower or 'aftermath' in title_lower:
            return 'EXTENDS'
        else:
            return 'SIMILAR_TO'  # Default


async def test_event_network():
    """Test event network with Hong Kong fire pages"""

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    builder = EventNetworkBuilder(db_pool, client)

    async with db_pool.acquire() as conn:
        # Get Hong Kong fire pages in chronological order
        pages_raw = await conn.fetch("""
            SELECT
                id, title, description, content_text, url,
                pub_time, created_at, embedding
            FROM core.pages
            WHERE title ILIKE '%hong kong%fire%' OR title ILIKE '%tai po%'
            ORDER BY created_at
        """)

        pages = [dict(p) for p in pages_raw]

    logger.info(f"\n{'#'*80}")
    logger.info(f"EVENT NETWORK TEST: Processing {len(pages)} pages")
    logger.info(f"{'#'*80}\n")

    # Process each page
    event_ids = []
    for i, page in enumerate(pages, 1):
        logger.info(f"[{i}/{len(pages)}]")
        event_id = await builder.process_page(page)
        event_ids.append(event_id)
        logger.info("")

    # Summary
    logger.info(f"\n{'#'*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'#'*80}\n")

    async with db_pool.acquire() as conn:
        # Count events
        event_count = await conn.fetchval("SELECT COUNT(*) FROM core.events")
        rel_count = await conn.fetchval("SELECT COUNT(*) FROM core.event_relationships")

        logger.info(f"Events created: {event_count}")
        logger.info(f"Relationships: {rel_count}\n")

        # Show events
        events = await conn.fetch("""
            SELECT id, title, coherence,
                   (SELECT COUNT(*) FROM core.page_events WHERE event_id = e.id) as page_count
            FROM core.events e
            ORDER BY created_at
        """)

        for event in events:
            logger.info(f"📌 {event['title']}")
            logger.info(f"   ID: {event['id']}")
            logger.info(f"   Coherence: {event['coherence']:.3f}")
            logger.info(f"   Pages: {event['page_count']}\n")

        # Show relationships
        if rel_count > 0:
            relationships = await conn.fetch("""
                SELECT
                    e1.title as event_a,
                    er.relationship_type,
                    e2.title as event_b,
                    er.confidence
                FROM core.event_relationships er
                JOIN core.events e1 ON er.event_id = e1.id
                JOIN core.events e2 ON er.related_event_id = e2.id
            """)

            logger.info("🔗 Relationships:")
            for rel in relationships:
                logger.info(f"   {rel['event_a']}")
                logger.info(f"      --[{rel['relationship_type']} ({rel['confidence']:.3f})]-->")
                logger.info(f"      {rel['event_b']}\n")

    await db_pool.close()
    logger.info("✅ Test complete\n")


if __name__ == '__main__':
    asyncio.run(test_event_network())
